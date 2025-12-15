#!/usr/bin/env python3
'''
This deduplication_ingest Lambda correctly finalizes deduplication by enforcing survivor-only semantics in
upload_staging and safely cleans up the job-scoped CTAS table.
'''
import os
import json
import time
import math
import logging
from typing import List, Dict

import boto3
from botocore.exceptions import ClientError

# common utilities used across your project (must exist in common/python and be available at runtime)
from common.utils import log, delete_iceberg_partition_rows, chunked_insert_upload_staging, get_job_input, find_key_recursively

# Environment variables (set by CDK)
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# Where workers write processed outputs (matches worker behavior)
PROCESSED_PREFIX_BASE = os.environ.get(
    "PROCESSED_PREFIX_BASE",
    "temp/image-upload"
)
# e.g. temp/image-upload/{job_id}/batches/deduplication-step/processed
PROCESSED_SUFFIX = "batches/deduplication-step/processed"

# Athena client for count verification
athena = boto3.client("athena")
s3 = boto3.client("s3")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def _s3_list_keys(bucket: str, prefix: str) -> List[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                keys.append(key)
    return keys

def _s3_read_json(bucket: str, key: str) -> Dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))

def _s3_read_jsonl(bucket: str, key: str):
    """Generator yielding parsed JSON objects from an S3 JSONL object."""
    resp = s3.get_object(Bucket=bucket, Key=key)
    for line in resp["Body"].iter_lines():
        if not line:
            continue
        yield json.loads(line.decode("utf-8"))

def _wait_for_athena(qid: str, poll_interval: float = 2.0, timeout_seconds: int = 600):
    start = time.time()
    while True:
        resp = athena.get_query_execution(QueryExecutionId=qid)
        state = resp["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            return True, resp
        if state in ("FAILED", "CANCELLED"):
            reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "")
            return False, reason
        if time.time() - start > timeout_seconds:
            return False, f"timeout after {timeout_seconds}s"
        time.sleep(poll_interval)

def _athena_count_job_rows(job_id: str) -> int:
    """Run a quick Athena count(*) for upload_staging WHERE job_id = '<job_id>'."""
    safe_job_id = job_id.replace("'", "''")
    sql = (
        f"SELECT count(*) as cnt FROM \"{ICEBERG_DATABASE_NAME}\".\"{UPLOAD_STAGING_TABLE_NAME}\" "
        f"WHERE job_id = '{safe_job_id}'"
    )
    q = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP
    )
    qid = q["QueryExecutionId"]
    ok, meta = _wait_for_athena(qid)
    if not ok:
        raise RuntimeError(f"Athena count query failed: {meta}")
    # fetch results
    res = athena.get_query_results(QueryExecutionId=qid, MaxResults=1)
    rows = res.get("ResultSet", {}).get("Rows", [])
    if len(rows) >= 2:
        val = rows[1]["Data"][0].get("VarCharValue")
        try:
            return int(val)
        except Exception:
            return 0
    return 0

def _collect_processed_shards(job_id: str, manifests: List[str]) -> Dict:
    """
    Given the list of manifest S3 URIs (from batching lambda), determine expected shard names,
    verify processed outputs exist, and aggregate summaries.
    Returns dict with:
      - 'shard_summaries': list of summary dicts
      - 'processed_jsonl_keys': list of s3 keys for processed jsonl files
      - 'total_rows_read': int
      - 'total_processed_rows': int
    """
    bucket = FILE_BUCKET_NAME
    processed_prefix = f"{PROCESSED_PREFIX_BASE}/{job_id}/{PROCESSED_SUFFIX}".rstrip("/")
    # Build expected shard names from manifest URIs (manifest filename contains shard name)
    expected_shard_names = []
    for m in manifests:
        # manifest is s3://bucket/.../manifest-shard-<name>.json
        try:
            _, key = m.replace("s3://", "").split("/", 1)
            # extract filename
            fname = key.split("/")[-1]
            # manifest-shard-<name>.json or manifest-shard-<name>-part.json
            if fname.startswith("manifest-shard-") and fname.endswith(".json"):
                shard_name = fname[len("manifest-shard-"):-len(".json")]
            else:
                # fallback: use full filename without extension
                shard_name = fname.rsplit(".", 1)[0]
            expected_shard_names.append(shard_name)
        except Exception:
            # if parsing fails, skip; we'll still scan processed prefix for available shards
            continue

    # List processed keys under processed_prefix
    processed_keys = _s3_list_keys(bucket, processed_prefix + "/")
    # Map shard -> keys
    shard_jsonl = {}
    shard_summary = {}
    shard_success = set()
    for k in processed_keys:
        name = k.split("/")[-1]
        if name.endswith(".jsonl") and name.startswith("shard-"):
            shard = name[len("shard-"):-len(".jsonl")]
            shard_jsonl[shard] = k
        elif name.endswith("-summary.json") and name.startswith("shard-"):
            shard = name[len("shard-"):-len("-summary.json")]
            shard_summary[shard] = k
        elif name.endswith("-SUCCESS") and name.startswith("shard-"):
            shard = name[len("shard-"):-len("-SUCCESS")]
            shard_success.add(shard)

    # If expected_shard_names is empty (manifest parsing failed), use discovered shards
    if not expected_shard_names:
        expected_shard_names = sorted(set(list(shard_jsonl.keys()) + list(shard_summary.keys()) + list(shard_success)))

    # Verify each expected shard has _SUCCESS and summary and jsonl
    missing = []
    shard_summaries = []
    processed_jsonl_keys = []
    total_rows_read = 0
    total_processed_rows = 0

    for shard in expected_shard_names:
        jsonl_key = shard_jsonl.get(shard)
        summary_key = shard_summary.get(shard)
        success_present = shard in shard_success
        if not (jsonl_key and summary_key and success_present):
            missing.append(shard)
            continue
        # read summary
        summary = _s3_read_json(bucket, summary_key)
        shard_summaries.append(summary)
        processed_jsonl_keys.append(jsonl_key)
        total_rows_read += int(summary.get("rows_read", 0))
        total_processed_rows += int(summary.get("processed_rows", 0))

    return {
        "missing_shards": missing,
        "shard_summaries": shard_summaries,
        "processed_jsonl_keys": processed_jsonl_keys,
        "total_rows_read": total_rows_read,
        "total_processed_rows": total_processed_rows
    }

def _read_all_processed_rows(bucket: str, jsonl_keys: List[str]):
    """Generator yielding all processed rows from the list of jsonl S3 keys."""
    for key in jsonl_keys:
        for row in _s3_read_jsonl(bucket, key):
            yield row

def _drop_ctas_table_if_exists(job_id):
    sanitized_job_id = ''.join(c if c.isalnum() else '_' for c in job_id)
    table_name = f"dedup_export_{sanitized_job_id}"
    sql = f'DROP TABLE IF EXISTS "{table_name}"'
    resp = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP
    )
    return resp["QueryExecutionId"]

def handler(event, context):
    # Validate input
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        manifests = event["manifests"]
    except KeyError as e:
        raise RuntimeError(
            f"Missing key in dedup ingest lambda: {e}, event={json.dumps(event)}"
        )

    if not job_id or job_id == "unknown":
        raise RuntimeError("Dedup reingest Lambda failed: missing job_id in event")

    if not manifests:
        raise RuntimeError("Reingest Lambda failed: missing manifests in event")
    if not isinstance(manifests, list):
        raise RuntimeError("Reingest Lambda failed: manifests must be a list of s3 URIs")

    log(job_id, user, event_type, f"Starting dedup reingest for job {job_id}", LOG_FIREHOSE_STREAM_NAME)

    # 1) Collect processed shard outputs and verify completeness
    try:
        collected = _collect_processed_shards(job_id, manifests)
    except Exception as e:
        log(job_id, user, event_type, f"Failed collecting processed shards: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    missing = collected["missing_shards"]
    if missing:
        err = f"Missing processed outputs for shards: {missing}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    total_rows_read = collected["total_rows_read"]
    total_processed_rows = collected["total_processed_rows"]
    processed_jsonl_keys = collected["processed_jsonl_keys"]

    log(job_id, user, event_type, f"Collected {len(processed_jsonl_keys)} processed shard files. rows_read={total_rows_read}, processed_rows={total_processed_rows}", LOG_FIREHOSE_STREAM_NAME)

    # 2) Verify original count via Athena
    try:
        original_count = _athena_count_job_rows(job_id)
    except Exception as e:
        log(job_id, user, event_type, f"Athena count failed for job {job_id}: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    log(job_id, user, event_type, f"Athena original_count={original_count} for job {job_id}", LOG_FIREHOSE_STREAM_NAME)

    # Basic sanity check: rows_read should equal original_count (or at least >= number of processed rows)
    if total_rows_read != original_count:
        err = f"Row count mismatch: Athena original_count={original_count}, workers rows_read={total_rows_read}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    # 3) Delete original partition rows in Iceberg
    try:
        delete_result = delete_iceberg_partition_rows(job_id,
                                                     ICEBERG_DATABASE_NAME,
                                                     UPLOAD_STAGING_TABLE_NAME,
                                                     ATHENA_OUTPUT_S3,
                                                     ATHENA_WORKGROUP)
        log(job_id, user, event_type, f"Deleted upload_staging partition for job {job_id}, result={delete_result}", LOG_FIREHOSE_STREAM_NAME)
    except Exception as e:
        log(job_id, user, event_type, f"Failed to delete upload_staging partition for job {job_id}: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    # 4) Read processed rows and insert back into upload_staging in chunks
    inserted_rows = 0
    try:
        rows_iter = _read_all_processed_rows(FILE_BUCKET_NAME, processed_jsonl_keys)
        chunk = []
        chunk_size = 200
        for r in rows_iter:
            if r.get("dedup_status") in ("survivor", "external_duplicate"):
                chunk.append(r)
            if len(chunk) >= chunk_size:
                all_failed, last_error = chunked_insert_upload_staging(chunk,
                                                                       ICEBERG_DATABASE_NAME,
                                                                       UPLOAD_STAGING_TABLE_NAME,
                                                                       ATHENA_WORKGROUP,
                                                                       ATHENA_OUTPUT_S3,
                                                                       chunk_size=chunk_size)
                if all_failed:
                    raise RuntimeError(f"chunked insert failed for a chunk; last_error={last_error}")
                inserted_rows += len(chunk)
                chunk = []
        # final chunk
        if chunk:
            all_failed, last_error = chunked_insert_upload_staging(chunk,
                                                                   ICEBERG_DATABASE_NAME,
                                                                   UPLOAD_STAGING_TABLE_NAME,
                                                                   ATHENA_WORKGROUP,
                                                                   ATHENA_OUTPUT_S3,
                                                                   chunk_size=chunk_size)
            if all_failed:
                raise RuntimeError(f"chunked insert failed for final chunk; last_error={last_error}")
            inserted_rows += len(chunk)
    except Exception as e:
        log(job_id, user, event_type, f"Failed inserting processed rows for job {job_id}: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    # 5) Optional verification: count rows after insert
    try:
        new_count = _athena_count_job_rows(job_id)
    except Exception as e:
        log(job_id, user, event_type, f"Athena count after insert failed for job {job_id}: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    log(job_id, user, event_type, f"Reingest complete for job {job_id}: inserted_rows={inserted_rows}, new_count={new_count}", LOG_FIREHOSE_STREAM_NAME)

    drop_qid = _drop_ctas_table_if_exists(job_id)
    dropped, meta_or_reason = _wait_for_athena(drop_qid)
    if not dropped:
        err = f"Failed to drop our created CTAS temp table for our current job id = {job_id}, reason = {meta_or_reason}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")

    skipped = total_processed_rows - inserted_rows
    log(job_id, user, event_type,
        f"Filtered out {skipped} non-survivor rows during dedup reingest",
        LOG_FIREHOSE_STREAM_NAME)

    return {
        "job_id": job_id,
        "reingest_done": True,
        "inserted_rows": inserted_rows,
        "original_count": original_count,
        "new_count": new_count
    }