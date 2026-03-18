#!/usr/bin/env python3
import os
import json
from typing import Dict, List

import boto3

from common.logging_utils import log
from common.iceberg_utils import delete_job_rows_from_table
from common.s3_utils import s3_list_keys, parse_s3_uri, s3_read_json, read_obj_with_retry
from common.table_schemas import UPLOAD_STAGING_TABLE_NAME

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[VAL_INGEST_PRE]"

s3 = boto3.client("s3")

def count_manifest_lines(manifests: List[str]) -> int:
    """
    Fallback: Count non-empty lines across all batching manifests (JSONL).
    """
    total = 0
    for uri in manifests:
        try:
            b, k = parse_s3_uri(uri, TASK_NAME)
        except Exception as e:
            raise ValueError(f"{TASK_NAME} Unable to parse s3 uri: {uri}, reason: {e}")

        resp = read_obj_with_retry(b, k, TASK_NAME)
        if resp is None:
            raise RuntimeError(f"{TASK_NAME} Failed to read {uri}, parsed bucket and key are {b} and {k}")

        body = resp["Body"]
        for line in body.iter_lines():
            if not line:
                continue
            if line.decode("utf-8-sig").strip():
                total += 1
    return total

def extract_expected_shards_from_manifests(manifests: List[str]) -> List[str]:
    """
    For validation, manifests are JSONL files named like batch-001.jsonl.
    Use the filename stem as shard name (batch-001).
    """
    expected: List[str] = []
    for m in manifests:

        try:
            _, key = parse_s3_uri(m, TASK_NAME)
        except Exception as e:
            raise ValueError(f"{TASK_NAME} Unable to parse s3 uri: {m}, reason: {e}")

        try:
            fname = key.split("/")[-1]
            shard_name = fname[:-len(".jsonl")] if fname.endswith(".jsonl") else fname.rsplit(".", 1)[0]
            expected.append(shard_name)
        except Exception:
            continue

    # stable unique preserve order
    seen = set()
    out = []
    for s in expected:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def collect_processed_shards(job_id: str, manifests: List[str]) -> Dict:
    """
    Validation worker output layout:
      {processed_prefix}/upload_staging/shard-<shard>.jsonl
      {processed_prefix}/shard-<shard>-summary.json
      {processed_prefix}/shard-<shard>-SUCCESS
    """
    bucket = FILE_BUCKET_NAME
    processed_prefix = f"temp/image-upload/{job_id}/batches/validation-step/processed"
    expected_shards = extract_expected_shards_from_manifests(manifests)
    processed_keys = s3_list_keys(bucket, processed_prefix + "/")

    shard_jsonl: Dict[str, str] = {}
    shard_summary: Dict[str, str] = {}
    shard_success = set()

    for k in processed_keys:
        name = k.split("/")[-1]

        if k.endswith(".jsonl") and "/upload_staging/" in k and name.startswith("shard-"):
            shard = name[len("shard-") : -len(".jsonl")]
            shard_jsonl[shard] = k
        elif name.startswith("shard-") and name.endswith("-summary.json"):
            shard = name[len("shard-") : -len("-summary.json")]
            shard_summary[shard] = k
        elif name.startswith("shard-") and name.endswith("-SUCCESS"):
            shard = name[len("shard-") : -len("-SUCCESS")]
            shard_success.add(shard)

    if not expected_shards:
        expected_shards = sorted(set(shard_jsonl) | set(shard_summary) | set(shard_success))

    missing: List[str] = []
    shards: List[Dict] = []
    total_rows_read = 0
    total_failed_rows = 0

    for shard in expected_shards:
        jsonl_key = shard_jsonl.get(shard)
        summary_key = shard_summary.get(shard)
        ok = shard in shard_success

        if not (jsonl_key and summary_key and ok):
            missing.append(shard)
            continue

        summary = s3_read_json(bucket, summary_key, TASK_NAME)
        rows_read = int(summary.get("rows_read", 0))
        failed_rows = int(summary.get("failed_rows", 0))
        processed_rows = int(summary.get("processed_rows", 0))

        total_rows_read += rows_read
        total_failed_rows += failed_rows

        shards.append(
            {
                "shard": shard,
                "kind": None,
                "rows_read": rows_read,
                "failed_rows": failed_rows,
                "processed_rows": processed_rows,
                "canonical_imagery_rows": None,
                "canonical_label_rows": None,
                "upload_staging_key": jsonl_key,
                "canonical_imagery_key": None,
                "canonical_labels_key": None,
                "image_labels_key": None
            }
        )

    return {
        "missing_shards": missing,
        "shards": shards,
        "total_rows_read": total_rows_read,
        "total_failed_rows": total_failed_rows,
        "processed_prefix": processed_prefix,
    }

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        manifests = event["manifests"]
        expected_count_in = event.get("expected_count")
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} missing job_id in event")
    if not manifests or not isinstance(manifests, list):
        raise RuntimeError(f"{TASK_NAME} manifests must be a non-empty list of s3 URIs")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} Starting validation pre-ingest for job {job_id}")

    # 0) Determine expected_count
    expected_count = None
    if isinstance(expected_count_in, int):
        expected_count = expected_count_in
    elif isinstance(expected_count_in, str) and expected_count_in.strip().isdigit():
        expected_count = int(expected_count_in.strip())

    if expected_count is None:
        # fallback: count manifest lines
        try:
            expected_count = count_manifest_lines(manifests)
        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Failed counting manifest lines: {e}", level="error")
            raise

    if expected_count <= 0:
        err = f"{TASK_NAME} expected_count is {expected_count} (manifests empty?)"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} expected_count={expected_count}")

    # 1) Collect processed outputs + verify completeness
    try:
        collected = collect_processed_shards(job_id, manifests)
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} Failed collecting processed shards: {e}", level="error")
        raise

    missing = collected["missing_shards"]
    if missing:
        err = f"{TASK_NAME} Missing processed outputs for shards: {missing}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    shards = collected["shards"]
    total_rows_read = collected["total_rows_read"]
    total_failed_rows = collected["total_failed_rows"]

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Collected {len(shards)} shard outputs. rows_read={total_rows_read}, failed_rows={total_failed_rows}")

    # 2) Verify counts: workers rows_read must equal expected_count
    if total_rows_read != expected_count:
        err = f"{TASK_NAME} Row count mismatch: expected_count={expected_count}, workers rows_read={total_rows_read}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    # 3) Delete upload_staging partition once (safe even if empty)
    try:
        delete_result = delete_job_rows_from_table(job_id,
                                                    TASK_NAME,
                                                    ICEBERG_DATABASE_NAME,
                                                    UPLOAD_STAGING_TABLE_NAME,
                                                    ATHENA_OUTPUT_S3,
                                                    ATHENA_WORKGROUP)
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Deleted upload_staging partition, result={delete_result}")
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Failed deleting upload_staging partition: {e}", level="error")
        raise

    return {
        "shards": shards,
        "original_count": int(expected_count),  # keep same field name used by post lambda
        "total_rows_read": int(total_rows_read),
        "total_failed_rows": int(total_failed_rows),
        "processed_prefix": collected.get("processed_prefix"),
        "ctas_table_name": None,
    }