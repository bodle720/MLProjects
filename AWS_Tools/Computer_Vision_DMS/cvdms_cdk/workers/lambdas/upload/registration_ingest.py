#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, Iterable, List

import boto3

from common.utils import (
    log,
    delete_iceberg_partition_rows,
    chunked_insert,
    s3_list_keys,
    wait_for_athena
)

# Environment variables (set by CDK)
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]

UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
CANONICAL_IMAGERY_TABLE_NAME = os.environ.get("CANONICAL_IMAGERY_TABLE_NAME", "canonical_imagery")

LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# Where workers write processed outputs
PROCESSED_PREFIX_BASE = os.environ.get("PROCESSED_PREFIX_BASE", "temp/image-upload")
# e.g. temp/image-upload/{job_id}/batches/registration/processed
PROCESSED_SUFFIX = os.environ.get("PROCESSED_SUFFIX", "batches/registration/processed")

# Label table names (not provided via env in your container_env; hardcode to match schema)
CANONICAL_BBOX_TABLE = "canonical_bounding_boxes"
CANONICAL_SEMANTIC_TABLE = "canonical_semantic_masks"
CANONICAL_INSTANCE_TABLE = "canonical_instance_annotations"

LABEL_TABLES: List[str] = [CANONICAL_BBOX_TABLE, CANONICAL_SEMANTIC_TABLE, CANONICAL_INSTANCE_TABLE]

athena = boto3.client("athena")
s3 = boto3.client("s3")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def _s3_read_json(bucket: str, key: str) -> Dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))

def _s3_read_jsonl(bucket: str, key: str) -> Iterable[Dict]:
    """Generator yielding parsed JSON objects from an S3 JSONL object."""
    resp = s3.get_object(Bucket=bucket, Key=key)
    for line in resp["Body"].iter_lines():
        if not line:
            continue
        yield json.loads(line.decode("utf-8"))

def _athena_count_job_rows(job_id: str) -> int:
    """COUNT(*) from upload_staging WHERE job_id='<job_id>'."""
    safe_job_id = job_id.replace("'", "''")
    sql = (
        f"SELECT count(*) as cnt FROM \"{ICEBERG_DATABASE_NAME}\".\"{UPLOAD_STAGING_TABLE_NAME}\" "
        f"WHERE job_id = '{safe_job_id}'"
    )
    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP,
    )["QueryExecutionId"]

    res = wait_for_athena(qid, poll=2.0, timeout=600)
    if res["state"] != "SUCCEEDED":
        raise RuntimeError(f"[REG_INGEST] Athena count failed: {res['metadata']}")

    out = athena.get_query_results(QueryExecutionId=qid)
    rows = out.get("ResultSet", {}).get("Rows", [])
    if len(rows) < 2 or not rows[1].get("Data"):
        return 0
    val = rows[1]["Data"][0].get("VarCharValue")
    return int(val) if val is not None else 0

def _drop_ctas_table_if_exists(job_id: str) -> str:
    """
    Drop the registration CTAS temp table if your batching lambda created one.
    This matches your earlier draft: reg_export_<sanitized_job_id>.
    """
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    table_name = f"{ICEBERG_DATABASE_NAME}.reg_export_{sanitized_job_id}"
    sql = f"DROP TABLE IF EXISTS {table_name}"
    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP,
    )["QueryExecutionId"]
    return qid

def _extract_expected_shards_from_manifests(manifests: List[str]) -> List[str]:
    expected = []
    for m in manifests:
        # s3://bucket/.../manifest-shard-<name>.json
        try:
            _, key = m.replace("s3://", "").split("/", 1)
            fname = key.split("/")[-1]
            if fname.startswith("manifest-shard-") and fname.endswith(".json"):
                shard_name = fname[len("manifest-shard-") : -len(".json")]
            else:
                shard_name = fname.rsplit(".", 1)[0]
            expected.append(shard_name)
        except Exception:
            continue
    # keep stable order, unique
    seen = set()
    out = []
    for s in expected:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _collect_processed_shards(job_id: str, manifests: List[str]) -> Dict:
    """
    Collect per-shard output keys for:
      - upload_staging jsonl
      - canonical_imagery jsonl
      - canonical_labels jsonl
      - summary json
      - success marker

    Returns:
      {
        missing_shards: [...],
        upload_jsonl_keys: [...],
        imagery_jsonl_keys: [...],
        labels_jsonl_keys: [...],
        shard_summaries: [...],
        total_rows_read: int,
        total_canon_imagery_rows: int,
        total_canon_label_rows: int,
      }
    """
    bucket = FILE_BUCKET_NAME
    processed_prefix = f"{PROCESSED_PREFIX_BASE}/{job_id}/{PROCESSED_SUFFIX}".rstrip("/")
    expected_shards = _extract_expected_shards_from_manifests(manifests)
    processed_keys = s3_list_keys(bucket, processed_prefix + "/")

    # shard -> key maps
    shard_upload = {}
    shard_imagery = {}
    shard_labels = {}
    shard_summary = {}
    shard_success = set()

    for k in processed_keys:
        # Examples:
        # .../upload_staging/shard-<shard>.jsonl
        # .../canonical_imagery/shard-<shard>.jsonl
        # .../canonical_labels/shard-<shard>.jsonl
        # .../shard-<shard>-summary.json
        # .../shard-<shard>-SUCCESS
        name = k.split("/")[-1]

        if name.startswith("shard-") and name.endswith(".jsonl"):
            shard = name[len("shard-") : -len(".jsonl")]
            if "/upload_staging/" in k:
                shard_upload[shard] = k
            elif "/canonical_imagery/" in k:
                shard_imagery[shard] = k
            elif "/canonical_labels/" in k:
                shard_labels[shard] = k

        elif name.startswith("shard-") and name.endswith("-summary.json"):
            shard = name[len("shard-") : -len("-summary.json")]
            shard_summary[shard] = k

        elif name.startswith("shard-") and name.endswith("-SUCCESS"):
            shard = name[len("shard-") : -len("-SUCCESS")]
            shard_success.add(shard)

    # If manifests parsing failed, infer from discovered shards
    if not expected_shards:
        expected_shards = sorted(
            set(shard_upload.keys())
            | set(shard_imagery.keys())
            | set(shard_labels.keys())
            | set(shard_summary.keys())
            | set(shard_success)
        )

    missing = []
    upload_keys = []
    imagery_keys = []
    labels_keys = []
    shard_summaries = []

    total_rows_read = 0
    total_canon_imagery_rows = 0
    total_canon_label_rows = 0

    for shard in expected_shards:
        u = shard_upload.get(shard)
        i = shard_imagery.get(shard)
        l = shard_labels.get(shard)
        s = shard_summary.get(shard)
        ok = shard in shard_success

        if not (u and i and l and s and ok):
            missing.append(shard)
            continue

        summary = _s3_read_json(bucket, s)
        shard_summaries.append(summary)

        upload_keys.append(u)
        imagery_keys.append(i)
        labels_keys.append(l)

        total_rows_read += int(summary.get("rows_read", 0))
        total_canon_imagery_rows += int(summary.get("canonical_imagery_rows", 0))
        total_canon_label_rows += int(summary.get("canonical_label_rows", 0))

    return {
        "missing_shards": missing,
        "upload_jsonl_keys": upload_keys,
        "imagery_jsonl_keys": imagery_keys,
        "labels_jsonl_keys": labels_keys,
        "shard_summaries": shard_summaries,
        "total_rows_read": total_rows_read,
        "total_canon_imagery_rows": total_canon_imagery_rows,
        "total_canon_label_rows": total_canon_label_rows,
    }

def _iter_rows_from_jsonl_keys(bucket: str, keys: List[str]) -> Iterable[Dict]:
    for key in keys:
        yield from _s3_read_jsonl(bucket, key)

def handler(event, context):
    # Validate input
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        manifests = event["registrationStage"]["manifests"]
        label_type = event.get("label_type", "unknown")
    except KeyError as e:
        raise RuntimeError(f"[REG_INGEST] Missing key in registration ingest lambda: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError("[REG_INGEST] Registration reingest lambda: missing job_id")
    if not manifests or not isinstance(manifests, list):
        raise RuntimeError("[REG_INGEST] Registration reingest lambda: manifests must be a list of s3 URIs")

    log(job_id, user, event_type, f"[REG_INGEST] Starting registration reingest for job {job_id} label_type={label_type}", LOG_FIREHOSE_STREAM_NAME)

    # 1) Collect processed shard outputs + verify completeness
    try:
        collected = _collect_processed_shards(job_id, manifests)
    except Exception as e:
        log(job_id, user, event_type, f"[REG_INGEST] Failed collecting processed shards: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    missing = collected["missing_shards"]
    if missing:
        err = f"[REG_INGEST] Missing processed outputs for shards: {missing}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    upload_keys = collected["upload_jsonl_keys"]
    imagery_keys = collected["imagery_jsonl_keys"]
    labels_keys = collected["labels_jsonl_keys"]

    total_rows_read = collected["total_rows_read"]
    total_canon_imagery_rows = collected["total_canon_imagery_rows"]
    total_canon_label_rows = collected["total_canon_label_rows"]

    log(
        job_id, user, event_type,
        f"[REG_INGEST] Collected {len(upload_keys)} shard outputs. "
        f"rows_read={total_rows_read}, canon_imagery_rows={total_canon_imagery_rows}, canon_label_rows={total_canon_label_rows}",
        LOG_FIREHOSE_STREAM_NAME
    )

    # 2) Verify original count via Athena (upload_staging rows before we delete)
    try:
        original_count = _athena_count_job_rows(job_id)
    except Exception as e:
        log(job_id, user, event_type, f"[REG_INGEST] Athena count failed for job {job_id}: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    log(job_id, user, event_type, f"[REG_INGEST] Athena original_count={original_count} for job {job_id}", LOG_FIREHOSE_STREAM_NAME)

    if total_rows_read != original_count:
        err = f"[REG_INGEST] Row count mismatch: Athena original_count={original_count}, workers rows_read={total_rows_read}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    # 3) Delete upload_staging partition rows for job_id
    try:
        delete_result = delete_iceberg_partition_rows(
            job_id,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
        log(job_id, user, event_type, f"[REG_INGEST] Deleted upload_staging partition for job {job_id}, result={delete_result}", LOG_FIREHOSE_STREAM_NAME)
    except Exception as e:
        log(job_id, user, event_type, f"[REG_INGEST] Failed to delete upload_staging partition for job {job_id}: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    # 4) Reinsert updated upload_staging rows
    inserted_upload = 0
    try:
        rows_iter = _iter_rows_from_jsonl_keys(FILE_BUCKET_NAME, upload_keys)
        chunk = []
        chunk_size = 200

        for r in rows_iter:
            chunk.append(r)
            if len(chunk) >= chunk_size:
                all_failed, last_error = chunked_insert(
                    chunk,
                    ICEBERG_DATABASE_NAME,
                    UPLOAD_STAGING_TABLE_NAME,
                    ATHENA_WORKGROUP,
                    ATHENA_OUTPUT_S3,
                    chunk_size=chunk_size,
                )
                if all_failed or last_error:
                    raise RuntimeError(f"[REG_INGEST] upload_staging chunked_insert failures; all_failed={all_failed}, last_error={last_error}")
                inserted_upload += len(chunk)
                chunk = []

        if chunk:
            all_failed, last_error = chunked_insert(
                chunk,
                ICEBERG_DATABASE_NAME,
                UPLOAD_STAGING_TABLE_NAME,
                ATHENA_WORKGROUP,
                ATHENA_OUTPUT_S3,
                chunk_size=chunk_size,
            )
            if all_failed or last_error:
                raise RuntimeError(f"[REG_INGEST] upload_staging chunked_insert failures; all_failed={all_failed}, last_error={last_error}")
            inserted_upload += len(chunk)

    except Exception as e:
        log(job_id, user, event_type, f"[REG_INGEST] Failed reinserting upload_staging rows: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    # 5) Insert canonical_imagery rows (may be 0 if everything failed registration)
    inserted_canon_imagery = 0
    try:
        rows_iter = _iter_rows_from_jsonl_keys(FILE_BUCKET_NAME, imagery_keys)
        chunk = []
        chunk_size = 200

        for r in rows_iter:
            chunk.append(r)
            if len(chunk) >= chunk_size:
                all_failed, last_error = chunked_insert(
                    chunk,
                    ICEBERG_DATABASE_NAME,
                    CANONICAL_IMAGERY_TABLE_NAME,
                    ATHENA_WORKGROUP,
                    ATHENA_OUTPUT_S3,
                    chunk_size=chunk_size,
                )
                if all_failed or last_error:
                    raise RuntimeError(f"[REG_INGEST] canonical_imagery chunked_insert failures; all_failed={all_failed}, last_error={last_error}")
                inserted_canon_imagery += len(chunk)
                chunk = []

        if chunk:
            all_failed, last_error = chunked_insert(
                chunk,
                ICEBERG_DATABASE_NAME,
                CANONICAL_IMAGERY_TABLE_NAME,
                ATHENA_WORKGROUP,
                ATHENA_OUTPUT_S3,
                chunk_size=chunk_size,
            )
            if all_failed or last_error:
                raise RuntimeError(f"[REG_INGEST] canonical_imagery chunked_insert failures; all_failed={all_failed}, last_error={last_error}")
            inserted_canon_imagery += len(chunk)

    except Exception as e:
        log(job_id, user, event_type, f"[REG_INGEST] Failed inserting canonical_imagery rows: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    # 6) Insert canonical label rows routed by __table (may be empty)
    inserted_labels_total = 0
    inserted_labels_by_table: Dict[str, int] = {t: 0 for t in LABEL_TABLES}

    try:
        # We stream and flush per-table chunks to avoid holding everything in memory.
        per_table_chunks: Dict[str, List[Dict]] = {t: [] for t in LABEL_TABLES}
        chunk_size = 200

        for row in _iter_rows_from_jsonl_keys(FILE_BUCKET_NAME, labels_keys):
            table = row.get("__table")
            if not table:
                # if your writer always includes __table for label rows, treat missing as bad data
                raise RuntimeError(f"[REG_INGEST] Label row missing __table routing field: {row}")

            if table not in LABEL_TABLES:
                raise RuntimeError(f"[REG_INGEST] Unsupported label table: {table}")

            # strip routing field
            row = dict(row)
            row.pop("__table", None)

            per_table_chunks[table].append(row)

            if len(per_table_chunks[table]) >= chunk_size:
                all_failed, last_error = chunked_insert(
                    per_table_chunks[table],
                    ICEBERG_DATABASE_NAME,
                    table,
                    ATHENA_WORKGROUP,
                    ATHENA_OUTPUT_S3,
                    chunk_size=chunk_size
                )
                if all_failed or last_error:
                    raise RuntimeError(f"[REG_INGEST] label table insert failures table={table}; all_failed={all_failed}, last_error={last_error}")
                inserted = len(per_table_chunks[table])
                inserted_labels_total += inserted
                inserted_labels_by_table[table] += inserted
                per_table_chunks[table] = []

        # flush any remaining
        for table, chunk in per_table_chunks.items():
            if not chunk:
                continue
            all_failed, last_error = chunked_insert(
                chunk,
                ICEBERG_DATABASE_NAME,
                table,
                ATHENA_WORKGROUP,
                ATHENA_OUTPUT_S3,
                chunk_size=chunk_size,
            )
            if all_failed or last_error:
                raise RuntimeError(f"[REG_INGEST] label table insert failures table={table}; all_failed={all_failed}, last_error={last_error}")
            inserted = len(chunk)
            inserted_labels_total += inserted
            inserted_labels_by_table[table] += inserted

    except Exception as e:
        log(job_id, user, event_type, f"[REG_INGEST] Failed inserting canonical label rows: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    # 7) Verify upload_staging count after reinsertion
    try:
        new_count = _athena_count_job_rows(job_id)
    except Exception as e:
        log(job_id, user, event_type, f"[REG_INGEST] Athena count after insert failed for job {job_id}: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    log(
        job_id, user, event_type,
        f"[REG_INGEST] Reingest complete for job {job_id}: "
        f"upload_inserted={inserted_upload}, upload_count={new_count}, "
        f"canon_imagery_inserted={inserted_canon_imagery}, labels_inserted={inserted_labels_total}, labels_by_table={inserted_labels_by_table}",
        LOG_FIREHOSE_STREAM_NAME
    )

    if new_count != original_count:
        raise RuntimeError(
            f"[REG_INGEST] Post-reinsert count mismatch: original_count={original_count}, new_count={new_count}."
        )

    # 8) Drop CTAS temp table if used by batching stage (safe no-op if not present)
    drop_qid = _drop_ctas_table_if_exists(job_id)
    drop_res = wait_for_athena(drop_qid, poll=2.0, timeout=600)
    if drop_res["state"] != "SUCCEEDED":
        resp = drop_res["metadata"]
        err = f"[REG_INGEST] Failed to drop CTAS temp table for job_id={job_id}, response={resp}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    return {
        "job_id": job_id,
        "reingest_done": True,
        "original_upload_count": original_count,
        "new_upload_count": new_count,
        "upload_rows_inserted": inserted_upload,
        "canonical_imagery_rows_inserted": inserted_canon_imagery,
        "canonical_label_rows_inserted": inserted_labels_total,
        "canonical_label_rows_by_table": inserted_labels_by_table,
    }
