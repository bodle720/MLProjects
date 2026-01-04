#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, List

from common.utils import (
    log,
    s3_list_keys,
    athena_count_job_rows,
    delete_iceberg_partition_rows,
)
from common.ingest import s3_read_json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Env
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

def _extract_expected_shards_from_manifests(manifests: List[str]) -> List[str]:
    """
    Try to extract shard names from batching manifests.
    Expected manifest pattern: .../manifest-shard-<name>.json
    """
    expected: List[str] = []
    for m in manifests:
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

    # stable unique
    seen = set()
    out = []
    for s in expected:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _collect_processed_shards(job_id: str, manifests: List[str]) -> Dict:
    """
    Locate per-shard dedup processed outputs
    We expect for each shard:
      - shard-<shard>.jsonl
      - shard-<shard>-summary.json
      - shard-<shard>-SUCCESS

    Returns:
      {
        missing_shards: [...],
        shards: [
          { shard, rows_read, processed_rows, upload_key, imagery_key, labels_key }
        ],
        total_rows_read: int,
        total_processed_rows: int,
      }
    """
    bucket = FILE_BUCKET_NAME
    processed_prefix = f"temp/image-upload/{job_id}/batches/deduplication-step/processed"
    expected_shards = _extract_expected_shards_from_manifests(manifests)
    processed_keys = s3_list_keys(bucket, processed_prefix + "/")

    shard_jsonl: Dict[str, str] = {}
    shard_summary: Dict[str, str] = {}
    shard_success = set()

    for k in processed_keys:
        name = k.split("/")[-1]
        if name.startswith("shard-") and name.endswith(".jsonl"):
            shard = name[len("shard-") : -len(".jsonl")]
            shard_jsonl[shard] = k
        elif name.startswith("shard-") and name.endswith("-summary.json"):
            shard = name[len("shard-") : -len("-summary.json")]
            shard_summary[shard] = k
        elif name.startswith("shard-") and name.endswith("-SUCCESS"):
            shard = name[len("shard-") : -len("-SUCCESS")]
            shard_success.add(shard)

    # if manifest parsing failed, infer from discovered shards
    if not expected_shards:
        expected_shards = sorted(set(shard_jsonl) | set(shard_summary) | set(shard_success))

    missing: List[str] = []
    shards: List[Dict] = []
    total_rows_read = 0
    total_processed_rows = 0

    for shard in expected_shards:
        jsonl_key = shard_jsonl.get(shard)
        summary_key = shard_summary.get(shard)
        ok = shard in shard_success

        if not (jsonl_key and summary_key and ok):
            missing.append(shard)
            continue

        summary = s3_read_json(bucket, summary_key)
        rows_read = int(summary.get("rows_read", 0))
        processed_rows = int(summary.get("processed_rows", 0))

        total_rows_read += rows_read
        total_processed_rows += processed_rows

        # IMPORTANT: include imagery_key/labels_key fields as nulls so Step Functions JSONPaths exist
        shards.append(
            {
                "shard": shard,
                "rows_read": rows_read,
                "processed_rows": processed_rows,
                "upload_key": jsonl_key,
                "imagery_key": None,
                "labels_key": None,
            }
        )

    return {
        "missing_shards": missing,
        "shards": shards,
        "total_rows_read": total_rows_read,
        "total_processed_rows": total_processed_rows,
        "processed_prefix": processed_prefix,
    }


def handler(event, context):
    # Validate input
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        manifests = event["manifests"]
    except KeyError as e:
        raise RuntimeError(f"[DEDUP_INGEST_PRE] Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError("[DEDUP_INGEST_PRE] missing job_id in event")
    if not manifests or not isinstance(manifests, list):
        raise RuntimeError("[DEDUP_INGEST_PRE] manifests must be a non-empty list of s3 URIs")

    log(job_id, user, event_type, f"[DEDUP_INGEST_PRE] Starting dedup pre-ingest for job {job_id}", LOG_FIREHOSE_STREAM_NAME)

    # 1) Collect processed shard outputs and verify completeness
    try:
        collected = _collect_processed_shards(job_id, manifests)
    except Exception as e:
        log(job_id, user, event_type, f"[DEDUP_INGEST_PRE] Failed collecting processed shards: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    missing = collected["missing_shards"]
    if missing:
        err = f"[DEDUP_INGEST_PRE] Missing processed outputs for shards: {missing}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    shards = collected["shards"]
    total_rows_read = collected["total_rows_read"]
    total_processed_rows = collected["total_processed_rows"]

    log(
        job_id,
        user,
        event_type,
        f"[DEDUP_INGEST_PRE] Collected {len(shards)} shard outputs. rows_read={total_rows_read}, processed_rows={total_processed_rows}",
        LOG_FIREHOSE_STREAM_NAME,
    )

    # 2) Verify original count via Athena (before deletion)
    try:
        original_count = athena_count_job_rows(
            job_id,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            "DEDUP_INGEST_PRE",
            athena_workgroup=ATHENA_WORKGROUP,
        )
    except Exception as e:
        log(job_id, user, event_type, f"[DEDUP_INGEST_PRE] Athena count failed: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    log(job_id, user, event_type, f"[DEDUP_INGEST_PRE] Athena original_count={original_count}", LOG_FIREHOSE_STREAM_NAME)

    if total_rows_read != original_count:
        err = f"[DEDUP_INGEST_PRE] Row count mismatch: original_count={original_count}, workers rows_read={total_rows_read}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    # 3) Delete original partition rows once (before Map inserts)
    try:
        delete_result = delete_iceberg_partition_rows(
            job_id,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
        log(job_id, user, event_type, f"[DEDUP_INGEST_PRE] Deleted upload_staging partition, result={delete_result}", LOG_FIREHOSE_STREAM_NAME)
    except Exception as e:
        log(job_id, user, event_type, f"[DEDUP_INGEST_PRE] Failed deleting upload_staging partition: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    # Return the per-shard plan for the Map state
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    ctas_table_name = f"dedup_export_{sanitized_job_id}"

    return {
        "shards": shards,
        "original_count": original_count,
        "total_rows_read": total_rows_read,
        "total_processed_rows": total_processed_rows,
        "processed_prefix": collected.get("processed_prefix"),
        "ctas_table_name":ctas_table_name
    }