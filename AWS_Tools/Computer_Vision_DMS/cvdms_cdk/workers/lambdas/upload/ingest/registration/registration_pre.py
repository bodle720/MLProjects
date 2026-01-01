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
from common.ingest import _s3_read_json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Env
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]

UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
CANONICAL_IMAGERY_TABLE_NAME = os.environ.get("CANONICAL_IMAGERY_TABLE_NAME", "canonical_imagery")

LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# Label table names (must match your schema + writer’s __table routing)
CANONICAL_BBOX_TABLE = "canonical_bounding_boxes"
CANONICAL_SEMANTIC_TABLE = "canonical_semantic_masks"
CANONICAL_INSTANCE_TABLE = "canonical_instance_annotations"
LABEL_TABLES: List[str] = [CANONICAL_BBOX_TABLE, CANONICAL_SEMANTIC_TABLE, CANONICAL_INSTANCE_TABLE]


def _extract_expected_shards_from_manifests(manifests: List[str]) -> List[str]:
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
    out: List[str] = []
    for s in expected:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _collect_processed_shards(job_id: str, manifests: List[str]) -> Dict:
    """
    Locate per-shard registration processed outputs

    We expect for each shard:
      - upload_staging/shard-<shard>.jsonl
      - canonical_imagery/shard-<shard>.jsonl
      - canonical_labels/shard-<shard>.jsonl
      - shard-<shard>-summary.json
      - shard-<shard>-SUCCESS

    Returns:
      {
        missing_shards: [...],
        shards: [
          {
            shard,
            rows_read,
            canonical_imagery_rows,
            canonical_label_rows,
            upload_key,
            imagery_key,
            labels_key,
          }
        ],
        total_rows_read: int,
        total_canon_imagery_rows: int,
        total_canon_label_rows: int,
        processed_prefix: str,
      }
    """
    bucket = FILE_BUCKET_NAME
    processed_prefix = f"temp/image-upload/{job_id}/batches/registration-step/processed"

    expected_shards = _extract_expected_shards_from_manifests(manifests)
    processed_keys = s3_list_keys(bucket, processed_prefix + "/")

    shard_upload: Dict[str, str] = {}
    shard_imagery: Dict[str, str] = {}
    shard_labels: Dict[str, str] = {}
    shard_summary: Dict[str, str] = {}
    shard_success = set()

    for k in processed_keys:
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

    missing: List[str] = []
    shards: List[Dict] = []

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
        rows_read = int(summary.get("rows_read", 0))
        canon_im_rows = int(summary.get("canonical_imagery_rows", 0))
        canon_lbl_rows = int(summary.get("canonical_label_rows", 0))

        total_rows_read += rows_read
        total_canon_imagery_rows += canon_im_rows
        total_canon_label_rows += canon_lbl_rows

        shards.append(
            {
                "shard": shard,
                "rows_read": rows_read,
                "canonical_imagery_rows": canon_im_rows,
                "canonical_label_rows": canon_lbl_rows,
                "upload_key": u,
                "imagery_key": i,
                "labels_key": l,
            }
        )

    return {
        "missing_shards": missing,
        "shards": shards,
        "total_rows_read": total_rows_read,
        "total_canon_imagery_rows": total_canon_imagery_rows,
        "total_canon_label_rows": total_canon_label_rows,
        "processed_prefix": processed_prefix,
    }


def handler(event, context):
    """
    Expected input (from your IngestStage pre Lambda TaskInput payload):
      { job_id, user, event_type, manifests, label_type?, data_source? }
    """
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        manifests = event["manifests"]
        label_type = event.get("label_type", "unknown")
    except KeyError as e:
        raise RuntimeError(f"[REG_INGEST_PRE] Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError("[REG_INGEST_PRE] missing job_id")
    if not manifests or not isinstance(manifests, list):
        raise RuntimeError("[REG_INGEST_PRE] manifests must be a non-empty list of s3 URIs")

    log(
        job_id,
        user,
        event_type,
        f"[REG_INGEST_PRE] Starting registration pre-ingest for job {job_id} label_type={label_type}",
        LOG_FIREHOSE_STREAM_NAME,
    )

    # 1) Collect processed shard outputs and verify completeness
    try:
        collected = _collect_processed_shards(job_id, manifests)
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            f"[REG_INGEST_PRE] Failed collecting processed shards: {e}",
            LOG_FIREHOSE_STREAM_NAME,
            error=str(e),
            level="error",
        )
        raise

    missing = collected["missing_shards"]
    if missing:
        err = f"[REG_INGEST_PRE] Missing processed outputs for shards: {missing}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    shards = collected["shards"]
    total_rows_read = collected["total_rows_read"]
    total_canon_imagery_rows = collected["total_canon_imagery_rows"]
    total_canon_label_rows = collected["total_canon_label_rows"]

    log(
        job_id,
        user,
        event_type,
        f"[REG_INGEST_PRE] Collected {len(shards)} shard outputs. "
        f"rows_read={total_rows_read}, canon_imagery_rows={total_canon_imagery_rows}, canon_label_rows={total_canon_label_rows}",
        LOG_FIREHOSE_STREAM_NAME,
    )

    # 2) Verify original count via Athena (before deletion)
    try:
        original_count = athena_count_job_rows(
            job_id,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            "REG_INGEST_PRE",
            athena_workgroup=ATHENA_WORKGROUP,
        )
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            f"[REG_INGEST_PRE] Athena count failed: {e}",
            LOG_FIREHOSE_STREAM_NAME,
            error=str(e),
            level="error",
        )
        raise

    log(
        job_id,
        user,
        event_type,
        f"[REG_INGEST_PRE] Athena original_count={original_count} for job {job_id}",
        LOG_FIREHOSE_STREAM_NAME,
    )

    if total_rows_read != original_count:
        err = f"[REG_INGEST_PRE] Row count mismatch: original_count={original_count}, workers rows_read={total_rows_read}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    # 3) Delete upload_staging partition rows once (before Map reinserts)
    try:
        delete_result = delete_iceberg_partition_rows(
            job_id,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
        log(
            job_id,
            user,
            event_type,
            f"[REG_INGEST_PRE] Deleted upload_staging partition, result={delete_result}",
            LOG_FIREHOSE_STREAM_NAME,
        )
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            f"[REG_INGEST_PRE] Failed deleting upload_staging partition: {e}",
            LOG_FIREHOSE_STREAM_NAME,
            error=str(e),
            level="error",
        )
        raise

    # 4) Precompute CTAS temp table name for post step
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    ctas_table_name = f"reg_export_{sanitized_job_id}"

    return {
        "shards": shards,
        "original_count": int(original_count),
        "total_rows_read": int(total_rows_read),
        "total_canon_imagery_rows": int(total_canon_imagery_rows),
        "total_canon_label_rows": int(total_canon_label_rows),
        "processed_prefix": collected.get("processed_prefix"),
        "ctas_table_name": ctas_table_name,
        "label_tables": LABEL_TABLES,  # handy for debugging; map can also hardcode
        "canonical_imagery_table": CANONICAL_IMAGERY_TABLE_NAME,
    }
