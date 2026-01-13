#!/usr/bin/env python3
import os
import json
from typing import Any, Dict

from common.logging_utils import log
from common.athena_utils import athena_count_job_rows, drop_table_if_exists

from common.table_schemas import (
    UPLOAD_STAGING_TABLE_NAME,
    CANONICAL_IMAGERY_TABLE_NAME,
    IMAGE_LABELS_TABLE_NAME,
    CANONICAL_BBOX_TABLE_NAME,
    CANONICAL_SEMANTIC_TABLE_NAME,
    CANONICAL_INSTANCE_TABLE_NAME,
)

ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[REG_INGEST_POST]"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def handler(event, context):
    """
    Expected Step Functions payload (per your IngestStage post payload):
      {
        job_id, user, event_type, label_type, data_source,
        pre: {
          expected_count,
          total_rows_read,
          total_canon_imagery_rows,
          total_canon_label_rows,
          total_image_labels_rows,
          processed_prefix,
          ctas_table_name,
          expected_target_shards,
          owner_shards
        }
      }
    """
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        pre: Dict[str, Any] = event["pre"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    expected_count = _safe_int(pre.get("expected_count"))
    total_rows_read = _safe_int(pre.get("total_rows_read"))
    total_canon_imagery_rows = _safe_int(pre.get("total_canon_imagery_rows"))
    total_canon_label_rows = _safe_int(pre.get("total_canon_label_rows"))
    total_image_labels_rows = _safe_int(pre.get("total_image_labels_rows"))

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Starting post-ingest verification for job={job_id}. "
            f"expected_count={expected_count} total_rows_read={total_rows_read} "
            f"canon_imagery_rows_expected={total_canon_imagery_rows} "
            f"canon_label_rows_total_expected={total_canon_label_rows} "
            f"image_labels_rows_expected={total_image_labels_rows}"
        ),
    )

    # ---- 1) Verify upload_staging row count for this job ----
    try:
        upload_count = athena_count_job_rows(
            job_id,
            f"{TASK_NAME}.count_upload_staging",
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} upload_staging count failed: {e}", level="error")
        raise

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} upload_staging job_count={upload_count}")

    # This is your strongest correctness check: every eligible row produced by CTAS should be back in upload_staging.
    if expected_count and int(upload_count) != int(expected_count):
        err = (
            f"{TASK_NAME} upload_staging count mismatch: expected_count={expected_count}, "
            f"athena_job_count={upload_count}"
        )
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    # ---- 2) Optional sanity: canonical_imagery count (GLOBAL table; count by data_source/job context is hard) ----
    # We don't have job_id in canonical_imagery, so we cannot do a direct count-by-job check with your current schema.
    # Instead, we only log what pre-lambda expected and skip hard verification.

    # ---- 3) Optional sanity: canonical label tables / image_labels ----
    # Same story:
    # - image_labels has no job_id, so we cannot count-by-job deterministically here.
    # - canonical_* label tables also have no job_id.
    # We rely on map idempotency (insert-only) + the upload_staging audit being correct.

    # ---- 4) Drop CTAS temp table created by registration batching (safe no-op if absent) ----
    ctas_table_name = pre.get("ctas_table_name")
    if not isinstance(ctas_table_name, str) or not ctas_table_name.strip():
        sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
        ctas_table_name = f"reg_export_{sanitized_job_id}"

    try:
        drop_table_if_exists(
            ICEBERG_DATABASE_NAME,
            ctas_table_name,
            f"{TASK_NAME}.drop_ctas",
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} CTAS drop failed table={ctas_table_name}: {e}",
            level="error",
        )
        raise

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Completed successfully for job {job_id}")

    return {
        "job_id": job_id,
        "verified_upload_staging_job_count": int(upload_count),
        "expected_count": int(expected_count),
        "total_rows_read": int(total_rows_read),
        "total_canon_imagery_rows_expected": int(total_canon_imagery_rows),
        "total_canon_label_rows_total_expected": int(total_canon_label_rows),
        "total_image_labels_rows_expected": int(total_image_labels_rows),
        "ctas_table_dropped": ctas_table_name,
    }
