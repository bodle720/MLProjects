#!/usr/bin/env python3
import os
import json
import logging

from common.utils import (
    log,
    athena_count_job_rows,
    wait_for_athena,
)
from common.ingest import _drop_ctas_table_if_exists

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Env
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]


def handler(event, context):
    """
    Expected input (from your TaskInput payload):
      {
        job_id, user, event_type, label_type, data_source,
        pre: {
          shards: [...],
          original_count: int,
          total_rows_read: int,
          total_processed_rows: int,
          processed_prefix: str,
          ctas_table_name: str
        }
      }
    """
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        pre = event["pre"]
    except KeyError as e:
        raise RuntimeError(f"[DEDUP_INGEST_POST] Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError("[DEDUP_INGEST_POST] missing job_id")
    if not isinstance(pre, dict):
        raise RuntimeError("[DEDUP_INGEST_POST] missing/invalid pre payload")

    original_count = pre.get("original_count")
    ctas_table_name = pre.get("ctas_table_name")

    if original_count is None:
        raise RuntimeError("[DEDUP_INGEST_POST] pre.original_count missing")
    if not ctas_table_name:
        raise RuntimeError("[DEDUP_INGEST_POST] pre.ctas_table_name missing")

    log(
        job_id,
        user,
        event_type,
        f"[DEDUP_INGEST_POST] Starting post-ingest verify for job {job_id} (original_count={original_count}, ctas_table={ctas_table_name})",
        LOG_FIREHOSE_STREAM_NAME,
    )

    # 1) Verify upload_staging count after Map inserts
    try:
        new_count = athena_count_job_rows(
            job_id,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            "DEDUP_INGEST_POST",
            athena_workgroup=ATHENA_WORKGROUP,
        )
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            f"[DEDUP_INGEST_POST] Athena count after inserts failed: {e}",
            LOG_FIREHOSE_STREAM_NAME,
            error=str(e),
            level="error",
        )
        raise

    log(
        job_id,
        user,
        event_type,
        f"[DEDUP_INGEST_POST] Athena new_count={new_count} for job {job_id}",
        LOG_FIREHOSE_STREAM_NAME,
    )

    if int(new_count) != int(original_count):
        err = f"[DEDUP_INGEST_POST] Post-insert count mismatch: original_count={original_count}, new_count={new_count}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    # 2) Drop CTAS temp table (safe no-op if not present)
    try:
        drop_qid = _drop_ctas_table_if_exists(
            ICEBERG_DATABASE_NAME,
            ctas_table_name,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
        drop_res = wait_for_athena(drop_qid, poll=2.0, timeout=600)
        if drop_res["state"] != "SUCCEEDED":
            resp = drop_res["metadata"]
            err = f"[DEDUP_INGEST_POST] Failed to drop CTAS temp table={ICEBERG_DATABASE_NAME}.{ctas_table_name}, resp={resp}"
            log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
            raise RuntimeError(err)
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            f"[DEDUP_INGEST_POST] CTAS drop failed: {e}",
            LOG_FIREHOSE_STREAM_NAME,
            error=str(e),
            level="error",
        )
        raise

    log(
        job_id,
        user,
        event_type,
        f"[DEDUP_INGEST_POST] Dedup ingest complete for job {job_id}: original_count={original_count}, new_count={new_count}",
        LOG_FIREHOSE_STREAM_NAME,
    )

    return {
        "job_id": job_id,
        "reingest_done": True,
        "original_count": int(original_count),
        "new_count": int(new_count),
        "ctas_table_dropped": True,
        "ctas_table_name": ctas_table_name,
    }
