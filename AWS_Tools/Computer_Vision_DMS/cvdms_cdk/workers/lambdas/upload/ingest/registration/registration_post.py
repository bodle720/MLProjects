#!/usr/bin/env python3
import os
import json
import logging
from typing import Any, Dict

from common.utils import log, athena_count_job_rows, wait_for_athena
from common.ingest import drop_ctas_table_if_exists

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Env
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

def _require(d: Dict[str, Any], key: str, task: str) -> Any:
    if key not in d:
        raise RuntimeError(f"[{task}] Missing key '{key}' in payload: {json.dumps(d)}")
    return d[key]

def handler(event, context):
    """
    Expected Step Functions payload (per your IngestStage post payload):
      {
        job_id, user, event_type, label_type, data_source,
        pre: {
          original_count, total_rows_read, ...,
          ctas_table_name (optional but recommended)
        }
      }
    """
    task = "REG_INGEST_POST"

    job_id = _require(event, "job_id", task)
    user = _require(event, "user", task)
    event_type = _require(event, "event_type", task)

    pre = _require(event, "pre", task)
    original_count = _require(pre, "original_count", task)

    log(job_id, user, event_type, f"[{task}] Starting post-ingest verification for job {job_id}", LOG_FIREHOSE_STREAM_NAME)

    # 1) Verify upload_staging row count after Map inserts
    try:
        new_count = athena_count_job_rows(
            job_id,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            task,
            athena_workgroup=ATHENA_WORKGROUP,
        )
    except Exception as e:
        log(job_id, user, event_type, f"[{task}] Athena count after inserts failed: {e}",
            LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    log(job_id, user, event_type, f"[{task}] upload_staging new_count={new_count}, original_count={original_count}", LOG_FIREHOSE_STREAM_NAME)

    if int(new_count) != int(original_count):
        err = f"[{task}] Post-reinsert count mismatch: original_count={original_count}, new_count={new_count}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    # 2) Drop CTAS temp table created by registration batching (safe no-op if absent)
    # Prefer name computed in pre-lambda; otherwise compute here.
    ctas_table_name = pre.get("ctas_table_name")
    if not ctas_table_name:
        sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
        ctas_table_name = f"reg_export_{sanitized_job_id}"

    try:
        drop_qid = drop_ctas_table_if_exists(
            ICEBERG_DATABASE_NAME,
            ctas_table_name,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
        res = wait_for_athena(drop_qid, poll=2.0, timeout=600)
        if res.get("state") != "SUCCEEDED":
            resp = res.get("metadata")
            err = f"[{task}] Failed to drop CTAS temp table for job_id={job_id}, table={ctas_table_name}, response={resp}"
            log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
            raise RuntimeError(err)

    except Exception as e:
        log(job_id, user, event_type, f"[{task}] CTAS drop failed table={ctas_table_name}: {e}",
            LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    log(job_id, user, event_type, f"[{task}] Completed successfully for job {job_id}", LOG_FIREHOSE_STREAM_NAME)

    # Note: Map results are discarded in the construct, so we can’t compute inserted_* totals here.
    # If we later keep Map results, we can aggregate them here.
    return {
        "job_id": job_id,
        "reingest_done": True,
        "original_upload_count": int(original_count),
        "new_upload_count": int(new_count),
        "ctas_table_dropped": ctas_table_name,
    }