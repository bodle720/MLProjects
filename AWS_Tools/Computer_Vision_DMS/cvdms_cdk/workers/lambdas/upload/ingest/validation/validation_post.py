#!/usr/bin/env python3
import os
import json

from common.logging_utils import log
from common.athena_utils import athena_count_job_rows
from common.table_schemas import UPLOAD_STAGING_TABLE_NAME

ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[VAL_INGEST_POST]"

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        pre = event["pre"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} missing job_id")
    if not isinstance(pre, dict):
        raise RuntimeError(f"{TASK_NAME} missing/invalid pre payload")

    original_count = pre.get("original_count")
    if original_count is None:
        raise RuntimeError(f"{TASK_NAME} pre.original_count missing")

    try:
        original_count = int(float(original_count))
    except Exception as e:
        raise RuntimeError(f"{TASK_NAME} pre.original_count is not a number ({original_count}): {e}")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Starting post-ingest verify for job {job_id} (expected_count={original_count})")

    # 1) Verify upload_staging count after Map inserts
    try:
        new_count = athena_count_job_rows(job_id,
                                        TASK_NAME,
                                        ICEBERG_DATABASE_NAME,
                                        UPLOAD_STAGING_TABLE_NAME,
                                        ATHENA_OUTPUT_S3,
                                        ATHENA_WORKGROUP,
                                        poll=5.0,
                                        timeout=850)
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Athena count after inserts failed: {e}", level="error")
        raise

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Athena new_count={new_count} for job {job_id}")

    if int(new_count) != int(original_count):
        err = f"{TASK_NAME} Post-insert count mismatch: expected_count={original_count}, new_count={new_count}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    # Validation ingest has no CTAS table to drop
    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Validation ingest complete for job {job_id}: expected_count={original_count}, new_count={new_count}")

    return {
        "job_id": job_id,
        "reingest_done": True,
        "expected_count": int(original_count),
        "new_count": int(new_count),
        "ctas_table_dropped": False,
        "ctas_table_name": None,
    }