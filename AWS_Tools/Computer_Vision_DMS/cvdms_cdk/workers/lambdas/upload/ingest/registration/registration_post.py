#!/usr/bin/env python3
import os
import json

from common.logging_utils import log
from common.athena_utils import athena_count_job_rows, drop_table_if_exists
from common.table_schemas import UPLOAD_STAGING_TABLE_NAME

ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[REG_INGEST_POST]"

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
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        pre = event["pre"]
        original_count = pre["original_count"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Starting post-ingest verification for job {job_id}")

    # 1) Verify upload_staging row count after Map inserts
    try:
        new_count = athena_count_job_rows(
                                        job_id,
                                        TASK_NAME,
                                        ICEBERG_DATABASE_NAME,
                                        UPLOAD_STAGING_TABLE_NAME,
                                        ATHENA_OUTPUT_S3,
                                        ATHENA_WORKGROUP
                                    )
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Athena count after inserts failed: {e}", level="error")
        raise

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} upload_staging new_count={new_count}, original_count={original_count}")

    if int(new_count) != int(original_count):
        err = f"{TASK_NAME} Post-reinsert count mismatch: original_count={original_count}, new_count={new_count}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    # 2) Drop CTAS temp table created by registration batching (safe no-op if absent)
    # Prefer name computed in pre-lambda; otherwise compute here.
    ctas_table_name = pre.get("ctas_table_name")
    if not ctas_table_name:
        sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
        ctas_table_name = f"reg_export_{sanitized_job_id}"

    try:
        drop_table_if_exists(ICEBERG_DATABASE_NAME,
                            ctas_table_name,
                            TASK_NAME,
                            ATHENA_OUTPUT_S3,
                            ATHENA_WORKGROUP)
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} CTAS drop failed table={ctas_table_name}: {e}", level="error")
        raise

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Completed successfully for job {job_id}")

    # Note: Map results are discarded in the construct, so we can’t compute inserted_* totals here.
    # If we later keep Map results, we can aggregate them here.
    return {
        "job_id": job_id,
        "reingest_done": True,
        "original_upload_count": int(original_count),
        "new_upload_count": int(new_count),
        "ctas_table_dropped": ctas_table_name,
    }