import os
import json

import boto3

# Lambda layer imports
from common.utils import (
    log,
    update_job_status,
    release_lock,
    delete_s3_prefix,
    delete_iceberg_partition_rows,
)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ["UPLOAD_STAGING_TABLE_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]

athena = boto3.client("athena")

def drop_ctas_table_if_exists(job_id):
    sanitized_job_id = ''.join(c if c.isalnum() else '_' for c in job_id)
    table_name = f"{ICEBERG_DATABASE_NAME}.dedup_export_{sanitized_job_id}"
    sql = f'DROP TABLE IF EXISTS {table_name}'
    resp = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP
    )
    return resp["QueryExecutionId"]

def handler(event, context):
    # DLQ messages come in as a batch from SQS
    total_records = 0
    num_processed_successfully = 0
    for record in event['Records']:
        total_records += 1

        update_success = False
        release_success = False
        s3_delete_success = False
        iceberg_job_rows_delete_success = False

        try:
            body = json.loads(record["body"])
        except Exception:
            print("[DLQ] Skipping non-JSON message")
            continue

        source = body.get("source")
        job_id = body.get("job_id")
        user = body.get("user")
        event_type = body.get("event_type")
        error = body.get("error")

        if source not in ("stepfunctions", "kickoff", "lambda"):
            print(f"[DLQ_PROCESSOR] Skipping non-workflow message, unknown source = {source}")
            continue

        if (job_id in (None, "unknown")) or (user is None) or (event_type is None):
            print(f"[DLQ_PROCESSOR] Ignoring non-job DLQ message: {body}")
            continue

        log(job_id, user, event_type, "[DLQ_PROCESSOR] Original failure reason", LOG_FIREHOSE_STREAM_NAME, error=str(error), level="error")

        # 1. Delete S3 temp files
        log(job_id, user, event_type, f"[DLQ_PROCESSOR] DLQ received valid error message: {body}", LOG_FIREHOSE_STREAM_NAME)

        prefix = f"temp/image-upload/{job_id}/"

        try:
            delete_s3_prefix(FILE_BUCKET_NAME, prefix)
        except Exception as e:
            log(job_id, user, event_type, "[DLQ_PROCESSOR] S3 cleanup failed", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        else:
            s3_delete_success = True
            log(job_id, user, event_type, f"[DLQ_PROCESSOR] Done deleting s3 temp files in dlq lambda processor", LOG_FIREHOSE_STREAM_NAME)

        # 2. Delete staging table rows im iceberg table
        try:
            delete_result = delete_iceberg_partition_rows(job_id,
                                                          ICEBERG_DATABASE_NAME,
                                                          UPLOAD_STAGING_TABLE_NAME,
                                                          ATHENA_OUTPUT_S3,
                                                          ATHENA_WORKGROUP)
        except Exception as e:
            log(job_id, user, event_type, "[DLQ_PROCESSOR] Iceberg cleanup failed", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        else:
            iceberg_job_rows_delete_success = True
            log(job_id, user, event_type, f"[DLQ_PROCESSOR] Done deleting iceberg staging row in dlq processor, results = {delete_result}", LOG_FIREHOSE_STREAM_NAME)

        # 3. mark job status as FAILED
        try:
            update_success, update_msg = update_job_status(job_id,
                                                           "FAILED",
                                                           JOB_TABLE_NAME,
                                                           LOG_FIREHOSE_STREAM_NAME,
                                                           user=user,
                                                           event_type=event_type,
                                                           error_msg=None)
        except Exception as e:
            log(job_id, user, event_type, "[DLQ_PROCESSOR] Updating job status failed", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        else:
            log(job_id, user, event_type, f"[DLQ_PROCESSOR] Done updating job status to FAILED. Success = {update_success}, msg = {update_msg}", LOG_FIREHOSE_STREAM_NAME)

        # 4. Unlock the infrastructure
        try:
            release_success, release_msg = release_lock(job_id,
                                                        LOCK_TABLE_NAME,
                                                        LOG_FIREHOSE_STREAM_NAME,
                                                        user=user,
                                                        event_type=event_type)
        except Exception as e:
            log(job_id, user, event_type, "[DLQ_PROCESSOR] Release lock failed", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        else:
            log(job_id, user, event_type, f"[DLQ_PROCESSOR] Done release lock attempt. Success = {release_success}, msg = {release_msg}", LOG_FIREHOSE_STREAM_NAME)

        # 5. Delete the CTAS table if it exists, might have already been done earlier.
        try:
            drop_ctas_table_if_exists(job_id)
        except Exception as e:
            pass
        else:
            log(job_id, user, event_type, f"[DLQ_PROCESSOR] Deleted the CTAS table for image-upload dedup task.", LOG_FIREHOSE_STREAM_NAME)

        if update_success and release_success and s3_delete_success and iceberg_job_rows_delete_success:
            num_processed_successfully += 1

    return {"status": "ok", "total_records": total_records, "successfully_processed_records": num_processed_successfully}