import os
import json

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import delete_s3_prefix
from common.general_utils.ddb_utils import update_job_status, release_lock
from common.general_utils.table_schemas import (
    CANONICAL_BBOX_TABLE_NAME,
    CANONICAL_SEMANTIC_TABLE_NAME,
    CANONICAL_INSTANCE_TABLE_NAME
)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]

TASK_NAME = "[DATASET_DLQ_PROCESSOR]"

LABEL_TABLES = {
    CANONICAL_BBOX_TABLE_NAME,
    CANONICAL_SEMANTIC_TABLE_NAME,
    CANONICAL_INSTANCE_TABLE_NAME,
}

def handler(event, context):
    total_records = 0
    num_processed_successfully = 0

    for record in event.get("Records", []):
        total_records += 1
        update_success = False
        release_success = False

        try:
            body = json.loads(record["body"])
        except Exception:
            print(f"{TASK_NAME} Skipping non-JSON message")
            continue

        source = body.get("source")
        job_id = body.get("job_id")
        user = body.get("user")
        event_type = body.get("event_type")
        error_msg = body.get("error")

        try:
            error_obj = json.loads(error_msg)
            cause = error_obj.get("Cause")
            if cause:
                cause_obj = json.loads(cause)
                error_msg = cause_obj.get("errorMessage", error_msg)
        except Exception:
            pass

        if source not in ("stepfunctions", "kickoff", "lambda"):
            print(f"{TASK_NAME} Skipping unknown source={source}")
            continue

        if (job_id in (None, "unknown")) or (user is None) or (event_type is None):
            print(f"{TASK_NAME} Ignoring non-job DLQ message: {body}")
            continue

        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} DLQ received message: {body}")

        # 1) Delete temp folder for this job
        prefix = f"temp/dataset-ops/{job_id}/"
        try:
            delete_s3_prefix(FILE_BUCKET_NAME, prefix, TASK_NAME)
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Deleted temp s3 prefix")
        except Exception:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Temp S3 cleanup failed", level="error")

        # 2) Mark job FAILED
        try:
            update_success, update_msg = update_job_status(
                job_id,
                "FAILED",
                JOB_TABLE_NAME,
                LOG_FIREHOSE_STREAM_NAME,
                user=user,
                event_type=event_type,
                error_msg=(error_msg or "")[:512]
            )
            if update_success:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Updated job status to FAILED.")
            else:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Failed to set job FAILED: {update_msg}", level="error")
        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Updating job status FAILED failed: {e}", level="error")

        # 3) Release global lock
        try:
            release_success, release_msg = release_lock(job_id, LOCK_TABLE_NAME, LOG_FIREHOSE_STREAM_NAME, user=user, event_type=event_type)
            if release_success:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Released lock.")
            else:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Release lock failed: {release_msg}", level="error")
        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Release lock failed: {e}", level="error")

        if update_success and release_success:
            num_processed_successfully += 1

    return {
        "status": "ok",
        "total_records": total_records,
        "successfully_processed_records": num_processed_successfully,
    }