import os

from common.logging_utils import log
from common.s3_utils import delete_s3_prefix
from common.ddb_utils import update_job_status, release_lock

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]

TASK_NAME = "[CLEANUP]"

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key in the cleanup lambda: {e}, event = {event}")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Starting cleanup lambda for job {job_id}")

    # 1. Delete S3 temp files
    prefix = f"temp/image-upload/{job_id}/"
    delete_s3_prefix(FILE_BUCKET_NAME, prefix, TASK_NAME)
    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Done deleting s3 temp files in cleanup lambda")

    # 2. make sure job status table is marked COMPLETED
    update_success, update_msg = update_job_status(job_id,
                                                  "COMPLETED",
                                                  JOB_TABLE_NAME,
                                                  LOG_FIREHOSE_STREAM_NAME,
                                                  user=user,
                                                  event_type=event_type,
                                                  error_msg=None)
    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Done updating job status to COMPLETED. Success = {update_success}, msg = {update_msg}")

    if not update_success:
        raise RuntimeError(f"{TASK_NAME} Failed to update job status: {update_msg}")

    # 3. Unlock the infrastructure
    release_success, release_msg = release_lock(job_id,
                                                 LOCK_TABLE_NAME,
                                                 LOG_FIREHOSE_STREAM_NAME,
                                                 user=user,
                                                 event_type=event_type)

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Done release lock attempt. Success = {release_success}, msg = {release_msg}")

    if not release_success:
        raise RuntimeError(f"{TASK_NAME} Failed to release lock: {release_msg}")

    return {
        "job_id": job_id,
        "cleanup_done": True
    }