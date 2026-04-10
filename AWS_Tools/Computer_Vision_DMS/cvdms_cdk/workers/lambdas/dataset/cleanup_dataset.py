import os

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import delete_s3_prefix
from common.general_utils.ddb_utils import update_job_status, release_lock

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]

TASK_NAME = "[DATASET_CLEANUP]"

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key in the cleanup lambda: {e}, event = {event}")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Starting cleanup lambda for job {job_id}")

    # ---------------------------------------------------------
    # 1. Delete S3 temp files (safe final cleanup)
    # ---------------------------------------------------------
    prefix = f"temp/dataset-ops/{job_id}/"

    delete_s3_prefix(FILE_BUCKET_NAME, prefix, TASK_NAME)

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Deleted S3 temp files under prefix {prefix}")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Cleanup completed successfully for job {job_id}")

    # ---------------------------------------------------------
    # 2. Release infrastructure lock
    # ---------------------------------------------------------
    try:
        release_success, release_msg = release_lock(
            job_id,
            LOCK_TABLE_NAME,
            LOG_FIREHOSE_STREAM_NAME,
            user=user,
            event_type=event_type
        )
    except Exception as e:
        release_success = False
        release_msg = f"exception:{e}"
        log(
            job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Lock release threw exception: {e}",
            level="error"
        )

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Lock release attempt complete. Success={release_success}, msg={release_msg}")

    if not release_success:
        if str(release_msg).startswith("lock_not_held_by_job_id:"):
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Proceeding with cleanup because lock is already not held by this job: {release_msg}",
                level="warning")
        else:
            raise RuntimeError(f"{TASK_NAME} Failed to release lock: {release_msg}")

    # ---------------------------------------------------------
    # 3. Mark job as COMPLETED
    # ---------------------------------------------------------
    update_success, update_msg = update_job_status(
        job_id,
        "COMPLETED",
        JOB_TABLE_NAME,
        LOG_FIREHOSE_STREAM_NAME,
        user=user,
        event_type=event_type,
        error_msg=None
    )

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Job status update to COMPLETED finished. Success={update_success}, msg={update_msg}")

    if not update_success:
        raise RuntimeError(f"{TASK_NAME} Failed to update job status: {update_msg}")

    return {
        "job_id": job_id,
        "cleanup_done": True
    }