import os
from typing import Any

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import delete_s3_prefix
from common.general_utils.ddb_utils import update_job_status, release_lock

# testing function for dlq processor
from common.testing_utils.dataset_testing import maybe_fail

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]

TASK_NAME = "[DATASET_CLEANUP]"

def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""

    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text

def handler(event, context):
    try:
        job_id = _require_nonempty_string(event.get("job_id"), field_name="job_id")
        user = _require_nonempty_string(event.get("user"), field_name="user")
        event_type = _require_nonempty_string(event.get("event_type"), field_name="event_type")
    except Exception as e:
        raise RuntimeError(f"{TASK_NAME} Invalid cleanup event: {e}; event={event!r}")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Starting cleanup for job_id={job_id}",
        level="info",
    )

    maybe_fail("cleanup_fail")
    # ---------------------------------------------------------
    # 1. Delete S3 temp files (safe final cleanup)
    # ---------------------------------------------------------
    prefix = f"temp/dataset-ops/{job_id}/"

    delete_s3_prefix(FILE_BUCKET_NAME, prefix, TASK_NAME)

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Deleted temp S3 files under prefix {prefix}",
        level="info",
    )

    # ---------------------------------------------------------
    # 2. Release infrastructure lock
    # ---------------------------------------------------------
    try:
        release_success, release_msg = release_lock(
            job_id,
            LOCK_TABLE_NAME,
            LOG_FIREHOSE_STREAM_NAME,
            user=user,
            event_type=event_type,
        )
    except Exception as e:
        release_success = False
        release_msg = f"exception:{e}"
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Lock release threw exception: {e}",
            level="error",
        )

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Lock release attempt finished. success={release_success}, msg={release_msg}",
        level="info",
    )

    if not release_success:
        if str(release_msg).startswith("lock_not_held_by_job_id:"):
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                (
                    f"{TASK_NAME} Proceeding because lock is already not held by this job: "
                    f"{release_msg}"
                ),
                level="warning",
            )
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
        error_msg=None,
    )

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Job status update to COMPLETED finished. success={update_success}, msg={update_msg}",
        level="info",
    )

    if not update_success:
        raise RuntimeError(f"{TASK_NAME} Failed to update job status: {update_msg}")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Cleanup completed successfully for job_id={job_id}",
        level="info",
    )

    return {
        "job_id": job_id,
        "cleanup_done": True,
        "temp_prefix_deleted": True,
        "lock_released": True,
        "job_marked_completed": True,
    }