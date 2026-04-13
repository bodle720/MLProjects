import os
import json
from typing import Any

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import delete_s3_prefix
from common.general_utils.ddb_utils import update_job_status, release_lock

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]

TASK_NAME = "[DATASET_DLQ_PROCESSOR]"
_VALID_SOURCES = {"stepfunctions", "kickoff", "lambda"}

def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def _extract_error_message(error_value: Any) -> str:
    """
    Normalize the DLQ error payload into a readable string.

    Handles:
    - plain strings
    - Step Functions-style JSON strings with nested Cause
    - dict-like payloads
    """
    if error_value is None:
        return ""

    # Already structured
    if isinstance(error_value, dict):
        cause = error_value.get("Cause")
        if isinstance(cause, str):
            try:
                cause_obj = json.loads(cause)
                if isinstance(cause_obj, dict):
                    return str(
                        cause_obj.get("errorMessage")
                        or cause_obj.get("cause")
                        or cause_obj.get("error")
                        or error_value
                    )
            except Exception:
                pass
        return str(
            error_value.get("errorMessage")
            or error_value.get("cause")
            or error_value.get("error")
            or error_value
        )

    # String payload
    text = str(error_value)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            cause = obj.get("Cause")
            if isinstance(cause, str):
                try:
                    cause_obj = json.loads(cause)
                    if isinstance(cause_obj, dict):
                        return str(
                            cause_obj.get("errorMessage")
                            or cause_obj.get("cause")
                            or cause_obj.get("error")
                            or text
                        )
                except Exception:
                    pass

            return str(
                obj.get("errorMessage")
                or obj.get("cause")
                or obj.get("error")
                or text
            )
    except Exception:
        pass

    return text

def _parse_record_body(record: dict[str, Any]) -> dict[str, Any] | None:
    body = record.get("body")
    if not isinstance(body, str) or not body.strip():
        return None

    try:
        parsed = json.loads(body)
    except Exception:
        return None

    return parsed if isinstance(parsed, dict) else None

def handler(event, context):
    total_records = 0
    successfully_processed_records = 0
    skipped_records = 0

    for record in event.get("Records", []):
        total_records += 1

        parsed = _parse_record_body(record)
        if parsed is None:
            print(f"{TASK_NAME} Skipping record with invalid or non-JSON body.")
            skipped_records += 1
            continue

        source = _optional_string(parsed.get("source"))
        job_id = _optional_string(parsed.get("job_id"))
        user = _optional_string(parsed.get("user"))
        event_type = _optional_string(parsed.get("event_type")) or "DATASET_OP"
        raw_error = parsed.get("error")
        error_msg = _extract_error_message(raw_error)

        if source not in _VALID_SOURCES:
            print(f"{TASK_NAME} Skipping unknown source={source!r}. body={parsed}")
            skipped_records += 1
            continue

        if job_id in (None, "unknown") or user is None:
            print(f"{TASK_NAME} Ignoring non-job or incomplete DLQ message: {parsed}")
            skipped_records += 1
            continue

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Processing DLQ message from source={source}",
            level="info",
        )

        # 1) Delete temp folder for this job
        prefix = f"temp/dataset-ops/{job_id}/"
        s3_cleanup_success = True
        try:
            delete_s3_prefix(FILE_BUCKET_NAME, prefix, TASK_NAME)
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Deleted temp S3 prefix {prefix}",
                level="info",
            )
        except Exception as e:
            s3_cleanup_success = False
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Temp S3 cleanup failed for prefix {prefix}: {e}",
                level="error",
            )

        # 2) Mark job FAILED
        update_success = False
        try:
            update_success, update_msg = update_job_status(
                job_id,
                "FAILED",
                JOB_TABLE_NAME,
                LOG_FIREHOSE_STREAM_NAME,
                user=user,
                event_type=event_type,
                error_msg=(error_msg or "")[:512],
            )
            if update_success:
                log(
                    job_id,
                    user,
                    event_type,
                    LOG_FIREHOSE_STREAM_NAME,
                    f"{TASK_NAME} Updated job status to FAILED.",
                    level="info",
                )
            else:
                log(
                    job_id,
                    user,
                    event_type,
                    LOG_FIREHOSE_STREAM_NAME,
                    f"{TASK_NAME} Failed to set job FAILED: {update_msg}",
                    level="error",
                )
        except Exception as e:
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Updating job status to FAILED threw exception: {e}",
                level="error",
            )

        # 3) Release global lock
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
            f"{TASK_NAME} Lock release attempt complete. success={release_success}, msg={release_msg}",
            level="info",
        )

        if not release_success:
            if str(release_msg).startswith("lock_not_held_by_job_id:"):
                log(
                    job_id,
                    user,
                    event_type,
                    LOG_FIREHOSE_STREAM_NAME,
                    f"{TASK_NAME} Lock is already not held by this job: {release_msg}",
                    level="warning",
                )
            else:
                log(
                    job_id,
                    user,
                    event_type,
                    LOG_FIREHOSE_STREAM_NAME,
                    f"{TASK_NAME} Lock failed to release: {release_msg}",
                    level="error",
                )

        lock_ok = release_success or str(release_msg).startswith("lock_not_held_by_job_id:")
        if update_success and lock_ok and s3_cleanup_success:
            successfully_processed_records += 1

    return {
        "status": "ok",
        "total_records": total_records,
        "successfully_processed_records": successfully_processed_records,
        "skipped_records": skipped_records,
    }