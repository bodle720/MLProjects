import os
import json
from typing import Any

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import delete_s3_prefix
from common.general_utils.ddb_utils import update_job_status, release_lock
from common.dataset_utils.dlq_helpers import (
    rollback_failed_create_or_update,
    finish_failed_delete,
)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
DATASETS_BUCKET_NAME = os.environ["DATASETS_BUCKET_NAME"]
DATASETS_TABLE_NAME = os.environ["DATASETS_TABLE_NAME"]
DATASET_VERSIONS_TABLE_NAME = os.environ["DATASET_VERSIONS_TABLE_NAME"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]

TASK_NAME = "[DATASET_DLQ_PROCESSOR]"
_VALID_SOURCES = {"stepfunctions", "kickoff", "lambda"}
_VALID_POLICIES = {
    "rollback_new_version",
    "complete_delete",
    "finalize_success",
    "kickoff_only",
}

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

def _resolve_effective_policy(
    *,
    source: str | None,
    task_type: str | None,
    failed_stage: str | None,
    dlq_policy: str | None,
) -> str:
    """
    Resolve the effective DLQ policy.

    Prefers the new explicit dlq_policy field, but falls back to source/task/stage
    for compatibility with older messages.
    """
    if source == "kickoff":
        return "kickoff_only"

    if dlq_policy in _VALID_POLICIES:
        return dlq_policy

    # Backward-compatible fallback behavior
    if failed_stage == "cleanup_task":
        return "finalize_success"

    if task_type in {"create_dataset", "update_dataset"}:
        return "rollback_new_version"

    if task_type == "delete_dataset":
        return "complete_delete"

    return "kickoff_only"

def _run_durable_policy(
    *,
    effective_policy: str,
    task_type: str | None,
    dataset_context: Any,
) -> dict[str, Any] | None:
    """
    Execute durable cleanup / rollback behavior based on effective policy.

    Returns a summary dict or None if no durable action was required.
    Raises only if the called helper raises unexpectedly; helper-level failures
    are normally captured inside the returned summary["errors"].
    """
    if effective_policy == "rollback_new_version":
        return rollback_failed_create_or_update(
            task_name=TASK_NAME,
            task_type=task_type or "",
            dataset_context=dataset_context,
            datasets_bucket_name=DATASETS_BUCKET_NAME,
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            athena_output_s3_uri=ATHENA_OUTPUT_S3,
            athena_workgroup=ATHENA_WORKGROUP,
            datasets_table_name=DATASETS_TABLE_NAME,
            dataset_versions_table_name=DATASET_VERSIONS_TABLE_NAME,
        )

    if effective_policy == "complete_delete":
        return finish_failed_delete(
            task_name=TASK_NAME,
            dataset_context=dataset_context,
            datasets_bucket_name=DATASETS_BUCKET_NAME,
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            athena_output_s3_uri=ATHENA_OUTPUT_S3,
            athena_workgroup=ATHENA_WORKGROUP,
            datasets_table_name=DATASETS_TABLE_NAME,
            dataset_versions_table_name=DATASET_VERSIONS_TABLE_NAME,
        )

    if effective_policy in {"finalize_success", "kickoff_only"}:
        return None

    raise ValueError(f"Unsupported effective_policy: {effective_policy!r}")

def _cleanup_temp_prefix(
    *,
    job_id: str,
    user: str,
    event_type: str,
) -> tuple[bool, str]:
    prefix = f"temp/dataset-ops/{job_id}/"

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
        return True, ""
    except Exception as e:
        msg = f"Temp S3 cleanup failed for prefix {prefix}: {type(e).__name__}: {e}"
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} {msg}",
            level="error",
        )
        return False, msg

def _release_global_lock(
    *,
    job_id: str,
    user: str,
    event_type: str,
) -> tuple[bool, str]:
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
        release_msg = f"exception:{type(e).__name__}:{e}"
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

    lock_ok = release_success or str(release_msg).startswith("lock_not_held_by_job_id:")
    if str(release_msg).startswith("lock_not_held_by_job_id:"):
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Lock is already not held by this job: {release_msg}",
            level="warning",
        )

    return lock_ok, str(release_msg)

def _update_job_terminal_status(
    *,
    job_id: str,
    user: str,
    event_type: str,
    status: str,
    error_msg: str | None,
) -> tuple[bool, str]:
    update_success, update_msg = update_job_status(
        job_id,
        status,
        JOB_TABLE_NAME,
        LOG_FIREHOSE_STREAM_NAME,
        user=user,
        event_type=event_type,
        error_msg=error_msg,
    )

    level = "info" if update_success else "error"
    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Job status update to {status} finished. success={update_success}, msg={update_msg}",
        level=level,
    )

    return update_success, str(update_msg)

def handler(event, context):
    total_records = 0
    successfully_processed_records = 0
    skipped_records = 0
    record_summaries: list[dict[str, Any]] = []
    fatal_errors: list[str] = []

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
        task_type = _optional_string(parsed.get("task_type"))
        dataset_context = parsed.get("dataset_context")
        failed_stage = _optional_string(parsed.get("failed_stage"))
        dlq_policy = _optional_string(parsed.get("dlq_policy"))
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

        effective_policy = _resolve_effective_policy(
            source=source,
            task_type=task_type,
            failed_stage=failed_stage,
            dlq_policy=dlq_policy,
        )

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            (
                f"{TASK_NAME} Processing DLQ message from source={source}, "
                f"task_type={task_type!r}, failed_stage={failed_stage!r}, "
                f"dlq_policy={dlq_policy!r}, effective_policy={effective_policy!r}"
            ),
            level="info",
        )

        errors: list[str] = []
        durable_cleanup_summary: dict[str, Any] | None = None

        # ---------------------------------------------------------
        # 1) Durable cleanup / rollback based on policy
        # ---------------------------------------------------------
        try:
            durable_cleanup_summary = _run_durable_policy(
                effective_policy=effective_policy,
                task_type=task_type,
                dataset_context=dataset_context,
            )
        except Exception as e:
            msg = f"Durable cleanup dispatcher failed: {type(e).__name__}: {e}"
            errors.append(msg)
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} {msg}",
                level="error",
            )

        if durable_cleanup_summary is not None and not durable_cleanup_summary.get("ok", False):
            for msg in durable_cleanup_summary.get("errors", []):
                log(
                    job_id,
                    user,
                    event_type,
                    LOG_FIREHOSE_STREAM_NAME,
                    f"{TASK_NAME} Durable cleanup issue: {msg}",
                    level="error",
                )
            errors.extend(durable_cleanup_summary.get("errors", []))

        # ---------------------------------------------------------
        # 2) Delete temp folder for this job
        # ---------------------------------------------------------
        temp_cleanup_success, temp_cleanup_msg = _cleanup_temp_prefix(
            job_id=job_id,
            user=user,
            event_type=event_type,
        )
        if not temp_cleanup_success:
            errors.append(temp_cleanup_msg)

        # ---------------------------------------------------------
        # 3) Terminal handling depends on effective policy
        # ---------------------------------------------------------
        if effective_policy == "finalize_success":
            # For cleanup-task failures, preserve durable dataset state.
            # Finalize success means:
            #   temp cleanup -> release lock -> mark COMPLETED
            lock_ok, release_msg = _release_global_lock(
                job_id=job_id,
                user=user,
                event_type=event_type,
            )
            if not lock_ok:
                errors.append(f"Lock failed to release: {release_msg}")

            update_success, update_msg = _update_job_terminal_status(
                job_id=job_id,
                user=user,
                event_type=event_type,
                status="COMPLETED",
                error_msg=None,
            )
            if not update_success:
                errors.append(f"Failed to set job COMPLETED: {update_msg}")

        else:
            # All other policies end in FAILED:
            #   temp cleanup -> mark FAILED -> release lock
            update_success, update_msg = _update_job_terminal_status(
                job_id=job_id,
                user=user,
                event_type=event_type,
                status="FAILED",
                error_msg=(error_msg or "")[:512],
            )
            if not update_success:
                errors.append(f"Failed to set job FAILED: {update_msg}")

            lock_ok, release_msg = _release_global_lock(
                job_id=job_id,
                user=user,
                event_type=event_type,
            )
            if not lock_ok:
                errors.append(f"Lock failed to release: {release_msg}")

        record_summary = {
            "job_id": job_id,
            "source": source,
            "task_type": task_type,
            "failed_stage": failed_stage,
            "dlq_policy": dlq_policy,
            "effective_policy": effective_policy,
            "durable_cleanup_summary": durable_cleanup_summary,
            "temp_cleanup_success": temp_cleanup_success,
            "errors": errors,
        }
        record_summaries.append(record_summary)

        if errors:
            fatal_errors.append(f"job_id={job_id}: " + " | ".join(errors))
        else:
            successfully_processed_records += 1

    if fatal_errors:
        raise RuntimeError(
            f"{TASK_NAME} one or more DLQ records failed cleanup: {' || '.join(fatal_errors)}"
        )

    return {
        "status": "ok",
        "total_records": total_records,
        "successfully_processed_records": successfully_processed_records,
        "skipped_records": skipped_records,
        "record_summaries": record_summaries,
    }