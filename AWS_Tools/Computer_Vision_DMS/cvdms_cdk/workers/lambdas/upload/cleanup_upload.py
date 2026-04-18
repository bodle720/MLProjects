import os
from typing import List, Dict

import boto3
from botocore.exceptions import ClientError

from common.general_utils.logging_utils import log
from common.general_utils.ddb_utils import update_job_status, release_lock

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]

TASK_NAME = "[UPLOAD_CLEANUP]"

s3 = boto3.client("s3")

def _s3_object_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def _list_temp_job_keys(job_id: str) -> List[str]:
    prefix = f"temp/image-upload/{job_id}/"
    paginator = s3.get_paginator("list_objects_v2")

    keys: List[str] = []
    for page in paginator.paginate(Bucket=FILE_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if isinstance(key, str) and key and not key.endswith("/"):
                keys.append(key)

    keys.sort()
    return keys

def _chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _delete_s3_keys_strict(bucket: str, keys: List[str], batch_size: int = 1000) -> Dict[str, object]:
    unique_keys = sorted({k.strip() for k in keys if isinstance(k, str) and k.strip()})
    if not unique_keys:
        return {
            "attempted": 0,
            "api_error_count": 0,
            "api_error_samples": [],
            "survivor_count": 0,
            "survivor_samples": [],
            "verify_error_count": 0,
            "verify_error_samples": [],
        }

    api_error_samples: List[str] = []

    for chunk in _chunked(unique_keys, batch_size):
        try:
            resp = s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in chunk]},
            )
            for err in resp.get("Errors", []) or []:
                key = str(err.get("Key", ""))
                code = str(err.get("Code", ""))
                msg = str(err.get("Message", ""))
                api_error_samples.append(f"{key}:{code}:{msg}")
        except Exception as e:
            sample = chunk[:10]
            api_error_samples.extend([f"{k}:delete_exception:{e}" for k in sample])

    survivors: List[str] = []
    verify_error_samples: List[str] = []

    for key in unique_keys:
        try:
            if _s3_object_exists(bucket, key):
                survivors.append(key)
        except Exception as e:
            verify_error_samples.append(f"{key}:verify_exception:{e}")

    return {
        "attempted": len(unique_keys),
        "api_error_count": len(api_error_samples),
        "api_error_samples": api_error_samples[:10],
        "survivor_count": len(survivors),
        "survivor_samples": survivors[:10],
        "verify_error_count": len(verify_error_samples),
        "verify_error_samples": verify_error_samples[:10],
    }

def _cleanup_temp_prefix_best_effort(job_id: str, user: str, event_type: str) -> None:
    """
    Best-effort cleanup of temp/image-upload/<job_id>/.
    This must never raise, because the job itself already succeeded.
    """
    prefix = f"temp/image-upload/{job_id}/"

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Starting best-effort temp cleanup for prefix s3://{FILE_BUCKET_NAME}/{prefix}",
    )

    try:
        temp_keys = _list_temp_job_keys(job_id)
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Temp cleanup listing failed for prefix {prefix}: {e}",
            level="warning",
        )
        return

    if not temp_keys:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} No temp files found under prefix {prefix}",
        )
        return

    try:
        result = _delete_s3_keys_strict(FILE_BUCKET_NAME, temp_keys)
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            (
                f"{TASK_NAME} Temp cleanup result for prefix {prefix}: "
                f"listed={len(temp_keys)} "
                f"attempted={result['attempted']} "
                f"api_error_count={result['api_error_count']} "
                f"survivor_count={result['survivor_count']} "
                f"verify_error_count={result['verify_error_count']} "
                f"api_error_samples={result['api_error_samples']} "
                f"survivor_samples={result['survivor_samples']} "
                f"verify_error_samples={result['verify_error_samples']}"
            ),
            level=(
                "warning"
                if (
                    result["api_error_count"]
                    or result["survivor_count"]
                    or result["verify_error_count"]
                )
                else "info"
            ),
        )
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Temp cleanup failed for prefix {prefix}: {e}",
            level="warning",
        )

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key in cleanup lambda: {e}, event={event}")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Starting cleanup lambda for job {job_id}",
    )

    # ---------------------------------------------------------
    # 1. Delete S3 temp files (best-effort final cleanup)
    # ---------------------------------------------------------
    _cleanup_temp_prefix_best_effort(job_id, user, event_type)

    # ---------------------------------------------------------
    # 2. Release infrastructure lock
    # ---------------------------------------------------------
    release_success, release_msg = release_lock(
        job_id,
        LOCK_TABLE_NAME,
        LOG_FIREHOSE_STREAM_NAME,
        user=user,
        event_type=event_type,
    )

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Lock release attempt complete. Success={release_success}, msg={release_msg}",
    )

    if not release_success:
        if str(release_msg).startswith("lock_not_held_by_job_id:"):
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Proceeding because lock is already not held by this job: {release_msg}",
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
        f"{TASK_NAME} Job status update to COMPLETED finished. Success={update_success}, msg={update_msg}",
    )

    if not update_success:
        raise RuntimeError(f"{TASK_NAME} Failed to update job status: {update_msg}")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Cleanup completed successfully for job {job_id}",
    )

    return {
        "job_id": job_id,
        "cleanup_done": True,
    }