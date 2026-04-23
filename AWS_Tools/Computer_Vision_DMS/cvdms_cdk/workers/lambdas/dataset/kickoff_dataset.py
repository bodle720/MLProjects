import os
import json
from typing import Tuple
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import s3_read_json

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]
DATASET_STATE_MACHINE_ARN = os.environ["DATASET_STATE_MACHINE_ARN"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
DATASET_DLQ_URL = os.environ["DATASET_DLQ_URL"]

ALLOWED_TASK_TYPES = {
    "create_dataset",
    "update_dataset",
    "delete_dataset",
}

TASK_NAME = "[DATASET_KICKOFF]"

sf = boto3.client("stepfunctions")
sqs = boto3.client("sqs")
ddb = boto3.client("dynamodb")


def assert_lock_held_by_job(job_id: str) -> Tuple[bool, str]:
    try:
        resp = ddb.get_item(
            TableName=LOCK_TABLE_NAME,
            Key={"lock_id": {"S": "global"}},
            ConsistentRead=True,
        )
        item = resp.get("Item")
        if not item:
            return False, "lock_item_missing"

        locked_attr = item.get("locked")
        locked_by_attr = item.get("locked_by")

        locked = False
        if locked_attr and "BOOL" in locked_attr:
            locked = bool(locked_attr["BOOL"])

        locked_by = ""
        if locked_by_attr and "S" in locked_by_attr:
            locked_by = locked_by_attr["S"] or ""

        if not locked:
            return False, "lock_not_held"

        if locked_by != job_id:
            return False, f"lock_held_by_other_job:{locked_by}"

        return True, ""
    except Exception as e:
        return False, f"ddb_get_item_error:{type(e).__name__}:{e}"


def send_to_dlq(job_id: str, user: str, event_type: str, error: str) -> Tuple[bool, str]:
    job_id = job_id or "unknown"
    user = user or "unknown"
    event_type = event_type or "DATASET_OP"

    try:
        sqs.send_message(
            QueueUrl=DATASET_DLQ_URL,
            MessageBody=json.dumps({
                "source": "kickoff",
                "job_id": job_id,
                "user": user,
                "event_type": event_type,
                "error": str(error),
            }),
        )
        return True, ""
    except Exception as e:
        msg = f"{TASK_NAME} Failed to send to DLQ: original_error={error}, exception={e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, msg, level="error")
        return False, msg


def fail(job_id: str, user: str, event_type: str, msg: str) -> dict:
    job_id = job_id or "unknown"
    user = user or "unknown"
    event_type = event_type or "DATASET_OP"

    ok, dlq_err = send_to_dlq(job_id, user, event_type, msg)
    if not ok:
        # Raise so the SQS-triggered Lambda invocation fails and the message is retried.
        raise RuntimeError(
            f"{TASK_NAME} Kickoff failed and DLQ send also failed. "
            f"original_error={msg}; dlq_error={dlq_err}"
        )

    return {
        "status": "failed",
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
    }


def handler(event, context):
    job_id = "unknown"
    user = "unknown"
    event_type = "DATASET_OP"

    records = event.get("Records", [])
    if not records:
        return fail(job_id, user, event_type, f"{TASK_NAME} No Records in event")

    sqs_rec = records[0]
    body = sqs_rec.get("body")
    if not body:
        return fail(job_id, user, event_type, f"{TASK_NAME} SQS record missing body")

    try:
        body_json = json.loads(body)
    except Exception as e:
        return fail(job_id, user, event_type, f"{TASK_NAME} Failed to parse SQS body as JSON: {e}")

    s3_records = body_json.get("Records", [])
    if not s3_records:
        return fail(job_id, user, event_type, f"{TASK_NAME} No S3 Records inside SQS body")

    s3_rec = s3_records[0]
    s3_info = s3_rec.get("s3", {})
    bucket = s3_info.get("bucket", {}).get("name")
    key = unquote_plus(s3_info.get("object", {}).get("key", ""))

    if not key.endswith("submission.json"):
        return fail(job_id, user, event_type, f"{TASK_NAME} Unexpected key: {key}")

    parts = key.split("/")
    if len(parts) >= 3 and parts[0] == "temp" and parts[1] == "dataset-ops":
        job_id = parts[2]

    if bucket != FILE_BUCKET_NAME:
        return fail(
            job_id,
            user,
            event_type,
            f"{TASK_NAME} Bucket mismatch: got {bucket}, expected {FILE_BUCKET_NAME}",
        )

    try:
        submission = s3_read_json(bucket, key, TASK_NAME)
        job_id = submission["job_id"]
        user = submission["user"]
        event_type = submission.get("event_type", "DATASET_OP")
        task_type = submission["task_type"]
        request = submission["request"]
        submission_s3_uri = submission["submission_s3_uri"]
        dataset_context = submission["dataset_context"]
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Could not initialize expected fields from submission.json: {e}",
            level="error",
        )
        return fail(job_id, user, event_type, f"{TASK_NAME} Invalid submission.json payload: {e}")

    if event_type != "DATASET_OP":
        return fail(job_id, user, event_type, f"{TASK_NAME} Invalid event_type: {event_type}")

    if not isinstance(task_type, str) or task_type not in ALLOWED_TASK_TYPES:
        return fail(job_id, user, event_type, f"{TASK_NAME} Invalid task_type: {task_type}")

    if not isinstance(request, dict):
        return fail(job_id, user, event_type, f"{TASK_NAME} request must be an object")

    if not isinstance(dataset_context, dict):
        return fail(job_id, user, event_type, f"{TASK_NAME} dataset_context must be an object")

    expected_submission_s3_uri = f"s3://{FILE_BUCKET_NAME}/temp/dataset-ops/{job_id}/submission.json"
    if submission_s3_uri != expected_submission_s3_uri:
        return fail(
            job_id,
            user,
            event_type,
            f"{TASK_NAME} submission_s3_uri mismatch: got {submission_s3_uri}, expected {expected_submission_s3_uri}",
        )

    ok, reason = assert_lock_held_by_job(job_id)
    if not ok:
        msg = f"{TASK_NAME} Lock mismatch for job_id={job_id}: {reason}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, msg, level="error")
        return fail(job_id, user, event_type, msg)

    # Deterministic execution name = idempotent per job.
    execution_name = job_id[:80]

    try:
        response = sf.start_execution(
            stateMachineArn=DATASET_STATE_MACHINE_ARN,
            name=execution_name,
            input=json.dumps({
                "job_id": job_id,
                "user": user,
                "event_type": event_type,
                "task_type": task_type,
                "submission_s3_uri": submission_s3_uri,
                "request": request,
                "dataset_context": dataset_context,
            }),
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ExecutionAlreadyExists":
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Execution already exists for job_id={job_id}; treating as duplicate kickoff and returning success.",
                level="warning",
            )
            return {
                "status": "ok",
                "job_id": job_id,
                "user": user,
                "event_type": event_type,
                "task_type": task_type,
                "submission_s3_uri": submission_s3_uri,
                "duplicate_start_suppressed": True,
            }

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Error starting dataset state machine: {e}",
            level="error",
        )
        return fail(job_id, user, event_type, f"{TASK_NAME} Failed to start dataset state machine: {e}")

    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Error starting dataset state machine: {e}",
            level="error",
        )
        return fail(job_id, user, event_type, f"{TASK_NAME} Failed to start dataset state machine: {e}")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Started dataset state machine execution {response['executionArn']}",
        level="info",
    )

    return {
        "status": "ok",
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "task_type": task_type,
        "submission_s3_uri": submission_s3_uri,
    }