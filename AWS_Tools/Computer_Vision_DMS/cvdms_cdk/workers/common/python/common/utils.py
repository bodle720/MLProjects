import json
import logging
from datetime import datetime, timezone
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

firehose = boto3.client("firehose")
dynamodb = boto3.resource("dynamodb")

def log(job_id, user, event_type, message, stream_name, warning=None, error=None, level="info"):
    entry = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "message": message,
        "warning": warning,
        "error": error,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    # CloudWatch log for operational visibility
    line = json.dumps(entry)
    if level.lower() == "error":
        logger.error(line)
    else:
        logger.info(line)

    # Firehose DirectPut (JSON line)
    try:
        firehose.put_record(
            DeliveryStreamName=stream_name,
            Record={"Data": (line + "\n").encode("utf-8")}
        )
    except Exception as e:
        # Do not fail the handler—the design prefers non-DLQ behavior.
        # Optionally log the failure; avoid recursion by not calling log() again.
        logger.error(json.dumps({
            "job_id": job_id,
            "user": user,
            "event_type": event_type,
            "message": "Failed to put log to Firehose",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }))

def update_job_status(job_id,
                      status,
                      job_table,
                      stream_name,
                      user = 'unknown',
                      event_type = 'unknown',
                      error_msg = None):

    valid_statuses = ['PENDING', 'IN_PROGRESS', 'FAILED', 'COMPLETED']
    if status not in valid_statuses:
        log(job_id, user, event_type, "Job status update failed.", stream_name, error=f"Failed to update job status because status {status} is invalid.", level="error")
        return False, f"invalid status: {status}"

    try:
        job_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #s = :s, #e = :e",
            ExpressionAttributeNames={"#s": "status", "#e": "error"},
            ExpressionAttributeValues={":s": status, ":e": error_msg},
            ConditionExpression="attribute_exists(job_id)",
        )
        return True, ""
    except ClientError as e:
        log(job_id, user, event_type, "Job status update failed.", stream_name, error=f"Failed to update job status due to error: {e}", level="error")
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ConditionalCheckFailedException":
            return False, f"job not found: {job_id}"
        return False, str(e)

def release_lock(job_id, lock_table_name):
    """
    Release lock only if current locked_by matches job_id (the job id holding the lock).
    Returns (True, "") on success.
    """
    lock_id = "global"

    try:
        lock_table = dynamodb.Table(lock_table_name)
        lock_table.update_item(
            Key={"lock_id": lock_id},
            UpdateExpression="SET locked = :false REMOVE locked_by",
            ConditionExpression="locked_by = :holder",
            ExpressionAttributeValues={":false": False, ":holder": job_id},
            ReturnValues="ALL_NEW",
        )
        return True, ""
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        logger.error(f"Failed releasing lock for job id {job_id}: {e}")
        if code == "ConditionalCheckFailedException":
            return False, f"lock_not_held_by_expected_holder: {job_id}"
        return False, f"dynamodb_error: {e}"