import logging
import boto3
from botocore.exceptions import ClientError

from common.logging_utils import log

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")

def update_job_status(job_id,
                      status,
                      job_table_name,
                      stream_name,
                      user = 'unknown',
                      event_type = 'unknown',
                      error_msg = None):

    valid_statuses = ['PENDING', 'IN_PROGRESS', 'FAILED', 'COMPLETED']
    if status not in valid_statuses:
        log(job_id, user, event_type, "Job status update failed.", stream_name, error=f"Failed to update job status because status {status} is invalid.", level="error")
        return False, f"invalid status: {status}"

    try:
        job_table = dynamodb.Table(job_table_name)
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

def release_lock(job_id,
                 lock_table_name,
                 stream_name,
                 user='unknown',
                 event_type='unknown'
                 ):
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
        error_msg = f"Failed releasing lock for job id {job_id}: {e}"
        log(job_id, user, event_type, error_msg, stream_name, error=error_msg, level="error")
        if code == "ConditionalCheckFailedException":
            return False, f"lock_not_held_by_job_id: {job_id}"
        return False, f"dynamodb_error: {e}"