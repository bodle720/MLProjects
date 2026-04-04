import boto3
from typing import Optional, Literal
from botocore.exceptions import ClientError

from common.general_utils.logging_utils import log

dynamodb_client = boto3.client("dynamodb")
ddb_resource = boto3.resource("dynamodb")

JobStatus = Literal["PENDING", "IN_PROGRESS", "FAILED", "COMPLETED"]

def update_job_status(job_id: str,
                      status: JobStatus,
                      job_table_name: str,
                      stream_name: str,
                      user: str = "unknown",
                      event_type: str = "unknown",
                      error_msg: Optional[str] = None) -> tuple[bool, str]:

    if not job_id:
        return False, "missing_job_id"
    if not job_table_name:
        return False, "missing_job_table_name"
    if not stream_name:
        return False, "missing_stream_name"

    if error_msg is not None:
        error_msg = str(error_msg)[:512]

    valid_statuses = {"PENDING", "IN_PROGRESS", "FAILED", "COMPLETED"}
    if status not in valid_statuses:
        log(job_id, user, event_type, stream_name,
            f"Failed to update job status: invalid status={status!r}",
            level="error")
        return False, f"invalid_status:{status}"

    expr_names = {"#s": "status"}
    expr_vals = {":s": {"S": status}}
    update_parts = ["#s = :s"]
    remove_parts = []

    if error_msg is not None:
        expr_names["#e"] = "error"
        expr_vals[":e"] = {"S": error_msg}
        update_parts.append("#e = :e")
    else:
        if status != "FAILED":
            expr_names["#e"] = "error"
            remove_parts.append("#e")

    update_expr = "SET " + ", ".join(update_parts)
    if remove_parts:
        update_expr += " REMOVE " + ", ".join(remove_parts)

    try:
        dynamodb_client.update_item(
            TableName=job_table_name,
            Key={"job_id": {"S": job_id}},
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_names,
            ExpressionAttributeValues=expr_vals,
            ConditionExpression="attribute_exists(job_id)",
        )
        return True, ""
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ConditionalCheckFailedException":
            msg = f"Job not found when updating status: job_id={job_id}"
            log(job_id, user, event_type, stream_name, msg, level="warning")
            return False, f"job_not_found:{job_id}"

        msg = f"Failed to update job status job_id={job_id} status={status}: {e}"
        log(job_id, user, event_type, stream_name, msg, level="error")
        return False, f"dynamodb_error:{code or 'unknown'}"

def release_lock(job_id: str,
                 lock_table_name: str,
                 stream_name: str,
                 user: str = 'unknown',
                 event_type: str = 'unknown') -> tuple[bool, str]:

    if not job_id:
        return False, "missing_job_id"
    if not lock_table_name:
        return False, "missing_lock_table_name"
    if not stream_name:
        return False, "missing_stream_name"

    lock_id = "global"
    try:
        dynamodb_client.update_item(
            TableName=lock_table_name,
            Key={"lock_id": {"S": lock_id}},
            UpdateExpression="SET locked = :false REMOVE locked_by",
            ConditionExpression="locked_by = :holder",
            ExpressionAttributeValues={
                ":false": {"BOOL": False},
                ":holder": {"S": job_id},
            },
            ReturnValues="ALL_NEW",
        )
        return True, ""
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ConditionalCheckFailedException":
            msg = f"Lock not held by job_id={job_id} (conditional check failed)"
            log(job_id, user, event_type, stream_name, msg, level="warning")
            return False, f"lock_not_held_by_job_id:{job_id}"

        msg = f"Failed releasing lock for job_id={job_id}: {e}"
        log(job_id, user, event_type, stream_name, msg, level="error")
        return False, f"dynamodb_error:{code or 'unknown'}"

def dataset_exists(dataset_id: str,
                   datasets_table_name) -> bool:

    datasets_table = ddb_resource.Table(datasets_table_name)
    resp = datasets_table.get_item(Key={"dataset_id": dataset_id}, ConsistentRead=True)
    return "Item" in resp