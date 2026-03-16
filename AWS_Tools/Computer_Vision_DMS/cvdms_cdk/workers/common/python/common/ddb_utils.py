import time
import boto3
from typing import Any, Dict, Optional, Literal
from botocore.exceptions import ClientError

from common.logging_utils import log

dynamodb = boto3.client("dynamodb")

DdbAttr = Dict[str, Any]
DdbItem = Dict[str, DdbAttr]

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
            remove_parts.append("error")

    update_expr = "SET " + ", ".join(update_parts)
    if remove_parts:
        update_expr += " REMOVE " + ", ".join(remove_parts)

    try:
        dynamodb.update_item(
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
        dynamodb.update_item(
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

def batch_get_dynamodb_items(table_name: str,
                             keys: list[str],
                             ddb_batch_get_max: int,
                             task_name: str) -> Dict[str, DdbItem]:
    if not isinstance(keys, list):
        raise TypeError(f"{task_name} keys must be a list[str], got {type(keys).__name__}")

    keys = list(set(keys))

    if not (1 <= ddb_batch_get_max <= 100):
        raise ValueError(f"{task_name} ddb_batch_get_max must be between 1 and 100 inclusive, got {ddb_batch_get_max}")

    results = {}

    for i in range(0, len(keys), ddb_batch_get_max):
        chunk = keys[i:i + ddb_batch_get_max]
        request_keys = [{"sha256": {"S": k}} for k in chunk]
        print(f"{task_name} requesting {len(request_keys)} keys from DynamoDB")
        request_items = {table_name: {"Keys": request_keys}}

        backoff = 1.0
        for attempt in range(30):
            try:
                resp = dynamodb.batch_get_item(RequestItems=request_items)

                for item in resp.get("Responses", {}).get(table_name, []):
                    sha = item.get("sha256", {}).get("S")
                    if sha:
                        results[sha] = item

                unprocessed = resp.get("UnprocessedKeys", {}).get(table_name, {}).get("Keys", [])
                print(f"{task_name} response count={len(resp.get('Responses', {}).get(table_name, []))}")
                print(f"{task_name} unprocessed count={len(unprocessed)}")

                if not unprocessed:
                    print(f"{task_name} breaking on attempt {attempt}")
                    break

                print(f"{task_name} retrying unprocessed keys: {len(unprocessed)} (attempt {attempt})")
                request_items = {table_name: {"Keys": unprocessed}}
                time.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                msg = e.response.get("Error", {}).get("Message", str(e))
                print(f"{task_name} batch_get_item ClientError code={code} message={msg}")

                retryable_codes = {
                    "ProvisionedThroughputExceededException",
                    "ThrottlingException",
                    "RequestLimitExceeded",
                    "InternalServerError",
                    "InternalServerException",
                }

                if code in retryable_codes:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 15.0)
                    continue

                raise

        else:
            raise RuntimeError(f"{task_name} DynamoDB batch_get_item exceeded retries for table {table_name}")

    return results