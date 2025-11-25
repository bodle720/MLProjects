import json
import logging
from datetime import datetime, timezone
import boto3
from botocore.exceptions import ClientError
from typing import Tuple
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

firehose = boto3.client("firehose")
dynamodb = boto3.resource("dynamodb")
athena = boto3.client("athena")
s3 = boto3.client("s3")

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

def _wait_for_query(client, query_execution_id, poll_interval=5, timeout_seconds=1800):
    """Poll Athena until query finishes; returns final state and metadata."""
    start = time.time()
    while True:
        resp = client.get_query_execution(QueryExecutionId=query_execution_id)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return {"state": state, "metadata": resp}
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Athena query {query_execution_id} timed out after {timeout_seconds}s")
        time.sleep(poll_interval)

def delete_iceberg_partition_rows(job_id: str,
                                    iceberg_db_name,
                                    image_upload_staging_table_name,
                                    athena_output_s3,
                                    athena_workgroup,
                                    poll_interval: int = 5,
                                    timeout_seconds: int = 1800,
                                    run_compaction: bool = True):
    """
    Delete all rows for a given job_id from an Iceberg table and optionally compact.
    Returns a dict with query ids and final states for DELETE and OPTIMIZE.
    """
    # Escape single quotes in job_id for SQL literal safety
    safe_job_id = job_id.replace("'", "''")
    full_table = f"{iceberg_db_name}.{image_upload_staging_table_name}"

    # 1) DELETE statement (Iceberg positional delete files)
    delete_sql = f"DELETE FROM {full_table} WHERE job_id = '{safe_job_id}'"
    delete_resp = athena.start_query_execution(
        QueryString=delete_sql,
        ResultConfiguration={"OutputLocation": athena_output_s3},
        WorkGroup=athena_workgroup
    )
    delete_qid = delete_resp["QueryExecutionId"]
    delete_result = _wait_for_query(athena, delete_qid, poll_interval=poll_interval, timeout_seconds=timeout_seconds)

    result = {
        "delete_query_id": delete_qid,
        "delete_state": delete_result["state"]
    }

    # 2) Optional: compact / rewrite data for that partition to remove position deletes
    #    Use OPTIMIZE ... REWRITE DATA USING BIN_PACK WHERE job_id = '...'
    #    (WHERE may only reference partition columns; job_id is partitioned in your table)
    if run_compaction and delete_result["state"] == "SUCCEEDED":
        optimize_sql = f"OPTIMIZE {full_table} REWRITE DATA USING BIN_PACK WHERE job_id = '{safe_job_id}'"
        opt_resp = athena.start_query_execution(
            QueryString=optimize_sql,
            ResultConfiguration={"OutputLocation": athena_output_s3},
            WorkGroup=athena_workgroup
        )
        opt_qid = opt_resp["QueryExecutionId"]
        opt_result = _wait_for_query(athena, opt_qid)
        result.update({
            "optimize_query_id": opt_qid,
            "optimize_state": opt_result["state"]
        })

    return result

def delete_s3_prefix(bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=obj["Key"])