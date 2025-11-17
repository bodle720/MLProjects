import json
import os
import logging
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]

firehose = boto3.client("firehose")
dynamodb = boto3.resource("dynamodb")

EVENT_TYPE = "TA_test"

def update_job_status(job_id,
                      status,
                      job_table,
                      error_msg=None):

    valid_statuses = ['PENDING', 'IN_PROGRESS', 'FAILED', 'COMPLETED']

    if status not in valid_statuses:
        return False, f"invalid status: {status}"

    try:
        job_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #s = :s, #e = :e",
            ExpressionAttributeNames={"#s": "status", "#e": "errors"},
            ExpressionAttributeValues={":s": status, ":e": error_msg},
            ConditionExpression="attribute_exists(job_id)",
        )
        return True, ""
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ConditionalCheckFailedException":
            return False, f"job not found: {job_id}"
        return False, str(e)

def log(job_id, user, message, warning=None, error=None, level="info"):
    entry = {
        "job_id": job_id,
        "user": user,
        "event_type": EVENT_TYPE,
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
            DeliveryStreamName=FIREHOSE_STREAM_NAME,
            Record={"Data": (line + "\n").encode("utf-8")}
        )
    except Exception as e:
        # Do not fail the handler—your design prefers non-DLQ behavior.
        # Optionally log the failure; avoid recursion by not calling log() again.
        logger.error(json.dumps({
            "job_id": job_id,
            "user": user,
            "event_type": EVENT_TYPE,
            "message": "Failed to put log to Firehose",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }))

def handler(event, context):
    job_table = dynamodb.Table(JOB_TABLE_NAME)

    try:
        # kickoff input
        job_id_ko = event['job_id']
        user_ko = event['user']
        label_types_ko =  event['label_types']

        # step1 result (preferred if you trust step1 for canonical values)
        step1 = event['step1']
        job_id_step1 = step1['job_id']
        user_step1 = step1['user']
        label_types_step1 = step1['label_types']
    except KeyError as e:
        raise Exception(f"Could not get needed keys: {e}")

    job_status_updated, job_msg = update_job_status(job_id_ko,
                                                    "COMPLETED",
                                                    job_table)

    if not job_status_updated:
        log(job_id_ko, user_ko, job_msg, error=job_msg, level="error")
        raise Exception(f"Could not set job status: {job_msg}")
    else:
        log(job_id_ko, user_ko, "Status of job set to COMPLETED in step function step 2")
        log(job_id_ko, user_ko, f"The user from kickoff is {user_ko}, the user from step 1 output is {user_step1}, state machine is done.")

    return {'statusCode': 200, 'job_id': job_id_ko, 'user': user_ko, 'label_types': label_types_ko}