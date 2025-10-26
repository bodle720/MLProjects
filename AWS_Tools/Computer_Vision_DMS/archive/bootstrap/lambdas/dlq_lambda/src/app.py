# -*- coding: utf-8 -*-
"""
DLQ Lambda to handle all failed events from thelifecycle, image opt, and sync queue.

1. Lifecycle Queue receives events formatted as:
    
    {"event_type": "DELETE_DATASET",
     "dataset_id": "my-sample-dataset",
     "job_id": <job uuid>}
    
    and
    
    {"event_type": "REMOVE_CLASS",
     "dataset_id": "my-sample-dataset",
     "class_name": "cat",
     "job_id": <job uuid>}
    
2. Image Ops Queue receives events formatted as:
    
    {"event_type": "IMAGE_UPLOAD",
     "datasets": <list of one or more non-empty dataset strings>,
     "job_id": <job uuid>}
    
    and
    
    {"event_type": "IMAGE_DELETE",
     "datasets": <list of one or more non-empty dataset strings>,
     "job_id": <job uuid>}
  
 3. Sync Queue receives events formatted as:
     
     {"event_type": "SYNC",
      "dataset_ids": <list of one or more non-empty dataset strings>,
      "job_id": <job uuid>}   
"""

import os
import time
import uuid
import json
from datetime import datetime, timezone

import boto3

# --- Environment variables ---
AWS_REGION = os.environ["AWS_REGION"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
S3_DATASETS_ROOT = os.environ["S3_DATASETS_ROOT"]
DDB_IMAGERY_TABLE = os.environ["DDB_IMAGERY_TABLE"]
DDB_DATASET_TABLE = os.environ["DDB_DATASET_TABLE"]
DDB_JOB_TABLE = os.environ["DDB_JOB_TABLE"]
LOG_GROUP_NAME = os.environ["LOG_GROUP_NAME"] # Main centralized log group all calls in this infrastructure log to.
AWS_LAMBDA_FUNCTION_NAME = os.environ["AWS_LAMBDA_FUNCTION_NAME"]

# --- AWS clients ---
s3 = boto3.client("s3", region_name=AWS_REGION)
ddb = boto3.client("dynamodb", region_name=AWS_REGION)
logs = boto3.client("logs")

# Generate a unique stream name per container, prevents race conditions on logs.
# Use Lambda function name + container-specific UUID
LOG_STREAM = f"{AWS_LAMBDA_FUNCTION_NAME}-{int(time.time())}-{uuid.uuid4()}"

_sequence_token = None
_stream_initialized = False

# --- Helpers---
def init_log_stream():
    global _stream_initialized
    if _stream_initialized:
        return
    try:
        logs.create_log_stream(
            logGroupName=LOG_GROUP_NAME,
            logStreamName=LOG_STREAM
        )
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
    _stream_initialized = True

def log_event_core(log_dict: dict):
    """
    Write a structured JSON log event into the container's dedicated stream.
    """
    global _sequence_token
    init_log_stream()

    event = {
        "timestamp": int(time.time() * 1000), # unix in milliseconds required by CloudWatch.
        "message": json.dumps(log_dict)
    }

    kwargs = {
        "logGroupName": LOG_GROUP_NAME,
        "logStreamName": LOG_STREAM,
        "logEvents": [event]
    }
    if _sequence_token:
        kwargs["sequenceToken"] = _sequence_token

    resp = logs.put_log_events(**kwargs)
    _sequence_token = resp.get("nextSequenceToken")

def log_event(job_id, message, level = 'INFO'):
    log_event_core({
                    "lambda": AWS_LAMBDA_FUNCTION_NAME,
                    "job_id": job_id,
                    "status": message,
                    'level': level,
                    'utc_time': datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                })
    
def job_update(job_id, status, summary=None):
    """Update job status in the Job table."""
    expr = "SET job_status = :s, updated_at = :t"
    values = {
        ":s": {"S": status},
        ":t": {"S": datetime.utcnow().isoformat()}
    }
    if summary:
        expr += ", job_summary = :sum"
        values[":sum"] = {"S": summary}
    ddb.update_item(
        TableName=DDB_JOB_TABLE,
        Key={"job_id": {"S": job_id}},
        UpdateExpression=expr,
        ExpressionAttributeValues=values
    )
    
# --- Lambda entrypoint ---
def lambda_handler(event, context):
    """
    DLQ Lambda entrypoint.
    Handles messages that failed from lifecycle, image ops, and sync queues.
    Logs the full payload and marks the job as FAILED in the Job table.
    """
    for record in event.get("Records", []):
        job_id = "unknown"
        try:
            # DLQ messages wrap the original body in record["body"]
            body = record.get("body")
            if not body:
                log_event(job_id, f"DLQ record missing body: {record}", level="ERROR")
                continue

            # Try to parse JSON
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                log_event(job_id, f"DLQ record body not valid JSON: {body}", level="ERROR")
                continue

            job_id = payload.get("job_id", "unknown")
            event_type = payload.get("event_type", "unknown")

            # Log the full payload
            log_event(job_id, f"DLQ received event_type={event_type}, payload={json.dumps(payload)}", level="INFO")

            # Update job status if we have a job_id
            if job_id != "unknown":
                try:
                    job_update(job_id, "FAILED", summary=f"Job landed in DLQ (event_type={event_type})")
                except Exception as update_err:
                    log_event(job_id, f"Failed to update job status in DLQ handler: {update_err}", level="ERROR")

        except Exception as e:
            log_event(job_id, f"Unexpected error in DLQ handler: {e}", level="ERROR")
            # Do not re-raise: we don’t want the DLQ Lambda itself to fail