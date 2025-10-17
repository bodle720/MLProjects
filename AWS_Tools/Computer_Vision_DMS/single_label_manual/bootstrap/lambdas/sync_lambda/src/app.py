# -*- coding: utf-8 -*-
"""
This Lambda handles one event from the API, which are routed here through the 
sync queue.

SYNC - keys are 'event_type', 'dataset_ids', 'job_id'
"""

import os
import time
import uuid
import json
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError
from helpers import assign_splits, build_csv

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
    
def unlock_dataset(dataset_id, job_id):
    """
    Unlock a dataset, clearing the lock and lock owner.
    """
    try:
        ddb.update_item(
            TableName=DDB_DATASET_TABLE,
            Key={'dataset_id': {'S': dataset_id}},
            UpdateExpression="SET locked = :val REMOVE locked_by",
            ExpressionAttributeValues={
                ":val": {"BOOL": False},
                ":job_id": {"S": job_id}
            },
            ConditionExpression="locked_by = :job_id"
        )
        log_event(job_id, f"Dataset {dataset_id} unlocked by job id {job_id}")
    except ClientError as e:
        log_event(job_id, f"Could not unlock dataset {dataset_id}: {e}", level = 'ERROR')

# --- Job status helper ---
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

def handle_sync(event):
    job_id = event["job_id"]
    dataset_ids = event["dataset_ids"]
    log_event(job_id, f"Handling SYNC job for datasets: {dataset_ids}")

    try:
        job_update(job_id, 'IN_PROGRESS', summary=f"Starting sync for {len(dataset_ids)} datasets")

        for dataset_id in dataset_ids:
            # Fetch dataset metadata
            ds_resp = ddb.get_item(
                TableName=DDB_DATASET_TABLE,
                Key={"dataset_id": {"S": dataset_id}},
                ConsistentRead=True
            )
            ds_item = ds_resp.get("Item")
            if not ds_item:
                raise Exception(f"Dataset {dataset_id} not found.")

            class_dict = json.loads(ds_item["class_to_id_dict"]["S"])

            # Query imagery table
            resp = ddb.query(
                TableName=DDB_IMAGERY_TABLE,
                IndexName="DatasetIndex",
                KeyConditionExpression="dataset_id = :d",
                ExpressionAttributeValues={":d": {"S": dataset_id}}
            )
            imagery_items = resp.get("Items", [])

            enriched = []
            for item in imagery_items:
                phash = item["phash"]["S"]
                label = item["label"]["S"]
                class_id = class_dict[label]
                
                enriched.append({"phash": phash,
                                 "label": label,
                                 "class_id": class_id})

            enriched = assign_splits(enriched)

            # Write manifests
            manifest_prefix = f"{S3_DATASETS_ROOT}/manifests/{dataset_id}/"
            json_key = manifest_prefix + "manifest.json"
            csv_key = manifest_prefix + "manifest.csv"

            s3.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=json_key,
                Body=json.dumps(enriched, indent=2).encode("utf-8"),
                ContentType="application/json"
            )
            csv_body = build_csv(enriched)
            
            s3.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=csv_key,
                Body=csv_body.encode("utf-8"),
                ContentType="text/csv"
            )

            # Mark dataset as synced
            ddb.update_item(
                TableName=DDB_DATASET_TABLE,
                Key={"dataset_id": {"S": dataset_id}},
                UpdateExpression="SET synced = :s",
                ExpressionAttributeValues={":s": {"BOOL": True}}
            )
            log_event(job_id, f"Wrote manifests and marked {dataset_id} as synced")

        job_update(job_id, 'COMPLETE', summary=f"Synced {len(dataset_ids)} datasets successfully.")

    except Exception as e:
        log_event(job_id, f"SYNC job failed: {e}", level='ERROR')
        job_update(job_id, 'FAILED', summary=f"Sync failed: {e}")
        raise
    finally:
        for dsid in dataset_ids:
            try:
                unlock_dataset(dsid, job_id)
            except Exception as unlock_err:
                log_event(job_id, f"Failed to unlock {dsid}: {unlock_err}", level='ERROR')

# --- Lambda entrypoint ---
def lambda_handler(event, context):
    """
    Lambda entrypoint for sync operations.
    Triggered by SQS messages from the sync queue.
    """
    for record in event.get("Records", []):
        try:
            body = json.loads(record["body"])
            job_id = body["job_id"]
            event_type = body.get("event_type")
            log_event(job_id, f"Received event: {event_type}")

            if event_type == "SYNC":
                handle_sync(body)
            else:
                log_event(job_id, f"Unknown event_type: {event_type}", level = 'ERROR')
        except Exception as e:
            log_event(job_id, f"Error processing record: {e}", level='ERROR')
            try:
                job_update(job_id, 'FAILED', summary=f'Handler exception: {e}')
            except Exception as update_err:
                log_event(job_id, f"Failed to update job status for {job_id}: {update_err}", level='ERROR')
            raise