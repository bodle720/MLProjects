# -*- coding: utf-8 -*-
"""
This Lambda handles two events from the API, which are routed here through the 
lifecycle queue.

1. DELETE_DATASET - keys are 'event_type', 'dataset_id', 'job_id'
2. REMOVE_CLASS - keys are 'event_type', 'dataset_id', 'job_id', 'class_name'
"""

import os
import time
import uuid
import json
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

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

# --- Event handlers ---
def handle_delete_dataset(event):
    dataset_id = event["dataset_id"]
    job_id = event["job_id"]

    log_event(job_id, f"Handling DELETE_DATASET event for dataset {dataset_id} and job id {job_id}")

    try:
        job_update(job_id, 'IN_PROGRESS', summary='Calling appropriate handler for dataset deletion.')

        # 1. Query imagery table for all images belonging to this dataset
        resp = ddb.query(
            TableName=DDB_IMAGERY_TABLE,
            IndexName="DatasetIndex",
            KeyConditionExpression="dataset_id = :d",
            ExpressionAttributeValues={":d": {"S": dataset_id}}
        )
        imagery_items = resp.get("Items", [])

        phashes_exts_to_check = [(item["phash"]["S"], item["extension"]["S"]) for item in imagery_items]

        # Delete imagery rows for this dataset
        for item in imagery_items:
            ddb.delete_item(
                TableName=DDB_IMAGERY_TABLE,
                Key={"dataset_phash": {"S": item["dataset_phash"]["S"]}}
            )

        # 2. For each phash, check if any other dataset references it
        for phash, ext in phashes_exts_to_check:
            resp = ddb.query(
                TableName=DDB_IMAGERY_TABLE,
                IndexName="PhashIndex",
                KeyConditionExpression="phash = :p",
                ExpressionAttributeValues={":p": {"S": phash}}
            )
            
            # when uploading, we guarantee phashes are unique.
            if not resp.get("Items"):  # no other dataset references this image
                key = f"{S3_DATASETS_ROOT}/images/{phash}.{ext}"
                try:
                    s3.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
                    log_event(job_id, f"Deleted image {key} from S3")
                except Exception as s3err:
                    log_event(job_id, f"Failed to delete {key}: {s3err}", level = 'ERROR')

        # 3. Delete manifest folder for this dataset
        manifest_prefix = f"{S3_DATASETS_ROOT}/manifests/{dataset_id}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=manifest_prefix):
            for obj in page.get("Contents", []):
                s3.delete_object(Bucket=S3_BUCKET_NAME, Key=obj["Key"])
                log_event(job_id, f"Deleted manifest object {obj['Key']}")

        # 4. Delete dataset row
        ddb.delete_item(
            TableName=DDB_DATASET_TABLE,
            Key={"dataset_id": {"S": dataset_id}}
        )

        # 5. Mark job complete
        job_update(job_id, "COMPLETE", f"Dataset {dataset_id} deleted successfully.")
        log_event(job_id, f"DELETE_DATASET job {job_id} completed for dataset {dataset_id}")

    except Exception as e:
        log_event(job_id, f"DELETE_DATASET failed for {dataset_id}: {e}", level = 'ERROR')
        job_update(job_id, "FAILED", f"Deletion failed for dataset {dataset_id}: {e}")
        try:
            unlock_dataset(dataset_id, job_id)
        except Exception as unlock_err:
            log_event(job_id, f"Failed to unlock {dataset_id} after error: {unlock_err}", level = 'ERROR')
        raise

def handle_remove_class_from_dataset(event):
    dataset_id = event["dataset_id"]
    class_name = event["class_name"]
    job_id = event["job_id"]
    
    log_event(job_id, f"Handling REMOVE_CLASS event for class {class_name} in dataset {dataset_id} and job id {job_id}")
    
    try:
        job_update(job_id, 'IN_PROGRESS', summary='Calling appropriate handler for removing a class.')

        # 1. Fetch dataset row
        resp = ddb.get_item(
            TableName=DDB_DATASET_TABLE,
            Key={"dataset_id": {"S": dataset_id}},
            ConsistentRead=True
        )
        item = resp.get("Item")
        if not item:
            raise Exception(f"Dataset {dataset_id} not found.")

        # Parse class_to_id_dict
        class_dict_str = item.get("class_to_id_dict", {}).get("S", "{}")
        try:
            class_dict = json.loads(class_dict_str)
        except json.JSONDecodeError:
            raise Exception(f"Corrupted class_to_id_dict for dataset {dataset_id}.")

        if class_name not in class_dict:
            raise Exception(f"Class '{class_name}' not found in dataset {dataset_id}.")

        # 2. Build new class_to_id_dict with sequential IDs
        new_classes = [c for c in class_dict.keys() if c != class_name]
        new_class_dict = {c: idx for idx, c in enumerate(new_classes)}

        # Update dataset row
        ddb.update_item(
            TableName=DDB_DATASET_TABLE,
            Key={"dataset_id": {"S": dataset_id}},
            UpdateExpression="SET class_to_id_dict = :c",
            ExpressionAttributeValues={":c": {"S": json.dumps(new_class_dict)}}
        )

        # 3. Query imagery table for all images in this dataset
        resp = ddb.query(
            TableName=DDB_IMAGERY_TABLE,
            IndexName="DatasetIndex",
            KeyConditionExpression="dataset_id = :d",
            ExpressionAttributeValues={":d": {"S": dataset_id}}
        )
        imagery_items = resp.get("Items", [])

        # 4. For each image in this class, delete imagery row and maybe S3 object
        for item in imagery_items:
            phash = item["phash"]["S"]
            label = item["label"]['S']
            if label == class_name:
                # Delete imagery row
                ddb.delete_item(
                    TableName=DDB_IMAGERY_TABLE,
                    Key={"dataset_phash": {"S": item["dataset_phash"]["S"]}}
                )

                # Check if phash is used by any other dataset
                resp2 = ddb.query(
                    TableName=DDB_IMAGERY_TABLE,
                    IndexName="PhashIndex",
                    KeyConditionExpression="phash = :p",
                    ExpressionAttributeValues={":p": {"S": phash}}
                )
                if not resp2.get("Items"):
                    ext = item["extension"]["S"]
                    key = f"{S3_DATASETS_ROOT}/images/{phash}.{ext}"
                    try:
                        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
                        log_event(job_id, f"Deleted image {key} from S3")
                    except Exception as s3err:
                        log_event(job_id, f"Failed to delete {key}: {s3err}", level = 'ERROR')

        # 5. Mark job complete and unlock dataset
        job_update(job_id, "COMPLETE", f"Class '{class_name}' removed from dataset {dataset_id}.")
        unlock_dataset(dataset_id, job_id)
        log_event(job_id, f"REMOVE_CLASS job {job_id} completed for dataset {dataset_id}")
    except Exception as e:
        log_event(job_id, f"REMOVE_CLASS failed for {dataset_id}, class {class_name}: {e}", level = 'ERROR')
        job_update(job_id, "FAILED", f"Class removal failed for dataset {dataset_id}, class '{class_name}': {e}")
        try:
            unlock_dataset(dataset_id, job_id)
        except Exception as unlock_err:
            log_event(job_id, f"Failed to unlock {dataset_id} after error: {unlock_err}", level = 'ERROR')
        raise

# --- Lambda entrypoint ---
def lambda_handler(event, context):
    """
    Lambda entrypoint for lifecycle operations.
    Triggered by SQS messages from the lifecycle queue.
    """
    for record in event.get("Records", []):
        try:
            body = json.loads(record["body"])
            job_id = body["job_id"]
            event_type = body.get("event_type")
            log_event(job_id, f"Received event: {event_type}")

            if event_type == "DELETE_DATASET":
                handle_delete_dataset(body)
            elif event_type == "REMOVE_CLASS":
                handle_remove_class_from_dataset(body)
            else:
                log_event(job_id, f"Unknown event_type: {event_type}", level = 'ERROR')
        except Exception as e:
            log_event(job_id, f"Error processing record: {e}", level='ERROR')
            try:
                job_update(job_id, 'FAILED', summary=f'Handler exception: {e}')
            except Exception as update_err:
                log_event(job_id, f"Failed to update job status for {job_id}: {update_err}", level='ERROR')
            raise