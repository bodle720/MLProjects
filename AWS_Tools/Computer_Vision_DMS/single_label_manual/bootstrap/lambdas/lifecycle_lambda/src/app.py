# -*- coding: utf-8 -*-
"""
This Lambda handles two events from the API, which are routed here through the 
lifecycle queue.

1. DELETE_DATASET - keys are 'event_type', 'dataset_id', 'job_id'
2. REMOVE_CLASS - keys are 'event_type', 'dataset_id', 'job_id', 'class_name'
"""

import os
import json
import boto3
import datetime
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Environment variables ---
AWS_REGION = os.environ["AWS_REGION"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
S3_DATASETS_ROOT = os.environ["S3_DATASETS_ROOT"]
DDB_IMAGERY_TABLE = os.environ["DDB_IMAGERY_TABLE"]
DDB_DATASET_TABLE = os.environ["DDB_DATASET_TABLE"]
DDB_JOB_TABLE = os.environ["DDB_JOB_TABLE"]

# --- AWS clients ---
s3 = boto3.client("s3", region_name=AWS_REGION)
ddb = boto3.client("dynamodb", region_name=AWS_REGION)

# --- Helpers for job and dataset status ---
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
        logger.info(f"[unlock_dataset] Dataset {dataset_id} unlocked by job {job_id}.")
    except ClientError as e:
        logger.warning(f"[unlock_dataset] Could not unlock dataset {dataset_id}: {e}")

def job_update(job_id, status, summary=None):
    """Update job status in the Job table."""
    expr = "SET job_status = :s, updated_at = :t"
    values = {
        ":s": {"S": status},
        ":t": {"S": datetime.datetime.utcnow().isoformat()}
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
    logger.info(f"[LifecycleLambda] Handling DELETE_DATASET for {dataset_id}, job {job_id}")

    try:
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
            
            # when uloading, we guarntee phashes are unique.
            if not resp.get("Items"):  # no other dataset references this image
                key = f"{S3_DATASETS_ROOT}/images/{phash}.{ext}"
                try:
                    s3.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
                    logger.info(f"[LifecycleLambda] Deleted image {key} from S3")
                except Exception as s3err:
                    logger.warning(f"[LifecycleLambda] Failed to delete {key}: {s3err}")

        # 3. Delete manifest folder for this dataset
        manifest_prefix = f"{S3_DATASETS_ROOT}/manifests/{dataset_id}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=manifest_prefix):
            for obj in page.get("Contents", []):
                s3.delete_object(Bucket=S3_BUCKET_NAME, Key=obj["Key"])
                logger.info(f"[LifecycleLambda] Deleted manifest object {obj['Key']}")

        # 4. Delete dataset row
        ddb.delete_item(
            TableName=DDB_DATASET_TABLE,
            Key={"dataset_id": {"S": dataset_id}}
        )

        # 5. Mark job complete
        job_update(job_id, "COMPLETE", f"Dataset {dataset_id} deleted successfully.")
        logger.info(f"[LifecycleLambda] DELETE_DATASET job {job_id} completed for dataset {dataset_id}")

    except Exception as e:
        logger.error(f"[LifecycleLambda] DELETE_DATASET failed for {dataset_id}: {e}")
        job_update(job_id, "FAILED", f"Deletion failed for dataset {dataset_id}: {e}")
        try:
            unlock_dataset(dataset_id, job_id)
        except Exception as unlock_err:
            logger.warning(f"[LifecycleLambda] Failed to unlock {dataset_id} after error: {unlock_err}")
        raise

def handle_remove_class_from_dataset(event):
    dataset_id = event["dataset_id"]
    class_name = event["class_name"]
    job_id = event["job_id"]
    logger.info(f"[LifecycleLambda] Handling REMOVE_CLASS for {dataset_id}, class {class_name}, job {job_id}")

    try:
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
                        logger.info(f"[LifecycleLambda] Deleted image {key} from S3")
                    except Exception as s3err:
                        logger.warning(f"[LifecycleLambda] Failed to delete {key}: {s3err}")

        # 5. Mark job complete and unlock dataset
        job_update(job_id, "COMPLETE", f"Class '{class_name}' removed from dataset {dataset_id}.")
        unlock_dataset(dataset_id, job_id)
        logger.info(f"[LifecycleLambda] REMOVE_CLASS job {job_id} completed for dataset {dataset_id}")

    except Exception as e:
        logger.error(f"[LifecycleLambda] REMOVE_CLASS failed for {dataset_id}, class {class_name}: {e}")
        job_update(job_id, "FAILED", f"Class removal failed for dataset {dataset_id}, class '{class_name}': {e}")
        try:
            unlock_dataset(dataset_id, job_id)
        except Exception as unlock_err:
            logger.warning(f"[LifecycleLambda] Failed to unlock {dataset_id} after error: {unlock_err}")
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
            event_type = body.get("event_type")
            logger.info(f"[LifecycleLambda] Received event: {event_type}")

            if event_type == "DELETE_DATASET":
                handle_delete_dataset(body)
            elif event_type == "REMOVE_CLASS":
                handle_remove_class_from_dataset(body)
            else:
                logger.warning(f"[LifecycleLambda] Unknown event_type: {event_type}")
        except Exception as e:
            logger.error(f"[LifecycleLambda] Error processing record: {e}")
            # Let the exception bubble up so SQS/Lambda retry/DLQ can handle it
            raise