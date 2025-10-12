# -*- coding: utf-8 -*-
"""
"""

import os
import json
import boto3
import logging
import datetime
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

# --- Job status helpers ---
def job_update(job_id, status, summary=None):
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

def job_complete(job_id, summary=None):
    job_update(job_id, "COMPLETE", summary)

def job_error(job_id, summary=None):
    job_update(job_id, "FAILED", summary)

# --- Dataset unlock helper ---
def unlock_dataset(dataset_id, job_id):
    try:
        ddb.update_item(
            TableName=DDB_DATASET_TABLE,
            Key={'dataset_id': {'S': dataset_id}},
            UpdateExpression="SET locked = :val REMOVE locked_by",
            ExpressionAttributeValues={":val": {"BOOL": False}}
        )
        logger.info(f"[unlock_dataset] Dataset {dataset_id} unlocked by job {job_id}.")
    except ClientError as e:
        logger.warning(f"[unlock_dataset] Could not unlock dataset {dataset_id}: {e}")

# --- Event handlers ---
def handle_image_upload(event):
    dataset_id = event["dataset_id"]
    job_id = event["job_id"]
    logger.info(f"[ImageOpsLambda] Handling IMAGE_UPLOAD for dataset {dataset_id}, job {job_id}")

    manifest_key = f"{S3_DATASETS_ROOT}/temp-images/{job_id}.json"

    try:
        # 1. Load manifest JSON from S3
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
        manifest = json.loads(resp["Body"].read().decode("utf-8"))
        images = manifest.get("images", [])

        if not images:
            raise Exception(f"Manifest {manifest_key} contained no images.")

        # 2. For each image in manifest
        for entry in images:
            phash = entry["phash"]
            label = entry["label"]
            filename = entry["filename"]

            dataset_phash = f"{dataset_id}#{phash}"

            # Insert imagery row into DDB_IMAGERY_TABLE
            ddb.put_item(
                TableName=DDB_IMAGERY_TABLE,
                Item={
                    "dataset_phash": {"S": dataset_phash},
                    "dataset_id": {"S": dataset_id},
                    "phash": {"S": phash},
                    "label": {"S": label},
                    "filename": {"S": filename},
                    "created_at": {"S": datetime.datetime.utcnow().isoformat()}
                }
            )

            # Move image from temp-images/ to images/
            src_key = f"{S3_DATASETS_ROOT}/temp-images/{phash}.png"
            dest_key = f"{S3_DATASETS_ROOT}/images/{phash}.png"

            # Copy to images/
            s3.copy_object(
                Bucket=S3_BUCKET_NAME,
                CopySource={"Bucket": S3_BUCKET_NAME, "Key": src_key},
                Key=dest_key,
                ContentType="image/png"
            )

            # Delete from temp-images/
            s3.delete_object(Bucket=S3_BUCKET_NAME, Key=src_key)
            logger.info(f"[ImageOpsLambda] Moved {src_key} -> {dest_key}")

        # 3. Delete manifest file
        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
        logger.info(f"[ImageOpsLambda] Deleted manifest {manifest_key}")

        # 4. Mark job complete and unlock dataset
        job_update(job_id, "COMPLETE", f"Uploaded {len(images)} images to dataset {dataset_id}.")
        unlock_dataset(dataset_id, job_id)
        logger.info(f"[ImageOpsLambda] IMAGE_UPLOAD job {job_id} completed for dataset {dataset_id}")

    except Exception as e:
        logger.error(f"[ImageOpsLambda] IMAGE_UPLOAD failed for {dataset_id}: {e}")
        job_error(job_id, f"Image upload failed for dataset {dataset_id}: {e}")
        try:
            unlock_dataset(dataset_id, job_id)
        except Exception as unlock_err:
            logger.warning(f"[ImageOpsLambda] Failed to unlock {dataset_id} after error: {unlock_err}")
        raise

def handle_remove_images(event):
    dataset_id = event["dataset_id"]
    job_id = event["job_id"]
    images = event["images"]  # list of phashes
    logger.info(f"[ImageOpsLambda] Handling REMOVE_IMAGES_FROM_DATASET for {dataset_id}, job {job_id}")

    try:
        # 1. For each phash in the list
        for phash in images:
            dataset_phash = f"{dataset_id}#{phash}"

            # Delete imagery row for this dataset + phash
            try:
                ddb.delete_item(
                    TableName=DDB_IMAGERY_TABLE,
                    Key={"dataset_phash": {"S": dataset_phash}}
                )
                logger.info(f"[ImageOpsLambda] Deleted imagery row {dataset_phash}")
            except Exception as ddb_err:
                logger.warning(f"[ImageOpsLambda] Failed to delete imagery row {dataset_phash}: {ddb_err}")

            # 2. Check if this phash is still referenced by any other dataset
            resp = ddb.query(
                TableName=DDB_IMAGERY_TABLE,
                IndexName="PhashIndex",  # new GSI on phash
                KeyConditionExpression="phash = :p",
                ExpressionAttributeValues={":p": {"S": phash}}
            )
            if not resp.get("Items"):  # no other dataset references this image
                key = f"{S3_DATASETS_ROOT}/images/{phash}.png"
                try:
                    s3.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
                    logger.info(f"[ImageOpsLambda] Deleted image {key} from S3")
                except Exception as s3err:
                    logger.warning(f"[ImageOpsLambda] Failed to delete {key}: {s3err}")

        # 3. Mark job complete and unlock dataset
        job_update(job_id, "COMPLETE", f"Removed {len(images)} images from dataset {dataset_id}.")
        unlock_dataset(dataset_id, job_id)
        logger.info(f"[ImageOpsLambda] REMOVE_IMAGES_FROM_DATASET job {job_id} completed for dataset {dataset_id}")

    except Exception as e:
        logger.error(f"[ImageOpsLambda] REMOVE_IMAGES_FROM_DATASET failed for {dataset_id}: {e}")
        job_error(job_id, f"Image removal failed for dataset {dataset_id}: {e}")
        try:
            unlock_dataset(dataset_id, job_id)
        except Exception as unlock_err:
            logger.warning(f"[ImageOpsLambda] Failed to unlock {dataset_id} after error: {unlock_err}")
        raise

# --- Lambda entrypoint ---
def lambda_handler(event, context):
    """
    Lambda entrypoint for image operations.
    Triggered by SQS messages from the image ops queue.
    """
    for record in event.get("Records", []):
        try:
            body = json.loads(record["body"])
            event_type = body.get("event_type")
            logger.info(f"[ImageOpsLambda] Received event: {event_type}")

            if event_type == "IMAGE_UPLOAD":
                handle_image_upload(body)
            elif event_type == "REMOVE_IMAGES_FROM_DATASET":
                handle_remove_images(body)
            else:
                logger.warning(f"[ImageOpsLambda] Unknown event_type: {event_type}")
        except Exception as e:
            logger.error(f"[ImageOpsLambda] Error processing record: {e}")
            raise