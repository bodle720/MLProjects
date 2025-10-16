# -*- coding: utf-8 -*-
"""
This Lambda handles two events from the API, which are routed here through the 
image ops queue.

1. IMAGE_UPLOAD - keys are 'event_type', 'datasets', 'job_id', which indicates the
                  location of the required manifest: f"{root}/temp-images/{job_id}.json"
                  The manifest is of the form {'job_id': '...',
                                               'datasets': ['...', ...],
                                               'images': [{
                                                           'phash': '...',
                                                           'original_filename': '...',
                                                           'label': '...',
                                                           'extension': '...', <-- one of 'tiff', 'jpeg' or 'png'
                                                           'attributes': {...}
                                                           },
                                                          ...]
                                               }
2. IMAGE_DELETE - keys are 'event_type', 'dataset_id', 'job_id'
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
    
def extension_to_mime(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext in ("tif", "tiff"):
        return "image/tiff"
    else:
        raise ValueError(f"Unsupported extension: {ext}")
  
#%%
# --- Event handlers ---
def handle_image_upload(body: dict):
    job_id = body["job_id"]
    datasets = body["datasets"]
    manifest_key = f"{S3_DATASETS_ROOT}/temp-images/{job_id}.json"

    try:
        job_update(job_id, "IN_PROGRESS", "Processing image upload manifest")

        # Load manifest from S3
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
        manifest = json.loads(resp["Body"].read())

        for ds in datasets:
            for img in manifest["images"]:
                # Write imagery row into DDB_IMAGERY_TABLE
                ddb.put_item(
                    TableName=DDB_IMAGERY_TABLE,
                    Item={
                        "dataset_id": {"S": ds},
                        "phash": {"S": img["phash"]},
                        "label": {"S": img["label"]},
                        "extension": {"S": img["extension"]},
                        "attributes": {"S": json.dumps(img.get("attributes", {}))}
                    }
                )

        job_update(job_id, "COMPLETED", f"Uploaded {len(manifest['images'])} images")
    except Exception as e:
        job_update(job_id, "FAILED", f"Image upload failed: {e}")
        raise
    finally:
        for ds in datasets:
            unlock_dataset(ds, job_id)
            
# def handle_image_upload(event):
#     dataset_id = event["dataset_id"]
#     job_id = event["job_id"]
#     logger.info(f"[ImageOpsLambda] Handling IMAGE_UPLOAD for dataset {dataset_id}, job {job_id}")

#     manifest_key = f"{S3_DATASETS_ROOT}/temp-images/{job_id}.json"

#     try:
#         # 1. Load manifest JSON from S3
#         resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
#         manifest = json.loads(resp["Body"].read().decode("utf-8"))
#         images = manifest.get("images", [])

#         if not images:
#             raise Exception(f"Manifest {manifest_key} contained no images.")

#         # 2. For each image in manifest
#         for entry in images:
#             phash = entry["phash"]
#             label = entry["label"]
#             filename = entry["filename"]
#             ext = entry["extension"] # one of {"jpg", "jpeg", "png", "tif", "tiff"}, lowercase
            
#             dataset_phash = f"{dataset_id}#{phash}"

#             # Insert imagery row into DDB_IMAGERY_TABLE
#             ddb.put_item(
#                 TableName=DDB_IMAGERY_TABLE,
#                 Item={
#                     "dataset_phash": {"S": dataset_phash},
#                     "dataset_id": {"S": dataset_id},
#                     "phash": {"S": phash},
#                     "label": {"S": label},
#                     "extension": {"S": ext},
#                     "filename": {"S": filename},
#                     "bands_count": {"N": str(entry["bands_count"])},
#                     "bands_map": {"S": json.dumps(entry["bands_map"])},
#                     "bands_source": {"S": entry["bands_source"]},
#                     "forced_split": {"S": entry["forced_split"]},
#                     "created_at": {"S": datetime.datetime.utcnow().isoformat()}
#                 }
#             )

#             # Move image from temp-images/ to images/
#             src_key = f"{S3_DATASETS_ROOT}/temp-images/{phash}.{ext}"
#             dest_key = f"{S3_DATASETS_ROOT}/images/{phash}.{ext}"

#             # Copy to images/
#             s3.copy_object(
#                 Bucket=S3_BUCKET_NAME,
#                 CopySource={"Bucket": S3_BUCKET_NAME, "Key": src_key},
#                 Key=dest_key,
#                 ContentType=extension_to_mime(ext)
#             )

#             # Delete from temp-images/
#             s3.delete_object(Bucket=S3_BUCKET_NAME, Key=src_key)
#             logger.info(f"[ImageOpsLambda] Moved {src_key} -> {dest_key}")

#         # 3. Delete manifest file
#         s3.delete_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
#         logger.info(f"[ImageOpsLambda] Deleted manifest {manifest_key}")

#         # 4. Mark job complete and unlock dataset
#         job_update(job_id, "COMPLETE", f"Uploaded {len(images)} images to dataset {dataset_id}.")
#         unlock_dataset(dataset_id, job_id)
#         logger.info(f"[ImageOpsLambda] IMAGE_UPLOAD job {job_id} completed for dataset {dataset_id}")

#     except Exception as e:
#         logger.error(f"[ImageOpsLambda] IMAGE_UPLOAD failed for {dataset_id}: {e}")
#         job_error(job_id, f"Image upload failed for dataset {dataset_id}: {e}")
#         try:
#             unlock_dataset(dataset_id, job_id)
#         except Exception as unlock_err:
#             logger.warning(f"[ImageOpsLambda] Failed to unlock {dataset_id} after error: {unlock_err}")
#         raise

# def handle_remove_images(event):
#     dataset_id = event["dataset_id"]
#     job_id = event["job_id"]
#     logger.info(f"[ImageOpsLambda] Handling REMOVE_IMAGES for {dataset_id}, job {job_id}")

#     manifest_key = f"{S3_DATASETS_ROOT}/temp-deletions/{job_id}.json"

#     try:
#         # 1. Load manifest JSON from S3
#         resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
#         manifest = json.loads(resp["Body"].read().decode("utf-8"))
#         images = manifest.get("images", [])

#         if not images:
#             raise Exception(f"Manifest {manifest_key} contained no images.")

#         # 2. For each phash in the manifest
#         for phash in images:
#             dataset_phash = f"{dataset_id}#{phash}"

#             # Delete imagery row
#             try:
#                 ddb.delete_item(
#                     TableName=DDB_IMAGERY_TABLE,
#                     Key={"dataset_phash": {"S": dataset_phash}}
#                 )
#                 logger.info(f"[ImageOpsLambda] Deleted imagery row {dataset_phash}")
#             except Exception as ddb_err:
#                 logger.warning(f"[ImageOpsLambda] Failed to delete imagery row {dataset_phash}: {ddb_err}")

#             # 3. Check if this phash is still referenced by any other dataset
#             resp = ddb.query(
#                 TableName=DDB_IMAGERY_TABLE,
#                 IndexName="PhashIndex",
#                 KeyConditionExpression="phash = :p",
#                 ExpressionAttributeValues={":p": {"S": phash}}
#             )
#             if not resp.get("Items"):
#                 key = f"{S3_DATASETS_ROOT}/images/{phash}.png"
#                 try:
#                     s3.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
#                     logger.info(f"[ImageOpsLambda] Deleted image {key} from S3")
#                 except Exception as s3err:
#                     logger.warning(f"[ImageOpsLambda] Failed to delete {key}: {s3err}")

#         # 4. Delete manifest file
#         s3.delete_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
#         logger.info(f"[ImageOpsLambda] Deleted manifest {manifest_key}")

#         # 5. Mark job complete and unlock dataset
#         job_update(job_id, "COMPLETE", f"Removed {len(images)} images from dataset {dataset_id}.")
#         unlock_dataset(dataset_id, job_id)
#         logger.info(f"[ImageOpsLambda] REMOVE_IMAGES job {job_id} completed for dataset {dataset_id}")

#     except Exception as e:
#         logger.error(f"[ImageOpsLambda] REMOVE_IMAGES failed for {dataset_id}: {e}")
#         job_error(job_id, f"Image removal failed for dataset {dataset_id}: {e}")
#         try:
#             unlock_dataset(dataset_id, job_id)
#         except Exception as unlock_err:
#             logger.warning(f"[ImageOpsLambda] Failed to unlock {dataset_id} after error: {unlock_err}")
#         raise

# --- Lambda entrypoint ---
def lambda_handler(event, context):
    """
    Lambda entrypoint for image operations.
    Triggered by SQS messages from the image ops queue.
    """
    for record in event.get("Records", []):
        try:
            body = json.loads(record["body"])
            job_id = body["job_id"]
            event_type = body.get("event_type")
            log_event(job_id, f"Received event: {event_type}")

            if event_type == "IMAGE_UPLOAD":
                handle_image_upload(body)
            elif event_type == "IMAGE_DELETE":
                handle_remove_images(body)
            else:
                log_event(job_id, f"Unknown event_type: {event_type}", level = 'ERROR')
        except Exception as e:
            log_event(job_id, f"Error processing record: {e}", level='ERROR')
            try:
                job_update(job_id, 'FAILED', summary=f'Handler exception: {e}')
            except Exception as update_err:
                log_event(job_id, f"Failed to update job status for {job_id}: {update_err}", level='ERROR')
            raise