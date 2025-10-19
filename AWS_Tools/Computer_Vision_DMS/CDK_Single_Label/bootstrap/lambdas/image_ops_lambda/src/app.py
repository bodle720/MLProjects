# -*- coding: utf-8 -*-
"""
This Lambda handles two events from the API, which are routed here through the 
image ops queue.

1. IMAGE_UPLOAD - keys are 'event_type', 'datasets', 'job_id', which indicates the
                  location of the required manifest: f"{root}/temp-images/{job_id}.json"
                  The manifest is of the form {'job_id': '...',
                                               'datasets': ['...', ...],
                                               'band_mapping': {...}
                                               'images': [{
                                                           'phash': '...',
                                                           'original_filename': '...',
                                                           'label': '...',
                                                           'extension': '...', <-- one of 'tiff', 'jpeg' or 'png'
                                                           'attributes': {...} <-- additional user defined attributes to add to the imagery table for each image in the manifest
                                                           },
                                                          ...]
                                               }
2. IMAGE_DELETE - keys are 'event_type', 'datasets', 'job_id'
                  The manifest is at f"{root}/temp-deletions/{job_id}.json"
                  The manifest is of the form:
                      manifest = {
                          "datasets": dataset_ids, <-- list of dataset ids
                          "job_id": job_id,
                          "phashes": phashes <-- list of non-empty string phashes
                      }
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
  
# --- Event Handlers ---
def handle_bulk_image_upload(body: dict):
    """
    Handle IMAGE_UPLOAD events:
    - Load manifest from S3
    - Insert imagery rows into DDB_IMAGERY_TABLE for each dataset
    - Copy each unique image from temp-images/ to images/ once (if not already present)
    - Delete temp files and manifest
    - Update job status and unlock datasets
    """
    job_id = body["job_id"]
    datasets = body["datasets"]
    manifest_key = f"{S3_DATASETS_ROOT}/temp-images/{job_id}.json"

    try:
        job_update(job_id, "IN_PROGRESS", "Processing image upload manifest")

        # Load manifest from S3
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
        manifest = json.loads(resp["Body"].read())

        images = manifest.get("images", [])
        band_mapping = manifest.get("band_mapping", {})  # match API side
        band_count = len(band_mapping)

        if not images:
            raise Exception(f"Manifest {manifest_key} contained no images.")

        copied = set()
        copied_count = 0
        already_existed_count = 0
        for img in images:
            phash = img["phash"]
            ext = img["extension"].lower().lstrip(".")
            original_filename = img.get("original_filename", "")
            attributes = img.get("attributes", {})
            already_exists = img.get("already_exists", False)

            # Only copy if not already present globally
            if not already_exists and phash not in copied:
                src_key = f"{S3_DATASETS_ROOT}/temp-images/{phash}.{ext}"
                dest_key = f"{S3_DATASETS_ROOT}/images/{phash}.{ext}"

                s3.copy_object(
                    Bucket=S3_BUCKET_NAME,
                    CopySource={"Bucket": S3_BUCKET_NAME, "Key": src_key},
                    Key=dest_key,
                    ContentType=extension_to_mime(ext)
                )
                s3.delete_object(Bucket=S3_BUCKET_NAME, Key=src_key)
                copied.add(phash)
                copied_count += 1
            else:
                already_existed_count += 1

            # Build common item fields
            base_item = {
                "phash": {"S": phash},
                "label": {"S": img["label"]},
                "extension": {"S": ext},
                "original_filename": {"S": original_filename},
                "band_mapping": {"S": json.dumps(band_mapping)},
                "band_count": {"N": str(band_count)},
                "uploaded_at": {"S": datetime.now(timezone.utc).isoformat()},
            }

            for k, v in attributes.items():
                base_item[k] = {"S": v}

            # Always insert dataset rows
            for ds in datasets:
                dataset_phash = f"{ds}#{phash}"
                item = {
                    "dataset_phash": {"S": dataset_phash},
                    "dataset_id": {"S": ds},
                    **base_item
                }
                # If already exists, silently replaces with new row.
                ddb.put_item(TableName=DDB_IMAGERY_TABLE, Item=item)

        # Log all successful copies        
        log_event(job_id, {
                    "copied_count": copied_count,
                    "already_existed_count": already_existed_count
                })

        # Delete manifest file
        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
        log_event(job_id, f"Deleted manifest {manifest_key}")
        job_update(job_id, "COMPLETED",
           f"Registered {len(images)} images ({copied_count} new, {already_existed_count} existing) across {len(datasets)} datasets")

    except Exception as e:
        job_update(job_id, "FAILED", f"Image upload failed: {e}")
        log_event(job_id, f"Image upload failed: {e}", level = 'ERROR')
        raise
    finally:
        for ds in datasets:
            unlock_dataset(ds, job_id)

def handle_bulk_remove_images(body: dict):
    """
    Handle IMAGE_DELETE events:
    - Load manifest from S3
    - Delete imagery rows from DDB_IMAGERY_TABLE for each dataset/phash
    - Capture extension before deletion so we can remove S3 object if no references remain
    - Delete manifest file
    - Update job status and unlock datasets
    """
    job_id = body["job_id"]
    manifest_key = f"{S3_DATASETS_ROOT}/temp-deletions/{job_id}.json"

    try:
        job_update(job_id, "IN_PROGRESS", "Processing image deletion manifest")

        # Load manifest from S3
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
        manifest = json.loads(resp["Body"].read())

        dataset_ids = manifest.get("datasets", [])
        phashes = manifest.get("phashes", [])

        if not dataset_ids or not phashes:
            raise Exception(f"Manifest {manifest_key} missing datasets or phashes.")

        deleted_rows = 0
        nonexistent_imagery_rows_to_delete = 0
        deleted_s3 = 0
        number_phashes_used_by_other_datasets = 0
        not_referenced_but_no_extension_keys = []
        
        for phash in phashes:
            ext = None  # capture extension from one of the rows we delete

            # Delete rows for each dataset
            for ds in dataset_ids:
                dataset_phash = f"{ds}#{phash}"
                try:
                    # Fetch extension before deleting (if row exists)
                    row = ddb.get_item(
                        TableName=DDB_IMAGERY_TABLE,
                        Key={"dataset_phash": {"S": dataset_phash}},
                        ProjectionExpression="extension"
                    ).get("Item")

                    if row and not ext:
                        ext = row["extension"]["S"].lower()

                    ddb.delete_item(
                        TableName=DDB_IMAGERY_TABLE,
                        Key={"dataset_phash": {"S": dataset_phash}},
                        ConditionExpression="attribute_exists(dataset_phash)"
                    )
                    deleted_rows += 1
                except ClientError as e:
                    code = e.response.get("Error", {}).get("Code")
                    if code == "ConditionalCheckFailedException":
                        nonexistent_imagery_rows_to_delete += 1
                    else:
                        raise Exception(f"Error deleting {dataset_phash}: {e}")

            # Check if phash is still referenced anywhere using PhashIndex
            resp = ddb.query(
                TableName=DDB_IMAGERY_TABLE,
                IndexName="PhashIndex",
                KeyConditionExpression="phash = :p",
                ExpressionAttributeValues={":p": {"S": phash}},
                ProjectionExpression="dataset_id"
            )
            still_referenced = bool(resp.get("Items"))

            if not still_referenced:
                if ext:
                    key = f"{S3_DATASETS_ROOT}/images/{phash}.{ext}"
                    try:
                        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
                        deleted_s3 += 1
                    except ClientError as e:
                        raise Exception(f"Error deleting {key}: {e}")
                else:
                    not_referenced_but_no_extension_keys.append(key)
            else:
                number_phashes_used_by_other_datasets += 1

        # Make logs
        if not_referenced_but_no_extension_keys:
            log_event(job_id, {
                "warning": "Unreferenced phashes with no extension",
                "sample_keys": not_referenced_but_no_extension_keys[:10],  # sample
                "count": len(not_referenced_but_no_extension_keys)
            }, level="WARNING")

        log_event(job_id, {
            "deleted_imagery_table_rows": deleted_rows,
            "nonexistent_rows_tried_to_delete": nonexistent_imagery_rows_to_delete,
            "deleted_from_s3_count": deleted_s3,
            "still_referenced": number_phashes_used_by_other_datasets
        })

        # Delete manifest file
        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=manifest_key)
        log_event(job_id, f"Deleted manifest {manifest_key}")

        job_update(job_id, "COMPLETED",
                   f"Deleted {deleted_rows} imagery rows "
                   f"({nonexistent_imagery_rows_to_delete} nonexistent), "
                   f"removed {deleted_s3} S3 objects, "
                   f"{number_phashes_used_by_other_datasets} still referenced")
        
    except Exception as e:
        job_update(job_id, "FAILED", f"Image deletion failed: {e}")
        log_event(job_id, f"Image deletion failed: {e}", level = 'ERROR')
        raise
    finally:
        # Unlock all datasets listed in manifest
        for ds in manifest.get("datasets", []):
            unlock_dataset(ds, job_id)

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
                handle_bulk_image_upload(body)
            elif event_type == "IMAGE_DELETE":
                handle_bulk_remove_images(body)
            else:
                log_event(job_id, f"Unknown event_type: {event_type}", level = 'ERROR')
        except Exception as e:
            log_event(job_id, f"Error processing record: {e}", level='ERROR')
            try:
                job_update(job_id, 'FAILED', summary=f'Handler exception: {e}')
            except Exception as update_err:
                log_event(job_id, f"Failed to update job status for {job_id}: {update_err}", level='ERROR')
            raise