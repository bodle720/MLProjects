import os
import json
import logging
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

# Lambda layer imports
from common.utils import update_job_status, log

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
UPLOAD_STATE_MACHINE_ARN = os.environ["UPLOAD_STATE_MACHINE_ARN"]
FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

EVENT_TYPE = "IMAGE_UPLOAD"
ALLOWED_LABEL_TYPES = ["string_labels", "bounding_boxes", "semantic_masks", "instance_annotations"]

firehose = boto3.client("firehose")
sf = boto3.client("stepfunctions")
s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

def list_keys(bucket, prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

def basenames_from_keys(keys, allowed_exts=None):
    files = []
    for k in keys:
        if k.endswith("/"):
            continue
        base, ext = os.path.splitext(os.path.basename(k))
        if allowed_exts is None or ext.lower() in allowed_exts:
            files.append(base)
    return files

def require_no_duplicates(name_list, kind):
    counts = {}
    for n in name_list:
        counts[n] = counts.get(n, 0) + 1
    dups = [n for n, c in counts.items() if c > 1]
    if dups:
        raise ValueError(f"Duplicate {kind} detected for basenames: {dups}")

def validate_labels(bucket, job_id, label_types, user):
    image_keys = list_keys(bucket, f"temp/image-upload/{job_id}/images/")
    image_bases = basenames_from_keys(
        image_keys, allowed_exts={".jpg", ".jpeg", ".png"}
    )
    require_no_duplicates(image_bases, "images")

    for label_type in label_types:
        if label_type in ["string_labels", "bounding_boxes", "instance_annotations"]:
            label_prefix = f"temp/image-upload/{job_id}/{label_type}/"
            label_keys = list_keys(bucket, label_prefix)
            label_bases = basenames_from_keys(label_keys, allowed_exts={".json"})
            require_no_duplicates(label_bases, f"{label_type}")

            if set(image_bases) != set(label_bases):
                missing_in_labels = sorted(set(image_bases) - set(label_bases))
                extra_in_labels = sorted(set(label_bases) - set(image_bases))
                error_msg = f"Mismatch for {label_type}. Missing labels for: {missing_in_labels}. Extra labels for: {extra_in_labels}"
                raise ValueError(error_msg)

            if len(image_bases) != len(label_bases):
                error_msg = f"Count mismatch for {label_type}. images={len(image_bases)} labels={len(label_bases)}"
                raise ValueError(error_msg)

        elif label_type == "semantic_masks":
            mask_prefix = f"temp/image-upload/{job_id}/semantic_masks/"
            mask_keys = list_keys(bucket, mask_prefix)

            mask_png_bases = basenames_from_keys(mask_keys, allowed_exts={".png"})
            mask_json_bases = basenames_from_keys(mask_keys, allowed_exts={".json"})
            require_no_duplicates(mask_png_bases, "semantic mask PNGs")
            require_no_duplicates(mask_json_bases, "semantic mask JSONs")

            if set(image_bases) != set(mask_png_bases) or set(image_bases) != set(mask_json_bases):
                missing_png = sorted(set(image_bases) - set(mask_png_bases))
                extra_png = sorted(set(mask_png_bases) - set(image_bases))
                missing_json = sorted(set(image_bases) - set(mask_json_bases))
                extra_json = sorted(set(mask_json_bases) - set(image_bases))
                error_msg = f"Semantic masks mismatch. PNG missing: {missing_png}, PNG extra: {extra_png}, JSON missing: {missing_json}, JSON extra: {extra_json}"
                raise ValueError(error_msg)

            if len(image_bases) != len(mask_png_bases) or len(image_bases) != len(mask_json_bases):
                error_msg = f"Semantic masks count mismatch. images={len(image_bases)}, pngs={len(mask_png_bases)} jsons={len(mask_json_bases)}"
                raise ValueError(error_msg)

        log(job_id, user, EVENT_TYPE, f"Found {len(image_bases)} images and labels for label type = {label_type}", FIREHOSE_STREAM_NAME)

def handler(event, context):

    job_table = dynamodb.Table(JOB_TABLE_NAME)

    # Guard: ensure there's at least one record
    records = event.get("Records", [])
    if not records:
        return {"status": "failed", "job_id": 'unknown', "error": "No Records in event"}

    # Use the first SQS record (batch_size=1 configured)
    sqs_rec = records[0]

    # SQS message body contains the S3 notification JSON as a string
    body = sqs_rec.get("body")
    if not body:
        return {"status": "failed", "job_id": 'unknown', "error": "SQS record missing body"}

    try:
        body_json = json.loads(body)
    except Exception as e:
        return {"status": "failed", "job_id": 'unknown', "error": f"Failed to parse SQS body as JSON: {str(e)}"}

    # Expect the S3 notification inside body_json["Records"][0]["s3"]
    s3_records = body_json.get("Records", [])
    if not s3_records:
        return {"status": "failed", "job_id": 'unknown', "error": "No S3 Records inside SQS body"}

    s3_rec = s3_records[0]
    s3_info = s3_rec.get("s3", {})
    bucket = s3_info.get("bucket", {}).get("name")
    key = unquote_plus(s3_info.get("object", {}).get("key", ""))

    if bucket != FILE_BUCKET_NAME:
        return {"status": "failed", "job_id": 'unknown', "error": f"Bucket mismatch in upload kickoff lambda"}

    # Now proceed with your existing logic to fetch job.json, validate, etc.
    job_id = None
    user = 'unknown'
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        job_data = json.loads(obj["Body"].read().decode("utf-8"))
        job_id = job_data["job_id"]
        user = job_data["user"]
        num_images = job_data["num_images"]
        source = job_data["source"]
        label_types = job_data["label_types"]
    except Exception as e:
        if job_id:
            log(job_id, user, EVENT_TYPE, "Upload Kickoff Lambda could not initialize job_id, user, num_images, source, and label_types from manifest", FIREHOSE_STREAM_NAME, error=str(e), level='error')
            update_job_status(job_id, "FAILED", job_table, FIREHOSE_STREAM_NAME, user = user, event_type = EVENT_TYPE, error_msg=str(e))
        return {"status": "failed", "error": f"Upload Kickoff Lambda could not initialize expected manifest fields: {str(e)}"}

    try:
        validate_labels(bucket, job_id, label_types, user)
    except Exception as e:
        log(job_id, user, EVENT_TYPE, "Error validating labels in kickoff lambda.", FIREHOSE_STREAM_NAME, error=str(e), level='error')
        update_job_status(job_id, "FAILED", job_table, FIREHOSE_STREAM_NAME, user=user, event_type=EVENT_TYPE, error_msg=str(e))
        return {"status": "failed", "job_id": job_id, "error": f"Error validating labels: {str(e)}"}

    try:
        response = sf.start_execution(
            stateMachineArn=UPLOAD_STATE_MACHINE_ARN,
            name = f"{job_id}-{int(datetime.now().timestamp() * 1000)}"[:80],
            input=json.dumps({
                "job_id": job_id,
                "user": user,
                "label_types": label_types,
                "source":source.lower()
            })
        )
    except Exception as e:
        log(job_id, user, EVENT_TYPE, "Error starting state machine for upload workflow.", FIREHOSE_STREAM_NAME, error=str(e), level='error')
        update_job_status(job_id, "FAILED", job_table, FIREHOSE_STREAM_NAME, user=user, event_type=EVENT_TYPE, error_msg=str(e))
        return {"status": "failed", "job_id": job_id, "error": f"Error starting the step function for uploading: {str(e)}"}

    log(job_id, user, EVENT_TYPE, f"Kickoff Lambda started state machine execution {response['executionArn']}", FIREHOSE_STREAM_NAME)

    return {"status": "ok",
            "job_id": job_id,
            "user": user,
            "event_type": EVENT_TYPE,
            "label_types": label_types,
            "num_images":num_images,
            "source":source}