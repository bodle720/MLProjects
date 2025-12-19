import os
import json
from datetime import datetime
from urllib.parse import unquote_plus

import boto3

# Lambda layer imports
from common.utils import log

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
UPLOAD_STATE_MACHINE_ARN = os.environ["UPLOAD_STATE_MACHINE_ARN"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
GLOBAL_DLQ_URL = os.environ["GLOBAL_DLQ_URL"]

sf = boto3.client("stepfunctions")
s3 = boto3.client("s3")
sqs = boto3.client("sqs")

def send_to_dlq(job_id, user, event_type, error):
    job_id = job_id or 'unknown'
    user = user or 'unknown'
    event_type = event_type or "IMAGE_UPLOAD"

    try:
        sqs.send_message(
            QueueUrl=GLOBAL_DLQ_URL,
            MessageBody=json.dumps({
                "source": "kickoff",
                "job_id": job_id,
                "user": user,
                "event_type": event_type,
                "error": str(error)
            })
        )
    except Exception as e:
        log(job_id, user, event_type, f"[UPLOAD_KICKOFF] Failed to send to DLQ: {str(error)}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')

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
        raise ValueError(f"[UPLOAD_KICKOFF] Duplicate {kind} detected for basenames: {dups}")

def validate_labels(bucket, job_id, label_type, user, event_type):
    image_keys = list_keys(bucket, f"temp/image-upload/{job_id}/images/")
    image_bases = basenames_from_keys(
        image_keys, allowed_exts={".jpg", ".jpeg", ".png"}
    )

    if not image_bases:
        raise ValueError("[UPLOAD_KICKOFF] No images found for job")

    require_no_duplicates(image_bases, "images")

    if label_type in ["string_labels", "bounding_boxes", "instance_annotations"]:
        label_prefix = f"temp/image-upload/{job_id}/{label_type}/"
        label_keys = list_keys(bucket, label_prefix)
        label_bases = basenames_from_keys(label_keys, allowed_exts={".json"})
        require_no_duplicates(label_bases, f"{label_type}")

        if set(image_bases) != set(label_bases):
            missing_in_labels = sorted(set(image_bases) - set(label_bases))
            extra_in_labels = sorted(set(label_bases) - set(image_bases))
            error_msg = f"[UPLOAD_KICKOFF] Mismatch for {label_type}. Missing labels for: {missing_in_labels}. Extra labels for: {extra_in_labels}"
            raise ValueError(error_msg)

        if len(image_bases) != len(label_bases):
            error_msg = f"[UPLOAD_KICKOFF] Count mismatch for {label_type}. images={len(image_bases)} labels={len(label_bases)}"
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
            error_msg = f"[UPLOAD_KICKOFF] Semantic masks mismatch. PNG missing: {missing_png}, PNG extra: {extra_png}, JSON missing: {missing_json}, JSON extra: {extra_json}"
            raise ValueError(error_msg)

        if len(image_bases) != len(mask_png_bases) or len(image_bases) != len(mask_json_bases):
            error_msg = f"[UPLOAD_KICKOFF] Semantic masks count mismatch. images={len(image_bases)}, pngs={len(mask_png_bases)} jsons={len(mask_json_bases)}"
            raise ValueError(error_msg)

    log(job_id, user, event_type, f"[UPLOAD_KICKOFF] Found {len(image_bases)} images and labels for label type = {label_type}", LOG_FIREHOSE_STREAM_NAME)

def fail(job_id, user, event_type, msg):
    job_id = job_id or "unknown"
    user = user or "unknown"
    event_type = event_type or "IMAGE_UPLOAD"
    send_to_dlq(job_id, user, event_type, msg)
    return {"status": "failed", "job_id": job_id, "user": user, "event_type": event_type}

def handler(event, context):
    job_id = "unknown"
    user = "unknown"
    event_type = "IMAGE_UPLOAD"

    # Guard: ensure there's at least one record
    records = event.get("Records", [])
    if not records:
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: No Records in event")

    # Use the first SQS record (batch_size=1 configured)
    sqs_rec = records[0]

    # SQS message body contains the S3 notification JSON as a string
    body = sqs_rec.get("body")
    if not body:
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: SQS record missing")

    try:
        body_json = json.loads(body)
    except Exception as e:
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: Failed to parse SQS body as JSON: {str(e)}")

    # Expect the S3 notification inside body_json["Records"][0]["s3"]
    s3_records = body_json.get("Records", [])
    if not s3_records:
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: No S3 Records inside SQS body")

    s3_rec = s3_records[0]
    s3_info = s3_rec.get("s3", {})
    bucket = s3_info.get("bucket", {}).get("name")
    key = unquote_plus(s3_info.get("object", {}).get("key", ""))

    # key: "temp/image-upload/<job_id>/job.json"
    if not key.endswith("job.json"):
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Unexpected key: {key}")

    parts = key.split("/")
    if len(parts) >= 3 and parts[0] == "temp" and parts[1] == "image-upload":
        job_id = parts[2]  # use this even before reading the manifest

    if bucket != FILE_BUCKET_NAME:
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: Bucket mismatch in upload kickoff lambda")

    # Now proceed with your existing logic to fetch job.json, validate, etc.
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        job_data = json.loads(obj["Body"].read().decode("utf-8"))
        job_id = job_data["job_id"]
        user = job_data["user"]
        data_source = job_data["data_source"]
        label_type = job_data["label_type"]
        event_type = job_data["event_type"]
    except Exception as e:
        log(job_id, user, "IMAGE_UPLOAD", "[UPLOAD_KICKOFF] Upload Kickoff Lambda could not initialize job_id, user, data_source, event_type, and label_type from manifest", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: could not initialize expected manifest fields: {str(e)}")

    if not isinstance(label_type, str):
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] label_type must be a str, got {type(label_type)}")

    try:
        validate_labels(bucket, job_id, label_type, user, event_type)
    except Exception as e:
        log(job_id, user, event_type, "[UPLOAD_KICKOFF] Error validating labels in kickoff lambda.", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: error validating labels: {str(e)}")

    try:
        response = sf.start_execution(
            stateMachineArn=UPLOAD_STATE_MACHINE_ARN,
            name = f"{job_id}-{int(datetime.now().timestamp() * 1000)}"[:80],
            input=json.dumps({
                "job_id": job_id,
                "user": user,
                "label_type": label_type,
                "data_source":data_source.lower(),
                "event_type":event_type,
            })
        )
    except Exception as e:
        log(job_id, user, event_type, "[UPLOAD_KICKOFF] Error starting state machine for upload workflow.", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: error starting the step function for uploading: {str(e)}")

    log(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda started state machine execution {response['executionArn']}", LOG_FIREHOSE_STREAM_NAME)

    return {"status": "ok",
            "job_id": job_id,
            "user": user,
            "label_type": label_type,
            "data_source":data_source,
            "event_type": event_type}