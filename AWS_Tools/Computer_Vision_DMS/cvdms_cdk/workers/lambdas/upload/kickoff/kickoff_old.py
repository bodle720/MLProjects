import os
import json
import logging
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
UPLOAD_STATE_MACHINE_ARN = os.environ["UPLOAD_STATE_MACHINE_ARN"]

EVENT_TYPE = "IMAGE_UPLOAD"
ALLOWED_LABEL_TYPES = ["string_labels", "bounding_boxes", "semantic_masks", "instance_annotations"]

sf = boto3.client("stepfunctions")
s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

def log(job_id,
        user,
        message,
        warning = None,
        error = None,
        level = "info"):

    entry = {
        "job_id": job_id,
        "user": user,
        "event_type": EVENT_TYPE,
        "message": message,
        "warning": warning,
        "error": error,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    if level.lower() == "error":
        logger.error(json.dumps(entry))
    else:
        logger.info(json.dumps(entry))

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

def list_label_types(bucket_name: str, job_id: str) -> list:
    """
    Return list of immediate subfolders under temp/image-upload/<job_id>/.
    Example return: ['string_labels', 'semantic_masks']
    Raises ValueError if none of the expected label types are found.
    """
    prefix = f"temp/image-upload/{job_id}/"
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    found = set()
    for page in page_iterator:
        for cp in page.get("CommonPrefixes", []):
            # CommonPrefix looks like "temp/image-upload/<job_id>/string_labels/"
            common = cp.get("Prefix")
            if not common:
                continue
            # extract the folder name after the job prefix
            suffix = common[len(prefix):].rstrip("/")  # e.g. "string_labels"
            if suffix:
                found.add(suffix)

    # Intersect with expected types (ignore other folders like "images/")
    label_types = sorted(found & ALLOWED_LABEL_TYPES)

    if not label_types:
        raise ValueError(f"No label subfolders found for job {job_id}; expected one of {sorted(ALLOWED_LABEL_TYPES)}")

    return label_types

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
        image_keys, allowed_exts={".jpg", ".jpeg", ".png", ".tif", ".tiff"}
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
                log(job_id, user, f"Mismatch for label type {label_type}", error = error_msg, level = 'error')
                raise ValueError(error_msg)

            if len(image_bases) != len(label_bases):
                error_msg = f"Count mismatch for {label_type}. images={len(image_bases)} labels={len(label_bases)}"
                log(job_id, user, f"Count mismatch for label type {label_type}", error = error_msg, level = 'error')
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
                log(job_id, user, f"Mask mismatch for label type {label_type}", error = error_msg, level = 'error')
                raise ValueError(error_msg)

            if len(image_bases) != len(mask_png_bases) or len(image_bases) != len(mask_json_bases):
                error_msg = f"Semantic masks count mismatch. images={len(image_bases)}, pngs={len(mask_png_bases)} jsons={len(mask_json_bases)}"
                log(job_id, user, f"Mask count mismatch for label type {label_type}", error = error_msg, level = 'error')
                raise ValueError(error_msg)

        log(job_id, user, f"Found {len(image_bases)} images and labels for {label_type}")

# ---------- Handler ----------
def handler(event, context):
    try:
        job_table = dynamodb.Table(JOB_TABLE_NAME)
    except ClientError as e:
        return {"status":"failed", "job_id": 'unknown', "error": f"Unable to create DynamoDB job table: {str(e)}"}

    # Guard: ensure there's at least one record
    records = event.get("Records", [])
    if not records:
        log("unknown", "unknown", "No Records in event", error="empty event", level="error")
        return {"status": "failed", "job_id": 'unknown', "error": "No Records in event"}

    # Use the first SQS record (batch_size=1 configured)
    sqs_rec = records[0]

    # SQS message body contains the S3 notification JSON as a string
    body = sqs_rec.get("body")
    if not body:
        log("unknown", "unknown", "SQS record missing body", error=str(sqs_rec), level="error")
        return {"status": "failed", "job_id": 'unknown', "error": "SQS record missing body"}

    try:
        body_json = json.loads(body)
    except Exception as e:
        log("unknown", "unknown", "Failed to parse SQS body as JSON", error=str(e), level="error")
        return {"status": "failed", "job_id": 'unknown', "error": f"Failed to parse SQS body as JSON: {str(e)}"}

    # Expect the S3 notification inside body_json["Records"][0]["s3"]
    s3_records = body_json.get("Records", [])
    if not s3_records:
        log("unknown", "unknown", "No S3 Records inside SQS body", error=str(body_json), level="error")
        return {"status": "failed", "job_id": 'unknown', "error": "No S3 Records inside SQS body"}

    s3_rec = s3_records[0]
    s3_info = s3_rec.get("s3", {})
    bucket = s3_info.get("bucket", {}).get("name")
    key = unquote_plus(s3_info.get("object", {}).get("key", ""))

    if bucket != FILE_BUCKET_NAME:
        log("unknown", "unknown", f"Bucket mismatch: got {bucket}", error=f"expected {FILE_BUCKET_NAME}", level="error")
        return {"status": "failed", "job_id": 'unknown', "error": f"Bucket mismatch in upload kickoff lambda"}

    # Now proceed with your existing logic to fetch job.json, validate, etc.
    try:
        job_id = user = 'unknown'
        obj = s3.get_object(Bucket=bucket, Key=key)
        job_data = json.loads(obj["Body"].read().decode("utf-8"))
        job_id = job_data["job_id"]
        user = job_data["user"]
    except Exception as e:
        log(job_id, user, "Upload Kickoff Lambda could not initialize job_id and user.", error=str(e), level="error")

        if job_id != 'unknown':
            job_status_updated, job_msg = update_job_status(job_id,
                                                          "FAILED",
                                                            job_table,
                                                           error_msg=str(e))

            if not job_status_updated:
                log(job_id, user, job_msg, error=job_msg, level="error")

        return {"status": "failed", "job_id": job_id, "error": f"Upload Kickoff Lambda could not initialize job_id and user: {str(e)}"}

    try:
        expected_key = f"temp/image-upload/{job_id}/job.json"
        if key != expected_key:
            log(job_id,
                user,
                f"The key found ({key}) does not match expected key of {expected_key}",
                error=f"The key found ({key}) does not match expected key of {expected_key}",
                level="error")

            return {"status": "failed", "job_id": job_id, "error": f"job_id in job.json does not match key path: {key} vs {expected_key}"}

        # get the label types for this upload workflow, extracted from tag names in s3
        try:
            label_types = list_label_types(FILE_BUCKET_NAME, job_id)
        except Exception as e:
            return {"status": "failed", "job_id": job_id, "error": f"Error listing label types: {str(e)}"}

        try:
            validate_labels(bucket, job_id, label_types, user)
        except Exception as e:
            return {"status": "failed", "job_id": job_id, "error": f"Error validating labels: {str(e)}"}

        # First try to start the workflow
        response = sf.start_execution(
            stateMachineArn=UPLOAD_STATE_MACHINE_ARN,
            name = f"{job_id}-{int(datetime.now().timestamp() * 1000)}"[:80],
            input=json.dumps({
                "job_id": job_id,
                "user": user,
                "label_types": label_types
            })
        )

        # If we got here, Step Function execution started successfully
        job_status_updated, job_msg = update_job_status(job_id,
                                               "IN_PROGRESS",
                                                        job_table)

        if not job_status_updated:
            log(job_id,
                user,
                job_msg,
                error=job_msg,
                level="error")

        log(job_id,
            user,
            f"Kickoff Lambda started state machine execution {response['executionArn']}. Job status update to IN_PROGRESS succeeded={job_status_updated}")

        return {"status": "ok",
                "job_id": job_id,
                "user": user,
                "event_type": EVENT_TYPE,
                "label_types": label_types}

    except Exception as e:
        job_status_updated, job_msg = update_job_status(job_id,
                                                       "FAILED",
                                                        job_table,
                                                        error_msg=str(e))

        if not job_status_updated:
            log(job_id,
                user,
                job_msg,
                error=job_msg,
                level="error")

        log(job_id,
            user,
            "Kickoff Lambda failed",
            error=str(e),
            level="error")

        return {"status": "failed", "job_id": job_id, "error": str(e)}