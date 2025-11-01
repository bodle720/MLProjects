import os
import logging
import json
import boto3
import datetime
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
UPLOAD_STATE_MACHINE_ARN = os.environ["UPLOAD_STATE_MACHINE_ARN"]

EXPECTED_JOB_TYPE = "IMAGE_UPLOAD"
ALLOWED_LABEL_TYPES = ["string_labels", "bounding_boxes", "semantic_masks", "instance_annotations"]

sf = boto3.client("stepfunctions")
s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

# ---------- Logging Helper ----------
def log(job_id, user, job_type, message, level="info"):
    logged_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {
        "job_id": job_id,
        "user": user,
        "job_type": job_type,
        "logged_at": logged_at,
        "message": message,
    }
    if level == "error":
        logger.error(json.dumps(entry))
    else:
        logger.info(json.dumps(entry))

# ---------- DynamoDB Helper ----------
def update_job_status(job_table, job_id, status, error_msg=None):
    valid_statuses = ['PENDING', 'IN_PROGRESS', 'FAILED', 'COMPLETED']
    if status not in valid_statuses:
        log(job_id, "UNK", EXPECTED_JOB_TYPE,
            f"Invalid status value when updating job: {status}", level="error")
        return False

    try:
        job_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #s = :s, #e = :e",
            ExpressionAttributeNames={"#s": "status", "#e": "errors"},
            ExpressionAttributeValues={":s": status, ":e": error_msg},
            ConditionExpression="attribute_exists(job_id)",
        )
        return True
    except ClientError as e:
        log(job_id, "UNK", EXPECTED_JOB_TYPE,
            f"Failed to update job {job_id} to {status}: {e}", level="error")
        return False

# ---------- Validation Helpers ----------
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

# ---------- Validation ----------
def validate_labels(bucket, job_id, label_type):
    image_keys = list_keys(bucket, f"temp/image-upload/{job_id}/images/")
    image_bases = basenames_from_keys(
        image_keys, allowed_exts={".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    require_no_duplicates(image_bases, "images")

    if label_type in ["string_labels", "bounding_boxes", "instance_annotations"]:
        label_prefix = f"temp/image-upload/{job_id}/{label_type}/"
        label_keys = list_keys(bucket, label_prefix)
        label_bases = basenames_from_keys(label_keys, allowed_exts={".json"})
        require_no_duplicates(label_bases, f"{label_type}")

        if set(image_bases) != set(label_bases):
            missing_in_labels = sorted(set(image_bases) - set(label_bases))
            extra_in_labels = sorted(set(label_bases) - set(image_bases))
            raise ValueError(
                f"Name mismatch for {label_type}. "
                f"Missing labels for: {missing_in_labels}; "
                f"Extra labels for: {extra_in_labels}"
            )

        if len(image_bases) != len(label_bases):
            raise ValueError(
                f"Count mismatch for {label_type}. images={len(image_bases)} labels={len(label_bases)}"
            )

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
            raise ValueError(
                "Semantic masks mismatch. "
                f"PNG missing: {missing_png}, PNG extra: {extra_png}; "
                f"JSON missing: {missing_json}, JSON extra: {extra_json}"
            )

        if len(image_bases) != len(mask_png_bases) or len(image_bases) != len(mask_json_bases):
            raise ValueError(
                f"Semantic masks count mismatch. images={len(image_bases)} "
                f"pngs={len(mask_png_bases)} jsons={len(mask_json_bases)}"
            )

# ---------- Handler ----------
def handler(event, context):
    job_id = None
    user = None
    job_table = None
    job_type = EXPECTED_JOB_TYPE

    # Step 1: Parse event and get job_id + job_table
    try:
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        if bucket != FILE_BUCKET_NAME:
            raise ValueError(f"Bucket mismatch: got {bucket} with key {key}, expected {FILE_BUCKET_NAME}")

        obj = s3.get_object(Bucket=bucket, Key=key)
        job_data = json.loads(obj["Body"].read())

        job_id = job_data["job_id"]
        job_table = dynamodb.Table(JOB_TABLE_NAME)

        job_type = job_data["job_type"]
        user = job_data["user"]
        label_type = job_data["label_type"]

    except Exception as e:
        # Fail fast: we don’t have both job_id and job_table, so just log
        log(job_id, user, job_type,
            f"Kickoff Lambda could not initialize job_id/job_table and related attributes: {str(e)}",
            level="error")

        if job_id and job_table:
            # We retrieved the job id and table, but attributes are missing.
            update_job_status(job_table, job_id, "FAILED", str(e))

        raise

    # Step 2: Validations and workflow start
    try:
        if job_type != EXPECTED_JOB_TYPE:
            raise ValueError(f"Unexpected job type: {job_type}")

        if label_type not in ALLOWED_LABEL_TYPES:
            raise ValueError(f"Unexpected label type: {label_type}")

        expected_key = f"temp/image-upload/{job_id}/job.json"
        if key != expected_key:
            raise ValueError(f"job_id in job.json does not match key path: {key} vs {expected_key}")

        validate_labels(bucket, job_id, label_type)

        # First try to start the workflow
        response = sf.start_execution(
            stateMachineArn=UPLOAD_STATE_MACHINE_ARN,
            name=f"{job_id}-{int(datetime.datetime.now().timestamp())}",
            input=json.dumps({
                "job_id": job_id,
                "user": user,
                "job_type": job_type,
                "label_type": label_type
            })
        )

        # If we got here, Step Function execution started successfully
        job_status_updated = update_job_status(job_table, job_id, "IN_PROGRESS")
        log(job_id, user, job_type,
            f"Kickoff Lambda started state machine execution {response['executionArn']}. Job status update to IN_PROGRESS succeeded={job_status_updated}")

        return {"status": "ok",
                "job_id": job_id,
                "user": user,
                "job_type": job_type,
                "label_type": label_type}

    except Exception as e:
        job_status_updated = update_job_status(job_table, job_id, "FAILED", str(e))
        log(job_id, user, job_type,
            f"Kickoff Lambda failed, job status updated to FAILED: {job_status_updated}, error: {str(e)}", "error")
        raise