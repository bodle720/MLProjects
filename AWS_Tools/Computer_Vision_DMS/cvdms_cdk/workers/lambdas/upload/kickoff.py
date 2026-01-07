import os
import json
from datetime import datetime
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

# Lambda layer imports, add to path to avoid pycharm complaining.
from common.logging_utils import log

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
UPLOAD_STATE_MACHINE_ARN = os.environ["UPLOAD_STATE_MACHINE_ARN"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
GLOBAL_DLQ_URL = os.environ["GLOBAL_DLQ_URL"]

sf = boto3.client("stepfunctions")
s3 = boto3.client("s3")
sqs = boto3.client("sqs")

ALLOWED_LABEL_TYPES = {
    "single-label",
    "multi-label",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
}

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

def fail(job_id, user, event_type, msg):
    job_id = job_id or "unknown"
    user = user or "unknown"
    event_type = event_type or "IMAGE_UPLOAD"
    send_to_dlq(job_id, user, event_type, msg)
    return {"status": "failed", "job_id": job_id, "user": user, "event_type": event_type}

def _parse_s3_uri(uri: str):
    # uri: s3://bucket/key
    rest = uri[5:]

    if "/" in rest:
        b, k = rest.split("/", 1)
        return b, k
    else:
        return None, None

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
        event_type = job_data.get("event_type", "IMAGE_UPLOAD")
        label_type = job_data["label_type"]
        data_source = job_data["data_source"]
        registration_time = job_data["registration_time"]
        original_manifest_s3_uri = job_data["original_manifest_s3_uri"]
    except Exception as e:
        log(job_id, user, "IMAGE_UPLOAD", "[UPLOAD_KICKOFF] Upload Kickoff Lambda could not initialize job_id, user, data_source, event_type, and/or label_type from manifest", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: could not initialize expected manifest fields: {str(e)}")

    if not isinstance(label_type, str) or label_type not in ALLOWED_LABEL_TYPES:
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Invalid label_type: {label_type}")

    if not isinstance(original_manifest_s3_uri, str) or not original_manifest_s3_uri.startswith("s3://"):
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Invalid S3 URI original_manifest_s3_uri: {original_manifest_s3_uri}")

    manifest_bucket, manifest_key = _parse_s3_uri(original_manifest_s3_uri)

    if manifest_bucket is None:
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: Manifest missing key: {original_manifest_s3_uri}")

    if manifest_bucket != FILE_BUCKET_NAME:
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: Manifest bucket mismatch in upload kickoff lambda, manifest is in bucket: {manifest_bucket}, expected {FILE_BUCKET_NAME}")

    if manifest_key != f"temp/image-upload/{job_id}/{job_id}.manifest":
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: Manifest key incorrect: got {manifest_key}, expected temp/image-upload/{job_id}/{job_id}.manifest")

    # Make sure we have <job id>.manifest located next to job.json: "temp/image-upload/<job_id>/<job_id>.manifest"
    try:
        head = s3.head_object(Bucket=manifest_bucket, Key=manifest_key)
        if head.get("ContentLength", 0) <= 0:
            return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Manifest is empty: {original_manifest_s3_uri}")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Missing manifest next to job.json: {original_manifest_s3_uri}")
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] head_object failed for manifest: {original_manifest_s3_uri}: {e}")

    # Start the upload step function.
    try:
        response = sf.start_execution(
            stateMachineArn=UPLOAD_STATE_MACHINE_ARN,
            name = f"{job_id}-{int(datetime.now().timestamp() * 1000)}"[:80],
            input=json.dumps({
                "job_id": job_id,
                "user": user,
                "event_type": event_type,
                "label_type": label_type,
                "data_source": str(data_source).lower(),
                "original_manifest_s3_uri": original_manifest_s3_uri,
                "registration_time": registration_time
            })
        )
    except Exception as e:
        log(job_id, user, event_type, "[UPLOAD_KICKOFF] Error starting state machine for upload workflow.", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
        return fail(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda failed: error starting the step function for uploading: {str(e)}")

    log(job_id, user, event_type, f"[UPLOAD_KICKOFF] Kickoff Lambda started state machine execution {response['executionArn']}", LOG_FIREHOSE_STREAM_NAME)

    return {
        "status": "ok",
        "job_id": job_id,
        "user": user,
        "label_type": label_type,
        "event_type": event_type,
        "data_source": data_source.lower(),
        "original_manifest_s3_uri": original_manifest_s3_uri
    }