import os
import boto3
from common.logging_utils import log
from common.s3_utils import delete_s3_prefix, parse_s3_uri

s3 = boto3.client("s3")

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# We can tune this constant
MAX_MEMORY_MB = 2048 # from the job definition for validation step
IMAGE_SIZE_MB = 3  # worst-case per image
SAFETY_FACTOR = 0.5  # 0.5 means use only use ~50% of memory for image data

max_images = int((MAX_MEMORY_MB * SAFETY_FACTOR) / IMAGE_SIZE_MB)
IMAGES_PER_BATCH = max(1, min(max_images, 200))

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        label_type = event["label_type"]
        data_source = event["data_source"]
        original_manifest_s3_uri = event["original_manifest_s3_uri"]
    except KeyError as e:
        raise RuntimeError(f"[VAL_FILE_BATCHING] Validation batching Lambda failed: missing required key {e}")

    log(job_id, user, event_type, f"[VAL_FILE_BATCHING] Starting batching of images for image upload validation job id {job_id}.", LOG_FIREHOSE_STREAM_NAME)
    manifest_prefix = f"temp/image-upload/{job_id}/batches/validation-step/manifests/"
    delete_s3_prefix(FILE_BUCKET_NAME, manifest_prefix)

    # Get the json lines from the original manifest
    # original_manifest_s3_uri: s3://bucket/key
    try:
        manifest_bucket, manifest_key = parse_s3_uri(original_manifest_s3_uri)
    except ValueError as e:
        raise RuntimeError(f"[VAL_FILE_BATCHING] Invalid original_manifest_s3_uri: {e}")

    resp = s3.get_object(Bucket=manifest_bucket, Key=manifest_key)

    batch_lines = []
    manifest_uris = []
    total = 0
    idx = 0

    #  This makes memory usage ~O(batch_size) instead of O(file_size).
    for raw in resp["Body"].iter_lines():
        if not raw:
            continue

        # Decode; handle possible BOM on first line
        line = raw.decode("utf-8-sig").strip()
        if not line:
            continue

        total += 1
        batch_lines.append(line)

        if len(batch_lines) >= IMAGES_PER_BATCH:
            idx += 1
            body = ("\n".join(batch_lines) + "\n").encode("utf-8")
            out_key = f"{manifest_prefix}batch-{idx:03d}.jsonl"

            s3.put_object(
                Bucket=FILE_BUCKET_NAME,
                Key=out_key,
                Body=body,
                ContentType="application/x-ndjson",
            )
            manifest_uris.append(f"s3://{FILE_BUCKET_NAME}/{out_key}")
            batch_lines = []

    # flush last partial batch
    if batch_lines:
        idx += 1
        body = ("\n".join(batch_lines) + "\n").encode("utf-8")
        out_key = f"{manifest_prefix}batch-{idx:03d}.jsonl"
        s3.put_object(Bucket=FILE_BUCKET_NAME, Key=out_key, Body=body, ContentType="application/x-ndjson")
        manifest_uris.append(f"s3://{FILE_BUCKET_NAME}/{out_key}")

    result = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "label_type": label_type,
        "data_source":data_source,
        "manifests": manifest_uris,
        "expected_count": total
    }

    msg = f"[VAL_FILE_BATCHING] Done batching {total} total images for image upload validation: label type = {label_type}, {IMAGES_PER_BATCH} images per batch."
    log(job_id, user, event_type, msg, LOG_FIREHOSE_STREAM_NAME)

    return result