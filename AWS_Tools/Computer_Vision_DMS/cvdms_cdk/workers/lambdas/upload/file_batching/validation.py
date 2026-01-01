import os

import boto3

from common.utils import log

s3 = boto3.client("s3")

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# We can tune this constant
MAX_MEMORY_MB = 2048 # from the job definition for validation step
IMAGE_SIZE_MB = 3  # worst-case per image
SAFETY_FACTOR = 0.5  # 0.5 means use only use ~50% of memory for image data

max_images = int((MAX_MEMORY_MB * SAFETY_FACTOR) / IMAGE_SIZE_MB)
IMAGES_PER_BATCH = min(max_images, 200)  # cap at 200 for sanity

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

    # Get the json lines from the original manifest
    # original_manifest_s3_uri: s3://bucket/key
    if not isinstance(original_manifest_s3_uri, str) or not original_manifest_s3_uri.startswith("s3://"):
        raise RuntimeError(f"[VAL_FILE_BATCHING] Invalid original_manifest_s3_uri: {original_manifest_s3_uri}")

    rest = original_manifest_s3_uri[5:]
    if "/" not in rest:
        raise RuntimeError(f"[VAL_FILE_BATCHING] original_manifest_s3_uri missing key: {original_manifest_s3_uri}")

    manifest_bucket, manifest_key = rest.split("/", 1)

    resp = s3.get_object(Bucket=manifest_bucket, Key=manifest_key)
    raw_text = resp["Body"].read().decode("utf-8-sig")

    # Keep only non-empty lines (client already ensured 1 JSON object per line)
    json_lines = [ln for ln in raw_text.splitlines() if ln.strip()]

    if not json_lines:
        raise RuntimeError(f"[VAL_FILE_BATCHING] Original manifest is empty or only blank lines: {original_manifest_s3_uri}")

    # Chunk into batches
    batches = [
        json_lines[i:i + IMAGES_PER_BATCH]
        for i in range(0, len(json_lines), IMAGES_PER_BATCH)
    ]

    manifest_uris = []
    for idx, batch in enumerate(batches, start=1):
        # make the manifest from the batch (JSONL content)
        # each element in `batch` is already a JSON string line

        manifest_body = ("\n".join(batch) + "\n").encode("utf-8")
        manifest_key = f"temp/image-upload/{job_id}/batches/validation-step/manifests/batch-{idx:03d}.jsonl"

        s3.put_object(
            Bucket=FILE_BUCKET_NAME,
            Key=manifest_key,
            Body=manifest_body,
            ContentType="application/x-ndjson"
        )

        manifest_uris.append(f"s3://{FILE_BUCKET_NAME}/{manifest_key}")

    result = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "label_type": label_type,
        "data_source":data_source,
        "manifests": manifest_uris,
        "expected_count": len(json_lines)
    }

    msg = f"[VAL_FILE_BATCHING] Done batching {len(json_lines)} total images for image upload validation: label type = {label_type}, {IMAGES_PER_BATCH} images per batch."
    log(job_id, user, event_type, msg, LOG_FIREHOSE_STREAM_NAME)

    return result