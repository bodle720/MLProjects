import os

from common.logging_utils import log
from common.s3_utils import delete_s3_prefix, parse_s3_uri, write_s3_obj, read_obj_with_retry

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[VAL_FILE_BATCHING]"

# We can tune these constants
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
        raise RuntimeError(f"{TASK_NAME} Validation batching Lambda failed: missing required key {e}")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} Starting batching of images for image upload validation job id {job_id}.")
    manifest_prefix = f"temp/image-upload/{job_id}/batches/validation-step/manifests/"
    main_prefix = f"temp/image-upload/{job_id}/batches/validation-step/"

    delete_s3_prefix(FILE_BUCKET_NAME, main_prefix, TASK_NAME)

    # Get the json lines from the original manifest
    # original_manifest_s3_uri: s3://bucket/key
    try:
        manifest_bucket, manifest_key = parse_s3_uri(original_manifest_s3_uri, TASK_NAME)
    except ValueError as e:
        raise RuntimeError(f"{TASK_NAME} Invalid original_manifest_s3_uri: {e}")

    resp = read_obj_with_retry(manifest_bucket, manifest_key, TASK_NAME)

    if resp is None:
        raise RuntimeError(f"{TASK_NAME} unable to load s3://{manifest_bucket}/{manifest_key} after retries")

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
            content = "\n".join(batch_lines) + "\n"
            out_key = f"{manifest_prefix}batch-{idx:03d}.jsonl"
            uri = write_s3_obj(FILE_BUCKET_NAME, out_key, content, "application/x-ndjson", TASK_NAME)
            manifest_uris.append(uri)
            batch_lines = []

    # flush last partial batch
    if batch_lines:
        idx += 1
        content = "\n".join(batch_lines) + "\n"
        out_key = f"{manifest_prefix}batch-{idx:03d}.jsonl"
        uri = write_s3_obj(FILE_BUCKET_NAME, out_key, content, "application/x-ndjson", TASK_NAME)
        manifest_uris.append(uri)

    result = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "label_type": label_type,
        "data_source":data_source,
        "manifests": manifest_uris,
        "expected_count": total
    }

    msg = f"{TASK_NAME} Done batching {total} total images for image upload validation: label type = {label_type}, {IMAGES_PER_BATCH} images per batch."
    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, msg)

    return result