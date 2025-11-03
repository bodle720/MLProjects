import os
import json
import boto3
import logging
import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

BUCKET = os.environ["FILE_BUCKET_NAME"]
# You can tune this constant
MAX_MEMORY_MB = 2048 # from the job definition for validation step
IMAGE_SIZE_MB = 3  # worst-case per image
SAFETY_FACTOR = 0.5  # only use ~50% of memory for image data

max_images = int((MAX_MEMORY_MB * SAFETY_FACTOR) / IMAGE_SIZE_MB)
IMAGES_PER_BATCH = min(max_images, 200)  # cap at 200 for sanity

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

def handler(event, context):

    job_id = user = 'UNKNOWN'

    try:
        job_id = event["job_id"]
        user = event["user"]
        job_type = event["job_type"]
        label_type = event["label_type"]
    except KeyError as e:
        log(job_id, user, "IMAGE_UPLOAD", f"Missing key(s) for batching event on image upload: {e}.", level="error")
        raise

    log(job_id, user, job_type, f"Starting batching of images for image upload validation job id {job_id}.")

    # Images are assumed to be under temp/image-upload/{job_id}/images/
    image_keys = []
    continuation_token = None
    prefix = f"temp/image-upload/{job_id}/images/"
    kwargs = {"Bucket": BUCKET, "Prefix": prefix}
    iteration_cap = 9_000_000 # for safety, likely unneeded
    iteration_count = 0
    while True and (iteration_count < iteration_cap):
        iteration_count += 1
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = s3.list_objects_v2(**kwargs)
        image_keys.extend([obj["Key"] for obj in resp.get("Contents", []) if not obj["Key"].endswith("/")])
        if not resp.get("IsTruncated"):
            break
        continuation_token = resp["NextContinuationToken"]

    if len(image_keys) == 0:
        log(job_id, user, "IMAGE_UPLOAD", f"No images found under {prefix}", level="error")
        raise

    # Chunk into batches
    batches = [
        image_keys[i:i + IMAGES_PER_BATCH]
        for i in range(0, len(image_keys), IMAGES_PER_BATCH)
    ]

    manifest_keys = []
    for idx, batch in enumerate(batches, start=1):
        manifest = {"images": batch}
        manifest_key = f"temp/image-upload/{job_id}/batches/validation-step/batch-{idx:03d}.json"

        s3.put_object(
            Bucket=BUCKET,
            Key=manifest_key,
            Body=json.dumps(manifest).encode("utf-8"),
            ContentType="application/json"
        )

        manifest_keys.append(f"s3://{BUCKET}/{manifest_key}")

    log(job_id, user, job_type, f"Done batching {len(image_keys)} total images for image upload validation: label type = {label_type}, manifest counts: {len(manifest_keys)} and {IMAGES_PER_BATCH} images per batch.")

    return {
        "job_id": job_id,
        "user": user,
        "job_type": job_type,
        "label_type": label_type,
        "manifests": manifest_keys
    }