import os
import json
import boto3
from math import ceil

s3 = boto3.client("s3")

BUCKET = os.environ["FILE_BUCKET_NAME"]
# You can tune this constant
MAX_MEMORY_MB = 2048 # from the job definition for validation step
IMAGE_SIZE_MB = 3  # worst-case per image
SAFETY_FACTOR = 0.5  # only use ~50% of memory for image data

max_images = int((MAX_MEMORY_MB * SAFETY_FACTOR) / IMAGE_SIZE_MB)
IMAGES_PER_BATCH = min(max_images, 200)  # cap at 200 for sanity

def handler(event, context):
    job_id = event["job_id"]
    user = event["user"]
    job_type = event["job_type"]
    label_type = event["label_type"]

    # Images are assumed to be under temp/image-upload/{job_id}/images/
    prefix = f"temp/image-upload/{job_id}/images/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)

    if "Contents" not in resp:
        raise RuntimeError(f"No images found under {prefix}")

    # Collect all image keys
    image_keys = [obj["Key"] for obj in resp["Contents"] if not obj["Key"].endswith("/")]

    # Chunk into batches
    batches = [
        image_keys[i:i + IMAGES_PER_BATCH]
        for i in range(0, len(image_keys), IMAGES_PER_BATCH)
    ]

    manifest_keys = []

    for idx, batch in enumerate(batches, start=1):
        manifest = {"images": batch}
        manifest_key = f"temp/image-upload/{job_id}/batches/batch-{idx:03d}.json"

        s3.put_object(
            Bucket=BUCKET,
            Key=manifest_key,
            Body=json.dumps(manifest).encode("utf-8"),
            ContentType="application/json"
        )

        manifest_keys.append(f"s3://{BUCKET}/{manifest_key}")

    return {
        "job_id": job_id,
        "user": user,
        "job_type": job_type,
        "label_type": label_type,
        "manifests": manifest_keys
    }
