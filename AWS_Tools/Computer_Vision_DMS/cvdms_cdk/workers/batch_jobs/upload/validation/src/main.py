import os
import io
import json
import hashlib
import time
from datetime import datetime, timezone

from PIL import Image
import boto3
from botocore.exceptions import ClientError

from common.utils import log, chunked_insert

# Env Variables from upload stack
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ["UPLOAD_STAGING_TABLE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# From the map state input
MANIFEST_S3_KEY = os.environ["MANIFEST_S3_KEY"]
JOB_ID = os.environ["JOB_ID"]
USER = os.environ["USER"]
LABEL_TYPE = os.environ["LABEL_TYPE"]
DATA_SOURCE = os.environ["DATA_SOURCE"]
EVENT_TYPE = os.environ["EVENT_TYPE"]

s3 = boto3.client("s3")
athena = boto3.client("athena")

# Image feature calculation helpers
def infer_dtype(img):
    mode = img.mode
    if mode in ("L", "RGB", "RGBA", "CMYK", "YCbCr"):
        return "uint8"
    if mode in ("I;16", "I;16B", "I;16L"):
        return "uint16"
    if mode == "I":
        return "int32"
    if mode == "F":
        return "float32"
    if mode == "1":
        return "bool"
    return str(mode)  # fallback

def validate_labels_presence(image_uuid):
    errors = []
    if LABEL_TYPE in ("string_labels", "bounding_boxes", "instance_annotations"):
        label_json = f"temp/image-upload/{JOB_ID}/{LABEL_TYPE}/{image_uuid}.json"
        try:
            s3.head_object(Bucket=FILE_BUCKET_NAME, Key=label_json)
        except ClientError as e:
            errors.append(f"Missing {LABEL_TYPE} for {image_uuid}: {e}")
    elif LABEL_TYPE == "semantic_masks":
        mask_png = f"temp/image-upload/{JOB_ID}/semantic_masks/{image_uuid}.png"
        mask_json = f"temp/image-upload/{JOB_ID}/semantic_masks/{image_uuid}.json"
        for k in (mask_png, mask_json):
            try:
                s3.head_object(Bucket=FILE_BUCKET_NAME, Key=k)
            except ClientError as e:
                errors.append(f"Missing semantic mask companion {k}: {e}")
    else:
        errors.append(f"Unrecognized label type in validation batch job: {LABEL_TYPE}")

    return errors

# def get_classes_present():
#     pass

# Main image processor
def process_image(image_key):
    # Extract UUID from filename
    image_uuid = os.path.splitext(os.path.basename(image_key))[0]

    row = {'job_id': JOB_ID,
           'image_id': image_uuid,
           "temp_source_ref": f"s3://{FILE_BUCKET_NAME}/{image_key}",
           "copy_to": None,
           "img_type": None,
           "img_height": None,
           "img_width": None,
           "num_channels": None,
           "dtype": None,
           "file_size_mb": 0.0, # a double value
           "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), # Athena preferred format.
           "data_source": DATA_SOURCE,
           "sha256_hash": None,
           "temp_string_labels_path": None,
           "temp_bbox_path": None,
           "temp_semantic_mask_path": None,
           "temp_instance_annotation_path": None,
           "classes_present": None,
           "validation_status": "pending",
           "validation_error": None,
           "dedup_status": "pending"}

    try:
        obj = s3.get_object(Bucket=FILE_BUCKET_NAME, Key=image_key)
    except ClientError as e:
        log(JOB_ID, USER, EVENT_TYPE, f"[VAL_JOB_DEF] Error getting s3 image key: {image_key}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
        row["validation_status"] = "failed"
        row["validation_error"] = str(e)
        return row

    data = obj["Body"].read()
    file_size_mb = round(len(data) / (1024 * 1024), 4)
    row["file_size_mb"] = float(file_size_mb)
    buf = io.BytesIO(data)

    buf.seek(0)
    sha = hashlib.sha256(buf.read()).hexdigest()
    buf.seek(0)

    row["sha256_hash"] = str(sha)

    try:
        img = Image.open(buf)
        img.load()
    except Exception as e:
        log(JOB_ID, USER, EVENT_TYPE, f"[VAL_JOB_DEF] Error using PIL to open image key: {image_key}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
        row["validation_status"] = "failed"
        row["validation_error"] = f"Cannot open image {image_key} in validation batch job: {e}"
        return row

    bands = len(img.getbands())
    if bands not in (1, 3):
        log(JOB_ID, USER, EVENT_TYPE, f"[VAL_JOB_DEF] Invalid band count for image key: {image_key}, count = {bands}, must be 1 or 3.", LOG_FIREHOSE_STREAM_NAME, error=f"Invalid band count: {bands}", level='error')
        row["validation_status"] = "failed"
        row["validation_error"] = f"Invalid band count for image key: {image_key}, count = {bands}, must be 1 or 3."
        return row
    else:
        row["num_channels"] = bands

    dtype = infer_dtype(img)
    width, height = img.size

    row["img_type"] = "L" if bands == 1 else "RGB"
    row["dtype"] = dtype
    row["img_height"] = int(height)
    row["img_width"] = int(width)

    # Validate labels
    label_presence_errors = validate_labels_presence(image_uuid)

    if label_presence_errors:
        log(JOB_ID, USER, EVENT_TYPE, f"[VAL_JOB_DEF] Error validating label presence for {image_key}", LOG_FIREHOSE_STREAM_NAME, error=", ".join(label_presence_errors), level='error')
        row["validation_status"] = "failed"
        row["validation_error"] = ", ".join(label_presence_errors)
        return row

    row["validation_status"] = "passed"
    row["temp_string_labels_path"] = f"temp/image-upload/{JOB_ID}/string_labels/{image_uuid}.json" if LABEL_TYPE == "string_labels" else None
    row["temp_bbox_path"] = f"temp/image-upload/{JOB_ID}/bounding_boxes/{image_uuid}.json" if LABEL_TYPE == "bounding_boxes" else None
    row["temp_semantic_mask_path"] = f"temp/image-upload/{JOB_ID}/semantic_masks/{image_uuid}.png" if LABEL_TYPE == "semantic_masks" else None
    row["temp_instance_annotation_path"] = f"temp/image-upload/{JOB_ID}/instance_annotations/{image_uuid}.json" if LABEL_TYPE == "instance_annotations" else None

    # get classes present
    row['classes_present'] = get_classes_present()

    return row

def read_manifest_with_retry(bucket, key, retries=5, delay=2):
    for attempt in range(retries):
        try:
            return s3.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
            raise

def main():
    bucket, key = MANIFEST_S3_KEY.replace("s3://", "").split("/", 1)
    obj = read_manifest_with_retry(bucket, key)
    manifest = json.loads(obj["Body"].read())
    images = manifest["images"]
    num_images = len(images)
    log(JOB_ID, USER, EVENT_TYPE, f"[VAL_JOB_DEF] Validation batch job starting: Job {JOB_ID} with manifest of {num_images} images, manifest located at {MANIFEST_S3_KEY}", LOG_FIREHOSE_STREAM_NAME)

    rows = []
    failed = 0
    for key in images:
        row = process_image(key)
        rows.append(row)
        if row["validation_status"] != "passed":
            failed += 1

    msg = f"[VAL_JOB_DEF] Image count that failed to process in validation batch job: {failed} images."
    log(JOB_ID, USER, EVENT_TYPE, msg, LOG_FIREHOSE_STREAM_NAME)

    if failed == len(images):
        raise

    if rows:
        all_failed, last_error = chunked_insert(rows,
                                               ICEBERG_DATABASE_NAME,
                                               UPLOAD_STAGING_TABLE_NAME,
                                               ATHENA_WORKGROUP,
                                               ATHENA_OUTPUT_S3,
                                               chunk_size=200)

        if last_error:
            log(JOB_ID, USER, EVENT_TYPE,
                f"[VAL_JOB_DEF] Athena insert failed for an image, and the last error was: {last_error}",
                LOG_FIREHOSE_STREAM_NAME, error=str(last_error), level='error')

        if all_failed:
            # Send to global DLQ.
            raise Exception(f"[VAL_JOB_DEF] Validation batch job failed for all images, total failed = {len(rows)}")

    log(JOB_ID, USER, EVENT_TYPE, f"[VAL_JOB_DEF] Completed processing: {len(rows)} rows written, {failed} failed", LOG_FIREHOSE_STREAM_NAME)

if __name__ == "__main__":
    main()