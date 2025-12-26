import os
import io
import uuid
import hashlib
import logging
import json

from datetime import datetime, timezone

import boto3
from PIL import Image
from botocore.exceptions import ClientError

from common.utils import chunked_insert
from helpers import read_manifest_with_retry, infer_dtype, create_and_save_labels

# Env Variables from upload stack
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ["UPLOAD_STAGING_TABLE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# From the map state input
MANIFEST_S3_URI = os.environ["MANIFEST_S3_URI"].strip()
JOB_ID = os.environ["JOB_ID"]
USER = os.environ["USER"]
LABEL_TYPE = os.environ["LABEL_TYPE"]
DATA_SOURCE = os.environ["DATA_SOURCE"]
EVENT_TYPE = os.environ["EVENT_TYPE"]

s3 = boto3.client("s3")
athena = boto3.client("athena")

if not MANIFEST_S3_URI.startswith("s3://") or MANIFEST_S3_URI.count("/") < 3:
    raise ValueError(f"Invalid MANIFEST_S3_URI: {MANIFEST_S3_URI}")

# Main image processor
def process_image(line):
    # Assign the image a uuid
    image_uuid = str(uuid.uuid4())

    # Get the source ref for the image from the line, note image is not necessarily in the file bucket at this point.
    temp_source_ref = line["source-ref"] # s3 uri of image, e.g. "s3://name-of-some-random-bucket/samples/coco/val2017/random-30-images/000000030828.jpg"
    bucket, key = temp_source_ref[5:].split("/", 1)  # remove "s3://"

    # Set up defaults for each column in upload staging table.
    row = {'job_id': JOB_ID,
           'image_id': image_uuid,
           "temp_source_ref": temp_source_ref,
           "img_type": None,
           "img_height": None,
           "img_width": None,
           "num_channels": None,
           "dtype": None,
           "file_size_mb": 0.0, # a double value
           "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), # Athena preferred format.
           "data_source": DATA_SOURCE,
           "sha256_hash": None,
           "string_labels": None,
           "temp_source_ref_bbox_meta": None,
           "temp_source_ref_semantic_png": None,
           "temp_source_ref_semantic_meta": None,
           "temp_source_ref_instance_png": None,
           "temp_source_ref_instance_meta": None,
           "classes_present": None,
           "validation_status": "pending",
           "validation_error": None,
           "dedup_status": "pending",
           "dedup_error": None,
           "registration_status": "pending",
           "registration_error": None,
           "matched_image_id": None}

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
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
        row["validation_status"] = "failed"
        row["validation_error"] = f"Cannot open image {temp_source_ref} in validation batch job: {e}"
        return row

    bands = len(img.getbands())
    if bands not in (1, 3):
        row["validation_status"] = "failed"
        row["validation_error"] = f"Invalid band count for image: {temp_source_ref}, count = {bands}, must be 1 or 3."
        return row
    else:
        row["num_channels"] = bands

    dtype = infer_dtype(img)
    width, height = img.size

    row["img_type"] = "L" if bands == 1 else "RGB"
    row["dtype"] = dtype
    row["img_height"] = int(height)
    row["img_width"] = int(width)

    # Here we can, depending on label type (if not "single-label" or "multi-label"), do the work to both create and move
    # the label files (bbox json or png masks and mask mappings) to the temp/ folder with newly created label uuids
    # per the following format:
    # for object detection:
    #    temp/image-upload/<job uuid>/object-detection/<label uuid>.json
    # or, for semantic segmentation:
    #    temp/image-upload/<job uuid>/semantic-segmentation/<label uuid>.png AND
    #    temp/image-upload/<job uuid>/semantic-segmentation/<label uuid>.json
    # or, for instance segmentation
    #    temp/image-upload/<job uuid>/instance-segmentation/<label uuid>.png
    #    temp/image-upload/<job uuid>/instance-segmentation/<label uuid>.json

    label_cols = {
        "single-label": [],
        "multi-label": [],
        "object-detection": ["temp_source_ref_bbox_meta"],
        "semantic-segmentation": ["temp_source_ref_semantic_png", "temp_source_ref_semantic_meta"],
        "instance-segmentation": ["temp_source_ref_instance_png", "temp_source_ref_instance_meta"],
    }

    col_names = label_cols.get(LABEL_TYPE)
    if col_names is None:
        row["validation_status"] = "failed"
        row["validation_error"] = f"Unsupported LABEL_TYPE: {LABEL_TYPE}"
        return row

    # returns a list and a str, must make sure the order of paths (if not obj detection) corresponds to png first, then meta mask map file as json
    paths, classes_present, error_msg = create_and_save_labels(line, LABEL_TYPE, JOB_ID, FILE_BUCKET_NAME) # paths is a list, even if single element (for object detection

    if error_msg:
        row["validation_status"] = "failed"
        row["validation_error"] = f"Unable to form and save label file(s): {error_msg}"
        return row

    if not classes_present:
        row["validation_status"] = "failed"
        row["validation_error"] = "Empty list of classes_present"
        return row

    row['classes_present'] = classes_present

    if LABEL_TYPE in ("single-label", "multi-label"):
        row["string_labels"] = classes_present

    if col_names:
        if len(paths) != len(col_names):
            row["validation_status"] = "failed"
            row["validation_error"] = f"Expected {len(col_names)} label paths for {LABEL_TYPE}, got {len(paths)}"
            return row

        for col_name, path in zip(col_names, paths):
            row[col_name] = path

    row["validation_status"] = "passed"

    return row

def main():
    bucket, key = MANIFEST_S3_URI[5:].split("/", 1)  # remove "s3://"

    obj = read_manifest_with_retry(bucket, key)
    if not obj:
        error_msg = f"[VAL_JOB_DEF] Could not read manifest file with bucket = {bucket} and key = {key}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    # Read in the json lines
    body = obj["Body"].read().decode("utf-8-sig")
    json_lines = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s:
            continue
        json_lines.append(json.loads(s))

    if len(json_lines) == 0:
        error_msg = f"[VAL_JOB_DEF] There are no images in {JOB_ID} for this batch job."
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    rows = []
    failed = 0
    for line in json_lines:
        row = process_image(line)
        rows.append(row)
        if row["validation_status"] != "passed":
            failed += 1

    all_failed, last_error = chunked_insert(rows,
                                           ICEBERG_DATABASE_NAME,
                                           UPLOAD_STAGING_TABLE_NAME,
                                           ATHENA_WORKGROUP,
                                           ATHENA_OUTPUT_S3,
                                           chunk_size=200)

    if last_error:
        error_msg = f"[VAL_JOB_DEF] Athena insert failed for an image, and the last error was: {last_error}"
        logging.error(error_msg)

    if all_failed:
        error_msg = f"[VAL_JOB_DEF] Validation batch job failed to upload to upload staging table for all images, total failed = {len(rows)}"
        logging.error(error_msg)
        raise Exception(error_msg)

    logging.info(f"[VAL_JOB_DEF] Completed processing: {len(rows)} rows written, {failed} failed validation")

if __name__ == "__main__":
    main()