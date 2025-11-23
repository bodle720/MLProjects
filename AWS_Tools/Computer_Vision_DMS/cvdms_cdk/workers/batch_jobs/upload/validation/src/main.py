import os, json, io, hashlib, logging, time
from datetime import datetime, timezone
import boto3
from PIL import Image
import imagehash
from botocore.exceptions import ClientError

from common.utils import log

# Env Variables from upload stack
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DB = os.environ["ICEBERG_DB"]
UPLOAD_STAGING_TABLE = os.environ["UPLOAD_STAGING_TABLE"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# From the map state input
MANIFEST_S3_KEY = os.environ["MANIFEST_S3_KEY"]
JOB_ID = os.environ["JOB_ID"]
USER = os.environ["USER"]
LABEL_TYPES = json.loads(os.environ["LABEL_TYPES"])
SOURCE = os.environ["SOURCE"]
EVENT_TYPE = os.environ["EVENT_TYPE"]

s3 = boto3.client("s3")
athena = boto3.client("athena")

# Image feature calculation helpers
def compute_phash_values(img):
    if len(img.getbands()) == 1:
        return str(imagehash.phash(img))
    elif len(img.getbands()) == 3:
        r, g, b = img.split()
        return f"{imagehash.phash(r)}|{imagehash.phash(g)}|{imagehash.phash(b)}"
    else:
        raise ValueError(f"Invalid band count: {len(img.getbands())}")

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
    for label_type in LABEL_TYPES:
        if label_type in ("string_labels", "bounding_boxes", "instance_annotations"):
            label_json = f"temp/image-upload/{JOB_ID}/{label_type}/{image_uuid}.json"
            try:
                s3.head_object(Bucket=FILE_BUCKET_NAME, Key=label_json)
            except ClientError as e:
                errors.append(f"Missing {label_type} for {image_uuid}: {e}")
        elif label_type == "semantic_masks":
            mask_png = f"temp/image-upload/{JOB_ID}/semantic_masks/{image_uuid}.png"
            mask_json = f"temp/image-upload/{JOB_ID}/semantic_masks/{image_uuid}.json"
            for k in (mask_png, mask_json):
                try:
                    s3.head_object(Bucket=FILE_BUCKET_NAME, Key=k)
                except ClientError as e:
                    errors.append(f"Missing semantic mask companion {k}: {e}")
        else:
            errors.append(f"Unrecognized label type in validation batch job: {label_type}")

    return errors

# Pushing to iceberg table helpers
def to_sql_value(v):
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "ARRAY[" + ", ".join("'" + str(x).replace("'", "''") + "'" for x in v) + "]"
    return "'" + str(v).replace("'", "''") + "'"

def wait_for_athena(query_execution_id, poll=1.5, timeout=900):
    """Poll Athena until query completes or times out. Returns True if succeeded, False otherwise."""
    start = time.time()
    while True:
        try:
            resp = athena.get_query_execution(QueryExecutionId=query_execution_id)
            state = resp["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                if state == "SUCCEEDED":
                    return True
                else:
                    reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "")
                    log(JOB_ID, USER, EVENT_TYPE,
                        f"Athena query {query_execution_id} ended with state {state}: {reason}",
                        LOG_FIREHOSE_STREAM_NAME, error=str(reason), level='error')
                    return False
            if time.time() - start > timeout:
                log(JOB_ID, USER, EVENT_TYPE,
                    f"Athena query {query_execution_id} timed out after {timeout} seconds",
                    LOG_FIREHOSE_STREAM_NAME, error="athena timeout", level='error')
                return False
            time.sleep(poll)
        except Exception as e:
            log(JOB_ID, USER, EVENT_TYPE,
                f"Error polling Athena query {query_execution_id}: {e}",
                LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
            return False

def chunked_insert(rows, chunk_size=200):
    """Insert rows into Iceberg in batches. If a batch fails, retry row-by-row."""
    columns = [
        "job_id", "image_id", "temp_source_ref", "copy_to",
        "img_type", "img_height", "img_width", "num_channels", "dtype",
        "file_size_mb", "uploaded_at", "source", "sha256_hash", "phash",
        "temp_string_labels_path", "temp_bbox_path", "temp_semantic_mask_path",
        "temp_instance_annotation_path", "validation_status", "validation_error",
        "dedup_status", "matched_image_id", "merge_action"
    ]
    table = f'"{ICEBERG_DB}"."{UPLOAD_STAGING_TABLE}"'

    for i in range(0, len(rows), chunk_size):
        batch = rows[i:i+chunk_size]
        values_clause = []
        for r in batch:
            values = [to_sql_value(r.get(c)) for c in columns]
            values_clause.append("(" + ", ".join(values) + ")")
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES " + ", ".join(values_clause)

        # first try to insert the btch, all or nothing is inserted in this athena call
        try:
            qid = athena.start_query_execution(
                QueryString=sql,
                ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
                WorkGroup=ATHENA_WORKGROUP
            )["QueryExecutionId"]

            success = wait_for_athena(qid)
            if not success:
                raise RuntimeError("Batch insert failed")

        except Exception as e:
            log(JOB_ID, USER, EVENT_TYPE,
                f"Athena batch insert failed for batch {i//chunk_size+1}: {e}",
                LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')

            # Retry row-by-row for this batch if batch insert failed due to a bad row.
            for r in batch:
                try:
                    values = [to_sql_value(r.get(c)) for c in columns]
                    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(values)})"
                    qid = athena.start_query_execution(
                        QueryString=sql,
                        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
                        WorkGroup=ATHENA_WORKGROUP
                    )["QueryExecutionId"]
                    wait_for_athena(qid)
                except Exception as row_e:
                    log(JOB_ID, USER, EVENT_TYPE,
                        f"Athena insert failed for image {r.get('image_id')}: {row_e}",
                        LOG_FIREHOSE_STREAM_NAME, error=str(row_e), level='error')

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
           "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), # timestamp, UTC upload time in ISO8601
           "source": SOURCE,
           "sha256_hash": None,
           "phash": None,
           "temp_string_labels_path": None,
           "temp_bbox_path": None,
           "temp_semantic_mask_path": None,
           "temp_instance_annotation_path": None,
           "validation_status": "pending",
           "validation_error": None,
           "dedup_status": "pending",
           "matched_image_id": None,
           "merge_action": None}

    try:
        obj = s3.get_object(Bucket=FILE_BUCKET_NAME, Key=image_key)
    except ClientError as e:
        log(JOB_ID, USER, EVENT_TYPE, f"Error getting s3 image key: {image_key}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
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
        log(JOB_ID, USER, EVENT_TYPE, f"Error using PIL to open image key: {image_key}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
        row["validation_status"] = "failed"
        row["validation_error"] = f"Cannot open image {image_key} in validation batch job: {e}"
        return row

    bands = len(img.getbands())
    if bands not in (1, 3):
        log(JOB_ID, USER, EVENT_TYPE, f"Invalid band count for image key: {image_key}, count = {bands}, must be 1 or 3.", LOG_FIREHOSE_STREAM_NAME, error=f"Invalid band count: {bands}", level='error')
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

    try:
        ph = compute_phash_values(img) # type str
    except Exception as e:
        log(JOB_ID, USER, EVENT_TYPE, f"phash error for image key: {image_key}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level='error')
        row["validation_status"] = "failed"
        row["validation_error"] = f"phash error for image key: {image_key}"
        return row
    else:
        row["phash"] = ph

    # Validate labels
    label_presence_errors = validate_labels_presence(image_uuid)

    if label_presence_errors:
        log(JOB_ID, USER, EVENT_TYPE, f"Error validating label presence for {image_key}", LOG_FIREHOSE_STREAM_NAME, error=", ".join(label_presence_errors), level='error')
        row["validation_status"] = "failed"
        row["validation_error"] = ", ".join(label_presence_errors)
        return row

    log(JOB_ID, USER, EVENT_TYPE, f"Successfully validated image {image_key}", LOG_FIREHOSE_STREAM_NAME)
    row["validation_status"] = "passed"
    row["temp_string_labels_path"] = f"temp/image-upload/{JOB_ID}/string_labels/{image_uuid}.json" if "string_labels" in LABEL_TYPES else None
    row["temp_bbox_path"] = f"temp/image-upload/{JOB_ID}/bounding_boxes/{image_uuid}.json" if "bounding_boxes" in LABEL_TYPES else None
    row["temp_semantic_mask_path"] = f"temp/image-upload/{JOB_ID}/semantic_masks/{image_uuid}.png" if "semantic_masks" in LABEL_TYPES else None
    row["temp_instance_annotation_path"] = f"temp/image-upload/{JOB_ID}/instance_annotations/{image_uuid}.json" if "instance_annotations" in LABEL_TYPES else None

    row["copy_to"] = f"s3://{FILE_BUCKET_NAME}/canonical/imagery/{os.path.basename(image_key)}"

    return row

def main():

    bucket, key = MANIFEST_S3_KEY.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    manifest = json.loads(obj["Body"].read())
    images = manifest["images"]
    num_images = len(images)
    log(JOB_ID, USER, EVENT_TYPE, f"Validation batch job starting: Job {JOB_ID} with manifest of {num_images} images, manifest located at {MANIFEST_S3_KEY}", LOG_FIREHOSE_STREAM_NAME)

    rows = []
    failed = 0
    for key in images:
        row = process_image(key)
        rows.append(row)
        if row["validation_status"] != "passed":
            failed += 1

    msg = f"Failed to process {failed} images in validation batch job."
    log(JOB_ID, USER, EVENT_TYPE, msg, LOG_FIREHOSE_STREAM_NAME)

    if rows:
        chunked_insert(rows, chunk_size=200)

    log(JOB_ID, USER, EVENT_TYPE, f"Completed processing: {len(rows)} rows written, {failed} failed", LOG_FIREHOSE_STREAM_NAME)

if __name__ == "__main__":
    main()