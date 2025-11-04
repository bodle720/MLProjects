import os, json, io, hashlib, logging, datetime, time
import boto3
from PIL import Image
import imagehash
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Env
BUCKET = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DB = os.environ["ICEBERG_DB"]
UPLOAD_STAGING_TABLE = os.environ["UPLOAD_STAGING_TABLE"]

MANIFEST_S3_KEY = os.environ["MANIFEST_S3_KEY"]
JOB_ID = os.environ["JOB_ID"]
USER = os.environ["USER"]
LABEL_TYPE = os.environ["LABEL_TYPE"]

s3 = boto3.client("s3")
athena = boto3.client("athena")

def compute_phash_values(img):
    if len(img.getbands()) == 1:
        return str(imagehash.phash(img))
    elif len(img.getbands()) == 3:
        r, g, b = img.split()
        return f"{imagehash.phash(r)}|{imagehash.phash(g)}|{imagehash.phash(b)}"
    else:
        raise ValueError(f"Invalid band count {len(img.getbands())}")

def infer_dtype(img):
    mode = img.mode
    if mode in ("L", "RGB"):
        return "uint8"
    if mode.startswith("I;16"):
        return "uint16"
    return mode

def validate_label_presence(image_uuid):
    errors = []
    if LABEL_TYPE in ("string_labels", "bounding_boxes", "instance_annotations"):
        label_json = f"temp/image-upload/{JOB_ID}/{LABEL_TYPE}/{image_uuid}.json"
        try:
            s3.head_object(Bucket=BUCKET, Key=label_json)
        except ClientError as e:
            errors.append(f"Missing {LABEL_TYPE} for {image_uuid}: {e}")
    elif LABEL_TYPE == "semantic_masks":
        mask_png = f"temp/image-upload/{JOB_ID}/semantic_masks/{image_uuid}.png"
        mask_json = f"temp/image-upload/{JOB_ID}/semantic_masks/{image_uuid}.json"
        for k in (mask_png, mask_json):
            try:
                s3.head_object(Bucket=BUCKET, Key=k)
            except ClientError as e:
                errors.append(f"Missing semantic mask companion {k}: {e}")
    return errors

def canonical_copy_target(image_key):
    return f"s3://{BUCKET}/canonical/imagery/{JOB_ID}/{os.path.basename(image_key)}"

def load_manifest():
    bucket, key = MANIFEST_S3_KEY.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())

def to_sql_value(v):
    if v is None:
        return "NULL"
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"

def chunked_insert(rows, chunk_size=200):
    columns = [
        "job_id", "image_id", "temp_source_ref", "copy_to",
        "img_type", "img_height", "img_width", "num_channels", "dtype",
        "file_size_mb", "uploaded_at", "source", "sha256_hash", "phash",
        "temp_string_labels_path", "temp_bbox_path", "temp_semantic_mask_path",
        "temp_instance_annotation_path", "validation_status", "validation_errors",
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
        qid = athena.start_query_execution(
            QueryString=sql,
            ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
            WorkGroup=ATHENA_WORKGROUP
        )["QueryExecutionId"]
        wait_for_athena(qid)

def wait_for_athena(query_execution_id, poll=1.5, timeout=900):
    start = time.time()
    while True:
        resp = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            if state != "SUCCEEDED":
                reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "")
                raise RuntimeError(f"Athena query {query_execution_id} {state}: {reason}")
            return
        if time.time() - start > timeout:
            raise TimeoutError(f"Athena query {query_execution_id} timed out")
        time.sleep(poll)

def process_image(image_key):
    errors = []
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=image_key)
    except ClientError as e:
        return {"row": None, "status": "failed", "errors": [f"s3 get failed: {e}"]}

    data = obj["Body"].read()
    file_size_mb = round(len(data) / (1024 * 1024), 4)
    buf = io.BytesIO(data)

    buf.seek(0)
    sha = hashlib.sha256(buf.read()).hexdigest()
    buf.seek(0)

    try:
        img = Image.open(buf)
        img.load()
    except Exception as e:
        errors.append(f"cannot open: {e}")
        return {"row": None, "status": "failed", "errors": errors}

    bands = len(img.getbands())
    if bands not in (1, 3):
        errors.append(f"invalid band_count {bands}")
    img_type = "L" if bands == 1 else "RGB"
    dtype = infer_dtype(img)
    width, height = img.size

    try:
        ph = compute_phash_values(img)
    except Exception as e:
        errors.append(f"phash error: {e}")
        ph = None

    # Extract UUID from filename
    image_uuid = os.path.splitext(os.path.basename(image_key))[0]

    # Validate labels
    errors.extend(validate_label_presence(image_uuid))

    status = "passed" if not errors else "failed"
    uploaded_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    temp_string = f"temp/image-upload/{JOB_ID}/string_labels/{image_uuid}.json" if LABEL_TYPE == "string_labels" else None
    temp_bbox = f"temp/image-upload/{JOB_ID}/bounding_boxes/{image_uuid}.json" if LABEL_TYPE == "bounding_boxes" else None
    temp_mask = f"temp/image-upload/{JOB_ID}/semantic_masks/{image_uuid}.png" if LABEL_TYPE == "semantic_masks" else None
    temp_inst = f"temp/image-upload/{JOB_ID}/instance_annotations/{image_uuid}.json" if LABEL_TYPE == "instance_annotations" else None

    row = {
        "job_id": JOB_ID,
        "image_id": image_uuid,  # now populated from filename UUID
        "temp_source_ref": f"s3://{BUCKET}/{image_key}",
        "copy_to": canonical_copy_target(image_key),
        "img_type": img_type,
        "img_height": height,
        "img_width": width,
        "num_channels": bands,
        "dtype": dtype,
        "file_size_mb": file_size_mb,
        "uploaded_at": uploaded_at,
        "source": None,
        "sha256_hash": sha,
        "phash": ph,
        "temp_string_labels_path": temp_string,
        "temp_bbox_path": temp_bbox,
        "temp_semantic_mask_path": temp_mask,
        "temp_instance_annotation_path": temp_inst,
        "validation_status": status,
        "validation_errors": json.dumps(errors) if errors else None,
        "dedup_status": "pending",
        "matched_image_id": None,
        "merge_action": "none"
    }
    return {"row": row, "status": status, "errors": errors}

def main():
    manifest = load_manifest()
    images = manifest["images"]
    log.info(f"Job {JOB_ID}: validating {len(images)} images")

    rows = []
    failed = 0
    for key in images:
        res = process_image(key)
        if res["row"]:
            rows.append(res["row"])
        if res["status"] != "passed":
            failed += 1
        log.info(f"{key}: {res['status']} errs={res['errors']}")

    if rows:
        chunked_insert(rows, chunk_size=200)
    log.info(f"Completed: {len(rows)} rows written, {failed} failed")


if __name__ == "__main__":
    main()



