# -*- coding: utf-8 -*-
import boto3, logging
from helpers import (
    load_image, validate_bands, compute_sha256,
    compute_phashes, extract_features
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

logger.info(json.dumps({
    "job_id": 12345,
    "status": "error",
    "reason": "Something went wrong"
}))

def job_update(job_id, status, summary=None):
    """Update job status in the Job table."""
    expr = "SET job_status = :s, updated_at = :t"
    values = {
        ":s": {"S": status},
        ":t": {"S": datetime.utcnow().isoformat()}
    }
    if summary:
        expr += ", job_summary = :sum"
        values[":sum"] = {"S": summary}
    ddb.update_item(
        TableName=DDB_JOB_TABLE,
        Key={"job_id": {"S": job_id}},
        UpdateExpression=expr,
        ExpressionAttributeValues=values
    )

def lambda_handler(event, context):
    bucket, key = parse_event(event)  # implement this
    obj = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = obj["Body"].read()

    img = load_image(file_bytes)
    if img is None:
        write_staging_record(key, error="load_failed")
        return {"status": "error"}

    if not validate_bands(img):
        write_staging_record(key, error="invalid_band")
        return {"status": "error"}

    sha256 = compute_sha256(file_bytes)
    if sha256_exists(sha256):  # implement this
        write_staging_record(key, sha256=sha256, duplicate=True)
        return {"status": "duplicate"}

    phashes = compute_phashes(img)
    features = extract_features(img)

    write_staging_record(key, sha256=sha256, phashes=phashes, features=features)
    return {"status": "ok", "sha256": sha256}

