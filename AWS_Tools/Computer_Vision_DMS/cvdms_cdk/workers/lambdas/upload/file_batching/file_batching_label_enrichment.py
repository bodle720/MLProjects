import os
import time
import boto3
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

UPLOAD_STAGING_TABLE = os.environ["UPLOAD_STAGING_TABLE"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]

def _query_staging_by_job_id(table, job_id: str):
    # Prefer a GSI on job_id; otherwise, replace with scan
    resp = table.query(
        IndexName="job_id-index",  # adjust if needed
        KeyConditionExpression=Key("job_id").eq(job_id)
    )
    items = resp.get("Items", [])
    while "LastEvaluatedKey" in resp:
        resp = table.query(
            IndexName="job_id-index",
            KeyConditionExpression=Key("job_id").eq(job_id),
            ExclusiveStartKey=resp["LastEvaluatedKey"]
        )
        items.extend(resp.get("Items", []))
    return items

def handler(event, context):
    """
    Input event:
    {
      "job_id": "...",
      "user": "...",
      "img_type": "...",
      "label_type": "single_label|multi_label|file"
    }
    """
    job_id = event.get("job_id")
    user = event.get("user")
    img_type = event.get("img_type")
    label_type = event.get("label_type")

    if not job_id or not user:
        raise ValueError("Missing job_id or user")

    staging_table = dynamodb.Table(UPLOAD_STAGING_TABLE)
    items = _query_staging_by_job_id(staging_table, job_id)

    manifests = []
    skipped = 0

    for it in items:
        if not it.get("duplicate_external"):
            continue  # only care about external duplicates

        temp_img_key = it.get("s3_key_temp")
        phash = it.get("phash")
        label_val = it.get("label_value")
        item_label_type = it.get("label_type", label_type)

        if not temp_img_key:
            skipped += 1
            continue

        # Build manifest entry for downstream batch job
        manifest_entry = {
            "job_id": job_id,
            "user": user,
            "img_type": img_type or it.get("img_type"),
            "phash": phash,
            "temp_img_key": temp_img_key,
            "label_type": item_label_type,
            "label_value": label_val,
            # include canonical reference if available
            "canonical_ref": it.get("canonical_ref"),
        }
        manifests.append(manifest_entry)

    result = {
        "job_id": job_id,
        "user": user,
        "img_type": img_type,
        "label_type": label_type,
        "manifests": manifests,
        "skipped_non_external": skipped,
        "batch_count": len(manifests),
        "created_at": int(time.time())
    }

    return result
