import os
import boto3
from boto3.dynamodb.conditions import Key

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
UPLOAD_STAGING_TABLE = os.environ["UPLOAD_STAGING_TABLE"]

def _delete_s3_prefix(bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=obj["Key"])

def _delete_staging_rows(table, job_id: str):
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

    for it in items:
        # Assuming PK is (job_id, s3_key_temp) or similar
        key = {"job_id": it["job_id"], "s3_key_temp": it["s3_key_temp"]}
        table.delete_item(Key=key)

def handler(event, context):
    """
    Input event:
    {
      "job_id": "...",
      "user": "..."
    }
    """
    job_id = event.get("job_id")
    if not job_id:
        raise ValueError("Missing job_id")

    # 1. Delete S3 temp files
    prefix = f"temp/uploads/{job_id}/"
    _delete_s3_prefix(FILE_BUCKET_NAME, prefix)

    # 2. Delete staging table rows
    staging_table = dynamodb.Table(UPLOAD_STAGING_TABLE)
    _delete_staging_rows(staging_table, job_id)

    return {
        "job_id": job_id,
        "cleanup_done": True
    }
