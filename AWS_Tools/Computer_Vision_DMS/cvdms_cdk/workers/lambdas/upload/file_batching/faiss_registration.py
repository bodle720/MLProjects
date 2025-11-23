import os
import json
import time
import uuid
from typing import Dict, Any, List, Optional

import boto3
from boto3.dynamodb.conditions import Key

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
UPLOAD_STAGING_TABLE = os.environ["UPLOAD_STAGING_TABLE"]
CANONICAL_IMAGERY_TABLE = os.environ["CANONICAL_IMAGERY_TABLE"]

# Helpers

def _canonical_image_key(user: str, job_id: str, temp_key: str) -> str:
    # Derive canonical path preserving filename
    basename = temp_key.split("/")[-1]
    return f"canonical/imagery/{user}/{job_id}/{basename}"

def _canonical_label_key(user: str, job_id: str, label_temp_key: str) -> str:
    label_basename = label_temp_key.split("/")[-1]
    return f"canonical/labels/{user}/{job_id}/{label_basename}"

def _copy_then_delete(src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
    s3.copy_object(
        Bucket=dst_bucket,
        Key=dst_key,
        CopySource={"Bucket": src_bucket, "Key": src_key}
    )
    s3.delete_object(Bucket=src_bucket, Key=src_key)

def _delete_if_exists(bucket: str, key: str) -> None:
    try:
        s3.delete_object(Bucket=bucket, Key=key)
    except s3.exceptions.NoSuchKey:
        pass
    except Exception:
        # best-effort cleanup; avoid failing the whole job on cleanup
        pass

def _put_canonical_row(table, item: Dict[str, Any]) -> None:
    table.put_item(Item=item)

def _query_staging_by_job_id(table, job_id: str) -> List[Dict[str, Any]]:
    # Prefer a GSI/partitioned by job_id. If not, change to scan with filter.
    resp = table.query(
        IndexName="job_id-index",  # adjust if needed
        KeyConditionExpression=Key("job_id").eq(job_id)
    )
    items = resp.get("Items", [])
    # Handle pagination if necessary
    while "LastEvaluatedKey" in resp:
        resp = table.query(
            IndexName="job_id-index",
            KeyConditionExpression=Key("job_id").eq(job_id),
            ExclusiveStartKey=resp["LastEvaluatedKey"]
        )
        items.extend(resp.get("Items", []))
    return items

def _is_true(val: Optional[Any]) -> bool:
    return bool(val) is True

# Lambda handler

def handler(event, context):
    """
    Expected event:
    {
      "job_id": "...",
      "user": "...",
      "img_type": "rgb|ir|...",   # optional passthrough
      "label_type": "single_label|multi_label|file",  # as produced upstream
      ... other fields
    }
    """
    job_id = event.get("job_id")
    user = event.get("user")
    img_type = event.get("img_type")  # passthrough to downstream if needed
    label_type = event.get("label_type")

    if not job_id or not user:
        raise ValueError("Missing required fields: job_id and user")

    staging_table = dynamodb.Table(UPLOAD_STAGING_TABLE)
    canonical_table = dynamodb.Table(CANONICAL_IMAGERY_TABLE)

    # Collect survivors and perform actions
    items = _query_staging_by_job_id(staging_table, job_id)

    manifests: List[Dict[str, Any]] = []
    registered_count = 0
    deleted_internal_dups = 0
    skipped_external_dups = 0

    for it in items:
        temp_img_key = it.get("s3_key_temp")
        phash = it.get("phash")
        dup_int = _is_true(it.get("duplicate_internal"))
        dup_ext = _is_true(it.get("duplicate_external"))
        label_val = it.get("label_value")  # string (for single/multi) or s3 key (for file)
        item_label_type = it.get("label_type", label_type)

        # Safety: require a temp image key
        if not temp_img_key:
            # Skip malformed items
            continue

        if dup_int:
            # Delete internal duplicates: remove temp file and staging record
            _delete_if_exists(FILE_BUCKET_NAME, temp_img_key)
            staging_table.delete_item(
                Key={"job_id": job_id, "s3_key_temp": temp_img_key}
            )
            deleted_internal_dups += 1
            continue

        if dup_ext:
            # Leave external duplicates for the next step
            skipped_external_dups += 1
            continue

        # Survivor: move image into canonical and register
        canonical_img_key = _canonical_image_key(user, job_id, temp_img_key)
        _copy_then_delete(FILE_BUCKET_NAME, temp_img_key, FILE_BUCKET_NAME, canonical_img_key)

        # Handle labels
        canonical_label_key = None
        labels_record: Optional[Dict[str, Any]] = None

        if item_label_type in ("single_label", "multi_label"):
            # Expect label_val as a string (single or comma-separated/multi JSON upstream)
            labels_record = {"type": item_label_type, "value": label_val}
        elif item_label_type == "file":
            # Expect label_val as temp S3 key for label file; move it
            if label_val:
                canonical_label_key = _canonical_label_key(user, job_id, label_val)
                _copy_then_delete(FILE_BUCKET_NAME, label_val, FILE_BUCKET_NAME, canonical_label_key)

        # Write canonical imagery record
        canonical_item = {
            "pk": f"IMG#{canonical_img_key}",        # or use a more domain-appropriate PK
            "sk": f"PHASH#{phash}",
            "s3_key_canonical": canonical_img_key,
            "phash": phash,
            "user": user,
            "job_id": job_id,
            "img_type": img_type or it.get("img_type"),
            "created_at": int(time.time()),
        }
        if labels_record:
            canonical_item["labels"] = labels_record
        if canonical_label_key:
            canonical_item["label_s3_key"] = canonical_label_key

        _put_canonical_row(canonical_table, canonical_item)

        # Add to FAISS manifests for step 10
        # Keep both canonical key and phash so the batch job can route/update index.
        manifests.append({
            "canonical_s3_key": canonical_img_key,
            "phash": phash,
            "img_type": img_type or it.get("img_type"),
            "user": user,
            "job_id": job_id,
        })

        # Remove staging record now that it’s registered
        staging_table.delete_item(
            Key={"job_id": job_id, "s3_key_temp": temp_img_key}
        )
        registered_count += 1

    result = {
        "job_id": job_id,
        "user": user,
        "img_type": img_type,
        "label_type": label_type,
        "manifests": manifests,  # consumed by Map in step 10
        "registered_count": registered_count,
        "deleted_internal_dups": deleted_internal_dups,
        "skipped_external_dups": skipped_external_dups,
    }
    return result
