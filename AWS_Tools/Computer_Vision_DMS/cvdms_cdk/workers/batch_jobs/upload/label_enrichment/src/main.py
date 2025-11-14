import os
import json
import time
import boto3
from boto3.dynamodb.conditions import Key

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
CANONICAL_IMAGERY_TABLE = os.environ["CANONICAL_IMAGERY_TABLE"]

def _copy_then_delete(src_key: str, dst_key: str):
    s3.copy_object(
        Bucket=FILE_BUCKET_NAME,
        Key=dst_key,
        CopySource={"Bucket": FILE_BUCKET_NAME, "Key": src_key}
    )
    s3.delete_object(Bucket=FILE_BUCKET_NAME, Key=src_key)

def _delete_if_exists(key: str):
    try:
        s3.delete_object(Bucket=FILE_BUCKET_NAME, Key=key)
    except Exception:
        pass

def _canonical_label_key(user: str, job_id: str, label_temp_key: str) -> str:
    basename = label_temp_key.split("/")[-1]
    return f"canonical/labels/{user}/{job_id}/{basename}"

def handler():
    """
    Expects MANIFESTS env var containing JSON array of manifest entries:
    [
      {
        "job_id": "...",
        "user": "...",
        "img_type": "...",
        "phash": "...",
        "temp_img_key": "...",
        "label_type": "single_label|multi_label|file",
        "label_value": "...",
        "canonical_ref": "IMG#canonical/imagery/..."
      }
    ]
    """
    manifests_str = os.environ.get("MANIFESTS")
    if not manifests_str:
        raise ValueError("No MANIFESTS provided")

    manifests = json.loads(manifests_str)
    canonical_table = dynamodb.Table(CANONICAL_IMAGERY_TABLE)

    enriched = 0
    skipped = 0
    deleted = 0

    for m in manifests:
        job_id = m["job_id"]
        user = m["user"]
        label_type = m.get("label_type")
        label_val = m.get("label_value")
        canonical_ref = m.get("canonical_ref")
        temp_img_key = m.get("temp_img_key")

        # Fetch canonical record
        if not canonical_ref:
            skipped += 1
            continue

        pk = canonical_ref
        resp = canonical_table.get_item(Key={"pk": pk})
        canonical_item = resp.get("Item")
        if not canonical_item:
            skipped += 1
            continue

        updated = False

        if label_type in ("single_label", "multi_label"):
            # Merge string labels
            existing_labels = canonical_item.get("labels", [])
            if isinstance(existing_labels, dict):
                existing_labels = [existing_labels]
            if label_val not in [l.get("value") for l in existing_labels]:
                existing_labels.append({"type": label_type, "value": label_val})
                canonical_item["labels"] = existing_labels
                updated = True

        elif label_type == "file" and label_val:
            # Move label file into canonical folder
            dst_key = _canonical_label_key(user, job_id, label_val)
            _copy_then_delete(label_val, dst_key)

            existing_files = canonical_item.get("label_files", [])
            if dst_key not in existing_files:
                existing_files.append(dst_key)
                canonical_item["label_files"] = existing_files
                updated = True

        if updated:
            canonical_item["updated_at"] = int(time.time())
            canonical_table.put_item(Item=canonical_item)
            enriched += 1
        else:
            # No new labels, delete temp files
            if label_type == "file" and label_val:
                _delete_if_exists(label_val)
            if temp_img_key:
                _delete_if_exists(temp_img_key)
            deleted += 1

    print(f"Enriched: {enriched}, Skipped: {skipped}, Deleted: {deleted}")

if __name__ == "__main__":
    handler()
