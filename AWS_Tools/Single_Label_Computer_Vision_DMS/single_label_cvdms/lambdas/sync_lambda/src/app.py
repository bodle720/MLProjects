# -*- coding: utf-8 -*-
"""
"""

import os
import json
import boto3
import logging
import datetime
from botocore.exceptions import ClientError

from helpers import extract_features_from_item, calculate_and_store_features, \
                    assign_splits, build_csv

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Environment variables ---
AWS_REGION = os.environ["AWS_REGION"]
S3_BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
S3_DATASETS_ROOT = os.environ["S3_DATASETS_ROOT"]
DDB_IMAGERY_TABLE = os.environ["DDB_IMAGERY_TABLE"]
DDB_DATASET_TABLE = os.environ["DDB_DATASET_TABLE"]
DDB_JOB_TABLE = os.environ["DDB_JOB_TABLE"]

# --- AWS clients ---
s3 = boto3.client("s3", region_name=AWS_REGION)
ddb = boto3.client("dynamodb", region_name=AWS_REGION)

# --- Job status helpers ---
def job_update(job_id, status, summary=None):
    expr = "SET job_status = :s, updated_at = :t"
    values = {
        ":s": {"S": status},
        ":t": {"S": datetime.datetime.utcnow().isoformat()}
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

def job_complete(job_id, summary=None):
    job_update(job_id, "COMPLETE", summary)

def job_error(job_id, summary=None):
    job_update(job_id, "FAILED", summary)

# --- Dataset unlock helper ---
def unlock_dataset(dataset_id, job_id):
    try:
        ddb.update_item(
            TableName=DDB_DATASET_TABLE,
            Key={'dataset_id': {'S': dataset_id}},
            UpdateExpression="SET locked = :val REMOVE locked_by",
            ExpressionAttributeValues={":val": {"BOOL": False}}
        )
        logger.info(f"[unlock_dataset] Dataset {dataset_id} unlocked by job {job_id}.")
    except ClientError as e:
        logger.warning(f"[unlock_dataset] Could not unlock dataset {dataset_id}: {e}")

# --- Event handler ---
def handle_sync(event):
    job_id = event["job_id"]
    dataset_ids = event["dataset_ids"]
    logger.info(f"[SyncLambda] Handling SYNC job {job_id} for datasets: {dataset_ids}")

    try:
        for dataset_id in dataset_ids:
            # 1. Fetch dataset metadata
            ds_resp = ddb.get_item(
                TableName=DDB_DATASET_TABLE,
                Key={"dataset_id": {"S": dataset_id}},
                ConsistentRead=True
            )
            ds_item = ds_resp.get("Item")
            if not ds_item:
                raise Exception(f"Dataset {dataset_id} not found.")
            class_dict = json.loads(ds_item["class_to_id_dict"]["S"])

            # 2. Query imagery table for this dataset
            resp = ddb.query(
                TableName=DDB_IMAGERY_TABLE,
                IndexName="DatasetIndex",
                KeyConditionExpression="dataset_id = :d",
                ExpressionAttributeValues={":d": {"S": dataset_id}}
            )
            imagery_items = resp.get("Items", [])

            enriched = []
            for item in imagery_items:
                phash = item["phash"]["S"]
                label = item["label"]["S"]

                # 3. Ensure features exist
                features = extract_features_from_item(item)
                if not features:
                    features = calculate_and_store_features(phash, dataset_id)

                # 4. Assign class_id
                class_id = class_dict[label]

                enriched.append({
                    "phash": phash,
                    "filename": item.get("filename", {}).get("S", ""),
                    "label": label,
                    "class_id": class_id,
                    "features": features
                })

            # 5. Assign splits
            enriched = assign_splits(enriched)

            # 6. Write manifests
            manifest_prefix = f"{S3_DATASETS_ROOT}/manifests/{dataset_id}/"
            json_key = manifest_prefix + "manifest.json"
            csv_key = manifest_prefix + "manifest.csv"

            s3.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=json_key,
                Body=json.dumps(enriched, indent=2).encode("utf-8"),
                ContentType="application/json"
            )
            csv_body = build_csv(enriched)
            s3.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=csv_key,
                Body=csv_body.encode("utf-8"),
                ContentType="text/csv"
            )

            logger.info(f"[SyncLambda] Wrote manifests for dataset {dataset_id}")

        # 7. Mark job complete and unlock datasets
        job_complete(job_id, f"Synced {len(dataset_ids)} datasets successfully.")
        for dsid in dataset_ids:
            unlock_dataset(dsid, job_id)

    except Exception as e:
        logger.error(f"[SyncLambda] SYNC job {job_id} failed: {e}")
        job_error(job_id, f"Sync failed: {e}")
        for dsid in dataset_ids:
            try:
                unlock_dataset(dsid, job_id)
            except Exception as unlock_err:
                logger.warning(f"[SyncLambda] Failed to unlock {dsid}: {unlock_err}")
        raise


# --- Lambda entrypoint ---
def lambda_handler(event, context):
    """
    Lambda entrypoint for sync operations.
    Triggered by SQS messages from the sync queue.
    """
    for record in event.get("Records", []):
        try:
            body = json.loads(record["body"])
            event_type = body.get("event_type")
            logger.info(f"[SyncLambda] Received event: {event_type}")

            if event_type == "SYNC":
                handle_sync(body)
            else:
                logger.warning(f"[SyncLambda] Unknown event_type: {event_type}")
        except Exception as e:
            logger.error(f"[SyncLambda] Error processing record: {e}")
            raise
