# -*- coding: utf-8 -*-
"""
"""

import os
import csv
import io
import json
import boto3
import logging
import datetime
from botocore.exceptions import ClientError

s3 = boto3.client("s3")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET = os.environ["S3_BUCKET_NAME"]
DATASETS_ROOT = os.environ["S3_DATASETS_ROOT"]
KEY = f"{DATASETS_ROOT}/dlq-logs/logs.csv"

def lambda_handler(event, context):
    records = event.get("Records", [])
    if not records:
        logger.info("No records in event.")
        return {"status": "empty"}

    # Step 1: Load existing CSV (if present)
    existing_rows = []
    header = []
    try:
        response = s3.get_object(Bucket=BUCKET, Key=KEY)
        body = response["Body"].read().decode("utf-8").splitlines()
        reader = csv.DictReader(body)
        existing_rows = list(reader)
        header = reader.fieldnames
        logger.info(f"Loaded existing CSV with {len(existing_rows)} rows.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.info("No existing logs.csv found, will create new one.")
            existing_rows = []
            header = []  # will infer from first batch
        else:
            raise

    # Step 2: Parse new records
    new_rows = []
    for record in records:
        try:
            body = json.loads(record.get("body", "{}"))
        except json.JSONDecodeError:
            body = {"raw_body": record.get("body")}

        # Add system fields
        body["messageId"] = record.get("messageId")
        body["dlq_timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"

        new_rows.append(body)

    # Step 3: Update header dynamically
    all_fieldnames = set(header or [])
    for row in new_rows:
        all_fieldnames.update(row.keys())
    fieldnames = sorted(all_fieldnames)  # stable order

    # Step 4: Merge old + new rows
    merged_rows = existing_rows + new_rows

    # Step 5: Write back to S3
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in merged_rows:
        writer.writerow(row)

    s3.put_object(
        Bucket=BUCKET,
        Key=KEY,
        Body=output.getvalue().encode("utf-8"),
        ContentType="text/csv"
    )

    logger.info(f"Appended {len(new_rows)} records to {KEY}.")
    return {"status": "ok", "appended": len(new_rows), "total": len(merged_rows)}

