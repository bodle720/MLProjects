import os
import logging
import json
import uuid
import boto3
import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
UPLOAD_STATE_MACHINE_ARN = os.environ["UPLOAD_STATE_MACHINE_ARN"]

# This lambda only expects one job type to trigger it.
EXPECTED_JOB_TYPE = 'IMAGE_UPLOAD'
ALLOWED_LABEL_TYPES = ["string_labels", "bounding_boxes", "semantic_masks", "instance_annotations"]

sf = boto3.client("stepfunctions")
s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

def handler(event, context):
    job_id = None
    created_at = None
    user = None

    try:
        job_table = dynamodb.Table(JOB_TABLE_NAME)
    except Exception as e:
        logger.error(json.dumps({
            "job_id": "UNK",
            "user": "UNK",
            "job_type": EXPECTED_JOB_TYPE,
            "logged_at": "UNK",
            "message": f"{EXPECTED_JOB_TYPE} failed in kickoff lambda due to inability to retrieve dynamodb job table: {str(e)}"
        }))
        raise

    try:
        created_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]

        if bucket != FILE_BUCKET_NAME:
            raise ValueError(f"Unexpected bucket: {bucket}")

        key = record["s3"]["object"]["key"]

        # Read job.json
        obj = s3.get_object(Bucket=bucket, Key=key)
        job_data = json.loads(obj["Body"].read())

        # Extract required fields
        job_id = job_data["job_id"]
        job_type = job_data["job_type"]

        if job_type != EXPECTED_JOB_TYPE:
            raise ValueError(f"Unexpected job type for this lambda: got {job_type}, expected {EXPECTED_JOB_TYPE}")

        label_type = job_data["label_type"]

        if label_type not in ALLOWED_LABEL_TYPES:
            raise ValueError(f"Unexpected label type: got {label_type}, expected one of {ALLOWED_LABEL_TYPES}")

        user = job_data["user"]

        # Write to DynamoDB
        job_table.put_item(Item={
            "job_id": job_id,
            "status": "IN_PROGRESS",
            "created_at": created_at,
            "user": user,
            "job_type": job_type,
            "summary": f"Job type {job_type} with label type {label_type}"
        })

        # Structured JSON log
        logger.info(json.dumps({
            "job_id": job_id,
            "user": user,
            "job_type": job_type,
            "logged_at": created_at,
            "message": "Kickoff Lambda starting state machine for image upload workflow."
        }))

        sf.start_execution(
            stateMachineArn=UPLOAD_STATE_MACHINE_ARN,
            name=f"{job_id}-{int(datetime.datetime.now().timestamp())}",
            input=json.dumps({
                "job_id": job_id,
                "user": user,
                "bucket": bucket,
                "key": key
            })
        )
        return {"status": "ok", "job_id": job_id}

    except Exception as e:
        # Fail fast: mark job as failed
        if job_id:
            job_table.put_item(Item={
                "job_id": job_id,
                "status": "FAILED",
                "summary": f"Failed to pass kickoff lambda image upload job. Error: {e}"
            })

        # Structured error log
        logger.error(json.dumps({
            "job_id": job_id if job_id else "UNK",
            "user": user if user else 'UNK',
            "job_type": EXPECTED_JOB_TYPE,
            "logged_at": created_at if created_at else 'UNK',
            "message": f"{EXPECTED_JOB_TYPE} failed due to error: {str(e)}"
        }))

        raise