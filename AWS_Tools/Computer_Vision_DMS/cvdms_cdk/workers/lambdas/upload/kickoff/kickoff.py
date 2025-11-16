import os
import json
import logging
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
UPLOAD_STATE_MACHINE_ARN = os.environ["UPLOAD_STATE_MACHINE_ARN"]

ALLOWED_LABEL_TYPES = ["string_labels", "bounding_boxes", "semantic_masks", "instance_annotations"]

sf = boto3.client("stepfunctions")
s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

# Logging Helper
def log(job_id,
        user,
        event_type,
        message,
        warning = None,
        error = None,
        level="info"):

    entry = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "message": message,
        "warning": warning,
        "error": error,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    if level == "error":
        logger.error(json.dumps(entry))
    else:
        logger.info(json.dumps(entry))