import json
import os
import logging
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

# Lambda layer import
from common.utils import update_job_status, log

FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]

firehose = boto3.client("firehose")
dynamodb = boto3.resource("dynamodb")

EVENT_TYPE = "TA_test"

def handler(event, context):
    job_table = dynamodb.Table(JOB_TABLE_NAME)

    try:
        job_id = event['job_id']
        user = event['user']
        label_types = event['label_types']
    except KeyError as e:
        raise Exception(f"Could not get needed keys: {e}")

    job_status_updated, job_msg = update_job_status(job_id,
                                                    "IN_PROGRESS",
                                                    job_table)

    if not job_status_updated:
        log(job_id,
            user,
            EVENT_TYPE,
            job_msg,
            FIREHOSE_STREAM_NAME,
            error=job_msg,
            level='error')
        raise Exception(f"Could not set job status: {job_msg}")
    else:
        log(job_id,
            user,
            EVENT_TYPE,
            "Status of job set to in progress in step function step 1",
            FIREHOSE_STREAM_NAME)

    return {'statusCode': 200, 'job_id': job_id, 'user': user, 'label_types': label_types}


