import json
import os
import logging
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

# Lambda layer import
from common.utils import log

FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
print('The firehose stream name is = ', FIREHOSE_STREAM_NAME)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]

firehose = boto3.client("firehose")
dynamodb = boto3.resource("dynamodb")

EVENT_TYPE = "IMAGE_UPLOAD"

def handler(event, context):
    try:
        job_id = event['job_id']
        user = event['user']
        label_types = event['label_types']
        job_table = dynamodb.Table(JOB_TABLE_NAME)
    except KeyError as e:
        raise Exception(f"Could not get needed keys: {e}")

    log(job_id, user, EVENT_TYPE, "Success in lambda tester 1", FIREHOSE_STREAM_NAME)

    return {'statusCode': 200, 'job_id': job_id, 'user': user, 'label_types': label_types}


