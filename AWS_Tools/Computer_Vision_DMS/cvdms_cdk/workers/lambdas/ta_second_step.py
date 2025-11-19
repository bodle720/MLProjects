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
        # kickoff input
        job_id_ko = event['job_id']
        user_ko = event['user']
        label_types_ko =  event['label_types']

        # step1 result (preferred if you trust step1 for canonical values)
        step1 = event['step1']
        job_id_step1 = step1['job_id']
        user_step1 = step1['user']
        label_types_step1 = step1['label_types']
    except KeyError as e:
        raise Exception(f"Could not get needed keys: {e}")

    job_status_updated, job_msg = update_job_status(job_id_ko,
                                                    "COMPLETED",
                                                    job_table)

    if not job_status_updated:
        log(job_id_ko,
            user_ko,
            EVENT_TYPE,
            job_msg,
            FIREHOSE_STREAM_NAME,
            error=job_msg,
            level='error')
        raise Exception(f"Could not set job status: {job_msg}")
    else:
        log(job_id_ko,
            user_ko,
            EVENT_TYPE,
            "Status of job set to COMPLETED in step function step 2",
            FIREHOSE_STREAM_NAME)

        log(job_id_ko,
            user_ko,
            EVENT_TYPE,
            f"The user from kickoff is {user_ko}, the user from step 1 output is {user_step1}, state machine is done.",
            FIREHOSE_STREAM_NAME)

    return {'statusCode': 200, 'job_id': job_id_ko, 'user': user_ko, 'label_types': label_types_ko}