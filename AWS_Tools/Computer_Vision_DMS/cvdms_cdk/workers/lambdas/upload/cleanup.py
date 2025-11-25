import os
import time
import boto3
from boto3.dynamodb.conditions import Key

from common.utils import log, update_job_status, release_lock, delete_s3_prefix, delete_iceberg_partition_rows

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
athena = boto3.client("athena")

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
ICEBERG_UPLOAD_STAGING_TABLE_NAME = os.environ["ICEBERG_UPLOAD_STAGING_TABLE_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DB_NAME = os.environ["ICEBERG_DB_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]

def handler(event, context):
    """
    Input event:
    {
      "job_id": "...",
      "user": "...",
      "event_type": "...",
      ...
    }
    """
    job_id = user = event_type = 'unknown'

    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
    except Exception as e:
        error_msg = f"Failed to parse event on cleanup call: {e}"
        log(job_id, user, event_type, error_msg, LOG_FIREHOSE_STREAM_NAME, error=error_msg, level="error")
        raise ValueError("Missing key in the cleanup lambda: {e}")

    log(job_id, user, event_type, f"Starting cleanup lambda for job {job_id}", LOG_FIREHOSE_STREAM_NAME)

    # 1. Delete S3 temp files
    prefix = f"temp/image-upload/{job_id}/"
    delete_s3_prefix(FILE_BUCKET_NAME, prefix)
    log(job_id, user, event_type, f"Done deleting s3 temp files in cleanup lambda", LOG_FIREHOSE_STREAM_NAME)

    # 2. Delete staging table rows im iceberg table
    delete_result = delete_iceberg_partition_rows(job_id,
                                                  ICEBERG_DB_NAME,
                                                  ICEBERG_UPLOAD_STAGING_TABLE_NAME,
                                                  ATHENA_OUTPUT_S3,
                                                  ATHENA_WORKGROUP
                                                  )
    log(job_id, user, event_type, f"Done deleting iceberg staging row, results = {delete_result}", LOG_FIREHOSE_STREAM_NAME)

    # 3. make sure job status table is marked COMPLETED
    update_success, update_msg = update_job_status(job_id,
                                                  "COMPLETED",
                                                  JOB_TABLE_NAME,
                                                  LOG_FIREHOSE_STREAM_NAME,
                                                  user=user,
                                                  event_type=event_type,
                                                  error_msg=None)
    log(job_id, user, event_type, f"Done updating job status to COMPLETED. Success = {update_success}, msg = {update_msg}", LOG_FIREHOSE_STREAM_NAME)

    # 4. Unlock the infrastructure
    release_success, release_msg = release_lock(job_id,
                                                 LOCK_TABLE_NAME,
                                                 LOG_FIREHOSE_STREAM_NAME,
                                                 user=user,
                                                 event_type=event_type)

    log(job_id, user, event_type, f"Done release lock attempt. Success = {release_success}, msg = {release_msg}", LOG_FIREHOSE_STREAM_NAME)

    return {
        "job_id": job_id,
        "cleanup_done": True
    }