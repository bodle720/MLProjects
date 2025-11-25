import os
import json
import boto3

# Lambda layer imports
from common.utils import log, update_job_status, release_lock, delete_s3_prefix, delete_iceberg_partition_rows

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
ICEBERG_UPLOAD_STAGING_TABLE_NAME = os.environ["ICEBERG_UPLOAD_STAGING_TABLE_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DB_NAME = os.environ["ICEBERG_DB_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]

dynamodb = boto3.resource('dynamodb')

def find_key_recursively(obj, target_key):
    """Search for target_key anywhere in a nested dict/list structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target_key:
                return v
            result = find_key_recursively(v, target_key)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_key_recursively(item, target_key)
            if result is not None:
                return result
    return None

def handler(event, context):
    # DLQ messages come in as a batch from SQS
    total_records = 0
    num_processed_successfully = 0
    for record in event['Records']:
        total_records += 1
        try:
            body = json.loads(record['body'])

            # Depending on how you structure messages, job_id may be nested
            job_id = find_key_recursively(body, "job_id")
            user = find_key_recursively(body, "user")
            event_type = find_key_recursively(body, "event_type")

            if job_id and user and event_type:

                # 1. Delete S3 temp files
                prefix = f"temp/image-upload/{job_id}/"
                delete_s3_prefix(FILE_BUCKET_NAME, prefix)
                log(job_id, user, event_type, f"Done deleting s3 temp files in dlq lambda processor", LOG_FIREHOSE_STREAM_NAME)

                # 2. Delete staging table rows im iceberg table
                delete_result = delete_iceberg_partition_rows(job_id,
                                                              ICEBERG_DB_NAME,
                                                              ICEBERG_UPLOAD_STAGING_TABLE_NAME,
                                                              ATHENA_OUTPUT_S3,
                                                              ATHENA_WORKGROUP
                                                              )
                log(job_id, user, event_type, f"Done deleting iceberg staging row in dlq processor, results = {delete_result}", LOG_FIREHOSE_STREAM_NAME)

                # 3. make sure job status table is marked COMPLETED
                update_success, update_msg = update_job_status(job_id,
                                                               "FAILED",
                                                               JOB_TABLE_NAME,
                                                               LOG_FIREHOSE_STREAM_NAME,
                                                               user=user,
                                                               event_type=event_type,
                                                               error_msg=None)
                log(job_id, user, event_type,
                    f"Done updating job status to FAILED. Success = {update_success}, msg = {update_msg}",
                    LOG_FIREHOSE_STREAM_NAME)

                # 4. Unlock the infrastructure
                release_success, release_msg = release_lock(job_id,
                                                            LOCK_TABLE_NAME,
                                                            LOG_FIREHOSE_STREAM_NAME,
                                                            user=user,
                                                            event_type=event_type)

                log(job_id, user, event_type,
                    f"Done release lock attempt. Success = {release_success}, msg = {release_msg}",
                    LOG_FIREHOSE_STREAM_NAME)

                if update_success and release_success:
                    num_processed_successfully += 1
            else:
                print(f"No job_id, user, or event_type found in DLQ message: {body}")

        except Exception as e:
            print(f"Error processing DLQ message: {e}")

    return {"status": "ok", "total_records": total_records, "successfully_processed_records": num_processed_successfully}