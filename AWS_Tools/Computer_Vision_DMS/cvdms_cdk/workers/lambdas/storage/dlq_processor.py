import os
import json
import boto3

# Lambda layer imports
from common.utils import log, update_job_status, release_lock

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

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
    job_table = dynamodb.Table(JOB_TABLE_NAME)

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

                # Fail the job since it's in the DLQ.
                update_successful, update_reason = update_job_status(job_id,
                                                                     "FAILED",
                                                                     job_table,
                                                                     LOG_FIREHOSE_STREAM_NAME,
                                                                     user=user,
                                                                     event_type=event_type,
                                                                     error_msg=f"Marked job {job_id} as FAILED due to DLQ message and released lock.")

                # We need to release the lock on the lock table as well
                release_successful, release_reason = release_lock(job_id,
                                                                  LOCK_TABLE_NAME)

                # Log the result with firehose.
                log_msg = f"Status switch to failed successful: {update_successful}, reason: {update_reason}, lock released: {release_successful}, reason: {release_reason}"
                level = "error" if not update_successful or not release_successful else "info"

                log(job_id,
                    user,
                    event_type,
                    log_msg,
                    LOG_FIREHOSE_STREAM_NAME,
                    level=level)

                if update_successful and release_successful:
                    num_processed_successfully += 1
            else:
                print(f"No job_id found in DLQ message: {body}")

        except Exception as e:
            print(f"Error processing DLQ message: {e}")

    return {"status": "ok", "total_records": total_records, "successfully_processed_records": num_processed_successfully}