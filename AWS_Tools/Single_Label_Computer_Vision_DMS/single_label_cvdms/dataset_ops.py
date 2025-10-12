# -*- coding: utf-8 -*-
"""
Main API functionality.
"""

import os
import sys
import logging
import uuid
import json
import boto3
from botocore.exceptions import ClientError
import datetime
from PIL import Image
import imagehash
from dotenv import load_dotenv

# --------------------------
# Define the logger.
# --------------------------
MAIN_DIR = os.path.dirname(__file__)
logging_save_to = os.path.join(MAIN_DIR, 'api_logs.txt')

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear() 
    
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(logging_save_to)
console_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
    
# --------------------------
# Load in the config.
# --------------------------
config_path = 'config.env'
loaded_env_config = load_dotenv(config_path)

if not loaded_env_config:
    logger.error(f'Failed to load environment config at {config_path}')
    sys.exit(1)

CONFIG = {}

region = os.getenv("AWS_REGION")

if not region:
    logger.error('AWS_REGION is missing from the config.')
    sys.exit(1)

region = region.lower()
CONFIG['AWS_REGION'] = region

bucket = os.getenv("S3_BUCKET_NAME")
root = os.getenv("S3_DATASETS_ROOT")

if not bucket:
    logger.error("S3_BUCKET_NAME is missing from the config")
    sys.exit(1)
        
if not root:
    logger.error("S3_DATASETS_ROOT is missing from the config")
    sys.exit(1)

# Normalize root (strip trailing slashes)
root = root.strip().strip("/")

CONFIG["S3_BUCKET_NAME"] = bucket
CONFIG["S3_DATASETS_ROOT"] = root

#------------------------------
# Get the DynamoDB parameters.
#------------------------------
DDB_IMAGERY_TABLE = os.getenv("DDB_IMAGERY_TABLE")
DDB_DATASET_TABLE = os.getenv("DDB_DATASET_TABLE")
DDB_JOB_TABLE = os.getenv("DDB_JOB_TABLE")

if (not DDB_IMAGERY_TABLE) or (not DDB_DATASET_TABLE) or (not DDB_JOB_TABLE):
    logger.error("DDB_IMAGERY_TABLE or DDB_DATASET_TABLE or DDB_JOB_TABLE is missing from the config")
    sys.exit(1)
    
CONFIG["DDB_IMAGERY_TABLE"] = DDB_IMAGERY_TABLE
CONFIG["DDB_DATASET_TABLE"] = DDB_DATASET_TABLE
CONFIG["DDB_JOB_TABLE"]     = DDB_JOB_TABLE

SQS_QUEUE_LIFECYCLE = os.getenv("SQS_QUEUE_LIFECYCLE")
SQS_QUEUE_IMAGE_OPS = os.getenv("SQS_QUEUE_IMAGE_OPS")
SQS_QUEUE_SYNC      = os.getenv("SQS_QUEUE_SYNC")
SQS_DLQ             = os.getenv("SQS_DLQ")

if (not SQS_QUEUE_LIFECYCLE) or (not SQS_QUEUE_IMAGE_OPS) or (not SQS_QUEUE_SYNC) or (not SQS_DLQ):
    logger.error("SQS_QUEUE_LIFECYCLE or SQS_QUEUE_IMAGE_OPS or SQS_QUEUE_SYNC or SQS_DLQ is missing from the config")
    sys.exit(1)
    
CONFIG["SQS_QUEUE_LIFECYCLE"] = SQS_QUEUE_LIFECYCLE
CONFIG["SQS_QUEUE_IMAGE_OPS"] = SQS_QUEUE_IMAGE_OPS
CONFIG["SQS_QUEUE_SYNC"]      = SQS_QUEUE_SYNC
CONFIG["SQS_DLQ"]             = SQS_DLQ

LAMBDA_LIFECYCLE = os.getenv("LAMBDA_LIFECYCLE")
LAMBDA_IMAGE_OPS = os.getenv("LAMBDA_IMAGE_OPS")
LAMBDA_SYNC      = os.getenv("LAMBDA_SYNC")

if (not LAMBDA_LIFECYCLE) or (not LAMBDA_IMAGE_OPS) or (not LAMBDA_SYNC):
    logger.error("LAMBDA_LIFECYCLE or LAMBDA_IMAGE_OPS or LAMBDA_SYNC is missing from the config")
    sys.exit(1)
    
CONFIG["LAMBDA_LIFECYCLE"] = LAMBDA_LIFECYCLE
CONFIG["LAMBDA_IMAGE_OPS"] = LAMBDA_IMAGE_OPS
CONFIG["LAMBDA_SYNC"]      = LAMBDA_SYNC
    
# --------------------------
# Define the clients.
# --------------------------
CLIENTS = {'s3': boto3.client("s3", region_name=CONFIG['AWS_REGION']),
           'sqs': boto3.client("sqs", region_name=CONFIG['AWS_REGION']),
           'ddb':boto3.client("dynamodb", region_name=CONFIG['AWS_REGION']),
           'lambda':boto3.client("lambda", region_name=CONFIG['AWS_REGION']),
           'iam':boto3.client("iam", region_name=CONFIG['AWS_REGION']),
           'sts':boto3.client("sts", region_name=CONFIG['AWS_REGION']),
           'ecr':boto3.client('ecr', region_name=CONFIG['AWS_REGION'])}

# --------------------------
# Create helper functions
# --------------------------
def get_lock_owner(dataset_id):
    ddb = CLIENTS['ddb']

    try:
        resp = ddb.get_item(
            TableName=CONFIG['DDB_DATASET_TABLE'],
            Key={'dataset_id': {'S': dataset_id}},
            ConsistentRead=True
        )
    except ClientError as e:
        logger.error(f"[get_lock_owner] Error retrieving lock owner for dataset {dataset_id}: {e}")
        raise

    item = resp.get('Item')
    if item and item.get('locked', {}).get('BOOL'):
        job_id = item.get('locked_by', {}).get('S')
        logger.info(f"[get_lock_owner] Dataset {dataset_id} is locked by job {job_id}.")
        return job_id

    logger.info(f"[get_lock_owner] Dataset {dataset_id} is not locked or not found.")
    return None

def dataset_exists(dataset_id):
    """
    Return True if the dataset_id exists in the Dataset table, False otherwise.
    """
    ddb = CLIENTS['ddb']
    table_name = CONFIG['DDB_DATASET_TABLE']

    try:
        resp = ddb.get_item(
            TableName=table_name,
            Key={'dataset_id': {'S': dataset_id}},
            ConsistentRead=True
        )
    except ClientError as e:
        logger.error(f"[dataset_exists] Error checking existence for dataset {dataset_id}: {e}")
        raise

    if 'Item' in resp and resp['Item']:
        logger.info(f"[dataset_exists] Dataset {dataset_id} exists in table {table_name}.")
        return True
    else:
        logger.info(f"[dataset_exists] Dataset {dataset_id} does not exist in table {table_name}.")
        return False

def dataset_locked(dataset_id):
    """
    Return True if the dataset is locked, False otherwise.
    Logs the status and, if locked, the job_id that holds the lock.
    """
    ddb = CLIENTS['ddb']

    try:
        resp = ddb.get_item(
            TableName=CONFIG['DDB_DATASET_TABLE'],
            Key={'dataset_id': {'S': dataset_id}},
            ConsistentRead=True
        )
    except ClientError as e:
        logger.error(f"[dataset_locked] Error checking lock for dataset {dataset_id}: {e}")
        raise

    item = resp.get('Item')
    if not item:
        logger.info(f"[dataset_locked] Dataset {dataset_id} not found in table {CONFIG['DDB_DATASET_TABLE']}. Treating as unlocked.")
        return False

    locked = item.get('locked', {}).get('BOOL', False)
    if locked:
        job_id = item.get('locked_by', {}).get('S')
        logger.info(f"[dataset_locked] Dataset {dataset_id} is locked by job {job_id}.")
        return True
    else:
        logger.info(f"[dataset_locked] Dataset {dataset_id} is not locked.")
        return False

def lock_dataset(dataset_id, job_id):
    """
    Attempt to lock a dataset for a specific job.
    Fails if the dataset is already locked.

    Parameters
    ----------
    dataset_id : str
        The dataset identifier.
    job_id : str
        The job requesting the lock.

    Raises
    ------
    Exception
        If the dataset is already locked or cannot be updated.
    """
    ddb = CLIENTS['ddb']
    table_name = CONFIG['DDB_DATASET_TABLE']

    # Fail fast if already locked
    if dataset_locked(dataset_id):
        raise Exception(f"[lock_dataset] Dataset {dataset_id} is already locked.")

    try:
        ddb.update_item(
            TableName=table_name,
            Key={'dataset_id': {'S': dataset_id}},
            UpdateExpression="SET locked = :val, locked_by = :job",
            ExpressionAttributeValues={
                ":val": {"BOOL": True},
                ":job": {"S": job_id}
            }
        )
        logger.info(f"[lock_dataset] Dataset {dataset_id} locked by job {job_id}.")
    except ClientError as e:
        logger.error(f"[lock_dataset] Failed to lock dataset {dataset_id}: {e}")
        raise

def unlock_dataset(dataset_id, job_id):
    """
    Unlock a dataset, clearing the lock and lock owner.

    Parameters
    ----------
    dataset_id : str
        The dataset identifier.
    job_id : str
        The job releasing the lock (for logging).
    """
    ddb = CLIENTS['ddb']
    table_name = CONFIG['DDB_DATASET_TABLE']

    try:
        ddb.update_item(
            TableName=table_name,
            Key={'dataset_id': {'S': dataset_id}},
            UpdateExpression="SET locked = :val REMOVE locked_by",
            ExpressionAttributeValues={":val": {"BOOL": False}}
        )
        logger.info(f"[unlock_dataset] Dataset {dataset_id} unlocked by job {job_id}.")
    except ClientError as e:
        logger.warning(f"[unlock_dataset] Could not unlock dataset {dataset_id}: {e}")


def set_job_status(job_id, status, message=None):
    """
    Update the job_status (and optionally job_summary) of a job in the Job table.

    Parameters
    ----------
    job_id : str
        The job identifier (primary key in the Job table).
    status : str
        The new status to set (e.g., 'COMPLETE', 'FAILED').
    message : str, optional
        If provided, also updates the job_summary field with this message.
    """
    ddb = CLIENTS['ddb']
    table_name = CONFIG['DDB_JOB_TABLE']

    try:
        if message:
            ddb.update_item(
                TableName=table_name,
                Key={'job_id': {'S': job_id}},
                UpdateExpression="SET job_status = :status, job_summary = :summary",
                ExpressionAttributeValues={
                    ":status": {"S": status},
                    ":summary": {"S": message}
                }
            )
            logger.info(f"[set_job_status] Job {job_id} -> {status}. Summary: {message}")
        else:
            ddb.update_item(
                TableName=table_name,
                Key={'job_id': {'S': job_id}},
                UpdateExpression="SET job_status = :status",
                ExpressionAttributeValues={":status": {"S": status}}
            )
            logger.info(f"[set_job_status] Job {job_id} -> {status}.")
    except ClientError as e:
        logger.error(f"[set_job_status] Failed to update job {job_id} to {status}: {e}")
        raise

def job_success(job_id, message=None):
    """
    Mark a job as COMPLETE and optionally update its summary.
    """
    set_job_status(job_id, "COMPLETE", message)

def job_error(job_id, message=None):
    """
    Mark a job as FAILED and optionally update its summary.
    """
    set_job_status(job_id, "FAILED", message)

def compute_phash(path):
    """Compute perceptual hash (phash) of an image file."""
    with Image.open(path) as img:
        return str(imagehash.phash(img))
    
# --------------------------
# Create main API functions
# --------------------------

def summarize_job(job_id):
    """
    Fetch a job record from the Job table and log a summary.

    Parameters
    ----------
    job_id : str
        The unique job identifier (primary key in the Job table).

    Returns
    -------
    dict or None
        A dictionary containing any of the following keys if present:
        - 'created_at'
        - 'event_type'
        - 'job_summary'
        - 'job_status'
        Returns None if the job does not exist.
    """
    ddb = CLIENTS['ddb']
    table_name = CONFIG['DDB_JOB_TABLE']

    try:
        resp = ddb.get_item(
            TableName=table_name,
            Key={'job_id': {'S': job_id}},
            ConsistentRead=True
        )
    except ClientError as e:
        logger.error(f"[summarize_job] Error retrieving job {job_id}: {e}")
        raise

    item = resp.get('Item')
    if not item:
        logger.info(f"[summarize_job] Job {job_id} not found in table {table_name}.")
        return None

    # Extract attributes if present
    summary = {}
    if 'created_at' in item:
        summary['created_at'] = item['created_at']['S']
    if 'event_type' in item:
        summary['event_type'] = item['event_type']['S']
    if 'job_summary' in item:
        summary['job_summary'] = item['job_summary']['S']
    if 'job_status' in item:
        summary['job_status'] = item['job_status']['S']

    # Log neatly
    logger.info(f"[summarize_job] Job {job_id} summary:")
    for k, v in summary.items():
        logger.info(f"  {k:11s}: {v}")

    return summary

def register_dataset(dataset_id, class_to_id_dict):
    """
    Register a new dataset by inserting into the Dataset table and
    tracking the operation as a Job in the Job table.
    While the dataset is being registered, it is locked (locked=True).
    """
    ddb = CLIENTS['ddb']
    dataset_table = CONFIG['DDB_DATASET_TABLE']
    job_table = CONFIG['DDB_JOB_TABLE']

    job_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()

    # Insert job row with PENDING status
    try:
        ddb.put_item(
            TableName=job_table,
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "REGISTER_DATASET"},
                "job_summary": {"S": f"Registering dataset {dataset_id}"},
                "job_status": {"S": "PENDING"}
            }
        )
    except ClientError as e:
        logger.error(f"[register_dataset] Failed to insert job {job_id}: {e}")
        raise

    locked = False
    try:
        # Validate class_to_id_dict
        if not isinstance(class_to_id_dict, dict) or not all(
            isinstance(k, str) and isinstance(v, int) and v >= 0 for k, v in class_to_id_dict.items()
        ):
            raise Exception("class_to_id_dict must be a dict[str,int>=0].")

        values = sorted(class_to_id_dict.values())
        expected = list(range(len(values)))
        if values != expected:
            raise Exception("class_to_id_dict values must be contiguous 0..N.")

        # Check if dataset already exists
        if dataset_exists(dataset_id):
            raise Exception(f"Dataset {dataset_id} already exists. Cannot register duplicate.")

        # Insert dataset row with locked=True and locked_by=job_id
        ddb.put_item(
            TableName=dataset_table,
            Item={
                "dataset_id": {"S": dataset_id},
                "locked": {"BOOL": True},
                "locked_by": {"S": job_id},
                "class_to_id_dict": {"S": json.dumps(class_to_id_dict)}
            },
            ConditionExpression="attribute_not_exists(dataset_id)"  # safety
        )
        locked = True

        # Mark job as COMPLETE
        job_success(job_id, f"Successfully registered dataset {dataset_id}.")
        logger.info(f"[register_dataset] Successfully registered dataset {dataset_id} under job {job_id}.")

        return job_id

    except Exception as e:
        logger.error(f"[register_dataset] Error registering dataset {dataset_id}: {e}")
        job_error(job_id, f"Failed to register dataset {dataset_id}: {e}")
        raise

    finally:
        if locked:
            unlock_dataset(dataset_id, job_id)

def add_class_to_dataset(dataset_id, class_name):
    """
    Add a new class to an existing dataset's class_to_id_dict.
    Tracks the operation as a Job in the Job table.
    Locks the dataset during modification to prevent concurrent access.
    """
    ddb = CLIENTS['ddb']
    dataset_table = CONFIG['DDB_DATASET_TABLE']
    job_table = CONFIG['DDB_JOB_TABLE']

    job_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()

    # Insert job row with PENDING status
    try:
        ddb.put_item(
            TableName=job_table,
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "ADD_CLASS_TO_DATASET"},
                "job_summary": {"S": f"Adding class '{class_name}' to dataset {dataset_id}"},
                "job_status": {"S": "PENDING"}
            }
        )
    except ClientError as e:
        logger.error(f"[add_class_to_dataset] Failed to insert job {job_id}: {e}")
        raise

    locked = False
    try:
        # Validate dataset existence
        if not dataset_exists(dataset_id):
            raise Exception(f"Dataset {dataset_id} does not exist. Please register it first.")

        # Fail immediately if dataset is locked by another job
        if dataset_locked(dataset_id):
            raise Exception(f"Dataset {dataset_id} is currently locked by another job. Try again later.")

        # Lock the dataset for this job
        lock_dataset(dataset_id, job_id)
        locked = True

        # Validate class_name
        if not isinstance(class_name, str):
            raise Exception(f"class_name must be a string. Got {type(class_name)}.")

        # Fetch current dataset row
        resp = ddb.get_item(
            TableName=dataset_table,
            Key={'dataset_id': {'S': dataset_id}},
            ConsistentRead=True
        )
        item = resp.get('Item')
        if not item:
            raise Exception(f"Dataset {dataset_id} not found after existence check.")

        # Parse class_to_id_dict
        class_dict_str = item.get('class_to_id_dict', {}).get('S', '{}')
        try:
            class_dict = json.loads(class_dict_str)
        except json.JSONDecodeError:
            raise Exception(f"Corrupted class_to_id_dict for dataset {dataset_id}.")

        # Ensure class_name not already present
        if class_name in class_dict:
            raise Exception(f"Class '{class_name}' already exists in dataset {dataset_id}.")

        # Compute next class ID
        next_id = max(class_dict.values(), default=-1) + 1
        class_dict[class_name] = next_id

        # Update dataset row
        ddb.update_item(
            TableName=dataset_table,
            Key={'dataset_id': {'S': dataset_id}},
            UpdateExpression="SET class_to_id_dict = :dict",
            ExpressionAttributeValues={":dict": {"S": json.dumps(class_dict)}}
        )

        # Mark job as COMPLETE
        job_success(job_id, f"Added class '{class_name}' with id {next_id} to dataset {dataset_id}.")
        logger.info(f"[add_class_to_dataset] Successfully added class '{class_name}' (id={next_id}) to dataset {dataset_id} under job {job_id}.")

        return job_id

    except Exception as e:
        logger.error(f"[add_class_to_dataset] Error while adding class to dataset {dataset_id}: {e}")
        job_error(job_id, f"Failed to add class '{class_name}' to dataset {dataset_id}: {e}")
        raise

    finally:
        if locked:
            unlock_dataset(dataset_id, job_id)

def delete_dataset(dataset_id):
    """
    Initiate deletion of a dataset by sending a DELETE_DATASET event to the lifecycle SQS queue.
    Tracks the operation as a Job in the Job table.

    Parameters
    ----------
    dataset_id : str
        The dataset identifier to delete.

    Returns
    -------
    str
        The job_id created for this deletion job.

    Raises
    ------
    Exception
        If the dataset does not exist, is already locked, or if a DynamoDB/SQS error occurs.
    """
    ddb = CLIENTS['ddb']
    sqs = CLIENTS['sqs']
    job_table = CONFIG['DDB_JOB_TABLE']
    lifecycle_queue_name = CONFIG['SQS_QUEUE_LIFECYCLE']

    job_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()

    # Insert job row with PENDING status
    try:
        ddb.put_item(
            TableName=job_table,
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "DELETE_DATASET"},
                "job_summary": {"S": f"Deleting dataset {dataset_id}"},
                "job_status": {"S": "PENDING"}
            }
        )
    except ClientError as e:
        logger.error(f"[delete_dataset] Failed to insert job {job_id}: {e}")
        raise

    sent_to_queue = False
    locked = False

    try:
        # Resolve the queue URL from the queue name
        resp_q = sqs.get_queue_url(QueueName=lifecycle_queue_name)
        queue_url = resp_q["QueueUrl"]

        # Check dataset existence
        if not dataset_exists(dataset_id):
            raise Exception(f"Dataset {dataset_id} does not exist. Cannot delete.")

        # Fail immediately if dataset is locked
        if dataset_locked(dataset_id):
            raise Exception(f"Dataset {dataset_id} is currently locked by another job. Try again later.")

        # Lock the dataset for this job
        lock_dataset(dataset_id, job_id)
        locked = True
        
        # Build the event payload
        event = {
            "event_type": "DELETE_DATASET",
            "dataset_id": dataset_id,
            "job_id": job_id
        }

        # Send event to lifecycle SQS queue
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(event)
        )
        sent_to_queue = True
        logger.info(f"[delete_dataset] Sent DELETE_DATASET event for dataset {dataset_id} to lifecycle queue under job {job_id}.")

        # Lambda will handle deletion and job status updates
        return job_id

    except Exception as e:
        # Mark job failed once here
        logger.error(f"[delete_dataset] Error during deletion init for dataset {dataset_id}: {e}")
        job_error(job_id, f"Deletion init failed for {dataset_id}: {e}")
        raise

    finally:
        # Only unlock if we failed to enqueue the event
        if locked and not sent_to_queue:
            unlock_dataset(dataset_id, job_id)

def remove_class_from_dataset(dataset_id, class_name):
    """
    Initiate removal of a class from a dataset by sending a REMOVE_CLASS_FROM_DATASET
    event to the lifecycle SQS queue. Tracks the operation as a Job in the Job table.

    Parameters
    ----------
    dataset_id : str
        The dataset identifier.
    class_name : str
        The class name to remove.

    Returns
    -------
    str
        The job_id created for this removal job.

    Raises
    ------
    Exception
        If the dataset does not exist, is locked, class_name is invalid or not present,
        or if a DynamoDB/SQS error occurs.
    """
    ddb = CLIENTS['ddb']
    sqs = CLIENTS['sqs']
    dataset_table = CONFIG['DDB_DATASET_TABLE']
    job_table = CONFIG['DDB_JOB_TABLE']
    lifecycle_queue_name = CONFIG['SQS_QUEUE_LIFECYCLE']

    job_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()

    # Insert job row with PENDING status
    try:
        ddb.put_item(
            TableName=job_table,
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "REMOVE_CLASS_FROM_DATASET"},
                "job_summary": {"S": f"Removing class '{class_name}' from dataset {dataset_id}"},
                "job_status": {"S": "PENDING"}
            }
        )
    except ClientError as e:
        logger.error(f"[remove_class_from_dataset] Failed to insert job {job_id}: {e}")
        raise

    sent_to_queue = False
    locked = False
    try:
        # Resolve the queue URL from the queue name
        resp_q = sqs.get_queue_url(QueueName=lifecycle_queue_name)
        queue_url = resp_q["QueueUrl"]

        # Check dataset existence
        if not dataset_exists(dataset_id):
            raise Exception(f"Dataset {dataset_id} does not exist. Cannot remove class.")

        # Fail immediately if dataset is locked
        if dataset_locked(dataset_id):
            raise Exception(f"Dataset {dataset_id} is currently locked by another job. Try again later.")

        # Lock the dataset for this job
        lock_dataset(dataset_id, job_id)
        locked = True

        # Validate class_name
        if not isinstance(class_name, str):
            raise Exception(f"class_name must be a string. Got {type(class_name)}.")

        # Fetch current dataset row
        resp = ddb.get_item(
            TableName=dataset_table,
            Key={'dataset_id': {'S': dataset_id}},
            ConsistentRead=True
        )
        item = resp.get('Item')
        if not item:
            raise Exception(f"Dataset {dataset_id} not found after existence check.")

        # Parse class_to_id_dict
        class_dict_str = item.get('class_to_id_dict', {}).get('S', '{}')
        try:
            class_dict = json.loads(class_dict_str)
        except json.JSONDecodeError:
            raise Exception(f"Corrupted class_to_id_dict for dataset {dataset_id}.")

        # Ensure class_name exists
        if class_name not in class_dict:
            raise Exception(f"Class '{class_name}' does not exist in dataset {dataset_id}.")

        if len(class_dict) == 1:
            raise Exception("Cannot remove the class because it's the only class in this dataset, call delete_dataset instead.")

        # Build the event payload
        event = {
            "event_type": "REMOVE_CLASS_FROM_DATASET",
            "dataset_id": dataset_id,
            "class_name": class_name,
            "job_id": job_id
        }

        # Send event to lifecycle SQS queue
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(event)
        )
        sent_to_queue = True
        logger.info(f"[remove_class_from_dataset] Sent REMOVE_CLASS_FROM_DATASET event for dataset {dataset_id}, class '{class_name}' under job {job_id}.")

        # Lambda will handle removal and job status updates
        return job_id

    except Exception as e:
        logger.error(f"[remove_class_from_dataset] Error during removal init for dataset {dataset_id}: {e}")
        job_error(job_id, f"Removal init failed for dataset {dataset_id}, class '{class_name}': {e}")
        raise

    finally:
        # Only unlock if we locked and did not enqueue successfully
        if locked and not sent_to_queue:
            unlock_dataset(dataset_id, job_id)
      
def remove_images_from_dataset(dataset_id, images, max_images=5000):
    """
    Remove images from a dataset by sending a single REMOVE_IMAGES_FROM_DATASET
    event to the image ops SQS queue. Enforces a maximum number of images per call.

    Parameters
    ----------
    dataset_id : str
        The dataset identifier.
    images : list[str]
        List of image pHashes to remove.
    max_images : int
        Maximum number of images allowed in a single request (default 5000).

    Returns
    -------
    str
        The job_id created for this removal job.

    Raises
    ------
    Exception
        If the dataset does not exist, is locked, images is invalid,
        exceeds max_images, or if a DynamoDB/SQS error occurs.
    """
    ddb = CLIENTS['ddb']
    sqs = CLIENTS['sqs']
    job_table = CONFIG['DDB_JOB_TABLE']
    image_ops_queue_name = CONFIG['SQS_QUEUE_IMAGE_OPS']

    queue_url = sqs.get_queue_url(QueueName=image_ops_queue_name)["QueueUrl"]

    # Pre-checks
    if not dataset_exists(dataset_id):
        raise Exception(f"Dataset {dataset_id} does not exist.")
        
    if dataset_locked(dataset_id):
        raise Exception(f"Dataset {dataset_id} is currently locked.")
        
    if not isinstance(images, list) or not all(isinstance(ph, str) for ph in images):
        raise Exception("images must be a list of strings (pHashes).")
        
    if len(images) == 0:
        raise Exception("images list cannot be empty.")
        
    if len(images) > max_images:
        raise Exception(f"Too many images: {len(images)} provided, "
                        f"maximum allowed is {max_images}. "
                        f"Split into multiple API calls.")

    job_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()
    locked = False
    sent_to_queue = False

    try:
        # Lock dataset
        lock_dataset(dataset_id, job_id)
        locked = True

        # Insert job row
        ddb.put_item(
            TableName=job_table,
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "REMOVE_IMAGES_FROM_DATASET"},
                "job_summary": {"S": f"Removing {len(images)} images from dataset {dataset_id}"},
                "job_status": {"S": "PENDING"}
            }
        )

        # Build event payload
        event = {
            "event_type": "REMOVE_IMAGES_FROM_DATASET",
            "dataset_id": dataset_id,
            "images": images,
            "job_id": job_id
        }

        # Send event
        sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(event)
        )
        sent_to_queue = True
        logger.info(f"[remove_images_from_dataset] Sent REMOVE_IMAGES_FROM_DATASET event "
                    f"for dataset {dataset_id}, {len(images)} images under job {job_id}.")

        return job_id

    except Exception as e:
        logger.error(f"[remove_images_from_dataset] Error: {e}")
        job_error(job_id, f"Removal init failed for dataset {dataset_id}: {e}")
        raise
    finally:
        if locked and not sent_to_queue:
            unlock_dataset(dataset_id, job_id)

def upload_images_to_dataset(dataset_id, images, labels):
    """
    Upload local images to the global temp-images folder and enqueue an IMAGE_UPLOAD job.

    Parameters
    ----------
    dataset_id : str
        The dataset identifier.
    images : list[str]
        List of local file paths to images.
    labels : list[str]
        List of class labels corresponding to each image.

    Returns
    -------
    str
        The job_id created for this upload job.
    """
    ddb = CLIENTS['ddb']
    s3 = CLIENTS['s3']
    sqs = CLIENTS['sqs']
    job_table = CONFIG['DDB_JOB_TABLE']
    dataset_table = CONFIG['DDB_DATASET_TABLE']
    image_ops_queue_name = CONFIG['SQS_QUEUE_IMAGE_OPS']
    bucket = CONFIG['S3_BUCKET_NAME']
    root = CONFIG['S3_DATASETS_ROOT']

    # Pre-checks
    if not dataset_exists(dataset_id):
        raise Exception(f"Dataset {dataset_id} does not exist.")
    if dataset_locked(dataset_id):
        raise Exception(f"Dataset {dataset_id} is currently locked.")
    if not isinstance(images, list) or not all(isinstance(p, str) for p in images):
        raise Exception("images must be a list of local file paths (strings).")
    if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
        raise Exception("labels must be a list of strings.")
    if len(images) != len(labels):
        raise Exception("images and labels must have the same length.")
    if not images:
        raise Exception("images list cannot be empty.")

    # Validate each path
    valid_exts = {".jpg", ".jpeg", ".png"}
    for path in images:
        if not os.path.exists(path):
            raise Exception(f"Image path does not exist: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext not in valid_exts:
            raise Exception(f"Unsupported image type for {path}. Must be jpg/jpeg/png.")

    # Fetch dataset class_to_id_dict
    resp = ddb.get_item(
        TableName=dataset_table,
        Key={'dataset_id': {'S': dataset_id}},
        ConsistentRead=True
    )
    item = resp.get('Item')
    if not item:
        raise Exception(f"Dataset {dataset_id} not found after existence check.")
    class_dict_str = item.get('class_to_id_dict', {}).get('S', '{}')
    try:
        class_dict = json.loads(class_dict_str)
    except json.JSONDecodeError:
        raise Exception(f"Corrupted class_to_id_dict for dataset {dataset_id}.")

    # Validate labels against class dict
    for label in labels:
        if label not in class_dict:
            raise Exception(f"Label '{label}' not found in dataset {dataset_id} classes.")

    job_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()
    locked = False
    sent_to_queue = False
    uploaded_keys = []

    try:
        # Lock dataset
        lock_dataset(dataset_id, job_id)
        locked = True

        # Insert job row
        ddb.put_item(
            TableName=job_table,
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "IMAGE_UPLOAD"},
                "job_summary": {"S": f"Uploading {len(images)} images to dataset {dataset_id}"},
                "job_status": {"S": "PENDING"}
            }
        )

        # Upload images to global temp-images/
        manifest = {"dataset_id": dataset_id, "job_id": job_id, "images": []}
        for path, label in zip(images, labels):
            phash = compute_phash(path)
            key = f"{root}/temp-images/{phash}.png"
            with open(path, "rb") as f:
                s3.put_object(Bucket=bucket, Key=key, Body=f, ContentType="image/png")
            uploaded_keys.append(key)
            manifest["images"].append({
                "phash": phash,
                "filename": os.path.basename(path),
                "label": label
            })

        # Upload manifest JSON
        manifest_key = f"{root}/temp-images/{job_id}.json"
        s3.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=json.dumps(manifest).encode("utf-8"),
            ContentType="application/json"
        )
        uploaded_keys.append(manifest_key)

        # Send event to image ops queue
        queue_url = sqs.get_queue_url(QueueName=image_ops_queue_name)["QueueUrl"]
        event = {"event_type": "IMAGE_UPLOAD", "dataset_id": dataset_id, "job_id": job_id}
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(event))
        sent_to_queue = True

        logger.info(f"[upload_images_to_dataset] Enqueued IMAGE_UPLOAD job {job_id} for dataset {dataset_id}.")
        return job_id

    except Exception as e:
        logger.error(f"[upload_images_to_dataset] Error: {e}")
        job_error(job_id, f"Image upload init failed for dataset {dataset_id}: {e}")

        # Cleanup uploaded files if enqueue failed
        if uploaded_keys and not sent_to_queue:
            for key in uploaded_keys:
                try:
                    s3.delete_object(Bucket=bucket, Key=key)
                    logger.info(f"[upload_images_to_dataset] Cleaned up {key} after failure.")
                except Exception as cleanup_err:
                    logger.warning(f"[upload_images_to_dataset] Failed to cleanup {key}: {cleanup_err}")
        raise
    finally:
        if locked and not sent_to_queue:
            unlock_dataset(dataset_id, job_id)

def sync_datasets(dataset_ids):
    """
    Initiate a sync operation for one or more datasets by sending a SYNC event
    to the sync SQS queue. Tracks the operation as a Job in the Job table.

    Parameters
    ----------
    dataset_ids : str | list[str]
        Either the string "all" to sync all datasets, or a list of dataset IDs.

    Returns
    -------
    str
        The job_id created for this sync job.

    Raises
    ------
    Exception
        If dataset_ids is invalid, or if any specified dataset does not exist or is locked.
    """
    ddb = CLIENTS['ddb']
    sqs = CLIENTS['sqs']
    dataset_table = CONFIG['DDB_DATASET_TABLE']
    job_table = CONFIG['DDB_JOB_TABLE']
    sync_queue_name = CONFIG['SQS_QUEUE_SYNC']

    queue_url = sqs.get_queue_url(QueueName=sync_queue_name)["QueueUrl"]

    resolved_ids = []
    if dataset_ids == "all":
        # Scan dataset table for all unlocked datasets
        resp = ddb.scan(TableName=dataset_table, ConsistentRead=True)
        for item in resp.get("Items", []):
            dsid = item["dataset_id"]["S"]
            locked = item.get("locked", {}).get("BOOL", False)
            if not locked:
                resolved_ids.append(dsid)
        if not resolved_ids:
            raise Exception("No unlocked datasets available to sync.")
    elif isinstance(dataset_ids, list) and all(isinstance(d, str) for d in dataset_ids):
        for dsid in dataset_ids:
            if not dataset_exists(dsid):
                raise Exception(f"Dataset {dsid} does not exist.")
            if dataset_locked(dsid):
                raise Exception(f"Dataset {dsid} is currently locked.")
            resolved_ids.append(dsid)
    else:
        raise Exception("dataset_ids must be 'all' or a list of strings.")

    job_id = str(uuid.uuid4())
    created_at = datetime.datetime.utcnow().isoformat()
    locked_ids = []
    sent_to_queue = False

    try:
        # Lock each dataset under this job_id
        for dsid in resolved_ids:
            lock_dataset(dsid, job_id)
            locked_ids.append(dsid)

        # Insert job row
        ddb.put_item(
            TableName=job_table,
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "SYNC"},
                "job_summary": {"S": f"Syncing {len(resolved_ids)} datasets"},
                "job_status": {"S": "PENDING"}
            }
        )

        # Build event
        event = {
            "event_type": "SYNC",
            "job_id": job_id,
            "dataset_ids": resolved_ids
        }

        # Send to sync queue
        sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(event))
        sent_to_queue = True
        logger.info(f"[sync_datasets] Enqueued SYNC job {job_id} for {len(resolved_ids)} datasets.")
        return job_id

    except Exception as e:
        logger.error(f"[sync_datasets] Error: {e}")
        job_error(job_id, f"Sync init failed: {e}")
        # Unlock any datasets we locked
        for dsid in locked_ids:
            try:
                unlock_dataset(dsid, job_id)
            except Exception as unlock_err:
                logger.warning(f"[sync_datasets] Failed to unlock {dsid}: {unlock_err}")
        raise
    finally:
        if not sent_to_queue:
            # If enqueue failed, ensure unlock
            for dsid in locked_ids:
                try:
                    unlock_dataset(dsid, job_id)
                except Exception as unlock_err:
                    logger.warning(f"[sync_datasets] Failed to unlock {dsid}: {unlock_err}")