# -*- coding: utf-8 -*-
"""
Main API functionality.
"""

import os
import re
import time
import datetime
import uuid
import json
import logging
from typing import Union, Optional

import boto3
from botocore.exceptions import ClientError

import api_helpers
 
# --------------------------
# Define the logger.
# --------------------------
main_dir = os.path.dirname(__file__)
    
base_logs = os.path.join(main_dir, "logs")
os.makedirs(base_logs, exist_ok=True)

logging_save_to = os.path.join(base_logs, "api_logs.txt")

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
        
class InfrastructureAPI:
    def __init__(self, infrastructure_name: str, region: str, profile_name: str):
        """
        Connect to a specific infrastructure namespace in SSM.
        Raises ValueError if the namespace doesn't exist.
        """
        
        self.infrastructure_name = infrastructure_name
        self.region = region
        self.profile_name = profile_name

        session = boto3.Session(profile_name=self.profile_name, region_name=self.region)

        # Load config from SSM
        ssm = session.client("ssm")
        self.config = api_helpers.load_config_from_ssm(ssm, self.infrastructure_name)

        # Validate region
        if self.config["AWS_REGION"] != self.region:
            raise ValueError(
                f"Region mismatch: user specified {self.region}, "
                f"but infrastructure {self.infrastructure_name} is in {self.config['AWS_REGION']}."
            )

        # Build clients
        self.clients = {
                "sts": session.client("sts"),
                "s3": session.client("s3"),
                "sqs": session.client("sqs"),
                "ddb": session.client("dynamodb"),
                "lambda": session.client("lambda"),
                "iam": session.client("iam"),
                "ecr": session.client("ecr"),
                "ssm": ssm,
                'logs': session.client("logs")
            }

        identity = self.clients["sts"].get_caller_identity()
        logger.info(f"Connected as {identity['Arn']} to infra {infrastructure_name}")
        
    def phash_exists(self, phash: str) -> bool:
        """
        Check if an image with the given phash already exists in the S3 images/ folder.
        Returns True if found, False otherwise.
        Note: This ignores extension — any object with this phash prefix counts as existing.
        """
        s3 = self.clients["s3"]
        bucket = self.config["S3_BUCKET_NAME"]
        root = self.config["S3_DATASETS_ROOT"]
    
        prefix = f"{root}/images/{phash}"
        paginator = s3.get_paginator("list_objects_v2")
    
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=1):
            if "Contents" in page and page["Contents"]:
                return True
        return False

    def dataset_exists(self, dataset_id: str) -> bool:
        ddb = self.clients["ddb"]

        try:
            resp = ddb.get_item(
                TableName=self.config["DDB_DATASET_TABLE"],
                Key={'dataset_id': {'S': dataset_id}},
                ConsistentRead=True
            )
        except ClientError as e:
            logger.error(f"[dataset_exists] Error checking {dataset_id}: {e}")
            raise

        return 'Item' in resp and bool(resp['Item'])
    
    def get_lock_owner(self, dataset_id: str):
        ddb = self.clients["ddb"]

        try:
            resp = ddb.get_item(
                TableName=self.config['DDB_DATASET_TABLE'],
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
    
    def dataset_locked(self, dataset_id: str):
        """
        Return True if the dataset is locked, False otherwise.
        Logs the status and, if locked, the job_id that holds the lock.
        """
        ddb = self.clients["ddb"]

        try:
            resp = ddb.get_item(
                TableName=self.config['DDB_DATASET_TABLE'],
                Key={'dataset_id': {'S': dataset_id}},
                ConsistentRead=True
            )
        except ClientError as e:
            logger.error(f"[dataset_locked] Error checking lock for dataset {dataset_id}: {e}")
            raise

        item = resp.get('Item')
        if not item:
            logger.info(f"[dataset_locked] Dataset {dataset_id} not found in table {self.config['DDB_DATASET_TABLE']}. Treating as unlocked.")
            return False

        locked = item.get('locked', {}).get('BOOL', False)
        if locked:
            job_id = item.get('locked_by', {}).get('S')
            logger.info(f"[dataset_locked] Dataset {dataset_id} is locked by job {job_id}.")
            return True
        else:
            logger.info(f"[dataset_locked] Dataset {dataset_id} is not locked.")
            return False
        
    def dataset_synced(self, dataset_id: str) -> bool:
        """
        Return True if the dataset is marked as synced, False otherwise.
        Logs the status for auditability.
        """
        ddb = self.clients["ddb"]
    
        try:
            resp = ddb.get_item(
                TableName=self.config['DDB_DATASET_TABLE'],
                Key={'dataset_id': {'S': dataset_id}},
                ConsistentRead=True
            )
        except ClientError as e:
            logger.error(f"[dataset_synced] Error checking sync status for dataset {dataset_id}: {e}")
            raise
    
        item = resp.get('Item')
        if not item:
            logger.info(
                f"[dataset_synced] Dataset {dataset_id} not found in table {self.config['DDB_DATASET_TABLE']}. "
                "Treating as not synced."
            )
            return False
    
        synced = item.get('synced', {}).get('BOOL', False)
        if synced:
            logger.info(f"[dataset_synced] Dataset {dataset_id} is marked as synced.")
            return True
        else:
            logger.info(f"[dataset_synced] Dataset {dataset_id} is not synced.")
            return False

    def lock_dataset(self, dataset_id: str, job_id: str):
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
        ddb = self.clients['ddb']

        # Fail fast if already locked
        if self.dataset_locked(dataset_id):
            raise Exception(f"[lock_dataset] Dataset {dataset_id} is already locked.")

        try:
            ddb.update_item(
                TableName=self.config['DDB_DATASET_TABLE'],
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
      
    def unlock_dataset(self, dataset_id: str, job_id: str):
          """Unlock a dataset, clearing the lock and lock owner."""
          ddb = self.clients['ddb']
  
          try:
              ddb.update_item(
                  TableName=self.config['DDB_DATASET_TABLE'],
                  Key={'dataset_id': {'S': dataset_id}},
                  UpdateExpression="SET locked = :val REMOVE locked_by",
                  ExpressionAttributeValues={":val": {"BOOL": False}}
              )
              logger.info(f"[unlock_dataset] Dataset {dataset_id} unlocked by job {job_id}.")
          except ClientError as e:
              logger.warning(f"[unlock_dataset] Could not unlock dataset {dataset_id}: {e}")
              
    def set_job_status(self, job_id: str, status: str, message: str = None):
        """Update the job_status (and optionally job_summary) of a job."""
        ddb = self.clients['ddb']

        try:
            if message:
                ddb.update_item(
                    TableName=self.config['DDB_JOB_TABLE'],
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
                    TableName=self.config['DDB_JOB_TABLE'],
                    Key={'job_id': {'S': job_id}},
                    UpdateExpression="SET job_status = :status",
                    ExpressionAttributeValues={":status": {"S": status}}
                )
                logger.info(f"[set_job_status] Job {job_id} -> {status}.")
        except ClientError as e:
            logger.error(f"[set_job_status] Failed to update job {job_id} to {status}: {e}")
            raise

    def job_success(self, job_id: str, message: str = None):
        """Mark a job as COMPLETE."""
        self.set_job_status(job_id, "COMPLETE", message)

    def job_error(self, job_id: str, message: str = None):
        """Mark a job as FAILED."""
        self.set_job_status(job_id, "FAILED", message)
                
    def summarize_job(self, job_id: str) -> dict | None:
        """Fetch a job record from the Job table and log a summary."""
        ddb = self.clients['ddb']

        try:
            resp = ddb.get_item(
                TableName=self.config['DDB_JOB_TABLE'],
                Key={'job_id': {'S': job_id}},
                ConsistentRead=True
            )
        except ClientError as e:
            logger.error(f"[summarize_job] Error retrieving job {job_id}: {e}")
            raise

        item = resp.get('Item')
        if not item:
            logger.info(f"[summarize_job] Job {job_id} not found in table {self.config['DDB_JOB_TABLE']}.")
            return None

        summary = {}
        if 'created_at' in item:
            summary['created_at'] = item['created_at']['S']
        if 'event_type' in item:
            summary['event_type'] = item['event_type']['S']
        if 'job_summary' in item:
            summary['job_summary'] = item['job_summary']['S']
        if 'job_status' in item:
            summary['job_status'] = item['job_status']['S']

        logger.info(f"[summarize_job] Job {job_id} summary:")
        for k, v in summary.items():
            logger.info(f"  {k:11s}: {v}")

        return summary

    def register_dataset(self, dataset_id: str, class_to_id_dict: dict[str, int], band_info: dict[str, str]) -> str:
        """Register a new dataset and track the operation as a Job."""
        
        if dataset_id.strip() == '':
            raise ValueError('The string all is a reserved key that cannot be used as a dataset id.')
            
        if dataset_id == 'all':
            raise ValueError('The string all is a reserved key that cannot be used as a dataset id.')
            
        ddb = self.clients['ddb']
        
        job_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()

        # Insert job row
        ddb.put_item(
            TableName=self.config['DDB_JOB_TABLE'],
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "REGISTER_DATASET"},
                "job_summary": {"S": f"Registering dataset {dataset_id}"},
                "job_status": {"S": "PENDING"}
            }
        )

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

            # Check existence
            if self.dataset_exists(dataset_id):
                raise Exception(f"Dataset {dataset_id} already exists.")

            band_info = {str(k): v for k, v in band_info.items()}
            api_helpers.validate_band_info(band_info)

            # Insert dataset row with lock
            ddb.put_item(
                TableName=self.config['DDB_DATASET_TABLE'],
                Item={
                    "dataset_id": {"S": dataset_id},
                    "locked": {"BOOL": True},
                    "synced": {"BOOL": True},
                    "locked_by": {"S": job_id},
                    "class_to_id_dict": {"S": json.dumps(class_to_id_dict)},
                    "band_info": {"S": json.dumps(band_info)}
                },
                ConditionExpression="attribute_not_exists(dataset_id)"
            )
            locked = True

            self.job_success(job_id, f"Successfully registered dataset {dataset_id}.")
            logger.info(f"[register_dataset] Registered dataset {dataset_id} under job {job_id}.")
            return job_id

        except Exception as e:
            logger.error(f"[register_dataset] Error registering dataset {dataset_id}: {e}")
            self.job_error(job_id, f"Failed to register dataset {dataset_id}: {e}")
            raise
        finally:
            if locked:
                self.unlock_dataset(dataset_id, job_id)

    def add_class_to_dataset(self, dataset_id: str, class_name: str) -> str:
        """Add a new class to an existing dataset."""
        ddb = self.clients['ddb']

        job_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()

        # Insert job row
        ddb.put_item(
            TableName=self.config['DDB_JOB_TABLE'],
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "ADD_CLASS"},
                "job_summary": {"S": f"Adding class '{class_name}' to dataset {dataset_id}"},
                "job_status": {"S": "PENDING"}
            }
        )

        locked = False
        try:
            if not self.dataset_exists(dataset_id):
                raise Exception(f"Dataset {dataset_id} does not exist.")

            if self.dataset_locked(dataset_id):
                raise Exception(f"Dataset {dataset_id} is currently locked.")

            self.lock_dataset(dataset_id, job_id)
            locked = True

            if not isinstance(class_name, str):
                raise Exception(f"class_name must be a string. Got {type(class_name)}.")

            resp = ddb.get_item(
                TableName=self.config['DDB_DATASET_TABLE'],
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

            if class_name in class_dict:
                raise Exception(f"Class '{class_name}' already exists in dataset {dataset_id}.")

            next_id = max(class_dict.values(), default=-1) + 1
            class_dict[class_name] = next_id

            ddb.update_item(
                TableName=self.config['DDB_DATASET_TABLE'],
                Key={'dataset_id': {'S': dataset_id}},
                UpdateExpression="SET class_to_id_dict = :dict",
                ExpressionAttributeValues={":dict": {"S": json.dumps(class_dict)}}
            )

            self.job_success(job_id, f"Added class '{class_name}' with id {next_id} to dataset {dataset_id}.")
            logger.info(f"[add_class_to_dataset] Added class '{class_name}' (id={next_id}) to dataset {dataset_id} under job {job_id}.")
            return job_id

        except Exception as e:
            logger.error(f"[add_class_to_dataset] Error adding class to dataset {dataset_id}: {e}")
            self.job_error(job_id, f"Failed to add class '{class_name}' to dataset {dataset_id}: {e}")
            raise
        finally:
            if locked:
                self.unlock_dataset(dataset_id, job_id)

    def delete_dataset(self, dataset_id: str) -> str:
        """Initiate deletion of a dataset by sending a DELETE_DATASET event to the lifecycle queue."""
        ddb = self.clients['ddb']
        sqs = self.clients['sqs']

        job_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()

        # Insert job row
        ddb.put_item(
            TableName=self.config['DDB_JOB_TABLE'],
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "DELETE_DATASET"},
                "job_summary": {"S": f"Deleting dataset {dataset_id}"},
                "job_status": {"S": "PENDING"}
            }
        )

        sent_to_queue = False
        locked = False
        try:
            queue_url = sqs.get_queue_url(QueueName=self.config['SQS_QUEUE_LIFECYCLE'])["QueueUrl"]

            if not self.dataset_exists(dataset_id):
                raise Exception(f"Dataset {dataset_id} does not exist.")

            if self.dataset_locked(dataset_id):
                raise Exception(f"Dataset {dataset_id} is currently locked.")

            self.lock_dataset(dataset_id, job_id)
            locked = True

            event = {"event_type": "DELETE_DATASET", "dataset_id": dataset_id, "job_id": job_id}
            sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(event))
            sent_to_queue = True
            logger.info(f"[delete_dataset] Sent DELETE_DATASET event for {dataset_id} under job {job_id}.")
            return job_id

        except Exception as e:
            logger.error(f"[delete_dataset] Error: {e}")
            self.job_error(job_id, f"Deletion init failed for {dataset_id}: {e}")
            raise
        finally:
            if locked and not sent_to_queue:
                self.unlock_dataset(dataset_id, job_id)
    
    def remove_class_from_dataset(self, dataset_id: str, class_name: str) -> str:
        """Initiate removal of a class from a dataset by sending a REMOVE_CLASS event."""
        ddb = self.clients['ddb']
        sqs = self.clients['sqs']

        job_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()

        # Insert job row
        ddb.put_item(
            TableName=self.config['DDB_JOB_TABLE'],
            Item={
                "job_id": {"S": job_id},
                "created_at": {"S": created_at},
                "event_type": {"S": "REMOVE_CLASS"},
                "job_summary": {"S": f"Removing class '{class_name}' from dataset {dataset_id}"},
                "job_status": {"S": "PENDING"}
            }
        )

        sent_to_queue = False
        locked = False
        try:
            queue_url = sqs.get_queue_url(QueueName=self.config['SQS_QUEUE_LIFECYCLE'])["QueueUrl"]

            if not self.dataset_exists(dataset_id):
                raise Exception(f"Dataset {dataset_id} does not exist.")

            if self.dataset_locked(dataset_id):
                raise Exception(f"Dataset {dataset_id} is currently locked.")

            self.lock_dataset(dataset_id, job_id)
            locked = True

            if not isinstance(class_name, str):
                raise Exception(f"class_name must be a string. Got {type(class_name)}.")

            resp = ddb.get_item(
                TableName=self.config['DDB_DATASET_TABLE'],
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

            if class_name not in class_dict:
                raise Exception(f"Class '{class_name}' does not exist in dataset {dataset_id}.")

            if len(class_dict) == 1:
                raise Exception("Cannot remove the only class; call delete_dataset instead.")

            event = {
                "event_type": "REMOVE_CLASS",
                "dataset_id": dataset_id,
                "class_name": class_name,
                "job_id": job_id
            }
            sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(event))
            sent_to_queue = True
            logger.info(f"[remove_class_from_dataset] Sent REMOVE_CLASS for {dataset_id}, class '{class_name}' under job {job_id}.")
            
            # Mark dataset as unsynced since a class removal job has been enqueued
            ddb.update_item(
                TableName=self.config['DDB_DATASET_TABLE'],
                Key={'dataset_id': {'S': dataset_id}},
                UpdateExpression="SET synced = :s",
                ExpressionAttributeValues={":s": {"BOOL": False}}
            )
            logger.info(f"[remove_class_from_dataset] Marked dataset {dataset_id} as unsynced.")

            return job_id

        except Exception as e:
            logger.error(f"[remove_class_from_dataset] Error: {e}")
            self.job_error(job_id, f"Removal init failed for {dataset_id}, class '{class_name}': {e}")
            raise
        finally:
            if locked and not sent_to_queue:
                self.unlock_dataset(dataset_id, job_id)
                 
    def upload_images_bulk(
                        self,
                        datasets: Union[str, list[str]],
                        images: list[str],
                        labels: list[str],
                        bands_mapping: dict[str, str],
                        attributes: Optional[dict[str, str]] = None
                    ) -> str:
        """
        Bulk upload images to one or more datasets with optional attributes.
    
        - datasets: a dataset id string, a list of dataset ids, or the string "all"
        - images: list of local file paths
        - labels: list of labels (same length as images)
        - bands_mapping: dict[str(str_index) -> str(band_name)] exactly matching VALID_BANDS, no duplicates
        - attributes: optional dict applied to each imagery item in the bulk upload manifest, can be used when
                      querying images from imagery table.
        Returns: job_id (string)
        """
    
        ddb = self.clients['ddb']
        s3 = self.clients['s3']
        sqs = self.clients['sqs']

        job_table = self.config['DDB_JOB_TABLE']
        dataset_table = self.config['DDB_DATASET_TABLE']
        image_ops_queue_name = self.config['SQS_QUEUE_IMAGE_OPS']
        bucket = self.config['S3_BUCKET_NAME']
        root = self.config['S3_DATASETS_ROOT']
    
        # --- Validate attributes ---
        reserved_keys = {
            "dataset_phash", "dataset_id", "phash", "label", "extension",
            "original_filename", "uploaded_at", "band_mapping", "band_count"
        }
    
        if attributes is not None:
            if not isinstance(attributes, dict):
                raise ValueError("attributes must be a dict[str,str] or None.")
    
            for k, v in attributes.items():
                # Ensure both key and value are strings
                if not isinstance(k, str) or not isinstance(v, str):
                    raise ValueError(f"Invalid attribute entry: {k} -> {v}. Keys and values must be strings.")
    
                # Reserved name check
                if k in reserved_keys:
                    raise ValueError(f"Attribute key '{k}' is reserved and cannot be used.")
    
                # Pattern check: no keys ending in _b<digits> or _B<digits>
                # e.g. mynewattr_b0, thisattr_B100
                if re.search(r"_[bB]\d+$", k):
                    raise ValueError(
                        f"Attribute key '{k}' is not allowed (keys ending with _b<digits> or _B<digits> are reserved)."
                    )

        # --- Resolve datasets ---
        if isinstance(datasets, str):
            if datasets == "all":
                # Use scan; in production consider pagination and a dedicated GSI if size grows
                resp = ddb.scan(TableName=dataset_table, ProjectionExpression="dataset_id")
                dataset_ids = [item["dataset_id"]["S"] for item in resp.get("Items", [])]
                if not dataset_ids:
                    raise Exception("No datasets found in dataset table.")
            else:
                if not datasets.strip():
                    raise ValueError("dataset string cannot be empty.")
                dataset_ids = [datasets]
        elif isinstance(datasets, list) and all(isinstance(ds, str) and ds.strip() for ds in datasets):
            dataset_ids = datasets
        else:
            raise ValueError("datasets must be 'all', a non-empty string, or a list of non-empty strings.")

        # --- Pre-checks on inputs ---
        
        if bands_mapping is None or not isinstance(bands_mapping, dict) or not bands_mapping:
            raise ValueError("bands_mapping is required and must be a non-empty dict.")
    
        bands_mapping = {str(k): v for k, v in bands_mapping.items()}
        api_helpers.validate_band_info(bands_mapping)

        if not isinstance(images, list) or not all(isinstance(p, str) for p in images):
            raise Exception("images must be a list of local file paths (strings).")
    
        if not isinstance(labels, list) or not all(isinstance(l, str) and l.strip() for l in labels):
            raise Exception("labels must be a list of non-empty strings.")
    
        if len(images) != len(labels):
            raise Exception("images and labels must have the same length.")
    
        if not images:
            raise Exception("images list cannot be empty.")
    
        # --- Validate images and extensions ---
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        canonical_ext_map = {".jpg": ".jpeg", ".jpeg": ".jpeg", ".png": ".png", ".tif": ".tiff", ".tiff": ".tiff"}
    
        for path in images:
            if not os.path.exists(path):
                raise Exception(f"Image path does not exist: {path}")
            ext = os.path.splitext(path)[1].lower()
            if ext not in valid_exts:
                raise Exception(f"Unsupported image type for {path}. Must be jpg/jpeg/png/tif/tiff.")
    
        # --- Validate datasets and load rows ---
        dataset_rows = {}
        for ds in dataset_ids:
            if not self.dataset_exists(ds):
                raise Exception(f"Dataset {ds} does not exist.")
            if self.dataset_locked(ds):
                raise Exception(f"Dataset {ds} is currently locked.")
            resp = ddb.get_item(TableName=dataset_table, Key={'dataset_id': {'S': ds}}, ConsistentRead=True)
            item = resp.get('Item')
            if not item:
                raise Exception(f"Dataset {ds} not found after existence check.")
            dataset_rows[ds] = item
    
        # --- Validate class dicts ---
        class_dicts = {}
        for ds, item in dataset_rows.items():
            class_dict_str = item.get("class_to_id_dict", {}).get("S", "{}")
            try:
                class_dict = json.loads(class_dict_str)
            except json.JSONDecodeError:
                raise Exception(f"Corrupted class_to_id_dict for dataset {ds}.")
            class_dicts[ds] = class_dict
    
            for label in labels:
                if label not in class_dict:
                    raise Exception(f"Label '{label}' not found in dataset {ds} classes.")
    
        # --- Validate band_info consistency across all selected datasets ---
        band_infos = {}
        for ds, item in dataset_rows.items():
            band_info_str = item.get("band_info", {}).get("S")
            if not band_info_str:
                raise Exception(f"Dataset {ds} missing band_info definition.")
            try:
                band_info = json.loads(band_info_str)
            except json.JSONDecodeError:
                raise Exception(f"Corrupted band_info for dataset {ds}.")
            band_infos[ds] = band_info
    
        # All band_info must be identical
        unique_band_infos = {json.dumps(bi, sort_keys=True) for bi in band_infos.values()}
        if len(unique_band_infos) != 1:
            raise Exception("Selected datasets do not share identical band_info definitions.")
        dataset_band_info = json.loads(list(unique_band_infos)[0])
    
        # bands_mapping must match dataset_band_info exactly
        if bands_mapping != dataset_band_info:
            raise Exception(f"bands_mapping {bands_mapping} does not match dataset band_info {dataset_band_info}")
    
        # --- Prepare job ---
        job_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()
    
        locked_datasets = set()
        uploaded_keys = []
        sent_to_queue = False
    
        try:
            # Lock all datasets up front
            for ds in dataset_ids:
                self.lock_dataset(ds, job_id)
                locked_datasets.add(ds)
    
            # Insert job row
            ddb.put_item(
                TableName=job_table,
                Item={
                    "job_id": {"S": job_id},
                    "created_at": {"S": created_at},
                    "event_type": {"S": "IMAGE_UPLOAD"},
                    "job_summary": {"S": f"Uploading {len(images)} images to {len(dataset_ids)} datasets"},
                    "job_status": {"S": "PENDING"}
                }
            )
    
            # Build manifest covering all datasets
            manifest = {
                "job_id": job_id,
                "datasets": dataset_ids,
                'bands_mapping':bands_mapping,
                "images": []
            }
    
            # Process each image
            used_phashes = set()
            for path, label in zip(images, labels):
                phash = api_helpers.compute_phash(path)
        
                if phash in used_phashes:
                    logger.info(f"[upload_images_bulk] Duplicate phash {phash} for {path} (label={label}) present in same bulk upload input set already; skipping.")
                    continue
                
                used_phashes.add(phash)
                
                ext = os.path.splitext(path)[1].lower()
                canonical_ext = canonical_ext_map[ext]
    
                # Extract band info and sanity-check
                desired_bands_order = [bands_mapping[str(i)] for i in range(len(bands_mapping))]
                appears_valid, reason = api_helpers.bands_appear_valid(path, desired_bands_order)
    
                if not appears_valid:
                    logger.info(f"[upload_images_bulk] The image {path} does not appear to have the correct band naming or structure required, reason = {reason}; skipping.")
                    continue
                    
                if self.phash_exists(phash):
                    # Already present globally: don’t upload, but include in manifest
                    logger.info(f"[upload_images_bulk] phash {phash} already present globally; marking as already_exists.")
                    manifest["images"].append({
                        "phash": phash,
                        "original_filename": os.path.basename(path),
                        "label": label,
                        "extension": canonical_ext.lstrip("."),
                        "attributes": attributes or {},
                        "already_exists": True
                    })
                else:
                    # Upload original file (no conversion), canonical extension
                    key = f"{root}/temp-images/{phash}{canonical_ext}"
                    with open(path, "rb") as f:
                        s3.put_object(
                            Bucket=bucket,
                            Key=key,
                            Body=f,
                            ContentType=api_helpers.extension_to_mime(canonical_ext)
                        )
                    uploaded_keys.append(key)
                    
                    manifest["images"].append({
                        "phash": phash,
                        "original_filename": os.path.basename(path),
                        "label": label,
                        "extension": canonical_ext.lstrip("."),
                        "attributes": attributes or {},
                        "already_exists": False
                    })
                        
            # Upload manifest and enqueue if we have any images
            if manifest["images"]:
                manifest_key = f"{root}/temp-images/{job_id}.json"
                s3.put_object(
                    Bucket=bucket,
                    Key=manifest_key,
                    Body=json.dumps(manifest).encode("utf-8"),
                    ContentType="application/json"
                )
                uploaded_keys.append(manifest_key)
    
                queue_url = sqs.get_queue_url(QueueName=image_ops_queue_name)["QueueUrl"]
                event = {"event_type": "IMAGE_UPLOAD", "datasets": dataset_ids, "job_id": job_id}
                sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(event))
                sent_to_queue = True
                
                for ds in dataset_ids:
                    ddb.update_item(
                        TableName=self.config['DDB_DATASET_TABLE'],
                        Key={'dataset_id': {'S': ds}},
                        UpdateExpression="SET synced = :s",
                        ExpressionAttributeValues={":s": {"BOOL": False}}
                    )
                    logger.info(f"[upload_images_bulk] Marked dataset {ds} as unsynced after enqueuing job {job_id}.")
                
                    
            logger.info(f"[upload_images_bulk] Enqueued IMAGE_UPLOAD job {job_id} for datasets {dataset_ids}.")
            return job_id
    
        except Exception as e:
            logger.error(f"[upload_images_bulk] Error: {e}")
            # Record job error
            try:
                self.job_error(job_id, f"Image upload init failed for datasets {dataset_ids}: {e}")
            except Exception as job_err:
                logger.warning(f"[upload_images_bulk] Failed to set job error for {job_id}: {job_err}")
    
            # Cleanup uploaded files if enqueue failed
            if uploaded_keys and not sent_to_queue:
                for key in uploaded_keys:
                    try:
                        s3.delete_object(Bucket=bucket, Key=key)
                        logger.info(f"[upload_images_bulk] Cleaned up {key} after failure.")
                    except Exception as cleanup_err:
                        logger.warning(f"[upload_images_bulk] Failed to cleanup {key}: {cleanup_err}")
            raise
    
        finally:
            # Unlock all datasets if not sent to queue
            if locked_datasets and not sent_to_queue:
                for ds in locked_datasets:
                    try:
                        self.unlock_dataset(ds, job_id)
                    except Exception as unlock_err:
                        logger.warning(f"[upload_images_bulk] Failed to unlock dataset {ds} for job {job_id}: {unlock_err}")
    
    def remove_images_bulk(
            self,
            datasets: Union[str, list[str]],
            phashes: Union[str, list[str]]
        ) -> str:
        """
        Remove images from one, many, or all datasets by uploading a deletion manifest to S3
        and sending an IMAGE_DELETE event to the image ops queue.
        """
        ddb = self.clients['ddb']
        s3 = self.clients['s3']
        sqs = self.clients['sqs']
        job_table = self.config['DDB_JOB_TABLE']
        dataset_table = self.config['DDB_DATASET_TABLE']
        image_ops_queue_name = self.config['SQS_QUEUE_IMAGE_OPS']
        bucket = self.config['S3_BUCKET_NAME']
        root = self.config['S3_DATASETS_ROOT']
    
        queue_url = sqs.get_queue_url(QueueName=image_ops_queue_name)["QueueUrl"]
    
        # --- Normalize phashes ---
        if isinstance(phashes, str):
            phashes = [phashes]
        if not isinstance(phashes, list) or not all(isinstance(ph, str) and ph.strip() for ph in phashes):
            raise ValueError("phashes must be a non-empty string or a list of non-empty strings.")
        if not phashes:
            raise ValueError("phashes list cannot be empty.")
    
        # --- Resolve datasets ---
        if isinstance(datasets, str):
            if datasets == "all":
                resp = ddb.scan(TableName=dataset_table, ProjectionExpression="dataset_id")
                dataset_ids = [item["dataset_id"]["S"] for item in resp.get("Items", [])]
                if not dataset_ids:
                    raise Exception("No datasets found in dataset table.")
            else:
                if not datasets.strip():
                    raise ValueError("dataset string cannot be empty.")
                dataset_ids = [datasets]
        elif isinstance(datasets, list) and all(isinstance(ds, str) and ds.strip() for ds in datasets):
            dataset_ids = datasets
        else:
            raise ValueError("datasets must be 'all', a non-empty string, or a list of non-empty strings.")
    
        job_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        locked_datasets = []
        sent_to_queue = False
        uploaded_keys: list[str] = []
    
        try:
            # Lock all datasets
            for ds in dataset_ids:
                if not self.dataset_exists(ds):
                    raise Exception(f"Dataset {ds} does not exist.")
                if self.dataset_locked(ds):
                    raise Exception(f"Dataset {ds} is currently locked.")
                self.lock_dataset(ds, job_id)
                locked_datasets.append(ds)
    
            # Insert job row
            summary = f"Removing {len(phashes)} images from {len(dataset_ids)} dataset(s)"
            ddb.put_item(
                TableName=job_table,
                Item={
                    "job_id": {"S": job_id},
                    "created_at": {"S": created_at},
                    "event_type": {"S": "IMAGE_DELETE"},
                    "job_summary": {"S": summary},
                    "job_status": {"S": "PENDING"}
                }
            )
    
            # Write manifest to S3
            manifest = {
                "datasets": dataset_ids,
                "job_id": job_id,
                "phashes": phashes
            }
            manifest_key = f"{root}/temp-deletions/{job_id}.json"
            s3.put_object(
                Bucket=bucket,
                Key=manifest_key,
                Body=json.dumps(manifest).encode("utf-8"),
                ContentType="application/json"
            )
            uploaded_keys.append(manifest_key)
    
            # Send lightweight event
            event = {"event_type": "IMAGE_DELETE", "datasets": dataset_ids, "job_id": job_id}
            sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(event))
            sent_to_queue = True
    
            for ds in dataset_ids:
                ddb.update_item(
                    TableName=self.config['DDB_DATASET_TABLE'],
                    Key={'dataset_id': {'S': ds}},
                    UpdateExpression="SET synced = :s",
                    ExpressionAttributeValues={":s": {"BOOL": False}}
                )
                logger.info(f"[remove_images_bulk] Marked dataset {ds} as unsynced after enqueuing job {job_id}.")
                
            logger.info(f"[remove_images_bulk] Enqueued IMAGE_DELETE job {job_id}, manifest {manifest_key}.")
            return job_id
    
        except Exception as e:
            logger.error(f"[remove_images_bulk] Error: {e}")
            self.job_error(job_id, f"Removal init failed: {e}")
    
            # Cleanup manifest if enqueue failed
            if uploaded_keys and not sent_to_queue:
                for key in uploaded_keys:
                    try:
                        s3.delete_object(Bucket=bucket, Key=key)
                        logger.info(f"[remove_images_bulk] Cleaned up {key} after failure.")
                    except Exception as cleanup_err:
                        logger.warning(f"[remove_images_bulk] Failed to cleanup {key}: {cleanup_err}")
            raise
        finally:
            if locked_datasets and not sent_to_queue:
                for ds in locked_datasets:
                    try:
                        self.unlock_dataset(ds, job_id)
                    except Exception as unlock_err:
                        logger.warning(f"[remove_images_bulk] Failed to unlock dataset {ds} for job {job_id}: {unlock_err}")
    
    def sync_datasets(self, dataset_ids: str | list[str]) -> str:
        """
        Initiate a sync operation for one or more datasets by sending a SYNC event
        to the sync SQS queue. Tracks the operation as a Job in the Job table.
        Skips datasets that are already marked as synced.
        """
        ddb = self.clients['ddb']
        sqs = self.clients['sqs']
        dataset_table = self.config['DDB_DATASET_TABLE']
        job_table = self.config['DDB_JOB_TABLE']
        sync_queue_name = self.config['SQS_QUEUE_SYNC']
    
        queue_url = sqs.get_queue_url(QueueName=sync_queue_name)["QueueUrl"]
    
        resolved_ids: list[str] = []
        skipped_ids: list[str] = []
    
        if dataset_ids == "all":
            # Scan dataset table for all datasets
            resp = ddb.scan(TableName=dataset_table, ConsistentRead=True)
            for item in resp.get("Items", []):
                dsid = item["dataset_id"]["S"]
                locked = item.get("locked", {}).get("BOOL", False)
                synced = item.get("synced", {}).get("BOOL", False)
    
                if locked:
                    logger.info(f"[sync_datasets] Skipping {dsid} because it is locked.")
                    skipped_ids.append(dsid)
                    continue
                if synced:
                    logger.info(f"[sync_datasets] Skipping {dsid} because it is already synced.")
                    skipped_ids.append(dsid)
                    continue
    
                resolved_ids.append(dsid)
    
            if not resolved_ids:
                raise Exception("No unlocked, unsynced datasets available to sync.")
    
        elif isinstance(dataset_ids, list) and all(isinstance(d, str) for d in dataset_ids):
            for dsid in dataset_ids:
                if not self.dataset_exists(dsid):
                    raise Exception(f"Dataset {dsid} does not exist.")
                if self.dataset_locked(dsid):
                    raise Exception(f"Dataset {dsid} is currently locked.")
    
                # Fetch dataset row to check synced flag
                resp = ddb.get_item(
                    TableName=dataset_table,
                    Key={'dataset_id': {'S': dsid}},
                    ConsistentRead=True
                )
                item = resp.get("Item")
                if not item:
                    raise Exception(f"Dataset {dsid} not found after existence check.")
    
                synced = item.get("synced", {}).get("BOOL", False)
                if synced:
                    logger.info(f"[sync_datasets] Skipping {dsid} because it is already synced.")
                    skipped_ids.append(dsid)
                    continue
    
                resolved_ids.append(dsid)
        else:
            raise Exception("dataset_ids must be 'all' or a list of strings.")
    
        job_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat()
        locked_ids: list[str] = []
        sent_to_queue = False
    
        if len(resolved_ids) == 0:
            raise Exception("dataset_ids is either an empty list or they are all already synced.")

        try:
            # Lock each dataset under this job_id
            for dsid in resolved_ids:
                self.lock_dataset(dsid, job_id)
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
            event = {"event_type": "SYNC", "job_id": job_id, "dataset_ids": resolved_ids}
    
            # Send to sync queue
            sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(event))
            sent_to_queue = True
    
            # Summary log
            logger.info(
                f"[sync_datasets] SYNC job {job_id} summary: "
                f"included={resolved_ids}, skipped={skipped_ids}"
            )
    
            return job_id
    
        except Exception as e:
            logger.error(f"[sync_datasets] Error: {e}")
            self.job_error(job_id, f"Sync init failed: {e}")
            # Unlock any datasets we locked
            for dsid in locked_ids:
                try:
                    self.unlock_dataset(dsid, job_id)
                except Exception as unlock_err:
                    logger.warning(f"[sync_datasets] Failed to unlock {dsid}: {unlock_err}")
            raise
        finally:
            if not sent_to_queue:
                for dsid in locked_ids:
                    try:
                        self.unlock_dataset(dsid, job_id)
                    except Exception as unlock_err:
                        logger.warning(f"[sync_datasets] Failed to unlock {dsid}: {unlock_err}")

 #%% left off here
    def query_logs(self,
                   filters: dict,
                   look_back_hours: float = 1.0,
                   timeout_seconds: int = 30,
                   poll_interval: int = 2,
                   limit: int = 1000):
        """
        Run a CloudWatch Logs Insights query with flexible filters.
        
        filters: dict of field -> value, e.g. {"job_id": "12345", "level": "ERROR"}
        look_back_hours: how many hours back from 'now' to search (can be fractional).
        timeout_seconds: max time to wait for query to complete.
        poll_interval: how often to poll for results.
        limit: max number of results to return.
        
        Example calls:
            Query by job_id only
            results = api.query_logs({"job_id": "12345"}, look_back_hours=2)
            
            Query by job_id and level
            results = api.query_logs({"job_id": "12345", "level": "ERROR"}, look_back_hours=6)
            
            Query all logs in last 30 minutes (no filters)
            results = api.query_logs({}, look_back_hours=0.5)

        """
        log_group_name = self.config['LOG_GROUP_NAME']
        logs = self.clients['logs']
    
        # Compute time window
        end_time = int(time.time())
        start_time = int(end_time - look_back_hours * 3600)
    
        # Build filter expression
        filter_exprs = [f'{field} = "{value}"' for field, value in filters.items()]
        filter_clause = " and ".join(filter_exprs) if filter_exprs else ""
    
        # Build query string
        query = f"""
        fields @timestamp, lambda, job_id, status, level, utc_time
        {"| filter " + filter_clause if filter_clause else ""}
        | sort @timestamp asc
        """
    
        start_query_response = logs.start_query(
            logGroupName=log_group_name,
            startTime=start_time,
            endTime=end_time,
            queryString=query,
            limit=limit
        )
    
        query_id = start_query_response["queryId"]
    
        # Poll with timeout
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            resp = logs.get_query_results(queryId=query_id)
            
            if resp["status"] == "Failed":
                logger.error(f"Query {query_id} failed: {resp}")
                return {"status": "Failed", "results": None, "raw": resp}
            
            elif resp["status"] == "Cancelled":
                logger.error(f"Query {query_id} cancelled: {resp}")
                return {"status": "Cancelled", "results": None, "raw": resp}
            
            elif resp["status"] == "Complete":
                logger.info(f"Query {query_id} completed successfully.")
                parsed = [
                    {col["field"]: col["value"] for col in row}
                    for row in resp["results"]
                ]
                return {"status": "Complete", "results": parsed, "raw": resp}
                              
            time.sleep(poll_interval)
    
        logger.warning(f"Query {query_id} timed out after {timeout_seconds}s")
        raise TimeoutError(f"Logs Insights query did not complete within {timeout_seconds} seconds")