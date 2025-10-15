# -*- coding: utf-8 -*-
"""
Main API functionality.
"""

import os
import logging
import uuid
import json
import boto3
from botocore.exceptions import ClientError
import datetime
import mimetypes

import api_helpers
 
# --------------------------
# Define the logger.
# --------------------------
main_dir = os.path.dirname(__file__)
    
logging_save_to = os.path.join(main_dir, 'logs.txt')
    
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

def extension_to_mime(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext in ("tif", "tiff"):
        return "image/tiff"
    else:
        raise ValueError(f"Unsupported extension: {ext}")
        
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
            }

        identity = self.clients["sts"].get_caller_identity()
        logger.info(f"Connected as {identity['Arn']} to infra {infrastructure_name}")
        
    def phash_exists(self, phash: str) -> bool:
        """
        Check if an image with the given phash already exists in the S3 images/ folder.
        Returns True if found, False otherwise.
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

            api_helpers.validate_band_info(band_info)

            # Insert dataset row with lock
            ddb.put_item(
                TableName=self.config['DDB_DATASET_TABLE'],
                Item={
                    "dataset_id": {"S": dataset_id},
                    "locked": {"BOOL": True},
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
                return job_id
    
            except Exception as e:
                logger.error(f"[remove_class_from_dataset] Error: {e}")
                self.job_error(job_id, f"Removal init failed for {dataset_id}, class '{class_name}': {e}")
                raise
            finally:
                if locked and not sent_to_queue:
                    self.unlock_dataset(dataset_id, job_id)
                    
         #%% left off here           
        def remove_images_from_dataset(self, dataset_id: str, images: list[str]) -> str:
            """
            Remove images from a dataset by uploading a deletion manifest to S3
            and sending a REMOVE_IMAGES event to the image ops queue.
            """
            ddb = self.clients['ddb']
            s3 = self.clients['s3']
            sqs = self.clients['sqs']
            job_table = self.config['DDB_JOB_TABLE']
            image_ops_queue_name = self.config['SQS_QUEUE_IMAGE_OPS']
            bucket = self.config['S3_BUCKET_NAME']
            root = self.config['S3_DATASETS_ROOT']
    
            queue_url = sqs.get_queue_url(QueueName=image_ops_queue_name)["QueueUrl"]
    
            # Pre-checks
            if not self.dataset_exists(dataset_id):
                raise Exception(f"Dataset {dataset_id} does not exist.")
            if self.dataset_locked(dataset_id):
                raise Exception(f"Dataset {dataset_id} is currently locked.")
            if not isinstance(images, list) or not all(isinstance(ph, str) for ph in images):
                raise Exception("images must be a list of strings (pHashes).")
            if len(images) == 0:
                raise Exception("images list cannot be empty.")
    
            job_id = str(uuid.uuid4())
            created_at = datetime.datetime.utcnow().isoformat()
            locked = False
            sent_to_queue = False
            uploaded_keys: list[str] = []
    
            try:
                # Lock dataset
                self.lock_dataset(dataset_id, job_id)
                locked = True
    
                # Insert job row
                ddb.put_item(
                    TableName=job_table,
                    Item={
                        "job_id": {"S": job_id},
                        "created_at": {"S": created_at},
                        "event_type": {"S": "REMOVE_IMAGES"},
                        "job_summary": {"S": f"Removing {len(images)} images from dataset {dataset_id}"},
                        "job_status": {"S": "PENDING"}
                    }
                )
    
                # Write manifest to S3
                manifest = {"dataset_id": dataset_id, "job_id": job_id, "images": images}
                manifest_key = f"{root}/temp-deletions/{job_id}.json"
                s3.put_object(
                    Bucket=bucket,
                    Key=manifest_key,
                    Body=json.dumps(manifest).encode("utf-8"),
                    ContentType="application/json"
                )
                uploaded_keys.append(manifest_key)
    
                # Send lightweight event
                event = {"event_type": "REMOVE_IMAGES", "dataset_id": dataset_id, "job_id": job_id}
                sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(event))
                sent_to_queue = True
    
                logger.info(f"[remove_images_from_dataset] Enqueued REMOVE_IMAGES job {job_id} "
                            f"for dataset {dataset_id}, manifest {manifest_key}.")
                return job_id
    
            except Exception as e:
                logger.error(f"[remove_images_from_dataset] Error: {e}")
                self.job_error(job_id, f"Removal init failed for dataset {dataset_id}: {e}")
    
                # Cleanup manifest if enqueue failed
                if uploaded_keys and not sent_to_queue:
                    for key in uploaded_keys:
                        try:
                            s3.delete_object(Bucket=bucket, Key=key)
                            logger.info(f"[remove_images_from_dataset] Cleaned up {key} after failure.")
                        except Exception as cleanup_err:
                            logger.warning(f"[remove_images_from_dataset] Failed to cleanup {key}: {cleanup_err}")
                raise
            finally:
                if locked and not sent_to_queue:
                    self.unlock_dataset(dataset_id, job_id)
                    
        def upload_images_to_dataset(
                                self,
                                dataset_id: str,
                                images: list[str],
                                labels: list[str],
                                bands_mapping: dict[str, str],
                                forced_split: str | None = None) -> str:
            """
            Upload local images to the global temp-images folder and enqueue an IMAGE_UPLOAD job.
            """
            
            ddb = self.clients['ddb']
            s3 = self.clients['s3']
            sqs = self.clients['sqs']
            job_table = self.config['DDB_JOB_TABLE']
            dataset_table = self.config['DDB_DATASET_TABLE']
            image_ops_queue_name = self.config['SQS_QUEUE_IMAGE_OPS']
            bucket = self.config['S3_BUCKET_NAME']
            root = self.config['S3_DATASETS_ROOT']
    
            # Pre-checks
            if bands_mapping is None:
                raise ValueError("bands_mapping is required for all uploads.")
            if forced_split not in (None, "training", "validation", "testing"):
                raise ValueError("forced_split must be one of None, 'training', 'validation', or 'testing'")
            if not self.dataset_exists(dataset_id):
                raise Exception(f"Dataset {dataset_id} does not exist.")
            if self.dataset_locked(dataset_id):
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
            valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            for path in images:
                if not os.path.exists(path):
                    raise Exception(f"Image path does not exist: {path}")
                ext = os.path.splitext(path)[1].lower()
                if ext not in valid_exts:
                    raise Exception(f"Unsupported image type for {path}. Must be jpg/jpeg/png/tif/tiff.")
    
            # Fetch dataset row
            resp = ddb.get_item(
                TableName=dataset_table,
                Key={'dataset_id': {'S': dataset_id}},
                ConsistentRead=True
            )
            item = resp.get('Item')
            if not item:
                raise Exception(f"Dataset {dataset_id} not found after existence check.")
    
            # Load dataset schema
            class_dict_str = item.get('class_to_id_dict', {}).get('S', '{}')
            dataset_band_info_str = item.get('band_info', {}).get('S')
            if not dataset_band_info_str:
                raise Exception(f"Dataset {dataset_id} missing band_info definition.")
            try:
                class_dict = json.loads(class_dict_str)
                dataset_band_info = json.loads(dataset_band_info_str)
            except json.JSONDecodeError:
                raise Exception(f"Corrupted dataset metadata for {dataset_id}.")
    
            # Validate labels
            for label in labels:
                if label not in class_dict:
                    raise Exception(f"Label '{label}' not found in dataset {dataset_id} classes.")
    
            # Validate bands_mapping against dataset definition
            if bands_mapping != dataset_band_info:
                raise Exception(f"bands_mapping {bands_mapping} does not match dataset band_info {dataset_band_info}")
    
            job_id = str(uuid.uuid4())
            created_at = datetime.datetime.utcnow().isoformat()
            locked = False
            sent_to_queue = False
            uploaded_keys: list[str] = []
    
            try:
                # Lock dataset
                self.lock_dataset(dataset_id, job_id)
                locked = True
    
                # Insert job row
                ddb.put_item(
                    TableName=job_table,
                    Item={
                        "job_id": {"S": job_id},
                        "created_at": {"S": created_at},
                        "event_type": {"S": "IMAGE_UPLOAD"},
                        "job_summary": {"S": f"Starting uploading of {len(images)} images to dataset {dataset_id}"},
                        "job_status": {"S": "PENDING"}
                    }
                )
    
                manifest = {"dataset_id": dataset_id, "job_id": job_id, "images": []}
    
                for path, label in zip(images, labels):
                    phash = api_helpers.compute_phash(path)
                    
                    if self.phash_exists(phash):
                        logger.info(f"[upload_images_to_dataset] phash of image at {path} in class {label} already present in images folder, skipping.")
                        continue
                    
                    ext = os.path.splitext(path)[1].lower()
    
                    # Extract band info
                    bands_info = api_helpers.extract_bands(path, [bands_mapping[str(i)] for i in range(len(bands_mapping))])
    
                    # Extra check for GeoTIFF
                    if ext in (".tif", ".tiff") and bands_info["bands_source"] == "gdal_metadata":
                        if bands_info["bands_map"] != dataset_band_info:
                            raise Exception(
                                f"GeoTIFF metadata {bands_info['bands_map']} does not match dataset band_info {dataset_band_info}"
                            )
    
                    # Upload original file (no conversion)
                    key = f"{root}/temp-images/{phash}{ext}"
                    with open(path, "rb") as f:
                        s3.put_object(
                            Bucket=bucket,
                            Key=key,
                            Body=f,
                            ContentType=extension_to_mime(ext)
                        )
                    uploaded_keys.append(key)
    
                    manifest["images"].append({
                        "phash": phash,
                        "filename": os.path.basename(path),
                        "label": label,
                        'extension': ext.replace('.',''),
                        "bands_count": bands_info["bands_count"],
                        "bands_map": bands_info["bands_map"],
                        "bands_source": bands_info["bands_source"],
                        "forced_split": forced_split
                    })
    
                # Upload manifest JSON
                if len(manifest["images"]) > 0:
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
                self.job_error(job_id, f"Image upload init failed for dataset {dataset_id}: {e}")
    
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
                    self.unlock_dataset(dataset_id, job_id)
                    
        def sync_datasets(self, dataset_ids: str | list[str]) -> str:
            """
            Initiate a sync operation for one or more datasets by sending a SYNC event
            to the sync SQS queue. Tracks the operation as a Job in the Job table.
            """
            ddb = self.clients['ddb']
            sqs = self.clients['sqs']
            dataset_table = self.config['DDB_DATASET_TABLE']
            job_table = self.config['DDB_JOB_TABLE']
            sync_queue_name = self.config['SQS_QUEUE_SYNC']
    
            queue_url = sqs.get_queue_url(QueueName=sync_queue_name)["QueueUrl"]
    
            resolved_ids: list[str] = []
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
                    if not self.dataset_exists(dsid):
                        raise Exception(f"Dataset {dsid} does not exist.")
                    if self.dataset_locked(dsid):
                        raise Exception(f"Dataset {dsid} is currently locked.")
                    resolved_ids.append(dsid)
            else:
                raise Exception("dataset_ids must be 'all' or a list of strings.")
    
            job_id = str(uuid.uuid4())
            created_at = datetime.datetime.utcnow().isoformat()
            locked_ids: list[str] = []
            sent_to_queue = False
    
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
                logger.info(f"[sync_datasets] Enqueued SYNC job {job_id} for {len(resolved_ids)} datasets.")
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