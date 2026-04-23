import uuid
import json
import logging

from typing import Literal, Any, Tuple, Optional, Dict
from datetime import datetime, timezone
from mypy_boto3_s3.client import S3Client
from botocore.exceptions import ClientError
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource

# Helpers for validating user inputs
from cvdms_platform.dataset.validator import (validate_create_dataset_inputs, validate_update_dataset_inputs,
                                                    validate_delete_dataset_inputs, validate_get_dataset_inputs)

# Helper to retrieve information about a dataset
from cvdms_platform.dataset.get_dataset_info import get_dataset_info

class DatasetClient:
    """
    High-level client to perform dataset operations.
    """
    def __init__(self,
                 *,
                 user: str,
                 file_bucket_name: str,
                 job_table_name: str,
                 lock_table_name: str,
                 datasets_table_name: str,
                 dataset_versions_table_name: str,
                 s3_client: S3Client,
                 dynamodb_resource: DynamoDBServiceResource):

        self.user = user
        self.file_bucket_name = file_bucket_name
        self.job_table_name = job_table_name
        self.lock_table_name = lock_table_name
        self.datasets_table_name = datasets_table_name
        self.dataset_versions_table_name = dataset_versions_table_name

        # Clients and resources
        self.s3_client = s3_client
        self.dynamodb_resource = dynamodb_resource

        # Get the DDB tables needed.
        self.job_table = self.dynamodb_resource.Table(self.job_table_name)
        self.lock_table = self.dynamodb_resource.Table(self.lock_table_name)
        self.datasets_table = self.dynamodb_resource.Table(self.datasets_table_name)
        self.dataset_versions_table = self.dynamodb_resource.Table(self.dataset_versions_table_name)

        # Constants
        self.event_type = "DATASET_OP"

    def get_dataset(self, *, dataset_id: str) -> dict[str, Any]:
        """
        Return dataset information for the latest version.

        Returns:
            {
                "dataset_info": {"exists": False},
                "latest_version_info": None,
            }
        if the dataset does not exist.

        Otherwise, returns:
            {
                "dataset_info": {... dataset-level immutable/current fields ...},
                "latest_version_info": {... latest version metadata ...},
            }
        """

        try:
            validated = validate_get_dataset_inputs(dataset_id=dataset_id)
            dataset_id = validated["dataset_id"]
        except Exception as e:
            logging.error(str(e))
            raise

        logging.info(f"Validated dataset_id successfully: {dataset_id}. Starting retrieval...")

        # Returns the latest versions info plus dataset wide immutable fields
        return get_dataset_info(
            datasets_table=self.datasets_table,
            dataset_versions_table=self.dataset_versions_table,
            dataset_id=dataset_id
        )

    def submit_create_dataset(self,
                                *,
                                dataset_id: str,
                                label_type: str,
                                description: str | None,
                                selection_config: dict,
                                split_strategy_name: str,
                                honor_source_splits: bool) -> dict:
        """
        Submits request to create a new dataset at version 1.

        High-level flow:
        1. validate the inputs
        2. verify the dataset does not exist
        3. submit the request to S3
        """

        # 1. Validate inputs
        logging.info("Validating inputs...")

        try:
            validated = validate_create_dataset_inputs(
                dataset_id=dataset_id,
                label_type=label_type,
                description=description,
                selection_config=selection_config,
                split_strategy_name=split_strategy_name,
                honor_source_splits=honor_source_splits
            )

            dataset_id = validated["dataset_id"]
            label_type = validated["label_type"]
            description = validated["description"]
            selection_config = validated["selection_config"]
            split_strategy_name = validated["split_strategy_name"]
            honor_source_splits = validated["honor_source_splits"]
        except Exception as e:
            logging.error(str(e))
            raise

        logging.info("Inputs validated.")

        # 2. Ensure dataset_id is not previously used
        logging.info("Checking if dataset already exists...")
        try:
            dataset_info = self.get_dataset(dataset_id=dataset_id)
        except Exception as e:
            logging.error(f"Failed loading dataset metadata for '{dataset_id}': {str(e)}")
            raise

        if dataset_info["dataset_info"]["exists"]:
            logging.error(f"Dataset '{dataset_id}' already exists, choose a different name.")
            raise ValueError(f"Dataset '{dataset_id}' already exists, choose a different name.")

        if honor_source_splits and split_strategy_name:
            logging.info(
                "Create request for dataset_id='%s' has honor_source_splits=True, "
                "so split_strategy_name='%s' will be ignored. "
                "Dataset splits will be assigned directly from image_source_membership.source_split. "
                "Images with conflicting non-empty source splits will be excluded, and images with no "
                "resolved non-empty source split will also be excluded.",
                dataset_id,
                split_strategy_name,
            )

        # 3. Submit task to S3
        payload = {"user": self.user,
                    "event_type": self.event_type,
                    "task_type": "create_dataset",
                    "request": {
                        "dataset_id": dataset_id,
                        "label_type": label_type,
                        "new_version": 1,
                        "description": description,
                        "selection_config": selection_config,
                        "split_strategy_name": split_strategy_name,
                        "honor_source_splits": honor_source_splits
                    }
                }

        submission = self._submit_job(payload=payload)

        return submission

    def submit_update_dataset(self,
                               *,
                               dataset_id: str,
                               operation: Literal["add", "remove"],
                               selection_config: dict[str, Any],
                               split_approach: Literal["maintain", "rebalance"] = "maintain",
                               split_strategy_name: str | None = None,
                               description: str | None = None) -> dict:
        """
        Submits request to create a new dataset version by adding or removing imagery to/from an existing dataset.

        High-level flow:
        1. validate inputs
        2. verify the dataset exists
        3. submit the request to S3
        """

        # 1. Validate inputs
        logging.info("Validating inputs...")

        try:
            validated = validate_update_dataset_inputs(
                dataset_id=dataset_id,
                operation=operation,
                selection_config=selection_config,
                split_approach=split_approach,
                split_strategy_name=split_strategy_name,
                description=description,
            )

            dataset_id = validated["dataset_id"]
            operation = validated["operation"]
            selection_config = validated["selection_config"]
            split_approach = validated["split_approach"]
            split_strategy_name = validated["split_strategy_name"]
            description = validated["description"]
        except Exception as e:
            logging.error(str(e))
            raise

        logging.info("Inputs validated.")

        # 2. Load current dataset state
        logging.info("Checking if dataset already exists...")
        try:
            dataset_info = self.get_dataset(dataset_id=dataset_id)
        except Exception as e:
            logging.error(f"Failed loading dataset metadata for '{dataset_id}': {str(e)}")
            raise

        if not dataset_info["dataset_info"]["exists"]:
            logging.error(f"Dataset '{dataset_id}' does not exist.")
            raise ValueError(f"Dataset '{dataset_id}' does not exist.")

        honor_source_splits = dataset_info["dataset_info"]["honor_source_splits"]

        if honor_source_splits and split_approach == "rebalance":
            raise ValueError(
                "You may not rebalance a dataset that has honor_source_splits=True."
            )

        if honor_source_splits and split_approach == "maintain" and split_strategy_name:
            logging.info(
                "Update request for dataset_id='%s' has honor_source_splits=True and "
                "split_approach='maintain', so split_strategy_name='%s' will be ignored. "
                "Existing retained rows keep their current split, and newly added rows will be "
                "assigned directly from image_source_membership.source_split. "
                "Images with conflicting non-empty source splits will be excluded, and images with no "
                "resolved non-empty source split will also be excluded.",
                dataset_id,
                split_strategy_name,
            )
        label_type = dataset_info["dataset_info"]["label_type"]
        current_version = dataset_info["dataset_info"]["latest_version"]
        new_version = current_version + 1

        # We must check that the selection config's allowed_classes is a subset of the dataset wide allowed_classes
        requested_allowed = set(selection_config["allowed_classes"])
        dataset_allowed = set(dataset_info["dataset_info"]["allowed_classes"])

        if not requested_allowed.issubset(dataset_allowed):
            raise ValueError(
                f"Requested update classes {sorted(requested_allowed)} must be a subset of "
                f"{sorted(dataset_allowed)}. To add classes, make a new dataset."
            )

        # 3. Submit task to S3
        payload = {"user": self.user,
                    "event_type": self.event_type,
                    "task_type": "update_dataset",
                    "request": {
                        "dataset_id": dataset_id,
                        "label_type": label_type,
                        "new_version": new_version,
                        "operation": operation,
                        "selection_config": selection_config,
                        "split_approach": split_approach,
                        "split_strategy_name": split_strategy_name,
                        "description": description,
                        "honor_source_splits": honor_source_splits
                    }
                }

        submission = self._submit_job(payload=payload)

        return submission

    def submit_delete_dataset_all_versions(self, *, dataset_id: str) -> dict:
        """
        Submits a request to delete a dataset and all its versions.

        1. validate inputs
        2. verify the dataset exists
        3. submit the request to S3
        """

        # 1. Validate inputs
        logging.info("Validating inputs")

        try:
            validated = validate_delete_dataset_inputs(dataset_id=dataset_id)
            dataset_id = validated["dataset_id"]
        except Exception as e:
            logging.error(str(e))
            raise

        # 2. Load existing dataset metadata and confirm dataset exists
        logging.info("Validation done. Now retrieving the dataset metadata...")
        try:
            dataset_record = self.get_dataset(dataset_id=dataset_id)
        except Exception as e:
            logging.error(f"Failed loading dataset metadata for '{dataset_id}': {str(e)}")
            raise

        if not dataset_record["dataset_info"]["exists"]:
            logging.error(f"Dataset '{dataset_id}' does not exist.")
            raise ValueError(f"Dataset '{dataset_id}' does not exist.")

        label_type = dataset_record["dataset_info"]["label_type"]

        # 3. Submit task to S3
        payload = {"user": self.user,
                   "event_type": self.event_type,
                   "task_type": "delete_dataset",
                   "request": {
                       "dataset_id": dataset_id,
                       "label_type": label_type,
                       "new_version": None
                        }
                   }

        submission = self._submit_job(payload=payload)

        return submission

    def _acquire_lock(self) -> Tuple[bool, str]:
        """
        Try to set locked = True, locked_by = holder (job_id).
        Returns (True, "") on success, (False, error_message) on failure.
        Uses conditional update so only one caller can win.
        """
        lock_id = "global"
        holder = str(uuid.uuid4())
        logging.info(f"Acquiring lock for lock {lock_id}, new potential holder = {holder}, used lock table name {self.lock_table_name}")
        try:
            self.lock_table.update_item(
                Key={"lock_id": lock_id},
                UpdateExpression="SET locked = :true, locked_by = :holder",
                ConditionExpression="attribute_not_exists(locked) OR locked = :false",
                ExpressionAttributeValues={":true": True, ":false": False, ":holder": holder},
                ReturnValues="ALL_NEW",
            )
            return True, holder
        except ClientError as e:
            logging.error(f"Unable to acquire lock for lock id {lock_id}, error message: {e}")
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                return False, "lock_already_held"
            return False, f"dynamodb_error: {e}"

    def _create_job_row(self,
                       job_id: str,
                       *,
                       task_type: str,
                       dataset_id: str,
                       summary: str = "") -> Tuple[bool, str]:
        """
        Insert initial job row with status=PENDING. Returns (True,"") or (False,error).
        Uses a condition to avoid overwriting an existing job_id.
        """
        item = {
            "job_id": job_id,
            "user": self.user,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "PENDING",
            "summary": summary,
            "event_type": self.event_type,
            "task_type": task_type,
            "dataset_id": dataset_id,
            "error": "",
        }

        try:
            self.job_table.put_item(Item=item, ConditionExpression="attribute_not_exists(job_id)")
            return True, ""
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                return False, "job_already_exists"
            return False, f"dynamodb_error: {e}"

    def _release_lock(self, expected_holder: str = "") -> Tuple[bool, str]:
        """
        Release lock only if current locked_by matches expected_holder (the job id holding the lock).
        Returns (True, "") on success.
        """
        lock_id = "global"
        try:
            self.lock_table.update_item(
                Key={"lock_id": lock_id},
                UpdateExpression="SET locked = :false REMOVE locked_by",
                ConditionExpression="locked_by = :holder",
                ExpressionAttributeValues={":false": False, ":holder": expected_holder},
                ReturnValues="ALL_NEW",
            )
            return True, ""
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            logging.error(f"Failed releasing lock for job id {expected_holder}: {e}")
            if code == "ConditionalCheckFailedException":
                return False, "lock_not_held_by_expected_holder"
            return False, f"dynamodb_error: {e}"

    def _update_job_status(self, job_id: str, status: str, error_msg: Optional[str] = None) -> Tuple[bool, str]:
        valid_statuses = ['PENDING', 'IN_PROGRESS', 'FAILED', 'COMPLETED']
        if status not in valid_statuses:
            logging.error(f"Failed updating job status because specified status was invalid: {status}")
            return False, f"invalid_status: {status}"

        try:
            self.job_table.update_item(
                Key={"job_id": job_id},
                UpdateExpression="SET #s = :s, #e = :e",
                ExpressionAttributeNames={"#s": "status", "#e": "error"},
                ExpressionAttributeValues={":s": status, ":e": error_msg or ""},
            )
            logging.info(f"Successfully updated job status to {status}")
            return True, "success"
        except ClientError as e:
            logging.error(f"Failed updating job status: {e}")
            return False, f"dynamodb_error: {e}"

    def _delete_temp_job_folder(self, job_id: str) -> Tuple[bool, str]:
        """
        Deletes all S3 objects under temp/dataset-ops/<job_id>/.
        Returns (True, "deleted") or (False, error_message).
        """
        prefix = f"temp/dataset-ops/{job_id}/"
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.file_bucket_name, Prefix=prefix)

            keys_to_delete = []
            for page in pages:
                for obj in page.get("Contents", []):
                    keys_to_delete.append({"Key": obj["Key"]})

            if not keys_to_delete:
                return True, "no_files_to_delete"

            # Delete in batches of 1000
            for i in range(0, len(keys_to_delete), 1000):
                batch = keys_to_delete[i:i + 1000]
                self.s3_client.delete_objects(
                    Bucket=self.file_bucket_name,
                    Delete={"Objects": batch}
                )

            return True, "deleted"
        except Exception as e:
            return False, f"delete_error: {e}"

    def _upload_submission_to_s3(self,
                               job_id: str,
                               payload: dict) -> Tuple[bool, str]:

        try:
            prefix = f"temp/dataset-ops/{job_id}"
            submission_key = f"{prefix}/submission.json"
            self.s3_client.put_object(
                Bucket=self.file_bucket_name,
                Key=submission_key,
                Body=json.dumps(payload).encode("utf-8"),
                ContentType="application/json"
            )

            logging.info(f"Upload of submission.json success: s3://{self.file_bucket_name}/{submission_key}")

            return True, "success"

        except Exception as e:
            delete_ok, delete_msg = self._delete_temp_job_folder(job_id)

            if not delete_ok:
                logging.error(f"Failed to delete temp folder for job {job_id} after a failed upload attempt. Delete error: {delete_msg}, upload error: {e}")
            else:
                logging.error(f"Upload failed: {e}")
                logging.info(f"Cleaned up temp folder for job {job_id}")

            return False, f"upload_error: {e}"

    def _submit_job(self, *, payload: dict) -> Dict:
        # 1) Try to acquire the global lock.
        ok, holder_or_err = self._acquire_lock()
        if not ok:
            return {"error": f"Failed to acquire lock: {holder_or_err}"}

        job_id = holder_or_err  # holder doubles as the job_id
        dataset_id = payload["request"]["dataset_id"]
        new_version = payload["request"]["new_version"]  # 1=create, N>1=update, None=delete
        label_type = payload["request"]["label_type"]
        honor_source_splits = payload["request"].get("honor_source_splits")

        logging.info(f"Acquired lock: {job_id}")

        # 2) Enrich payload with job-scoped context before upload.
        payload["job_id"] = job_id
        payload["submission_s3_uri"] = (
            f"s3://{self.file_bucket_name}/temp/dataset-ops/{job_id}/submission.json"
        )
        payload["dataset_context"] = {
            "dataset_id": dataset_id,
            "new_version": new_version,
            "label_type": label_type,
            "honor_source_splits": honor_source_splits,
        }

        # 3) Create initial job row.
        task_type = payload["task_type"]
        job_summary = payload["request"].get("description")
        if not job_summary:
            if task_type == "delete_dataset":
                job_summary = f"Deleting dataset_id={dataset_id}"
            else:
                job_summary = "No description provided."

        ok, err = self._create_job_row(
            job_id,
            task_type=task_type,
            dataset_id=dataset_id,
            summary=job_summary,
        )
        if not ok:
            logging.error(f"Failed to create job row: {err}")
            rel_ok, rel_msg = self._release_lock(expected_holder=job_id)
            return {
                "error": (
                    f"Failed to create job row: {err}. "
                    f"Lock released: {rel_ok}, release msg: {rel_msg}"
                )
            }

        logging.info(
            f"Created job row in job table for {self.event_type} event with status=PENDING."
        )

        # 4) Upload submission.json to S3.
        logging.info("Uploading submission file to S3.")
        ok, msg = self._upload_submission_to_s3(job_id, payload)
        if not ok:
            logging.error(f"Failed to upload submission file to S3: {msg}")
            job_ok, job_msg = self._update_job_status(job_id, "FAILED", error_msg=msg)
            rel_ok, rel_msg = self._release_lock(expected_holder=job_id)
            return {
                "error": (
                    f"Failed upload step to S3: {msg}. "
                    f"Updated job status to FAILED attempt: {job_ok}, msg: {job_msg}. "
                    f"Lock released: {rel_ok}, release msg: {rel_msg}"
                )
            }

        logging.info("Done uploading submission.json to S3.")

        # 5) Mark submitted/in progress.
        # IMPORTANT: if this fails, do NOT release the lock, because server-side work
        # may already be progressing after the successful S3 upload.
        job_ok, job_msg = self._update_job_status(job_id, "IN_PROGRESS")
        if not job_ok:
            logging.error(
                f"Failed to update job status to IN_PROGRESS for job_id={job_id}: {job_msg}. "
                "Submission was already uploaded, so server-side work may already be progressing. "
                "Not releasing the lock."
            )
            raise RuntimeError(
                    f"Submission uploaded successfully, but failed to update job status to "
                    f"IN_PROGRESS: {job_msg}. "
                    "Server-side work may already be progressing; inspect the job before retrying.")

        return {"submission_status": "success", "job_id": job_id}