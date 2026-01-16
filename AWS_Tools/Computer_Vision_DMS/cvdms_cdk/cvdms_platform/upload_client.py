'''
Ground Truth JSONL input semantics expected by validate_manifest() / _gt_row_to_v1()

General (all label types)
- Input is JSON Lines (one JSON object per non-empty line).
- Strict JSON parsing: NaN/Infinity are rejected.
- Each line MUST include "source-ref" as a strict s3://bucket/key URI (no surrounding whitespace).
- "source-ref" MUST end with one of: .jpg, .jpeg, .png (case-insensitive).

Skip rules (ONLY for: single-label, multi-label, object-detection)
- single-label: skip if class-name is empty after strip()
- multi-label:  skip if class-map is an empty dict {}
- object-detection: skip if annotations is an empty list []
- semantic-segmentation / instance-segmentation: no skip permitted; missing/invalid => error

Label-type specifics:

1) single-label
  Required keys:
    - "single-label-metadata": { "class-name": <string> }
  Rules:
    - class-name is normalized: strip().lower()
    - class-name MUST be non-empty to keep the row
    - class-name MUST NOT be in _RESERVED_CLASS_NAMES_LC (currently {"bg","background"})

2) multi-label
  Required keys:
    - "multi-label-metadata": { "class-map": { <id_str>: <class_name_str>, ... } }
  IMPORTANT ASSUMPTION:
    - class-map is scoped to the IMAGE (not dataset-wide): every present value is a label.
    - The GT "multi-label": [...] field is treated as redundant/ignored.
  Rules:
    - class-map values MUST be non-empty strings
    - values normalized: strip().lower()
    - values MUST NOT be in _RESERVED_CLASS_NAMES_LC
    - v1 labels = sorted(set(values)) for determinism
    - skip if class-map is {}

3) object-detection
  Required keys:
    - "object-detection": { "annotations": [ {class_id, top, left, height, width}, ... ] }
    - "object-detection-metadata": { "class-map": { <class_id_str>: <class_name_str>, ... } }
  Rules:
    - annotations must be a list; skip if []
    - each annotation requires fields: class_id, top, left, height, width
    - top/left/height/width accept int/float/numeric-string; must be finite (no NaN/Infinity)
    - top,left >= 0; height,width > 0
    - class_id must be integer-like and must exist in class-map
    - class-map values normalized: strip().lower(); must be non-empty and not reserved

4) semantic-segmentation
  Required keys:
    - "semantic-segmentation-ref": <s3://... .png>
    - "semantic-segmentation-ref-metadata": {
          "internal-color-map": {
              <id>: {"class-name": <string>, "hex-color": "#RRGGBB", ...}, ...
          }
      }
  Background policy:
    - internal-color-map MUST include exactly one class-name equal to "bg" or "background"
      (case-insensitive). v1 output EXCLUDES that background entry.
  Rules:
    - mask ref must end with .png (case-insensitive)
    - each class-name must be non-empty after strip().lower()
    - class-name must be unique (case-insensitive) across entries
    - hex-color must be valid "#RRGGBB"
    - no skip permitted

5) instance-segmentation
  Required keys:
    - "instance-segmentation-metadata": { "worker-response-ref": <s3://... .json> }
  Rules:
    - worker-response-ref must be a valid S3 URI ending in .json
    - no skip permitted
'''

import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict

from botocore.exceptions import ClientError
from mypy_boto3_s3.client import S3Client
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource

from cvdms_platform.upload_client_utils import validate_manifest

_ALLOWED_LABEL_TYPES = {"single-label", "multi-label", "object-detection", "semantic-segmentation", "instance-segmentation"}

class UploadClient:
    """
    High-level client to create an upload job with a global lock and
    seed the job row in DynamoDB.
    """
    def __init__(self,
                 *,
                 user: str,
                 file_bucket_name: str,
                 job_table_name: str,
                 lock_table_name: str,
                 s3_client: S3Client,
                 dynamodb_resource: DynamoDBServiceResource):

        self.event_type = "IMAGE_UPLOAD"
        self.user = user
        self.file_bucket_name = file_bucket_name
        self.job_table_name = job_table_name
        self.lock_table_name = lock_table_name

        self.s3 = s3_client
        self.dynamodb = dynamodb_resource

        self.job_table = self.dynamodb.Table(self.job_table_name)
        self.lock_table = self.dynamodb.Table(self.lock_table_name)

    # small helper to mark job status (callable by client code)
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

    def _create_job_row(self,
                       job_id: str,
                       *,
                       summary: str = "") -> Tuple[bool, str]:
        """
        Insert initial job row with status=PENDING. Returns (True,"") or (False,error).
        Uses a condition to avoid overwriting an existing job_id.
        """
        item = {
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "PENDING",
            "summary": summary,
            "event_type": self.event_type,
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

    def _delete_temp_job_folder(self, job_id: str) -> Tuple[bool, str]:
        """
        Deletes all S3 objects under temp/image-upload/<job_id>/.
        Returns (True, "deleted") or (False, error_message).
        """
        prefix = f"temp/image-upload/{job_id}/"
        try:
            paginator = self.s3.get_paginator("list_objects_v2")
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
                self.s3.delete_objects(
                    Bucket=self.file_bucket_name,
                    Delete={"Objects": batch}
                )

            return True, "deleted"
        except Exception as e:
            return False, f"delete_error: {e}"

    def _upload_files_to_s3(self,
                           job_id: str,
                           label_type: str,
                           manifest_path: str,
                           data_source: str = "") -> Tuple[bool, str]:

        prefix = f"temp/image-upload/{job_id}"

        # Upload the local manifest (manifest_path) and job.json to the prefix.
        try:
            # Upload manifest_path from local drive to S3.
            local_manifest = Path(manifest_path)
            if not local_manifest.exists() or not local_manifest.is_file():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

            manifest_key = f"{prefix}/{job_id}.manifest"

            # Streaming upload from disk (no file_obj.read()), better memory usage.
            self.s3.upload_file(
                Filename=str(local_manifest),
                Bucket=self.file_bucket_name,
                Key=manifest_key,
                ExtraArgs={
                    "ContentType": "application/x-ndjson",  # JSON Lines / NDJSON
                },
            )

            logging.info(f"Upload of manifest success: s3://{self.file_bucket_name}/{manifest_key}")

            # Upload job.json
            job_manifest = {
                "job_id": job_id,
                "user": self.user,
                "event_type": self.event_type,
                "label_type": label_type,
                "data_source":data_source,
                "registration_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "original_manifest_s3_uri": f"s3://{self.file_bucket_name}/{manifest_key}"
            }
            job_json_key = f"{prefix}/job.json"
            self.s3.put_object(
                Bucket=self.file_bucket_name,
                Key=job_json_key,
                Body=json.dumps(job_manifest).encode("utf-8"),
                ContentType="application/json"
            )

            logging.info(f"Upload of job.json success: s3://{self.file_bucket_name}/{job_json_key}")

            return True, "success"

        except Exception as e:
            delete_ok, delete_msg = self._delete_temp_job_folder(job_id)

            if not delete_ok:
                logging.error(f"Failed to delete temp folder for job {job_id} after a failed upload attempt. Delete error: {delete_msg}, upload error: {e}")
            else:
                logging.error(f"Upload failed: {e}")
                logging.info(f"Cleaned up temp folder for job {job_id}")

            return False, f"upload_error: {e}"

    def start_upload_job(self,
                          manifest_path: str,
                          label_type: str,
                          *,
                          job_summary: str = "",
                          data_source: str = "") -> Dict:
        """
        High-level operation a caller will use. Steps:
          1) try to acquire lock
          2) create job row with status=PENDING
          3) read, validate, and uploaded standardized manifest and return job_id for caller to continue
        Returns {"job_id": ...} on success; {"error": ...} on failure.

        This method keeps errors explicit so callers can decide to retry or inspect.
        """
        if label_type not in _ALLOWED_LABEL_TYPES:
            return {"error": f"Invalid label type: {label_type}, must be one of {_ALLOWED_LABEL_TYPES}"}

        # try to acquire lock
        ok, holder_or_err = self._acquire_lock()
        if not ok:
            logging.error(f"Failed to acquire lock: {holder_or_err}")
            return {"error": f"could not acquire lock: {holder_or_err}"}

        job_id = holder_or_err  # we used holder as generated job id in acquire lock
        logging.info(f"Acquired lock: {job_id}")

        # create job row
        ok, err = self._create_job_row(job_id, summary=job_summary)
        if not ok:
            logging.error(f"Failed to create job row: {err}")
            # release lock before returning
            self._release_lock(expected_holder=job_id)
            return {"error": f"could not create job row: {err}"}

        logging.info(f"Created job row in job table for {self.event_type} event and is status: PENDING.")

        # Validate the manifest. Ensure it has the expected structure and s3 uri's are valid (formatted correctly).
        validation_dict = validate_manifest(manifest_path, label_type)
        if not validation_dict.get('success'):
            err = validation_dict.get('error')
            # mark job failed and release lock
            logging.error(f"Failed to load and validate manifest: {err}")
            self._update_job_status(job_id, "FAILED", error_msg=err)
            self._release_lock(expected_holder=job_id)
            return {"error": err}
        else:
            manifest_path = validation_dict['local_path']

        # At this point we have job_id, PENDING row, and manifest json loaded in.
        skipped_count = validation_dict["skipped_count"]
        kept_count = validation_dict["kept_count"]
        total_nonempty = validation_dict["total_nonempty"]

        logging.info(f'Manifest validated, skipped row count = {skipped_count}, kept row count = {kept_count}, total nonempty row count = {total_nonempty}')
        if skipped_count > 0:
            logging.info(f"Skipped rows, so needed to save filtered manifest to: {manifest_path}")

        logging.info('Uploading files to S3.')
        ok, msg = self._upload_files_to_s3(job_id,
                                          label_type,
                                          manifest_path,
                                          data_source=data_source)
        if not ok:
            logging.error(f"Failed to upload files to S3: {msg}")
            self._update_job_status(job_id, "FAILED", error_msg=msg)
            self._release_lock(expected_holder=job_id)
            return {"error": f"Failed upload step: {msg}"}

        logging.info("Done uploading manifest and job.json to S3.")
        self._update_job_status(job_id, "IN_PROGRESS", error_msg=msg)

        return {"submission_status": "success", "job_id": job_id}