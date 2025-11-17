# cvdms_platform/clients.py
import csv
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict

import boto3
from botocore.exceptions import ClientError


ISO_NOW = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class UploadClient:
    """
    High-level client to create an upload job with a global lock and
    seed the job row in DynamoDB. Networking and actual S3 uploads
    are left to other helpers; this class orchestrates locks + job creation.

    Constructor accepts either resource names (common for local scripts)
    and builds boto3 clients/resources internally.
    """

    def __init__(self,
                 *,
                 region_name: str,
                 file_bucket: str,
                 job_table: str,
                 lock_table: str,
                 s3_client: Optional[boto3.client] = None,
                 dynamodb_resource: Optional[boto3.resource] = None):
        self.region_name = region_name
        self.file_bucket = file_bucket
        self.job_table_name = job_table
        self.lock_table_name = lock_table

        self.s3 = s3_client or boto3.client("s3", region_name=region_name)
        self.dynamodb = dynamodb_resource or boto3.resource("dynamodb", region_name=region_name)

        self.job_table = self.dynamodb.Table(self.job_table_name)
        self.lock_table = self.dynamodb.Table(self.lock_table_name)

    def _generate_job_id(self) -> str:
        return str(uuid.uuid4())

    def acquire_lock(self, lock_id: str = "global", holder: Optional[str] = None) -> Tuple[bool, str]:
        """
        Try to set locked = True, locked_by = holder (job_id).
        Returns (True, "") on success, (False, error_message) on failure.
        Uses conditional update so only one caller can win.
        """
        if holder is None:
            holder = self._generate_job_id()

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
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                return False, "lock_already_held"
            return False, f"dynamodb_error: {e}"

    def release_lock(self, lock_id: str = "global", expected_holder: Optional[str] = None) -> Tuple[bool, str]:
        """
        Release lock only if current locked_by matches expected_holder (if provided).
        Returns (True, "") on success.
        """
        try:
            if expected_holder:
                self.lock_table.update_item(
                    Key={"lock_id": lock_id},
                    UpdateExpression="SET locked = :false REMOVE locked_by",
                    ConditionExpression="locked_by = :holder",
                    ExpressionAttributeValues={":false": False, ":holder": expected_holder},
                    ReturnValues="ALL_NEW",
                )
            else:
                # not checking holder; use only when safe
                self.lock_table.update_item(
                    Key={"lock_id": lock_id},
                    UpdateExpression="SET locked = :false REMOVE locked_by",
                    ExpressionAttributeValues={":false": False},
                    ReturnValues="ALL_NEW",
                )
            return True, ""
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                return False, "lock_not_held_by_expected_holder"
            return False, f"dynamodb_error: {e}"

    def create_job_row(self,
                       job_id: str,
                       *,
                       summary: str = "",
                       job_type: str = "IMAGE_UPLOAD",
                       dataset_id: Optional[str] = None) -> Tuple[bool, str]:
        """
        Insert initial job row with status=PENDING. Returns (True,"") or (False,error).
        Uses a condition to avoid overwriting an existing job_id.
        """
        item = {
            "job_id": job_id,
            "created_at": ISO_NOW(),
            "status": "PENDING",
            "summary": summary,
            "job_type": job_type,
            "errors": "",
        }
        if dataset_id:
            item["dataset_id"] = dataset_id

        try:
            self.job_table.put_item(Item=item, ConditionExpression="attribute_not_exists(job_id)")
            return True, ""
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                return False, "job_already_exists"
            return False, f"dynamodb_error: {e}"

    def _read_paths_from_csv(self, csv_path: str) -> Tuple[bool, list]:
        """
        Read CSV expected to contain one local path per row (first column).
        Returns (True, [paths...]) or (False, error_message).
        """
        try:
            paths = []
            with open(csv_path, newline="") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    path = row[0].strip()
                    if path:
                        paths.append(path)
            return True, paths
        except Exception as e:
            return False, f"csv_read_error: {e}"

    def start_upload_job_from_csv(self,
                                  csv_path: str,
                                  *,
                                  summary: str = "",
                                  dataset_id: Optional[str] = None,
                                  job_type: str = "IMAGE_UPLOAD",
                                  lock_id: str = "global") -> Tuple[bool, Dict]:
        """
        High-level operation a caller will use. Steps:
          1) try to acquire lock
          2) create job row with status=PENDING
          3) read csv and return job_id for caller to continue (actual file uploads implemented elsewhere)
        Returns (True, {"job_id": ...}) on success; (False, {"error": ...}) on failure.

        This method keeps errors explicit so callers can decide to retry or inspect.
        """
        # try to acquire lock
        ok, holder_or_err = self.acquire_lock(lock_id=lock_id)
        if not ok:
            return False, {"error": f"could_not_acquire_lock: {holder_or_err}"}

        job_id = holder_or_err  # we used holder as generated job id in acquire_lock

        # create job row
        ok, err = self.create_job_row(job_id, summary=summary, job_type=job_type, dataset_id=dataset_id)
        if not ok:
            # release lock before returning
            self.release_lock(lock_id=lock_id, expected_holder=job_id)
            return False, {"error": f"could_not_create_job_row: {err}"}

        # read csv for local paths (uploads happen elsewhere)
        ok, paths_or_err = self._read_paths_from_csv(csv_path)
        if not ok:
            # mark job failed and release lock
            msg = paths_or_err
            self.job_table.update_item(
                Key={"job_id": job_id},
                UpdateExpression="SET #s = :s, #e = :e",
                ExpressionAttributeNames={"#s": "status", "#e": "errors"},
                ExpressionAttributeValues={":s": "FAILED", ":e": msg},
            )
            self.release_lock(lock_id=lock_id, expected_holder=job_id)
            return False, {"error": f"could_not_read_csv: {msg}"}

        # At this point we have job_id, PENDING row, and the list of local paths.
        # Caller can either: upload files and then call a "finish" method, or this class
        # can provide an upload helper. For now return job_id and paths.
        return True, {"job_id": job_id, "paths": paths_or_err}

    # small helper to mark job status (callable by client code)
    def update_job_status(self, job_id: str, status: str, error_msg: Optional[str] = None) -> Tuple[bool, str]:
        valid_statuses = ['PENDING', 'IN_PROGRESS', 'FAILED', 'COMPLETED']
        if status not in valid_statuses:
            return False, f"invalid_status: {status}"
        try:
            self.job_table.update_item(
                Key={"job_id": job_id},
                UpdateExpression="SET #s = :s, #e = :e",
                ExpressionAttributeNames={"#s": "status", "#e": "errors"},
                ExpressionAttributeValues={":s": status, ":e": error_msg or ""},
            )
            return True, ""
        except ClientError as e:
            return False, f"dynamodb_error: {e}"
