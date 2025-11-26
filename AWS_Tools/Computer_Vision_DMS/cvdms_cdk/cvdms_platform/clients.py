import pandas as pd
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict
import logging
import time
import boto3
from botocore.exceptions import ClientError
from botocore.client import BaseClient
from boto3.resources.base import ServiceResource

ISO_NOW = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

VALID_LABEL_COLUMNS = {
    "string_labels",
    "bounding_boxes",
    "semantic_masks",
    "mask_map",
    "instance_annotations"
}

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
                 s3_client: BaseClient,
                 dynamodb_resource: ServiceResource):

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
    def update_job_status(self, job_id: str, status: str, error_msg: Optional[str] = None) -> Tuple[bool, str]:
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

    def acquire_lock(self) -> Tuple[bool, str]:
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
            logging.error('Unable to acquire lock for lock id {lock_id}, error message: {e}')
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                return False, "lock_already_held"
            return False, f"dynamodb_error: {e}"

    def release_lock(self, expected_holder: str = "") -> Tuple[bool, str]:
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

    def create_job_row(self,
                       job_id: str,
                       *,
                       summary: str = "") -> Tuple[bool, str]:
        """
        Insert initial job row with status=PENDING. Returns (True,"") or (False,error).
        Uses a condition to avoid overwriting an existing job_id.
        """
        item = {
            "job_id": job_id,
            "created_at": ISO_NOW(),
            "status": "PENDING",
            "summary": summary,
            "event_type": self.event_type,
            "errors": "",
        }

        try:
            self.job_table.put_item(Item=item, ConditionExpression="attribute_not_exists(job_id)")
            return True, ""
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                return False, "job_already_exists"
            return False, f"dynamodb_error: {e}"

    def _load_and_validate_csv(self, csv_path: str) -> Tuple[bool, Dict]:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return False, {"csv_read_error": str(e)}

        if "path" not in df.columns:
            return False, {"missing_column": "'path' column is required"}

        unexpected_cols = set(df.columns) - VALID_LABEL_COLUMNS - {"path"}
        if unexpected_cols:
            return False, {"unexpected_columns": sorted(unexpected_cols)}

        if ("semantic_masks" in df.columns) ^ ("mask_map" in df.columns):
            return False, {"mask_column_mismatch": "Both 'semantic_masks' and 'mask_map' must be present together"}

        df["path"] = df["path"].astype(str).str.strip()
        df = df[df["path"].notnull()]
        df = df.drop_duplicates(subset=["path"])
        df = df[df["path"].apply(os.path.exists)]

        if "string_labels" in df.columns:
            df["string_labels"] = (
                df["string_labels"]
                .astype(str)  # ensure string type
                .str.strip()  # remove leading/trailing spaces
                .str.lower()  # lowercase
                .str.replace(r"\s+", "_")  # replace internal spaces with underscores
            )

        image_format_validator = lambda path: os.path.splitext(path)[1].lower() in [".jpg", ".jpeg", ".png"]

        df = df[df["path"].map(image_format_validator)]

        if df.empty:
            return False, {"no_valid_rows": "CSV is empty or contains no valid image paths after filtering (e.g., extensions must be jpeg, jpg, or png, case insensitive)"}

        error_dict = {}

        for idx, row in df.iterrows():
            row_errors = []

            # bounding_boxes
            if "bounding_boxes" in row and pd.notna(row["bounding_boxes"]):
                bb_path = str(row["bounding_boxes"]).strip()
                if not os.path.exists(bb_path):
                    row_errors.append("bounding_box_missing")
                elif not bb_path.endswith(".json"):
                    row_errors.append("bounding_box_not_json")

            # semantic_masks
            if "semantic_masks" in row and pd.notna(row["semantic_masks"]):
                mask_path = str(row["semantic_masks"]).strip()
                if not os.path.exists(mask_path):
                    row_errors.append("semantic_mask_missing")
                elif not mask_path.lower().endswith(".png"):
                    row_errors.append("semantic_mask_not_png")

                # mask_map
                try:
                    mask_map_str = str(row["mask_map"]).strip()
                    mask_map = json.loads(mask_map_str)
                    if "0" not in mask_map or mask_map["0"].lower() != "bg":
                        row_errors.append("mask_map_missing_bg")
                    keys = sorted(map(int, mask_map.keys()))
                    if keys != list(range(len(keys))):
                        row_errors.append("mask_map_keys_not_sequential")
                except Exception:
                    row_errors.append("mask_map_invalid_json")

            # instance_annotations
            if "instance_annotations" in row and pd.notna(row["instance_annotations"]):
                ia_path = str(row["instance_annotations"]).strip()
                if not os.path.exists(ia_path):
                    row_errors.append("instance_annotation_missing")
                elif not ia_path.endswith(".json"):
                    row_errors.append("instance_annotation_not_json")

            # accumulate errors
            for err in row_errors:
                if err not in error_dict:
                    error_dict[err] = {"rows_affected": 0}
                error_dict[err]["rows_affected"] += 1

        self.df = df

        if error_dict:
            return False, error_dict

        return True, {"message": "success"}

    def delete_temp_job_folder(self, job_id: str) -> Tuple[bool, str]:
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

    def upload_files_to_s3(self, job_id: str, source: str = "") -> Tuple[bool, str]:
        """
        Uploads all files referenced in self.df to the appropriate S3 temp folder for the given job_id.
        Returns (True, "success") or (False, error_message).
        """
        if not hasattr(self, "df"):
            return False, "No DataFrame loaded. Run start_upload_job_from_csv first."

        prefix = f"temp/image-upload/{job_id}"
        manifest_prefix = "temp/image-upload"
        try:
            for _, row in self.df.iterrows():
                image_path = row["path"]
                base_uuid = str(uuid.uuid4())
                image_ext = os.path.splitext(image_path)[1].lower()
                image_key = f"{prefix}/images/{base_uuid}{image_ext}"

                # Upload image
                self.s3.upload_file(image_path, self.file_bucket_name, image_key)

                # Upload string_labels
                if "string_labels" in row and pd.notna(row["string_labels"]):
                    labels = [l.strip() for l in str(row["string_labels"]).split(",") if l.strip()]
                    label_obj = {"string_labels": labels}
                    label_key = f"{prefix}/string_labels/{base_uuid}.json"
                    self.s3.put_object(
                        Bucket=self.file_bucket_name,
                        Key=label_key,
                        Body=json.dumps(label_obj).encode("utf-8"),
                        ContentType="application/json"
                    )

                # Upload bounding_boxes
                if "bounding_boxes" in row and pd.notna(row["bounding_boxes"]):
                    bb_path = str(row["bounding_boxes"]).strip()
                    bb_key = f"{prefix}/bounding_boxes/{base_uuid}.json"
                    self.s3.upload_file(bb_path, self.file_bucket_name, bb_key)

                # Upload semantic_masks and mask_map
                if "semantic_masks" in row and pd.notna(row["semantic_masks"]):
                    mask_path = str(row["semantic_masks"]).strip()
                    mask_key = f"{prefix}/semantic_masks/{base_uuid}.png"
                    self.s3.upload_file(mask_path, self.file_bucket_name, mask_key)

                    mask_map_str = str(row["mask_map"]).strip()
                    mask_map_key = f"{prefix}/semantic_masks/{base_uuid}.json"
                    self.s3.put_object(
                        Bucket=self.file_bucket_name,
                        Key=mask_map_key,
                        Body=mask_map_str.encode("utf-8"),
                        ContentType="application/json"
                    )

                # Upload instance_annotations
                if "instance_annotations" in row and pd.notna(row["instance_annotations"]):
                    ia_path = str(row["instance_annotations"]).strip()
                    ia_key = f"{prefix}/instance_annotations/{base_uuid}.json"
                    self.s3.upload_file(ia_path, self.file_bucket_name, ia_key)

            # Upload job.json manifest
            job_manifest = {
                "job_id": job_id,
                "user": self.user,
                "num_images": len(self.df),
                "source":source,
                "event_type": self.event_type,
                "label_types": [col for col in self.df.columns if col in VALID_LABEL_COLUMNS and col != "mask_map"]
            }
            manifest_key = f"{manifest_prefix}/job.json"
            self.s3.put_object(
                Bucket=self.file_bucket_name,
                Key=manifest_key,
                Body=json.dumps(job_manifest).encode("utf-8"),
                ContentType="application/json"
            )

            return True, "success"

        except Exception as e:
            delete_ok, delete_msg = self.delete_temp_job_folder(job_id)

            if not delete_ok:
                logging.error(f"Failed to delete temp folder for job {job_id} after a failed upload attempt. Delete error: {delete_msg}, upload error: {e}")

            return False, f"upload_error: {e}"

    def start_upload_job_from_csv(self,
                                  csv_path: str,
                                  *,
                                  summary: str = "",
                                  source: str = "") -> Tuple[bool, Dict]:
        """
        High-level operation a caller will use. Steps:
          1) try to acquire lock
          2) create job row with status=PENDING
          3) read csv and return job_id for caller to continue (actual file uploads implemented elsewhere)
        Returns (True, {"job_id": ...}) on success; (False, {"error": ...}) on failure.

        This method keeps errors explicit so callers can decide to retry or inspect.
        """
        # try to acquire lock
        ok, holder_or_err = self.acquire_lock()
        if not ok:
            logging.error(f"Failed to acquire lock: {holder_or_err}")
            return False, {"error": f"could_not_acquire_lock: {holder_or_err}"}

        job_id = holder_or_err  # we used holder as generated job id in acquire_lock
        logging.info(f"Acquired lock: {job_id}")

        # create job row
        ok, err = self.create_job_row(job_id, summary=summary)
        if not ok:
            logging.error(f"Failed to create job row: {err}")
            # release lock before returning
            self.release_lock(expected_holder=job_id)
            return False, {"error": f"could_not_create_job_row: {err}"}

        logging.info(f"Created job row in job table for {self.event_type} event and is status: PENDING.")

        # read csv and assign self.df if ok.
        ok, errors_dict = self._load_and_validate_csv(csv_path)
        if not ok:
            # mark job failed and release lock
            logging.error(f"Failed to load and validate csv: {errors_dict}")
            msg = json.dumps(errors_dict)
            self.update_job_status(job_id, "FAILED", error_msg=msg)
            self.release_lock(expected_holder=job_id)
            return False, {"error": errors_dict}

        # At this point we have job_id, PENDING row, and self.df to upload to temp folder.
        logging.info("CSV loaded in and validated. Uploading to S3...")
        ok, msg = self.upload_files_to_s3(job_id, source=source)
        if not ok:
            logging.error(f"Failed to upload files to S3: {msg}")
            self.update_job_status(job_id, "FAILED", error_msg=msg)
            self.release_lock(expected_holder=job_id)
            return False, {"error": f"Failed upload step: {msg}"}

        logging.info("Done uploading files to S3.")
        self.update_job_status(job_id, "IN_PROGRESS", error_msg=msg)

        return True, {"submission_status": "success", "job_id": job_id}

class LogClient:
    """
    High-level client to query logs.
    """
    def __init__(self,
                 *,
                 glue_db_name: str,
                 glue_table_name: str,
                 log_bucket_name: str,
                 athena_client: BaseClient):

        self.glue_db_name = glue_db_name
        self.glue_table_name = glue_table_name
        self.log_bucket_name = log_bucket_name
        self.athena = athena_client


    def _run_query(self, query: str) -> Dict:
        resp = self.athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": self.glue_db_name},
            ResultConfiguration={
                "OutputLocation": f"s3://{self.log_bucket_name}/athena-results/"
            }
        )
        qid = resp["QueryExecutionId"]

        # poll until finished
        while True:
            status = self.athena.get_query_execution(QueryExecutionId=qid)
            state = status["QueryExecution"]["Status"]["State"]
            if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                break
            time.sleep(2)

        if state != "SUCCEEDED":
            raise RuntimeError(f"Athena query failed with state {state}")

        return self.athena.get_query_results(QueryExecutionId=qid)

    def get_logs_by_job_id(self, job_id: str) -> Tuple[bool, Dict, Optional[pd.DataFrame]]:
        try:
            # Step 1: repair partitions so new data is visible
            self._run_query(f"MSCK REPAIR TABLE {self.glue_table_name}")

            # Step 2: run the actual log query
            results = self._run_query(
                f"""
                SELECT *
                FROM {self.glue_table_name}
                WHERE job_id = '{job_id}'
                ORDER BY timestamp DESC
                """
            )

            # Extract column names
            columns = [col["Name"] for col in results["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]

            # Extract rows (skip the header row)
            rows = results["ResultSet"]["Rows"][1:]
            data = []
            for row in rows:
                values = []
                for cell in row.get("Data", []):
                    values.append(cell.get("VarCharValue"))
                data.append(values)

            # Build DataFrame if we have data
            df = pd.DataFrame(data, columns=columns) if data else None

            return True, results, df

        except Exception as e:
            logging.error(f"Error querying Athena: {e}")
            return False, {"error": str(e)}, None
