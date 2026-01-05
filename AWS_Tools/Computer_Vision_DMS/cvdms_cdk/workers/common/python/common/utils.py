import time
import json
import math
import logging
from decimal import Decimal
from typing import List, Sequence
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

INT_COLS = {"img_height", "img_width", "num_channels"}
FLOAT_COLS = {"file_size_mb"}

logger = logging.getLogger()
logger.setLevel(logging.INFO)

firehose = boto3.client("firehose")
dynamodb = boto3.resource("dynamodb")
athena = boto3.client("athena")
s3 = boto3.client("s3")

UPLOAD_STAGING_COLS = [
    "job_id", "image_id", "temp_source_ref",
    "img_type", "img_height", "img_width", "num_channels", "dtype",
    "file_size_mb", "uploaded_at", "data_source", "sha256_hash",
    "string_labels", "temp_source_ref_bbox_meta", "temp_source_ref_semantic_png",
    "temp_source_ref_semantic_meta", "temp_source_ref_instance_png", "temp_source_ref_instance_meta",
    "classes_present", "validation_status", "validation_error",
    "dedup_status", "dedup_error", "registration_status", "registration_error", "matched_image_id"
]

CANONICAL_IMAGERY_COLS = [
    "image_id", "source_ref", "img_type",
    "img_height", "img_width", "num_channels", "dtype",
    "file_size_mb", "uploaded_at", "data_source", "sha256_hash",
    "string_labels", "bbox_annotation_ids", "semantic_mask_ids",
    "instance_annotation_ids"
]

CANONICAL_BBOX_COLS = ["bbox_annotation_id", "image_id", "source_ref_meta", "classes_present"]
CANONICAL_SEMANTIC_COLS = ["semantic_mask_id", "image_id", "source_ref_png", "source_ref_meta", "classes_present"]
CANONICAL_INSTANCE_COLS = ["instance_annotation_id", "image_id", "source_ref_png", "source_ref_meta", "classes_present"]

def log(job_id, user, event_type, message, stream_name, warning=None, error=None, level="info"):
    entry = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "message": message,
        "warning": warning,
        "error": error,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    # CloudWatch log for operational visibility
    line = json.dumps(entry)
    if level.lower() == "error":
        logger.error(line)
    else:
        logger.info(line)

    # Firehose DirectPut (JSON line)
    try:
        firehose.put_record(
            DeliveryStreamName=stream_name,
            Record={"Data": (line + "\n").encode("utf-8")}
        )
    except Exception as e:
        # Do not fail the handler—the design prefers non-DLQ behavior.
        # Optionally log the failure; avoid recursion by not calling log() again.
        logger.error(json.dumps({
            "job_id": job_id,
            "user": user,
            "event_type": event_type,
            "message": "Failed to put log to Firehose",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }))

def update_job_status(job_id,
                      status,
                      job_table_name,
                      stream_name,
                      user = 'unknown',
                      event_type = 'unknown',
                      error_msg = None):

    valid_statuses = ['PENDING', 'IN_PROGRESS', 'FAILED', 'COMPLETED']
    if status not in valid_statuses:
        log(job_id, user, event_type, "Job status update failed.", stream_name, error=f"Failed to update job status because status {status} is invalid.", level="error")
        return False, f"invalid status: {status}"

    try:
        job_table = dynamodb.Table(job_table_name)
        job_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #s = :s, #e = :e",
            ExpressionAttributeNames={"#s": "status", "#e": "error"},
            ExpressionAttributeValues={":s": status, ":e": error_msg},
            ConditionExpression="attribute_exists(job_id)",
        )
        return True, ""
    except ClientError as e:
        log(job_id, user, event_type, "Job status update failed.", stream_name, error=f"Failed to update job status due to error: {e}", level="error")
        code = e.response.get("Error", {}).get("Code", "")
        if code == "ConditionalCheckFailedException":
            return False, f"job not found: {job_id}"
        return False, str(e)

def release_lock(job_id,
                 lock_table_name,
                 stream_name,
                 user='unknown',
                 event_type='unknown'
                 ):
    """
    Release lock only if current locked_by matches job_id (the job id holding the lock).
    Returns (True, "") on success.
    """
    lock_id = "global"
    try:
        lock_table = dynamodb.Table(lock_table_name)
        lock_table.update_item(
            Key={"lock_id": lock_id},
            UpdateExpression="SET locked = :false REMOVE locked_by",
            ConditionExpression="locked_by = :holder",
            ExpressionAttributeValues={":false": False, ":holder": job_id},
            ReturnValues="ALL_NEW",
        )
        return True, ""
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        error_msg = f"Failed releasing lock for job id {job_id}: {e}"
        log(job_id, user, event_type, error_msg, stream_name, error=error_msg, level="error")
        if code == "ConditionalCheckFailedException":
            return False, f"lock_not_held_by_job_id: {job_id}"
        return False, f"dynamodb_error: {e}"

def delete_s3_prefix(bucket: str, prefix: str, batch_size: int = 100) -> None:
    """
    Delete all objects under s3://{bucket}/{prefix} in batches (default 100).
    Raises on any AWS error or if S3 reports per-key delete errors.

    Notes:
      - S3 DeleteObjects supports up to 1000 keys per request; we default to 100.
      - list_objects_v2 can return pages without "Contents".
    """
    if batch_size < 1 or batch_size > 1000:
        raise ValueError(f"batch_size must be between 1 and 1000, got {batch_size}")

    paginator = s3.get_paginator("list_objects_v2")
    to_delete: list[dict] = []

    def flush() -> None:
        nonlocal to_delete
        if not to_delete:
            return

        try:
            resp = s3.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})
        except ClientError as e:
            logger.error(
                f"Failed to delete objects for s3://{bucket}/{prefix} "
                f"(batch_size={len(to_delete)}): {e}"
            )
            raise

        # DeleteObjects can succeed but still report per-key errors.
        errors = resp.get("Errors", [])
        if errors:
            # Log a small sample to avoid huge logs
            sample = errors[:10]
            logger.error(
                f"S3 reported {len(errors)} delete error(s) for s3://{bucket}/{prefix}. "
                f"Sample: {sample}"
            )
            raise RuntimeError(
                f"S3 delete_objects returned {len(errors)} errors for s3://{bucket}/{prefix}"
            )

        to_delete = []

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                to_delete.append({"Key": obj["Key"]})
                if len(to_delete) >= batch_size:
                    flush()

        flush()
    except ClientError as e:
        logger.error(f"Error while listing/deleting s3://{bucket}/{prefix}: {e}")
        raise

def s3_list_keys(bucket: str, prefix: str) -> List[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                keys.append(key)
    return keys

def wait_for_athena(query_execution_id,
                    poll=1.5,
                    timeout=900):
    """Poll Athena until query completes or times out. Returns True if succeeded, False otherwise."""
    start = time.time()
    while True:
        try:
            resp = athena.get_query_execution(QueryExecutionId=query_execution_id)
            state = resp["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                return {"state": state, "metadata": resp}
            if time.time() - start > timeout:
                return {"state": state, "metadata": resp}
            time.sleep(poll)
        except Exception as e:
            raise Exception(f'Exception in wait_for_athena: {e}')

def _escape_sql_string(s: str) -> str:
    return s.replace("'", "''")

def _coerce_int(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return None  # avoid True -> 1 surprises
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if math.isfinite(v) and v.is_integer():
            return int(v)
        return None
    if isinstance(v, Decimal):
        try:
            f = float(v)
        except Exception:
            return None
        return int(f) if (math.isfinite(f) and f.is_integer()) else None
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        if math.isfinite(f) and f.is_integer():
            return int(f)
        return None
    return None

def _coerce_float(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float, Decimal)):
        try:
            f = float(v)
        except Exception:
            return None
        return f if math.isfinite(f) else None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None

def to_sql_value(r, c):
    v = r.get(c)

    # NULL handling (note: empty string is NOT NULL generally, except uploaded_at below)
    if v is None:
        return "NULL"

    # avoid True -> 1, False -> 0
    if isinstance(v, bool):
        return "NULL"

    # int columns
    if c in INT_COLS:
        iv = _coerce_int(v)
        return "NULL" if iv is None else str(iv)

    # float/double columns
    if c in FLOAT_COLS:
        fv = _coerce_float(v)
        return "NULL" if fv is None else str(fv)

    # timestamp column
    if c == "uploaded_at":
        if not v:
            return "NULL"
        # expect string like "YYYY-MM-DD HH:MM:SS"
        if isinstance(v, str):
            return f"TIMESTAMP '{_escape_sql_string(v.strip())}'"
        return "NULL"

    # numeric (non-special)
    if isinstance(v, (int, float, Decimal)):
        fv = float(v)
        return "NULL" if not math.isfinite(fv) else str(v)

    # arrays
    if isinstance(v, list):
        if len(v) == 0:
            # pick type based on column
            # all your arrays here are array<string>
            return "CAST(ARRAY[] AS ARRAY(VARCHAR))"
        return "ARRAY[" + ", ".join("'" + _escape_sql_string(str(x)) + "'" for x in v) + "]"

    # default string
    return "'" + _escape_sql_string(str(v)) + "'"

def _athena_error_details(qid: str) -> str:
    qe = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
    st = qe.get("Status", {})
    ae = st.get("AthenaError")
    if ae:
        return f"{st.get('StateChangeReason','unknown')} | AthenaError={ae}"
    return st.get("StateChangeReason", "unknown")

def _row_type_summary(r: dict, cols: list[str]) -> str:
    parts = []
    for c in cols:
        v = r.get(c)
        parts.append(f"{c}={type(v).__name__}")
    return ", ".join(parts)

def _run_athena(sql: str, op: str, athena_output_s3: str, athena_workgroup: str) -> None:
    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": athena_output_s3},
        WorkGroup=athena_workgroup
    )["QueryExecutionId"]

    res = wait_for_athena(qid)
    if res["state"] != "SUCCEEDED":
        reason = _athena_error_details(qid)
        raise RuntimeError(f"{op} failed qid={qid}, reason={reason}")

def _table_columns(table_name: str) -> list[str]:
    if table_name == "upload_staging":
        return UPLOAD_STAGING_COLS
    if table_name == "canonical_imagery":
        return CANONICAL_IMAGERY_COLS
    if table_name == "canonical_bounding_boxes":
        return CANONICAL_BBOX_COLS
    if table_name == "canonical_semantic_masks":
        return CANONICAL_SEMANTIC_COLS
    if table_name == "canonical_instance_annotations":
        return CANONICAL_INSTANCE_COLS
    raise ValueError(f"Table name not recognized: {table_name}")


def _table_key_columns(table_name: str) -> list[str]:
    """
    Key columns used for idempotent delete-then-insert.
    Note: Iceberg/Athena doesn't enforce PK uniqueness, so we implement "upsert" via delete-by-key.
    """
    if table_name == "upload_staging":
        # scoped by job_id + image_id (and image_id is already job-scoped in your current scheme)
        return ["job_id", "image_id"]
    if table_name == "canonical_imagery":
        return ["image_id"]
    if table_name == "canonical_bounding_boxes":
        return ["bbox_annotation_id"]
    if table_name == "canonical_semantic_masks":
        return ["semantic_mask_id"]
    if table_name == "canonical_instance_annotations":
        return ["instance_annotation_id"]
    raise ValueError(f"Table name not recognized: {table_name}")

def _build_insert_sql(batch: list[dict], table: str, columns: list[str]) -> str:
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise TypeError("columns must be a list[str]")

    values_clause = []
    for r in batch:
        values = [to_sql_value(r, c) for c in columns]
        values_clause.append("(" + ", ".join(values) + ")")

    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES " + ", ".join(values_clause)

def _build_delete_sql_by_keys(batch: list[dict], table: str, key_cols: list[str]) -> str:
    """
    Builds a targeted delete statement for the IDs in this batch.
    - upload_staging: DELETE WHERE job_id='..' AND image_id IN (...)
    - other tables:   DELETE WHERE <id_col> IN (...)
    """
    if not batch:
        raise ValueError("batch is empty")
    if not isinstance(key_cols, list) or not all(isinstance(c, str) for c in key_cols):
        raise TypeError("key_cols must be a list[str]")

    # Special case: upload_staging uses job_id + image_id
    if key_cols == ["job_id", "image_id"]:
        job_id = batch[0].get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise RuntimeError("delete-then-insert(upload_staging): missing/invalid job_id in batch[0]")

        safe_job_id = _escape_sql_string(job_id.strip())

        # Collect unique image_ids
        seen = set()
        uniq_ids: list[str] = []
        for r in batch:
            iid = r.get("image_id")

            if r.get("job_id") != job_id:
                raise RuntimeError("delete-then-insert(upload_staging): mixed job_id in batch")

            if not isinstance(iid, str) or not iid.strip():
                raise RuntimeError("delete-then-insert(upload_staging): missing/invalid image_id in batch row")

            if iid not in seen:
                seen.add(iid)
                uniq_ids.append(iid)

        in_list = ", ".join("'" + _escape_sql_string(i.strip()) + "'" for i in uniq_ids)
        return f"DELETE FROM {table} WHERE job_id = '{safe_job_id}' AND image_id IN ({in_list})"

    # General case: single id column IN (...)
    if len(key_cols) != 1:
        raise RuntimeError(f"Unsupported key_cols shape for {table}: {key_cols}")

    id_col = key_cols[0]

    seen = set()
    uniq_ids: list[str] = []
    for r in batch:
        v = r.get(id_col)
        if not isinstance(v, str) or not v.strip():
            raise RuntimeError(f"delete-then-insert({table}): missing/invalid {id_col} in batch row")
        if v not in seen:
            seen.add(v)
            uniq_ids.append(v)

    in_list = ", ".join("'" + _escape_sql_string(i.strip()) + "'" for i in uniq_ids)
    return f"DELETE FROM {table} WHERE {id_col} IN ({in_list})"

def chunked_insert(rows,
                  iceberg_db_name,
                  table_name,
                  athena_workgroup,
                  athena_output_s3,
                  chunk_size=200):
    """
    Idempotent chunk writer:
      - DELETE by key(s) for the chunk
      - INSERT the chunk

    This makes replays safe even if:
      - Athena INSERT succeeded but Lambda/polling failed
      - the state is manually re-run / redriven
      - the shard is rerun after partial progress

    Returns (all_failed, last_error) to preserve your existing call sites.
    """
    if not isinstance(chunk_size, int):
        return True, f"chunk_size must be int, got {type(chunk_size).__name__}"
    if not (0 < chunk_size <= 1000):
        return True, f"chunk_size must be 1..1000, got {chunk_size}"

    if not rows:
        return False, ""

    columns = _table_columns(table_name)
    key_cols = _table_key_columns(table_name)

    table = f"\"{iceberg_db_name}\".\"{table_name}\""

    last_error = ""

    for i in range(0, len(rows), chunk_size):
        batch = rows[i:i + chunk_size]
        if not batch:
            continue

        try:
            delete_sql = _build_delete_sql_by_keys(batch, table, key_cols)
            _run_athena(delete_sql, op=f"DELETE({table_name} chunk)", athena_output_s3=athena_output_s3, athena_workgroup=athena_workgroup)

            insert_sql = _build_insert_sql(batch, table, columns)
            _run_athena(insert_sql, op=f"INSERT({table_name} chunk)", athena_output_s3=athena_output_s3, athena_workgroup=athena_workgroup)

        except Exception as e:
            last_error = str(e)
            return True, last_error

    return False, last_error

def delete_iceberg_partition_rows(job_id: str,
                                    iceberg_db_name,
                                    table_name,
                                    athena_output_s3,
                                    athena_workgroup,
                                    poll_interval: int = 5,
                                    timeout_seconds: int = 1800,
                                    run_compaction: bool = True):
    """
    Delete all rows for a given job_id from an Iceberg table and optionally compact.
    Returns a dict with query ids and final states for DELETE and OPTIMIZE.
    """
    # Escape single quotes in job_id for SQL literal safety
    safe_job_id = job_id.replace("'", "''")
    full_table = f"{iceberg_db_name}.{table_name}"

    # 1) DELETE statement (Iceberg positional delete files)
    delete_sql = f"DELETE FROM {full_table} WHERE job_id = '{safe_job_id}'"
    delete_resp = athena.start_query_execution(
        QueryString=delete_sql,
        ResultConfiguration={"OutputLocation": athena_output_s3},
        WorkGroup=athena_workgroup
    )
    delete_qid = delete_resp["QueryExecutionId"]
    delete_result = wait_for_athena(delete_qid, poll=poll_interval, timeout=timeout_seconds)

    if delete_result["state"] != "SUCCEEDED":
        raise RuntimeError(f"[VAL_INGEST_PRE] DELETE failed: {delete_result}")

    result = {
        "delete_query_id": delete_qid,
        "delete_state": delete_result["state"],
        "delete_resp": delete_result["metadata"]
    }

    # 2) Optional: compact / rewrite data for that partition to remove position deletes
    #    Use OPTIMIZE ... REWRITE DATA USING BIN_PACK WHERE job_id = '...'
    #    (WHERE may only reference partition columns; job_id is partitioned in your table)
    if run_compaction and delete_result["state"] == "SUCCEEDED":
        optimize_sql = f"OPTIMIZE {full_table} REWRITE DATA USING BIN_PACK WHERE job_id = '{safe_job_id}'"
        opt_resp = athena.start_query_execution(
            QueryString=optimize_sql,
            ResultConfiguration={"OutputLocation": athena_output_s3},
            WorkGroup=athena_workgroup
        )
        opt_qid = opt_resp["QueryExecutionId"]
        opt_result = wait_for_athena(opt_qid)
        result.update({
            "optimize_query_id": opt_qid,
            "optimize_state": opt_result["state"]
        })

    return result

def athena_count_job_rows(job_id: str,
                           db_name: str,
                           table_name: str,
                           athena_output_s3: str,
                           task_name: str,
                           athena_workgroup: str = "primary") -> int:
    """COUNT(*) from upload_staging WHERE job_id='<job_id>'."""
    safe_job_id = job_id.replace("'", "''")
    sql = (
        f"SELECT count(*) as cnt FROM \"{db_name}\".\"{table_name}\" "
        f"WHERE job_id = '{safe_job_id}'"
    )
    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": athena_output_s3},
        WorkGroup=athena_workgroup,
    )["QueryExecutionId"]

    res = wait_for_athena(qid, poll=2.0, timeout=600)
    if res["state"] != "SUCCEEDED":
        raise RuntimeError(f"[{task_name}] Athena count failed: {res['metadata']}")

    out = athena.get_query_results(QueryExecutionId=qid)
    rows = out.get("ResultSet", {}).get("Rows", [])
    if len(rows) < 2 or not rows[1].get("Data"):
        return 0
    val = rows[1]["Data"][0].get("VarCharValue")

    return int(val) if val is not None else 0

def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not isinstance(s3_uri, str) or not s3_uri.startswith("s3://") or s3_uri.count("/") < 3:
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    b, k = s3_uri[5:].split("/", 1)
    return b, k