import time
import json
import math
import logging
from decimal import Decimal
from typing import Tuple, Any, Optional, List, Dict
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
    "job_id", "image_id", "temp_source_ref", "copy_to",
    "img_type", "img_height", "img_width", "num_channels", "dtype",
    "file_size_mb", "uploaded_at", "data_source", "sha256_hash",
    "temp_string_labels_path", "temp_bbox_path", "temp_semantic_mask_path",
    "temp_instance_annotation_path", "validation_status", "validation_error",
    "dedup_status"
]

CANONICAL_IMAGERY_COLS = [
    "image_id", "source_ref", "img_type",
    "img_height", "img_width", "num_channels", "dtype",
    "file_size_mb", "uploaded_at", "data_source", "sha256_hash",
    "string_labels", "bboxes", "semantic_masks",
    "instance_annotations"
]

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
        return "ARRAY[" + ", ".join("'" + _escape_sql_string(str(x)) + "'" for x in v) + "]"

    # default string
    return "'" + _escape_sql_string(str(v)) + "'"

def chunked_insert(rows,
                  iceberg_db_name,
                  table_name,
                  athena_workgroup,
                  athena_output_s3,
                  chunk_size=200):
    """Insert rows into Iceberg in batches. If a batch fails, retry row-by-row."""

    assert chunk_size > 0

    if table_name == 'upload_staging':
        columns = UPLOAD_STAGING_COLS
    elif table_name == 'canonical_imagery':
        columns = CANONICAL_IMAGERY_COLS
    else:
        raise Exception(f'Table name not recognized: {table_name}')

    table = f'"{iceberg_db_name}"."{table_name}"'
    all_failed = False
    fail_count = 0
    last_error = ""
    for i in range(0, len(rows), chunk_size):
        batch = rows[i:i+chunk_size]
        values_clause = []
        for r in batch:
            values = [to_sql_value(r, c) for c in columns]
            values_clause.append("(" + ", ".join(values) + ")")
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES " + ", ".join(values_clause)

        # first try to insert the batch, all or nothing is inserted in this athena call
        try:
            qid = athena.start_query_execution(
                QueryString=sql,
                ResultConfiguration={"OutputLocation": athena_output_s3},
                WorkGroup=athena_workgroup
            )["QueryExecutionId"]

            wait_res = wait_for_athena(qid)
            success = wait_res['state'] == 'SUCCEEDED'
            if not success:
                raise RuntimeError("Batch insert failed")

        except Exception as e:
            # Retry row-by-row for this batch if batch insert failed due to a bad row.
            for r in batch:
                try:
                    values = [to_sql_value(r, c) for c in columns]
                    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(values)})"
                    qid = athena.start_query_execution(
                        QueryString=sql,
                        ResultConfiguration={"OutputLocation": athena_output_s3},
                        WorkGroup=athena_workgroup
                    )["QueryExecutionId"]
                    wait_for_athena(qid)

                except Exception as last_error:
                    fail_count += 1

    if fail_count == len(rows):
        all_failed = True

    return all_failed, str(last_error)

def delete_iceberg_partition_rows(job_id: str,
                                    iceberg_db_name,
                                    image_upload_staging_table_name,
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
    full_table = f"{iceberg_db_name}.{image_upload_staging_table_name}"

    # 1) DELETE statement (Iceberg positional delete files)
    delete_sql = f"DELETE FROM {full_table} WHERE job_id = '{safe_job_id}'"
    delete_resp = athena.start_query_execution(
        QueryString=delete_sql,
        ResultConfiguration={"OutputLocation": athena_output_s3},
        WorkGroup=athena_workgroup
    )
    delete_qid = delete_resp["QueryExecutionId"]
    delete_result = wait_for_athena(delete_qid, poll=poll_interval, timeout=timeout_seconds)

    result = {
        "delete_query_id": delete_qid,
        "delete_state": delete_result["state"]
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

def delete_s3_prefix(bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=obj["Key"])

# Helpers to extract event keys we will need regardless of input type.
def find_key_recursively(obj: Any, target_key: str, max_depth: int = 6, max_nodes: int = 10000) -> Optional[Any]:
    """
    Search for target_key anywhere in a nested dict/list structure.
    Bounded by max_depth and max_nodes to avoid runaway traversal.
    Returns the first found value or None.
    """
    nodes_visited = 0

    def _recurse(o: Any, depth: int) -> Optional[Any]:
        nonlocal nodes_visited
        if nodes_visited >= max_nodes:
            return None
        nodes_visited += 1

        if depth < 0:
            return None
        if isinstance(o, dict):
            # check direct hit first for deterministic behavior
            if target_key in o:
                return o[target_key]
            for k, v in o.items():
                # skip trivial scalar values to reduce work
                if isinstance(v, (dict, list)):
                    res = _recurse(v, depth - 1)
                    if res is not None:
                        return res
        elif isinstance(o, list):
            for item in o:
                if isinstance(item, (dict, list)):
                    res = _recurse(item, depth - 1)
                    if res is not None:
                        return res
        return None

    return _recurse(obj, max_depth)

def _extract_from_container_env(event: Any) -> Dict[str, str]:
    """
    If event is an ECS/Batch job detail or similar with Container.Environment list,
    return a dict of env vars.
    """
    try:
        job_detail = event[0] if isinstance(event, list) else event
        env_list = job_detail.get("Container", {}).get("Environment", [])
        if not env_list:
            # Some shapes use job_detail['container']['environment'] (lowercase)
            env_list = job_detail.get("container", {}).get("environment", [])
        env_map = {e["Name"]: e["Value"] for e in env_list if "Name" in e and "Value" in e}
        return env_map
    except Exception:
        return {}

def _dig_for_key(event: Any, key: str) -> Any:
    """
    Try to find `key` in several likely places:
      - top-level: event[key]
      - nested under any stage keys (e.g., event['validationStage'][key])
      - nested under 'detail' (CloudWatch events)
    Returns None if not found.
    """
    if not isinstance(event, dict):
        return None

    # 1) top-level
    if key in event:
        return event[key]

    # 2) under top-level stage keys (common pattern: event['validationStage']['job_id'])
    for k, v in event.items():
        if isinstance(v, dict) and key in v:
            return v[key]

    # 3) under 'detail' (CloudWatch/Batch event)
    if "detail" in event and isinstance(event["detail"], dict) and key in event["detail"]:
        return event["detail"][key]

    return None

def _normalize_label_types(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            # not JSON, maybe comma-separated
            return [s.strip() for s in raw.split(",") if s.strip()]
    return [str(raw)]

def get_job_input(event: Any) -> Dict[str, Any]:
    """
    Return a normalized dict with keys:
      job_id, user, label_types (list), data_source, event_type
    """
    # 1) Try container env extraction (ECS/Batch job detail)
    env_map = _extract_from_container_env(event)
    if env_map:
        job_id = env_map.get("JOB_ID") or env_map.get("job_id")
        user = env_map.get("USER") or env_map.get("user")
        label_types_raw = env_map.get("LABEL_TYPES") or env_map.get("label_types")
        data_source = env_map.get("DATA_SOURCE") or env_map.get("data_source")
        event_type = env_map.get("EVENT_TYPE") or env_map.get("event_type")
        return {
            "job_id": job_id,
            "user": user,
            "label_types": _normalize_label_types(label_types_raw),
            "data_source": data_source,
            "event_type": event_type
        }

    # 2) direct keys and stage-nested lookups (fast)
    job_id = _dig_for_key(event, "job_id")
    user = _dig_for_key(event, "user")
    label_types_raw = _dig_for_key(event, "label_types") or _dig_for_key(event, "labelTypes")
    data_source = _dig_for_key(event, "data_source")
    event_type = _dig_for_key(event, "event_type") or _dig_for_key(event, "eventType")

    # 3) recursive fallback (bounded)
    if job_id is None:
        job_id = find_key_recursively(event, "job_id")
    if user is None:
        user = find_key_recursively(event, "user")
    if label_types_raw is None:
        label_types_raw = find_key_recursively(event, "label_types") or find_key_recursively(event, "labelTypes")
    if data_source is None:
        data_source = find_key_recursively(event, "data_source")
    if event_type is None:
        event_type = find_key_recursively(event, "event_type") or find_key_recursively(event, "eventType")

    # log when recursive fallback was used for observability
    # (only log at debug/info level to avoid noise)
    logger.debug("get_job_input: used recursive fallback for job_id=%s user=%s", job_id, user)

    # normalize and return defaults
    return {
        "job_id": job_id or "unknown",
        "user": user or "unknown",
        "label_types": _normalize_label_types(label_types_raw),
        "data_source": data_source or "unknown",
        "event_type": event_type or "unknown"
    }