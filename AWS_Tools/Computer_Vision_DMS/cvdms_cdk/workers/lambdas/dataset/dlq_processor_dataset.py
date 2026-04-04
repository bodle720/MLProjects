import os
import json
from typing import Dict, List, Iterable, Tuple, Optional

import boto3

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import (
    delete_s3_prefix,
    parse_s3_uri
)
from common.general_utils.ddb_utils import update_job_status, release_lock
from common.general_utils.iceberg_utils import escape_sql_string
from common.general_utils.athena_utils import run_athena, athena_fetch_all_rows
from common.general_utils.table_schemas import (
    TABLES,
    CANONICAL_IMAGERY_TABLE_NAME,
    IMAGE_LABELS_TABLE_NAME,
    CANONICAL_BBOX_TABLE_NAME,
    CANONICAL_SEMANTIC_TABLE_NAME,
    CANONICAL_INSTANCE_TABLE_NAME
)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]

TASK_NAME = "[DLQ_PROCESSOR_DATASET]"

s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

LABEL_TABLES = {
    CANONICAL_BBOX_TABLE_NAME,
    CANONICAL_SEMANTIC_TABLE_NAME,
    CANONICAL_INSTANCE_TABLE_NAME,
}

def chunked(lst: List, n: int) -> Iterable[List]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def delete_s3_keys_best_effort(bucket: str, keys: List[str], batch_size: int = 1000) -> Tuple[int, int]:
    if not keys:
        return 0, 0
    deleted = 0
    errors_total = 0
    for chunk in chunked(keys, batch_size):
        try:
            resp = s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in chunk]},
            )
            deleted += len(chunk)
            errors = resp.get("Errors", []) or []
            errors_total += len(errors)
        except Exception:
            errors_total += len(chunk)
    return deleted, errors_total

def _safe_s3_key_from_uri(uri: str) -> Optional[str]:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        return None
    try:
        b, k = parse_s3_uri(uri, TASK_NAME)
        if b != FILE_BUCKET_NAME:
            return None
        return k
    except Exception:
        return None

def _delete_canonical_imagery_rows(image_ids: List[str]) -> None:
    if not image_ids:
        return
    for chunk in chunked(image_ids, 500):
        in_list = ", ".join(f"'{escape_sql_string(i)}'" for i in chunk)
        sql = f'DELETE FROM "{ICEBERG_DATABASE_NAME}"."{CANONICAL_IMAGERY_TABLE_NAME}" WHERE image_id IN ({in_list})'
        run_athena(sql, TASK_NAME, ATHENA_OUTPUT_S3, ATHENA_WORKGROUP, poll=2.0, timeout=1800)

def _delete_image_labels_for_images(image_ids: List[str]) -> None:
    """
    Safe for *new images* (images created by this job).
    """
    if not image_ids:
        return
    for chunk in chunked(image_ids, 500):
        in_list = ", ".join(f"'{escape_sql_string(i)}'" for i in chunk)
        sql = f'DELETE FROM "{ICEBERG_DATABASE_NAME}"."{IMAGE_LABELS_TABLE_NAME}" WHERE image_id IN ({in_list})'
        run_athena(sql, TASK_NAME, ATHENA_OUTPUT_S3, ATHENA_WORKGROUP, poll=2.0, timeout=1800)

def _extract_label_s3_keys(table: str, rows: List[Dict]) -> List[str]:
    """
    From canonical label table row fields, extract canonical label S3 keys to delete (file bucket only).
    """
    keys: List[str] = []
    seen = set()
    for r in rows:
        # bbox table: source_ref_meta
        # semantic/instance: source_ref_png + source_ref_meta
        for col in ("source_ref_png", "source_ref_meta"):
            v = r.get(col)
            k = _safe_s3_key_from_uri(v) if isinstance(v, str) else None
            if k and k not in seen:
                seen.add(k)
                keys.append(k)
    return keys

def _label_key_col(table: str) -> str:
    schema = TABLES.get(table)
    if not schema or not schema.key_cols:
        raise RuntimeError(f"{TASK_NAME} missing schema/key_cols for table={table}")
    if len(schema.key_cols) != 1:
        raise RuntimeError(f"{TASK_NAME} expected single key col for table={table}, got {schema.key_cols}")
    return schema.key_cols[0]

def _label_ids_from_rows(table: str, rows: List[Dict]) -> List[str]:
    kcol = _label_key_col(table)
    out: List[str] = []
    seen = set()
    for r in rows:
        v = r.get(kcol)
        if isinstance(v, str) and v.strip() and v.strip() not in seen:
            seen.add(v.strip())
            out.append(v.strip())
    return out

def _is_label_id_referenced(label_type: str, label_id: str) -> bool:
    """
    Checks whether ANY image_labels row references (label_type,label_id).
    If referenced, do NOT delete canonical label row/object.
    """
    lt = escape_sql_string(label_type)
    lid = escape_sql_string(label_id)
    sql = f"""
    SELECT COUNT(*) AS c
    FROM "{ICEBERG_DATABASE_NAME}"."{IMAGE_LABELS_TABLE_NAME}"
    WHERE label_type = '{lt}'
      AND label_id = '{lid}'
    """
    qid, _ = run_athena(sql, TASK_NAME, ATHENA_OUTPUT_S3, ATHENA_WORKGROUP, poll=2.0, timeout=300)
    rows = athena_fetch_all_rows(qid)
    # athena_fetch_all_rows in the codebase returns list[dict] with keys matching select aliases
    # Be defensive:
    for r in rows:
        v = r.get("c") or r.get("_col0") or r.get("count") or r.get("count(*)")
        try:
            if int(v) > 0:
                return True
        except Exception:
            continue
    return False

def _delete_canonical_label_rows_if_orphaned(table: str, label_type: str, rows: List[Dict]) -> Tuple[List[str], int]:
    """
    Deletes label table rows only if orphaned (no image_labels refs).
    Returns (deleted_label_ids, skipped_count).
    """
    if not rows:
        return [], 0

    kcol = _label_key_col(table)
    ids = _label_ids_from_rows(table, rows)
    if not ids:
        return [], 0

    to_delete: List[str] = []
    skipped = 0

    for lid in ids:
        try:
            if _is_label_id_referenced(label_type, lid):
                skipped += 1
            else:
                to_delete.append(lid)
        except Exception:
            skipped += 1

    for chunk in chunked(to_delete, 500):
        in_list = ", ".join(f"'{escape_sql_string(i)}'" for i in chunk)
        sql = f'DELETE FROM "{ICEBERG_DATABASE_NAME}"."{table}" WHERE {kcol} IN ({in_list})'
        run_athena(sql, TASK_NAME, ATHENA_OUTPUT_S3, ATHENA_WORKGROUP, poll=2.0, timeout=1800)

    return to_delete, skipped

def handler(event, context):
    total_records = 0
    num_processed_successfully = 0

    for record in event.get("Records", []):
        total_records += 1
        update_success = False
        release_success = False

        try:
            body = json.loads(record["body"])
        except Exception:
            print(f"{TASK_NAME} Skipping non-JSON message")
            continue

        source = body.get("source")
        job_id = body.get("job_id")
        user = body.get("user")
        event_type = body.get("event_type")
        error_msg = body.get("error")

        try:
            error_obj = json.loads(error_msg)
            cause = error_obj.get("Cause")
            if cause:
                cause_obj = json.loads(cause)
                error_msg = cause_obj.get("errorMessage", error_msg)
        except Exception:
            pass

        if source not in ("stepfunctions", "kickoff", "lambda"):
            print(f"{TASK_NAME} Skipping unknown source={source}")
            continue

        if (job_id in (None, "unknown")) or (user is None) or (event_type is None):
            print(f"{TASK_NAME} Ignoring non-job DLQ message: {body}")
            continue

        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} DLQ received message: {body}")

        # 1) Delete temp folder for this job
        prefix = f"temp/dataset-ops/{job_id}/"
        try:
            delete_s3_prefix(FILE_BUCKET_NAME, prefix, TASK_NAME)
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Deleted temp s3 prefix")
        except Exception:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Temp S3 cleanup failed", level="error")

        # 2) Mark job FAILED
        try:
            update_success, update_msg = update_job_status(
                job_id,
                "FAILED",
                JOB_TABLE_NAME,
                LOG_FIREHOSE_STREAM_NAME,
                user=user,
                event_type=event_type,
                error_msg=(error_msg or "")[:512]
            )
            if update_success:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Updated job status to FAILED.")
            else:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Failed to set job FAILED: {update_msg}", level="error")
        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Updating job status FAILED failed: {e}", level="error")

        # 3) Release global lock
        try:
            release_success, release_msg = release_lock(job_id, LOCK_TABLE_NAME, LOG_FIREHOSE_STREAM_NAME, user=user, event_type=event_type)
            if release_success:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Released lock.")
            else:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Release lock failed: {release_msg}", level="error")
        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Release lock failed: {e}", level="error")

        if update_success and release_success:
            num_processed_successfully += 1

    return {
        "status": "ok",
        "total_records": total_records,
        "successfully_processed_records": num_processed_successfully,
    }