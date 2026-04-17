import os
import json
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Iterable, Tuple, Optional, Set

import boto3
from botocore.exceptions import ClientError

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import (
    parse_s3_uri,
    s3_list_keys,
    s3_read_jsonl_list,
)
from common.general_utils.ddb_utils import update_job_status, release_lock
from common.general_utils.iceberg_utils import escape_sql_string
from common.general_utils.athena_utils import run_athena, athena_fetch_all_rows
from common.general_utils.table_schemas import (
    TABLES,
    UPLOAD_STAGING_TABLE_NAME,
    CANONICAL_IMAGERY_TABLE_NAME,
    IMAGE_LABELS_TABLE_NAME,
    IMAGE_SOURCE_MEMBERSHIP_TABLE_NAME,
    CANONICAL_BBOX_TABLE_NAME,
    CANONICAL_SEMANTIC_TABLE_NAME,
    CANONICAL_INSTANCE_TABLE_NAME,
)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
SHA256_TABLE_NAME = os.environ["SHA256_TABLE_NAME"]

TASK_NAME = "[UPLOAD_DLQ_PROCESSOR]"

# Quiescence behavior
QUIESCENCE_POLL_SEC = 10
QUIESCENCE_MAX_WAIT_SEC = 60
QUIESCENCE_REQUIRED_EMPTY_POLLS = 2
ACTIVE_MARKER_STALE_SEC = 20 * 60

# Narrow Iceberg commit retry for rollback DML
COMMIT_RETRY_ATTEMPTS = 4
COMMIT_RETRY_BASE_SLEEP_SEC = 2.0
COMMIT_RETRY_JITTER_SEC = 0.5

s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

LABEL_TABLES = {
    CANONICAL_BBOX_TABLE_NAME,
    CANONICAL_SEMANTIC_TABLE_NAME,
    CANONICAL_INSTANCE_TABLE_NAME,
}

LABEL_TYPE_BY_TABLE = {
    CANONICAL_BBOX_TABLE_NAME: "object-detection",
    CANONICAL_SEMANTIC_TABLE_NAME: "semantic-segmentation",
    CANONICAL_INSTANCE_TABLE_NAME: "instance-segmentation",
}

def chunked(lst: List, n: int) -> Iterable[List]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _is_retryable_iceberg_commit_error(exc: Exception | str) -> bool:
    text = str(exc)
    return (
        "ICEBERG_COMMIT_ERROR" in text
        or "Failed to commit Iceberg update" in text
    )

def _sleep_with_backoff(attempt_index: int) -> None:
    delay = COMMIT_RETRY_BASE_SLEEP_SEC * (2 ** attempt_index)
    delay += random.uniform(0.0, COMMIT_RETRY_JITTER_SEC)
    time.sleep(delay)

def _run_athena_with_commit_retry(
    sql: str,
    *,
    op_name: str,
    poll: float = 2.0,
    timeout: float = 1800,
) -> Tuple[str, Dict]:
    last_exc: Optional[Exception] = None

    for attempt in range(COMMIT_RETRY_ATTEMPTS):
        try:
            return run_athena(
                sql,
                op_name,
                ATHENA_OUTPUT_S3,
                ATHENA_WORKGROUP,
                poll=poll,
                timeout=timeout,
            )
        except Exception as e:
            last_exc = e
            retryable = _is_retryable_iceberg_commit_error(e)
            is_last = attempt >= (COMMIT_RETRY_ATTEMPTS - 1)

            if retryable and not is_last:
                _sleep_with_backoff(attempt)
                continue

            raise

    if last_exc:
        raise last_exc

    raise RuntimeError(f"{TASK_NAME} unexpected rollback Athena failure in {op_name}")

def _safe_s3_key_from_uri(uri: str) -> Optional[str]:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        return None
    try:
        bucket, key = parse_s3_uri(uri, TASK_NAME)
        if bucket != FILE_BUCKET_NAME:
            return None
        return key
    except Exception:
        return None

def _read_json_key(key: str) -> Dict:
    resp = s3.get_object(Bucket=FILE_BUCKET_NAME, Key=key)
    body = resp["Body"].read().decode("utf-8")
    return json.loads(body)

def _list_registration_processed_keys(job_id: str) -> List[str]:
    prefix = f"temp/image-upload/{job_id}/batches/registration-step/processed/"
    try:
        return s3_list_keys(FILE_BUCKET_NAME, prefix)
    except Exception:
        return []

def _list_active_worker_markers(job_id: str) -> List[Tuple[str, datetime]]:
    prefix = f"temp/image-upload/{job_id}/worker-markers/"
    paginator = s3.get_paginator("list_objects_v2")

    out: List[Tuple[str, datetime]] = []
    for page in paginator.paginate(Bucket=FILE_BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "/active/" not in key or key.endswith("/"):
                continue
            last_modified = obj.get("LastModified")
            if isinstance(last_modified, datetime):
                out.append((key, last_modified))
    return out

def _wait_for_worker_quiescence_or_raise(job_id: str, user: str, event_type: str) -> None:
    start = time.time()
    empty_polls = 0

    while True:
        markers = _list_active_worker_markers(job_id)
        fresh: List[str] = []
        stale: List[str] = []

        now = _utc_now()
        for key, last_modified in markers:
            age_sec = (now - last_modified).total_seconds()
            if age_sec > ACTIVE_MARKER_STALE_SEC:
                stale.append(key)
            else:
                fresh.append(key)

        if fresh:
            empty_polls = 0
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Waiting for worker quiescence; fresh active markers={len(fresh)} sample={fresh[:5]}",
                level="warning",
            )
        else:
            empty_polls += 1
            if stale:
                log(
                    job_id,
                    user,
                    event_type,
                    LOG_FIREHOSE_STREAM_NAME,
                    f"{TASK_NAME} Proceeding with stale active markers only; stale_count={len(stale)} sample={stale[:5]}",
                    level="warning",
                )
            if empty_polls >= QUIESCENCE_REQUIRED_EMPTY_POLLS:
                return

        if (time.time() - start) >= QUIESCENCE_MAX_WAIT_SEC:
            if fresh:
                sample = fresh[:5]
                raise RuntimeError(
                    f"{TASK_NAME} rollback blocked by fresh active worker markers; count={len(fresh)} sample={sample}"
                )
            return

        time.sleep(QUIESCENCE_POLL_SEC)

def _athena_fetch_rows_1col(qid: str) -> List[str]:
    out: List[str] = []
    rows = athena_fetch_all_rows(qid)
    for r in rows:
        if not isinstance(r, dict):
            continue
        for value in r.values():
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
                break
    return out

def _athena_fetch_rows_2col(qid: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    rows = athena_fetch_all_rows(qid)
    for r in rows:
        if not isinstance(r, dict):
            continue
        vals = list(r.values())
        if len(vals) < 2:
            continue
        a = vals[0].strip() if isinstance(vals[0], str) else ""
        b = vals[1].strip() if isinstance(vals[1], str) else ""
        if a and b:
            out.append((a, b))
    return out

def _query_new_image_ids_from_upload_staging(job_id: str) -> List[str]:
    safe_job_id = escape_sql_string(job_id)
    sql = f"""
    SELECT image_id
    FROM "{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE_NAME}"
    WHERE job_id = '{safe_job_id}'
      AND dedup_status = 'passed'
    """
    qid, _ = run_athena(
        sql,
        f"{TASK_NAME} query_new_image_ids",
        ATHENA_OUTPUT_S3,
        ATHENA_WORKGROUP,
        poll=2.0,
        timeout=300,
    )
    vals = _athena_fetch_rows_1col(qid)
    return sorted(set(vals))

def _query_new_sha_mappings_from_upload_staging(job_id: str) -> List[Tuple[str, str]]:
    safe_job_id = escape_sql_string(job_id)
    sql = f"""
    SELECT sha256_hash, image_id
    FROM "{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE_NAME}"
    WHERE job_id = '{safe_job_id}'
      AND dedup_status = 'passed'
      AND sha256_hash IS NOT NULL
      AND sha256_hash <> ''
      AND image_id IS NOT NULL
      AND image_id <> ''
    """
    qid, _ = run_athena(
        sql,
        f"{TASK_NAME} query_new_sha_mappings",
        ATHENA_OUTPUT_S3,
        ATHENA_WORKGROUP,
        poll=2.0,
        timeout=300,
    )
    rows = _athena_fetch_rows_2col(qid)
    return sorted(set(rows))

def _query_canonical_image_keys_for_image_ids(image_ids: List[str]) -> List[str]:
    """
    Third authoritative source of canonical image object keys:
    query live canonical_imagery rows for the new image_ids before deleting those rows.
    """
    if not image_ids:
        return []

    out: Set[str] = set()

    for chunk in chunked(sorted(set(image_ids)), 500):
        in_list = ", ".join(f"'{escape_sql_string(i)}'" for i in chunk)
        sql = f"""
        SELECT source_ref
        FROM "{ICEBERG_DATABASE_NAME}"."{CANONICAL_IMAGERY_TABLE_NAME}"
        WHERE image_id IN ({in_list})
          AND source_ref IS NOT NULL
          AND source_ref <> ''
        """
        qid, _ = run_athena(
            sql,
            f"{TASK_NAME} query_canonical_image_keys_for_image_ids",
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
            poll=2.0,
            timeout=300,
        )
        uris = _athena_fetch_rows_1col(qid)
        for uri in uris:
            key = _safe_s3_key_from_uri(uri)
            if key:
                out.add(key)

    return sorted(out)

def delete_sha256_entries_for_job(mappings: List[Tuple[str, str]]) -> Tuple[int, int, int]:
    """
    Delete sha256 entries only if they still point at the image_id created by THIS job.
    Returns (deleted, skipped_not_matching, errors).
    """
    deleted = 0
    skipped = 0
    errors = 0

    for sha, iid in mappings:
        try:
            dynamodb.delete_item(
                TableName=SHA256_TABLE_NAME,
                Key={"sha256": {"S": sha}},
                ConditionExpression="image_id = :iid",
                ExpressionAttributeValues={":iid": {"S": iid}},
            )
            deleted += 1
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                skipped += 1
                continue
            errors += 1

    return deleted, skipped, errors

def _load_batch_rollback_seeds(
    processed_keys: List[str],
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Reads rollback seed JSONs written by the registration batch worker.
    Returns:
      - new_image_ids
      - canonical_image_keys_to_delete
      - sha256 mappings to delete: (sha256, image_id)

    We intentionally do NOT blindly delete canonical label object keys from these
    seeds because those fingerprint-addressed objects may be shared.
    """
    rollback_seed_keys = [k for k in processed_keys if "/rollback-batch/" in k and k.endswith(".json")]
    rollback_seed_keys.sort()

    new_image_ids: Set[str] = set()
    canonical_image_keys: Set[str] = set()
    sha_mappings: Set[Tuple[str, str]] = set()

    for key in rollback_seed_keys:
        try:
            payload = _read_json_key(key)
        except Exception:
            continue

        for image_id in payload.get("new_image_ids", []) or []:
            if isinstance(image_id, str) and image_id.strip():
                new_image_ids.add(image_id.strip())

        for image_key in payload.get("canonical_image_keys_to_delete", []) or []:
            if isinstance(image_key, str) and image_key.strip():
                canonical_image_keys.add(image_key.strip())

        for row in payload.get("sha256_mappings_to_delete", []) or []:
            sha = row.get("sha256")
            iid = row.get("image_id")
            if isinstance(sha, str) and sha.strip() and isinstance(iid, str) and iid.strip():
                sha_mappings.add((sha.strip(), iid.strip()))

    return sorted(new_image_ids), sorted(canonical_image_keys), sorted(sha_mappings)

def _load_canonical_image_object_keys_from_processed_rows(processed_keys: List[str]) -> List[str]:
    """
    Secondary source of canonical image object keys:
    read processed canonical_imagery shard outputs and extract source_ref URIs.
    """
    imagery_jsonls = [k for k in processed_keys if "/canonical_imagery/" in k and k.endswith(".jsonl")]
    imagery_jsonls.sort()

    keys: Set[str] = set()
    if not imagery_jsonls:
        return []

    for row in s3_read_jsonl_list(FILE_BUCKET_NAME, imagery_jsonls, f"{TASK_NAME}.read_canonical_imagery"):
        uri = row.get("source_ref")
        key = _safe_s3_key_from_uri(uri) if isinstance(uri, str) else None
        if key:
            keys.add(key)

    return sorted(keys)

def _load_exact_rollback_targets(processed_keys: List[str]) -> Tuple[Set[Tuple[str, str, str]], List[Tuple[str, str]]]:
    """
    Reads rollback plan JSONs written by registration ingest map lambda and returns:
      - exact image_labels keys to delete: (image_id, label_type, label_id)
      - exact image_source_membership keys to delete: (image_id, data_source)
    """
    rollback_keys = [k for k in processed_keys if "/rollback/" in k and k.endswith(".json")]
    rollback_keys.sort()

    image_label_keys: Set[Tuple[str, str, str]] = set()
    image_source_membership_keys: Set[Tuple[str, str]] = set()

    for key in rollback_keys:
        try:
            payload = _read_json_key(key)
        except Exception:
            continue

        for row in payload.get("image_labels_to_delete", []) or []:
            iid = row.get("image_id")
            lt = row.get("label_type")
            lid = row.get("label_id")
            if (
                isinstance(iid, str) and iid.strip()
                and isinstance(lt, str) and lt.strip()
                and isinstance(lid, str) and lid.strip()
            ):
                image_label_keys.add((iid.strip(), lt.strip(), lid.strip()))

        for row in payload.get("image_source_memberships_to_delete", []) or []:
            iid = row.get("image_id")
            ds = row.get("data_source")
            if isinstance(iid, str) and iid.strip() and isinstance(ds, str) and ds.strip():
                image_source_membership_keys.add((iid.strip(), ds.strip()))

    return image_label_keys, sorted(image_source_membership_keys)

def _delete_canonical_imagery_rows(image_ids: List[str]) -> None:
    if not image_ids:
        return
    for chunk in chunked(image_ids, 500):
        in_list = ", ".join(f"'{escape_sql_string(i)}'" for i in chunk)
        sql = f'DELETE FROM "{ICEBERG_DATABASE_NAME}"."{CANONICAL_IMAGERY_TABLE_NAME}" WHERE image_id IN ({in_list})'
        _run_athena_with_commit_retry(sql, op_name=f"{TASK_NAME} delete_canonical_imagery")

def _delete_image_labels_for_images(image_ids: List[str]) -> None:
    if not image_ids:
        return
    for chunk in chunked(image_ids, 500):
        in_list = ", ".join(f"'{escape_sql_string(i)}'" for i in chunk)
        sql = f'DELETE FROM "{ICEBERG_DATABASE_NAME}"."{IMAGE_LABELS_TABLE_NAME}" WHERE image_id IN ({in_list})'
        _run_athena_with_commit_retry(sql, op_name=f"{TASK_NAME} delete_image_labels_for_images")

def _delete_image_source_memberships_for_images(image_ids: List[str]) -> None:
    if not image_ids:
        return
    for chunk in chunked(image_ids, 500):
        in_list = ", ".join(f"'{escape_sql_string(i)}'" for i in chunk)
        sql = f'DELETE FROM "{ICEBERG_DATABASE_NAME}"."{IMAGE_SOURCE_MEMBERSHIP_TABLE_NAME}" WHERE image_id IN ({in_list})'
        _run_athena_with_commit_retry(sql, op_name=f"{TASK_NAME} delete_image_source_memberships_for_images")

def _delete_exact_image_labels(keys: Set[Tuple[str, str, str]]) -> None:
    if not keys:
        return

    rows = sorted(keys)
    for chunk in chunked(rows, 200):
        clauses = []
        for image_id, label_type, label_id in chunk:
            clauses.append(
                f"(image_id = '{escape_sql_string(image_id)}' "
                f"AND label_type = '{escape_sql_string(label_type)}' "
                f"AND label_id = '{escape_sql_string(label_id)}')"
            )
        where_sql = " OR ".join(clauses)
        sql = f'DELETE FROM "{ICEBERG_DATABASE_NAME}"."{IMAGE_LABELS_TABLE_NAME}" WHERE {where_sql}'
        _run_athena_with_commit_retry(sql, op_name=f"{TASK_NAME} delete_exact_image_labels")

def _delete_exact_image_source_memberships(keys: List[Tuple[str, str]]) -> None:
    if not keys:
        return

    for chunk in chunked(keys, 200):
        clauses = []
        for image_id, data_source in chunk:
            clauses.append(
                f"(image_id = '{escape_sql_string(image_id)}' "
                f"AND data_source = '{escape_sql_string(data_source)}')"
            )
        where_sql = " OR ".join(clauses)
        sql = f'DELETE FROM "{ICEBERG_DATABASE_NAME}"."{IMAGE_SOURCE_MEMBERSHIP_TABLE_NAME}" WHERE {where_sql}'
        _run_athena_with_commit_retry(sql, op_name=f"{TASK_NAME} delete_exact_image_source_memberships")

def _load_owner_label_rows(processed_keys: List[str]) -> Dict[str, List[Dict]]:
    owner_jsonls = [k for k in processed_keys if "/canonical_labels_by_fingerprint/" in k and k.endswith(".jsonl")]
    owner_jsonls.sort()
    out: Dict[str, List[Dict]] = {t: [] for t in LABEL_TABLES}

    if not owner_jsonls:
        return out

    for row in s3_read_jsonl_list(FILE_BUCKET_NAME, owner_jsonls, f"{TASK_NAME}.read_owner_labels"):
        table = row.get("__table")
        if not isinstance(table, str) or table.strip() not in LABEL_TABLES:
            continue
        r2 = dict(row)
        r2.pop("__table", None)
        out[table.strip()].append(r2)

    return out

def _extract_label_s3_keys(rows: List[Dict]) -> List[str]:
    keys: List[str] = []
    seen = set()
    for r in rows:
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
    for r in rows:
        v = r.get("c") or r.get("_col0") or r.get("count") or r.get("count(*)")
        try:
            if int(v) > 0:
                return True
        except Exception:
            continue
    return False

def _delete_canonical_label_rows_if_orphaned(table: str, label_type: str, rows: List[Dict]) -> Tuple[List[str], int]:
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
        _run_athena_with_commit_retry(
            sql,
            op_name=f"{TASK_NAME} delete_canonical_label_rows_if_orphaned:{table}",
        )

    return to_delete, skipped

def _drop_ctas_tables_best_effort(job_id: str) -> None:
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    candidates = [
        f"reg_export_{sanitized_job_id}",
        f"dedup_export_{sanitized_job_id}",
    ]
    for tname in candidates:
        try:
            sql = f'DROP TABLE IF EXISTS "{ICEBERG_DATABASE_NAME}"."{tname}"'
            run_athena(sql, TASK_NAME, ATHENA_OUTPUT_S3, ATHENA_WORKGROUP, poll=2.0, timeout=300)
        except Exception:
            pass

def _normalize_error_message(error_msg: object) -> str:
    if not isinstance(error_msg, str):
        return str(error_msg)

    msg = error_msg
    try:
        error_obj = json.loads(msg)
        cause = error_obj.get("Cause")
        if cause:
            cause_obj = json.loads(cause)
            msg = cause_obj.get("errorMessage", msg)
    except Exception:
        pass
    return msg

def _s3_object_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def delete_s3_keys_strict(bucket: str, keys: List[str], batch_size: int = 1000) -> Dict[str, object]:
    """
    Strict S3 cleanup:
      1) issue delete_objects in batches
      2) HEAD every key afterward
      3) report survivors as critical rollback failures
    """
    unique_keys = sorted({k.strip() for k in keys if isinstance(k, str) and k.strip()})
    if not unique_keys:
        return {
            "attempted": 0,
            "api_error_count": 0,
            "api_error_samples": [],
            "survivor_count": 0,
            "survivor_samples": [],
            "verify_error_count": 0,
            "verify_error_samples": [],
        }

    api_error_samples: List[str] = []

    for chunk in chunked(unique_keys, batch_size):
        try:
            resp = s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in chunk]},
            )
            for err in resp.get("Errors", []) or []:
                key = str(err.get("Key", ""))
                code = str(err.get("Code", ""))
                msg = str(err.get("Message", ""))
                api_error_samples.append(f"{key}:{code}:{msg}")
        except Exception as e:
            sample = chunk[:10]
            api_error_samples.extend([f"{k}:delete_exception:{e}" for k in sample])

    survivors: List[str] = []
    verify_error_samples: List[str] = []

    for key in unique_keys:
        try:
            if _s3_object_exists(bucket, key):
                survivors.append(key)
        except Exception as e:
            verify_error_samples.append(f"{key}:verify_exception:{e}")

    return {
        "attempted": len(unique_keys),
        "api_error_count": len(api_error_samples),
        "api_error_samples": api_error_samples[:10],
        "survivor_count": len(survivors),
        "survivor_samples": survivors[:10],
        "verify_error_count": len(verify_error_samples),
        "verify_error_samples": verify_error_samples[:10],
    }

def _process_one_record(record: Dict[str, object]) -> None:
    try:
        body = json.loads(record["body"])
    except Exception:
        raise RuntimeError(f"{TASK_NAME} non-JSON DLQ message")

    source = body.get("source")
    job_id = body.get("job_id")
    user = body.get("user")
    event_type = body.get("event_type")
    error_msg = _normalize_error_message(body.get("error"))

    if source not in ("stepfunctions", "kickoff", "lambda"):
        raise RuntimeError(f"{TASK_NAME} unknown source={source!r}")

    if (job_id in (None, "unknown")) or (user is None) or (event_type is None):
        raise RuntimeError(f"{TASK_NAME} invalid job-shaped DLQ message: {body}")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} DLQ received message: {body}")

    # 0) Wait for in-flight workers to quiesce before rollback.
    _wait_for_worker_quiescence_or_raise(job_id, user, event_type)

    # 1) Gather rollback sources AFTER quiescence.
    processed_keys = _list_registration_processed_keys(job_id)

    batch_new_image_ids, batch_canonical_image_keys, batch_sha_mappings = _load_batch_rollback_seeds(processed_keys)
    processed_canonical_image_keys = _load_canonical_image_object_keys_from_processed_rows(processed_keys)
    rollback_label_keys, rollback_source_membership_keys = _load_exact_rollback_targets(processed_keys)

    # Use rollback-batch new_image_ids as the primary authoritative source.
    # Also union with upload_staging query as a best-effort supplement.
    try:
        queried_new_image_ids = _query_new_image_ids_from_upload_staging(job_id)
    except Exception as e:
        queried_new_image_ids = []
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} query_new_image_ids_from_upload_staging failed; continuing with seed-only source: {e}",
            level="warning",
        )

    try:
        queried_sha_mappings = _query_new_sha_mappings_from_upload_staging(job_id)
    except Exception as e:
        queried_sha_mappings = []
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} query_new_sha_mappings_from_upload_staging failed; continuing with seed-only source: {e}",
            level="warning",
        )

    new_image_ids = sorted(set(batch_new_image_ids) | set(queried_new_image_ids))
    batch_sha_mappings = sorted(set(batch_sha_mappings) | set(queried_sha_mappings))

    # Third authoritative source: live canonical_imagery rows, queried BEFORE deleting those rows.
    try:
        table_canonical_image_keys = _query_canonical_image_keys_for_image_ids(new_image_ids)
    except Exception as e:
        table_canonical_image_keys = []
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} query_canonical_image_keys_for_image_ids failed; continuing with other key sources: {e}",
            level="warning",
        )

    canonical_image_keys_to_delete = sorted(
        set(batch_canonical_image_keys)
        | set(processed_canonical_image_keys)
        | set(table_canonical_image_keys)
    )

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Canonical image cleanup sources: "
            f"batch_new_image_ids={len(batch_new_image_ids)} "
            f"queried_new_image_ids={len(queried_new_image_ids)} "
            f"rollback_seed_keys={len(batch_canonical_image_keys)} "
            f"processed_row_keys={len(processed_canonical_image_keys)} "
            f"table_row_keys={len(table_canonical_image_keys)} "
            f"union_keys={len(canonical_image_keys_to_delete)} "
            f"sample_union_keys={canonical_image_keys_to_delete[:5]}"
        ),
    )

    critical_failures: List[str] = []

    # 2) Registration-side rollback
    if rollback_label_keys:
        try:
            _delete_exact_image_labels(rollback_label_keys)
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Deleted exact image_labels rollback keys count={len(rollback_label_keys)}",
            )
        except Exception as e:
            critical_failures.append(f"exact image_labels rollback failed: {e}")

    if rollback_source_membership_keys:
        try:
            _delete_exact_image_source_memberships(rollback_source_membership_keys)
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Deleted exact image_source_membership rollback keys count={len(rollback_source_membership_keys)}",
            )
        except Exception as e:
            critical_failures.append(f"exact image_source_membership rollback failed: {e}")

    # Delete row-owned linkage for new images first.
    if new_image_ids:
        try:
            _delete_image_labels_for_images(new_image_ids)
            _delete_image_source_memberships_for_images(new_image_ids)
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Deleted image_labels + image_source_membership for {len(new_image_ids)} new image(s)",
            )
        except Exception as e:
            critical_failures.append(f"new-image linkage rollback failed: {e}")

    # Strict canonical image object cleanup BEFORE deleting canonical_imagery rows,
    # so retries can still query live table rows if this step fails.
    canonical_image_objects_ok = True
    if canonical_image_keys_to_delete:
        try:
            s3_delete_result = delete_s3_keys_strict(FILE_BUCKET_NAME, canonical_image_keys_to_delete)
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                (
                    f"{TASK_NAME} Canonical image object cleanup result: "
                    f"attempted={s3_delete_result['attempted']} "
                    f"api_error_count={s3_delete_result['api_error_count']} "
                    f"survivor_count={s3_delete_result['survivor_count']} "
                    f"verify_error_count={s3_delete_result['verify_error_count']} "
                    f"api_error_samples={s3_delete_result['api_error_samples']} "
                    f"survivor_samples={s3_delete_result['survivor_samples']} "
                    f"verify_error_samples={s3_delete_result['verify_error_samples']}"
                ),
                level=(
                    "warning"
                    if (
                        s3_delete_result["api_error_count"]
                        or s3_delete_result["survivor_count"]
                        or s3_delete_result["verify_error_count"]
                    )
                    else "info"
                ),
            )

            if (
                s3_delete_result["api_error_count"]
                or s3_delete_result["survivor_count"]
                or s3_delete_result["verify_error_count"]
            ):
                canonical_image_objects_ok = False
                critical_failures.append(
                    "canonical image object cleanup failed: "
                    f"attempted={s3_delete_result['attempted']} "
                    f"api_error_count={s3_delete_result['api_error_count']} "
                    f"survivor_count={s3_delete_result['survivor_count']} "
                    f"verify_error_count={s3_delete_result['verify_error_count']} "
                    f"survivor_samples={s3_delete_result['survivor_samples']}"
                )
        except Exception as e:
            canonical_image_objects_ok = False
            critical_failures.append(f"canonical image object cleanup failed: {e}")

    # Only delete canonical_imagery rows if object cleanup was clean.
    if new_image_ids and canonical_image_objects_ok:
        try:
            _delete_canonical_imagery_rows(new_image_ids)
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Deleted canonical_imagery rows for {len(new_image_ids)} new image(s)",
            )
        except Exception as e:
            critical_failures.append(f"canonical_imagery row rollback failed: {e}")
    elif new_image_ids and not canonical_image_objects_ok:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Skipping canonical_imagery row deletion because canonical object cleanup was not clean; preserving table-based key discovery for retry.",
            level="warning",
        )

    owner_rows_by_table = _load_owner_label_rows(processed_keys)
    for table, rows in owner_rows_by_table.items():
        if not rows:
            continue

        label_type = LABEL_TYPE_BY_TABLE.get(table)
        if not label_type:
            continue

        try:
            deleted_ids, skipped_rows = _delete_canonical_label_rows_if_orphaned(table, label_type, rows)
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Canonical label rollback table={table}: deleted={len(deleted_ids)} skipped(referenced_or_unknown)={skipped_rows}",
            )

            if deleted_ids:
                kcol = _label_key_col(table)
                deleted_id_set = set(deleted_ids)
                rows_deleted = [r for r in rows if r.get(kcol) in deleted_id_set]
                label_s3_keys = _extract_label_s3_keys(rows_deleted)

                if label_s3_keys:
                    label_delete_result = delete_s3_keys_strict(FILE_BUCKET_NAME, label_s3_keys)
                    log(
                        job_id,
                        user,
                        event_type,
                        LOG_FIREHOSE_STREAM_NAME,
                        (
                            f"{TASK_NAME} Canonical label object cleanup table={table}: "
                            f"attempted={label_delete_result['attempted']} "
                            f"api_error_count={label_delete_result['api_error_count']} "
                            f"survivor_count={label_delete_result['survivor_count']} "
                            f"verify_error_count={label_delete_result['verify_error_count']} "
                            f"survivor_samples={label_delete_result['survivor_samples']}"
                        ),
                        level=(
                            "warning"
                            if (
                                label_delete_result["api_error_count"]
                                or label_delete_result["survivor_count"]
                                or label_delete_result["verify_error_count"]
                            )
                            else "info"
                        ),
                    )

                    if (
                        label_delete_result["api_error_count"]
                        or label_delete_result["survivor_count"]
                        or label_delete_result["verify_error_count"]
                    ):
                        critical_failures.append(
                            f"canonical label object cleanup failed table={table}: "
                            f"attempted={label_delete_result['attempted']} "
                            f"api_error_count={label_delete_result['api_error_count']} "
                            f"survivor_count={label_delete_result['survivor_count']} "
                            f"verify_error_count={label_delete_result['verify_error_count']} "
                            f"survivor_samples={label_delete_result['survivor_samples']}"
                        )

        except Exception as e:
            critical_failures.append(f"canonical label rollback failed table={table}: {e}")

    # 3) SHA rollback
    try:
        d, s, e = delete_sha256_entries_for_job(batch_sha_mappings)
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} SHA256 rollback: deleted={d} skipped(not-matching)={s} errors={e} candidates={len(batch_sha_mappings)}",
        )
        if e:
            critical_failures.append(f"sha256 rollback errors={e}")
    except Exception as e:
        critical_failures.append(f"sha256 rollback failed: {e}")

    # 4) Non-critical cleanup
    try:
        _drop_ctas_tables_best_effort(job_id)
    except Exception:
        pass

    # 5) If any critical rollback step failed, raise so SQS retries later.
    if critical_failures:
        joined = " | ".join(critical_failures[:8])
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Critical rollback failures: {joined}",
            level="error",
        )
        raise RuntimeError(f"{TASK_NAME} Critical rollback failures: {joined}")

    # 6) Mark job FAILED only after rollback succeeds.
    update_success, update_msg = update_job_status(
        job_id,
        "FAILED",
        JOB_TABLE_NAME,
        LOG_FIREHOSE_STREAM_NAME,
        user=user,
        event_type=event_type,
        error_msg=(error_msg or "")[:512],
    )
    if not update_success:
        raise RuntimeError(f"{TASK_NAME} Failed to set job FAILED: {update_msg}")
    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Updated job status to FAILED.")

    # 7) Release lock only after rollback + status update succeed.
    release_success, release_msg = release_lock(
        job_id,
        LOCK_TABLE_NAME,
        LOG_FIREHOSE_STREAM_NAME,
        user=user,
        event_type=event_type,
    )
    if not release_success:
        if str(release_msg).startswith("lock_not_held_by_job_id:"):
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Proceeding because lock already not held by this job: {release_msg}",
                level="warning",
            )
        else:
            raise RuntimeError(f"{TASK_NAME} Release lock failed: {release_msg}")
    else:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Released lock.")

def handler(event, context):
    total_records = 0
    num_processed_successfully = 0

    for record in event.get("Records", []):
        total_records += 1
        try:
            _process_one_record(record)
            num_processed_successfully += 1
        except Exception as e:
            # Raise so the SQS-triggered Lambda retries later.
            try:
                body = json.loads(record.get("body", "{}"))
                job_id = body.get("job_id") or "unknown"
                user = body.get("user") or "unknown"
                event_type = body.get("event_type") or "IMAGE_UPLOAD"
                log(
                    job_id,
                    user,
                    event_type,
                    LOG_FIREHOSE_STREAM_NAME,
                    f"{TASK_NAME} Record processing failed and will be retried: {e}",
                    level="error",
                )
            except Exception:
                pass
            raise

    return {
        "status": "ok",
        "total_records": total_records,
        "successfully_processed_records": num_processed_successfully,
    }