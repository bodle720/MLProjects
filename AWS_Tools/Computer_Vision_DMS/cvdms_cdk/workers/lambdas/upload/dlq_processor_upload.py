import os
import json
import time
import random
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Iterable, Tuple, Optional, Set, Any

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
QUIESCENCE_MAX_WAIT_SEC = 600
QUIESCENCE_REQUIRED_EMPTY_POLLS = 2
ACTIVE_MARKER_STALE_SEC = 20 * 60

# Batch status interpretation for active markers written by Batch workers.
BATCH_LIVE_STATUSES = {"SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"}
BATCH_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED"}

# Narrow Iceberg commit retry for rollback DML
COMMIT_RETRY_ATTEMPTS = 4
COMMIT_RETRY_BASE_SLEEP_SEC = 2.0
COMMIT_RETRY_JITTER_SEC = 0.5

# Low remaining-time guardrails for better logs and fewer opaque timeouts
LOW_TIME_WARN_MS = 180_000
LOW_TIME_ABORT_MS = 60_000

s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")
batch_client = boto3.client("batch")

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


def _remaining_ms(context: Any) -> Optional[int]:
    try:
        return int(context.get_remaining_time_in_millis())
    except Exception:
        return None


def _log_traceback(job_id: str, user: str, event_type: str, prefix: str, exc: Exception) -> None:
    tb = traceback.format_exc()
    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} {prefix}: {exc}; traceback={tb[:12000]}",
        level="error",
    )


def _log_phase_start(job_id: str, user: str, event_type: str, phase: str, detail: str = "") -> float:
    msg = f"{TASK_NAME} START phase={phase}"
    if detail:
        msg += f" {detail}"
    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, msg)
    return time.monotonic()


def _log_phase_done(job_id: str, user: str, event_type: str, phase: str, started_at: float, detail: str = "") -> None:
    elapsed = time.monotonic() - started_at
    msg = f"{TASK_NAME} DONE phase={phase} elapsed_s={elapsed:.1f}"
    if detail:
        msg += f" {detail}"
    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, msg)


def _check_time_budget(context: Any, job_id: str, user: str, event_type: str, phase: str) -> None:
    remaining = _remaining_ms(context)
    if remaining is None:
        return
    if remaining <= LOW_TIME_ABORT_MS:
        raise RuntimeError(
            f"{TASK_NAME} aborting before phase={phase} due to low remaining time: remaining_ms={remaining}"
        )
    if remaining <= LOW_TIME_WARN_MS:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Low remaining time before phase={phase}: remaining_ms={remaining}",
            level="warning",
        )


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


def _try_read_json_key(key: str) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        payload = _read_json_key(key)
        if not isinstance(payload, dict):
            return None, "non_dict_payload"
        return payload, None
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return None, "missing"
        return None, f"client_error:{code}"
    except Exception as e:
        return None, f"{type(e).__name__}:{e}"


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


def _describe_batch_jobs_statuses(job_ids: List[str]) -> Tuple[Dict[str, str], Set[str]]:
    """
    Returns:
      - status_by_job_id
      - unresolved_job_ids: job IDs not returned by DescribeJobs; treat as terminal/missing
    """
    unique_ids = sorted({j.strip() for j in job_ids if isinstance(j, str) and j.strip()})
    status_by_job_id: Dict[str, str] = {}
    unresolved: Set[str] = set()

    if not unique_ids:
        return status_by_job_id, unresolved

    for chunk in chunked(unique_ids, 100):
        resp = batch_client.describe_jobs(jobs=chunk)
        returned: Set[str] = set()

        for job in resp.get("jobs", []) or []:
            jid = job.get("jobId")
            status = job.get("status")
            if isinstance(jid, str) and jid.strip():
                jid = jid.strip()
                returned.add(jid)
                if isinstance(status, str) and status.strip():
                    status_by_job_id[jid] = status.strip()

        unresolved.update(set(chunk) - returned)

    return status_by_job_id, unresolved


def _wait_for_worker_quiescence_or_raise(job_id: str, user: str, event_type: str) -> None:
    """
    Batch-backed active markers now block rollback only while AWS Batch still reports
    the exact job as live. Non-batch or unreadable markers still use the old stale-timeout
    fallback because we have no better control-plane source of truth for them yet.
    """
    start = time.time()
    empty_polls = 0

    while True:
        markers = _list_active_worker_markers(job_id)
        now = _utc_now()

        # Batch-backed markers: decide with Batch control-plane status, not marker age.
        batch_markers: List[Tuple[str, str]] = []

        # Non-batch/unknown markers: fall back to freshness-vs-stale age.
        fresh_nonbatch: List[str] = []
        stale_nonbatch: List[str] = []

        for key, last_modified in markers:
            age_sec = (now - last_modified).total_seconds()
            payload, read_err = _try_read_json_key(key)

            # Marker disappeared between list and read. Treat as gone.
            if read_err == "missing":
                continue

            if payload:
                worker_kind = str(payload.get("worker_kind") or "").strip().lower()
                batch_job_id = str(payload.get("batch_job_id") or "").strip()

                if worker_kind == "batch" and batch_job_id:
                    batch_markers.append((key, batch_job_id))
                    continue

            # No usable batch identity: fall back to age-based logic.
            if age_sec > ACTIVE_MARKER_STALE_SEC:
                stale_nonbatch.append(key)
            else:
                if read_err:
                    fresh_nonbatch.append(f"{key}:marker_read_error={read_err}")
                else:
                    fresh_nonbatch.append(key)

        batch_live: List[str] = []
        batch_terminal: List[str] = []
        batch_unknown: List[str] = []

        if batch_markers:
            try:
                status_by_job_id, unresolved_job_ids = _describe_batch_jobs_statuses(
                    [jid for _, jid in batch_markers]
                )

                for key, jid in batch_markers:
                    status = status_by_job_id.get(jid)
                    if status in BATCH_LIVE_STATUSES:
                        batch_live.append(f"{key}:job_id={jid}:status={status}")
                    elif status in BATCH_TERMINAL_STATUSES:
                        batch_terminal.append(f"{key}:job_id={jid}:status={status}")
                    elif jid in unresolved_job_ids:
                        # Not returned by DescribeJobs. Treat as terminal/missing, not blocking.
                        batch_terminal.append(f"{key}:job_id={jid}:status=MISSING")
                    else:
                        batch_unknown.append(f"{key}:job_id={jid}:status={status or 'UNKNOWN'}")

            except Exception as e:
                # If Batch status lookup fails, be conservative: treat batch markers as blocking.
                batch_unknown.extend([f"{key}:job_id={jid}:describe_error" for key, jid in batch_markers])
                log(
                    job_id,
                    user,
                    event_type,
                    LOG_FIREHOSE_STREAM_NAME,
                    f"{TASK_NAME} Batch DescribeJobs failed during quiescence; falling back to blocking on batch markers this poll: {e}",
                    level="warning",
                )

        blocking = fresh_nonbatch + batch_live + batch_unknown

        if blocking:
            empty_polls = 0
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                (
                    f"{TASK_NAME} Waiting for worker quiescence; "
                    f"blocking_count={len(blocking)} "
                    f"live_batch_count={len(batch_live)} "
                    f"unknown_batch_count={len(batch_unknown)} "
                    f"fresh_nonbatch_count={len(fresh_nonbatch)} "
                    f"terminal_batch_count={len(batch_terminal)} "
                    f"stale_nonbatch_count={len(stale_nonbatch)} "
                    f"sample_blocking={blocking[:5]}"
                ),
                level="warning",
            )
        else:
            empty_polls += 1
            if batch_terminal or stale_nonbatch:
                log(
                    job_id,
                    user,
                    event_type,
                    LOG_FIREHOSE_STREAM_NAME,
                    (
                        f"{TASK_NAME} Quiescence poll clear; "
                        f"terminal_batch_count={len(batch_terminal)} "
                        f"stale_nonbatch_count={len(stale_nonbatch)} "
                        f"required_empty_polls={QUIESCENCE_REQUIRED_EMPTY_POLLS} "
                        f"current_empty_polls={empty_polls} "
                        f"sample_terminal_batch={batch_terminal[:5]} "
                        f"sample_stale_nonbatch={stale_nonbatch[:5]}"
                    ),
                    level="warning",
                )

            if empty_polls >= QUIESCENCE_REQUIRED_EMPTY_POLLS:
                return

        if (time.time() - start) >= QUIESCENCE_MAX_WAIT_SEC:
            if blocking:
                raise RuntimeError(
                    f"{TASK_NAME} rollback blocked by active workers after max wait; "
                    f"blocking_count={len(blocking)} "
                    f"live_batch_count={len(batch_live)} "
                    f"unknown_batch_count={len(batch_unknown)} "
                    f"fresh_nonbatch_count={len(fresh_nonbatch)} "
                    f"sample_blocking={blocking[:5]}"
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
) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str]]]:
    """
    Returns:
      new_image_ids,
      canonical_image_keys_to_delete,
      canonical_label_keys_to_delete,   # safe because reg worker only writes NEW-ONLY label object keys
      sha256_mappings_to_delete
    """
    rollback_seed_keys = [k for k in processed_keys if "/rollback-batch/" in k and k.endswith(".json")]
    rollback_seed_keys.sort()

    new_image_ids: Set[str] = set()
    canonical_image_keys: Set[str] = set()
    canonical_label_keys: Set[str] = set()
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

        for label_key in payload.get("canonical_label_keys_to_delete", []) or []:
            if isinstance(label_key, str) and label_key.strip():
                canonical_label_keys.add(label_key.strip())

        for row in payload.get("sha256_mappings_to_delete", []) or []:
            sha = row.get("sha256")
            iid = row.get("image_id")
            if isinstance(sha, str) and sha.strip() and isinstance(iid, str) and iid.strip():
                sha_mappings.add((sha.strip(), iid.strip()))

    return (
        sorted(new_image_ids),
        sorted(canonical_image_keys),
        sorted(canonical_label_keys),
        sorted(sha_mappings),
    )


def _load_canonical_image_object_keys_from_processed_rows(processed_keys: List[str]) -> List[str]:
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


def _find_referenced_label_ids(label_type: str, label_ids: List[str]) -> Set[str]:
    if not label_ids:
        return set()

    referenced: Set[str] = set()
    safe_label_type = escape_sql_string(label_type)

    for chunk in chunked(sorted(set(label_ids)), 500):
        in_list = ", ".join(f"'{escape_sql_string(i)}'" for i in chunk)
        sql = f"""
        SELECT DISTINCT label_id
        FROM "{ICEBERG_DATABASE_NAME}"."{IMAGE_LABELS_TABLE_NAME}"
        WHERE label_type = '{safe_label_type}'
          AND label_id IN ({in_list})
        """
        qid, _ = run_athena(
            sql,
            f"{TASK_NAME} find_referenced_label_ids:{label_type}",
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
            poll=2.0,
            timeout=300,
        )
        referenced.update(_athena_fetch_rows_1col(qid))

    return referenced


def _delete_canonical_label_rows_if_orphaned(table: str, label_type: str, rows: List[Dict]) -> Tuple[List[str], int]:
    if not rows:
        return [], 0

    kcol = _label_key_col(table)
    ids = _label_ids_from_rows(table, rows)
    if not ids:
        return [], 0

    referenced = _find_referenced_label_ids(label_type, ids)
    to_delete = [lid for lid in ids if lid not in referenced]
    skipped = len(ids) - len(to_delete)

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


def _temp_job_prefix(job_id: str) -> str:
    return f"temp/image-upload/{job_id}/"


def _list_temp_job_keys(job_id: str) -> List[str]:
    prefix = _temp_job_prefix(job_id)
    keys = s3_list_keys(FILE_BUCKET_NAME, prefix)
    return sorted(
        k for k in keys
        if isinstance(k, str) and k.strip() and not k.endswith("/")
    )


def _cleanup_temp_prefix_best_effort(
    job_id: str,
    user: str,
    event_type: str,
    context: Any = None,
) -> None:
    """
    Best-effort cleanup of the entire temp/image-upload/<job_id>/ subtree.
    This runs only after critical rollback succeeds and after job status is set
    to FAILED. It must never raise, to avoid retry loops after correctness-
    critical rollback already completed.
    """
    remaining = _remaining_ms(context)
    if remaining is not None and remaining <= LOW_TIME_ABORT_MS:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Skipping temp prefix cleanup due to low remaining time: remaining_ms={remaining}",
            level="warning",
        )
        return

    prefix = _temp_job_prefix(job_id)
    phase = _log_phase_start(
        job_id,
        user,
        event_type,
        "cleanup_temp_prefix",
        detail=f"prefix=s3://{FILE_BUCKET_NAME}/{prefix}",
    )

    try:
        temp_keys = _list_temp_job_keys(job_id)
    except Exception as e:
        _log_traceback(job_id, user, event_type, "list temp prefix keys failed (best-effort)", e)
        _log_phase_done(
            job_id,
            user,
            event_type,
            "cleanup_temp_prefix",
            phase,
            detail="listed=0 skipped_due_to_list_error=true",
        )
        return

    if not temp_keys:
        _log_phase_done(
            job_id,
            user,
            event_type,
            "cleanup_temp_prefix",
            phase,
            detail="listed=0 attempted=0",
        )
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Temp prefix cleanup found no keys under s3://{FILE_BUCKET_NAME}/{prefix}",
        )
        return

    try:
        delete_result = delete_s3_keys_strict(FILE_BUCKET_NAME, temp_keys)
        _log_phase_done(
            job_id,
            user,
            event_type,
            "cleanup_temp_prefix",
            phase,
            detail=(
                f"listed={len(temp_keys)} attempted={delete_result['attempted']} "
                f"api_error_count={delete_result['api_error_count']} "
                f"survivor_count={delete_result['survivor_count']} "
                f"verify_error_count={delete_result['verify_error_count']}"
            ),
        )
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            (
                f"{TASK_NAME} Temp prefix cleanup result: "
                f"prefix=s3://{FILE_BUCKET_NAME}/{prefix} "
                f"listed={len(temp_keys)} "
                f"attempted={delete_result['attempted']} "
                f"api_error_count={delete_result['api_error_count']} "
                f"survivor_count={delete_result['survivor_count']} "
                f"verify_error_count={delete_result['verify_error_count']} "
                f"api_error_samples={delete_result['api_error_samples']} "
                f"survivor_samples={delete_result['survivor_samples']} "
                f"verify_error_samples={delete_result['verify_error_samples']}"
            ),
            level=(
                "warning"
                if (
                    delete_result["api_error_count"]
                    or delete_result["survivor_count"]
                    or delete_result["verify_error_count"]
                )
                else "info"
            ),
        )
    except Exception as e:
        _log_traceback(job_id, user, event_type, "temp prefix cleanup failed (best-effort)", e)


def _process_one_record(record: Dict[str, object], context: Any = None) -> None:
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

    total_started = time.monotonic()
    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} DLQ received message: {body}")

    # 0) Wait for in-flight workers to quiesce before rollback.
    _check_time_budget(context, job_id, user, event_type, "quiescence")
    phase = _log_phase_start(job_id, user, event_type, "quiescence")
    _wait_for_worker_quiescence_or_raise(job_id, user, event_type)
    _log_phase_done(job_id, user, event_type, "quiescence", phase)

    # 1) Gather rollback sources AFTER quiescence.
    _check_time_budget(context, job_id, user, event_type, "gather_rollback_sources")
    phase = _log_phase_start(job_id, user, event_type, "gather_rollback_sources")
    processed_keys = _list_registration_processed_keys(job_id)

    (
        batch_new_image_ids,
        batch_canonical_image_keys,
        batch_canonical_label_keys,
        batch_sha_mappings,
    ) = _load_batch_rollback_seeds(processed_keys)

    processed_canonical_image_keys = _load_canonical_image_object_keys_from_processed_rows(processed_keys)
    rollback_label_keys, rollback_source_membership_keys = _load_exact_rollback_targets(processed_keys)

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

    canonical_label_seed_keys_to_delete = sorted(set(batch_canonical_label_keys))

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
    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Canonical label object seed cleanup sources: "
            f"seed_only_keys={len(canonical_label_seed_keys_to_delete)} "
            f"sample_seed_keys={canonical_label_seed_keys_to_delete[:5]}"
        ),
    )
    _log_phase_done(
        job_id,
        user,
        event_type,
        "gather_rollback_sources",
        phase,
        detail=(
            f"processed_keys={len(processed_keys)} rollback_label_keys={len(rollback_label_keys)} "
            f"rollback_source_membership_keys={len(rollback_source_membership_keys)} new_image_ids={len(new_image_ids)} "
            f"sha_candidates={len(batch_sha_mappings)} canonical_image_keys={len(canonical_image_keys_to_delete)} "
            f"canonical_label_seed_keys={len(canonical_label_seed_keys_to_delete)}"
        ),
    )

    critical_failures: List[str] = []

    # 2) Registration-side rollback

    if rollback_label_keys:
        try:
            _check_time_budget(context, job_id, user, event_type, "delete_exact_image_labels")
            phase = _log_phase_start(
                job_id,
                user,
                event_type,
                "delete_exact_image_labels",
                detail=f"count={len(rollback_label_keys)}",
            )
            _delete_exact_image_labels(rollback_label_keys)
            _log_phase_done(
                job_id,
                user,
                event_type,
                "delete_exact_image_labels",
                phase,
                detail=f"count={len(rollback_label_keys)}",
            )
        except Exception as e:
            _log_traceback(job_id, user, event_type, "delete_exact_image_labels failed", e)
            critical_failures.append(f"exact image_labels rollback failed: {e}")

    if rollback_source_membership_keys:
        try:
            _check_time_budget(context, job_id, user, event_type, "delete_exact_image_source_memberships")
            phase = _log_phase_start(
                job_id,
                user,
                event_type,
                "delete_exact_image_source_memberships",
                detail=f"count={len(rollback_source_membership_keys)}",
            )
            _delete_exact_image_source_memberships(rollback_source_membership_keys)
            _log_phase_done(
                job_id,
                user,
                event_type,
                "delete_exact_image_source_memberships",
                phase,
                detail=f"count={len(rollback_source_membership_keys)}",
            )
        except Exception as e:
            _log_traceback(job_id, user, event_type, "delete_exact_image_source_memberships failed", e)
            critical_failures.append(f"exact image_source_membership rollback failed: {e}")

    if new_image_ids:
        try:
            _check_time_budget(context, job_id, user, event_type, "delete_new_image_linkage")
            phase = _log_phase_start(
                job_id,
                user,
                event_type,
                "delete_new_image_linkage",
                detail=f"count={len(new_image_ids)}",
            )
            _delete_image_labels_for_images(new_image_ids)
            _delete_image_source_memberships_for_images(new_image_ids)
            _log_phase_done(
                job_id,
                user,
                event_type,
                "delete_new_image_linkage",
                phase,
                detail=f"count={len(new_image_ids)}",
            )
        except Exception as e:
            _log_traceback(job_id, user, event_type, "delete_new_image_linkage failed", e)
            critical_failures.append(f"new-image linkage rollback failed: {e}")

    canonical_image_objects_ok = True
    if canonical_image_keys_to_delete:
        try:
            _check_time_budget(context, job_id, user, event_type, "delete_canonical_image_objects")
            phase = _log_phase_start(
                job_id,
                user,
                event_type,
                "delete_canonical_image_objects",
                detail=f"count={len(canonical_image_keys_to_delete)}",
            )
            s3_delete_result = delete_s3_keys_strict(FILE_BUCKET_NAME, canonical_image_keys_to_delete)
            _log_phase_done(
                job_id,
                user,
                event_type,
                "delete_canonical_image_objects",
                phase,
                detail=(
                    f"attempted={s3_delete_result['attempted']} api_error_count={s3_delete_result['api_error_count']} "
                    f"survivor_count={s3_delete_result['survivor_count']} verify_error_count={s3_delete_result['verify_error_count']}"
                ),
            )
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
            _log_traceback(job_id, user, event_type, "delete_canonical_image_objects failed", e)
            critical_failures.append(f"canonical image object cleanup failed: {e}")

    if new_image_ids and canonical_image_objects_ok:
        try:
            _check_time_budget(context, job_id, user, event_type, "delete_canonical_imagery_rows")
            phase = _log_phase_start(
                job_id,
                user,
                event_type,
                "delete_canonical_imagery_rows",
                detail=f"count={len(new_image_ids)}",
            )
            _delete_canonical_imagery_rows(new_image_ids)
            _log_phase_done(
                job_id,
                user,
                event_type,
                "delete_canonical_imagery_rows",
                phase,
                detail=f"count={len(new_image_ids)}",
            )
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Deleted canonical_imagery rows for {len(new_image_ids)} new image(s)",
            )
        except Exception as e:
            _log_traceback(job_id, user, event_type, "delete_canonical_imagery_rows failed", e)
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

    owner_rows_by_table: Dict[str, List[Dict]] = {t: [] for t in LABEL_TABLES}
    try:
        _check_time_budget(context, job_id, user, event_type, "load_owner_label_rows")
        phase = _log_phase_start(job_id, user, event_type, "load_owner_label_rows")
        owner_rows_by_table = _load_owner_label_rows(processed_keys)
        owner_summary = {table: len(rows) for table, rows in owner_rows_by_table.items() if rows}
        _log_phase_done(
            job_id,
            user,
            event_type,
            "load_owner_label_rows",
            phase,
            detail=f"tables={owner_summary} total={sum(owner_summary.values())}",
        )
    except Exception as e:
        _log_traceback(job_id, user, event_type, "load_owner_label_rows failed", e)
        critical_failures.append(f"load owner label rows failed: {e}")

    # Keep row deletion safety exactly as before: orphan-check only.
    for table, rows in owner_rows_by_table.items():
        if not rows:
            continue

        label_type = LABEL_TYPE_BY_TABLE.get(table)
        if not label_type:
            continue

        try:
            _check_time_budget(context, job_id, user, event_type, f"canonical_label_orphan_check:{table}")
            phase = _log_phase_start(
                job_id,
                user,
                event_type,
                f"canonical_label_orphan_check:{table}",
                detail=f"rows={len(rows)} label_type={label_type}",
            )
            deleted_ids, skipped_rows = _delete_canonical_label_rows_if_orphaned(table, label_type, rows)
            _log_phase_done(
                job_id,
                user,
                event_type,
                f"canonical_label_orphan_check:{table}",
                phase,
                detail=f"deleted={len(deleted_ids)} skipped={skipped_rows}",
            )
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Canonical label rollback table={table}: deleted={len(deleted_ids)} skipped(referenced_or_unknown)={skipped_rows}",
            )

            if deleted_ids:
                _check_time_budget(context, job_id, user, event_type, f"delete_canonical_label_objects:{table}")
                kcol = _label_key_col(table)
                deleted_id_set = set(deleted_ids)
                rows_deleted = [r for r in rows if r.get(kcol) in deleted_id_set]
                label_s3_keys = _extract_label_s3_keys(rows_deleted)

                if label_s3_keys:
                    phase = _log_phase_start(
                        job_id,
                        user,
                        event_type,
                        f"delete_canonical_label_objects:{table}",
                        detail=f"count={len(label_s3_keys)}",
                    )
                    label_delete_result = delete_s3_keys_strict(FILE_BUCKET_NAME, label_s3_keys)
                    _log_phase_done(
                        job_id,
                        user,
                        event_type,
                        f"delete_canonical_label_objects:{table}",
                        phase,
                        detail=(
                            f"attempted={label_delete_result['attempted']} api_error_count={label_delete_result['api_error_count']} "
                            f"survivor_count={label_delete_result['survivor_count']} verify_error_count={label_delete_result['verify_error_count']}"
                        ),
                    )
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
            _log_traceback(job_id, user, event_type, f"canonical label rollback failed table={table}", e)
            critical_failures.append(f"canonical label rollback failed table={table}: {e}")

    # NEW: delete seed-listed canonical label OBJECTS even when owner rows are absent.
    # These keys are safe because the registration worker now records only NEW-ONLY keys.
    if canonical_label_seed_keys_to_delete:
        try:
            _check_time_budget(context, job_id, user, event_type, "delete_seed_canonical_label_objects")
            phase = _log_phase_start(
                job_id,
                user,
                event_type,
                "delete_seed_canonical_label_objects",
                detail=f"count={len(canonical_label_seed_keys_to_delete)}",
            )
            label_seed_delete_result = delete_s3_keys_strict(FILE_BUCKET_NAME, canonical_label_seed_keys_to_delete)
            _log_phase_done(
                job_id,
                user,
                event_type,
                "delete_seed_canonical_label_objects",
                phase,
                detail=(
                    f"attempted={label_seed_delete_result['attempted']} "
                    f"api_error_count={label_seed_delete_result['api_error_count']} "
                    f"survivor_count={label_seed_delete_result['survivor_count']} "
                    f"verify_error_count={label_seed_delete_result['verify_error_count']}"
                ),
            )
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                (
                    f"{TASK_NAME} Canonical label object seed cleanup result: "
                    f"attempted={label_seed_delete_result['attempted']} "
                    f"api_error_count={label_seed_delete_result['api_error_count']} "
                    f"survivor_count={label_seed_delete_result['survivor_count']} "
                    f"verify_error_count={label_seed_delete_result['verify_error_count']} "
                    f"api_error_samples={label_seed_delete_result['api_error_samples']} "
                    f"survivor_samples={label_seed_delete_result['survivor_samples']} "
                    f"verify_error_samples={label_seed_delete_result['verify_error_samples']}"
                ),
                level=(
                    "warning"
                    if (
                        label_seed_delete_result["api_error_count"]
                        or label_seed_delete_result["survivor_count"]
                        or label_seed_delete_result["verify_error_count"]
                    )
                    else "info"
                ),
            )

            if (
                label_seed_delete_result["api_error_count"]
                or label_seed_delete_result["survivor_count"]
                or label_seed_delete_result["verify_error_count"]
            ):
                critical_failures.append(
                    "canonical label seed object cleanup failed: "
                    f"attempted={label_seed_delete_result['attempted']} "
                    f"api_error_count={label_seed_delete_result['api_error_count']} "
                    f"survivor_count={label_seed_delete_result['survivor_count']} "
                    f"verify_error_count={label_seed_delete_result['verify_error_count']} "
                    f"survivor_samples={label_seed_delete_result['survivor_samples']}"
                )
        except Exception as e:
            _log_traceback(job_id, user, event_type, "delete_seed_canonical_label_objects failed", e)
            critical_failures.append(f"canonical label seed object cleanup failed: {e}")

    try:
        _check_time_budget(context, job_id, user, event_type, "sha256_rollback")
        phase = _log_phase_start(job_id, user, event_type, "sha256_rollback", detail=f"candidates={len(batch_sha_mappings)}")
        d, s, e = delete_sha256_entries_for_job(batch_sha_mappings)
        _log_phase_done(job_id, user, event_type, "sha256_rollback", phase, detail=f"deleted={d} skipped={s} errors={e}")
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
        _log_traceback(job_id, user, event_type, "sha256 rollback failed", e)
        critical_failures.append(f"sha256 rollback failed: {e}")

    try:
        _check_time_budget(context, job_id, user, event_type, "drop_ctas_tables")
        phase = _log_phase_start(job_id, user, event_type, "drop_ctas_tables")
        _drop_ctas_tables_best_effort(job_id)
        _log_phase_done(job_id, user, event_type, "drop_ctas_tables", phase)
    except Exception as e:
        _log_traceback(job_id, user, event_type, "drop_ctas_tables failed", e)

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

    try:
        _check_time_budget(context, job_id, user, event_type, "update_job_failed")
        phase = _log_phase_start(job_id, user, event_type, "update_job_failed")
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
        _log_phase_done(job_id, user, event_type, "update_job_failed", phase)
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Updated job status to FAILED.")
    except Exception as e:
        _log_traceback(job_id, user, event_type, "update_job_failed failed", e)
        raise

    # Best-effort only: do not raise from temp cleanup. The correctness-critical
    # rollback above has already completed successfully at this point.
    _cleanup_temp_prefix_best_effort(
        job_id=job_id,
        user=user,
        event_type=event_type,
        context=context,
    )

    try:
        _check_time_budget(context, job_id, user, event_type, "release_lock")
        phase = _log_phase_start(job_id, user, event_type, "release_lock")
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
        _log_phase_done(job_id, user, event_type, "release_lock", phase)
    except Exception as e:
        _log_traceback(job_id, user, event_type, "release_lock failed", e)
        raise

    total_elapsed = time.monotonic() - total_started
    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Rollback complete for job {job_id}. total_elapsed_s={total_elapsed:.1f}",
    )


def handler(event, context):
    total_records = 0
    num_processed_successfully = 0

    for record in event.get("Records", []):
        total_records += 1
        try:
            _process_one_record(record, context=context)
            num_processed_successfully += 1
        except Exception as e:
            try:
                body = json.loads(record.get("body", "{}"))
                job_id = body.get("job_id") or "unknown"
                user = body.get("user") or "unknown"
                event_type = body.get("event_type") or "IMAGE_UPLOAD"
                _log_traceback(job_id, user, event_type, "Record processing failed and will be retried", e)
            except Exception:
                pass
            raise

    return {
        "status": "ok",
        "total_records": total_records,
        "successfully_processed_records": num_processed_successfully,
    }