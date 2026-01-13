import os
import json
from typing import Dict, List, Iterable, Tuple, Optional

import boto3
from botocore.exceptions import ClientError

from common.logging_utils import log
from common.s3_utils import (
    delete_s3_prefix,
    parse_s3_uri,
    s3_list_keys,
    s3_read_jsonl_list,
)
from common.ddb_utils import update_job_status, release_lock
from common.iceberg_utils import escape_sql_string
from common.athena_utils import run_athena, athena_fetch_all_rows
from common.table_schemas import (
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
SHA256_TABLE_NAME = os.environ["SHA256_TABLE_NAME"]

TASK_NAME = "[DLQ_PROCESSOR]"

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

def _load_new_sha_mappings_from_processed_upload_staging(processed_keys: List[str]) -> List[Tuple[str, str]]:
    """
    Returns list of (sha256, image_id) mappings that this job attempted to register
    for NEW canonical images (dedup_status='passed').
    We read registration processed outputs (temp) so this works even if ingest never ran.
    """
    upload_jsonls = [k for k in processed_keys if "/upload_staging/" in k and k.endswith(".jsonl")]
    upload_jsonls.sort()
    if not upload_jsonls:
        return []

    out: List[Tuple[str, str]] = []
    seen = set()

    for row in s3_read_jsonl_list(FILE_BUCKET_NAME, upload_jsonls, f"{TASK_NAME}.read_upload_staging_for_sha"):
        # Only NEW canonical images produce a sha mapping in reg worker
        if row.get("dedup_status") != "passed":
            continue

        # The worker sets registration_status="passed" on success for NEW images
        if row.get("registration_status") != "passed":
            continue

        sha = row.get("sha256_hash")
        iid = row.get("image_id")
        if not (isinstance(sha, str) and sha.strip() and isinstance(iid, str) and iid.strip()):
            continue

        key = (sha.strip(), iid.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(key)

    return out


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
                # Only delete if the mapping still points at the image_id we created
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


def _list_registration_processed_keys(job_id: str) -> List[str]:
    prefix = f"temp/image-upload/{job_id}/batches/registration-step/processed/"
    try:
        return s3_list_keys(FILE_BUCKET_NAME, prefix)
    except Exception:
        return []

def _load_canonical_imagery_outputs(job_id: str, processed_keys: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns (new_image_ids, canonical_image_s3_keys)
    """
    canon_keys = [k for k in processed_keys if "/canonical_imagery/" in k and k.endswith(".jsonl")]
    canon_keys.sort()
    if not canon_keys:
        return [], []

    new_image_ids: List[str] = []
    image_s3_keys: List[str] = []
    seen_ids = set()
    seen_s3 = set()

    for row in s3_read_jsonl_list(FILE_BUCKET_NAME, canon_keys, f"{TASK_NAME}.read_canonical_imagery"):
        iid = row.get("image_id")
        if isinstance(iid, str) and iid.strip() and iid.strip() not in seen_ids:
            seen_ids.add(iid.strip())
            new_image_ids.append(iid.strip())

        src = row.get("source_ref")
        k = _safe_s3_key_from_uri(src) if isinstance(src, str) else None
        if k and k not in seen_s3:
            seen_s3.add(k)
            image_s3_keys.append(k)

    return new_image_ids, image_s3_keys

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

def _load_owner_label_rows(processed_keys: List[str]) -> Dict[str, List[Dict]]:
    """
    Reads canonical label owner-part jsonls and groups rows by __table.
    """
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
    # athena_fetch_all_rows in your codebase returns list[dict] with keys matching select aliases
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

        if source not in ("stepfunctions", "kickoff", "lambda"):
            print(f"{TASK_NAME} Skipping unknown source={source}")
            continue

        if (job_id in (None, "unknown")) or (user is None) or (event_type is None):
            print(f"{TASK_NAME} Ignoring non-job DLQ message: {body}")
            continue

        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} DLQ received message: {body}", level="error")

        # ---- NEW ORDER: collect rollback targets BEFORE deleting temp prefix ----
        processed_keys = _list_registration_processed_keys(job_id)

        # 2) Roll back canonical writes best-effort (registration side-effects)
        #    Only “safe” deletions:
        #      - canonical_imagery + image_labels for NEW images created by this job
        #      - canonical label rows + objects only if orphaned (no remaining image_labels refs)
        try:
            new_image_ids, canon_image_s3_keys = _load_canonical_imagery_outputs(job_id, processed_keys)

            # Delete Iceberg rows for new images
            if new_image_ids:
                try:
                    _delete_image_labels_for_images(new_image_ids)
                    _delete_canonical_imagery_rows(new_image_ids)
                    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Deleted canonical_imagery + image_labels for {len(new_image_ids)} new image(s)")
                except Exception as e:
                    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Iceberg rollback for new images failed: {e}", level="error")

            # Delete canonical image S3 objects for new images
            if canon_image_s3_keys:
                deleted_est, errors = delete_s3_keys_best_effort(FILE_BUCKET_NAME, canon_image_s3_keys)
                msg = f"{TASK_NAME} Deleted canonical image objects: attempted={len(canon_image_s3_keys)} deleted_est={deleted_est} errors={errors}"
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, msg, level=("warning" if errors else "info"))

            # Canonical label rollback (only if orphaned)
            # Note: label_type strings must match your IMAGE_LABELS label_type values.
            # If your label_type values differ, adjust LABEL_TYPE_BY_TABLE.
            LABEL_TYPE_BY_TABLE = {
                CANONICAL_BBOX_TABLE_NAME: "object-detection",
                CANONICAL_SEMANTIC_TABLE_NAME: "semantic-segmentation",
                CANONICAL_INSTANCE_TABLE_NAME: "instance-segmentation",
            }

            owner_rows_by_table = _load_owner_label_rows(processed_keys)

            # We only consider deleting label rows/objects that were part of this job’s owner outputs.
            for table, rows in owner_rows_by_table.items():
                if not rows:
                    continue

                label_type = LABEL_TYPE_BY_TABLE.get(table)
                if not label_type:
                    continue

                # If orphaned, delete label table rows. If deleted, delete their S3 objects.
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
                        # delete S3 objects only for the rows whose ids we deleted
                        kcol = _label_key_col(table)
                        rows_deleted = [r for r in rows if (r.get(kcol) in set(deleted_ids))]
                        label_s3_keys = _extract_label_s3_keys(table, rows_deleted)
                        if label_s3_keys:
                            d_est, errs = delete_s3_keys_best_effort(FILE_BUCKET_NAME, label_s3_keys)
                            msg = f"{TASK_NAME} Deleted canonical label objects table={table}: attempted={len(label_s3_keys)} deleted_est={d_est} errors={errs}"
                            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, msg, level=("warning" if errs else "info"))

                except Exception as e:
                    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Canonical label rollback failed table={table}: {e}", level="error")

        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Rollback orchestration failed: {e}", level="error")

        # 2.5) Drop CTAS temp tables best-effort
        try:
            _drop_ctas_tables_best_effort(job_id)
        except Exception:
            pass

        # 1) Delete temp folder for this job (always, after we used it for rollback)
        prefix = f"temp/image-upload/{job_id}/"
        try:
            delete_s3_prefix(FILE_BUCKET_NAME, prefix, TASK_NAME)
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Deleted temp s3 prefix")
        except Exception:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Temp S3 cleanup failed", level="error")

        # 3) Mark job FAILED
        try:
            update_success, update_msg = update_job_status(
                job_id,
                "FAILED",
                JOB_TABLE_NAME,
                LOG_FIREHOSE_STREAM_NAME,
                user=user,
                event_type=event_type,
                error_msg=None,
            )
            if update_success:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Updated job status to FAILED.")
            else:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Failed to set job FAILED: {update_msg}", level="error")
        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Updating job status FAILED failed: {e}", level="error")

        # 4) Remove any registered hashes (conditional ownership) best-effort
        try:
            mappings = _load_new_sha_mappings_from_processed_upload_staging(processed_keys)
            d, s, e = delete_sha256_entries_for_job(mappings)
            log(
                job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} SHA256 rollback: deleted={d} skipped(not-matching)={s} errors={e} candidates={len(mappings)}"
            )
        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} SHA256 rollback failed: {e}",
                level="error")

        # 5) Release global lock
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
