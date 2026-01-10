import os
import json
from typing import Dict, List, Iterable, Tuple, Optional

import boto3
from botocore.exceptions import ClientError

from common.logging_utils import log
from common.s3_utils import delete_s3_prefix, parse_s3_uri, get_key_basename
from common.ddb_utils import update_job_status, release_lock
from common.iceberg_utils import escape_sql_string
from common.athena_utils import run_athena, athena_fetch_all_rows
from common.table_schemas import CANONICAL_IMAGERY_TABLE_NAME, CANONICAL_BBOX_TABLE_NAME, \
                                 CANONICAL_SEMANTIC_TABLE_NAME, CANONICAL_INSTANCE_TABLE_NAME, \
                                 UPLOAD_STAGING_TABLE_NAME

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

def split_ext(name: str) -> Tuple[str, str]:
    if "." not in name:
        return name, ""
    stem, ext = name.rsplit(".", 1)
    return stem, ext.lower()

def get_job_sha256s(job_id: str) -> List[str]:
    safe_job = escape_sql_string(job_id)
    db = ICEBERG_DATABASE_NAME
    t = UPLOAD_STAGING_TABLE_NAME

    sql = f"""
    SELECT sha256_hash
    FROM "{db}"."{t}"
    WHERE job_id = '{safe_job}'
      AND registration_status = 'passed'
      AND sha256_hash IS NOT NULL
    """

    qid, _ = run_athena(sql,
                         TASK_NAME,
                         ATHENA_OUTPUT_S3,
                         ATHENA_WORKGROUP,
                         poll=2.0,
                         timeout=900)

    rows = athena_fetch_all_rows(qid)
    out = []
    for r in rows:
        v = r.get("sha256_hash")
        if v:
            out.append(v)
    # dedupe preserve order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def delete_sha256_entries_for_job(job_id: str, shas: List[str]) -> tuple[int, int]:
    deleted = 0
    skipped = 0

    for sha in shas:
        try:
            dynamodb.delete_item(
                TableName=SHA256_TABLE_NAME,
                Key={"sha256": {"S": sha}},
                ConditionExpression="job_id = :j",
                ExpressionAttributeValues={":j": {"S": job_id}},
            )
            deleted += 1
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                # either doesn't exist or belongs to some other job => don't delete
                skipped += 1
                continue
            raise

    return deleted, skipped

def get_registered_upload_rows(job_id: str) -> List[Dict[str, Optional[str]]]:
    """
    Pull enough columns from upload_staging to compute rollback targets.
    """
    safe_job = escape_sql_string(job_id)
    db = ICEBERG_DATABASE_NAME
    t = UPLOAD_STAGING_TABLE_NAME

    sql = f"""
    SELECT
      image_id,
      data_source,
      temp_source_ref,
      temp_source_ref_bbox_meta,
      temp_source_ref_semantic_png,
      temp_source_ref_semantic_meta,
      temp_source_ref_instance_png,
      temp_source_ref_instance_meta
    FROM "{db}"."{t}"
    WHERE job_id = '{safe_job}'
      AND registration_status = 'passed'
    """

    qid, _ = run_athena(sql,
                         TASK_NAME,
                         ATHENA_OUTPUT_S3,
                         ATHENA_WORKGROUP,
                         poll=2.0,
                         timeout=900)

    rows = athena_fetch_all_rows(qid)
    return rows

def derive_canonical_image_key(data_source: str, image_id: str, temp_source_ref: str) -> Optional[str]:
    """
    canonical/imagery/<data_source>/<image_id>.<ext>
    ext is derived from temp_source_ref filename.
    """
    if not (data_source and image_id and temp_source_ref):
        return None
    try:
        _, src_key = parse_s3_uri(temp_source_ref, TASK_NAME)
        fname = get_key_basename(src_key)
        _, ext = split_ext(fname)
        if ext not in ("png", "jpg", "jpeg"):
            return None
        return f"canonical/imagery/{data_source}/{image_id}.{ext}"
    except Exception:
        return None

def label_uuid_from_temp_ref(uri: str) -> Optional[str]:
    if not uri:
        return None
    try:
        _, key = parse_s3_uri(uri, TASK_NAME)
        fname = get_key_basename(key)
        stem, _ = split_ext(fname)
        return stem or None
    except Exception:
        return None

def derive_canonical_label_keys(row: Dict[str, Optional[str]]) -> List[str]:
    """
    Based on which temp label refs exist, derive canonical label keys deterministically.
    """
    out: List[str] = []

    bbox = row.get("temp_source_ref_bbox_meta")
    sem_png = row.get("temp_source_ref_semantic_png")
    sem_meta = row.get("temp_source_ref_semantic_meta")
    ins_png = row.get("temp_source_ref_instance_png")
    ins_meta = row.get("temp_source_ref_instance_meta")

    # object detection
    if bbox:
        lu = label_uuid_from_temp_ref(bbox)
        if lu:
            out.append(f"canonical/labels/object-detection/{lu}.json")

    # semantic
    if sem_png or sem_meta:
        lu1 = label_uuid_from_temp_ref(sem_png) if sem_png else None
        lu2 = label_uuid_from_temp_ref(sem_meta) if sem_meta else None
        lu = lu1 or lu2
        # only add if we have a UUID; if mismatch, still best-effort delete both UUIDs
        if lu1 and lu2 and lu1 != lu2:
            out.append(f"canonical/labels/semantic-segmentation/{lu1}.png")
            out.append(f"canonical/labels/semantic-segmentation/{lu1}.json")
            out.append(f"canonical/labels/semantic-segmentation/{lu2}.png")
            out.append(f"canonical/labels/semantic-segmentation/{lu2}.json")
        elif lu:
            out.append(f"canonical/labels/semantic-segmentation/{lu}.png")
            out.append(f"canonical/labels/semantic-segmentation/{lu}.json")

    # instance
    if ins_png or ins_meta:
        lu1 = label_uuid_from_temp_ref(ins_png) if ins_png else None
        lu2 = label_uuid_from_temp_ref(ins_meta) if ins_meta else None
        lu = lu1 or lu2
        if lu1 and lu2 and lu1 != lu2:
            out.append(f"canonical/labels/instance-segmentation/{lu1}.png")
            out.append(f"canonical/labels/instance-segmentation/{lu1}.json")
            out.append(f"canonical/labels/instance-segmentation/{lu2}.png")
            out.append(f"canonical/labels/instance-segmentation/{lu2}.json")
        elif lu:
            out.append(f"canonical/labels/instance-segmentation/{lu}.png")
            out.append(f"canonical/labels/instance-segmentation/{lu}.json")

    # de-dupe preserve order
    seen = set()
    uniq = []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq

def chunked(iterable: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def delete_s3_keys_best_effort(bucket: str, keys: List[str], batch_size: int = 1000) -> Tuple[int, int]:
    """
    Returns (deleted_count_estimate, error_count).
    """
    if not keys:
        return 0, 0

    deleted = 0
    errors_total = 0

    for chunk in chunked(keys, batch_size):
        try:
            resp = s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in chunk]}
            )
            deleted += len(chunk)
            errors = resp.get("Errors", []) or []
            if errors:
                errors_total += len(errors)
        except Exception:
            # treat entire chunk as "errorish", but keep going
            errors_total += len(chunk)

    return deleted, errors_total

def delete_iceberg_by_image_ids(table_name: str, image_ids: List[str], chunk_size: int = 500) -> None:
    """
    DELETE FROM <table> WHERE image_id IN ('...', ...), chunked to keep SQL size sane.
    """
    if not image_ids:
        return

    db = ICEBERG_DATABASE_NAME
    for chunk in chunked(image_ids, chunk_size):
        in_list = ", ".join(f"'{escape_sql_string(i)}'" for i in chunk)
        sql = f'DELETE FROM "{db}"."{table_name}" WHERE image_id IN ({in_list})'
        _, _ = run_athena(sql,
                            TASK_NAME,
                            ATHENA_OUTPUT_S3,
                            ATHENA_WORKGROUP,
                            poll=2.0,
                            timeout=1800)

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

        # Original failure reason
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Original failure reason", level="error")
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} DLQ received message: {body}")

        # 1) Delete temp folder for this job (always)
        prefix = f"temp/image-upload/{job_id}/"
        try:
            delete_s3_prefix(FILE_BUCKET_NAME, prefix, TASK_NAME)
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Deleted temp s3 prefix")
        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Temp S3 cleanup failed", level="error")

        # 2) Roll back canonical writes (Iceberg + S3) best-effort
        try:
            registered_rows = get_registered_upload_rows(job_id)
        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Failed querying upload_staging for rollback targets", level="error")
            registered_rows = []

        # If we have rows that claim registration passed, rollback the canonical side-effects
        if registered_rows:
            try:
                # image_ids for table deletes
                image_ids: List[str] = []
                # s3 keys to delete
                s3_keys_to_delete: List[str] = []

                for r in registered_rows:
                    image_id = r.get("image_id")
                    if image_id:
                        image_ids.append(image_id)

                    # Determine canonical image key (use row's data_source, else message data_source)
                    row_ds = r.get("data_source") or ""
                    temp_ref = r.get("temp_source_ref") or ""
                    img_key = derive_canonical_image_key(row_ds, image_id or "", temp_ref)
                    if img_key:
                        s3_keys_to_delete.append(img_key)

                    # Label keys (based on temp label refs)
                    s3_keys_to_delete.extend(derive_canonical_label_keys(r))

                # de-dupe image_ids preserve order
                seen = set()
                uniq_ids = []
                for i in image_ids:
                    if i not in seen:
                        seen.add(i)
                        uniq_ids.append(i)
                image_ids = uniq_ids

                # 2a) delete canonical iceberg rows (tables first to avoid dangling refs)
                try:
                    delete_iceberg_by_image_ids(CANONICAL_IMAGERY_TABLE_NAME, image_ids)
                    delete_iceberg_by_image_ids(CANONICAL_BBOX_TABLE_NAME, image_ids)
                    delete_iceberg_by_image_ids(CANONICAL_SEMANTIC_TABLE_NAME, image_ids)
                    delete_iceberg_by_image_ids(CANONICAL_INSTANCE_TABLE_NAME, image_ids)
                    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Deleted canonical iceberg rows for {len(image_ids)} image_id(s)")
                except Exception as e:
                    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Canonical Iceberg cleanup failed", level="error")

                # 2b) delete canonical s3 objects
                try:
                    # de-dupe keys preserve order
                    seenk = set()
                    uniqk = []
                    for k in s3_keys_to_delete:
                        if k and k not in seenk:
                            seenk.add(k)
                            uniqk.append(k)

                    deleted_est, errors = delete_s3_keys_best_effort(FILE_BUCKET_NAME, uniqk, batch_size=1000)
                    msg = f"{TASK_NAME} Deleted canonical S3 objects: attempted={len(uniqk)} deleted_est={deleted_est} errors={errors}"
                    if errors:
                        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, msg, level='warning')
                    else:
                        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, msg)
                except Exception as e:
                    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Canonical S3 cleanup failed", level="error")

            except Exception as e:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Rollback orchestration failed", level="error")
        else:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} No registration-passed rows found; skipping canonical rollback")

        # 3) Mark job FAILED
        try:
            update_success, update_msg = update_job_status(job_id,
                                                            "FAILED",
                                                            JOB_TABLE_NAME,
                                                            LOG_FIREHOSE_STREAM_NAME,
                                                            user=user,
                                                            event_type=event_type,
                                                            error_msg=None)

            if update_success:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Updated job to status to FAILED successfully.")
            else:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} Failed to set job status to FAILED, message = {update_msg}", level="error")

        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Updating job status FAILED with exception: {e}", level="error")

        # 4) Remove any registered hashes in the hash table if error occurred after registration
        try:
            shas = get_job_sha256s(job_id)
            d, s = delete_sha256_entries_for_job(job_id, shas)
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} SHA256 rollback: deleted={d} skipped={s} candidates={len(shas)}")
        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} SHA256 rollback failed: {e}", level="error")

        # 5) Release global lock
        try:
            release_success, release_msg = release_lock(job_id,
                                                        LOCK_TABLE_NAME,
                                                        LOG_FIREHOSE_STREAM_NAME,
                                                        user=user,
                                                        event_type=event_type)

            if release_success:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Release lock attempt success.")
            else:
                log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} Release lock failed, message = {release_msg}", level="error")

        except Exception as e:
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Release lock failed with exception: {e}", level="error")

        # Did we at least fail the job + unlock?
        if update_success and release_success:
            num_processed_successfully += 1

    return {
        "status": "ok",
        "total_records": total_records,
        "successfully_processed_records": num_processed_successfully,
    }