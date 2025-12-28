import os
import json
import logging
from typing import Dict, List, Iterable, Tuple, Optional

import boto3

from common.utils import (
    log,
    update_job_status,
    release_lock,
    delete_s3_prefix,
    wait_for_athena,
)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ["UPLOAD_STAGING_TABLE_NAME"]
LOCK_TABLE_NAME = os.environ["LOCK_TABLE_NAME"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]

# canonical table names (stable per your schema)
CANONICAL_IMAGERY_TABLE = "canonical_imagery"
CANONICAL_BBOX_TABLE = "canonical_bounding_boxes"
CANONICAL_SEMANTIC_TABLE = "canonical_semantic_masks"
CANONICAL_INSTANCE_TABLE = "canonical_instance_annotations"

athena = boto3.client("athena")
s3 = boto3.client("s3")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def _escape_sql_string(s: str) -> str:
    return s.replace("'", "''")

def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    # expects s3://bucket/key
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    bucket_key = uri[len("s3://"):]
    bucket, key = bucket_key.split("/", 1)
    return bucket, key

def _basename(key: str) -> str:
    return key.rsplit("/", 1)[-1]

def _split_ext(name: str) -> Tuple[str, str]:
    if "." not in name:
        return name, ""
    stem, ext = name.rsplit(".", 1)
    return stem, ext.lower()

def _athena_query(sql: str, poll: float = 2.0, timeout: int = 900) -> str:
    q = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP,
    )
    qid = q["QueryExecutionId"]
    res = wait_for_athena(qid, poll=poll, timeout=timeout)
    if res["state"] != "SUCCEEDED":
        meta = res.get("metadata")
        raise RuntimeError(f"Athena query failed state={res['state']} sql={sql} meta={meta}")
    return qid

def _athena_fetch_all_rows(qid: str) -> List[Dict[str, Optional[str]]]:
    """
    Returns list[dict] mapping column_name -> VarCharValue (strings).
    Note: Athena returns everything as strings here.
    """
    rows_out: List[Dict[str, Optional[str]]] = []
    next_token = None
    header: List[str] = []

    while True:
        kwargs = {"QueryExecutionId": qid}
        if next_token:
            kwargs["NextToken"] = next_token

        resp = athena.get_query_results(**kwargs)
        rs = resp.get("ResultSet", {})
        rows = rs.get("Rows", [])

        # first page contains header row
        if not header:
            if not rows:
                return []
            header = [c.get("VarCharValue", "") for c in rows[0].get("Data", [])]
            rows = rows[1:]  # drop header row

        for r in rows:
            data = r.get("Data", [])
            item = {}
            for i, col in enumerate(header):
                v = data[i].get("VarCharValue") if i < len(data) else None
                item[col] = v
            rows_out.append(item)

        next_token = resp.get("NextToken")
        if not next_token:
            break

    return rows_out

def _get_registered_upload_rows(job_id: str) -> List[Dict[str, Optional[str]]]:
    """
    Pull enough columns from upload_staging to compute rollback targets.
    """
    safe_job = _escape_sql_string(job_id)
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

    qid = _athena_query(sql, poll=2.0, timeout=900)
    return _athena_fetch_all_rows(qid)

def _derive_canonical_image_key(data_source: str, image_id: str, temp_source_ref: str) -> Optional[str]:
    """
    canonical/imagery/<data_source>/<image_id>.<ext>
    ext is derived from temp_source_ref filename.
    """
    if not (data_source and image_id and temp_source_ref):
        return None
    try:
        _, src_key = _parse_s3_uri(temp_source_ref)
        fname = _basename(src_key)
        _, ext = _split_ext(fname)
        if ext not in ("png", "jpg", "jpeg"):
            return None
        return f"canonical/imagery/{data_source}/{image_id}.{ext}"
    except Exception:
        return None

def _label_uuid_from_temp_ref(uri: str) -> Optional[str]:
    if not uri:
        return None
    try:
        _, key = _parse_s3_uri(uri)
        fname = _basename(key)
        stem, _ = _split_ext(fname)
        return stem or None
    except Exception:
        return None

def _derive_canonical_label_keys(row: Dict[str, Optional[str]]) -> List[str]:
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
        lu = _label_uuid_from_temp_ref(bbox)
        if lu:
            out.append(f"canonical/labels/object-detection/{lu}.json")

    # semantic
    if sem_png or sem_meta:
        lu1 = _label_uuid_from_temp_ref(sem_png) if sem_png else None
        lu2 = _label_uuid_from_temp_ref(sem_meta) if sem_meta else None
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
        lu1 = _label_uuid_from_temp_ref(ins_png) if ins_png else None
        lu2 = _label_uuid_from_temp_ref(ins_meta) if ins_meta else None
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

def _chunked(iterable: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def _delete_s3_keys_best_effort(bucket: str, keys: List[str], batch_size: int = 1000) -> Tuple[int, int]:
    """
    Returns (deleted_count_estimate, error_count).
    """
    if not keys:
        return 0, 0

    deleted = 0
    errors_total = 0

    for chunk in _chunked(keys, batch_size):
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

def _delete_iceberg_by_image_ids(table_name: str, image_ids: List[str], chunk_size: int = 500) -> None:
    """
    DELETE FROM <table> WHERE image_id IN ('...', ...), chunked to keep SQL size sane.
    """
    if not image_ids:
        return

    db = ICEBERG_DATABASE_NAME
    for chunk in _chunked(image_ids, chunk_size):
        in_list = ", ".join(f"'{_escape_sql_string(i)}'" for i in chunk)
        sql = f'DELETE FROM "{db}"."{table_name}" WHERE image_id IN ({in_list})'
        _athena_query(sql, poll=2.0, timeout=1800)

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
            print("[DLQ_PROCESSOR] Skipping non-JSON message")
            continue

        source = body.get("source")
        job_id = body.get("job_id")
        user = body.get("user")
        event_type = body.get("event_type")
        error = body.get("error")

        if source not in ("stepfunctions", "kickoff", "lambda"):
            print(f"[DLQ_PROCESSOR] Skipping unknown source={source}")
            continue

        if (job_id in (None, "unknown")) or (user is None) or (event_type is None):
            print(f"[DLQ_PROCESSOR] Ignoring non-job DLQ message: {body}")
            continue

        # Original failure reason
        log(job_id, user, event_type, "[DLQ_PROCESSOR] Original failure reason", LOG_FIREHOSE_STREAM_NAME, error=str(error), level="error")
        log(job_id, user, event_type, f"[DLQ_PROCESSOR] DLQ received message: {body}", LOG_FIREHOSE_STREAM_NAME)

        # 1) Delete temp folder for this job (always)
        prefix = f"temp/image-upload/{job_id}/"
        try:
            delete_s3_prefix(FILE_BUCKET_NAME, prefix)
            temp_delete_success = True
            log(job_id, user, event_type, "[DLQ_PROCESSOR] Deleted temp s3 prefix", LOG_FIREHOSE_STREAM_NAME)
        except Exception as e:
            canonical_cleanup_success = False
            log(job_id, user, event_type, "[DLQ_PROCESSOR] Temp S3 cleanup failed", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")

        # 2) Roll back canonical writes (Iceberg + S3) best-effort
        #    We DO NOT delete upload_staging anymore.
        try:
            registered_rows = _get_registered_upload_rows(job_id)
        except Exception as e:
            canonical_cleanup_success = False
            log(job_id, user, event_type, "[DLQ_PROCESSOR] Failed querying upload_staging for rollback targets", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
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
                    img_key = _derive_canonical_image_key(row_ds, image_id or "", temp_ref)
                    if img_key:
                        s3_keys_to_delete.append(img_key)

                    # Label keys (based on temp label refs)
                    s3_keys_to_delete.extend(_derive_canonical_label_keys(r))

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
                    _delete_iceberg_by_image_ids(CANONICAL_IMAGERY_TABLE, image_ids)
                    _delete_iceberg_by_image_ids(CANONICAL_BBOX_TABLE, image_ids)
                    _delete_iceberg_by_image_ids(CANONICAL_SEMANTIC_TABLE, image_ids)
                    _delete_iceberg_by_image_ids(CANONICAL_INSTANCE_TABLE, image_ids)
                    log(job_id, user, event_type, f"[DLQ_PROCESSOR] Deleted canonical iceberg rows for {len(image_ids)} image_id(s)", LOG_FIREHOSE_STREAM_NAME)
                except Exception as e:
                    canonical_cleanup_success = False
                    log(job_id, user, event_type, "[DLQ_PROCESSOR] Canonical Iceberg cleanup failed", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")

                # 2b) delete canonical s3 objects
                try:
                    # de-dupe keys preserve order
                    seenk = set()
                    uniqk = []
                    for k in s3_keys_to_delete:
                        if k and k not in seenk:
                            seenk.add(k)
                            uniqk.append(k)

                    deleted_est, errors = _delete_s3_keys_best_effort(FILE_BUCKET_NAME, uniqk, batch_size=1000)
                    msg = f"[DLQ_PROCESSOR] Deleted canonical S3 objects: attempted={len(uniqk)} deleted_est={deleted_est} errors={errors}"
                    if errors:
                        canonical_cleanup_success = False
                        log(job_id, user, event_type, msg, LOG_FIREHOSE_STREAM_NAME, warning="Some canonical S3 deletes reported errors")
                    else:
                        log(job_id, user, event_type, msg, LOG_FIREHOSE_STREAM_NAME)
                except Exception as e:
                    canonical_cleanup_success = False
                    log(job_id, user, event_type, "[DLQ_PROCESSOR] Canonical S3 cleanup failed", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")

            except Exception as e:
                canonical_cleanup_success = False
                log(job_id, user, event_type, "[DLQ_PROCESSOR] Rollback orchestration failed", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        else:
            log(job_id, user, event_type, "[DLQ_PROCESSOR] No registration-passed rows found; skipping canonical rollback", LOG_FIREHOSE_STREAM_NAME)

        # 3) Mark job FAILED (keep this)
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
            log(job_id, user, event_type, f"[DLQ_PROCESSOR] Updated job status to FAILED. success={update_success}, msg={update_msg}", LOG_FIREHOSE_STREAM_NAME)
        except Exception as e:
            log(job_id, user, event_type, "[DLQ_PROCESSOR] Updating job status FAILED", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")

        # 4) Release global lock (keep this)
        try:
            release_success, release_msg = release_lock(
                job_id,
                LOCK_TABLE_NAME,
                LOG_FIREHOSE_STREAM_NAME,
                user=user,
                event_type=event_type,
            )
            log(job_id, user, event_type, f"[DLQ_PROCESSOR] Release lock attempt. success={release_success}, msg={release_msg}", LOG_FIREHOSE_STREAM_NAME)
        except Exception as e:
            log(job_id, user, event_type, "[DLQ_PROCESSOR] Release lock failed", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")

        # Did we at least fail the job + unlock?
        if update_success and release_success:
            num_processed_successfully += 1

    return {
        "status": "ok",
        "total_records": total_records,
        "successfully_processed_records": num_processed_successfully,
    }
