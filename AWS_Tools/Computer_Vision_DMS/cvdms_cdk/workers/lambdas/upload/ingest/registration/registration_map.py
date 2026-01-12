#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Tuple, Iterable, Set
import boto3

from common.logging_utils import log
from common.s3_utils import s3_read_jsonl_list
from common.athena_utils import run_athena
from common.iceberg_utils import chunked_insert, chunked_insert_where_not_exists
from common.table_schemas import (
    UPLOAD_STAGING_TABLE_NAME,
    CANONICAL_IMAGERY_TABLE_NAME,
    IMAGE_LABELS_TABLE_NAME,
    CANONICAL_BBOX_TABLE_NAME,
    CANONICAL_SEMANTIC_TABLE_NAME,
    CANONICAL_INSTANCE_TABLE_NAME,
)

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[REG_INGEST_MAP]"
CHUNK_SIZE = 200

LABEL_TABLES = {
    CANONICAL_BBOX_TABLE_NAME,
    CANONICAL_SEMANTIC_TABLE_NAME,
    CANONICAL_INSTANCE_TABLE_NAME,
}

athena = boto3.client("athena")


def _escape_sql_string(s: str) -> str:
    return s.replace("'", "''")


def _chunks(lst: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def _athena_fetch_rows_3col(qid: str) -> List[Tuple[str, str, str]]:
    """
    Fetch all rows (image_id, label_type, label_id) from a SELECT.
    Assumes the query selects exactly those 3 columns in that order.
    """
    out: List[Tuple[str, str, str]] = []
    paginator = athena.get_paginator("get_query_results")

    first = True
    for page in paginator.paginate(QueryExecutionId=qid):
        rows = page["ResultSet"]["Rows"]
        for r in rows:
            # Skip header row once
            if first:
                first = False
                continue
            data = r.get("Data", [])
            if len(data) < 3:
                continue
            a = (data[0].get("VarCharValue") or "").strip()
            b = (data[1].get("VarCharValue") or "").strip()
            c = (data[2].get("VarCharValue") or "").strip()
            if a and b and c:
                out.append((a, b, c))
    return out


def fetch_existing_image_label_keys(
    image_ids: List[str],
    label_types: List[str],
) -> Set[Tuple[str, str, str]]:
    """
    Returns a set of (image_id, label_type, label_id) that already exist in Iceberg.
    Chunked to avoid mega-IN queries.
    """
    if not image_ids or not label_types:
        return set()

    full_table = f"\"{ICEBERG_DATABASE_NAME}\".\"{IMAGE_LABELS_TABLE_NAME}\""

    lt_in = ", ".join("'" + _escape_sql_string(x) + "'" for x in sorted(set(label_types)))

    existing: Set[Tuple[str, str, str]] = set()

    # Tune chunk size based on query length; 200 is usually safe.
    for chunk in _chunks(sorted(set(image_ids)), 200):
        iid_in = ", ".join("'" + _escape_sql_string(x) + "'" for x in chunk)
        sql = f"""
        SELECT image_id, label_type, label_id
        FROM {full_table}
        WHERE label_type IN ({lt_in})
          AND image_id IN ({iid_in})
        """

        qid, _ = run_athena(
            sql,
            f"{TASK_NAME} fetch_existing_image_labels",
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
            poll=2.0,
            timeout=300,
        )
        for t in _athena_fetch_rows_3col(qid):
            existing.add(t)

    return existing


def _iter_jsonl(bucket: str, key: str, task: str) -> Iterable[Dict]:
    # s3_read_jsonl_list takes list of keys
    return s3_read_jsonl_list(bucket, [key], task)


def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        label_type = event["label_type"]
        shard = event["shard"]
        upload_staging_key = event["upload_staging_key"]
        canonical_imagery_key = event["canonical_imagery_key"]
        canonical_labels_key = event["canonical_labels_key"]
        image_labels_key = event["image_labels_key"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} missing job_id")
    if not shard:
        raise RuntimeError(f"{TASK_NAME} missing shard")
    if not upload_staging_key:
        raise RuntimeError(f"{TASK_NAME} missing upload_staging_key")
    if not canonical_imagery_key:
        raise RuntimeError(f"{TASK_NAME} missing canonical_imagery_key")
    if not canonical_labels_key:
        raise RuntimeError(f"{TASK_NAME} missing canonical_labels_key")
    if not image_labels_key:
        raise RuntimeError(f"{TASK_NAME} missing image_labels_key")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Start ingest shard={shard}")

    # ---------------------------
    # 0) Load image_labels rows and compute "existing" set for enriched/no_op
    # ---------------------------
    incoming_image_labels: List[Dict[str, str]] = []
    incoming_keys: Set[Tuple[str, str, str]] = set()
    incoming_image_ids: Set[str] = set()
    incoming_label_types: Set[str] = set()

    any_img_labels = False
    for r in _iter_jsonl(FILE_BUCKET_NAME, image_labels_key, f"{TASK_NAME}.read_image_labels"):
        any_img_labels = True
        iid = r.get("image_id")
        lt = r.get("label_type")
        lid = r.get("label_id")
        if not (isinstance(iid, str) and iid.strip() and isinstance(lt, str) and lt.strip() and isinstance(lid, str) and lid.strip()):
            continue
        key = (iid.strip(), lt.strip(), lid.strip())
        if key in incoming_keys:
            continue
        incoming_keys.add(key)
        incoming_image_labels.append({"image_id": key[0], "label_type": key[1], "label_id": key[2]})
        incoming_image_ids.add(key[0])
        incoming_label_types.add(key[1])

    # If the file exists but is empty, any_img_labels will be False; that's OK.
    existing_keys: Set[Tuple[str, str, str]] = set()
    if incoming_image_ids and incoming_label_types:
        existing_keys = fetch_existing_image_label_keys(list(incoming_image_ids), list(incoming_label_types))

    # Compute which keys are truly "new" vs already present
    new_keys = incoming_keys - existing_keys

    # ---------------------------
    # 1) Load upload_staging rows, compute external_dup status using new_keys
    # ---------------------------
    upload_rows: List[Dict] = []
    enriched_count = 0
    noop_count = 0
    failed_count = 0

    for row in _iter_jsonl(FILE_BUCKET_NAME, upload_staging_key, f"{TASK_NAME}.read_upload_staging"):
        # We only adjust statuses for external_duplicate rows that are not failed
        dstat = row.get("dedup_status")
        rstat = row.get("registration_status")

        if dstat == "external_duplicate" and rstat != "failed":
            matched = row.get("matched_image_id")
            if isinstance(matched, str) and matched.strip():
                target_image_id = matched.strip()
            else:
                # If this happens, it’s a real worker bug; mark failed.
                row["registration_status"] = "failed"
                row["registration_error"] = "missing matched_image_id for external_duplicate"
                failed_count += 1
                upload_rows.append(row)
                continue

            # Determine the mapping keys this row would create
            row_new = False
            if label_type in ("single-label", "multi-label"):
                labs = row.get("string_labels")
                if isinstance(labs, list):
                    for lab in labs:
                        if isinstance(lab, str) and lab.strip():
                            k = (target_image_id, "string-label", lab.strip().lower())
                            if k in new_keys:
                                row_new = True
                                break
            else:
                fp = row.get("label_fingerprint")
                if isinstance(fp, str) and fp.strip():
                    k = (target_image_id, label_type, fp.strip())
                    if k in new_keys:
                        row_new = True
                else:
                    # structured type but missing fingerprint => worker should have failed it;
                    # make it failed here to keep audit truthful.
                    row["registration_status"] = "failed"
                    row["registration_error"] = "missing label_fingerprint for structured external_duplicate"
                    failed_count += 1
                    upload_rows.append(row)
                    continue

            if row_new:
                row["registration_status"] = "enriched"
                row["registration_error"] = None
                enriched_count += 1
            else:
                row["registration_status"] = "no_op"
                row["registration_error"] = None
                noop_count += 1

        upload_rows.append(row)

    # Ingest upload_staging (delete-then-insert by (job_id,image_id) batches)
    if upload_rows:
        ok, err = chunked_insert(
            upload_rows,
            f"{TASK_NAME}.upload_staging",
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_WORKGROUP,
            ATHENA_OUTPUT_S3,
            chunk_size=CHUNK_SIZE,
        )
        if not ok:
            raise RuntimeError(f"{TASK_NAME} upload_staging chunked_insert failed: {err}")

    # ---------------------------
    # 2) canonical_imagery (delete-then-insert; may be empty if shard is all external dups)
    # ---------------------------
    canon_img_rows: List[Dict] = []
    for row in _iter_jsonl(FILE_BUCKET_NAME, canonical_imagery_key, f"{TASK_NAME}.read_canonical_imagery"):
        canon_img_rows.append(row)

    if canon_img_rows:
        ok, err = chunked_insert(
            canon_img_rows,
            f"{TASK_NAME}.canonical_imagery",
            ICEBERG_DATABASE_NAME,
            CANONICAL_IMAGERY_TABLE_NAME,
            ATHENA_WORKGROUP,
            ATHENA_OUTPUT_S3,
            chunk_size=CHUNK_SIZE,
        )
        if not ok:
            raise RuntimeError(f"{TASK_NAME} canonical_imagery chunked_insert failed: {err}")

    # ---------------------------
    # 3) image_labels (INSERT ONLY WHERE NOT EXISTS)
    # ---------------------------
    if incoming_image_labels:
        ok, err = chunked_insert_where_not_exists(
            incoming_image_labels,
            f"{TASK_NAME}.image_labels",
            ICEBERG_DATABASE_NAME,
            IMAGE_LABELS_TABLE_NAME,
            ATHENA_WORKGROUP,
            ATHENA_OUTPUT_S3,
            chunk_size=CHUNK_SIZE,
        )
        if not ok:
            raise RuntimeError(f"{TASK_NAME} image_labels insert-only failed: {err}")

    # ---------------------------
    # 4) canonical label tables routed by __table (INSERT ONLY WHERE NOT EXISTS)
    # ---------------------------
    per_table: Dict[str, List[Dict]] = {t: [] for t in LABEL_TABLES}
    any_canon_labels = False

    for row in _iter_jsonl(FILE_BUCKET_NAME, canonical_labels_key, f"{TASK_NAME}.read_canonical_labels"):
        any_canon_labels = True
        table = row.get("__table")
        if not isinstance(table, str) or not table.strip():
            raise RuntimeError(f"{TASK_NAME} canonical label row missing __table: {row}")
        table = table.strip()

        if table not in LABEL_TABLES:
            raise RuntimeError(f"{TASK_NAME} unsupported canonical label table in __table: {table}")

        r2 = dict(row)
        r2.pop("__table", None)
        per_table[table].append(r2)

        if len(per_table[table]) >= CHUNK_SIZE:
            chunk = per_table[table]
            ok, err = chunked_insert_where_not_exists(
                chunk,
                f"{TASK_NAME}.canonical_labels.{table}",
                ICEBERG_DATABASE_NAME,
                table,
                ATHENA_WORKGROUP,
                ATHENA_OUTPUT_S3,
                chunk_size=CHUNK_SIZE,
            )
            if not ok:
                raise RuntimeError(f"{TASK_NAME} canonical label insert-only failed table={table}: {err}")
            per_table[table] = []

    # flush remaining
    for table, chunk in per_table.items():
        if not chunk:
            continue
        ok, err = chunked_insert_where_not_exists(
            chunk,
            f"{TASK_NAME}.canonical_labels.{table}",
            ICEBERG_DATABASE_NAME,
            table,
            ATHENA_WORKGROUP,
            ATHENA_OUTPUT_S3,
            chunk_size=CHUNK_SIZE,
        )
        if not ok:
            raise RuntimeError(f"{TASK_NAME} canonical label insert-only failed table={table}: {err}")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Done shard={shard}: "
            f"upload_rows={len(upload_rows)} "
            f"canon_imagery_rows={len(canon_img_rows)} "
            f"image_labels_incoming={len(incoming_image_labels)} "
            f"image_labels_new={len(new_keys)} "
            f"external_dup_enriched={enriched_count} "
            f"external_dup_no_op={noop_count} "
            f"external_dup_failed={failed_count} "
            f"canon_labels_seen={any_canon_labels}"
        ),
    )

    return {
        "job_id": job_id,
        "shard": shard,
        "upload_rows": len(upload_rows),
        "canonical_imagery_rows": len(canon_img_rows),
        "image_labels_incoming": len(incoming_image_labels),
        "image_labels_new": len(new_keys),
        "external_dup_enriched": enriched_count,
        "external_dup_no_op": noop_count,
        "external_dup_failed": failed_count,
    }
