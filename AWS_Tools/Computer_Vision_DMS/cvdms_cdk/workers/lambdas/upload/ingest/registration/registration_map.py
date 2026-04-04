#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Tuple, Iterable, Set
import boto3

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import s3_read_jsonl_list, s3_list_keys
from common.general_utils.athena_utils import run_athena
from common.general_utils.iceberg_utils import chunked_insert, chunked_insert_where_not_exists
from common.general_utils.table_schemas import (
    UPLOAD_STAGING_TABLE_NAME,
    CANONICAL_IMAGERY_TABLE_NAME,
    IMAGE_LABELS_TABLE_NAME,
    CANONICAL_BBOX_TABLE_NAME,
    CANONICAL_SEMANTIC_TABLE_NAME,
    CANONICAL_INSTANCE_TABLE_NAME
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
        yield lst[i : i + n]

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
            if first:
                first = False
                continue  # header
            data = r.get("Data", [])
            if len(data) < 3:
                continue
            a = (data[0].get("VarCharValue") or "").strip()
            b = (data[1].get("VarCharValue") or "").strip()
            c = (data[2].get("VarCharValue") or "").strip()
            if a and b and c:
                out.append((a, b, c))
    return out

def fetch_existing_image_label_keys(image_ids: List[str], label_types: List[str]) -> Set[Tuple[str, str, str]]:
    """
    Returns a set of (image_id, label_type, label_id) that already exist in Iceberg.
    Chunked to avoid mega-IN queries.
    """
    if not image_ids or not label_types:
        return set()

    full_table = f"\"{ICEBERG_DATABASE_NAME}\".\"{IMAGE_LABELS_TABLE_NAME}\""

    lt_in = ", ".join("'" + _escape_sql_string(x) + "'" for x in sorted(set(label_types)))
    existing: Set[Tuple[str, str, str]] = set()

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

def _iter_jsonl_keys(keys: List[str], task: str) -> Iterable[Dict]:
    # keys are S3 keys (NOT URIs)
    return s3_read_jsonl_list(FILE_BUCKET_NAME, keys, task)

def _list_owner_label_jsonl_keys(owner_prefix: str) -> List[str]:
    """
    owner_prefix is an S3 *prefix*, e.g.
      temp/image-upload/<job_id>/batches/registration-step/processed/canonical_labels_by_fingerprint/owner-000123/
    Returns sorted JSONL keys under that prefix.
    """
    prefix = owner_prefix.rstrip("/") + "/"
    keys = s3_list_keys(FILE_BUCKET_NAME, prefix)
    out = [k for k in keys if k.endswith(".jsonl")]
    out.sort()
    return out

def _ingest_label_owner_shard(
    *,
    job_id: str,
    user: str,
    event_type: str,
    shard: str,
    owner_prefix: str,
) -> Dict:
    """
    Insert-only ingest for canonical label tables from owner shard prefix.
    """
    label_keys = _list_owner_label_jsonl_keys(owner_prefix)
    if not label_keys:
        # legal: owner shard might be empty depending on data distribution
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Owner shard={shard} empty (no jsonl parts)")
        return {"job_id": job_id, "shard": shard, "kind": "label_owner", "label_parts": 0, "label_rows_ingested": 0}

    per_table: Dict[str, List[Dict]] = {t: [] for t in LABEL_TABLES}
    attempted_to_ingest = 0

    for row in _iter_jsonl_keys(label_keys, f"{TASK_NAME}.read_owner_labels"):
        table = row.get("__table")
        if not isinstance(table, str) or not table.strip():
            raise RuntimeError(f"{TASK_NAME} owner label row missing __table: {row}")
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
            attempted_to_ingest += len(chunk)
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
        attempted_to_ingest += len(chunk)

    log(job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Done kind=label_owner shard={shard} parts={len(label_keys)} attempted_to_ingest={attempted_to_ingest}")

    return {"job_id": job_id, "shard": shard, "kind": "label_owner", "label_parts": len(label_keys), "attempted_to_ingest": attempted_to_ingest}

def _ingest_target_shard(
    *,
    job_id: str,
    user: str,
    event_type: str,
    label_type: str,
    shard: str,
    upload_staging_key: str,
    canonical_imagery_key: str,
    image_labels_key: str,
) -> Dict:
    """
    Target shard ingest:
      - upload_staging: delete-then-insert
      - canonical_imagery: delete-then-insert
      - image_labels: insert-only (where-not-exists)
      - sets registration_status enriched/no_op for external_duplicate rows based on whether any candidate-new image_labels keys are detected before insert
    """
    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Start kind=target shard={shard}")

    # 0) Load image_labels rows and compute "existing" set for enriched/no_op
    incoming_image_labels: List[Dict[str, str]] = []
    incoming_keys: Set[Tuple[str, str, str]] = set()
    incoming_image_ids: Set[str] = set()
    incoming_label_types: Set[str] = set()

    for r in _iter_jsonl_keys([image_labels_key], f"{TASK_NAME}.read_image_labels"):
        iid = r.get("image_id")
        lt = r.get("label_type")
        lid = r.get("label_id")
        if not (
            isinstance(iid, str) and iid.strip()
            and isinstance(lt, str) and lt.strip()
            and isinstance(lid, str) and lid.strip()
        ):
            continue
        key = (iid.strip(), lt.strip(), lid.strip())
        if key in incoming_keys:
            continue
        incoming_keys.add(key)
        incoming_image_labels.append({"image_id": key[0], "label_type": key[1], "label_id": key[2]})
        incoming_image_ids.add(key[0])
        incoming_label_types.add(key[1])

    existing_keys: Set[Tuple[str, str, str]] = set()
    if incoming_image_ids and incoming_label_types:
        existing_keys = fetch_existing_image_label_keys(list(incoming_image_ids), list(incoming_label_types))

    new_keys = incoming_keys - existing_keys

    # 1) Load upload_staging rows, compute external_dup status using new_keys
    upload_rows: List[Dict] = []
    enriched_count = 0
    noop_count = 0
    failed_count = 0

    for row in _iter_jsonl_keys([upload_staging_key], f"{TASK_NAME}.read_upload_staging"):
        dstat = row.get("dedup_status")
        rstat = row.get("registration_status")

        if dstat == "external_duplicate" and rstat not in ("failed", "enriched", "no_op"):
            matched = row.get("matched_image_id")
            if isinstance(matched, str) and matched.strip():
                target_image_id = matched.strip()
            else:
                row["registration_status"] = "failed"
                row["registration_error"] = "missing matched_image_id for external_duplicate"
                failed_count += 1
                upload_rows.append(row)
                continue

            row_new = False
            if label_type in ("single-label", "multi-label"):
                usable_labels = [
                    lab.strip().lower()
                    for lab in (row.get("string_labels") or [])
                    if isinstance(lab, str) and lab.strip()
                ]

                if not usable_labels:
                    row["registration_status"] = "failed"
                    row["registration_error"] = "missing usable string_labels for external_duplicate"
                    failed_count += 1
                    upload_rows.append(row)
                    continue

                for lab in usable_labels:
                    k = (target_image_id, "string-label", lab)
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

    # upload_staging (delete-then-insert)
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

    # 2) canonical_imagery (delete-then-insert; may be empty)
    canon_img_rows: List[Dict] = []
    for row in _iter_jsonl_keys([canonical_imagery_key], f"{TASK_NAME}.read_canonical_imagery"):
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

    # 3) image_labels (insert-only)
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

    log(job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Done kind=target shard={shard}: "
            f"upload_rows={len(upload_rows)} "
            f"canon_imagery_rows={len(canon_img_rows)} "
            f"image_labels_attempted={len(incoming_image_labels)} "
            f"image_labels_candidate_new={len(new_keys)} "
            f"external_dup_enriched={enriched_count} "
            f"external_dup_no_op={noop_count} "
            f"external_dup_failed={failed_count}"
        ))

    return {
        "job_id": job_id,
        "shard": shard,
        "kind": "target",
        "upload_rows": len(upload_rows),
        "canonical_imagery_rows": len(canon_img_rows),
        "image_labels_attempted": len(incoming_image_labels),
        "image_labels_candidate_new": len(new_keys),
        "external_dup_enriched": enriched_count,
        "external_dup_no_op": noop_count,
        "external_dup_failed": failed_count,
    }

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        label_type = event.get("label_type")  # present for target shards (and at top-level job input)
        shard = event["shard"]
        kind = event["kind"]
        upload_staging_key = event.get("upload_staging_key")
        canonical_imagery_key = event.get("canonical_imagery_key")
        canonical_labels_key = event.get("canonical_labels_key")  # for owner shards this is a prefix
        image_labels_key = event.get("image_labels_key")
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} missing job_id")
    if not shard:
        raise RuntimeError(f"{TASK_NAME} missing shard")

    # Dispatch by kind
    if kind == "label_owner":
        if not isinstance(canonical_labels_key, str) or not canonical_labels_key.strip():
            raise RuntimeError(f"{TASK_NAME} kind=label_owner requires canonical_labels_key (owner prefix)")
        return _ingest_label_owner_shard(
            job_id=job_id,
            user=user,
            event_type=event_type,
            shard=shard,
            owner_prefix=canonical_labels_key,
        )
    elif kind != "target":
        raise RuntimeError(f"{TASK_NAME} unsupported kind={kind}")

    # default: kind=target
    if not isinstance(label_type, str) or not label_type.strip():
        raise RuntimeError(f"{TASK_NAME} kind=target requires label_type")
    if not upload_staging_key:
        raise RuntimeError(f"{TASK_NAME} kind=target missing upload_staging_key")
    if not canonical_imagery_key:
        raise RuntimeError(f"{TASK_NAME} kind=target missing canonical_imagery_key")
    if not image_labels_key:
        raise RuntimeError(f"{TASK_NAME} kind=target missing image_labels_key")

    # canonical_labels_key should be None for target shards now (don’t require it)
    return _ingest_target_shard(
        job_id=job_id,
        user=user,
        event_type=event_type,
        label_type=label_type.strip(),
        shard=shard,
        upload_staging_key=upload_staging_key,
        canonical_imagery_key=canonical_imagery_key,
        image_labels_key=image_labels_key
    )