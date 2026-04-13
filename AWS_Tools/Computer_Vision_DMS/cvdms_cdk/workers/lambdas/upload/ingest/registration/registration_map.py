#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Tuple, Iterable, Set, Optional
import boto3

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import (
    s3_read_jsonl_list,
    s3_list_keys,
    write_s3_obj,
)
from common.general_utils.athena_utils import run_athena
from common.general_utils.iceberg_utils import chunked_insert, chunked_insert_where_not_exists
from common.general_utils.table_schemas import (
    UPLOAD_STAGING_TABLE_NAME,
    CANONICAL_IMAGERY_TABLE_NAME,
    IMAGE_LABELS_TABLE_NAME,
    IMAGE_SOURCE_MEMBERSHIP_TABLE_NAME,
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
        yield lst[i : i + n]

def _athena_fetch_rows_3col(qid: str, *, allow_empty_third: bool = False) -> List[Tuple[str, str, Optional[str]]]:
    """
    Fetch all rows from a SELECT of exactly 3 columns in order.
    Returns tuples of (col1, col2, col3_or_none).
    """
    out: List[Tuple[str, str, Optional[str]]] = []
    paginator = athena.get_paginator("get_query_results")

    first = True
    for page in paginator.paginate(QueryExecutionId=qid):
        rows = page["ResultSet"]["Rows"]
        for r in rows:
            if first:
                first = False
                continue  # header

            data = r.get("Data", [])
            if len(data) < 2:
                continue

            a = (data[0].get("VarCharValue") or "").strip()
            b = (data[1].get("VarCharValue") or "").strip()
            c_raw = data[2].get("VarCharValue") if len(data) >= 3 else None
            c = c_raw.strip() if isinstance(c_raw, str) else None

            if not a or not b:
                continue
            if (not allow_empty_third) and (not c):
                continue

            out.append((a, b, c if c else None))

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
        for iid, lt, lid in _athena_fetch_rows_3col(qid, allow_empty_third=False):
            existing.add((iid, lt, lid or ""))

    return existing

def fetch_existing_image_source_memberships(
    image_ids: List[str],
    data_sources: List[str],
) -> Dict[Tuple[str, str], Optional[str]]:
    """
    Returns a mapping:
      (image_id, data_source) -> source_split_or_none
    """
    if not image_ids or not data_sources:
        return {}

    full_table = f"\"{ICEBERG_DATABASE_NAME}\".\"{IMAGE_SOURCE_MEMBERSHIP_TABLE_NAME}\""

    ds_in = ", ".join("'" + _escape_sql_string(x) + "'" for x in sorted(set(data_sources)))
    existing: Dict[Tuple[str, str], Optional[str]] = {}

    for chunk in _chunks(sorted(set(image_ids)), 200):
        iid_in = ", ".join("'" + _escape_sql_string(x) + "'" for x in chunk)
        sql = f"""
        SELECT image_id, data_source, source_split
        FROM {full_table}
        WHERE data_source IN ({ds_in})
          AND image_id IN ({iid_in})
        """

        qid, _ = run_athena(
            sql,
            f"{TASK_NAME} fetch_existing_image_source_memberships",
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
            poll=2.0,
            timeout=300,
        )
        for iid, ds, split in _athena_fetch_rows_3col(qid, allow_empty_third=True):
            existing[(iid, ds)] = split

    return existing

def _iter_jsonl_keys(keys: List[str], task: str) -> Iterable[Dict]:
    return s3_read_jsonl_list(FILE_BUCKET_NAME, keys, task)

def _list_owner_label_jsonl_keys(owner_prefix: str) -> List[str]:
    """
    owner_prefix is an S3 prefix like:
      temp/image-upload/<job_id>/batches/registration-step/processed/canonical_labels_by_fingerprint/owner-000123/
    """
    prefix = owner_prefix.rstrip("/") + "/"
    keys = s3_list_keys(FILE_BUCKET_NAME, prefix)
    out = [k for k in keys if k.endswith(".jsonl")]
    out.sort()
    return out

def _normalize_source_split(value: object) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"{TASK_NAME} source_split must be str or None, got {type(value).__name__}")

    s = value.strip().lower()
    if s == "":
        return None
    if s not in {"train", "val", "test"}:
        raise RuntimeError(f"{TASK_NAME} invalid source_split={value!r}")
    return s

def _rollback_plan_key(job_id: str, shard: str) -> str:
    return f"temp/image-upload/{job_id}/batches/registration-step/processed/rollback/target-shard-{shard}.json"

def _write_target_rollback_plan(
    *,
    job_id: str,
    shard: str,
    image_label_keys_to_delete: Set[Tuple[str, str, str]],
    image_source_memberships_to_delete: List[Dict[str, Optional[str]]],
) -> str:
    """
    Persist the exact candidate-new keys before insert-only writes, so DLQ can
    safely delete them later if this job fails after partial registration ingest.
    """
    payload = {
        "job_id": job_id,
        "shard": shard,
        "kind": "target",
        "image_labels_to_delete": [
            {"image_id": iid, "label_type": lt, "label_id": lid}
            for (iid, lt, lid) in sorted(image_label_keys_to_delete)
        ],
        "image_source_memberships_to_delete": sorted(
            image_source_memberships_to_delete,
            key=lambda r: (r["image_id"], r["data_source"]),
        ),
    }

    key = _rollback_plan_key(job_id, shard)
    write_s3_obj(
        FILE_BUCKET_NAME,
        key,
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n",
        "application/json",
        TASK_NAME,
    )
    return key

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
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Owner shard={shard} empty (no jsonl parts)")
        return {
            "job_id": job_id,
            "shard": shard,
            "kind": "label_owner",
            "label_parts": 0,
            "label_rows_ingested": 0,
        }

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

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Done kind=label_owner shard={shard} parts={len(label_keys)} attempted_to_ingest={attempted_to_ingest}",
    )

    return {
        "job_id": job_id,
        "shard": shard,
        "kind": "label_owner",
        "label_parts": len(label_keys),
        "attempted_to_ingest": attempted_to_ingest,
    }

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
    image_source_membership_key: str,
) -> Dict:
    """
    Target shard ingest:
      - upload_staging: delete-then-insert
      - canonical_imagery: delete-then-insert
      - image_labels: insert-only
      - image_source_membership: insert-only
      - sets registration_status enriched/no_op for external_duplicate rows based on whether
        any candidate-new image_labels OR source-membership rows are detected before insert
      - writes a rollback plan with exact candidate-new keys before insert-only writes
    """
    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Start kind=target shard={shard}")

    # 0a) Load image_labels rows and compute existing/new
    incoming_image_labels: List[Dict[str, str]] = []
    incoming_label_keys: Set[Tuple[str, str, str]] = set()
    incoming_image_ids_for_labels: Set[str] = set()
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
        if key in incoming_label_keys:
            continue

        incoming_label_keys.add(key)
        incoming_image_labels.append(
            {"image_id": key[0], "label_type": key[1], "label_id": key[2]}
        )
        incoming_image_ids_for_labels.add(key[0])
        incoming_label_types.add(key[1])

    existing_label_keys: Set[Tuple[str, str, str]] = set()
    if incoming_image_ids_for_labels and incoming_label_types:
        existing_label_keys = fetch_existing_image_label_keys(
            list(incoming_image_ids_for_labels),
            list(incoming_label_types),
        )

    new_label_keys = incoming_label_keys - existing_label_keys

    # 0b) Load image_source_membership rows and compute existing/new/conflicts
    incoming_source_memberships: List[Dict[str, Optional[str]]] = []
    incoming_source_key_to_split: Dict[Tuple[str, str], Optional[str]] = {}
    incoming_image_ids_for_sources: Set[str] = set()
    incoming_data_sources: Set[str] = set()

    for r in _iter_jsonl_keys([image_source_membership_key], f"{TASK_NAME}.read_image_source_membership"):
        iid = r.get("image_id")
        ds = r.get("data_source")
        ss = _normalize_source_split(r.get("source_split"))

        if not (isinstance(iid, str) and iid.strip() and isinstance(ds, str) and ds.strip()):
            continue

        key = (iid.strip(), ds.strip())
        if key in incoming_source_key_to_split:
            prev_split = incoming_source_key_to_split[key]
            if prev_split != ss:
                raise RuntimeError(
                    f"{TASK_NAME} conflicting incoming source_split for {key}: {prev_split} vs {ss}"
                )
            continue

        incoming_source_key_to_split[key] = ss
        incoming_source_memberships.append(
            {"image_id": key[0], "data_source": key[1], "source_split": ss}
        )
        incoming_image_ids_for_sources.add(key[0])
        incoming_data_sources.add(key[1])

    existing_source_memberships: Dict[Tuple[str, str], Optional[str]] = {}
    if incoming_image_ids_for_sources and incoming_data_sources:
        existing_source_memberships = fetch_existing_image_source_memberships(
            list(incoming_image_ids_for_sources),
            list(incoming_data_sources),
        )

    new_source_membership_keys: Set[Tuple[str, str]] = set()
    conflicting_source_membership_keys: Dict[Tuple[str, str], Tuple[Optional[str], Optional[str]]] = {}

    for key, incoming_split in incoming_source_key_to_split.items():
        if key not in existing_source_memberships:
            new_source_membership_keys.add(key)
            continue

        existing_split = existing_source_memberships[key]
        if existing_split != incoming_split:
            conflicting_source_membership_keys[key] = (existing_split, incoming_split)

    # Candidate-new source membership rows for rollback + insert
    memberships_to_insert = [
        r for r in incoming_source_memberships
        if (r["image_id"], r["data_source"]) in new_source_membership_keys
        and (r["image_id"], r["data_source"]) not in conflicting_source_membership_keys
    ]

    # Persist rollback plan BEFORE insert-only writes.
    rollback_plan_key = _write_target_rollback_plan(
        job_id=job_id,
        shard=shard,
        image_label_keys_to_delete=new_label_keys,
        image_source_memberships_to_delete=memberships_to_insert,
    )

    # 1) Load upload_staging rows, compute external_duplicate status using new labels + new source memberships
    upload_rows: List[Dict] = []
    enriched_count = 0
    noop_count = 0
    failed_count = 0
    source_membership_conflict_count = 0

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

            row_data_source = row.get("data_source")
            if not (isinstance(row_data_source, str) and row_data_source.strip()):
                row["registration_status"] = "failed"
                row["registration_error"] = "missing data_source for external_duplicate"
                failed_count += 1
                upload_rows.append(row)
                continue

            membership_key = (target_image_id, row_data_source.strip())

            if membership_key in conflicting_source_membership_keys:
                existing_split, incoming_split = conflicting_source_membership_keys[membership_key]
                row["registration_status"] = "failed"
                row["registration_error"] = (
                    f"conflicting source_split for existing image_source_membership "
                    f"{membership_key}: existing={existing_split!r}, incoming={incoming_split!r}"
                )
                failed_count += 1
                source_membership_conflict_count += 1
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
                    if k in new_label_keys:
                        row_new = True
                        break
            else:
                fp = row.get("label_fingerprint")
                if isinstance(fp, str) and fp.strip():
                    k = (target_image_id, label_type, fp.strip())
                    if k in new_label_keys:
                        row_new = True
                else:
                    row["registration_status"] = "failed"
                    row["registration_error"] = "missing label_fingerprint for structured external_duplicate"
                    failed_count += 1
                    upload_rows.append(row)
                    continue

            if membership_key in new_source_membership_keys:
                row_new = True

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

    # 2) canonical_imagery (delete-then-insert)
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

    # 4) image_source_membership (insert-only, excluding conflicts)
    if memberships_to_insert:
        ok, err = chunked_insert_where_not_exists(
            memberships_to_insert,
            f"{TASK_NAME}.image_source_membership",
            ICEBERG_DATABASE_NAME,
            IMAGE_SOURCE_MEMBERSHIP_TABLE_NAME,
            ATHENA_WORKGROUP,
            ATHENA_OUTPUT_S3,
            chunk_size=CHUNK_SIZE,
        )
        if not ok:
            raise RuntimeError(f"{TASK_NAME} image_source_membership insert-only failed: {err}")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Done kind=target shard={shard}: "
            f"upload_rows={len(upload_rows)} "
            f"canon_imagery_rows={len(canon_img_rows)} "
            f"image_labels_attempted={len(incoming_image_labels)} "
            f"image_labels_candidate_new={len(new_label_keys)} "
            f"image_source_membership_attempted={len(incoming_source_memberships)} "
            f"image_source_membership_candidate_new={len(new_source_membership_keys)} "
            f"image_source_membership_conflicts={len(conflicting_source_membership_keys)} "
            f"external_dup_enriched={enriched_count} "
            f"external_dup_no_op={noop_count} "
            f"external_dup_failed={failed_count} "
            f"rollback_plan_key=s3://{FILE_BUCKET_NAME}/{rollback_plan_key}"
        ),
    )

    return {
        "job_id": job_id,
        "shard": shard,
        "kind": "target",
        "upload_rows": len(upload_rows),
        "canonical_imagery_rows": len(canon_img_rows),
        "image_labels_attempted": len(incoming_image_labels),
        "image_labels_candidate_new": len(new_label_keys),
        "image_source_membership_attempted": len(incoming_source_memberships),
        "image_source_membership_candidate_new": len(new_source_membership_keys),
        "image_source_membership_conflicts": len(conflicting_source_membership_keys),
        "external_dup_enriched": enriched_count,
        "external_dup_no_op": noop_count,
        "external_dup_failed": failed_count,
        "source_membership_conflict_rows": source_membership_conflict_count,
        "rollback_plan_key": rollback_plan_key,
    }

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        label_type = event.get("label_type")
        shard = event["shard"]
        kind = event["kind"]
        upload_staging_key = event.get("upload_staging_key")
        canonical_imagery_key = event.get("canonical_imagery_key")
        canonical_labels_key = event.get("canonical_labels_key")
        image_labels_key = event.get("image_labels_key")
        image_source_membership_key = event.get("image_source_membership_key")
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} missing job_id")
    if not shard:
        raise RuntimeError(f"{TASK_NAME} missing shard")

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

    if not isinstance(label_type, str) or not label_type.strip():
        raise RuntimeError(f"{TASK_NAME} kind=target requires label_type")
    if not upload_staging_key:
        raise RuntimeError(f"{TASK_NAME} kind=target missing upload_staging_key")
    if not canonical_imagery_key:
        raise RuntimeError(f"{TASK_NAME} kind=target missing canonical_imagery_key")
    if not image_labels_key:
        raise RuntimeError(f"{TASK_NAME} kind=target missing image_labels_key")
    if not image_source_membership_key:
        raise RuntimeError(f"{TASK_NAME} kind=target missing image_source_membership_key")

    return _ingest_target_shard(
        job_id=job_id,
        user=user,
        event_type=event_type,
        label_type=label_type.strip(),
        shard=shard,
        upload_staging_key=upload_staging_key,
        canonical_imagery_key=canonical_imagery_key,
        image_labels_key=image_labels_key,
        image_source_membership_key=image_source_membership_key,
    )