#!/usr/bin/env python3
import os
import json
from typing import Dict, List

import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timezone

from common.logging_utils import log
from common.s3_utils import s3_read_jsonl_list
from common.iceberg_utils import chunked_insert
from common.table_schemas import CANONICAL_IMAGERY_TABLE_NAME, CANONICAL_BBOX_TABLE_NAME, \
                                 CANONICAL_SEMANTIC_TABLE_NAME, CANONICAL_INSTANCE_TABLE_NAME, \
                                 UPLOAD_STAGING_TABLE_NAME

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
SHA256_TABLE_NAME = os.environ["SHA256_TABLE_NAME"]

LABEL_TABLES: List[str] = [CANONICAL_BBOX_TABLE_NAME, CANONICAL_SEMANTIC_TABLE_NAME, CANONICAL_INSTANCE_TABLE_NAME]
TASK_NAME = "[REG_INGEST_MAP]"
CHUNK_SIZE = 200

dynamodb = boto3.client("dynamodb")

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def put_sha256_mapping(sha: str, image_id: str, job_id: str, data_source: str | None) -> None:
    """
    Create sha256 -> canonical image_id mapping iff sha256 not already present.
    Idempotent:
      - If already present with same (image_id, job_id) => ok (replay)
      - If already present with different image_id => conflict => raise
    """
    if not sha or not image_id:
        return

    try:
        dynamodb.put_item(
            TableName=SHA256_TABLE_NAME,
            Item={
                "sha256": {"S": sha},
                "image_id": {"S": image_id},
                "job_id": {"S": job_id},
                "data_source": {"S": (data_source or "")},
                "created_at": {"S": utc_now_iso()},
            },
            ConditionExpression="attribute_not_exists(#k)",
            ExpressionAttributeNames={"#k": "sha256"},
        )
        return

    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code != "ConditionalCheckFailedException":
            raise

        # Exists already: must verify it's the same mapping (idempotent replay),
        # otherwise someone else already registered this sha (external dup).
        resp = dynamodb.get_item(
            TableName=SHA256_TABLE_NAME,
            Key={"sha256": {"S": sha}},
            ConsistentRead=True,
        )
        item = resp.get("Item") or {}
        existing_image_id = item.get("image_id", {}).get("S")
        existing_job_id = item.get("job_id", {}).get("S")

        if existing_image_id == image_id and existing_job_id == job_id:
            return  # replay-safe

        # If it exists with different image_id, that means it's truly an external duplicate
        # (or a logic bug if you expected to be creating it).
        raise RuntimeError(
            f"{TASK_NAME} sha256 already registered: sha={sha} "
            f"existing_image_id={existing_image_id} new_image_id={image_id} "
            f"existing_job_id={existing_job_id} new_job_id={job_id}"
        )

def register_sha256_for_canonical_rows(rows: List[Dict], job_id: str, data_source: str | None) -> int:
    """
    rows are canonical_imagery rows; expects keys sha256_hash + image_id.
    Returns number of mapping attempts (successful puts or verified replays).
    """
    n = 0
    for r in rows:
        sha = r.get("sha256_hash")
        image_id = r.get("image_id")
        if isinstance(sha, str) and sha.strip() and isinstance(image_id, str) and image_id.strip():
            put_sha256_mapping(sha.strip(), image_id.strip(), job_id, data_source)
            n += 1
    return n

def flush_chunk(rows: List[Dict], table_name: str, task_name: str) -> None:
    ok, err = chunked_insert(
                            rows,
                            task_name,
                            ICEBERG_DATABASE_NAME,
                            table_name,
                            ATHENA_WORKGROUP,
                            ATHENA_OUTPUT_S3,
                            chunk_size=CHUNK_SIZE,
                        )
    if not ok:
        raise RuntimeError(f"{task_name} chunked insert failures table={table_name}; error={err}")

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        data_source = event["data_source"]
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
    if not upload_staging_key or not canonical_imagery_key or not canonical_labels_key:
        raise RuntimeError(
            f"{TASK_NAME} missing shard keys (upload/imagery/labels). "
            f"upload_staging_key={upload_staging_key}, canonical_imagery_key={canonical_imagery_key}, canonical_labels_key={canonical_labels_key}"
        )

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Ingesting shard={shard}")

    inserted_upload = 0
    inserted_canon_imagery = 0
    inserted_labels_total = 0
    inserted_labels_by_table: Dict[str, int] = {t: 0 for t in LABEL_TABLES}
    chunk_size = 200

    # 1) upload_staging
    try:
        chunk: List[Dict] = []
        for row in s3_read_jsonl_list(FILE_BUCKET_NAME, [upload_staging_key], TASK_NAME):
            chunk.append(row)
            if len(chunk) >= chunk_size:
                flush_chunk(chunk, UPLOAD_STAGING_TABLE_NAME, f"{TASK_NAME}.upload_staging")
                inserted_upload += len(chunk)
                chunk = []
        if chunk:
            flush_chunk(chunk, UPLOAD_STAGING_TABLE_NAME, f"{TASK_NAME}.upload_staging")
            inserted_upload += len(chunk)
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} upload_staging insert failed shard={shard}: {e}", level="error")
        raise

    # 2) canonical_imagery
    try:
        chunk = []
        sha_registered = 0

        for row in s3_read_jsonl_list(FILE_BUCKET_NAME, [canonical_imagery_key], TASK_NAME):
            chunk.append(row)
            if len(chunk) >= chunk_size:
                flush_chunk(chunk, CANONICAL_IMAGERY_TABLE_NAME, f"{TASK_NAME}.canonical_imagery")
                inserted_canon_imagery += len(chunk)
                sha_registered += register_sha256_for_canonical_rows(chunk, job_id, data_source)
                chunk = []
        if chunk:
            flush_chunk(chunk, CANONICAL_IMAGERY_TABLE_NAME, f"{TASK_NAME}.canonical_imagery")
            inserted_canon_imagery += len(chunk)
            sha_registered += register_sha256_for_canonical_rows(chunk, job_id, data_source)
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} canonical_imagery insert failed shard={shard}: {e}", level="error")
        raise

    # 3) canonical labels routed by __table (streaming per-table)
    try:
        per_table_chunks: Dict[str, List[Dict]] = {t: [] for t in LABEL_TABLES}

        for row in s3_read_jsonl_list(FILE_BUCKET_NAME, [canonical_labels_key], TASK_NAME):
            table = row.get("__table")
            if not table:
                raise RuntimeError(f"{TASK_NAME} label row missing __table routing field: {row}")
            if table not in LABEL_TABLES:
                raise RuntimeError(f"{TASK_NAME} unsupported label table: {table}")

            # strip routing field
            row = dict(row)
            row.pop("__table", None)

            per_table_chunks[table].append(row)

            if len(per_table_chunks[table]) >= chunk_size:
                chunk_to_flush = per_table_chunks[table]
                flush_chunk(chunk_to_flush, table, f"{TASK_NAME}.labels.{table}")
                inserted = len(chunk_to_flush)
                inserted_labels_total += inserted
                inserted_labels_by_table[table] += inserted
                per_table_chunks[table] = []

        # flush remaining
        for table, chunk in per_table_chunks.items():
            if not chunk:
                continue
            flush_chunk(chunk, table, f"{TASK_NAME}.labels.{table}")
            inserted = len(chunk)
            inserted_labels_total += inserted
            inserted_labels_by_table[table] += inserted

    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} label inserts failed shard={shard}: {e}", level="error")
        raise

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Done shard={shard}: "
        f"upload_inserted={inserted_upload}, canon_imagery_inserted={inserted_canon_imagery}, "
        f"labels_inserted={inserted_labels_total}, labels_by_table={inserted_labels_by_table}"
    )

    # Return per-shard counts (Map result_path is DISCARD in your construct, but returning is still useful for debugging/tests)
    return {
        "job_id": job_id,
        "shard": shard,
        "upload_rows_inserted": inserted_upload,
        "sha_registered": sha_registered,
        "canonical_imagery_rows_inserted": inserted_canon_imagery,
        "canonical_label_rows_inserted": inserted_labels_total,
        "canonical_label_rows_by_table": inserted_labels_by_table,
    }