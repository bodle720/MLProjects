#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, List

import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timezone

from common.utils import log, chunked_insert
from common.ingest import iter_rows_from_jsonl_keys

dynamodb = boto3.client("dynamodb")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Env
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
SHA256_TABLE_NAME = os.environ["SHA256_TABLE_NAME"]

UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
CANONICAL_IMAGERY_TABLE_NAME = os.environ.get("CANONICAL_IMAGERY_TABLE_NAME", "canonical_imagery")
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# Label table names must match writer’s __table routing
CANONICAL_BBOX_TABLE = "canonical_bounding_boxes"
CANONICAL_SEMANTIC_TABLE = "canonical_semantic_masks"
CANONICAL_INSTANCE_TABLE = "canonical_instance_annotations"
LABEL_TABLES: List[str] = [CANONICAL_BBOX_TABLE, CANONICAL_SEMANTIC_TABLE, CANONICAL_INSTANCE_TABLE]
CHUNK_SIZE = 200

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _put_sha256_mapping(sha: str, image_id: str, job_id: str, data_source: str | None) -> None:
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
                "created_at": {"S": _utc_now_iso()},
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
            f"[REG_INGEST_MAP] sha256 already registered: sha={sha} "
            f"existing_image_id={existing_image_id} new_image_id={image_id} "
            f"existing_job_id={existing_job_id} new_job_id={job_id}"
        )

def _register_sha256_for_canonical_rows(rows: List[Dict], job_id: str, data_source: str | None) -> int:
    """
    rows are canonical_imagery rows; expects keys sha256_hash + image_id.
    Returns number of mapping attempts (successful puts or verified replays).
    """
    n = 0
    for r in rows:
        sha = r.get("sha256_hash")
        image_id = r.get("image_id")
        if isinstance(sha, str) and sha.strip() and isinstance(image_id, str) and image_id.strip():
            _put_sha256_mapping(sha.strip(), image_id.strip(), job_id, data_source)
            n += 1
    return n

def _flush_chunk(rows: List[Dict], table_name: str, task_name: str) -> None:
    all_failed, last_error = chunked_insert(
        rows,
        ICEBERG_DATABASE_NAME,
        table_name,
        ATHENA_WORKGROUP,
        ATHENA_OUTPUT_S3,
        chunk_size=CHUNK_SIZE,
    )
    if all_failed or last_error:
        raise RuntimeError(
            f"[{task_name}] chunked_insert failures table={table_name}; "
            f"all_failed={all_failed}, last_error={last_error}"
        )

def handler(event, context):
    """
    Map item input (from your IngestStage Map item_selector):
      {
        job_id, user, event_type, label_type, data_source,
        shard, rows_read,
        upload_key, imagery_key, labels_key
      }
    """
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        data_source = event["data_source"]
        shard = event["shard"]
        upload_key = event["upload_key"]
        imagery_key = event.get("imagery_key")
        labels_key = event.get("labels_key")
    except KeyError as e:
        raise RuntimeError(f"[REG_INGEST_MAP] Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError("[REG_INGEST_MAP] missing job_id")
    if not shard:
        raise RuntimeError("[REG_INGEST_MAP] missing shard")
    if not upload_key or not imagery_key or not labels_key:
        raise RuntimeError(
            f"[REG_INGEST_MAP] missing shard keys (upload/imagery/labels). "
            f"upload_key={upload_key}, imagery_key={imagery_key}, labels_key={labels_key}"
        )

    log(job_id, user, event_type, f"[REG_INGEST_MAP] Ingesting shard={shard}", LOG_FIREHOSE_STREAM_NAME)

    inserted_upload = 0
    inserted_canon_imagery = 0
    inserted_labels_total = 0
    inserted_labels_by_table: Dict[str, int] = {t: 0 for t in LABEL_TABLES}
    chunk_size = 200

    # 1) upload_staging
    try:
        chunk: List[Dict] = []
        for row in iter_rows_from_jsonl_keys(FILE_BUCKET_NAME, [upload_key]):
            chunk.append(row)
            if len(chunk) >= chunk_size:
                _flush_chunk(chunk, UPLOAD_STAGING_TABLE_NAME, "REG_INGEST_MAP.upload_staging")
                inserted_upload += len(chunk)
                chunk = []
        if chunk:
            _flush_chunk(chunk, UPLOAD_STAGING_TABLE_NAME, "REG_INGEST_MAP.upload_staging")
            inserted_upload += len(chunk)
    except Exception as e:
        log(job_id, user, event_type, f"[REG_INGEST_MAP] upload_staging insert failed shard={shard}: {e}",
            LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    # 2) canonical_imagery
    try:
        chunk = []
        sha_registered = 0

        for row in iter_rows_from_jsonl_keys(FILE_BUCKET_NAME, [imagery_key]):
            chunk.append(row)
            if len(chunk) >= chunk_size:
                _flush_chunk(chunk, CANONICAL_IMAGERY_TABLE_NAME, "REG_INGEST_MAP.canonical_imagery")
                inserted_canon_imagery += len(chunk)
                sha_registered += _register_sha256_for_canonical_rows(chunk, job_id, data_source)
                chunk = []
        if chunk:
            _flush_chunk(chunk, CANONICAL_IMAGERY_TABLE_NAME, "REG_INGEST_MAP.canonical_imagery")
            inserted_canon_imagery += len(chunk)
            sha_registered += _register_sha256_for_canonical_rows(chunk, job_id, data_source)
    except Exception as e:
        log(job_id, user, event_type, f"[REG_INGEST_MAP] canonical_imagery insert failed shard={shard}: {e}",
            LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    # 3) canonical labels routed by __table (streaming per-table)
    try:
        per_table_chunks: Dict[str, List[Dict]] = {t: [] for t in LABEL_TABLES}

        for row in iter_rows_from_jsonl_keys(FILE_BUCKET_NAME, [labels_key]):
            table = row.get("__table")
            if not table:
                raise RuntimeError(f"[REG_INGEST_MAP] label row missing __table routing field: {row}")
            if table not in LABEL_TABLES:
                raise RuntimeError(f"[REG_INGEST_MAP] unsupported label table: {table}")

            # strip routing field
            row = dict(row)
            row.pop("__table", None)

            per_table_chunks[table].append(row)

            if len(per_table_chunks[table]) >= chunk_size:
                chunk_to_flush = per_table_chunks[table]
                _flush_chunk(chunk_to_flush, table, f"REG_INGEST_MAP.labels.{table}")
                inserted = len(chunk_to_flush)
                inserted_labels_total += inserted
                inserted_labels_by_table[table] += inserted
                per_table_chunks[table] = []

        # flush remaining
        for table, chunk in per_table_chunks.items():
            if not chunk:
                continue
            _flush_chunk(chunk, table, f"REG_INGEST_MAP.labels.{table}")
            inserted = len(chunk)
            inserted_labels_total += inserted
            inserted_labels_by_table[table] += inserted

    except Exception as e:
        log(job_id, user, event_type, f"[REG_INGEST_MAP] label inserts failed shard={shard}: {e}",
            LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    log(
        job_id,
        user,
        event_type,
        f"[REG_INGEST_MAP] Done shard={shard}: "
        f"upload_inserted={inserted_upload}, canon_imagery_inserted={inserted_canon_imagery}, "
        f"labels_inserted={inserted_labels_total}, labels_by_table={inserted_labels_by_table}",
        LOG_FIREHOSE_STREAM_NAME,
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