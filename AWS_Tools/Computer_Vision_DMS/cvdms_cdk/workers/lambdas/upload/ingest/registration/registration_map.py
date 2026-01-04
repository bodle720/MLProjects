#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, List

from common.utils import log, chunked_insert
from common.ingest import iter_rows_from_jsonl_keys

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Env
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]

UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
CANONICAL_IMAGERY_TABLE_NAME = os.environ.get("CANONICAL_IMAGERY_TABLE_NAME", "canonical_imagery")
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# Label table names must match writer’s __table routing
CANONICAL_BBOX_TABLE = "canonical_bounding_boxes"
CANONICAL_SEMANTIC_TABLE = "canonical_semantic_masks"
CANONICAL_INSTANCE_TABLE = "canonical_instance_annotations"
LABEL_TABLES: List[str] = [CANONICAL_BBOX_TABLE, CANONICAL_SEMANTIC_TABLE, CANONICAL_INSTANCE_TABLE]

def _flush_chunk(rows: List[Dict], table_name: str, task_name: str) -> None:
    all_failed, last_error = chunked_insert(
        rows,
        ICEBERG_DATABASE_NAME,
        table_name,
        ATHENA_WORKGROUP,
        ATHENA_OUTPUT_S3,
        chunk_size=len(rows),
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
        for row in iter_rows_from_jsonl_keys(FILE_BUCKET_NAME, [imagery_key]):
            chunk.append(row)
            if len(chunk) >= chunk_size:
                _flush_chunk(chunk, CANONICAL_IMAGERY_TABLE_NAME, "REG_INGEST_MAP.canonical_imagery")
                inserted_canon_imagery += len(chunk)
                chunk = []
        if chunk:
            _flush_chunk(chunk, CANONICAL_IMAGERY_TABLE_NAME, "REG_INGEST_MAP.canonical_imagery")
            inserted_canon_imagery += len(chunk)
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
        "canonical_imagery_rows_inserted": inserted_canon_imagery,
        "canonical_label_rows_inserted": inserted_labels_total,
        "canonical_label_rows_by_table": inserted_labels_by_table,
    }