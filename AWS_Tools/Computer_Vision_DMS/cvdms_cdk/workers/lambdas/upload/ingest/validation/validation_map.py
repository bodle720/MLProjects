#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, List

from common.utils import log, chunked_insert
from common.ingest import _iter_rows_from_jsonl_keys

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Env
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

CHUNK_SIZE = int(os.environ.get("INGEST_CHUNK_SIZE", "200"))


def handler(event, context):
    """
    Expected Step Functions per-item input (from item_selector):
      {
        job_id, user, event_type, label_type, data_source,
        shard, rows_read,
        upload_key,
        imagery_key (None),
        labels_key (None)
      }
    """
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        shard = event["shard"]
        upload_key = event["upload_key"]
        rows_read = event.get("rows_read")
    except KeyError as e:
        raise RuntimeError(f"[VAL_INGEST_MAP] Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError("[VAL_INGEST_MAP] missing job_id")
    if not shard:
        raise RuntimeError("[VAL_INGEST_MAP] missing shard")
    if not upload_key:
        raise RuntimeError("[VAL_INGEST_MAP] missing upload_key")

    log(
        job_id,
        user,
        event_type,
        f"[VAL_INGEST_MAP] Start shard={shard} upload_key=s3://{FILE_BUCKET_NAME}/{upload_key} rows_read={rows_read}",
        LOG_FIREHOSE_STREAM_NAME,
    )

    inserted_rows = 0
    try:
        # Stream rows from this shard’s processed jsonl (already validated/augmented)
        rows_iter = _iter_rows_from_jsonl_keys(FILE_BUCKET_NAME, [upload_key])

        chunk: List[Dict] = []
        for r in rows_iter:
            chunk.append(r)
            if len(chunk) >= CHUNK_SIZE:
                all_failed, last_error = chunked_insert(
                    chunk,
                    ICEBERG_DATABASE_NAME,
                    UPLOAD_STAGING_TABLE_NAME,
                    ATHENA_WORKGROUP,
                    ATHENA_OUTPUT_S3,
                    chunk_size=CHUNK_SIZE,
                )
                if all_failed or last_error:
                    raise RuntimeError(
                        f"[VAL_INGEST_MAP] chunked_insert failures shard={shard}; "
                        f"all_failed={all_failed}, last_error={last_error}"
                    )
                inserted_rows += len(chunk)
                chunk = []

        # flush last
        if chunk:
            all_failed, last_error = chunked_insert(
                chunk,
                ICEBERG_DATABASE_NAME,
                UPLOAD_STAGING_TABLE_NAME,
                ATHENA_WORKGROUP,
                ATHENA_OUTPUT_S3,
                chunk_size=CHUNK_SIZE,
            )
            if all_failed or last_error:
                raise RuntimeError(
                    f"[VAL_INGEST_MAP] chunked_insert failures shard={shard}; "
                    f"all_failed={all_failed}, last_error={last_error}"
                )
            inserted_rows += len(chunk)

    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            f"[VAL_INGEST_MAP] Failed shard={shard}: {e}",
            LOG_FIREHOSE_STREAM_NAME,
            error=str(e),
            level="error",
        )
        raise

    log(
        job_id,
        user,
        event_type,
        f"[VAL_INGEST_MAP] Done shard={shard} inserted_rows={inserted_rows}",
        LOG_FIREHOSE_STREAM_NAME,
    )

    return {
        "job_id": job_id,
        "shard": shard,
        "inserted_rows": inserted_rows,
        "upload_key": upload_key,
    }
