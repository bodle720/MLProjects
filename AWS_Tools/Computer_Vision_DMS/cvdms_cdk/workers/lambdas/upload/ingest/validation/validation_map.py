#!/usr/bin/env python3
import os
import json
from typing import Dict, List

from common.logging_utils import log
from common.iceberg_utils import chunked_insert
from common.s3_utils import s3_read_jsonl_list

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

CHUNK_SIZE = 200

def handler(event, context):
    """
    Expected Step Functions per-item input (from item_selector):
      {
        job_id, user, event_type, label_type, data_source,
        shard, rows_read,
        upload_staging_key,
        imagery_key (None),
        labels_key (None)
      }
    """
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        shard = event["shard"]
        upload_staging_key = event["upload_staging_key"]
        rows_read = event.get("rows_read")
    except KeyError as e:
        raise RuntimeError(f"[VAL_INGEST_MAP] Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError("[VAL_INGEST_MAP] missing job_id")
    if not shard:
        raise RuntimeError("[VAL_INGEST_MAP] missing shard")
    if not upload_staging_key:
        raise RuntimeError("[VAL_INGEST_MAP] missing upload_staging_key")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"[VAL_INGEST_MAP] Start shard={shard} upload_staging_key=s3://{FILE_BUCKET_NAME}/{upload_staging_key} rows_read={rows_read}"
    )

    inserted_rows = 0
    try:
        # Stream rows from this shard’s processed jsonl (already validated/augmented)
        rows_iter = s3_read_jsonl_list(FILE_BUCKET_NAME, [upload_staging_key], "[VAL_INGEST_MAP]")

        chunk: List[Dict] = []
        for r in rows_iter:
            chunk.append(r)
            if len(chunk) >= CHUNK_SIZE:
                ok, err = chunked_insert(
                    chunk,
                    "[VAL_INGEST_MAP]",
                    ICEBERG_DATABASE_NAME,
                    UPLOAD_STAGING_TABLE_NAME,
                    ATHENA_WORKGROUP,
                    ATHENA_OUTPUT_S3,
                    chunk_size=CHUNK_SIZE,
                )
                if not ok:
                    raise RuntimeError(
                        f"[VAL_INGEST_MAP] chunked_insert failed shard={shard}; err={err}"
                    )
                inserted_rows += len(chunk)
                chunk = []

        # flush last
        if chunk:
            ok, err = chunked_insert(
                chunk,
                "[VAL_INGEST_MAP]",
                ICEBERG_DATABASE_NAME,
                UPLOAD_STAGING_TABLE_NAME,
                ATHENA_WORKGROUP,
                ATHENA_OUTPUT_S3,
                chunk_size=CHUNK_SIZE,
            )
            if not ok:
                raise RuntimeError(
                    f"[VAL_INGEST_MAP] chunked_insert failed shard={shard}; err={err}"
                )
            inserted_rows += len(chunk)

    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"[VAL_INGEST_MAP] Failed shard={shard}: {e}",
            level="error"
        )
        raise

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"[VAL_INGEST_MAP] Done shard={shard} inserted_rows={inserted_rows}"
    )

    if rows_read is not None and inserted_rows != int(float(rows_read)):
        raise RuntimeError(f"[VAL_INGEST_MAP] inserted_rows={inserted_rows} != rows_read={rows_read}")

    return {
        "job_id": job_id,
        "shard": shard,
        "inserted_rows": inserted_rows,
        "upload_staging_key": upload_staging_key,
    }