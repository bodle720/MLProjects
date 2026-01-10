#!/usr/bin/env python3
import os
import json
from typing import Dict, List

from common.logging_utils import log
from common.s3_utils import s3_read_jsonl_list
from common.iceberg_utils import chunked_insert
from common.table_schemas import UPLOAD_STAGING_TABLE_NAME

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[DEDUP_INGEST_MAP]"
CHUNK_SIZE = 200

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        shard = event["shard"]
        rows_read = event["rows_read"]
        upload_staging_key = event["upload_staging_key"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} missing job_id")
    if not shard:
        raise RuntimeError(f"{TASK_NAME} missing shard")
    if not upload_staging_key:
        raise RuntimeError(f"{TASK_NAME} missing upload_staging_key")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Start shard={shard} upload_staging_key=s3://{FILE_BUCKET_NAME}/{upload_staging_key} rows_read={rows_read}"
    )

    inserted_rows = 0
    try:
        # Stream rows from this shard’s processed jsonl
        rows_iter = s3_read_jsonl_list(FILE_BUCKET_NAME, [upload_staging_key], TASK_NAME)

        chunk: List[Dict] = []
        for r in rows_iter:
            chunk.append(r)
            if len(chunk) >= CHUNK_SIZE:
                ok, err = chunked_insert(
                                        chunk,
                                        TASK_NAME,
                                        ICEBERG_DATABASE_NAME,
                                        UPLOAD_STAGING_TABLE_NAME,
                                        ATHENA_WORKGROUP,
                                        ATHENA_OUTPUT_S3,
                                        chunk_size=CHUNK_SIZE,
                                    )
                if not ok:
                    raise RuntimeError(
                        f"{TASK_NAME} chunked_insert failures shard={shard}; "
                        f"error={err}"
                    )
                inserted_rows += len(chunk)
                chunk = []

        # flush last
        if chunk:
            ok, err = chunked_insert(
                chunk,
                TASK_NAME,
                ICEBERG_DATABASE_NAME,
                UPLOAD_STAGING_TABLE_NAME,
                ATHENA_WORKGROUP,
                ATHENA_OUTPUT_S3,
                chunk_size=CHUNK_SIZE,
            )
            if not ok:
                raise RuntimeError(
                    f"{TASK_NAME} chunked_insert failures shard={shard}; "
                    f"error={err}"
                )
            inserted_rows += len(chunk)

    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Failed shard={shard}: {e}",
            level="error"
        )
        raise

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Done shard={shard} inserted_rows={inserted_rows}"
    )

    if rows_read is not None and int(inserted_rows) != int(rows_read):
        raise RuntimeError(f"{TASK_NAME} inserted_rows={inserted_rows} != rows_read={rows_read}")

    # We discard Map results in the state machine, but returning is still useful for CloudWatch debugging.
    return {
        "job_id": job_id,
        "shard": shard,
        "inserted_rows": inserted_rows,
        "upload_staging_key": upload_staging_key,
    }