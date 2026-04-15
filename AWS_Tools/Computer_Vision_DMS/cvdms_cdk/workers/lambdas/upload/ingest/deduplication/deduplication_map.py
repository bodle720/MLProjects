#!/usr/bin/env python3
import os
import json
from datetime import datetime, timezone

import boto3

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import s3_read_jsonl_list
from common.general_utils.iceberg_utils import chunked_insert
from common.general_utils.table_schemas import UPLOAD_STAGING_TABLE_NAME

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[DEDUP_INGEST_MAP]"
STAGE_NAME = "dedup-ingest"
CHUNK_SIZE = 200

s3 = boto3.client("s3")

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _active_marker_key(job_id: str, shard: str) -> str:
    return f"temp/image-upload/{job_id}/worker-markers/{STAGE_NAME}/active/{shard}.json"

def _completed_marker_key(job_id: str, shard: str) -> str:
    return f"temp/image-upload/{job_id}/worker-markers/{STAGE_NAME}/completed/{shard}.json"

def _write_json_marker(key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=FILE_BUCKET_NAME,
        Key=key,
        Body=(json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"),
        ContentType="application/json",
    )

def _delete_marker_best_effort(key: str) -> None:
    try:
        s3.delete_object(Bucket=FILE_BUCKET_NAME, Key=key)
    except Exception:
        pass

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

    active_key = _active_marker_key(job_id, shard)
    completed_key = _completed_marker_key(job_id, shard)

    _write_json_marker(
        active_key,
        {
            "job_id": job_id,
            "stage": STAGE_NAME,
            "shard": shard,
            "request_id": getattr(context, "aws_request_id", None),
            "started_at": _iso_now(),
            "rows_read": rows_read,
            "upload_staging_key": upload_staging_key,
        },
    )

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Start shard={shard} upload_staging_key=s3://{FILE_BUCKET_NAME}/{upload_staging_key} rows_read={rows_read}",
    )

    try:
        rows_iter = s3_read_jsonl_list(FILE_BUCKET_NAME, [upload_staging_key], TASK_NAME)

        ok, err = chunked_insert(
            rows_iter,
            TASK_NAME,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_WORKGROUP,
            ATHENA_OUTPUT_S3,
            chunk_size=CHUNK_SIZE,
        )
        if not ok:
            raise RuntimeError(
                f"{TASK_NAME} chunked_insert failed shard={shard}; err={err}, upload key={upload_staging_key}"
            )

        _write_json_marker(
            completed_key,
            {
                "job_id": job_id,
                "stage": STAGE_NAME,
                "shard": shard,
                "request_id": getattr(context, "aws_request_id", None),
                "completed_at": _iso_now(),
                "rows_read": rows_read,
                "upload_staging_key": upload_staging_key,
            },
        )

    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Failed shard={shard} with upload key={upload_staging_key}: {e}",
            level="error",
        )
        raise

    finally:
        _delete_marker_best_effort(active_key)

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Done shard={shard}")

    return {
        "job_id": job_id,
        "shard": shard,
        "upload_staging_key": upload_staging_key,
        "active_marker_key": active_key,
        "completed_marker_key": completed_key,
    }