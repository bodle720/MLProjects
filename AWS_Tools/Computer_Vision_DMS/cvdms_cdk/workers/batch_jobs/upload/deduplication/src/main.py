#!/usr/bin/env python3
import os
import json
import time
import logging
import math
from decimal import Decimal
from collections import defaultdict
from datetime import datetime, date

import pyarrow.dataset as ds
import s3fs
import boto3
from botocore.exceptions import ClientError

from common.utils import log

# Environment variables (provided by BatchingStage)
MANIFEST_S3_URI = os.environ.get("MANIFEST_S3_URI")
JOB_ID = os.environ.get("JOB_ID", "unknown")
USER = os.environ.get("USER", "unknown")
DATA_SOURCE = os.environ.get("DATA_SOURCE", "unknown")
EVENT_TYPE = os.environ.get("EVENT_TYPE", "unknown")
FILE_BUCKET_NAME = os.environ.get("FILE_BUCKET_NAME")
LOG_FIREHOSE_STREAM_NAME = os.environ.get("LOG_FIREHOSE_STREAM_NAME")
SHA256_TABLE_NAME = os.environ.get("SHA256_TABLE_NAME")

if not FILE_BUCKET_NAME:
    raise RuntimeError("[DEDUP_JOB_DEF] FILE_BUCKET_NAME not set")
if not LOG_FIREHOSE_STREAM_NAME:
    raise RuntimeError("[DEDUP_JOB_DEF] LOG_FIREHOSE_STREAM_NAME not set")
if not SHA256_TABLE_NAME:
    raise RuntimeError("[DEDUP_JOB_DEF] SHA256_TABLE_NAME not set")

PROCESSED_PREFIX = f"temp/image-upload/{JOB_ID}/batches/deduplication-step/processed"

DDB_BATCH_GET_MAX = 100

MAX_ROWS_IN_MEMORY = int(os.environ.get("DEDUP_MAX_ROWS_IN_MEMORY", "200000"))
MAX_GROUP_SIZE = int(os.environ.get("DEDUP_MAX_GROUP_SIZE", "10000"))

s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def to_jsonable(v):
    if v is None:
        return None

    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()

    if hasattr(v, "as_py"):
        v = v.as_py()

    if isinstance(v, datetime):
        return v.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(v, date):
        return v.isoformat()

    if isinstance(v, Decimal):
        f = float(v)
        return None if not math.isfinite(f) else f

    if isinstance(v, float):
        return None if not math.isfinite(v) else v

    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")

    return v

def read_manifest_with_retry(bucket, key, retries=5, delay=2):
    for attempt in range(retries):
        try:
            return s3.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey" and attempt < retries - 1:
                time.sleep(delay)
                continue
            raise

def s3_read_json(s3_uri):
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    resp = read_manifest_with_retry(bucket, key)
    return json.loads(resp["Body"].read().decode("utf-8"))

def write_s3_text(bucket, key, text, content_type="application/json"):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)

def normalize_row(row: dict) -> dict:
    for k, v in list(row.items()):
        row[k] = to_jsonable(v)
    return row

def read_parquet_rows_from_s3_uris(s3_uris):
    fs = s3fs.S3FileSystem()
    for uri in s3_uris:
        path = uri.replace("s3://", "")
        dataset = ds.dataset(path, filesystem=fs, format="parquet")
        scanner = dataset.scanner(batch_size=10_000, use_threads=True)
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                yield normalize_row(row)

def pick_representative(group):
    if len(group) == 1:
        return group[0]

    def key_fn(r):
        ts = r.get("uploaded_at") or "9999-12-31 23:59:59"
        return ts, r.get("image_id") or ""

    return min(group, key=key_fn)

def batch_get_dynamodb_items(table_name, keys):
    results = {}

    for i in range(0, len(keys), DDB_BATCH_GET_MAX):
        chunk = keys[i:i + DDB_BATCH_GET_MAX]
        request_keys = [{"sha256": {"S": k}} for k in chunk]
        request_items = {table_name: {"Keys": request_keys}}

        backoff = 1.0
        for attempt in range(15):
            try:
                resp = dynamodb.batch_get_item(RequestItems=request_items)

                for item in resp.get("Responses", {}).get(table_name, []):
                    sha = item.get("sha256", {}).get("S")
                    if sha:
                        results[sha] = item

                unprocessed = resp.get("UnprocessedKeys", {}).get(table_name, {}).get("Keys", [])
                if not unprocessed:
                    break

                request_items = {table_name: {"Keys": unprocessed}}
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code in ("AccessDeniedException", "UnrecognizedClientException"):
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        else:
            raise RuntimeError(f"[DEDUP_JOB_DEF] DynamoDB batch_get_item exceeded retries for table {table_name}")

    return results

def process_manifest(manifest):
    files = manifest.get("files", [])
    shard_name = manifest.get("shard_prefix", "shard")

    total_rows = 0
    groups = defaultdict(list)
    row_iter = read_parquet_rows_from_s3_uris(files)

    for r in row_iter:
        total_rows += 1
        sha = r.get("sha256_hash")
        if not sha:
            # carry forward; dedup will skip these later stages
            # keep dedup_status as-is unless caller wants to set it elsewhere
            groups[f"__MISSING_SHA__{total_rows}"].append(r)
            continue

        groups[sha].append(r)

        if total_rows > MAX_ROWS_IN_MEMORY:
            raise RuntimeError(f"[DEDUP_JOB_DEF] Shard {shard_name} exceeded MAX_ROWS_IN_MEMORY ({MAX_ROWS_IN_MEMORY})")

    processed_rows = []
    representatives = []  # list of (sha, rep_image_id)
    internal_dup_count = 0

    warned_big_group = False  # cap warning to once per shard

    for sha, group in groups.items():
        if sha.startswith("__MISSING_SHA__"):
            processed_rows.extend(group)
            continue

        if (not warned_big_group) and (len(group) > MAX_GROUP_SIZE):
            warned_big_group = True
            logger.warning(
                "[DEDUP_JOB_DEF] Shard %s has a large sha group: sha=%s size=%d exceeds MAX_GROUP_SIZE=%d",
                shard_name, sha, len(group), MAX_GROUP_SIZE
            )

        rep = pick_representative(group)
        rep_image_id = rep.get("image_id")
        rep["dedup_status"] = "passed"
        processed_rows.append(rep)

        if rep_image_id:
            representatives.append((sha, rep_image_id))

        for r in group:
            if r.get("image_id") == rep_image_id:
                continue
            r["dedup_status"] = "internal_duplicate"
            processed_rows.append(r)
            internal_dup_count += 1

    sha_list = [s for s, _ in representatives]
    ddb_map = batch_get_dynamodb_items(SHA256_TABLE_NAME, sha_list) if sha_list else {}

    external_dup_count = 0

    rep_index = {}
    for r in processed_rows:
        if r.get("sha256_hash") and r.get("image_id"):
            rep_index[(r["sha256_hash"], r["image_id"])] = r

    for sha, rep_image_id in representatives:
        item = ddb_map.get(sha)
        if not item:
            continue

        matched_image_id = item.get("image_id", {}).get("S")
        rep_row = rep_index.get((sha, rep_image_id))
        if rep_row:
            rep_row["dedup_status"] = "external_duplicate"
            rep_row["matched_image_id"] = matched_image_id
            external_dup_count += 1

    summary = {
        "job_id": JOB_ID,
        "shard_name": shard_name,
        "rows_read": total_rows,
        "internal_duplicates": internal_dup_count,
        "external_duplicates": external_dup_count,
        "representatives_checked": len(representatives),
        "processed_rows": len(processed_rows),
    }

    return processed_rows, summary

def write_processed_outputs(shard_name, processed_rows, summary):
    bucket = FILE_BUCKET_NAME
    jsonl_key = f"{PROCESSED_PREFIX}/shard-{shard_name}.jsonl"
    summary_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-summary.json"
    success_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-SUCCESS"

    body = "\n".join(json.dumps(r) for r in processed_rows)
    if len(body) > 50_000_000:
        raise RuntimeError("[DEDUP_JOB_DEF] JSONL too large for put_object; implement multipart upload")

    write_s3_text(bucket, jsonl_key, body, content_type="application/x-ndjson")
    write_s3_text(bucket, summary_key, json.dumps(summary), content_type="application/json")
    write_s3_text(bucket, success_key, "", content_type="text/plain")

    return {
        "jsonl": f"s3://{bucket}/{jsonl_key}",
        "summary": f"s3://{bucket}/{summary_key}",
        "success": f"s3://{bucket}/{success_key}",
    }

def main():
    start = time.time()
    if not MANIFEST_S3_URI:
        raise RuntimeError("[DEDUP_JOB_DEF] MANIFEST_S3_URI not set in environment")

    manifest = s3_read_json(MANIFEST_S3_URI)
    shard_name = manifest.get("shard_prefix", "shard")

    # START LOG (one line)
    log(
        JOB_ID, USER, EVENT_TYPE,
        f"[DEDUP_JOB_DEF] start shard={shard_name} manifest={MANIFEST_S3_URI}",
        LOG_FIREHOSE_STREAM_NAME
    )

    try:
        processed_rows, summary = process_manifest(manifest)
        write_processed_outputs(shard_name, processed_rows, summary)
    except Exception as e:
        # ERROR LOG (one line)
        log(
            JOB_ID, USER, EVENT_TYPE,
            f"[DEDUP_JOB_DEF] error shard={shard_name} err={e}",
            LOG_FIREHOSE_STREAM_NAME,
            error=str(e),
            level="error",
        )
        raise

    elapsed = time.time() - start

    # FINISH LOG (one line with counts + elapsed)
    log(
        JOB_ID, USER, EVENT_TYPE,
        (
            f"[DEDUP_JOB_DEF] done shard={shard_name} "
            f"rows_read={summary['rows_read']} "
            f"internal_duplicates={summary['internal_duplicates']} "
            f"external_duplicates={summary['external_duplicates']} "
            f"processed_rows={summary['processed_rows']} "
            f"time_s={elapsed:.1f}"
        ),
        LOG_FIREHOSE_STREAM_NAME
    )

if __name__ == "__main__":
    main()
