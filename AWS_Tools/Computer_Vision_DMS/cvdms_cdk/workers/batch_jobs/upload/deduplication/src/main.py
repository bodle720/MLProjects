#!/usr/bin/env python3
'''
This Batch job reads one dedup shard, deterministically marks internal duplicates, checks survivor hashes against the
canonical SHA table for external duplicates, and writes shard-local dedup results for later global reconciliation.
'''
import os
import json
import time
import logging
import math
from decimal import Decimal
from collections import defaultdict
from datetime import datetime, date

import pyarrow as pa
import pyarrow.dataset as ds
import s3fs
import boto3
from botocore.exceptions import ClientError

# The logging helper must be available in the image or layer
from common.utils import log

# Environment variables (provided by BatchingStage)
MANIFEST_S3_KEY = os.environ.get("MANIFEST_S3_KEY")
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

# Output prefix base (processed outputs will be written under this + /{job_id}/)
PROCESSED_PREFIX_BASE = os.environ.get(
    "PROCESSED_PREFIX_BASE",
    "temp/image-upload"
)

# Derived processed prefix for this job
PROCESSED_PREFIX = f"{PROCESSED_PREFIX_BASE}/{JOB_ID}/batches/deduplication-step/processed"

# DynamoDB batch limits
DDB_BATCH_GET_MAX = 100

# Safety limits (tunable via env)
MAX_ROWS_IN_MEMORY = int(os.environ.get("DEDUP_MAX_ROWS_IN_MEMORY", "200000"))
MAX_GROUP_SIZE = int(os.environ.get("DEDUP_MAX_GROUP_SIZE", "10000"))

s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def to_jsonable(v):
    if v is None:
        return None

    # pandas.Timestamp and similar
    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()

    # pyarrow scalar -> python value
    if hasattr(v, "as_py"):
        v = v.as_py()

    if isinstance(v, datetime):
        return v.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(v, date):
        return v.isoformat()

    if isinstance(v, Decimal):
        # the schema uses double for file_size_mb; Decimal can be cast safely
        f = float(v)
        return None if not math.isfinite(f) else f

    if isinstance(v, float):
        return None if not math.isfinite(v) else v

    if isinstance(v, (bytes, bytearray)):
        # safest: decode if we know encoding; otherwise base64
        return v.decode("utf-8", errors="replace")

    return v

def read_manifest_with_retry(bucket, key, retries=5, delay=2):
    for attempt in range(retries):
        try:
            return s3.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                if attempt < retries - 1:
                    time.sleep(delay)
                    continue
            raise

def s3_read_json(s3_uri):
    """Read a JSON object from s3://bucket/key and return parsed JSON."""
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    resp = read_manifest_with_retry(bucket, key)
    return json.loads(resp["Body"].read().decode("utf-8"))

def write_s3_text(bucket, key, text, content_type="application/json"):
    """Write text to S3."""
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)

def normalize_row(row: dict) -> dict:
    for k, v in list(row.items()):
        row[k] = to_jsonable(v)
    return row

def read_parquet_rows_from_s3_uris(s3_uris):
    """
    Generator yielding dict rows from a list of s3://... parquet URIs.
    Requires pyarrow + s3fs in the container.
    """
    fs = s3fs.S3FileSystem()
    for uri in s3_uris:
        path = uri.replace("s3://", "")  # IMPORTANT when passing filesystem=
        try:
            dataset = ds.dataset(path, filesystem=fs, format="parquet")
            scanner = dataset.scanner(
                batch_size=10_000,  # max rows per RecordBatch
                use_threads=True
            )
            for batch in scanner.to_batches():
                for row in batch.to_pylist():
                    yield normalize_row(row)
        except Exception as e:
            logger.error("[DEDUP_JOB_DEF] Failed to read parquet from %s: %s", uri, e)
            raise

def pick_representative(group):
    """
    Deterministic representative selection:
    1) earliest uploaded_at (string compare works for 'YYYY-MM-DD HH:MM:SS')
    2) tie-breaker: lexicographically smallest image_id
    """

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
        for attempt in range(15):  # keep this small
            try:
                resp = dynamodb.batch_get_item(RequestItems=request_items)

                for item in resp.get("Responses", {}).get(table_name, []):
                    sha = item.get("sha256", {}).get("S")
                    if sha:
                        results[sha] = item

                unprocessed = resp.get("UnprocessedKeys", {}).get(table_name, {}).get("Keys", [])
                if not unprocessed:
                    break  # done with this chunk

                request_items = {table_name: {"Keys": unprocessed}}
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code in ("AccessDeniedException", "UnrecognizedClientException"):
                    raise  # fail fast; don't backoff forever
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        else:
            raise RuntimeError(f"[DEDUP_JOB_DEF] DynamoDB batch_get_item exceeded retries for table {table_name}")

    return results

def process_manifest(manifest):
    """
    manifest: dict with keys 'job_id', 'shard_prefix', 'files'
    Returns: processed_rows (list of dicts), summary dict
    """
    files = manifest.get("files", [])
    shard_name = manifest.get("shard_prefix", "shard")
    total_rows = 0
    groups = defaultdict(list)

    row_iter = read_parquet_rows_from_s3_uris(files)

    # Accumulate rows grouped by sha256_hash
    for r in row_iter:
        total_rows += 1
        sha = r.get("sha256_hash")
        if not sha:
            # mark missing sha as validation failure
            r["validation_status"] = r.get("validation_status", "failed")
            r["validation_error"] = r.get("validation_error", "missing sha256_hash")
            r["dedup_status"] = r.get("dedup_status", "pending")
            r["dedup_error"] = r.get("dedup_error", "")
            groups[f"__MISSING_SHA__{total_rows}"].append(r)
            continue

        groups[sha].append(r)

        if total_rows > MAX_ROWS_IN_MEMORY:
            raise RuntimeError(f"[DEDUP_JOB_DEF] Shard {shard_name} exceeded MAX_ROWS_IN_MEMORY ({MAX_ROWS_IN_MEMORY})")

    processed_rows = []
    representatives = []  # list of (sha, rep_image_id)
    internal_dup_count = 0

    for sha, group in groups.items():
        if sha.startswith("__MISSING_SHA__"):
            for r in group:
                processed_rows.append(r)
            continue

        if len(group) > MAX_GROUP_SIZE:
            logger.warning(
                f"[DEDUP_JOB_DEF] SHA group {sha} size {len(group)} exceeds MAX_GROUP_SIZE={MAX_GROUP_SIZE}"
            )

        rep = pick_representative(group)
        rep_image_id = rep.get("image_id")
        rep["dedup_status"] = "survivor"
        processed_rows.append(rep)

        if rep_image_id:
            representatives.append((sha, rep_image_id))

        for r in group:
            if r.get("image_id") == rep_image_id:
                continue
            r["dedup_status"] = "internal_duplicate"
            processed_rows.append(r)
            internal_dup_count += 1

    # Query DynamoDB for representatives' sha values
    sha_list = [s for s, _ in representatives]
    ddb_map = {}
    if sha_list:
        ddb_map = batch_get_dynamodb_items(SHA256_TABLE_NAME, sha_list)

    external_dup_count = 0

    # Build an index for fast lookup of the representative row
    # key: (sha256_hash, image_id) -> row dict
    rep_index = {}
    for r in processed_rows:
        if r.get("sha256_hash") and r.get("image_id"):
            rep_index[(r["sha256_hash"], r["image_id"])] = r

    # Apply external duplicate marks in O(R)
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
        "processed_rows": len(processed_rows)
    }

    return processed_rows, summary

def write_processed_outputs(job_id, shard_name, processed_rows, summary):
    """
    Writes:
      - JSONL file: processed rows
      - summary JSON
      - _SUCCESS marker
    to PROCESSED_PREFIX
    """
    bucket = FILE_BUCKET_NAME
    jsonl_key = f"{PROCESSED_PREFIX}/shard-{shard_name}.jsonl"
    summary_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-summary.json"
    success_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-SUCCESS"

    # stream JSONL content
    # write in chunks to avoid building a huge string in memory
    # but boto3 put_object expects full body; for large outputs consider multipart upload

    body = "\n".join(json.dumps(r) for r in processed_rows)
    if len(body) > 50_000_000:
        raise RuntimeError("[DEDUP_JOB_DEF] JSONL too large for put_object; implement multipart upload")

    write_s3_text(bucket, jsonl_key, body, content_type="application/x-ndjson")
    write_s3_text(bucket, summary_key, json.dumps(summary), content_type="application/json")
    write_s3_text(bucket, success_key, "", content_type="text/plain")

    return {
        "jsonl": f"s3://{bucket}/{jsonl_key}",
        "summary": f"s3://{bucket}/{summary_key}",
        "success": f"s3://{bucket}/{success_key}"
    }

def main():
    start = time.time()
    if not MANIFEST_S3_KEY:
        raise RuntimeError("[DEDUP_JOB_DEF] MANIFEST_S3_KEY not set in environment")

    manifest = s3_read_json(MANIFEST_S3_KEY)
    shard_name = manifest.get("shard_prefix", "shard")
    log(JOB_ID, USER, EVENT_TYPE, f"[DEDUP_JOB_DEF] Batch worker starting for job {JOB_ID}, shard {shard_name}, manifest {MANIFEST_S3_KEY}, pyarrow={pa.__version__}", LOG_FIREHOSE_STREAM_NAME)

    try:
        processed_rows, summary = process_manifest(manifest)
    except Exception as e:
        log(JOB_ID, USER, EVENT_TYPE, f"[DEDUP_JOB_DEF] Batch worker failed processing manifest {MANIFEST_S3_KEY}: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    try:
        write_processed_outputs(JOB_ID, shard_name, processed_rows, summary)
    except Exception as e:
        log(JOB_ID, USER, EVENT_TYPE, f"[DEDUP_JOB_DEF] Batch worker failed writing outputs for shard {shard_name}: {e}", LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    elapsed = time.time() - start
    log(JOB_ID, USER, EVENT_TYPE, f"[DEDUP_JOB_DEF] Batch worker completed shard {shard_name}: rows_read={summary['rows_read']}, internal_duplicates={summary['internal_duplicates']}, external_duplicates={summary['external_duplicates']}, time_s={elapsed:.1f}", LOG_FIREHOSE_STREAM_NAME)

if __name__ == "__main__":
    main()