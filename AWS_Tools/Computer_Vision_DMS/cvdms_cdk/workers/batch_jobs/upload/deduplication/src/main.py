#!/usr/bin/env python3
import os
import json
import time
import logging
from collections import defaultdict

from common.logging_utils import log
from common.s3_utils import parse_s3_uri, s3_read_json, write_s3_obj, read_parquet_rows_from_s3_uris
from common.ddb_utils import batch_get_dynamodb_items

# Environment variables (provided by BatchingStage)
MANIFEST_S3_URI = os.environ.get("MANIFEST_S3_URI")
JOB_ID = os.environ.get("JOB_ID", "unknown")
USER = os.environ.get("USER", "unknown")
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
MAX_ROWS_IN_MEMORY = 200000
MAX_GROUP_SIZE = 10000

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def pick_representative(group):
    if len(group) == 1:
        return group[0]

    def key_fn(r):
        ts = r.get("uploaded_at") or "9999-12-31 23:59:59"
        return ts, r.get("image_id") or ""

    return min(group, key=key_fn)

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
    ddb_map = batch_get_dynamodb_items(SHA256_TABLE_NAME, sha_list, DDB_BATCH_GET_MAX, "[DEDUP_JOB_DEF]") if sha_list else {}

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

    body = "\n".join(json.dumps(r) for r in processed_rows) + "\n"
    if len(body) > 50_000_000:
        raise RuntimeError("[DEDUP_JOB_DEF] JSONL too large for put_object; implement multipart upload")

    write_s3_obj(bucket, jsonl_key, body, "application/x-ndjson", "[DEDUP_JOB_DEF]")
    write_s3_obj(bucket, summary_key, json.dumps(summary), "application/json", "[DEDUP_JOB_DEF]")
    write_s3_obj(bucket, success_key, "", "text/plain", "[DEDUP_JOB_DEF]")

    return {
        "jsonl": f"s3://{bucket}/{jsonl_key}",
        "summary": f"s3://{bucket}/{summary_key}",
        "success": f"s3://{bucket}/{success_key}",
    }

def main():
    start = time.time()
    if not MANIFEST_S3_URI:
        raise RuntimeError("[DEDUP_JOB_DEF] MANIFEST_S3_URI not set in environment")

    mb, mk = parse_s3_uri(MANIFEST_S3_URI, "[DEDUP_JOB_DEF]")
    manifest = s3_read_json(mb, mk, "[DEDUP_JOB_DEF]")

    shard_name = manifest.get("shard_prefix", "shard")

    # START LOG (one line)
    log(JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,f"[DEDUP_JOB_DEF] start shard={shard_name} manifest={MANIFEST_S3_URI}")


    try:
        processed_rows, summary = process_manifest(manifest)
        write_processed_outputs(shard_name, processed_rows, summary)
    except Exception as e:
        # ERROR LOG (one line)
        log(JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,f"[DEDUP_JOB_DEF] error shard={shard_name} err={e}", level="error")
        raise

    elapsed = time.time() - start

    # FINISH LOG (one line with counts + elapsed)
    log(
        JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,
        (
            f"[DEDUP_JOB_DEF] done shard={shard_name} "
            f"rows_read={summary['rows_read']} "
            f"internal_duplicates={summary['internal_duplicates']} "
            f"external_duplicates={summary['external_duplicates']} "
            f"processed_rows={summary['processed_rows']} "
            f"time_s={elapsed:.1f}"
        )
    )

if __name__ == "__main__":
    main()