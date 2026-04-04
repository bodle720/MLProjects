#!/usr/bin/env python3
import os
import json
import time
import hashlib
from typing import Any
from collections import defaultdict
from datetime import datetime, timezone

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import parse_s3_uri, s3_read_json, write_s3_obj
from common.upload_utils.upload_ddb_utils import batch_get_dynamodb_items
from common.general_utils.s3fs_utils import (
    read_parquet_rows_from_s3_uris,
    jsonl_stream_to_s3
)

MANIFEST_S3_URI = os.environ["MANIFEST_S3_URI"].strip()
JOB_ID = os.environ["JOB_ID"]
USER = os.environ["USER"]
EVENT_TYPE = os.environ["EVENT_TYPE"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
SHA256_TABLE_NAME = os.environ["SHA256_TABLE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[DEDUP_JOB_DEF]"

if not FILE_BUCKET_NAME:
    raise RuntimeError(f"{TASK_NAME} FILE_BUCKET_NAME not set")
if not LOG_FIREHOSE_STREAM_NAME:
    raise RuntimeError(f"{TASK_NAME} LOG_FIREHOSE_STREAM_NAME not set")
if not SHA256_TABLE_NAME:
    raise RuntimeError(f"{TASK_NAME} SHA256_TABLE_NAME not set")

PROCESSED_PREFIX = f"temp/image-upload/{JOB_ID}/batches/deduplication-step/processed"
DDB_BATCH_GET_MAX = 25
MAX_ROWS_IN_MEMORY = 200000
MAX_GROUP_SIZE = 10000

def ts_sortable(v: Any) -> str:
    """
    Return a sortable timestamp string.
    - datetime -> ISO-ish string in UTC
    - string -> stripped string
    - None/unknown -> far-future sentinel
    """
    if v is None:
        return "9999-12-31 23:59:59"

    if isinstance(v, datetime):
        # Ensure tz-aware then convert to UTC
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        v = v.astimezone(timezone.utc)
        # keep same style you already use elsewhere
        return v.strftime("%Y-%m-%d %H:%M:%S")

    # some parquet readers may return numbers or other types
    s = str(v).strip()
    return s if s else "9999-12-31 23:59:59"

def pick_representative(group):
    if len(group) == 1:
        return group[0]

    def key_fn(r):
        ts = ts_sortable(r.get("uploaded_at"))
        return (ts, r.get("image_id") or "")

    return min(group, key=key_fn)

def norm_string_labels(row: dict) -> list[str]:
    labels = row.get("string_labels")
    if not isinstance(labels, list) or not labels:
        labels = row.get("classes_present")
    if not isinstance(labels, list):
        return []
    out = []
    seen = set()
    for x in labels:
        s = str(x).strip().lower()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out)

def label_signature(row: dict) -> str:
    """
    Deterministic signature for "are these labels identical?"
    - Prefer label_fingerprint when present (OD/semantic/instance)
    - Else hash normalized string labels (single/multi)
    """
    fp = row.get("label_fingerprint")
    if isinstance(fp, str) and fp.strip():
        return f"fp:{fp.strip()}"

    labels = norm_string_labels(row)
    if labels:
        blob = ("|".join(labels)).encode("utf-8")
        return "str:" + hashlib.sha256(blob).hexdigest()

    # Should not happen for validation_status=passed, but treat as conflict-ish
    return "__MISSING_LABEL_SIG__"

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
            r["dedup_status"] = "failed"
            r["dedup_error"] = "missing sha256_hash"
            groups[f"__MISSING_SHA__{total_rows}"].append(r)
            continue

        if r.get("validation_status") != "passed":
            r["dedup_status"] = "failed"
            r["dedup_error"] = f"skipped dedup because validation_status={r.get('validation_status')}"
            groups[f"__VAL_FAILED__{total_rows}"].append(r)
            continue

        groups[sha].append(r)

        if total_rows > MAX_ROWS_IN_MEMORY:
            raise RuntimeError(f"{TASK_NAME} Shard {shard_name} exceeded MAX_ROWS_IN_MEMORY ({MAX_ROWS_IN_MEMORY})")

    processed_rows = []
    representatives = []  # list of (sha, rep_image_id)
    internal_dup_count = 0

    warned_big_group = False  # cap warning to once per shard

    for sha, group in groups.items():
        if sha.startswith("__MISSING_SHA__") or sha.startswith("__VAL_FAILED__"):
            processed_rows.extend(group)
            continue

        if (not warned_big_group) and (len(group) > MAX_GROUP_SIZE):
            warned_big_group = True
            log(JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Shard {shard_name} has a large sha group: sha={sha} size={len(group)} exceeds MAX_GROUP_SIZE={MAX_GROUP_SIZE}",
                level='warning')

        sigs = {label_signature(r) for r in group}
        if "__MISSING_LABEL_SIG__" in sigs or len(sigs) > 1:
            for r in group:
                r["dedup_status"] = "failed"
                error_msg = "string_labels or classes_present value must be a non-empty list" if "__MISSING_LABEL_SIG__" in sigs else "duplicate images with different labels not allowed"
                r["dedup_error"] = error_msg
                r["matched_image_id"] = None
            processed_rows.extend(group)
            continue

        rep = pick_representative(group)
        rep_image_id = rep.get("image_id")
        rep["dedup_status"] = "passed"
        processed_rows.append(rep)

        if rep_image_id:
            representatives.append((sha, rep_image_id))

        for r in group:
            if r is rep:
                continue
            r["dedup_status"] = "internal_duplicate"
            processed_rows.append(r)
            internal_dup_count += 1

    sha_list = list({s for s, _ in representatives})

    print(f"{TASK_NAME} unique SHA count = {len(sha_list)}")
    ddb_map = batch_get_dynamodb_items(SHA256_TABLE_NAME, sha_list, DDB_BATCH_GET_MAX, TASK_NAME) if sha_list else {}

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
        if not matched_image_id:
            continue

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

    jsonl_stream_to_s3(bucket, jsonl_key, processed_rows)

    write_s3_obj(bucket, summary_key, json.dumps(summary), "application/json", TASK_NAME)
    write_s3_obj(bucket, success_key, "", "text/plain", TASK_NAME)

    return {
        "jsonl": f"s3://{bucket}/{jsonl_key}",
        "summary": f"s3://{bucket}/{summary_key}",
        "success": f"s3://{bucket}/{success_key}",
    }

def main():
    start = time.time()
    if not MANIFEST_S3_URI:
        raise RuntimeError(f"{TASK_NAME} MANIFEST_S3_URI not set in environment")

    mb, mk = parse_s3_uri(MANIFEST_S3_URI, TASK_NAME)
    manifest = s3_read_json(mb, mk, TASK_NAME)

    shard_name = manifest.get("shard_prefix", "shard")

    log(JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} start shard={shard_name} manifest={MANIFEST_S3_URI}")

    try:
        processed_rows, summary = process_manifest(manifest)
        write_processed_outputs(shard_name, processed_rows, summary)
    except Exception as e:
        log(JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} error shard={shard_name} err={e}", level="error")
        raise

    elapsed = time.time() - start

    log(
        JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} done shard={shard_name} "
            f"rows_read={summary['rows_read']} "
            f"internal_duplicates={summary['internal_duplicates']} "
            f"external_duplicates={summary['external_duplicates']} "
            f"processed_rows={summary['processed_rows']} "
            f"time_s={elapsed:.1f}"
        )
    )

if __name__ == "__main__":
    main()