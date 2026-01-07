#!/usr/bin/env python3
import os
import io
import re
import json
import time
import hashlib

import boto3
from PIL import Image
from botocore.exceptions import ClientError

from common.logging_utils import log
from common.s3_utils import parse_s3_uri
from helpers import infer_dtype, create_and_save_labels, stable_uuid5, read_manifest_with_retry

# Env Variables from upload stack
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# From the map state input
MANIFEST_S3_URI = os.environ["MANIFEST_S3_URI"].strip()
JOB_ID = os.environ["JOB_ID"]
USER = os.environ["USER"]
LABEL_TYPE = os.environ["LABEL_TYPE"]
DATA_SOURCE = os.environ["DATA_SOURCE"]
EVENT_TYPE = os.environ["EVENT_TYPE"]
REGISTRATION_TIME = os.environ["REGISTRATION_TIME"]

PROCESSED_PREFIX = f"temp/image-upload/{JOB_ID}/batches/validation-step/processed"

s3 = boto3.client("s3")

def _manifest_shard_name(manifest_s3_uri: str) -> str:
    _, key = parse_s3_uri(manifest_s3_uri)
    fname = key.rsplit("/", 1)[-1]
    # batch-001.jsonl -> batch-001
    m = re.match(r"^(.*)\.jsonl$", fname)
    return m.group(1) if m else fname

def _write_s3_text(bucket: str, key: str, text: str, content_type: str) -> None:
    body = text.encode("utf-8")
    # Safety: put_object limit is 5GB, but you may have your own guard; 200 rows should be small.
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)

def process_image(line: dict, shard_name: str, line_idx: int) -> dict:

    # derive deterministic image uuid from job + source-ref
    temp_source_ref = line.get("source-ref")

    # Deterministic per-occurrence ID (unique even for duplicates)
    image_id = stable_uuid5(f"{JOB_ID}|{shard_name}|{line_idx}")

    if not isinstance(temp_source_ref, str) or not temp_source_ref.startswith("s3://") or line.get('issue'):
        # malformed line -> failed row
        error_msg = line.get("issue") or f"missing/invalid temp source-ref: {temp_source_ref}"
        return {
            "job_id": JOB_ID,
            "image_id": image_id,
            "temp_source_ref": temp_source_ref,
            "img_type": None,
            "img_height": None,
            "img_width": None,
            "num_channels": None,
            "dtype": None,
            "file_size_mb": 0.0,
            "uploaded_at": REGISTRATION_TIME,
            "data_source": DATA_SOURCE,
            "sha256_hash": None,
            "string_labels": None,
            "temp_source_ref_bbox_meta": None,
            "temp_source_ref_semantic_png": None,
            "temp_source_ref_semantic_meta": None,
            "temp_source_ref_instance_png": None,
            "temp_source_ref_instance_meta": None,
            "label_fingerprint": None,
            "classes_present": None,
            "validation_status": "failed",
            "validation_error": error_msg,
            "dedup_status": "pending",
            "dedup_error": None,
            "registration_status": "pending",
            "registration_error": None,
            "matched_image_id": None,
        }

    # Defaults for upload_staging row
    row = {
        "job_id": JOB_ID,
        "image_id": image_id,
        "temp_source_ref": temp_source_ref,
        "img_type": None,
        "img_height": None,
        "img_width": None,
        "num_channels": None,
        "dtype": None,
        "file_size_mb": 0.0,
        "uploaded_at": REGISTRATION_TIME,
        "data_source": DATA_SOURCE,
        "sha256_hash": None,
        "string_labels": None,
        "temp_source_ref_bbox_meta": None,
        "temp_source_ref_semantic_png": None,
        "temp_source_ref_semantic_meta": None,
        "temp_source_ref_instance_png": None,
        "temp_source_ref_instance_meta": None,
        "label_fingerprint": None,
        "classes_present": None,
        "validation_status": "pending",
        "validation_error": None,
        "dedup_status": "pending",
        "dedup_error": None,
        "registration_status": "pending",
        "registration_error": None,
        "matched_image_id": None,
    }

    # Fetch image bytes from source-ref
    try:
        bucket, key = parse_s3_uri(temp_source_ref)
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        row["validation_status"] = "failed"
        row["validation_error"] = str(e)
        return row

    data = obj["Body"].read()
    row["file_size_mb"] = float(round(len(data) / (1024 * 1024), 4))

    buf = io.BytesIO(data)
    buf.seek(0)
    sha = hashlib.sha256(buf.read()).hexdigest()
    row["sha256_hash"] = sha
    buf.seek(0)

    # Open image
    try:
        img = Image.open(buf)
        img.load()
    except Exception as e:
        row["validation_status"] = "failed"
        row["validation_error"] = f"Cannot open image {temp_source_ref}: {e}"
        return row

    bands = len(img.getbands())
    if bands not in (1, 3):
        row["validation_status"] = "failed"
        row["validation_error"] = f"Invalid band count: {bands}, must be 1 or 3"
        return row

    row["num_channels"] = bands
    row["img_type"] = "L" if bands == 1 else "RGB"
    row["dtype"] = infer_dtype(img)
    width, height = img.size
    row["img_width"] = int(width)
    row["img_height"] = int(height)

    label_cols = {
        "single-label": [],
        "multi-label": [],
        "object-detection": ["temp_source_ref_bbox_meta"],
        "semantic-segmentation": ["temp_source_ref_semantic_png", "temp_source_ref_semantic_meta"],
        "instance-segmentation": ["temp_source_ref_instance_png", "temp_source_ref_instance_meta"],
    }

    col_names = label_cols.get(LABEL_TYPE)
    if col_names is None:
        row["validation_status"] = "failed"
        row["validation_error"] = f"Unsupported LABEL_TYPE: {LABEL_TYPE}"
        return row

    paths, classes_present, label_fingerprint, error_msg = create_and_save_labels(
        line=line,
        label_type=LABEL_TYPE,
        job_id=JOB_ID,
        file_bucket_name=FILE_BUCKET_NAME
    )

    if error_msg:
        row["validation_status"] = "failed"
        row["validation_error"] = f"Unable to form/save label files: {error_msg}"
        return row

    if not classes_present:
        row["validation_status"] = "failed"
        row["validation_error"] = "Empty list of classes_present"
        return row

    row["classes_present"] = classes_present
    if LABEL_TYPE in ("single-label", "multi-label"):
        row["string_labels"] = classes_present

    if col_names:
        if len(paths) != len(col_names):
            row["validation_status"] = "failed"
            row["validation_error"] = f"Expected {len(col_names)} label paths, got {len(paths)}"
            return row

        for col_name, path in zip(col_names, paths):
            row[col_name] = path

    if label_fingerprint:
        row["label_fingerprint"] = label_fingerprint

    row["validation_status"] = "passed"
    return row

def write_processed_outputs(shard_name: str, processed_rows: list[dict], summary: dict) -> None:
    bucket = FILE_BUCKET_NAME

    jsonl_key = f"{PROCESSED_PREFIX}/upload_staging/shard-{shard_name}.jsonl"
    summary_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-summary.json"
    success_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-SUCCESS"

    body = "\n".join(json.dumps(r) for r in processed_rows) + "\n"
    _write_s3_text(bucket, jsonl_key, body, content_type="application/x-ndjson")
    _write_s3_text(bucket, summary_key, json.dumps(summary), content_type="application/json")
    # write SUCCESS last
    _write_s3_text(bucket, success_key, "", content_type="text/plain")

def main():
    start = time.time()

    shard_name = _manifest_shard_name(MANIFEST_S3_URI)

    log(JOB_ID, USER, EVENT_TYPE,
        f"[VAL_JOB_DEF] start shard={shard_name} manifest={MANIFEST_S3_URI} label_type={LABEL_TYPE}",
        LOG_FIREHOSE_STREAM_NAME)

    mb, mk = parse_s3_uri(MANIFEST_S3_URI)

    obj = read_manifest_with_retry(mb, mk)
    if not obj:
        raise RuntimeError(f"[VAL_JOB_DEF] Could not read manifest: {MANIFEST_S3_URI}")

    processed_rows = []
    total = 0
    failed = 0
    for line_idx, line_bytes in enumerate(obj["Body"].iter_lines(), start = 0):
        if not line_bytes:
            continue

        s = line_bytes.decode("utf-8-sig").strip()
        if not s:
            continue

        try:
            line = json.loads(s)
        except json.JSONDecodeError as e:
            # malformed JSON -> failed row
            line = {"issue": f"invalid JSON: {e.msg} (pos={e.pos}), raw =  {s}"}

        if not isinstance(line, dict):
            # non-dict JSON -> failed row (still preserves line count)
            line = {"issue": f"expected dict, got {type(line).__name__}, raw =  {s}"}

        row = process_image(line, shard_name=shard_name, line_idx=line_idx)

        total += 1
        if row["validation_status"] != "passed":
            failed += 1
        processed_rows.append(row)

    if total == 0:
        raise RuntimeError(f"[VAL_JOB_DEF] No images found in manifest for shard={shard_name}")

    summary = {
        "job_id": JOB_ID,
        "shard_name": shard_name,
        "label_type": LABEL_TYPE,
        "rows_read": total,
        "failed_rows": failed,
        "processed_rows": len(processed_rows),
        "manifest": MANIFEST_S3_URI,
    }

    write_processed_outputs(shard_name, processed_rows, summary)

    elapsed = time.time() - start
    log(JOB_ID, USER, EVENT_TYPE,
        f"[VAL_JOB_DEF] done shard={shard_name} rows_read={total} failed={failed} processed_rows={len(processed_rows)} time_s={elapsed:.1f}",
        LOG_FIREHOSE_STREAM_NAME)

if __name__ == "__main__":
    main()
