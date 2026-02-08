#!/usr/bin/env python3

'''
Memory is O(batch size), not O(1), because process_image accumulates the whole shard in memory. We can change to a more strategic
processing and writing strategy, but as long as shard size and memory size are reasonable, we will keep it this way. Note for future change.
'''

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
from common.s3_utils import parse_s3_uri, write_s3_obj, read_obj_with_retry
from helpers import infer_dtype, create_and_save_labels, stable_uuid5, parse_json_object_line
from quality_helpers import compute_image_quality_features

MANIFEST_S3_URI = os.environ["MANIFEST_S3_URI"].strip()
JOB_ID = os.environ["JOB_ID"]
USER = os.environ["USER"]
LABEL_TYPE = os.environ["LABEL_TYPE"]
DATA_SOURCE = os.environ["DATA_SOURCE"]
EVENT_TYPE = os.environ["EVENT_TYPE"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
SHA256_TABLE_NAME = os.environ["SHA256_TABLE_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
REGISTRATION_TIME = os.environ["REGISTRATION_TIME"]

TASK_NAME = "[VAL_JOB_DEF]"
PROCESSED_PREFIX = f"temp/image-upload/{JOB_ID}/batches/validation-step/processed"

s3 = boto3.client("s3")

def manifest_shard_name(manifest_s3_uri: str) -> str:
    try:
        _, key = parse_s3_uri(manifest_s3_uri, TASK_NAME)
    except ValueError as e:
        raise RuntimeError(f"{TASK_NAME} Invalid manifest_s3_uri: {e}")

    fname = key.rsplit("/", 1)[-1]
    # batch-001.jsonl -> batch-001
    m = re.match(r"^(.*)\.jsonl$", fname)
    return m.group(1) if m else fname

def process_image(line: dict, shard_name: str, line_idx: int) -> dict:

    # derive deterministic image uuid from job + source-ref
    temp_source_ref = line.get("source_ref")

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
            "file_size_mb": None,
            "uploaded_at": REGISTRATION_TIME,
            "data_source": DATA_SOURCE,
            "sha256_hash": None,
            "luma_mean": None,
            "luma_p10": None,
            "luma_p90": None,
            "dark_frac": None,
            "bright_frac": None,
            "contrast_luma_std": None,
            "contrast_luma_p90_p10": None,
            "blur_laplacian_var": None,
            "sat_mean": None,
            "colorfulness": None,
            "lighting_bucket": None,
            "blur_bucket": None,
            "contrast_bucket": None,
            "color_bucket": None,
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
        "file_size_mb": None,
        "uploaded_at": REGISTRATION_TIME,
        "data_source": DATA_SOURCE,
        "sha256_hash": None,
        "luma_mean": None,
        "luma_p10": None,
        "luma_p90": None,
        "dark_frac": None,
        "bright_frac": None,
        "contrast_luma_std": None,
        "contrast_luma_p90_p10": None,
        "blur_laplacian_var": None,
        "sat_mean": None,
        "colorfulness": None,
        "lighting_bucket": None,
        "blur_bucket": None,
        "contrast_bucket": None,
        "color_bucket": None,
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

    try:
        bucket, key = parse_s3_uri(temp_source_ref, TASK_NAME)
    except ValueError as e:
        row["validation_status"] = "failed"
        row["validation_error"] = str(e)
        return row

    # Fetch image bytes from source-ref
    try:
        obj = read_obj_with_retry(bucket, key, TASK_NAME)
    except ClientError as e:
        row["validation_status"] = "failed"
        row["validation_error"] = str(e)
        return row

    if not obj:
        row["validation_status"] = "failed"
        row["validation_error"] = "Unable to read object from S3 with retry"
        return row

    data = obj["Body"].read()
    row["file_size_mb"] = float(round(len(data) / (1024 * 1024), 4))
    row["sha256_hash"] = hashlib.sha256(data).hexdigest()

    buf = io.BytesIO(data)
    buf.seek(0)

    # Open image
    try:
        img = Image.open(buf)
        img.load()
    except Image.DecompressionBombError as e:
        row["validation_status"] = "failed"
        row["validation_error"] = f"DecompressionBombError: {e}"
        return row
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

    quality_feats = compute_image_quality_features(img)
    row.update(quality_feats)

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
    jsonl_key = f"{PROCESSED_PREFIX}/upload_staging/shard-{shard_name}.jsonl"
    summary_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-summary.json"
    success_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-SUCCESS"

    body = "\n".join(json.dumps(r) for r in processed_rows) + "\n"
    write_s3_obj(FILE_BUCKET_NAME,
                  jsonl_key,
                  body,
                  "application/x-ndjson",
                 TASK_NAME)
    write_s3_obj(FILE_BUCKET_NAME,
                  summary_key,
                  json.dumps(summary),
                  "application/json",
                  TASK_NAME)
    # write SUCCESS last
    write_s3_obj(FILE_BUCKET_NAME,
                  success_key,
                  "",
                  "text/plain",
                  TASK_NAME)

def main():
    start = time.time()

    shard_name = manifest_shard_name(MANIFEST_S3_URI)

    log(JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} start shard={shard_name} manifest={MANIFEST_S3_URI} label_type={LABEL_TYPE}")

    try:
        mb, mk = parse_s3_uri(MANIFEST_S3_URI, TASK_NAME)
    except ValueError as e:
        raise RuntimeError(f"{TASK_NAME} Invalid MANIFEST_S3_URI: {e}")

    obj = read_obj_with_retry(mb, mk, TASK_NAME)
    if not obj:
        raise RuntimeError(f"{TASK_NAME} Could not read manifest: {MANIFEST_S3_URI}")

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
            line = parse_json_object_line(s)
        except Exception as e:
            # malformed JSON -> failed row
            line = {"issue": f"invalid JSON: {e}, raw =  {s}"}

        if not isinstance(line, dict):
            # non-dict JSON -> failed row (still preserves line count)
            line = {"issue": f"expected dict, got {type(line).__name__}, raw =  {s}"}

        row = process_image(line, shard_name=shard_name, line_idx=line_idx)

        total += 1
        if row["validation_status"] != "passed":
            failed += 1
        processed_rows.append(row)

    if total == 0:
        raise RuntimeError(f"{TASK_NAME} No images found in manifest for shard={shard_name}")

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
    log(JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} done shard={shard_name} rows_read={total} failed={failed} processed_rows={len(processed_rows)} time_s={elapsed:.1f}")

if __name__ == "__main__":
    main()