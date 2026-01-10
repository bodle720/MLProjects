#!/usr/bin/env python3
import os
import json
import time
from typing import Any, Dict, Iterable, List, Tuple

import boto3
import pyarrow as pa
import s3fs

from common.logging_utils import log
from common.s3_utils import write_s3_obj, parse_s3_uri, s3_read_json, read_parquet_rows_from_s3_uris, jsonl_content_converter

from helpers import (
    build_canonical_image_dest,
    build_canonical_label_dests,
    build_canonical_imagery_row,
    build_label_table_row,
    copy_objects_or_raise,
    cleanup_copied_best_effort
)

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

TASK_NAME = "[REG_JOB_DEF]"

if not MANIFEST_S3_URI:
    raise RuntimeError(f"{TASK_NAME} MANIFEST_S3_URI not set")
if not FILE_BUCKET_NAME:
    raise RuntimeError(f"{TASK_NAME} FILE_BUCKET_NAME not set")
if not LOG_FIREHOSE_STREAM_NAME:
    raise RuntimeError(f"{TASK_NAME} LOG_FIREHOSE_STREAM_NAME not set")

# This is the “stage processed area” analogous to deduplication-step
PROCESSED_PREFIX = f"temp/image-upload/{JOB_ID}/batches/registration-step/processed"

# Safety
MAX_ROWS_IN_MEMORY = 200000

s3 = boto3.client("s3")

def write_jsonl_to_s3(bucket: str, key: str, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Stream JSONL to S3 via s3fs (avoids building huge strings in memory).
    """
    fs = s3fs.S3FileSystem()
    # s3fs path form: "bucket/key"
    path = f"{bucket}/{key}"
    with fs.open(path, "wb") as f:
        for r in rows:
            f.write((json.dumps(r) + "\n").encode("utf-8"))

def process_manifest(manifest: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      - updated_upload_rows: list[dict] (all rows, carried forward)
      - canonical_imagery_rows: list[dict] (only successfully registered eligible rows)
      - canonical_label_rows: list[dict] (only for successfully registered eligible rows; includes __table)
      - summary: dict
    """
    files = manifest.get("files", []) or []
    shard_name = manifest.get("shard_prefix", "shard")

    total_rows = 0
    eligible_rows = 0
    skipped_rows = 0
    reg_passed = 0
    reg_failed = 0

    updated_upload_rows: List[Dict[str, Any]] = []
    canonical_imagery_rows: List[Dict[str, Any]] = []
    canonical_label_rows: List[Dict[str, Any]] = []

    for row in read_parquet_rows_from_s3_uris(files):
        total_rows += 1
        if total_rows > MAX_ROWS_IN_MEMORY:
            raise RuntimeError(f"{TASK_NAME} Shard {shard_name} exceeded MAX_ROWS_IN_MEMORY={MAX_ROWS_IN_MEMORY}")

        # Carry-forward default: do not touch registration_* unless we actually process the row.
        # registration_status is expected to be 'pending' until we complete processing for eligible rows.

        vstat = row.get("validation_status")
        dstat = row.get("dedup_status")

        if vstat != "passed" or dstat != "passed":
            skipped_rows += 1
            updated_upload_rows.append(row)
            continue

        eligible_rows += 1

        # Process eligible row with per-row error capture (do not fail entire shard for one bad row)
        copied_dst_keys: List[str] = []
        try:
            image_id = row.get("image_id")
            if not image_id:
                raise RuntimeError("Missing image_id")

            temp_image_uri = row.get("temp_source_ref")
            if not temp_image_uri:
                raise RuntimeError("Missing temp_source_ref")

            # Build canonical destinations
            canonical_image_key, canonical_image_uri = build_canonical_image_dest(
                FILE_BUCKET_NAME, DATA_SOURCE, image_id, temp_image_uri
            )
            copied_dst_keys.append(canonical_image_key)

            label_dst_keys, label_dst_uris, label_uuid = build_canonical_label_dests(
                FILE_BUCKET_NAME,
                LABEL_TYPE,
                row.get("temp_source_ref_bbox_meta"),
                row.get("temp_source_ref_semantic_png"),
                row.get("temp_source_ref_semantic_meta"),
                row.get("temp_source_ref_instance_png"),
                row.get("temp_source_ref_instance_meta"),
            )

            copied_dst_keys.extend(label_dst_keys)

            # Build copy plan (source URIs -> canonical keys)
            copy_plan: List[Tuple[str, str, str, str]] = []

            # image copy
            src_b, src_k = parse_s3_uri(temp_image_uri, TASK_NAME)
            copy_plan.append((src_b, src_k, FILE_BUCKET_NAME, canonical_image_key))

            # label copies (if any)
            if LABEL_TYPE == "object-detection":
                src_b, src_k = parse_s3_uri(row.get("temp_source_ref_bbox_meta"), TASK_NAME)
                # only one dest
                copy_plan.append((src_b, src_k, FILE_BUCKET_NAME, label_dst_keys[0]))

            elif LABEL_TYPE == "semantic-segmentation":
                src_b1, src_k1 = parse_s3_uri(row.get("temp_source_ref_semantic_png"), TASK_NAME)
                src_b2, src_k2 = parse_s3_uri(row.get("temp_source_ref_semantic_meta"), TASK_NAME)
                # dst keys include png and json
                dst_png = next(k for k in label_dst_keys if k.endswith(".png"))
                dst_json = next(k for k in label_dst_keys if k.endswith(".json"))
                copy_plan.append((src_b1, src_k1, FILE_BUCKET_NAME, dst_png))
                copy_plan.append((src_b2, src_k2, FILE_BUCKET_NAME, dst_json))

            elif LABEL_TYPE == "instance-segmentation":
                src_b1, src_k1 = parse_s3_uri(row.get("temp_source_ref_instance_png"), TASK_NAME)
                src_b2, src_k2 = parse_s3_uri(row.get("temp_source_ref_instance_meta"), TASK_NAME)
                dst_png = next(k for k in label_dst_keys if k.endswith(".png"))
                dst_json = next(k for k in label_dst_keys if k.endswith(".json"))
                copy_plan.append((src_b1, src_k1, FILE_BUCKET_NAME, dst_png))
                copy_plan.append((src_b2, src_k2, FILE_BUCKET_NAME, dst_json))

            else:
                # single/multi label: no label copies
                pass

            # Perform all copies; if any fails, raise and cleanup
            copy_objects_or_raise(copy_plan)

            # Build canonical imagery + label table rows
            canon_img_row = build_canonical_imagery_row(
                DATA_SOURCE, row, canonical_image_uri, LABEL_TYPE, REGISTRATION_TIME, label_uuid
            )

            label_row = None
            if LABEL_TYPE in ("object-detection", "semantic-segmentation", "instance-segmentation"):
                if not label_uuid:
                    raise RuntimeError("label_uuid could not be determined for label-type requiring labels")
                label_row = build_label_table_row(
                    FILE_BUCKET_NAME,
                    LABEL_TYPE,
                    image_id,
                    label_uuid,
                    label_dst_uris,
                    row.get("classes_present"),
                )

            # Mark upload staging row as successfully registered
            row["registration_status"] = "passed"
            row["registration_error"] = None
            reg_passed += 1

            canonical_imagery_rows.append(canon_img_row)
            if label_row:
                canonical_label_rows.append(label_row)

        except Exception as e:
            # best-effort cleanup of anything we copied for this row
            cleanup_copied_best_effort(FILE_BUCKET_NAME, copied_dst_keys)

            row["registration_status"] = "failed"
            row["registration_error"] = str(e)
            reg_failed += 1

        updated_upload_rows.append(row)

    summary = {
        "job_id": JOB_ID,
        "shard_name": shard_name,
        "label_type": LABEL_TYPE,
        "rows_read": total_rows,
        "eligible_rows": eligible_rows,
        "skipped_rows": skipped_rows,
        "registration_passed": reg_passed,
        "registration_failed": reg_failed,
        "canonical_imagery_rows": len(canonical_imagery_rows),
        "canonical_label_rows": len(canonical_label_rows),
    }

    return updated_upload_rows, canonical_imagery_rows, canonical_label_rows, summary

def write_outputs(shard_name: str,
                  updated_upload_rows: List[Dict[str, Any]],
                  canonical_imagery_rows: List[Dict[str, Any]],
                  canonical_label_rows: List[Dict[str, Any]],
                  summary: Dict[str, Any]) -> None:

    bucket = FILE_BUCKET_NAME

    upload_key = f"{PROCESSED_PREFIX}/upload_staging/shard-{shard_name}.jsonl"
    imagery_key = f"{PROCESSED_PREFIX}/canonical_imagery/shard-{shard_name}.jsonl"
    labels_key = f"{PROCESSED_PREFIX}/canonical_labels/shard-{shard_name}.jsonl"

    summary_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-summary.json"
    success_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-SUCCESS"

    write_s3_obj(bucket, upload_key, jsonl_content_converter(updated_upload_rows), "application/x-ndjson", TASK_NAME)
    write_s3_obj(bucket, imagery_key, jsonl_content_converter(canonical_imagery_rows), "application/x-ndjson", TASK_NAME)

    # always write a labels file (may be empty) for predictable ingest logic
    write_s3_obj(bucket, labels_key, jsonl_content_converter(canonical_label_rows), "application/x-ndjson", TASK_NAME)

    write_s3_obj(bucket, summary_key, json.dumps(summary, separators=(",", ":"), ensure_ascii=False) + "\n", "application/json", TASK_NAME)
    write_s3_obj(bucket, success_key, b"", "text/plain", TASK_NAME)

def main():
    start = time.time()

    mb, mk = parse_s3_uri(MANIFEST_S3_URI, TASK_NAME)
    manifest = s3_read_json(mb, mk, TASK_NAME)

    shard_name = manifest["shard_prefix"]

    # Start log (one line)
    log(JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Start shard={shard_name} manifest={MANIFEST_S3_URI} label_type={LABEL_TYPE} pyarrow={pa.__version__}")

    try:
        updated_upload_rows, canonical_imagery_rows, canonical_label_rows, summary = process_manifest(manifest)
        write_outputs(shard_name, updated_upload_rows, canonical_imagery_rows, canonical_label_rows, summary)

        elapsed = time.time() - start
        # Finish log (one line)
        log(JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Finish shard={shard_name} rows={summary['rows_read']} eligible={summary['eligible_rows']} "
            f"passed={summary['registration_passed']} failed={summary['registration_failed']} "
            f"canon_imagery={summary['canonical_imagery_rows']} canon_labels={summary['canonical_label_rows']} "
            f"time_s={elapsed:.1f}")

    except Exception as e:
        # Error log (one line)
        log(JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} ERROR shard={shard_name} manifest={MANIFEST_S3_URI}: {e}", level="error")
        raise

if __name__ == "__main__":
    main()
