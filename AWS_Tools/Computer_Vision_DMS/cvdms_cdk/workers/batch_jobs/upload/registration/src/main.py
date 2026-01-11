#!/usr/bin/env python3
import os
import json
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import boto3
import pyarrow as pa
import s3fs

from common.logging_utils import log
from common.s3_utils import (
    write_s3_obj,
    parse_s3_uri,
    s3_read_json,
    read_parquet_rows_from_s3_uris,
)

from helpers import (
    build_canonical_image_dest,
    build_canonical_label_dests_by_fingerprint,
    build_canonical_imagery_row,
    build_canonical_label_table_row,
    build_image_label_rows,
    copy_objects_or_raise,
    cleanup_copied_best_effort,
    jsonl_stream_to_s3,
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
if not SHA256_TABLE_NAME:
    raise RuntimeError(f"{TASK_NAME} SHA256_TABLE_NAME not set")

PROCESSED_PREFIX = f"temp/image-upload/{JOB_ID}/batches/registration-step/processed"

MAX_ROWS_IN_MEMORY = 200000

s3 = boto3.client("s3")
ddb = boto3.client("dynamodb")


def _ddb_put_sha256_mapping_or_raise(sha256_hash: str, image_id: str) -> None:
    """
    Register sha256 -> canonical image_id mapping for NEW canonical images only.
    Uses a conditional put to avoid overwriting an existing mapping.
    """
    if not sha256_hash or not image_id:
        raise RuntimeError("cannot write sha256 mapping: missing sha256_hash or image_id")

    # Assumes SHA256 table partition key is 'sha256_hash' and has attribute 'image_id' stored as S.
    # Adjust attribute names if your DDB table differs.
    ddb.put_item(
        TableName=SHA256_TABLE_NAME,
        Item={
            "sha256_hash": {"S": sha256_hash},
            "image_id": {"S": image_id},
        },
        ConditionExpression="attribute_not_exists(sha256_hash)",
    )


def process_manifest(manifest: Dict[str, Any]) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """
    Returns:
      - updated_upload_rows: list[dict] (rows we read, with registration_status updated when processed)
      - canonical_imagery_rows: list[dict]
      - canonical_label_rows: list[dict] (rows include "__table" routing)
      - image_labels_rows: list[dict]
      - summary: dict
    """
    files = manifest.get("files", []) or []
    shard_name = manifest.get("shard_prefix", "shard")

    total_rows = 0
    eligible_rows = 0
    skipped_rows = 0

    reg_passed = 0
    reg_failed = 0
    reg_enriched = 0
    reg_noop = 0

    updated_upload_rows: List[Dict[str, Any]] = []
    canonical_imagery_rows: List[Dict[str, Any]] = []
    canonical_label_rows: List[Dict[str, Any]] = []
    image_labels_rows: List[Dict[str, Any]] = []

    # Dedup within shard outputs
    seen_canonical_images: set[str] = set()
    seen_canonical_labels: set[str] = set()  # fingerprint ids
    seen_image_labels: set[Tuple[str, str, str]] = set()  # (image_id, label_type, label_id)

    for row in read_parquet_rows_from_s3_uris(files):
        total_rows += 1
        if total_rows > MAX_ROWS_IN_MEMORY:
            raise RuntimeError(f"{TASK_NAME} Shard {shard_name} exceeded MAX_ROWS_IN_MEMORY={MAX_ROWS_IN_MEMORY}")

        vstat = row.get("validation_status")
        dstat = row.get("dedup_status")

        # Registration stage should only be handed eligible rows by CTAS,
        # but we keep this guard in case of manual/testing mistakes.
        if vstat != "passed" or dstat not in ("passed", "external_duplicate", "internal_duplicate"):
            skipped_rows += 1
            updated_upload_rows.append(row)
            continue

        if dstat == "internal_duplicate":
            # Intentionally ignore; keep row unchanged so audit shows it wasn't registered.
            skipped_rows += 1
            updated_upload_rows.append(row)
            continue

        eligible_rows += 1

        copied_dst_keys: List[str] = []
        per_row_emitted_any = False

        try:
            if dstat == "passed":
                # New canonical image
                image_id = row.get("image_id")
                if not image_id:
                    raise RuntimeError("Missing image_id")

                temp_image_uri = row.get("temp_source_ref")
                if not temp_image_uri:
                    raise RuntimeError("Missing temp_source_ref")

                sha256_hash = row.get("sha256_hash")
                if not sha256_hash:
                    raise RuntimeError("Missing sha256_hash")

                # Copy image to canonical location
                canonical_image_key, canonical_image_uri = build_canonical_image_dest(
                    FILE_BUCKET_NAME, image_id, temp_image_uri
                )
                copied_dst_keys.append(canonical_image_key)

                copy_plan: List[Tuple[str, str, str, str]] = []
                src_b, src_k = parse_s3_uri(temp_image_uri, TASK_NAME)
                copy_plan.append((src_b, src_k, FILE_BUCKET_NAME, canonical_image_key))

                # Label handling (files + label table row + image_labels rows)
                fingerprint = row.get("label_fingerprint")
                label_copy_plan: List[Tuple[str, str, str, str]] = []
                label_dst_keys: List[str] = []
                label_dst_uris: List[str] = []
                label_row: Optional[Dict[str, Any]] = None

                if LABEL_TYPE in ("object-detection", "semantic-segmentation", "instance-segmentation"):
                    if not fingerprint:
                        raise RuntimeError("Missing label_fingerprint for label-type requiring label files")

                    label_dst_keys, label_dst_uris, label_copy_plan = build_canonical_label_dests_by_fingerprint(
                        file_bucket=FILE_BUCKET_NAME,
                        label_type=LABEL_TYPE,
                        fingerprint=fingerprint,
                        temp_bbox_meta_uri=row.get("temp_source_ref_bbox_meta"),
                        temp_semantic_png_uri=row.get("temp_source_ref_semantic_png"),
                        temp_semantic_meta_uri=row.get("temp_source_ref_semantic_meta"),
                        temp_instance_png_uri=row.get("temp_source_ref_instance_png"),
                        temp_instance_meta_uri=row.get("temp_source_ref_instance_meta"),
                    )

                    copied_dst_keys.extend(label_dst_keys)
                    copy_plan.extend(label_copy_plan)

                    # Canonical label table row (no image_id in model A)
                    if fingerprint not in seen_canonical_labels:
                        label_row = build_canonical_label_table_row(
                            label_type=LABEL_TYPE,
                            fingerprint=fingerprint,
                            canonical_label_uris=label_dst_uris,
                            classes_present=row.get("classes_present"),
                        )
                        if label_row:
                            canonical_label_rows.append(label_row)
                            seen_canonical_labels.add(fingerprint)

                # Execute all copies for this row
                copy_objects_or_raise(copy_plan)

                # Canonical imagery row (table schema has only base fields)
                if image_id not in seen_canonical_images:
                    canon_img_row = build_canonical_imagery_row(
                        row=row,
                        canonical_image_uri=canonical_image_uri,
                        registration_time=REGISTRATION_TIME,
                    )
                    canonical_imagery_rows.append(canon_img_row)
                    seen_canonical_images.add(image_id)

                # Image_labels rows
                for ilr in build_image_label_rows(
                    job_label_type=LABEL_TYPE,
                    target_image_id=image_id,
                    string_labels=row.get("string_labels"),
                    fingerprint=fingerprint,
                ):
                    key = (ilr["image_id"], ilr["label_type"], ilr["label_id"])
                    if key not in seen_image_labels:
                        image_labels_rows.append(ilr)
                        seen_image_labels.add(key)

                # Register sha256->image_id mapping in DDB (only for NEW canonical images)
                _ddb_put_sha256_mapping_or_raise(sha256_hash=sha256_hash, image_id=image_id)

                row["registration_status"] = "passed"
                row["registration_error"] = None
                reg_passed += 1

            elif dstat == "external_duplicate":
                # Label enrichment against matched_image_id
                target_image_id = row.get("matched_image_id")
                if not target_image_id:
                    raise RuntimeError("Missing matched_image_id for external_duplicate row")

                fingerprint = row.get("label_fingerprint")

                # For external duplicate, we never copy the image and never write canonical_imagery.
                # We may copy/register labels (OD/seg) if needed, and we always emit image_labels mappings
                # (deduped within shard).

                # 1) Image_labels mappings
                emitted_this_row = 0
                for ilr in build_image_label_rows(
                    job_label_type=LABEL_TYPE,
                    target_image_id=target_image_id,
                    string_labels=row.get("string_labels"),
                    fingerprint=fingerprint,
                ):
                    key = (ilr["image_id"], ilr["label_type"], ilr["label_id"])
                    if key not in seen_image_labels:
                        image_labels_rows.append(ilr)
                        seen_image_labels.add(key)
                        emitted_this_row += 1

                # 2) If this is a structured label type, also ensure label files exist in canonical paths
                #    and emit canonical label row once per fingerprint (per shard).
                copied_dst_keys = []
                if LABEL_TYPE in ("object-detection", "semantic-segmentation", "instance-segmentation"):
                    if not fingerprint:
                        raise RuntimeError("Missing label_fingerprint for external_duplicate structured label")

                    # Copy label files to canonical location (content-addressed by fingerprint)
                    label_dst_keys, label_dst_uris, label_copy_plan = build_canonical_label_dests_by_fingerprint(
                        file_bucket=FILE_BUCKET_NAME,
                        label_type=LABEL_TYPE,
                        fingerprint=fingerprint,
                        temp_bbox_meta_uri=row.get("temp_source_ref_bbox_meta"),
                        temp_semantic_png_uri=row.get("temp_source_ref_semantic_png"),
                        temp_semantic_meta_uri=row.get("temp_source_ref_semantic_meta"),
                        temp_instance_png_uri=row.get("temp_source_ref_instance_png"),
                        temp_instance_meta_uri=row.get("temp_source_ref_instance_meta"),
                    )
                    copied_dst_keys.extend(label_dst_keys)

                    # Copy (safe because deterministic keys; overwrites are acceptable)
                    # NOTE: This may re-copy files even if they exist. That's fine operationally.
                    copy_objects_or_raise(label_copy_plan)

                    # Emit canonical label row only once per fingerprint per shard
                    if fingerprint not in seen_canonical_labels:
                        label_row = build_canonical_label_table_row(
                            label_type=LABEL_TYPE,
                            fingerprint=fingerprint,
                            canonical_label_uris=label_dst_uris,
                            classes_present=row.get("classes_present"),
                        )
                        if label_row:
                            canonical_label_rows.append(label_row)
                            seen_canonical_labels.add(fingerprint)
                            emitted_this_row += 1  # count as "work performed"

                # Status selection
                if emitted_this_row > 0:
                    row["registration_status"] = "enriched"
                    row["registration_error"] = None
                    reg_enriched += 1
                else:
                    row["registration_status"] = "no_op"
                    row["registration_error"] = None
                    reg_noop += 1

            else:
                # Shouldn't happen due to earlier guard
                skipped_rows += 1

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
        "registration_enriched": reg_enriched,
        "registration_no_op": reg_noop,
        "canonical_imagery_rows": len(canonical_imagery_rows),
        "canonical_label_rows": len(canonical_label_rows),
        "image_labels_rows": len(image_labels_rows),
    }

    return updated_upload_rows, canonical_imagery_rows, canonical_label_rows, image_labels_rows, summary


def write_outputs(
    shard_name: str,
    updated_upload_rows: List[Dict[str, Any]],
    canonical_imagery_rows: List[Dict[str, Any]],
    canonical_label_rows: List[Dict[str, Any]],
    image_labels_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    bucket = FILE_BUCKET_NAME

    upload_key = f"{PROCESSED_PREFIX}/upload_staging/shard-{shard_name}.jsonl"
    imagery_key = f"{PROCESSED_PREFIX}/canonical_imagery/shard-{shard_name}.jsonl"
    labels_key = f"{PROCESSED_PREFIX}/canonical_labels/shard-{shard_name}.jsonl"
    image_labels_key = f"{PROCESSED_PREFIX}/image_labels/shard-{shard_name}.jsonl"

    summary_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-summary.json"
    success_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-SUCCESS"

    # Stream to S3 to avoid huge in-memory strings
    jsonl_stream_to_s3(bucket, upload_key, updated_upload_rows)
    jsonl_stream_to_s3(bucket, imagery_key, canonical_imagery_rows)

    # Always write a labels file (possibly empty) for predictable ingest logic
    jsonl_stream_to_s3(bucket, labels_key, canonical_label_rows)

    # Always write image_labels file (possibly empty)
    jsonl_stream_to_s3(bucket, image_labels_key, image_labels_rows)

    write_s3_obj(bucket, summary_key, json.dumps(summary, separators=(",", ":"), ensure_ascii=False) + "\n",
                 "application/json", TASK_NAME)
    write_s3_obj(bucket, success_key, b"", "text/plain", TASK_NAME)


def main():
    start = time.time()

    mb, mk = parse_s3_uri(MANIFEST_S3_URI, TASK_NAME)
    manifest = s3_read_json(mb, mk, TASK_NAME)

    shard_name = manifest.get("shard_prefix", "shard")

    log(
        JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Start shard={shard_name} manifest={MANIFEST_S3_URI} label_type={LABEL_TYPE} pyarrow={pa.__version__}"
    )

    try:
        updated_upload_rows, canonical_imagery_rows, canonical_label_rows, image_labels_rows, summary = process_manifest(manifest)
        write_outputs(shard_name, updated_upload_rows, canonical_imagery_rows, canonical_label_rows, image_labels_rows, summary)

        elapsed = time.time() - start
        log(
            JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Finish shard={shard_name} rows={summary['rows_read']} eligible={summary['eligible_rows']} "
            f"passed={summary['registration_passed']} enriched={summary['registration_enriched']} no_op={summary['registration_no_op']} "
            f"failed={summary['registration_failed']} canon_imagery={summary['canonical_imagery_rows']} "
            f"canon_labels={summary['canonical_label_rows']} image_labels={summary['image_labels_rows']} "
            f"time_s={elapsed:.1f}"
        )

    except Exception as e:
        log(
            JOB_ID, USER, EVENT_TYPE, LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} ERROR shard={shard_name} manifest={MANIFEST_S3_URI}: {e}",
            level="error"
        )
        raise


if __name__ == "__main__":
    main()
