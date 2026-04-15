#!/usr/bin/env python3
import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import boto3
from botocore.exceptions import ClientError

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import (
    write_s3_obj,
    parse_s3_uri,
    s3_read_json,
)
from common.general_utils.s3fs_utils import (
    read_parquet_rows_from_s3_uris,
    jsonl_stream_to_s3,
)

from helpers import (
    build_canonical_image_dest,
    build_canonical_label_dests_by_fingerprint,
    build_canonical_imagery_row,
    build_canonical_label_table_row,
    build_image_label_rows,
    build_image_source_membership_row,
    copy_objects_or_raise,
    fingerprint_owner_shard_id,
    build_owner_label_output_key,
    sha_mapping_row
)

MANIFEST_S3_URI = os.environ["MANIFEST_S3_URI"].strip()
JOB_ID = os.environ["JOB_ID"]
USER = os.environ["USER"]
LABEL_TYPE = os.environ["LABEL_TYPE"]
DATA_SOURCE = os.environ["DATA_SOURCE"]

SOURCE_SPLIT = os.environ["SOURCE_SPLIT"].strip().lower()
if SOURCE_SPLIT == "__none__":
    SOURCE_SPLIT = None

if SOURCE_SPLIT not in {None, "train", "val", "test"}:
    raise ValueError(f"Invalid SOURCE_SPLIT env value: {SOURCE_SPLIT!r}")

PATH_PREFIX = os.environ["PATH_PREFIX"]
EVENT_TYPE = os.environ["EVENT_TYPE"]
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
SHA256_TABLE_NAME = os.environ["SHA256_TABLE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
REGISTRATION_TIME = os.environ["REGISTRATION_TIME"]

TASK_NAME = "[REG_JOB_DEF]"
STAGE_NAME = "registration-batch"

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
OWNER_SHARDS = 512  # keep aligned with batching MAX_SHARDS

ddb = boto3.client("dynamodb")
s3 = boto3.client("s3")

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _active_marker_key(shard_name: str) -> str:
    return f"temp/image-upload/{JOB_ID}/worker-markers/{STAGE_NAME}/active/{shard_name}.json"

def _completed_marker_key(shard_name: str) -> str:
    return f"temp/image-upload/{JOB_ID}/worker-markers/{STAGE_NAME}/completed/{shard_name}.json"

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

def _rollback_seed_key(shard_name: str) -> str:
    return f"{PROCESSED_PREFIX}/rollback-batch/shard-{shard_name}.json"

def _write_batch_rollback_seed(
    *,
    shard_name: str,
    new_image_ids: List[str],
    canonical_image_keys_to_delete: List[str],
    canonical_label_keys_to_delete: List[str],
    sha256_mappings_to_delete: List[Dict[str, str]],
) -> str:
    payload = {
        "job_id": JOB_ID,
        "shard": shard_name,
        "kind": "registration_batch",
        "new_image_ids": sorted(set(new_image_ids)),
        "canonical_image_keys_to_delete": sorted(set(canonical_image_keys_to_delete)),
        "canonical_label_keys_to_delete": sorted(set(canonical_label_keys_to_delete)),
        "sha256_mappings_to_delete": sorted(
            sha256_mappings_to_delete,
            key=lambda r: (r["sha256"], r["image_id"]),
        ),
    }

    key = _rollback_seed_key(shard_name)
    write_s3_obj(
        FILE_BUCKET_NAME,
        key,
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n",
        "application/json",
        TASK_NAME,
    )
    return key

def put_sha256_mapping_idempotent(sha256_hash: str, image_id: str) -> None:
    """
    Register sha256 -> canonical image_id mapping for NEW canonical images only.
    Uses a conditional put to avoid overwriting an existing mapping.
    """
    if not sha256_hash or not image_id:
        raise RuntimeError(
            f"{TASK_NAME} cannot write sha256 mapping: missing sha256_hash or image_id"
        )

    try:
        ddb.put_item(
            TableName=SHA256_TABLE_NAME,
            Item={
                "sha256": {"S": sha256_hash},
                "image_id": {"S": image_id},
            },
            ConditionExpression="attribute_not_exists(sha256)",
        )
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise

        resp = ddb.get_item(
            TableName=SHA256_TABLE_NAME,
            Key={"sha256": {"S": sha256_hash}},
            ConsistentRead=True,
        )
        existing = resp.get("Item", {}).get("image_id", {}).get("S")

        if not existing:
            raise RuntimeError(
                f"{TASK_NAME} sha256_hash exists but could not read existing image_id"
            )

        if existing == image_id:
            return

        raise RuntimeError(
            f"{TASK_NAME} sha256_hash already mapped to a different image_id: {existing}"
        )

def _execute_side_effects(
    *,
    copy_plan_all: List[Tuple[str, str, str, str]],
    sha_mappings_to_put: List[Dict[str, str]],
) -> None:
    if copy_plan_all:
        copy_objects_or_raise(copy_plan_all)

    for row in sha_mappings_to_put:
        put_sha256_mapping_idempotent(
            sha256_hash=row["sha256"],
            image_id=row["image_id"],
        )

def plan_manifest(manifest: Dict[str, Any]) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Tuple[str, str, str, str]],   # copy_plan_all
    List[Dict[str, str]],              # sha_mappings_to_put
    List[str],                         # new_image_ids
    List[str],                         # canonical_image_keys_to_delete
    List[str],                         # canonical_label_keys_to_delete
]:
    """
    Planning only. NO side effects here.
    """
    files = manifest.get("files", []) or []
    shard_name = manifest.get("shard_prefix", "shard")

    total_rows = 0
    registration_candidate_rows = 0
    skipped_rows = 0

    new_canonical_images_registered = 0
    reg_failed = 0

    updated_upload_rows: List[Dict[str, Any]] = []
    canonical_imagery_rows: List[Dict[str, Any]] = []
    image_labels_rows: List[Dict[str, Any]] = []
    image_source_membership_rows: List[Dict[str, Any]] = []

    canonical_label_rows_by_owner: Dict[str, List[Dict[str, Any]]] = {}

    copy_plan_all: List[Tuple[str, str, str, str]] = []
    sha_mappings_to_put: List[Dict[str, str]] = []

    new_image_ids: List[str] = []
    rollback_canonical_image_keys: List[str] = []
    rollback_canonical_label_keys: List[str] = []

    seen_canonical_images: set[str] = set()
    seen_new_image_ids: set[str] = set()
    seen_image_labels: set[Tuple[str, str, str]] = set()
    seen_fingerprints: set[str] = set()
    seen_copy_ops: set[Tuple[str, str, str, str]] = set()
    seen_sha_mappings: set[Tuple[str, str]] = set()

    # One upload job has one data_source/source_split, but dedupe membership rows anyway.
    seen_source_memberships: Dict[Tuple[str, str], Any] = {}

    for row in read_parquet_rows_from_s3_uris(files):
        total_rows += 1
        if total_rows > MAX_ROWS_IN_MEMORY:
            raise RuntimeError(
                f"{TASK_NAME} Shard {shard_name} exceeded MAX_ROWS_IN_MEMORY={MAX_ROWS_IN_MEMORY}"
            )

        vstat = row.get("validation_status")
        dstat = row.get("dedup_status")

        if vstat != "passed":
            skipped_rows += 1
            row["registration_status"] = "failed"
            row["registration_error"] = f"skipped registration because validation_status={vstat}"
            reg_failed += 1
            updated_upload_rows.append(row)
            continue

        if dstat == "internal_duplicate":
            skipped_rows += 1
            row["registration_status"] = "failed"
            row["registration_error"] = "skipped registration because dedup_status=internal_duplicate"
            reg_failed += 1
            updated_upload_rows.append(row)
            continue

        if dstat not in ("passed", "external_duplicate"):
            skipped_rows += 1
            dedup_err = row.get("dedup_error")
            if isinstance(dedup_err, str) and dedup_err.strip():
                row["registration_error"] = (
                    f"skipped registration because dedup failed: {dedup_err.strip()}"
                )
            else:
                row["registration_error"] = f"skipped registration because dedup_status={dstat}"

            row["registration_status"] = "failed"
            reg_failed += 1
            updated_upload_rows.append(row)
            continue

        registration_candidate_rows += 1

        try:
            if dstat in ("passed", "external_duplicate"):
                usable_labels = [
                    lab.strip().lower()
                    for lab in (row.get("string_labels") or [])
                    if isinstance(lab, str) and lab.strip()
                ]

                if LABEL_TYPE in ("single-label", "multi-label") and not usable_labels:
                    raise RuntimeError(f"{TASK_NAME} Missing usable string_labels for {LABEL_TYPE}")

            if dstat == "passed":
                image_id = row.get("image_id")
                if not image_id:
                    raise RuntimeError(f"{TASK_NAME} Missing image_id")

                temp_image_uri = row.get("temp_source_ref")
                if not temp_image_uri:
                    raise RuntimeError(f"{TASK_NAME} Missing temp_source_ref")

                sha256_hash = row.get("sha256_hash")
                if not sha256_hash:
                    raise RuntimeError(f"{TASK_NAME} Missing sha256_hash")

                canonical_image_key, canonical_image_uri = build_canonical_image_dest(
                    FILE_BUCKET_NAME,
                    image_id,
                    temp_image_uri,
                    DATA_SOURCE,
                    PATH_PREFIX,
                )

                copy_plan: List[Tuple[str, str, str, str]] = []
                src_b, src_k = parse_s3_uri(temp_image_uri, TASK_NAME)
                copy_plan.append((src_b, src_k, FILE_BUCKET_NAME, canonical_image_key))
                rollback_canonical_image_keys.append(canonical_image_key)

                if image_id not in seen_new_image_ids:
                    seen_new_image_ids.add(image_id)
                    new_image_ids.append(image_id)

                fingerprint = row.get("label_fingerprint")

                if LABEL_TYPE in ("object-detection", "semantic-segmentation", "instance-segmentation"):
                    if not fingerprint:
                        raise RuntimeError(
                            f"{TASK_NAME} Missing label_fingerprint for label-type requiring label files"
                        )

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

                    copy_plan.extend(label_copy_plan)
                    rollback_canonical_label_keys.extend(label_dst_keys)

                    if fingerprint not in seen_fingerprints:
                        label_row = build_canonical_label_table_row(
                            label_type=LABEL_TYPE,
                            fingerprint=fingerprint,
                            canonical_label_uris=label_dst_uris,
                            classes_present=row.get("classes_present"),
                        )
                        if label_row:
                            owner = fingerprint_owner_shard_id(fingerprint, OWNER_SHARDS)
                            canonical_label_rows_by_owner.setdefault(owner, []).append(label_row)
                            seen_fingerprints.add(fingerprint)

                if image_id not in seen_canonical_images:
                    canon_img_row = build_canonical_imagery_row(
                        row=row,
                        canonical_image_uri=canonical_image_uri,
                        registration_time=REGISTRATION_TIME,
                    )
                    canonical_imagery_rows.append(canon_img_row)
                    seen_canonical_images.add(image_id)

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

                membership_row = build_image_source_membership_row(
                    image_id=image_id,
                    data_source=DATA_SOURCE,
                    source_split=SOURCE_SPLIT,
                )
                membership_key = (membership_row["image_id"], membership_row["data_source"])
                membership_split = membership_row["source_split"]

                if membership_key in seen_source_memberships:
                    prev_split = seen_source_memberships[membership_key]
                    if prev_split != membership_split:
                        raise RuntimeError(
                            f"{TASK_NAME} conflicting source_split for {membership_key}: "
                            f"{prev_split} vs {membership_split}"
                        )
                else:
                    image_source_membership_rows.append(membership_row)
                    seen_source_memberships[membership_key] = membership_split

                for cp in copy_plan:
                    if cp not in seen_copy_ops:
                        seen_copy_ops.add(cp)
                        copy_plan_all.append(cp)

                sha_key = (sha256_hash, image_id)
                if sha_key not in seen_sha_mappings:
                    seen_sha_mappings.add(sha_key)
                    sha_mappings_to_put.append(
                        sha_mapping_row(sha256_hash=sha256_hash, image_id=image_id)
                    )

                row["registration_status"] = "passed"
                row["registration_error"] = None
                new_canonical_images_registered += 1

            elif dstat == "external_duplicate":
                target_image_id = row.get("matched_image_id")
                if not target_image_id:
                    raise RuntimeError(
                        f"{TASK_NAME} Missing matched_image_id for external_duplicate row"
                    )

                fingerprint = row.get("label_fingerprint")

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

                if LABEL_TYPE in ("object-detection", "semantic-segmentation", "instance-segmentation"):
                    if not fingerprint:
                        raise RuntimeError(
                            f"{TASK_NAME} Missing label_fingerprint for external_duplicate structured label"
                        )

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

                    rollback_canonical_label_keys.extend(label_dst_keys)

                    for cp in label_copy_plan:
                        if cp not in seen_copy_ops:
                            seen_copy_ops.add(cp)
                            copy_plan_all.append(cp)

                    if fingerprint not in seen_fingerprints:
                        label_row = build_canonical_label_table_row(
                            label_type=LABEL_TYPE,
                            fingerprint=fingerprint,
                            canonical_label_uris=label_dst_uris,
                            classes_present=row.get("classes_present"),
                        )
                        if label_row:
                            owner = fingerprint_owner_shard_id(fingerprint, OWNER_SHARDS)
                            canonical_label_rows_by_owner.setdefault(owner, []).append(label_row)
                            seen_fingerprints.add(fingerprint)

                membership_row = build_image_source_membership_row(
                    image_id=target_image_id,
                    data_source=DATA_SOURCE,
                    source_split=SOURCE_SPLIT,
                )
                membership_key = (membership_row["image_id"], membership_row["data_source"])
                membership_split = membership_row["source_split"]

                if membership_key in seen_source_memberships:
                    prev_split = seen_source_memberships[membership_key]
                    if prev_split != membership_split:
                        raise RuntimeError(
                            f"{TASK_NAME} conflicting source_split for {membership_key}: "
                            f"{prev_split} vs {membership_split}"
                        )
                else:
                    image_source_membership_rows.append(membership_row)
                    seen_source_memberships[membership_key] = membership_split

            else:
                skipped_rows += 1

        except Exception as e:
            row["registration_status"] = "failed"
            row["registration_error"] = str(e)
            reg_failed += 1

        updated_upload_rows.append(row)

    owner_shards_touched = sorted(canonical_label_rows_by_owner.keys())

    summary = {
        "job_id": JOB_ID,
        "shard_name": shard_name,
        "label_type": LABEL_TYPE,
        "rows_read": total_rows,
        "registration_candidate_rows": registration_candidate_rows,
        "skipped_rows": skipped_rows,
        "new_canonical_images_registered": new_canonical_images_registered,
        "registration_failed": reg_failed,
        "canonical_imagery_rows": len(canonical_imagery_rows),
        "image_labels_rows": len(image_labels_rows),
        "image_source_membership_rows": len(image_source_membership_rows),
        "canonical_label_owner_shards_touched": owner_shards_touched,
        "canonical_label_rows_total": sum(len(v) for v in canonical_label_rows_by_owner.values()),
    }

    return (
        updated_upload_rows,
        canonical_imagery_rows,
        canonical_label_rows_by_owner,
        image_labels_rows,
        image_source_membership_rows,
        summary,
        copy_plan_all,
        sha_mappings_to_put,
        new_image_ids,
        rollback_canonical_image_keys,
        rollback_canonical_label_keys,
    )

def write_outputs(
    shard_name: str,
    updated_upload_rows: List[Dict[str, Any]],
    canonical_imagery_rows: List[Dict[str, Any]],
    canonical_label_rows_by_owner: Dict[str, List[Dict[str, Any]]],
    image_labels_rows: List[Dict[str, Any]],
    image_source_membership_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    bucket = FILE_BUCKET_NAME

    upload_key = f"{PROCESSED_PREFIX}/upload_staging/shard-{shard_name}.jsonl"
    imagery_key = f"{PROCESSED_PREFIX}/canonical_imagery/shard-{shard_name}.jsonl"
    image_labels_key = f"{PROCESSED_PREFIX}/image_labels/shard-{shard_name}.jsonl"
    image_source_membership_key = (
        f"{PROCESSED_PREFIX}/image_source_membership/shard-{shard_name}.jsonl"
    )

    summary_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-summary.json"
    success_key = f"{PROCESSED_PREFIX}/shard-{shard_name}-SUCCESS"

    jsonl_stream_to_s3(bucket, upload_key, updated_upload_rows)
    jsonl_stream_to_s3(bucket, imagery_key, canonical_imagery_rows)
    jsonl_stream_to_s3(bucket, image_labels_key, image_labels_rows)
    jsonl_stream_to_s3(bucket, image_source_membership_key, image_source_membership_rows)

    for owner_shard_id, rows in canonical_label_rows_by_owner.items():
        if not rows:
            continue
        owner_key = build_owner_label_output_key(
            processed_prefix=PROCESSED_PREFIX,
            owner_shard_id=owner_shard_id,
            source_target_shard=shard_name,
        )
        jsonl_stream_to_s3(bucket, owner_key, rows)

    write_s3_obj(
        bucket,
        summary_key,
        json.dumps(summary, separators=(",", ":"), ensure_ascii=False) + "\n",
        "application/json",
        TASK_NAME,
    )
    write_s3_obj(bucket, success_key, b"", "text/plain", TASK_NAME)

def main():
    start = time.time()

    mb, mk = parse_s3_uri(MANIFEST_S3_URI, TASK_NAME)
    manifest = s3_read_json(mb, mk, TASK_NAME)

    shard_name = manifest.get("shard_prefix", "shard")
    active_key = _active_marker_key(shard_name)
    completed_key = _completed_marker_key(shard_name)

    _write_json_marker(
        active_key,
        {
            "job_id": JOB_ID,
            "stage": STAGE_NAME,
            "shard": shard_name,
            "request_id": None,
            "started_at": _iso_now(),
            "manifest_s3_uri": MANIFEST_S3_URI,
            "label_type": LABEL_TYPE,
            "data_source": DATA_SOURCE,
            "source_split": SOURCE_SPLIT,
        },
    )

    log(
        JOB_ID,
        USER,
        EVENT_TYPE,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Start shard={shard_name} manifest={MANIFEST_S3_URI} label_type={LABEL_TYPE}",
    )

    try:
        (
            updated_upload_rows,
            canonical_imagery_rows,
            canonical_label_rows_by_owner,
            image_labels_rows,
            image_source_membership_rows,
            summary,
            copy_plan_all,
            sha_mappings_to_put,
            new_image_ids,
            rollback_canonical_image_keys,
            rollback_canonical_label_keys,
        ) = plan_manifest(manifest)

        rollback_seed_key = _write_batch_rollback_seed(
            shard_name=shard_name,
            new_image_ids=new_image_ids,
            canonical_image_keys_to_delete=rollback_canonical_image_keys,
            canonical_label_keys_to_delete=rollback_canonical_label_keys,
            sha256_mappings_to_delete=sha_mappings_to_put,
        )

        _execute_side_effects(
            copy_plan_all=copy_plan_all,
            sha_mappings_to_put=sha_mappings_to_put,
        )

        write_outputs(
            shard_name,
            updated_upload_rows,
            canonical_imagery_rows,
            canonical_label_rows_by_owner,
            image_labels_rows,
            image_source_membership_rows,
            summary,
        )

        _write_json_marker(
            completed_key,
            {
                "job_id": JOB_ID,
                "stage": STAGE_NAME,
                "shard": shard_name,
                "completed_at": _iso_now(),
                "rollback_seed_key": rollback_seed_key,
            },
        )

        elapsed = time.time() - start
        log(
            JOB_ID,
            USER,
            EVENT_TYPE,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Finish shard={shard_name} rows={summary['rows_read']} "
            f"registration_candidate_rows={summary['registration_candidate_rows']} "
            f"new_canonical_images_registered={summary['new_canonical_images_registered']} "
            f"registration_failed={summary['registration_failed']} "
            f"canon_imagery={summary['canonical_imagery_rows']} "
            f"image_labels={summary['image_labels_rows']} "
            f"image_source_membership={summary['image_source_membership_rows']} "
            f"canon_label_rows={summary['canonical_label_rows_total']} "
            f"owner_shards={len(summary['canonical_label_owner_shards_touched'])} "
            f"new_image_ids={len(new_image_ids)} "
            f"rollback_seed=s3://{FILE_BUCKET_NAME}/{rollback_seed_key} "
            f"time_s={elapsed:.1f}",
        )

    except Exception as e:
        log(
            JOB_ID,
            USER,
            EVENT_TYPE,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} ERROR shard={shard_name} manifest={MANIFEST_S3_URI}: {e}",
            level="error",
        )
        raise

    finally:
        _delete_marker_best_effort(active_key)

if __name__ == "__main__":
    main()