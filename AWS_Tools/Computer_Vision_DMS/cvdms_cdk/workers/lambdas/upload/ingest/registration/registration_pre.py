#!/usr/bin/env python3
import os
import json
from typing import Dict, List

from common.logging_utils import log
from common.s3_utils import s3_list_keys, s3_read_json, parse_s3_uri

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[REG_INGEST_PRE]"

def extract_expected_shards_from_manifests(manifests: List[str]) -> List[str]:
    """
    Try to extract shard names from batching manifests.
    Expected manifest pattern: .../manifest-shard-<name>.json
    """
    expected: List[str] = []
    for m in manifests:
        _, key = parse_s3_uri(m, TASK_NAME)
        fname = key.split("/")[-1]
        if fname.startswith("manifest-shard-") and fname.endswith(".json"):
            shard_name = fname[len("manifest-shard-") : -len(".json")]
        else:
            shard_name = fname.rsplit(".", 1)[0]
        expected.append(shard_name)

    # stable unique
    seen = set()
    out: List[str] = []
    for s in expected:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def collect_processed_shards(job_id: str, manifests: List[str]) -> Dict:
    """
    Locate per-shard registration processed outputs.

    We expect for each shard:
      - upload_staging/shard-<shard>.jsonl
      - canonical_imagery/shard-<shard>.jsonl
      - canonical_labels/shard-<shard>.jsonl
      - image_labels/shard-<shard>.jsonl
      - shard-<shard>-summary.json
      - shard-<shard>-SUCCESS
    """
    bucket = FILE_BUCKET_NAME
    processed_prefix = f"temp/image-upload/{job_id}/batches/registration-step/processed"
    expected_shards = extract_expected_shards_from_manifests(manifests)

    processed_keys = s3_list_keys(bucket, processed_prefix + "/")

    shard_upload: Dict[str, str] = {}
    shard_imagery: Dict[str, str] = {}
    shard_labels: Dict[str, str] = {}
    shard_image_labels: Dict[str, str] = {}
    shard_summary: Dict[str, str] = {}
    shard_success = set()

    for k in processed_keys:
        name = k.split("/")[-1]

        # Per-table JSONLs live under subfolders
        if name.startswith("shard-") and name.endswith(".jsonl"):
            shard = name[len("shard-") : -len(".jsonl")]
            if "/upload_staging/" in k:
                shard_upload[shard] = k
            elif "/canonical_imagery/" in k:
                shard_imagery[shard] = k
            elif "/canonical_labels/" in k:
                shard_labels[shard] = k
            elif "/image_labels/" in k:
                shard_image_labels[shard] = k

        # Summary/SUCCESS live at the processed_prefix root
        elif name.startswith("shard-") and name.endswith("-summary.json"):
            shard = name[len("shard-") : -len("-summary.json")]
            shard_summary[shard] = k

        elif name.startswith("shard-") and name.endswith("-SUCCESS"):
            shard = name[len("shard-") : -len("-SUCCESS")]
            shard_success.add(shard)

    # If manifest parsing failed, infer from discovered shards
    if not expected_shards:
        expected_shards = sorted(
            set(shard_upload)
            | set(shard_imagery)
            | set(shard_labels)
            | set(shard_image_labels)
            | set(shard_summary)
            | set(shard_success)
        )

    missing: List[str] = []
    shards: List[Dict] = []

    total_rows_read = 0
    total_canon_imagery_rows = 0
    total_canon_label_rows = 0
    total_image_labels_rows = 0

    for shard in expected_shards:
        up_k = shard_upload.get(shard)
        img_k = shard_imagery.get(shard)
        lab_k = shard_labels.get(shard)
        img_lab_k = shard_image_labels.get(shard)
        sum_k = shard_summary.get(shard)
        ok = shard in shard_success

        if not (up_k and img_k and lab_k and img_lab_k and sum_k and ok):
            missing.append(shard)
            continue

        summary = s3_read_json(bucket, sum_k, TASK_NAME)
        rows_read = int(summary.get("rows_read", 0))
        canon_im_rows = int(summary.get("canonical_imagery_rows", 0))
        canon_lbl_rows = int(summary.get("canonical_label_rows", 0))
        img_lbl_rows = int(summary.get("image_labels_rows", 0))

        total_rows_read += rows_read
        total_canon_imagery_rows += canon_im_rows
        total_canon_label_rows += canon_lbl_rows
        total_image_labels_rows += img_lbl_rows

        shards.append(
            {
                "shard": shard,
                "rows_read": rows_read,
                "canonical_imagery_rows": canon_im_rows,
                "canonical_label_rows": canon_lbl_rows,
                "image_labels_rows": img_lbl_rows,
                "upload_staging_key": up_k,
                "canonical_imagery_key": img_k,
                "canonical_labels_key": lab_k,
                "image_labels_key": img_lab_k,
            }
        )

    return {
        "missing_shards": missing,
        "shards": shards,
        "total_rows_read": total_rows_read,
        "total_canon_imagery_rows": total_canon_imagery_rows,
        "total_canon_label_rows": total_canon_label_rows,
        "total_image_labels_rows": total_image_labels_rows,
        "processed_prefix": processed_prefix,
        "expected_shards": expected_shards
    }

def handler(event, context):
    """
    Expected input:
      {
        job_id, user, event_type, manifests,
        label_type?, data_source?,
        expected_count  (eligible_rows from reg batching)
      }
    """
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        manifests = event["manifests"]
        expected_count = int(event["expected_count"])
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} missing job_id")
    if not manifests or not isinstance(manifests, list):
        raise RuntimeError(f"{TASK_NAME} manifests must be a non-empty list of s3 URIs")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Starting registration pre-ingest for job {job_id}")

    collected = collect_processed_shards(job_id, manifests)

    missing = collected["missing_shards"]
    if missing:
        err = f"{TASK_NAME} Missing processed outputs for shards: {missing}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    shards = collected["shards"]
    total_rows_read = int(collected["total_rows_read"])

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Collected {len(shards)} shard outputs. total_rows_read={total_rows_read} "
        f"canon_imagery_rows={collected['total_canon_imagery_rows']} "
        f"canon_label_rows={collected['total_canon_label_rows']} "
        f"image_labels_rows={collected['total_image_labels_rows']}")

    # Validate against expected eligible row count (best check for registration)
    if total_rows_read != expected_count:
        raise RuntimeError(
            f"{TASK_NAME} Eligible row count mismatch: expected_count={expected_count}, "
            f"workers total_rows_read={total_rows_read}"
        )

    # Precompute CTAS temp table name for post step
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    ctas_table_name = f"reg_export_{sanitized_job_id}"

    return {
        "shards": shards,
        "expected_count": expected_count,
        "total_rows_read": total_rows_read,
        "total_canon_imagery_rows": int(collected["total_canon_imagery_rows"]),
        "total_canon_label_rows": int(collected["total_canon_label_rows"]),
        "total_image_labels_rows": int(collected["total_image_labels_rows"]),
        "processed_prefix": collected.get("processed_prefix"),
        "ctas_table_name": ctas_table_name,
    }