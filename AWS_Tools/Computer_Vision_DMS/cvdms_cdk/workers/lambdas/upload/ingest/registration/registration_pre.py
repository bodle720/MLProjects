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
        try:
            _, key = parse_s3_uri(m, TASK_NAME)
        except Exception:
            raise

        fname = key.split("/")[-1]
        if fname.startswith("manifest-shard-") and fname.endswith(".json"):
            shard_name = fname[len("manifest-shard-"): -len(".json")]
        else:
            shard_name = fname.rsplit(".", 1)[0]

        expected.append(shard_name)

    # stable unique
    seen = set()
    out = []
    for s in expected:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _extract_owner_shard_id_from_key(k: str) -> str:
    # expects .../canonical_labels_by_fingerprint/owner-XXXXXX/part-....jsonl
    parts = k.split("/")
    for p in parts:
        if p.startswith("owner-"):
            return p[len("owner-") :]
    return ""

def collect_processed_shards(job_id: str, manifests: List[str], user: str, event_type: str) -> Dict:
    """
    Locate per-shard registration processed outputs.

    Target-image shards (same as before, minus canonical_labels):
      - upload_staging/shard-<shard>.jsonl
      - canonical_imagery/shard-<shard>.jsonl
      - image_labels/shard-<shard>.jsonl
      - shard-<shard>-summary.json
      - shard-<shard>-SUCCESS

    Canonical label rows are now fingerprint-owner sharded:
      - canonical_labels_by_fingerprint/owner-<owner>/part-<targetShard>.jsonl
      (no SUCCESS marker per owner shard)
    """
    bucket = FILE_BUCKET_NAME
    processed_prefix = f"temp/image-upload/{job_id}/batches/registration-step/processed"
    expected_target_shards = extract_expected_shards_from_manifests(manifests)

    processed_keys = s3_list_keys(bucket, processed_prefix + "/")

    # ---- target-image shard outputs ----
    shard_upload: Dict[str, str] = {}
    shard_imagery: Dict[str, str] = {}
    shard_image_labels: Dict[str, str] = {}
    shard_summary: Dict[str, str] = {}
    shard_success = set()

    # ---- fingerprint-owner outputs ----
    # owner_id -> list of jsonl keys (parts)
    owner_parts: Dict[str, List[str]] = {}

    for k in processed_keys:
        name = k.split("/")[-1]

        # Canonical label owner-part jsonls
        if "/canonical_labels_by_fingerprint/" in k and name.endswith(".jsonl"):
            owner_id = _extract_owner_shard_id_from_key(k)
            if owner_id:
                owner_parts.setdefault(owner_id, []).append(k)
            continue

        # Per-table JSONLs live under subfolders (target shards)
        if name.startswith("shard-") and name.endswith(".jsonl"):
            shard = name[len("shard-") : -len(".jsonl")]
            if "/upload_staging/" in k:
                shard_upload[shard] = k
            elif "/canonical_imagery/" in k:
                shard_imagery[shard] = k
            elif "/image_labels/" in k:
                shard_image_labels[shard] = k

        # Summary/SUCCESS live at the processed_prefix root
        elif name.startswith("shard-") and name.endswith("-summary.json"):
            shard = name[len("shard-") : -len("-summary.json")]
            shard_summary[shard] = k

        elif name.startswith("shard-") and name.endswith("-SUCCESS"):
            shard = name[len("shard-") : -len("-SUCCESS")]
            shard_success.add(shard)

    # If manifest parsing failed, infer target shards from discovered keys
    if not expected_target_shards:
        expected_target_shards = sorted(
            set(shard_upload)
            | set(shard_imagery)
            | set(shard_image_labels)
            | set(shard_summary)
            | set(shard_success)
        )

    # sort owner part lists for stable ordering
    for owner_id in owner_parts:
        owner_parts[owner_id] = sorted(owner_parts[owner_id])

    missing_target: List[str] = []
    shards: List[Dict] = []

    total_rows_read = 0
    total_canon_imagery_rows = 0
    total_image_labels_rows = 0

    # canonical label totals now come from summary["canonical_label_rows_total"]
    total_canon_label_rows = 0

    expected_owner_shards_from_summaries = set()

    # ---- build target-image shard items (Map kind=target) ----
    for shard in expected_target_shards:
        up_k = shard_upload.get(shard)
        img_k = shard_imagery.get(shard)
        img_lab_k = shard_image_labels.get(shard)
        sum_k = shard_summary.get(shard)
        ok = shard in shard_success

        # NOTE: no canonical_labels file required anymore
        if not (up_k and img_k and img_lab_k and sum_k and ok):
            missing_target.append(shard)
            continue

        summary = s3_read_json(bucket, sum_k, TASK_NAME)
        rows_read = int(summary.get("rows_read", 0))
        canon_im_rows = int(summary.get("canonical_imagery_rows", 0))
        img_lbl_rows = int(summary.get("image_labels_rows", 0))
        canon_lbl_rows = int(summary.get("canonical_label_rows_total", 0))  # <-- changed

        owner_shards_touched = summary.get("canonical_label_owner_shards_touched", [])
        if isinstance(owner_shards_touched, list):
            for o in owner_shards_touched:
                if o:
                    expected_owner_shards_from_summaries.add(str(o).rjust(6, "0"))

        total_rows_read += rows_read
        total_canon_imagery_rows += canon_im_rows
        total_image_labels_rows += img_lbl_rows
        total_canon_label_rows += canon_lbl_rows

        shards.append(
            {
                "kind": "target",
                "shard": shard,
                "rows_read": rows_read,
                "canonical_imagery_rows": canon_im_rows,
                "image_labels_rows": img_lbl_rows,
                "upload_staging_key": up_k,
                "canonical_imagery_key": img_k,
                "canonical_labels_key": None,  # <-- important
                "image_labels_key": img_lab_k,
            }
        )

    # ---- build fingerprint-owner shard items (Map kind=label_owner) ----
    # Each item points at the owner prefix; shard ingest can list all jsonl under it.
    owner_shards: List[str] = sorted(owner_parts.keys())

    actual_owner_shards = set(owner_shards)
    missing_owner_shards = expected_owner_shards_from_summaries - actual_owner_shards

    if missing_owner_shards:
        raise RuntimeError(
            f"{TASK_NAME} Missing canonical label owner shard outputs: {sorted(missing_owner_shards)}. "
            f"Expected from worker summaries but not found in processed prefix."
        )

    unexpected_owner_shards = actual_owner_shards - expected_owner_shards_from_summaries
    if unexpected_owner_shards:
        log(job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Warning: unexpected owner shard outputs discovered: {sorted(unexpected_owner_shards)}",
            level="warning")

    total_owner_label_files = 0
    for owner_id in owner_shards:
        parts = owner_parts.get(owner_id, [])
        if not parts:
            continue
        total_owner_label_files += len(parts)

        owner_prefix = f"{processed_prefix}/canonical_labels_by_fingerprint/owner-{owner_id}/"

        shards.append(
            {
                "kind": "label_owner",
                "shard": f"owner-{owner_id}",
                "owner_shard_id": owner_id,
                "owner_prefix": owner_prefix,
                "parts_count": len(parts),
                # Pattern A keys: only canonical_labels_key is used for this kind
                "upload_staging_key": None,
                "canonical_imagery_key": None,
                "canonical_labels_key": owner_prefix,  # prefix, not a file
                "image_labels_key": None,
            }
        )

    return {
        "missing_target_shards": missing_target,
        "shards": shards,
        "total_rows_read": total_rows_read,
        "total_canon_imagery_rows": total_canon_imagery_rows,
        "total_canon_label_rows": total_canon_label_rows,
        "total_image_labels_rows": total_image_labels_rows,
        "processed_prefix": processed_prefix,
        "expected_target_shards": expected_target_shards,
        "owner_shards": owner_shards,
        "total_owner_label_files": total_owner_label_files,
    }

def handler(event, context):
    """
    Expected input:
      {
        job_id, user, event_type, manifests,
        label_type?, data_source?,
        total_rows  (total_rows from reg batching)
        Note: Ingest stage injects expected_count.$ from $.registrationStage.total_rows, hence the naming mismatch. Wiring is correct.
      }
    """
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        manifests = event["manifests"]
        total_rows = int(event["expected_count"])
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    if not job_id or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} missing job_id")
    if not manifests or not isinstance(manifests, list):
        raise RuntimeError(f"{TASK_NAME} manifests must be a non-empty list of s3 URIs")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Starting registration pre-ingest for job {job_id}")

    collected = collect_processed_shards(job_id, manifests, user, event_type)

    missing = collected["missing_target_shards"]
    if missing:
        err = f"{TASK_NAME} Missing processed outputs for target shards: {missing}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    shards = collected["shards"]
    total_rows_read = int(collected["total_rows_read"])

    log(job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Collected {len(shards)} map items "
        f"(target_shards={len(collected['expected_target_shards'])}, owner_shards={len(collected['owner_shards'])}). "
        f"total_rows_read={total_rows_read} "
        f"canon_imagery_rows={collected['total_canon_imagery_rows']} "
        f"canon_label_rows_total={collected['total_canon_label_rows']} "
        f"image_labels_rows={collected['total_image_labels_rows']} "
        f"owner_label_files={collected['total_owner_label_files']}")

    # Validate against total row count (best check for registration)
    if total_rows_read != total_rows:
        raise RuntimeError(
            f"{TASK_NAME} Total row count mismatch: total_rows={total_rows}, "
            f"workers total_rows_read={total_rows_read}"
        )

    # Precompute CTAS temp table name for post step
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    ctas_table_name = f"reg_export_{sanitized_job_id}"

    return {
        "shards": shards,  # mixed kinds: target + label_owner
        "total_rows": total_rows,
        "total_rows_read": total_rows_read,
        "total_canon_imagery_rows": int(collected["total_canon_imagery_rows"]),
        "total_canon_label_rows": int(collected["total_canon_label_rows"]),
        "total_image_labels_rows": int(collected["total_image_labels_rows"]),
        "processed_prefix": collected.get("processed_prefix"),
        "ctas_table_name": ctas_table_name,
        "expected_target_shards": collected.get("expected_target_shards"),
        "owner_shards": collected.get("owner_shards"),
    }