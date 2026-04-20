#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Any, Tuple

import boto3

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import (
    s3_list_keys,
    s3_read_json,
    parse_s3_uri,
    read_obj_with_retry,
    write_s3_obj,
)

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
INGEST_HANDOFF_FILE_NAME = os.environ.get("INGEST_HANDOFF_FILE_NAME", "map-items.jsonl")

TASK_NAME = "[REG_INGEST_PRE]"

# --------------------------------------------------------------------
# Grouping knobs.
# Registration mutates the most tables, so keep these conservative.
# --------------------------------------------------------------------
GROUPING_ENABLED = os.environ["GROUPING_ENABLED"].strip().lower() == "true"

# Target-shard grouping (upload_staging + canonical_imagery + image_labels + image_source_membership)
TARGET_TARGET_ROWS = int(os.environ["TARGET_ROWS"])
TARGET_TARGET_BYTES = int(os.environ["TARGET_BYTES"])
MAX_TARGET_ROWS = int(os.environ["MAX_ROWS"])
MAX_TARGET_BYTES = int(os.environ["MAX_BYTES"])

# Owner-shard grouping (canonical_labels_by_fingerprint only)
TARGET_OWNER_BYTES = int(os.environ["TARGET_OWNER_BYTES"])
MAX_OWNER_BYTES = int(os.environ["MAX_OWNER_BYTES"])
TARGET_OWNER_PARTS = int(os.environ["TARGET_OWNER_PARTS"])
MAX_OWNER_PARTS = int(os.environ["MAX_OWNER_PARTS"])

# Guardrail while materializing merged JSONLs in pre-lambda.
MAX_MATERIALIZED_GROUP_BYTES = int(os.environ["MAX_MATERIALIZED_GROUP_BYTES"])

s3 = boto3.client("s3")

def _require_event_key(event: dict, key: str):
    if key not in event:
        raise RuntimeError(f"{TASK_NAME} Missing key: {key!r}, event={json.dumps(event)}")
    return event[key]


def _head_size_bytes(bucket: str, key: str) -> int:
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return int(resp.get("ContentLength", 0))
    except Exception as e:
        raise RuntimeError(f"{TASK_NAME} Failed head_object for s3://{bucket}/{key}: {e}")


def _read_key_bytes(bucket: str, key: str) -> bytes:
    resp = read_obj_with_retry(bucket, key, TASK_NAME)
    if resp is None:
        raise RuntimeError(f"{TASK_NAME} Failed to read s3://{bucket}/{key}")
    data = resp["Body"].read()
    if not isinstance(data, (bytes, bytearray)):
        raise RuntimeError(f"{TASK_NAME} Unexpected non-bytes read from s3://{bucket}/{key}")
    return bytes(data)


def _ensure_trailing_newline(data: bytes) -> bytes:
    if not data:
        return b""
    return data if data.endswith(b"\n") else (data + b"\n")


def _write_bytes_key(key: str, body: bytes, content_type: str = "application/x-ndjson") -> None:
    s3.put_object(
        Bucket=FILE_BUCKET_NAME,
        Key=key,
        Body=body,
        ContentType=content_type,
    )


def read_batch_plan_items(bucket: str, key: str) -> List[Dict[str, Any]]:
    resp = read_obj_with_retry(bucket, key, TASK_NAME)
    if resp is None:
        raise RuntimeError(f"{TASK_NAME} Failed to read batch plan s3://{bucket}/{key}")

    items: List[Dict[str, Any]] = []
    for raw in resp["Body"].iter_lines():
        if not raw:
            continue

        line = raw.decode("utf-8-sig").strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except Exception as e:
            raise RuntimeError(f"{TASK_NAME} Invalid JSONL in batch plan s3://{bucket}/{key}: {e}")

        if not isinstance(obj, dict):
            raise RuntimeError(f"{TASK_NAME} Expected dict items in batch plan, got {type(obj).__name__}")

        manifest = obj.get("manifest")
        if not isinstance(manifest, str) or not manifest.startswith("s3://"):
            raise RuntimeError(f"{TASK_NAME} Batch plan item missing valid manifest URI: {obj}")

        items.append(obj)

    if not items:
        raise RuntimeError(f"{TASK_NAME} Batch plan is empty: s3://{bucket}/{key}")

    return items


def extract_expected_shards_from_batch_items(batch_items: List[Dict[str, Any]]) -> List[str]:
    """
    Prefer explicit shard from the batching handoff item.
    Fallback to manifest filename:
      .../manifest-shard-<name>.json -> <name>
    """
    expected: List[str] = []

    for item in batch_items:
        shard = item.get("shard")
        if isinstance(shard, str) and shard.strip():
            expected.append(shard.strip())
            continue

        manifest_uri = item["manifest"]
        _, key = parse_s3_uri(manifest_uri, TASK_NAME)

        fname = key.split("/")[-1]
        if fname.startswith("manifest-shard-") and fname.endswith(".json"):
            shard_name = fname[len("manifest-shard-") : -len(".json")]
        else:
            shard_name = fname.rsplit(".", 1)[0]

        expected.append(shard_name)

    seen = set()
    out = []
    for s in expected:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _extract_owner_shard_id_from_key(k: str) -> str:
    for part in k.split("/"):
        if part.startswith("owner-"):
            return part[len("owner-") :]
    return ""


def collect_processed_shards(
    job_id: str,
    batch_items: List[Dict[str, Any]],
    user: str,
    event_type: str,
) -> Dict[str, Any]:
    """
    Target-image shards:
      - upload_staging/shard-<shard>.jsonl
      - canonical_imagery/shard-<shard>.jsonl
      - image_labels/shard-<shard>.jsonl
      - image_source_membership/shard-<shard>.jsonl
      - shard-<shard>-summary.json
      - shard-<shard>-SUCCESS

    Canonical label rows are fingerprint-owner sharded:
      - canonical_labels_by_fingerprint/owner-<owner>/part-<targetShard>.jsonl
    """
    bucket = FILE_BUCKET_NAME
    processed_prefix = f"temp/image-upload/{job_id}/batches/registration-step/processed"
    expected_target_shards = extract_expected_shards_from_batch_items(batch_items)

    processed_keys = s3_list_keys(bucket, processed_prefix + "/")

    shard_upload: Dict[str, str] = {}
    shard_imagery: Dict[str, str] = {}
    shard_image_labels: Dict[str, str] = {}
    shard_image_source_membership: Dict[str, str] = {}
    shard_summary: Dict[str, str] = {}
    shard_success = set()

    owner_parts: Dict[str, List[str]] = {}

    for k in processed_keys:
        name = k.split("/")[-1]

        if "/canonical_labels_by_fingerprint/" in k and name.endswith(".jsonl"):
            owner_id = _extract_owner_shard_id_from_key(k)
            if owner_id:
                owner_parts.setdefault(owner_id, []).append(k)
            continue

        if name.startswith("shard-") and name.endswith(".jsonl"):
            shard = name[len("shard-") : -len(".jsonl")]
            if "/upload_staging/" in k:
                shard_upload[shard] = k
            elif "/canonical_imagery/" in k:
                shard_imagery[shard] = k
            elif "/image_labels/" in k:
                shard_image_labels[shard] = k
            elif "/image_source_membership/" in k:
                shard_image_source_membership[shard] = k

        elif name.startswith("shard-") and name.endswith("-summary.json"):
            shard = name[len("shard-") : -len("-summary.json")]
            shard_summary[shard] = k

        elif name.startswith("shard-") and name.endswith("-SUCCESS"):
            shard = name[len("shard-") : -len("-SUCCESS")]
            shard_success.add(shard)

    if not expected_target_shards:
        expected_target_shards = sorted(
            set(shard_upload)
            | set(shard_imagery)
            | set(shard_image_labels)
            | set(shard_image_source_membership)
            | set(shard_summary)
            | set(shard_success)
        )

    for owner_id in owner_parts:
        owner_parts[owner_id] = sorted(owner_parts[owner_id])

    missing_target: List[str] = []
    target_shards: List[Dict[str, Any]] = []
    owner_shard_items: List[Dict[str, Any]] = []

    total_rows_read = 0
    total_canon_imagery_rows = 0
    total_image_labels_rows = 0
    total_image_source_membership_rows = 0
    total_canon_label_rows = 0

    expected_owner_shards_from_summaries = set()

    for shard in expected_target_shards:
        up_k = shard_upload.get(shard)
        img_k = shard_imagery.get(shard)
        img_lab_k = shard_image_labels.get(shard)
        img_src_k = shard_image_source_membership.get(shard)
        sum_k = shard_summary.get(shard)
        ok = shard in shard_success

        if not (up_k and img_k and img_lab_k and img_src_k and sum_k and ok):
            missing_target.append(shard)
            continue

        summary = s3_read_json(bucket, sum_k, TASK_NAME)
        rows_read = int(summary.get("rows_read", 0))
        canon_im_rows = int(summary.get("canonical_imagery_rows", 0))
        img_lbl_rows = int(summary.get("image_labels_rows", 0))
        img_src_rows = int(summary.get("image_source_membership_rows", 0))
        canon_lbl_rows = int(summary.get("canonical_label_rows_total", 0))

        owner_shards_touched = summary.get("canonical_label_owner_shards_touched", [])
        if isinstance(owner_shards_touched, list):
            for o in owner_shards_touched:
                if o:
                    expected_owner_shards_from_summaries.add(str(o).rjust(6, "0"))

        upload_size = _head_size_bytes(bucket, up_k)
        imagery_size = _head_size_bytes(bucket, img_k)
        image_labels_size = _head_size_bytes(bucket, img_lab_k)
        image_source_membership_size = _head_size_bytes(bucket, img_src_k)

        total_rows_read += rows_read
        total_canon_imagery_rows += canon_im_rows
        total_image_labels_rows += img_lbl_rows
        total_image_source_membership_rows += img_src_rows
        total_canon_label_rows += canon_lbl_rows

        target_shards.append(
            {
                "kind": "target",
                "shard": shard,
                "rows_read": rows_read,
                "canonical_imagery_rows": canon_im_rows,
                "image_labels_rows": img_lbl_rows,
                "image_source_membership_rows": img_src_rows,
                "upload_staging_key": up_k,
                "canonical_imagery_key": img_k,
                "canonical_labels_key": None,
                "image_labels_key": img_lab_k,
                "image_source_membership_key": img_src_k,
                "upload_staging_size_bytes": upload_size,
                "canonical_imagery_size_bytes": imagery_size,
                "image_labels_size_bytes": image_labels_size,
                "image_source_membership_size_bytes": image_source_membership_size,
                "total_target_bytes": (
                    upload_size
                    + imagery_size
                    + image_labels_size
                    + image_source_membership_size
                ),
            }
        )

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
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Warning: unexpected owner shard outputs discovered: {sorted(unexpected_owner_shards)}",
            level="warning",
        )

    total_owner_label_files = 0
    for owner_id in owner_shards:
        parts = owner_parts.get(owner_id, [])
        if not parts:
            continue

        total_owner_label_files += len(parts)
        owner_prefix = f"{processed_prefix}/canonical_labels_by_fingerprint/owner-{owner_id}/"
        owner_bytes = 0
        for p in parts:
            owner_bytes += _head_size_bytes(bucket, p)

        owner_shard_items.append(
            {
                "kind": "label_owner",
                "shard": f"owner-{owner_id}",
                "rows_read": None,
                "owner_shard_id": owner_id,
                "owner_prefix": owner_prefix,
                "parts_count": len(parts),
                "owner_part_keys": list(parts),
                "canonical_labels_size_bytes": owner_bytes,
                "upload_staging_key": None,
                "canonical_imagery_key": None,
                "canonical_labels_key": owner_prefix,
                "image_labels_key": None,
                "image_source_membership_key": None,
            }
        )

    return {
        "missing_target_shards": missing_target,
        "target_shards": target_shards,
        "owner_shard_items": owner_shard_items,
        "total_rows_read": total_rows_read,
        "total_canon_imagery_rows": total_canon_imagery_rows,
        "total_canon_label_rows": total_canon_label_rows,
        "total_image_labels_rows": total_image_labels_rows,
        "total_image_source_membership_rows": total_image_source_membership_rows,
        "processed_prefix": processed_prefix,
        "target_shard_count": len(expected_target_shards),
        "owner_shard_count": len(owner_shards),
        "total_owner_label_files": total_owner_label_files,
    }


def _close_group(groups: List[List[Dict[str, Any]]], current: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if current:
        groups.append(current)
    return []


def group_target_shards(shards: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not GROUPING_ENABLED or len(shards) <= 1:
        return [[s] for s in shards]

    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_rows = 0
    current_bytes = 0

    for shard in shards:
        shard_rows = int(shard.get("rows_read", 0))
        shard_bytes = int(shard.get("total_target_bytes", 0))

        if not current:
            current = [shard]
            current_rows = shard_rows
            current_bytes = shard_bytes
            continue

        current_at_target = (
            current_rows >= TARGET_TARGET_ROWS
            or current_bytes >= TARGET_TARGET_BYTES
        )

        would_rows = current_rows + shard_rows
        would_bytes = current_bytes + shard_bytes
        would_exceed_hard_cap = (
            would_rows > MAX_TARGET_ROWS
            or would_bytes > MAX_TARGET_BYTES
        )

        if current_at_target or would_exceed_hard_cap:
            current = _close_group(groups, current)
            current = [shard]
            current_rows = shard_rows
            current_bytes = shard_bytes
            continue

        current.append(shard)
        current_rows = would_rows
        current_bytes = would_bytes

    _close_group(groups, current)
    return groups


def group_owner_shards(owner_items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not GROUPING_ENABLED or len(owner_items) <= 1:
        return [[o] for o in owner_items]

    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_bytes = 0
    current_parts = 0

    for owner in owner_items:
        owner_bytes = int(owner.get("canonical_labels_size_bytes", 0))
        owner_parts = int(owner.get("parts_count", 0))

        if not current:
            current = [owner]
            current_bytes = owner_bytes
            current_parts = owner_parts
            continue

        current_at_target = (
            current_bytes >= TARGET_OWNER_BYTES
            or current_parts >= TARGET_OWNER_PARTS
        )

        would_bytes = current_bytes + owner_bytes
        would_parts = current_parts + owner_parts
        would_exceed_hard_cap = (
            would_bytes > MAX_OWNER_BYTES
            or would_parts > MAX_OWNER_PARTS
        )

        if current_at_target or would_exceed_hard_cap:
            current = _close_group(groups, current)
            current = [owner]
            current_bytes = owner_bytes
            current_parts = owner_parts
            continue

        current.append(owner)
        current_bytes = would_bytes
        current_parts = would_parts

    _close_group(groups, current)
    return groups


def _materialize_grouped_jsonl_file(
    *,
    source_keys: List[str],
    dest_key: str,
) -> Tuple[str, int]:
    if not source_keys:
        raise RuntimeError(f"{TASK_NAME} cannot materialize empty source_keys for {dest_key}")

    parts: List[bytes] = []
    total_bytes = 0

    for key in source_keys:
        blob = _ensure_trailing_newline(_read_key_bytes(FILE_BUCKET_NAME, key))
        total_bytes += len(blob)

        if total_bytes > MAX_MATERIALIZED_GROUP_BYTES:
            raise RuntimeError(
                f"{TASK_NAME} grouped materialization would exceed "
                f"MAX_MATERIALIZED_GROUP_BYTES={MAX_MATERIALIZED_GROUP_BYTES}: "
                f"dest_key={dest_key} total_bytes={total_bytes}"
            )

        parts.append(blob)

    body = b"".join(parts)
    _write_bytes_key(dest_key, body)
    return dest_key, len(body)


def _materialize_grouped_target_item(
    *,
    job_id: str,
    group_index: int,
    group_shards: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not group_shards:
        raise RuntimeError(f"{TASK_NAME} cannot materialize empty target group")

    if len(group_shards) == 1:
        only = group_shards[0]
        return {
            "kind": "target",
            "shard": f"group-target-{group_index:05d}",
            "source_shards": [only["shard"]],
            "grouped": False,
            "source_shard_count": 1,
            "rows_read": int(only["rows_read"]),
            "canonical_imagery_rows": int(only["canonical_imagery_rows"]),
            "image_labels_rows": int(only["image_labels_rows"]),
            "image_source_membership_rows": int(only["image_source_membership_rows"]),
            "upload_staging_key": only["upload_staging_key"],
            "canonical_imagery_key": only["canonical_imagery_key"],
            "canonical_labels_key": None,
            "image_labels_key": only["image_labels_key"],
            "image_source_membership_key": only["image_source_membership_key"],
        }

    source_shards = [s["shard"] for s in group_shards]

    upload_key, _ = _materialize_grouped_jsonl_file(
        source_keys=[s["upload_staging_key"] for s in group_shards],
        dest_key=(
            f"temp/image-upload/{job_id}/batches/registration-step/ingest-handoff/grouped/"
            f"upload_staging/group-{group_index:05d}.jsonl"
        ),
    )
    imagery_key, _ = _materialize_grouped_jsonl_file(
        source_keys=[s["canonical_imagery_key"] for s in group_shards],
        dest_key=(
            f"temp/image-upload/{job_id}/batches/registration-step/ingest-handoff/grouped/"
            f"canonical_imagery/group-{group_index:05d}.jsonl"
        ),
    )
    image_labels_key, _ = _materialize_grouped_jsonl_file(
        source_keys=[s["image_labels_key"] for s in group_shards],
        dest_key=(
            f"temp/image-upload/{job_id}/batches/registration-step/ingest-handoff/grouped/"
            f"image_labels/group-{group_index:05d}.jsonl"
        ),
    )
    image_source_membership_key, _ = _materialize_grouped_jsonl_file(
        source_keys=[s["image_source_membership_key"] for s in group_shards],
        dest_key=(
            f"temp/image-upload/{job_id}/batches/registration-step/ingest-handoff/grouped/"
            f"image_source_membership/group-{group_index:05d}.jsonl"
        ),
    )

    return {
        "kind": "target",
        "shard": f"group-target-{group_index:05d}",
        "source_shards": source_shards,
        "grouped": True,
        "source_shard_count": len(group_shards),
        "rows_read": sum(int(s.get("rows_read", 0)) for s in group_shards),
        "canonical_imagery_rows": sum(int(s.get("canonical_imagery_rows", 0)) for s in group_shards),
        "image_labels_rows": sum(int(s.get("image_labels_rows", 0)) for s in group_shards),
        "image_source_membership_rows": sum(int(s.get("image_source_membership_rows", 0)) for s in group_shards),
        "upload_staging_key": upload_key,
        "canonical_imagery_key": imagery_key,
        "canonical_labels_key": None,
        "image_labels_key": image_labels_key,
        "image_source_membership_key": image_source_membership_key,
    }


def _materialize_grouped_owner_item(
    *,
    job_id: str,
    group_index: int,
    group_owner_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not group_owner_items:
        raise RuntimeError(f"{TASK_NAME} cannot materialize empty owner group")

    if len(group_owner_items) == 1:
        only = group_owner_items[0]
        return {
            "kind": "label_owner",
            "shard": f"group-owner-{group_index:05d}",
            "source_shards": [only["shard"]],
            "grouped": False,
            "source_shard_count": 1,
            "rows_read": None,
            "owner_shard_ids": [only["owner_shard_id"]],
            "canonical_labels_key": only["canonical_labels_key"],
            "upload_staging_key": None,
            "canonical_imagery_key": None,
            "image_labels_key": None,
            "image_source_membership_key": None,
        }

    owner_ids = [o["owner_shard_id"] for o in group_owner_items]
    source_shards = [o["shard"] for o in group_owner_items]
    source_keys: List[str] = []
    total_parts = 0

    for o in group_owner_items:
        part_keys = o.get("owner_part_keys") or []
        if not isinstance(part_keys, list):
            raise RuntimeError(f"{TASK_NAME} owner_part_keys malformed for owner item: {o}")
        total_parts += len(part_keys)
        for p in part_keys:
            if isinstance(p, str) and p.strip():
                source_keys.append(p.strip())

    grouped_prefix = (
        f"temp/image-upload/{job_id}/batches/registration-step/ingest-handoff/grouped/"
        f"canonical_labels/owner-group-{group_index:05d}/"
    )
    grouped_file_key = f"{grouped_prefix}part-merged.jsonl"

    _materialize_grouped_jsonl_file(
        source_keys=source_keys,
        dest_key=grouped_file_key,
    )

    return {
        "kind": "label_owner",
        "shard": f"group-owner-{group_index:05d}",
        "source_shards": source_shards,
        "grouped": True,
        "source_shard_count": len(group_owner_items),
        "rows_read": None,
        "owner_shard_ids": owner_ids,
        "grouped_owner_parts_count": total_parts,
        "canonical_labels_key": grouped_prefix,
        "upload_staging_key": None,
        "canonical_imagery_key": None,
        "image_labels_key": None,
        "image_source_membership_key": None,
    }


def build_grouped_ingest_items(
    job_id: str,
    target_shards: List[Dict[str, Any]],
    owner_shard_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    target_groups = group_target_shards(target_shards)
    owner_groups = group_owner_shards(owner_shard_items)

    grouped_items: List[Dict[str, Any]] = []

    for idx, group in enumerate(target_groups):
        grouped_items.append(
            _materialize_grouped_target_item(
                job_id=job_id,
                group_index=idx,
                group_shards=group,
            )
        )

    for idx, group in enumerate(owner_groups):
        grouped_items.append(
            _materialize_grouped_owner_item(
                job_id=job_id,
                group_index=idx,
                group_owner_items=group,
            )
        )

    return grouped_items


def write_ingest_handoff(job_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    handoff_prefix = f"temp/image-upload/{job_id}/batches/registration-step/ingest-handoff/"
    handoff_key = f"{handoff_prefix}{INGEST_HANDOFF_FILE_NAME}"

    body = "\n".join(json.dumps(item, separators=(",", ":")) for item in items) + "\n"
    plan_s3_uri = write_s3_obj(
        FILE_BUCKET_NAME,
        handoff_key,
        body,
        "application/x-ndjson",
        TASK_NAME,
    )

    return {
        "plan_bucket": FILE_BUCKET_NAME,
        "plan_key": handoff_key,
        "plan_s3_uri": plan_s3_uri,
        "item_count": len(items),
    }


def handler(event, context):
    job_id = _require_event_key(event, "job_id")
    user = _require_event_key(event, "user")
    event_type = _require_event_key(event, "event_type")
    batch_plan_bucket = _require_event_key(event, "batch_plan_bucket")
    batch_plan_key = _require_event_key(event, "batch_plan_key")
    expected_count = _require_event_key(event, "expected_count")

    try:
        total_rows = int(expected_count)
    except Exception as e:
        raise RuntimeError(f"{TASK_NAME} expected_count is not an int-like value ({expected_count}): {e}")

    if not job_id or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} missing job_id")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Starting registration pre-ingest for job {job_id}",
    )

    try:
        batch_items = read_batch_plan_items(batch_plan_bucket, batch_plan_key)
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Failed reading batch handoff plan s3://{batch_plan_bucket}/{batch_plan_key}: {e}",
            level="error",
        )
        raise

    try:
        collected = collect_processed_shards(job_id, batch_items, user, event_type)
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Failed collecting processed shards: {e}",
            level="error",
        )
        raise

    missing = collected["missing_target_shards"]
    if missing:
        err = f"{TASK_NAME} Missing processed outputs for target shards: {missing}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    target_shards = collected["target_shards"]
    owner_shard_items = collected["owner_shard_items"]
    total_rows_read = int(collected["total_rows_read"])

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Collected raw registration outputs "
        f"(target_shards={len(target_shards)}, owner_shards={len(owner_shard_items)}). "
        f"total_rows_read={total_rows_read} "
        f"canon_imagery_rows={collected['total_canon_imagery_rows']} "
        f"canon_label_rows_total={collected['total_canon_label_rows']} "
        f"image_labels_rows={collected['total_image_labels_rows']} "
        f"image_source_membership_rows={collected['total_image_source_membership_rows']} "
        f"owner_label_files={collected['total_owner_label_files']}"
    )

    if total_rows_read != total_rows:
        raise RuntimeError(
            f"{TASK_NAME} Total row count mismatch: total_rows={total_rows}, "
            f"workers total_rows_read={total_rows_read}"
        )

    try:
        grouped_items = build_grouped_ingest_items(job_id, target_shards, owner_shard_items)
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Failed building grouped ingest items: {e}",
            level="error",
        )
        raise

    grouped_target_items = [x for x in grouped_items if x.get("kind") == "target"]
    grouped_owner_items = [x for x in grouped_items if x.get("kind") == "label_owner"]

    grouped_total_rows_read = sum(int(x.get("rows_read", 0) or 0) for x in grouped_target_items)
    grouped_total_canon_imagery_rows = sum(int(x.get("canonical_imagery_rows", 0) or 0) for x in grouped_target_items)
    grouped_total_image_labels_rows = sum(int(x.get("image_labels_rows", 0) or 0) for x in grouped_target_items)
    grouped_total_image_source_membership_rows = sum(int(x.get("image_source_membership_rows", 0) or 0) for x in grouped_target_items)

    if grouped_total_rows_read != total_rows_read:
        raise RuntimeError(
            f"{TASK_NAME} Grouped target rows mismatch: grouped_total_rows_read={grouped_total_rows_read}, "
            f"workers total_rows_read={total_rows_read}"
        )

    if grouped_total_canon_imagery_rows != int(collected["total_canon_imagery_rows"]):
        raise RuntimeError(
            f"{TASK_NAME} Grouped canonical_imagery row mismatch: grouped={grouped_total_canon_imagery_rows}, "
            f"workers={collected['total_canon_imagery_rows']}"
        )

    if grouped_total_image_labels_rows != int(collected["total_image_labels_rows"]):
        raise RuntimeError(
            f"{TASK_NAME} Grouped image_labels row mismatch: grouped={grouped_total_image_labels_rows}, "
            f"workers={collected['total_image_labels_rows']}"
        )

    if grouped_total_image_source_membership_rows != int(collected["total_image_source_membership_rows"]):
        raise RuntimeError(
            f"{TASK_NAME} Grouped image_source_membership row mismatch: grouped={grouped_total_image_source_membership_rows}, "
            f"workers={collected['total_image_source_membership_rows']}"
        )

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Grouped ingest units "
        f"(target_shards={len(target_shards)} -> target_groups={len(grouped_target_items)}, "
        f"owner_shards={len(owner_shard_items)} -> owner_groups={len(grouped_owner_items)}). "
        f"grouping_enabled={GROUPING_ENABLED} "
        f"target_rows_target={TARGET_TARGET_ROWS} "
        f"target_bytes_target={TARGET_TARGET_BYTES} "
        f"max_target_rows={MAX_TARGET_ROWS} "
        f"max_target_bytes={MAX_TARGET_BYTES} "
        f"target_owner_bytes={TARGET_OWNER_BYTES} "
        f"max_owner_bytes={MAX_OWNER_BYTES}"
    )

    try:
        handoff = write_ingest_handoff(job_id, grouped_items)
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Failed writing ingest handoff: {e}",
            level="error",
        )
        raise

    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    ctas_table_name = f"reg_export_{sanitized_job_id}"

    return {
        "plan_bucket": handoff["plan_bucket"],
        "plan_key": handoff["plan_key"],
        "plan_s3_uri": handoff["plan_s3_uri"],
        "item_count": handoff["item_count"],
        "total_rows": total_rows,
        "total_rows_read": total_rows_read,
        "total_canon_imagery_rows": int(collected["total_canon_imagery_rows"]),
        "total_canon_label_rows": int(collected["total_canon_label_rows"]),
        "total_image_labels_rows": int(collected["total_image_labels_rows"]),
        "total_image_source_membership_rows": int(collected["total_image_source_membership_rows"]),
        "processed_prefix": collected.get("processed_prefix"),
        "ctas_table_name": ctas_table_name,
        "target_shard_count": int(collected["target_shard_count"]),
        "owner_shard_count": int(collected["owner_shard_count"]),
        "total_owner_label_files": int(collected["total_owner_label_files"]),
        "grouped_target_item_count": int(len(grouped_target_items)),
        "grouped_owner_item_count": int(len(grouped_owner_items)),
    }