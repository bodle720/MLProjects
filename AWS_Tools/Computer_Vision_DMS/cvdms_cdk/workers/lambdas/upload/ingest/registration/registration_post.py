#!/usr/bin/env python3
import os
import json
from typing import Any, Dict

from common.general_utils.logging_utils import log
from common.general_utils.athena_utils import drop_table_if_exists

ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[REG_INGEST_POST]"


def _require_event_key(event: dict, key: str):
    if key not in event:
        raise RuntimeError(f"{TASK_NAME} Missing key: {key!r}, event={json.dumps(event)}")
    return event[key]


def _require_pre_dict(event: dict) -> Dict[str, Any]:
    pre = _require_event_key(event, "pre")
    if not isinstance(pre, dict):
        raise RuntimeError(f"{TASK_NAME} invalid pre payload type: expected dict, got {type(pre).__name__}")
    return pre


def _require_pre_int(pre: Dict[str, Any], key: str, *, min_value: int | None = None) -> int:
    if key not in pre:
        raise RuntimeError(f"{TASK_NAME} pre.{key} missing")

    raw = pre[key]
    try:
        if isinstance(raw, bool):
            raise ValueError("bool is not a valid integer payload")
        value = int(raw)
    except Exception as e:
        raise RuntimeError(f"{TASK_NAME} pre.{key} is not an int-like value ({raw!r}): {e}")

    if min_value is not None and value < min_value:
        raise RuntimeError(f"{TASK_NAME} pre.{key} must be >= {min_value}, got {value}")

    return value


def _derive_ctas_table_name(job_id: str, pre: Dict[str, Any]) -> str:
    ctas_table_name = pre.get("ctas_table_name")
    if isinstance(ctas_table_name, str) and ctas_table_name.strip():
        return ctas_table_name.strip()

    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    return f"reg_export_{sanitized_job_id}"


def handler(event, context):
    try:
        job_id = _require_event_key(event, "job_id")
        user = _require_event_key(event, "user")
        event_type = _require_event_key(event, "event_type")
        pre = _require_pre_dict(event)
    except Exception:
        raise

    if not isinstance(job_id, str) or not job_id.strip() or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} invalid job_id: {job_id!r}")
    if not isinstance(user, str) or not user.strip():
        raise RuntimeError(f"{TASK_NAME} invalid user: {user!r}")
    if not isinstance(event_type, str) or not event_type.strip():
        raise RuntimeError(f"{TASK_NAME} invalid event_type: {event_type!r}")

    total_rows = _require_pre_int(pre, "total_rows", min_value=1)
    total_rows_read = _require_pre_int(pre, "total_rows_read", min_value=1)
    total_canon_imagery_rows = _require_pre_int(pre, "total_canon_imagery_rows", min_value=0)
    total_canon_label_rows = _require_pre_int(pre, "total_canon_label_rows", min_value=0)
    total_image_labels_rows = _require_pre_int(pre, "total_image_labels_rows", min_value=0)
    total_image_source_membership_rows = _require_pre_int(
        pre,
        "total_image_source_membership_rows",
        min_value=0,
    )

    target_shard_count = int(pre.get("target_shard_count", 0) or 0)
    owner_shard_count = int(pre.get("owner_shard_count", 0) or 0)
    total_owner_label_files = int(pre.get("total_owner_label_files", 0) or 0)

    if total_rows_read != total_rows:
        raise RuntimeError(
            f"{TASK_NAME} total_rows mismatch: pre.total_rows={total_rows}, "
            f"pre.total_rows_read={total_rows_read}"
        )

    ctas_table_name = _derive_ctas_table_name(job_id, pre)

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Starting registration post-ingest finalization for job={job_id}. "
            f"total_rows={total_rows} total_rows_read={total_rows_read} "
            f"canon_imagery_rows_expected={total_canon_imagery_rows} "
            f"canon_label_rows_total_expected={total_canon_label_rows} "
            f"image_labels_rows_expected={total_image_labels_rows} "
            f"image_source_membership_rows_expected={total_image_source_membership_rows} "
            f"target_shard_count={target_shard_count} "
            f"owner_shard_count={owner_shard_count} "
            f"total_owner_label_files={total_owner_label_files} "
            f"ctas_table={ctas_table_name}"
        ),
    )

    try:
        drop_table_if_exists(
            ICEBERG_DATABASE_NAME,
            ctas_table_name,
            f"{TASK_NAME}.drop_ctas",
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
    except Exception as e:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} CTAS drop failed table={ctas_table_name}: {e}",
            level="error",
        )
        raise RuntimeError(f"{TASK_NAME} CTAS drop failed for table={ctas_table_name}: {e}")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Completed registration post-ingest finalization for job={job_id}. "
            f"total_rows={total_rows} total_rows_read={total_rows_read} "
            f"canon_imagery_rows_expected={total_canon_imagery_rows} "
            f"canon_label_rows_total_expected={total_canon_label_rows} "
            f"image_labels_rows_expected={total_image_labels_rows} "
            f"image_source_membership_rows_expected={total_image_source_membership_rows} "
            f"target_shard_count={target_shard_count} "
            f"owner_shard_count={owner_shard_count} "
            f"total_owner_label_files={total_owner_label_files} "
            f"ctas_table_dropped={ctas_table_name}"
        ),
    )

    return {
        "job_id": job_id,
        "total_rows": total_rows,
        "total_rows_read": total_rows_read,
        "total_canon_imagery_rows_expected": total_canon_imagery_rows,
        "total_canon_label_rows_total_expected": total_canon_label_rows,
        "total_image_labels_rows_expected": total_image_labels_rows,
        "total_image_source_membership_rows_expected": total_image_source_membership_rows,
        "target_shard_count": target_shard_count,
        "owner_shard_count": owner_shard_count,
        "total_owner_label_files": total_owner_label_files,
        "ctas_table_name": ctas_table_name,
        "ctas_drop_ok": True,
    }