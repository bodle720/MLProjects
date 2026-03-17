#!/usr/bin/env python3
import os
import json
from typing import Any, Dict

from common.logging_utils import log
from common.athena_utils import drop_table_if_exists

ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[REG_INGEST_POST]"

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        pre: Dict[str, Any] = event["pre"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Missing key: {e}, event={json.dumps(event)}")

    total_rows = _safe_int(pre.get("total_rows"))
    total_rows_read = _safe_int(pre.get("total_rows_read"))
    total_canon_imagery_rows = _safe_int(pre.get("total_canon_imagery_rows"))
    total_canon_label_rows = _safe_int(pre.get("total_canon_label_rows"))
    total_image_labels_rows = _safe_int(pre.get("total_image_labels_rows"))

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Starting post-ingest cleanup for job={job_id}. "
            f"total_rows={total_rows} total_rows_read={total_rows_read} "
            f"canon_imagery_rows_expected={total_canon_imagery_rows} "
            f"canon_label_rows_total_expected={total_canon_label_rows} "
            f"image_labels_rows_expected={total_image_labels_rows}"
        )
    )

    ctas_table_name = pre.get("ctas_table_name")
    if not isinstance(ctas_table_name, str) or not ctas_table_name.strip():
        sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
        ctas_table_name = f"reg_export_{sanitized_job_id}"

    ctas_drop_ok = True
    try:
        drop_table_if_exists(
            ICEBERG_DATABASE_NAME,
            ctas_table_name,
            f"{TASK_NAME}.drop_ctas",
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
    except Exception as e:
        ctas_drop_ok = False
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} CTAS drop failed table={ctas_table_name}: {e}",
            level="error",
        )

    if ctas_drop_ok:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Completed registration post-ingest cleanup for job {job_id}"
        )
    else:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Completed registration post-ingest with CTAS cleanup warning for job {job_id}",
            level="warning"
        )

    return {
        "job_id": job_id,
        "total_rows": total_rows,
        "total_rows_read": total_rows_read,
        "total_canon_imagery_rows_expected": total_canon_imagery_rows,
        "total_canon_label_rows_total_expected": total_canon_label_rows,
        "total_image_labels_rows_expected": total_image_labels_rows,
        "ctas_table_name": ctas_table_name,
        "ctas_drop_ok": ctas_drop_ok,
    }