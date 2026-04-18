#!/usr/bin/env python3
import os
import json
from typing import Dict, List, Any

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import (
    s3_list_keys,
    s3_read_json,
    parse_s3_uri,
    read_obj_with_retry,
    write_s3_obj,
)

from common.general_utils.table_schemas import UPLOAD_STAGING_TABLE_NAME
from common.upload_utils.upload_athena_utils import athena_count_job_rows
from common.upload_utils.upload_iceberg_utils import delete_job_rows_from_table

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
INGEST_HANDOFF_FILE_NAME = os.environ.get("INGEST_HANDOFF_FILE_NAME", "map-items.jsonl")

TASK_NAME = "[DEDUP_INGEST_PRE]"

def _require_event_key(event: dict, key: str):
    if key not in event:
        raise RuntimeError(f"{TASK_NAME} Missing key: {key!r}, event={json.dumps(event)}")
    return event[key]

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

def collect_processed_shards(job_id: str, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Expect for each dedup shard:
      - shard-<shard>.jsonl
      - shard-<shard>-summary.json
      - shard-<shard>-SUCCESS
    """
    bucket = FILE_BUCKET_NAME
    processed_prefix = f"temp/image-upload/{job_id}/batches/deduplication-step/processed"
    expected_shards = extract_expected_shards_from_batch_items(batch_items)
    processed_keys = s3_list_keys(bucket, processed_prefix + "/")

    shard_jsonl: Dict[str, str] = {}
    shard_summary: Dict[str, str] = {}
    shard_success = set()

    for k in processed_keys:
        name = k.split("/")[-1]
        if name.startswith("shard-") and name.endswith(".jsonl"):
            shard = name[len("shard-") : -len(".jsonl")]
            shard_jsonl[shard] = k
        elif name.startswith("shard-") and name.endswith("-summary.json"):
            shard = name[len("shard-") : -len("-summary.json")]
            shard_summary[shard] = k
        elif name.startswith("shard-") and name.endswith("-SUCCESS"):
            shard = name[len("shard-") : -len("-SUCCESS")]
            shard_success.add(shard)

    if not expected_shards:
        expected_shards = sorted(set(shard_jsonl) | set(shard_summary) | set(shard_success))

    missing: List[str] = []
    shards: List[Dict[str, Any]] = []
    total_rows_read = 0
    total_processed_rows = 0

    for shard in expected_shards:
        jsonl_key = shard_jsonl.get(shard)
        summary_key = shard_summary.get(shard)
        ok = shard in shard_success

        if not (jsonl_key and summary_key and ok):
            missing.append(shard)
            continue

        summary = s3_read_json(bucket, summary_key, TASK_NAME)
        rows_read = int(summary.get("rows_read", 0))
        processed_rows = int(summary.get("processed_rows", 0))

        total_rows_read += rows_read
        total_processed_rows += processed_rows

        shards.append(
            {
                "shard": shard,
                "kind": "deduplication",
                "rows_read": rows_read,
                "processed_rows": processed_rows,
                "canonical_imagery_rows": None,
                "canonical_label_rows": None,
                "upload_staging_key": jsonl_key,
                "canonical_imagery_key": None,
                "canonical_labels_key": None,
                "image_labels_key": None,
                "image_source_membership_key": None,
            }
        )

    return {
        "missing_shards": missing,
        "shards": shards,
        "total_rows_read": total_rows_read,
        "total_processed_rows": total_processed_rows,
        "processed_prefix": processed_prefix,
    }

def write_ingest_handoff(job_id: str, shards: List[Dict[str, Any]]) -> Dict[str, Any]:
    handoff_prefix = f"temp/image-upload/{job_id}/batches/deduplication-step/ingest-handoff/"
    handoff_key = f"{handoff_prefix}{INGEST_HANDOFF_FILE_NAME}"

    body = "\n".join(json.dumps(item, separators=(",", ":")) for item in shards) + "\n"
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
        "item_count": len(shards),
    }

def handler(event, context):
    job_id = _require_event_key(event, "job_id")
    user = _require_event_key(event, "user")
    event_type = _require_event_key(event, "event_type")
    batch_plan_bucket = _require_event_key(event, "batch_plan_bucket")
    batch_plan_key = _require_event_key(event, "batch_plan_key")

    if not job_id or job_id == "unknown":
        raise RuntimeError(f"{TASK_NAME} missing job_id in event")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Starting dedup pre-ingest for job {job_id}")

    # 1) Read batching-stage handoff plan
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

    # 2) Collect processed shard outputs and verify completeness
    try:
        collected = collect_processed_shards(job_id, batch_items)
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Failed collecting processed shards: {e}", level="error")
        raise

    missing = collected["missing_shards"]
    if missing:
        err = f"{TASK_NAME} Missing processed outputs for shards: {missing}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    shards = collected["shards"]
    total_rows_read = collected["total_rows_read"]
    total_processed_rows = collected["total_processed_rows"]

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Collected {len(shards)} shard outputs. rows_read={total_rows_read}, processed_rows={total_processed_rows}",
    )

    # 3) Verify original count via Athena before deletion
    try:
        original_count = athena_count_job_rows(
            job_id,
            TASK_NAME,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Athena count failed: {e}", level="error")
        raise

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Athena original_count={original_count}")

    if total_rows_read != original_count:
        err = f"{TASK_NAME} Row count mismatch: original_count={original_count}, workers rows_read={total_rows_read}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    if total_processed_rows != total_rows_read:
        raise RuntimeError(f"{TASK_NAME} processed_rows({total_processed_rows}) != rows_read({total_rows_read})")

    # 4) Delete original partition rows once, before map inserts
    try:
        delete_result = delete_job_rows_from_table(
            job_id,
            TASK_NAME,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
        )
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Deleted upload_staging partition, result={delete_result}")
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Failed deleting upload_staging partition: {e}", level="error")
        raise

    # 5) Write ingest handoff JSONL for Distributed Map
    try:
        handoff = write_ingest_handoff(job_id, shards)
    except Exception as e:
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Failed writing ingest handoff: {e}", level="error")
        raise

    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    ctas_table_name = f"dedup_export_{sanitized_job_id}"

    return {
        "plan_bucket": handoff["plan_bucket"],
        "plan_key": handoff["plan_key"],
        "plan_s3_uri": handoff["plan_s3_uri"],
        "item_count": handoff["item_count"],
        "original_count": int(original_count),
        "total_rows_read": int(total_rows_read),
        "total_processed_rows": int(total_processed_rows),
        "processed_prefix": collected.get("processed_prefix"),
        "ctas_table_name": ctas_table_name,
    }