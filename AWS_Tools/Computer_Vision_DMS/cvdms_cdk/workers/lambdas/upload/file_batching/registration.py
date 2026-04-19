import os
import json
import math
import boto3

from common.general_utils.logging_utils import log
from common.general_utils.athena_utils import (
    drop_table_if_exists,
    run_athena,
    athena_get_int_scalar,
)
from common.general_utils.s3_utils import delete_s3_prefix, write_s3_obj
from common.general_utils.table_schemas import UPLOAD_STAGING_TABLE_NAME

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
BATCH_HANDOFF_FILE_NAME = os.environ.get("BATCH_HANDOFF_FILE_NAME", "map-items.jsonl")

TASK_NAME = "[REG_FILE_BATCHING]"

# Tunables from config/env
WORKER_MEMORY_MB = int(os.environ.get("WORKER_MEMORY_MB", "4096"))
ESTIMATED_ITEM_SIZE_KB = float(os.environ.get("ESTIMATED_ITEM_SIZE_KB", "2"))
MEMORY_SAFETY_FACTOR = float(os.environ.get("MEMORY_SAFETY_FACTOR", "0.5"))
MIN_ITEMS_PER_SHARD = int(os.environ.get("MIN_ITEMS_PER_SHARD", "500"))
MAX_ITEMS_PER_SHARD = int(os.environ.get("MAX_ITEMS_PER_SHARD", "1000"))

if WORKER_MEMORY_MB <= 0:
    raise RuntimeError(f"{TASK_NAME} WORKER_MEMORY_MB must be > 0, got {WORKER_MEMORY_MB}")
if ESTIMATED_ITEM_SIZE_KB <= 0:
    raise RuntimeError(f"{TASK_NAME} ESTIMATED_ITEM_SIZE_KB must be > 0, got {ESTIMATED_ITEM_SIZE_KB}")
if not (0 < MEMORY_SAFETY_FACTOR <= 1):
    raise RuntimeError(
        f"{TASK_NAME} MEMORY_SAFETY_FACTOR must be in (0, 1], got {MEMORY_SAFETY_FACTOR}"
    )
if MIN_ITEMS_PER_SHARD <= 0:
    raise RuntimeError(f"{TASK_NAME} MIN_ITEMS_PER_SHARD must be > 0, got {MIN_ITEMS_PER_SHARD}")
if MAX_ITEMS_PER_SHARD <= 0:
    raise RuntimeError(f"{TASK_NAME} MAX_ITEMS_PER_SHARD must be > 0, got {MAX_ITEMS_PER_SHARD}")
if MIN_ITEMS_PER_SHARD > MAX_ITEMS_PER_SHARD:
    raise RuntimeError(
        f"{TASK_NAME} MIN_ITEMS_PER_SHARD ({MIN_ITEMS_PER_SHARD}) cannot exceed "
        f"MAX_ITEMS_PER_SHARD ({MAX_ITEMS_PER_SHARD})"
    )

# Keep separate for now unless we later add it to config/env
MAX_SHARDS = 512

if MAX_SHARDS <= 0:
    raise RuntimeError(f"{TASK_NAME} MAX_SHARDS must be > 0, got {MAX_SHARDS}")

s3 = boto3.client("s3")

def _require_event_key(event: dict, key: str):
    if key not in event:
        raise RuntimeError(f"{TASK_NAME} missing required key {key!r}; event={json.dumps(event)}")
    return event[key]

def choose_target_rows_per_shard() -> int:
    usable_kb = WORKER_MEMORY_MB * 1024.0 * MEMORY_SAFETY_FACTOR
    estimated_rows_from_memory = int(usable_kb / ESTIMATED_ITEM_SIZE_KB)

    target = max(
        MIN_ITEMS_PER_SHARD,
        min(estimated_rows_from_memory, MAX_ITEMS_PER_SHARD),
    )
    return max(1, target)

def compute_num_shards(total_rows: int, target_rows: int) -> int:
    if total_rows <= 0:
        return 1
    n = int(math.ceil(total_rows / float(target_rows)))
    return max(1, min(n, MAX_SHARDS))

def generate_count_sql(job_id: str) -> str:
    table = f'"{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE_NAME}"'
    safe_job_id = job_id.replace("'", "''")

    return f"""
    SELECT CAST(count(*) AS bigint) AS c
    FROM {table}
    WHERE job_id = '{safe_job_id}'
    """

def generate_start_athena_ctas_sql(job_id: str, export_s3_prefix: str, num_shards: int) -> str:
    table = f'"{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE_NAME}"'
    safe_job_id = job_id.replace("'", "''")
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)

    tmp_table = f"{ICEBERG_DATABASE_NAME}.reg_export_{sanitized_job_id}"
    export_location = f"s3://{FILE_BUCKET_NAME}/{export_s3_prefix.rstrip('/')}/"

    num_shards = max(1, int(num_shards))

    return f"""
    CREATE TABLE {tmp_table}
    WITH (
        format = 'PARQUET',
        external_location = '{export_location}',
        partitioned_by = ARRAY['shard_id']
    ) AS
    SELECT
        job_id,
        image_id,
        temp_source_ref,
        img_type,
        img_height,
        img_width,
        num_channels,
        dtype,
        file_size_mb,
        CAST(uploaded_at AS timestamp(3)) AS uploaded_at,
        data_source,
        source_split,
        sha256_hash,
        luma_mean,
        luma_p10,
        luma_p90,
        dark_frac,
        bright_frac,
        contrast_luma_std,
        contrast_luma_p90_p10,
        blur_laplacian_var,
        sat_mean,
        colorfulness,
        lighting_bucket,
        blur_bucket,
        contrast_bucket,
        color_bucket,
        string_labels,
        temp_source_ref_bbox_meta,
        temp_source_ref_semantic_png,
        temp_source_ref_semantic_meta,
        temp_source_ref_instance_png,
        temp_source_ref_instance_meta,
        label_fingerprint,
        classes_present,
        validation_status,
        validation_error,
        dedup_status,
        dedup_error,
        registration_status,
        registration_error,
        matched_image_id,
        CASE
          WHEN dedup_status = 'external_duplicate' AND matched_image_id IS NOT NULL THEN matched_image_id
          ELSE image_id
        END AS target_image_id,
        lpad(
          CAST(
            mod(
              from_base(
                substr(
                  replace(
                    coalesce(
                      CASE
                        WHEN dedup_status = 'external_duplicate' AND matched_image_id IS NOT NULL THEN matched_image_id
                        ELSE image_id
                      END,
                      '00000000'
                    ),
                    '-',''
                  ),
                  1, 8
                ),
                16
              ),
              {num_shards}
            ) AS varchar
          ),
          6,
          '0'
        ) AS shard_id
    FROM {table}
    WHERE job_id = '{safe_job_id}'
    """

def list_export_files_by_shard(export_prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    files_by_shard: dict[str, list[str]] = {}
    sample_keys: list[str] = []
    export_prefix = export_prefix.rstrip("/") + "/"

    for page in paginator.paginate(Bucket=FILE_BUCKET_NAME, Prefix=export_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if len(sample_keys) < 3:
                sample_keys.append(key)

            if key.endswith("/"):
                continue

            leaf = key.split("/")[-1]
            if leaf.startswith("_") or leaf.startswith("."):
                continue

            shard_id = None
            for part in key.split("/"):
                if part.startswith("shard_id="):
                    shard_id = part.split("=", 1)[1]
                    break

            if shard_id is None:
                raise RuntimeError(f"{TASK_NAME} Unable to extract shard_id from export key: {key}")

            files_by_shard.setdefault(shard_id, []).append(f"s3://{FILE_BUCKET_NAME}/{key}")

    return files_by_shard, sample_keys

def write_manifest(job_id: str, shard_name: str, files: list[str], manifest_prefix: str) -> str:
    manifest = {
        "job_id": job_id,
        "shard_prefix": shard_name,
        "files": files,
    }
    manifest_key = f"{manifest_prefix}manifest-shard-{shard_name}.json"
    return write_s3_obj(
        FILE_BUCKET_NAME,
        manifest_key,
        json.dumps(manifest, separators=(",", ":")),
        "application/json",
        TASK_NAME,
    )

def handler(event, context):
    job_id = _require_event_key(event, "job_id")
    user = _require_event_key(event, "user")
    event_type = _require_event_key(event, "event_type")
    label_type = _require_event_key(event, "label_type")
    data_source = _require_event_key(event, "data_source")
    source_split = _require_event_key(event, "source_split")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Starting registration batching for job {job_id}",
    )

    main_prefix = f"temp/image-upload/{job_id}/batches/registration-step/"
    export_prefix_base = f"{main_prefix}export/"
    manifest_prefix = f"{main_prefix}manifests/"
    handoff_prefix = f"{main_prefix}handoff/"
    handoff_key = f"{handoff_prefix}{BATCH_HANDOFF_FILE_NAME}"

    delete_s3_prefix(FILE_BUCKET_NAME, main_prefix, TASK_NAME)

    # 0) COUNT all rows
    try:
        count_sql = generate_count_sql(job_id)
        qid, _ = run_athena(
            count_sql,
            TASK_NAME,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
            poll=2.0,
            timeout=300,
        )
        total_rows = athena_get_int_scalar(qid, TASK_NAME)
    except Exception as e:
        err = f"{TASK_NAME} Failed to count rows for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    if total_rows <= 0:
        raise RuntimeError(f"{TASK_NAME} No upload_staging rows found for registration for job_id={job_id}")

    target_rows = choose_target_rows_per_shard()
    num_shards = compute_num_shards(total_rows, target_rows)

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} total_rows={total_rows}, target_rows_per_shard={target_rows}, num_shards={num_shards}",
    )

    # 1) CTAS export partitioned by shard_id
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    table_name = f"reg_export_{sanitized_job_id}"

    try:
        drop_table_if_exists(
            ICEBERG_DATABASE_NAME,
            table_name,
            TASK_NAME,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
            poll=3.0,
            timeout=900,
        )
    except Exception as e:
        err = f"{TASK_NAME} Failed to drop CTAS table if it exists for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    sql = generate_start_athena_ctas_sql(job_id, export_prefix_base, num_shards)
    try:
        run_athena(
            sql,
            TASK_NAME,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
            poll=3.0,
            timeout=900,
        )
    except Exception as e:
        err = f"{TASK_NAME} CTAS failed for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Athena CTAS succeeded for job {job_id}, export prefix={export_prefix_base}",
    )

    # 2) List exported files and group by shard_id
    files_by_shard, sample_keys = list_export_files_by_shard(export_prefix_base)
    if not files_by_shard:
        err = f"{TASK_NAME} No exported files found under {export_prefix_base}, sample keys: {sample_keys}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    # 3) Write per-shard manifests + handoff JSONL
    handoff_lines: list[str] = []
    manifest_count = 0

    try:
        for shard_id, files in sorted(files_by_shard.items()):
            if not files:
                continue

            manifest_s3_uri = write_manifest(job_id, shard_id, files, manifest_prefix)
            manifest_count += 1

            handoff_item = {
                "manifest": manifest_s3_uri,
                "shard": shard_id,
            }
            handoff_lines.append(json.dumps(handoff_item, separators=(",", ":")))

    except Exception as e:
        err = f"{TASK_NAME} Failed writing manifests/handoff for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    handoff_body = "\n".join(handoff_lines) + "\n"
    plan_s3_uri = write_s3_obj(
        FILE_BUCKET_NAME,
        handoff_key,
        handoff_body,
        "application/x-ndjson",
        TASK_NAME,
    )

    result = {
        "plan_bucket": FILE_BUCKET_NAME,
        "plan_key": handoff_key,
        "plan_s3_uri": plan_s3_uri,
        "item_count": manifest_count,
        "manifest_count": manifest_count,
        "total_rows": total_rows,
        "num_shards": num_shards,
        "target_rows_per_shard": target_rows,
        "ctas_table": table_name,
        "export_prefix": export_prefix_base,
    }

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Completed for job {job_id}. "
            f"Created {manifest_count} manifests from {len(files_by_shard)} shard partitions. "
            f"handoff_s3_uri={plan_s3_uri}"
        ),
    )

    return result