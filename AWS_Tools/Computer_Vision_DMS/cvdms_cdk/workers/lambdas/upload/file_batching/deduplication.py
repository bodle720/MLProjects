import os
import json
import math

import boto3

from common.general_utils.logging_utils import log
from common.general_utils.s3_utils import delete_s3_prefix, write_s3_obj
from common.general_utils.athena_utils import run_athena, athena_get_int_scalar, drop_table_if_exists
from common.general_utils.table_schemas import UPLOAD_STAGING_TABLE_NAME
from common.upload_utils.upload_athena_utils import athena_count_job_rows

# Environment variables provided by BatchingStage
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]  # a URI
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]
BATCH_HANDOFF_FILE_NAME = os.environ.get("BATCH_HANDOFF_FILE_NAME", "map-items.jsonl")

TASK_NAME = "[DEDUP_FILE_BATCHING]"

# Tunables
AVG_ROW_KB = 2.0
MEMORY_SAFETY_FACTOR = 0.5
MIN_ROWS_PER_SHARD = 1000
MAX_ROWS_PER_SHARD = 20000
MAX_PREFIX_LENGTH = 3
JOB_MEMORY_MB = 2048

s3 = boto3.client("s3")

def _require_event_key(event: dict, key: str):
    if key not in event:
        raise RuntimeError(f"{TASK_NAME} Batching Lambda failed: missing required key {key!r}")
    return event[key]

def generate_start_athena_max_shard_sql(job_id: str, prefix_len: int) -> str:
    table = f'"{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE_NAME}"'
    safe_job_id = job_id.replace("'", "''")

    return f"""
    SELECT max(cnt) AS max_cnt
    FROM (
      SELECT substr(sha256_hash, 1, {prefix_len}) AS sha_prefix, count(*) AS cnt
      FROM {table}
      WHERE job_id = '{safe_job_id}'
        AND sha256_hash IS NOT NULL
        AND sha256_hash <> ''
      GROUP BY 1
    )
    """

def generate_start_athena_ctas_sql(job_id: str, export_s3_prefix: str, prefix_len: int) -> str:
    table = f'"{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE_NAME}"'
    safe_job_id = job_id.replace("'", "''")
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    tmp_table = f"{ICEBERG_DATABASE_NAME}.dedup_export_{sanitized_job_id}"
    export_location = f"s3://{FILE_BUCKET_NAME}/{export_s3_prefix.rstrip('/')}/"

    return f"""
    CREATE TABLE {tmp_table}
    WITH (
        format = 'PARQUET',
        external_location = '{export_location}',
        partitioned_by = ARRAY['sha_prefix']
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
          WHEN sha256_hash IS NULL OR sha256_hash = '' THEN '__MISSING__'
          ELSE substr(sha256_hash, 1, {prefix_len})
        END AS sha_prefix
    FROM {table}
    WHERE job_id = '{safe_job_id}'
    """

def list_export_files(export_prefix: str):
    """
    List objects under the export prefix and group them by sha_prefix partition.
    Returns dict: {sha_prefix: [s3://.../key, ...], ...}
    """
    paginator = s3.get_paginator("list_objects_v2")
    files_by_prefix: dict[str, list[str]] = {}
    export_prefix = export_prefix.rstrip("/") + "/"
    kwargs = {"Bucket": FILE_BUCKET_NAME, "Prefix": export_prefix}
    sample_keys: list[str] = []

    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if len(sample_keys) < 10:
                sample_keys.append(key)

            if key.endswith("/"):
                continue

            name = key.split("/")[-1]
            if name.startswith("_") or name.startswith("."):
                continue

            sha_prefix = None
            for part in key.split("/"):
                if part.startswith("sha_prefix="):
                    sha_prefix = part.split("=", 1)[1]
                    break

            if not sha_prefix:
                continue

            files_by_prefix.setdefault(sha_prefix, []).append(f"s3://{FILE_BUCKET_NAME}/{key}")

    return files_by_prefix, sample_keys

def write_manifest(job_id: str, shard_name: str, files: list[str], manifest_prefix: str) -> str:
    manifest = {
        "job_id": job_id,
        "shard_prefix": shard_name,
        "files": files,
    }
    manifest_key = f"{manifest_prefix}manifest-shard-{shard_name}.json"
    body = json.dumps(manifest, separators=(",", ":"))
    return write_s3_obj(FILE_BUCKET_NAME, manifest_key, body, "application/json", TASK_NAME)

def choose_prefix_length(
    total_rows: int,
    job_memory_mb: int = JOB_MEMORY_MB,
    avg_row_kb: float = AVG_ROW_KB,
    safety_factor: float = MEMORY_SAFETY_FACTOR,
    min_rows: int = MIN_ROWS_PER_SHARD,
    max_rows: int = MAX_ROWS_PER_SHARD,
    max_prefix_len: int = MAX_PREFIX_LENGTH,
):
    usable_mb = job_memory_mb * safety_factor
    usable_kb = usable_mb * 1024.0
    if avg_row_kb <= 0:
        avg_row_kb = 2.0

    estimated_rows = int(usable_kb / avg_row_kb)
    target = max(min_rows, min(estimated_rows, max_rows))

    if total_rows <= 0:
        return 1, target

    prefixes_needed = math.ceil(total_rows / target)
    for p in range(1, max_prefix_len + 1):
        if (16 ** p) >= prefixes_needed:
            return p, target

    return max_prefix_len, target

def handler(event, context):
    job_id = _require_event_key(event, "job_id")
    user = _require_event_key(event, "user")
    event_type = _require_event_key(event, "event_type")
    label_type = _require_event_key(event, "label_type")
    data_source = _require_event_key(event, "data_source")
    source_split = _require_event_key(event, "source_split")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Starting dedup batching for job {job_id}")

    main_prefix = f"temp/image-upload/{job_id}/batches/deduplication-step/"
    export_prefix_base = f"{main_prefix}export/"
    manifest_prefix = f"{main_prefix}manifests/"
    handoff_prefix = f"{main_prefix}handoff/"
    handoff_key = f"{handoff_prefix}{BATCH_HANDOFF_FILE_NAME}"

    delete_s3_prefix(FILE_BUCKET_NAME, main_prefix, TASK_NAME)

    try:
        total_rows = athena_count_job_rows(
            job_id,
            TASK_NAME,
            ICEBERG_DATABASE_NAME,
            UPLOAD_STAGING_TABLE_NAME,
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
            poll=2.0,
            timeout=300,
        )
    except Exception as e:
        err = f"{TASK_NAME} Failed to count rows from upload staging table for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Estimated total rows for job {job_id} = {total_rows} rows")

    prefix_len, target_rows = choose_prefix_length(
        total_rows,
        job_memory_mb=JOB_MEMORY_MB,
        avg_row_kb=AVG_ROW_KB,
        safety_factor=MEMORY_SAFETY_FACTOR,
        min_rows=MIN_ROWS_PER_SHARD,
        max_rows=MAX_ROWS_PER_SHARD,
        max_prefix_len=MAX_PREFIX_LENGTH,
    )

    probe_prefix_len = prefix_len
    max_cnt = None
    for p in range(probe_prefix_len, MAX_PREFIX_LENGTH + 1):
        sql = generate_start_athena_max_shard_sql(job_id, p)
        try:
            qid, _ = run_athena(
                sql,
                f"{TASK_NAME} PROBE MAX SHARD",
                ATHENA_OUTPUT_S3,
                ATHENA_WORKGROUP,
                poll=2.0,
                timeout=300,
            )
        except Exception as e:
            err = f"{TASK_NAME} Failed max shard probe in Athena for p={p}, {job_id}: {e}"
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
            raise

        max_cnt = athena_get_int_scalar(qid, TASK_NAME)
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Probe max shard rows for prefix_len={p}: max_cnt={max_cnt} (target={target_rows})",
        )

        if max_cnt <= target_rows:
            prefix_len = p
            break

    if max_cnt is not None and max_cnt > target_rows and prefix_len == MAX_PREFIX_LENGTH:
        warn = (
            f"{TASK_NAME} WARNING: even at MAX_PREFIX_LENGTH={prefix_len}, "
            f"max shard rows={max_cnt} > target_rows={target_rows}. "
            f"Proceeding; consider increasing MAX_PREFIX_LENGTH or raising target_rows/job memory."
        )
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, warn, level="warning")

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Chosen sha_prefix length = {prefix_len} (target rows per shard = {target_rows})",
    )

    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    table_name = f"dedup_export_{sanitized_job_id}"
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
        err = f"{TASK_NAME} Failed to drop CTAS table if exists for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    sql = generate_start_athena_ctas_sql(job_id, export_prefix_base, prefix_len)
    try:
        run_athena(
            sql,
            f"{TASK_NAME} MAKE CTAS TABLE",
            ATHENA_OUTPUT_S3,
            ATHENA_WORKGROUP,
            poll=3.0,
            timeout=900,
        )
    except Exception as e:
        err = f"{TASK_NAME} Failed to make CTAS table and export for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Athena CTAS succeeded for job {job_id}, export prefix={export_prefix_base}",
    )

    try:
        files_by_prefix, sample_keys = list_export_files(export_prefix_base)
    except Exception as e:
        err = f"{TASK_NAME} Failed listing export files for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    if not files_by_prefix:
        err = (
            f"{TASK_NAME} No exported files found for job {job_id} under prefix "
            f"{export_prefix_base}, sample of keys: {sample_keys}"
        )
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    total_export_files = sum(len(v) for v in files_by_prefix.values())
    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        f"{TASK_NAME} Found {total_export_files} export data files across {len(files_by_prefix)} shard prefixes",
    )

    handoff_lines: list[str] = []
    manifest_count = 0

    try:
        for shard_prefix, files in sorted(files_by_prefix.items()):
            if not files:
                continue

            manifest_s3_uri = write_manifest(job_id, shard_prefix, files, manifest_prefix)
            manifest_count += 1

            handoff_item = {
                "manifest": manifest_s3_uri,
                "shard": shard_prefix,
            }
            handoff_lines.append(json.dumps(handoff_item, separators=(",", ":")))

            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Wrote manifest for shard {shard_prefix} with {len(files)} files: {manifest_s3_uri}",
            )

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
        "target_rows_per_shard": target_rows,
        "sha_prefix_length": prefix_len,
        "ctas_table": table_name,
        "export_prefix": export_prefix_base,
    }

    log(
        job_id,
        user,
        event_type,
        LOG_FIREHOSE_STREAM_NAME,
        (
            f"{TASK_NAME} Batching Lambda completed for job {job_id}. "
            f"Created {manifest_count} manifests. handoff_s3_uri={plan_s3_uri}"
        ),
    )

    return result