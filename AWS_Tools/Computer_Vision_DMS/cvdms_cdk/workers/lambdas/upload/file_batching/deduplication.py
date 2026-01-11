import os
import json
import math

import boto3

from common.logging_utils import log
from common.s3_utils import delete_s3_prefix
from common.athena_utils import run_athena, athena_get_int_scalar, athena_count_job_rows, drop_table_if_exists
from common.table_schemas import UPLOAD_STAGING_TABLE_NAME

# Environment variables provided by BatchingStage
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"] # a URI
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[DEDUP_FILE_BATCHING]"

# Tunables (hardcoded)
# Average row size in KB (conservative metadata estimate). Tune if you have better numbers.
AVG_ROW_KB = 2.0
# Safety factor for memory usage (0.0-1.0)
MEMORY_SAFETY_FACTOR = 0.5
# Minimum and maximum target rows per shard
MIN_ROWS_PER_SHARD = 1000
MAX_ROWS_PER_SHARD = 20000
# Max prefix length (hex chars). 1 => 16 prefixes, 2 => 256, 3 => 4096, 4 => 65536
MAX_PREFIX_LENGTH = 3

# Job memory (MB) used to compute target rows per shard.
JOB_MEMORY_MB = 2048

s3 = boto3.client("s3")

def generate_start_athena_max_shard_sql(job_id, prefix_len):
    table = f'"{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE_NAME}"'
    safe_job_id = job_id.replace("'", "''")

    sql = f"""
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
    return sql

def generate_start_athena_ctas_sql(job_id, export_s3_prefix, prefix_len):
    table = f'"{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE_NAME}"'
    safe_job_id = job_id.replace("'", "''")
    sanitized_job_id = ''.join(c if c.isalnum() else '_' for c in job_id)
    tmp_table = f"{ICEBERG_DATABASE_NAME}.dedup_export_{sanitized_job_id}"
    export_location = f"s3://{FILE_BUCKET_NAME}/{export_s3_prefix.rstrip('/')}/"

    sql = f"""
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
        sha256_hash,
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
    return sql

def list_export_files(export_prefix):
    """
    List objects under the export prefix and group them by sha_prefix partition.
    Returns dict: {sha_prefix: [s3://.../key, ...], ...}
    """
    paginator = s3.get_paginator("list_objects_v2")
    files_by_prefix = {}
    export_prefix = export_prefix.rstrip("/") + "/"
    kwargs = {"Bucket": FILE_BUCKET_NAME, "Prefix": export_prefix}
    sample_keys = []
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if len(sample_keys) < 3:
                sample_keys.append(key)

            if key.endswith("/"):
                continue

            if key.split("/")[-1].startswith("_") or key.split("/")[-1].startswith("."):
                continue

            # Try to extract sha_prefix from key path like ".../sha_prefix=aa/part-000.parquet"
            sha_prefix = None
            parts = key.split("/")
            for p in parts:
                if p.startswith("sha_prefix="):
                    sha_prefix = p.split("=", 1)[1]
                    break
            if not sha_prefix:
                raise RuntimeError(f"{TASK_NAME} Unable to extract sha_prefix from export key: {key}")
            files_by_prefix.setdefault(sha_prefix, []).append(f"s3://{FILE_BUCKET_NAME}/{key}")

    return files_by_prefix, sample_keys

def write_manifest(job_id, shard_name, files, manifest_prefix):
    """
    Write a manifest JSON for a shard. Returns s3 uri.
    Manifest shape:
    {
      "job_id": "<job_id>",
      "shard_prefix": "<sha_prefix or composite>",
      "files": ["s3://...","s3://..."]
    }
    """
    manifest = {
        "job_id": job_id,
        "shard_prefix": shard_name,
        "files": files
    }
    manifest_key = f"{manifest_prefix}manifest-shard-{shard_name}.json"
    body = json.dumps(manifest)
    s3.put_object(Bucket=FILE_BUCKET_NAME, Key=manifest_key, Body=body.encode("utf-8"), ContentType="application/json")
    return f"s3://{FILE_BUCKET_NAME}/{manifest_key}"

def choose_prefix_length(total_rows,
                          job_memory_mb=JOB_MEMORY_MB,
                          avg_row_kb=AVG_ROW_KB,
                          safety_factor=MEMORY_SAFETY_FACTOR,
                          min_rows=MIN_ROWS_PER_SHARD,
                          max_rows=MAX_ROWS_PER_SHARD,
                          max_prefix_len=MAX_PREFIX_LENGTH):
    """
    Choose smallest prefix length P such that expected_rows_per_shard <= target_rows_per_shard.
    target_rows_per_shard is computed from memory and avg_row_kb, bounded by min/max.
    """
    # compute target rows from memory
    usable_mb = job_memory_mb * safety_factor
    # convert MB to KB
    usable_kb = usable_mb * 1024.0
    # estimate rows
    if avg_row_kb <= 0:
        avg_row_kb = 2.0
    estimated_rows = int(usable_kb / avg_row_kb)
    target = max(min_rows, min(estimated_rows, max_rows))

    # compute needed prefixes
    if total_rows <= 0:
        return 1, target
    prefixes_needed = math.ceil(total_rows / target)
    # find smallest P where 16^P >= prefixes_needed
    for p in range(1, max_prefix_len + 1):
        if (16 ** p) >= prefixes_needed:
            return p, target
    # fallback to max_prefix_len
    return max_prefix_len, target

def handler(event, context):
    # Validate input
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        label_type = event["label_type"]
        data_source = event["data_source"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Batching Lambda failed: missing required key {e}")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Starting dedup batching for job {job_id}")

    # Prepare prefixes
    export_prefix_base = f"temp/image-upload/{job_id}/batches/deduplication-step/export/"
    manifest_prefix = f"temp/image-upload/{job_id}/batches/deduplication-step/manifests/"
    main_prefix = f"temp/image-upload/{job_id}/batches/deduplication-step/"

    delete_s3_prefix(FILE_BUCKET_NAME, main_prefix, TASK_NAME)

    # 0) Run COUNT(*) to estimate rows
    try:
        total_rows = athena_count_job_rows(job_id,
                                          TASK_NAME,
                                          ICEBERG_DATABASE_NAME,
                                          UPLOAD_STAGING_TABLE_NAME,
                                          ATHENA_OUTPUT_S3,
                                          ATHENA_WORKGROUP,
                                          poll=2.0,
                                          timeout=300)
    except Exception as e:
        err = f"{TASK_NAME} Failed to count rows from upload staging table for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} Estimated total rows for job {job_id} = {total_rows} rows")

    # 1) choose prefix length P dynamically
    prefix_len, target_rows = choose_prefix_length(
        total_rows,
        job_memory_mb=JOB_MEMORY_MB,
        avg_row_kb=AVG_ROW_KB,
        safety_factor=MEMORY_SAFETY_FACTOR,
        min_rows=MIN_ROWS_PER_SHARD,
        max_rows=MAX_ROWS_PER_SHARD,
        max_prefix_len=MAX_PREFIX_LENGTH
    )

    # 1.5) Post-check: ensure max shard size <= target_rows by probing Athena.
    # This prevents a pathological prefix bucket from creating a huge manifest/shard.
    probe_prefix_len = prefix_len
    max_cnt = None
    for p in range(probe_prefix_len, MAX_PREFIX_LENGTH + 1):
        sql = generate_start_athena_max_shard_sql(job_id, p)
        try:
            qid, _ = run_athena(sql,
                                 f"{TASK_NAME} PROBE MAX SHARD",
                                 ATHENA_OUTPUT_S3,
                                 ATHENA_WORKGROUP,
                                 poll=2.0,
                                 timeout=300)
        except Exception as e:
            err = f"{TASK_NAME} Failed max shard probe in Athena for p = {p}, {job_id}: {e}"
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
            raise

        max_cnt = athena_get_int_scalar(qid, TASK_NAME)
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} Probe max shard rows for prefix_len={p}: max_cnt={max_cnt} (target={target_rows})")

        if max_cnt <= target_rows:
            prefix_len = p
            break

    # If still too big even at max prefix len, keep max and warn (can still proceed, but shard may be large).
    if max_cnt is not None and max_cnt > target_rows and prefix_len ==  MAX_PREFIX_LENGTH:
        warn = (f"{TASK_NAME} WARNING: even at MAX_PREFIX_LENGTH={prefix_len}, "
                f"max shard rows={max_cnt} > target_rows={target_rows}. "
                f"Proceeding; consider increasing MAX_PREFIX_LENGTH or raising target_rows/job memory.")
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, warn, level="warning")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} Chosen sha_prefix length = {prefix_len} (target rows per shard = {target_rows})")

    # 2) Run CTAS to export partitioned files with sha_prefix
    sanitized_job_id = ''.join(c if c.isalnum() else '_' for c in job_id)
    table_name = f"dedup_export_{sanitized_job_id}"
    try:
        drop_table_if_exists(ICEBERG_DATABASE_NAME,
                             table_name,
                             TASK_NAME,
                             ATHENA_OUTPUT_S3,
                             ATHENA_WORKGROUP,
                             poll=3.0,
                             timeout=900)
    except Exception as e:
        err = f"{TASK_NAME} Failed to drop CTAS table if exists for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    sql = generate_start_athena_ctas_sql(job_id, export_prefix_base, prefix_len)
    try:
        run_athena(sql,
                   f"{TASK_NAME} MAKE CTAS TABLE",
                   ATHENA_OUTPUT_S3,
                   ATHENA_WORKGROUP,
                   poll=3.0,
                   timeout=900)
    except Exception as e:
        err = f"{TASK_NAME} Failed to make CTAS table and export for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Athena CTAS succeeded for job {job_id}, export prefix = {export_prefix_base}")

    # 3) List exported files and group by sha_prefix
    try:
        files_by_prefix, sample_keys = list_export_files(export_prefix_base)
    except Exception as e:
        err = f"{TASK_NAME} Failed listing export files for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    if not files_by_prefix:
        err = f"{TASK_NAME} No exported files found for job {job_id} under prefix {export_prefix_base}, sample of keys: {sample_keys}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    # 4) Create manifests.
    manifest_uris = []
    try:
        for shard_prefix, files in sorted(files_by_prefix.items()):
            if not files:
                continue

            manifest_s3_uri = write_manifest(job_id, shard_prefix, files, manifest_prefix)
            manifest_uris.append(manifest_s3_uri)
            log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Wrote manifest for shard {shard_prefix} with {len(files)} files: {manifest_s3_uri}")

    except Exception as e:
        err = f"{TASK_NAME} Failed writing manifests for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    # 5) Return the expected shape for BatchingStage
    result = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "label_type": label_type,
        "data_source": data_source,
        "manifests": manifest_uris
    }

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Batching Lambda completed for job {job_id}. Created {len(manifest_uris)} manifests.")

    return result