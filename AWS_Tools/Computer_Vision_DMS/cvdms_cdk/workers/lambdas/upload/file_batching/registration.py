import os
import json
import math

import boto3

from common.logging_utils import log
from common.athena_utils import drop_table_if_exists, run_athena, athena_count_job_rows
from common.s3_utils import delete_s3_prefix
from common.table_schemas import UPLOAD_STAGING_TABLE_NAME

FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]  # e.g. s3://<bucket>/athena-results/
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[REG_FILE_BATCHING]"

# Tunables
AVG_ROW_KB = 2.0
MEMORY_SAFETY_FACTOR = 0.5
MIN_ROWS_PER_SHARD = 1000
MAX_ROWS_PER_SHARD = 20000
JOB_MEMORY_MB = 2048

# Hard cap to prevent creating absurdly many partitions
MAX_SHARDS = 4096
MAX_FILES_PER_MANIFEST = 500  # or 1000

s3 = boto3.client("s3")

def generate_start_athena_ctas_sql(job_id: str, export_s3_prefix: str, num_shards: int) -> str:
    table = f'"{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE_NAME}"'
    safe_job_id = job_id.replace("'", "''")
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)

    tmp_table = f"{ICEBERG_DATABASE_NAME}.reg_export_{sanitized_job_id}"
    export_location = f"s3://{FILE_BUCKET_NAME}/{export_s3_prefix.rstrip('/')}/"

    num_shards = max(1, num_shards)

    sql = f"""
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
        lpad(CAST(mod(from_base(substr(replace(coalesce(image_id, ''), '-', ''), 1, 8), 16), {num_shards}) AS varchar), 6, '0') AS shard_id
    FROM {table}
    WHERE job_id = '{safe_job_id}'
    """
    return sql

def choose_target_rows_per_shard(total_rows: int) -> int:
    usable_mb = JOB_MEMORY_MB * MEMORY_SAFETY_FACTOR
    usable_kb = usable_mb * 1024.0
    avg_row_kb = AVG_ROW_KB if AVG_ROW_KB > 0 else 2.0

    estimated_rows = int(usable_kb / avg_row_kb)
    target = max(MIN_ROWS_PER_SHARD, min(estimated_rows, MAX_ROWS_PER_SHARD))

    # In pathological small-memory cases, ensure >= 1
    return max(1, target)

def compute_num_shards(total_rows: int, target_rows: int) -> int:
    if total_rows <= 0:
        return 1
    n = int(math.ceil(total_rows / float(target_rows)))
    if n < 1:
        n = 1
    if n > MAX_SHARDS:
        n = MAX_SHARDS
    return n

def list_export_files_by_shard(export_prefix: str):
    """
    Group exported Parquet files by shard_id partition.
    Expected key path like: .../shard_id=000123/part-....parquet
    Returns dict: {shard_id: [s3://.../key, ...], ...}
    """
    paginator = s3.get_paginator("list_objects_v2")
    files_by_shard = {}
    sample_keys = []
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

            if not shard_id:
                raise RuntimeError(f"{TASK_NAME} Unable to extract shard_id from export key: {key}")

            files_by_shard.setdefault(shard_id, []).append(f"s3://{FILE_BUCKET_NAME}/{key}")

    return files_by_shard, sample_keys

def write_manifest(job_id: str, shard_name: str, files, manifest_prefix: str) -> str:
    # Keep same manifest shape as your other stages for reuse in map workers:
    # shard_prefix is now shard_id
    manifest = {"job_id": job_id, "shard_prefix": shard_name, "files": files}
    manifest_key = f"{manifest_prefix}manifest-shard-{shard_name}.json"
    s3.put_object(
        Bucket=FILE_BUCKET_NAME,
        Key=manifest_key,
        Body=json.dumps(manifest).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{FILE_BUCKET_NAME}/{manifest_key}"

def handler(event, context):
    try:
        job_id = event["job_id"]
        user = event["user"]
        event_type = event["event_type"]
        label_type = event["label_type"]
        data_source = event["data_source"]
    except KeyError as e:
        raise RuntimeError(f"{TASK_NAME} Batching Lambda failed: missing required key {e}")

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Starting registration batching for job {job_id}")

    export_prefix_base = f"temp/image-upload/{job_id}/batches/registration-step/export/"
    manifest_prefix = f"temp/image-upload/{job_id}/batches/registration-step/manifests/"
    main_prefix = f"temp/image-upload/{job_id}/batches/registration-step/"
    delete_s3_prefix(FILE_BUCKET_NAME, main_prefix, TASK_NAME)

    # 0) COUNT(*)
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

    if total_rows <= 0:
        raise RuntimeError(f"{TASK_NAME} No rows in upload_staging for job_id={job_id}")

    target_rows = choose_target_rows_per_shard(total_rows)
    num_shards = compute_num_shards(total_rows, target_rows)

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} total_rows={total_rows}, target_rows_per_shard={target_rows}, num_shards={num_shards}")

    # 1) CTAS export partitioned by shard_id
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    table_name = f"reg_export_{sanitized_job_id}"
    try:
        drop_table_if_exists(ICEBERG_DATABASE_NAME,
                                 table_name,
                                 TASK_NAME,
                                 ATHENA_OUTPUT_S3,
                                 ATHENA_WORKGROUP,
                                 poll=3.0,
                                 timeout=900)
    except Exception as e:
        err = f"{TASK_NAME} Failed to drop CTAS table if it exists for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    sql = generate_start_athena_ctas_sql(job_id, export_prefix_base, num_shards)
    try:
        run_athena(sql,
                   TASK_NAME,
                   ATHENA_OUTPUT_S3,
                   ATHENA_WORKGROUP,
                   poll=3.0,
                   timeout=900)
    except Exception as e:
        err = f"{TASK_NAME} Failed to start CTAS table for job {job_id}: {e}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, f"{TASK_NAME} Athena CTAS succeeded for job {job_id}, export prefix = {export_prefix_base}")

    # 2) List exported files and group by shard_id
    files_by_shard, sample_keys = list_export_files_by_shard(export_prefix_base)

    if not files_by_shard:
        err = f"{TASK_NAME} No exported files found for job {job_id} under prefix {export_prefix_base}, sample keys: {sample_keys}"
        log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME, err, level="error")
        raise RuntimeError(err)

    # 3) Write manifests (one per shard_id)
    manifest_uris = []
    for shard_id, files in sorted(files_by_shard.items()):
        if not files:
            continue

        if len(files) > MAX_FILES_PER_MANIFEST:
            for part_idx in range(0, len(files), MAX_FILES_PER_MANIFEST):
                sub = files[part_idx:part_idx + MAX_FILES_PER_MANIFEST]
                shard_name2 = f"{shard_id}{part_idx // MAX_FILES_PER_MANIFEST:04d}"
                manifest_uris.append(write_manifest(job_id, shard_name2, sub, manifest_prefix))

        manifest_s3_uri = write_manifest(job_id, shard_id, files, manifest_prefix)
        manifest_uris.append(manifest_s3_uri)

    result = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "label_type": label_type,
        "data_source": data_source,
        "manifests": manifest_uris
    }

    log(job_id, user, event_type, LOG_FIREHOSE_STREAM_NAME,f"{TASK_NAME} Completed for job {job_id}. Created {len(manifest_uris)} manifests.")

    return result