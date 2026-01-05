import os
import json
import math

import boto3

from common.utils import log, wait_for_athena, delete_s3_prefix
from common.ingest import drop_ctas_table_if_exists

# Environment variables provided by BatchingStage
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]  # e.g. s3://<bucket>/athena-results/
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
UPLOAD_STAGING_TABLE_NAME = os.environ.get("UPLOAD_STAGING_TABLE_NAME", "upload_staging")
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# Tunables (defaults; overridable via env)
AVG_ROW_KB = float(os.environ.get("REG_AVG_ROW_KB", "2.0"))
MEMORY_SAFETY_FACTOR = float(os.environ.get("REG_MEMORY_SAFETY_FACTOR", "0.5"))
MIN_ROWS_PER_SHARD = int(os.environ.get("REG_MIN_ROWS_PER_SHARD", "1000"))
MAX_ROWS_PER_SHARD = int(os.environ.get("REG_MAX_ROWS_PER_SHARD", "20000"))
JOB_MEMORY_MB = int(os.environ.get("REG_JOB_MEMORY_MB", "512"))

# Hard cap to prevent creating absurdly many partitions
MAX_SHARDS = int(os.environ.get("REG_MAX_SHARDS", "4096"))

s3 = boto3.client("s3")
athena = boto3.client("athena")

def _start_athena_count(job_id: str) -> str:
    table = f'"{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE_NAME}"'
    safe_job_id = job_id.replace("'", "''")
    sql = f"SELECT count(*) as cnt FROM {table} WHERE job_id = '{safe_job_id}'"
    resp = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP,
    )
    return resp["QueryExecutionId"]

def _read_count_from_athena_result(qid: str) -> int:
    resp = athena.get_query_results(QueryExecutionId=qid, MaxResults=2)
    rows = resp.get("ResultSet", {}).get("Rows", [])
    if len(rows) < 2:
        return 0
    val = rows[1]["Data"][0].get("VarCharValue")
    try:
        return int(val)
    except Exception:
        return 0

def _choose_target_rows_per_shard(total_rows: int) -> int:
    usable_mb = JOB_MEMORY_MB * MEMORY_SAFETY_FACTOR
    usable_kb = usable_mb * 1024.0
    avg_row_kb = AVG_ROW_KB if AVG_ROW_KB > 0 else 2.0

    estimated_rows = int(usable_kb / avg_row_kb)
    target = max(MIN_ROWS_PER_SHARD, min(estimated_rows, MAX_ROWS_PER_SHARD))

    # In pathological small-memory cases, ensure >= 1
    return max(1, target)

def _compute_num_shards(total_rows: int, target_rows: int) -> int:
    if total_rows <= 0:
        return 1
    n = int(math.ceil(total_rows / float(target_rows)))
    if n < 1:
        n = 1
    if n > MAX_SHARDS:
        n = MAX_SHARDS
    return n

def _start_athena_ctas(job_id: str, export_s3_prefix: str, num_shards: int) -> str:
    """
    Export job_id partition into Parquet files partitioned by shard_id (0..num_shards-1),
    where shard_id is derived from hashing image_id. No sha256 dependency.
    """
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
        classes_present,
        validation_status,
        validation_error,
        dedup_status,
        dedup_error,
        registration_status,
        registration_error,
        matched_image_id,
        lpad(CAST(mod(from_base(substr(to_hex(xxhash64(to_utf8(coalesce(image_id, '')))), 1, 8), 16), {num_shards}) AS varchar), 6, '0') AS shard_id
    FROM {table}
    WHERE job_id = '{safe_job_id}'
    """

    resp = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP,
    )
    return resp["QueryExecutionId"]

def _list_export_files_by_shard(export_prefix: str):
    """
    Group exported Parquet files by shard_id partition.
    Expected key path like: .../shard_id=000123/part-....parquet
    Returns dict: {shard_id: [s3://.../key, ...], ...}
    """
    paginator = s3.get_paginator("list_objects_v2")
    files_by_shard = {}
    all_keys = []
    export_prefix = export_prefix.rstrip("/") + "/"

    for page in paginator.paginate(Bucket=FILE_BUCKET_NAME, Prefix=export_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            all_keys.append(key)

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
                raise RuntimeError(f"[REG_FILE_BATCHING] Unable to extract shard_id from export key: {key}")

            files_by_shard.setdefault(shard_id, []).append(f"s3://{FILE_BUCKET_NAME}/{key}")

    return files_by_shard, all_keys

def _write_manifest(job_id: str, shard_name: str, files, manifest_prefix: str) -> str:
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
        raise RuntimeError(f"[REG_FILE_BATCHING] Batching Lambda failed: missing required key {e}")

    log(job_id, user, event_type, f"[REG_FILE_BATCHING] Starting registration batching for job {job_id}", LOG_FIREHOSE_STREAM_NAME)

    export_prefix_base = f"temp/image-upload/{job_id}/batches/registration-step/export/"
    manifest_prefix = f"temp/image-upload/{job_id}/batches/registration-step/manifests/"
    delete_s3_prefix(FILE_BUCKET_NAME, manifest_prefix)

    # 0) COUNT(*)
    count_qid = _start_athena_count(job_id)
    athena_res = wait_for_athena(count_qid, poll=2.0, timeout=300)
    if athena_res["state"] != "SUCCEEDED":
        resp = athena_res["metadata"]
        err = f"[REG_FILE_BATCHING] Athena COUNT failed for job {job_id}. Response = {resp}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    total_rows = _read_count_from_athena_result(count_qid)
    target_rows = _choose_target_rows_per_shard(total_rows)
    num_shards = _compute_num_shards(total_rows, target_rows)

    log(
        job_id, user, event_type,
        f"[REG_FILE_BATCHING] total_rows={total_rows}, target_rows_per_shard={target_rows}, num_shards={num_shards}",
        LOG_FIREHOSE_STREAM_NAME
    )

    # 1) CTAS export partitioned by shard_id
    sanitized_job_id = "".join(c if c.isalnum() else "_" for c in job_id)
    table_name = f"reg_export_{sanitized_job_id}"

    drop_qid = drop_ctas_table_if_exists(ICEBERG_DATABASE_NAME,
                                         table_name,
                                         ATHENA_OUTPUT_S3,
                                         ATHENA_WORKGROUP)

    drop_res = wait_for_athena(drop_qid, poll=3.0, timeout=900)
    if drop_res["state"] != "SUCCEEDED":
        resp = drop_res["metadata"]
        err = f"[REG_FILE_BATCHING] Failed to drop CTAS temp table for job {job_id}. Response = {resp}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    delete_s3_prefix(FILE_BUCKET_NAME, export_prefix_base)

    qid = _start_athena_ctas(job_id, export_prefix_base, num_shards)
    ctas_res = wait_for_athena(qid, poll=3.0, timeout=900)
    if ctas_res["state"] != "SUCCEEDED":
        resp = ctas_res["metadata"]
        err = f"[REG_FILE_BATCHING] Athena CTAS failed for job {job_id}. Response = {resp}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    log(job_id, user, event_type, f"[REG_FILE_BATCHING] Athena CTAS succeeded for job {job_id}, export prefix = {export_prefix_base}", LOG_FIREHOSE_STREAM_NAME)

    # 2) List exported files and group by shard_id
    files_by_shard, all_keys = _list_export_files_by_shard(export_prefix_base)

    if not files_by_shard:
        err = f"[REG_FILE_BATCHING] No exported files found for job {job_id} under prefix {export_prefix_base}, sample keys: {all_keys[:10]}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    # 3) Write manifests (one per shard_id)
    manifest_uris = []
    for shard_id, files in sorted(files_by_shard.items()):
        if not files:
            continue
        manifest_s3_uri = _write_manifest(job_id, shard_id, files, manifest_prefix)
        manifest_uris.append(manifest_s3_uri)

    result = {
        "job_id": job_id,
        "user": user,
        "event_type": event_type,
        "label_type": label_type,
        "data_source": data_source,
        "manifests": manifest_uris,
    }

    log(job_id, user, event_type, f"[REG_FILE_BATCHING] Completed for job {job_id}. Created {len(manifest_uris)} manifests.", LOG_FIREHOSE_STREAM_NAME)

    return result