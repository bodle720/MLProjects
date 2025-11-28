import os
import json
import math
import time
import boto3
from botocore.exceptions import ClientError

from common.utils import log, get_job_input

# Environment variables provided by BatchingStage
FILE_BUCKET_NAME = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]  # e.g. s3://<bucket>/athena-results/
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
UPLOAD_STAGING_TABLE = os.environ.get("UPLOAD_STAGING_TABLE", "upload_staging")
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

# Tunables (hardcoded, but can be overridden via extra_lambda_env by making env vars)
# Average row size in KB (conservative metadata estimate). Tune if you have better numbers.
AVG_ROW_KB = 2.0
# Safety factor for memory usage (0.0-1.0)
MEMORY_SAFETY_FACTOR = 0.5
# Minimum and maximum target rows per shard
MIN_ROWS_PER_SHARD = 1000
MAX_ROWS_PER_SHARD = 20000
# Max prefix length (hex chars). 1 => 16 prefixes, 2 => 256, 3 => 4096, 4 => 65536
MAX_PREFIX_LENGTH = 3

# Job memory (MB) used to compute target rows per shard. Default to 512 to match the job def.
JOB_MEMORY_MB = 512

# S3 layout base for dedup exports and manifests
EXPORT_BASE_PREFIX = "temp/image-upload"  # final path: temp/image-upload/{job_id}/batches/deduplication-step/...

s3 = boto3.client("s3")
athena = boto3.client("athena")

def _start_athena_count(job_id):
    """Run a fast COUNT(*) to estimate number of rows for job_id."""
    table = f'"{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE}"'

    safe_job_id = job_id.replace("'", "''")
    sql = (
        f"SELECT count(*) as cnt FROM {table} "
        f"WHERE job_id = '{safe_job_id}'"
    )
    resp = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP
    )
    return resp["QueryExecutionId"]

def _start_athena_ctas(job_id, export_s3_prefix, prefix_len):
    """
    Run CTAS to export upload_staging rows for job_id to export_s3_prefix,
    adding sha_prefix = substr(sha256_hash,1,prefix_len) for partitioning.
    Returns QueryExecutionId.
    """
    table = f'"{ICEBERG_DATABASE_NAME}"."{UPLOAD_STAGING_TABLE}"'

    safe_job_id = job_id.replace("'", "''")
    export_location = f"s3://{FILE_BUCKET_NAME}/{export_s3_prefix.rstrip('/')}/"
    # sanitize table name for CTAS (replace non-alnum with underscore)
    tmp_table = f"dedup_export_{''.join(c if c.isalnum() else '_' for c in job_id)}_{prefix_len}"
    sql = (
        f"CREATE TABLE {tmp_table} "
        f"WITH (format='PARQUET', external_location = '{export_location}') AS "
        f"SELECT *, substr(sha256_hash, 1, {prefix_len}) AS sha_prefix "
        f"FROM {table} "
        f"WHERE job_id = '{safe_job_id}'"
    )
    resp = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP
    )
    return resp["QueryExecutionId"]

def _wait_for_athena(qid, poll_interval=3, timeout_seconds=900):
    start = time.time()
    while True:
        resp = athena.get_query_execution(QueryExecutionId=qid)
        state = resp["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            return True, resp
        if state in ("FAILED", "CANCELLED"):
            reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "")
            return False, reason
        if time.time() - start > timeout_seconds:
            return False, f"timeout after {timeout_seconds}s"
        time.sleep(poll_interval)

def _read_count_from_athena_result(qid):
    """Read the single-row count result from Athena query execution output."""
    # Get results
    resp = athena.get_query_results(QueryExecutionId=qid, MaxResults=1)
    rows = resp.get("ResultSet", {}).get("Rows", [])
    # First row is header; second row contains the value if present
    if len(rows) >= 2:
        # value is in rows[1]['Data'][0]['VarCharValue']
        val = rows[1]["Data"][0].get("VarCharValue")
        try:
            return int(val)
        except Exception:
            return 0
    return 0

def _list_export_files(export_prefix):
    """
    List objects under the export prefix and group them by sha_prefix partition.
    Returns dict: {sha_prefix: [s3://.../key, ...], ...}
    """
    paginator = s3.get_paginator("list_objects_v2")
    prefix = export_prefix.rstrip("/") + "/"
    files_by_prefix = {}
    kwargs = {"Bucket": FILE_BUCKET_NAME, "Prefix": prefix}
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            # Try to extract sha_prefix from key path like ".../sha_prefix=aa/part-000.parquet"
            sha_prefix = None
            parts = key.split("/")
            for p in parts:
                if p.startswith("sha_prefix="):
                    sha_prefix = p.split("=", 1)[1]
                    break
            if not sha_prefix:
                sha_prefix = "zz"
            files_by_prefix.setdefault(sha_prefix, []).append(f"s3://{FILE_BUCKET_NAME}/{key}")
    return files_by_prefix

def _write_manifest(job_id, shard_name, files, manifest_prefix):
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
    manifest_key = f"{manifest_prefix.rstrip('/')}/manifest-shard-{shard_name}.json"
    body = json.dumps(manifest)
    s3.put_object(Bucket=FILE_BUCKET_NAME, Key=manifest_key, Body=body.encode("utf-8"), ContentType="application/json")
    return f"s3://{FILE_BUCKET_NAME}/{manifest_key}"

def _choose_prefix_length(total_rows, job_memory_mb=JOB_MEMORY_MB, avg_row_kb=AVG_ROW_KB,
                          safety_factor=MEMORY_SAFETY_FACTOR,
                          min_rows=MIN_ROWS_PER_SHARD, max_rows=MAX_ROWS_PER_SHARD,
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
        job_input = get_job_input(event)
        job_id = job_input["job_id"]
        user = job_input["user"]
        event_type = job_input["event_type"]
        label_types = job_input["label_types"] # a list
        source = job_input.get("source", "unknown")
    except KeyError as e:
        raise RuntimeError(f"Batching Lambda failed: missing required key {e}")

    log(job_id, user, event_type, f"Starting dedup batching for job {job_id}", LOG_FIREHOSE_STREAM_NAME)

    # Prepare prefixes
    export_prefix_base = f"{EXPORT_BASE_PREFIX}/{job_id}/batches/deduplication-step/export"
    manifest_prefix = f"{EXPORT_BASE_PREFIX}/{job_id}/batches/deduplication-step/manifests"
    processed_prefix = f"{EXPORT_BASE_PREFIX}/{job_id}/batches/deduplication-step/processed"

    # 0) Run COUNT(*) to estimate rows
    try:
        count_qid = _start_athena_count(job_id)
    except Exception as e:
        err = f"Failed to start Athena COUNT for job {job_id}: {e}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    ok, meta_or_reason = _wait_for_athena(count_qid, poll_interval=2, timeout_seconds=300)
    if not ok:
        err = f"Athena COUNT failed for job {job_id}: {meta_or_reason}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=str(meta_or_reason), level="error")
        raise RuntimeError(err)

    total_rows = _read_count_from_athena_result(count_qid)
    log(job_id, user, event_type, f"Estimated total rows for job {job_id} = {total_rows}", LOG_FIREHOSE_STREAM_NAME)

    # 1) choose prefix length P dynamically
    prefix_len, target_rows = _choose_prefix_length(
        total_rows,
        job_memory_mb=int(os.environ.get("DEDUP_JOB_MEMORY_MB", JOB_MEMORY_MB)),
        avg_row_kb=float(os.environ.get("DEDUP_AVG_ROW_KB", AVG_ROW_KB)),
        safety_factor=float(os.environ.get("DEDUP_MEMORY_SAFETY_FACTOR", MEMORY_SAFETY_FACTOR)),
        min_rows=int(os.environ.get("DEDUP_MIN_ROWS_PER_SHARD", MIN_ROWS_PER_SHARD)),
        max_rows=int(os.environ.get("DEDUP_MAX_ROWS_PER_SHARD", MAX_ROWS_PER_SHARD)),
        max_prefix_len=int(os.environ.get("DEDUP_MAX_PREFIX_LENGTH", MAX_PREFIX_LENGTH))
    )

    log(job_id, user, event_type, f"Chosen sha_prefix length = {prefix_len} (target rows per shard = {target_rows})", LOG_FIREHOSE_STREAM_NAME)

    # 2) Run CTAS to export partitioned files with sha_prefix
    try:
        qid = _start_athena_ctas(job_id, export_prefix_base, prefix_len)
    except Exception as e:
        err = f"Failed to start Athena CTAS for job {job_id}: {e}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    success, meta_or_reason = _wait_for_athena(qid)
    if not success:
        err = f"Athena CTAS failed for job {job_id}: {meta_or_reason}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=str(meta_or_reason), level="error")
        raise RuntimeError(err)

    log(job_id, user, event_type, f"Athena CTAS succeeded for job {job_id}, export prefix = {export_prefix_base}", LOG_FIREHOSE_STREAM_NAME)

    # 3) List exported files and group by sha_prefix
    try:
        files_by_prefix = _list_export_files(export_prefix_base)
    except Exception as e:
        err = f"Failed listing export files for job {job_id}: {e}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    if not files_by_prefix:
        err = f"No exported files found for job {job_id} under prefix {export_prefix_base}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=err, level="error")
        raise RuntimeError(err)

    # 4) Create manifests. If a single sha_prefix has too many files (edge case), split that prefix into multiple manifests
    manifest_keys = []
    try:
        for shard_prefix, files in sorted(files_by_prefix.items()):
            if not files:
                continue
            # If a single shard has too many files/rows, split into sub-manifests of reasonable size.
            # We use file count as a proxy for rows; split_size_files chosen conservatively.
            split_size_files = int(max(1, math.ceil(len(files) / max(1, math.ceil(total_rows / target_rows))))) if total_rows > 0 else 100
            # cap split_size_files to avoid huge manifests
            split_size_files = max(50, min(split_size_files, 2000))

            if len(files) <= split_size_files:
                manifest_s3_uri = _write_manifest(job_id, shard_prefix, files, manifest_prefix)
                manifest_keys.append(manifest_s3_uri)
                log(job_id, user, event_type, f"Wrote manifest for shard {shard_prefix} with {len(files)} files: {manifest_s3_uri}", LOG_FIREHOSE_STREAM_NAME)
            else:
                # split into multiple manifests for this shard_prefix
                for i in range(0, len(files), split_size_files):
                    chunk = files[i:i + split_size_files]
                    sub_name = f"{shard_prefix}-{i//split_size_files+1}"
                    manifest_s3_uri = _write_manifest(job_id, sub_name, chunk, manifest_prefix)
                    manifest_keys.append(manifest_s3_uri)
                    log(job_id, user, event_type, f"Wrote manifest for shard {shard_prefix} part {sub_name} with {len(chunk)} files: {manifest_s3_uri}", LOG_FIREHOSE_STREAM_NAME)
    except Exception as e:
        err = f"Failed writing manifests for job {job_id}: {e}"
        log(job_id, user, event_type, err, LOG_FIREHOSE_STREAM_NAME, error=str(e), level="error")
        raise

    # 5) Return the expected shape for BatchingStage
    result = {
        "job_id": job_id,
        "user": user,
        "label_types": json.dumps(label_types),
        "source": source,
        "event_type": event_type,
        "manifests": manifest_keys
    }

    log(job_id, user, event_type, f"Batching Lambda completed for job {job_id}. Created {len(manifest_keys)} manifests.", LOG_FIREHOSE_STREAM_NAME)

    return result