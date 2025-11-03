import os, json, logging, time
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Env
BUCKET = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DB = os.environ["ICEBERG_DB"]
UPLOAD_STAGING_TABLE = os.environ["UPLOAD_STAGING_TABLE"]

athena = boto3.client("athena")
s3 = boto3.client("s3")

# Tunables
PHASH_PREFIX_LEN = int(os.environ.get("PHASH_PREFIX_LEN", "10"))  # starting prefix length
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "400"))     # upper cap per similarity batch
MANIFEST_PREFIX = "batches/internal-dedup"                        # temp manifest prefix under job folder

def _wait_athena(qid, poll=1.5, timeout=900):
    start = time.time()
    while True:
        resp = athena.get_query_execution(QueryExecutionId=qid)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            if state != "SUCCEEDED":
                reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "")
                raise RuntimeError(f"Athena {qid} {state}: {reason}")
            return
        if time.time() - start > timeout:
            raise TimeoutError(f"Athena {qid} timed out")
        time.sleep(poll)

def _query(sql):
    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP
    )["QueryExecutionId"]
    _wait_athena(qid)
    res = athena.get_query_results(QueryExecutionId=qid)
    rows = res.get("ResultSet", {}).get("Rows", [])
    headers = [c["VarCharValue"] for c in rows[0]["Data"]]
    out = []
    for r in rows[1:]:
        vals = []
        for i, c in enumerate(r["Data"]):
            vals.append(c.get("VarCharValue"))
        out.append(dict(zip(headers, vals)))
    return out

def _sql_ident(db, table):
    return f'"{db}"."{table}"'

def _delete_temp_and_staging(job_id):
    prefix = f"temp/image-upload/{job_id}/"
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        objs = resp.get("Contents", [])
        if objs:
            keys.extend([{"Key": o["Key"]} for o in objs])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    for i in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys[i:i+1000]})

    table = _sql_ident(ICEBERG_DB, UPLOAD_STAGING_TABLE)
    sql = f"DELETE FROM {table} WHERE job_id = '{job_id}'"
    _query(sql)

def _integrity_checks(job_id):
    table = _sql_ident(ICEBERG_DB, UPLOAD_STAGING_TABLE)
    sql = f"""
    SELECT COUNT(*) AS total,
           SUM(CASE WHEN validation_status = 'passed' THEN 1 ELSE 0 END) AS passed
    FROM {table}
    WHERE job_id = '{job_id}'
    """
    res = _query(sql)[0]
    total = int(res["total"])
    passed = int(res["passed"])
    if total == 0:
        return False, "No rows found in upload_staging for job"
    if passed != total:
        return False, f"Not all rows passed validation (passed={passed}, total={total})"

    sql2 = f"""
    SELECT img_type, COUNT(*) AS c
    FROM {table}
    WHERE job_id = '{job_id}'
    GROUP BY img_type
    """
    rows = _query(sql2)
    if len(rows) != 1:
        return False, f"Mixed img_type detected: {', '.join([r['img_type'] for r in rows])}"
    return True, rows[0]["img_type"]

def _mark_internal_duplicates_exact(job_id):
    table = _sql_ident(ICEBERG_DB, UPLOAD_STAGING_TABLE)
    sql_sha = f"""
    UPDATE {table}
    SET dedup_status = 'internal_duplicate'
    WHERE job_id = '{job_id}'
      AND validation_status = 'passed'
      AND image_id NOT IN (
          SELECT image_id FROM (
            SELECT image_id,
                   ROW_NUMBER() OVER (PARTITION BY sha256_hash ORDER BY image_id) AS rn
            FROM {table}
            WHERE job_id = '{job_id}' AND validation_status = 'passed'
          ) t WHERE t.rn = 1
      )
      AND sha256_hash IN (
          SELECT sha256_hash FROM {table}
          WHERE job_id = '{job_id}' AND validation_status = 'passed'
          GROUP BY sha256_hash HAVING COUNT(*) >= 2
      )
    """
    _query(sql_sha)

    sql_ph = f"""
    UPDATE {table}
    SET dedup_status = 'internal_duplicate'
    WHERE job_id = '{job_id}'
      AND validation_status = 'passed'
      AND (dedup_status IS NULL OR dedup_status = 'pending')
      AND image_id NOT IN (
          SELECT image_id FROM (
            SELECT image_id,
                   ROW_NUMBER() OVER (PARTITION BY phash ORDER BY image_id) AS rn
            FROM {table}
            WHERE job_id = '{job_id}'
              AND validation_status = 'passed'
              AND (dedup_status IS NULL OR dedup_status = 'pending')
          ) t WHERE t.rn = 1
      )
      AND phash IN (
          SELECT phash FROM {table}
          WHERE job_id = '{job_id}'
            AND validation_status = 'passed'
            AND (dedup_status IS NULL OR dedup_status = 'pending')
          GROUP BY phash HAVING COUNT(*) >= 2
      )
    """
    _query(sql_ph)

    sql_norm = f"""
    UPDATE {table}
    SET dedup_status = 'pending'
    WHERE job_id = '{job_id}'
      AND validation_status = 'passed'
      AND dedup_status IS NULL
    """
    _query(sql_norm)

def _fetch_survivors(job_id):
    table = _sql_ident(ICEBERG_DB, UPLOAD_STAGING_TABLE)
    sql = f"""
    SELECT image_id, phash, temp_source_ref
    FROM {table}
    WHERE job_id = '{job_id}'
      AND validation_status = 'passed'
      AND dedup_status = 'pending'
    """
    return _query(sql)

def _phash_prefix(phash, img_type, prefix_len):
    if img_type == "L":
        return phash[:prefix_len]
    parts = phash.split("|")
    if len(parts) != 3:
        return "MALFORMED"
    return "|".join(p[:prefix_len] for p in parts)

def _make_bins_iterative(rows, img_type, max_size, start_prefix_len):
    bins = {"": rows}
    prefix_len = start_prefix_len

    while True:
        new_bins = {}
        too_big = False
        for prefix, items in bins.items():
            if len(items) > max_size:
                too_big = True
                for r in items:
                    pfx = _phash_prefix(r["phash"], img_type, prefix_len)
                    new_bins.setdefault(pfx, []).append(r)
            else:
                new_bins[prefix] = items
        bins = new_bins
        prefix_len += 1
        if not too_big or prefix_len > 64:  # safeguard: stop if prefix too long
            break

    return {k: v for k, v in bins.items() if len(v) >= 2}

def _emit_manifests(job_id, bins):
    manifest_s3_uris = []
    for prefix, items in bins.items():
        manifest = {
            "job_id": job_id,
            "phash_prefix": prefix,
            "images": [
                {"image_id": r["image_id"], "temp_source_ref": r["temp_source_ref"], "phash": r["phash"]}
                for r in items
            ]
        }
        safe_prefix = prefix.replace("|","_")
        key = f"temp/image-upload/{job_id}/{MANIFEST_PREFIX}/phash-{safe_prefix}.json"
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=json.dumps(manifest).encode("utf-8"),
            ContentType="application/json"
        )
        manifest_s3_uris.append(f"s3://{BUCKET}/{key}")
    return manifest_s3_uris

def handler(event, context):
    job_id = event["job_id"]
    user = event.get("user", "unknown")

    log.info(f"Internal dedup batching start for job {job_id}")

    ok, info = _integrity_checks(job_id)
    if not ok:
        log.error(f"Integrity check failed for job {job_id}: {info}. Cleaning up.")
        _delete_temp_and_staging(job_id)
        # Signal failure to Step Functions by raising
        raise RuntimeError(f"Job {job_id} failed integrity: {info}")

    img_type = info  # 'L' or 'RGB'
    _mark_internal_duplicates_exact(job_id)

    survivors = _fetch_survivors(job_id)
    if not survivors:
        log.info(f"No survivors remain after exact duplicate marking for job {job_id}.")
        # Return empty manifests; downstream can short-circuit
        return {
            "job_id": job_id,
            "user": user,
            "manifests": [],
            "img_type": img_type
        }

    # Iteratively split bins until all are ≤ MAX_BATCH_SIZE
    bins = _make_bins_iterative(
        survivors, img_type, MAX_BATCH_SIZE, PHASH_PREFIX_LEN
    )
    manifests = _emit_manifests(job_id, bins)

    log.info(
        f"Prepared {len(manifests)} internal-dedup manifests for job {job_id} "
        f"(bins={len(bins)}, img_type={img_type})"
    )

    return {
        "job_id": job_id,
        "user": user,
        "manifests": manifests,
        "img_type": img_type
    }
