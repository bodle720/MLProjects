import os, json, logging, time
import boto3

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Env
BUCKET = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DB = os.environ["ICEBERG_DB"]
UPLOAD_STAGING_TABLE = os.environ["UPLOAD_STAGING_TABLE"]
SHA256_TABLE = os.environ["SHA256_TABLE"]
PHASH_TABLE = os.environ["PHASH_TABLE"]

athena = boto3.client("athena")
s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

# Tunables
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "400"))
MANIFEST_PREFIX = "batches/external-dedup"

def _wait_athena(qid, poll=1.5, timeout=900):
    start = time.time()
    while True:
        resp = athena.get_query_execution(QueryExecutionId=qid)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED","FAILED","CANCELLED"):
            if state != "SUCCEEDED":
                reason = resp["QueryExecution"]["Status"].get("StateChangeReason","")
                raise RuntimeError(f"Athena {qid} {state}: {reason}")
            return
        if time.time()-start > timeout:
            raise TimeoutError(f"Athena {qid} timed out")
        time.sleep(poll)

def _exec(sql):
    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP
    )["QueryExecutionId"]
    _wait_athena(qid)

def _sql_ident(db, table):
    return f'"{db}"."{table}"'

def _fetch_survivors(job_id):
    table = _sql_ident(ICEBERG_DB, UPLOAD_STAGING_TABLE)
    sql = f"""
    SELECT image_id, sha256_hash, phash, img_type
    FROM {table}
    WHERE job_id='{job_id}'
      AND validation_status='passed'
      AND (dedup_status IS NULL OR dedup_status='pending')
    """
    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP
    )["QueryExecutionId"]
    _wait_athena(qid)
    res = athena.get_query_results(QueryExecutionId=qid)
    rows = res.get("ResultSet",{}).get("Rows",[])
    headers = [c["VarCharValue"] for c in rows[0]["Data"]]
    out=[]
    for r in rows[1:]:
        vals=[c.get("VarCharValue") for c in r["Data"]]
        out.append(dict(zip(headers,vals)))
    return out

def _mark_external_duplicates(job_id, survivors):
    sha_table = dynamodb.Table(SHA256_TABLE)
    phash_table = dynamodb.Table(PHASH_TABLE)
    table = _sql_ident(ICEBERG_DB, UPLOAD_STAGING_TABLE)

    dup_ids = []
    for s in survivors:
        sha = s["sha256_hash"]
        phash = s["phash"]
        image_id = s["image_id"]

        # Check sha256
        resp = sha_table.get_item(Key={"sha256": sha})
        if "Item" in resp:
            matched_id = resp["Item"]["image_id"]
            sql = f"""
            UPDATE {table}
            SET dedup_status='external_duplicate',
                matched_image_id='{matched_id}'
            WHERE job_id='{job_id}' AND image_id='{image_id}'
            """
            _exec(sql)
            dup_ids.append(image_id)
            continue

        # Check phash
        resp = phash_table.get_item(Key={"phash": phash})
        if "Item" in resp:
            matched_id = resp["Item"]["image_id"]
            sql = f"""
            UPDATE {table}
            SET dedup_status='external_duplicate',
                matched_image_id='{matched_id}'
            WHERE job_id='{job_id}' AND image_id='{image_id}'
            """
            _exec(sql)
            dup_ids.append(image_id)

    return dup_ids

def _assign_bins(survivors, img_type):
    # survivors not marked as external duplicates
    bins = {}
    for s in survivors:
        phash = s["phash"]
        if img_type=="L":
            prefix = phash[:10]
        else:
            parts = phash.split("|")
            prefix = "|".join(p[:10] for p in parts)
        bins.setdefault(prefix,[]).append(s)
    return bins

def _emit_manifests(job_id, bins):
    manifest_s3_uris=[]
    for prefix,items in bins.items():
        for i in range(0,len(items),MAX_BATCH_SIZE):
            chunk=items[i:i+MAX_BATCH_SIZE]
            manifest={
                "job_id":job_id,
                "phash_prefix":prefix,
                "images":[
                    {"image_id":r["image_id"],"phash":r["phash"]}
                    for r in chunk
                ]
            }
            safe_prefix=prefix.replace("|","_")
            key=f"temp/image-upload/{job_id}/{MANIFEST_PREFIX}/phash-{safe_prefix}-part{i//MAX_BATCH_SIZE+1}.json"
            s3.put_object(
                Bucket=BUCKET,
                Key=key,
                Body=json.dumps(manifest).encode("utf-8"),
                ContentType="application/json"
            )
            manifest_s3_uris.append(f"s3://{BUCKET}/{key}")
    return manifest_s3_uris

def _canonical_indices_exist(img_type: str) -> bool:
    """Check if any canonical FAISS indices exist in S3 for this image type."""
    if img_type == "L":
        prefix = "canonical/indices/gray/"
    else:
        prefix = "canonical/indices/rgb/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=1)
    return "Contents" in resp and len(resp["Contents"]) > 0

def handler(event,context):
    job_id=event["job_id"]
    user=event.get("user","unknown")
    log.info(f"External dedup batching start for job {job_id}")

    survivors=_fetch_survivors(job_id)
    if not survivors:
        log.info("No survivors to process")
        return {"job_id":job_id,"user":user,"manifests":[]}

    # Mark external duplicates
    dup_ids=_mark_external_duplicates(job_id,survivors)
    survivors=[s for s in survivors if s["image_id"] not in dup_ids]

    if not survivors:
        log.info("All survivors were external duplicates")
        return {"job_id":job_id,"user":user,"manifests":[]}

    img_type=survivors[0]["img_type"]

    if not _canonical_indices_exist(img_type):
        log.info(f"No canonical indices found for img_type={img_type}. "
                 f"Skipping external dedup for job {job_id}.")
        return {
            "job_id": job_id,
            "user": user,
            "manifests": [],   # empty list means Map state is a pass
            "img_type": img_type
        }

    bins=_assign_bins(survivors,img_type)
    manifests=_emit_manifests(job_id,bins)

    log.info(f"Prepared {len(manifests)} external-dedup manifests for job {job_id}")
    return {"job_id":job_id,
            "user":user,
            "manifests":manifests,
            "img_type":img_type,
            "label_type": }
