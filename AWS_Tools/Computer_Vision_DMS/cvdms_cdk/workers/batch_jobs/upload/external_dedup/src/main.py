import os, json, logging, boto3, time
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Environment variables
BUCKET = os.environ["FILE_BUCKET_NAME"]
MANIFEST_S3_KEY = os.environ["MANIFEST_S3_KEY"]
JOB_ID = os.environ["JOB_ID"]
USER = os.environ.get("USER", "unknown")
ICEBERG_DB = os.environ["ICEBERG_DB"]
UPLOAD_STAGING_TABLE = os.environ["UPLOAD_STAGING_TABLE"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
IMG_TYPE = os.environ["IMG_TYPE"]

# Thresholds
GRAY_THRESHOLD = 8
RGB_THRESHOLD = 24

s3 = boto3.client("s3")
athena = boto3.client("athena")

def _wait(qid, poll=1.5, timeout=900):
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
    _wait(qid)

def _load_manifest():
    bucket, key = MANIFEST_S3_KEY.replace("s3://","").split("/",1)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)

def _phash_to_bits(phash, img_type):
    if img_type == "L":
        return np.array([int(b) for b in bin(int(phash,16))[2:].zfill(64)], dtype="float32")
    else:
        parts = phash.split("|")
        bits = "".join(parts)
        return np.array([int(b) for b in bin(int(bits,16))[2:].zfill(192)], dtype="float32")

def _load_faiss_index_and_mapping(prefix, img_type):
    if img_type == "L":
        base_key = f"canonical/indices/gray/{prefix}"
    else:
        base_key = f"canonical/indices/rgb/{prefix}"

    # Download index
    index_key = f"{base_key}.index"
    local_index = f"/tmp/{prefix}.index"
    s3.download_file(BUCKET, index_key, local_index)
    index = faiss.read_index(local_index)

    # Download mapping
    mapping_key = f"{base_key}.ids.json"
    local_map = f"/tmp/{prefix}.ids.json"
    s3.download_file(BUCKET, mapping_key, local_map)
    with open(local_map, "r") as f:
        id_map = json.load(f)  # keys are str(row_id), values are canonical image_id

    return index, id_map

def main():
    manifest = _load_manifest()
    job_id = manifest["job_id"]
    prefix = manifest["phash_prefix"]
    images = manifest["images"]

    if not images:
        log.info("No images in manifest")
        return

    index, id_map = _load_faiss_index_and_mapping(prefix, IMG_TYPE)

    survivors = []
    for img in images:
        vec = _phash_to_bits(img["phash"], IMG_TYPE).reshape(1,-1)
        D, I = index.search(vec, 1)
        dist = int(D[0][0])
        if (IMG_TYPE=="L" and dist <= GRAY_THRESHOLD) or (IMG_TYPE=="RGB" and dist <= RGB_THRESHOLD):
            row_id = str(I[0][0])
            matched_id = id_map.get(row_id, row_id)  # fallback to row_id if not found
            sql = f"""
            UPDATE "{ICEBERG_DB}"."{UPLOAD_STAGING_TABLE}"
            SET dedup_status='external_duplicate',
                matched_image_id='{matched_id}'
            WHERE job_id='{job_id}' AND image_id='{img["image_id"]}'
            """
            _exec(sql)
            log.info(f"Image {img['image_id']} marked external_duplicate (matched {matched_id})")
        else:
            survivors.append(img["image_id"])

    log.info(f"Job {job_id}: {len(images)} checked, {len(survivors)} survivors remain")

if __name__=="__main__":
    main()


