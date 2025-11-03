import os, json, logging, time
import boto3
from itertools import combinations

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Environment
BUCKET = os.environ["FILE_BUCKET_NAME"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ICEBERG_DB = os.environ["ICEBERG_DB"]
UPLOAD_STAGING_TABLE = os.environ["UPLOAD_STAGING_TABLE"]

MANIFEST_S3_KEY = os.environ["MANIFEST_S3_KEY"]
JOB_ID = os.environ["JOB_ID"]

s3 = boto3.client("s3")
athena = boto3.client("athena")

# Thresholds
GRAY_THRESHOLD = 8       # bits out of 64
RGB_THRESHOLD = 24       # bits out of 192 (≈8 per channel)

def _wait(qid):
    while True:
        resp = athena.get_query_execution(QueryExecutionId=qid)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED","FAILED","CANCELLED"):
            if state != "SUCCEEDED":
                reason = resp["QueryExecution"]["Status"].get("StateChangeReason","")
                raise RuntimeError(f"Athena {qid} {state}: {reason}")
            return
        time.sleep(1.5)

def _exec(sql):
    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_S3},
        WorkGroup=ATHENA_WORKGROUP
    )["QueryExecutionId"]
    _wait(qid)

def hamming_distance_hex(h1, h2):
    return bin(int(h1,16) ^ int(h2,16)).count("1")

def normalize_phash(phash):
    parts = phash.split("|")
    if len(parts) == 1:
        return parts[0], "L"   # grayscale
    elif len(parts) == 3:
        return "".join(parts), "RGB"  # concatenate to 192 bits
    else:
        raise ValueError(f"Malformed phash: {phash}")

def main():
    # Load manifest
    bucket, key = MANIFEST_S3_KEY.replace("s3://","").split("/",1)
    manifest = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    images = manifest["images"]

    log.info(f"Job {JOB_ID}: internal dedup similarity on {len(images)} images")

    # Normalize phashes
    for img in images:
        norm, mode = normalize_phash(img["phash"])
        img["phash_norm"] = norm
        img["mode"] = mode

    duplicates = set()
    survivors = set(img["image_id"] for img in images)

    # Pairwise comparisons
    for a, b in combinations(images, 2):
        if a["mode"] != b["mode"]:
            continue  # skip cross-type comparisons
        dist = hamming_distance_hex(a["phash_norm"], b["phash_norm"])
        if a["mode"] == "L":
            if dist <= GRAY_THRESHOLD:
                duplicates.add(b["image_id"])
        else:  # RGB
            if dist <= RGB_THRESHOLD:
                duplicates.add(b["image_id"])

    # Update staging table
    if duplicates:
        table = f'"{ICEBERG_DB}"."{UPLOAD_STAGING_TABLE}"'
        dup_list = ",".join(f"'{d}'" for d in duplicates)
        sql = f"""
        UPDATE {table}
        SET dedup_status='internal_duplicate'
        WHERE job_id='{JOB_ID}' AND image_id IN ({dup_list})
        """
        _exec(sql)
        log.info(f"Marked {len(duplicates)} images as internal duplicates")
    else:
        log.info("No internal duplicates found")

if __name__=="__main__":
    main()
