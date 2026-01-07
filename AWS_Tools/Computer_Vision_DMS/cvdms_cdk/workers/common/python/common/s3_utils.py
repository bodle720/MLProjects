import json
import logging
from typing import Dict, List, Iterator

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

def delete_s3_prefix(bucket: str, prefix: str, batch_size: int = 100) -> None:
    """
    Delete all objects under s3://{bucket}/{prefix} in batches (default 100).
    Raises on any AWS error or if S3 reports per-key delete errors.

    Notes:
      - S3 DeleteObjects supports up to 1000 keys per request; we default to 100.
      - list_objects_v2 can return pages without "Contents".
    """
    if batch_size < 1 or batch_size > 1000:
        raise ValueError(f"batch_size must be between 1 and 1000, got {batch_size}")

    paginator = s3.get_paginator("list_objects_v2")
    to_delete: list[dict] = []

    def flush() -> None:
        nonlocal to_delete
        if not to_delete:
            return

        try:
            resp = s3.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})
        except ClientError as e:
            logger.error(
                f"Failed to delete objects for s3://{bucket}/{prefix} "
                f"(batch_size={len(to_delete)}): {e}"
            )
            raise

        # DeleteObjects can succeed but still report per-key errors.
        errors = resp.get("Errors", [])
        if errors:
            # Log a small sample to avoid huge logs
            sample = errors[:10]
            logger.error(
                f"S3 reported {len(errors)} delete error(s) for s3://{bucket}/{prefix}. "
                f"Sample: {sample}"
            )
            raise RuntimeError(
                f"S3 delete_objects returned {len(errors)} errors for s3://{bucket}/{prefix}"
            )

        to_delete = []

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                to_delete.append({"Key": obj["Key"]})
                if len(to_delete) >= batch_size:
                    flush()

        flush()
    except ClientError as e:
        logger.error(f"Error while listing/deleting s3://{bucket}/{prefix}: {e}")
        raise

def s3_list_keys(bucket: str, prefix: str) -> List[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                keys.append(key)
    return keys

def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not isinstance(s3_uri, str) or not s3_uri.startswith("s3://") or s3_uri.count("/") < 3:
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    b, k = s3_uri[5:].split("/", 1)
    return b, k

def s3_read_json(bucket: str, key: str) -> Dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))

def s3_read_jsonl(
    bucket: str,
    key: str,
    *,
    encoding: str = "utf-8",          # normal decode
    allow_bom: bool = True,           # handle UTF-8 BOM on first line
    strict_json: bool = True,         # if False: skip malformed JSON lines instead of raising
) -> Iterator[Dict]:
    """
    Stream JSON objects from an S3 JSONL/NDJSON file.

    - Uses resp["Body"].iter_lines() so it does NOT load the whole object into memory.
    - Skips empty/whitespace-only lines.
    - Optionally strips UTF-8 BOM on the first non-empty line (common when files are saved as utf-8-sig).
    - Raises a helpful error with line number if JSON is invalid (unless strict_json=False).
    """
    resp = s3.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]

    saw_first_content_line = False

    for lineno, raw in enumerate(body.iter_lines(), start=1):
        if not raw:
            continue

        # Decode bytes -> str
        try:
            line = raw.decode(encoding)
        except UnicodeDecodeError as e:
            msg = f"s3://{bucket}/{key} line {lineno}: decode failed ({encoding}): {e}"
            if strict_json:
                raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end, msg) from e
            continue

        line = line.strip()
        if not line:
            continue

        # Strip BOM only on the first non-empty content line
        if allow_bom and not saw_first_content_line:
            # \ufeff is the Unicode BOM char that shows up when a UTF-8 BOM is decoded
            if line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")
            saw_first_content_line = True

        # Parse JSON
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            msg = (
                f"s3://{bucket}/{key} line {lineno}: invalid JSON: {e.msg} "
                f"(pos={e.pos})"
            )
            if strict_json:
                raise ValueError(msg) from e
            continue

        if not isinstance(obj, dict):
            msg = f"s3://{bucket}/{key} line {lineno}: expected JSON object, got {type(obj).__name__}"
            if strict_json:
                raise ValueError(msg)
            continue

        yield obj

def s3_read_jsonl_list(bucket: str, jsonl_keys: list[str]) -> Iterator[Dict]:
    for key in jsonl_keys:
        yield from s3_read_jsonl(bucket, key)