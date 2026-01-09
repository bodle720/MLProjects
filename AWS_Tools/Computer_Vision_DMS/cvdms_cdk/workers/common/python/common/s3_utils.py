import json
import time
import logging
import math
from decimal import Decimal
from datetime import datetime, date
from typing import List, Iterator, Union, Any, Iterable, Mapping, Optional, Sequence

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionClosedError
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

s3 = boto3.client("s3")

_TRANSIENT_CODES = {
    "Throttling", "ThrottlingException",
    "RequestTimeout", "RequestTimeoutException",
    "SlowDown",
    "InternalError", "ServiceUnavailable",
}

def delete_s3_prefix(bucket: str, prefix: str, task_name: str, batch_size: int = 100) -> None:
    """
    Delete all objects under s3://{bucket}/{prefix} in batches (default 100).
    Raises on any AWS error or if S3 reports per-key delete errors.

    Notes:
      - S3 DeleteObjects supports up to 1000 keys per request; we default to 100.
      - list_objects_v2 can return pages without "Contents".
    """
    if batch_size < 1 or batch_size > 1000:
        raise ValueError(f"{task_name} batch_size must be between 1 and 1000, got {batch_size}")

    if prefix.strip() == "":
        raise ValueError(f"{task_name} refusing to delete empty prefix for bucket {bucket}")

    paginator = s3.get_paginator("list_objects_v2")
    to_delete: list[dict[str, str]] = []

    def flush() -> None:
        nonlocal to_delete
        if not to_delete:
            return

        try:
            resp = s3.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})
        except ClientError as e:
            logger.error(
                f"{task_name} Failed to delete objects for s3://{bucket}/{prefix} "
                f"(batch_size={len(to_delete)}): {e}"
            )
            raise

        # DeleteObjects can succeed but still report per-key errors.
        errors = resp.get("Errors", [])
        if errors:
            # Log a small sample to avoid huge logs
            sample = errors[:10]
            logger.error(
                f"{task_name} S3 reported {len(errors)} delete error(s) for s3://{bucket}/{prefix}. "
                f"Sample: {sample}"
            )
            raise RuntimeError(
                f"{task_name} S3 delete_objects returned {len(errors)} errors for s3://{bucket}/{prefix}"
            )

        to_delete = []

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                k = obj.get("Key")
                if not k:
                    continue
                to_delete.append({"Key": k})
                if len(to_delete) >= batch_size:
                    flush()

        flush()
    except ClientError as e:
        logger.error(f"{task_name} Error while listing/deleting s3://{bucket}/{prefix}: {e}")
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

def parse_s3_uri(s3_uri: str, task_name: str) -> tuple[str, str]:
    if not isinstance(s3_uri, str):
        raise ValueError(f"{task_name} s3_uri must be a str, got {type(s3_uri).__name__}")

    s3_uri = s3_uri.strip()
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"{task_name} invalid s3 uri (missing s3://): {s3_uri!r}")

    rest = s3_uri[5:]
    if "/" not in rest:
        raise ValueError(f"{task_name} invalid s3 uri (missing key): {s3_uri!r}")

    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"{task_name} invalid s3 uri (empty bucket or key): {s3_uri!r}")

    return bucket, key

def s3_read_json(bucket: str, key: str, task_name: str) -> dict[str, Any]:
    resp = read_obj_with_retry(bucket, key, task_name)
    if resp is None:
        raise RuntimeError(f"{task_name} unable to load s3://{bucket}/{key} after retries")

    try:
        raw = resp["Body"].read()
        text = raw.decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"{task_name} failed reading/decoding s3://{bucket}/{key}: {e}") from e

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{task_name} invalid JSON in s3://{bucket}/{key}: {e.msg} (pos={e.pos})") from e

    if not isinstance(obj, dict):
        raise ValueError(f"{task_name} expected JSON object in s3://{bucket}/{key}, got {type(obj).__name__}")

    return obj

def s3_read_jsonl(bucket: str,
                key: str,
                task_name: str,
                *,
                encoding: str = "utf-8",
                allow_bom: bool = True,
                strict_json: bool = True) -> Iterator[dict[str, Any]]:
    """
    Stream JSON objects from an S3 JSONL/NDJSON file (one JSON object per line).
    Yields only dict-shaped JSON objects.
    """
    resp = read_obj_with_retry(bucket, key, task_name)

    if resp is None:
        raise RuntimeError(f"{task_name} unable to load s3://{bucket}/{key} after retries")

    body = resp["Body"]

    saw_first_content_line = False

    for lineno, raw in enumerate(body.iter_lines(), start=1):
        if not raw:
            continue

        try:
            line = raw.decode(encoding)
        except UnicodeDecodeError as e:
            msg = f"{task_name} s3://{bucket}/{key} line {lineno}: decode failed ({encoding}): {e}"
            if strict_json:
                raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end, msg) from e
            continue

        line = line.strip()
        if not line:
            continue

        if allow_bom and not saw_first_content_line:
            if line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")
            saw_first_content_line = True

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            msg = f"{task_name} s3://{bucket}/{key} line {lineno}: invalid JSON: {e.msg} (pos={e.pos})"
            if strict_json:
                raise ValueError(msg) from e
            continue

        if not isinstance(obj, dict):
            msg = f"{task_name} s3://{bucket}/{key} line {lineno}: expected JSON object, got {type(obj).__name__}"
            if strict_json:
                raise ValueError(msg)
            continue

        yield obj

def s3_read_jsonl_list(bucket: str,
                        jsonl_keys: Sequence[str],
                        task_name: str,
                        *,
                        encoding: str = "utf-8",
                        allow_bom: bool = True,
                        strict_json: bool = True) -> Iterator[dict[str, Any]]:
    for key in jsonl_keys:
        yield from s3_read_jsonl(
            bucket,
            key,
            task_name,
            encoding=encoding,
            allow_bom=allow_bom,
            strict_json=strict_json
        )

def write_s3_obj(bucket: str,
                key: str,
                content: Union[str, bytes, bytearray, memoryview],
                content_type: str,
                task_name: str,
                encoding: str = "utf-8",
                retries: int = 3,
                delay: Union[int, float] = 1.0) -> str:

    if isinstance(content, str):
        body = content.encode(encoding)
    elif isinstance(content, bytes):
        body = content
    elif isinstance(content, (bytearray, memoryview)):
        body = bytes(content)
    else:
        raise TypeError(f"{task_name} content must be str/bytes-like, got {type(content).__name__}")

    delay_s = float(delay)
    for attempt in range(retries):
        try:
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
            return f"s3://{bucket}/{key}"

        except (EndpointConnectionError, ConnectionClosedError) as e:
            # network/transient
            if attempt < retries - 1:
                time.sleep(delay_s)
                continue
            raise RuntimeError(f"{task_name} put_object failed for s3://{bucket}/{key}: {e}") from e

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in _TRANSIENT_CODES and attempt < retries - 1:
                time.sleep(delay_s)
                continue
            raise RuntimeError(f"{task_name} put_object failed for s3://{bucket}/{key}: {e}") from e

    # unreachable
    raise RuntimeError(f"{task_name} unexpected: retries exhausted for s3://{bucket}/{key}")

def read_obj_with_retry(bucket: str,
                        key: str,
                        task_name: str,
                        retries: int = 5,
                        delay: Union[float, int] = 2.0) -> Optional[Mapping[str, Any]]:

    if not isinstance(retries, int) or retries < 1:
        raise ValueError(f"{task_name} retries must be an integer and 1 or higher, got {retries}")

    try:
        delay_s = float(delay)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{task_name} delay must be int or float, got {delay!r}") from e

    if delay_s < 0:
        raise ValueError(f"{task_name} delay must be >= 0, got {delay_s}")

    for attempt in range(retries):
        try:
            return s3.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "NoSuchKey" and attempt < retries - 1:
                time.sleep(delay_s)
                continue
            raise RuntimeError(f"{task_name} error loading s3://{bucket}/{key}: {e}") from e

    return None

def to_jsonable(v: Any) -> Any:
    if v is None:
        return None

    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()

    if hasattr(v, "as_py"):
        v = v.as_py()

    if isinstance(v, datetime):
        return v.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(v, date):
        return v.isoformat()

    if isinstance(v, Decimal):
        f = float(v)
        return None if not math.isfinite(f) else f

    if isinstance(v, float):
        return None if not math.isfinite(v) else v

    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")

    return v

def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: to_jsonable(v) for k, v in row.items()}

def read_parquet_rows_from_s3_uris(s3_uris: Iterable[str]) -> Iterator[dict[str, Any]]:
    fs = s3fs.S3FileSystem()
    for uri in s3_uris:
        path = uri.replace("s3://", "", 1)
        dataset = ds.dataset(path, filesystem=fs, format="parquet")
        scanner = dataset.scanner(batch_size=10_000, use_threads=True)
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                yield normalize_row(row)

def get_key_basename(key: str) -> str:
    return key.rsplit("/", 1)[-1]