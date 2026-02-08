import json
import logging
import math
from decimal import Decimal
from datetime import datetime, date
from typing import Iterator, Any, Iterable, Dict

import boto3

import pyarrow.dataset as ds
import s3fs

logger = logging.getLogger(__name__)

s3 = boto3.client("s3")

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

def jsonl_stream_to_s3(bucket: str, key: str, rows: Iterable[Dict[str, Any]]) -> None:
    fs = s3fs.S3FileSystem()
    path = f"s3://{bucket}/{key}"
    with fs.open(path, "wb", block_size=8 * 1024 * 1024) as f:  # 8MB parts
        for r in rows:
            f.write((json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8"))

def read_parquet_rows_from_s3_uris(s3_uris: Iterable[str]) -> Iterator[dict[str, Any]]:
    fs = s3fs.S3FileSystem()
    for uri in s3_uris:
        path = uri.replace("s3://", "", 1)
        dataset = ds.dataset(path, filesystem=fs, format="parquet")
        scanner = dataset.scanner(batch_size=10_000, use_threads=True)
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                yield normalize_row(row)