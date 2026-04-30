"""
S3 I/O helpers for CVDMS training projects.

This module intentionally stays small and dependency-light. It provides common
utilities for reading text, JSON, JSONL, and bytes from S3, plus S3 URI parsing.

Higher-level modules such as metadata.py, manifests.py, and image_loading.py
should use these helpers rather than duplicating boto3 access logic.
"""

import json
from dataclasses import dataclass
from typing import Any, Iterable

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

@dataclass(frozen=True)
class S3Uri:
    """
    Parsed representation of an S3 URI.

    Example:
        s3://my-bucket/path/to/file.json

    becomes:
        bucket = "my-bucket"
        key = "path/to/file.json"
    """

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

def make_s3_client(
    *,
    profile_name: str | None = None,
    region_name: str | None = None,
) -> BaseClient:
    """
    Create a boto3 S3 client.

    If profile_name is provided, a boto3 Session is created with that profile.
    Otherwise, boto3 uses the normal AWS credential chain.

    This is useful for local PyCharm runs where a named AWS profile is often used,
    while still working naturally in SageMaker or other AWS runtimes.
    """
    if profile_name:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        return session.client("s3")

    if region_name:
        return boto3.client("s3", region_name=region_name)

    return boto3.client("s3")

def parse_s3_uri(uri: str) -> S3Uri:
    """
    Parse an S3 URI into bucket/key components.

    Args:
        uri: S3 URI in the form s3://bucket/key

    Returns:
        S3Uri(bucket=..., key=...)

    Raises:
        TypeError: if uri is not a string
        ValueError: if uri is not a valid S3 URI
    """
    if not isinstance(uri, str):
        raise TypeError(f"Expected S3 URI string, got {type(uri).__name__}")

    text = uri.strip()
    if not text:
        raise ValueError("S3 URI cannot be empty")

    if not text.startswith("s3://"):
        raise ValueError(f"Expected S3 URI starting with 's3://', got: {uri!r}")

    without_scheme = text[5:]
    bucket, sep, key = without_scheme.partition("/")

    if not bucket:
        raise ValueError(f"S3 URI is missing bucket: {uri!r}")

    if not sep or not key:
        raise ValueError(f"S3 URI is missing key: {uri!r}")

    key = key.lstrip("/")
    if not key:
        raise ValueError(f"S3 URI is missing non-empty key: {uri!r}")

    return S3Uri(bucket=bucket, key=key)

def read_s3_bytes(
    uri: str,
    *,
    s3_client: BaseClient | None = None,
) -> bytes:
    """
    Read an S3 object as bytes.

    Args:
        uri: S3 URI to read
        s3_client: optional boto3 S3 client

    Returns:
        Object body as bytes
    """
    s3 = s3_client or boto3.client("s3")
    parsed = parse_s3_uri(uri)

    try:
        obj = s3.get_object(Bucket=parsed.bucket, Key=parsed.key)
        return obj["Body"].read()
    except ClientError as exc:
        raise RuntimeError(f"Failed to read S3 object: {parsed.uri}") from exc

def read_s3_text(
    uri: str,
    *,
    s3_client: BaseClient | None = None,
    encoding: str = "utf-8-sig",
) -> str:
    """
    Read an S3 object as text.

    Uses utf-8-sig by default so files with a UTF-8 BOM are handled safely.
    """
    data = read_s3_bytes(uri, s3_client=s3_client)
    return data.decode(encoding)

def read_s3_json(
    uri: str,
    *,
    s3_client: BaseClient | None = None,
) -> dict[str, Any]:
    """
    Read an S3 JSON object.

    Raises:
        ValueError if the JSON root is not an object/dict.
    """
    text = read_s3_text(uri, s3_client=s3_client)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {uri}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object at {uri}, got {type(payload).__name__}"
        )

    return payload

def iter_s3_jsonl(
    uri: str,
    *,
    s3_client: BaseClient | None = None,
    strict: bool = True,
) -> Iterable[dict[str, Any]]:
    """
    Iterate over JSONL records from S3.

    Args:
        uri: S3 URI of the JSONL file
        s3_client: optional boto3 S3 client
        strict:
            If True, invalid/blank rows raise errors.
            Blank lines are always ignored.
            If False, invalid JSON rows are skipped.

    Yields:
        One dict per JSONL line.

    Notes:
        This reads the object body line-by-line instead of loading the entire file
        into memory as text first. That is better for larger manifests.
    """
    s3 = s3_client or boto3.client("s3")
    parsed = parse_s3_uri(uri)

    try:
        obj = s3.get_object(Bucket=parsed.bucket, Key=parsed.key)
    except ClientError as exc:
        raise RuntimeError(f"Failed to open S3 JSONL object: {parsed.uri}") from exc

    body = obj["Body"]

    for line_number, raw_line in enumerate(body.iter_lines(), start=1):
        if not raw_line:
            continue

        try:
            line = raw_line.decode("utf-8-sig" if line_number == 1 else "utf-8").strip()
        except UnicodeDecodeError as exc:
            if strict:
                raise ValueError(
                    f"Could not decode JSONL line {line_number} from {parsed.uri}: {exc}"
                ) from exc
            continue

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise ValueError(
                    f"Invalid JSONL record in {parsed.uri} at line {line_number}: {exc}"
                ) from exc
            continue

        if not isinstance(record, dict):
            if strict:
                raise ValueError(
                    f"Expected JSON object in {parsed.uri} at line {line_number}, "
                    f"got {type(record).__name__}"
                )
            continue

        yield record

def read_s3_jsonl(
    uri: str,
    *,
    s3_client: BaseClient | None = None,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """
    Read a JSONL file from S3 into a list.

    For very large files, prefer iter_s3_jsonl().
    """
    return list(iter_s3_jsonl(uri, s3_client=s3_client, strict=strict))

def s3_object_exists(
    uri: str,
    *,
    s3_client: BaseClient | None = None,
) -> bool:
    """
    Return True if the S3 object exists, otherwise False.

    Permission note:
        If the caller lacks permission to HeadObject, this may raise instead of
        returning False.
    """
    s3 = s3_client or boto3.client("s3")
    parsed = parse_s3_uri(uri)

    try:
        s3.head_object(Bucket=parsed.bucket, Key=parsed.key)
        return True
    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))

        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False

        raise RuntimeError(f"Failed to check S3 object existence: {parsed.uri}") from exc

def join_s3_uri(bucket: str, key: str) -> str:
    """
    Build an S3 URI from a bucket and key.

    Args:
        bucket: S3 bucket name
        key: S3 object key

    Returns:
        s3://bucket/key
    """
    bucket_text = str(bucket).strip()
    key_text = str(key).strip().lstrip("/")

    if not bucket_text:
        raise ValueError("bucket cannot be empty")

    if not key_text:
        raise ValueError("key cannot be empty")

    return f"s3://{bucket_text}/{key_text}"