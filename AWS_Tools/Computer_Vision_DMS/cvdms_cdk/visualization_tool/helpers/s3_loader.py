"""
S3 loading utilities for the local CVDMS Dataset Viewer.

This module is intentionally small and UI-agnostic. It handles:
- Creating boto3 sessions/clients
- Listing available CVDMS dataset IDs
- Listing available versions for a dataset
- Reading JSON objects from S3

Expected S3 layout:

    s3://<datasets-bucket>/datasets/<dataset_id>/v<version>/visualization/*.json
"""

import json
import re
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound


_DATASET_PREFIX = "datasets/"
_VERSION_RE = re.compile(r"^v(\d+)$")


@dataclass(frozen=True)
class S3ObjectRef:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


class S3LoadError(RuntimeError):
    """Raised when the viewer cannot load required S3 data."""


def create_boto3_session(
    *,
    profile_name: str | None = None,
    region_name: str | None = None,
) -> boto3.Session:
    """
    Create a boto3 session.

    Parameters
    ----------
    profile_name:
        Optional AWS profile name. If None or empty, boto3 uses its default
        credential resolution chain.
    region_name:
        Optional AWS region.

    Returns
    -------
    boto3.Session
    """
    clean_profile = profile_name.strip() if isinstance(profile_name, str) else None
    clean_region = region_name.strip() if isinstance(region_name, str) else None

    try:
        if clean_profile:
            return boto3.Session(profile_name=clean_profile, region_name=clean_region)
        return boto3.Session(region_name=clean_region)
    except ProfileNotFound as exc:
        raise S3LoadError(f"AWS profile not found: {clean_profile!r}") from exc


def create_s3_client(
    *,
    profile_name: str | None = None,
    region_name: str | None = None,
) -> BaseClient:
    """
    Create an S3 client from a boto3 session.
    """
    session = create_boto3_session(
        profile_name=profile_name,
        region_name=region_name,
    )

    try:
        return session.client("s3")
    except NoCredentialsError as exc:
        raise S3LoadError(
            "AWS credentials were not found. Configure credentials or provide an AWS profile."
        ) from exc


def _list_common_prefixes(
    s3_client: BaseClient,
    *,
    bucket: str,
    prefix: str,
) -> list[str]:
    """
    List immediate child prefixes under a given S3 prefix.

    Uses Delimiter='/' so this behaves like listing folders.
    """
    prefixes: list[str] = []
    continuation_token: str | None = None

    while True:
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": prefix,
            "Delimiter": "/",
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        try:
            response = s3_client.list_objects_v2(**kwargs)
        except ClientError as exc:
            raise S3LoadError(
                f"Failed to list S3 prefixes under s3://{bucket}/{prefix}: {exc}"
            ) from exc

        for item in response.get("CommonPrefixes", []):
            child_prefix = item.get("Prefix")
            if child_prefix:
                prefixes.append(child_prefix)

        if not response.get("IsTruncated"):
            break

        continuation_token = response.get("NextContinuationToken")
        if not continuation_token:
            break

    return prefixes


def _s3_key_exists(
    s3_client: BaseClient,
    *,
    bucket: str,
    key: str,
) -> bool:
    """
    Return True if the S3 object exists, False if it does not exist.
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise S3LoadError(f"Failed to check S3 object s3://{bucket}/{key}: {exc}") from exc


def list_dataset_ids(
    s3_client: BaseClient,
    *,
    bucket: str,
    require_visualization_artifacts: bool = False,
) -> list[str]:
    """
    Discover dataset IDs under:

        datasets/<dataset_id>/

    If require_visualization_artifacts=True, only returns dataset IDs that appear
    to have at least one version containing visualization/overview.json.
    """
    bucket = bucket.strip()
    if not bucket:
        raise ValueError("bucket cannot be empty")

    prefixes = _list_common_prefixes(
        s3_client,
        bucket=bucket,
        prefix=_DATASET_PREFIX,
    )

    dataset_ids: list[str] = []

    for prefix in prefixes:
        # "datasets/my_dataset/" -> "my_dataset"
        parts = prefix.rstrip("/").split("/")
        if len(parts) != 2:
            continue

        dataset_id = parts[-1].strip()
        if not dataset_id:
            continue

        if require_visualization_artifacts:
            versions = list_dataset_versions(
                s3_client,
                bucket=bucket,
                dataset_id=dataset_id,
                require_visualization_artifacts=True,
            )
            if not versions:
                continue

        dataset_ids.append(dataset_id)

    return sorted(set(dataset_ids))


def list_dataset_versions(
    s3_client: BaseClient,
    *,
    bucket: str,
    dataset_id: str,
    require_visualization_artifacts: bool = True,
) -> list[int]:
    """
    Discover available versions under:

        datasets/<dataset_id>/v1/
        datasets/<dataset_id>/v2/
        ...

    By default, only versions containing visualization/overview.json are returned.
    """
    bucket = bucket.strip()
    dataset_id = dataset_id.strip()

    if not bucket:
        raise ValueError("bucket cannot be empty")
    if not dataset_id:
        raise ValueError("dataset_id cannot be empty")

    dataset_prefix = f"datasets/{dataset_id}/"
    prefixes = _list_common_prefixes(
        s3_client,
        bucket=bucket,
        prefix=dataset_prefix,
    )

    versions: list[int] = []

    for prefix in prefixes:
        # "datasets/my_dataset/v2/" -> "v2"
        version_folder = prefix.rstrip("/").split("/")[-1]
        match = _VERSION_RE.match(version_folder)
        if not match:
            continue

        version = int(match.group(1))

        if require_visualization_artifacts:
            overview_key = (
                f"datasets/{dataset_id}/v{version}/visualization/overview.json"
            )
            if not _s3_key_exists(s3_client, bucket=bucket, key=overview_key):
                continue

        versions.append(version)

    return sorted(set(versions))


def read_text_from_s3(
    s3_client: BaseClient,
    *,
    bucket: str,
    key: str,
) -> str:
    """
    Read an S3 object as UTF-8 text.
    """
    bucket = bucket.strip()
    key = key.strip()

    if not bucket:
        raise ValueError("bucket cannot be empty")
    if not key:
        raise ValueError("key cannot be empty")

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        raw = response["Body"].read()
        return raw.decode("utf-8")
    except ClientError as exc:
        raise S3LoadError(f"Failed to read s3://{bucket}/{key}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise S3LoadError(f"Object is not valid UTF-8 text: s3://{bucket}/{key}") from exc


def read_json_from_s3(
    s3_client: BaseClient,
    *,
    bucket: str,
    key: str,
) -> dict[str, Any]:
    """
    Read an S3 object and parse it as a JSON object.

    The visualization Lambda writes JSON objects, not JSON arrays, so this
    function enforces a dictionary return type.
    """
    text = read_text_from_s3(s3_client, bucket=bucket, key=key)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise S3LoadError(f"Object is not valid JSON: s3://{bucket}/{key}") from exc

    if not isinstance(payload, dict):
        raise S3LoadError(
            f"Expected JSON object at s3://{bucket}/{key}, got {type(payload).__name__}"
        )

    return payload


def build_visualization_key(
    *,
    dataset_id: str,
    version: int,
    filename: str,
) -> str:
    """
    Build the canonical S3 key for a visualization artifact.
    """
    dataset_id = dataset_id.strip()
    filename = filename.strip()

    if not dataset_id:
        raise ValueError("dataset_id cannot be empty")
    if version < 1:
        raise ValueError("version must be >= 1")
    if not filename:
        raise ValueError("filename cannot be empty")

    return f"datasets/{dataset_id}/v{version}/visualization/{filename}"