# -*- coding: utf-8 -*-
"""
Helper functions for the bootstrap process.
"""

import re
import logging

from botocore.exceptions import ClientError
import botocore.session

# Misc. helpers

def normalize_root(s3_root: str) -> str:
    """Strip leading/trailing slashes and collapse multiple consecutive slashes."""
    root = s3_root.strip().strip("/")
    root = re.sub(r"/+", "/", root)
    return root

VALID_REGIONS = [
    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
    "af-south-1", "ap-east-1", "ap-south-1", "ap-south-2",
    "ap-southeast-1", "ap-southeast-2", "ap-southeast-3", "ap-southeast-4",
    "ap-northeast-1", "ap-northeast-2", "ap-northeast-3",
    "ca-central-1", "ca-west-1",
    "eu-central-1", "eu-central-2", "eu-west-1", "eu-west-2", "eu-west-3",
    "eu-north-1", "eu-south-1", "eu-south-2",
    "il-central-1", "me-south-1", "me-central-1", "sa-east-1"
]

# Validation helpers

def is_profile_name_valid(profile_name: str) -> str | None:
    """
    Validate that the given AWS profile name is a non-empty string
    and exists in the local AWS configuration.

    Returns:
        None if the profile name is valid.
        A descriptive error string if invalid.
    """
    if not isinstance(profile_name, str) or not profile_name.strip():
        return "Invalid profile name: it must be a non-empty string."

    session = botocore.session.get_session()
    available_profiles = session.available_profiles

    if profile_name not in available_profiles:
        return (f"Profile '{profile_name}' not found in AWS config. "
                f"Available profiles: {available_profiles}")

    return None

def is_valid_region(region: str) -> str | None:
    if region not in VALID_REGIONS:
        return f"Invalid region: {region}. Must be one of {', '.join(VALID_REGIONS)}"
    return None

def is_valid_cdk_stack_name(name: str) -> str | None:
    if not isinstance(name, str):
        return "Stack name must be a string."
    if not name.strip():
        return "Stack name cannot be empty or whitespace."
    if len(name) > 128:
        return f"Stack name too long ({len(name)} > 128)."
    pattern = r'^[A-Za-z](?:[A-Za-z0-9-]*[A-Za-z0-9])?$'
    if not re.match(pattern, name):
        return ("Invalid stack name format. Must start with a letter, "
                "contain only letters/numbers/hyphens, and not end with a hyphen.")
    return None

def is_valid_bucket_name(bucket_name: str) -> str | None:
    if not isinstance(bucket_name, str):
        return "Bucket name must be a string."
    if not (3 <= len(bucket_name) <= 63):
        return f"Bucket name length {len(bucket_name)} is invalid (must be 3–63)."
    if not bucket_name.strip():
        return "Bucket name cannot be empty or whitespace."
    pattern = re.compile(
        r'^(?!\d+\.\d+\.\d+\.\d+$)(?!.*\.\.)(?!.*\.-)(?!.*-\.)'
        r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$'
    )
    if not pattern.match(bucket_name):
        return ("Invalid bucket name format. Must be lowercase letters, numbers, dots, hyphens; "
                "no consecutive periods; no dash next to period; not an IP address.")
    return None

def is_valid_root(s3_root: str) -> str | None:
    if not isinstance(s3_root, str):
        return "Root must be a string."
    if not s3_root.strip():
        return "Root cannot be empty or whitespace."
    if len(s3_root) > 1024:
        return f"Root length {len(s3_root)} exceeds 1024 characters."
    if s3_root.startswith("s3://") or s3_root.startswith("arn:") or "://" in s3_root:
        return "Root must not be a URL or ARN."
    if s3_root != normalize_root(s3_root):
        return "Root is not normalized (check for leading/trailing/double slashes)."
    pattern = re.compile(r'^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$')
    if not pattern.match(s3_root):
        return ("Invalid root format. Segments must be alphanumeric/dot/underscore/hyphen, "
                "separated by single slashes.")
    return None
        
def ensure_bucket_and_root(clients: dict, region: str, s3_bucket: str, s3_root: str) -> str | None:
    """
    Ensure the S3 bucket exists and that the given root prefix is empty.

    Returns:
        None if the bucket/root combination is valid and safe to use.
        A descriptive error string if the bucket already contains objects or an unexpected error occurs.
    """
    s3 = clients["s3"]

    try:
        # Check if bucket exists
        s3.head_bucket(Bucket=s3_bucket)
        logging.info(f"S3 bucket {s3_bucket} already exists, checking root prefix.")

        # Check if root prefix is empty
        resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=f"{s3_root}/", MaxKeys=1)
        if "Contents" in resp:
            first_key = resp["Contents"][0]["Key"]
            return (f"Bucket {s3_bucket} already has objects under {s3_root}/ "
                    f"(e.g. {first_key}). Root must be empty.")
        logging.info(f"Root prefix {s3_root}/ is empty. Safe to proceed.")
        return None

    except ClientError as e:
        error_code = str(e.response["Error"]["Code"])
        if error_code in ("404", "NoSuchBucket", "NotFound"):
            logging.info(f"Creating S3 bucket {s3_bucket}.")
            if region == "us-east-1":
                s3.create_bucket(Bucket=s3_bucket)
            else:
                s3.create_bucket(
                    Bucket=s3_bucket,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
            return None
        else:
            return f"Unexpected error ensuring bucket {s3_bucket}: {e}"