# -*- coding: utf-8 -*-
"""
Utilitiy functions to assist in validating the infrastructure resource names.
"""

import re
import logging
from botocore.exceptions import ClientError

VALID_REGIONS = [
    "us-east-1",      # N. Virginia
    "us-east-2",      # Ohio
    "us-west-1",      # N. California
    "us-west-2",      # Oregon
    "af-south-1",     # Cape Town
    "ap-east-1",      # Hong Kong
    "ap-south-1",     # Mumbai
    "ap-south-2",     # Hyderabad
    "ap-southeast-1", # Singapore
    "ap-southeast-2", # Sydney
    "ap-southeast-3", # Jakarta
    "ap-southeast-4", # Melbourne
    "ap-northeast-1", # Tokyo
    "ap-northeast-2", # Seoul
    "ap-northeast-3", # Osaka
    "ca-central-1",   # Central Canada
    "ca-west-1",      # Calgary
    "eu-central-1",   # Frankfurt
    "eu-central-2",   # Zurich
    "eu-west-1",      # Ireland
    "eu-west-2",      # London
    "eu-west-3",      # Paris
    "eu-north-1",     # Stockholm
    "eu-south-1",     # Milan
    "eu-south-2",     # Spain
    "il-central-1",   # Tel Aviv
    "me-south-1",     # Bahrain
    "me-central-1",   # UAE
    "sa-east-1"       # São Paulo
]

class ValidationError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("Validation failed")

    def __str__(self):
        return "Validation failed with the following errors:\n" + "\n".join(f"- {e}" for e in self.errors)

def run_all_validations(config):
    errors = []

    if err := validate_infrastructure_name(config["INFRASTRUCTURE_NAME"]):
        errors.append(err)

    if err := validate_bucket_name_and_root(config["S3_BUCKET_NAME"], config["S3_DATASETS_ROOT"]):
        errors.append(err)

    if err := validate_table(config["DDB_IMAGERY_TABLE"]):
        errors.append(err)
    if err := validate_table(config["DDB_DATASET_TABLE"]):
        errors.append(err)
    if err := validate_table(config["DDB_JOB_TABLE"]):
        errors.append(err)

    if err := validate_queue(config["SQS_QUEUE_LIFECYCLE"]):
        errors.append(err)
    if err := validate_queue(config["SQS_QUEUE_IMAGE_OPS"]):
        errors.append(err)
    if err := validate_queue(config["SQS_QUEUE_SYNC"]):
        errors.append(err)
    if err := validate_queue(config["SQS_QUEUE_DLQ"]):
        errors.append(err)

    if err := validate_lambda(config["LAMBDA_LIFECYCLE"]):
        errors.append(err)
    if err := validate_lambda(config["LAMBDA_IMAGE_OPS"]):
        errors.append(err)
    if err := validate_lambda(config["LAMBDA_SYNC"]):
        errors.append(err)
    if err := validate_lambda(config["LAMBDA_DLQ"]):
        errors.append(err)

    if err := validate_role_name(config["LIFECYCLE_ROLE_NAME"]):
        errors.append(err)
    if err := validate_role_name(config["IMAGE_OPS_ROLE_NAME"]):
        errors.append(err)
    if err := validate_role_name(config["SYNC_ROLE_NAME"]):
        errors.append(err)
    if err := validate_role_name(config["DLQ_ROLE_NAME"]):
        errors.append(err)

    if err := validate_img_name(config["IMAGE_NAME_LIFECYCLE"]):
        errors.append(err)
    if err := validate_img_name(config["IMAGE_NAME_IMAGE_OPS"]):
        errors.append(err)
    if err := validate_img_name(config["IMAGE_NAME_SYNC"]):
        errors.append(err)
    if err := validate_img_name(config["IMAGE_NAME_DLQ"]):
        errors.append(err)

    if err := validate_log_group(config["LOG_GROUP_NAME"]):
        errors.append(err)
            
    if errors:
        raise ValidationError(errors)

def validate_log_group(name: str):
    # Allowed: letters, numbers, underscore, hyphen, slash, period
    LOG_GROUP_REGEX = re.compile(r'^[A-Za-z0-9_\-./]{1,512}$')
    
    if not LOG_GROUP_REGEX.match(name):
        return f"Invalid CloudWatch Log Group name: {name}"
    
    return None

def validate_infrastructure_name(name: str):
    """
    Validate project name for use in SSM Parameter Store path segments and tags.
    """
    INFRASTRUCTURE_NAME_REGEX = re.compile(r"^[A-Za-z0-9._-]+$")
    if not INFRASTRUCTURE_NAME_REGEX.match(name):
        return f"Invalid project name: {name}. Must contain only letters, numbers, '.', '-', '_'"
    if len(name) > 256:
        return f"Invalid project name: {name}. Must be <= 256 characters"
    return None

def normalize_root(bucket_root: str) -> str:
    """Strip leading/trailing slashes and collapse multiple consecutive slashes."""
    root = bucket_root.strip().strip("/")
    root = re.sub(r"/+", "/", root)
    return root

def validate_bucket_name_and_root(bucket_name: str, bucket_root: str):
    """
    Validate S3 bucket name and root prefix.
    Returns None if valid, or an error string if invalid.
    """

    BUCKET_REGEX = re.compile(r"^(?!\d+\.)[a-z0-9][a-z0-9\-\.]{1,61}[a-z0-9]$")
    if not BUCKET_REGEX.match(bucket_name):
        return f"Invalid S3 bucket name: {bucket_name}"

    if bucket_root.startswith("/") or bucket_root.startswith("s3://"):
        return f"Invalid S3 root path: {bucket_root}. Must be relative, without leading '/' or 's3://'"

    # Check that the root is already normalized
    normalized = normalize_root(bucket_root)
    if bucket_root != normalized:
        return (
            f"Invalid S3 root path: {bucket_root}. "
            f"Must already be normalized (expected '{normalized}')"
        )

    return None

def validate_table(name: str):
    TABLE_NAME_REGEX = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{2,254}$")
    if not TABLE_NAME_REGEX.match(name):
        return f"Invalid DynamoDB table name: {name}"
    return None

def validate_queue(name: str):
    QUEUE_NAME_REGEX = re.compile(r"^[A-Za-z0-9_-]{1,80}(\.fifo)?$")
    if not QUEUE_NAME_REGEX.match(name):
        return f"Invalid SQS queue name: {name}"
    return None

def validate_lambda(name: str):
    FUNC_NAME_REGEX = re.compile(r"^[a-zA-Z0-9-_]{1,64}$")
    if not FUNC_NAME_REGEX.match(name):
        return f"Invalid Lambda function name: {name}"
    return None

def validate_role_name(name: str):
    ROLE_NAME_REGEX = re.compile(r"^[\w+=,.@-]{1,64}$")
    if not ROLE_NAME_REGEX.match(name):
        return f"Invalid IAM role name: {name}"
    return None

def validate_img_name(name: str):
    ECR_REPO_REGEX = re.compile(r"^[a-z0-9]+(?:[._/-][a-z0-9]+)*$")
    if not ECR_REPO_REGEX.match(name):
        return f"Invalid ECR image name: {name}"
    return None

# Existence checks
# --- Low-level existence checks (return True/False) ---

def bucket_exists(s3, bucket_name):
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            return False
        elif code in ("403", "AccessDenied"):
            logging.error(f"Access denied to bucket {bucket_name}")
            raise Exception(f"Access denied to bucket {bucket_name}")
        raise

def prefix_empty(s3, bucket_name, prefix):
    resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
    return "Contents" not in resp

def table_exists(ddb, table_name):
    try:
        ddb.describe_table(TableName=table_name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            return False
        raise

def queue_exists(sqs, queue_name):
    try:
        sqs.get_queue_url(QueueName=queue_name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("AWS.SimpleQueueService.NonExistentQueue", "QueueDoesNotExist"):
            return False
        raise

def image_exists(ecr_client, repo_name):
    try:
        ecr_client.describe_repositories(repositoryNames=[repo_name])
        return True
    except ecr_client.exceptions.RepositoryNotFoundException:
        return False

def lambda_exists(lambda_client, function_name):
    try:
        lambda_client.get_function(FunctionName=function_name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            return False
        raise

def role_exists(iam, role_name):
    try:
        iam.get_role(RoleName=role_name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            return False
        raise

def log_group_exists(logs_client, log_group_name):
    try:
        resp = logs_client.describe_log_groups(
            logGroupNamePrefix=log_group_name,
            limit=1
        )
        for lg in resp.get("logGroups", []):
            if lg["logGroupName"] == log_group_name:
                return True
        return False
    except ClientError as e:
        logging.error(f"Error describing log groups: {e}")
        raise
        
# --- High-level check_* wrappers (return None or error string) ---

def check_log_group(logs_client, log_group_name):
    try:
        if log_group_exists(logs_client, log_group_name):
            return f"CloudWatch Log Group {log_group_name} already exists"
    except Exception as e:
        return f"Error checking CloudWatch Log Group {log_group_name}: {e}"
    return None

def check_bucket_and_root(s3, bucket_name, bucket_root):
    try:
        if bucket_exists(s3, bucket_name):
            if not prefix_empty(s3, bucket_name, f"{bucket_root}/"):
                return f"S3 bucket {bucket_name} already has contents under {bucket_root}/"
    except Exception as e:
        return f"Error checking S3 bucket {bucket_name}: {e}"
    return None

def check_table(ddb, table_name):
    try:
        if table_exists(ddb, table_name):
            return f"DynamoDB table {table_name} already exists"
    except Exception as e:
        return f"Error checking DynamoDB table {table_name}: {e}"
    return None

def check_queue(sqs, queue_name):
    try:
        if queue_exists(sqs, queue_name):
            return f"SQS queue {queue_name} already exists"
    except Exception as e:
        return f"Error checking SQS queue {queue_name}: {e}"
    return None

def check_lambda(lambda_client, fn):
    try:
        if lambda_exists(lambda_client, fn):
            return f"Lambda {fn} already exists"
    except Exception as e:
        return f"Error checking Lambda {fn}: {e}"
    return None

def check_role(iam, role_name):
    try:
        if role_exists(iam, role_name):
            return f"IAM role {role_name} already exists"
    except Exception as e:
        return f"Error checking IAM role {role_name}: {e}"
    return None

def check_image(ecr_client, repo_name):
    try:
        if image_exists(ecr_client, repo_name):
            return f"ECR repo {repo_name} already exists"
    except Exception as e:
        return f"Error checking ECR repo {repo_name}: {e}"
    return None

# --- Aggregator using walrus operator ---

def run_existence_checks(config, clients):
    errors = []

    # Bucket special case
    if err := check_bucket_and_root(clients["s3"], config["S3_BUCKET_NAME"], config["S3_DATASETS_ROOT"]):
        errors.append(err)

    # DynamoDB
    for key in ["DDB_IMAGERY_TABLE", "DDB_DATASET_TABLE", "DDB_JOB_TABLE"]:
        if err := check_table(clients["ddb"], config[key]):
            errors.append(err)

    # SQS
    for key in ["SQS_QUEUE_LIFECYCLE", "SQS_QUEUE_IMAGE_OPS", "SQS_QUEUE_SYNC", "SQS_QUEUE_DLQ"]:
        if err := check_queue(clients["sqs"], config[key]):
            errors.append(err)

    # Lambdas
    for key in ["LAMBDA_LIFECYCLE", "LAMBDA_IMAGE_OPS", "LAMBDA_SYNC", "LAMBDA_DLQ"]:
        if err := check_lambda(clients["lambda"], config[key]):
            errors.append(err)

    # IAM roles
    for key in ["LIFECYCLE_ROLE_NAME", "IMAGE_OPS_ROLE_NAME", "SYNC_ROLE_NAME", "DLQ_ROLE_NAME"]:
        if err := check_role(clients["iam"], config[key]):
            errors.append(err)

    # ECR repos
    for key in ["IMAGE_NAME_LIFECYCLE", "IMAGE_NAME_IMAGE_OPS", "IMAGE_NAME_SYNC", "IMAGE_NAME_DLQ"]:
        if err := check_image(clients["ecr"], config[key]):
            errors.append(err)

    if err := check_log_group(clients["logs"], config['LOG_GROUP_NAME']):
        errors.append(err)
            
    if errors:
        raise ValidationError(errors)