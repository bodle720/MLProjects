# -*- coding: utf-8 -*-
"""
This script will build the CDK stack with the given name.
Use admin privileges to build and specify the following inputs:

Inputs:
    - profile_name: The name of the profile when doing `aws sso login --profile <profile name>`
    - region:       The AWS region the stack will be built.
    - stack_name:   String name of the stack built with CDK. e.g. my-first-dms
    - s3_bucket:    The bucket to hold the imagery and related files. e.g. my-unique-bucket-name
    - s3_root:      The root directory inside s3_bucket that will store all imagery and related files.
                    e.g. path/to/the/project/files
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timezone

import boto3

from bootstrap_helpers import is_profile_name_valid, is_valid_region, \
                              is_valid_cdk_stack_name, is_valid_bucket_name, \
                              is_valid_root, ensure_bucket_and_root

MAIN_DIR = os.path.dirname(__file__)
base_logs = os.path.join(MAIN_DIR, "logs")
os.makedirs(base_logs, exist_ok=True)

logging_save_to = os.path.join(base_logs, "bootstrap_builder_logs.txt")

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(logging_save_to)
console_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
        
def main():   
    # Get current UTC time and log beginning.
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f'----------Beginning Bootstrapping Process. Time = {now_utc} UTC----------')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_name", required=True, help="The profile name used when running the script; it should have admin privileges. Found in /.aws/config")
    parser.add_argument("--region", required=True, help="The AWS region where the stack will reside.")
    parser.add_argument("--stack_name", required=True, help="The string name of your stack.")
    parser.add_argument("--s3_bucket", required=True, help="The bucket name in S3 to hold all files related to stack.")
    parser.add_argument("--s3_root", required=True, help="The directory path in your bucket of choice; it must be empty.")

    args = parser.parse_args()

    profile_name = args.profile_name
    region = args.region.lower()
    stack_name = args.stack_name
    s3_bucket = args.s3_bucket
    s3_root = args.s3_root

    errors = []
    
    if err := is_profile_name_valid(profile_name):
        errors.append(err)
        
    if err := is_valid_region(region):
        errors.append(err)
        
    if err := is_valid_cdk_stack_name(stack_name):
        errors.append(err)
        
    if err := is_valid_bucket_name(s3_bucket):
        errors.append(err)
        
    if err := is_valid_root(s3_root):
        errors.append(err)

    clients = None
    try:
        session = boto3.Session(profile_name=profile_name, region_name=region)
        clients = {
            "sts": session.client("sts"),
            "s3": session.client("s3"),
            "sqs": session.client("sqs"),
            "ddb": session.client("dynamodb"),
            "lambda": session.client("lambda"),
            "iam": session.client("iam"),
            "ecr": session.client("ecr"),
            "logs": session.client("logs"),
            "ssm": session.client("ssm"),
        }
    except Exception as e:
        errors.append(f"Failed to create boto3 session/clients: {e}")
    
    if clients and not errors:
        if err := ensure_bucket_and_root(clients, region, s3_bucket, s3_root):
            errors.append(err)
        
    if errors:
        logger.error("Bootstrap validation failed with the following issues:")
        for e in errors:
            logger.error(f" - {e}")
        logger.error(f"----------End of bootstrap attempt with {len(errors)} errors----------")
        sys.exit(1)
        
    logger.info("All bootstrap validations passed successfully. Proceeding to CDK deploy…")
            
if __name__ == "__main__":
    main()
