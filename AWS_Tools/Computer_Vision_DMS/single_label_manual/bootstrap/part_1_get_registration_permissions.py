# -*- coding: utf-8 -*-
"""
Create permissions to run part 2: validating and registering the infrastructure.
"""

import os
import sys
import json
import argparse
import logging

from helpers.validation_helpers import VALID_REGIONS

def generate_validation_policy(account_id: str, region: str) -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "SSMParameterStoreAccess",
                "Effect": "Allow",
                "Action": [
                    "ssm:GetParametersByPath",
                    "ssm:PutParameter"
                ],
                "Resource": f"arn:aws:ssm:{region}:{account_id}:parameter/cv-datasets/single-label/*"
            },
            {
                "Sid": "S3ReadAccess",
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket",
                    "s3:GetBucketLocation"
                ],
                "Resource": "arn:aws:s3:::*"
            },
            {
                "Sid": "DynamoDBDescribe",
                "Effect": "Allow",
                "Action": "dynamodb:DescribeTable",
                "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/*"
            },
            {
                "Sid": "SQSDescribe",
                "Effect": "Allow",
                "Action": [
                    "sqs:GetQueueUrl",
                    "sqs:GetQueueAttributes"
                ],
                "Resource": f"arn:aws:sqs:{region}:{account_id}:*"
            },
            {
                "Sid": "LambdaDescribe",
                "Effect": "Allow",
                "Action": "lambda:GetFunction",
                "Resource": f"arn:aws:lambda:{region}:{account_id}:function:*"
            },
            {
                "Sid": "IAMReadRole",
                "Effect": "Allow",
                "Action": "iam:GetRole",
                "Resource": f"arn:aws:iam::{account_id}:role/*"
            },
            {
                "Sid": "ECRDescribe",
                "Effect": "Allow",
                "Action": "ecr:DescribeRepositories",
                "Resource": f"arn:aws:ecr:{region}:{account_id}:repository/*"
            },
            {
                "Sid": "STSCallerIdentity",
                "Effect": "Allow",
                "Action": "sts:GetCallerIdentity",
                "Resource": "*"
            },
            {
                "Sid": "CheckLogGroupExistence",
                "Effect": "Allow",
                "Action": "logs:DescribeLogGroups",
                "Resource": "*"
            }
        ]
    }

if __name__ == "__main__":
    main_dir = os.path.dirname(__file__)
    base_logs = os.path.join(main_dir, "logs")
    os.makedirs(base_logs, exist_ok=True)

    logging_save_to = os.path.join(base_logs, "part_1_logs.txt")

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True, help="AWS Account ID (12 digits)")
    parser.add_argument("--region", required=True, help="AWS Region for the infrastructure")
    args = parser.parse_args()

    account_id = args.account
    region = args.region.lower()

    # Validate account ID
    if not (account_id.isdigit() and len(account_id) == 12):
        logger.error(f"Invalid account id: {account_id}")
        sys.exit(1)

    # Validate region
    if region not in VALID_REGIONS:
        logger.error(f"Invalid region: {region}. Must be one of {', '.join(VALID_REGIONS)}")
        sys.exit(1)

    os.makedirs("generated_permissions", exist_ok=True)
    output_path = os.path.join("generated_permissions", "config_validation_policy.json")

    with open(output_path, "w") as f:
        json.dump(generate_validation_policy(account_id, region), f, indent=2)

    logger.info(f"Policy for running part 2 written to {output_path}")