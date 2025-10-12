# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 22:20:03 2025

@author: brian
"""

import os
import re
import sys
import json
import logging
from dotenv import load_dotenv
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


def load_config(config_path):
    loaded_env_config = load_dotenv(config_path)

    if not loaded_env_config:
        logging.error(f'Failed to load environment config at {config_path}')
        sys.exit(1)
    else:
        logging.info('Loaded in config.')
    
    config = {}
    
    #-----------------------
    # Get the AWS region.
    #-----------------------
    region = os.getenv("AWS_REGION")
    if not region:
        logging.error(f'AWS_REGION is missing from the config, must be in {VALID_REGIONS}')
        sys.exit(1)
    
    region = region.lower()
    if region not in VALID_REGIONS:
        logging.error(f'Invalid region: {region}, must be in {VALID_REGIONS}')
        sys.exit(1)

    config['AWS_REGION'] = region
    logging.info(f"Using AWS region: {region}")
    
    #-----------------------
    # Get the S3 parameters.
    #-----------------------
    bucket = os.getenv("S3_BUCKET_NAME")
    root = os.getenv("S3_DATASETS_ROOT")
    
    # Validate bucket name
    BUCKET_REGEX = re.compile(r"^(?!\d+\.)[a-z0-9][a-z0-9\-\.]{1,61}[a-z0-9]$")
    if not bucket:
        logging.error("S3_BUCKET_NAME is missing from the config")
        sys.exit(1)
    if not BUCKET_REGEX.match(bucket):
        logging.error(f"Invalid S3 bucket name: {bucket}")
        sys.exit(1)
    
    # Validate root path
    if not root:
        logging.error("S3_DATASETS_ROOT is missing from the config")
        sys.exit(1)
    if root.startswith("/") or root.startswith("s3://"):
        logging.error(f"Invalid S3 root path: {root}. Must be a relative path without leading '/' or 's3://'")
        sys.exit(1)
    
    # Normalize root (strip trailing slashes)
    root = root.strip().strip("/")
    
    config["S3_BUCKET_NAME"] = bucket
    config["S3_DATASETS_ROOT"] = root

    logging.info(f"Using S3 bucket: {bucket} and datasets root: {root} for a full URI of s3://{bucket}/{root}")
    
    #------------------------------
    # Get the DynamoDB parameters.
    #------------------------------
    DDB_IMAGERY_TABLE = os.getenv("DDB_IMAGERY_TABLE")
    DDB_DATASET_TABLE = os.getenv("DDB_DATASET_TABLE")
    DDB_JOB_TABLE = os.getenv("DDB_JOB_TABLE")
    
    # DynamoDB table name rules
    TABLE_NAME_REGEX = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{2,254}$")
    
    def validate_table(name, var_name):
        if not name:
            logging.error(f"{var_name} is missing from the config")
            sys.exit(1)
        if not TABLE_NAME_REGEX.match(name):
            logging.error(f"Invalid DynamoDB table name for {var_name}: {name}")
            sys.exit(1)
        logging.info(f"Using {var_name}: {name}")
        return name
    
    config["DDB_IMAGERY_TABLE"] = validate_table(DDB_IMAGERY_TABLE, "DDB_IMAGERY_TABLE")
    config["DDB_DATASET_TABLE"] = validate_table(DDB_DATASET_TABLE, "DDB_DATASET_TABLE")
    config["DDB_JOB_TABLE"]     = validate_table(DDB_JOB_TABLE, "DDB_JOB_TABLE")
    
    # ------------------------------
    # Get the SQS parameters.
    # ------------------------------
    SQS_QUEUE_LIFECYCLE = os.getenv("SQS_QUEUE_LIFECYCLE")
    SQS_QUEUE_IMAGE_OPS = os.getenv("SQS_QUEUE_IMAGE_OPS")
    SQS_QUEUE_SYNC      = os.getenv("SQS_QUEUE_SYNC")
    SQS_QUEUE_DLQ             = os.getenv("SQS_QUEUE_DLQ")
    
    # SQS queue name rules: 1–80 chars, alphanumeric, hyphen, underscore, period
    QUEUE_NAME_REGEX = re.compile(r"^[A-Za-z0-9_-]{1,80}(\.fifo)?$")
    
    def validate_queue(value, var_name):
        if not value:
            logging.error(f"{var_name} is missing from the config")
            sys.exit(1)
    
        # Accept only a queue name
        if not QUEUE_NAME_REGEX.match(value):
            logging.error(f"{var_name} is not a valid SQS queue name: {value}")
            sys.exit(1)
    
        logging.info(f"Using {var_name}: {value}")
        return value
    
    config["SQS_QUEUE_LIFECYCLE"] = validate_queue(SQS_QUEUE_LIFECYCLE, "SQS_QUEUE_LIFECYCLE")
    config["SQS_QUEUE_IMAGE_OPS"] = validate_queue(SQS_QUEUE_IMAGE_OPS, "SQS_QUEUE_IMAGE_OPS")
    config["SQS_QUEUE_SYNC"]      = validate_queue(SQS_QUEUE_SYNC, "SQS_QUEUE_SYNC")
    config["SQS_QUEUE_DLQ"]       = validate_queue(SQS_QUEUE_DLQ, "SQS_QUEUE_DLQ")

    # ------------------------------
    # Get the Lambda parameters.
    # ------------------------------
    LAMBDA_LIFECYCLE = os.getenv("LAMBDA_LIFECYCLE")
    LAMBDA_IMAGE_OPS = os.getenv("LAMBDA_IMAGE_OPS")
    LAMBDA_SYNC      = os.getenv("LAMBDA_SYNC")
    LAMBDA_DLQ      = os.getenv("LAMBDA_DLQ")

    # Lambda function name rules: 1–64 chars, letters, numbers, hyphens, underscores
    FUNC_NAME_REGEX = re.compile(r"^[a-zA-Z0-9-_]{1,64}$")
    
    def validate_lambda(value, var_name):
        if not value:
            logging.error(f"{var_name} is missing from the config")
            sys.exit(1)
    
        if not FUNC_NAME_REGEX.match(value):
            logging.error(f"{var_name} is not a valid Lambda function name or ARN: {value}")
            sys.exit(1)
    
        logging.info(f"Using {var_name}: {value}")
        return value
    
    config["LAMBDA_LIFECYCLE"] = validate_lambda(LAMBDA_LIFECYCLE, "LAMBDA_LIFECYCLE")
    config["LAMBDA_IMAGE_OPS"] = validate_lambda(LAMBDA_IMAGE_OPS, "LAMBDA_IMAGE_OPS")
    config["LAMBDA_SYNC"]      = validate_lambda(LAMBDA_SYNC, "LAMBDA_SYNC")
    config["LAMBDA_DLQ"]      = validate_lambda(LAMBDA_DLQ, "LAMBDA_DLQ")

    # ------------------------------
    # Lambda execution role names.
    # ------------------------------
    LIFECYCLE_ROLE_NAME = os.getenv("LIFECYCLE_ROLE_NAME")
    IMAGE_OPS_ROLE_NAME = os.getenv("IMAGE_OPS_ROLE_NAME")
    SYNC_ROLE_NAME      = os.getenv("SYNC_ROLE_NAME")
    DLQ_ROLE_NAME      = os.getenv("DLQ_ROLE_NAME")

    # IAM role name regex (per AWS docs)
    ROLE_NAME_REGEX = re.compile(r"^[\w+=,.@-]{1,64}$")
    
    def validate_role_name(value, var_name):
        if not value:
            logging.error(f"{var_name} is missing")
            sys.exit(1)
        if not ROLE_NAME_REGEX.match(value):
            logging.error(f"Invalid IAM role name for {var_name}: {value}")
            sys.exit(1)
        logging.info(f"Using {var_name}: {value}")
        return value
    
    config["LIFECYCLE_ROLE_NAME"] = validate_role_name(LIFECYCLE_ROLE_NAME, "LIFECYCLE_ROLE_NAME")
    config["IMAGE_OPS_ROLE_NAME"] = validate_role_name(IMAGE_OPS_ROLE_NAME, "IMAGE_OPS_ROLE_NAME")
    config["SYNC_ROLE_NAME"]      = validate_role_name(SYNC_ROLE_NAME, "SYNC_ROLE_NAME")
    config["DLQ_ROLE_NAME"]      = validate_role_name(DLQ_ROLE_NAME, "DLQ_ROLE_NAME")

    # ------------------------------
    # Lambda image names.
    # ------------------------------
    IMAGE_NAME_LIFECYCLE = os.getenv("IMAGE_NAME_LIFECYCLE")
    IMAGE_NAME_IMAGE_OPS = os.getenv("IMAGE_NAME_IMAGE_OPS")
    IMAGE_NAME_SYNC      = os.getenv("IMAGE_NAME_SYNC")
    IMAGE_NAME_DLQ      = os.getenv("IMAGE_NAME_DLQ")

    IMG_NAME_REGEX = re.compile(r"^[\w+=,.@-]{1,64}$")
    
    def validate_img_name(value, var_name):
        if not value:
            logging.error(f"{var_name} is missing")
            sys.exit(1)
        if not IMG_NAME_REGEX.match(value):
            logging.error(f"Invalid ECR Image name for {var_name}: {value}")
            sys.exit(1)
        logging.info(f"Using {var_name}: {value}")
        return value
    
    config["IMAGE_NAME_LIFECYCLE"] = validate_img_name(IMAGE_NAME_LIFECYCLE, "IMAGE_NAME_LIFECYCLE")
    config["IMAGE_NAME_IMAGE_OPS"] = validate_img_name(IMAGE_NAME_IMAGE_OPS, "IMAGE_NAME_IMAGE_OPS")
    config["IMAGE_NAME_SYNC"]      = validate_img_name(IMAGE_NAME_SYNC, "IMAGE_NAME_SYNC")
    config["IMAGE_NAME_DLQ"]      = validate_img_name(IMAGE_NAME_DLQ, "IMAGE_NAME_DLQ")

    return config

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
            sys.exit(1)
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
  
def queue_exists(sqs, q):
    try:
        if q.startswith("https://"):
            sqs.get_queue_attributes(QueueUrl=q, AttributeNames=["QueueArn"])
        else:
            sqs.get_queue_url(QueueName=q)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("AWS.SimpleQueueService.NonExistentQueue", "QueueDoesNotExist"):
            return False
        raise
        
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

def image_exists(ecr_client, repo_name):
    try:
        ecr_client.describe_repositories(repositoryNames=[repo_name])
        return True
    except ecr_client.exceptions.RepositoryNotFoundException:
        return False
    
def validate_config(config, clients):
    
    identity = clients['sts'].get_caller_identity()
    logging.info(f"Running as {identity['Arn']} in account {identity['Account']}")

    if bucket_exists(clients['s3'], config['S3_BUCKET_NAME']):
        if not prefix_empty(clients['s3'], config['S3_BUCKET_NAME'], f"{config['S3_DATASETS_ROOT']}/"):
            sys.exit(f"Error: Bucket {config['S3_BUCKET_NAME']} already has contents under {config['S3_DATASETS_ROOT']}/")
            
    for table in [config['DDB_IMAGERY_TABLE'], config['DDB_DATASET_TABLE'], config['DDB_JOB_TABLE']]:
        if table_exists(clients['ddb'], table):
            sys.exit(f"Error: DynamoDB table {table} already exists")
            
    for q in [config['SQS_QUEUE_LIFECYCLE'], config['SQS_QUEUE_IMAGE_OPS'], config['SQS_QUEUE_SYNC'], config['SQS_QUEUE_DLQ']]:
        if queue_exists(clients['sqs'], q):
            sys.exit(f"Error: SQS queue {q} already exists")
            
    for fn in [config['LAMBDA_LIFECYCLE'], config['LAMBDA_IMAGE_OPS'], config['LAMBDA_SYNC'], config['LAMBDA_DLQ']]:
        if lambda_exists(clients['lambda'], fn):
            sys.exit(f"Error: Lambda {fn} already exists")
            
    for role in [config['LIFECYCLE_ROLE_NAME'], config['IMAGE_OPS_ROLE_NAME'], config['SYNC_ROLE_NAME'], config['DLQ_ROLE_NAME']]:
        if role_exists(clients['iam'], role):
            sys.exit(f"Error: IAM role {role} already exists")
            
    for img_name in [config['IMAGE_NAME_LIFECYCLE'], config['IMAGE_NAME_IMAGE_OPS'], config['IMAGE_NAME_SYNC'], config['IMAGE_NAME_DLQ']]:
        if image_exists(clients['ecr'], img_name):
            sys.exit(f"Error: ECR Image name {img_name} already exists")
            
    logging.info("Names validated")

def create_roles(config, clients):
    aws_region = config['AWS_REGION']
    account_id = clients['sts'].get_caller_identity()['Account']
    images_arn = f"arn:aws:s3:::{config['S3_BUCKET_NAME']}/{config['S3_DATASETS_ROOT']}/images/*"
    manifests_arn = f"arn:aws:s3:::{config['S3_BUCKET_NAME']}/{config['S3_DATASETS_ROOT']}/manifests/*"
    temp_images_arn = f"arn:aws:s3:::{config['S3_BUCKET_NAME']}/{config['S3_DATASETS_ROOT']}/temp-images/*" 
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    lifecycle_ops_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket"
              ],
                "Resource": f"arn:aws:s3:::{config['S3_BUCKET_NAME']}",
                "Condition": {
                  "StringLike": {
                    "s3:prefix": [
                      f"{config['S3_DATASETS_ROOT']}/images/*",
                      f"{config['S3_DATASETS_ROOT']}/manifests/*"
                    ]
                  }
                }
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:DeleteObject"
                ],
                "Resource": [images_arn, manifests_arn]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:PutItem",
                    "dynamodb:GetItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:BatchWriteItem"
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_DATASET_TABLE']}",
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_DATASET_TABLE']}/index/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:UpdateItem",
                    "dynamodb:GetItem"
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_IMAGERY_TABLE']}",
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_IMAGERY_TABLE']}/index/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:Query",
                    "dynamodb:UpdateItem",
                    "dynamodb:GetItem"
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_JOB_TABLE']}",
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_JOB_TABLE']}/index/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:GetQueueAttributes",
                    "sqs:GetQueueUrl",
                    "sqs:ChangeMessageVisibility"
                ],
                "Resource": f"arn:aws:sqs:{aws_region}:{account_id}:{config['SQS_QUEUE_LIFECYCLE']}"
            }
        ]
    }
    
    img_ops_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket"
              ],
                "Resource": f"arn:aws:s3:::{config['S3_BUCKET_NAME']}",
                "Condition": {
                  "StringLike": {
                    "s3:prefix": [
                      f"{config['S3_DATASETS_ROOT']}/images/*",
                      f"{config['S3_DATASETS_ROOT']}/temp-images/*"
                    ]
                  }
                }
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:DeleteObject",
                    "s3:PutObject"
                ],
                "Resource": images_arn
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:DeleteObject",
                    "s3:GetObject"
                ],
                "Resource": temp_images_arn
            },
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:PutItem",
                    "dynamodb:Query",
                    "dynamodb:BatchWriteItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:GetItem"
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_DATASET_TABLE']}",
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_DATASET_TABLE']}/index/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:DeleteItem",
                    "dynamodb:Query",
                    "dynamodb:UpdateItem",
                    "dynamodb:GetItem",
                    "dynamodb:PutItem"
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_IMAGERY_TABLE']}",
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_IMAGERY_TABLE']}/index/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:Query",
                    "dynamodb:UpdateItem",
                    "dynamodb:GetItem"
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_JOB_TABLE']}",
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_JOB_TABLE']}/index/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:GetQueueAttributes",
                    "sqs:GetQueueUrl",
                    "sqs:ChangeMessageVisibility"
                ],
                "Resource": f"arn:aws:sqs:{aws_region}:{account_id}:{config['SQS_QUEUE_IMAGE_OPS']}"
            }
        ]
    }
    
    sync_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket"
              ],
                "Resource": f"arn:aws:s3:::{config['S3_BUCKET_NAME']}",
                "Condition": {
                  "StringLike": {
                    "s3:prefix": [
                      f"{config['S3_DATASETS_ROOT']}/manifests/*"
                    ]
                  }
                }
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:DeleteObject",
                    "s3:PutObject",
                    "s3:GetObject"
                ],
                "Resource": manifests_arn
            },
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:Query",
                    "dynamodb:UpdateItem",
                    "dynamodb:GetItem"
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_DATASET_TABLE']}",
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_DATASET_TABLE']}/index/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:Query",
                    "dynamodb:UpdateItem",
                    "dynamodb:GetItem"
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_IMAGERY_TABLE']}",
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_IMAGERY_TABLE']}/index/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "dynamodb:UpdateItem",
                    "dynamodb:GetItem"
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_JOB_TABLE']}",
                    f"arn:aws:dynamodb:{aws_region}:{account_id}:table/{config['DDB_JOB_TABLE']}/index/*"
                ]
            },
            {
                "Effect": "Allow",
                "Action": [
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:GetQueueAttributes",
                    "sqs:GetQueueUrl",
                    "sqs:ChangeMessageVisibility"
                ],
                "Resource": f"arn:aws:sqs:{aws_region}:{account_id}:{config['SQS_QUEUE_SYNC']}"
            }
        ]
    }
        
    dlq_policy = {
          "Version": "2012-10-17",
          "Statement": [
            {
              "Sid": "CloudWatchLogs",
              "Effect": "Allow",
              "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
              ],
              "Resource": "*"
            },
            {
              "Effect": "Allow",
              "Action": "s3:ListBucket",
              "Resource": f"arn:aws:s3:::{config['S3_BUCKET_NAME']}",
              "Condition": {
                "StringLike": {
                  "s3:prefix": [
                    f"{config['S3_DATASETS_ROOT']}/dlq-logs/*"
                  ]
                }
              }
            },
            {
              "Sid": "SQSPermissions",
              "Effect": "Allow",
              "Action": [
                "sqs:ReceiveMessage",
                "sqs:DeleteMessage",
                "sqs:GetQueueAttributes",
                "sqs:GetQueueUrl"
              ],
              "Resource": f"arn:aws:sqs:{aws_region}:{account_id}:{config['SQS_QUEUE_DLQ']}"
            },
            {
              "Sid": "S3Permissions",
              "Effect": "Allow",
              "Action": [
                "s3:GetObject",
                "s3:PutObject"
              ],
              "Resource": f"arn:aws:s3:::{config['S3_BUCKET_NAME']}/{config['S3_DATASETS_ROOT']}/dlq-logs/*"
            }
          ]
        }

    # ---------------------------------------
    # Create the Lifecycle Ops policy.
    # ---------------------------------------
    resp = clients['iam'].create_role(
        RoleName=config['LIFECYCLE_ROLE_NAME'],
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description=f"Role for {config['LIFECYCLE_ROLE_NAME']}",
    )
    
    clients['iam'].put_role_policy(
        RoleName=config['LIFECYCLE_ROLE_NAME'],
        PolicyName=f"{config['LIFECYCLE_ROLE_NAME']}-inline-execution",
        PolicyDocument=json.dumps(lifecycle_ops_policy)
    )
    
    logging.info(f"Created role {config['LIFECYCLE_ROLE_NAME']} and attached inline execution policy.")
    
    config['LIFECYCLE_ROLE_ARN'] = resp['Role']['Arn']
    
    # ---------------------------------------
    # Create the Image Ops policy.
    # ---------------------------------------
    resp = clients['iam'].create_role(
        RoleName=config['IMAGE_OPS_ROLE_NAME'],
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description=f"Role for {config['IMAGE_OPS_ROLE_NAME']}",
    )
    
    clients['iam'].put_role_policy(
        RoleName=config['IMAGE_OPS_ROLE_NAME'],
        PolicyName=f"{config['IMAGE_OPS_ROLE_NAME']}-inline-execution",
        PolicyDocument=json.dumps(img_ops_policy)
    )
    
    logging.info(f"Created role {config['IMAGE_OPS_ROLE_NAME']} and attached inline execution policy.")
    
    config['IMAGE_OPS_ROLE_ARN'] = resp['Role']['Arn']

    # ---------------------------------------
    # Create the Sync policy.
    # ---------------------------------------
    resp = clients['iam'].create_role(
        RoleName=config['SYNC_ROLE_NAME'],
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description=f"Role for {config['SYNC_ROLE_NAME']}",
    )
     
    clients['iam'].put_role_policy(
        RoleName=config['SYNC_ROLE_NAME'],
        PolicyName=f"{config['SYNC_ROLE_NAME']}-inline-execution",
        PolicyDocument=json.dumps(sync_policy)
    )

    logging.info(f"Created role {config['SYNC_ROLE_NAME']} and attached inline execution policy.")
    
    config['SYNC_ROLE_ARN'] = resp['Role']['Arn']
    
    # ---------------------------------------
    # Create the DLQ policy.
    # ---------------------------------------
    resp = clients['iam'].create_role(
        RoleName=config['DLQ_ROLE_NAME'],
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Description=f"Role for {config['DLQ_ROLE_NAME']}",
    )
     
    clients['iam'].put_role_policy(
        RoleName=config['DLQ_ROLE_NAME'],
        PolicyName=f"{config['DLQ_ROLE_NAME']}-inline-execution",
        PolicyDocument=json.dumps(dlq_policy)
    )

    logging.info(f"Created role {config['DLQ_ROLE_NAME']} and attached inline execution policy.")
    
    config['DLQ_ROLE_ARN'] = resp['Role']['Arn']

def make_bucket(config, clients):
    s3 = clients['s3']
    sts = clients['sts']
    account_id = sts.get_caller_identity()['Account']

    bucket_name = config['S3_BUCKET_NAME']

    try:
        s3.head_bucket(Bucket=bucket_name)
        logging.info(f"S3 bucket {bucket_name} already exists.")
    except s3.exceptions.ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            logging.info(f"Creating S3 bucket {bucket_name}.")
            
            if config['AWS_REGION'] == "us-east-1":
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': config['AWS_REGION']}
                )

        else:
            raise

    # Bucket policy
    bucket_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DenyImagesAndManifestsExceptLambdas",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:*",
                "Resource": [
                    f"arn:aws:s3:::{config['S3_BUCKET_NAME']}/{config['S3_DATASETS_ROOT']}/images/*",
                    f"arn:aws:s3:::{config['S3_BUCKET_NAME']}/{config['S3_DATASETS_ROOT']}/manifests/*"
                ],
                "Condition": {
                    "StringNotLike": {
                        "aws:PrincipalArn": [
                            f"arn:aws:iam::{account_id}:role/{config['LIFECYCLE_ROLE_NAME']}",
                            f"arn:aws:iam::{account_id}:role/{config['IMAGE_OPS_ROLE_NAME']}",
                            f"arn:aws:iam::{account_id}:role/{config['SYNC_ROLE_NAME']}"
                        ]
                    }
                }
            }
        ]
    }


    # Attach bucket policy
    s3.put_bucket_policy(
        Bucket=bucket_name,
        Policy=json.dumps(bucket_policy)
    )
    logging.info(f"Attached bucket policy to {bucket_name}.")

def make_queues(config, clients):
    sqs = clients['sqs']

    # First make the DLQ Queue
    try:
        # Try to get the queue URL
        response = sqs.get_queue_url(QueueName=config['SQS_QUEUE_DLQ'])
        queue_url = response['QueueUrl']
        logging.info(f"SQS queue {config['SQS_QUEUE_DLQ']} already exists at {queue_url}.")
    except sqs.exceptions.QueueDoesNotExist:
        # Create the queue if it doesn't exist
        logging.info(f"Creating SQS queue {config['SQS_QUEUE_DLQ']}.")
        
        attributes = {}
        attributes["MessageRetentionPeriod"] = "1209600"  # 14 days

        response = sqs.create_queue(
            QueueName=config['SQS_QUEUE_DLQ'],
            Attributes=attributes
        )
        queue_url = response['QueueUrl']
        logging.info(f"Created SQS queue {config['SQS_QUEUE_DLQ']} at {queue_url}.")

    # Store the ARN in config for later use
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=['QueueArn']
    )
    config["SQS_QUEUE_DLQ_ARN"] = attrs['Attributes']['QueueArn']
    
    # Make the remaining queues
    queue_key_names = [
        "SQS_QUEUE_LIFECYCLE",
        "SQS_QUEUE_IMAGE_OPS",
        "SQS_QUEUE_SYNC"]
    
    for queue_key in queue_key_names:
        qname = config[queue_key]
        try:
            # Try to get the queue URL
            response = sqs.get_queue_url(QueueName=qname)
            queue_url = response['QueueUrl']
            logging.info(f"SQS queue {qname} already exists at {queue_url}.")
        except sqs.exceptions.QueueDoesNotExist:
            # Create the queue if it doesn't exist
            logging.info(f"Creating SQS queue {qname}.")

            response = sqs.create_queue(
                QueueName=qname,
                Attributes={
                            "RedrivePolicy": json.dumps({
                                "deadLetterTargetArn": config['SQS_QUEUE_DLQ_ARN'],
                                "maxReceiveCount": "5"
                            })
                        }
            )
            queue_url = response['QueueUrl']
            logging.info(f"Created SQS queue {qname} at {queue_url}.")

        # Store the ARN in config for later use
        attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['QueueArn']
        )
        config[f"{queue_key}_ARN"] = attrs['Attributes']['QueueArn']

def make_tables(config, clients):
    ddb = clients['ddb']

    tables = [
        {
            "name": config['DDB_DATASET_TABLE'],
            "params": {
                "AttributeDefinitions": [
                    {"AttributeName": "dataset_id", "AttributeType": "S"}
                ],
                "KeySchema": [
                    {"AttributeName": "dataset_id", "KeyType": "HASH"}
                ],
                "BillingMode": "PAY_PER_REQUEST"
            }
        },
        {
            "name": config['DDB_IMAGERY_TABLE'],
            "params": {
                "AttributeDefinitions": [
                    {"AttributeName": "dataset_phash", "AttributeType": "S"},
                    {"AttributeName": "dataset_id", "AttributeType": "S"},
                    {"AttributeName": "phash", "AttributeType": "S"}
                ],
                "KeySchema": [
                    {"AttributeName": "dataset_phash", "KeyType": "HASH"}
                ],
                "BillingMode": "PAY_PER_REQUEST",
                "GlobalSecondaryIndexes": [
                    {
                        "IndexName": "DatasetIndex",
                        "KeySchema": [
                            {"AttributeName": "dataset_id", "KeyType": "HASH"},
                            {"AttributeName": "phash", "KeyType": "RANGE"}
                        ],
                        "Projection": {"ProjectionType": "ALL"}
                    },
                    {
                        "IndexName": "PhashIndex",
                        "KeySchema": [
                            {"AttributeName": "phash", "KeyType": "HASH"}
                        ],
                        "Projection": {"ProjectionType": "ALL"}
                    }
                ]
            }
        },

        {
            "name": config['DDB_JOB_TABLE'],
            "params": {
                "AttributeDefinitions": [
                    {"AttributeName": "jobId", "AttributeType": "S"}
                ],
                "KeySchema": [
                    {"AttributeName": "jobId", "KeyType": "HASH"}
                ],
                "BillingMode": "PAY_PER_REQUEST"
            }
        }
    ]

    for table in tables:
        try:
            ddb.create_table(TableName=table["name"], **table["params"])
            logging.info(f"Creating DynamoDB table {table['name']}")
            waiter = ddb.get_waiter("table_exists")
            waiter.wait(TableName=table["name"])
            logging.info(f"DynamoDB table {table['name']} is active.")
        except ddb.exceptions.ResourceInUseException:
            logging.info(f"DynamoDB table {table['name']} already exists, skipping.")

    logging.info("DynamoDB tables created.")   