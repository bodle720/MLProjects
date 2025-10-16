# -*- coding: utf-8 -*-
"""
Validate the infrastructure and register with AWS SSM Parameter Store.
"""

import os
import sys
import logging
import boto3
import argparse
import uuid
import json

from helpers.validation_helpers import VALID_REGIONS, ValidationError, run_all_validations,\
                                       run_existence_checks, normalize_root

def random_short_hash(length: int = 6) -> str:
    return uuid.uuid4().hex[:length]

def generate_bootstrap_policy(account_id, config):
    # return dict with the bootstrap policy JSON
        
    bootstrap_policy = {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid": "STSIdentity",
          "Effect": "Allow",
          "Action": "sts:GetCallerIdentity",
          "Resource": "*"
        },
        {
          "Sid": "IAMRoleManagement",
          "Effect": "Allow",
          "Action": [
            "iam:CreateRole",
            "iam:PutRolePolicy",
            "iam:GetRole",
            "iam:PassRole"
          ],
          "Resource": [
            f"arn:aws:iam::{account_id}:role/{config['LIFECYCLE_ROLE_NAME']}",
            f"arn:aws:iam::{account_id}:role/{config['IMAGE_OPS_ROLE_NAME']}",
            f"arn:aws:iam::{account_id}:role/{config['SYNC_ROLE_NAME']}",
            f"arn:aws:iam::{account_id}:role/{config['DLQ_ROLE_NAME']}"
          ],
          "Condition": {
            "StringEquals": {
              "iam:PassedToService": "lambda.amazonaws.com"
            }
          }
        },
        {
          "Sid": "S3BucketManagement",
          "Effect": "Allow",
          "Action": [
            "s3:CreateBucket",
            "s3:GetBucketLocation",
            "s3:HeadBucket",
            "s3:PutBucketPolicy",
            "s3:PutBucketAcl",
            "s3:ListBucket"
          ],
          "Resource": [
            f"arn:aws:s3:::{config['S3_BUCKET_NAME']}",
            f"arn:aws:s3:::{config['S3_BUCKET_NAME']}/*"
          ]
        },
        {
          "Sid": "SQSManagement",
          "Effect": "Allow",
          "Action": [
            "sqs:CreateQueue",
            "sqs:GetQueueUrl",
            "sqs:GetQueueAttributes",
            "sqs:SetQueueAttributes"
          ],
          "Resource": [
            f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_LIFECYCLE']}",
            f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_IMAGE_OPS']}",
            f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_SYNC']}",
            f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_DLQ']}"
          ]
        },
        {
          "Sid": "DynamoDBManagement",
          "Effect": "Allow",
          "Action": [
            "dynamodb:CreateTable",
            "dynamodb:DescribeTable"
          ],
          "Resource": [
            f"arn:aws:dynamodb:{config['AWS_REGION']}:{account_id}:table/{config['DDB_IMAGERY_TABLE']}",
            f"arn:aws:dynamodb:{config['AWS_REGION']}:{account_id}:table/{config['DDB_DATASET_TABLE']}",
            f"arn:aws:dynamodb:{config['AWS_REGION']}:{account_id}:table/{config['DDB_JOB_TABLE']}"
          ]
        },
        {
          "Sid": "LambdaManagement",
          "Effect": "Allow",
          "Action": [
            "lambda:CreateFunction",
            "lambda:UpdateFunctionCode",
            "lambda:UpdateFunctionConfiguration",
            "lambda:GetFunction",
            "lambda:GetFunctionConfiguration",
            "lambda:CreateEventSourceMapping",
            "lambda:PutFunctionConcurrency",
            "lambda:GetLayerVersion"
          ],
          "Resource": [
            f"arn:aws:lambda:{config['AWS_REGION']}:{account_id}:function:{config['LAMBDA_LIFECYCLE']}",
            f"arn:aws:lambda:{config['AWS_REGION']}:{account_id}:function:{config['LAMBDA_IMAGE_OPS']}",
            f"arn:aws:lambda:{config['AWS_REGION']}:{account_id}:function:{config['LAMBDA_SYNC']}",
            f"arn:aws:lambda:{config['AWS_REGION']}:{account_id}:function:{config['LAMBDA_DLQ']}"
          ]
        },
        {
          "Sid": "ECRRepositoryAndPush",
          "Effect": "Allow",
          "Action": [
            "ecr:CreateRepository",
            "ecr:DescribeRepositories",
            "ecr:GetAuthorizationToken",
            "ecr:BatchCheckLayerAvailability",
            "ecr:InitiateLayerUpload",
            "ecr:UploadLayerPart",
            "ecr:CompleteLayerUpload",
            "ecr:PutImage"
          ],
          "Resource": [
            f"arn:aws:ecr:{config['AWS_REGION']}:{account_id}:repository/{config['IMAGE_NAME_LIFECYCLE']}",
            f"arn:aws:ecr:{config['AWS_REGION']}:{account_id}:repository/{config['IMAGE_NAME_IMAGE_OPS']}",
            f"arn:aws:ecr:{config['AWS_REGION']}:{account_id}:repository/{config['IMAGE_NAME_SYNC']}",
            f"arn:aws:ecr:{config['AWS_REGION']}:{account_id}:repository/{config['IMAGE_NAME_DLQ']}"
          ]
        },
        {
          "Sid": "MakeLogGroup",
          "Effect": "Allow",
          "Action": [
            "logs:CreateLogGroup",
            "logs:PutRetentionPolicy",
            "logs:DescribeLogGroups"
          ],
          "Resource": "*"
        },
        {
          "Sid": "AllowSSMParamAccess",
          "Effect": "Allow",
          "Action": [
            "ssm:GetParametersByPath",
            "ssm:GetParameters",
            "ssm:GetParameter"
          ],
          "Resource": f"arn:aws:ssm:{config['AWS_REGION']}:{account_id}:parameter/cv-datasets/single-label/{config['INFRASTRUCTURE_NAME']}/infrastructure/*"
        }
      ]
    }
    
    return bootstrap_policy

def generate_api_policy(account_id, config):
    # return dict with the api-user policy JSON
    
    api_policy = {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid": "DatasetTableAccess",
          "Effect": "Allow",
          "Action": [
            "dynamodb:GetItem",
            "dynamodb:UpdateItem",
            "dynamodb:Scan",
            "dynamodb:DescribeTable"
          ],
          "Resource": f"arn:aws:dynamodb:{config['AWS_REGION']}:{account_id}:table/{config['DDB_DATASET_TABLE']}"
        },
        {
          "Sid": "JobTableAccess",
          "Effect": "Allow",
          "Action": [
            "dynamodb:PutItem",
            "dynamodb:UpdateItem",
            "dynamodb:DescribeTable"
          ],
          "Resource": f"arn:aws:dynamodb:{config['AWS_REGION']}:{account_id}:table/{config['DDB_JOB_TABLE']}"
        },
        {
          "Sid": "LifecycleQueueAccess",
          "Effect": "Allow",
          "Action": [
            "sqs:GetQueueUrl",
            "sqs:SendMessage"
          ],
          "Resource": f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_LIFECYCLE']}"
        },
        {
          "Sid": "ImageOpsQueueAccess",
          "Effect": "Allow",
          "Action": [
            "sqs:GetQueueUrl",
            "sqs:SendMessage"
          ],
          "Resource": f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_IMAGE_OPS']}"
        },
        {
          "Sid": "SyncQueueAccess",
          "Effect": "Allow",
          "Action": [
            "sqs:GetQueueUrl",
            "sqs:SendMessage"
          ],
          "Resource": f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_SYNC']}"
        },
        {
          "Sid": "S3TempImagesAccess",
          "Effect": "Allow",
          "Action": [
            "s3:PutObject",
            "s3:DeleteObject"
          ],
          "Resource": [
            f"arn:aws:s3:::{config['S3_BUCKET_NAME']}/{config['S3_DATASETS_ROOT']}/temp-images/*"
          ]
        },
        {
          "Sid": "AllowLogQuery",
          "Effect": "Allow",
          "Action": [
            "logs:StartQuery",
            "logs:GetQueryResults",
          ],
          "Resource": "*"
        },
        {
          "Sid": "AllowSSMParamAccess",
          "Effect": "Allow",
          "Action": [
            "ssm:GetParametersByPath",
            "ssm:GetParameters",
            "ssm:GetParameter"
          ],
          "Resource": f"arn:aws:ssm:{config['AWS_REGION']}:{account_id}:parameter/cv-datasets/single-label/{config['INFRASTRUCTURE_NAME']}/infrastructure/*"
        }
      ]
    }
    return api_policy

def generate_teardown_policy(account_id, config):
    # return dict with the teardown policy JSON

    teardown_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3Teardown",
                "Effect": "Allow",
                "Action": [
                    "s3:DeleteBucketPolicy",
                    "s3:DeleteObject",
                    "s3:DeleteObjectVersion",
                    "s3:ListBucket"
                ],
                "Resource": [
                    f"arn:aws:s3:::{config['S3_BUCKET_NAME']}",
                    f"arn:aws:s3:::{config['S3_BUCKET_NAME']}/{config['S3_DATASETS_ROOT']}/*"
                ]
            },
            {
                "Sid": "LambdaTeardown",
                "Effect": "Allow",
                "Action": [
                    "lambda:DeleteFunction",
                    "lambda:DeleteEventSourceMapping",
                    "lambda:ListEventSourceMappings"
                ],
                "Resource": "*"
            },
            {
                "Sid": "ECRTeardown",
                "Effect": "Allow",
                "Action": [
                    "ecr:DeleteRepository",
                    "ecr:DescribeRepositories"
                ],
                "Resource": [
                    f"arn:aws:ecr:{config['AWS_REGION']}:{account_id}:repository/{config['IMAGE_NAME_LIFECYCLE']}",
                    f"arn:aws:ecr:{config['AWS_REGION']}:{account_id}:repository/{config['IMAGE_NAME_IMAGE_OPS']}",
                    f"arn:aws:ecr:{config['AWS_REGION']}:{account_id}:repository/{config['IMAGE_NAME_SYNC']}",
                    f"arn:aws:ecr:{config['AWS_REGION']}:{account_id}:repository/{config['IMAGE_NAME_DLQ']}"
                ]
            },
            {
                "Sid": "DynamoDBTeardown",
                "Effect": "Allow",
                "Action": [
                    "dynamodb:DeleteTable",
                    "dynamodb:DescribeTable"
                ],
                "Resource": [
                    f"arn:aws:dynamodb:{config['AWS_REGION']}:{account_id}:table/{config['DDB_IMAGERY_TABLE']}",
                    f"arn:aws:dynamodb:{config['AWS_REGION']}:{account_id}:table/{config['DDB_DATASET_TABLE']}",
                    f"arn:aws:dynamodb:{config['AWS_REGION']}:{account_id}:table/{config['DDB_JOB_TABLE']}"
                ]
            },
            {
                "Sid": "SQSTeardown",
                "Effect": "Allow",
                "Action": [
                    "sqs:DeleteQueue",
                    "sqs:GetQueueUrl",
                    "sqs:GetQueueAttributes"
                ],
                "Resource": [
                    f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_LIFECYCLE']}",
                    f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_IMAGE_OPS']}",
                    f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_SYNC']}",
                    f"arn:aws:sqs:{config['AWS_REGION']}:{account_id}:{config['SQS_QUEUE_DLQ']}"
                ]
            },
            {
                "Sid": "IAMTeardown",
                "Effect": "Allow",
                "Action": [
                    "iam:DeleteRole",
                    "iam:DeleteRolePolicy",
                    "iam:DetachRolePolicy",
                    "iam:ListAttachedRolePolicies",
                    "iam:ListRolePolicies",
                    "iam:ListInstanceProfilesForRole",
                    "iam:RemoveRoleFromInstanceProfile"
                ],
                "Resource": [
                    f"arn:aws:iam::{account_id}:role/{config['LIFECYCLE_ROLE_NAME']}",
                    f"arn:aws:iam::{account_id}:role/{config['IMAGE_OPS_ROLE_NAME']}",
                    f"arn:aws:iam::{account_id}:role/{config['SYNC_ROLE_NAME']}",
                    f"arn:aws:iam::{account_id}:role/{config['DLQ_ROLE_NAME']}",
                    f"arn:aws:iam::{account_id}:instance-profile/*"
                ]
            },
            {
                "Sid": "STSCallerIdentity",
                "Effect": "Allow",
                "Action": "sts:GetCallerIdentity",
                "Resource": "*"
            },
            {
                "Sid": "DeleteLogGroup",
                "Effect": "Allow",
                "Action": "logs:DeleteLogGroup",
                "Resource": f"arn:aws:logs:{config['AWS_REGION']}:{account_id}:log-group:{config['LOG_GROUP_NAME']}"
            },
            {
                "Sid": "SSMRemoveParameterStore",
                "Effect": "Allow",
                "Action": [
                    "ssm:DeleteParameter",
                    "ssm:DeleteParameters"
                ],
                "Resource": f"arn:aws:ssm:{config['AWS_REGION']}:{account_id}:parameter/cv-datasets/single-label/{config['INFRASTRUCTURE_NAME']}/*"
            }
        ]
    }

    return teardown_policy

if __name__ == "__main__":
    # -------------------------------
    # Configure logging settings
    # -------------------------------
    main_dir = os.path.dirname(__file__)
    base_logs = os.path.join(main_dir, "logs")
    os.makedirs(base_logs, exist_ok=True)
        
    logging_save_to = os.path.join(base_logs, "part_2_logs.txt")
    
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
    parser.add_argument("--region",
                        required=True,
                        help=f"Region for the infrastructure. Must be one of: {', '.join(VALID_REGIONS)}")
    parser.add_argument("--infrastructure_name",
                        required=True,
                        help="Namespace for the infrastructure, must not exist.")
    parser.add_argument("--bucket_name",
                        required=True,
                        help="S3 Bucket to store the DMS.")
    parser.add_argument("--bucket_root",
                        required=True,
                        help="Location in the S3 bucket to store the dataset.")
    parser.add_argument("--profile_name",
                        required=True,
                        help="AWS profile to use for credentials")

    args = parser.parse_args()
    region = args.region.lower()
    infrastructure_name = args.infrastructure_name.lower()
    bucket_name = args.bucket_name.lower()
    bucket_root = normalize_root(args.bucket_root.lower())
    profile_name = args.profile_name

    if region not in VALID_REGIONS:
        raise ValueError(f'Invalid region {region}.')
    
    session = boto3.Session(profile_name=profile_name, region_name=region)
    
    # -------------------------------
    # Get clients.
    # -------------------------------    
    clients = {
            "sts": session.client("sts"),
            "s3": session.client("s3"),
            "sqs": session.client("sqs"),
            "ddb": session.client("dynamodb"),
            "lambda": session.client("lambda"),
            "iam": session.client("iam"),
            "ecr": session.client("ecr"),
            "ssm": session.client("ssm"),
            'logs': session.client("logs")
        }
        
    identity = clients['sts'].get_caller_identity()
    account_id = identity["Account"]

    logger.info(f"Running as {identity['Arn']} in account {account_id}")

    # -------------------------------
    # Build the resource names.
    # -------------------------------
    config = {
        # Core project context
        "AWS_REGION": region,
        "INFRASTRUCTURE_NAME": infrastructure_name,
        "S3_BUCKET_NAME": bucket_name,
        "S3_DATASETS_ROOT": bucket_root,
    
        # Log Group
        "LOG_GROUP_NAME": f"{infrastructure_name}/loggroup-{random_short_hash()}",

        # SQS Queues
        "SQS_QUEUE_LIFECYCLE": f"{infrastructure_name}-lifecyclequeue-{random_short_hash()}",
        "SQS_QUEUE_IMAGE_OPS": f"{infrastructure_name}-imgopsqueue-{random_short_hash()}",
        "SQS_QUEUE_SYNC": f"{infrastructure_name}-syncqueue-{random_short_hash()}",
        "SQS_QUEUE_DLQ": f"{infrastructure_name}-dlq-{random_short_hash()}",
    
        # Lambda Functions
        "LAMBDA_LIFECYCLE": f"{infrastructure_name}-lifecycleworker-{random_short_hash()}",
        "LAMBDA_IMAGE_OPS": f"{infrastructure_name}-imgopsworker-{random_short_hash()}",
        "LAMBDA_SYNC": f"{infrastructure_name}-syncworker-{random_short_hash()}",
        "LAMBDA_DLQ": f"{infrastructure_name}-dlqworker-{random_short_hash()}",
    
        # DynamoDB Tables
        "DDB_IMAGERY_TABLE": f"{infrastructure_name}-imagerytable-{random_short_hash()}",
        "DDB_DATASET_TABLE": f"{infrastructure_name}-datasettable-{random_short_hash()}",
        "DDB_JOB_TABLE": f"{infrastructure_name}-jobtable-{random_short_hash()}",
    
        # IAM Roles
        "LIFECYCLE_ROLE_NAME": f"{infrastructure_name}-lifecyclerole-{random_short_hash()}",
        "IMAGE_OPS_ROLE_NAME": f"{infrastructure_name}-imgopsrole-{random_short_hash()}",
        "SYNC_ROLE_NAME": f"{infrastructure_name}-syncrole-{random_short_hash()}",
        "DLQ_ROLE_NAME": f"{infrastructure_name}-dlqrole-{random_short_hash()}",
    
        # ECR Images
        "IMAGE_NAME_LIFECYCLE": f"{infrastructure_name}-lifecycleimg-{random_short_hash()}",
        "IMAGE_NAME_IMAGE_OPS": f"{infrastructure_name}-imgopsimg-{random_short_hash()}",
        "IMAGE_NAME_SYNC": f"{infrastructure_name}-syncimg-{random_short_hash()}",
        "IMAGE_NAME_DLQ": f"{infrastructure_name}-dlqimg-{random_short_hash()}",
    }
    
    # -------------------------------
    # Validate the resource names.
    # -------------------------------
    
    # Make sure the infrastructure_name namespace does not already exist, if it does, error out.
    resp = clients['ssm'].get_parameters_by_path(
        Path=f"/cv-datasets/single-label/{infrastructure_name}/",
        MaxResults=1
    )
    if resp.get("Parameters"):
        sys.exit(f"Error: Namespace /cv-datasets/single-label/{infrastructure_name} already exists in Parameter Store, choose different infrastructure_name")

    try:
        run_all_validations(config)       # regex/name checks
        run_existence_checks(config, clients)  # AWS existence checks
    except ValidationError as e:
        logging.error(e)   # prints all errors
        sys.exit(1)
            
    logger.info("Validation complete. Registering parameters in SSM...")    
    
    for key, value in config.items():
        clients['ssm'].put_parameter(
            Name=f"/cv-datasets/single-label/{infrastructure_name}/infrastructure/{key}",
            Value=value,
            Type="String",
            Overwrite=False,
            Description=f"{key} for infrastructure {infrastructure_name}"
        )
        
        logger.info(f"Registered {key} -> /cv-datasets/single-label/{infrastructure_name}/infrastructure/{key}")
    
    logger.info('Config registered.')
    
    os.makedirs(f"generated_permissions/{infrastructure_name}", exist_ok=True)

    with open(f"generated_permissions/{infrastructure_name}/bootstrap-policy.json", "w") as f:
        json.dump(generate_bootstrap_policy(account_id, config), f, indent=2)

    with open(f"generated_permissions/{infrastructure_name}/api-user-policy.json", "w") as f:
        json.dump(generate_api_policy(account_id, config), f, indent=2)
        
    with open(f"generated_permissions/{infrastructure_name}/teardown-policy.json", "w") as f:
        json.dump(generate_teardown_policy(account_id, config), f, indent=2)

    logger.info("Policies written to generated/permissions/")