# -*- coding: utf-8 -*-
"""
Utilitiy functions to assist in setting up the infrastructure.
"""

import json
import logging

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

    role_arns = {}
    
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
    
    role_arns['LIFECYCLE_ROLE_ARN'] = resp['Role']['Arn']
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
    
    role_arns['IMAGE_OPS_ROLE_ARN'] = resp['Role']['Arn']

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
    
    role_arns['SYNC_ROLE_ARN'] = resp['Role']['Arn']

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
    
    role_arns['DLQ_ROLE_ARN'] = resp['Role']['Arn']
    
    return role_arns

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

    queue_arns = {}
    
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
    queue_arns['SQS_QUEUE_DLQ_ARN'] = attrs['Attributes']['QueueArn']
    
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
        queue_arns[f"{queue_key}_ARN"] = attrs['Attributes']['QueueArn']
    
    return queue_arns

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