# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:29:14 2025

@author: brian
"""

# to do

# perhaps change phash existence check in api like in the sync lambda?

# write out sync logic in the lambda and its helpers

# make a api function to query and return the phash of images fitting certain conditions/ttribute values

# make api function to query all logs related to a: dataset, lambda (evn dlq), job id

# make an api call that will accept phashes from a query say, and then assign them to a dataset if not already beloinging to it
#       this will let us 'transfer images from one set to another'

# verify lambda execution roles

# verify api role, teardown role and bootstrap role

#%% Notes

    tables = [
        {
            "name_key": 'DDB_DATASET_TABLE',
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
            "name_key": 'DDB_IMAGERY_TABLE',
            "params": {
                "AttributeDefinitions": [
                    {"AttributeName": "dataset_unique_id", "AttributeType": "S"},
                    {"AttributeName": "dataset_id", "AttributeType": "S"},
                    {"AttributeName": "unique_id", "AttributeType": "S"}
                ],
                "KeySchema": [
                    {"AttributeName": "dataset_unique_id", "KeyType": "HASH"}
                ],
                "BillingMode": "PAY_PER_REQUEST",
                "GlobalSecondaryIndexes": [
                    {
                        "IndexName": "DatasetIndex",
                        "KeySchema": [
                            {"AttributeName": "dataset_id", "KeyType": "HASH"},
                            {"AttributeName": "unique_id", "KeyType": "RANGE"}
                        ],
                        "Projection": {"ProjectionType": "ALL"}
                    },
                    {
                        "IndexName": "UniqueIdIndex",
                        "KeySchema": [
                            {"AttributeName": "unique_id", "KeyType": "HASH"}
                        ],
                        "Projection": {"ProjectionType": "ALL"}
                    }
                ]
            }
        },

        {
            "name_key": 'DDB_JOB_TABLE',
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
    
    
    lambda_defs = {'LAMBDA_LIFECYCLE': {'image_name_key': 'IMAGE_NAME_LIFECYCLE',
                                               'path': "lambdas/lifecycle_lambda",
                                               'role': role_arns['LIFECYCLE_ROLE_ARN'],
                                               'queue_arn': queue_arns['SQS_QUEUE_LIFECYCLE_ARN'],
                                               'memory_mb': 512,
                                               'timeout_sec': 300,
                                               'queue_visibility': 360,
                                               'batch_size': 10},
                   'LAMBDA_IMAGE_OPS': {'image_name_key': 'IMAGE_NAME_IMAGE_OPS',
                                                'path': "lambdas/image_ops_lambda",
                                                'role': role_arns['IMAGE_OPS_ROLE_ARN'],
                                                'queue_arn': queue_arns['SQS_QUEUE_IMAGE_OPS_ARN'],
                                                'memory_mb': 1024,
                                                'timeout_sec': 600,
                                                'queue_visibility': 660,
                                                'batch_size': 10},
                   'LAMBDA_SYNC':      {'image_name_key': 'IMAGE_NAME_SYNC',
                                                'path': "lambdas/sync_lambda",
                                                'role': role_arns['SYNC_ROLE_ARN'],
                                                'queue_arn': queue_arns['SQS_QUEUE_SYNC_ARN'],
                                                'memory_mb': 3072,
                                                'timeout_sec': 900,
                                                'queue_visibility': 960,
                                                'batch_size': 1},
                   'LAMBDA_DLQ':      {'image_name_key': 'IMAGE_NAME_DLQ',
                                                'path': "lambdas/dlq_lambda",
                                                'role': role_arns['DLQ_ROLE_ARN'],
                                                'queue_arn': queue_arns['SQS_QUEUE_DLQ_ARN'],
                                                'memory_mb': 512,
                                                'timeout_sec': 60,
                                                'queue_visibility': 90,
                                                'batch_size': 10}}