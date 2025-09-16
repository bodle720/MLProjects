# -*- coding: utf-8 -*-
"""
DynamoDB Helpers.
"""

import logging
from botocore.exceptions import ClientError

def create_table(dynamodb_resource,
                 table_name,
                 partition_key,
                 sort_key,
                 partition_key_type = "S",
                 sort_key_type = "S",
                 rcu = 5,
                 wcu = 5):
    """
    Creates a DynamoDB table with the specified name, partition key, and sort key.

    Validates that the key types are one of the allowed scalar types: 'S' (String), 'N' (Number), or 'B' (Binary).
    If the table already exists, logs that fact and returns True.
    If the table is created successfully, waits until it's active, logs success, and returns True.
    If the key types are invalid or any error occurs during creation, logs the error and returns False.

    Parameters:
        dynamodb_resource (boto3.resource): A DynamoDB resource object.
        table_name (str): The name of the table to create.
        partition_key (str): The name of the partition key attribute.
        sort_key (str): The name of the sort key attribute.
        partition_key_type (str): The type of the partition key ('S', 'N', or 'B'). Defaults to 'S'.
        sort_key_type (str): The type of the sort key ('S', 'N', or 'B'). Defaults to 'S'.
        rcu (int): Read capacity units for the table. Defaults to 5.
        wcu (int): Write capacity units for the table. Defaults to 5.

    Returns:
        bool: True if the table exists or was created successfully; False if validation or creation fails.
    """
    
    if partition_key_type not in ["S", "N", "B"]:
        return False
    
    if sort_key_type not in ["S", "N", "B"]:
        return False
    
    try:
        existing_tables = [t.name for t in dynamodb_resource.tables.all()]
        if table_name in existing_tables:
            logging.info(f"Table '{table_name}' already exists.")
            return True

        table = dynamodb_resource.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': partition_key, 'KeyType': 'HASH'},
                {'AttributeName': sort_key, 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': partition_key, 'AttributeType': 'S'},
                {'AttributeName': sort_key, 'AttributeType': 'S'}
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': rcu,
                'WriteCapacityUnits': wcu
            }
        )

        table.wait_until_exists()
        logging.info(f"Table '{table_name}' created successfully.")
        return True

    except ClientError as e:
        logging.error(f"Failed to create table '{table_name}': {e.response['Error']['Message']}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error while creating table '{table_name}': {str(e)}")
        return False
