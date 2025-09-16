# -*- coding: utf-8 -*-
"""
Helper script for the Python worker. Add here whatever functionality you need.
"""
import json
import boto3
import logging
from botocore.exceptions import ClientError

def upload_dict_to_s3(data, bucket_name, object_key):
    """
    Upload a JSON-serializable Python object to an S3 bucket.

    Parameters:
    - data (dict): The Python object to serialize and upload.
    - bucket_name (str): Name of the target S3 bucket.
    - object_key (str): S3 object key (e.g., 'folder/subfolder/file.json').

    Returns:
    - True if upload succeeded, False otherwise.
    """
    s3_client = boto3.client('s3')

    try:
        json_bytes = json.dumps(data).encode('utf-8')
        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=json_bytes)
    except ClientError as e:
        logging.error(f"Failed to upload JSON to S3: {e}")
        return False

    return True