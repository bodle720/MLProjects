# -*- coding: utf-8 -*-
"""
Helper functions relating to uploading and downloading objects on S3.
"""

import os
import boto3
import logging
import json
import pandas as pd
from botocore.exceptions import ClientError
from io import StringIO, BytesIO

def summarize_buckets():
    """
    Summarize all S3 buckets in the account, including key attributes.

    Attributes reported per bucket:
    - Name
    - Creation date
    - Region
    - Bucket policy (or 'No policy')
    - Public access block configuration (or 'Not configured')
    - Versioning status (Enabled/Suspended/Not configured)

    Returns:
    - List of dictionaries summarizing each bucket.
    """
    s3 = boto3.client('s3')

    try:
        response = s3.list_buckets()
    except ClientError:
        return []

    summaries = []

    for bucket in response.get('Buckets', []):
        bucket_name = bucket['Name']
        summary = {
            'Name': bucket_name,
            'Created': bucket['CreationDate'].strftime('%Y-%m-%d %H:%M:%S'),
            'Region': 'Unknown',
            'Policy': 'No policy',
            'Versioning': 'Not configured'
        }

        # Get bucket region
        try:
            location = s3.get_bucket_location(Bucket=bucket_name)
            region = location.get('LocationConstraint') or 'us-east-1'
            summary['Region'] = region
        except ClientError:
            pass

        # Get bucket policy
        try:
            policy = s3.get_bucket_policy(Bucket=bucket_name)
            summary['Policy'] = policy['Policy']
        except ClientError:
            pass

        # Get versioning status
        try:
            versioning = s3.get_bucket_versioning(Bucket=bucket_name)
            status = versioning.get('Status')
            summary['Versioning'] = status if status else 'Not configured'
        except ClientError:
            pass

        summaries.append(summary)

    return summaries

def create_bucket(bucket_name, region = None):
    """
    Create an S3 bucket in a specified region (credit: boto3 docs)

    If a region is not specified, the bucket is created in the S3 default
    region of us-east-1.

    :param bucket_name: Bucket to create
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """

    # Create bucket
    if region == 'us-east-1':
        region = None
        
    try:
        if region is None:
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def delete_s3_bucket(bucket_name, force=False):
    """
    Delete an S3 bucket. Optionally delete all objects (and versions) before removal.

    Parameters:
    - bucket_name (str): Name of the S3 bucket to delete.
    - force (bool): If True, delete all objects and versions before deleting the bucket.

    Returns:
    - True if deletion succeeded, False otherwise.
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    versioning = s3.BucketVersioning(bucket_name)

    try:
        if force:
            # Delete all objects
            bucket.objects.all().delete()

            # If versioning is enabled, delete all versions too
            if versioning.status == 'Enabled':
                bucket.object_versions.delete()

        # Delete the bucket itself
        bucket.delete()
        logging.info(f"Bucket '{bucket_name}' deleted successfully.")
        return True

    except ClientError as e:
        logging.error(f"Failed to delete bucket '{bucket_name}': {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return False

def upload_local_file_to_s3(local_path, bucket_name, object_name = None):
    """
    Upload a file to an S3 bucket (credit: boto3 docs)
    If no object_name (the key, a path in the s3 bucket), then placed
    in the root directory of the bucket using the name of the file.

    :param local_path: File to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name. If not specified then local_path is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use local_path
    if object_name is None:
        object_name = os.path.basename(local_path)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_path, bucket_name, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    
    return True

def download_s3_obj_to_local_file(bucket_name, object_key, local_path):
    """
    Download an object from S3 and save it to a local file.

    Parameters:
    - bucket_name (str): Name of the S3 bucket.
    - object_key (str): Key of the object to download (e.g., 'folder/file.txt').
    - local_path (str): Local file path to save the downloaded content.

    Returns:
    - True if download succeeded, False otherwise.
    """
    s3_client = boto3.client('s3')

    try:
        with open(local_path, 'wb') as f:
            s3_client.download_fileobj(bucket_name, object_key, f)
    except ClientError as e:
        logging.error(f"Failed to download object from S3: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error while saving file: {e}")
        return False

    return True

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
    
def upload_text_to_s3(text_body, bucket_name, object_key):
    """
    Upload a list of text lines to an S3 bucket as a plain text file.

    Parameters:
    - text_body (str): Text to upload.
    - bucket_name (str): Name of the target S3 bucket.
    - object_key (str): S3 object key (e.g., 'folder/subfolder/file.txt').

    Returns:
    - True if upload succeeded, False otherwise.
    """
    s3_client = boto3.client('s3')

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=text_body.encode('utf-8'),
            ContentType='text/plain'
        )
    except ClientError as e:
        logging.error(f"Failed to upload text to S3: {e}")
        return False

    return True

def upload_dataframe_to_s3(df, bucket_name, object_key, encoding='utf-8', index=False):
    """
    Upload a pandas DataFrame to an S3 bucket as a CSV file.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to upload.
    - bucket_name (str): Name of the target S3 bucket.
    - object_key (str): S3 object key (e.g., 'folder/subfolder/data.csv').
    - encoding (str): Character encoding for CSV (default: 'utf-8').
    - index (bool): Whether to include the DataFrame index in the CSV (default: False).

    Returns:
    - True if upload succeeded, False otherwise.
    """
    s3_client = boto3.client('s3')

    try:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=index, encoding=encoding)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=csv_buffer.getvalue().encode(encoding),
            ContentType='text/csv'
        )
    except ClientError as e:
        logging.error(f"Failed to upload DataFrame to S3: {e}")
        return False

    return True

def get_dict_from_s3(bucket_name, object_key, encoding='utf-8'):
    """
    Pull a JSON file from S3 and deserialize it into a Python dict.

    Parameters:
    - bucket_name (str): Name of the S3 bucket.
    - object_key (str): Key of the object to download (e.g., 'folder/file.json').
    - encoding (str): Character encoding to decode the file (default: 'utf-8').

    Returns:
    - Deserialized Python object (dict) if successful, None otherwise.
    """
    s3_client = boto3.client('s3')

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        content = response['Body'].read().decode(encoding)
        return json.loads(content)
    except (ClientError, json.JSONDecodeError) as e:
        logging.error(f"Failed to download or parse JSON from S3: {e}")
        return None
        
def get_text_from_s3(bucket_name, object_key, encoding='utf-8'):
    """
    Pull a plain text file from S3 and return its contents.

    Parameters:
    - bucket_name (str): Name of the S3 bucket.
    - object_key (str): Key of the object to download (e.g., 'folder/file.txt').
    - encoding (str): Character encoding to decode the file (default: 'utf-8').

    Returns:
    - String if successful, None if pull fails.
    """
    s3_client = boto3.client('s3')

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        content = response['Body'].read().decode(encoding)
        return content
    except ClientError as e:
        logging.error(f"Failed to download text from S3: {e}")
        return None

def get_dataframe_from_s3(bucket_name, object_key, encoding='utf-8', delimiter=','):
    """
    Download a CSV file from S3 and load it into a pandas DataFrame.

    Parameters:
    - bucket_name (str): Name of the S3 bucket.
    - object_key (str): Key of the object to download (e.g., 'folder/data.csv').
    - encoding (str): Character encoding for the CSV file (default: 'utf-8').
    - delimiter (str): CSV delimiter (default: ',').

    Returns:
    - pandas.DataFrame if successful, None otherwise.
    """
    s3_client = boto3.client('s3')

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        return pd.read_csv(response['Body'], encoding=encoding, delimiter=delimiter)
    except ClientError as e:
        logging.error(f"Failed to download CSV from S3: {e}")
    except pd.errors.ParserError as e:
        logging.error(f"Failed to parse CSV content: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while loading CSV: {e}")

    return None

def upload_df_or_dict_as_parquet_to_s3(data, bucket_name, object_key, engine='pyarrow', compression='snappy', index=False, max_level=2):
    """
    Upload a Python dictionary or pandas DataFrame to S3 as a Parquet file.

    Parameters:
    - data (dict or pandas.DataFrame): Input data to upload.
        If dict, it will be normalized into a DataFrame.
    - bucket_name (str): Name of the target S3 bucket.
    - object_key (str): S3 object key (e.g., 'folder/data.parquet').
    - engine (str): Parquet engine to use ('pyarrow' or 'fastparquet').
    - compression (str): Compression codec for Parquet (default: 'snappy').
    - index (bool): Whether to include the DataFrame index in the Parquet file (default: False).
    - max_level (int): Depth to flatten nested dictionaries (used only if input is dict).

    Returns:
    - True if upload succeeded, False otherwise.
    """
    s3_client = boto3.client('s3')

    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pd.json_normalize(data, sep='_', max_level=max_level)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise TypeError("Input must be a dict or pandas DataFrame")

        # Write Parquet to in-memory buffer
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, engine=engine, compression=compression, index=index)

        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=object_key,
            Body=parquet_buffer.getvalue(),
            ContentType='application/octet-stream'
        )
    except (ClientError, TypeError) as e:
        logging.error(f"Failed to upload Parquet to S3: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during Parquet upload: {e}")
        return False

    return True

def get_df_or_dict_parquet_from_s3(bucket_name, object_key, return_as_dict=False, engine='pyarrow'):
    """
    Download a Parquet file from S3 and return its contents as either a DataFrame or a list of dictionaries.

    Parameters:
    - bucket_name (str): Name of the S3 bucket.
    - object_key (str): Key of the Parquet object (e.g., 'folder/data.parquet').
    - return_as_dict (bool): Should the function return a dict (True) or pd.DataFrame (False)
    - engine (str): Parquet decoding engine ('pyarrow' or 'fastparquet').

    Returns:
    - pandas.DataFrame or dict, depending on return_as_dict.
    - None if download or parsing fails.
    """
    s3_client = boto3.client('s3')

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        parquet_bytes = BytesIO(response['Body'].read())
        df = pd.read_parquet(parquet_bytes, engine=engine)

        if return_as_dict:
            return df.to_dict(orient='records')
        
        return df

    except ClientError as e:
        logging.error(f"Failed to download Parquet from S3: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while reading Parquet: {e}")

    return None