# -*- coding: utf-8 -*-
"""
Helper script for the Python worker. Add here whatever functionality you need.
"""

import boto3
import logging
import pandas as pd
from botocore.exceptions import ClientError
from io import BytesIO

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