# -*- coding: utf-8 -*-
"""
Helper script for the Python worker. Add here whatever functionality you need.
"""

import boto3
import logging
import pandas as pd
from botocore.exceptions import ClientError
from io import BytesIO
from decimal import Decimal

def rounder(val):
    
    val = str(val)
    
    if float(val) >= 1:
        precision = '0.01'
        num_after_allowed = 2
    else:
        precision = '0.0001'
        num_after_allowed = 4
        
    corrected_value = str(float(Decimal(val).quantize(Decimal(precision))))
    before_decimal, after_decimal = corrected_value.split('.')  # Split at the first period
    after_decimal = after_decimal[:num_after_allowed]  # Keep at most 2 or 4 characters
    ans = float(f"{before_decimal}.{after_decimal}")
    
    return ans

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function will calculate the Heikin-Ashi candle values and return them
    in the form of a pandas.DataFrame, with columns open_h, high_h, low_h, and
    close_h.
    
    Arguments
    ----------
    :param df: pandas.DataFrame with columns: open, high, low, close.
    :type df: pandas.DataFrame
    
    Returns
    ----------
    :return: pandas.DataFrame with Heikin Ashi candle values.
    :rtype: pandas.DataFrame
    '''

    df_original_ix = df.index
    df = df.reset_index(drop=True)
    ha_df = pd.DataFrame(index=df.index)  # Create a new DataFrame for Heikin-Ashi
    
    # Calculate Heikin-Ashi close (average of open, high, low, close of regular candles)
    ha_df['close_h'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Calculate Heikin-Ashi open (average of previous Heikin-Ashi open and haikinashi close)
    ha_df['open_h'] = df['open']  # Initialize with original open values
    for i in range(1, len(df)):
        ha_df.loc[i, 'open_h'] = (ha_df.loc[i-1, 'open_h'] + ha_df.loc[i-1, 'close_h']) / 2
    
    
    # Heikin-Ashi high (max of HA open, HA close, regular high)
    ha_df['high_h'] = ha_df[['open_h', 'close_h']].join(df['high']).max(axis=1)
    
    # Heikin-Ashi low (min of HA open, HA close, regular low)
    ha_df['low_h'] = ha_df[['open_h', 'close_h']].join(df['low']).min(axis=1)

    ha_df.index = df_original_ix
    
    return ha_df

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