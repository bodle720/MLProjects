# -*- coding: utf-8 -*-
"""
A sample Python script that will run a worker function on AWS Batch
through a Docker container.
"""
import psutil
import logging
import threading
import time
import os
import json
import argparse
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import pandas_ta as ta

import helpers

# Configure logger. Send logs to stdout, preserving only the message text.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_features(bucket_name,
                input_key,
                dt,
                tf_int,
                symb):
    
    # Can use symb for logging.
    
    # Load in the DataFrame.
    # df will have columns: Date, open, high, low, close, and volume, with Date being strings format 'yyyymmdd'
    df = helpers.get_df_or_dict_parquet_from_s3(bucket_name, input_key)

    # Make the Date the index and a Datetime object we can use for filtering.
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df.set_index('Date', inplace=True)

    # Trim the daily DataFrame to the required day.
    cutoff_date = pd.to_datetime(dt, format='%Y%m%d')
    df = df[df.index <= cutoff_date]
    
    if len(df) == 0:
        return {}
    
    # Sample to appropriate timeframe.
    NUM_PAST_VALUES = 30
    RSI_PERIODS = [2, 5, 7, 9, 14, 21, 25, 30]  
        
    df.columns = df.columns.str.lower()
    
    num_required_past_bars = 1101 # roughly based on needing for the 5 day timeframe, calculating the 50 sma, need 250 bars of daily date

    start_index = len(df) - num_required_past_bars

    if start_index < 0:
        return {}
                
    df = df.iloc[start_index:]

    # Stores the features.
    tf_dict = {}
    tf_slice = df.copy(deep=True)
    
    rows_to_trim = len(tf_slice) % tf_int
    tf_slice = tf_slice.iloc[rows_to_trim:].copy()
    
    # Assign each their groups, first the earlier dates.
    group_labels = tf_slice.index.to_series().reset_index(drop=True).index // tf_int
    tf_slice['group'] = group_labels
         
    tf_df = tf_slice.iloc[::-1].groupby('group').agg({'open': 'last',   
                                                    'high': 'max',  
                                                    'low': 'min',     
                                                    'close': 'first',
                                                    'volume': 'sum'})
    
    # now add in the Heikin Ashi candles for this timeframe
    df_ha = helpers.calculate_heikin_ashi(tf_df)
    tf_df = pd.concat([tf_df, df_ha], axis=1)
            
    cols_to_round = ['open', 'high', 'low', 'close', 'close_h', 'open_h', 'high_h', 'low_h']
    for col in cols_to_round:
        tf_df[col] = tf_df[col].map(helpers.rounder)
    
    # RSI features as an example.
    for rsi_length in RSI_PERIODS:
        rsi_series = ta.rsi(tf_df['close'], length = rsi_length).tail(NUM_PAST_VALUES).dropna().map(helpers.rounder)
        tf_dict[f'rsi_{rsi_length}'] = rsi_series.to_list()
        
    # Add the last Hekinashi close price.
    lastHACandles = tf_df[['open_h', 'high_h', 'low_h', 'close_h']].tail(NUM_PAST_VALUES).to_dict(orient='records')
    tf_dict['last_ha_close'] = lastHACandles[-1]['close_h']
    
    return tf_dict

def process_job(bucket_name, input_key, output_key, dt, tf_int, symb):
    
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket_name, Key=output_key)
        logger.info(f"Output exists at s3://{bucket_name}/{output_key}; skipping.")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise
            
            
    process = psutil.Process()
    peak_mem = 0
    
    peak_cpu = 0.0
    cpu_sum = 0.0
    cpu_count = 0
    
    # Prime CPU percent measurement
    process.cpu_percent(None)

    # Event to signal threads to stop
    stop_event = threading.Event()
    
    def track_memory():
        nonlocal peak_mem
        while not stop_event.is_set():
            try:
                mem = process.memory_info().rss
                peak_mem = max(peak_mem, mem)
                time.sleep(0.5)
            except Exception:
                break

    def track_cpu():
        nonlocal peak_cpu, cpu_sum, cpu_count
        while not stop_event.is_set():
            try:
                cpu = process.cpu_percent(interval=0.5)
                peak_cpu = max(peak_cpu, cpu)
                cpu_sum += cpu
                cpu_count += 1
            except Exception:
                break
    
    mem_thread = threading.Thread(target=track_memory, daemon=True)
    cpu_thread = threading.Thread(target=track_cpu, daemon=True)
    
    mem_thread.start()
    cpu_thread.start()
    
    features = get_features(bucket_name,
                            input_key,
                            dt,
                            int(tf_int), 
                            symb)
    
    helpers.upload_df_or_dict_as_parquet_to_s3(features, bucket_name, output_key)
    
    # Signal threads to stop and give them a moment to exit
    stop_event.set()
    mem_thread.join(timeout=1)
    cpu_thread.join(timeout=1)
    
    # Log final metrics
    peak_mib = peak_mem / (1024 ** 2)
    avg_cpu = cpu_sum / cpu_count if cpu_count else 0.0
    
    logger.info(f"[MEMORY_METRIC] peak_memory_mib={peak_mib:.2f}")
    logger.info(f"[CPU_METRIC] peak_cpu_pct={peak_cpu:.2f}, avg_cpu_pct={avg_cpu:.2f}")
    
def main():
    parser = argparse.ArgumentParser(
        description="Process one job or array-job slice."
    )

    # Array job flags.
    parser.add_argument("--manifest-s3-bucket", help="S3 bucket holding the jobs manifest")
    parser.add_argument("--manifest-key",       help="S3 object key to the JSON manifest")

    # Legacy positional args
    parser.add_argument("bucket_name", nargs="?",
                        help="S3 bucket name for input/output")
    parser.add_argument("input_key",   nargs="?",
                        help="S3 key for input parquet")
    parser.add_argument("output_key",  nargs="?",
                        help="S3 key for output parquet")
    parser.add_argument("dt",          nargs="?",
                        help="Date (dt)")
    parser.add_argument("tf_int",      nargs="?",
                        help="Timeframe integer")
    parser.add_argument("symb",        nargs="?",
                        help="Symbol")
    
    args = parser.parse_args()

    # Decide which mode: batch or testing.
    has_manifest = args.manifest_s3_bucket and args.manifest_key
    has_legacy   = all([args.bucket_name, args.input_key,
                        args.output_key, args.dt,
                        args.tf_int, args.symb])
    
    if has_manifest:
        # Array mode
        
        # AWS_BATCH_JOB_ARRAY_INDEX is automatically set per container in an array job.
        # Because each manifest chunk is stored separately, no offset calculation is needed.
        idx = int(os.environ["AWS_BATCH_JOB_ARRAY_INDEX"]) # Chunked, so no need for index.

        s3 = boto3.client("s3")
        obj = s3.get_object(
            Bucket=args.manifest_s3_bucket,
            Key=args.manifest_key
        )
        jobs = json.loads(obj["Body"].read())

        job = jobs[idx]
        
        process_job(
            job["bucket"],
            job["input_key"],
            job["output_key"],
            job["dt"],
            job["tf_int"],
            job["symb"]
        )

    elif has_legacy:
        # Legacy mode, good for command line local script testing.
        process_job(
            args.bucket_name,
            args.input_key,
            args.output_key,
            args.dt,
            args.tf_int,
            args.symb
        )
    else:
        parser.error(
            "Either provide BOTH --manifest-s3-bucket & --manifest-key for array mode, "
            "OR provide 6 positional args for legacy mode."
        )
      
if __name__ == "__main__":
    main()