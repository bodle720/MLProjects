# -*- coding: utf-8 -*-
"""
Helper functions for some post-batch processing analysis of the worker calls.
Requires AWS CLI to be installed and configured with appropriate permissions.
"""

import boto3
import subprocess
import json
import time
from datetime import datetime, timedelta, timezone
from typing import List

def list_s3_files_with_extension(bucket_name, prefix, extension):
    """
    Recursively list all files in an S3 bucket under a given prefix that match a specific file extension.

    This function uses Boto3's paginator to efficiently traverse all objects under the specified prefix,
    including nested folders, and filters them by the provided file extension.

    Parameters:
        bucket_name (str): Name of the S3 bucket.
        prefix (str): Folder path (prefix) within the bucket to search. Should end with '/' if targeting a folder.
        extension (str): File extension to match (e.g., '.csv', '.json', '.parquet').

    Returns:
        List[str]: A list of object keys (file paths) that match the given extension.

    Example:
        >>> list_s3_files_with_extension('my-bucket', 'data/', '.csv')
        ['data/file1.csv', 'data/archive/2023/file2.csv']
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    matched_keys: List[str] = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith(extension):
                matched_keys.append(key)

    return matched_keys

def start_insights_query(log_group, start_ts, end_ts, query_file, region):
    cmd = [
            "aws",
            "--region", region,       
            "logs", "start-query",
            "--log-group-name", log_group,
            "--start-time",    str(start_ts),
            "--end-time",      str(end_ts),
            "--query-string", f"file://{query_file}"
            ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"start-query failed: {proc.stderr}")
    return json.loads(proc.stdout)["queryId"]

def get_query_results(query_id, region):
    cmd = [
        "aws",
        "--region", region,
        "logs", "get-query-results",
        "--query-id", query_id
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"get-query-results failed: {proc.stderr}")
    return json.loads(proc.stdout)

def extract_metrics(results):
    """
    Returns a dict of { field_name: float(value) } for the first row.
    If there are no rows, returns an empty dict.
    Assumes metrics are in the first row and all values are numeric.
    """
    rows = results.get("results", [])
    if not rows:
        return {}            # <— guard against empty results

    row = rows[0]
    mets = {}
    for cell in row:
        mets[cell["field"]] = float(cell["value"])
    return mets

def run_query_and_wait(log_group, start_ts, end_ts, query_file, region):
    """
    Kicks off a Logs Insights query, polls until completion,
    and returns the extracted metrics dict.
    """
    qid = start_insights_query(log_group, start_ts, end_ts, query_file, region)
    print(f"Started Logs Insights query for '{query_file}': {qid}")

    while True:
        res = get_query_results(qid, region)
        status = res["status"]
        if status == "Complete":
            metrics = extract_metrics(res)
            if not metrics:
                print(f"No data returned for '{query_file}' in the given time window.")
            return metrics
        if status in ("Failed", "Cancelled"):
            raise RuntimeError(f"Logs Insights query {status}")
        print(f"Waiting for query '{query_file}'… status={status}")
        time.sleep(5)

def get_epoch_window(hours_ago_start, hours_ago_end):
    """
    Returns (start_ts, end_ts) as Unix epoch seconds  
    where start_ts = now - hours_ago_start  
          end_ts   = now - hours_ago_end

    Example: get_epoch_window(7, 3) → [
      7 hours ago … up to … 3 hours ago ] window.
    """
    
    if hours_ago_end >= hours_ago_start:
        raise Exception("Hours ago end param must be strictly less than hours ago start param.")
        
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_ago_start)
    end   = now - timedelta(hours=hours_ago_end)
    
    epoch_start = int(start.timestamp())
    epoch_end = int(end.timestamp())

    formatted_start = start.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    formatted_end = end.strftime("%Y-%m-%d %H:%M:%S") + " UTC"

    return epoch_start, epoch_end, formatted_start, formatted_end