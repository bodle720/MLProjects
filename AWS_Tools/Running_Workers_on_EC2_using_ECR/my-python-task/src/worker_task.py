# -*- coding: utf-8 -*-
"""
A sample Python script that will run a worker function on EC2 instances
through a Docker container.
"""

import sys
import pandas as pd

from helpers import upload_df_or_dict_as_parquet_to_s3

def worker(input_value):
    # Simulate data processing with pandas.
    df = pd.DataFrame({
        'input': [input_value],
        'result': [int(input_value) * 2]
    })
    return df

if __name__ == "__main__":
    
    # Expect two arguments: input_value and bucket_name.
    if len(sys.argv) != 3:
        raise ValueError("Usage: python worker_task.py <input_value> <bucket_name>")
        
    # Input args are passed by docker run command.
    input_value = sys.argv[1]
    bucket_name = sys.argv[2]
    
    # Do the work.
    df = worker(input_value)
    
    # This is where I want to save the result in S3.
    object_key = f"ec2_results/output_{input_value}.parquet"
        
    upload_df_or_dict_as_parquet_to_s3(df,
                                       bucket_name,
                                       object_key)
    
