# -*- coding: utf-8 -*-
"""
A sample Python script that will run a worker function on AWS Batch
through a Docker container.

Called by first lambda in the step function, that outputssomethign like:
    
    resp = {
        'x':x,
        'y':y,
        'z':z,
        f'x^2 + y^2 = {z}': 'success',
        'explanation': f'sqrt z = {sqrt_z}, meaning {sqrt_z}^2 = {x}^2 + {y}^2',
        'The environment variable is: ': some_val
    }

    return {
        'statusCode': 200,
        'body': json.dumps(resp)
    }
"""
import psutil
import logging
import threading
import time
import boto3
from botocore.exceptions import ClientError
import json
from helpers import upload_dict_to_s3

# Configure logger. Send logs to stdout, preserving only the message text.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_features():
    tf_dict = {'message': 'inside the get features funciton',
               'x1': 100,
               'y1': 90,
               'z1': 0}
    
    w = 0
    for _ in range(1000):
        w += 1
        
    tf_dict['w'] = w
    time.sleep(10)
    
    return tf_dict

def process_job(bucket_name, output_key):#, l1_x, l1_y, l1_z):
    
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
    
    features_dict = get_features()
    
    # features_dict['xfroml1'] = l1_x
    # features_dict['yfroml1'] = l1_y
    # features_dict['zfroml1'] = l1_z

    upload_dict_to_s3(features_dict, bucket_name, output_key)
    
    # Signal threads to stop and give them a moment to exit
    stop_event.set()
    mem_thread.join(timeout=1)
    cpu_thread.join(timeout=1)
    
    # Log final metrics
    peak_mib = peak_mem / (1024 ** 2)
    avg_cpu = cpu_sum / cpu_count if cpu_count else 0.0
    
    logger.info(f"[MEMORY_METRIC] peak_memory_mib={peak_mib:.2f}")
    logger.info(f"[CPU_METRIC] peak_cpu_pct={peak_cpu:.2f}, avg_cpu_pct={avg_cpu:.2f}")
    
def lambda_handler(event, context):
    
    # get the input
    print(event)
    # l1_x = event['x']
    # l1_y = event['y']
    # l1_z = event['z']
    
    bucket_name = 'batch-demo32'
    key = 'fromlambda_step_function/mystepoutput.json'
    
    process_job(bucket_name, key)#, l1_x, l1_y, l1_z)
    
    logging.info(f'{key} saved to bucket {bucket_name}')
    
    return {
     "statusCode": 200,
     "body": f"Logging works, check s3 for the file at  {key} in {bucket_name}!"
     # 'xfromlambda1': l1_x,
     # 'yfromlambda1': l1_y,
     # 'zfromlambda1': l1_z
 }
      