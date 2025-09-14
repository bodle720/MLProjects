# -*- coding: utf-8 -*-
"""
This script is meant to help you understand your worker call better in terms of 
memory and CPU utilization. First, we verify the number of output files we expect

Then, we utilize the .sql predefined queries in this 
folder to search for and compile the metrics we desire, tailored to the logs inside
the worker function. In order to do so, we must specify an epoch start and end timestamp.
Instead, I have redefined the problem as simply specifying how many hours looking back from now
we would like our search window to begin and end. This is needed so AWS can filter the
logs we want.

Having a better understanding of CPU and memory utilization for your task is crucial
for resource allocation in the compute environment and when defining the job. This script
helps you retroactively assign a number to it after perhaps running a small subset of the 
workload to get some sample output to work with.

NOTE: This script uses placeholder values and does not expose any sensitive AWS credentials or identifiers.
"""

import analysis_helpers

#%% Set some parameters. 
   
AWS_REGION         = "us-east-1"
LOG_GROUP          = "/your/jobs/log/group/xxxxx" # Where you chose your workers to log  in the register_job_definition call.
MEMORY_QUERY_FILE  = "memory-query.sql"
CPU_QUERY_FILE     = "cpu-query.sql"
start_hrs_ago      = 7 # How many hours back from now do you want your window to start searching in?
end_hrs_ago        = 0 # How many hours back from now do you want your window to end searching in? Must be < start_hrs_ago.

#%% Get the window of time we will search in: from 'start_hrs_ago' hours ago to 'end_hrs_ago' hours ago.

START_TS, END_TS, start, end = analysis_helpers.get_epoch_window(hours_ago_start = start_hrs_ago,
                                                                 hours_ago_end = end_hrs_ago)

print(f"Querying logs from {START_TS} to {END_TS} (epoch seconds), or from {start} to {end}.")

#%% Now let's count how many files were saved from our workers in the bucket we chose and the folder we chose for the output.

bucket = 'your-batch-output-bucket-xxxxx'
prefix = 'batch_out/'  # This includes all subfolders under 'prefix/'
extension = '.parquet' # Each worker saved one parquet file.

saved_files = analysis_helpers.list_s3_files_with_extension(bucket, prefix, extension)
print(f'Num files saved from batch run with .parquet output = {len(saved_files):,}')

#%% Get the Memory Metrics for each completed worker.

mem_metrics = analysis_helpers.run_query_and_wait(LOG_GROUP, START_TS, END_TS, MEMORY_QUERY_FILE, AWS_REGION)
print("--- Memory Metrics ---")
print(f"\tJob Count : {mem_metrics['count']:,} jobs")
print(f"\tAvg Memory Usage : {mem_metrics['avg_memory']:.2f} MiB")
print(f"\tMin Memory Usage : {mem_metrics['min_memory']:.2f} MiB")
print(f"\tMax Memory Usage: {mem_metrics['max_memory']:.2f} MiB\n")

#%% Get the CPU utilization Metrics for each completed worker.

cpu_metrics = analysis_helpers.run_query_and_wait(LOG_GROUP, START_TS, END_TS, CPU_QUERY_FILE, AWS_REGION)
print("--- CPU Percent Utilization Metrics ---")
print(f"\tJob Count : {cpu_metrics['count']:,} jobs")
print(f"\tAvg Peak CPU Usage : {cpu_metrics['avg_peak_cpu']:.2f}%")
print(f"\tMin Peak CPU Usage : {cpu_metrics['min_peak_cpu']:.2f}%")
print(f"\tMax Peak CPU Usage: {cpu_metrics['max_peak_cpu']:.2f}%")
print(f"\tAvg mean CPU Usage : {cpu_metrics['avg_avg_cpu']:.2f}%")
print(f"\tMin mean CPU Usage : {cpu_metrics['min_avg_cpu']:.2f}%")
print(f"\tMax mean CPU Usage: {cpu_metrics['max_avg_cpu']:.2f}%")
