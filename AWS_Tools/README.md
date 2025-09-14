# AWS Python Tutorial Suite

This directory contains Python applications leveraging various AWS services. Some serve both as step-by-step tutorials and as frameworks you can adapt for your own needs—swap in your own logic, buckets, and VPC settings in place of the existing examples.

## Projects

1. **Running_Workers_on_EC2_using_ECR**  
   A comprehensive walkthrough demonstrating how to:
   - Build a Docker image containing a Python worker.  
   - Push that image to Amazon ECR.  
   - Launch EC2 instances with a UserData bootstrap script.  
   - Use AWS Systems Manager (SSM) to pull and run the Docker container on each instance.  
   - Save each worker’s output as a Parquet file under `ec2_results/output_{input_value}.parquet` in S3.  
   - Terminate the EC2 instances automatically once tasks complete.  

   Use this code as a template for running containerized workloads on EC2 without manual SSH or SCP.

2. **S3_Functionalities**  
   A focused collection of scripts showing how to:
   - Upload and download various object types (JSON, CSV, Parquet, etc.) to and from Amazon S3.  
   - Normalize Python dictionaries into pandas DataFrames.  
   - Write DataFrames to Parquet and store them in S3.  
   - Retrieve objects and load them back into your Python environment.  

   Includes reusable helper functions (`helpers.py`) for common S3 operations.

3. **Parallel_Computing_with_Batch**  
   A scalable template for parallel workloads using **AWS Batch**, originally built for stock feature generation but easily adaptable to other domains.
   
   - Programmatically configure **AWS Batch** via `boto3`, including Job Queues, Job Definitions, and Compute Environments.  
   - Launch thousands of jobs in parallel to process financial data or any other repeatable task.  
   - Read and write data to **Amazon S3** for downstream model training, backtesting, or archival.  
   - Monitor and analyze worker performance—track **CPU and memory usage** to optimize job definitions and compute resource allocation.



