# EC2 Docker Worker Walkthrough

An end-to-end, educational Python demo that shows how to build a Docker image locally, push it to ECR, spin up EC2 instances, and run containerized worker tasks on those instances via AWS Systems Manager (SSM)—all without manually copying files to the servers. It is intended to be a framework that is easily modified for your own needs.
This walkthrough is provided **for educational purposes only**. Use at your own risk. It is intended to be run in an IDE from cell to cell.

---

## Description

This repository contains:

- **runner.py**  
  A script that  
  1. Builds a Docker image and tags it for ECR.  
  2. Pushes the image to an ECR repository.  
  3. Launches three EC2 instances with UserData bootstrap.  
  4. Uses SSM `SendCommand` to pull and run the container on each instance, passing an integer and bucket name.  
  5. Polls S3 for Parquet outputs under `ec2_results/output_{input_value}.parquet`.  
  6. Terminates all EC2 instances and optionally downloads results.

- **helpers.py**  
  A module with `upload_df_or_dict_as_parquet_to_s3()`, converting dicts/DataFrames into Parquet and uploading to S3.

- **worker_task.py**  
  A simple container entrypoint that  
  1. Reads two arguments (`input_value`, `bucket_name`).  
  2. Doubles the integer.  
  3. Uploads the result as a Pandas DataFrame (as a Parquet file) via the helper.

- **Dockerfile**  
  Builds a slim Python image, installs requirements, and sets `worker_task.py` as the ENTRYPOINT.

---

## Prerequisites

- AWS account with an IAM user or role configured locally (`~/.aws/credentials` or env vars).  
- S3 bucket created for outputs (prefix: `ec2_results/`).  
- IAM Role for EC2 with **SSM** and **S3** permissions, attached via an Instance Profile.  
- Docker and AWS CLI installed locally.  
- Python 3.8+ environment

### Security Group Requirements

The security group attached to your EC2 instances must allow outbound traffic on:

- **TCP 443** – HTTPS calls to Amazon ECR (pull/push images) and S3 (GET/PUT objects)  
- **UDP and TCP 53** – DNS resolution so the instance can resolve ECR/S3 hostnames  

Because AWS Security Groups are stateful, you only need to specify these outbound rules.  Return traffic is permitted automatically.

## Configuration

At the top of `runner.py`, set:

```python
region                = 'us-east-1'              
repo_name             = 'ec2-worker-example'     
BUCKET_NAME           = '<YOUR_BUCKET_NAME>'     
SUBNET_ID             = '<YOUR_SUBNET_ID>'       
SECURITY_GROUP_ID     = '<YOUR_SECURITY_GROUP_ID>'
INSTANCE_PROFILE_NAME = '<YOUR_INSTANCE_PROFILE_NAME>'
