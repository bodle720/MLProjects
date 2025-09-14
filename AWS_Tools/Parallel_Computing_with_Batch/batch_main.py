# -*- coding: utf-8 -*-
"""
Main AWS Batch runner script. Run block by block.
You can run each block successfully after changing into the Parallel_Computing_with_Batch directory.
"""

#%% Imports.

import json
import math
import boto3
import subprocess
from botocore.exceptions import ClientError

import batch_helpers

#%% Configuration block for AWS Batch job setup.
# Replace placeholder values with your actual resource IDs or use environment variables for automation.

AWS_REGION          = "us-east-1"
ACCOUNT_ID          =  boto3.client("sts").get_caller_identity()["Account"]
ECR_REPO_NAME       = "batch-docker-img"
BATCH_JOB_DEF_NAME  = "my-batch-job-def"
JOB_QUEUE_NAME      = "my-batch-queue"
BATCH_CE_NAME       = "my-batch-ce"
VPC_SUBNETS         = ["subnet-xxxxxxxxxxxxx","subnet-xxxxxxxxxxxx"]
SECURITY_GROUPS     = ["sg-xxxxxxxxxx"]
INSTANCE_ROLE       = f"arn:aws:iam::{ACCOUNT_ID}:instance-profile/YourInstanceRolexxxxxxxx" # Using the default will enable CloudWatch logs.
JOB_DEF_ROLE        = f"arn:aws:iam::{ACCOUNT_ID}:role/YourJobDefinitionRolexxxxxxxx"
S3_BUCKET_NAME      = "your-data-bucket-xxxxx"
ECR_IMAGE_URI       = f"{ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO_NAME}:latest"
AWS_LOGSGROUP       = "/your/custom/logs/group" # Go into CloudWatch and make your logs group this name.
 
#%% Create the image name (repository) in ECR.

ecr = boto3.client('ecr', region_name=AWS_REGION)

try:
    ecr.create_repository(repositoryName=ECR_REPO_NAME)
except ecr.exceptions.RepositoryAlreadyExistsException:
    print(f'ECR Repo {ECR_REPO_NAME} already exists.')

#%% Login to ECR.

# Fetch the ECR login password.
pw_proc = subprocess.run(
                        [
                            "aws", "ecr", "get-login-password",
                            "--region", AWS_REGION
                        ],
                        capture_output=True,
                        text=True,
                        check=True
                        )

password = pw_proc.stdout

# Pass it to docker login via stdin.
ecr_login_res = subprocess.run(
                                [
                                    "docker", "login",
                                    "--username", "AWS",
                                    "--password-stdin",
                                    f"{ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com"
                                ],
                                input=password,
                                capture_output=True,
                                text=True,
                                check=True
                            )

print("STDOUT:\n", ecr_login_res.stdout)
print("STDERR:\n", ecr_login_res.stderr)

#%% Build the docker image and give it the name given by repo_name. This stores it locally in private repo.
# Be sure to cd into my-batch-task folder first.

docker_build_res = subprocess.run(
                                    [
                                        "docker",
                                        "build",
                                        "-t", ECR_REPO_NAME,
                                        "./my-batch-task"
                                    ],
                                    capture_output=True,
                                    text=True
                                    )

print("STDOUT:\n", docker_build_res.stdout)
print("STDERR:\n", docker_build_res.stderr)

#%% Rename the image to the AWS URI (makes a new reference to the image, but with appropriate name to push to ECR).

docker_tag_res = subprocess.run(
                                ["docker", "tag", ECR_REPO_NAME, ECR_IMAGE_URI],
                                capture_output=True,
                                text=True
                                )


print("STDOUT:\n", docker_tag_res.stdout)
print("STDERR:\n", docker_tag_res.stderr)

#%% Push the local copy to ECR for later use.

docker_push_res = subprocess.run(
                                ["docker", "push", ECR_IMAGE_URI],
                                capture_output=True,
                                text=True
                                )


print("STDOUT:\n", docker_push_res.stdout)
print("STDERR:\n", docker_push_res.stderr)

#%% Make the Compute Environment and wait for it to fully spin up.

batch = boto3.client("batch", region_name=AWS_REGION)

try:
    batch.create_compute_environment(
        computeEnvironmentName=BATCH_CE_NAME,
        type="MANAGED",
        state="ENABLED",
        computeResources={
            "type": "SPOT",
            "allocationStrategy":"SPOT_PRICE_CAPACITY_OPTIMIZED",
            "minvCpus": 0, # You should keep this at 0 to prevent idle EC2 instance charging you.
            "maxvCpus": 128, # Depends on account limits. Might need to request a quota increase.
            "desiredvCpus": 128, 
            "instanceTypes": ["c5.large", "c5.xlarge",
                             "c5d.large", "c5d.xlarge",
                             "c6i.large", "c6i.xlarge"],
            # "instanceTypes": ["t3.micro", "t3.small", # These instance types are suitable for low-memory tasks.
            #                  "t3a.micro", "t3a.small",
            #                  "c6g.medium"],
            "subnets": VPC_SUBNETS,
            "securityGroupIds": SECURITY_GROUPS,
            "instanceRole": INSTANCE_ROLE
        },
        serviceRole= "" # Auto assigned, replace with your AWS Batch service role ARN if required.
    )
except ClientError as e:
    code = e.response['Error']['Code']
    message = e.response['Error']['Message']
    if code == 'ClientException' and 'already exists' in message:
        print(f'Compute environment {BATCH_CE_NAME} already exists.')
    else:
        raise

ce_ready = batch_helpers.wait_for_compute_env_valid(batch, BATCH_CE_NAME, timeout=300, interval=20)

if ce_ready:
    print('Compute Environment ready!')
else:
    print('Compute Environment not ready...')
    
#%% Make and attach a Job Queue to the previous Compute Environment and wait for it to show as valid.

try:
    batch.create_job_queue(
        jobQueueName=JOB_QUEUE_NAME,
        state="ENABLED",
        priority=1,
        computeEnvironmentOrder=[{"order": 1, "computeEnvironment": BATCH_CE_NAME}],
        jobQueueType="ECS"
    )
    print(f"Job queue {JOB_QUEUE_NAME} created.")
except ClientError as e:
    code = e.response["Error"]["Code"]
    msg = e.response["Error"]["Message"]
    if code == "ClientException" and "already exists" in msg:
        print(f"Job queue {JOB_QUEUE_NAME} already exists.")
    else:
        raise

jq_ready = batch_helpers.wait_for_queue_valid(batch, JOB_QUEUE_NAME, timeout=300, interval=20)

if jq_ready:
    print('Job Queue ready!')
else:
    print('Job Queue not ready...')
    
#%% Make the job definition, which will define the job that runs and its requirements.

try:
    response = batch.register_job_definition(
                                jobDefinitionName=BATCH_JOB_DEF_NAME,
                                type="container",
                                containerProperties={
                                    "image": ECR_IMAGE_URI,
                                    "vcpus": 1,
                                    "memory": 2048, # The hard limit (in MiB) of memory to present to the container. (see post_batch_analysis/)
                                    "command": [],  # we’ll supply args at submit time
                                    "jobRoleArn": JOB_DEF_ROLE,
                                    "logConfiguration": {
                                    "logDriver": "awslogs",
                                     "options": {
                                        "awslogs-group": AWS_LOGSGROUP,  # You can customize this, but be sure to make the CloudWatch log group first.
                                        "awslogs-region": AWS_REGION,          
                                        "awslogs-stream-prefix": "batch"}  # Prefix for log stream names.
                                         }
                                    }
                                )

                                
    job_def_arn = response["jobDefinitionArn"]
    revision    = response["revision"]
    print(f"Registered job definition {job_def_arn} (revision {revision})")
except ClientError as e:
    code    = e.response["Error"]["Code"]
    message = e.response["Error"]["Message"]
    print(f"Failed to register job definition: {code} – {message}")
    raise Exception(f'Error making job definition: {e}')

#%% Load in the dates and ticker symbols we wish to process features for. You must be in the AWS_Batch directory for this.

with open(r"data/dates.txt", "r") as f:
    dates = [line.strip() for line in f]
    
with open(r"data/symbols.txt", "r") as f:
    symbols = [line.strip() for line in f]

#%% Make a list of jobs to later send to the job queue for completion.

jobs = []
for symb in symbols:
    for dt in dates:
        for tf_int in [1,2,3,4,5]:
            tf_int = str(tf_int)
            jobs.append({
                "bucket":     S3_BUCKET_NAME,
                "input_key":  f"dfs/{symb}.parquet",
                "output_key": f"batch_out/{symb}/{dt}/{tf_int}.parquet",
                "dt":         dt,
                "tf_int":     tf_int,
                "symb":       symb,
            })
    
total_jobs = len(jobs)
print(f"Total of {total_jobs:,} jobs")  

#%% Upload the manifest of jobs for the array job. This contains all unique required task definitions.
# We will upload the manifest in chunks as each worker will need to load the entire manifest in, which will
# quickly use up significant amounts of memory. This way, we split it and manage memory more efficiently
# and keep costs lower.

# We could do the following commented out line of code. In that case, we would need a start index flag for the worker,
# but insteead we will save memory and chunk up the manifest into multiple S3 keys, as described above.
# s3.put_object(Bucket=S3_BUCKET_NAME, Key="manifest/jobs.json", Body=json.dumps(jobs))

CHUNK_SIZE = 10_000
num_chunks = math.ceil(total_jobs / CHUNK_SIZE)
s3 = boto3.client("s3")

for chunk_id in range(num_chunks):
    start_index = chunk_id * CHUNK_SIZE # 0, 10,000; 3,000; ... ; 2360000
    end = start_index + CHUNK_SIZE
    
    # use this for size if not breaking manifest up.
    # size = min(CHUNK_SIZE, total_jobs - start_index) # 10,000; 10,000; 10,000; ... 7,910 (= total_jobs%10000)
    
    chunk_jobs = jobs[start_index:end]
    manifest_key = f"manifests/chunk_{chunk_id}.json"
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=manifest_key, Body=json.dumps(chunk_jobs))

    batch.submit_job(
          jobName=f"worker-chunk-{chunk_id}",
          jobQueue=JOB_QUEUE_NAME,
          jobDefinition=BATCH_JOB_DEF_NAME,
          arrayProperties={"size": len(chunk_jobs)},
          containerOverrides={"command": [
            "--manifest-s3-bucket", S3_BUCKET_NAME,
            "--manifest-key",      manifest_key
          ]} 
        )
    
    print(f"Submitted chunk {chunk_id+1}/{num_chunks} as array size={len(chunk_jobs)}")