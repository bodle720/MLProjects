# -*- coding: utf-8 -*-

"""
Main runner script: 

1. Builds a Docker image and pushes it to ECR.  
2. Launches three EC2 instances.  
3. Runs the container on each instance with a different integer argument and an S3 bucket name.  
4. The worker function inside the container:
   - Accepts an integer and the bucket name.  
   - Doubles the integer (e.g., 101 -> 202, 202 -> 404).  
   - Saves the result as a Parquet file to S3 under 
     ec2_results/output_{input_value}.parquet.  

5. After all tasks complete, the script:
   - Terminates the EC2 instances.  
   - (Optionally) downloads and prints the Parquet outputs.  

Prerequisites:
- AWS credentials configured  
- Docker & AWS CLI installed  

Placeholders to replace before running:
- `<BUCKET_NAME>`
- `<SUBNET_ID>`
- `<SECURITY_GROUP_ID>`
- `<INSTANCE_PROFILE_NAME>`
"""

#%% Imports and definitions.

import time
import textwrap
import boto3
import subprocess
import pandas as pd
from botocore.exceptions import ClientError
from io import BytesIO

region = 'us-east-1' # Change to your region if necessary.
repo_name = 'ec2-worker-example' # Name what you want. This is the base name of the Docker image we will create.

BUCKET_NAME = '' # The S3 bucket where output will be stored. A folder called 'ec2_results' will be created in this bucket, so make sure it doesn't already exist.
SUBNET_ID = '' # Where the server runs, implies the VPC.
SECURITY_GROUP_ID = '' # Firewall rules for the EC2 server itself.
INSTANCE_PROFILE_NAME = '' # The AWS instance profile name that contains S3 and SSM permissions.

#%% Define the image URI on AWS ECR and connect to ECR.

account_id = boto3.client('sts').get_caller_identity()['Account']
image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"

ecr = boto3.client('ecr', region_name=region)

#%% Create the image name (repository) in ECR.

try:
    ecr.create_repository(repositoryName=repo_name)
except ecr.exceptions.RepositoryAlreadyExistsException:
    print(f'Repo {repo_name} already exists.')

#%% Grab AWS password for the currently logged in account and pass into docker login command to log into ECR.
# Subprocess should run in current Python working directory. If not, you can use cwd arg for .run() command.

result = subprocess.run(f"aws ecr get-login-password --region {region} | \
docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com", shell=True, capture_output=True, text=True)

print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

#%% Build the docker image and give it the name given by repo_name. This stores it locally in private repo.

result = subprocess.run(f"docker build -t {repo_name} .", shell=True, capture_output=True, text=True)

print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

#%% Rename the image to the AWS URI (makes a new reference to the image, but with appropriate name to push to ECR).

result = subprocess.run(f"docker tag {repo_name} {image_uri}", shell=True, capture_output=True, text=True)

print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

#%% Push the local copy to ECR for later use.

result = subprocess.run(f"docker push {image_uri}", shell=True, capture_output=True, text=True)

print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

#%% Define helper funtions to make sure instances are managed and pass all checks. Optional.

ssm = boto3.client('ssm', region_name=region)

def wait_until_managed(instance_id, timeout=600, interval=10):
    print(f"\tWaiting for {instance_id} to register with SSM...")

    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = ssm.describe_instance_information(
            Filters=[
                {
                    "Key": "InstanceIds",
                    "Values": [instance_id]
                }
            ]
        )

        if resp.get('InstanceInformationList'):
            print("\tSuccess.")
            return True

        print(f"\tWaiting {interval} seconds to check again…")
        time.sleep(interval)

    raise TimeoutError(f"\t{instance_id} never showed up as managed")

def wait_for_status_ok(instance_id, timeout=600, interval=10):
    print(f"\tWaiting for {instance_id} to pass checks...")

    ec2 = boto3.client("ec2", region_name=region)
    deadline = time.time() + timeout

    while time.time() < deadline:
        resp = ec2.describe_instance_status(
            InstanceIds=[instance_id],
            IncludeAllInstances=True
        )
        statuses = resp.get("InstanceStatuses", [])
        if statuses:
            s = statuses[0]
            if s["SystemStatus"]["Status"] == "ok" and s["InstanceStatus"]["Status"] == "ok":
                print("\tSuccess.")
                return True
        
        print(f"\tWaiting {interval} seconds to check again…")
        time.sleep(interval)

    raise TimeoutError(f"\t{instance_id} status checks never passed")
    
#%% Configure and launch EC2 instances. SubnetId and SecurityGroupIds will imply
# which VPC to use and what rules to control to/from traffic on the server, respectively.

ec2 = boto3.resource('ec2', region_name=region)

# UserData indicates one-time boot-time setup instructions for the instance.
user_data = textwrap.dedent('''\
#!/bin/bash
yum update -y

if command -v amazon-linux-extras &> /dev/null; then
  amazon-linux-extras install docker -y
else
  dnf install docker -y
fi

systemctl enable docker amazon-ssm-agent
systemctl start  docker amazon-ssm-agent
''')

inputs = ['101', '202', '303']  # Example inputs that will be passed to each server for Python worker to accept.
instances = []
for input_value in inputs:
    print(f'Creating instance with input value: {input_value}')
    
    # Make the instance and run it.
    instance = ec2.create_instances(
        ImageId='ami-00ca32bbc84273381',  # Amazon Linux 2 AMI in the us-east-1 region (as of 9/2025). Yours may differ.
        InstanceType='t2.micro', # CPU architecture to use.
        MinCount=1, # Create at least one instance.
        MaxCount=1, # Create at most one instance.
        SubnetId = SUBNET_ID, 
        SecurityGroupIds=[SECURITY_GROUP_ID],
        IamInstanceProfile={'Name': INSTANCE_PROFILE_NAME},
        UserData=user_data
    )[0]
    
    instance.wait_until_running()
    instance.reload()
    wait_until_managed(instance.id)

    instances.append((instance.id, input_value))

#%% Let all checks pass before continuing.

print('Waiting for checks to pass.')
for instance_id, _ in instances:
    print(f'On instance id = {instance_id}')
    wait_for_status_ok(instance_id)

#%% Now, actually run the container on the three running servers using AWS Systems Manager (previously SSM).

# Send the tasks to each server/EC2 instance.
for instance_id, input_value in instances:
    print('Launching container on instance: ', instance_id)
    
    # Indicate commands needed after image is registered.
    commands = [            
              # Authenticate with ECR (no sudo).
              f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com",
            
              # Pull image and run it.
              f"docker pull {image_uri}",
              f"docker run --rm {image_uri} {input_value} {BUCKET_NAME}"
            ]
    ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": commands}
    )

#%% Give sufficient time to complete tasks, with a max of 200 seconds.
          
s3 = boto3.client('s3')
st_time = time.time()

while (time.time() - st_time) < 200:
    found = 0
    for input_value in inputs:
        try:
            s3.head_object(Bucket=BUCKET_NAME, Key=f"ec2_results/output_{input_value}.parquet")
            print(f"Output file found for input {input_value}")
            found += 1
        except s3.exceptions.ClientError:
            print(f"Output not yet found for input {input_value}")
    
    if found == len(inputs):
        print("All outputs obtained from servers!")
        break
    else:
        print("Waiting 10 seconds for task completion...")
        time.sleep(10)
else:
    print("Timeout reached. Some outputs may be missing.")

#%% Terminate (not just stop) the instances. This deletes them.

instance_ids = [instance_id for instance_id, _ in instances]
ec2_client = boto3.client('ec2', region_name=region)
ec2_client.terminate_instances(InstanceIds=instance_ids)

print(f"Terminated instances: {instance_ids}")

#%% Let's verify results are in S3.

for input_value in inputs:
    object_key = f"ec2_results/output_{input_value}.parquet"
    
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
        parquet_bytes = BytesIO(response['Body'].read())
        df = pd.read_parquet(parquet_bytes, engine='pyarrow')
    except ClientError as e:
        print(f"Failed to download Parquet from S3: {e}")
    except Exception as e:
        print(f"Unexpected error while reading Parquet: {e}")
    else:    
        print(f'Result for input value = {input_value}')
        print(df)