#!/usr/bin/env python3
import aws_cdk as cdk
import boto3
import os
import sys
from stacks.storage_stack import StorageStack
from stacks.upload_stack import ImageUploadStack

APP_NAME = "cvdms"

# Guardrail: fail fast if parameters already exist under this app_name
ssm = boto3.client("ssm")
resp = ssm.get_parameters_by_path(Path=f"/{APP_NAME}/", MaxResults=1)
if resp.get("Parameters"):
    sys.exit(f"Guardrail: SSM parameters already exist under /{APP_NAME}/. "
             f"Pick a different APP_NAME hardcoded in app.py or clean up before deploying.")

app = cdk.App()
env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION"),
)

storage = StorageStack(app,
                       f"{APP_NAME}-storage",
                        app_name=APP_NAME,
                       env=env)

ImageUploadStack(
    app,
    f"{APP_NAME}-upload",
    app_name=APP_NAME,
    file_bucket=storage.file_bucket,
    iceberg_bucket=storage.iceberg_bucket,
    job_table=storage.job_table,
    sha256_table=storage.sha256_table,
    lock_table=storage.lock_table,
    global_dlq=storage.global_dlq,
    athena_database=storage.athena_database,
    app_log_group = storage.app_log_group,
    env=env
)

app.synth()