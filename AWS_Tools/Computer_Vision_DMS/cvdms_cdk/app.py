#!/usr/bin/env python3
import aws_cdk as cdk
import os
import sys

from config import CONFIG
from stacks.logging_stack import LoggingStack
from stacks.storage_stack import StorageStack
from stacks.upload_stack import ImageUploadStack

APP_NAME = CONFIG.app_name

app = cdk.App()
env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION"),
)

logging_stack = LoggingStack(app,
                           f"{APP_NAME}-LoggingStack",
                            app_name=APP_NAME,
                            env=env)

storage_stack = StorageStack(app,
                       f"{APP_NAME}-StorageStack",
                         app_name=APP_NAME,
                         common_utils_layer=logging_stack.common_utils_layer,
                         firehose_delivery_stream=logging_stack.firehose_delivery_stream,
                         env=env)

upload_stack = ImageUploadStack(
    app,
    f"{APP_NAME}-UploadStack",
    app_name=APP_NAME,
    common_utils_layer=logging_stack.common_utils_layer,
    file_bucket=storage_stack.file_bucket,
    iceberg_bucket=storage_stack.iceberg_bucket,
    job_table=storage_stack.job_table,
    sha256_table=storage_stack.sha256_table,
    lock_table=storage_stack.lock_table,
    global_dlq=storage_stack.global_dlq,
    iceberg_database_name=storage_stack.iceberg_database_name,
    upload_events_queue=storage_stack.upload_events_queue,
    firehose_delivery_stream=logging_stack.firehose_delivery_stream,
    env=env
)

app.synth()