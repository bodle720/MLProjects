#!/usr/bin/env python3
import aws_cdk as cdk
import os
import sys

from config import CONFIG
from stacks.logging_stack import LoggingStack
from stacks.storage_stack import StorageStack
# from stacks.upload_stack import ImageUploadStack

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
                       env=env)

# ImageUploadStack(
#     app,
#     f"{APP_NAME}-UploadStack",
#     app_name=APP_NAME,
#     file_bucket=storage.file_bucket,
#     iceberg_bucket=storage.iceberg_bucket,
#     job_table=storage.job_table,
#     sha256_table=storage.sha256_table,
#     phash_table=storage.phash_table,
#     lock_table=storage.lock_table,
#     global_dlq=storage.global_dlq,
#     athena_database_name=storage.athena_database_name,
#     app_log_group = storage.app_log_group,
#     upload_events_queue =storage.upload_events_queue,
#     env=env
# )

app.synth()