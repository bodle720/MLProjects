#!/usr/bin/env python3
# cdk deploy cvdmsv1-LoggingStack cvdmsv1-StorageStack --profile <profile name>
# cdk deploy cvdmsv1-UploadStack cvdmsv1-DatasetStack --profile <profile name>

import os

import aws_cdk as cdk

from config import CONFIG

from stacks.main_stacks.logging_stack import LoggingStack
from stacks.main_stacks.storage_stack import StorageStack
from stacks.main_stacks.upload_stack import UploadStack
from stacks.main_stacks.dataset_stack import DatasetStack

APP_NAME = CONFIG.app_name

app = cdk.App()
env = cdk.Environment(account=os.getenv("CDK_DEFAULT_ACCOUNT"),
                      region=os.getenv("CDK_DEFAULT_REGION"))

logging_stack = LoggingStack(app,
                           f"{APP_NAME}-LoggingStack",
                            app_name=APP_NAME,
                            env=env)

storage_stack = StorageStack(app,
                           f"{APP_NAME}-StorageStack",
                             app_name=APP_NAME,
                             env=env)

upload_stack = UploadStack(app,
                            f"{APP_NAME}-UploadStack",
                            app_name=APP_NAME,
                            file_bucket=storage_stack.file_bucket,
                            iceberg_bucket=storage_stack.iceberg_bucket,
                            job_table=storage_stack.job_table,
                            sha256_table=storage_stack.sha256_table,
                            lock_table=storage_stack.lock_table,
                            iceberg_database_name=storage_stack.iceberg_database_name,
                            firehose_delivery_stream=logging_stack.firehose_delivery_stream,
                            upload_events_queue=storage_stack.upload_events_queue,
                            env=env)

dataset_stack = DatasetStack(app,
                                f"{APP_NAME}-DatasetStack",
                                app_name=APP_NAME,
                                file_bucket=storage_stack.file_bucket,
                                datasets_bucket=storage_stack.datasets_bucket,
                                iceberg_bucket=storage_stack.iceberg_bucket,
                                job_table=storage_stack.job_table,
                                datasets_table=storage_stack.datasets_table,
                                dataset_versions_table=storage_stack.dataset_versions_table,
                                lock_table=storage_stack.lock_table,
                                iceberg_database_name=storage_stack.iceberg_database_name,
                                firehose_delivery_stream=logging_stack.firehose_delivery_stream,
                                dataset_events_queue=storage_stack.dataset_events_queue,
                                env=env)

app.synth()