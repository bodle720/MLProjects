#!/usr/bin/env python3
import aws_cdk as cdk

from stacks.storage_stack import StorageStack
from stacks.upload_stack import ImageUploadStack

app = cdk.App()

storage = StorageStack(app, "CvdmsStorageStack")

ImageUploadStack(
    app, "CvdmsImageUploadStack",
    file_bucket=storage.file_bucket,
    iceberg_bucket=storage.iceberg_bucket,
    job_table=storage.job_table,
    sha256_table=storage.sha256_table,
    lock_table=storage.lock_table,
    global_dlq=storage.global_dlq,
    athena_database=storage.athena_database,   # <-- pass it in
)


app.synth()