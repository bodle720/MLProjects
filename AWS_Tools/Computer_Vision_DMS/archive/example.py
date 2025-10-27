# -*- coding: utf-8 -*-
"""
API Example
"""

from cv_platform.client import CVPlatformClient

client = CVPlatformClient(
    bucket_name="my-datalake-bucket",
    splitter_job_def="splitter:1",
    job_queue="cpu-job-queue"
)

# Upload an image
s3_uri = client.upload_image("dataset-01", "local_image.jpg")

# Generate manifests
job_id = client.generate_manifests("dataset-01", "classification")

# Query imagery metadata
qid = client.query_imagery("SELECT * FROM cv_datalake.imagery_metadata LIMIT 10")
print("Athena query started:", qid)


