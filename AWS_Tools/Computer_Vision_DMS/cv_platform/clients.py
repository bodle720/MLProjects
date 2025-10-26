# -*- coding: utf-8 -*-
"""
Clients
"""

import boto3

class CVPlatformClient:
    def __init__(self, bucket_name, splitter_job_def, job_queue, workgroup="primary"):
        self.s3 = boto3.client("s3")
        self.batch = boto3.client("batch")
        self.athena = boto3.client("athena")
        self.bucket = bucket_name
        self.splitter_job_def = splitter_job_def
        self.job_queue = job_queue
        self.workgroup = workgroup

    def upload_image(self, dataset_id, local_path, key=None):
        key = key or f"raw/{dataset_id}/{local_path.split('/')[-1]}"
        self.s3.upload_file(local_path, self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def generate_manifests(self, dataset_id, task_type):
        resp = self.batch.submit_job(
            jobName=f"split-{dataset_id}",
            jobQueue=self.job_queue,
            jobDefinition=self.splitter_job_def,
            containerOverrides={
                "command": [
                    "python", "/app/split.py",
                    "--dataset_id", dataset_id,
                    "--task_type", task_type,
                    "--bucket", self.bucket
                ]
            }
        )
        return resp["jobId"]

    def query_imagery(self, sql):
        resp = self.athena.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": "cv_datalake"},
            WorkGroup=self.workgroup,
            ResultConfiguration={"OutputLocation": f"s3://{self.bucket}/athena-results/"}
        )
        return resp["QueryExecutionId"]
