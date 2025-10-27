# -*- coding: utf-8 -*-
"""
Core Stack
"""

# stacks/core_stack.py
from aws_cdk import Stack, Duration, RemovalPolicy
from constructs import Construct
from aws_cdk.aws_s3 import Bucket, BucketEncryption, BlockPublicAccess
from aws_cdk.aws_dynamodb import Table, BillingMode, Attribute, AttributeType
from aws_cdk.aws_sqs import Queue, DeadLetterQueue
from aws_cdk.aws_logs import LogGroup, RetentionDays
from aws_cdk import aws_ssm as ssm   # <-- add this import

class CoreStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        regular_bucket = Bucket(self, "RegularBucket",
            versioned=True,
            block_public_access=BlockPublicAccess.BLOCK_ALL,
            encryption=BucketEncryption.S3_MANAGED,
            lifecycle_rules=[{
                "id": "TempImagesCleanup",
                "prefix": "temp-images/",
                "expiration": Duration.days(7)
            }],
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        datalake_bucket = Bucket(self, "DataLakeBucket",
            versioned=True,
            block_public_access=BlockPublicAccess.BLOCK_ALL,
            encryption=BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        pipeline_dlq = Queue(self, "PipelineDLQ", retention_period=Duration.days(14))
        ingest_dlq = Queue(self, "IngestDLQ", retention_period=Duration.days(14))
        ingest_queue = Queue(self, "IngestQueue",
            dead_letter_queue=DeadLetterQueue(queue=ingest_dlq, max_receive_count=5),
            visibility_timeout=Duration.minutes(10)
        )

        jobs = Table(self, "JobsTable",
            table_name="jobs",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            partition_key=Attribute(name="job_id", type=AttributeType.STRING),
            removal_policy=RemovalPolicy.DESTROY
        )

        lock = Table(self, "LockTable",
            table_name="lock",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            partition_key=Attribute(name="singleton", type=AttributeType.STRING),
            removal_policy=RemovalPolicy.DESTROY
        )

        datasets = Table(self, "DatasetsTable",
            table_name="datasets",
            billing_mode=BillingMode.PAY_PER_REQUEST,
            partition_key=Attribute(name="dataset_id", type=AttributeType.STRING),
            removal_policy=RemovalPolicy.DESTROY
        )

        central_logs = LogGroup(self, "CentralLogGroup",
            log_group_name="/cv/central",
            retention=RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY
        )

        # Expose resources as attributes
        self.buckets = {"regular": regular_bucket, "datalake": datalake_bucket}
        self.queues = {"ingest": ingest_queue, "ingestDlq": ingest_dlq, "pipelineDlq": pipeline_dlq}
        self.tables = {"jobs": jobs, "lock": lock, "datasets": datasets}
        self.logs = {"central": central_logs}

        # --- NEW: Write key values into SSM Parameter Store ---
        ssm.StringParameter(self, "RegularBucketParam",
            parameter_name="/cv-platform/regular-bucket",
            string_value=regular_bucket.bucket_name
        )

        ssm.StringParameter(self, "DataLakeBucketParam",
            parameter_name="/cv-platform/datalake-bucket",
            string_value=datalake_bucket.bucket_name
        )

        ssm.StringParameter(self, "JobsTableParam",
            parameter_name="/cv-platform/jobs-table",
            string_value=jobs.table_name
        )

        ssm.StringParameter(self, "DatasetsTableParam",
            parameter_name="/cv-platform/datasets-table",
            string_value=datasets.table_name
        )



