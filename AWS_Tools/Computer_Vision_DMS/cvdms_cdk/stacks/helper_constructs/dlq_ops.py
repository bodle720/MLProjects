from constructs import Construct

from aws_cdk import (
    Duration,
    RemovalPolicy,
    aws_iam as iam,
    aws_sqs as sqs,
    aws_lambda as _lambda,
    aws_ssm as ssm,
    aws_lambda_event_sources as event_sources,
    aws_kinesisfirehose as firehose,
    aws_s3 as s3
)

from config_models import DLQOpsConfig

class DLQOps(Construct):
    def __init__(self,
                 scope: Construct,
                 construct_id: str,
                 *,
                 name: str,
                 app_name: str,
                 dlq_processor_env_vars: dict,
                 region: str,
                 account: str,
                 dlq_ops_config: DLQOpsConfig,
                 iceberg_database_name: str,
                 common_utils_layer: _lambda.LayerVersion,
                 file_bucket: s3.Bucket,
                 firehose_delivery_stream: firehose.CfnDeliveryStream):

        super().__init__(scope, construct_id)

        self.name = name
        self.app_name = app_name
        self.dlq_processor_env_vars = dlq_processor_env_vars
        self.region = region
        self.account = account


        # DLQ processor params
        self.dlq_processor_path = dlq_ops_config.dlq_processor.path # from the CONFIG
        self.dlq_processor_handler = dlq_ops_config.dlq_processor.handler # from the CONFIG
        self.dlq_processor_memory_size = dlq_ops_config.dlq_processor.memory_size # from the CONFIG
        self.dlq_processor_timeout_sec = dlq_ops_config.dlq_processor.timeout_sec # from the CONFIG

        # SQS params
        self.dlq_retention_period_days = dlq_ops_config.sqs_queue.retention_period_days
        self.dlq_visibility_timeout_minutes = dlq_ops_config.sqs_queue.visibility_timeout_minutes

        self.iceberg_database_name = iceberg_database_name
        self.common_utils_layer = common_utils_layer

        self.file_bucket = file_bucket
        self.firehose_delivery_stream = firehose_delivery_stream

        # Make the DLQ
        self.dlq = self.make_dlq()

        # Make the DLQ processor
        self.dlq_processor = self.make_dlq_processor()

        # Add the param to SSM
        self.add_to_ssm()

    def make_dlq(self):
        dlq = sqs.Queue(self, f"{self.name}_DeadLetterQueue",
                        retention_period=Duration.days(self.dlq_retention_period_days),
                        visibility_timeout=Duration.minutes(self.dlq_visibility_timeout_minutes),
                        removal_policy=RemovalPolicy.DESTROY)

        return dlq

    def make_dlq_processor(self):
        # Make a lambda that polls the dlq and processes the messages
        dlq_processor = _lambda.Function(
            self,
            f"{self.name}_DLQProcessor",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=self.dlq_processor_handler,
            code=_lambda.Code.from_asset(self.dlq_processor_path),
            layers=[self.common_utils_layer],
            memory_size=self.dlq_processor_memory_size,
            timeout=Duration.seconds(self.dlq_processor_timeout_sec),
            environment=self.dlq_processor_env_vars
        )

        # Assign general permissions
        # 1) S3: Athena results write only to athena-results/
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:PutObject", "s3:GetObject"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/athena-results/*"]
        ))
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"]
        ))

        # 2) Athena: start and poll queries in the workgroup
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "athena:StartQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "athena:StopQueryExecution",
            ],
            resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"]
        ))

        # 3) Firehose logging
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[self.firehose_delivery_stream.attr_arn]
        ))

        # 4) Glue metadata read (catalog, DB, and tables)
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "glue:GetDatabase", "glue:GetDatabases",
                "glue:GetTable", "glue:GetTables",
                "glue:GetPartition", "glue:GetPartitions",
                "glue:GetTableVersion", "glue:GetTableVersions"
            ],
            resources=[
                f"arn:aws:glue:{self.region}:{self.account}:catalog",
                f"arn:aws:glue:{self.region}:{self.account}:database/{self.iceberg_database_name}",
                f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/*"
            ]
        ))

        dlq_processor.add_event_source(event_sources.SqsEventSource(self.dlq, batch_size=10))
        self.dlq.grant_consume_messages(dlq_processor)

        return dlq_processor

    def add_to_ssm(self):
        ssm.StringParameter(self, f"{self.name}_DlqNameParam",
                            parameter_name=f"/cvdms/{self.app_name}/{self.name}/{self.name}_dlq_name",
                            string_value=self.dlq.queue_name
                            )