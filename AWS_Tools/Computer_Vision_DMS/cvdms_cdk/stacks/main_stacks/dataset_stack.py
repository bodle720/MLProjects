import uuid
from constructs import Construct
from stacks.helper_constructs.dlq_ops import DLQOps

from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_lambda_event_sources as event_sources,
    aws_iam as iam,
    aws_sqs as sqs,
    aws_dynamodb as dynamodb,
    aws_kinesisfirehose as firehose,
    aws_s3_notifications as s3n,
    aws_ssm as ssm
)

from config import CONFIG
from config_models import LambdaConfig

class DatasetStack(Stack):
    def __init__(self,
                scope: Construct,
                construct_id: str,
                *,
                app_name: str,
                common_utils_layer: _lambda.LayerVersion,
                file_bucket: s3.Bucket,
                datasets_bucket: s3.Bucket,
                iceberg_bucket: s3.Bucket,
                job_table: dynamodb.Table,
                datasets_table: dynamodb.Table,
                dataset_versions_table: dynamodb.Table,
                lock_table: dynamodb.Table,
                iceberg_database_name: str,
                firehose_delivery_stream: firehose.CfnDeliveryStream,
                **kwargs) -> None:

        super().__init__(scope, construct_id, **kwargs)

        # Variables from Storage/Logging stack and app name.
        self.app_name = app_name
        self.common_utils_layer = common_utils_layer

        self.file_bucket = file_bucket
        self.datasets_bucket = datasets_bucket
        self.iceberg_bucket = iceberg_bucket

        self.job_table = job_table
        self.datasets_table = datasets_table
        self.dataset_versions_table = dataset_versions_table
        self.lock_table = lock_table

        self.iceberg_database_name = iceberg_database_name
        self.firehose_delivery_stream = firehose_delivery_stream

        # Make the SQS Queue that will receive dataset events
        self.dataset_events_queue = self.make_dataset_events_queue()

        # Make the dlq
        self.dlq = self.make_dlq_assign_permissions()

        # Make cleanup lambda to run once entire dataset job is done.
        cleanup_task = self._make_cleanup_task(CONFIG.dataset.cleanup_lambda)

        workflow_definition = None
        # sfn.Chain.start(my_first_task) \
        #     .next(cleanup_task)

        dataset_state_machine = sfn.StateMachine(self, "DatasetStateMachine",
                                                definition_body=sfn.DefinitionBody.from_chainable(workflow_definition),
                                                timeout=Duration.hours(
                                                    CONFIG.dataset.dataset_state_machine.duration_hours)
                                                )

        dataset_state_machine.role.add_to_principal_policy(
            iam.PolicyStatement(actions=["sqs:SendMessage"], resources=[self.dlq.queue_arn]))

        dataset_state_machine.apply_removal_policy(RemovalPolicy.DESTROY)

        # Make kickoff lambda to trigger on submission.json
        self._make_kickoff_lambda(dataset_state_machine, CONFIG.dataset.kickoff_lambda)

    def make_dataset_events_queue(self):
        dataset_events_dlq = sqs.Queue(
            self, "DatasetEventsDLQ",
            retention_period=Duration.days(14)
        )

        dataset_events_queue = sqs.Queue(
            self, "DatasetEventsQueue",
            visibility_timeout=Duration.minutes(CONFIG.dataset.events_queue.visibility_timeout_minutes),
            retention_period=Duration.days(CONFIG.dataset.events_queue.retention_period_days),
            dead_letter_queue=sqs.DeadLetterQueue(
                queue=dataset_events_dlq,
                max_receive_count=1
            )
        )

        self.file_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.SqsDestination(dataset_events_queue),
            s3.NotificationKeyFilter(prefix="temp/dataset-ops/", suffix="/submission.json")
        )

        # Add it to SSM
        ssm.StringParameter(self, "DatasetEventsQueueNameParam",
                            parameter_name=f"/cvdms/{self.app_name}/dataset/dataset_events_queue_name",
                            string_value=dataset_events_queue.queue_name)

        return dataset_events_queue

    def make_dlq_assign_permissions(self):
        dlq_processor_env_vars = {
            "JOB_TABLE_NAME": self.job_table.table_name,
            "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
            "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
            "LOCK_TABLE_NAME": self.lock_table.table_name,
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DATABASE_NAME": self.iceberg_database_name,
            "ATHENA_OUTPUT_S3": f"s3://{self.file_bucket.bucket_name}/athena-results/"
        }

        dlq_out = DLQOps(self,
                         "dataset_dlq",
                         name="dataset",
                         app_name=self.app_name,
                         dlq_processor_env_vars=dlq_processor_env_vars,
                         region=self.region,
                         account=self.account,
                         dlq_ops_config=CONFIG.dataset.dlq_ops,
                         iceberg_database_name=self.iceberg_database_name,
                         common_utils_layer=self.common_utils_layer,
                         file_bucket=self.file_bucket,
                         firehose_delivery_stream=self.firehose_delivery_stream)

        dlq = dlq_out.dlq
        dlq_processor = dlq_out.dlq_processor

        # Assign proper permissions
        self.lock_table.grant_read_write_data(dlq_processor)
        self.job_table.grant_read_write_data(dlq_processor)

        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:DeleteObject"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/temp/dataset-ops/*"]
        ))

        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
            conditions={"StringLike": {"s3:prefix": ["temp/dataset-ops/*"]}}
        ))

        return dlq

    def _make_dlq_chain(self) -> sfn.Chain:
        suffix = uuid.uuid4().hex[:8]

        make_dlq_message = sfn.Pass(
            self, f"MakeDLQMessage_{suffix}",
            parameters={
                "source": "stepfunctions",
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "event_type.$": "$.event_type",
                "error.$": "States.JsonToString($.errorInfo)",
            },
            result_path="$.dlqMessage"
        )

        send_to_dlq = tasks.CallAwsService(
            self, f"SendToDLQ_{suffix}",
            service="sqs",
            action="sendMessage",
            parameters={
                "QueueUrl": self.dlq.queue_url,
                "MessageBody.$": "States.JsonToString($.dlqMessage)"
            },
            iam_resources=[self.dlq.queue_arn],
        )

        fail = sfn.Fail(self, f"WorkflowFailed_{suffix}", cause="SentToDatasetDLQ", error="WorkflowError")

        return sfn.Chain.start(make_dlq_message).next(send_to_dlq).next(fail)

    def _make_cleanup_task(self,
                           cleanup_config: LambdaConfig):
        cleanup_lambda = _lambda.Function(
            self,
            "CleanupLambdaDataset",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=cleanup_config.handler,
            code=_lambda.Code.from_asset(cleanup_config.path),
            layers=[self.common_utils_layer],
            memory_size=cleanup_config.memory_size,
            timeout=Duration.seconds(cleanup_config.timeout_sec),
            environment={
                "JOB_TABLE_NAME": self.job_table.table_name,
                "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
                "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
                "LOCK_TABLE_NAME": self.lock_table.table_name,
                "ATHENA_WORKGROUP": "primary"
            }
        )

        # 1) DynamoDB
        self.lock_table.grant_read_write_data(cleanup_lambda)
        self.job_table.grant_read_write_data(cleanup_lambda)

        # 2) S3: delete temp files under temp/dataset-ops/ and read them
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:DeleteObject"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/temp/dataset-ops/*"]
        ))
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
            conditions={"StringLike": {"s3:prefix": ["temp/dataset-ops/*"]}}
        ))

        # 3) S3: Athena results write only to athena-results/
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:PutObject"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/athena-results/*"]
        ))
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"]
        ))

        # 4) Athena: start and poll queries in the workgroup
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults"],
            resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"]
        ))

        # 5) Firehose logging
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[self.firehose_delivery_stream.attr_arn]
        ))

        # 6) Glue metadata read (catalog, DB, and tables)
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
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

        cleanup_task = tasks.LambdaInvoke(
            self, "CleanupTaskDataset",
            lambda_function=cleanup_lambda,
            result_path="$.cleanup",
            output_path="$",
            payload=sfn.TaskInput.from_object({
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "event_type.$": "$.event_type"
            }),
            payload_response_only=True)

        cleanup_task.add_retry(backoff_rate=2.0, max_attempts=2, interval=Duration.seconds(2))

        cleanup_task.add_catch(
            handler=self._make_dlq_chain(),  # fresh chain instance
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        return cleanup_task

    def _make_kickoff_lambda(self,
                             dataset_state_machine,
                             kickoff_config: LambdaConfig):
        # Make Kickoff lambda
        kickoff_lambda = _lambda.Function(
            self,
            "KickoffLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=kickoff_config.handler,
            code=_lambda.Code.from_asset(kickoff_config.path),
            layers=[self.common_utils_layer],
            memory_size=kickoff_config.memory_size,
            timeout=Duration.seconds(kickoff_config.timeout_sec),
            environment={
                "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
                "LOCK_TABLE_NAME": self.lock_table.table_name,
                "DATASET_STATE_MACHINE_ARN": dataset_state_machine.state_machine_arn,
                "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
                "DATASET_DLQ_URL": self.dlq.queue_url
            }
        )

        dataset_state_machine.grant_start_execution(kickoff_lambda)

        # Permissions for the kickoff lambda
        self.job_table.grant_read_write_data(kickoff_lambda)
        self.file_bucket.grant_read(kickoff_lambda)
        self.lock_table.grant_read_data(kickoff_lambda)

        # ensure S3 bucket-level list and get-location are permitted
        kickoff_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
        ))

        # explicitly allow GetObject on the athena-results prefix only if you will read it;
        # otherwise GetObject on whole bucket is already covered by grant_read above.
        kickoff_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/*"],
        ))

        kickoff_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[self.firehose_delivery_stream.attr_arn],  # use ARN provided by the L1
        ))

        # Trigger: S3 event for job.json, add the queue as an event source
        kickoff_lambda.add_event_source(event_sources.SqsEventSource(self.dataset_events_queue, batch_size=1))
        self.dataset_events_queue.grant_consume_messages(kickoff_lambda)