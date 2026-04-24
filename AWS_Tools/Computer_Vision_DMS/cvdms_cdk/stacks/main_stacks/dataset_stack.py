import uuid
import json
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
    aws_ssm as ssm,
    aws_dynamodb as dynamodb,
    aws_kinesisfirehose as firehose
)

from config import CONFIG
from config_models import LambdaConfig

class DatasetStack(Stack):
    def __init__(self,
                 scope: Construct,
                 construct_id: str,
                 *,
                 app_name: str,
                 file_bucket: s3.Bucket,
                 datasets_bucket: s3.Bucket,
                 iceberg_bucket: s3.Bucket,
                 job_table: dynamodb.Table,
                 datasets_table: dynamodb.Table,
                 dataset_versions_table: dynamodb.Table,
                 lock_table: dynamodb.Table,
                 iceberg_database_name: str,
                 firehose_delivery_stream: firehose.CfnDeliveryStream,
                 dataset_events_queue: sqs.Queue,
                 **kwargs) -> None:

        super().__init__(scope, construct_id, **kwargs)

        # Shared resources / config
        self.app_name = app_name
        self.common_utils_layer = self.make_common_utils_layer()

        self.file_bucket = file_bucket
        self.datasets_bucket = datasets_bucket
        self.iceberg_bucket = iceberg_bucket

        self.job_table = job_table
        self.datasets_table = datasets_table
        self.dataset_versions_table = dataset_versions_table
        self.lock_table = lock_table

        self.iceberg_database_name = iceberg_database_name
        self.firehose_delivery_stream = firehose_delivery_stream

        # SQS queue receiving dataset submission events
        self.dataset_events_queue = dataset_events_queue

        # Make a SSM testing param that various lambdas in the dataset flow can call on to test the
        # dlq processor. Params can be manually changed at will in the console in SSM.
        self.dataset_testing_ssm_param_name = f"/cvdms/{app_name}/dataset/testing/fail_control"
        self.make_ssm_testing_fail_control_param()

        # Workflow DLQ + processor
        self.dlq = self.make_dlq_assign_permissions()

        # Step Functions tasks
        create_task = self._make_dataset_lambda_task(
            task_id="CreateDatasetTask",
            lambda_id="CreateDatasetLambda",
            lambda_config=CONFIG.dataset.create_lambda,
            stage_name="create_task",
            dlq_policy="rollback_new_version"
        )

        update_task = self._make_dataset_lambda_task(
            task_id="UpdateDatasetTask",
            lambda_id="UpdateDatasetLambda",
            lambda_config=CONFIG.dataset.update_lambda,
            stage_name="update_task",
            dlq_policy="rollback_new_version"
        )

        delete_task = self._make_dataset_lambda_task(
            task_id="DeleteDatasetTask",
            lambda_id="DeleteDatasetLambda",
            lambda_config=CONFIG.dataset.delete_lambda,
            stage_name="delete_task",
            dlq_policy="complete_delete"
        )

        visualization_task = self._make_dataset_lambda_task(
            task_id="GenerateVisualizationTask",
            lambda_id="GenerateVisualizationLambda",
            lambda_config=CONFIG.dataset.visualization_lambda,
            stage_name="visualization_task",
            dlq_policy="rollback_new_version"
        )

        cleanup_task = self._make_cleanup_task(CONFIG.dataset.cleanup_lambda)

        invalid_task_type = sfn.Fail(
            self,
            "InvalidDatasetTaskType",
            cause="Unsupported task_type in dataset submission",
            error="InvalidTaskType"
        )

        visualization_task.next(cleanup_task)

        workflow_definition = sfn.Chain.start(
            sfn.Choice(self, "RouteDatasetTaskType")
            .when(
                sfn.Condition.string_equals("$.task_type", "create_dataset"),
                create_task.next(visualization_task)
            )
            .when(
                sfn.Condition.string_equals("$.task_type", "update_dataset"),
                update_task.next(visualization_task)
            )
            .when(
                sfn.Condition.string_equals("$.task_type", "delete_dataset"),
                delete_task.next(cleanup_task)
            )
            .otherwise(invalid_task_type)
        )

        dataset_state_machine = sfn.StateMachine(
            self,
            "DatasetStateMachine",
            definition_body=sfn.DefinitionBody.from_chainable(workflow_definition),
            timeout=Duration.hours(CONFIG.dataset.dataset_state_machine.duration_hours)
        )

        dataset_state_machine.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=["sqs:SendMessage"],
                resources=[self.dlq.queue_arn]
            )
        )

        dataset_state_machine.apply_removal_policy(RemovalPolicy.DESTROY)

        # Kickoff lambda starts the workflow from SQS messages
        self._make_kickoff_lambda(dataset_state_machine, CONFIG.dataset.kickoff_lambda)

    def make_ssm_testing_fail_control_param(self):
        ssm.StringParameter(
            self,
            "DatasetTestingFailControlParam",
            parameter_name=self.dataset_testing_ssm_param_name,
            string_value=json.dumps(
                {
                    "enabled": False,
                    "failpoint_name": None
                }
            ),
        )

    def make_dlq_assign_permissions(self):
        dlq_processor_env_vars = {
            "JOB_TABLE_NAME": self.job_table.table_name,
            "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
            "DATASETS_BUCKET_NAME": self.datasets_bucket.bucket_name,
            "ICEBERG_BUCKET_NAME": self.iceberg_bucket.bucket_name,
            "DATASETS_TABLE_NAME": self.datasets_table.table_name,
            "DATASET_VERSIONS_TABLE_NAME": self.dataset_versions_table.table_name,
            "LOCK_TABLE_NAME": self.lock_table.table_name,
            "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DATABASE_NAME": self.iceberg_database_name,
            "ATHENA_OUTPUT_S3": f"s3://{self.file_bucket.bucket_name}/athena-results/"
        }

        dlq_out = DLQOps(
            self,
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
            firehose_delivery_stream=self.firehose_delivery_stream
        )

        dlq = dlq_out.dlq
        dlq_processor = dlq_out.dlq_processor

        # DynamoDB
        self.lock_table.grant_read_write_data(dlq_processor)
        self.job_table.grant_read_write_data(dlq_processor)
        self.datasets_table.grant_read_write_data(dlq_processor)
        self.dataset_versions_table.grant_read_write_data(dlq_processor)

        # File bucket temp dataset submissions
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:DeleteObject"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/temp/dataset-ops/*"]
        ))
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
            conditions={"StringLike": {"s3:prefix": ["temp/dataset-ops/*"]}}
        ))

        # Dataset artifacts bucket
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:DeleteObject", "s3:PutObject"],
            resources=[f"arn:aws:s3:::{self.datasets_bucket.bucket_name}/datasets/*"]
        ))
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[f"arn:aws:s3:::{self.datasets_bucket.bucket_name}"],
            conditions={"StringLike": {"s3:prefix": ["datasets/*"]}}
        ))

        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
            resources=[f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}/*"]
        ))
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}"]
        ))

        return dlq

    def _make_dlq_chain(self, *, failed_stage: str, dlq_policy: str) -> sfn.Chain:
        suffix = uuid.uuid4().hex[:8]

        make_dlq_message = sfn.Pass(
            self,
            f"MakeDLQMessage_{suffix}",
            parameters={
                "source": "stepfunctions",
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "event_type.$": "$.event_type",
                "task_type.$": "$.task_type",
                "dataset_context.$": "$.dataset_context",
                "error.$": "States.JsonToString($.errorInfo)",
                "failed_stage": failed_stage,
                "dlq_policy": dlq_policy,
            },
            result_path="$.dlqMessage"
        )

        send_to_dlq = tasks.CallAwsService(
            self,
            f"SendToDLQ_{suffix}",
            service="sqs",
            action="sendMessage",
            parameters={
                "QueueUrl": self.dlq.queue_url,
                "MessageBody.$": "States.JsonToString($.dlqMessage)"
            },
            iam_resources=[self.dlq.queue_arn],
        )

        fail = sfn.Fail(
            self,
            f"WorkflowFailed_{suffix}",
            cause="SentToDatasetDLQ",
            error="WorkflowError"
        )

        return sfn.Chain.start(make_dlq_message).next(send_to_dlq).next(fail)

    def _dataset_lambda_env(self) -> dict[str, str]:
        return {
            "JOB_TABLE_NAME": self.job_table.table_name,
            "DATASETS_TABLE_NAME": self.datasets_table.table_name,
            "DATASET_VERSIONS_TABLE_NAME": self.dataset_versions_table.table_name,
            "LOCK_TABLE_NAME": self.lock_table.table_name,
            "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
            "DATASETS_BUCKET_NAME": self.datasets_bucket.bucket_name,
            "ICEBERG_BUCKET_NAME": self.iceberg_bucket.bucket_name,
            "ICEBERG_DATABASE_NAME": self.iceberg_database_name,
            "ATHENA_WORKGROUP": "primary",
            "ATHENA_OUTPUT_S3": f"s3://{self.file_bucket.bucket_name}/athena-results/",
            "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
            "DATASET_TESTING_SSM_PARAM_NAME": self.dataset_testing_ssm_param_name
        }

    def _grant_common_dataset_lambda_permissions(self, fn: _lambda.Function) -> None:
        # DynamoDB
        self.job_table.grant_read_write_data(fn)
        self.datasets_table.grant_read_write_data(fn)
        self.dataset_versions_table.grant_read_write_data(fn)
        self.lock_table.grant_read_write_data(fn)

        # File bucket: read temp submissions and Athena results
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
            resources=[
                f"arn:aws:s3:::{self.file_bucket.bucket_name}/temp/dataset-ops/*",
                f"arn:aws:s3:::{self.file_bucket.bucket_name}/athena-results/*",
            ]
        ))
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"]
        ))

        # Dataset artifacts bucket
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
            resources=[f"arn:aws:s3:::{self.datasets_bucket.bucket_name}/datasets/*"]
        ))
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.datasets_bucket.bucket_name}"]
        ))

        # Iceberg bucket
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
            resources=[f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}/*"]
        ))
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}"]
        ))

        # Athena
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "athena:StartQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "athena:StopQueryExecution",
            ],
            resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"]
        ))

        # Glue
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "glue:GetDatabase", "glue:GetDatabases",
                "glue:GetTable", "glue:GetTables",
                "glue:GetPartition", "glue:GetPartitions",
                "glue:GetTableVersion", "glue:GetTableVersions",
                "glue:CreateTable", "glue:UpdateTable", "glue:DeleteTable",
                "glue:BatchCreatePartition", "glue:BatchDeletePartition",
            ],
            resources=[
                f"arn:aws:glue:{self.region}:{self.account}:catalog",
                f"arn:aws:glue:{self.region}:{self.account}:database/{self.iceberg_database_name}",
                f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/*",
            ]
        ))

        # Firehose logging
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[self.firehose_delivery_stream.attr_arn]
        ))

        # Permission to read the testing ssm param
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["ssm:GetParameter", "ssm:GetParameters"],
                resources=["*"],
            )
        )

    def _make_dataset_lambda_task(self,
                                  *,
                                  task_id: str,
                                  lambda_id: str,
                                  lambda_config: LambdaConfig,
                                  stage_name: str,
                                  dlq_policy: str) -> tasks.LambdaInvoke:
        fn = _lambda.Function(
            self,
            lambda_id,
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=lambda_config.handler,
            code=_lambda.Code.from_asset(lambda_config.path),
            layers=[self.common_utils_layer],
            memory_size=lambda_config.memory_size,
            timeout=Duration.seconds(lambda_config.timeout_sec),
            environment=self._dataset_lambda_env()
        )

        self._grant_common_dataset_lambda_permissions(fn)

        task = tasks.LambdaInvoke(
            self,
            task_id,
            lambda_function=fn,
            result_path=f"$.{task_id}",
            output_path="$",
            payload=sfn.TaskInput.from_object({
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "event_type.$": "$.event_type",
                "task_type.$": "$.task_type",
                "submission_s3_uri.$": "$.submission_s3_uri",
                "dataset_context.$": "$.dataset_context",
                "request.$": "$.request"
            }),
            payload_response_only=True
        )

        task.add_catch(
            handler=self._make_dlq_chain(failed_stage = stage_name, dlq_policy = dlq_policy),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        return task

    def _make_cleanup_task(self, cleanup_config: LambdaConfig):
        cleanup_lambda = _lambda.Function(
            self,
            "CleanupLambdaDataset",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=cleanup_config.handler,
            code=_lambda.Code.from_asset(cleanup_config.path),
            layers=[self.common_utils_layer],
            memory_size=cleanup_config.memory_size,
            timeout=Duration.seconds(cleanup_config.timeout_sec),
            environment=self._dataset_lambda_env()
        )

        self._grant_common_dataset_lambda_permissions(cleanup_lambda)

        cleanup_task = tasks.LambdaInvoke(
            self,
            "CleanupTaskDataset",
            lambda_function=cleanup_lambda,
            result_path="$.cleanup",
            output_path="$",
            payload=sfn.TaskInput.from_object({
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "event_type.$": "$.event_type",
                "task_type.$": "$.task_type",
                "request.$": "$.request",
                "dataset_context.$": "$.dataset_context",
                "submission_s3_uri.$": "$.submission_s3_uri",
            }),
            payload_response_only=True
        )

        cleanup_task.add_catch(
            handler=self._make_dlq_chain(failed_stage = "cleanup_task", dlq_policy = "finalize_success"),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        return cleanup_task

    def _make_kickoff_lambda(self,
                             dataset_state_machine: sfn.StateMachine,
                             kickoff_config: LambdaConfig):
        kickoff_lambda = _lambda.Function(
            self,
            "KickoffLambdaDataset",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=kickoff_config.handler,
            code=_lambda.Code.from_asset(kickoff_config.path),
            layers=[self.common_utils_layer],
            memory_size=kickoff_config.memory_size,
            timeout=Duration.seconds(kickoff_config.timeout_sec),
            environment={
                "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
                "JOB_TABLE_NAME": self.job_table.table_name,
                "LOCK_TABLE_NAME": self.lock_table.table_name,
                "DATASET_STATE_MACHINE_ARN": dataset_state_machine.state_machine_arn,
                "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
                "DATASET_DLQ_URL": self.dlq.queue_url
            }
        )

        dataset_state_machine.grant_start_execution(kickoff_lambda)

        self.job_table.grant_read_write_data(kickoff_lambda)
        self.lock_table.grant_read_write_data(kickoff_lambda)
        self.file_bucket.grant_read(kickoff_lambda)

        kickoff_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
        ))

        kickoff_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[self.firehose_delivery_stream.attr_arn],
        ))

        kickoff_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["sqs:SendMessage"],
            resources=[self.dlq.queue_arn],
        ))

        kickoff_lambda.add_event_source(
            event_sources.SqsEventSource(self.dataset_events_queue, batch_size=1)
        )
        self.dataset_events_queue.grant_consume_messages(kickoff_lambda)

    def make_common_utils_layer(self):
        # Create a Lambda Layer from the common utilities
        common_layer = _lambda.LayerVersion(
            self,
            "CommonUtilsLayerDataset",
            code=_lambda.Code.from_asset("workers/common"),
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_11],
            description="Shared utilities for all Lambda functions in the dataset flow"
        )
        return common_layer