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
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_sqs as sqs,
    aws_dynamodb as dynamodb,
    aws_kinesisfirehose as firehose,
)

from config import CONFIG
from stacks.helper_constructs.batching_stage import BatchingStage
from stacks.helper_constructs.ingest_stage import IngestStage
from config_models import ComputeEnvConfig, LambdaConfig

class UploadStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        app_name: str,
        common_utils_layer: _lambda.LayerVersion,
        file_bucket: s3.Bucket,
        iceberg_bucket: s3.Bucket,
        job_table: dynamodb.Table,
        sha256_table: dynamodb.Table,
        lock_table: dynamodb.Table,
        iceberg_database_name: str,
        firehose_delivery_stream: firehose.CfnDeliveryStream,
        upload_events_queue: sqs.Queue,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Variables from Storage/Logging stack and app name.
        self.app_name = app_name
        self.file_bucket = file_bucket
        self.iceberg_bucket = iceberg_bucket
        self.job_table = job_table
        self.sha256_table = sha256_table
        self.lock_table = lock_table
        self.iceberg_database_name = iceberg_database_name
        self.firehose_delivery_stream = firehose_delivery_stream
        self.common_utils_layer = common_utils_layer

        # Make the SQS Queue that will receive upload events.
        self.upload_events_queue = upload_events_queue

        # Make the DLQ.
        self.dlq = self.make_dlq_assign_permissions()

        # Creates Batch compute environment and job queue pointing to the compute environment.
        job_queue = self._make_compute_env(CONFIG.upload.compute_env)

        # ------------------------------------------------------------------
        # Batching stages (now return compact S3 plan pointers, not inline arrays)
        # ------------------------------------------------------------------
        validation_stage = BatchingStage(
            self,
            "validationStage",
            stage_name="validationStage",
            config=CONFIG.upload.validation,
            common_utils_layer=self.common_utils_layer,
            file_bucket=self.file_bucket,
            iceberg_bucket=self.iceberg_bucket,
            job_table=self.job_table,
            sha256_table=self.sha256_table,
            job_queue=job_queue,
            iceberg_database_name=self.iceberg_database_name,
            ce_maxv_cpus=CONFIG.upload.compute_env.maxv_cpus,
            region=self.region,
            account=self.account,
            dlq_chain_factory=self._make_dlq_chain,
            firehose_delivery_stream_name=self.firehose_delivery_stream.ref,
            firehose_delivery_stream_attr_arn=self.firehose_delivery_stream.attr_arn,
        )

        deduplication_stage = BatchingStage(
            self,
            "deduplicationStage",
            stage_name="deduplicationStage",
            config=CONFIG.upload.deduplication,
            common_utils_layer=self.common_utils_layer,
            file_bucket=self.file_bucket,
            iceberg_bucket=self.iceberg_bucket,
            job_table=self.job_table,
            sha256_table=self.sha256_table,
            job_queue=job_queue,
            iceberg_database_name=self.iceberg_database_name,
            ce_maxv_cpus=CONFIG.upload.compute_env.maxv_cpus,
            region=self.region,
            account=self.account,
            dlq_chain_factory=self._make_dlq_chain,
            firehose_delivery_stream_name=self.firehose_delivery_stream.ref,
            firehose_delivery_stream_attr_arn=self.firehose_delivery_stream.attr_arn,
        )

        registration_stage = BatchingStage(
            self,
            "registrationStage",
            stage_name="registrationStage",
            config=CONFIG.upload.registration,
            common_utils_layer=self.common_utils_layer,
            file_bucket=self.file_bucket,
            iceberg_bucket=self.iceberg_bucket,
            job_table=self.job_table,
            sha256_table=self.sha256_table,
            job_queue=job_queue,
            iceberg_database_name=self.iceberg_database_name,
            ce_maxv_cpus=CONFIG.upload.compute_env.maxv_cpus,
            region=self.region,
            account=self.account,
            dlq_chain_factory=self._make_dlq_chain,
            firehose_delivery_stream_name=self.firehose_delivery_stream.ref,
            firehose_delivery_stream_attr_arn=self.firehose_delivery_stream.attr_arn,
        )

        # ------------------------------------------------------------------
        # Ingest stages (now take compact batch plan pointers and build their own
        # S3-backed ingest map plans in pre-lambdas)
        # ------------------------------------------------------------------
        validation_ingest_stage = IngestStage(
            self,
            "validationIngestStage",
            stage_name="validationIngestStage",
            config=CONFIG.upload.validation_ingest,
            common_utils_layer=self.common_utils_layer,
            file_bucket=self.file_bucket,
            iceberg_bucket=self.iceberg_bucket,
            job_table=self.job_table,
            lock_table=self.lock_table,
            sha256_table=self.sha256_table,
            iceberg_database_name=self.iceberg_database_name,
            region=self.region,
            account=self.account,
            dlq_chain_factory=self._make_dlq_chain,
            firehose_delivery_stream_name=self.firehose_delivery_stream.ref,
            firehose_delivery_stream_attr_arn=self.firehose_delivery_stream.attr_arn,
            batch_plan_bucket_path="$.validationStage.plan_bucket",
            batch_plan_key_path="$.validationStage.plan_key",
            batch_plan_s3_uri_path="$.validationStage.plan_s3_uri",
            expected_count_path="$.validationStage.expected_count",
        )

        deduplication_ingest_stage = IngestStage(
            self,
            "deduplicationIngestStage",
            stage_name="deduplicationIngestStage",
            config=CONFIG.upload.deduplication_ingest,
            common_utils_layer=self.common_utils_layer,
            file_bucket=self.file_bucket,
            iceberg_bucket=self.iceberg_bucket,
            job_table=self.job_table,
            lock_table=self.lock_table,
            sha256_table=self.sha256_table,
            iceberg_database_name=self.iceberg_database_name,
            region=self.region,
            account=self.account,
            dlq_chain_factory=self._make_dlq_chain,
            firehose_delivery_stream_name=self.firehose_delivery_stream.ref,
            firehose_delivery_stream_attr_arn=self.firehose_delivery_stream.attr_arn,
            batch_plan_bucket_path="$.deduplicationStage.plan_bucket",
            batch_plan_key_path="$.deduplicationStage.plan_key",
            batch_plan_s3_uri_path="$.deduplicationStage.plan_s3_uri",
        )

        registration_ingest_stage = IngestStage(
            self,
            "registrationIngestStage",
            stage_name="registrationIngestStage",
            config=CONFIG.upload.registration_ingest,
            common_utils_layer=self.common_utils_layer,
            file_bucket=self.file_bucket,
            iceberg_bucket=self.iceberg_bucket,
            job_table=self.job_table,
            lock_table=self.lock_table,
            sha256_table=self.sha256_table,
            iceberg_database_name=self.iceberg_database_name,
            region=self.region,
            account=self.account,
            dlq_chain_factory=self._make_dlq_chain,
            firehose_delivery_stream_name=self.firehose_delivery_stream.ref,
            firehose_delivery_stream_attr_arn=self.firehose_delivery_stream.attr_arn,
            batch_plan_bucket_path="$.registrationStage.plan_bucket",
            batch_plan_key_path="$.registrationStage.plan_key",
            batch_plan_s3_uri_path="$.registrationStage.plan_s3_uri",
            expected_count_path="$.registrationStage.total_rows",
        )

        # Make cleanup lambda to run once the entire upload job is done.
        cleanup_task = self._make_cleanup_task(CONFIG.upload.cleanup_lambda)

        workflow_definition = (
            sfn.Chain.start(validation_stage.batching_task)
            .next(validation_stage.map_state)
            .next(validation_ingest_stage.pre_ingest_task)
            .next(validation_ingest_stage.map_state)
            .next(validation_ingest_stage.post_ingest_task)
            .next(deduplication_stage.batching_task)
            .next(deduplication_stage.map_state)
            .next(deduplication_ingest_stage.pre_ingest_task)
            .next(deduplication_ingest_stage.map_state)
            .next(deduplication_ingest_stage.post_ingest_task)
            .next(registration_stage.batching_task)
            .next(registration_stage.map_state)
            .next(registration_ingest_stage.pre_ingest_task)
            .next(registration_ingest_stage.map_state)
            .next(registration_ingest_stage.post_ingest_task)
            .next(cleanup_task)
        )

        upload_state_machine = sfn.StateMachine(
            self,
            "UploadStateMachine",
            definition_body=sfn.DefinitionBody.from_chainable(workflow_definition),
            timeout=Duration.hours(CONFIG.upload.upload_state_machine.duration_hours),
        )

        upload_state_machine.role.add_to_principal_policy(
            iam.PolicyStatement(actions=["sqs:SendMessage"], resources=[self.dlq.queue_arn])
        )

        upload_state_machine.apply_removal_policy(RemovalPolicy.DESTROY)

        # Distributed Map child workflow permissions. The ingest helper already exposes
        # these as required state machine policies, but adding this once globally is safe
        # and also covers the batching Distributed Maps.
        upload_state_machine.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=[
                    "states:StartExecution",
                    "states:DescribeExecution",
                    "states:StopExecution",
                ],
                resources=["*"],
            )
        )

        upload_state_machine.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=[
                    "events:PutRule",
                    "events:PutTargets",
                    "events:DescribeRule",
                ],
                resources=[
                    f"arn:aws:events:{self.region}:{self.account}:rule/StepFunctionsGetEventsForBatchJobsRule"
                ],
            )
        )

        upload_state_machine.role.add_to_principal_policy(
            iam.PolicyStatement(
                actions=[
                    "batch:SubmitJob",
                    "batch:DescribeJobs",
                    "batch:TerminateJob",
                ],
                resources=["*"],
            )
        )

        for stage in [validation_stage, deduplication_stage, registration_stage]:
            # allow the state machine role to pass the job role to Batch/ECS
            upload_state_machine.role.add_to_principal_policy(
                iam.PolicyStatement(
                    actions=["iam:PassRole"],
                    resources=[stage.job_role.role_arn],
                )
            )

            for stmt in stage.state_machine_policy_statements:
                upload_state_machine.role.add_to_principal_policy(stmt)

        # IngestStage uses a CustomState for the Distributed Map, so attach its required
        # state machine role permissions explicitly.
        for ingest_stage in [
            validation_ingest_stage,
            deduplication_ingest_stage,
            registration_ingest_stage,
        ]:
            for stmt in ingest_stage.state_machine_policy_statements:
                upload_state_machine.role.add_to_principal_policy(stmt)

        # Make kickoff lambda to trigger on job.json upload.
        self._make_kickoff_lambda(upload_state_machine, CONFIG.upload.kickoff_lambda)

    def make_dlq_assign_permissions(self):
        dlq_processor_env_vars = {
            "JOB_TABLE_NAME": self.job_table.table_name,
            "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
            "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
            "LOCK_TABLE_NAME": self.lock_table.table_name,
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DATABASE_NAME": self.iceberg_database_name,
            "ATHENA_OUTPUT_S3": f"s3://{self.file_bucket.bucket_name}/athena-results/",
            "SHA256_TABLE_NAME": self.sha256_table.table_name,
        }

        dlq_out = DLQOps(
            self,
            "upload_dlq",
            name="upload",
            app_name=self.app_name,
            dlq_processor_env_vars=dlq_processor_env_vars,
            region=self.region,
            account=self.account,
            dlq_ops_config=CONFIG.upload.dlq_ops,
            iceberg_database_name=self.iceberg_database_name,
            common_utils_layer=self.common_utils_layer,
            file_bucket=self.file_bucket,
            firehose_delivery_stream=self.firehose_delivery_stream,
        )

        dlq = dlq_out.dlq
        dlq_processor = dlq_out.dlq_processor

        # DynamoDB
        self.lock_table.grant_read_write_data(dlq_processor)
        self.job_table.grant_read_write_data(dlq_processor)
        self.sha256_table.grant_read_write_data(dlq_processor)

        # File bucket:
        # - temp/image-upload/* : read rollback artifacts / processed outputs
        # - canonical/*        : delete canonical image + label objects written by failed job
        dlq_processor.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:DeleteObject"],
                resources=[
                    f"arn:aws:s3:::{self.file_bucket.bucket_name}/temp/image-upload/*",
                    f"arn:aws:s3:::{self.file_bucket.bucket_name}/canonical/*",
                ],
            )
        )

        dlq_processor.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
                conditions={
                    "StringLike": {
                        "s3:prefix": [
                            "temp/image-upload/*",
                            "canonical/*",
                        ]
                    }
                },
            )
        )

        dlq_processor.add_to_role_policy(
            iam.PolicyStatement(
                actions=["batch:DescribeJobs"],
                resources=["*"],
            )
        )

        # Iceberg bucket:
        # canonical/* covers:
        # - canonical/imagery/*
        # - canonical/image-labels/*
        # - canonical/bounding-boxes/*
        # - canonical/semantic-masks/*
        # - canonical/instance-annotations/*
        dlq_processor.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:DeleteObject", "s3:PutObject"],
                resources=[
                    f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}/upload_staging/*",
                    f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}/canonical/*",
                    f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}/image_source_membership/*",
                ],
            )
        )

        dlq_processor.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}"],
                conditions={
                    "StringLike": {
                        "s3:prefix": [
                            "upload_staging/*",
                            "canonical/*",
                            "image_source_membership/*",
                        ]
                    }
                },
            )
        )

        # Glue delete for CTAS temp tables
        dlq_processor.add_to_role_policy(
            iam.PolicyStatement(
                actions=["glue:DeleteTable"],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/dedup_export_*",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/reg_export_*",
                ],
            )
        )

        # Glue metadata write for rollback-mutated Iceberg tables
        dlq_processor.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:CreateTable",
                    "glue:UpdateTable",
                    "glue:DeleteTable",
                    "glue:BatchCreatePartition",
                    "glue:BatchDeletePartition",
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{self.iceberg_database_name}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/upload_staging",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/canonical_imagery",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/image_labels",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/image_source_membership",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/canonical_bounding_boxes",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/canonical_semantic_masks",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/canonical_instance_annotations",
                ],
            )
        )

        return dlq

    def _make_dlq_chain(self) -> sfn.Chain:
        suffix = uuid.uuid4().hex[:8]

        make_dlq_message = sfn.Pass(
            self,
            f"MakeDLQMessage_{suffix}",
            parameters={
                "source": "stepfunctions",
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "event_type.$": "$.event_type",
                "error.$": "States.JsonToString($.errorInfo)",
            },
            result_path="$.dlqMessage",
        )

        send_to_dlq = tasks.CallAwsService(
            self,
            f"SendToDLQ_{suffix}",
            service="sqs",
            action="sendMessage",
            parameters={
                "QueueUrl": self.dlq.queue_url,
                "MessageBody.$": "States.JsonToString($.dlqMessage)",
            },
            iam_resources=[self.dlq.queue_arn],
        )

        fail = sfn.Fail(self, f"WorkflowFailed_{suffix}", cause="SentToUploadDLQ", error="WorkflowError")

        return sfn.Chain.start(make_dlq_message).next(send_to_dlq).next(fail)

    def _make_compute_env(self, ce_config: ComputeEnvConfig):
        # Use the default VPC (public subnets included)
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # Create a security group for Batch instances
        batch_sg = ec2.SecurityGroup(
            self,
            "BatchSecurityGroup",
            vpc=vpc,
            description="Security group for Batch compute environment",
            allow_all_outbound=True,
        )

        # The role the EC2 nodes take on, needed to register with clusters, pull containers, etc.
        instance_role = iam.Role(
            self,
            "BatchInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonEC2ContainerServiceforEC2Role"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
            ],
        )

        compute_env = batch.ManagedEc2EcsComputeEnvironment(
            self,
            "ComputeEnv",
            vpc=vpc,
            spot=True,
            allocation_strategy=batch.AllocationStrategy.SPOT_PRICE_CAPACITY_OPTIMIZED,
            spot_bid_percentage=100,
            minv_cpus=ce_config.minv_cpus,
            maxv_cpus=ce_config.maxv_cpus,
            instance_types=[ec2.InstanceType(i) for i in ce_config.instance_types],
            security_groups=[batch_sg],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_role=instance_role,
            images=[batch.EcsMachineImage(image_type=batch.EcsMachineImageType.ECS_AL2023)],
        )

        job_queue = batch.JobQueue(
            self,
            "JobQueue",
            priority=1,
            compute_environments=[
                batch.OrderedComputeEnvironment(
                    compute_environment=compute_env,
                    order=1,
                )
            ],
        )
        return job_queue

    def _make_cleanup_task(self, cleanup_config: LambdaConfig):
        cleanup_lambda = _lambda.Function(
            self,
            "CleanupLambdaUpload",
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
                "ATHENA_WORKGROUP": "primary",
            },
        )

        # 1) DynamoDB
        self.lock_table.grant_read_write_data(cleanup_lambda)
        self.job_table.grant_read_write_data(cleanup_lambda)

        # 2) S3: delete temp files under temp/image-upload/ and read them
        cleanup_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:DeleteObject"],
                resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/temp/image-upload/*"],
            )
        )
        cleanup_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
                conditions={"StringLike": {"s3:prefix": ["temp/image-upload/*"]}},
            )
        )

        # 3) S3: Athena results write only to athena-results/
        cleanup_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject"],
                resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/athena-results/*"],
            )
        )
        cleanup_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
            )
        )

        # 4) Athena: start and poll queries in the workgroup
        cleanup_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults"],
                resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"],
            )
        )

        # 5) Firehose logging
        cleanup_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
                resources=[self.firehose_delivery_stream.attr_arn],
            )
        )

        # 6) Glue metadata read (catalog, DB, and tables)
        cleanup_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:GetDatabase",
                    "glue:GetDatabases",
                    "glue:GetTable",
                    "glue:GetTables",
                    "glue:GetPartition",
                    "glue:GetPartitions",
                    "glue:GetTableVersion",
                    "glue:GetTableVersions",
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{self.iceberg_database_name}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/*",
                ],
            )
        )

        # 7) Glue metadata write for upload_staging (required when Athena DELETE/OPTIMIZE updates Iceberg metadata)
        cleanup_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:CreateTable",
                    "glue:UpdateTable",
                    "glue:DeleteTable",
                    "glue:BatchCreatePartition",
                    "glue:BatchDeletePartition",
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{self.iceberg_database_name}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/upload_staging",
                ],
            )
        )

        # 8) S3: read and delete Iceberg files for upload_staging and canonical prefix
        cleanup_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:DeleteObject", "s3:PutObject"],
                resources=[
                    f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}/upload_staging/*",
                    f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}/canonical/*",
                ],
            )
        )
        cleanup_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}"],
                conditions={"StringLike": {"s3:prefix": ["upload_staging/*", "canonical/*"]}},
            )
        )

        cleanup_task = tasks.LambdaInvoke(
            self,
            "CleanupTask",
            lambda_function=cleanup_lambda,
            result_path="$.cleanup",
            output_path="$",
            payload=sfn.TaskInput.from_object(
                {
                    "job_id.$": "$.job_id",
                    "user.$": "$.user",
                    "event_type.$": "$.event_type",
                }
            ),
            payload_response_only=True,
        )

        cleanup_task.add_retry(backoff_rate=2.0, max_attempts=2, interval=Duration.seconds(2))

        cleanup_task.add_catch(
            handler=self._make_dlq_chain(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        return cleanup_task

    def _make_kickoff_lambda(self, upload_state_machine, kickoff_config: LambdaConfig):
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
                "UPLOAD_STATE_MACHINE_ARN": upload_state_machine.state_machine_arn,
                "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
                "UPLOAD_DLQ_URL": self.dlq.queue_url,
            },
        )

        upload_state_machine.grant_start_execution(kickoff_lambda)

        # Permissions for the kickoff lambda
        self.job_table.grant_read_write_data(kickoff_lambda)
        self.file_bucket.grant_read(kickoff_lambda)
        self.lock_table.grant_read_data(kickoff_lambda)

        kickoff_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
            )
        )

        kickoff_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/*"],
            )
        )

        kickoff_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
                resources=[self.firehose_delivery_stream.attr_arn],
            )
        )

        kickoff_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sqs:SendMessage"],
                resources=[self.dlq.queue_arn],
            )
        )

        kickoff_lambda.add_event_source(event_sources.SqsEventSource(self.upload_events_queue, batch_size=1))
        self.upload_events_queue.grant_consume_messages(kickoff_lambda)