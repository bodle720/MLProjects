import uuid
from constructs import Construct

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
    aws_kinesisfirehose as firehose
)

from config import CONFIG
from stacks.helper_constructs.batching_stage import BatchingStage
from stacks.helper_constructs.ingest_stage import IngestStage
from config_models import ComputeEnvConfig, LambdaConfig

class ImageUploadStack(Stack):
    def __init__(self,
                 scope: Construct,
                 construct_id: str,
                 app_name: str,
                 common_utils_layer: _lambda.LayerVersion,
                 file_bucket: s3.Bucket,
                 iceberg_bucket: s3.Bucket,
                 job_table: dynamodb.Table,
                 sha256_table: dynamodb.Table,
                 lock_table: dynamodb.Table,
                 global_dlq: sqs.Queue,
                 iceberg_database_name: str,
                 upload_events_queue: sqs.Queue,
                 firehose_delivery_stream: firehose.CfnDeliveryStream,  # L1 type
                 **kwargs) -> None:

        super().__init__(scope, construct_id, **kwargs)

        # Variables from Storage stack and app name.
        self.app_name = app_name
        self.file_bucket = file_bucket
        self.iceberg_bucket = iceberg_bucket
        self.job_table = job_table
        self.sha256_table = sha256_table
        self.lock_table = lock_table
        self.global_dlq = global_dlq
        self.iceberg_database_name = iceberg_database_name
        self.upload_events_queue = upload_events_queue
        self.firehose_delivery_stream = firehose_delivery_stream
        self.common_utils_layer = common_utils_layer

        # Creates Batch compute environment and job queue pointing to the compute environment.
        job_queue = self._make_compute_env(CONFIG.compute_env)

        validation_stage = BatchingStage(
            self, "validationStage",
            stage_name="validationStage",
            config=CONFIG.validation,
            common_utils_layer=self.common_utils_layer,
            file_bucket=self.file_bucket,
            iceberg_bucket=self.iceberg_bucket,
            job_table=self.job_table,
            sha256_table=self.sha256_table,
            job_queue=job_queue,
            iceberg_database_name=self.iceberg_database_name,
            ce_maxv_cpus=CONFIG.compute_env.maxv_cpus,
            region=self.region,
            account=self.account,
            dlq_chain_factory=self._make_dlq_chain,
            firehose_delivery_stream_name=self.firehose_delivery_stream.ref,
            firehose_delivery_stream_attr_arn=self.firehose_delivery_stream.attr_arn
        )

        deduplication_stage = BatchingStage(
            self, "deduplicationStage",
            stage_name="deduplicationStage",
            config=CONFIG.deduplication,
            common_utils_layer=self.common_utils_layer,
            file_bucket=self.file_bucket,
            iceberg_bucket=self.iceberg_bucket,
            job_table=self.job_table,
            sha256_table=self.sha256_table,
            job_queue=job_queue,
            iceberg_database_name=self.iceberg_database_name,
            ce_maxv_cpus=CONFIG.compute_env.maxv_cpus,
            region=self.region,
            account=self.account,
            dlq_chain_factory=self._make_dlq_chain,
            firehose_delivery_stream_name=self.firehose_delivery_stream.ref,
            firehose_delivery_stream_attr_arn=self.firehose_delivery_stream.attr_arn
        )

        registration_stage = BatchingStage(
            self, "registrationStage",
            stage_name="registrationStage",
            config=CONFIG.registration,
            common_utils_layer=self.common_utils_layer,
            file_bucket=self.file_bucket,
            iceberg_bucket=self.iceberg_bucket,
            job_table=self.job_table,
            sha256_table=self.sha256_table,
            job_queue=job_queue,
            iceberg_database_name=self.iceberg_database_name,
            ce_maxv_cpus=CONFIG.compute_env.maxv_cpus,
            region=self.region,
            account=self.account,
            dlq_chain_factory=self._make_dlq_chain,
            firehose_delivery_stream_name=self.firehose_delivery_stream.ref,
            firehose_delivery_stream_attr_arn=self.firehose_delivery_stream.attr_arn
        )

        validation_ingest_stage = IngestStage(
            self, "validationIngestStage",
            stage_name="validationIngestStage",
            config=CONFIG.validation_ingest,
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
            manifest_path="$.validationStage.manifests",
            expected_count_path="$.validationStage.expected_count"
        )

        deduplication_ingest_stage = IngestStage(
            self, "deduplicationIngestStage",
            stage_name="deduplicationIngestStage",
            config=CONFIG.deduplication_ingest,
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
            manifest_path="$.deduplicationStage.manifests"
        )

        registration_ingest_stage = IngestStage(
            self, "registrationIngestStage",
            stage_name="registrationIngestStage",
            config=CONFIG.registration_ingest,
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
            manifest_path="$.registrationStage.manifests",
            expected_count_path="$.registrationStage.eligible_rows"
        )

        # Make cleanup lambda to run once entire upload job is done.
        cleanup_task = self._make_cleanup_task(CONFIG.cleanup_lambda)

        workflow_definition = sfn.Chain.start(validation_stage.batching_task) \
            .next(validation_stage.map_state) \
            .next(validation_ingest_stage.pre_ingest_task) \
            .next(validation_ingest_stage.map_state) \
            .next(validation_ingest_stage.post_ingest_task) \
            .next(deduplication_stage.batching_task) \
            .next(deduplication_stage.map_state) \
            .next(deduplication_ingest_stage.pre_ingest_task) \
            .next(deduplication_ingest_stage.map_state) \
            .next(deduplication_ingest_stage.post_ingest_task) \
            .next(registration_stage.batching_task) \
            .next(registration_stage.map_state) \
            .next(registration_ingest_stage.pre_ingest_task) \
            .next(registration_ingest_stage.map_state) \
            .next(registration_ingest_stage.post_ingest_task) \
            .next(cleanup_task)

        upload_state_machine = sfn.StateMachine(self, "UploadStateMachine",
                              definition_body=sfn.DefinitionBody.from_chainable(workflow_definition),
                              timeout=Duration.hours(CONFIG.upload_state_machine.duration_hours)
                              )

        upload_state_machine.role.add_to_principal_policy(
            iam.PolicyStatement(actions=["sqs:SendMessage"], resources=[self.global_dlq.queue_arn]))

        upload_state_machine.apply_removal_policy(RemovalPolicy.DESTROY)

        upload_state_machine.role.add_to_principal_policy(iam.PolicyStatement(
            actions=["batch:TerminateJob"],
            resources=["*"],
        ))

        for stage in [validation_stage, deduplication_stage, registration_stage]:
            upload_state_machine.role.add_to_principal_policy(iam.PolicyStatement(
                actions=["batch:SubmitJob", "batch:DescribeJobs"],
                resources=[job_queue.job_queue_arn, stage.job_def.attr_job_definition_arn]
            ))

            # allow the state machine role to pass the job role to Batch/ECS
            upload_state_machine.role.add_to_principal_policy(iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[stage.job_role.role_arn]
            ))

        # Make kickoff lambda to trigger on job.json upload
        self._make_kickoff_lambda(upload_state_machine, CONFIG.kickoff_lambda)

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
                "QueueUrl": self.global_dlq.queue_url,
                "MessageBody.$": "States.JsonToString($.dlqMessage)"
            },
            iam_resources=[self.global_dlq.queue_arn],
        )

        fail = sfn.Fail(self, f"WorkflowFailed_{suffix}", cause="SentToGlobalDLQ", error="WorkflowError")

        return sfn.Chain.start(make_dlq_message).next(send_to_dlq).next(fail)

    def _make_compute_env(self,
                          ce_config: ComputeEnvConfig):

        # Use the default VPC (public subnets included)
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # Create a security group for Batch instances
        batch_sg = ec2.SecurityGroup(
            self, "BatchSecurityGroup",
            vpc=vpc,
            description="Security group for Batch compute environment",
            allow_all_outbound=True  # needed so jobs can reach S3/ECR
        )

        # The role the EC2 nodes take on, needed to register with clusters, pull containers, etc
        # Things for the node to do, not necessarily the job running on it.
        instance_role = iam.Role(
            self, "BatchInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonEC2ContainerServiceforEC2Role"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
            ]
        )

        # Make the compute environment
        compute_env = batch.ManagedEc2EcsComputeEnvironment(
            self, "ComputeEnv",
            vpc=vpc,
            # Spot configuration
            spot=True,
            allocation_strategy=batch.AllocationStrategy.SPOT_PRICE_CAPACITY_OPTIMIZED,
            spot_bid_percentage=100,
            # Scaling limits
            minv_cpus=ce_config.minv_cpus,
            maxv_cpus=ce_config.maxv_cpus,
            # Instances
            instance_types=[ec2.InstanceType(i) for i in ce_config.instance_types],
            security_groups=[batch_sg],
            # Optional: restrict subnets
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC), # ec2.SubnetType.PRIVATE_WITH_EGRESS for prod, safer
            # Roles
            instance_role=instance_role,
            # Image, per AWS Health update for 2026
            images=[batch.EcsMachineImage(image_type=batch.EcsMachineImageType.ECS_AL2023)]
        )

        # The job queue for the above compute environment.
        job_queue = batch.JobQueue(
                                self,
                            "JobQueue",
                                priority=1, # If multiple queues share the compute env, this queue takes first priority.
                                compute_environments=[
                                    batch.OrderedComputeEnvironment(
                                        compute_environment=compute_env,
                                        order=1 # Ensures this is the first and only compute environment batch will try.
                                    )
                                ]
                            )
        return job_queue

    def _make_cleanup_task(self,
                           cleanup_config: LambdaConfig):
        cleanup_lambda = _lambda.Function(
            self,
            "CleanupLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=cleanup_config.handler,
            code=_lambda.Code.from_asset(cleanup_config.path),
            layers = [self.common_utils_layer],
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

        # 2) S3: delete temp files under temp/image-upload/ and read them
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:DeleteObject"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/temp/image-upload/*"]
        ))
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
            conditions={"StringLike": {"s3:prefix": ["temp/image-upload/*"]}}
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

        # 7) Glue metadata write for upload_staging (required when Athena DELETE/OPTIMIZE updates Iceberg metadata)
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "glue:CreateTable", "glue:UpdateTable", "glue:DeleteTable",
                "glue:BatchCreatePartition", "glue:BatchDeletePartition"
            ],
            resources=[f"arn:aws:glue:{self.region}:{self.account}:catalog",
                       f"arn:aws:glue:{self.region}:{self.account}:database/{self.iceberg_database_name}",
                       f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/upload_staging"
                       ]
        ))

        # 8) S3: read and delete Iceberg files for upload_staging prefix
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:DeleteObject", "s3:PutObject"],
            resources=[f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}/upload_staging/*"]
        ))
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}"],
            conditions={"StringLike": {"s3:prefix": ["upload_staging/*"]}}
        ))

        cleanup_task = tasks.LambdaInvoke(
            self, "CleanupTask",
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
                            upload_state_machine,
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
                    "UPLOAD_STATE_MACHINE_ARN": upload_state_machine.state_machine_arn,
                    "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
                    "GLOBAL_DLQ_URL": self.global_dlq.queue_url
                }
            )

            upload_state_machine.grant_start_execution(kickoff_lambda)

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
            kickoff_lambda.add_event_source(event_sources.SqsEventSource(self.upload_events_queue, batch_size=1))
            self.upload_events_queue.grant_consume_messages(kickoff_lambda)