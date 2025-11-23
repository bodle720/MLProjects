from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    Size,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_s3_notifications as s3n,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_lambda_event_sources as event_sources,
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_sqs as sqs,
    aws_logs as logs,
    aws_dynamodb as dynamodb,
    aws_kinesisfirehose as firehose,
    aws_ecs as ecs,
    aws_ecr_assets as ecr_assets
)
from constructs import Construct
from typing import Callable
import uuid

from config import CONFIG
from config_models import ComputeEnvConfig, KickoffLambdaConfig, CleanupLambdaConfig, StageConfig

class BatchingStage(Construct):
    def __init__(self, scope: Construct, id: str, *,
                 stage_name: str,
                 config: StageConfig,
                 common_utils_layer: _lambda.LayerVersion,
                 file_bucket: s3.Bucket,
                 job_table: dynamodb.Table,
                 sha256_table: dynamodb.Table,
                 phash_table: dynamodb.Table,
                 job_queue: batch.JobQueue,
                 athena_database_name: str,
                 ce_maxv_cpus: int,
                 region: str,
                 account: str,
                 global_dlq: sqs.Queue,
                 dlq_chain_factory: Callable[[], sfn.Chain],
                 firehose_delivery_stream_name: str,
                 extra_lambda_env: dict = None,
                 extra_permissions: list[iam.PolicyStatement] = None,
                 extra_container_env: dict = None,
                 extra_map_state_params: dict = None):

        '''
        Batching param explanation:
        The batching Lambda returns a dict with manifests, job_id, user, label_types. The Map state iterates over manifests,
        builds a per‑item input with those values, and the Batch task pulls them into container environment variables via
        JsonPath.string_at
        '''

        super().__init__(scope, id)

        # Lambda env vars
        lambda_env = {
            "FILE_BUCKET_NAME": file_bucket.bucket_name,
            "ATHENA_OUTPUT_S3": f"s3://{file_bucket.bucket_name}/athena-results/",
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DB": athena_database_name,
            "UPLOAD_STAGING_TABLE": "upload_staging",
            "SHA256_TABLE": sha256_table.table_name,
            "PHASH_TABLE": phash_table.table_name,
            "LOG_FIREHOSE_STREAM_NAME": firehose_delivery_stream_name
        }

        if extra_lambda_env:
            lambda_env.update(extra_lambda_env)

        batching_fn = _lambda.Function(
            self, f"{stage_name}BatchingLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=config.file_batching.handler,
            code=_lambda.Code.from_asset(config.file_batching.path),
            layers=[common_utils_layer],
            dead_letter_queue=global_dlq,
            memory_size=config.file_batching.memory_size,
            timeout=Duration.minutes(config.file_batching.timeout_min),
            environment=lambda_env
        )

        # baseline grants and policies
        sha256_table.grant_read_data(batching_fn)
        phash_table.grant_read_data(batching_fn)
        file_bucket.grant_read_write(batching_fn)

        batching_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket","s3:GetBucketLocation"],
            resources=[file_bucket.bucket_arn]
        ))
        batching_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["athena:StartQueryExecution","athena:GetQueryExecution","athena:GetQueryResults"],
            resources=[f"arn:aws:athena:{region}:{account}:workgroup/primary"]
        ))

        if extra_permissions:
            for stmt in extra_permissions:
                batching_fn.add_to_role_policy(stmt)

        batching_task = tasks.LambdaInvoke(
            self, f"{stage_name}BatchingTask",
            lambda_function=batching_fn,
            result_path=f"$.{stage_name}",
            output_path="$",
            payload_response_only=True # this means Lambda’s return is also merged into the state output, not just nested.
        )

        batching_task.add_retry(backoff_rate=2.0, max_attempts=2, interval=Duration.seconds(2))

        # if the batching Lambda fails, Step Functions will capture the error info under $.errorInfo and transition to the DLQ state.
        batching_task.add_catch(
            handler=dlq_chain_factory(),  # fresh chain instance
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        job_role = iam.Role(
            self, f"{stage_name}JobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )
        file_bucket.grant_read_write(job_role)
        job_table.grant_read_write_data(job_role)
        phash_table.grant_read_data(job_role)
        sha256_table.grant_read_data(job_role)
        job_role.add_to_policy(iam.PolicyStatement(
            actions=["athena:StartQueryExecution","athena:GetQueryExecution","athena:GetQueryResults"],
            resources=[f"arn:aws:athena:{region}:{account}:workgroup/primary"]
        ))
        if extra_permissions:
            for stmt in extra_permissions:
                job_role.add_to_policy(stmt)

        # build/publish local Docker image from a local path
        image_asset = ecr_assets.DockerImageAsset(self, f"{stage_name}TaskImage",
                                                  directory=config.batch_task_job_def.directory,
                                                  file=config.batch_task_job_def.file
                                                  )
        container_image = ecs.ContainerImage.from_registry(image_asset.image_uri)

        job_def = batch.EcsJobDefinition(
            self,
            f"{stage_name}JobDef",
            retry_attempts=5,
            timeout=Duration.hours(2),
            container=batch.EcsEc2ContainerDefinition(
                self,
                f"{stage_name}containerDefn",
                image=container_image,
                memory=Size.mebibytes(config.batch_task_job_def.memory_limit_mib),
                cpu=int(config.batch_task_job_def.vcpus * 1024),
                job_role=job_role
            )
        )

        container_env = {
            "MANIFEST_S3_KEY": sfn.JsonPath.string_at("$.manifest"), # means Step Functions will substitute the values from the per‑item input into environment variables for the container, see below
            "JOB_ID": sfn.JsonPath.string_at("$.job_id"),
            "USER": sfn.JsonPath.string_at("$.user"),
            "LABEL_TYPES": sfn.JsonPath.string_at("$.label_types"),
            "SOURCE": sfn.JsonPath.string_at("$.source"),
            "EVENT_TYPE": sfn.JsonPath.string_at("$.event_type"),
            "FILE_BUCKET_NAME": file_bucket.bucket_name,
            "ATHENA_OUTPUT_S3": f"s3://{file_bucket.bucket_name}/athena-results/",
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DB": athena_database_name,
            "UPLOAD_STAGING_TABLE": "upload_staging",
            "LOG_FIREHOSE_STREAM_NAME": firehose_delivery_stream_name
        }
        if extra_container_env:
            container_env.update(extra_container_env)

        batch_task = tasks.BatchSubmitJob(
            self, f"{stage_name}BatchTask",
            job_definition_arn=job_def.job_definition_arn,
            job_queue_arn=job_queue.job_queue_arn,
            job_name=f"{stage_name.lower()}-batch",
            container_overrides=tasks.BatchContainerOverrides(
                environment=container_env
            ),
            integration_pattern=sfn.IntegrationPattern.RUN_JOB
        )

        batch_task.add_catch(
            handler=dlq_chain_factory(),  # fresh chain instance
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        # Map state (wired to Batch task)
        params = {
                "manifest.$": "$$.Map.Item.Value",
                "job_id.$": "$.job_id", # keys ending with .$ tell Step Functions “this value comes from a JSONPath expression.”
                "user.$": "$.user",
                "label_types.$": "$.label_types",
                "source.$": "$.source",
                "event_type.$": "$.event_type"
        }

        if extra_map_state_params:
            params.update(extra_map_state_params)

        map_state = sfn.Map(
            self, f"{stage_name}MapState",
            items_path=f"$.{stage_name}.manifests",
            item_selector=params,
            max_concurrency=max(1, min(50, int(ce_maxv_cpus / max(1, config.batch_task_job_def.vcpus)))),
        )

        per_item = sfn.Chain.start(batch_task)
        map_state.iterator(per_item)

        self.batching_task = batching_task
        self.map_state = map_state
        self.job_def = job_def

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
                 phash_table: dynamodb.Table,
                 lock_table: dynamodb.Table,
                 global_dlq: sqs.Queue,
                 athena_database_name: str,
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
        self.phash_table = phash_table
        self.lock_table = lock_table
        self.global_dlq = global_dlq
        self.athena_database_name = athena_database_name
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
            job_table=self.job_table,
            sha256_table=self.sha256_table,
            phash_table=self.phash_table,
            job_queue=job_queue,
            athena_database_name=self.athena_database_name,
            ce_maxv_cpus=CONFIG.compute_env.maxv_cpus,
            region=self.region,
            account=self.account,
            global_dlq=self.global_dlq,
            dlq_chain_factory=self._make_dlq_chain,
            firehose_delivery_stream_name=self.firehose_delivery_stream.ref,
        )

        # Make cleanup lambda
        cleanup_task = self._make_cleanup_task(CONFIG.cleanup_lambda)

        workflow_definition = sfn.Chain.start(validation_stage.batching_task) \
            .next(validation_stage.map_state) \
            .next(cleanup_task)

        upload_state_machine = sfn.StateMachine(self, "UploadStateMachine",
                              definition_body=sfn.DefinitionBody.from_chainable(workflow_definition),
                              timeout=Duration.hours(CONFIG.upload_state_machine.duration_hours)
                              )

        upload_state_machine.role.add_to_principal_policy(
            iam.PolicyStatement(actions=["sqs:SendMessage"], resources=[self.global_dlq.queue_arn]))

        upload_state_machine.apply_removal_policy(RemovalPolicy.DESTROY)

        for stage in [validation_stage]:#, internal_dedup_stage, external_dedup_stage, faiss_registration_stage, label_enrichment_stage]:
            # grant the state machine's role permission to submit that stage's job
            stage.job_def.grant_submit_job(upload_state_machine.role, job_queue)

        # Make kickoff lambda to trigger on job.json upload
        self._make_kickoff_lambda(upload_state_machine, CONFIG.kickoff_lambda)

    def _make_dlq_chain(self) -> sfn.Chain:
        suffix = uuid.uuid4().hex[:8]
        send_to_dlq = tasks.CallAwsService(
            self, f"SendToDLQ_{suffix}",
            service="sqs",
            action="sendMessage",
            parameters={
                "QueueUrl": self.global_dlq.queue_url,
                "MessageBody.$": 'States.JsonToString({"job_id": $.job_id, "user": $.user, "error": $.errorInfo})'
            },
            iam_resources=[self.global_dlq.queue_arn],
        )
        send_to_dlq_fail = sfn.Fail(self, f"SendToDLQFail_{suffix}", cause="StepFailed", error="StepError")
        return sfn.Chain.start(send_to_dlq).next(send_to_dlq_fail)

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
        # An IAM Role assumes by AWS Batch itself (not the jobs)
        # The purpose is to let Batch manage the compute environment: creating, scaling, managing clusters
        batch_service_role = iam.Role(
            self, "BatchServiceRole",
            assumed_by=iam.ServicePrincipal("batch.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSBatchServiceRole")]
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
            service_role=batch_service_role,  # IAM Role for the compute environment
            instance_role=instance_role
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
                           cleanup_config: CleanupLambdaConfig):
        cleanup_lambda = _lambda.Function(
            self,
            "CleanupLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=cleanup_config.handler,
            code=_lambda.Code.from_asset(cleanup_config.path),
            layers = [self.common_utils_layer],
            dead_letter_queue=self.global_dlq,
            memory_size=cleanup_config.memory_size,
            timeout=Duration.seconds(cleanup_config.timeout_sec),
            environment={
                "JOB_TABLE_NAME": self.job_table.table_name,
                "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
                "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref
            }
        )

        # Permissions for the kickoff lambda
        self.job_table.grant_read_write_data(cleanup_lambda)
        self.file_bucket.grant_read(cleanup_lambda)

        # ensure S3 bucket-level list and get-location are permitted
        cleanup_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
        ))

        cleanup_task = tasks.LambdaInvoke(
            self, "CleanupTask",
            lambda_function=cleanup_lambda,
            result_path="$.cleanup",
            output_path="$",
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
                            kickoff_config: KickoffLambdaConfig):
            # Make Kickoff lambda
            kickoff_lambda = _lambda.Function(
                self,
                "KickoffLambda",
                runtime=_lambda.Runtime.PYTHON_3_11,
                handler=kickoff_config.handler,
                code=_lambda.Code.from_asset(kickoff_config.path),
                layers=[self.common_utils_layer],
                dead_letter_queue=self.global_dlq,
                memory_size=kickoff_config.memory_size,
                timeout=Duration.seconds(kickoff_config.timeout_sec),
                environment={
                    "JOB_TABLE_NAME": self.job_table.table_name,
                    "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
                    "UPLOAD_STATE_MACHINE_ARN": upload_state_machine.state_machine_arn,
                    "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref
                }
            )

            upload_state_machine.grant_start_execution(kickoff_lambda)

            # Permissions for the kickoff lambda
            self.job_table.grant_read_write_data(kickoff_lambda)
            self.file_bucket.grant_read(kickoff_lambda)

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