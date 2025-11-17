from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
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
    aws_kinesisfirehose as firehose
)
from constructs import Construct

from config import CONFIG
from config_models import ComputeEnvConfig, KickoffLambdaConfig, CleanupLambdaConfig
from stacks.upload_stack_utils import BatchingStage

class ImageUploadStack(Stack):
    def __init__(self,
                 scope: Construct,
                 construct_id: str,
                 app_name: str,
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
        # Note:  use firehose_delivery_stream.ref (name) or firehose_delivery_stream.attr_arn (ARN)

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
        # Creates Batch compute environment and job queue pointing to the compute environment.
        # job_queue = self._make_compute_env(CONFIG.compute_env)

        # validation_stage = BatchingStage(
        #     self, "validationStage",
        #     stage_name="validationStage",
        #     config=CONFIG.validation,
        #     file_bucket=self.file_bucket,
        #     job_table=self.job_table,
        #     sha256_table=self.sha256_table,
        #     phash_table=self.phash_table,
        #     job_queue=job_queue,
        #     athena_database_name=self.athena_database_name,
        #     ce_maxv_cpus=CONFIG.compute_env.maxv_cpus,
        #     region=self.region,
        #     account=self.account,
        #     global_dlq=self.global_dlq
        # )
        #
        # internal_dedup_stage = BatchingStage(
        #     self, "internalDedupStage",
        #     stage_name="internalDedupStage",
        #     config=CONFIG.internal_dedup,
        #     file_bucket=self.file_bucket,
        #     job_table=self.job_table,
        #     sha256_table=self.sha256_table,
        #     phash_table=self.phash_table,
        #     job_queue=job_queue,
        #     athena_database_name=self.athena_database_name,
        #     ce_maxv_cpus=CONFIG.compute_env.maxv_cpus,
        #     region=self.region,
        #     account=self.account,
        #     global_dlq=self.global_dlq
        # )
        #
        #
        # external_dedup_stage = BatchingStage(
        #     self, "externalDedupStage",
        #     stage_name="externalDedupStage",
        #     config=CONFIG.external_dedup,
        #     file_bucket=self.file_bucket,
        #     job_table=self.job_table,
        #     sha256_table=self.sha256_table,
        #     phash_table=self.phash_table,
        #     job_queue=job_queue,
        #     athena_database_name=self.athena_database_name,
        #     ce_maxv_cpus=CONFIG.compute_env.maxv_cpus,
        #     region=self.region,
        #     account=self.account,
        #     global_dlq=self.global_dlq,
        #     extra_container_env={'IMG_TYPE':sfn.JsonPath.string_at("$.img_type")},
        #     extra_map_state_params={"img_type.$": "$.img_type"}
        # )
        #
        # faiss_registration_stage = BatchingStage(
        #     self, "faissRegistrationStage",
        #     stage_name="faissRegistrationStage",
        #     config=CONFIG.faiss_registration,
        #     file_bucket=self.file_bucket,
        #     job_table=self.job_table,
        #     sha256_table=self.sha256_table,
        #     phash_table=self.phash_table,
        #     job_queue=job_queue,
        #     athena_database_name=self.athena_database_name,
        #     ce_maxv_cpus=CONFIG.compute_env.maxv_cpus,
        #     region=self.region,
        #     account=self.account,
        #     global_dlq=self.global_dlq,
        # )
        #
        # label_enrichment_stage = BatchingStage(
        #     self, "labelEnrichmentStage",
        #     stage_name="labelEnrichmentStage",
        #     config=CONFIG.label_enrichment,
        #     file_bucket=self.file_bucket,
        #     job_table=self.job_table,
        #     sha256_table=self.sha256_table,
        #     phash_table=self.phash_table,
        #     job_queue=job_queue,
        #     athena_database_name=self.athena_database_name,
        #     ce_maxv_cpus=CONFIG.compute_env.maxv_cpus,
        #     region=self.region,
        #     account=self.account,
        #     global_dlq=self.global_dlq,
        # )
        #
        # # Make cleanup lambda
        # cleanup_task = self._make_cleanup_task(CONFIG.cleanup_lambda)

        # workflow_definition = (
        #     validation_stage.batching_task
        #         .next(validation_stage.map_state)
        #         .next(internal_dedup_stage.batching_task)
        #         .next(internal_dedup_stage.map_state)
        #         .next(external_dedup_stage.batching_task)
        #         .next(external_dedup_stage.map_state)
        #         .next(faiss_registration_stage.batching_task)
        #         .next(faiss_registration_stage.map_state)
        #         .next(label_enrichment_stage.batching_task)
        #         .next(label_enrichment_stage.map_state)
        #         .next(cleanup_task)
        # )

        ta_first_step_lambda = _lambda.Function(
            self, "taFirstStepSM",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="ta_first_step.handler",
            code=_lambda.Code.from_asset("workers/lambdas"),
            timeout=Duration.minutes(10),
            environment={
                    "ICEBERG_BUCKET_NAME": iceberg_bucket.bucket_name,
                    "ICEBERG_DATABASE_NAME": athena_database_name,
                    "S3_ATHENA_OUTPUT_URI": f"s3://{file_bucket.bucket_name}/athena-results/"
                }
        )
        workflow_definition = (
            ta_first_step_lambda
        )
        upload_state_machine = sfn.StateMachine(self, "UploadStateMachine",
                              definition_body=sfn.DefinitionBody.from_chainable(workflow_definition),
                              timeout=Duration.hours(CONFIG.upload_state_machine.duration_hours)
                              )

        upload_state_machine.apply_removal_policy(RemovalPolicy.DESTROY)

        # after upload_state_machine is created
        # for stage in (validation_stage, internal_dedup_stage, external_dedup_stage, faiss_registration_stage,
        #               label_enrichment_stage):
        #     # grant the state machine's role permission to submit this stage's job
        #     stage.job_def.grant_submit_job(upload_state_machine.role, job_queue)

        # Make kickoff lambda to trigger on job.json upload
        self._make_kickoff_lambda(upload_state_machine,
                                  CONFIG.kickoff_lambda)

    # def _make_compute_env(self,
    #                       ce_config: ComputeEnvConfig):
    #
    #     # Use the default VPC (public subnets included)
    #     vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)
    #
    #     # Create a security group for Batch instances
    #     batch_sg = ec2.SecurityGroup(
    #         self, "BatchSecurityGroup",
    #         vpc=vpc,
    #         description="Security group for Batch compute environment",
    #         allow_all_outbound=True  # needed so jobs can reach S3/ECR
    #     )
    #     # An IAM Role assumes by AWS Batch itself (not the jobs)
    #     # The purpose is to let Batch manage the compute environment: creating, scaling, managing clusters
    #     batch_service_role = iam.Role(
    #         self, "BatchServiceRole",
    #         assumed_by=iam.ServicePrincipal("batch.amazonaws.com"),
    #         managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSBatchServiceRole")]
    #     )
    #
    #     # The role the EC2 nodes take on, needed to register with clusters, pull containers, etc
    #     # Things for the node to do, not necessarily the job running on it.
    #     instance_role = iam.Role(
    #         self, "BatchInstanceRole",
    #         assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
    #         managed_policies=[
    #             iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonEC2ContainerServiceforEC2Role"),
    #             iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"),
    #         ]
    #     )
    #
    #     # Make the compute environment
    #     compute_env = batch.ManagedEc2EcsComputeEnvironment(
    #         self, "ComputeEnv",
    #         vpc=vpc,
    #         # Spot configuration
    #         spot=True,
    #         allocation_strategy=batch.AllocationStrategy.SPOT_PRICE_CAPACITY_OPTIMIZED,
    #         spot_bid_percentage=100,
    #         # Scaling limits
    #         minv_cpus=ce_config.minv_cpus,
    #         maxv_cpus=ce_config.maxv_cpus,
    #         # Instances
    #         instance_types=[ec2.InstanceType(i) for i in ce_config.instance_types],
    #         security_groups=[batch_sg],
    #         # Optional: restrict subnets
    #         vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC), # ec2.SubnetType.PRIVATE_WITH_EGRESS for prod
    #         # Roles
    #         service_role=batch_service_role,  # IAM Role for the compute environment
    #         instance_role=instance_role
    #     )
    #
    #     # The job queue for the above compute environment.
    #     job_queue = batch.JobQueue(
    #                             self,
    #                         "JobQueue",
    #                             priority=1, # If multiple queues share the compute env, this queue takes first priority.
    #                             compute_environments=[
    #                                 batch.OrderedComputeEnvironment(
    #                                     compute_environment=compute_env,
    #                                     order=1 # Ensures this is the first and only compute environment batch will try.
    #                                 )
    #                             ]
    #                         )
    #     return job_queue
    #
    # def _make_cleanup_task(self,
    #                        cleanup_config: CleanupLambdaConfig):
    #     cleanup_lambda = _lambda.Function(
    #         self,
    #         "CleanupLambda",
    #         runtime=_lambda.Runtime.PYTHON_3_11,
    #         handler=cleanup_config.handler,
    #         code=_lambda.Code.from_asset(cleanup_config.path),
    #         dead_letter_queue=self.global_dlq,
    #         memory_size=cleanup_config.memory_size,
    #         timeout=Duration.seconds(cleanup_config.timeout_sec),
    #         environment={
    #             "JOB_TABLE_NAME": self.job_table.table_name,
    #             "FILE_BUCKET_NAME": self.file_bucket.bucket_name
    #         }
    #     )
    #
    #     # Permissions for the kickoff lambda
    #     self.job_table.grant_read_write_data(cleanup_lambda)
    #     self.file_bucket.grant_read(cleanup_lambda)
    #
    #     # ensure S3 bucket-level list and get-location are permitted
    #     cleanup_lambda.add_to_role_policy(
    #         iam.PolicyStatement(
    #             actions=[
    #                 "s3:ListBucket",
    #                 "s3:GetBucketLocation"
    #             ],
    #             resources=[
    #                 f"arn:aws:s3:::{self.file_bucket.bucket_name}"
    #             ]
    #         )
    #     )
    #
    #     cleanup_task = tasks.LambdaInvoke(
    #         self, "CleanupTask",
    #         lambda_function=cleanup_lambda,
    #         output_path="$.Payload"
    #     )
    #
    #     return cleanup_task

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