from config import CONFIG
from config_models import ComputeEnvConfig
from stacks.upload_stack_utils import BatchingStage

from aws_cdk import (
    Stack,
    Duration,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_s3_notifications as s3n,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_sqs as sqs,
    aws_logs as logs,
    aws_dynamodb as dynamodb,
)
from constructs import Construct

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
                 app_log_group: logs.LogGroup,
                 **kwargs) -> None:

        super().__init__(scope, construct_id, **kwargs)

        # Creates Batch compute environment and job queue pointing to the compute environment.
        job_queue = self._make_compute_env(CONFIG.compute_env)

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
        self.app_log_group = app_log_group

        validation_stage = BatchingStage(
            self, "validationStage",
            stage_name="validationStage",
            config=CONFIG.validation,
            file_bucket=self.file_bucket,
            job_table=self.job_table,
            log_group=self.app_log_group,
            sha256_table=self.sha256_table,
            phash_table=self.phash_table,
            job_queue=job_queue,
            athena_database_name=self.athena_database_name,
            region=self.region,
            account=self.account,
            global_dlq=self.global_dlq
        )

        internal_dedup_stage = BatchingStage(
            self, "ExternalDedup",
            stage_name="ExternalDedup",
            config=CONFIG['external_dedup'],
            file_bucket=self.file_bucket,
            job_table=self.job_table,
            log_group=self.app_log_group,
            sha256_table=self.sha256_table,
            phash_table=self.phash_table,
            job_queue=job_queue,
            athena_database=self.athena_database,
            region=self.region,
            account=self.account,
            global_dlq=self.global_dlq
        )

        external_dedup_stage = BatchingStage(
            self, "ExternalDedup",
            stage_name="ExternalDedup",
            config=CONFIG['external_dedup'],
            file_bucket=self.file_bucket,
            job_table=self.job_table,
            log_group=self.app_log_group,
            sha256_table=self.sha256_table,
            phash_table=self.phash_table,
            job_queue=job_queue,
            athena_database=self.athena_database,
            region=self.region,
            account=self.account,
            global_dlq=self.global_dlq
        )

        faiss_registration_stage = BatchingStage(
            self, "ExternalDedup",
            stage_name="ExternalDedup",
            config=CONFIG['external_dedup'],
            file_bucket=self.file_bucket,
            job_table=self.job_table,
            log_group=self.app_log_group,
            sha256_table=self.sha256_table,
            phash_table=self.phash_table,
            job_queue=job_queue,
            athena_database=self.athena_database,
            region=self.region,
            account=self.account,
            global_dlq=self.global_dlq
        )

        label_enrichment_stage = BatchingStage(
            self, "LabelEnrichment",
            stage_name="LabelEnrichment",
            config=CONFIG['label_enrichment'],
            file_bucket=self.file_bucket,
            job_table=self.job_table,
            log_group=self.app_log_group,
            sha256_table=self.sha256_table,
            phash_table=self.phash_table,
            job_queue=job_queue,
            athena_database=self.athena_database,
            region=self.region,
            account=self.account,
            global_dlq=self.global_dlq,
            extra_env={
                "CANONICAL_LABELS_TABLE": self.canonical_labels_table.table_name
            },
            extra_permissions=[
                iam.PolicyStatement(
                    actions=["dynamodb:PutItem", "dynamodb:UpdateItem"],
                    resources=[self.canonical_labels_table.table_arn]
                )
            ],
            extra_container_env={
                "CANONICAL_LABELS_TABLE": self.canonical_labels_table.table_name
            }
        )

        # Make cleanup lambda
        cleanup_task = self._make_cleanup_task()

        workflow_definition = validation_stage.batching_task
                .next(validation_stage.map_state)
                .next(internal_dedup_stage.batching_task)
                .next(internal_dedup_stage.map_state)
                .next(external_dedup_stage.batching_task)
                .next(external_dedup_stage.map_state)
                .next(faiss_registration_stage.batching_task)
                .next(faiss_registration_stage.map_state)
                .next(label_enrichment_stage.batching_task)
                .next(label_enrichment_stage.map_state)
                .next(cleanup_task)

        upload_state_machine = sfn.StateMachine(
            self,
            "UploadStateMachine",
            definition=workflow_definition,
            timeout=Duration.hours(CONFIG.upload_state_machine.duration_hours)
        )

        # Make kickoff lambda to trigger on job.json upload
        self._make_kickoff_lambda(upload_state_machine)

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
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonEC2ContainerServiceforEC2Role")
            ]
        )

        # The mechanism that attaches the instance role to EC2 at launch.
        instance_profile = iam.CfnInstanceProfile(self, "BatchInstanceProfile", roles=[instance_role.role_name])

        # Make the compute environment
        compute_env = batch.ComputeEnvironment(
            self, "ComputeEnv",
            service_role=batch_service_role,
            compute_resources=batch.ComputeResources(
                type=batch.ComputeResourceType.SPOT,
                allocation_strategy=batch.AllocationStrategy.SPOT_PRICE_CAPACITY_OPTIMIZED,
                vpc=vpc,
                minv_cpus=ce_config.minv_cpus,
                desiredv_cpus=ce_config.desiredv_cpus,
                maxv_cpus=ce_config.maxv_cpus,
                instance_role=instance_profile.attr_arn,  # attach instance profile
                instance_types=[ec2.InstanceType(i) for i in ce_config.instance_types],
                vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),  # or PRIVATE_WITH_EGRESS for prod
                security_groups=[batch_sg]
            )
        )

        # The job queue for the above compute environment.
        job_queue = batch.JobQueue(
                                self,
                            "JobQueue",
                                priority=1, # If multiple queues share the compute env, this queue takes first priority.
                                compute_environments=[
                                    batch.JobQueueComputeEnvironment(
                                        compute_environment=compute_env,
                                        order=1 # Ensures this is the first and only compute environment batch will try.
                                    )
                                ]
                            )
        return job_queue

    def _make_cleanup_task(self):
        cleanup_lambda = None

        cleanup_task = tasks.LambdaInvoke(
            self, "CleanupTask",
            lambda_function=cleanup_lambda,
            output_path="$.Payload"
        )

        return cleanup_task

    def _make_kickoff_lambda(self,
                            upload_state_machine):
            # Make Kickoff lambda
            kickoff_lambda = _lambda.Function(
                self,
                "KickoffLambda",
                runtime=_lambda.Runtime.PYTHON_3_11,
                handler=CONFIG.kickoff_lambda.handler,
                code=_lambda.Code.from_asset(CONFIG.kickoff_lambda.path),
                dead_letter_queue=self.global_dlq,
                log_group=self.app_log_group,
                memory_size=CONFIG.kickoff_lambda.memory_size,
                timeout=Duration.seconds(CONFIG.kickoff_lambda.timeout_sec),
                environment={
                    "JOB_TABLE_NAME": self.job_table.table_name,
                    "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
                    "UPLOAD_STATE_MACHINE_ARN": upload_state_machine.state_machine_arn
                }
            )

            upload_state_machine.grant_start_execution(kickoff_lambda)

            # Permissions for the kickoff lambda
            self.job_table.grant_read_write_data(kickoff_lambda)
            self.app_log_group.grant_write(kickoff_lambda)
            self.file_bucket.grant_read(kickoff_lambda)

            # ensure S3 bucket-level list and get-location are permitted
            kickoff_lambda.add_to_role_policy(
                iam.PolicyStatement(
                    actions=[
                        "s3:ListBucket",
                        "s3:GetBucketLocation"
                    ],
                    resources=[
                        f"arn:aws:s3:::{self.file_bucket.bucket_name}"
                    ]
                )
            )

            # explicitly allow GetObject on the athena-results prefix only if you will read it;
            # otherwise GetObject on whole bucket is already covered by grant_read above.
            kickoff_lambda.add_to_role_policy(
                iam.PolicyStatement(
                    actions=[
                        "s3:GetObject"
                    ],
                    resources=[
                        f"arn:aws:s3:::{self.file_bucket.bucket_name}/*"
                    ]
                )
            )

            # Trigger: S3 event for job.json
            self.file_bucket.add_event_notification(
                s3.EventType.OBJECT_CREATED,
                s3n.LambdaDestination(kickoff_lambda,
                                      dead_letter_queue=self.global_dlq),
                s3.NotificationKeyFilter(prefix="temp/image-upload/", suffix="job.json")
            )