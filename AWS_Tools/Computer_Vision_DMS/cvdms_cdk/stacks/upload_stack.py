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

CONFIG = {'compute_env': {'minv_cpus': 0,
                          'desiredv_cpus': 0,
                          'maxv_cpus': 64,
                          'instance_types': ['m5.large', 'm5.xlarge']},
          'upload_state_machine': {'duration_hours': 2},
          'validation': {'file_batching':{'path':'lambdas/upload/batching',
                                          'handler': 'file_batching_validation.handler',
                                          'memory_size': 512,
                                          'timeout_min': 5},
                         'batch_task_job_def': {'vcpus': 1,
                                                'memory_limit_mib': 2048,
                                                'path':'lambdas/upload/validation'}},
          'internal_dedup':{'file_batching':{'path':'lambdas/upload/batching',
                                              'handler': 'file_batching_internal_dedup.handler',
                                              'memory_size': 512,
                                              'timeout_min': 5},
                            'batch_task_job_def': {'vcpus': 1,
                                                   'memory_limit_mib': 2048,
                                                   'path': 'lambdas/upload/internal_dedup'}
                            },
          'external_dedup':{},
          'registration':{},
          'kickoff_lambda':{'path':'lambdas/upload/kickoff',
                            'handler': 'kickoff.handler',
                            'memory_size': 512,
                            'timeout_sec': 30}
          }


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
                 athena_database: str,
                 app_log_group: logs.LogGroup,
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
        self.athena_database = athena_database
        self.app_log_group = app_log_group

        ##############################################################
        # Make the validation workflow
        ##############################################################

        # Gives job_queue pointing to the compute environment
        job_queue = self._make_compute_env()

        # Gives file_batching_lambda_validation, a lambda function to batch up files for validation
        file_batching_lambda_validation = self._make_file_batching_lambda_validation()

        # Gives file_batching_task_validation, a task that invokes file_batching_lambda_validation
        file_batching_task_validation = self._make_file_batching_task_validation(file_batching_lambda_validation)

        # Gives validation_job_role, an IAM role assumed by ECS that the validation task needs
        validation_job_role = self._make_validation_job_role()

        # Gives batch_task_validation, a task for submitting Batch jobs.
        batch_task_validation = self._make_batch_task_validation(validation_job_role,
                                                                 job_queue)

        # Gives map_state_validation, the initial map state that iterates over the batch jobs
        map_state_validation = self._make_map_state_validation()

        # Gives file_batching_lambda_internal_dedup, a lambda function to batch up files for internal deduplication
        file_batching_lambda_internal_dedup = self._make_file_batching_lambda_internal_dedup()

        # Gives file_batching_task_internal_dedup, a task that invokes file_batching_lambda_internal_dedup
        file_batching_task_internal_dedup = self._make_file_batching_task_internal_dedup(file_batching_lambda_internal_dedup)

        internal_dedup_job_role = self._make_internal_dedup_job_role()

        batch_task_internal_dedup = self._make_batch_task_internal_dedup(internal_dedup_job_role,
                                                                          job_queue)

        map_state_internal_dedup = self._make_map_state_internal_dedup()

        # Gives workflow_definition, the definition to feed into the state machin construction next.
        workflow_definition = self._make_workflow_definition(map_state_validation,
                                                              batch_task_validation,
                                                              file_batching_task_validation,
                                                              file_batching_task_internal_dedup,
                                                             map_state_internal_dedup,
                                                             batch_task_internal_dedup)

        # Gives upload_state_machine using self.workflow_definition
        upload_state_machine = self._make_upload_state_machine(workflow_definition)

        # Last step: Kickoff lambda starts the state machine via an S3 event.
        kickoff_lambda = self._make_kickoff_lambda(upload_state_machine)

    def _make_compute_env(self):
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
                minv_cpus=CONFIG['compute_env']['minv_cpus'],
                desiredv_cpus=CONFIG['compute_env']['desiredv_cpus'],
                maxv_cpus=CONFIG['compute_env']['maxv_cpus'],
                instance_role=instance_profile.attr_arn,  # attach instance profile
                instance_types=[ec2.InstanceType(i) for i in CONFIG['compute_env']['instance_types']],
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

    def _make_file_batching_lambda_validation(self):
        file_batching_lambda_validation = _lambda.Function(
            self,
            "FileBatchingLambdaValidation",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=CONFIG['validation']['file_batching']['handler'],
            code=_lambda.Code.from_asset(CONFIG['validation']['file_batching']['path']),
            dead_letter_queue=self.global_dlq,
            log_group=self.app_log_group,
            memory_size=CONFIG['validation']['file_batching']['memory_size'],
            timeout=Duration.minutes(CONFIG['validation']['file_batching']['timeout_min']),
            environment={
                "FILE_BUCKET_NAME": self.file_bucket.bucket_name
            }
        )

        # Permissions for batching lambda
        self.file_bucket.grant_read_write(file_batching_lambda_validation)
        self.app_log_group.grant_write(file_batching_lambda_validation)

        file_batching_lambda_validation.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[self.file_bucket.bucket_arn]
            )
        )

        return file_batching_lambda_validation

    def _make_file_batching_task_validation(self, file_batching_lambda_validation):
        file_batching_task_validation = tasks.LambdaInvoke(
            self,
            "CreateManifests",
            lambda_function=file_batching_lambda_validation,
            output_path="$.Payload"
        )

        return file_batching_task_validation

    def _make_validation_job_role(self):
        validation_job_role = iam.Role(
            self, "ValidationJobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )

        self.file_bucket.grant_read(validation_job_role)
        self.job_table.grant_read_write_data(validation_job_role)
        self.app_log_group.grant_write(validation_job_role)

        return validation_job_role

    def _make_batch_task_validation(self,
                                    validation_job_role,
                                    job_queue):
        batch_job_def = batch.JobDefinition(
            self, "ValidationJobDef",
            container=batch.JobDefinitionContainer(
                image=batch.EcrImage.from_asset(CONFIG['validation']['batch_task_job_def']['path']),
                vcpus=CONFIG['validation']['batch_task_job_def']['vcpus'],
                memory_limit_mib=CONFIG['validation']['batch_task_job_def']['memory_limit_mib'],
                job_role=validation_job_role
            )
        )

        batch_task_validation = tasks.BatchSubmitJob(
            self,
            "ValidateBatch",
            job_definition=batch_job_def,
            job_queue=job_queue,
            job_name="validate-batch",
            container_overrides=tasks.BatchContainerOverrides(
                environment={
                    "MANIFEST_S3_KEY": sfn.JsonPath.string_at("$.manifest"),
                    "JOB_ID": sfn.JsonPath.string_at("$.job_id"),
                    "USER": sfn.JsonPath.string_at("$.user"),
                    "JOB_TYPE": sfn.JsonPath.string_at("$.job_type"),
                    "LABEL_TYPE": sfn.JsonPath.string_at("$.label_type"),
                    'FILE_BUCKET_NAME': self.file_bucket.bucket_name,
                    'ATHENA_OUTPUT_S3':f"s3://{self.file_bucket.bucket_name}/athena-results/",
                    'ATHENA_WORKGROUP':"primary",
                    'ICEBERG_DB': self.athena_database,
                    'UPLOAD_STAGING_TABLE':"upload_staging"
                }
            ),
            integration_pattern=sfn.IntegrationPattern.RUN_JOB
        )

        return batch_task_validation

    def _make_map_state_validation(self):
        map_state_validation = sfn.Map(
            self,
            "ProcessBatches",
            items_path="$.manifests",
            parameters={
                # $$MAP_ITEM is the current array element (the manifest string)
                "manifest.$": "$$MAP_ITEM",
                # assign manifest key in the iteration to the s3 uri pointing to the manifest.
                # pull job_id from the parent scope and assign to job_id an iteration
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "job_type.$": "$.job_type",
                "label_type.$": "$.label_type"
            }
        )

        return map_state_validation

    def _make_workflow_definition(self,
                                  map_state_validation,
                                  batch_task_validation,
                                  file_batching_task_validation,
                                  file_batching_task_internal_dedup,
                                  map_state_internal_dedup,
                                  batch_task_internal_dedup):

        map_state_validation.iterator(batch_task_validation)
        map_state_internal_dedup.iterator(batch_task_internal_dedup)

        return (
            file_batching_task_validation
            .next(map_state_validation)
            .next(file_batching_task_internal_dedup)
            .next(map_state_internal_dedup)
        )

    def _make_upload_state_machine(self, workflow_definition):
        upload_state_machine = sfn.StateMachine(
            self,
            "UploadStateMachine",
            definition=workflow_definition,
            timeout=Duration.hours(CONFIG['upload_state_machine']['duration_hours'])
        )

        return upload_state_machine

    def _make_file_batching_lambda_internal_dedup(self):
        file_batching_lambda_internal_dedup = _lambda.Function(
            self,
            "FileBatchingLambdaInternalDedup",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=CONFIG['internal_dedup']['file_batching']['handler'],
            code=_lambda.Code.from_asset(CONFIG['internal_dedup']['file_batching']['path']),
            dead_letter_queue=self.global_dlq,
            log_group=self.app_log_group,
            memory_size=CONFIG['internal_dedup']['file_batching']['memory_size'],
            timeout=Duration.minutes(CONFIG['internal_dedup']['file_batching']['timeout_min']),
            environment={
                "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
                "ATHENA_OUTPUT_S3":f"s3://{self.file_bucket.bucket_name}/athena-results/",
                "ATHENA_WORKGROUP": "primary",
                "ICEBERG_DB":self.athena_database,
                "UPLOAD_STAGING_TABLE":"upload_staging"
            }
        )

        # Permissions for batching lambda
        self.file_bucket.grant_read_write(file_batching_lambda_internal_dedup)
        self.app_log_group.grant_write(file_batching_lambda_internal_dedup)

        file_batching_lambda_internal_dedup.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[self.file_bucket.bucket_arn]
            )
        )

        file_batching_lambda_internal_dedup.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults"
                ],
                resources=[
                    f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"]
            )
        )

        return file_batching_lambda_internal_dedup

    def _make_file_batching_task_internal_dedup(self,
                                               file_batching_lambda_internal_dedup):
        file_batching_task_internal_dedup = tasks.LambdaInvoke(
            self,
            "CreateManifestsInternalDedup",
            lambda_function=file_batching_lambda_internal_dedup,
            output_path="$.Payload"
        )

        return file_batching_task_internal_dedup

    def _make_internal_dedup_job_role(self):
        internal_dedup_job_role = iam.Role(
            self, "InternalDedupJobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )

        self.file_bucket.grant_read_write(internal_dedup_job_role)
        self.job_table.grant_read_write_data(internal_dedup_job_role)
        self.app_log_group.grant_write(internal_dedup_job_role)

        internal_dedup_job_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults"
                ],
                resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"]
            )
        )

        return internal_dedup_job_role

    def _make_batch_task_internal_dedup(self,
                                    internal_dedup_job_role,
                                    job_queue):
        batch_job_def = batch.JobDefinition(
            self, "InternalDedupJobDef",
            container=batch.JobDefinitionContainer(
                image=batch.EcrImage.from_asset(CONFIG['internal_dedup']['batch_task_job_def']['path']),
                vcpus=CONFIG['internal_dedup']['batch_task_job_def']['vcpus'],
                memory_limit_mib=CONFIG['internal_dedup']['batch_task_job_def']['memory_limit_mib'],
                job_role=internal_dedup_job_role
            )
        )

        batch_task_internal_dedup = tasks.BatchSubmitJob(
            self,
            "InternalDedupBatch",
            job_definition=batch_job_def,
            job_queue=job_queue,
            job_name="internal-dedup-batch",
            container_overrides=tasks.BatchContainerOverrides(
                environment={
                    "MANIFEST_S3_KEY": sfn.JsonPath.string_at("$.manifest"),
                    "JOB_ID": sfn.JsonPath.string_at("$.job_id"),
                    "USER": sfn.JsonPath.string_at("$.user"),
                    "JOB_TYPE": sfn.JsonPath.string_at("$.job_type"),
                    "LABEL_TYPE": sfn.JsonPath.string_at("$.label_type"),
                    'FILE_BUCKET_NAME': self.file_bucket.bucket_name,
                    'ATHENA_OUTPUT_S3':f"s3://{self.file_bucket.bucket_name}/athena-results/",
                    'ATHENA_WORKGROUP':"primary",
                    'ICEBERG_DB': self.athena_database,
                    'UPLOAD_STAGING_TABLE':"upload_staging"
                }
            ),
            integration_pattern=sfn.IntegrationPattern.RUN_JOB
        )

        return batch_task_internal_dedup

    def _make_map_state_internal_dedup(self):
        map_state_internal_dedup = sfn.Map(
            self,
            "ProcessBatchesInternalDedup",
            items_path="$.manifests",
            parameters={
                "manifest.$": "$$MAP_ITEM",
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "job_type.$": "$.job_type",
                "label_type.$": "$.label_type"
            }
        )

        return map_state_internal_dedup



















    def _make_kickoff_lambda(self,
                             upload_state_machine):
        # Make Kickoff lambda
        kickoff_lambda = _lambda.Function(
            self,
            "KickoffLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=CONFIG['kickoff_lambda']['handler'],
            code=_lambda.Code.from_asset(CONFIG['kickoff_lambda']['path']),
            dead_letter_queue=self.global_dlq,
            log_group=self.app_log_group,
            memory_size=CONFIG['kickoff_lambda']['memory_size'],
            timeout=Duration.seconds(CONFIG['kickoff_lambda']['timeout_sec']),
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

        return kickoff_lambda
