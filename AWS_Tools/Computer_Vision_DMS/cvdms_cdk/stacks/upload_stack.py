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
                 lock_table: dynamodb.Table,
                 global_dlq: sqs.Queue,
                 athena_database: str,
                 app_log_group: logs.LogGroup,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

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
                minv_cpus=0,
                desiredv_cpus=0,
                maxv_cpus=64,
                instance_role=instance_profile.attr_arn,  # attach instance profile
                instance_types=[ec2.InstanceType("m5.large"), ec2.InstanceType("m5.xlarge")],
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

        # Make the validation lambda
        validation_lambda = _lambda.DockerImageFunction(
            self,
            "ValidationLambda",
            code=_lambda.DockerImageCode.from_image_asset("lambdas/upload/validation"),
            memory_size=1024,
            timeout=cdk.Duration.minutes(15),
            log_group=app_log_group  # 👈 force logs into the shared group
        )

        validation_lambda = _lambda.Function(
                                            self,
                                            "ValidationLambda",
                                            runtime=_lambda.Runtime.PYTHON_3_12,
                                            handler="validation.handler",
                                            code=_lambda.Code.from_asset(f"lambdas/upload/validation"),
                                            timeout=Duration.minutes(15),
                                            environment={
                                                    "FILES_BUCKET": file_bucket.bucket_name,
                                                    "SHA256_TABLE_NAME": sha256_table.table_name,
                                                    "JOB_TABLE_NAME": job_table.table_name,
                                                    "ICEBERG_BUCKET_NAME": iceberg_bucket.bucket_name,
                                                     }
                                            )

        label_lambda = mk_lambda(
            "LabelEnrichmentLambda", "label_enrichment",
            {
                "FILES_BUCKET": file_bucket.bucket_name,
                "SHA256_TABLE": sha256_table.table_name,
                "JOB_TABLE": job_table.table_name,
            }
        )

        reg_lambda = mk_lambda(
            "RegistrationLambda", "registration",
            {
                "FILES_BUCKET": file_bucket.bucket_name,
                "SHA256_TABLE": sha256_table.table_name,
                "LOCK_TABLE": lock_table.table_name,
                "JOB_TABLE": job_table.table_name,
            }
        )

        cleanup_lambda = mk_lambda(
            "CleanupLambda", "cleanup",
            {"FILES_BUCKET": file_bucket.bucket_name},
            timeout=2
        )

        # S3 event -> Validation Lambda
        file_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(validation_lambda),
            s3.NotificationKeyFilter(prefix="temp/image-upload/", suffix="job.json")
        )




        # Use the default ECS instance role (already has AmazonEC2ContainerServiceforEC2Role)
        # CDK will automatically create an instance profile for you if you don't specify one.


        # Example job definition
        job_role = iam.Role(
            self, "BatchJobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )

        # Grants for job role
        file_bucket.grant_read_write(job_role)
        iceberg_bucket.grant_read_write(job_role)
        sha256_table.grant_read_write_data(job_role)
        job_table.grant_read_write_data(job_role)
        lock_table.grant_read_write_data(job_role)

        job_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "glue:GetDatabase",
                "glue:GetTables",
                "glue:GetTable",
                "glue:UpdateTable",
                "glue:CreateTable",
                "glue:GetPartitions",
            ],
            resources=[
                f"arn:aws:glue:{self.region}:{self.account}:catalog",
                f"arn:aws:glue:{self.region}:{self.account}:database/{athena_database}",
                f"arn:aws:glue:{self.region}:{self.account}:table/{athena_database}/*",
            ],
        ))

        job_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "athena:StartQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "athena:GetWorkGroup",
            ],
            resources=[
                f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"
            ],
        ))

        internal_job = batch.JobDefinition(
            self, "InternalDedupJob",
            container=batch.JobDefinitionContainer(
                image=batch.EcrImage.from_registry("ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/internal-dedup:latest"),
                vcpus=2,
                memory_limit_mib=4096,
                job_role=job_role
            )
        )

        external_job = batch.JobDefinition(
            self, "ExternalDedupJob",
            container=batch.JobDefinitionContainer(
                image=batch.EcrImage.from_registry("ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/external-dedup:latest"),
                vcpus=2,
                memory_limit_mib=4096,
                job_role=job_role
            )
        )

        # ------------------------------
        # Step Functions workflow + DLQ
        # ------------------------------
        def catch(task):
            return task.add_catch(
                tasks.SqsSendMessage(
                    self, "SendToDLQ",
                    queue=global_dlq,
                    message_body=sfn.TaskInput.from_json_path_at("$.Error")
                ),
                result_path="$.Error"
            )

        definition = (
            catch(tasks.LambdaInvoke(self,"RunValidation",lambda_function=validation_lambda,output_path="$.Payload"))
            .next(catch(tasks.BatchSubmitJob(self,"RunInternalDedup",job_definition=internal_job,job_name="internal",job_queue=job_queue)))
            .next(catch(tasks.BatchSubmitJob(self,"RunExternalDedup",job_definition=external_job,job_name="external",job_queue=job_queue)))
            .next(catch(tasks.LambdaInvoke(self,"RunLabel",lambda_function=label_lambda,output_path="$.Payload")))
            .next(catch(tasks.LambdaInvoke(self,"RunRegistration",lambda_function=reg_lambda,output_path="$.Payload")))
            .next(catch(tasks.LambdaInvoke(self,"RunCleanup",lambda_function=cleanup_lambda,output_path="$.Payload")))
        )

        sfn.StateMachine(
            self, "ImageUploadWorkflow",
            definition=definition,
            timeout=Duration.hours(2)
        )

        # ------------------------------
        # IAM grants for Lambdas
        # ------------------------------
        sha256_table.grant_read_write_data(validation_lambda)
        sha256_table.grant_read_write_data(reg_lambda)
        job_table.grant_read_write_data(validation_lambda)
        lock_table.grant_read_write_data(reg_lambda)

        file_bucket.grant_read(validation_lambda)
        file_bucket.grant_read_write(label_lambda)
        file_bucket.grant_read_write(reg_lambda)
        file_bucket.grant_read_write(cleanup_lambda)

        for fn in [validation_lambda, label_lambda, reg_lambda]:
            # Glue: restrict to your catalog + specific database + its tables
            fn.add_to_role_policy(iam.PolicyStatement(
                actions=[
                    "glue:GetDatabase",
                    "glue:GetTables",
                    "glue:GetTable",
                    "glue:GetPartitions",
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{athena_database}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{athena_database}/*",
                ],
            ))

            # Athena: restrict to the workgroup you actually use
            fn.add_to_role_policy(iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:GetWorkGroup",
                ],
                resources=[
                    f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"
                ],
            ))

            # S3: restrict to your file and iceberg buckets
            fn.add_to_role_policy(iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                resources=[
                    file_bucket.bucket_arn,
                    f"{file_bucket.bucket_arn}/*",
                    iceberg_bucket.bucket_arn,
                    f"{iceberg_bucket.bucket_arn}/*",
                ],
            ))