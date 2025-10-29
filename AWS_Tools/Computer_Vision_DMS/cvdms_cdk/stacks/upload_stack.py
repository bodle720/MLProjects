from aws_cdk import (
    Stack, Duration,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_s3_notifications as s3n,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_sqs as sqs,
)
from constructs import Construct


class ImageUploadStack(Stack):

    def __init__(self, scope: Construct, id: str, *,
                 file_bucket: s3.Bucket,
                 iceberg_bucket: s3.Bucket,
                 job_table,
                 sha256_table,
                 lock_table,
                 global_dlq,
                 athena_database,
                 **kw):
        super().__init__(scope, id, **kw)

        # ------------------------------
        # Isolated VPC for Batch
        # ------------------------------
        # Use the default VPC (public subnets included)
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)


        # batch_sg = ec2.SecurityGroup(
        #     self, "BatchSG",
        #     vpc=vpc,
        #     allow_all_outbound=True,
        #     description="Security group for Batch compute"
        # )

        # ------------------------------
        # Lambdas (outside VPC)
        # ------------------------------
        def mk_lambda(name, handler, env, timeout=5):
            return _lambda.Function(
                self, name,
                runtime=_lambda.Runtime.PYTHON_3_12,
                handler=f"{handler}.handler",
                code=_lambda.Code.from_asset(f"lambdas/{handler}"),
                timeout=Duration.minutes(timeout),
                environment=env
            )

        validation_lambda = mk_lambda(
            "ValidationLambda", "validation",
            {
                "FILES_BUCKET": file_bucket.bucket_name,
                "SHA256_TABLE": sha256_table.table_name,
                "JOB_TABLE": job_table.table_name,
                "ICEBERG_BUCKET": iceberg_bucket.bucket_name,
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

        # S3 event → Validation Lambda
        file_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(validation_lambda),
            s3.NotificationKeyFilter(prefix="temp/image-upload/", suffix="job.json")
        )

        # ------------------------------
        # Batch: compute environment, queue, roles
        # ------------------------------
        inst_role = iam.Role(
            self, "BatchInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"
                )
            ]
        )

        inst_profile = iam.CfnInstanceProfile(
            self, "BatchInstProf",
            roles=[inst_role.role_name]
        )

        # Use the default ECS instance role (already has AmazonEC2ContainerServiceforEC2Role)
        # CDK will automatically create an instance profile for you if you don't specify one.

        compute_env = batch.ComputeEnvironment(
            self, "CheapComputeEnv",
            compute_resources=batch.ComputeResources(
                type=batch.ComputeResourceType.SPOT,
                allocation_strategy=batch.AllocationStrategy.SPOT_PRICE_CAPACITY_OPTIMIZED,
                vpc=vpc,
                minv_cpus=0,
                desiredv_cpus=0,
                maxv_cpus=32,
                instance_types=[ec2.InstanceType("m5.large")],
                vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
                # no explicit instance_role → Batch uses the default ECS instance role
            )
        )

        job_queue = batch.JobQueue(
            self, "CheapJobQueue",
            compute_environments=[
                batch.JobQueueComputeEnvironment(
                    compute_environment=compute_env,
                    order=1
                )
            ]
        )

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