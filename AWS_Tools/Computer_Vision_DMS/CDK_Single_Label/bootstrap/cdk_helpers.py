# -*- coding: utf-8 -*-
"""
CDK Funtionality.
"""

# app.py
import aws_cdk as cdk
from stacks.image_stack import ImageStack
from aws_cdk import (
    aws_logs as logs,
    aws_lambda as _lambda,
    Duration,
    Stack
)
from constructs import Construct
app = cdk.App()

ImageStack(
    app,
    "ImageInfraStack-dev",
    # You can parameterize these if needed via context or environment
    env=cdk.Environment(account="123456789012", region="us-east-1"),
    predefined_bucket_base_name="my-predefined-datasets-bucket"  # base name, stack adds uniqueness
)

app.synth()

#%%

# stacks/image_stack.py
from aws_cdk import (
    Stack, Duration, RemovalPolicy, CfnOutput,
    aws_dynamodb as dynamodb,
    aws_logs as logs,
    aws_sqs as sqs,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_s3 as s3,
    aws_events as events,
    aws_events_targets as targets,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_ec2 as ec2,
    aws_batch as batch,
)
from constructs import Construct

class ImageStack(Stack):
    def __init__(self, scope: Construct, id: str, predefined_bucket_base_name: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # ----------------------------
        # 1) Core: S3 datasets bucket
        # ----------------------------
        # Bucket name: predefined base + account + region for uniqueness and auditability.
        bucket_name = f"{predefined_bucket_base_name}-{self.account}-{self.region}"
        bucket = s3.Bucket(
            self, "DatasetsBucket",
            bucket_name=bucket_name,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
        )


#%%

        # Explicit log group
        central_log_group = logs.LogGroup(
            self, "CentralLogGroup",
            log_group_name="/my/app/central-logs",
            retention=logs.RetentionDays.ONE_WEEK,  # optional
            removal_policy=cdk.RemovalPolicy.DESTROY  # ensures teardown deletes it
        )

        # Lambda function
        # fn = _lambda.Function(
        #     self, "MyFn",
        #     runtime=_lambda.Runtime.PYTHON_3_11,
        #     handler="app.handler",
        #     code=_lambda.Code.from_asset("lambda_src"),
        #     timeout=Duration.seconds(30),
        #     environment={
        #         "LOG_GROUP_NAME": central_log_group.log_group_name
        #     }
        # )

        # Grant Lambda permission to write to the central log group
        central_log_group.grant_write(fn)
        
#%%
        # Import the existing S3 bucket 
        bucket = s3.Bucket.from_bucket_name(self, "ImportedBucket", bucket_name)

        # ENforce restriction policies
        bucket.add_to_resource_policy(
        iam.PolicyStatement(
            sid="RestrictDatasetRoot",
            effect=iam.Effect.DENY,
            principals=[iam.AnyPrincipal()],
            actions=["s3:*"],
            resources=[f"{bucket.bucket_arn}/{dataset_root}/*"],
            conditions={
                "StringNotLike": {
                    "aws:PrincipalArn": [
                        f"arn:aws:iam::{self.account}:role/ImageInfraStack-dev-*"
                        ]
                    }
                }
            )
        )
    
        bucket.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowTempImages",
                effect=iam.Effect.ALLOW,
                principals=[iam.AnyPrincipal()],
                actions=["s3:PutObject", "s3:GetObject", "s3:AbortMultipartUpload"],
                resources=[f"{bucket.bucket_arn}/{dataset_root}/temp-images/*"]
            )
        )
    

        # Dataset root layout (prefixes). We’ll use these in IAM resource scoping.
        dataset_root = "cv-datasets/single-label/rgb-gray-only"
        images_prefix = f"{dataset_root}/images"
        temp_images_prefix = f"{dataset_root}/temp-images"
        upload_manifests_prefix = f"{dataset_root}/uploads-manifests"
        delete_manifests_prefix = f"{dataset_root}/delete-manifests"

        # S3 ARNs for prefix-scoped permissions (object-level grants)
        bucket_arn = f"arn:aws:s3:::{bucket.bucket_name}"
        images_objects_arn = f"{bucket_arn}/{images_prefix}/*"
        temp_images_objects_arn = f"{bucket_arn}/{temp_images_prefix}/*"
        upload_manifests_objects_arn = f"{bucket_arn}/{upload_manifests_prefix}/*"
        delete_manifests_objects_arn = f"{bucket_arn}/{delete_manifests_prefix}/*"

        # -----------------------------------
        # 2) Centralized CloudWatch log group
        # -----------------------------------
        # Mirrors your boto3 setup: create a single log group and set retention.
        log_group = logs.LogGroup(
            self, "StackLogGroup",
            log_group_name=f"/image-infra/{self.stack_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN
        )

        # ----------------------------
        # 3) DynamoDB tables (3 total)
        # ----------------------------
        # Dataset table: one row per dataset_id; includes 'locked' boolean attribute.
        dataset_table = dynamodb.Table(
            self, "DatasetTable",
            partition_key=dynamodb.Attribute(name="dataset_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN
        )

        # Imagery table: partition key dataset_unique_id; two GSIs for dataset_id+unique_id and unique_id alone.
        imagery_table = dynamodb.Table(
            self, "ImageryTable",
            partition_key=dynamodb.Attribute(name="dataset_unique_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN
        )
        imagery_table.add_global_secondary_index(
            index_name="DatasetIndex",
            partition_key=dynamodb.Attribute(name="dataset_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="unique_id", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.ALL,
        )
        imagery_table.add_global_secondary_index(
            index_name="UniqueIdIndex",
            partition_key=dynamodb.Attribute(name="unique_id", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # Job table: tracks jobs by jobId.
        job_table = dynamodb.Table(
            self, "JobTable",
            partition_key=dynamodb.Attribute(name="jobId", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN
        )

        # -----------------------------
        # 4) SQS: DLQ, lifecycle, sync
        # -----------------------------
        dlq = sqs.Queue(
            self, "Dlq",
            queue_name=f"{self.stack_name}-dlq",
            retention_period=Duration.days(14)
        )

        lifecycle_queue = sqs.Queue(
            self, "LifecycleQueue",
            queue_name=f"{self.stack_name}-lifecycle",
            dead_letter_queue=sqs.DeadLetterQueue(queue=dlq, max_receive_count=5),
        )

        sync_queue = sqs.Queue(
            self, "SyncQueue",
            queue_name=f"{self.stack_name}-sync",
            dead_letter_queue=sqs.DeadLetterQueue(queue=dlq, max_receive_count=5),
        )

        # ------------------------------------------------
        # 5) IAM: managed policy for CloudWatch log access
        # ------------------------------------------------
        lambda_basic_logs = iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
        # Note: This only grants logging (CloudWatch Logs). All data access is explicitly scoped below.

        # ----------------------------------------------------------
        # 6) Dedicated roles per Lambda with least-privilege policies
        # ----------------------------------------------------------

        # Helper: attach inline policy to a role
        def attach_inline_policy(role: iam.Role, name: str, statements: list[iam.PolicyStatement]):
            policy = iam.Policy(self, name, statements=statements)
            role.attach_inline_policy(policy)
            return policy

        # Lifecycle Lambda role
        lifecycle_role = iam.Role(
            self, "LifecycleLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[lambda_basic_logs],
            description="Least-privilege role for lifecycle operations."
        )
        # Permissions per your spec:
        # - Dataset table: read/write/delete
        dataset_table.grant_read_write_data(lifecycle_role)
        # - Imagery table: read + query (grant_read_data covers Get/Scan/Query)
        imagery_table.grant_read_data(lifecycle_role)
        # - Job table: read/write
        job_table.grant_read_write_data(lifecycle_role)
        # - SQS: consume lifecycle queue for polling pattern
        lifecycle_queue.grant_consume_messages(lifecycle_role)
        # - S3: write manifests under delete-manifests (for chaining delete jobs when needed)
        attach_inline_policy(
            lifecycle_role,
            "LifecycleS3ManifestWrite",
            [
                iam.PolicyStatement(
                    actions=["s3:PutObject", "s3:AbortMultipartUpload"],
                    resources=[delete_manifests_objects_arn, upload_manifests_objects_arn]
                )
            ]
        )

        # Sync submitter Lambda role (orchestrator that reads/writes Job table, reads Dataset, queries Imagery)
        sync_submit_role = iam.Role(
            self, "SyncSubmitLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[lambda_basic_logs],
            description="Least-privilege role for submitting sync jobs and reading dataset/imagery metadata."
        )
        job_table.grant_read_write_data(sync_submit_role)
        dataset_table.grant_read_data(sync_submit_role)
        imagery_table.grant_read_data(sync_submit_role)
        sync_queue.grant_consume_messages(sync_submit_role)

        # Sync worker: we’ll run this as an AWS Batch container job.
        # Job role for the batch container with the exact access you listed.
        batch_job_role = iam.Role(
            self, "BatchJobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")],
            description="IAM role assumed by Batch job containers for dataset sync."
        )
        # - S3 dataset root manifests folder: list/read/get/write/overwrite
        attach_inline_policy(
            batch_job_role,
            "BatchS3DatasetRootAccess",
            [
                iam.PolicyStatement(
                    actions=["s3:ListBucket"],
                    resources=[bucket_arn],
                    conditions={"StringLike": {"s3:prefix": [upload_manifests_prefix, delete_manifests_prefix, f"{dataset_root}/*"]}}
                ),
                iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"],
                    resources=[
                        upload_manifests_objects_arn,
                        delete_manifests_objects_arn,
                        images_objects_arn,
                        temp_images_objects_arn,
                        f"{bucket_arn}/{dataset_root}/*",
                    ],
                ),
            ]
        )
        # - Imagery table: read + query
        imagery_table.grant_read_data(batch_job_role)
        # - Job table: read/write/update
        job_table.grant_read_write_data(batch_job_role)

        # Upload step-1 (manifest reader/chunker) Lambda role
        upload_chunk_role = iam.Role(
            self, "UploadChunkLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[lambda_basic_logs],
            description="Role for reading upload manifests and temp-images to produce batches."
        )
        # - S3 read of upload manifests and temp-images to build batches
        attach_inline_policy(
            upload_chunk_role,
            "UploadChunkS3Read",
            [
                iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:ListBucket"],
                    resources=[bucket_arn, upload_manifests_objects_arn, temp_images_objects_arn],
                )
            ]
        )
        # - Job table (optional) if you record batch progress; otherwise skip
        job_table.grant_read_write_data(upload_chunk_role)

        # Upload map-state worker Lambda role
        upload_worker_role = iam.Role(
            self, "UploadWorkerLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[lambda_basic_logs],
            description="Role for processing upload batches: read temp-images, write canonical images, write imagery rows."
        )
        # - S3: read temp-images, write to images/
        attach_inline_policy(
            upload_worker_role,
            "UploadWorkerS3RW",
            [
                iam.PolicyStatement(actions=["s3:GetObject"], resources=[temp_images_objects_arn]),
                iam.PolicyStatement(actions=["s3:PutObject", "s3:AbortMultipartUpload"], resources=[images_objects_arn]),
            ]
        )
        # - Imagery table: write calculated attributes/features; read & query for safety
        imagery_table.grant_read_write_data(upload_worker_role)
        dataset_table.grant_read_data(upload_worker_role)
        job_table.grant_read_write_data(upload_worker_role)

        # Delete step-1 (manifest reader/chunker) Lambda role
        delete_chunk_role = iam.Role(
            self, "DeleteChunkLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[lambda_basic_logs],
            description="Role for reading delete manifests and producing delete batches."
        )
        attach_inline_policy(
            delete_chunk_role,
            "DeleteChunkS3Read",
            [
                iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:ListBucket"],
                    resources=[bucket_arn, delete_manifests_objects_arn],
                )
            ]
        )
        # - Tables: read/query all; update job table
        dataset_table.grant_read_data(delete_chunk_role)
        imagery_table.grant_read_data(delete_chunk_role)
        job_table.grant_read_write_data(delete_chunk_role)

        # Delete map-state worker Lambda role
        delete_worker_role = iam.Role(
            self, "DeleteWorkerLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[lambda_basic_logs],
            description="Role for deleting imagery and objects per batch; updates job table."
        )
        # - S3: delete images in images/ and temp-images if cleanup required
        attach_inline_policy(
            delete_worker_role,
            "DeleteWorkerS3Delete",
            [
                iam.PolicyStatement(actions=["s3:DeleteObject"], resources=[images_objects_arn, temp_images_objects_arn]),
                iam.PolicyStatement(actions=["s3:GetObject"], resources=[images_objects_arn, temp_images_objects_arn]),
            ]
        )
        # - Imagery table: delete/read
        imagery_table.grant_read_write_data(delete_worker_role)
        # - Job table: read/write/update
        job_table.grant_read_write_data(delete_worker_role)
        # - Dataset table: read/query (for good measure)
        dataset_table.grant_read_data(delete_worker_role)

        # ---------------------------------------------------
        # 7) Lambdas: code, handlers, env, event source maps
        # ---------------------------------------------------
        
        # to the following for auto ecr image build and push and make lambda
        #%%
        from aws_cdk import aws_lambda as _lambda
        # IAM role for Lambda
        lambda_role = iam.Role(
            self, "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )

        # Lambda from Dockerfile (CDK builds & pushes image)
        fn = _lambda.DockerImageFunction(
            self, "MyImageLambda",
            code=_lambda.DockerImageCode.from_image_asset("lambda_docker_dir"),
            role=lambda_role,
            memory_size=128,
            timeout=core.Duration.seconds(20),
            description="A Lambda function created from a Docker image.",
            environment={
                "MY_ENV_VAR": "value"
            },
            architecture=_lambda.Architecture.X86_64
        )
#%%
        lifecycle_lambda = _lambda.Function(
            self, "LifecycleLambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lifecycle.handler",
            code=_lambda.Code.from_asset("lambda/lifecycle"),
            role=lifecycle_role,
            memory_size=1024,
            timeout=Duration.minutes(5),
            environment={
                "DATASET_TABLE": dataset_table.table_name,
                "IMAGERY_TABLE": imagery_table.table_name,
                "JOB_TABLE": job_table.table_name,
                "BUCKET": bucket.bucket_name,
                "DATASET_ROOT": dataset_root,
                "LIFECYCLE_QUEUE_URL": lifecycle_queue.queue_url,
            }
        )
        lifecycle_lambda.add_event_source_mapping(
            "LifecycleQueueMapping",
            event_source_arn=lifecycle_queue.queue_arn,
            batch_size=10,
            enabled=True,
        )

        # Sync submitter lambda (reads SQS sync events, starts Batch jobs)
        sync_lambda = _lambda.Function(
            self, "SyncSubmitLambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="sync_submit.handler",
            code=_lambda.Code.from_asset("lambda/sync_submit"),
            role=sync_submit_role,
            memory_size=1024,
            timeout=Duration.minutes(5),
            environment={
                "DATASET_TABLE": dataset_table.table_name,
                "IMAGERY_TABLE": imagery_table.table_name,
                "JOB_TABLE": job_table.table_name,
                "BUCKET": bucket.bucket_name,
                "DATASET_ROOT": dataset_root,
                "SYNC_QUEUE_URL": sync_queue.queue_url,
            }
        )
        sync_lambda.add_event_source_mapping(
            "SyncQueueMapping",
            event_source_arn=sync_queue.queue_arn,
            batch_size=5,
            enabled=True,
        )

        # Upload chunker Lambda (Step 1 for upload workflow)
        upload_chunk_lambda = _lambda.Function(
            self, "UploadChunkLambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="upload_chunk.handler",
            code=_lambda.Code.from_asset("lambda/upload_chunk"),
            role=upload_chunk_role,
            memory_size=1024,
            timeout=Duration.minutes(5),
            environment={
                "BUCKET": bucket.bucket_name,
                "DATASET_ROOT": dataset_root,
                "BATCH_SIZE": "1000"
            }
        )

        # Upload worker Lambda (Map state)
        upload_worker_lambda = _lambda.Function(
            self, "UploadWorkerLambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="upload_worker.handler",
            code=_lambda.Code.from_asset("lambda/upload_worker"),
            role=upload_worker_role,
            memory_size=1024,
            timeout=Duration.minutes(15),
            environment={
                "DATASET_TABLE": dataset_table.table_name,
                "IMAGERY_TABLE": imagery_table.table_name,
                "JOB_TABLE": job_table.table_name,
                "BUCKET": bucket.bucket_name,
                "DATASET_ROOT": dataset_root,
            }
        )

        # Delete chunker Lambda (Step 1 for delete workflow)
        delete_chunk_lambda = _lambda.Function(
            self, "DeleteChunkLambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="delete_chunk.handler",
            code=_lambda.Code.from_asset("lambda/delete_chunk"),
            role=delete_chunk_role,
            memory_size=1024,
            timeout=Duration.minutes(5),
            environment={
                "BUCKET": bucket.bucket_name,
                "DATASET_ROOT": dataset_root,
                "BATCH_SIZE": "1000"
            }
        )

        # Delete worker Lambda (Map state)
        delete_worker_lambda = _lambda.Function(
            self, "DeleteWorkerLambda",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="delete_worker.handler",
            code=_lambda.Code.from_asset("lambda/delete_worker"),
            role=delete_worker_role,
            memory_size=1024,
            timeout=Duration.minutes(15),
            environment={
                "DATASET_TABLE": dataset_table.table_name,
                "IMAGERY_TABLE": imagery_table.table_name,
                "JOB_TABLE": job_table.table_name,
                "BUCKET": bucket.bucket_name,
                "DATASET_ROOT": dataset_root,
            }
        )

        # ---------------------------------------
        # 8) AWS Batch for sync (single heavy job)
        # ---------------------------------------
        vpc = ec2.Vpc(self, "BatchVpc", nat_gateways=1)
        sg = ec2.SecurityGroup(self, "BatchSecurityGroup", vpc=vpc, allow_all_outbound=True)

        compute_env = batch.ComputeEnvironment(
            self, "BatchComputeEnv",
            compute_environment_name=f"{self.stack_name}-compute",
            compute_resources=batch.ComputeResources(
                type=batch.ComputeResourceType.SPOT,  # choose ON_DEMAND if preferred
                allocation_strategy=batch.AllocationStrategy.BEST_FIT_PROGRESSIVE,
                vpc=vpc,
                security_groups=[sg],
                minv_cpus=0,
                desiredv_cpus=0,
                maxv_cpus=128,
                instance_types=[ec2.InstanceType("r5.4xlarge"), ec2.InstanceType("m5.4xlarge")],
                subnets=vpc.private_subnets,
            ),
        )

        job_queue = batch.JobQueue(
            self, "BatchJobQueue",
            job_queue_name=f"{self.stack_name}-job-queue",
            compute_environments=[batch.JobQueueComputeEnvironment(compute_environment=compute_env, order=1)],
            priority=1
        )

        job_def = batch.JobDefinition(
            self, "SyncJobDefinition",
            job_definition_name=f"{self.stack_name}-sync-job",
            container=batch.JobDefinitionContainer(
                # Replace with your ECR image containing the sync worker
                image=batch.EcrImage.from_registry("public.ecr.aws/docker/library/python:3.12"),
                vcpus=8,
                memory_limit_mib=32768,
                job_role=batch_job_role,
                environment={
                    "DATASET_TABLE": dataset_table.table_name,
                    "IMAGERY_TABLE": imagery_table.table_name,
                    "JOB_TABLE": job_table.table_name,
                    "BUCKET": bucket.bucket_name,
                    "DATASET_ROOT": dataset_root,
                },
            ),
            platform_capabilities=[batch.PlatformCapabilities.EC2],
        )

        # Permission for sync submitter Lambda to submit/describe Batch jobs
        sync_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["batch:SubmitJob", "batch:DescribeJobs"],
            resources=[job_def.job_definition_arn, job_queue.job_queue_arn]
        ))

        # ----------------------------------------------------
        # 9) Step Functions: upload and delete with Map states
        # ----------------------------------------------------
        # Upload workflow
        upload_read = tasks.LambdaInvoke(
            self, "ReadUploadManifest",
            lambda_function=upload_chunk_lambda,
            payload_response_only=True,
        )
        upload_map = sfn.Map(
            self, "UploadMap",
            max_concurrency=50,
            items_path="$.batches",
        ).iterator(tasks.LambdaInvoke(
            self, "ProcessUploadBatch",
            lambda_function=upload_worker_lambda,
            payload_response_only=True,
            payload=sfn.TaskInput.from_object({"batch": sfn.JsonPath.string_at("$")})
        ))
        upload_sm = sfn.StateMachine(
            self, "UploadStateMachine",
            definition=upload_read.next(upload_map),
            timeout=Duration.hours(2),
            logs=sfn.LogOptions(destination=log_group, level=sfn.LogLevel.ALL),
        )

        # Delete workflow
        delete_read = tasks.LambdaInvoke(
            self, "ReadDeleteManifest",
            lambda_function=delete_chunk_lambda,
            payload_response_only=True,
        )
        delete_map = sfn.Map(
            self, "DeleteMap",
            max_concurrency=50,
            items_path="$.batches",
        ).iterator(tasks.LambdaInvoke(
            self, "ProcessDeleteBatch",
            lambda_function=delete_worker_lambda,
            payload_response_only=True,
            payload=sfn.TaskInput.from_object({"batch": sfn.JsonPath.string_at("$")})
        ))
        delete_sm = sfn.StateMachine(
            self, "DeleteStateMachine",
            definition=delete_read.next(delete_map),
            timeout=Duration.hours(2),
            logs=sfn.LogOptions(destination=log_group, level=sfn.LogLevel.ALL),
        )

        # ---------------------------------------------
        # 10) EventBridge rules: trigger workflows on S3
        # ---------------------------------------------
        # Allow EventBridge to start state machines.
        upload_sm.grant_start_execution(iam.ServicePrincipal("events.amazonaws.com"))
        delete_sm.grant_start_execution(iam.ServicePrincipal("events.amazonaws.com"))

        # Rules that match manifest uploads under dataset_root
        upload_rule = events.Rule(
            self, "UploadManifestRule",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [bucket.bucket_name]},
                    "object": {"key": [{"prefix": upload_manifests_prefix + "/"}]},
                },
            ),
            targets=[targets.SfnStateMachine(upload_sm)]
        )

        delete_rule = events.Rule(
            self, "DeleteManifestRule",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [bucket.bucket_name]},
                    "object": {"key": [{"prefix": delete_manifests_prefix + "/"}]},
                },
            ),
            targets=[targets.SfnStateMachine(delete_sm)]
        )

        # ----------------
        # 11) CFN outputs
        # ----------------
        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "DatasetRoot", value=dataset_root)
        CfnOutput(self, "DatasetTableName", value=dataset_table.table_name)
        CfnOutput(self, "ImageryTableName", value=imagery_table.table_name)
        CfnOutput(self, "JobTableName", value=job_table.table_name)
        CfnOutput(self, "LifecycleQueueUrl", value=lifecycle_queue.queue_url)
        CfnOutput(self, "SyncQueueUrl", value=sync_queue.queue_url)
        CfnOutput(self, "UploadStateMachineArn", value=upload_sm.state_machine_arn)
        CfnOutput(self, "DeleteStateMachineArn", value=delete_sm.state_machine_arn)
        CfnOutput(self, "BatchJobQueueArn", value=job_queue.job_queue_arn)
        CfnOutput(self, "SyncJobDefinitionArn", value=job_def.job_definition_arn)
        CfnOutput(self, "LogGroupName", value=log_group.log_group_name)


