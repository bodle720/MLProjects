from typing import Callable
from constructs import Construct

from aws_cdk import (
    RemovalPolicy,
    Duration,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_batch as batch,
    aws_iam as iam,
    aws_logs as logs,
    aws_dynamodb as dynamodb,
    aws_ecr_assets as ecr_assets
)

from config_models import BatchingStageConfig

class BatchingStage(Construct):
    def __init__(self,
                 scope: Construct,
                 construct_id: str, *,
                 stage_name: str,
                 config: BatchingStageConfig,
                 common_utils_layer: _lambda.LayerVersion,
                 file_bucket: s3.Bucket,
                 iceberg_bucket: s3.Bucket,
                 job_table: dynamodb.Table,
                 sha256_table: dynamodb.Table,
                 job_queue: batch.JobQueue,
                 iceberg_database_name: str,
                 ce_maxv_cpus: int,
                 region: str,
                 account: str,
                 dlq_chain_factory: Callable[[], sfn.Chain],
                 firehose_delivery_stream_name: str,
                 firehose_delivery_stream_attr_arn: str,
                 extra_lambda_env: dict = None,
                 extra_permissions: list[iam.PolicyStatement] = None,
                 extra_container_env: dict = None,
                 extra_map_state_params: dict = None):

        '''
        Batching param explanation:
        The batching Lambda returns a dict with manifests, job_id, user, label_type. The Map state iterates over manifests,
        builds a per‑item input with those values, and the Batch task pulls them into container environment variables via
        JsonPath.string_at
        '''

        super().__init__(scope, construct_id)

        # Lambda env vars
        lambda_env = {
            "FILE_BUCKET_NAME": file_bucket.bucket_name,
            "ATHENA_OUTPUT_S3": f"s3://{file_bucket.bucket_name}/athena-results/",
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DATABASE_NAME": iceberg_database_name,
            "UPLOAD_STAGING_TABLE_NAME": "upload_staging",
            "CANONICAL_IMAGERY_TABLE_NAME": "canonical_imagery",
            "SHA256_TABLE_NAME": sha256_table.table_name,
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
            memory_size=config.file_batching.memory_size,
            timeout=Duration.seconds(config.file_batching.timeout_sec),
            environment=lambda_env
        )

        # baseline grants and policies
        sha256_table.grant_read_data(batching_fn)
        file_bucket.grant_read_write(batching_fn)

        batching_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket","s3:GetBucketLocation"],
            resources=[file_bucket.bucket_arn,
                       iceberg_bucket.bucket_arn]
        ))

        batching_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=[f"arn:aws:s3:::{iceberg_bucket.bucket_name}/upload_staging/*"]
        ))

        batching_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["athena:StartQueryExecution","athena:GetQueryExecution","athena:GetQueryResults"],
            resources=[f"arn:aws:athena:{region}:{account}:workgroup/primary"]
        ))

        batching_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[firehose_delivery_stream_attr_arn]
        ))

        batching_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:DeleteObject"],
            resources=[f"arn:aws:s3:::{file_bucket.bucket_name}/temp/image-upload/*"]
        ))

        batching_fn.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "glue:GetDatabase",
                "glue:GetDatabases",
                "glue:GetTable",
                "glue:GetTables",
                "glue:GetPartition",
                "glue:GetPartitions",
                "glue:GetTableVersion",
                "glue:GetTableVersions",
                "glue:CreateTable",
                "glue:UpdateTable",
                "glue:DeleteTable",
                "glue:CreatePartition",
                "glue:BatchCreatePartition",
                "glue:BatchDeletePartition",
                "glue:DeletePartition"
            ],
            resources=[
                f"arn:aws:glue:{region}:{account}:catalog",
                f"arn:aws:glue:{region}:{account}:database/{iceberg_database_name}",
                f"arn:aws:glue:{region}:{account}:table/{iceberg_database_name}/*"
            ]
        ))

        if extra_permissions:
            for stmt in extra_permissions:
                batching_fn.add_to_role_policy(stmt)

        batching_task = tasks.LambdaInvoke(
            self, f"{stage_name}BatchingTask",
            lambda_function=batching_fn,
            result_path=f"$.{stage_name}",
            output_path="$",
            payload_response_only=True # means the Lambda’s response is used as the task output
        )

        batching_task.add_retry(backoff_rate=2.0, max_attempts=2, interval=Duration.seconds(2))

        # if the batching Lambda fails, Step Functions will capture the error info under $.errorInfo and transition to the DLQ state.
        batching_task.add_catch(
            handler=dlq_chain_factory(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        job_role = iam.Role(
            self, f"{stage_name}JobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )

        # Firehose logging (unchanged)
        job_role.add_to_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[firehose_delivery_stream_attr_arn]
        ))

        # DynamoDB tables
        job_table.grant_read_write_data(job_role)
        sha256_table.grant_read_data(job_role)

        # Athena: start queries and poll results (scoped to workgroup)
        job_role.add_to_policy(iam.PolicyStatement(
            actions=["athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults"],
            resources=[f"arn:aws:athena:{region}:{account}:workgroup/primary"]
        ))

        # Glue metadata read (catalog, database, and all tables in the DB)
        job_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "glue:GetDatabase", "glue:GetDatabases",
                "glue:GetTable", "glue:GetTables",
                "glue:GetPartition", "glue:GetPartitions",
                "glue:GetTableVersion", "glue:GetTableVersions"
            ],
            resources=[
                f"arn:aws:glue:{region}:{account}:catalog",
                f"arn:aws:glue:{region}:{account}:database/{iceberg_database_name}",
                f"arn:aws:glue:{region}:{account}:table/{iceberg_database_name}/*"
            ]
        ))

        # Glue metadata write for upload_staging (required when INSERT/DELETE/OPTIMIZE updates Iceberg metadata)
        job_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "glue:CreateTable", "glue:UpdateTable", "glue:DeleteTable",
                "glue:BatchCreatePartition", "glue:BatchDeletePartition"
            ],
            resources=[
                f"arn:aws:glue:{region}:{account}:catalog",
                f"arn:aws:glue:{region}:{account}:database/{iceberg_database_name}",
                f"arn:aws:glue:{region}:{account}:table/{iceberg_database_name}/upload_staging"
            ]
        ))

        # S3: Athena results write only to athena-results/ prefix in file_bucket
        job_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:PutObject"],
            resources=[f"arn:aws:s3:::{file_bucket.bucket_name}/athena-results/*"]
        ))
        # allow listing only for athena-results prefix on all buckets (need to be able to load in images everywhere)
        job_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::*"]
        ))

        # S3: read/write/delete Iceberg files for upload_staging prefix
        job_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
            resources=[f"arn:aws:s3:::{iceberg_bucket.bucket_name}/upload_staging/*"]
        ))
        # allow listing only for upload_staging prefix on iceberg_bucket
        job_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[f"arn:aws:s3:::{iceberg_bucket.bucket_name}"],
            conditions={"StringLike": {"s3:prefix": ["upload_staging/*"]}}
        ))

        job_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:PutObject"],
            resources=[f"arn:aws:s3:::{file_bucket.bucket_name}/temp/image-upload/*"]
        ))

        job_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "s3:GetObject",
                "s3:GetObjectVersion",
            ],
            resources=[
                "arn:aws:s3:::*/*",
            ],
        ))

        # Allow the Batch job to write to canonical outputs in file bucket
        job_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "s3:PutObject",
                "s3:AbortMultipartUpload",  # good to include for some SDK behaviors
            ],
            resources=[
                f"arn:aws:s3:::{file_bucket.bucket_name}/canonical/*",
            ],
        ))

        job_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[f"arn:aws:s3:::{file_bucket.bucket_name}"],
            conditions={"StringLike": {"s3:prefix": ["temp/image-upload/*"]}}
        ))

        if extra_permissions:
            for stmt in extra_permissions:
                job_role.add_to_policy(stmt)

        # build/publish local Docker image from a local path
        image_asset = ecr_assets.DockerImageAsset(self, f"{stage_name}TaskImage",
                                                  directory=config.batch_task_job_def.directory,
                                                  file=config.batch_task_job_def.file
                                                  )
        log_group = logs.LogGroup(
            self,
            f"{stage_name}LogGroup",
            log_group_name=f"/aws/batch/{stage_name}",
            removal_policy=RemovalPolicy.DESTROY
        )

        job_def = batch.CfnJobDefinition(
            self, f"{stage_name}JobDef",
            type="container",
            container_properties={
                "image": image_asset.image_uri,
                "cpu": int(config.batch_task_job_def.vcpus * 1024),
                "vcpus": config.batch_task_job_def.vcpus,  # job-level vCPUs Batch uses for placement
                "memory": config.batch_task_job_def.memory_limit_mib,
                "jobRoleArn": job_role.role_arn,
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": log_group.log_group_name,
                        "awslogs-region": region,
                        "awslogs-stream-prefix": f"{stage_name}-batch"
                    }
                },
            },
            retry_strategy={"attempts": 5},
            timeout={"attemptDurationSeconds": int(Duration.hours(2).to_seconds())}
        )

        container_env = {
            "MANIFEST_S3_URI": sfn.JsonPath.string_at("$.manifest"), # means Step Functions will substitute the values from the per‑item input into environment variables for the container, see below
            "JOB_ID": sfn.JsonPath.string_at("$.job_id"),
            "USER": sfn.JsonPath.string_at("$.user"),
            "LABEL_TYPE": sfn.JsonPath.string_at("$.label_type"),
            "DATA_SOURCE": sfn.JsonPath.string_at("$.data_source"),
            "EVENT_TYPE": sfn.JsonPath.string_at("$.event_type"),
            "FILE_BUCKET_NAME": file_bucket.bucket_name,
            "SHA256_TABLE_NAME": sha256_table.table_name,
            "ATHENA_OUTPUT_S3": f"s3://{file_bucket.bucket_name}/athena-results/",
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DATABASE_NAME": iceberg_database_name,
            "UPLOAD_STAGING_TABLE_NAME": "upload_staging",
            "CANONICAL_IMAGERY_TABLE_NAME": "canonical_imagery",
            "LOG_FIREHOSE_STREAM_NAME": firehose_delivery_stream_name,
            "AWS_REGION": region,
            "AWS_DEFAULT_REGION": region
        }
        if extra_container_env:
            container_env.update(extra_container_env)

        batch_task = tasks.BatchSubmitJob(
            self, f"{stage_name}BatchTask",
            job_definition_arn=job_def.attr_job_definition_arn,
            job_queue_arn=job_queue.job_queue_arn,
            job_name=f"{stage_name.lower()}-batch",
            container_overrides=tasks.BatchContainerOverrides(
                environment=container_env
            ),
            integration_pattern=sfn.IntegrationPattern.RUN_JOB
        )

        # Map state (wired to Batch task)
        params = {
                "manifest.$": "$$.Map.Item.Value",
                "job_id.$": "$.job_id", # keys ending with .$ tell Step Functions “this value comes from a JSONPath expression.”
                "user.$": "$.user",
                "event_type.$": "$.event_type",
                "label_type.$": "$.label_type",
                "data_source.$": "$.data_source"
        }

        if extra_map_state_params:
            params.update(extra_map_state_params)

        map_state = sfn.Map(
            self, f"{stage_name}MapState",
            items_path=f"$.{stage_name}.manifests",
            item_selector=params,
            result_path=sfn.JsonPath.DISCARD,
            output_path="$",
            max_concurrency=max(1, min(50, int(ce_maxv_cpus / max(1, config.batch_task_job_def.vcpus)))),
        )

        map_state.add_catch(
            handler=dlq_chain_factory(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        per_item = sfn.Chain.start(batch_task)
        map_state.iterator(per_item)

        self.batching_task = batching_task
        self.map_state = map_state
        self.job_def = job_def
        self.job_role = job_role