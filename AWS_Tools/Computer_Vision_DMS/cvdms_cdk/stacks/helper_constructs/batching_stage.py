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
    aws_ecr_assets as ecr_assets,
)

from config_models import BatchingStageConfig

class BatchingStage(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
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
        extra_lambda_env: dict | None = None,
        extra_permissions: list[iam.PolicyStatement] | None = None,
        extra_container_env: dict | None = None,
        extra_map_state_params: dict | None = None,
    ):
        """
        New batching contract:
        - The batching Lambda writes a JSONL handoff plan to S3.
        - Each JSONL line is one map item, typically containing at minimum:
              {"manifest": "s3://..."}
          plus any stage-specific item fields.
        - The batching Lambda returns only a compact pointer payload, for example:
              {
                  "plan_bucket": "...",
                  "plan_key": "temp/image-upload/<job_id>/batches/<stage>/handoff/map-items.jsonl",
                  "plan_s3_uri": "s3://.../map-items.jsonl",
                  "item_count": 123,
                  ...optional small summary fields...
              }
        - Step Functions Distributed Map reads the JSONL plan directly from S3, so large
          per-stage manifest arrays are never carried inline in execution state.
        """
        super().__init__(scope, construct_id)

        # kept for signature compatibility with existing callers
        _ = ce_maxv_cpus

        # ------------------------------------------------------------------
        # Batching lambda
        # ------------------------------------------------------------------
        lambda_env = {
            "FILE_BUCKET_NAME": file_bucket.bucket_name,
            "ATHENA_OUTPUT_S3": f"s3://{file_bucket.bucket_name}/athena-results/",
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DATABASE_NAME": iceberg_database_name,
            "LOG_FIREHOSE_STREAM_NAME": firehose_delivery_stream_name,
            "BATCH_STAGE_NAME": stage_name,
            "BATCH_HANDOFF_FILE_NAME": "map-items.jsonl",
        }
        if extra_lambda_env:
            lambda_env.update(extra_lambda_env)

        batching_fn = _lambda.Function(
            self,
            f"{stage_name}BatchingLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=config.file_batching.handler,
            code=_lambda.Code.from_asset(config.file_batching.path),
            layers=[common_utils_layer],
            memory_size=config.file_batching.memory_size,
            timeout=Duration.seconds(config.file_batching.timeout_sec),
            environment=lambda_env,
        )

        sha256_table.grant_read_data(batching_fn)
        file_bucket.grant_read_write(batching_fn)

        batching_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[file_bucket.bucket_arn, iceberg_bucket.bucket_arn],
            )
        )
        batching_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[f"arn:aws:s3:::{iceberg_bucket.bucket_name}/upload_staging/*"],
            )
        )
        batching_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:StopQueryExecution",
                ],
                resources=[f"arn:aws:athena:{region}:{account}:workgroup/primary"],
            )
        )
        batching_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
                resources=[firehose_delivery_stream_attr_arn],
            )
        )
        batching_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:DeleteObject"],
                resources=[f"arn:aws:s3:::{file_bucket.bucket_name}/temp/image-upload/*"],
            )
        )
        batching_fn.add_to_role_policy(
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
                    "glue:CreateTable",
                    "glue:UpdateTable",
                    "glue:DeleteTable",
                    "glue:CreatePartition",
                    "glue:BatchCreatePartition",
                    "glue:BatchDeletePartition",
                    "glue:DeletePartition",
                ],
                resources=[
                    f"arn:aws:glue:{region}:{account}:catalog",
                    f"arn:aws:glue:{region}:{account}:database/{iceberg_database_name}",
                    f"arn:aws:glue:{region}:{account}:table/{iceberg_database_name}/*",
                ],
            )
        )

        if extra_permissions:
            for stmt in extra_permissions:
                batching_fn.add_to_role_policy(stmt)

        batching_task = tasks.LambdaInvoke(
            self,
            f"{stage_name}BatchingTask",
            lambda_function=batching_fn,
            result_path=f"$.{stage_name}",
            output_path="$",
            payload_response_only=True,
        )
        batching_task.add_retry(
            backoff_rate=2.0,
            max_attempts=2,
            interval=Duration.seconds(2),
        )
        batching_task.add_catch(
            handler=dlq_chain_factory(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        # ------------------------------------------------------------------
        # Batch worker role + job definition
        # ------------------------------------------------------------------
        job_role = iam.Role(
            self,
            f"{stage_name}JobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )

        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
                resources=[firehose_delivery_stream_attr_arn],
            )
        )

        job_table.grant_read_write_data(job_role)
        sha256_table.grant_read_write_data(job_role)

        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:StopQueryExecution",
                ],
                resources=[f"arn:aws:athena:{region}:{account}:workgroup/primary"],
            )
        )

        job_role.add_to_policy(
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
                    f"arn:aws:glue:{region}:{account}:catalog",
                    f"arn:aws:glue:{region}:{account}:database/{iceberg_database_name}",
                    f"arn:aws:glue:{region}:{account}:table/{iceberg_database_name}/*",
                ],
            )
        )

        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:CreateTable",
                    "glue:UpdateTable",
                    "glue:DeleteTable",
                    "glue:BatchCreatePartition",
                    "glue:BatchDeletePartition",
                ],
                resources=[
                    f"arn:aws:glue:{region}:{account}:catalog",
                    f"arn:aws:glue:{region}:{account}:database/{iceberg_database_name}",
                    f"arn:aws:glue:{region}:{account}:table/{iceberg_database_name}/upload_staging",
                ],
            )
        )

        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject"],
                resources=[f"arn:aws:s3:::{file_bucket.bucket_name}/athena-results/*"],
            )
        )
        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetBucketLocation"],
                resources=[file_bucket.bucket_arn, iceberg_bucket.bucket_arn],
            )
        )
        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                resources=[
                    f"arn:aws:s3:::{iceberg_bucket.bucket_name}/upload_staging/*",
                    f"arn:aws:s3:::{iceberg_bucket.bucket_name}/canonical/*",
                ],
            )
        )
        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[f"arn:aws:s3:::{iceberg_bucket.bucket_name}"],
                conditions={"StringLike": {"s3:prefix": ["upload_staging/*", "canonical/*"]}},
            )
        )
        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                resources=[f"arn:aws:s3:::{file_bucket.bucket_name}/temp/image-upload/*"],
            )
        )
        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:GetObjectVersion"],
                resources=["arn:aws:s3:::*/*"],
            )
        )
        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject", "s3:AbortMultipartUpload"],
                resources=[f"arn:aws:s3:::{file_bucket.bucket_name}/canonical/*"],
            )
        )
        job_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[f"arn:aws:s3:::{file_bucket.bucket_name}"],
                conditions={"StringLike": {"s3:prefix": ["temp/image-upload/*"]}},
            )
        )

        if extra_permissions:
            for stmt in extra_permissions:
                job_role.add_to_policy(stmt)

        image_asset = ecr_assets.DockerImageAsset(
            self,
            f"{stage_name}TaskImage",
            directory=config.batch_task_job_def.directory,
            file=config.batch_task_job_def.file,
        )

        log_group = logs.LogGroup(
            self,
            f"{stage_name}LogGroup",
            log_group_name=f"/aws/batch/{stage_name}",
            removal_policy=RemovalPolicy.DESTROY,
        )

        job_def = batch.CfnJobDefinition(
            self,
            f"{stage_name}JobDef",
            type="container",
            container_properties={
                "image": image_asset.image_uri,
                "cpu": int(config.batch_task_job_def.vcpus * 1024),
                "vcpus": config.batch_task_job_def.vcpus,
                "memory": config.batch_task_job_def.memory_limit_mib,
                "jobRoleArn": job_role.role_arn,
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": log_group.log_group_name,
                        "awslogs-region": region,
                        "awslogs-stream-prefix": f"{stage_name}-batch",
                    },
                },
            },
            retry_strategy={"attempts": 1},
            timeout={"attemptDurationSeconds": int(Duration.hours(2).to_seconds())},
        )

        container_env = {
            "MANIFEST_S3_URI": sfn.JsonPath.string_at("$.manifest"),
            "JOB_ID": sfn.JsonPath.string_at("$.job_id"),
            "USER": sfn.JsonPath.string_at("$.user"),
            "LABEL_TYPE": sfn.JsonPath.string_at("$.label_type"),
            "DATA_SOURCE": sfn.JsonPath.string_at("$.data_source"),
            "SOURCE_SPLIT": sfn.JsonPath.string_at("$.source_split"),
            "PATH_PREFIX": sfn.JsonPath.string_at("$.path_prefix"),
            "EVENT_TYPE": sfn.JsonPath.string_at("$.event_type"),
            "FILE_BUCKET_NAME": file_bucket.bucket_name,
            "SHA256_TABLE_NAME": sha256_table.table_name,
            "ATHENA_OUTPUT_S3": f"s3://{file_bucket.bucket_name}/athena-results/",
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DATABASE_NAME": iceberg_database_name,
            "LOG_FIREHOSE_STREAM_NAME": firehose_delivery_stream_name,
            "AWS_REGION": region,
            "AWS_DEFAULT_REGION": region,
            "REGISTRATION_TIME": sfn.JsonPath.string_at("$.registration_time"),
        }
        if extra_container_env:
            container_env.update(extra_container_env)

        params = {
            "manifest.$": "$$.Map.Item.Value.manifest",
            "job_id.$": "$.job_id",
            "user.$": "$.user",
            "label_type.$": "$.label_type",
            "data_source.$": "$.data_source",
            "source_split.$": "$.source_split",
            "path_prefix.$": "$.path_prefix",
            "event_type.$": "$.event_type",
            "registration_time.$": "$.registration_time",
        }
        if extra_map_state_params:
            params.update(extra_map_state_params)

        map_state_json = {
            "Type": "Map",
            "ItemReader": {
                "Resource": "arn:aws:states:::s3:getObject",
                "ReaderConfig": {
                    "InputType": "JSONL",
                },
                "Parameters": {
                    "Bucket.$": f"$.{stage_name}.plan_bucket",
                    "Key.$": f"$.{stage_name}.plan_key",
                },
            },
            "ItemSelector": params,
            "ItemProcessor": {
                "ProcessorConfig": {
                    "Mode": "DISTRIBUTED",
                    "ExecutionType": "STANDARD",
                },
                "StartAt": f"{stage_name}BatchTask",
                "States": {
                    f"{stage_name}BatchTask": {
                        "Type": "Task",
                        "Resource": "arn:aws:states:::batch:submitJob.sync",
                        "Parameters": {
                            "JobDefinition": job_def.attr_job_definition_arn,
                            "JobName": f"{stage_name.lower()}-batch",
                            "JobQueue": job_queue.job_queue_arn,
                            "ContainerOverrides": {
                                "Environment": [
                                    {"Name": "MANIFEST_S3_URI", "Value.$": "$.manifest"},
                                    {"Name": "JOB_ID", "Value.$": "$.job_id"},
                                    {"Name": "USER", "Value.$": "$.user"},
                                    {"Name": "LABEL_TYPE", "Value.$": "$.label_type"},
                                    {"Name": "DATA_SOURCE", "Value.$": "$.data_source"},
                                    {"Name": "SOURCE_SPLIT", "Value.$": "$.source_split"},
                                    {"Name": "PATH_PREFIX", "Value.$": "$.path_prefix"},
                                    {"Name": "EVENT_TYPE", "Value.$": "$.event_type"},
                                    {"Name": "FILE_BUCKET_NAME", "Value": file_bucket.bucket_name},
                                    {"Name": "SHA256_TABLE_NAME", "Value": sha256_table.table_name},
                                    {"Name": "ATHENA_OUTPUT_S3",
                                     "Value": f"s3://{file_bucket.bucket_name}/athena-results/"},
                                    {"Name": "ATHENA_WORKGROUP", "Value": "primary"},
                                    {"Name": "ICEBERG_DATABASE_NAME", "Value": iceberg_database_name},
                                    {"Name": "LOG_FIREHOSE_STREAM_NAME", "Value": firehose_delivery_stream_name},
                                    {"Name": "AWS_REGION", "Value": region},
                                    {"Name": "AWS_DEFAULT_REGION", "Value": region},
                                    {"Name": "REGISTRATION_TIME", "Value.$": "$.registration_time"},
                                ]
                            },
                        },
                        "Retry": [
                            {
                                "ErrorEquals": ["Batch.ServerException"],
                                "IntervalSeconds": 10,
                                "MaxAttempts": 2,
                                "BackoffRate": 2.0,
                            }
                        ],
                        "End": True,
                    }
                },
            },
            "MaxConcurrency": config.map_max_concurrency,
            "ResultPath": None,
            "OutputPath": "$",
        }

        distributed_map = sfn.CustomState(
            self,
            f"{stage_name}DistributedMapState",
            state_json=map_state_json,
        )

        distributed_map.add_catch(
            handler=dlq_chain_factory(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        # Explicit state machine permissions needed for the Distributed Map plan reader.
        self.state_machine_policy_statements = [
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[f"arn:aws:s3:::{file_bucket.bucket_name}/temp/image-upload/*"],
            ),
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[file_bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["temp/image-upload/*"]}},
            ),
        ]

        self.batching_task = batching_task
        self.map_state = distributed_map
        self.distributed_map = distributed_map
        self.job_def = job_def
        self.job_role = job_role