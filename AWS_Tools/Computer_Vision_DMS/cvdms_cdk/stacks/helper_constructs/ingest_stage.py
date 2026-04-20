from typing import Callable
from constructs import Construct

from aws_cdk import (
    Duration,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_iam as iam,
    aws_dynamodb as dynamodb,
)

from config_models import IngestStageConfig

class IngestStage(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        stage_name: str,
        config: IngestStageConfig,
        common_utils_layer: _lambda.LayerVersion,
        file_bucket: s3.Bucket,
        iceberg_bucket: s3.Bucket,
        job_table: dynamodb.Table,
        lock_table: dynamodb.Table,
        sha256_table: dynamodb.Table,
        iceberg_database_name: str,
        region: str,
        account: str,
        dlq_chain_factory: Callable[[], sfn.Chain],
        firehose_delivery_stream_name: str,
        firehose_delivery_stream_attr_arn: str,
        batch_plan_bucket_path: str,
        batch_plan_key_path: str,
        batch_plan_s3_uri_path: str | None = None,
        expected_count_path: str | None = None,
        upload_testing_ssm_param_name: str = None
    ):
        super().__init__(scope, construct_id)

        self.stage_name = stage_name
        self.config = config
        self.common_utils_layer = common_utils_layer
        self.file_bucket = file_bucket
        self.iceberg_bucket = iceberg_bucket
        self.job_table = job_table
        self.lock_table = lock_table
        self.sha256_table = sha256_table
        self.iceberg_database_name = iceberg_database_name
        self.region = region
        self.account = account
        self.dlq_chain_factory = dlq_chain_factory
        self.firehose_delivery_stream_name = firehose_delivery_stream_name
        self.firehose_delivery_stream_attr_arn = firehose_delivery_stream_attr_arn
        self.batch_plan_bucket_path = batch_plan_bucket_path
        self.batch_plan_key_path = batch_plan_key_path
        self.batch_plan_s3_uri_path = batch_plan_s3_uri_path
        self.expected_count_path = expected_count_path
        self.upload_testing_ssm_param_name = upload_testing_ssm_param_name
        ingest_map_result_path = f"$.{stage_name}MapResults"

        lambda_env = {
            "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
            "ATHENA_OUTPUT_S3": f"s3://{self.file_bucket.bucket_name}/athena-results/",
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DATABASE_NAME": self.iceberg_database_name,
            "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream_name,
            "SHA256_TABLE_NAME": self.sha256_table.table_name,
            "INGEST_HANDOFF_FILE_NAME": "map-items.jsonl",
            "GROUPING_ENABLED": "true" if config.pre_ingest_grouping.grouping_enabled else "false",
            "TARGET_ROWS": str(config.pre_ingest_grouping.target_rows),
            "TARGET_BYTES": str(config.pre_ingest_grouping.target_bytes),
            "MAX_ROWS": str(config.pre_ingest_grouping.max_rows),
            "MAX_BYTES": str(config.pre_ingest_grouping.max_bytes),
            "MAX_MATERIALIZED_GROUP_BYTES": str(config.pre_ingest_grouping.max_materialized_group_bytes),
            "UPLOAD_TESTING_SSM_PARAM_NAME": self.upload_testing_ssm_param_name,
        }

        if config.pre_ingest_grouping.target_owner_bytes is not None:
            lambda_env["TARGET_OWNER_BYTES"] = str(config.pre_ingest_grouping.target_owner_bytes)

        if config.pre_ingest_grouping.max_owner_bytes is not None:
            lambda_env["MAX_OWNER_BYTES"] = str(config.pre_ingest_grouping.max_owner_bytes)

        if config.pre_ingest_grouping.target_owner_parts is not None:
            lambda_env["TARGET_OWNER_PARTS"] = str(config.pre_ingest_grouping.target_owner_parts)

        if config.pre_ingest_grouping.max_owner_parts is not None:
            lambda_env["MAX_OWNER_PARTS"] = str(config.pre_ingest_grouping.max_owner_parts)

        # --------------------------------------
        # Form the 'pre' lambda task.
        # --------------------------------------
        pre_fn = _lambda.Function(
            self,
            f"{stage_name}PreLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=config.pre_ingest_lambda.handler,
            code=_lambda.Code.from_asset(config.pre_ingest_lambda.path),
            layers=[common_utils_layer],
            memory_size=config.pre_ingest_lambda.memory_size,
            timeout=Duration.seconds(config.pre_ingest_lambda.timeout_sec),
            environment=lambda_env,
        )

        self.apply_ingest_permissions(pre_fn)

        payload_obj = {
            "job_id.$": "$.job_id",
            "user.$": "$.user",
            "event_type.$": "$.event_type",
            "label_type.$": "$.label_type",
            "data_source.$": "$.data_source",
            "source_split.$": "$.source_split",
            "batch_plan_bucket.$": self.batch_plan_bucket_path,
            "batch_plan_key.$": self.batch_plan_key_path,
        }
        if self.batch_plan_s3_uri_path:
            payload_obj["batch_plan_s3_uri.$"] = self.batch_plan_s3_uri_path
        if self.expected_count_path:
            payload_obj["expected_count.$"] = self.expected_count_path

        pre_ingest_task = tasks.LambdaInvoke(
            self,
            f"{stage_name}PreLambdaTask",
            lambda_function=pre_fn,
            result_path=f"$.{stage_name}PreLambdaTask",
            output_path="$",
            task_timeout=sfn.Timeout.duration(Duration.seconds(config.pre_ingest_lambda.timeout_sec)),
            payload_response_only=True,
            payload=sfn.TaskInput.from_object(payload_obj),
        )

        pre_ingest_task.add_catch(
            handler=self.dlq_chain_factory(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        # --------------------------------------
        # Form the map lambda function.
        # --------------------------------------
        map_fn = _lambda.Function(
            self,
            f"{stage_name}MapLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=config.map_ingest_lambda.handler,
            code=_lambda.Code.from_asset(config.map_ingest_lambda.path),
            layers=[common_utils_layer],
            memory_size=config.map_ingest_lambda.memory_size,
            timeout=Duration.seconds(config.map_ingest_lambda.timeout_sec),
            environment=lambda_env,
        )

        self.apply_ingest_permissions(map_fn)

        # --------------------------------------
        # Distributed Map over an S3 JSONL handoff file produced by the pre-lambda.
        # We intentionally store the map result under a named field instead of relying
        # on ResultPath: null / DISCARD semantics for CustomState.
        # --------------------------------------
        map_state_json = {
            "Type": "Map",
            "ItemReader": {
                "Resource": "arn:aws:states:::s3:getObject",
                "ReaderConfig": {
                    "InputType": "JSONL",
                },
                "Parameters": {
                    "Bucket.$": f"$.{stage_name}PreLambdaTask.plan_bucket",
                    "Key.$": f"$.{stage_name}PreLambdaTask.plan_key",
                },
            },
            "ItemProcessor": {
                "ProcessorConfig": {
                    "Mode": "DISTRIBUTED",
                    "ExecutionType": "STANDARD",
                },
                "StartAt": f"{stage_name}MapLambdaInvoke",
                "States": {
                    f"{stage_name}MapLambdaInvoke": {
                        "Type": "Task",
                        "Resource": "arn:aws:states:::lambda:invoke",
                        "OutputPath": "$.Payload",
                        "Parameters": {
                            "FunctionName": map_fn.function_arn,
                            "Payload.$": "$",
                        },
                        "Retry": [
                            {
                                "ErrorEquals": [
                                    "Lambda.ServiceException",
                                    "Lambda.AWSLambdaException",
                                    "Lambda.SdkClientException",
                                ],
                                "IntervalSeconds": 2,
                                "MaxAttempts": 2,
                                "BackoffRate": 2.0,
                            }
                        ],
                        "End": True,
                    }
                },
            },
            "ItemSelector": {
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "event_type.$": "$.event_type",
                "label_type.$": "$.label_type",
                "data_source.$": "$.data_source",
                "source_split.$": "$.source_split",
                "shard.$": "$$.Map.Item.Value.shard",
                "kind.$": "$$.Map.Item.Value.kind",
                "rows_read.$": "$$.Map.Item.Value.rows_read",
                "upload_staging_key.$": "$$.Map.Item.Value.upload_staging_key",
                "canonical_imagery_key.$": "$$.Map.Item.Value.canonical_imagery_key",
                "canonical_labels_key.$": "$$.Map.Item.Value.canonical_labels_key",
                "image_labels_key.$": "$$.Map.Item.Value.image_labels_key",
                "image_source_membership_key.$": "$$.Map.Item.Value.image_source_membership_key",
            },
            "MaxConcurrency": config.map_max_concurrency,
            "ResultPath": ingest_map_result_path,
            "OutputPath": "$",
        }

        map_state = sfn.CustomState(
            self,
            f"{stage_name}DistributedMapState",
            state_json=map_state_json,
        )

        map_state.add_catch(
            handler=self.dlq_chain_factory(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        # --------------------------------------
        # Form the 'post' lambda task.
        # --------------------------------------
        post_fn = _lambda.Function(
            self,
            f"{stage_name}PostLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=config.post_ingest_lambda.handler,
            code=_lambda.Code.from_asset(config.post_ingest_lambda.path),
            layers=[common_utils_layer],
            memory_size=config.post_ingest_lambda.memory_size,
            timeout=Duration.seconds(config.post_ingest_lambda.timeout_sec),
            environment=lambda_env,
        )

        self.apply_ingest_permissions(post_fn)

        post_ingest_task = tasks.LambdaInvoke(
            self,
            f"{stage_name}PostLambdaTask",
            lambda_function=post_fn,
            result_path=f"$.{stage_name}PostLambdaTask",
            output_path="$",
            task_timeout=sfn.Timeout.duration(Duration.seconds(config.post_ingest_lambda.timeout_sec)),
            payload_response_only=True,
            payload=sfn.TaskInput.from_object(
                {
                    "job_id.$": "$.job_id",
                    "user.$": "$.user",
                    "event_type.$": "$.event_type",
                    "label_type.$": "$.label_type",
                    "data_source.$": "$.data_source",
                    "source_split.$": "$.source_split",
                    "pre.$": f"$.{stage_name}PreLambdaTask",
                }
            ),
        )

        post_ingest_task.add_retry(
            backoff_rate=2.0,
            max_attempts=2,
            interval=Duration.seconds(2),
        )

        post_ingest_task.add_catch(
            handler=self.dlq_chain_factory(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        # These policies must be attached to the *state machine role* in upload_stack.py
        # because the Distributed Map ItemReader and the custom Lambda invoke live in ASL,
        # not in a typed CDK task that auto-grants them.
        self.state_machine_policy_statements = [
            iam.PolicyStatement(
                actions=["lambda:InvokeFunction"],
                resources=[map_fn.function_arn],
            ),
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/temp/image-upload/*"],
            ),
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[self.file_bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["temp/image-upload/*"]}},
            ),
            iam.PolicyStatement(
                actions=[
                    "states:StartExecution",
                    "states:DescribeExecution",
                    "states:StopExecution",
                ],
                resources=["*"],
            ),
        ]

        self.pre_fn = pre_fn
        self.map_fn = map_fn
        self.post_fn = post_fn
        self.pre_ingest_task = pre_ingest_task
        self.map_state = map_state
        self.post_ingest_task = post_ingest_task

    def apply_ingest_permissions(self, lambda_fn: _lambda.Function) -> None:
        # 1) DynamoDB
        self.lock_table.grant_read_write_data(lambda_fn)
        self.job_table.grant_read_write_data(lambda_fn)
        self.sha256_table.grant_read_write_data(lambda_fn)

        # 2) S3 (file bucket): temp artifacts + bucket listing/location
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:DeleteObject", "s3:PutObject"],
                resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/temp/image-upload/*"],
            )
        )
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[self.file_bucket.bucket_arn],
            )
        )

        # 3) S3 (file bucket): Athena results prefix
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject", "s3:GetObject", "s3:AbortMultipartUpload"],
                resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/athena-results/*"],
            )
        )

        # 4) Athena: start and poll queries in the workgroup
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:StopQueryExecution",
                ],
                resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"],
            )
        )

        # 5) Firehose logging
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
                resources=[self.firehose_delivery_stream_attr_arn],
            )
        )

        # 6) Glue metadata read (catalog, DB, and tables)
        lambda_fn.add_to_role_policy(
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
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{self.iceberg_database_name}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/*",
                ],
            )
        )

        # 7) Glue metadata write (Iceberg tables update via Athena INSERT/DELETE/OPTIMIZE)
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:CreateTable",
                    "glue:UpdateTable",
                    "glue:DeleteTable",
                    "glue:BatchCreatePartition",
                    "glue:BatchDeletePartition",
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{self.iceberg_database_name}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/*",
                ],
            )
        )

        # Allow deleting only your CTAS temp tables (union of patterns)
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["glue:DeleteTable"],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/dedup_export_*",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/reg_export_*",
                ],
            )
        )

        # 8) S3 (iceberg bucket): read/write/delete Iceberg files; list limited prefixes
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:DeleteObject", "s3:PutObject"],
                resources=[f"arn:aws:s3:::{self.iceberg_bucket.bucket_name}/*"],
            )
        )
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[self.iceberg_bucket.bucket_arn],
                conditions={
                    "StringLike": {
                        "s3:prefix": ["canonical/*", "upload_staging/*", "image_source_membership/*"]
                    }
                },
            )
        )

        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["ssm:GetParameter", "ssm:GetParameters"],
                resources=["*"],
            )
        )