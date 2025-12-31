from typing import Callable
from constructs import Construct

from aws_cdk import (
    Duration,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_iam as iam,
    aws_dynamodb as dynamodb
)

from config_models import IngestStageConfig

class IngestStage(Construct):
    def __init__(self,
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
                 iceberg_database_name: str,
                 region: str,
                 account: str,
                 dlq_chain_factory: Callable[[], sfn.Chain],
                 firehose_delivery_stream_name: str,
                 firehose_delivery_stream_attr_arn: str):

        super().__init__(scope, construct_id)

        self.stage_name = stage_name
        self.config = config
        self.common_utils_layer = common_utils_layer
        self.file_bucket = file_bucket
        self.iceberg_bucket = iceberg_bucket
        self.job_table = job_table
        self.lock_table = lock_table
        self.iceberg_database_name = iceberg_database_name
        self.region = region
        self.account = account
        self.dlq_chain_factory = dlq_chain_factory
        self.firehose_delivery_stream_name = firehose_delivery_stream_name
        self.firehose_delivery_stream_attr_arn = firehose_delivery_stream_attr_arn

        lambda_env = {
                "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
                "ATHENA_OUTPUT_S3": f"s3://{self.file_bucket.bucket_name}/athena-results/",
                "ATHENA_WORKGROUP": "primary",
                "ICEBERG_DATABASE_NAME": self.iceberg_database_name,
                "UPLOAD_STAGING_TABLE_NAME": "upload_staging",
                "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream_name,
                "CANONICAL_IMAGERY_TABLE_NAME": "canonical_imagery"
            }

        # --------------------------------------
        # Form the 'pre' lambda task.
        # --------------------------------------
        pre_fn = _lambda.Function(
            self, f"{stage_name}PreLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=config.pre_ingest_lambda.handler,
            code=_lambda.Code.from_asset(config.pre_ingest_lambda.path),
            layers=[common_utils_layer],
            memory_size=config.pre_ingest_lambda.memory_size,
            timeout=Duration.seconds(config.pre_ingest_lambda.timeout_sec),
            environment=lambda_env
        )

        self.apply_ingest_permissions(pre_fn)

        pre_ingest_task = tasks.LambdaInvoke(
            self, f"{stage_name}PreLambdaTask",
            lambda_function=pre_fn,
            result_path=f"$.{stage_name}PreLambdaTask",
            output_path="$",
            timeout=Duration.seconds(config.pre_ingest_lambda.timeout_sec),
            payload_response_only=True)

        pre_ingest_task.add_catch(
            handler=self.dlq_chain_factory(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        # --------------------------------------
        # Form the map task.
        # --------------------------------------
        map_fn = _lambda.Function(
            self, f"{stage_name}MapLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=config.map_ingest_lambda.handler,
            code=_lambda.Code.from_asset(config.map_ingest_lambda.path),
            layers=[common_utils_layer],
            memory_size=config.map_ingest_lambda.memory_size,
            timeout=Duration.seconds(config.map_ingest_lambda.timeout_sec),
            environment=lambda_env
        )

        self.apply_ingest_permissions(map_fn)

        map_task = tasks.LambdaInvoke(
            self, f"{stage_name}MapLambdaTask",
            lambda_function=map_fn,
            result_path=sfn.JsonPath.DISCARD,
            output_path="$",
            timeout=Duration.seconds(config.map_ingest_lambda.timeout_sec),
            payload_response_only=True)

        map_state = sfn.Map(
            self, f"{stage_name}MapState",
            items_path=f"$.{stage_name}PreLambdaTask.shards",
            item_selector={
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "event_type.$": "$.event_type",
                "label_type.$": "$.label_type",
                "data_source.$": "$.data_source",

                "shard.$": "$$.Map.Item.Value.shard",
                "rows_read.$": "$$.Map.Item.Value.rows_read",

                # always present for both modes
                "upload_key.$": "$$.Map.Item.Value.upload_key",

                # optional for registration mode
                "imagery_key.$": "$$.Map.Item.Value.imagery_key",
                "labels_key.$": "$$.Map.Item.Value.labels_key",
            },
            result_path=sfn.JsonPath.DISCARD,
            output_path="$",
            max_concurrency=config.map_max_concurrency
        )

        map_state.add_catch(
            handler=self.dlq_chain_factory(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        map_state.iterator(sfn.Chain.start(map_task))

        # --------------------------------------
        # Form the 'post' lambda task.
        # --------------------------------------
        post_fn = _lambda.Function(
            self, f"{stage_name}PostLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=config.post_ingest_lambda.handler,
            code=_lambda.Code.from_asset(config.post_ingest_lambda.path),
            layers=[common_utils_layer],
            memory_size=config.post_ingest_lambda.memory_size,
            timeout=Duration.seconds(config.post_ingest_lambda.timeout_sec),
            environment=lambda_env
        )

        self.apply_ingest_permissions(post_fn)

        post_ingest_task = tasks.LambdaInvoke(
            self, f"{stage_name}PostLambdaTask",
            lambda_function=post_fn,
            result_path=f"$.{stage_name}PostLambdaTask",
            output_path="$",
            timeout=Duration.seconds(config.post_ingest_lambda.timeout_sec),
            payload_response_only=True)

        post_ingest_task.add_catch(
            handler=self.dlq_chain_factory(),
            errors=["States.ALL"],
            result_path="$.errorInfo",
        )

        self.pre_ingest_task = pre_ingest_task
        self.map_state = map_state
        self.post_ingest_task = post_ingest_task

    def apply_ingest_permissions(self, lambda_fn: _lambda.Function) -> None:

        # 1) DynamoDB
        self.lock_table.grant_read_write_data(lambda_fn)
        self.job_table.grant_read_write_data(lambda_fn)

        # 2) S3 (file bucket): temp artifacts + bucket listing/location
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:DeleteObject"],
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
                actions=["athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults"],
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
                    "glue:GetDatabase", "glue:GetDatabases",
                    "glue:GetTable", "glue:GetTables",
                    "glue:GetPartition", "glue:GetPartitions",
                    "glue:GetTableVersion", "glue:GetTableVersions",
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{self.iceberg_database_name}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{self.iceberg_database_name}/*",
                ],
            )
        )

        # 7) Glue metadata write (Iceberg tables update via Athena INSERT/DELETE/OPTIMIZE)
        # Use the broader registration scope (all tables in the DB), since it subsumes upload_staging-only.
        lambda_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:CreateTable", "glue:UpdateTable", "glue:DeleteTable",
                    "glue:BatchCreatePartition", "glue:BatchDeletePartition",
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
        # Use broad object-level access across bucket (registration needs canonical/*; dedup needs upload_staging/*).
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
                conditions={"StringLike": {"s3:prefix": ["canonical/*", "upload_staging/*"]}},
            )
        )