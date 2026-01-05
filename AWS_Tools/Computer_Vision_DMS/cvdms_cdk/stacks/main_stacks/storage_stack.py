from re import sub
from constructs import Construct

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CustomResource,
    aws_iam as iam,
    aws_s3 as s3,
    aws_sqs as sqs,
    aws_logs as logs,
    aws_dynamodb as dynamodb,
    aws_lambda as _lambda,
    custom_resources as cr,
    aws_ssm as ssm,
    aws_s3_notifications as s3n,
    aws_kinesisfirehose as firehose,
    aws_lambda_event_sources as event_sources
)

from config import CONFIG

class StorageStack(Stack):

    def __init__(self,
                 scope: Construct,
                 construct_id: str,
                 app_name: str,
                 common_utils_layer: _lambda.LayerVersion,
                 firehose_delivery_stream: firehose.CfnDeliveryStream,
                 **kwargs) -> None:
        '''
        This stack makes the file bucket that will store all files and the iceberg bucket, which will store all
        Iceberg table data. It makes the GLue database and relevant tables via a DDL lambda call at deploy time
        for said Iceberg tables.
        '''

        # The super call accepts env and initializes the self.account and self.region values
        # inside the base Stack class. So e can call them in this subclass.
        super().__init__(scope, construct_id, **kwargs)

        self.common_utils_layer = common_utils_layer
        self.firehose_delivery_stream = firehose_delivery_stream

        # Derive a unique glue database name from the stack name to store the iceberg table schema
        iceberg_database_name = sub(r'[^a-z0-9_]', '_', construct_id.lower()) + "_imagery_db"

        # File bucket (S3 file bucket to hold files)
        file_bucket = s3.Bucket(
            self, "S3FileBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    prefix="temp/",
                    expiration=Duration.days(15)
                ),
                s3.LifecycleRule(
                    prefix="athena-results/",
                    expiration=Duration.days(15)
                )
            ]
        )

        # Iceberg bucket
        iceberg_bucket = s3.Bucket(
            self, "S3IcebergTablesBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        # DynamoDB tables
        # Lock table
        lock_table = dynamodb.Table(
            self, "LockTable",
            partition_key=dynamodb.Attribute(name="lock_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        seed_lock = cr.AwsCustomResource(
            self, "SeedLockRow",
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=[lock_table.table_arn]
            ),
            on_create=cr.AwsSdkCall(
                service="DynamoDB",
                action="putItem",
                parameters={
                    "TableName": lock_table.table_name,
                    "Item": {
                        "lock_id": {"S": "global"},
                        "locked": {"BOOL": False}
                    },
                    # Only insert if lock_id doesn't already exist
                    "ConditionExpression": "attribute_not_exists(lock_id)"
                },
                physical_resource_id=cr.PhysicalResourceId.of("SeedLockRowResource"),
            ),
        )

        seed_lock.node.add_dependency(lock_table)

        # Datasets table
        datasets_table = dynamodb.Table(
            self, "DatasetsTable",
            partition_key=dynamodb.Attribute(name="dataset_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        # Job table
        job_table = dynamodb.Table(
            self, "JobsTable",
            partition_key=dynamodb.Attribute(name="job_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        # GSIs for job_table
        job_table.add_global_secondary_index(
            index_name="status-createdAt-index",
            partition_key=dynamodb.Attribute(name="status", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="created_at", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.ALL
        )

        job_table.add_global_secondary_index(
            index_name="jobType-createdAt-index",
            partition_key=dynamodb.Attribute(name="job_type", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="created_at", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.ALL
        )

        job_table.add_global_secondary_index(
            index_name="datasetId-createdAt-index",
            partition_key=dynamodb.Attribute(name="dataset_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="created_at", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.ALL
        )

        # SHA256 lookup table
        sha256_table = dynamodb.Table(
            self, "Sha256LookupTable",
            partition_key=dynamodb.Attribute(name="sha256", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        # Lambda for Iceberg DDL, always auto deleted, lambdas cannot be retained on cdk destroy.
        # DDL = Data Definition Language, defines the database schema, a subset language of SQL.
        ddl_lambda = _lambda.Function(
            self, "LambdaIcebergDDL",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_asset(CONFIG.storage.ddl_lambda_path),
            timeout=Duration.minutes(15),
            environment={
                    "ICEBERG_BUCKET_NAME": iceberg_bucket.bucket_name,
                    "ICEBERG_DATABASE_NAME": iceberg_database_name,
                    "S3_ATHENA_OUTPUT_URI": f"s3://{file_bucket.bucket_name}/athena-results/"
                }
        )

        # Ensure explicit log group for the DDL lambda so we can destroy it
        logs.LogRetention(self, f"{ddl_lambda.node.id}LogGroup",
                          log_group_name=f"/aws/lambda/{ddl_lambda.function_name}",
                          retention=logs.RetentionDays.THREE_DAYS,
                          removal_policy=RemovalPolicy.DESTROY
                          )

        file_bucket.grant_read_write(ddl_lambda)
        iceberg_bucket.grant_read_write(ddl_lambda)

        # IAM policy for Athena + Glue, scoped down
        ddl_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:StopQueryExecution"
                ],
                resources=[
                    f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"
                ]
            )
        )

        # The Athena SQL commands translates them into Glue API calls under the hood,
        # which will store the schema in the Glue Data Catalog, a component of AWS Glue.
        # GLue will store the catalog/schema that allows queries to "understand" the
        # tables' schemas (it is a schema registry).
        ddl_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:GetDatabase",
                    "glue:CreateDatabase",
                    "glue:GetTable",
                    "glue:CreateTable",
                    "glue:UpdateTable",
                    "glue:GetPartition",
                    "glue:CreatePartition"
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{iceberg_database_name}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{iceberg_database_name}/*"
                ]
            )
        )

        # ensure bucket-level list and get-location plus prefix object access
        ddl_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:ListBucket",
                    "s3:GetBucketLocation"
                ],
                resources=[
                    f"arn:aws:s3:::{file_bucket.bucket_name}",
                    f"arn:aws:s3:::{iceberg_bucket.bucket_name}"
                ]
            )
        )

        ddl_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject"
                ],
                resources=[
                    f"arn:aws:s3:::{file_bucket.bucket_name}/athena-results/*",
                    f"arn:aws:s3:::{file_bucket.bucket_name}/*",
                    f"arn:aws:s3:::{iceberg_bucket.bucket_name}/*"
                ]
            )
        )

        # -------------------------------------------------------------------
        # Create a provider Lambda that we control. This Lambda will be invoked
        # as the custom resource provider and can in turn invoke the ddl_lambda.
        # -------------------------------------------------------------------
        provider_ddl_fn = _lambda.SingletonFunction(self, "IcebergDDLProviderFn",
                                       uuid="6f1a8f2e-1c9b-4a2a-9f6b-0d5b7e4f1234",
                                       runtime=_lambda.Runtime.PYTHON_3_11,
                                       handler="custom_resource_provider_ddl.handler",
                                       code=_lambda.Code.from_asset(CONFIG.storage.provider_ddl_lambda_path),
                                       timeout=Duration.minutes(14),
                                       memory_size=256,
                                       environment={
                                           "DDL_FUNCTION_NAME": ddl_lambda.function_name,
                                       }
                                       )

        logs.LogRetention(self, f"{provider_ddl_fn.node.id}LogGroup",
                          log_group_name=f"/aws/lambda/{provider_ddl_fn.function_name}",
                          retention=logs.RetentionDays.ONE_DAY,
                          removal_policy=RemovalPolicy.DESTROY
                          )

        provider_ddl_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["lambda:InvokeFunction"],
            resources=[ddl_lambda.function_arn]
        ))

        # -------------------------------------------------------------------
        # Create the custom resource provider and the custom resource.
        # Using cr.Provider gives you a CloudFormation CustomResource backed by
        # a Lambda function (provider_ddl_fn) that you control.
        # -------------------------------------------------------------------
        provider_ddl = cr.Provider(self, "IcebergDDLProvider",
                               on_event_handler=provider_ddl_fn
                               )

        # The custom resource that triggers provider to run on create/update/delete as you define in provider logic
        CustomResource(self, "RunIcebergDDL",
                       service_token=provider_ddl.service_token,
                       removal_policy=RemovalPolicy.DESTROY
                       )

        # Global DLQ for async failures (S3->Lambda, Lambda async invokes, etc.)
        dlq = sqs.Queue(
            self, "GlobalDeadLetterQueue",
            retention_period=Duration.days(14),
            visibility_timeout=Duration.minutes(5),
            removal_policy=RemovalPolicy.DESTROY
        )

        cleanup_fn = _lambda.Function(
            self, "DatabaseCleanupLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="delete_database.handler",
            code=_lambda.Code.from_asset(CONFIG.storage.delete_db_lambda_path),
            timeout=Duration.minutes(10),
            environment={
                "ICEBERG_DATABASE_NAME": iceberg_database_name
            }
        )

        logs.LogRetention(self, f"{cleanup_fn.node.id}LogGroup",
                          log_group_name=f"/aws/lambda/{cleanup_fn.function_name}",
                          retention=logs.RetentionDays.THREE_DAYS,
                          removal_policy=RemovalPolicy.DESTROY
                          )

        cleanup_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:GetDatabases",
                    "glue:GetDatabase",
                    "glue:DeleteDatabase",
                    "glue:GetTables",
                    "glue:GetTable",
                    "glue:DeleteTable",
                    "glue:GetUserDefinedFunctions",
                    "glue:DeleteUserDefinedFunction"
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{iceberg_database_name}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{iceberg_database_name}/*",
                    f"arn:aws:glue:{self.region}:{self.account}:userDefinedFunction/{iceberg_database_name}/*"
                ]
            )
        )

        # Provider Lambda that will invoke the cleanup lambda on Delete
        provider_cleanup_fn = _lambda.SingletonFunction(self, "GlueCleanupProviderFn",
                                       uuid="6f1a8f2e-1b9b-4a2a-9f6b-0d5b7e4f4321",
                                       runtime=_lambda.Runtime.PYTHON_3_11,
                                       handler="custom_resource_provider_cleanup.handler",  # see provider example below
                                       code=_lambda.Code.from_asset(CONFIG.storage.provider_cleanup_lambda_path),
                                       timeout=Duration.minutes(14),
                                       memory_size=256,
                                       environment={
                                           "CLEANUP_FUNCTION_NAME": cleanup_fn.function_name
                                       }
                                       )

        logs.LogRetention(self, f"{provider_cleanup_fn.node.id}LogGroup",
                          log_group_name=f"/aws/lambda/{provider_cleanup_fn.function_name}",
                          retention=logs.RetentionDays.ONE_DAY,
                          removal_policy=RemovalPolicy.DESTROY
                          )

        # grant provider permission to invoke the cleanup lambda
        provider_cleanup_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["lambda:InvokeFunction"],
            resources=[cleanup_fn.function_arn]
        ))

        # Create a provider backed by our provider lambda
        provider_cleanup = cr.Provider(self, "GlueCleanupProvider",
                               on_event_handler=provider_cleanup_fn
                               )

        CustomResource(self, "DropGlueDatabaseOnDelete",
                       service_token=provider_cleanup.service_token,
                       removal_policy=RemovalPolicy.DESTROY
                       )

        upload_events_queue = sqs.Queue(self, "UploadEventsQueue",
                                        visibility_timeout=Duration.minutes(5),
                                        retention_period=Duration.days(4))

        file_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.SqsDestination(upload_events_queue),
            s3.NotificationKeyFilter(prefix="temp/image-upload/", suffix="/job.json")
        )

        # Make a lambda that polls the dlq and processes the messages
        dlq_processor = _lambda.Function(
            self,
            "DLQProcessor",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=CONFIG.dlq_processor.handler,
            code=_lambda.Code.from_asset(CONFIG.dlq_processor.path),
            layers=[self.common_utils_layer],
            memory_size=CONFIG.dlq_processor.memory_size,
            timeout=Duration.seconds(CONFIG.dlq_processor.timeout_sec),
            environment={
                "JOB_TABLE_NAME": job_table.table_name,
                "FILE_BUCKET_NAME": file_bucket.bucket_name,
                "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
                "UPLOAD_STAGING_TABLE_NAME": "upload_staging",
                "LOCK_TABLE_NAME": lock_table.table_name,
                "ATHENA_WORKGROUP": "primary",
                "ICEBERG_DATABASE_NAME": iceberg_database_name,
                "ATHENA_OUTPUT_S3": f"s3://{file_bucket.bucket_name}/athena-results/",
                "SHA256_TABLE_NAME": sha256_table.table_name
            }
        )

        # 1) DynamoDB
        lock_table.grant_read_write_data(dlq_processor)
        job_table.grant_read_write_data(dlq_processor)
        sha256_table.grant_read_write_data(dlq_processor)

        # 2) S3: delete temp files under temp/image-upload/ and read them
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:DeleteObject"],
            resources=[f"arn:aws:s3:::{file_bucket.bucket_name}/temp/image-upload/*"]
        ))
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[f"arn:aws:s3:::{file_bucket.bucket_name}"],
            conditions={"StringLike": {"s3:prefix": ["temp/image-upload/*"]}}
        ))

        # 3) S3: Athena results write only to athena-results/
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:PutObject", "s3:GetObject"],
            resources=[f"arn:aws:s3:::{file_bucket.bucket_name}/athena-results/*"]
        ))
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{file_bucket.bucket_name}"]
        ))

        # 4) Athena: start and poll queries in the workgroup
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults"],
            resources=[f"arn:aws:athena:{self.region}:{self.account}:workgroup/primary"]
        ))

        # 5) Firehose logging
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[self.firehose_delivery_stream.attr_arn]
        ))

        # 6) Glue metadata read (catalog, DB, and tables)
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "glue:GetDatabase", "glue:GetDatabases",
                "glue:GetTable", "glue:GetTables",
                "glue:GetPartition", "glue:GetPartitions",
                "glue:GetTableVersion", "glue:GetTableVersions"
            ],
            resources=[
                f"arn:aws:glue:{self.region}:{self.account}:catalog",
                f"arn:aws:glue:{self.region}:{self.account}:database/{iceberg_database_name}",
                f"arn:aws:glue:{self.region}:{self.account}:table/{iceberg_database_name}/*"
            ]
        ))

        # 7) Glue metadata write for upload_staging (required when Athena DELETE/OPTIMIZE updates Iceberg metadata)
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "glue:CreateTable", "glue:UpdateTable", "glue:DeleteTable",
                "glue:BatchCreatePartition", "glue:BatchDeletePartition"
            ],
            resources=[f"arn:aws:glue:{self.region}:{self.account}:catalog",
                       f"arn:aws:glue:{self.region}:{self.account}:database/{iceberg_database_name}",
                       f"arn:aws:glue:{self.region}:{self.account}:table/{iceberg_database_name}/upload_staging"
                       ]
        ))

        # 8) S3: read and delete Iceberg files for upload_staging prefix
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject", "s3:DeleteObject", "s3:PutObject"],
            resources=[f"arn:aws:s3:::{iceberg_bucket.bucket_name}/upload_staging/*"]
        ))
        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"],
            resources=[f"arn:aws:s3:::{iceberg_bucket.bucket_name}"],
            conditions={"StringLike": {"s3:prefix": ["upload_staging/*"]}}
        ))

        dlq_processor.add_to_role_policy(iam.PolicyStatement(
            actions=["glue:DeleteTable"],
            resources=[
                f"arn:aws:glue:{self.region}:{self.account}:table/{iceberg_database_name}/dedup_export_*"
            ],
        ))

        dlq_processor.add_event_source(event_sources.SqsEventSource(dlq, batch_size=10))
        dlq.grant_consume_messages(dlq_processor)

        # Expose constructs for cross-stack wiring
        self.file_bucket = file_bucket
        self.iceberg_bucket = iceberg_bucket
        self.job_table = job_table
        self.sha256_table = sha256_table
        self.lock_table = lock_table
        self.global_dlq = dlq
        self.datasets_table = datasets_table
        self.iceberg_database_name = iceberg_database_name
        self.upload_events_queue = upload_events_queue

        # SSM params
        # Buckets
        ssm.StringParameter(self, "FileBucketNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/file_bucket_name",
                            string_value=file_bucket.bucket_name
                            )

        ssm.StringParameter(self, "IcebergBucketNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/iceberg_bucket_name",
                            string_value=iceberg_bucket.bucket_name
                            )

        # Glue / Athena
        ssm.StringParameter(self, "AthenaDatabaseNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/iceberg_database_name",
                            string_value=iceberg_database_name
                            )

        # DynamoDB Tables
        ssm.StringParameter(self, "JobTableNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/job_table_name",
                            string_value=job_table.table_name
                            )

        ssm.StringParameter(self, "DatasetsTableNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/datasets_table_name",
                            string_value=datasets_table.table_name
                            )

        ssm.StringParameter(self, "LockTableNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/lock_table_name",
                            string_value=lock_table.table_name
                            )

        ssm.StringParameter(self, "Sha256TableNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/sha256_table_name",
                            string_value=sha256_table.table_name
                            )

        # Queues
        ssm.StringParameter(self, "GlobalDlqNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/global_dlq_name",
                            string_value=dlq.queue_name
                            )

        ssm.StringParameter(self, "UploadEventsQueueNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/upload_events_queue_name",
                            string_value=upload_events_queue.queue_name
                            )