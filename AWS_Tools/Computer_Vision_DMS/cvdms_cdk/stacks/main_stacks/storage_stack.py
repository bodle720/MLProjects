from re import sub
from constructs import Construct
import hashlib

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CustomResource,
    aws_iam as iam,
    aws_s3 as s3,
    aws_logs as logs,
    aws_dynamodb as dynamodb,
    aws_lambda as _lambda,
    custom_resources as cr,
    aws_ssm as ssm,
    aws_sqs as sqs,
    aws_s3_notifications as s3n
)

from config import CONFIG

class StorageStack(Stack):

    def __init__(self,
                 scope: Construct,
                 construct_id: str,
                 *,
                 app_name: str,
                 **kwargs) -> None:
        '''
        This stack makes the file bucket that will store all files and the iceberg bucket, which will store all
        Iceberg table data. It makes the GLue database and relevant tables via a DDL lambda call at deploy time
        for said Iceberg tables.
        '''

        # The super call accepts env and initializes the self.account and self.region values
        # inside the base Stack class. So e can call them in this subclass.
        super().__init__(scope, construct_id, **kwargs)

        # Derive a unique glue database name from the stack name to store the iceberg table schema.
        # Make it deterministic, collision resistant, and length bounded.
        raw = sub(r"[^a-z0-9_]", "_", construct_id.lower())
        raw = sub(r"_+", "_", raw).strip("_")
        if not raw:
            raw = "db"
        elif raw[0].isdigit():
            raw = f"db_{raw}"

        h = hashlib.sha1(construct_id.encode("utf-8")).hexdigest()[:8]
        suffix = "_imagery_db"
        max_len = 255

        base = f"{raw}_{h}"
        base = base[: max_len - len(suffix)]
        iceberg_database_name = f"{base}{suffix}"

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

        # Datasets bucket
        datasets_bucket = s3.Bucket(
            self, "DatasetsBucket",
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

        datasets_table.add_global_secondary_index(
            index_name="createdAt-index",
            partition_key=dynamodb.Attribute(name="created_at", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.ALL
        )

        # Dataset versions table (one row per dataset_id + version snapshot)
        dataset_versions_table = dynamodb.Table(
            self,
            "DatasetVersionsTable",
            partition_key=dynamodb.Attribute(name="dataset_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="version", type=dynamodb.AttributeType.NUMBER),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Job table
        job_table = dynamodb.Table(
            self, "JobsTable",
            partition_key=dynamodb.Attribute(name="job_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            # time_to_live_attribute="ttl" # Unix epoch timestamp (seconds)
        )

        # GSIs for job_table
        job_table.add_global_secondary_index(
            index_name="status-createdAt-index",
            partition_key=dynamodb.Attribute(name="status", type=dynamodb.AttributeType.STRING),
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
                    "ATHENA_OUTPUT_S3": f"s3://{file_bucket.bucket_name}/athena-results/"
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

        # Let's make the event queues the Upload and Dataset stacks will need

        # Upload Queue for upload kickoff to poll
        upload_events_dlq = sqs.Queue(
            self, "UploadEventsDLQ",
            retention_period=Duration.days(14)
        )

        upload_events_queue = sqs.Queue(
            self, "UploadEventsQueue",
            visibility_timeout=Duration.minutes(CONFIG.upload.events_queue.visibility_timeout_minutes),
            retention_period=Duration.days(CONFIG.upload.events_queue.retention_period_days),
            dead_letter_queue=sqs.DeadLetterQueue(
                queue=upload_events_dlq,
                max_receive_count=1
            )
        )

        file_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.SqsDestination(upload_events_queue),
            s3.NotificationKeyFilter(prefix="temp/image-upload/", suffix="/job.json")
        )

        # Make the Dataset stack events queue for the dataset kickoff lambda
        dataset_events_dlq = sqs.Queue(
            self,
            "DatasetEventsDLQ",
            retention_period=Duration.days(14)
        )

        dataset_events_queue = sqs.Queue(
            self,
            "DatasetEventsQueue",
            visibility_timeout=Duration.minutes(CONFIG.dataset.events_queue.visibility_timeout_minutes),
            retention_period=Duration.days(CONFIG.dataset.events_queue.retention_period_days),
            dead_letter_queue=sqs.DeadLetterQueue(
                queue=dataset_events_dlq,
                max_receive_count=1
            )
        )

        file_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.SqsDestination(dataset_events_queue),
            s3.NotificationKeyFilter(
                prefix="temp/dataset-ops/",
                suffix="/submission.json"
            )
        )

        # Expose constructs for cross-stack wiring
        self.file_bucket = file_bucket
        self.iceberg_bucket = iceberg_bucket
        self.datasets_bucket = datasets_bucket

        self.job_table = job_table
        self.sha256_table = sha256_table
        self.lock_table = lock_table

        self.iceberg_database_name = iceberg_database_name

        # Expose the datasets table as well
        self.datasets_table = datasets_table
        self.dataset_versions_table = dataset_versions_table

        # Expose the queues
        self.upload_events_queue = upload_events_queue
        self.dataset_events_queue = dataset_events_queue

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

        ssm.StringParameter(self, "DatasetsBucketNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/datasets_bucket_name",
                            string_value=datasets_bucket.bucket_name
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

        ssm.StringParameter(self, "DatasetVersionsTableNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/dataset_versions_table_name",
                            string_value=dataset_versions_table.table_name
                            )

        ssm.StringParameter(self, "LockTableNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/lock_table_name",
                            string_value=lock_table.table_name
                            )

        ssm.StringParameter(self, "Sha256TableNameParam",
                            parameter_name=f"/cvdms/{app_name}/storage/sha256_table_name",
                            string_value=sha256_table.table_name
                            )

        ssm.StringParameter(self, "UploadEventsQueueNameParam",
                            parameter_name=f"/cvdms/{app_name}/upload/upload_events_queue_name",
                            string_value=upload_events_queue.queue_name)

        ssm.StringParameter(self,"DatasetEventsQueueNameParam",
            parameter_name=f"/cvdms/{app_name}/dataset/dataset_events_queue_name",
            string_value=dataset_events_queue.queue_name)