from aws_cdk import (
    Stack,
    Duration,
    CfnOutput,
    aws_iam as iam,
    aws_s3 as s3,
    aws_sqs as sqs,
    aws_dynamodb as dynamodb,
    aws_lambda as _lambda,
    custom_resources as cr
)
from constructs import Construct
import re

class StorageStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Derive a unique database name from the stack name
        db_name = re.sub(r'[^a-z0-9_]', '_', construct_id.lower()) + "_imagery_db"

        # 1. File bucket (d3 file bucket)
        file_bucket = s3.Bucket(
            self, "FileBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            lifecycle_rules=[
                s3.LifecycleRule(
                    prefix="temp/",
                    expiration=Duration.days(15)
                ),
                s3.LifecycleRule(
                    prefix="athena-results/",
                    expiration=Duration.days(15)
                ),
                s3.LifecycleRule(
                    prefix="temp/image-upload/",
                    expiration=Duration.days(30)
                )
            ]
        )

        # 2. Iceberg bucket
        iceberg_bucket = s3.Bucket(
            self, "IcebergBucket",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED
        )

        # 3. DynamoDB tables

        # Lock table
        lock_table = dynamodb.Table(
            self, "LockTable",
            partition_key=dynamodb.Attribute(name="lock_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )

        # Datasets table
        datasets_table = dynamodb.Table(
            self, "DatasetsTable",
            partition_key=dynamodb.Attribute(name="dataset_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )

        # Job table
        job_table = dynamodb.Table(
            self, "JobTable",
            partition_key=dynamodb.Attribute(name="job_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )

        # GSIs for job_table
        job_table.add_global_secondary_index(
            index_name="status-createdAt-index",
            partition_key=dynamodb.Attribute(name="status", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="created_at", type=dynamodb.AttributeType.STRING)
        )

        job_table.add_global_secondary_index(
            index_name="jobType-createdAt-index",
            partition_key=dynamodb.Attribute(name="job_type", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="created_at", type=dynamodb.AttributeType.STRING)
        )

        job_table.add_global_secondary_index(
            index_name="datasetId-createdAt-index",
            partition_key=dynamodb.Attribute(name="dataset_id", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="created_at", type=dynamodb.AttributeType.STRING)
        )

        # SHA256 lookup table
        sha256_table = dynamodb.Table(
            self, "Sha256LookupTable",
            partition_key=dynamodb.Attribute(name="sha256", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )

        # 4. Lambda for Iceberg DDL
        ddl_lambda = _lambda.Function(
            self, "IcebergDDL",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_asset("lambdas/storage/iceberg_ddl"),
            timeout=Duration.minutes(5),
            environment={
                "ICEBERG_BUCKET": iceberg_bucket.bucket_name,
                "ICEBERG_DATABASE": db_name,
                "ATHENA_OUTPUT": f"s3://{file_bucket.bucket_name}/athena-results/"
            }
        )

        file_bucket.grant_read_write(ddl_lambda)
        iceberg_bucket.grant_read_write(ddl_lambda)

        # IAM policy for Athena + Glue, scoped down
        ddl_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
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
                    "glue:CreateTable"
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{db_name}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{db_name}/*"
                ]
            )
        )

        # Custom resource to invoke the DDL Lambda at deploy time, after a destroy
        # it will reinvoke the lambda. So, iceberg table creation happens once, not
        # for an update, however.
        cr.AwsCustomResource(
            self, "RunIcebergDDL",
            on_create=cr.AwsSdkCall(
                service="Lambda",
                action="invoke",
                parameters={"FunctionName": ddl_lambda.function_name},
                physical_resource_id=cr.PhysicalResourceId.of("IcebergDDLRun")
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements([
                iam.PolicyStatement(
                    actions=["lambda:InvokeFunction"],
                    resources=[ddl_lambda.function_arn]
                )
            ])
        )

        # Global DLQ for async failures (S3→Lambda, Lambda async invokes, etc.)
        dlq = sqs.Queue(
            self, "GlobalDeadLetterQueue",
            retention_period=Duration.days(14),
            visibility_timeout=Duration.minutes(5)
        )

        # Expose constructs for cross-stack wiring
        self.file_bucket = file_bucket
        self.iceberg_bucket = iceberg_bucket
        self.lock_table = lock_table
        self.datasets_table = datasets_table
        self.job_table = job_table
        self.sha256_table = sha256_table
        self.global_dlq = dlq
        self.athena_database = db_name