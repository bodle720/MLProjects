from aws_cdk import (
    Stack,
    Duration,
    CfnOutput,
    aws_iam as iam,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    aws_lambda as _lambda,
    custom_resources as cr
)
from constructs import Construct

class StorageStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

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
                "ICEBERG_DATABASE": "imagery_db",
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
                    f"arn:aws:glue:{self.region}:{self.account}:database/imagery_db",
                    f"arn:aws:glue:{self.region}:{self.account}:table/imagery_db/*"
                ]
            )
        )

        # Custom resource to invoke the DDL Lambda at deploy time
        cr.AwsCustomResource(
            self, "RunIcebergDDL",
            on_create=cr.AwsSdkCall(
                service="Lambda",
                action="invoke",
                parameters={"FunctionName": ddl_lambda.function_name},
                physical_resource_id=cr.PhysicalResourceId.of("IcebergDDLRun")
            ),
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE
            )
        )

        # Outputs (for cross-stack references)
        CfnOutput(self, "FileBucketName", value=file_bucket.bucket_name, export_name="FileBucketName")
        CfnOutput(self, "FileBucketArn", value=file_bucket.bucket_arn, export_name="FileBucketArn")

        CfnOutput(self, "IcebergBucketName", value=iceberg_bucket.bucket_name, export_name="IcebergBucketName")
        CfnOutput(self, "IcebergBucketArn", value=iceberg_bucket.bucket_arn, export_name="IcebergBucketArn")

        CfnOutput(self, "LockTableName", value=lock_table.table_name, export_name="LockTableName")
        CfnOutput(self, "DatasetsTableName", value=datasets_table.table_name, export_name="DatasetsTableName")
        CfnOutput(self, "JobTableName", value=job_table.table_name, export_name="JobTableName")
        CfnOutput(self, "Sha256TableName", value=sha256_table.table_name, export_name="Sha256TableName")

        CfnOutput(self, "IcebergDDLFunctionName", value=ddl_lambda.function_name, export_name="IcebergDDLFunctionName")
        CfnOutput(self, "IcebergDDLFunctionArn", value=ddl_lambda.function_arn, export_name="IcebergDDLFunctionArn")

        CfnOutput(self, "AthenaDatabase", value="imagery_db", export_name="AthenaDatabase")
        CfnOutput(self, "AthenaOutputLocation", value=f"s3://{file_bucket.bucket_name}/athena-results/",
                  export_name="AthenaOutputLocation")