from datetime import datetime
from constructs import Construct
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_iam as iam,
    aws_s3 as s3,
    aws_logs as logs,
    aws_lambda as _lambda,
    aws_kinesisfirehose as firehose,
    aws_glue as glue,
)

from config import CONFIG

class LoggingStack(Stack):
    def __init__(self,
                 scope: Construct,
                 construct_id: str,
                 app_name: str,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # --- S3 bucket for Parquet logs (auto-delete on destroy) ---
        log_bucket = s3.Bucket(self, "LogsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True
        )

        # --- Lambda: transform function to normalize logs to stable JSON ---
        transform_fn = _lambda.Function(self, "FirehoseTransformLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="log_transformer.handler",
            code=_lambda.Code.from_asset(CONFIG.logging.transform_lambda_path),
            timeout=Duration.seconds(30),
            memory_size=256
        )

        # Create explicit LogGroup for the transform Lambda so we can destroy it on stack delete
        # Lambda auto-creates /aws/lambda/<name>, but creating a LogGroup with the exact name
        # and DESTROY removal policy ensures the logs are removed with the stack.
        transform_log_group = logs.LogGroup(self, "TransformLogGroup",
            log_group_name=f"/aws/lambda/{transform_fn.function_name}",
            removal_policy=RemovalPolicy.DESTROY,
            retention=logs.RetentionDays.ONE_DAY,  # short retention for transform lambda logs
        )

        # Allow Firehose to invoke the transform lambda
        transform_fn.grant_invoke(iam.ServicePrincipal("firehose.amazonaws.com"))

        # --- Glue Database and Table for Data Format Conversion / Athena ---
        glue_db = glue.CfnDatabase(self, "LogsGlueDatabase",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=f"{app_name.lower()}_logs_db"
            )
        )

        # Glue table: minimal structure for normalized logs; partitioned by year/month/day
        glue_table = glue.CfnTable(self, "LogsGlueTable",
            catalog_id=self.account,
            database_name=glue_db.ref,
            table_input=glue.CfnTable.TableInputProperty(
                name=f"{app_name.lower()}_logs_table",
                table_type="EXTERNAL_TABLE",
                parameters={
                    "classification": "parquet",
                },
                partition_keys=[
                    glue.CfnTable.ColumnProperty(name="year", type="int"),
                    glue.CfnTable.ColumnProperty(name="month", type="int"),
                    glue.CfnTable.ColumnProperty(name="day", type="int"),
                ],
                storage_descriptor=glue.CfnTable.StorageDescriptorProperty(
                    columns=[
                        glue.CfnTable.ColumnProperty(name="job_id", type="string"),
                        glue.CfnTable.ColumnProperty(name="event_type", type="string"),
                        glue.CfnTable.ColumnProperty(name="message", type="string"),
                        glue.CfnTable.ColumnProperty(name="warnings", type="string"),
                        glue.CfnTable.ColumnProperty(name="errors", type="string"),
                        glue.CfnTable.ColumnProperty(name="timestamp", type="timestamp"),
                    ],
                    input_format="org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                    output_format="org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                    serde_info=glue.CfnTable.SerdeInfoProperty(
                        serialization_library="org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                        parameters={}
                    ),
                    location=f"s3://{log_bucket.bucket_name}/logs/",
                    stored_as_sub_directories=False
                )
            )
        )

        # --- IAM role for Firehose (access to S3, Lambda invoke, Glue/Catalog) ---
        firehose_role = iam.Role(self, "FirehoseServiceRole",
            assumed_by=iam.ServicePrincipal("firehose.amazonaws.com"),
            inline_policies={
                "FirehoseS3WritePolicy": iam.PolicyDocument(statements=[
                    iam.PolicyStatement(
                        actions=["s3:PutObject", "s3:PutObjectAcl", "s3:AbortMultipartUpload", "s3:ListBucket", "s3:GetBucketLocation"],
                        resources=[log_bucket.bucket_arn, f"{log_bucket.bucket_arn}/*"],
                    )
                ]),
                "FirehoseLambdaInvokePolicy": iam.PolicyDocument(statements=[
                    iam.PolicyStatement(
                        actions=["lambda:InvokeFunction", "lambda:GetFunctionConfiguration"],
                        resources=[transform_fn.function_arn],
                    )
                ]),
                "FirehoseGluePolicy": iam.PolicyDocument(statements=[
                    iam.PolicyStatement(
                        actions=[
                            "glue:GetTable",
                            "glue:GetTableVersion",
                            "glue:GetTableVersions",
                            "glue:GetDatabase",
                            "glue:CreatePartition",
                            "glue:GetPartition",
                            "glue:BatchCreatePartition"
                        ],
                        resources=["*"]
                    )
                ])
            }
        )

        # Grant bucket write to role via granting the role the s3:PutObject etc (already via inline policy)
        log_bucket.grant_read_write(firehose_role)

        # --- Firehose delivery stream (CfnDeliveryStream because high-level L2 lacks some config) ---
        # Configure DataFormatConversionConfiguration for Parquet conversion with Glue.
        # The transform Lambda will normalize incoming records into JSON; DataFormatConversion converts JSON->Parquet.
        delivery_stream = firehose.CfnDeliveryStream(self, "LogsDeliveryStream",
                             delivery_stream_type="DirectPut",
                             extended_s3_destination_configuration=firehose.CfnDeliveryStream.ExtendedS3DestinationConfigurationProperty(
                                 bucket_arn=log_bucket.bucket_arn,
                                 role_arn=firehose_role.role_arn,
                                 prefix="logs/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/",
                                 buffering_hints=firehose.CfnDeliveryStream.BufferingHintsProperty(
                                     interval_in_seconds=60,
                                     size_in_m_bs=5
                                 ),
                                 compression_format="UNCOMPRESSED",
                                 processing_configuration=firehose.CfnDeliveryStream.ProcessingConfigurationProperty(
                                     enabled=True,
                                     processors=[
                                         firehose.CfnDeliveryStream.ProcessorProperty(
                                             type="Lambda",
                                             parameters=[
                                                 firehose.CfnDeliveryStream.ProcessorParameterProperty(
                                                     parameter_name="LambdaArn",
                                                     parameter_value=transform_fn.function_arn
                                                 ),
                                                 firehose.CfnDeliveryStream.ProcessorParameterProperty(
                                                     parameter_name="NumberOfRetries",
                                                     parameter_value="3"
                                                 ),
                                                 firehose.CfnDeliveryStream.ProcessorParameterProperty(
                                                     parameter_name="RoleArn",
                                                     parameter_value=firehose_role.role_arn
                                                 ),
                                             ]
                                         )
                                     ]
                                 ),
                                 data_format_conversion_configuration=firehose.CfnDeliveryStream.DataFormatConversionConfigurationProperty(
                                     enabled=True,
                                     input_format_configuration=firehose.CfnDeliveryStream.InputFormatConfigurationProperty(
                                         deserializer=firehose.CfnDeliveryStream.DeserializerProperty(
                                             open_x_json_ser_de=firehose.CfnDeliveryStream.OpenXJsonSerDeProperty(
                                                 case_insensitive=False,
                                                 column_to_json_key_mappings={}
                                             )
                                         )
                                     ),
                                     output_format_configuration=firehose.CfnDeliveryStream.OutputFormatConfigurationProperty(
                                         serializer=firehose.CfnDeliveryStream.SerializerProperty(
                                             parquet_ser_de=firehose.CfnDeliveryStream.ParquetSerDeProperty()
                                         )
                                     ),
                                     schema_configuration=firehose.CfnDeliveryStream.SchemaConfigurationProperty(
                                         database_name=glue_db.ref,
                                         table_name=glue_table.table_input.name,
                                         region=self.region,
                                         role_arn=firehose_role.role_arn,
                                         version_id="LATEST"
                                     )
                                 )
                             )
                        )

        # Allow Firehose to read the Lambda source mapping permission explicitly (AWS requires permission for lambda:InvokeFunction)
        # transform_fn.grant_invoke(...) done above

        # Glue table depends on bucket and delivery stream indirectly; ensure correct creation order
        glue_table.add_depends_on(glue_db)

        # Outputs
        self.log_bucket = log_bucket
        self.firehose_delivery_stream_name = delivery_stream.ref
        self.glue_db_name = glue_db.ref
        self.glue_table_name = glue_table.table_input.name

