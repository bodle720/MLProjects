import uuid
from constructs import Construct
from stacks.helper_constructs.dlq_ops import DLQOps

from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_lambda_event_sources as event_sources,
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_sqs as sqs,
    aws_dynamodb as dynamodb,
    aws_kinesisfirehose as firehose,
    aws_s3_notifications as s3n,
    aws_ssm as ssm
)

from config import CONFIG
from config_models import LambdaConfig

class DatasetStack(Stack):
    def __init__(self,
                 scope: Construct,
                 construct_id: str,
                 *,
                 app_name: str,
                 common_utils_layer: _lambda.LayerVersion,
                 file_bucket: s3.Bucket,
                 iceberg_bucket: s3.Bucket,
                 job_table: dynamodb.Table,
                 sha256_table: dynamodb.Table,
                 lock_table: dynamodb.Table,
                 iceberg_database_name: str,
                 firehose_delivery_stream: firehose.CfnDeliveryStream,  # L1 type
                 **kwargs) -> None:

        super().__init__(scope, construct_id, **kwargs)
