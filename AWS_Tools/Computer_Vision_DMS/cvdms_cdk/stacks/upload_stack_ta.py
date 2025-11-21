from aws_cdk import (
    Stack,
    RemovalPolicy,
    Duration,
    aws_s3 as s3,
    aws_lambda as _lambda,
    aws_s3_notifications as s3n,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_lambda_event_sources as event_sources,
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_sqs as sqs,
    aws_logs as logs,
    aws_dynamodb as dynamodb,
    aws_kinesisfirehose as firehose
)
from constructs import Construct

from config import CONFIG
from config_models import ComputeEnvConfig, KickoffLambdaConfig, CleanupLambdaConfig

class ImageUploadStack(Stack):
    def __init__(self,
                 scope: Construct,
                 construct_id: str,
                 app_name: str,
                 common_utils_layer: _lambda.LayerVersion,
                 file_bucket: s3.Bucket,
                 iceberg_bucket: s3.Bucket,
                 job_table: dynamodb.Table,
                 sha256_table: dynamodb.Table,
                 phash_table: dynamodb.Table,
                 lock_table: dynamodb.Table,
                 global_dlq: sqs.Queue,
                 athena_database_name: str,
                 upload_events_queue: sqs.Queue,
                 firehose_delivery_stream: firehose.CfnDeliveryStream,  # L1 type
                 **kwargs) -> None:

        super().__init__(scope, construct_id, **kwargs)

        self.app_name = app_name
        self.file_bucket = file_bucket
        self.iceberg_bucket = iceberg_bucket
        self.job_table = job_table
        self.sha256_table = sha256_table
        self.phash_table = phash_table
        self.lock_table = lock_table
        self.global_dlq = global_dlq
        self.athena_database_name = athena_database_name
        self.upload_events_queue = upload_events_queue
        self.firehose_delivery_stream = firehose_delivery_stream
        self.common_utils_layer = common_utils_layer

        self.send_to_dlq = tasks.CallAwsService(self, "SendToDLQ",
                                           service="sqs",
                                           action="sendMessage",
                                           parameters={
                                               "QueueUrl": self.global_dlq.queue_url,
                                               "MessageBody.$": "$"  # send the entire failed state as the message body
                                           },
                                           iam_resources=[self.global_dlq.queue_arn]
                                           )

        send_to_dlq_fail = sfn.Fail(self, "SendToDLQFail", cause="StepFailed", error="StepError")
        self.send_to_dlq.next(send_to_dlq_fail)

        step1_task = self._make_first_step_lambda()
        step2_task = self._make_second_step_lambda()

        workflow_definition = (
            step1_task
            .next(step2_task)
        )

        upload_state_machine = sfn.StateMachine(self, "UploadStateMachine",
                              definition_body=sfn.DefinitionBody.from_chainable(workflow_definition),
                              timeout=Duration.hours(CONFIG.upload_state_machine.duration_hours)
                              )

        upload_state_machine.role.add_to_principal_policy(
            iam.PolicyStatement(actions=["sqs:SendMessage"], resources=[self.global_dlq.queue_arn]))

        upload_state_machine.apply_removal_policy(RemovalPolicy.DESTROY)

        self._make_kickoff_lambda(upload_state_machine, CONFIG.kickoff_lambda)

    def _make_kickoff_lambda(self,
                            upload_state_machine,
                            kickoff_config: KickoffLambdaConfig):
        # Make Kickoff lambda
        kickoff_lambda = _lambda.Function(
            self,
            "KickoffLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=kickoff_config.handler,
            code=_lambda.Code.from_asset(kickoff_config.path),
            layers=[self.common_utils_layer],
            dead_letter_queue=self.global_dlq,
            memory_size=kickoff_config.memory_size,
            timeout=Duration.seconds(kickoff_config.timeout_sec),
            environment={
                "JOB_TABLE_NAME": self.job_table.table_name,
                "FILE_BUCKET_NAME": self.file_bucket.bucket_name,
                "UPLOAD_STATE_MACHINE_ARN": upload_state_machine.state_machine_arn,
                "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref
            }
        )

        upload_state_machine.grant_start_execution(kickoff_lambda)

        # Permissions for the kickoff lambda
        self.job_table.grant_read_write_data(kickoff_lambda)
        self.file_bucket.grant_read(kickoff_lambda)

        # ensure S3 bucket-level list and get-location are permitted
        kickoff_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
        ))

        # explicitly allow GetObject on the athena-results prefix only if you will read it;
        # otherwise GetObject on whole bucket is already covered by grant_read above.
        kickoff_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/*"],
        ))

        kickoff_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[self.firehose_delivery_stream.attr_arn],  # use ARN provided by the L1
        ))

        # Trigger: S3 event for job.json, add the queue as an event source
        kickoff_lambda.add_event_source(event_sources.SqsEventSource(self.upload_events_queue, batch_size=1))
        self.upload_events_queue.grant_consume_messages(kickoff_lambda)

    def _make_first_step_lambda(self):
        ta_first_step_lambda = _lambda.Function(
            self,
            "FirstStepLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="ta_first_step.handler",
            code=_lambda.Code.from_asset("workers/lambdas"),
            layers=[self.common_utils_layer],
            dead_letter_queue=self.global_dlq,
            memory_size=256,
            timeout=Duration.seconds(60),
            environment={
                "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
                "JOB_TABLE_NAME": self.job_table.table_name
            }
        )

        self.job_table.grant_read_write_data(ta_first_step_lambda)
        self.file_bucket.grant_read(ta_first_step_lambda)

        ta_first_step_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"]
        ))

        ta_first_step_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/*"]
        ))

        ta_first_step_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[self.firehose_delivery_stream.attr_arn]
        ))

        step1_task = tasks.LambdaInvoke(self, "Step1Task",
                                        lambda_function=ta_first_step_lambda,
                                        result_path="$.step1",
                                        output_path="$",
                                        payload_response_only=True)

        step1_task.add_retry(backoff_rate=2.0, max_attempts=2, interval=Duration.seconds(2))

        step1_task.add_catch(self.send_to_dlq,
                             errors=["States.ALL"],
                             result_path="$.errorInfo"
                             )

        return step1_task

    def _make_second_step_lambda(self):
        ta_second_step_lambda = _lambda.Function(
            self,
            "SecondStepLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="ta_second_step.handler",
            code=_lambda.Code.from_asset("workers/lambdas"),
            layers=[self.common_utils_layer],
            dead_letter_queue=self.global_dlq,
            memory_size=256,
            timeout=Duration.seconds(60),
            environment={
                "LOG_FIREHOSE_STREAM_NAME": self.firehose_delivery_stream.ref,
                "JOB_TABLE_NAME": self.job_table.table_name
            }
        )

        self.job_table.grant_read_write_data(ta_second_step_lambda)
        self.file_bucket.grant_read(ta_second_step_lambda)

        ta_second_step_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket", "s3:GetBucketLocation"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}"],
        ))

        ta_second_step_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=[f"arn:aws:s3:::{self.file_bucket.bucket_name}/*"],
        ))

        ta_second_step_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
            resources=[self.firehose_delivery_stream.attr_arn],  # use ARN provided by the L1
        ))

        step2_task = tasks.LambdaInvoke(self, "Step2Task",
                                        lambda_function=ta_second_step_lambda,
                                        result_path="$.step2",
                                        output_path="$",
                                        payload_response_only=True)

        step2_task.add_retry(backoff_rate=2.0, max_attempts=2, interval=Duration.seconds(2))

        step2_task.add_catch(self.send_to_dlq,
                             errors=["States.ALL"],
                             result_path="$.errorInfo"
                             )

        return step2_task