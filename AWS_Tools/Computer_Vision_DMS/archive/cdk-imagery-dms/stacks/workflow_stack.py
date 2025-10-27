# -*- coding: utf-8 -*-
"""
Workflow Stack
"""

# stacks/workflow_stack.py
from aws_cdk import Stack, Duration
from constructs import Construct
from aws_cdk.aws_lambda import DockerImageFunction, DockerImageCode, Architecture
from aws_cdk.aws_sqs import Queue
from aws_cdk.aws_s3 import Bucket
from aws_cdk.aws_events import Rule
from aws_cdk.aws_events_targets import SfnStateMachine, RuleTargetInput
from aws_cdk.aws_stepfunctions import StateMachine, Choice, Condition, JsonPath, DefinitionBody, LogOptions, LogLevel
from aws_cdk.aws_stepfunctions_tasks import DynamoUpdateItem, DynamoAttributeValue, BatchSubmitJob, SqsSendMessage
from aws_cdk.aws_dynamodb import Table
from aws_cdk.aws_logs import LogGroup

class WorkflowStack(Stack):
    def __init__(self, scope: Construct, id: str, *,
                 buckets: dict, queues: dict, tables: dict,
                 datalake: dict, batch: dict, logs: dict, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        central_log_group: LogGroup = logs["central"]
        regular_bucket: Bucket = buckets["regular"]
        ingest_queue: Queue = queues["ingest"]
        pipeline_dlq: Queue = queues["pipelineDlq"]
        jobs_table: Table = tables["jobs"]
        lock_table: Table = tables["lock"]

        # Lambda built & pushed automatically from local Docker context
        polling_fn = DockerImageFunction(self, "PollingLambda",
            code=DockerImageCode.from_image_asset("./lambdas/polling"),
            architecture=Architecture.X86_64,
            timeout=Duration.minutes(3),
            environment={
                "REGULAR_BUCKET": regular_bucket.bucket_name,
                "JOBS_TABLE": jobs_table.table_name,
                "PIPELINE_DLQ_URL": pipeline_dlq.queue_url,
                "CENTRAL_LOG_GROUP": central_log_group.log_group_name
            }
        )

        regular_bucket.grant_read_write(polling_fn)
        jobs_table.grant_read_write_data(polling_fn)
        ingest_queue.grant_consume_messages(polling_fn)
        pipeline_dlq.grant_send_messages(polling_fn)
        central_log_group.grant_write(polling_fn)

        polling_fn.add_event_source_mapping("PollingSqsMapping",
            event_source_arn=ingest_queue.queue_arn,
            batch_size=10
        )

        rule = Rule(self, "StagedUploadReadyRule",
            event_pattern={"source": ["cv.pipeline"], "detail-type": ["STAGED_UPLOAD_READY"]}
        )

        # Step Functions tasks (no hardcoded ARNs; use ComputeStack outputs)
        job_defs = batch["jobDefs"]

        acquire_lock = DynamoUpdateItem(self, "AcquireLock",
            table=lock_table,
            key={"singleton": DynamoAttributeValue.from_string("global")},
            update_expression="SET locked = :true, locked_by = :job",
            condition_expression="attribute_not_exists(locked) OR locked = :false",
            expression_attribute_values={
                ":true": DynamoAttributeValue.from_boolean(True),
                ":false": DynamoAttributeValue.from_boolean(False),
                ":job": DynamoAttributeValue.from_string(JsonPath.string_at("$.job_id")),
            },
            result_path=JsonPath.DISCARD
        )

        submit_embeddings = BatchSubmitJob(self, "SubmitEmbeddingJob",
            job_definition_arn=job_defs["embeddingsArn"],
            job_name=JsonPath.string_at("$.job_id"),
            job_queue_arn=batch["gpuQueueArn"],
            container_overrides=[{
                "command": ["python", "/app/embed.py",
                            "--manifest", JsonPath.string_at("$.manifest_s3_uri"),
                            "--output", JsonPath.string_at("$.staging_embeddings_s3_prefix"),
                            "--job_id", JsonPath.string_at("$.job_id")]
            }],
            payload=JsonPath.object_at("$")
        )

        small_dedup = BatchSubmitJob(self, "SmallBatchDedup",
            job_definition_arn=job_defs["smallDedupArn"],
            job_name=JsonPath.string_at("$.job_id"),
            job_queue_arn=batch["cpuQueueArn"],
            container_overrides=[{
                "command": ["python", "/app/small_dedup.py",
                            "--staging", JsonPath.string_at("$.staging_embeddings_s3_prefix"),
                            "--job_id", JsonPath.string_at("$.job_id")]
            }],
            payload=JsonPath.object_at("$")
        )

        large_dedup = BatchSubmitJob(self, "LargeBatchDedup",
            job_definition_arn=job_defs["largeDedupArn"],
            job_name=JsonPath.string_at("$.job_id"),
            job_queue_arn=batch["cpuQueueArn"],
            container_overrides=[{
                "command": ["python", "/app/large_dedup.py",
                            "--staging", JsonPath.string_at("$.staging_embeddings_s3_prefix"),
                            "--job_id", JsonPath.string_at("$.job_id")]
            }],
            payload=JsonPath.object_at("$")
        )

        global_dedup = BatchSubmitJob(self, "GlobalDedup",
            job_definition_arn=job_defs["globalDedupArn"],
            job_name=JsonPath.string_at("$.job_id"),
            job_queue_arn=batch["cpuQueueArn"],
            container_overrides=[{
                "command": ["python", "/app/global_dedup.py",
                            "--faiss", JsonPath.string_at("$.faiss_index_s3_uri"),
                            "--staging", JsonPath.string_at("$.staging_embeddings_s3_prefix"),
                            "--job_id", JsonPath.string_at("$.job_id")]
            }],
            payload=JsonPath.object_at("$")
        )

        persist_job = BatchSubmitJob(self, "PersistJob",
            job_definition_arn=job_defs["persistArn"],
            job_name=JsonPath.string_at("$.job_id"),
            job_queue_arn=batch["cpuQueueArn"],
            container_overrides=[{
                "command": ["python", "/app/persist.py",
                            "--imagery_table", JsonPath.string_at("$.imagery_table"),
                            "--embeddings_table", JsonPath.string_at("$.embeddings_table"),
                            "--images_bucket", regular_bucket.bucket_name,
                            "--faiss_out", JsonPath.string_at("$.faiss_index_s3_uri"),
                            "--job_id", JsonPath.string_at("$.job_id")]
            }],
            payload=JsonPath.object_at("$")
        )

        release_lock = DynamoUpdateItem(self, "ReleaseLock",
            table=lock_table,
            key={"singleton": DynamoAttributeValue.from_string("global")},
            update_expression="SET locked = :false, locked_by = :null",
            expression_attribute_values={
                ":false": DynamoAttributeValue.from_boolean(False),
                ":null": DynamoAttributeValue.from_null(),
            },
            result_path=JsonPath.DISCARD
        )

        send_failure_to_dlq = SqsSendMessage(self, "SendFailureToDLQ",
            queue=pipeline_dlq,
            message_body=JsonPath.object_at("$")
        )

        branch = Choice(self, "SmallOrLargeBatch") \
            .when(Condition.number_less_than_equals("$.batch_size", 1000),
                  small_dedup.next(global_dedup)) \
            .otherwise(large_dedup.next(global_dedup))

        definition = acquire_lock.add_catch(send_failure_to_dlq, result_path="$.error") \
            .next(submit_embeddings.add_catch(send_failure_to_dlq, result_path="$.error")) \
            .next(branch.add_catch(send_failure_to_dlq, result_path="$.error")) \
            .next(persist_job.add_catch(send_failure_to_dlq, result_path="$.error")) \
            .next(release_lock.add_catch(send_failure_to_dlq, result_path="$.error"))

        sm = StateMachine(self, "PipelineStateMachine",
            definition_body=DefinitionBody.from_chainable(definition),
            timeout=Duration.hours(2),
            logs=LogOptions(
                destination=central_log_group,
                level=LogLevel.ALL,
                include_execution_data=True
            )
        )

        rule.add_target(SfnStateMachine(sm,
            input=RuleTargetInput.from_event_path("$.detail"),
            dead_letter_queue=pipeline_dlq
        ))

        pipeline_dlq.grant_send_messages(sm)
        central_log_group.grant_write(sm.role)



