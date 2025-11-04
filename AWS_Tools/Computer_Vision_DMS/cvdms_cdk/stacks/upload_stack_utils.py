from constructs import Construct

from aws_cdk import (
    Duration,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_batch as batch,
    aws_s3 as s3,
    aws_logs as logs,
    aws_dynamodb as dynamodb,
    aws_sqs as sqs
)

from config_models import StageConfig

class BatchingStage(Construct):
    def __init__(self, scope: Construct, id: str, *,
                 stage_name: str,
                 config: StageConfig,
                 file_bucket: s3.Bucket,
                 job_table: dynamodb.Table,
                 log_group: logs.LogGroup,
                 sha256_table: dynamodb.Table,
                 phash_table: dynamodb.Table,
                 job_queue: batch.JobQueue,
                 athena_database_name: str,
                 region: str,
                 account: str,
                 global_dlq: sqs.Queue,
                 extra_lambda_env: dict = None,
                 extra_permissions: list[iam.PolicyStatement] = None,
                 extra_container_env: dict = None):

        super().__init__(scope, id)

        # Lambda env vars
        lambda_env = {
            "FILE_BUCKET_NAME": file_bucket.bucket_name,
            "ATHENA_OUTPUT_S3": f"s3://{file_bucket.bucket_name}/athena-results/",
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DB": athena_database_name,
            "UPLOAD_STAGING_TABLE": "upload_staging",
            "SHA256_TABLE": sha256_table.table_name,
            "PHASH_TABLE": phash_table.table_name
        }

        if extra_lambda_env:
            lambda_env.update(extra_lambda_env)

        # --- Batching Lambda ---
        batching_fn = _lambda.Function(
            self, f"{stage_name}BatchingLambda",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler=config.file_batching.handler,
            code=_lambda.Code.from_asset(config.file_batching.path),
            dead_letter_queue=global_dlq,
            log_group=log_group,
            memory_size=config.file_batching.memory_size,
            timeout=Duration.minutes(config.file_batching.timeout_min),
            environment=lambda_env
        )

        # baseline grants
        sha256_table.grant_read_data(batching_fn)
        phash_table.grant_read_data(batching_fn)
        file_bucket.grant_read_write(batching_fn)
        log_group.grant_write(batching_fn)

        # baseline policies
        batching_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["s3:ListBucket","s3:GetBucketLocation"],
            resources=[file_bucket.bucket_arn]
        ))
        batching_fn.add_to_role_policy(iam.PolicyStatement(
            actions=["athena:StartQueryExecution","athena:GetQueryExecution","athena:GetQueryResults"],
            resources=[f"arn:aws:athena:{region}:{account}:workgroup/primary"]
        ))

        if extra_permissions:
            for stmt in extra_permissions:
                batching_fn.add_to_role_policy(stmt)

        # 2. LambdaInvoke task
        batching_task = tasks.LambdaInvoke(
            self, f"{stage_name}BatchingTask",
            lambda_function=batching_fn,
            output_path="$.Payload"
        )

        # 3. Job Role
        job_role = iam.Role(
            self, f"{stage_name}JobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )
        file_bucket.grant_read_write(job_role)
        job_table.grant_read_write_data(job_role)
        log_group.grant_write(job_role)
        phash_table.grant_read_data(job_role)
        sha256_table.grant_read_data(job_role)
        job_role.add_to_policy(iam.PolicyStatement(
            actions=["athena:StartQueryExecution","athena:GetQueryExecution","athena:GetQueryResults"],
            resources=[f"arn:aws:athena:{region}:{account}:workgroup/primary"]
        ))
        if extra_permissions:
            for stmt in extra_permissions:
                job_role.add_to_policy(stmt)

        # 4. Batch job definition + task
        job_def = batch.JobDefinition(
            self, f"{stage_name}JobDef",
            container=batch.JobDefinitionContainer(
                image=batch.EcrImage.from_asset(config.batch_task_job_def.path),
                vcpus=config.batch_task_job_def.vcpus,
                memory_limit_mib=config.batch_task_job_def.memory_limit_mib,
                job_role=job_role
            )
        )

        # baseline container env
        container_env = {
            "MANIFEST_S3_KEY": sfn.JsonPath.string_at("$.manifest"),
            "JOB_ID": sfn.JsonPath.string_at("$.job_id"),
            "USER": sfn.JsonPath.string_at("$.user"),
            "LABEL_TYPE": sfn.JsonPath.string_at("$.label_type"),
            "FILE_BUCKET_NAME": file_bucket.bucket_name,
            "ATHENA_OUTPUT_S3": f"s3://{file_bucket.bucket_name}/athena-results/",
            "ATHENA_WORKGROUP": "primary",
            "ICEBERG_DB": athena_database_name,
            "UPLOAD_STAGING_TABLE": "upload_staging"
        }
        if extra_container_env:
            container_env.update(extra_container_env)

        batch_task = tasks.BatchSubmitJob(
            self, f"{stage_name}BatchTask",
            job_definition=job_def,
            job_queue=job_queue,
            job_name=f"{stage_name.lower()}-batch",
            container_overrides=tasks.BatchContainerOverrides(
                environment=container_env
            ),
            integration_pattern=sfn.IntegrationPattern.RUN_JOB
        )

        # 5. Map state (wired to Batch task)
        map_state = sfn.Map(
            self, f"{stage_name}MapState",
            items_path="$.manifests",
            parameters={
                "manifest.$": "$$MAP_ITEM", # $$MAP_ITEM is the current array element (the manifest string)
                # assign manifest key in the iteration to the s3 uri pointing to the manifest.
                # pull job_id from the parent scope and assign to job_id an iteration
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "label_type.$": "$.label_type"
            }
        )
        map_state.iterator(batch_task)

        # Expose entrypoints
        self.batching_task = batching_task
        self.map_state = map_state


