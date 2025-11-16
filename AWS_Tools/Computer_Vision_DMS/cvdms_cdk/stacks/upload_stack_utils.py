from aws_cdk import (
    Duration,
    Size,
    aws_lambda as _lambda,
    aws_iam as iam,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_batch as batch,
    aws_s3 as s3,
    aws_logs as logs,
    aws_dynamodb as dynamodb,
    aws_sqs as sqs,
    aws_ecs as ecs,
    aws_ecr_assets as ecr_assets
)

from constructs import Construct

from config_models import StageConfig

class BatchingStage(Construct):
    def __init__(self, scope: Construct, id: str, *,
                 stage_name: str,
                 config: StageConfig,
                 file_bucket: s3.Bucket,
                 job_table: dynamodb.Table,
                 sha256_table: dynamodb.Table,
                 phash_table: dynamodb.Table,
                 job_queue: batch.JobQueue,
                 athena_database_name: str,
                 ce_maxv_cpus: int,
                 region: str,
                 account: str,
                 global_dlq: sqs.Queue,
                 extra_lambda_env: dict = None,
                 extra_permissions: list[iam.PolicyStatement] = None,
                 extra_container_env: dict = None,
                 extra_map_state_params: dict = None):

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
            memory_size=config.file_batching.memory_size,
            timeout=Duration.minutes(config.file_batching.timeout_min),
            environment=lambda_env
        )

        # baseline grants
        sha256_table.grant_read_data(batching_fn)
        phash_table.grant_read_data(batching_fn)
        file_bucket.grant_read_write(batching_fn)

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
        # build/publish local Docker image from a local path
        image_asset = ecr_assets.DockerImageAsset(self, f"{stage_name}TaskImage",
                                                  directory=config.batch_task_job_def.path
                                                  )
        container_image = ecs.ContainerImage.from_registry(image_asset.image_uri)

        job_def = batch.EcsJobDefinition(
            self,
            f"{stage_name}JobDef",
            retry_attempts=5,
            timeout=Duration.hours(2),
            container=batch.EcsEc2ContainerDefinition(
                self,
                f"{stage_name}containerDefn",
                image=container_image,
                memory=Size.mebibytes(config.batch_task_job_def.memory_limit_mib),
                cpu=int(config.batch_task_job_def.vcpus * 1024),
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
            job_definition_arn=job_def.job_definition_arn,
            job_queue_arn=job_queue.job_queue_arn,
            job_name=f"{stage_name.lower()}-batch",
            container_overrides=tasks.BatchContainerOverrides(
                environment=container_env
            ),
            integration_pattern=sfn.IntegrationPattern.RUN_JOB
        )

        # 5. Map state (wired to Batch task)
        params = {
                "manifest.$": "$$.Map.Item.Value",
                "job_id.$": "$.job_id",
                "user.$": "$.user",
                "label_type.$": "$.label_type"
            }

        if extra_map_state_params:
            params.update(extra_map_state_params)

        map_state = sfn.Map(
            self, f"{stage_name}MapState",
            items_path="$.manifests",
            item_selector=params,
            max_concurrency=max(1, min(50, int(ce_maxv_cpus / max(1, config.batch_task_job_def.vcpus))))
        )

        # build iterator chain from the single batch task
        iterator_chain = sfn.Chain.start(batch_task)

        # attach to the Map using the most-compatible API available
        # prefer ItemProcessor + StateMachineFragment when both exist and StateMachineFragment is instantiable
        try:
            can_use_itemprocessor = hasattr(sfn, "ItemProcessor") and hasattr(sfn, "StateMachineFragment")
            if can_use_itemprocessor:
                # try to instantiate to ensure StateMachineFragment isn't abstract in this runtime
                try:
                    fragment = sfn.StateMachineFragment(self, "MapProcessorFragment", definition=iterator_chain)
                    # prefer explicit ItemProcessor when available
                    if hasattr(sfn, "ItemProcessor"):
                        map_state.item_processor(sfn.ItemProcessor(processor=fragment))
                    else:
                        # some CDK versions accept the fragment directly
                        map_state.item_processor(fragment)
                except TypeError:
                    # StateMachineFragment is abstract in this CDK; fall back to iterator
                    map_state.iterator(iterator_chain)
            else:
                # no ItemProcessor/StateMachineFragment support on this CDK - use the older API
                map_state.iterator(iterator_chain)
        except Exception:
            # last-resort fallback to the old API
            map_state.iterator(iterator_chain)

        # Expose entrypoints
        self.batching_task = batching_task
        self.map_state = map_state
        self.job_def = job_def


