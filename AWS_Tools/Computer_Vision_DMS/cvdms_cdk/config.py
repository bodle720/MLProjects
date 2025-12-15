from config_models import (
    AppConfig, ComputeEnvConfig, UploadStateMachineConfig,
    StageConfig, FileBatchingConfig, BatchTaskJobDefConfig,
    KickoffLambdaConfig, CleanupLambdaConfig, StorageConfig,
    LoggingConfig, DLQProcessorConfig, DedupLambdaConfig
)

CONFIG = AppConfig(
    app_name="cvdmsv1",
    logging=LoggingConfig(
        transform_lambda_path="workers/lambdas/logging",
        transform_lambda_duration_sec=60,
        transform_lambda_memory_size=256,
        firehose_interval_in_seconds=60,
        firehose_size_in_m_bs=64
    ),
    storage=StorageConfig(
        ddl_lambda_path="workers/lambdas/storage/iceberg_ddl",
        delete_db_lambda_path="workers/lambdas/storage",
        provider_ddl_lambda_path="workers/lambdas/storage",
        provider_cleanup_lambda_path = "workers/lambdas/storage"
    ),
    dlq_processor=DLQProcessorConfig(
            path="workers/lambdas/storage",
            handler="dlq_processor.handler",
            memory_size=512,
            timeout_sec=30
        ),
    compute_env=ComputeEnvConfig(
        minv_cpus=0,
        maxv_cpus=64,
        instance_types=["m5.large", "m5.xlarge", "m5.2xlarge"]
    ),
    upload_state_machine=UploadStateMachineConfig(duration_hours=2),
    validation=StageConfig(
        file_batching=FileBatchingConfig(
            path="workers/lambdas/upload/file_batching",
            handler="validation.handler",
            memory_size=512,
            timeout_min=5
        ),
        batch_task_job_def=BatchTaskJobDefConfig(
            vcpus=1,
            memory_limit_mib=2048,
            directory="workers",
            file="batch_jobs/upload/validation/Dockerfile"
        )
    ),
    deduplication=StageConfig(
        file_batching=FileBatchingConfig(
            path="workers/lambdas/upload/file_batching",
            handler="deduplication.handler",
            memory_size=512,
            timeout_min=15
        ),
        batch_task_job_def=BatchTaskJobDefConfig(
            vcpus=1,
            memory_limit_mib=2048,
            directory="workers",
            file="batch_jobs/upload/deduplication/Dockerfile"
        )
    ),
    kickoff_lambda=KickoffLambdaConfig(
        path="workers/lambdas/upload",
        handler="kickoff.handler",
        memory_size=512,
        timeout_sec=30
    ),
    cleanup_lambda=CleanupLambdaConfig(
        path="workers/lambdas/upload",
        handler="cleanup.handler",
        memory_size=512,
        timeout_sec=30
    ),
    dedup_ingest_lambda = DedupLambdaConfig(
        path="workers/lambdas/upload",
        handler="deduplication_ingest.handler",
        memory_size=512,
        timeout_sec=30
    )
)