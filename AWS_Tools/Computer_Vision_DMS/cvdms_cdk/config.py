from config_models import (
    AppConfig, ComputeEnvConfig, UploadStateMachineConfig,
    StageConfig, FileBatchingConfig, BatchTaskJobDefConfig,
    KickoffLambdaConfig
)

CONFIG = AppConfig(
    compute_env=ComputeEnvConfig(
        minv_cpus=0,
        desiredv_cpus=0,
        maxv_cpus=64,
        instance_types=["m5.large", "m5.xlarge"]
    ),
    upload_state_machine=UploadStateMachineConfig(duration_hours=2),
    validation=StageConfig(
        file_batching=FileBatchingConfig(
            path="lambdas/upload/file_batching",
            handler="file_batching_validation.handler",
            memory_size=512,
            timeout_min=5
        ),
        batch_task_job_def=BatchTaskJobDefConfig(
            vcpus=1,
            memory_limit_mib=2048,
            path="lambdas/upload/validation"
        )
    ),
    internal_dedup=StageConfig(
        file_batching=FileBatchingConfig(
            path="lambdas/upload/file_batching",
            handler="file_batching_internal_dedup.handler",
            memory_size=512,
            timeout_min=5
        ),
        batch_task_job_def=BatchTaskJobDefConfig(
            vcpus=1,
            memory_limit_mib=2048,
            path="lambdas/upload/internal_dedup"
        )
    ),
    external_dedup=StageConfig(
        file_batching=FileBatchingConfig(
            path="lambdas/upload/file_batching",
            handler="file_batching_external_dedup.handler",
            memory_size=512,
            timeout_min=5
        ),
        batch_task_job_def=BatchTaskJobDefConfig(
            vcpus=1,
            memory_limit_mib=4096,
            path="lambdas/upload/external_dedup"
        )
    ),
    kickoff_lambda=KickoffLambdaConfig(
        path="lambdas/upload/kickoff",
        handler="kickoff.handler",
        memory_size=512,
        timeout_sec=30
    )
)
