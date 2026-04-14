from typing import List
from dataclasses import dataclass

####################################################################
# Config for the Logging Stack
####################################################################
@dataclass
class LoggingConfig:
    transform_lambda_path: str
    firehose_interval_in_seconds: int
    firehose_size_in_m_bs: int
    transform_lambda_duration_sec: int
    transform_lambda_memory_size: int

####################################################################
# Config for the Storage Stack
####################################################################
@dataclass
class StorageConfig:
    ddl_lambda_path: str
    delete_db_lambda_path: str
    provider_ddl_lambda_path: str
    provider_cleanup_lambda_path: str

####################################################################
# Reusable Configs for general tasks, e.g. Making a compute
# environment or a lambda function
####################################################################
@dataclass
class ComputeEnvConfig:
    minv_cpus: int
    maxv_cpus: int
    instance_types: List[str]

@dataclass
class LambdaConfig:
    path: str
    handler: str
    memory_size: int
    timeout_sec: int

@dataclass
class SQSConfig:
    retention_period_days: int
    visibility_timeout_minutes: int

@dataclass
class DLQOpsConfig:
    dlq_processor: LambdaConfig
    sqs_queue: SQSConfig

####################################################################
# Upload flow specific configs
####################################################################
@dataclass
class StateMachineConfig:
    duration_hours: int

@dataclass
class BatchTaskJobDefConfig:
    vcpus: int
    memory_limit_mib: int
    directory: str
    file: str

@dataclass
class BatchingStageConfig:
    file_batching: LambdaConfig
    map_max_concurrency: int
    batch_task_job_def: BatchTaskJobDefConfig

@dataclass
class IngestStageConfig:
    pre_ingest_lambda: LambdaConfig
    map_ingest_lambda: LambdaConfig
    map_max_concurrency: int
    post_ingest_lambda: LambdaConfig

####################################################################
# Config for the Upload Stack
####################################################################
@dataclass
class UploadConfig:
    upload_state_machine: StateMachineConfig
    kickoff_lambda: LambdaConfig
    compute_env: ComputeEnvConfig
    dlq_ops: DLQOpsConfig
    events_queue: SQSConfig
    validation: BatchingStageConfig
    deduplication: BatchingStageConfig
    registration: BatchingStageConfig
    validation_ingest: IngestStageConfig
    deduplication_ingest: IngestStageConfig
    registration_ingest: IngestStageConfig
    cleanup_lambda: LambdaConfig

@dataclass
class DatasetConfig:
    dataset_state_machine: StateMachineConfig
    kickoff_lambda: LambdaConfig
    dlq_ops: DLQOpsConfig
    events_queue: SQSConfig
    create_lambda: LambdaConfig
    update_lambda: LambdaConfig
    delete_lambda: LambdaConfig
    visualization_lambda: LambdaConfig
    cleanup_lambda: LambdaConfig

@dataclass
class AppConfig:
    app_name: str
    logging: LoggingConfig
    storage: StorageConfig
    upload: UploadConfig
    dataset: DatasetConfig