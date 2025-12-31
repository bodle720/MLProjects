from typing import List
from dataclasses import dataclass

@dataclass
class LoggingConfig:
    transform_lambda_path: str
    firehose_interval_in_seconds: int
    firehose_size_in_m_bs: int
    transform_lambda_duration_sec: int
    transform_lambda_memory_size: int

@dataclass
class StorageConfig:
    ddl_lambda_path: str
    delete_db_lambda_path: str
    provider_ddl_lambda_path: str
    provider_cleanup_lambda_path: str

@dataclass
class UploadStateMachineConfig:
    duration_hours: int

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
class BatchTaskJobDefConfig:
    vcpus: int
    memory_limit_mib: int
    directory: str
    file: str

@dataclass
class BatchingStageConfig:
    file_batching: LambdaConfig
    batch_task_job_def: BatchTaskJobDefConfig

@dataclass
class IngestStageConfig:
    pre_ingest_lambda: LambdaConfig
    map_ingest_lambda: LambdaConfig
    map_max_concurrency: int
    post_ingest_lambda: LambdaConfig

@dataclass
class AppConfig:
    app_name: str
    logging: LoggingConfig
    storage: StorageConfig
    dlq_processor: LambdaConfig
    compute_env: ComputeEnvConfig
    upload_state_machine: UploadStateMachineConfig
    validation: BatchingStageConfig
    deduplication: BatchingStageConfig
    registration: BatchingStageConfig
    dedup_ingest: IngestStageConfig
    registration_ingest: IngestStageConfig
    kickoff_lambda: LambdaConfig
    cleanup_lambda: LambdaConfig