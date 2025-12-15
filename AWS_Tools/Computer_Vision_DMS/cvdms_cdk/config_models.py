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
class FileBatchingConfig:
    path: str
    handler: str
    memory_size: int
    timeout_min: int

@dataclass
class BatchTaskJobDefConfig:
    vcpus: int
    memory_limit_mib: int
    directory: str
    file: str

@dataclass
class StageConfig:
    file_batching: FileBatchingConfig
    batch_task_job_def: BatchTaskJobDefConfig

@dataclass
class ComputeEnvConfig:
    minv_cpus: int
    maxv_cpus: int
    instance_types: List[str]

@dataclass
class UploadStateMachineConfig:
    duration_hours: int

@dataclass
class KickoffLambdaConfig:
    path: str
    handler: str
    memory_size: int
    timeout_sec: int

@dataclass
class DLQProcessorConfig:
    path: str
    handler: str
    memory_size: int
    timeout_sec: int

@dataclass
class CleanupLambdaConfig:
    path: str
    handler: str
    memory_size: int
    timeout_sec: int

@dataclass
class AppConfig:
    app_name: str
    logging: LoggingConfig
    storage: StorageConfig
    dlq_processor: DLQProcessorConfig
    compute_env: ComputeEnvConfig
    upload_state_machine: UploadStateMachineConfig
    validation: StageConfig
    deduplication: StageConfig
    kickoff_lambda: KickoffLambdaConfig
    cleanup_lambda: CleanupLambdaConfig