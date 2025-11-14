from dataclasses import dataclass
from typing import List

from aws_cdk.aws_sns import LoggingConfig

@dataclass
class LoggingConfig:
    transform_lambda_path: str
    interval_in_seconds: int
    size_in_m_bs: int

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
    path: str

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
    compute_env: ComputeEnvConfig
    upload_state_machine: UploadStateMachineConfig
    validation: StageConfig
    internal_dedup: StageConfig
    external_dedup: StageConfig
    faiss_registration: StageConfig
    label_enrichment: StageConfig
    kickoff_lambda: KickoffLambdaConfig
    cleanup_lambda: CleanupLambdaConfig
