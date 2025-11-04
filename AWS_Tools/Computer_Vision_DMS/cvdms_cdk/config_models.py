from dataclasses import dataclass
from typing import List

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
    desiredv_cpus: int
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
class AppConfig:
    compute_env: ComputeEnvConfig
    upload_state_machine: UploadStateMachineConfig
    validation: StageConfig
    internal_dedup: StageConfig
    external_dedup: StageConfig
    registration: dict
    kickoff_lambda: KickoffLambdaConfig
