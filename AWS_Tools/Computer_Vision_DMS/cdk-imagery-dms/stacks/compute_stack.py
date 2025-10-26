# -*- coding: utf-8 -*-
"""
Compute Stack
"""

# stacks/compute_stack.py
from aws_cdk import Stack
from constructs import Construct
from aws_cdk.aws_ec2 import Vpc
from aws_cdk.aws_batch import CfnComputeEnvironment, CfnJobQueue, CfnJobDefinition
from aws_cdk.aws_ecr_assets import DockerImageAsset
from aws_cdk.aws_ecr_assets import DockerImageAsset
from aws_cdk.aws_batch import CfnJobDefinition

from aws_cdk import (
    aws_batch as batch,
    aws_ssm as ssm,
)

class ComputeStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        vpc = Vpc(self, "BatchVpc", max_azs=2)

        cpu_env = CfnComputeEnvironment(self, "CpuSpotEnv",
            type="MANAGED",
            compute_environment_name="cpu-spot-env",
            compute_resources={
                "type": "SPOT",
                "maxvCpus": 256,
                "minvCpus": 0,
                "desiredvCpus": 0,
                "instanceTypes": ["c5.large", "c5.xlarge"],
                "subnets": [s.subnet_id for s in vpc.private_subnets]
            },
            state="ENABLED"
        )

        gpu_env = CfnComputeEnvironment(self, "GpuSpotEnv",
            type="MANAGED",
            compute_environment_name="gpu-spot-env",
            compute_resources={
                "type": "SPOT",
                "maxvCpus": 128,
                "minvCpus": 0,
                "desiredvCpus": 0,
                "instanceTypes": ["g4dn.xlarge", "g5.xlarge"],
                "subnets": [s.subnet_id for s in vpc.private_subnets]
            },
            state="ENABLED"
        )

        cpu_queue = CfnJobQueue(self, "CpuJobQueue",
            job_queue_name="cpu-job-queue",
            priority=1,
            compute_environment_order=[{"order": 1, "computeEnvironment": cpu_env.attr_compute_environment_arn}],
            state="ENABLED"
        )

        gpu_queue = CfnJobQueue(self, "GpuJobQueue",
            job_queue_name="gpu-job-queue",
            priority=2,
            compute_environment_order=[{"order": 1, "computeEnvironment": gpu_env.attr_compute_environment_arn}],
            state="ENABLED"
        )

        # Build Docker images from local folders (no hardcoded URIs)
        embeddings_img = DockerImageAsset(self, "EmbeddingsImage", directory="./batch/embeddings")
        small_dedup_img = DockerImageAsset(self, "SmallDedupImage", directory="./batch/small_dedup")
        large_dedup_img = DockerImageAsset(self, "LargeDedupImage", directory="./batch/large_dedup")
        global_dedup_img = DockerImageAsset(self, "GlobalDedupImage", directory="./batch/global_dedup")
        persist_img = DockerImageAsset(self, "PersistImage", directory="./batch/persist")

        # Helper for job definition creation
        def job_def(name: str, image_uri: str, vcpus: int, memory: int, gpu: bool = False,
                    stream_prefix: str = None) -> CfnJobDefinition:
            container = {
                "image": image_uri,
                "vcpus": vcpus,
                "memory": memory,
                "command": ["python", "/app/main.py"],  # overridden at submit
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/cv/central",
                        "awslogs-region": self.region,
                        "awslogs-stream-prefix": stream_prefix or f"batch/{name}"
                    }
                }
            }
            if gpu:
                container["resourceRequirements"] = [{"type": "GPU", "value": "1"}]

            return CfnJobDefinition(self, f"{name.capitalize()}JobDef",
                job_definition_name=name,
                type="container",
                platform_capabilities=["EC2"],
                container_properties=container,
                retry_strategy={"attempts": 2}
            )

        embeddings_jd = job_def("embeddings", embeddings_img.image_uri, vcpus=4, memory=8192, gpu=True, stream_prefix="batch/embeddings")
        small_dedup_jd = job_def("small-dedup", small_dedup_img.image_uri, vcpus=2, memory=4096, stream_prefix="batch/small-dedup")
        large_dedup_jd = job_def("large-dedup", large_dedup_img.image_uri, vcpus=4, memory=8192, stream_prefix="batch/large-dedup")
        global_dedup_jd = job_def("global-dedup", global_dedup_img.image_uri, vcpus=4, memory=8192, stream_prefix="batch/global-dedup")
        persist_jd = job_def("persist", persist_img.image_uri, vcpus=4, memory=8192, stream_prefix="batch/persist")

        # Expose ARNs so WorkflowStack can submit jobs without hardcoding
        self.outputs = {
            "cpuQueueArn": cpu_queue.attr_job_queue_arn,
            "gpuQueueArn": gpu_queue.attr_job_queue_arn,
            "jobDefs": {
                "embeddingsArn": embeddings_jd.attr_job_definition_arn,
                "smallDedupArn": small_dedup_jd.attr_job_definition_arn,
                "largeDedupArn": large_dedup_jd.attr_job_definition_arn,
                "globalDedupArn": global_dedup_jd.attr_job_definition_arn,
                "persistArn": persist_jd.attr_job_definition_arn
            }
        }
        
        
        splitter_img = DockerImageAsset(self, "SplitterImage", directory="./batch/splitter")
        
        splitter_jd = CfnJobDefinition(self, "SplitterJobDef",
            job_definition_name="splitter",
            type="container",
            platform_capabilities=["EC2"],
            container_properties={
                "image": splitter_img.image_uri,
                "vcpus": 2,
                "memory": 4096,
                "command": ["python", "/app/split.py",
                            "--dataset_id", "Ref::dataset_id",
                            "--task_type", "Ref::task_type"],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/cv/central",
                        "awslogs-region": self.region,
                        "awslogs-stream-prefix": "batch/splitter"
                    }
                }
            },
            retry_strategy={"attempts": 1}
        )
        
        self.outputs["splitterJobArn"] = splitter_jd.attr_job_definition_arn
        
        # Write job queue and job def into SSM
        ssm.StringParameter(
            self, "CpuJobQueueParam",
            parameter_name="/cv-platform/cpu-job-queue",
            string_value=cpu_queue.job_queue_name
        )
        
        ssm.StringParameter(
            self, "SplitterJobDefParam",
            parameter_name="/cv-platform/splitter-job-def",
            string_value=splitter_jd.attr_job_definition_arn
        )
                
        
        



