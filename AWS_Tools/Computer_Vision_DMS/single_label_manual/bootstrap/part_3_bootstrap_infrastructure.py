# -*- coding: utf-8 -*-
import os
import sys
import logging
import boto3
import argparse     
 
from helpers.validation_helpers import VALID_REGIONS
from helpers.ecr_docker_helpers import build_and_push_docker_image_to_ecr
from helpers.lambda_helpers import create_lambda_function
from helpers.bootstrap_helpers import create_roles, make_bucket, \
                                     make_queues, make_tables

def load_config_from_ssm(ssm_client, infrastructure_name: str) -> dict:
    """
    Loads all parameters for a given infrastructure_name from SSM Parameter Store
    and returns them as a dict.

    Raises:
        ValueError: if no parameters are found for the given infrastructure_name.
    """
    prefix = f"/cv-datasets/single-label/{infrastructure_name}/infrastructure/"
    config = {}

    paginator = ssm_client.get_paginator("get_parameters_by_path")
    for page in paginator.paginate(Path=prefix, Recursive=True, WithDecryption=True):
        for param in page.get("Parameters", []):
            key = param["Name"].split("/")[-1]  # last segment is the config key
            config[key] = param["Value"]

    if not config:
        raise ValueError(
            f"No parameters found under {prefix}. "
            f"Did you run part 2 to register this infrastructure?"
        )

    return config

def store_arns_in_ssm(ssm_client, infrastructure_name, role_arns, queue_arns):
    prefix = f"/cv-datasets/single-label/{infrastructure_name}/infrastructure/"
    
    for key, value in {**role_arns, **queue_arns}.items():
        ssm_client.put_parameter(
            Name=f"{prefix}{key}",
            Value=value,
            Type="String",
            Overwrite=True,  # allow updates if re-run
            Description=f"{key} for {infrastructure_name}"
        )
        logging.info(f"Stored {key} -> {prefix}{key}")

if __name__ == "__main__":
    # -------------------------------
    # Configure logging settings
    # -------------------------------
    main_dir = os.path.dirname(__file__)
    base_logs = os.path.join(main_dir, "logs")
    os.makedirs(base_logs, exist_ok=True)
        
    logging_save_to = os.path.join(base_logs, "part_3_logs.txt")
    
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear() 
        
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(logging_save_to)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--infrastructure_name", required=True,
                        help="The infrastructure namespace to bootstrap (must already exist in SSM).")
    
    parser.add_argument("--region", required=True, help="Region where infra was registered")
    parser.add_argument("--profile_name", required=True, help="AWS profile to use for credentials")

    args = parser.parse_args()
    infrastructure_name = args.infrastructure_name
    region = args.region.lower()
    profile_name = args.profile_name

    if region not in VALID_REGIONS:
        logging.error(f'Invalid region: {region}')
        sys.exit(1)
        
    session = boto3.Session(profile_name=profile_name, region_name=region)
    
    ssm = session.client("ssm")
    config = load_config_from_ssm(ssm, infrastructure_name)
    
    if region != config['AWS_REGION']:
        logger.error(f"Region mismatch, the region specified ({region}) does not math the infrastructure region ({config['AWS_REGION']})")
        sys.exit(1)
        
    logger.info(f"Loaded {len(config)} parameters for {infrastructure_name}")
     
    # -------------------------------
    # Get clients and validate config.
    # -------------------------------
    
    clients = {
            "sts": session.client("sts"),
            "s3": session.client("s3"),
            "sqs": session.client("sqs"),
            "ddb": session.client("dynamodb"),
            "lambda": session.client("lambda"),
            "iam": session.client("iam"),
            "ecr": session.client("ecr")
        }
    
    # -------------------------------
    # Make the roles for each Lambda.
    # -------------------------------
    logger.info('Creating lambda roles.')
    role_arns = create_roles(config, clients)
    
    # -------------------------------
    # Make the S3 bucket if it doesn't exist.
    # -------------------------------
    logger.info('Making S3 bucket.')
    make_bucket(config, clients)
    
    # -------------------------------
    # Make the SQS Queues.
    # -------------------------------
    logger.info('Making SQS queues.')
    queue_arns = make_queues(config, clients)
    
    # Store the ARNs in the Parameter Store for future teardown
    store_arns_in_ssm(ssm, infrastructure_name, role_arns, queue_arns)
    
    # -------------------------------
    # Make the DynamoDB Tables.
    # -------------------------------
    logger.info('Making DynamoDB tables.')
    make_tables(config, clients)
    
    # -------------------------------
    # Make the Lambda functions.
    # -------------------------------
    logger.info('Making Lambda functions.')

    account_id = clients['sts'].get_caller_identity()["Account"]
    lambda_client = clients['lambda']
    ecr_client = clients['ecr']
    region = config['AWS_REGION']
    local_tag = ecr_tag = 'latest'
    for_lambda_fn = True
    from_docker = True

    env_vars = {'Variables': {'AWS_REGION': config['AWS_REGION'],
                              'S3_BUCKET_NAME': config['S3_BUCKET_NAME'],
                              'S3_DATASETS_ROOT': config['S3_DATASETS_ROOT'],
                              'DDB_IMAGERY_TABLE': config['DDB_IMAGERY_TABLE'],
                              'DDB_DATASET_TABLE': config['DDB_DATASET_TABLE'],
                              'DDB_JOB_TABLE': config['DDB_JOB_TABLE']}}
    
    lambda_defs = {config['LAMBDA_LIFECYCLE']: {'image_name': config['IMAGE_NAME_LIFECYCLE'],
                                               'path': "lambdas/lifecycle_lambda",
                                               'role': role_arns['LIFECYCLE_ROLE_ARN'],
                                               'queue_arn': queue_arns['SQS_QUEUE_LIFECYCLE_ARN'],
                                               'memory_mb': 512,
                                               'timeout_sec': 300,
                                               'queue_visibility': 360,
                                               'batch_size': 10},
                   config['LAMBDA_IMAGE_OPS']: {'image_name': config['IMAGE_NAME_IMAGE_OPS'],
                                                'path': "lambdas/image_ops_lambda",
                                                'role': role_arns['IMAGE_OPS_ROLE_ARN'],
                                                'queue_arn': queue_arns['SQS_QUEUE_IMAGE_OPS_ARN'],
                                                'memory_mb': 1024,
                                                'timeout_sec': 600,
                                                'queue_visibility': 660,
                                                'batch_size': 10},
                   config['LAMBDA_SYNC']:      {'image_name': config['IMAGE_NAME_SYNC'],
                                                'path': "lambdas/sync_lambda",
                                                'role': role_arns['SYNC_ROLE_ARN'],
                                                'queue_arn': queue_arns['SQS_QUEUE_SYNC_ARN'],
                                                'memory_mb': 3072,
                                                'timeout_sec': 900,
                                                'queue_visibility': 960,
                                                'batch_size': 1},
                   config['LAMBDA_DLQ']:      {'image_name': config['IMAGE_NAME_DLQ'],
                                                'path': "lambdas/dlq_lambda",
                                                'role': role_arns['DLQ_ROLE_ARN'],
                                                'queue_arn': queue_arns['SQS_QUEUE_DLQ_ARN'],
                                                'memory_mb': 512,
                                                'timeout_sec': 60,
                                                'queue_visibility': 90,
                                                'batch_size': 10}}
    
    for function_name, info in lambda_defs.items():
                    
        image_name = info['image_name']
        path = info['path']
        role = info['role']
        queue_arn = info['queue_arn']
        memory_mb = info['memory_mb']
        timeout_sec = info['timeout_sec']
        batch_size = info['batch_size']
        
        ecr_image_uri = build_and_push_docker_image_to_ecr(ecr_client,
                                                           region, 
                                                           account_id,
                                                           image_name,
                                                           path,
                                                           local_tag,
                                                           ecr_tag,
                                                           for_lambda_fn)
        
            
        response = create_lambda_function(lambda_client,
                                        from_docker,
                                        ecr_image_uri,
                                        function_name,
                                        role,
                                        handler=None,
                                        env_vars=env_vars,
                                        runtime='python3.12',
                                        timeout=timeout_sec,
                                        memory_size=memory_mb,
                                        description=f"Single Label project lambda for {function_name}.")
        
        logger.info(f"Created Lambda {function_name} with image {ecr_image_uri}")
        
        function_arn = response['FunctionArn']
        
        lambda_client.create_event_source_mapping(
            EventSourceArn=queue_arn,
            FunctionName=function_arn,
            BatchSize=batch_size,
            Enabled=True
        )
        
        logger.info(f"Mapped {queue_arn} -> {function_arn}")
        
        if function_name in [config['LAMBDA_DLQ'], config['LAMBDA_SYNC']]:
            lambda_client.put_function_concurrency(
                                FunctionName=function_name,
                                ReservedConcurrentExecutions=1
                            )
    
        # update queue visibility
        queue_name = queue_arn.split(":")[-1]
        queue_url = clients['sqs'].get_queue_url(QueueName=queue_name)["QueueUrl"] # Convert ARN -> URL
    
        clients['sqs'].set_queue_attributes(
            QueueUrl=queue_url,
            Attributes={
                "VisibilityTimeout": str(info["queue_visibility"])
            }
        )
        logger.info(f"Set VisibilityTimeout={info['queue_visibility']}s for {queue_name}")        