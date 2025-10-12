# -*- coding: utf-8 -*-
import os
import logging
import boto3
from helpers.setup_helpers import load_config, validate_config, create_roles, make_bucket, \
                                  make_queues, make_tables
           
from helpers.ecr_docker_helpers import build_and_push_docker_image_to_ecr
from helpers.lambda_helpers import create_lambda_function

if __name__ == "__main__":
    # -------------------------------
    # Configure logging settings
    # -------------------------------
    main_dir = os.path.dirname(__file__)
    logging_save_to = os.path.join(main_dir, 'logs.txt')
    
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
    
    logger.info('Loading config.')
    config = load_config(os.path.join(main_dir, 'config.env'))
        
    # -------------------------------
    # Get clients and validate config.
    # -------------------------------
    clients = {'s3': boto3.client("s3", region_name=config['AWS_REGION']),
               'sqs': boto3.client("sqs", region_name=config['AWS_REGION']),
               'ddb':boto3.client("dynamodb", region_name=config['AWS_REGION']),
               'lambda':boto3.client("lambda", region_name=config['AWS_REGION']),
               'iam':boto3.client("iam", region_name=config['AWS_REGION']),
               'sts':boto3.client("sts", region_name=config['AWS_REGION']),
               'ecr':boto3.client('ecr', region_name=config['AWS_REGION'])}
    
    logger.info('Validating config.')
    validate_config(config, clients)

    # -------------------------------
    # Make the roles for each Lambda.
    # -------------------------------
    logger.info('Creating lambda roles.')
    create_roles(config, clients)
    
    # -------------------------------
    # Make the S3 bucket if it doesn't exist.
    # -------------------------------
    logger.info('Making S3 bucket.')
    make_bucket(config, clients)
    
    # -------------------------------
    # Make the SQS Queues.
    # -------------------------------
    logger.info('Making SQS queues.')
    make_queues(config, clients)
    
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
                                               'role': config['LIFECYCLE_ROLE_ARN'],
                                               'queue_arn': config['SQS_QUEUE_LIFECYCLE_ARN'],
                                               'memory_mb': 512,
                                               'timeout_sec': 300,
                                               'queue_visibility': 360,
                                               'batch_size': 10},
                   config['LAMBDA_IMAGE_OPS']: {'image_name': config['IMAGE_NAME_IMAGE_OPS'],
                                                'path': "lambdas/image_ops_lambda",
                                                'role': config['IMAGE_OPS_ROLE_ARN'],
                                                'queue_arn': config['SQS_QUEUE_IMAGE_OPS_ARN'],
                                                'memory_mb': 1024,
                                                'timeout_sec': 600,
                                                'queue_visibility': 660,
                                                'batch_size': 10},
                   config['LAMBDA_SYNC']:      {'image_name': config['IMAGE_NAME_SYNC'],
                                                'path': "lambdas/sync_lambda",
                                                'role': config['SYNC_ROLE_ARN'],
                                                'queue_arn': config['SQS_QUEUE_SYNC_ARN'],
                                                'memory_mb': 3072,
                                                'timeout_sec': 900,
                                                'queue_visibility': 960,
                                                'batch_size': 1},
                   config['LAMBDA_DLQ']:      {'image_name': config['IMAGE_NAME_DLQ'],
                                                'path': "lambdas/dlq_lambda",
                                                'role': config['DLQ_ROLE_ARN'],
                                                'queue_arn': config['SQS_QUEUE_DLQ_ARN'],
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
