# -*- coding: utf-8 -*-
"""
Lambda helpers.
"""

import logging
import time
            
def validate_layer_arns(lambda_client, layer_arns):
    """
    Validates that each Lambda layer ARN exists and is accessible.

    Args:
        lambda_client: Boto3 Lambda client.
        layer_arns (list): List of layer ARNs to validate.

    Returns:
        list: Validated layer ARNs.

    Raises:
        ValueError: If any layer ARN is invalid or inaccessible.
    """
    valid_layers = []
    for arn in layer_arns:
        try:
            # Extract layer name and version from ARN
            parts = arn.split(":")
            if len(parts) < 8 or not parts[-1].isdigit():
                raise ValueError(f"Invalid layer ARN format: {arn}")
            layer_name = ":".join(parts[:-1])
            version = int(parts[-1])

            # Validate existence
            lambda_client.get_layer_version(
                LayerName=layer_name,
                VersionNumber=version
            )
            valid_layers.append(arn)
        except Exception as e:
            raise ValueError(f"Layer ARN '{arn}' is invalid or inaccessible: {e}")
            
    return valid_layers

def wait_for_lambda_creation(lambda_client, function_name, max_wait=300, poll_interval=5):
    """
    Waits until the Lambda function is in 'Active' state.

    Args:
        lambda_client: Boto3 Lambda client.
        function_name (str): Name of the Lambda function.
        max_wait (int): Max time to wait in seconds.
        poll_interval (int): Time between polls in seconds.

    Raises:
        TimeoutError: If function doesn't become active within max_wait.
    """
    waited = 0
    while waited < max_wait:
        response = lambda_client.get_function_configuration(FunctionName=function_name)
        status = response.get("State")
        reason = response.get("StateReasonCode", "")
        if status == "Active":
            return
        logging.info(f"Waiting for Lambda '{function_name}' to become Active... (State: {status}, Reason: {reason})")
        time.sleep(poll_interval)
        waited += poll_interval
    raise TimeoutError(f"Lambda function '{function_name}' did not become Active within {max_wait} seconds.")

def create_lambda_function(lambda_client,
                         from_docker,
                         code_source,
                         function_name,
                         lambda_role_arn,
                         handler=None,
                         env_vars=None,
                         runtime='python3.12',
                         timeout=20,
                         memory_size=128,
                         layers_to_attach = [],
                         description="A Lambda function created from a ZIP or Docker image."):
    """
    Creates an AWS Lambda function using either a ZIP archive or a Docker image.

    For ZIP-based functions, this reads the ZIP file from disk and sets the runtime and handler.
    For Docker-based functions, this uses the ECR image URI and omits runtime and handler.

    Parameters:
        lambda_client (boto3.client): A boto3 Lambda client instance.
        from_docker (bool): If True, creates the function from a Docker image in ECR.
        code_source (str): Path to ZIP file (if from_docker=False) or an ECR image URI (if from_docker=True).
        function_name (str): Name of the Lambda function to create.
        lambda_role_arn (str): IAM role ARN that the Lambda function will assume.
        handler (str, optional): Entry point for ZIP-based Lambda (e.g., 'my_worker.do_stuff' or 'app.lambda_handler').
        env_vars (dict, optional): Environment variables structured as {'Variables': {...}}.
        runtime (str, optional): Runtime identifier for ZIP-based Lambda (default: 'python3.12').
        timeout (int): Timeout in seconds for function execution (default: 20).
        memory_size (int): Memory size for the task in units of MB. 128 is default, more might be needed for larger images or more dependencies.
        layers_to_attach (list): A list of lambda layer ARNs to attach to this function.
        description (str): Description of the Lambda function.

    Returns:
        dict: Response from `create_function`, including metadata like function ARN.

    Raises:
        FileNotFoundError: If the ZIP file cannot be read.
        botocore.exceptions.ClientError: If Lambda creation fails.
    """
    if from_docker:
        code = {'ImageUri': code_source}
        package_type = 'Image'
    else:
        try:
            with open(code_source, 'rb') as f:
                zipped_code = f.read()
            code = {'ZipFile': zipped_code}
            package_type = 'Zip'
        except Exception as e:
            logging.error(f"Failed to read ZIP file at '{code_source}': {e}")
            raise

    # Build request payload dynamically
    kwargs = {
        'FunctionName': function_name,
        'Role': lambda_role_arn,
        'Code': code,
        'Description': description,
        'PackageType': package_type,
        'Timeout': timeout,
        'MemorySize': memory_size,
        'Architectures': ['x86_64'],
        'LoggingConfig': {'LogFormat': 'JSON'}
    }

    if env_vars:
        kwargs['Environment'] = env_vars
        
    if not from_docker:
        kwargs['Runtime'] = runtime
        kwargs['Handler'] = handler

    try:
        response = lambda_client.create_function(**kwargs)
        logging.info(f"Lambda function '{function_name}' created successfully.")
    except Exception as e:
        logging.error(f"Failed to create Lambda function '{function_name}': {e}")
        raise
        
    if layers_to_attach:
        layers_to_attach = validate_layer_arns(lambda_client, layers_to_attach)
        
        wait_for_lambda_creation(lambda_client, function_name, max_wait=300, poll_interval=5)
        
        try:
            lambda_client.update_function_configuration(
                FunctionName=function_name,
                Layers=layers_to_attach
            )
            logging.info(f"Layer(s) {layers_to_attach} attached to function: {function_name}")
        except Exception as e:
            logging.error(f"Failed to attach layers to function '{function_name}': {e}")
            raise
            
    return response