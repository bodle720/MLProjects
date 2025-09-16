# -*- coding: utf-8 -*-
"""
Lambda helpers.
"""

import os
import logging
from pprint import pprint
import zipfile
import subprocess
import shutil
import uuid
import time

def describe_a_lambda(lambda_client,
                      lambda_function_name):
    """
    Retrieves and displays metadata about an AWS Lambda function.

    Supports both ZIP-based and container image-based Lambda functions.
    Uses safe access patterns and pretty printing for readability.

    Parameters:
        lambda_client (boto3.client): A boto3 Lambda client instance.
        lambda_function_name (str): Name of the Lambda function to describe.

    Raises:
        botocore.exceptions.ClientError: If the function does not exist or access is denied.
    """

    try:
        response = lambda_client.get_function(FunctionName=lambda_function_name)
        config = response.get('Configuration', {})
        code = response.get('Code', {})

        print(f"\nLambda Function: {lambda_function_name}")
        print(f"Role: {config.get('Role')}")
        print(f"Handler: {config.get('Handler')}")
        print(f"Code Size: {config.get('CodeSize')} bytes")
        print(f"Description: {config.get('Description')}")
        print(f"Last Modified: {config.get('LastModified')}")
        print(f"Package Type: {config.get('PackageType')}")  # 'Zip' or 'Image'
        print(f"State: {config.get('State')}")

        env_vars = config.get('Environment', {}).get('Variables')
        if env_vars:
            print("Environment Variables:")
            pprint(env_vars)

        layers = config.get('Layers')
        if layers:
            print("Layers:")
            pprint(layers)

        print("Code Location:")
        pprint(code)

    except Exception as e:
        logging.error(f"Failed to describe Lambda function '{lambda_function_name}': {e}")
        raise

def publish_new_lambda_layer(lambda_client,
                             layer_name,
                             requirements,
                             description,
                             runtime = "python3.12"):
    """
    Builds and publishes an AWS Lambda layer using a Linux-compatible Docker container.
    
    Args:
        lambda_client: Boto3 Lambda client.
        layer_name (str): Name of the Lambda layer to publish.
        requirements (list): List of Python packages to include in the layer.
        description (str): Description of the layer.
        runtime (str): Compatible runtime (default: "python3.12").

    Returns:
        str: The ARN of the published Lambda layer version.

    Raises:
        ValueError: If a layer with the same name already exists.
        RuntimeError: If Docker build fails.
        
    Example use:
        lambda_client = boto3.client("lambda", region_name="us-east-1")

        layer_arn = publish_lambda_layer(
            lambda_client=lambda_client,
            layer_name="numpy-pillow-layer",
            requirements=["numpy", "pillow"],
            description="Numpy and Pillow packages for image processing workflows."
        )
        
        print(f"Published layer ARN: {layer_arn}")

    """

    # Check if layer name already exists
    existing_layers = lambda_client.list_layers(MaxItems=50).get("Layers", [])
    for layer in existing_layers:
        if layer["LayerName"] == layer_name:
            raise ValueError(f"Layer '{layer_name}' already exists.")

    # Create temporary directory
    temp_id = str(uuid.uuid4())[:8]
    build_dir = f"layer_build_{temp_id}"
    python_dir = os.path.join(build_dir, "python")
    os.makedirs(python_dir, exist_ok=True)

    zip_path = f"{build_dir}.zip"

    try:

        # Build layer by installing dependencies inside Docker container environment (Windows is not compatible for linux builds)
        # - run pip install in AWS SAM build image
        # - mounts build_dir at /var/task
        # - installs into /var/task/python
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{os.path.abspath(build_dir)}:/var/task",
            "public.ecr.aws/sam/build-python3.12",
            "bash", "-c",
            # upgrade pip, then install your requirements
            f"pip install --upgrade pip && "
            f"pip install {' '.join(requirements)} -t /var/task/python"
        ]

        result = subprocess.run(docker_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed:\n{result.stderr}")
            
        # Zip contents
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(build_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, build_dir)
                    zf.write(full_path, arcname)

        # Publish layer
        with open(zip_path, "rb") as f:
            response = lambda_client.publish_layer_version(
                LayerName=layer_name,
                Description=description,
                Content={"ZipFile": f.read()},
                CompatibleRuntimes=[runtime]
            )

        layer_arn = response["LayerVersionArn"]
        logging.info(f"Layer published: {layer_arn}")
        return layer_arn
    
    except Exception as e:
        logging.error(f'An exception happened: {e}')
        raise
    finally:
        # Cleanup temp files
        shutil.rmtree(build_dir, ignore_errors=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
def zip_folder_of_lambda_function_contents(source_dir,
                                           save_to):
    """
    Zips the contents of a Lambda function source directory into a deployment-ready archive.

    This is used to prepare the `code_source` argument for the Lambda creation function create_lambda_function below (non-Docker style).
    Only the contents of the folder are zipped—**not** the folder itself, as required by the AWS Lambda interpreter.

    Args:
        source_dir (str): Path to the folder containing Lambda source files.
        save_to (str): Path to save the resulting ZIP file.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        ValueError: If the source directory is empty.
    """

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files_to_zip = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, start=source_dir)
            files_to_zip.append((full_path, arcname))

    if not files_to_zip:
        raise ValueError(f"No files found in source directory: {source_dir}")

    with zipfile.ZipFile(save_to, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for full_path, arcname in files_to_zip:
            zipf.write(full_path, arcname)

    logging.info(f"Lambda deployment package created: {save_to}")

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