# -*- coding: utf-8 -*-
"""
Helpers for AWS Lambda.
"""

import logging
import subprocess
import re
import os

def make_new_repo(ecr_client,
                  region, 
                  account_id,
                  ecr_repo_name):
    """
    Creates a new Amazon ECR repository with security best practices enabled.
    
    If the repository already exists, retrieves its metadata instead of failing.
    Automatically enables image scanning on push and AES256 encryption at rest.
    
    Parameters:
        ecr_client (boto3.client): A boto3 ECR client instance.
        region (str): AWS region where the ECR repository should be created (e.g., 'us-west-2').
        account_id (str): AWS account ID used to construct the repository URI.
        ecr_repo_name (str): Name of the ECR repository. Must match ECR naming rules (lowercase letters, numbers, hyphens, underscores).
    
    Raises:
        ValueError: If the repository name is invalid.
        Exception: If repository creation fails for reasons other than it already existing.
    
    Returns:
        dict: Metadata for the ECR repository, including keys like 'repositoryUri', 'repositoryArn', and 'registryId'.
    """
    
    if not re.match(r'^[a-z0-9-_]+$', ecr_repo_name):
        raise ValueError(f"Invalid ECR repo name: {ecr_repo_name}")

    try:
        response = ecr_client.create_repository(repositoryName=ecr_repo_name,
                                          imageScanningConfiguration={'scanOnPush': True},
                                          encryptionConfiguration={'encryptionType': 'AES256'})
        repo_info = response['repository']
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        logging.info(f'ECR Repo {ecr_repo_name} already exists.')
        repo_info = ecr_client.describe_repositories(repositoryNames=[ecr_repo_name])['repositories'][0]
    except Exception as e:
        raise Exception(f'Error making ECR repo {ecr_repo_name}: {e}')
        
    return repo_info

def login_ecr(region, account_id):
    """
    Authenticates Docker with Amazon ECR using the AWS CLI.
    
    This function retrieves a temporary authentication token via `aws ecr get-login-password`
    and pipes it into `docker login` using `--password-stdin`. It enables subsequent Docker
    operations like tagging and pushing images to the specified ECR registry.
    
    Parameters:
        region (str): AWS region where the ECR registry is hosted (e.g., 'us-west-2').
        account_id (str): AWS account ID used to construct the ECR registry URI.
    
    Returns:
        bool: True if Docker login succeeds, False otherwise.
    
    Logs:
        - Info-level message on successful login.
        - Error-level message on failure, including stderr output.
    """
    try:
        # Get ECR login password
        pw_proc = subprocess.run(
            ["aws", "ecr", "get-login-password", "--region", region],
            capture_output=True,
            text=True,
            check=True
        )
        password = pw_proc.stdout.strip()

        # Docker login
        login_proc = subprocess.run(
            ["docker", "login", "--username", "AWS", "--password-stdin", f"{account_id}.dkr.ecr.{region}.amazonaws.com"],
            input=password,
            capture_output=True,
            text=True
        )

        if login_proc.returncode == 0 and "Login Succeeded" in login_proc.stdout:
            logging.info(f"Docker login success, stdout:\n{login_proc.stdout}")
            return True
        else:
            logging.error(f"Docker login failed, stderr:\n{login_proc.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get ECR login password: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during ECR login: {e}")
        return False


def build_image_locally(base_image_name, tag='latest', path='.', for_lambda_fn = True):
    """
    Builds a Docker image locally from a specified directory containing a Dockerfile.
    
    This function validates the image name and tag, then runs `docker build` to create
    a tagged image. It raises an exception if the build fails, and logs the outcome.
    
    Parameters:
        base_image_name (str): Base name for the Docker image (e.g., 'my-lambda-task').
        tag (str): Docker tag to apply (default: 'latest'). Must follow Docker tag naming rules.
        path (str): Path to the directory containing the Dockerfile (e.g., './my-docker-task').
        for_lambda_fn: set to True if trying to build a lambda function, False if not, e.g. for a Batch worker.
        for_lambda_fn (bool): If True, builds with Lambda-compatible settings (disables BuildKit, forces linux/amd64).
                                Set to False for workers for AWS Batch.
    Raises:
        ValueError: If the image name or tag is invalid.
        subprocess.CalledProcessError: If the Docker build command fails.
    
    Returns:
        str: Full image name including tag (e.g., 'my-lambda-task:latest').
    """
    
    image_name = f"{base_image_name}:{tag}"

    if not re.match(r'^[\w][\w.-]{0,127}$', tag):
        raise ValueError(f"Invalid Docker tag: {tag}")
        
    if not re.match(r'^[\w][\w.-]{0,127}$', base_image_name):
        raise ValueError(f"Invalid Docker image name: {base_image_name}")
    
    if for_lambda_fn:
        build_command = [
                        "docker", "build",
                        "--platform", "linux/amd64",
                        "-t", image_name, path]
                    
        env = {"DOCKER_BUILDKIT": "0", **dict(os.environ)}
        
        try:
            subprocess.run(build_command, env=env, check=True, text=True, capture_output=True)
            logging.info("Docker build completed successfully.")
            return image_name
        except subprocess.CalledProcessError as e:
            logging.error(f"Docker build failed for image {image_name}:\n{e.stderr}")
            raise
    else:
        # works for batch
        build_command = ["docker", "build", "-t", image_name, path]
    
        try:
            subprocess.run(build_command, capture_output=True, text=True, check=True)
            logging.info(f"Docker build succeeded for image: {image_name}")
            return image_name
        except subprocess.CalledProcessError as e:
            logging.error(f"Docker build failed for image {image_name}:\n{e.stderr}")
            raise 
            
def tag_image_to_ecr_uri(local_image_name, ecr_repo_uri, repo_tag='latest'):
    """
    Tags a local Docker image with the ECR URI for pushing.

    Parameters:
        local_image_name (str): Name of the tagged and locally built image (e.g., 'my-image:latest').
        ecr_repo_uri (str): ECR repository URI (e.g., '123456789012.dkr.ecr.us-west-2.amazonaws.com/my-repo').
                            Can be obtained from output from make_new_repo via repo_info['repositoryUri']
        repo_tag (str): Tag to apply on the ECR image (default: 'latest').

    Returns:
        str: Fully qualified ECR image URI (e.g., '.../my-repo:latest').
    """

    if not re.match(r'^[\w][\w.-]{0,127}$', repo_tag):
        raise ValueError(f"Invalid Docker tag: {repo_tag}")

    ecr_image_uri = f"{ecr_repo_uri}:{repo_tag}"

    try:
        subprocess.run(
            ["docker", "tag", local_image_name, ecr_image_uri],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Tagged image '{local_image_name}' as '{ecr_image_uri}'")
        return ecr_image_uri
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to tag image '{local_image_name}' as '{ecr_image_uri}':\n{e.stderr}")
        raise

def push_local_image_to_ecr(ecr_image_uri):
    """
    Pushes a locally tagged Docker image to the specified Amazon ECR repository.

    This function runs `docker push` on the provided image URI, which must include
    both the repository and tag (e.g., '123456789012.dkr.ecr.us-west-2.amazonaws.com/my-repo:latest').

    Parameters:
        ecr_image_uri (str): Fully qualified ECR image URI including tag.

    Raises:
        subprocess.CalledProcessError: If the Docker push command fails.

    Logs:
        - Info-level message on successful push.
        - Error-level message on failure, including stderr output.
    """
    try:
        subprocess.run(
            ["docker", "push", ecr_image_uri],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Docker push succeeded for image: {ecr_image_uri}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Docker push failed for image {ecr_image_uri}:\n{e.stderr}")
        raise
    
def build_and_push_docker_image_to_ecr(ecr_client,
                                   region, 
                                   account_id,
                                   ecr_repo_name,
                                   path_to_folder_containing_dockerfile,
                                   local_tag,
                                   ecr_tag,
                                   for_lambda_fn):
    """
    Builds a Docker image locally and pushes it to Amazon ECR.

    This function orchestrates the full lifecycle:
    - Creates the ECR repository if it doesn't exist
    - Authenticates Docker to ECR
    - Builds the image from a local Dockerfile
    - Tags the image with the full ECR URI
    - Pushes the image to ECR

    Parameters:
        ecr_client (boto3.client): A boto3 ECR client instance.
        region (str): AWS region where the ECR repository is hosted.
        account_id (str): AWS account ID used to construct the ECR URI.
        ecr_repo_name (str): Name of the ECR repository. Also used as the base image name.
        path_to_folder_containing_dockerfile (str): Path to the folder containing the Dockerfile.
        local_tag (str): Local tag to apply during build (e.g., 'latest').
        ecr_tag (str): Tag to apply when pushing to ECR (e.g., 'v1').

    Raises:
        Exception: If any step in the process fails.
    """
    try:
        # Create or retrieve ECR repository
        print('Making the ECR repo.')
        repo_info = make_new_repo(ecr_client, region, account_id, ecr_repo_name)
        ecr_repo_uri = repo_info.get('repositoryUri')
        if not ecr_repo_uri:
            raise ValueError("Missing 'repositoryUri' in ECR repo metadata.")

        # Authenticate Docker to ECR
        logging.info('Logging in to the ECR repo.')
        login_success = login_ecr(region, account_id)
        if not login_success:
            raise RuntimeError("Docker login to ECR failed.")

        # Build the Docker image locally
        print('Building the Docker image locally.')
        local_image_name_with_tag = build_image_locally(
            base_image_name=ecr_repo_name,
            tag=local_tag,
            path=path_to_folder_containing_dockerfile,
            for_lambda_fn=for_lambda_fn
        )

        # Tag the image with the full ECR URI
        print('Tagging the local image with the ECR URI.')
        ecr_image_uri = tag_image_to_ecr_uri(
            local_image_name=local_image_name_with_tag,
            ecr_repo_uri=ecr_repo_uri,
            repo_tag=ecr_tag
        )

        # Push the image to ECR
        print('Pushing the image to ECR.')
        push_local_image_to_ecr(ecr_image_uri)

        print(f"Successfully built and pushed image to ECR: {ecr_image_uri}")
        return ecr_image_uri

    except Exception as e:
        print(f"Failed to build and push Docker image to ECR: {e}")
        raise