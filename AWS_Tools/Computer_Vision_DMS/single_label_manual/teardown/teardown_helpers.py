# -*- coding: utf-8 -*-
"""
Teardown helpers.
"""

import logging
from botocore.exceptions import ClientError

# -------------------------------
# S3 helpers
# -------------------------------
def delete_bucket_policy(s3, bucket_name):
    try:
        s3.delete_bucket_policy(Bucket=bucket_name)
        logging.info(f"Deleted bucket policy on {bucket_name}.")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchBucketPolicy", "NoSuchBucket"):  # policy absent or bucket missing
            logging.info(f"No bucket policy to delete for {bucket_name} (code={code}).")
        elif code == "AccessDenied":
            logging.error(f"Access denied deleting bucket policy on {bucket_name}. Ensure teardown role has s3:DeleteBucketPolicy.")
            raise
        else:
            logging.error(f"Failed to delete bucket policy on {bucket_name}: {e}")
            raise

def delete_all_objects_under_prefix(s3, bucket_name, prefix):
    logging.info(f"Deleting all S3 objects under s3://{bucket_name}/{prefix}...")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    total_deleted = 0
    for page in pages:
        objects = page.get("Contents", [])
        if not objects:
            continue
        # Batch delete up to 1000 at a time
        to_delete = [{"Key": obj["Key"]} for obj in objects]
        resp = s3.delete_objects(
            Bucket=bucket_name,
            Delete={"Objects": to_delete, "Quiet": True}
        )
        deleted = resp.get("Deleted", [])
        total_deleted += len(deleted)

    logging.info(f"Deleted {total_deleted} objects under s3://{bucket_name}/{prefix}.")

# -------------------------------
# Lambda helpers
# -------------------------------
def delete_event_source_mappings(lambda_client, function_name):
    try:
        paginator = lambda_client.get_paginator("list_event_source_mappings")
        pages = paginator.paginate(FunctionName=function_name)
        count = 0
        for page in pages:
            for mapping in page.get("EventSourceMappings", []):
                uuid = mapping["UUID"]
                lambda_client.delete_event_source_mapping(UUID=uuid)
                logging.info(f"Deleted event source mapping {uuid} for {function_name}.")
                count += 1
        if count == 0:
            logging.info(f"No event source mappings found for {function_name}.")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code == "ResourceNotFoundException":
            logging.info(f"Function {function_name} not found when listing mappings.")
        else:
            logging.error(f"Failed listing/deleting mappings for {function_name}: {e}")
            raise


def delete_lambda(lambda_client, function_name):
    try:
        lambda_client.delete_function(FunctionName=function_name)
        logging.info(f"Deleted Lambda function {function_name}.")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code == "ResourceNotFoundException":
            logging.info(f"Lambda function {function_name} already deleted.")
        else:
            logging.error(f"Failed to delete Lambda {function_name}: {e}")
            raise
 
# -------------------------------
# ECR helper
# -------------------------------
def delete_ecr_repo(ecr_client, repo_name):
    try:
        ecr_client.delete_repository(repositoryName=repo_name, force=True)
        logging.info(f"Deleted ECR repository {repo_name} (force=True).")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code == "RepositoryNotFoundException":
            logging.info(f"ECR repository {repo_name} not found (already deleted).")
        else:
            logging.error(f"Failed to delete ECR repository {repo_name}: {e}")
            raise

# -------------------------------
# DynamoDB helper
# -------------------------------
def delete_ddb_table(ddb_client, table_name):
    try:
        ddb_client.delete_table(TableName=table_name)
        logging.info(f"Initiated deletion of DynamoDB table {table_name}.")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code == "ResourceNotFoundException":
            logging.info(f"DynamoDB table {table_name} already deleted.")
        else:
            logging.error(f"Failed to delete DynamoDB table {table_name}: {e}")
            raise


# -------------------------------
# SQS helpers
# -------------------------------
def get_queue_url(sqs_client, queue_name):
    try:
        return sqs_client.get_queue_url(QueueName=queue_name)["QueueUrl"]
    except sqs_client.exceptions.QueueDoesNotExist:
        return None


def delete_queue(sqs_client, queue_name):
    url = get_queue_url(sqs_client, queue_name)
    if not url:
        logging.info(f"SQS queue {queue_name} already deleted.")
        return
    try:
        sqs_client.delete_queue(QueueUrl=url)
        logging.info(f"Deleted SQS queue {queue_name}.")
    except ClientError as e:
        logging.error(f"Failed to delete SQS queue {queue_name}: {e}")
        raise


# -------------------------------
# IAM helpers
# -------------------------------
def delete_role_and_inline_policies(iam_client, role_name):
    """
    Delete an IAM role and clean up all inline policies, managed policies,
    and instance profile attachments before deleting the role itself.
    """

    try:
        # 1. Delete all inline policies
        try:
            inline_policies = iam_client.list_role_policies(RoleName=role_name).get("PolicyNames", [])
            for pol_name in inline_policies:
                iam_client.delete_role_policy(RoleName=role_name, PolicyName=pol_name)
                logging.info(f"Deleted inline policy {pol_name} from role {role_name}.")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "NoSuchEntity":
                logging.info(f"No inline policies found on role {role_name}.")
            else:
                logging.error(f"Failed to list/delete inline policies on {role_name}: {e}")
                raise

        # 2. Detach any attached managed policies
        attached = iam_client.list_attached_role_policies(RoleName=role_name).get("AttachedPolicies", [])
        for pol in attached:
            iam_client.detach_role_policy(RoleName=role_name, PolicyArn=pol["PolicyArn"])
            logging.info(f"Detached managed policy {pol['PolicyArn']} from role {role_name}.")

        # 3. Remove from any instance profiles
        profiles = iam_client.list_instance_profiles_for_role(RoleName=role_name).get("InstanceProfiles", [])
        for prof in profiles:
            iam_client.remove_role_from_instance_profile(
                InstanceProfileName=prof["InstanceProfileName"],
                RoleName=role_name
            )
            logging.info(f"Removed role {role_name} from instance profile {prof['InstanceProfileName']}.")

        # 4. Finally delete the role
        iam_client.delete_role(RoleName=role_name)
        logging.info(f"Deleted IAM role {role_name}.")

    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code == "NoSuchEntity":
            logging.info(f"IAM role {role_name} already deleted.")
        else:
            logging.error(f"Failed to delete IAM role {role_name}: {e}")
            raise

            raise