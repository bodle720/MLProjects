# -*- coding: utf-8 -*-
"""
Teardown the infrastructure described in the config.
"""

import os
import sys
import argparse
import logging
import boto3

import helpers.teardown_helpers as td
from helpers.setup_helpers import load_config_from_ssm


def delete_parameters_for_namespace(ssm_client, infrastructure_name: str):
    """Delete all parameters under the infra namespace in SSM."""
    prefix = f"/cv-datasets/single-label/{infrastructure_name}/infrastructure/"
    paginator = ssm_client.get_paginator("get_parameters_by_path")
    for page in paginator.paginate(Path=prefix, Recursive=True):
        names = [p["Name"] for p in page.get("Parameters", [])]
        if names:
            ssm_client.delete_parameters(Names=names)
            logging.info(f"Deleted {len(names)} parameters under {prefix}.")
    logging.info(f"Finished deleting all parameters under {prefix}.")


if __name__ == "__main__":
    # -------------------------------
    # Parse CLI args
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--infrastructure_name", required=True,
                        help="The infrastructure namespace to teardown (must exist in SSM).")
    parser.add_argument("--region", required=True,
                        help="The AWS region where the infrastructure was deployed.")
    args = parser.parse_args()

    # -------------------------------
    # Configure logging
    # -------------------------------
    main_dir = os.path.dirname(__file__)

    logging_save_to = os.path.join(main_dir, "logs.txt")
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

    # -------------------------------
    # Load config from SSM
    # -------------------------------
    ssm = boto3.client("ssm", region_name=args.region)
    try:
        config = load_config_from_ssm(ssm, args.infrastructure_name)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Validate region matches infra config
    infra_region = config["AWS_REGION"]
    if infra_region != args.region:
        logger.error(
            f"Region mismatch: user specified {args.region}, "
            f"but infrastructure {args.infrastructure_name} is in {infra_region}."
        )
        sys.exit(1)

    logger.info(f"Loaded config for {args.infrastructure_name} in region {infra_region}.")

    # -------------------------------
    # Build clients
    # -------------------------------
    clients = {
        "s3": boto3.client("s3", region_name=infra_region),
        "sqs": boto3.client("sqs", region_name=infra_region),
        "ddb": boto3.client("dynamodb", region_name=infra_region),
        "lambda": boto3.client("lambda", region_name=infra_region),
        "iam": boto3.client("iam", region_name=infra_region),
        "sts": boto3.client("sts", region_name=infra_region),
        "ecr": boto3.client("ecr", region_name=infra_region),
        "ssm": ssm,
    }
    account_id = clients["sts"].get_caller_identity()["Account"]
    logger.info(f"Caller account: {account_id}")

    # -------------------------------
    # Teardown sequence (same as before)
    # -------------------------------
    bucket_name = config["S3_BUCKET_NAME"]
    td.delete_bucket_policy(clients["s3"], bucket_name)
    td.delete_all_objects_under_prefix(clients["s3"], bucket_name, config["S3_DATASETS_ROOT"] + "/")

    for fn in [config["LAMBDA_LIFECYCLE"], config["LAMBDA_IMAGE_OPS"],
               config["LAMBDA_SYNC"], config["LAMBDA_DLQ"]]:
        td.delete_event_source_mappings(clients["lambda"], fn)
        td.delete_lambda(clients["lambda"], fn)

    for repo in [config["IMAGE_NAME_LIFECYCLE"], config["IMAGE_NAME_IMAGE_OPS"],
                 config["IMAGE_NAME_SYNC"], config["IMAGE_NAME_DLQ"]]:
        td.delete_ecr_repo(clients["ecr"], repo)

    for table in [config["DDB_IMAGERY_TABLE"], config["DDB_DATASET_TABLE"], config["DDB_JOB_TABLE"]]:
        td.delete_ddb_table(clients["ddb"], table)

    for qn in [config["SQS_QUEUE_LIFECYCLE"], config["SQS_QUEUE_IMAGE_OPS"],
               config["SQS_QUEUE_SYNC"], config["SQS_QUEUE_DLQ"]]:
        td.delete_queue(clients["sqs"], qn)

    for role_name_key in ["LIFECYCLE_ROLE_NAME", "IMAGE_OPS_ROLE_NAME",
                          "SYNC_ROLE_NAME", "DLQ_ROLE_NAME"]:
        td.delete_role_and_inline_policies(clients["iam"], config[role_name_key])

    delete_parameters_for_namespace(clients["ssm"], args.infrastructure_name)

    logger.info("Teardown complete. All resources and parameters removed.")

    