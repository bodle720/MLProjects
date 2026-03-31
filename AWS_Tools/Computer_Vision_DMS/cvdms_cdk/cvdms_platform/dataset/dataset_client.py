import logging
from typing import Literal, Any
from mypy_boto3_s3.client import S3Client
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource

# Helpers for validating user inputs
from cvdms_platform.dataset.utils.validator import (validate_create_dataset_inputs, validate_update_dataset_inputs,
                                                    validate_delete_dataset_inputs, validate_get_dataset_inputs)

# Helper to retrieve information about a dataset
from cvdms_platform.dataset.utils.get_dataset_info import get_dataset_info

# Helper to upload JSON request to S3
from cvdms_platform.dataset.utils.upload_submission import upload_submission

LABEL_TYPE_TO_MEMBERSHIP_TABLE = {
    "single-label": "single_label",
    "multi-label": "multi_label",
    "object-detection": "object_detection",
    "semantic-segmentation": "semantic_segmentation",
    "instance-segmentation": "instance_segmentation"
}

class DatasetClient:
    """
    High-level client to perform dataset operations.
    """
    def __init__(self,
                 *,
                 user: str,
                 datasets_table_name: str,
                 dataset_versions_table_name: str,
                 s3_client: S3Client,
                 dynamodb_resource: DynamoDBServiceResource):

        self.user = user
        self.datasets_table_name = datasets_table_name
        self.dataset_versions_table_name = dataset_versions_table_name
        self.s3_client = s3_client
        self.dynamodb_resource = dynamodb_resource

    def get_dataset(self, *, dataset_id: str) -> dict[str, Any]:
        """
        Return dataset information for the latest version.

        Returns:
            {"exists": False}
        if the dataset does not exist.

        Otherwise, returns a dict with dataset-level metadata, latest version
        metadata, split counts, and artifact pointers.
        """

        validated = validate_get_dataset_inputs(dataset_id=dataset_id)
        dataset_id = validated["dataset_id"]
        logging.info(f"Validated dataset_id successfully: {dataset_id}")

        return get_dataset_info(
            dynamodb_resource=self.dynamodb_resource,
            datasets_table_name=self.datasets_table_name,
            dataset_versions_table_name=self.dataset_versions_table_name,
            dataset_id=dataset_id
        )

    def submit_create_dataset(self,
                                *,
                                dataset_id: str,
                                label_type: str,
                                description: str,
                                selection_config: dict,
                                split_strategy_name: str) -> dict:
        """
        Submits request to create a new dataset at version 1.

        High-level flow:
        1. validate inputs
        2. verify the dataset does not exist
        3. submit the request to S3
        """

        # 1. Validate inputs
        logging.info("Validating inputs...")
        validated = validate_create_dataset_inputs(
            dataset_id=dataset_id,
            label_type=label_type,
            description=description,
            selection_config=selection_config,
            split_strategy_name=split_strategy_name
        )

        dataset_id = validated["dataset_id"]
        label_type = validated["label_type"]
        description = validated["description"]
        selection_config = validated["selection_config"]
        split_strategy_name = validated["split_strategy_name"]

        logging.info("Inputs validated.")

        # 2. Ensure dataset_id is not previously used
        logging.info("Checking if dataset already exists...")
        dataset_info = self.get_dataset(dataset_id=dataset_id)
        if dataset_info["exists"]:
            logging.error(f"Dataset '{dataset_id}' already exists, choose a different name.")
            raise ValueError(f"Dataset '{dataset_id}' already exists, choose a different name.")

        # 3. Submit task to S3
        payload = {}
        submission = upload_submission(payload=payload)

        return submission

    def submit_update_dataset(self,
                               *,
                               dataset_id: str,
                               operation: Literal["add", "remove"],
                               selection_config: dict[str, Any],
                               split_approach: Literal["maintain", "rebalance"] = "maintain",
                               split_strategy_name: str | None = None,
                               description: str | None = None) -> dict:
        """
        Submits request to create a new dataset version by adding or removing imagery to/from an existing dataset.

        High-level flow:
        1. validate inputs
        2. verify the dataset exists
        3. submit the request to S3
        """

        # 1. Validate inputs
        logging.info("Validating inputs...")

        validated = validate_update_dataset_inputs(
            dataset_id=dataset_id,
            operation=operation,
            selection_config=selection_config,
            split_approach=split_approach,
            split_strategy_name=split_strategy_name,
            description=description,
        )

        dataset_id = validated["dataset_id"]
        operation = validated["operation"]
        selection_config = validated["selection_config"]
        split_approach = validated["split_approach"]
        split_strategy_name = validated["split_strategy_name"]
        description = validated["description"]

        logging.info("Inputs validated.")

        # 2. Load current dataset state
        logging.info("Checking if dataset already exists...")
        dataset_info = self.get_dataset(dataset_id=dataset_id)

        if not dataset_info["exists"]:
            logging.error(f"Dataset '{dataset_id}' does not exist.")
            raise ValueError(f"Dataset '{dataset_id}' does not exist.")

        # 3. Submit task to S3
        payload = {}
        submission = upload_submission(payload=payload)

        return submission

    def submit_delete_dataset_all_versions(self, *, dataset_id: str) -> dict:
        """
        Submits a request to delete a dataset and all its versions.

        1. validate inputs
        2. verify the dataset exists
        3. submit the request to S3
        """

        # 1. Validate inputs
        try:
            logging.info("Validating inputs")
            validated = validate_delete_dataset_inputs(dataset_id=dataset_id)
            dataset_id = validated["dataset_id"]
        except Exception as e:
            logging.error(f"Failed validating inputs: {str(e)}")
            raise

        # 2. Load existing dataset metadata and confirm dataset exists
        logging.info("Validation done. Now retrieving the dataset metadata...")
        try:
            dataset_record = self.get_dataset(dataset_id=dataset_id)
        except Exception as e:
            logging.error(f"Failed loading dataset metadata for '{dataset_id}': {e}")
            raise

        if not dataset_record["exists"]:
            logging.error(f"Dataset '{dataset_id}' does not exist.")
            raise

        # 3. Submit task to S3
        payload = {}
        submission = upload_submission(payload=payload)

        return submission