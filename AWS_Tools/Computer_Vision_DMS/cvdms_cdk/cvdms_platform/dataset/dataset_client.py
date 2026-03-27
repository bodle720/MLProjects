from typing import Literal, Any
from mypy_boto3_s3.client import S3Client
from mypy_boto3_athena.client import AthenaClient
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource

from cvdms_platform.dataset.split_strategies.stratified_v1 import assign_splits_stratified_v1
from cvdms_platform.dataset.utils.validator import validate_create_dataset_inputs, validate_update_dataset_inputs
from cvdms_platform.dataset.utils.resolve_imagery_db import resolve_candidate_imagery
from cvdms_platform.dataset.utils.resolve_membership_db import resolve_dataset_membership
from cvdms_platform.dataset.utils.add_membership import write_membership_rows
from cvdms_platform.dataset.utils.write_artifacts import write_dataset_artifacts
from cvdms_platform.dataset.utils.ddb_update import write_dataset_ddb_records

from cvdms_platform.dataset.utils.get_dataset import get_dataset_info
from cvdms_platform.dataset.utils.update_dataset_membership import get_updated_split_rows

LABEL_TYPE_TO_MEMBERSHIP_TABLE = {
    "single-label": "single_label",
    "multi-label": "multi_label",
    "object-detection": "object_detection",
    "semantic-segmentation": "semantic_segmentation",
    "instance-segmentation": "instance_segmentation",
}

class DatasetClient:
    """
    High-level client to perform dataset operations.
    """
    def __init__(self,
                 *,
                 user: str,
                 datasets_bucket_name: str,
                 datasets_table_name: str,
                 dataset_versions_table_name: str,
                 s3_client: S3Client,
                 dynamodb_resource: DynamoDBServiceResource,
                 iceberg_database_name: str,
                 athena_client: AthenaClient,
                 file_bucket_name: str):

        self.user = user
        self.datasets_bucket_name = datasets_bucket_name
        self.datasets_table_name = datasets_table_name
        self.dataset_versions_table_name = dataset_versions_table_name

        self.s3 = s3_client
        self.dynamodb = dynamodb_resource
        self.athena = athena_client

        self.datasets_table = self.dynamodb.Table(self.datasets_table_name)
        self.iceberg_database_name = iceberg_database_name
        self.athena_output_s3_uri = f"s3://{file_bucket_name}/athena-results/"
        self.canonical_imagery_table_name = "canonical_imagery"

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        """
        Return dataset information for the latest version.

        Returns:
            {"exists": False}
        if the dataset does not exist.

        Otherwise returns a dict with dataset-level metadata, latest version
        metadata, split counts, and artifact pointers.
        """
        return get_dataset_info(
            dynamodb_resource=self.dynamodb,
            datasets_table_name=self.datasets_table_name,
            dataset_versions_table_name=self.dataset_versions_table_name,
            dataset_id=dataset_id
        )

    def create_dataset(self,
                        *,
                        dataset_id: str,
                        label_type: str,
                        description: str,
                        selection_config: dict,
                        split_strategy_name: str) -> dict:
        """
        Create a new dataset at version 1.

        High-level flow:
        1. validate inputs
        2. build selection SQL
        3. resolve and normalize candidate rows from Athena
        4. assign train/val/test splits
        5. write Iceberg membership rows
        6. write S3 dataset artifacts
        7. write DynamoDB dataset metadata
        """
        # Validate inputs
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

        # Ensure globally unique dataset_id
        existing = self.datasets_table.get_item(Key={"dataset_id": dataset_id}).get("Item")
        if existing:
            raise ValueError(f"Dataset '{dataset_id}' already exists.")

        # Build the SQL for obtaining canonical imagery selection and retrieve the normalized candidates from Iceberg
        selection_sql, candidates = resolve_candidate_imagery(self.iceberg_database_name,
                                                               label_type,
                                                               selection_config,
                                                               self.athena,
                                                               self.athena_output_s3_uri)

        if not candidates:
            raise ValueError(
                f"Dataset '{dataset_id}' selection returned zero candidate rows."
            )

        if split_strategy_name == "stratified_v1":
            split_rows = assign_splits_stratified_v1(candidates=candidates)
        else:
            raise ValueError(f"Split strategy '{split_strategy_name}' not supported.")

        membership_result = write_membership_rows(
            athena_client=self.athena,
            iceberg_database_name=self.iceberg_database_name,
            athena_output_s3_uri=self.athena_output_s3_uri,
            dataset_id=dataset_id,
            version=1,
            dataset_label_type=label_type,
            split_rows=split_rows
        )

        artifact_result = write_dataset_artifacts(
            s3_client=self.s3,
            dataset_bucket_name=self.datasets_bucket_name,
            dataset_id=dataset_id,
            version=1,
            label_type=label_type,
            split_strategy_name=split_strategy_name,
            selection_sql=selection_sql,
            selection_config=selection_config,
            split_rows=split_rows
        )

        ddb_result = write_dataset_ddb_records(
            new_dataset=True,
            dynamodb_resource=self.dynamodb,
            datasets_table_name=self.datasets_table_name,
            dataset_versions_table_name=self.dataset_versions_table_name,
            dataset_id=dataset_id,
            new_version=1,
            label_type=label_type,
            description=description,
            split_strategy_name=split_strategy_name,
            created_by=self.user,
            operation="create",
            split_approach="initial",
            selection_config=selection_config,
            split_rows=split_rows,
            artifact_result=artifact_result
        )

        return {
            "dataset_id": dataset_id,
            "version": 1,
            "label_type": label_type,
            "description": description,
            "split_strategy_name": split_strategy_name,
            "candidate_count": len(candidates),
            "membership_row_count": membership_result["row_count"],
            "membership_table_name": membership_result["table_name"],
            "artifact_result": artifact_result,
            "ddb_result": ddb_result
        }

    def update_dataset(self,
                       *,
                       dataset_id: str,
                       operation: Literal["add", "remove"],
                       selection_config: dict[str, Any],
                       split_approach: Literal["maintain", "rebalance"] = "maintain",
                       split_strategy_name: str | None = None,
                       description: str | None = None) -> dict:
        """
        Create a new dataset version by adding or removing imagery.

        High-level flow:
        1. validate inputs
        2. load existing dataset metadata
        3. determine latest version and dataset invariants
        4. build selection SQL and resolve rows for both imagery to add/remove and existing dataset membership rows
        5. derive the next dataset image set
        6. assign or preserve train/val/test splits
        7. write Iceberg membership rows for the new version
        8. write S3 dataset artifacts for the new version
        9. write DynamoDB dataset metadata for the new version
        """
        # 1. Validate inputs
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

        # 2. Load current dataset state
        dataset_info = self.get_dataset(dataset_id)

        if not dataset_info["exists"]:
            raise ValueError(f"Dataset '{dataset_id}' does not exist.")

        # 3. Determine latest version
        latest_version = dataset_info["latest_version"]
        new_version = latest_version + 1
        label_type = dataset_info["label_type"]

        # Resolve effective split strategy for this update
        if split_approach == "maintain":
            effective_split_strategy_name = dataset_info["latest_version_split_strategy"]
        else:
            # rebalance path: validator already ensured this is provided
            effective_split_strategy_name = split_strategy_name

        # Resolve effective description for the new version
        if description is None:
            effective_description = dataset_info.get("latest_version_description")
        else:
            effective_description = description

        # 4. Build the SQL for obtaining canonical imagery selection and retrieve the normalized candidates from Iceberg
        # This represents the images we want to add or remove from the dataset in making the new version.
        selection_sql_for_update, selected_imagery_rows = resolve_candidate_imagery(self.iceberg_database_name,
                                                                                   label_type,
                                                                                   selection_config,
                                                                                   self.athena,
                                                                                   self.athena_output_s3_uri)

        if not selected_imagery_rows:
            raise ValueError(
                f"Selection returned zero candidate imagery rows to {operation} to/from dataset {dataset_id}."
            )

        dataset_membership_table_name = LABEL_TYPE_TO_MEMBERSHIP_TABLE.get(label_type)
        if dataset_membership_table_name is None:
            raise ValueError(f"Unsupported dataset label_type: {label_type!r}")

        membership_sql, current_rows = resolve_dataset_membership(iceberg_database_name=self.iceberg_database_name,
                                                                    dataset_membership_table_name=dataset_membership_table_name,
                                                                    canonical_imagery_table_name=self.canonical_imagery_table_name,
                                                                    dataset_id=dataset_id,
                                                                    version=latest_version,
                                                                    label_type=label_type,
                                                                    mode="minimal" if split_approach == "maintain" else "enriched",
                                                                    athena_client=self.athena,
                                                                    athena_output_s3_uri=self.athena_output_s3_uri)

        if not current_rows:
            raise ValueError(
                f"Selection returned zero current rows in dataset {dataset_id}."
            )

        # 5 and 6: get final dataset after applying operation, and assign split to each row according to specifications
        split_rows = get_updated_split_rows(selected_imagery_rows,
                                            current_rows,
                                            operation,
                                            split_approach,
                                            effective_split_strategy_name)

        if not split_rows:
            raise ValueError(
                f"After {operation} operation, {dataset_id} had no rows for the newest updated version."
            )

        # 7. Write the rows to the iceberg dataset tables
        membership_result = write_membership_rows(
            athena_client=self.athena,
            iceberg_database_name=self.iceberg_database_name,
            athena_output_s3_uri=self.athena_output_s3_uri,
            dataset_id=dataset_id,
            version=new_version,
            dataset_label_type=label_type,
            split_rows=split_rows
        )

        # 8. Write the S3 artifacts
        artifact_result = write_dataset_artifacts(
            s3_client=self.s3,
            dataset_bucket_name=self.datasets_bucket_name,
            dataset_id=dataset_id,
            version=new_version,
            label_type=label_type,
            split_strategy_name=effective_split_strategy_name,
            selection_sql=selection_sql_for_update,
            selection_config=selection_config,
            split_rows=split_rows
        )

        # 9. Write the new ddb dataset-version row and update the dataset table's row.
        ddb_result = write_dataset_ddb_records(
            new_dataset=False,
            dynamodb_resource=self.dynamodb,
            datasets_table_name=self.datasets_table_name,
            dataset_versions_table_name=self.dataset_versions_table_name,
            dataset_id=dataset_id,
            new_version=new_version,
            label_type=label_type,
            description=effective_description,
            split_strategy_name=effective_split_strategy_name,
            created_by=self.user,
            operation=operation,
            split_approach=split_approach,
            selection_config=selection_config,
            split_rows=split_rows,
            artifact_result=artifact_result
        )

        return {
            "dataset_id": dataset_id,
            "new_version": new_version,
            "label_type": label_type,
            "description": effective_description,
            "effective_split_strategy_name": effective_split_strategy_name,
            f"candidate_imagery_count_to_{operation}": len(selected_imagery_rows),
            "preexisting_membership_count": len(current_rows),
            "final_membership_row_count": membership_result["row_count"],
            "membership_table_name": membership_result["table_name"],
            "artifact_result": artifact_result,
            "ddb_result": ddb_result
        }