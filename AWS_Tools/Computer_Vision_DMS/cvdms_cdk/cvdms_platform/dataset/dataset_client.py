from boto3.dynamodb.types import TypeSerializer
from mypy_boto3_s3.client import S3Client
from mypy_boto3_athena.client import AthenaClient
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource

from cvdms_platform.dataset.split_strategies.stratified_v1 import assign_splits_stratified_v1

from cvdms_platform.dataset.utils.validator_utils import validate_create_dataset_inputs
from cvdms_platform.dataset.utils.sql_utils import build_selection_sql
from cvdms_platform.dataset.utils.athena_utils import resolve_candidates
from cvdms_platform.dataset.utils.membership_utils import write_membership_rows
from cvdms_platform.dataset.utils.artifact_utils import write_dataset_artifacts
from cvdms_platform.dataset.utils.ddb_utils import write_dataset_ddb_records

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

        # Build the SQL for obtaining membership
        selection_sql = build_selection_sql(
            iceberg_database_name=self.iceberg_database_name,
            dataset_label_type=label_type,
            selection_config=selection_config
        )

        candidates = resolve_candidates(athena_client=self.athena,
                                        iceberg_database_name=self.iceberg_database_name,
                                        athena_output_s3_uri=self.athena_output_s3_uri,
                                        selection_sql=selection_sql)

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
            dynamodb_resource=self.dynamodb,
            datasets_table_name=self.datasets_table_name,
            dataset_versions_table_name=self.dataset_versions_table_name,
            dataset_id=dataset_id,
            version=1,
            label_type=label_type,
            description=description,
            split_strategy_name=split_strategy_name,
            selection_config=selection_config,
            split_rows=split_rows,
            created_by=self.user,
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
            "ddb_result": ddb_result,
        }