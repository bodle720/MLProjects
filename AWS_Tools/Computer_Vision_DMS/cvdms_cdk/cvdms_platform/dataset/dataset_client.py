from boto3.dynamodb.types import TypeSerializer
from mypy_boto3_s3.client import S3Client
from mypy_boto3_athena.client import AthenaClient
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource

from cvdms_platform.dataset.input_validators import validate_create_dataset_inputs
from cvdms_platform.dataset.sql_utils import build_selection_sql

_ALLOWED_LABEL_TYPES = {"single-label", "multi-label", "object-detection", "semantic-segmentation", "instance-segmentation"}
_ALLOWED_SPLIT_STRATEGIES = {"stratified_v1"}
_serializer = TypeSerializer()

def _to_ddb_item(item: dict) -> dict:
    return {k: _serializer.serialize(v) for k, v in item.items()}

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
                 athena_client: AthenaClient):

        self.user = user
        self.datasets_bucket_name = datasets_bucket_name
        self.datasets_table_name = datasets_table_name
        self.dataset_versions_table_name = dataset_versions_table_name

        self.s3 = s3_client
        self.dynamodb = dynamodb_resource
        self.athena = athena_client

        self.datasets_table = self.dynamodb.Table(self.datasets_table_name)
        self.dataset_versions_table = self.dynamodb.Table(self.dataset_versions_table_name)
        self.iceberg_database_name = iceberg_database_name

    def create_dataset(self,
                        *,
                        dataset_id: str,
                        label_type: str,
                        description: str,
                        selection_config: dict,
                        split_strategy_name: str) -> dict:

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


        # Create and enter the DDB rows
        # created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        # version = 1
        #
        # dataset_item = {
        #     "dataset_id": dataset_id,
        #     "label_type": label_type,
        #     "created_at": created_at,
        #     "latest_version": version,
        #     "description": description,
        #     "created_by": self.user
        # }
        #
        # dataset_version_item = {
        #     "dataset_id": dataset_id,
        #     "version": version,
        #     "created_at": created_at,
        #     "total_image_count": 0,
        #     "train_image_count": 0,
        #     "val_image_count": 0,
        #     "test_image_count": 0,
        #     "split_strategy": split_strategy_name,
        #     "selection_config": selection_config,
        #     "created_by": self.user
        # }
        #
        # try:
        #     self.dynamodb.meta.client.transact_write_items(
        #         TransactItems=[
        #             {
        #                 "Put": {
        #                     "TableName": self.datasets_table_name,
        #                     "Item": _to_ddb_item(dataset_item),
        #                     "ConditionExpression": "attribute_not_exists(dataset_id)"
        #                 }
        #             },
        #             {
        #                 "Put": {
        #                     "TableName": self.dataset_versions_table_name,
        #                     "Item": _to_ddb_item(dataset_version_item),
        #                     "ConditionExpression": "attribute_not_exists(dataset_id) AND attribute_not_exists(version)"
        #                 }
        #             }
        #         ]
        #     )
        # except ClientError as e:
        #     error_code = e.response.get("Error", {}).get("Code", "")
        #     if error_code == "TransactionCanceledException":
        #         raise ValueError(f"Dataset '{dataset_id}' already exists or version 1 already exists.") from e
        #     raise

        # return {
        #     "dataset_id": dataset_id,
        #     "label_type": label_type,
        #     "description": description,
        #     "created_at": created_at,
        #     "latest_version": version,
        #     "split_strategy_name": split_strategy_name,
        #     "selection_config": selection_config
        # }
