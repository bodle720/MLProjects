import time
from typing import Any

from boto3.dynamodb.types import TypeSerializer
from mypy_boto3_s3.client import S3Client
from mypy_boto3_athena.client import AthenaClient
from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource

from cvdms_platform.dataset.input_validators import validate_create_dataset_inputs
from cvdms_platform.dataset.sql_utils import build_selection_sql

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

        candidates = self._resolve_candidates(selection_sql=selection_sql)

        if not candidates:
            raise ValueError(
                f"Dataset '{dataset_id}' selection returned zero candidate rows."
            )

        return {
            "dataset_id": dataset_id,
            "label_type": label_type,
            "candidate_count": len(candidates),
            "selection_sql": selection_sql,
            "candidates_preview": candidates[:5],
        }

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

    def _resolve_candidates(self, *, selection_sql: str) -> list[dict]:
        """
        Execute the selection SQL in Athena and return all candidate rows as a list of dicts.
        Values are normalized into expected Python types.
        """
        query_execution_id = self._start_athena_query(selection_sql=selection_sql)
        self._wait_for_athena_query(query_execution_id=query_execution_id)
        raw_rows = self._fetch_athena_results(query_execution_id=query_execution_id)
        return [self._normalize_candidate_row(row) for row in raw_rows]

    def _start_athena_query(self, *, selection_sql: str) -> str:
        """
        Start an Athena query in the Iceberg database and return the QueryExecutionId.
        """
        response = self.athena.start_query_execution(
            QueryString=selection_sql,
            QueryExecutionContext={"Database": self.iceberg_database_name},
        )
        return response["QueryExecutionId"]

    def _wait_for_athena_query(
            self,
            *,
            query_execution_id: str,
            poll_interval_seconds: float = 1.0,
            timeout_seconds: int = 900,
    ) -> None:
        """
        Poll Athena until the query succeeds, fails, or times out.
        """
        start = time.time()

        while True:
            response = self.athena.get_query_execution(QueryExecutionId=query_execution_id)
            status = response["QueryExecution"]["Status"]["State"]

            if status == "SUCCEEDED":
                return

            if status in {"FAILED", "CANCELLED"}:
                reason = response["QueryExecution"]["Status"].get(
                    "StateChangeReason",
                    "Unknown Athena error.",
                )
                raise RuntimeError(
                    f"Athena query {query_execution_id} ended with status {status}: {reason}"
                )

            if time.time() - start > timeout_seconds:
                try:
                    self.athena.stop_query_execution(QueryExecutionId=query_execution_id)
                except Exception:
                    pass

                raise TimeoutError(
                    f"Athena query {query_execution_id} did not finish within {timeout_seconds} seconds."
                )

            time.sleep(poll_interval_seconds)

    def _fetch_athena_results(self, *, query_execution_id: str) -> list[dict]:
        """
        Fetch all Athena result rows and return them as a list of dicts.
        Assumes the first row on the first page is the header row.
        """
        rows_out: list[dict] = []
        next_token: str | None = None
        column_names: list[str] | None = None
        is_first_page = True

        while True:
            kwargs: dict[str, Any] = {"QueryExecutionId": query_execution_id}
            if next_token:
                kwargs["NextToken"] = next_token

            response = self.athena.get_query_results(**kwargs)
            result_set = response["ResultSet"]
            rows = result_set.get("Rows", [])

            if is_first_page:
                if not rows:
                    return []

                header_row = rows[0]
                column_names = [
                    col.get("VarCharValue", "")
                    for col in header_row.get("Data", [])
                ]
                data_rows = rows[1:]
                is_first_page = False
            else:
                data_rows = rows

            for row in data_rows:
                rows_out.append(
                    self._athena_row_to_dict(
                        column_names=column_names or [],
                        row=row,
                    )
                )

            next_token = response.get("NextToken")
            if not next_token:
                break

        return rows_out

    @staticmethod
    def _athena_row_to_dict(*, column_names: list[str], row: dict) -> dict:
        """
        Convert a single Athena row to a Python dict keyed by column name.
        Missing values become None.
        """
        data = row.get("Data", [])
        out: dict[str, Any] = {}

        for idx, col_name in enumerate(column_names):
            if idx >= len(data):
                out[col_name] = None
                continue

            cell = data[idx]
            out[col_name] = cell.get("VarCharValue")

        return out

    @staticmethod
    def _normalize_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize Athena string-valued result cells into expected Python types.
        """
        int_fields = {
            "img_height",
            "img_width",
            "num_channels",
        }

        float_fields = {
            "file_size_mb",
            "luma_mean",
            "luma_p10",
            "luma_p90",
            "dark_frac",
            "bright_frac",
            "contrast_luma_std",
            "contrast_luma_p90_p10",
            "blur_laplacian_var",
            "sat_mean",
            "colorfulness",
        }

        normalized = dict(row)

        for field in int_fields:
            value = normalized.get(field)
            normalized[field] = None if value is None else int(value)

        for field in float_fields:
            value = normalized.get(field)
            normalized[field] = None if value is None else float(value)

        return normalized