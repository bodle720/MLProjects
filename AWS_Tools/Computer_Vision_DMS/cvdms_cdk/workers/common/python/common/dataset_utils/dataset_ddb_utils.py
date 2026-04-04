from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import boto3
from botocore.exceptions import ClientError

dynamodb_resource = boto3.resource("dynamodb")

def write_ddb_artifacts(
    *,
    new_dataset: bool,
    datasets_table_name: str,
    dataset_versions_table_name: str,
    dataset_id: str,
    new_version: int,
    label_type: str,
    description: str,
    split_strategy_name: str,
    created_by: str,
    operation: str,
    split_approach: str,
    selection_config: dict[str, Any],
    split_rows: list[dict[str, Any]],
    artifact_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Write the canonical dataset and dataset_version DynamoDB records.

    Behavior:
    - new_dataset=True:
        create the dataset row and create version row new_version
    - new_dataset=False:
        update the existing dataset row to point at new_version and create the
        new version row new_version

    This should only be called after:
    1. membership rows were successfully inserted into Iceberg
    2. dataset artifacts were successfully written to S3
    """
    if new_version < 1:
        raise ValueError(f"new_version must be >= 1, got {new_version}")

    if new_dataset and new_version != 1:
        raise ValueError(
            f"new_dataset=True requires new_version=1, got {new_version}"
        )

    if not new_dataset and new_version == 1:
        raise ValueError(
            "new_dataset=False cannot be used with new_version=1"
        )

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    count_summary = build_split_count_summary(split_rows=split_rows)

    dataset_item: dict[str, Any] = {
        "dataset_id": dataset_id,
        "latest_version": new_version,
        "label_type": label_type,
        "latest_version_created_at": created_at,
        "latest_version_description": description,
        "last_modified_by": created_by,
    }
    if new_dataset:
        dataset_item["created_at"] = created_at
        dataset_item["created_by"] = created_by

    dataset_version_item: dict[str, Any] = {
        "dataset_id": dataset_id,
        "version": new_version,
        "label_type": label_type,
        "created_at": created_at,
        "operation": operation,
        "split_approach": split_approach,
        "split_strategy_name": split_strategy_name,
        "version_description": description,
        "selection_config": selection_config,
        "created_by": created_by,
        "total_image_count": count_summary["total_image_count"],
        "total_train_count": count_summary["total_train_count"],
        "total_val_count": count_summary["total_val_count"],
        "total_test_count": count_summary["total_test_count"],
    }

    if artifact_result:
        if artifact_result.get("base_prefix") is not None:
            dataset_version_item["version_s3_prefix"] = artifact_result["base_prefix"]

        if artifact_result.get("selection_sql_uri") is not None:
            dataset_version_item["selection_sql_uri"] = artifact_result["selection_sql_uri"]

        if artifact_result.get("selection_config_uri") is not None:
            dataset_version_item["selection_config_uri"] = artifact_result["selection_config_uri"]

        if artifact_result.get("metadata_json_uri") is not None:
            dataset_version_item["metadata_json_uri"] = artifact_result["metadata_json_uri"]

        if artifact_result.get("membership_enriched_csv_uri") is not None:
            dataset_version_item["membership_enriched_csv_uri"] = artifact_result["membership_enriched_csv_uri"]

        if artifact_result.get("manifest_uris") is not None:
            dataset_version_item["manifest_uris"] = artifact_result["manifest_uris"]

    transact_write_dataset_and_version(
        new_dataset=new_dataset,
        datasets_table_name=datasets_table_name,
        dataset_versions_table_name=dataset_versions_table_name,
        dataset_item=dataset_item,
        dataset_version_item=dataset_version_item,
        expected_previous_version=None if new_dataset else (new_version - 1),
    )

    return {
        "dataset_item": dataset_item,
        "dataset_version_item": dataset_version_item,
    }

def build_split_count_summary(*, split_rows: list[dict[str, Any]]) -> dict[str, int]:
    split_counts = Counter()

    for row in split_rows:
        split = _require_valid_split(row.get("split"))
        split_counts[split] += 1

    return {
        "total_image_count": len(split_rows),
        "total_train_count": split_counts.get("train", 0),
        "total_val_count": split_counts.get("val", 0),
        "total_test_count": split_counts.get("test", 0),
    }

def transact_write_dataset_and_version(
    *,
    new_dataset: bool,
    datasets_table_name: str,
    dataset_versions_table_name: str,
    dataset_item: dict[str, Any],
    dataset_version_item: dict[str, Any],
    expected_previous_version: int | None,
) -> None:
    """
    Transactionally write:
    - dataset row (Put for new dataset, Update for existing dataset)
    - dataset_version row (always Put)

    For updates, expected_previous_version protects against concurrent writers.
    """
    if new_dataset:
        dataset_action = {
            "Put": {
                "TableName": datasets_table_name,
                "Item": to_ddb_item(dataset_item),
                "ConditionExpression": "attribute_not_exists(dataset_id)",
            }
        }
    else:
        if expected_previous_version is None:
            raise ValueError(
                "expected_previous_version is required when new_dataset=False"
            )

        dataset_action = {
            "Update": {
                "TableName": datasets_table_name,
                "Key": to_ddb_item({"dataset_id": dataset_item["dataset_id"]}),
                "ConditionExpression": (
                    "attribute_exists(dataset_id) AND latest_version = :expected_prev"
                ),
                "UpdateExpression": (
                    "SET latest_version = :new_version, "
                    "latest_version_created_at = :latest_version_created_at, "
                    "latest_version_description = :latest_version_description, "
                    "last_modified_by = :last_modified_by"
                ),
                "ExpressionAttributeValues": to_ddb_item(
                    {
                        ":expected_prev": expected_previous_version,
                        ":new_version": dataset_item["latest_version"],
                        ":latest_version_created_at": dataset_item["latest_version_created_at"],
                        ":latest_version_description": dataset_item["latest_version_description"],
                        ":last_modified_by": dataset_item["last_modified_by"],
                    }
                ),
            }
        }

    version_action = {
        "Put": {
            "TableName": dataset_versions_table_name,
            "Item": to_ddb_item(dataset_version_item),
            "ConditionExpression": (
                "attribute_not_exists(dataset_id) AND attribute_not_exists(version)"
            ),
        }
    }

    try:
        dynamodb_resource.meta.client.transact_write_items(
            TransactItems=[dataset_action, version_action]
        )
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        dataset_id = dataset_item.get("dataset_id", "<unknown>")
        version = dataset_version_item.get("version", "<unknown>")

        if error_code == "TransactionCanceledException":
            if new_dataset:
                raise ValueError(
                    f"Dataset '{dataset_id}' already exists or version {version} already exists."
                ) from e
            raise ValueError(
                f"Failed to advance dataset '{dataset_id}' to version {version}. "
                f"The dataset may not exist, the previous version may not match, "
                f"or version {version} already exists."
            ) from e

        raise

def to_ddb_item(value: Any) -> dict[str, Any]:
    """
    Convert a plain Python dict into the low-level DynamoDB attribute-value format.
    """
    if not isinstance(value, dict):
        raise TypeError(f"to_ddb_item expects dict input, got {type(value).__name__}")

    return {k: _to_ddb_attr(v) for k, v in value.items()}

def _to_ddb_attr(value: Any) -> dict[str, Any]:
    if value is None:
        return {"NULL": True}

    if isinstance(value, bool):
        return {"BOOL": value}

    if isinstance(value, str):
        return {"S": value}

    if isinstance(value, int):
        return {"N": str(value)}

    if isinstance(value, float):
        return {"N": str(Decimal(str(value)))}

    if isinstance(value, Decimal):
        return {"N": str(value)}

    if isinstance(value, list):
        return {"L": [_to_ddb_attr(v) for v in value]}

    if isinstance(value, dict):
        return {"M": {k: _to_ddb_attr(v) for k, v in value.items()}}

    raise TypeError(f"Unsupported type for DynamoDB serialization: {type(value).__name__}")

def _require_valid_split(value: Any) -> str:
    split = str(value).strip() if value is not None else ""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {value!r}")
    return split