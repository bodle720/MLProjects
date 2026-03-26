from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from botocore.exceptions import ClientError

def write_dataset_ddb_records(
    *,
    dynamodb_resource: Any,
    datasets_table_name: str,
    dataset_versions_table_name: str,
    dataset_id: str,
    version: int,
    label_type: str,
    description: str,
    split_strategy_name: str,
    selection_config: dict[str, Any],
    split_rows: list[dict[str, Any]],
    created_by: str,
    artifact_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Write the canonical dataset and dataset_version DynamoDB records.

    This should be called only after:
    1. membership rows were successfully inserted into Iceberg
    2. dataset artifacts were successfully written to S3

    Uses a transactional write so the dataset row and version row appear together.
    """
    if version < 1:
        raise ValueError(f"version must be >= 1, got {version}")

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    count_summary = build_split_count_summary(split_rows=split_rows)

    dataset_item = build_dataset_item(
        dataset_id=dataset_id,
        label_type=label_type,
        description=description,
        created_by=created_by,
        created_at=created_at,
        latest_version=version,
    )

    dataset_version_item = build_dataset_version_item(
        dataset_id=dataset_id,
        version=version,
        created_at=created_at,
        total_image_count=count_summary["total_image_count"],
        train_image_count=count_summary["train_image_count"],
        val_image_count=count_summary["val_image_count"],
        test_image_count=count_summary["test_image_count"],
        split_strategy_name=split_strategy_name,
        selection_config=selection_config,
        created_by=created_by,
        artifact_result=artifact_result,
    )

    transact_put_dataset_and_version(
        dynamodb_resource=dynamodb_resource,
        datasets_table_name=datasets_table_name,
        dataset_versions_table_name=dataset_versions_table_name,
        dataset_item=dataset_item,
        dataset_version_item=dataset_version_item,
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
        "train_image_count": split_counts.get("train", 0),
        "val_image_count": split_counts.get("val", 0),
        "test_image_count": split_counts.get("test", 0),
    }

def build_dataset_item(
    *,
    dataset_id: str,
    label_type: str,
    description: str,
    created_by: str,
    created_at: str,
    latest_version: int,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "label_type": label_type,
        "created_at": created_at,
        "latest_version": latest_version,
        "description": description,
        "created_by": created_by,
    }

def build_dataset_version_item(
    *,
    dataset_id: str,
    version: int,
    created_at: str,
    total_image_count: int,
    train_image_count: int,
    val_image_count: int,
    test_image_count: int,
    split_strategy_name: str,
    selection_config: dict[str, Any],
    created_by: str,
    artifact_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "dataset_id": dataset_id,
        "version": version,
        "created_at": created_at,
        "total_image_count": total_image_count,
        "train_image_count": train_image_count,
        "val_image_count": val_image_count,
        "test_image_count": test_image_count,
        "split_strategy": split_strategy_name,
        "selection_config": selection_config,
        "created_by": created_by,
    }

    if artifact_result:
        # Store helpful artifact pointers if present.
        if artifact_result.get("base_prefix") is not None:
            item["base_prefix"] = artifact_result["base_prefix"]

        if artifact_result.get("selection_sql_uri") is not None:
            item["selection_sql_uri"] = artifact_result["selection_sql_uri"]

        if artifact_result.get("selection_config_uri") is not None:
            item["selection_config_uri"] = artifact_result["selection_config_uri"]

        if artifact_result.get("metadata_json_uri") is not None:
            item["metadata_json_uri"] = artifact_result["metadata_json_uri"]

        if artifact_result.get("membership_enriched_csv_uri") is not None:
            item["membership_enriched_csv_uri"] = artifact_result["membership_enriched_csv_uri"]

        if artifact_result.get("manifest_uris") is not None:
            item["manifest_uris"] = artifact_result["manifest_uris"]

    return item

def transact_put_dataset_and_version(
    *,
    dynamodb_resource: Any,
    datasets_table_name: str,
    dataset_versions_table_name: str,
    dataset_item: dict[str, Any],
    dataset_version_item: dict[str, Any],
) -> None:
    """
    Transactionally create:
    - dataset row
    - dataset_version row

    Intended for initial dataset creation (version 1).
    """
    try:
        dynamodb_resource.meta.client.transact_write_items(
            TransactItems=[
                {
                    "Put": {
                        "TableName": datasets_table_name,
                        "Item": to_ddb_item(dataset_item),
                        "ConditionExpression": "attribute_not_exists(dataset_id)",
                    }
                },
                {
                    "Put": {
                        "TableName": dataset_versions_table_name,
                        "Item": to_ddb_item(dataset_version_item),
                        "ConditionExpression": "attribute_not_exists(dataset_id) AND attribute_not_exists(version)",
                    }
                },
            ]
        )
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "TransactionCanceledException":
            dataset_id = dataset_item.get("dataset_id", "<unknown>")
            version = dataset_version_item.get("version", "<unknown>")
            raise ValueError(
                f"Dataset '{dataset_id}' already exists or version {version} already exists."
            ) from e
        raise

def to_ddb_item(value: Any) -> dict[str, Any]:
    """
    Convert a plain Python dict into the low-level DynamoDB attribute-value format.

    This mirrors the structure expected by transact_write_items(..., Item=...).
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
        # DynamoDB does not accept native float directly in the higher-level serializer sense,
        # so convert through Decimal-safe string representation.
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