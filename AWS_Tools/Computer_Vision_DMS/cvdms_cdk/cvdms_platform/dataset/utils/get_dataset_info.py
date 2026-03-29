from decimal import Decimal
from typing import Any

from botocore.exceptions import ClientError

def get_dataset_info(
    *,
    dynamodb_resource: Any,
    datasets_table_name: str,
    dataset_versions_table_name: str,
    dataset_id: str,
) -> dict[str, Any]:
    """
    Return normalized dataset information for the latest version.

    Returns:
        {"exists": False}
    if the dataset row does not exist.

    Otherwise returns:
        {
            "exists": True,
            ...
        }

    This helper is intentionally DDB-only. It relies on the dataset row for
    dataset-level state and the latest dataset_versions row for current version
    metadata, counts, and artifact locations.
    """
    datasets_table = dynamodb_resource.Table(datasets_table_name)
    dataset_versions_table = dynamodb_resource.Table(dataset_versions_table_name)

    dataset_item = get_dataset_row(
        datasets_table=datasets_table,
        dataset_id=dataset_id,
    )
    if dataset_item is None:
        return {"exists": False}

    latest_version = _coerce_required_int(
        dataset_item.get("latest_version"),
        field_name="latest_version",
    )

    dataset_version_item = get_dataset_version_row(
        dataset_versions_table=dataset_versions_table,
        dataset_id=dataset_id,
        version=latest_version,
    )

    # This should not normally happen if writes are correct, but fail loudly so
    # the inconsistency is visible rather than returning partial truth.
    if dataset_version_item is None:
        raise ValueError(
            f"Dataset '{dataset_id}' exists, but latest version row "
            f"{latest_version} was not found in '{dataset_versions_table_name}'."
        )

    return build_get_dataset_result(
        dataset_item=dataset_item,
        dataset_version_item=dataset_version_item,
    )

def get_dataset_row(
    *,
    datasets_table: Any,
    dataset_id: str,
) -> dict[str, Any] | None:
    try:
        response = datasets_table.get_item(
            Key={"dataset_id": dataset_id},
            ConsistentRead=True,
        )
    except ClientError as e:
        raise ValueError(
            f"Failed reading dataset row for dataset_id='{dataset_id}'."
        ) from e

    item = response.get("Item")
    if item is None:
        return None

    return normalize_ddb_value(item)

def get_dataset_version_row(
    *,
    dataset_versions_table: Any,
    dataset_id: str,
    version: int,
) -> dict[str, Any] | None:
    try:
        response = dataset_versions_table.get_item(
            Key={"dataset_id": dataset_id, "version": version},
            ConsistentRead=True,
        )
    except ClientError as e:
        raise ValueError(
            f"Failed reading dataset version row for dataset_id='{dataset_id}', "
            f"version={version}."
        ) from e

    item = response.get("Item")
    if item is None:
        return None

    return normalize_ddb_value(item)

def build_get_dataset_result(
    *,
    dataset_item: dict[str, Any],
    dataset_version_item: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": True,

        # Dataset-level fields
        "dataset_id": dataset_item["dataset_id"],
        "label_type": dataset_item["label_type"],
        "created_at": dataset_item.get("created_at"),
        "created_by": dataset_item.get("created_by"),
        "latest_version": _coerce_required_int(
            dataset_item.get("latest_version"),
            field_name="latest_version",
        ),
        "latest_version_created_at": dataset_item.get("latest_version_created_at"),
        "latest_version_description": dataset_item.get("latest_version_description"),
        "last_modified_by": dataset_item.get("last_modified_by"),

        # Latest-version fields
        "version_created_at": dataset_version_item.get("created_at"),
        "operation": dataset_version_item.get("operation"),
        "split_approach": dataset_version_item.get("split_approach"),
        "latest_version_split_strategy": dataset_version_item.get("split_strategy_name"),
        "version_description": dataset_version_item.get("version_description"),

        # Counts
        "total_image_count": _coerce_required_int(
            dataset_version_item.get("total_image_count"),
            field_name="total_image_count",
        ),
        "total_train_count": _coerce_required_int(
            dataset_version_item.get("total_train_count"),
            field_name="total_train_count",
        ),
        "total_val_count": _coerce_required_int(
            dataset_version_item.get("total_val_count"),
            field_name="total_val_count",
        ),
        "total_test_count": _coerce_required_int(
            dataset_version_item.get("total_test_count"),
            field_name="total_test_count",
        ),

        # Artifact pointers
        "version_s3_prefix": dataset_version_item.get("version_s3_prefix"),
        "selection_sql_uri": dataset_version_item.get("selection_sql_uri"),
        "selection_config_uri": dataset_version_item.get("selection_config_uri"),
        "metadata_json_uri": dataset_version_item.get("metadata_json_uri"),
        "membership_enriched_csv_uri": dataset_version_item.get("membership_enriched_csv_uri"),
        "manifest_uris": dataset_version_item.get("manifest_uris"),

        # Useful raw metadata that update_dataset will likely want later
        "selection_config": dataset_version_item.get("selection_config"),
    }

    return result

def normalize_ddb_value(value: Any) -> Any:
    """
    Recursively normalize values returned by boto3 DynamoDB resource.Table(...).
    In particular, convert Decimal -> int/float where appropriate.
    """
    if isinstance(value, Decimal):
        if value % 1 == 0:
            return int(value)
        return float(value)

    if isinstance(value, list):
        return [normalize_ddb_value(v) for v in value]

    if isinstance(value, dict):
        return {k: normalize_ddb_value(v) for k, v in value.items()}

    return value

def _coerce_required_int(value: Any, *, field_name: str) -> int:
    value = normalize_ddb_value(value)

    if isinstance(value, bool):
        raise ValueError(f"Field '{field_name}' must be an int, got bool")

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)

    raise ValueError(f"Field '{field_name}' must be an int, got {value!r}")