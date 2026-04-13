import logging
from decimal import Decimal
from typing import Any

from botocore.exceptions import ClientError

def get_dataset_info(
    *,
    datasets_table: Any,
    dataset_versions_table: Any,
    dataset_id: str,
) -> dict[str, Any]:
    """
    Return normalized dataset information for the latest version.

    This helper is intentionally DDB-only. It relies on the dataset row for
    dataset-level state and the latest dataset_versions row for current version
    metadata, counts, and artifact locations.
    """
    logging.info("Getting dataset row.")
    dataset_item = get_dataset_row(
        datasets_table=datasets_table,
        dataset_id=dataset_id,
    )

    if dataset_item is None:
        return {
            "dataset_info": {"exists": False},
            "latest_version_info": None,
        }

    latest_version = _coerce_required_int(
        dataset_item.get("latest_version"),
        field_name="latest_version",
    )

    logging.info("Getting dataset version row.")
    dataset_version_item = get_dataset_version_row(
        dataset_versions_table=dataset_versions_table,
        dataset_id=dataset_id,
        version=latest_version,
    )

    # This should not normally happen if writes are correct, but fail loudly so
    # the inconsistency is visible rather than returning partial truth.
    if dataset_version_item is None:
        logging.error(
            f"Dataset '{dataset_id}' exists, but latest version row "
            f"{latest_version} was not found in dataset versions table."
        )
        raise ValueError(
            f"Dataset '{dataset_id}' exists, but latest version row "
            f"{latest_version} was not found in dataset versions table."
        )

    logging.info("Rows obtained. Building and returning result.")
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
    dataset_id = _coerce_required_string(
        dataset_item.get("dataset_id"),
        field_name="dataset_id",
    )
    latest_version = _coerce_required_int(
        dataset_item.get("latest_version"),
        field_name="latest_version",
    )
    label_type = _coerce_required_string(
        dataset_item.get("label_type"),
        field_name="label_type",
    )
    allowed_classes = _coerce_required_string_list(
        dataset_item.get("allowed_classes"),
        field_name="allowed_classes",
    )
    honor_source_splits = _coerce_required_bool(
        dataset_item.get("honor_source_splits"),
        field_name="honor_source_splits",
    )

    version = _coerce_required_int(
        dataset_version_item.get("version"),
        field_name="version",
    )
    total_image_count = _coerce_required_int(
        dataset_version_item.get("total_image_count"),
        field_name="total_image_count",
    )
    total_train_count = _coerce_required_int(
        dataset_version_item.get("total_train_count"),
        field_name="total_train_count",
    )
    total_val_count = _coerce_required_int(
        dataset_version_item.get("total_val_count"),
        field_name="total_val_count",
    )
    total_test_count = _coerce_required_int(
        dataset_version_item.get("total_test_count"),
        field_name="total_test_count",
    )

    latest_version_honor_source_splits = dataset_version_item.get("honor_source_splits")
    if latest_version_honor_source_splits is None:
        latest_version_honor_source_splits = honor_source_splits
    else:
        latest_version_honor_source_splits = _coerce_required_bool(
            latest_version_honor_source_splits,
            field_name="latest_version_info.honor_source_splits",
        )

    result: dict[str, Any] = {
        "dataset_info": {
            "exists": True,
            "dataset_id": dataset_id,
            "latest_version": latest_version,
            "label_type": label_type,
            "allowed_classes": allowed_classes,
            "honor_source_splits": honor_source_splits,
            "created_at": _coerce_optional_string(dataset_item.get("created_at")),
            "created_by": _coerce_optional_string(dataset_item.get("created_by")),
            "last_modified_by": _coerce_optional_string(dataset_item.get("last_modified_by")),
            "dataset_description": _coerce_optional_string(dataset_item.get("dataset_description")),
        },
        "latest_version_info": {
            "version": version,
            "created_at": _coerce_optional_string(dataset_version_item.get("created_at")),
            "description": _coerce_optional_string(dataset_version_item.get("description")),
            "operation": _coerce_optional_string(dataset_version_item.get("operation")),
            "split_approach": _coerce_optional_string(dataset_version_item.get("split_approach")),
            "split_strategy_name": _coerce_optional_string(dataset_version_item.get("split_strategy_name")),
            "honor_source_splits": latest_version_honor_source_splits,
            "effective_split_mode": _coerce_optional_string(dataset_version_item.get("effective_split_mode")),
            "total_image_count": total_image_count,
            "total_train_count": total_train_count,
            "total_val_count": total_val_count,
            "total_test_count": total_test_count,
            "version_s3_prefix": _coerce_optional_string(dataset_version_item.get("version_s3_prefix")),
            "selection_sql_uri": _coerce_optional_string(dataset_version_item.get("selection_sql_uri")),
            "selection_config_uri": _coerce_optional_string(dataset_version_item.get("selection_config_uri")),
            "metadata_json_uri": _coerce_optional_string(dataset_version_item.get("metadata_json_uri")),
            "membership_enriched_csv_uri": _coerce_optional_string(dataset_version_item.get("membership_enriched_csv_uri")),
            "manifest_uris": _coerce_optional_dict(dataset_version_item.get("manifest_uris")),
            "selection_config": _coerce_optional_dict(dataset_version_item.get("selection_config")),
        },
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

def _coerce_required_bool(value: Any, *, field_name: str) -> bool:
    value = normalize_ddb_value(value)

    if isinstance(value, bool):
        return value

    raise ValueError(f"Field '{field_name}' must be a bool, got {value!r}")

def _coerce_required_string(value: Any, *, field_name: str) -> str:
    value = normalize_ddb_value(value)

    if isinstance(value, str):
        text = value.strip()
        if text:
            return text

    raise ValueError(f"Field '{field_name}' must be a non-empty string, got {value!r}")

def _coerce_optional_string(value: Any) -> str | None:
    value = normalize_ddb_value(value)

    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        return text or None

    return str(value).strip() or None

def _coerce_required_string_list(value: Any, *, field_name: str) -> list[str]:
    value = normalize_ddb_value(value)

    if not isinstance(value, list):
        raise ValueError(f"Field '{field_name}' must be a list[str], got {value!r}")

    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"Field '{field_name}' must contain only strings, got {item!r}"
            )
        text = item.strip()
        if not text:
            continue
        out.append(text)

    if not out:
        raise ValueError(f"Field '{field_name}' must be a non-empty list[str]")

    return out

def _coerce_optional_dict(value: Any) -> dict[str, Any] | None:
    value = normalize_ddb_value(value)

    if value is None:
        return None

    if not isinstance(value, dict):
        raise ValueError(f"Expected dict or None, got {value!r}")

    return value