from collections import Counter
from datetime import datetime, timezone
from typing import Any

import boto3
from boto3.dynamodb.types import TypeSerializer
from botocore.exceptions import ClientError

dynamodb_resource = boto3.resource("dynamodb")
dynamodb_client = boto3.client("dynamodb")

_VALID_SPLITS = {"train", "val", "test"}
_VALID_SPLIT_APPROACHES = {"initial", "maintain", "rebalance"}

_serializer = TypeSerializer()

def to_ddb_item(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"to_ddb_item expects dict input, got {type(value).__name__}")
    return {k: _serializer.serialize(v) for k, v in value.items()}

def write_ddb_artifacts(
    *,
    new_dataset: bool,
    datasets_table_name: str,
    dataset_versions_table_name: str,
    dataset_id: str,
    new_version: int,
    label_type: str,
    dataset_description: str | None,
    version_description: str | None,
    split_strategy_name: str | None,
    honor_source_splits: bool,
    created_by: str,
    operation: str,
    split_approach: str,
    selection_config: dict[str, Any],
    split_rows: list[dict[str, Any]],
    artifact_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Write the canonical dataset and dataset_version DynamoDB records.

    Contract:
    - datasets table:
        * dataset_id
        * latest_version
        * label_type
        * allowed_classes              (immutable after create)
        * honor_source_splits          (immutable after create)
        * created_at                   (create only)
        * created_by                   (create only)
        * last_modified_by
        * dataset_description          (immutable after create)

    - dataset_versions table:
        * dataset_id
        * version
        * label_type
        * created_at
        * operation
        * split_approach
        * split_strategy_name
        * honor_source_splits
        * effective_split_mode
        * description                  (version-specific)
        * selection_config             (exact request config for this version)
        * created_by
        * split counts
        * artifact pointers

    Behavior:
    - new_dataset=True:
        create the dataset row and create version row 1
    - new_dataset=False:
        update ONLY mutable dataset-row fields (latest_version, last_modified_by)
        and create the new version row

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

    if not isinstance(selection_config, dict):
        raise TypeError("selection_config must be a dict")

    if not isinstance(honor_source_splits, bool):
        raise TypeError("honor_source_splits must be a bool")

    split_approach = _normalize_required_string(split_approach, field_name="split_approach")
    if split_approach not in _VALID_SPLIT_APPROACHES:
        raise ValueError(
            f"split_approach must be one of {sorted(_VALID_SPLIT_APPROACHES)}, got {split_approach!r}"
        )

    normalized_split_strategy_name = _normalize_optional_string(split_strategy_name)
    effective_split_mode = _derive_effective_split_mode(
        new_dataset=new_dataset,
        split_approach=split_approach,
        honor_source_splits=honor_source_splits,
        split_strategy_name=normalized_split_strategy_name,
    )

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    canonical_split_rows, duplicate_collapsed_count = _canonicalize_split_rows_for_metadata_or_raise(
        split_rows=split_rows,
    )
    count_summary = build_split_count_summary(split_rows=canonical_split_rows)

    allowed_classes = selection_config.get("allowed_classes")
    if not isinstance(allowed_classes, list) or not allowed_classes:
        raise ValueError("selection_config.allowed_classes must be a non-empty list")

    # Immutable dataset-level description: only written on create.
    if new_dataset:
        effective_dataset_description = _normalize_optional_string(dataset_description)
        if effective_dataset_description is None:
            effective_dataset_description = _default_dataset_description(
                label_type=label_type,
                honor_source_splits=honor_source_splits,
                created_at=created_at,
            )
    else:
        effective_dataset_description = None  # must not be written/updated

    # Version-level description: written for every version.
    effective_version_description = _normalize_optional_string(version_description)
    if effective_version_description is None:
        effective_version_description = _default_version_description(
            new_dataset=new_dataset,
            label_type=label_type,
            operation=operation,
            split_approach=split_approach,
            version=new_version,
            honor_source_splits=honor_source_splits,
            split_strategy_name=normalized_split_strategy_name,
            created_at=created_at,
        )

    dataset_item: dict[str, Any] = {
        "dataset_id": dataset_id,
        "latest_version": new_version,
        "label_type": label_type,
        "last_modified_by": created_by,
    }

    if new_dataset:
        dataset_item["allowed_classes"] = allowed_classes
        dataset_item["honor_source_splits"] = honor_source_splits
        dataset_item["created_at"] = created_at
        dataset_item["created_by"] = created_by
        dataset_item["dataset_description"] = effective_dataset_description

    dataset_version_item: dict[str, Any] = {
        "dataset_id": dataset_id,
        "version": new_version,
        "label_type": label_type,
        "created_at": created_at,
        "operation": operation,
        "split_approach": split_approach,
        "split_strategy_name": normalized_split_strategy_name,
        "honor_source_splits": honor_source_splits,
        "effective_split_mode": effective_split_mode,
        "description": effective_version_description,
        "selection_config": selection_config,
        "created_by": created_by,
        "total_image_count": count_summary["total_image_count"],
        "total_train_count": count_summary["total_train_count"],
        "total_val_count": count_summary["total_val_count"],
        "total_test_count": count_summary["total_test_count"],
    }

    if duplicate_collapsed_count > 0:
        dataset_version_item["duplicate_collapsed_count"] = duplicate_collapsed_count

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
                    "last_modified_by = :last_modified_by"
                ),
                "ExpressionAttributeValues": to_ddb_item(
                    {
                        ":expected_prev": expected_previous_version,
                        ":new_version": dataset_item["latest_version"],
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
        dynamodb_client.transact_write_items(
            TransactItems=[dataset_action, version_action]
        )
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        dataset_id = dataset_item.get("dataset_id", "<unknown>")
        version = dataset_version_item.get("version", "<unknown>")

        if error_code == "TransactionCanceledException":
            cancellation_reasons = e.response.get("CancellationReasons", [])
            reasons_text = "; ".join(
                f"{idx}:{reason.get('Code')}:{reason.get('Message')}"
                for idx, reason in enumerate(cancellation_reasons)
            ) or "<no cancellation reasons returned>"

            if new_dataset:
                raise RuntimeError(
                    f"Create transaction failed for dataset '{dataset_id}' version {version}. "
                    f"CancellationReasons={reasons_text}"
                ) from e

            raise RuntimeError(
                f"Update transaction failed for dataset '{dataset_id}' version {version}. "
                f"CancellationReasons={reasons_text}"
            ) from e

        raise

def _require_valid_split(value: Any) -> str:
    split = str(value).strip() if value is not None else ""
    if split not in _VALID_SPLITS:
        raise ValueError(f"Invalid split: {value!r}")
    return split

def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def _normalize_required_string(value: Any, *, field_name: str) -> str:
    text = _normalize_optional_string(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text

def _derive_effective_split_mode(
    *,
    new_dataset: bool,
    split_approach: str,
    honor_source_splits: bool,
    split_strategy_name: str | None,
) -> str:
    if honor_source_splits:
        return "honor_source_splits"

    if new_dataset:
        if not split_strategy_name:
            raise ValueError(
                "split_strategy_name is required for dataset create when honor_source_splits=False"
            )
        return split_strategy_name

    if split_approach == "rebalance":
        if not split_strategy_name:
            raise ValueError(
                "split_strategy_name is required when split_approach='rebalance' "
                "and honor_source_splits=False"
            )
        return split_strategy_name

    if split_approach == "maintain":
        return split_strategy_name or "maintain"

    if split_approach == "initial":
        if not split_strategy_name:
            raise ValueError(
                "split_strategy_name is required when split_approach='initial' "
                "and honor_source_splits=False"
            )
        return split_strategy_name

    raise ValueError(f"Unsupported split_approach: {split_approach!r}")

def _canonicalize_split_rows_for_metadata_or_raise(
    *,
    split_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    Defensive metadata-only canonicalization.

    Enforce one logical row per image_id:
    - exact duplicate rows collapse
    - conflicting duplicate rows raise

    Returns:
    - canonical rows sorted by image_id
    - number of exact duplicates collapsed
    """
    canonical_by_image_id: dict[str, dict[str, Any]] = {}
    duplicate_collapsed_count = 0

    for row in split_rows:
        image_id = _normalize_required_string(row.get("image_id"), field_name="split_rows[].image_id")
        split = _require_valid_split(row.get("split"))

        normalized_row = {**row, "image_id": image_id, "split": split}

        existing = canonical_by_image_id.get(image_id)
        if existing is None:
            canonical_by_image_id[image_id] = normalized_row
            continue

        if existing == normalized_row:
            duplicate_collapsed_count += 1
            continue

        raise ValueError(
            f"Conflicting duplicate split rows for image_id={image_id!r}. "
            f"First row={existing!r}; duplicate row={normalized_row!r}"
        )

    canonical_rows = sorted(
        canonical_by_image_id.values(),
        key=lambda r: r["image_id"],
    )

    return canonical_rows, duplicate_collapsed_count

def _default_dataset_description(
    *,
    label_type: str,
    honor_source_splits: bool,
    created_at: str,
) -> str:
    if honor_source_splits:
        return f"{label_type} dataset honoring source splits created at {created_at}"
    return f"{label_type} dataset created at {created_at}"

def _default_version_description(
    *,
    new_dataset: bool,
    label_type: str,
    operation: str,
    split_approach: str,
    version: int,
    honor_source_splits: bool,
    split_strategy_name: str | None,
    created_at: str,
) -> str:
    if new_dataset:
        if honor_source_splits:
            return (
                f"Initial {label_type} dataset version v{version} honoring source splits "
                f"created at {created_at}"
            )
        if split_strategy_name:
            return (
                f"Initial {label_type} dataset version v{version} "
                f"({split_strategy_name}) created at {created_at}"
            )
        return f"Initial {label_type} dataset version v{version} created at {created_at}"

    if honor_source_splits:
        return (
            f"Dataset update v{version} ({operation}, {split_approach}) honoring "
            f"source splits created at {created_at}"
        )

    if split_strategy_name:
        return (
            f"Dataset update v{version} ({operation}, {split_approach}, {split_strategy_name}) "
            f"created at {created_at}"
        )

    return (
        f"Dataset update v{version} ({operation}, {split_approach}) "
        f"created at {created_at}"
    )