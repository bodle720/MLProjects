from typing import Any

from cvdms_platform.dataset.utils.athena_utils import (
    start_athena_query,
    wait_for_athena_query
)

_MEMBERSHIP_TABLE_BY_LABEL_TYPE: dict[str, str] = {
    "single-label": "single_label",
    "multi-label": "multi_label",
    "object-detection": "object_detection",
    "semantic-segmentation": "semantic_segmentation",
    "instance-segmentation": "instance_segmentation",
}

_VALID_SPLITS = {"train", "val", "test"}

def write_membership_rows(
    *,
    athena_client: Any,
    iceberg_database_name: str,
    athena_output_s3_uri: str,
    dataset_id: str,
    version: int,
    dataset_label_type: str,
    split_rows: list[dict[str, Any]],
    chunk_size: int = 500,
) -> dict[str, Any]:
    """
    Build and insert dataset membership rows into the correct Iceberg table.

    Returns a small summary dict.
    """
    membership_rows = build_membership_rows(
        dataset_id=dataset_id,
        version=version,
        dataset_label_type=dataset_label_type,
        split_rows=split_rows,
    )

    if not membership_rows:
        raise ValueError("No membership rows were built from split_rows.")

    table_name = get_membership_table_name(dataset_label_type=dataset_label_type)

    insert_membership_rows(
        athena_client=athena_client,
        iceberg_database_name=iceberg_database_name,
        athena_output_s3_uri=athena_output_s3_uri,
        table_name=table_name,
        dataset_label_type=dataset_label_type,
        membership_rows=membership_rows,
        chunk_size=chunk_size,
    )

    return {
        "table_name": table_name,
        "row_count": len(membership_rows),
        "dataset_id": dataset_id,
        "version": version,
        "dataset_label_type": dataset_label_type,
    }

def build_membership_rows(
    *,
    dataset_id: str,
    version: int,
    dataset_label_type: str,
    split_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Transform split_rows into the minimal membership schema for the target table.
    """
    if version < 1:
        raise ValueError(f"version must be >= 1, got {version}")

    out: list[dict[str, Any]] = []

    for row in split_rows:
        image_id = _require_nonempty_string(row.get("image_id"), field_name="image_id")
        split = _require_valid_split(row.get("split"))

        if dataset_label_type == "single-label":
            label = _require_nonempty_string(row.get("label"), field_name="label")
            out.append(
                {
                    "dataset_id": dataset_id,
                    "version": version,
                    "image_id": image_id,
                    "label": label,
                    "split": split,
                }
            )
            continue

        if dataset_label_type == "multi-label":
            labels = _normalize_string_array(
                row.get("labels"),
                field_name="labels",
                require_nonempty=True,
            )
            out.append(
                {
                    "dataset_id": dataset_id,
                    "version": version,
                    "image_id": image_id,
                    "labels": labels,
                    "split": split,
                }
            )
            continue

        if dataset_label_type == "object-detection":
            bbox_annotation_ids = _normalize_string_array(
                row.get("bbox_annotation_ids"),
                field_name="bbox_annotation_ids",
                require_nonempty=True,
            )
            out.append(
                {
                    "dataset_id": dataset_id,
                    "version": version,
                    "image_id": image_id,
                    "bbox_annotation_ids": bbox_annotation_ids,
                    "split": split,
                }
            )
            continue

        if dataset_label_type == "semantic-segmentation":
            semantic_mask_ids = _normalize_string_array(
                row.get("semantic_mask_ids"),
                field_name="semantic_mask_ids",
                require_nonempty=True,
            )
            out.append(
                {
                    "dataset_id": dataset_id,
                    "version": version,
                    "image_id": image_id,
                    "semantic_mask_ids": semantic_mask_ids,
                    "split": split,
                }
            )
            continue

        if dataset_label_type == "instance-segmentation":
            instance_annotation_ids = _normalize_string_array(
                row.get("instance_annotation_ids"),
                field_name="instance_annotation_ids",
                require_nonempty=True,
            )
            out.append(
                {
                    "dataset_id": dataset_id,
                    "version": version,
                    "image_id": image_id,
                    "instance_annotation_ids": instance_annotation_ids,
                    "split": split,
                }
            )
            continue

        raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}")

    return out

def get_membership_table_name(*, dataset_label_type: str) -> str:
    try:
        return _MEMBERSHIP_TABLE_BY_LABEL_TYPE[dataset_label_type]
    except KeyError as e:
        raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}") from e

def insert_membership_rows(
    *,
    athena_client: Any,
    iceberg_database_name: str,
    athena_output_s3_uri: str,
    table_name: str,
    dataset_label_type: str,
    membership_rows: list[dict[str, Any]],
    chunk_size: int = 500,
) -> None:
    """
    Insert membership rows into the correct Iceberg table using Athena INSERT INTO.

    Uses chunking to avoid oversized SQL statements.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    column_names = get_membership_column_names(dataset_label_type=dataset_label_type)

    for i in range(0, len(membership_rows), chunk_size):
        chunk = membership_rows[i : i + chunk_size]
        sql = build_membership_insert_sql(
            iceberg_database_name=iceberg_database_name,
            table_name=table_name,
            dataset_label_type=dataset_label_type,
            column_names=column_names,
            membership_rows=chunk,
        )

        query_execution_id = start_athena_query(
            athena_client=athena_client,
            iceberg_database_name=iceberg_database_name,
            athena_output_s3_uri=athena_output_s3_uri,
            selection_sql=sql,
        )
        wait_for_athena_query(
            athena_client=athena_client,
            query_execution_id=query_execution_id,
        )

def get_membership_column_names(*, dataset_label_type: str) -> list[str]:
    if dataset_label_type == "single-label":
        return ["dataset_id", "version", "image_id", "label", "split"]

    if dataset_label_type == "multi-label":
        return ["dataset_id", "version", "image_id", "labels", "split"]

    if dataset_label_type == "object-detection":
        return ["dataset_id", "version", "image_id", "bbox_annotation_ids", "split"]

    if dataset_label_type == "semantic-segmentation":
        return ["dataset_id", "version", "image_id", "semantic_mask_ids", "split"]

    if dataset_label_type == "instance-segmentation":
        return ["dataset_id", "version", "image_id", "instance_annotation_ids", "split"]

    raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}")

def build_membership_insert_sql(
    *,
    iceberg_database_name: str,
    table_name: str,
    dataset_label_type: str,
    column_names: list[str],
    membership_rows: list[dict[str, Any]],
) -> str:
    if not membership_rows:
        raise ValueError("membership_rows cannot be empty")

    values_sql = ",\n".join(
        _membership_row_to_values_sql(
            dataset_label_type=dataset_label_type,
            row=row,
        )
        for row in membership_rows
    )

    columns_sql = ", ".join(column_names)

    return f"""
INSERT INTO {iceberg_database_name}.{table_name} ({columns_sql})
VALUES
{values_sql}
""".strip() + "\n"

def _membership_row_to_values_sql(
    *,
    dataset_label_type: str,
    row: dict[str, Any],
) -> str:
    dataset_id_sql = _sql_quote(row["dataset_id"])
    version_sql = str(int(row["version"]))
    image_id_sql = _sql_quote(row["image_id"])
    split_sql = _sql_quote(row["split"])

    if dataset_label_type == "single-label":
        label_sql = _sql_quote(row["label"])
        return f"({dataset_id_sql}, {version_sql}, {image_id_sql}, {label_sql}, {split_sql})"

    if dataset_label_type == "multi-label":
        labels_sql = _sql_array_literal(row["labels"])
        return f"({dataset_id_sql}, {version_sql}, {image_id_sql}, {labels_sql}, {split_sql})"

    if dataset_label_type == "object-detection":
        bbox_ids_sql = _sql_array_literal(row["bbox_annotation_ids"])
        return f"({dataset_id_sql}, {version_sql}, {image_id_sql}, {bbox_ids_sql}, {split_sql})"

    if dataset_label_type == "semantic-segmentation":
        semantic_ids_sql = _sql_array_literal(row["semantic_mask_ids"])
        return f"({dataset_id_sql}, {version_sql}, {image_id_sql}, {semantic_ids_sql}, {split_sql})"

    if dataset_label_type == "instance-segmentation":
        instance_ids_sql = _sql_array_literal(row["instance_annotation_ids"])
        return f"({dataset_id_sql}, {version_sql}, {image_id_sql}, {instance_ids_sql}, {split_sql})"

    raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}")

def _normalize_string_array(
    value: Any,
    *,
    field_name: str,
    require_nonempty: bool,
) -> list[str]:
    if value is None:
        values: list[str] = []
    elif isinstance(value, list):
        values = [str(v).strip() for v in value if str(v).strip()]
    else:
        raise TypeError(f"{field_name} must be list[str] | None, got {type(value).__name__}")

    # dedupe + deterministic sort
    values = sorted(set(values))

    if require_nonempty and not values:
        raise ValueError(f"{field_name} must be non-empty")

    return values

def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be None")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text

def _require_valid_split(value: Any) -> str:
    split = _require_nonempty_string(value, field_name="split")
    if split not in _VALID_SPLITS:
        raise ValueError(f"Invalid split: {split!r}")
    return split

def _sql_array_literal(values: list[str]) -> str:
    if not values:
        return "ARRAY[]"
    inner = ", ".join(_sql_quote(v) for v in values)
    return f"ARRAY[{inner}]"

def _sql_quote(value: str) -> str:
    return f"'{_sql_escape_literal(value)}'"

def _sql_escape_literal(value: str) -> str:
    return value.replace("'", "''")