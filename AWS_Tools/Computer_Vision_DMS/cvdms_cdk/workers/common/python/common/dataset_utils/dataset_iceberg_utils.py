from typing import Any, Union

from common.general_utils.athena_utils import run_athena
from common.general_utils.iceberg_utils import (
    normalize_string_array,
    require_nonempty_string,
    build_insert_sql,
)
from common.general_utils.table_schemas import TABLES, TableSchema

_MEMBERSHIP_TABLE_BY_LABEL_TYPE: dict[str, str] = {
    "single-label": "single_label",
    "multi-label": "multi_label",
    "object-detection": "object_detection",
    "semantic-segmentation": "semantic_segmentation",
    "instance-segmentation": "instance_segmentation",
}

_VALID_SPLITS = {"train", "val", "test"}
_EXPECTED_MEMBERSHIP_KEY_COLS = ["dataset_id", "version", "image_id"]

_ICEBERG_COMMIT_RETRY_MARKERS = (
    "ICEBERG_COMMIT_ERROR",
    "Failed to commit Iceberg update",
)


def _is_retryable_iceberg_commit_error(exc: Exception | str) -> bool:
    text = str(exc)
    return any(marker in text for marker in _ICEBERG_COMMIT_RETRY_MARKERS)


def _escape_sql_string(value: str) -> str:
    return value.replace("'", "''")


def _assert_membership_table_key_shape(table_name: str) -> TableSchema:
    schema = TABLES.get(table_name)
    if schema is None:
        raise ValueError(f"Unknown membership table schema: {table_name}")

    if list(schema.key_cols) != _EXPECTED_MEMBERSHIP_KEY_COLS:
        raise ValueError(
            f"Membership table {table_name} must use key_cols="
            f"{_EXPECTED_MEMBERSHIP_KEY_COLS}, got {schema.key_cols}"
        )

    return schema


def _dedupe_membership_rows_or_raise(
    rows: list[dict[str, Any]],
    *,
    dataset_label_type: str,
) -> list[dict[str, Any]]:
    """
    Enforce the logical invariant:
      one membership row per image_id within a single dataset_id/version write.

    Behavior:
    - exact duplicate row for the same image_id -> collapse
    - conflicting row for the same image_id -> raise
    """
    deduped: list[dict[str, Any]] = []
    seen_by_image_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        image_id = row["image_id"]
        existing = seen_by_image_id.get(image_id)

        if existing is None:
            seen_by_image_id[image_id] = row
            deduped.append(row)
            continue

        if existing == row:
            continue

        raise ValueError(
            f"Conflicting duplicate membership row for image_id={image_id!r} "
            f"in dataset_label_type={dataset_label_type!r}. "
            f"First row={existing!r}; duplicate row={row!r}"
        )

    # Deterministic order for stable outputs / easier debugging
    deduped.sort(key=lambda r: str(r["image_id"]).strip())
    return deduped


def _run_sql_with_commit_retry(
    *,
    sql: str,
    query_name: str,
    athena_output_s3_uri: str,
    athena_workgroup: str,
    poll: Union[int, float] = 5,
    timeout: Union[int, float] = 1800,
    commit_retry_attempts: int = 4,
) -> None:
    last_exc: Exception | None = None

    for attempt in range(commit_retry_attempts):
        try:
            run_athena(
                sql,
                query_name,
                athena_output_s3_uri,
                athena_workgroup,
                poll,
                timeout,
            )
            return
        except Exception as e:
            last_exc = e
            retryable = _is_retryable_iceberg_commit_error(e)
            is_last_attempt = attempt >= (commit_retry_attempts - 1)

            if retryable and not is_last_attempt:
                continue

            raise

    if last_exc is not None:
        raise last_exc


def _delete_existing_dataset_version_rows(
    *,
    task_name: str,
    full_table: str,
    dataset_id: str,
    version: int,
    athena_output_s3_uri: str,
    athena_workgroup: str,
) -> None:
    safe_dataset_id = _escape_sql_string(dataset_id)

    delete_sql = (
        f"DELETE FROM {full_table} "
        f"WHERE dataset_id = '{safe_dataset_id}' AND version = {version}"
    )

    _run_sql_with_commit_retry(
        sql=delete_sql,
        query_name=f"{task_name} DELETE_DATASET_VERSION",
        athena_output_s3_uri=athena_output_s3_uri,
        athena_workgroup=athena_workgroup,
    )


def _insert_membership_rows_in_batches(
    *,
    rows: list[dict[str, Any]],
    schema: TableSchema,
    full_table: str,
    task_name: str,
    athena_output_s3_uri: str,
    athena_workgroup: str,
    chunk_size: int,
) -> None:
    if not (1 <= chunk_size <= 1000):
        raise ValueError(f"chunk_size must be 1..1000, got {chunk_size}")

    for chunk_no, start in enumerate(range(0, len(rows), chunk_size), start=1):
        batch = rows[start:start + chunk_size]
        insert_sql = build_insert_sql(batch, full_table, task_name, schema)

        _run_sql_with_commit_retry(
            sql=insert_sql,
            query_name=f"{task_name} INSERT_MEMBERSHIP_CHUNK_{chunk_no}",
            athena_output_s3_uri=athena_output_s3_uri,
            athena_workgroup=athena_workgroup,
        )


def write_dataset_membership(
    *,
    task_name: str,
    iceberg_database_name: str,
    athena_output_s3_uri: str,
    athena_workgroup: str,
    dataset_id: str,
    version: int,
    dataset_label_type: str,
    split_rows: list[dict[str, Any]],
    chunk_size: int = 500,
) -> dict[str, Any]:
    """
    Build and replace dataset membership rows for the target dataset/version
    in the correct Iceberg table.

    Important behavior:
    - builds canonical membership rows from split_rows
    - collapses exact duplicates by image_id and rejects conflicting duplicates
    - deletes the entire existing dataset/version slice before inserting
      the current canonical row set
    - this gives whole-version replacement semantics, which is safer for retries
    """
    table_name = _MEMBERSHIP_TABLE_BY_LABEL_TYPE.get(dataset_label_type)
    if not table_name:
        raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}")

    schema = _assert_membership_table_key_shape(table_name)

    membership_rows = build_membership_rows(
        dataset_id=dataset_id,
        version=version,
        dataset_label_type=dataset_label_type,
        split_rows=split_rows,
    )

    membership_rows = _dedupe_membership_rows_or_raise(
        membership_rows,
        dataset_label_type=dataset_label_type,
    )

    if not membership_rows:
        raise ValueError("No membership rows were built from split_rows.")

    full_table = f"\"{iceberg_database_name}\".\"{table_name}\""

    # Whole-version replacement semantics:
    # clear any existing rows for this dataset/version, then write the canonical set.
    _delete_existing_dataset_version_rows(
        task_name=task_name,
        full_table=full_table,
        dataset_id=dataset_id,
        version=version,
        athena_output_s3_uri=athena_output_s3_uri,
        athena_workgroup=athena_workgroup,
    )

    _insert_membership_rows_in_batches(
        rows=membership_rows,
        schema=schema,
        full_table=full_table,
        task_name=task_name,
        athena_output_s3_uri=athena_output_s3_uri,
        athena_workgroup=athena_workgroup,
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
        image_id = require_nonempty_string(row.get("image_id"), field_name="image_id")

        split = require_nonempty_string(row.get("split"), field_name="split")
        if split not in _VALID_SPLITS:
            raise ValueError(f"Invalid split: {split!r}")

        if dataset_label_type == "single-label":
            label = require_nonempty_string(row.get("label"), field_name="label")
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
            labels = normalize_string_array(
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
            bbox_annotation_ids = normalize_string_array(
                row.get("bbox_annotation_ids"),
                field_name="bbox_annotation_ids",
                require_nonempty=True,
            )
            classes_present = normalize_string_array(
                row.get("classes_present"),
                field_name="classes_present",
                require_nonempty=True,
            )
            out.append(
                {
                    "dataset_id": dataset_id,
                    "version": version,
                    "image_id": image_id,
                    "bbox_annotation_ids": bbox_annotation_ids,
                    "classes_present": classes_present,
                    "split": split,
                }
            )
            continue

        if dataset_label_type == "semantic-segmentation":
            semantic_mask_ids = normalize_string_array(
                row.get("semantic_mask_ids"),
                field_name="semantic_mask_ids",
                require_nonempty=True,
            )
            classes_present = normalize_string_array(
                row.get("classes_present"),
                field_name="classes_present",
                require_nonempty=True,
            )
            out.append(
                {
                    "dataset_id": dataset_id,
                    "version": version,
                    "image_id": image_id,
                    "semantic_mask_ids": semantic_mask_ids,
                    "classes_present": classes_present,
                    "split": split,
                }
            )
            continue

        if dataset_label_type == "instance-segmentation":
            instance_annotation_ids = normalize_string_array(
                row.get("instance_annotation_ids"),
                field_name="instance_annotation_ids",
                require_nonempty=True,
            )
            classes_present = normalize_string_array(
                row.get("classes_present"),
                field_name="classes_present",
                require_nonempty=True,
            )
            out.append(
                {
                    "dataset_id": dataset_id,
                    "version": version,
                    "image_id": image_id,
                    "instance_annotation_ids": instance_annotation_ids,
                    "classes_present": classes_present,
                    "split": split,
                }
            )
            continue

        raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}")

    return out