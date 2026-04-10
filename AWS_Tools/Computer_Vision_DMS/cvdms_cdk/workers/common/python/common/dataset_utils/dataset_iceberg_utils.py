from typing import Any

from common.general_utils.iceberg_utils import normalize_string_array, require_nonempty_string, chunked_insert

_MEMBERSHIP_TABLE_BY_LABEL_TYPE: dict[str, str] = {
    "single-label": "single_label",
    "multi-label": "multi_label",
    "object-detection": "object_detection",
    "semantic-segmentation": "semantic_segmentation",
    "instance-segmentation": "instance_segmentation"
}

_VALID_SPLITS = {"train", "val", "test"}

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

    table_name = _MEMBERSHIP_TABLE_BY_LABEL_TYPE.get(dataset_label_type)
    if not table_name:
        raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}")

    ok, err = chunked_insert(
        rows=membership_rows,
        task_name=task_name,
        iceberg_db_name=iceberg_database_name,
        table_name=table_name,
        athena_workgroup=athena_workgroup,
        athena_output_s3=athena_output_s3_uri,
        chunk_size=chunk_size,
        allow_empty=False,
    )

    if not ok:
        raise RuntimeError(f"{task_name} failed writing dataset membership rows: {err}")

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