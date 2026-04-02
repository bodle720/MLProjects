from typing import Any, Literal, Union

from common.general_utils.athena_utils import (
    run_athena,
    athena_fetch_all_rows,
    parse_optional_int,
    parse_optional_float,
    parse_athena_array_string,
    parse_optional_string,
)

DatasetLabelType = Literal[
    "single-label",
    "multi-label",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
]

MembershipMode = Literal["minimal", "enriched"]

LABEL_TYPE_TO_MEMBERSHIP_FIELD: dict[DatasetLabelType, str] = {
    "single-label": "label",
    "multi-label": "labels",
    "object-detection": "bbox_annotation_ids",
    "semantic-segmentation": "semantic_mask_ids",
    "instance-segmentation": "instance_annotation_ids",
}

_LABEL_TYPES_WITH_CLASSES_PRESENT: set[DatasetLabelType] = {
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
}

def resolve_sql(sql: str,
                task_name: str,
                athena_output_s3_uri: str,
                athena_workgroup: str,
                poll: Union[int, float] = 1.5,
                timeout: Union[int, float] = 900) -> list[dict[str, Any]]:
    """
    Execute the selection SQL in Athena and return normalized membership rows.
    """
    query_execution_id, _ = run_athena(
        sql,
        task_name,
        athena_output_s3_uri,
        athena_workgroup,
        poll=poll,
        timeout=timeout
    )

    raw_rows = athena_fetch_all_rows(query_execution_id)

    return raw_rows

def resolve_dataset_membership(*,
                                iceberg_database_name: str,
                                dataset_membership_table_name: str,
                                canonical_imagery_table_name: str,
                                dataset_id: str,
                                version: int,
                                label_type: DatasetLabelType,
                                mode: MembershipMode,
                                athena_output_s3_uri: str,
                                athena_workgroup: str,
                                task_name: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Build SQL for current dataset-version membership and return normalized rows.

    Modes:
    - minimal:
        returns the fields needed to preserve membership rows directly
        during maintain flows:
            dataset_id, version, image_id, split, dataset_label_type,
            <label payload field>, and classes_present for the 3 structured task types

    - enriched:
        returns the same membership payload fields plus canonical imagery
        features needed for split recomputation / rebalance flows
    """
    if mode not in {"minimal", "enriched"}:
        raise ValueError(f"Unsupported mode: {mode!r}")

    if label_type not in LABEL_TYPE_TO_MEMBERSHIP_FIELD:
        raise ValueError(f"Unsupported label_type: {label_type!r}")

    if not dataset_id or not str(dataset_id).strip():
        raise ValueError("dataset_id must be a non-empty string.")

    if type(version) is not int or version < 1:
        raise ValueError("version must be an integer >= 1.")

    membership_sql = build_dataset_membership_sql(iceberg_database_name=iceberg_database_name,
                                                    dataset_membership_table_name=dataset_membership_table_name,
                                                    canonical_imagery_table_name=canonical_imagery_table_name,
                                                    dataset_id=dataset_id,
                                                    version=version,
                                                    label_type=label_type,
                                                    mode=mode)

    raw_rows = resolve_sql(membership_sql,
                           f"{task_name} RESOLVE DS MEMBERSHIP SQL",
                           athena_output_s3_uri,
                           athena_workgroup)

    normalized_rows = [
        normalize_membership_row(
            row=row,
            label_type=label_type,
            mode=mode
        )
        for row in raw_rows
    ]

    return membership_sql, normalized_rows

def normalize_membership_row(
    *,
    row: dict[str, Any],
    label_type: DatasetLabelType,
    mode: MembershipMode,
) -> dict[str, Any]:
    """
    Normalize Athena string-valued result cells into expected Python types and
    enforce the membership-row contract.
    """
    membership_field = LABEL_TYPE_TO_MEMBERSHIP_FIELD[label_type]

    int_fields = {
        "version",
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

    array_fields = {
        "labels",
        "bbox_annotation_ids",
        "semantic_mask_ids",
        "instance_annotation_ids",
        "classes_present",
    }

    normalized: dict[str, Any] = {}

    for key, value in row.items():
        if key in int_fields:
            normalized[key] = parse_optional_int(value, field_name=key)
        elif key in float_fields:
            normalized[key] = parse_optional_float(value, field_name=key)
        elif key in array_fields:
            normalized[key] = parse_athena_array_string(value, field_name=key)
        else:
            normalized[key] = parse_optional_string(value)

    _require_nonempty_string(normalized.get("dataset_id"), "dataset_id")
    _require_positive_int(normalized.get("version"), "version")
    _require_nonempty_string(normalized.get("image_id"), "image_id")
    _require_nonempty_string(normalized.get("dataset_label_type"), "dataset_label_type")
    _require_valid_split(normalized.get("split"))

    if normalized["dataset_label_type"] != label_type:
        raise ValueError(
            f"dataset_label_type mismatch: expected {label_type!r}, "
            f"got {normalized['dataset_label_type']!r}"
        )

    _require_membership_payload(
        value=normalized.get(membership_field),
        field_name=membership_field,
        label_type=label_type,
    )

    if label_type in _LABEL_TYPES_WITH_CLASSES_PRESENT:
        _require_nonempty_string_array(
            normalized.get("classes_present"),
            field_name="classes_present",
        )

    if mode == "enriched":
        _require_nonempty_string(normalized.get("source_ref"), "source_ref")
        _require_nonempty_string(normalized.get("sha256_hash"), "sha256_hash")
        _require_nonempty_string(normalized.get("data_source"), "data_source")
        _require_nonempty_string(normalized.get("lighting_bucket"), "lighting_bucket")
        _require_nonempty_string(normalized.get("blur_bucket"), "blur_bucket")
        _require_nonempty_string(normalized.get("contrast_bucket"), "contrast_bucket")
        _require_nonempty_string(normalized.get("color_bucket"), "color_bucket")

        if normalized.get("img_height") is None:
            raise ValueError("Enriched membership row missing img_height.")
        if normalized.get("img_width") is None:
            raise ValueError("Enriched membership row missing img_width.")

    return normalized

def build_dataset_membership_sql(*,
                                iceberg_database_name: str,
                                dataset_membership_table_name: str,
                                canonical_imagery_table_name: str,
                                dataset_id: str,
                                version: int,
                                label_type: DatasetLabelType,
                                mode: MembershipMode) -> str:

    dataset_id_sql = _sql_quote(dataset_id)
    membership_field = LABEL_TYPE_TO_MEMBERSHIP_FIELD[label_type]
    include_classes_present = label_type in _LABEL_TYPES_WITH_CLASSES_PRESENT
    dataset_label_type_sql = _sql_quote(label_type)

    classes_present_select = ",\n    m.classes_present" if include_classes_present else ""

    if mode == "minimal":
        return f"""
SELECT
    m.dataset_id,
    m.version,
    m.image_id,
    {dataset_label_type_sql} AS dataset_label_type,
    m.{membership_field}{classes_present_select},
    m.split
FROM "{iceberg_database_name}"."{dataset_membership_table_name}" AS m
WHERE m.dataset_id = {dataset_id_sql}
  AND m.version = {version}
ORDER BY m.image_id
""".strip()

    return f"""
SELECT
    m.dataset_id,
    m.version,
    m.image_id,
    {dataset_label_type_sql} AS dataset_label_type,
    m.{membership_field}{classes_present_select},
    m.split,

    ci.source_ref,
    ci.img_type,
    ci.img_height,
    ci.img_width,
    ci.num_channels,
    ci.dtype,
    ci.file_size_mb,
    ci.uploaded_at,
    ci.data_source,
    ci.sha256_hash,

    ci.luma_mean,
    ci.luma_p10,
    ci.luma_p90,
    ci.dark_frac,
    ci.bright_frac,
    ci.contrast_luma_std,
    ci.contrast_luma_p90_p10,
    ci.blur_laplacian_var,
    ci.sat_mean,
    ci.colorfulness,

    ci.lighting_bucket,
    ci.blur_bucket,
    ci.contrast_bucket,
    ci.color_bucket

FROM "{iceberg_database_name}"."{dataset_membership_table_name}" AS m
INNER JOIN "{iceberg_database_name}"."{canonical_imagery_table_name}" AS ci
    ON m.image_id = ci.image_id
WHERE m.dataset_id = {dataset_id_sql}
  AND m.version = {version}
ORDER BY m.image_id
""".strip()

def _require_membership_payload(
    *,
    value: Any,
    field_name: str,
    label_type: DatasetLabelType,
) -> None:
    if label_type == "single-label":
        _require_nonempty_string(value, field_name)
        return

    _require_nonempty_string_array(value, field_name=field_name)

def _require_nonempty_string_array(value: Any, *, field_name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Field '{field_name}' must be a non-empty list[str].")

    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Field '{field_name}' contains an invalid string value.")

def _require_nonempty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")

def _require_positive_int(value: Any, field_name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"Field '{field_name}' must be an integer >= 1.")

def _require_valid_split(value: Any) -> None:
    if value not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split value: {value!r}")

def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"