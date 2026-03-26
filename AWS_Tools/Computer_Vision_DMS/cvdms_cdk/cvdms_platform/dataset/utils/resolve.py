from typing import Any

from cvdms_platform.dataset.utils.athena_utils import (
    start_athena_query,
    wait_for_athena_query,
    fetch_athena_results,
    parse_optional_string,
    parse_optional_float,
    parse_optional_int,
    parse_athena_array_string
)

def resolve_candidates(
    *,
    athena_client: Any,
    iceberg_database_name: str,
    athena_output_s3_uri: str,
    selection_sql: str
) -> list[dict[str, Any]]:
    """
    Execute the selection SQL in Athena and return normalized candidate rows.
    """
    query_execution_id = start_athena_query(
        athena_client=athena_client,
        iceberg_database_name=iceberg_database_name,
        athena_output_s3_uri=athena_output_s3_uri,
        selection_sql=selection_sql
    )
    wait_for_athena_query(
        athena_client=athena_client,
        query_execution_id=query_execution_id
    )
    raw_rows = fetch_athena_results(
        athena_client=athena_client,
        query_execution_id=query_execution_id
    )
    return [normalize_candidate_row(row) for row in raw_rows]

def normalize_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize Athena result cells into expected Python types for dataset candidates.

    Expected conventions from selection SQL:
    - single-label:
        label: str | None
        labels: list[str] == []
    - multi-label:
        label: None
        labels: list[str]
    - object-detection:
        bbox_annotation_ids: list[str]
    - semantic-segmentation:
        semantic_mask_ids: list[str]
    - instance-segmentation:
        instance_annotation_ids: list[str]
    - classes_present:
        list[str] across all task types
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

    array_fields = {
        "classes_present",
        "labels",
        "bbox_annotation_ids",
        "semantic_mask_ids",
        "instance_annotation_ids",
    }

    string_fields = {
        "image_id",
        "source_ref",
        "sha256_hash",
        "img_type",
        "dtype",
        "uploaded_at",
        "data_source",
        "lighting_bucket",
        "blur_bucket",
        "contrast_bucket",
        "color_bucket",
        "dataset_label_type",
        "label",
    }

    normalized: dict[str, Any] = dict(row)

    for field in string_fields:
        if field in normalized:
            normalized[field] = parse_optional_string(normalized.get(field))

    for field in int_fields:
        if field in normalized:
            normalized[field] = parse_optional_int(
                normalized.get(field),
                field_name=field,
            )

    for field in float_fields:
        if field in normalized:
            normalized[field] = parse_optional_float(
                normalized.get(field),
                field_name=field,
            )

    for field in array_fields:
        if field in normalized:
            normalized[field] = parse_athena_array_string(
                normalized.get(field),
                field_name=field,
            )

    normalized.setdefault("label", None)
    normalized.setdefault("labels", [])
    normalized.setdefault("classes_present", [])
    normalized.setdefault("bbox_annotation_ids", [])
    normalized.setdefault("semantic_mask_ids", [])
    normalized.setdefault("instance_annotation_ids", [])

    dataset_label_type = normalized.get("dataset_label_type")

    if dataset_label_type == "single-label":
        if not normalized["label"]:
            raise ValueError(
                f"single-label candidate must include non-empty 'label': {normalized!r}"
            )
        if len(normalized["classes_present"]) != 1:
            raise ValueError(
                "single-label candidate must have classes_present of length 1: "
                f"{normalized!r}"
            )

    elif dataset_label_type == "multi-label":
        if normalized["label"] is not None:
            raise ValueError(
                f"multi-label candidate should not include scalar 'label': {normalized!r}"
            )
        if len(normalized["labels"]) < 1:
            raise ValueError(
                f"multi-label candidate must include non-empty 'labels': {normalized!r}"
            )

    elif dataset_label_type == "object-detection":
        if len(normalized["bbox_annotation_ids"]) < 1:
            raise ValueError(
                "object-detection candidate must include non-empty "
                f"'bbox_annotation_ids': {normalized!r}"
            )

    elif dataset_label_type == "semantic-segmentation":
        if len(normalized["semantic_mask_ids"]) < 1:
            raise ValueError(
                "semantic-segmentation candidate must include non-empty "
                f"'semantic_mask_ids': {normalized!r}"
            )

    elif dataset_label_type == "instance-segmentation":
        if len(normalized["instance_annotation_ids"]) < 1:
            raise ValueError(
                "instance-segmentation candidate must include non-empty "
                f"'instance_annotation_ids': {normalized!r}"
            )

    return normalized