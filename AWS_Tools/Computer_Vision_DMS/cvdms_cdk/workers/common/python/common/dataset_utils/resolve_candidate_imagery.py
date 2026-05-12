from typing import Any, Union

from common.general_utils.athena_utils import (
    run_athena,
    athena_fetch_all_rows,
    parse_optional_string,
    parse_optional_float,
    parse_optional_int,
    parse_athena_array_string,
)

def resolve_sql(
    sql: str,
    task_name: str,
    athena_output_s3_uri: str,
    athena_workgroup: str,
    poll: Union[int, float] = 1.5,
    timeout: Union[int, float] = 900,
) -> list[dict[str, Any]]:
    """
    Execute the selection SQL in Athena and return raw result rows.
    """
    query_execution_id, _ = run_athena(
        sql,
        task_name,
        athena_output_s3_uri,
        athena_workgroup,
        poll=poll,
        timeout=timeout,
    )

    raw_rows = athena_fetch_all_rows(query_execution_id)
    return raw_rows

######################################################################################
# Main entrypoint
######################################################################################
def resolve_candidate_imagery(
    *,
    iceberg_database_name: str,
    label_type: str,
    selection_config: dict[str, Any],
    athena_output_s3_uri: str,
    athena_workgroup: str,
    task_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Resolve normalized candidate rows for dataset creation/update.

    Important provenance behavior:
    - source provenance now comes from image_source_membership, not canonical_imagery
    - source filtering is applied against image_source_membership.data_source using
      selection_config["allowed_sources"]
    - each candidate row includes:
        * data_sources: list[str]
        * source_splits_present: list[str] of distinct non-empty train/val/test values
        * resolved_source_split: str | None
        * source_split_status: "resolved" | "unresolved" | "inconsistent"
    """
    sql = build_selection_sql(
        iceberg_database_name=iceberg_database_name,
        dataset_label_type=label_type,
        selection_config=selection_config,
    )

    raw_rows = resolve_sql(
        sql,
        f"{task_name} RESOLVE CAND IMG SQL",
        athena_output_s3_uri,
        athena_workgroup,
    )

    allowed_classes = selection_config["allowed_classes"]
    candidates = [
        normalize_candidate_row(
            row,
            allowed_classes=allowed_classes,
        )
        for row in raw_rows
    ]

    return sql, candidates

######################################################################################
# Candidate normalization
######################################################################################
def normalize_candidate_row(
    row: dict[str, Any],
    *,
    allowed_classes: list[str],
) -> dict[str, Any]:
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
    - provenance:
        data_sources: list[str]
        source_splits_present: list[str] of non-empty train/val/test
        resolved_source_split: str | None
        source_split_status: "resolved" | "unresolved" | "inconsistent"
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
        "data_sources",
        "source_splits_present",
    }

    string_fields = {
        "image_id",
        "source_ref",
        "sha256_hash",
        "img_type",
        "dtype",
        "uploaded_at",
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
    normalized.setdefault("data_sources", [])
    normalized.setdefault("source_splits_present", [])

    normalized["data_sources"] = _normalize_string_array(
        normalized.get("data_sources", []),
    )
    normalized["source_splits_present"] = _normalize_source_split_array(
        normalized.get("source_splits_present", []),
    )

    # Compatibility shim for downstream code that still expects a scalar data_source.
    # Only set it when there is exactly one relevant source in scope.
    normalized["data_source"] = (
        normalized["data_sources"][0]
        if len(normalized["data_sources"]) == 1
        else None
    )

    if len(normalized["source_splits_present"]) == 1:
        normalized["resolved_source_split"] = normalized["source_splits_present"][0]
        normalized["source_split_status"] = "resolved"
    elif len(normalized["source_splits_present"]) == 0:
        normalized["resolved_source_split"] = None
        normalized["source_split_status"] = "unresolved"
    else:
        normalized["resolved_source_split"] = None
        normalized["source_split_status"] = "inconsistent"

    dataset_label_type = normalized.get("dataset_label_type")

    if dataset_label_type in {
        "object-detection",
        "semantic-segmentation",
        "instance-segmentation",
    }:
        normalized["classes_present"] = _filter_classes_present_to_allowed(
            normalized.get("classes_present", []),
            allowed_classes,
        )

        if len(normalized["classes_present"]) < 1:
            raise ValueError(
                f"{dataset_label_type} candidate must include non-empty filtered "
                f"'classes_present': {normalized!r}"
            )

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

def _filter_classes_present_to_allowed(
    values: list[str],
    allowed_classes: list[str],
) -> list[str]:
    allowed = set(str(v).strip().lower() for v in allowed_classes)
    out: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        if text not in allowed:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)

    return out

def _normalize_string_array(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)

    return sorted(out)

def _normalize_source_split_array(values: list[Any]) -> list[str]:
    valid = {"train", "val", "test"}
    out: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip().lower()
        if not text:
            continue
        if text not in valid:
            raise ValueError(f"Invalid non-empty source split from Athena result: {text!r}")
        if text in seen:
            continue
        seen.add(text)
        out.append(text)

    return sorted(out)

######################################################################################
# SQL builder
######################################################################################
def build_selection_sql(
    *,
    iceberg_database_name: str,
    dataset_label_type: str,
    selection_config: dict[str, Any],
) -> str:
    """
    Build Athena SQL that resolves candidate dataset membership rows from canonical tables.

    Important provenance behavior:
    - canonical_imagery no longer stores data_source
    - provenance now comes from image_source_membership
    - if allowed_sources is provided, provenance is scoped to only those membership rows
    - returned SQL always includes:
        data_sources: array<string>
        source_splits_present: array<string>
    """
    query_label_type = _map_dataset_label_type_to_query_label_type(dataset_label_type)
    common_filters = _build_common_filter_clauses(
        selection_config=selection_config,
        imagery_alias="ci",
    )

    if dataset_label_type == "single-label":
        base_candidate_select = _build_single_label_base_candidate_select(
            iceberg_database_name=iceberg_database_name,
            query_label_type=query_label_type,
            selection_config=selection_config,
            common_filters=common_filters,
        )
    elif dataset_label_type == "multi-label":
        base_candidate_select = _build_multi_label_base_candidate_select(
            iceberg_database_name=iceberg_database_name,
            query_label_type=query_label_type,
            selection_config=selection_config,
            common_filters=common_filters,
        )
    elif dataset_label_type == "object-detection":
        base_candidate_select = _build_object_detection_base_candidate_select(
            iceberg_database_name=iceberg_database_name,
            query_label_type=query_label_type,
            selection_config=selection_config,
            common_filters=common_filters,
        )
    elif dataset_label_type == "semantic-segmentation":
        base_candidate_select = _build_semantic_segmentation_base_candidate_select(
            iceberg_database_name=iceberg_database_name,
            query_label_type=query_label_type,
            selection_config=selection_config,
            common_filters=common_filters,
        )
    elif dataset_label_type == "instance-segmentation":
        base_candidate_select = _build_instance_segmentation_base_candidate_select(
            iceberg_database_name=iceberg_database_name,
            query_label_type=query_label_type,
            selection_config=selection_config,
            common_filters=common_filters,
        )
    else:
        raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}")

    source_membership_cte, source_membership_join_type = _build_source_membership_cte(
        iceberg_database_name=iceberg_database_name,
        selection_config=selection_config,
    )

    sql = f"""
WITH
{source_membership_cte},
base_candidates AS (
{base_candidate_select}
)
SELECT
    bc.*,
    COALESCE(sm.data_sources, CAST(ARRAY[] AS ARRAY(VARCHAR))) AS data_sources,
    COALESCE(sm.source_splits_present, CAST(ARRAY[] AS ARRAY(VARCHAR))) AS source_splits_present
FROM base_candidates bc
{source_membership_join_type} source_membership sm
    ON bc.image_id = sm.image_id
"""
    return sql.strip() + "\n"

def _map_dataset_label_type_to_query_label_type(dataset_label_type: str) -> str:
    """
    single-label and multi-label both intentionally read from canonical
    image_labels rows whose label_type is 'string-label'.
    """
    if dataset_label_type in {"single-label", "multi-label"}:
        return "string-label"
    return dataset_label_type

def _build_common_ci_select_list(*, dataset_label_type: str) -> str:
    return f"""
    ci.image_id,
    ci.source_ref,
    ci.sha256_hash,
    ci.img_type,
    ci.img_height,
    ci.img_width,
    ci.num_channels,
    ci.dtype,
    ci.file_size_mb,
    ci.uploaded_at,
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
    ci.color_bucket,
    '{_sql_escape_literal(dataset_label_type)}' AS dataset_label_type"""

def _build_common_ci_group_by_list() -> str:
    return """
    ci.image_id,
    ci.source_ref,
    ci.sha256_hash,
    ci.img_type,
    ci.img_height,
    ci.img_width,
    ci.num_channels,
    ci.dtype,
    ci.file_size_mb,
    ci.uploaded_at,
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
    ci.color_bucket"""

def _build_source_membership_cte(
    *,
    iceberg_database_name: str,
    selection_config: dict[str, Any],
) -> tuple[str, str]:
    allowed_sources = _get_allowed_sources(selection_config)

    where_sql = ""
    join_type = "LEFT JOIN"

    if allowed_sources:
        where_sql = f"\n    WHERE ism.data_source IN ({_sql_list(allowed_sources)})"
        join_type = "JOIN"

    cte_sql = f"""
source_membership AS (
    SELECT
        ism.image_id,
        ARRAY_SORT(ARRAY_DISTINCT(ARRAY_AGG(ism.data_source))) AS data_sources,
        ARRAY_SORT(
            FILTER(
                ARRAY_DISTINCT(
                    ARRAY_AGG(
                        CASE
                            WHEN TRIM(COALESCE(ism.source_split, '')) = '' THEN NULL
                            ELSE LOWER(TRIM(ism.source_split))
                        END
                    )
                ),
                x -> x IS NOT NULL
            )
        ) AS source_splits_present
    FROM {iceberg_database_name}.image_source_membership ism
{where_sql}
    GROUP BY ism.image_id
)""".strip()

    return cte_sql, join_type

def _build_single_label_base_candidate_select(
    *,
    iceberg_database_name: str,
    query_label_type: str,
    selection_config: dict[str, Any],
    common_filters: list[str],
) -> str:
    """
    Returns one row per image with exactly one distinct selected string label.
    Images with 0 or >1 distinct selected labels are excluded.
    """
    class_list_sql = _sql_list(selection_config["allowed_classes"])
    common_ci_cols = _build_common_ci_select_list(dataset_label_type="single-label")
    common_ci_group_by = _build_common_ci_group_by_list()
    where_sql = _join_where_clauses(
        [
            *common_filters,
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"il.label_id IN ({class_list_sql})",
        ]
    )

    sql = f"""
SELECT
{common_ci_cols},
    ei.label AS label,
    CAST(NULL AS array(varchar)) AS labels,
    ARRAY[ei.label] AS classes_present,
    CAST(NULL AS array(varchar)) AS bbox_annotation_ids,
    CAST(NULL AS array(varchar)) AS semantic_mask_ids,
    CAST(NULL AS array(varchar)) AS instance_annotation_ids
FROM {iceberg_database_name}.canonical_imagery ci
JOIN (
    SELECT
        image_id,
        MIN(label_id) AS label
    FROM (
        SELECT DISTINCT
            ci.image_id,
            il.label_id
        FROM {iceberg_database_name}.canonical_imagery ci
        JOIN {iceberg_database_name}.image_labels il
            ON ci.image_id = il.image_id
        {where_sql}
    ) filtered_links
    GROUP BY image_id
    HAVING COUNT(*) = 1
) ei
    ON ci.image_id = ei.image_id
GROUP BY
{common_ci_group_by},
    ei.label
"""
    return sql.strip()

def _build_multi_label_base_candidate_select(
    *,
    iceberg_database_name: str,
    query_label_type: str,
    selection_config: dict[str, Any],
    common_filters: list[str],
) -> str:
    """
    Returns one row per image with a deduped sorted array of selected string labels.
    """
    class_list_sql = _sql_list(selection_config["allowed_classes"])
    common_ci_cols = _build_common_ci_select_list(dataset_label_type="multi-label")
    common_ci_group_by = _build_common_ci_group_by_list()
    where_sql = _join_where_clauses(
        [
            *common_filters,
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"il.label_id IN ({class_list_sql})",
        ]
    )

    sql = f"""
SELECT
{common_ci_cols},
    CAST(NULL AS varchar) AS label,
    ARRAY_SORT(ARRAY_AGG(fl.label_id)) AS labels,
    ARRAY_SORT(ARRAY_AGG(fl.label_id)) AS classes_present,
    CAST(NULL AS array(varchar)) AS bbox_annotation_ids,
    CAST(NULL AS array(varchar)) AS semantic_mask_ids,
    CAST(NULL AS array(varchar)) AS instance_annotation_ids
FROM {iceberg_database_name}.canonical_imagery ci
JOIN (
    SELECT DISTINCT
        ci.image_id,
        il.label_id
    FROM {iceberg_database_name}.canonical_imagery ci
    JOIN {iceberg_database_name}.image_labels il
        ON ci.image_id = il.image_id
    {where_sql}
) fl
    ON ci.image_id = fl.image_id
GROUP BY
{common_ci_group_by}
"""
    return sql.strip()

def _build_object_detection_base_candidate_select(
    *,
    iceberg_database_name: str,
    query_label_type: str,
    selection_config: dict[str, Any],
    common_filters: list[str],
) -> str:
    class_list_sql = _sql_list(
        [str(v).strip().lower() for v in selection_config["allowed_classes"]]
    )
    common_ci_cols = _build_common_ci_select_list(dataset_label_type="object-detection")
    common_ci_group_by = _build_common_ci_group_by_list()
    where_sql = _join_where_clauses(
        [
            *common_filters,
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"LOWER(TRIM(CAST(t.class_name AS varchar))) IN ({class_list_sql})",
        ]
    )

    sql = f"""
SELECT
{common_ci_cols},
    CAST(NULL AS varchar) AS label,
    CAST(NULL AS array(varchar)) AS labels,
    ARRAY_SORT(ARRAY_DISTINCT(ARRAY_AGG(fl.class_name))) AS classes_present,
    ARRAY_SORT(ARRAY_DISTINCT(ARRAY_AGG(fl.bbox_annotation_id))) AS bbox_annotation_ids,
    CAST(NULL AS array(varchar)) AS semantic_mask_ids,
    CAST(NULL AS array(varchar)) AS instance_annotation_ids
FROM {iceberg_database_name}.canonical_imagery ci
JOIN (
    SELECT DISTINCT
        ci.image_id,
        bb.bbox_annotation_id,
        LOWER(TRIM(CAST(t.class_name AS varchar))) AS class_name
    FROM {iceberg_database_name}.canonical_imagery ci
    JOIN {iceberg_database_name}.image_labels il
        ON ci.image_id = il.image_id
    JOIN {iceberg_database_name}.canonical_bounding_boxes bb
        ON il.label_id = bb.bbox_annotation_id
    CROSS JOIN UNNEST(bb.classes_present) AS t(class_name)
    {where_sql}
) fl
    ON ci.image_id = fl.image_id
GROUP BY
{common_ci_group_by}
"""
    return sql.strip()

def _build_semantic_segmentation_base_candidate_select(
    *,
    iceberg_database_name: str,
    query_label_type: str,
    selection_config: dict[str, Any],
    common_filters: list[str],
) -> str:
    class_list_sql = _sql_list(
        [str(v).strip().lower() for v in selection_config["allowed_classes"]]
    )
    common_ci_cols = _build_common_ci_select_list(dataset_label_type="semantic-segmentation")
    common_ci_group_by = _build_common_ci_group_by_list()
    where_sql = _join_where_clauses(
        [
            *common_filters,
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"LOWER(TRIM(CAST(t.class_name AS varchar))) IN ({class_list_sql})",
        ]
    )

    sql = f"""
SELECT
{common_ci_cols},
    CAST(NULL AS varchar) AS label,
    CAST(NULL AS array(varchar)) AS labels,
    ARRAY_SORT(ARRAY_DISTINCT(ARRAY_AGG(fl.class_name))) AS classes_present,
    CAST(NULL AS array(varchar)) AS bbox_annotation_ids,
    ARRAY_SORT(ARRAY_DISTINCT(ARRAY_AGG(fl.semantic_mask_id))) AS semantic_mask_ids,
    CAST(NULL AS array(varchar)) AS instance_annotation_ids
FROM {iceberg_database_name}.canonical_imagery ci
JOIN (
    SELECT DISTINCT
        ci.image_id,
        sm.semantic_mask_id,
        LOWER(TRIM(CAST(t.class_name AS varchar))) AS class_name
    FROM {iceberg_database_name}.canonical_imagery ci
    JOIN {iceberg_database_name}.image_labels il
        ON ci.image_id = il.image_id
    JOIN {iceberg_database_name}.canonical_semantic_masks sm
        ON il.label_id = sm.semantic_mask_id
    CROSS JOIN UNNEST(sm.classes_present) AS t(class_name)
    {where_sql}
) fl
    ON ci.image_id = fl.image_id
GROUP BY
{common_ci_group_by}
"""
    return sql.strip()

def _build_instance_segmentation_base_candidate_select(
    *,
    iceberg_database_name: str,
    query_label_type: str,
    selection_config: dict[str, Any],
    common_filters: list[str],
) -> str:
    class_list_sql = _sql_list(
        [str(v).strip().lower() for v in selection_config["allowed_classes"]]
    )
    common_ci_cols = _build_common_ci_select_list(dataset_label_type="instance-segmentation")
    common_ci_group_by = _build_common_ci_group_by_list()
    where_sql = _join_where_clauses(
        [
            *common_filters,
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"LOWER(TRIM(CAST(t.class_name AS varchar))) IN ({class_list_sql})",
        ]
    )

    sql = f"""
SELECT
{common_ci_cols},
    CAST(NULL AS varchar) AS label,
    CAST(NULL AS array(varchar)) AS labels,
    ARRAY_SORT(ARRAY_DISTINCT(ARRAY_AGG(fl.class_name))) AS classes_present,
    CAST(NULL AS array(varchar)) AS bbox_annotation_ids,
    CAST(NULL AS array(varchar)) AS semantic_mask_ids,
    ARRAY_SORT(ARRAY_DISTINCT(ARRAY_AGG(fl.instance_annotation_id))) AS instance_annotation_ids
FROM {iceberg_database_name}.canonical_imagery ci
JOIN (
    SELECT DISTINCT
        ci.image_id,
        ia.instance_annotation_id,
        LOWER(TRIM(CAST(t.class_name AS varchar))) AS class_name
    FROM {iceberg_database_name}.canonical_imagery ci
    JOIN {iceberg_database_name}.image_labels il
        ON ci.image_id = il.image_id
    JOIN {iceberg_database_name}.canonical_instance_annotations ia
        ON il.label_id = ia.instance_annotation_id
    CROSS JOIN UNNEST(ia.classes_present) AS t(class_name)
    {where_sql}
) fl
    ON ci.image_id = fl.image_id
GROUP BY
{common_ci_group_by}
"""
    return sql.strip()

def _build_common_filter_clauses(
    *,
    selection_config: dict[str, Any],
    imagery_alias: str,
) -> list[str]:
    clauses: list[str] = []

    upload_date_range = selection_config.get("upload_date_range")
    if upload_date_range:
        start_date, end_date = upload_date_range
        clauses.append(
            f"DATE({imagery_alias}.uploaded_at) BETWEEN DATE '{_sql_escape_literal(start_date)}' "
            f"AND DATE '{_sql_escape_literal(end_date)}'"
        )

    width_range = selection_config.get("width_range")
    if width_range:
        min_width, max_width = width_range
        clauses.append(f"{imagery_alias}.img_width BETWEEN {min_width} AND {max_width}")

    height_range = selection_config.get("height_range")
    if height_range:
        min_height, max_height = height_range
        clauses.append(f"{imagery_alias}.img_height BETWEEN {min_height} AND {max_height}")

    lighting_buckets = selection_config.get("lighting_buckets")
    if lighting_buckets:
        clauses.append(f"{imagery_alias}.lighting_bucket IN ({_sql_list(lighting_buckets)})")

    blur_buckets = selection_config.get("blur_buckets")
    if blur_buckets:
        clauses.append(f"{imagery_alias}.blur_bucket IN ({_sql_list(blur_buckets)})")

    contrast_buckets = selection_config.get("contrast_buckets")
    if contrast_buckets:
        clauses.append(f"{imagery_alias}.contrast_bucket IN ({_sql_list(contrast_buckets)})")

    color_buckets = selection_config.get("color_buckets")
    if color_buckets:
        clauses.append(f"{imagery_alias}.color_bucket IN ({_sql_list(color_buckets)})")

    return clauses

def _get_allowed_sources(selection_config: dict[str, Any]) -> list[str]:
    values = selection_config.get("allowed_sources")
    if not values:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)

    return sorted(out)

def _join_where_clauses(clauses: list[str]) -> str:
    clauses = [c for c in clauses if c]
    if not clauses:
        return ""
    return "\nWHERE " + "\n  AND ".join(clauses)

def _sql_list(values: list[str]) -> str:
    return ", ".join(_sql_quote(v) for v in values)

def _sql_quote(value: str) -> str:
    return f"'{_sql_escape_literal(value)}'"

def _sql_escape_literal(value: str) -> str:
    return value.replace("'", "''")