from typing import Any

def build_selection_sql(
    *,
    iceberg_database_name: str,
    dataset_label_type: str,
    selection_config: dict[str, Any],
) -> str:
    """
    Build Athena SQL that resolves candidate dataset membership rows from canonical tables.

    Returns one row per image-label pair, including:
    - all analysis-relevant canonical_imagery fields
    - sha256_hash for leakage-aware grouping during split assignment
    - dataset_label_type
    - label_id
    - classes_present for structured-label tasks
    - task-specific artifact references (annotation_ref / mask_ref / mask_meta_ref)
    """
    query_label_type = _map_dataset_label_type_to_query_label_type(dataset_label_type)

    select_clause, from_clause, where_clauses = _build_base_query_parts(
        iceberg_database_name=iceberg_database_name,
        dataset_label_type=dataset_label_type,
        query_label_type=query_label_type,
        selection_config=selection_config,
    )

    common_filters = _build_common_filter_clauses(
        selection_config=selection_config,
        imagery_alias="ci",
    )
    where_clauses.extend(common_filters)

    where_sql = ""
    if where_clauses:
        where_sql = "\nWHERE " + "\n  AND ".join(where_clauses)

    sql = f"""{select_clause}
{from_clause}{where_sql}
"""
    return sql.strip() + "\n"

def _map_dataset_label_type_to_query_label_type(dataset_label_type: str) -> str:
    if dataset_label_type in {"single-label", "multi-label"}:
        return "string-label"
    return dataset_label_type

def _build_common_select_columns(*, dataset_label_type: str) -> str:
    """
    Common SELECT columns pulled from canonical_imagery plus common membership fields.
    """
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
    ci.data_source,
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
    '{_sql_escape_literal(dataset_label_type)}' AS dataset_label_type,
    il.label_id"""

def _build_base_query_parts(
    *,
    iceberg_database_name: str,
    dataset_label_type: str,
    query_label_type: str,
    selection_config: dict[str, Any],
) -> tuple[str, str, list[str]]:
    """
    Returns:
    - select_clause
    - from_clause
    - where_clauses
    """
    class_list_sql = _sql_list(selection_config["allowed_classes"])
    common_cols = _build_common_select_columns(dataset_label_type=dataset_label_type)

    if dataset_label_type in {"single-label", "multi-label"}:
        select_clause = f"""
SELECT
{common_cols},
    ARRAY[il.label_id] AS classes_present,
    CAST(NULL AS varchar) AS annotation_ref,
    CAST(NULL AS varchar) AS mask_ref,
    CAST(NULL AS varchar) AS mask_meta_ref
"""
        from_clause = f"""
FROM {iceberg_database_name}.canonical_imagery ci
JOIN {iceberg_database_name}.image_labels il
    ON ci.image_id = il.image_id
"""
        where_clauses = [
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"il.label_id IN ({class_list_sql})",
        ]
        return select_clause, from_clause, where_clauses

    if dataset_label_type == "object-detection":
        select_clause = f"""
SELECT
{common_cols},
    bb.classes_present,
    bb.source_ref_meta AS annotation_ref,
    CAST(NULL AS varchar) AS mask_ref,
    CAST(NULL AS varchar) AS mask_meta_ref
"""
        from_clause = f"""
FROM {iceberg_database_name}.canonical_imagery ci
JOIN {iceberg_database_name}.image_labels il
    ON ci.image_id = il.image_id
JOIN {iceberg_database_name}.canonical_bounding_boxes bb
    ON il.label_id = bb.bbox_annotation_id
"""
        where_clauses = [
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"EXISTS (SELECT 1 FROM UNNEST(bb.classes_present) AS t(class_name) WHERE class_name IN ({class_list_sql}))",
        ]
        return select_clause, from_clause, where_clauses

    if dataset_label_type == "semantic-segmentation":
        select_clause = f"""
SELECT
{common_cols},
    sm.classes_present,
    CAST(NULL AS varchar) AS annotation_ref,
    sm.source_ref_png AS mask_ref,
    sm.source_ref_meta AS mask_meta_ref
"""
        from_clause = f"""
FROM {iceberg_database_name}.canonical_imagery ci
JOIN {iceberg_database_name}.image_labels il
    ON ci.image_id = il.image_id
JOIN {iceberg_database_name}.canonical_semantic_masks sm
    ON il.label_id = sm.semantic_mask_id
"""
        where_clauses = [
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"EXISTS (SELECT 1 FROM UNNEST(sm.classes_present) AS t(class_name) WHERE class_name IN ({class_list_sql}))",
        ]
        return select_clause, from_clause, where_clauses

    if dataset_label_type == "instance-segmentation":
        select_clause = f"""
SELECT
{common_cols},
    ia.classes_present,
    CAST(NULL AS varchar) AS annotation_ref,
    ia.source_ref_png AS mask_ref,
    ia.source_ref_meta AS mask_meta_ref
"""
        from_clause = f"""
FROM {iceberg_database_name}.canonical_imagery ci
JOIN {iceberg_database_name}.image_labels il
    ON ci.image_id = il.image_id
JOIN {iceberg_database_name}.canonical_instance_annotations ia
    ON il.label_id = ia.instance_annotation_id
"""
        where_clauses = [
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"EXISTS (SELECT 1 FROM UNNEST(ia.classes_present) AS t(class_name) WHERE class_name IN ({class_list_sql}))",
        ]
        return select_clause, from_clause, where_clauses

    raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}")

def _build_common_filter_clauses(
    *,
    selection_config: dict[str, Any],
    imagery_alias: str,
) -> list[str]:
    clauses: list[str] = []

    allowed_sources = selection_config.get("allowed_sources")
    if allowed_sources:
        clauses.append(
            f"{imagery_alias}.data_source IN ({_sql_list(allowed_sources)})"
        )

    upload_date_range = selection_config.get("upload_date_range")
    if upload_date_range:
        start_date, end_date = upload_date_range
        clauses.append(
            f"DATE({imagery_alias}.uploaded_at) BETWEEN DATE '{_sql_escape_literal(start_date)}' AND DATE '{_sql_escape_literal(end_date)}'"
        )

    width_range = selection_config.get("width_range")
    if width_range:
        min_width, max_width = width_range
        clauses.append(
            f"{imagery_alias}.img_width BETWEEN {min_width} AND {max_width}"
        )

    height_range = selection_config.get("height_range")
    if height_range:
        min_height, max_height = height_range
        clauses.append(
            f"{imagery_alias}.img_height BETWEEN {min_height} AND {max_height}"
        )

    lighting_buckets = selection_config.get("lighting_buckets")
    if lighting_buckets:
        clauses.append(
            f"{imagery_alias}.lighting_bucket IN ({_sql_list(lighting_buckets)})"
        )

    blur_buckets = selection_config.get("blur_buckets")
    if blur_buckets:
        clauses.append(
            f"{imagery_alias}.blur_bucket IN ({_sql_list(blur_buckets)})"
        )

    contrast_buckets = selection_config.get("contrast_buckets")
    if contrast_buckets:
        clauses.append(
            f"{imagery_alias}.contrast_bucket IN ({_sql_list(contrast_buckets)})"
        )

    color_buckets = selection_config.get("color_buckets")
    if color_buckets:
        clauses.append(
            f"{imagery_alias}.color_bucket IN ({_sql_list(color_buckets)})"
        )

    return clauses

def _sql_list(values: list[str]) -> str:
    return ", ".join(_sql_quote(v) for v in values)

def _sql_quote(value: str) -> str:
    return f"'{_sql_escape_literal(value)}'"

def _sql_escape_literal(value: str) -> str:
    return value.replace("'", "''")