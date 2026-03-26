from typing import Any

def build_selection_sql(
    *,
    iceberg_database_name: str,
    dataset_label_type: str,
    selection_config: dict[str, Any],
) -> str:
    """
    Build Athena SQL that resolves candidate dataset membership rows from canonical tables.

    New conventions:
    - returns exactly one candidate row per image
    - deduplicates repeated image_labels links
    - single-label:
        * label: varchar
        * classes_present: array<string> of length 1
    - multi-label:
        * labels: array<string>
        * classes_present: same deduped label array
    - object-detection / semantic-segmentation / instance-segmentation:
        * task-specific *_ids array<string>
        * classes_present = deduped union of classes across linked artifacts
    """
    query_label_type = _map_dataset_label_type_to_query_label_type(dataset_label_type)
    common_filters = _build_common_filter_clauses(
        selection_config=selection_config,
        imagery_alias="ci",
    )

    if dataset_label_type == "single-label":
        return _build_single_label_sql(
            iceberg_database_name=iceberg_database_name,
            query_label_type=query_label_type,
            selection_config=selection_config,
            common_filters=common_filters,
        )

    if dataset_label_type == "multi-label":
        return _build_multi_label_sql(
            iceberg_database_name=iceberg_database_name,
            query_label_type=query_label_type,
            selection_config=selection_config,
            common_filters=common_filters,
        )

    if dataset_label_type == "object-detection":
        return _build_object_detection_sql(
            iceberg_database_name=iceberg_database_name,
            query_label_type=query_label_type,
            selection_config=selection_config,
            common_filters=common_filters,
        )

    if dataset_label_type == "semantic-segmentation":
        return _build_semantic_segmentation_sql(
            iceberg_database_name=iceberg_database_name,
            query_label_type=query_label_type,
            selection_config=selection_config,
            common_filters=common_filters,
        )

    if dataset_label_type == "instance-segmentation":
        return _build_instance_segmentation_sql(
            iceberg_database_name=iceberg_database_name,
            query_label_type=query_label_type,
            selection_config=selection_config,
            common_filters=common_filters,
        )

    raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}")

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
    '{_sql_escape_literal(dataset_label_type)}' AS dataset_label_type"""

def _build_common_ci_group_by_list() -> str:
    """
    GROUP BY list for all canonical_imagery fields selected above.
    """
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
    ci.color_bucket"""

def _build_single_label_sql(
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
WITH filtered_links AS (
    SELECT DISTINCT
        ci.image_id,
        il.label_id
    FROM {iceberg_database_name}.canonical_imagery ci
    JOIN {iceberg_database_name}.image_labels il
        ON ci.image_id = il.image_id
    {where_sql}
),
eligible_images AS (
    SELECT
        image_id,
        MIN(label_id) AS label
    FROM filtered_links
    GROUP BY image_id
    HAVING COUNT(*) = 1
)
SELECT
{common_ci_cols},
    ei.label AS label,
    CAST(NULL AS array(varchar)) AS labels,
    ARRAY[ei.label] AS classes_present,
    CAST(NULL AS array(varchar)) AS bbox_annotation_ids,
    CAST(NULL AS array(varchar)) AS semantic_mask_ids,
    CAST(NULL AS array(varchar)) AS instance_annotation_ids
FROM {iceberg_database_name}.canonical_imagery ci
JOIN eligible_images ei
    ON ci.image_id = ei.image_id
GROUP BY
{common_ci_group_by},
    ei.label
"""
    return sql.strip() + "\n"

def _build_multi_label_sql(
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
WITH filtered_links AS (
    SELECT DISTINCT
        ci.image_id,
        il.label_id
    FROM {iceberg_database_name}.canonical_imagery ci
    JOIN {iceberg_database_name}.image_labels il
        ON ci.image_id = il.image_id
    {where_sql}
)
SELECT
{common_ci_cols},
    CAST(NULL AS varchar) AS label,
    ARRAY_SORT(ARRAY_AGG(fl.label_id)) AS labels,
    ARRAY_SORT(ARRAY_AGG(fl.label_id)) AS classes_present,
    CAST(NULL AS array(varchar)) AS bbox_annotation_ids,
    CAST(NULL AS array(varchar)) AS semantic_mask_ids,
    CAST(NULL AS array(varchar)) AS instance_annotation_ids
FROM {iceberg_database_name}.canonical_imagery ci
JOIN filtered_links fl
    ON ci.image_id = fl.image_id
GROUP BY
{common_ci_group_by}
"""
    return sql.strip() + "\n"

def _build_object_detection_sql(
    *,
    iceberg_database_name: str,
    query_label_type: str,
    selection_config: dict[str, Any],
    common_filters: list[str],
) -> str:
    class_list_sql = _sql_list(selection_config["allowed_classes"])
    common_ci_cols = _build_common_ci_select_list(dataset_label_type="object-detection")
    common_ci_group_by = _build_common_ci_group_by_list()
    where_sql = _join_where_clauses(
        [
            *common_filters,
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"EXISTS (SELECT 1 FROM UNNEST(bb.classes_present) AS t(class_name) WHERE class_name IN ({class_list_sql}))",
        ]
    )

    sql = f"""
WITH filtered_links AS (
    SELECT DISTINCT
        ci.image_id,
        bb.bbox_annotation_id,
        bb.classes_present
    FROM {iceberg_database_name}.canonical_imagery ci
    JOIN {iceberg_database_name}.image_labels il
        ON ci.image_id = il.image_id
    JOIN {iceberg_database_name}.canonical_bounding_boxes bb
        ON il.label_id = bb.bbox_annotation_id
    {where_sql}
)
SELECT
{common_ci_cols},
    CAST(NULL AS varchar) AS label,
    CAST(NULL AS array(varchar)) AS labels,
    ARRAY_SORT(ARRAY_DISTINCT(FLATTEN(ARRAY_AGG(fl.classes_present)))) AS classes_present,
    ARRAY_SORT(ARRAY_AGG(fl.bbox_annotation_id)) AS bbox_annotation_ids,
    CAST(NULL AS array(varchar)) AS semantic_mask_ids,
    CAST(NULL AS array(varchar)) AS instance_annotation_ids
FROM {iceberg_database_name}.canonical_imagery ci
JOIN filtered_links fl
    ON ci.image_id = fl.image_id
GROUP BY
{common_ci_group_by}
"""
    return sql.strip() + "\n"

def _build_semantic_segmentation_sql(
    *,
    iceberg_database_name: str,
    query_label_type: str,
    selection_config: dict[str, Any],
    common_filters: list[str],
) -> str:
    class_list_sql = _sql_list(selection_config["allowed_classes"])
    common_ci_cols = _build_common_ci_select_list(dataset_label_type="semantic-segmentation")
    common_ci_group_by = _build_common_ci_group_by_list()
    where_sql = _join_where_clauses(
        [
            *common_filters,
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"EXISTS (SELECT 1 FROM UNNEST(sm.classes_present) AS t(class_name) WHERE class_name IN ({class_list_sql}))",
        ]
    )

    sql = f"""
WITH filtered_links AS (
    SELECT DISTINCT
        ci.image_id,
        sm.semantic_mask_id,
        sm.classes_present
    FROM {iceberg_database_name}.canonical_imagery ci
    JOIN {iceberg_database_name}.image_labels il
        ON ci.image_id = il.image_id
    JOIN {iceberg_database_name}.canonical_semantic_masks sm
        ON il.label_id = sm.semantic_mask_id
    {where_sql}
)
SELECT
{common_ci_cols},
    CAST(NULL AS varchar) AS label,
    CAST(NULL AS array(varchar)) AS labels,
    ARRAY_SORT(ARRAY_DISTINCT(FLATTEN(ARRAY_AGG(fl.classes_present)))) AS classes_present,
    CAST(NULL AS array(varchar)) AS bbox_annotation_ids,
    ARRAY_SORT(ARRAY_AGG(fl.semantic_mask_id)) AS semantic_mask_ids,
    CAST(NULL AS array(varchar)) AS instance_annotation_ids
FROM {iceberg_database_name}.canonical_imagery ci
JOIN filtered_links fl
    ON ci.image_id = fl.image_id
GROUP BY
{common_ci_group_by}
"""
    return sql.strip() + "\n"

def _build_instance_segmentation_sql(
    *,
    iceberg_database_name: str,
    query_label_type: str,
    selection_config: dict[str, Any],
    common_filters: list[str],
) -> str:
    class_list_sql = _sql_list(selection_config["allowed_classes"])
    common_ci_cols = _build_common_ci_select_list(dataset_label_type="instance-segmentation")
    common_ci_group_by = _build_common_ci_group_by_list()
    where_sql = _join_where_clauses(
        [
            *common_filters,
            f"il.label_type = '{_sql_escape_literal(query_label_type)}'",
            f"EXISTS (SELECT 1 FROM UNNEST(ia.classes_present) AS t(class_name) WHERE class_name IN ({class_list_sql}))",
        ]
    )

    sql = f"""
WITH filtered_links AS (
    SELECT DISTINCT
        ci.image_id,
        ia.instance_annotation_id,
        ia.classes_present
    FROM {iceberg_database_name}.canonical_imagery ci
    JOIN {iceberg_database_name}.image_labels il
        ON ci.image_id = il.image_id
    JOIN {iceberg_database_name}.canonical_instance_annotations ia
        ON il.label_id = ia.instance_annotation_id
    {where_sql}
)
SELECT
{common_ci_cols},
    CAST(NULL AS varchar) AS label,
    CAST(NULL AS array(varchar)) AS labels,
    ARRAY_SORT(ARRAY_DISTINCT(FLATTEN(ARRAY_AGG(fl.classes_present)))) AS classes_present,
    CAST(NULL AS array(varchar)) AS bbox_annotation_ids,
    CAST(NULL AS array(varchar)) AS semantic_mask_ids,
    ARRAY_SORT(ARRAY_AGG(fl.instance_annotation_id)) AS instance_annotation_ids
FROM {iceberg_database_name}.canonical_imagery ci
JOIN filtered_links fl
    ON ci.image_id = fl.image_id
GROUP BY
{common_ci_group_by}
"""
    return sql.strip() + "\n"

def _build_common_filter_clauses(
    *,
    selection_config: dict[str, Any],
    imagery_alias: str,
) -> list[str]:
    clauses: list[str] = []

    allowed_sources = selection_config.get("allowed_sources")
    if allowed_sources:
        clauses.append(f"{imagery_alias}.data_source IN ({_sql_list(allowed_sources)})")

    upload_date_range = selection_config.get("upload_date_range")
    if upload_date_range:
        start_date, end_date = upload_date_range
        clauses.append(
            f"DATE({imagery_alias}.uploaded_at) BETWEEN DATE '{_sql_escape_literal(start_date)}' AND DATE '{_sql_escape_literal(end_date)}'"
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