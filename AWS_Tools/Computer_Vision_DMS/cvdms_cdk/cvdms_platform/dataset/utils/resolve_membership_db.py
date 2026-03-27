from typing import Any, Literal
from mypy_boto3_athena.client import AthenaClient

from cvdms_platform.dataset.utils.athena_utils import resolve_sql

def resolve_dataset_membership(
    *,
    iceberg_database_name: str,
    dataset_membership_table_name: str,
    canonical_imagery_table_name: str,
    dataset_id: str,
    version: int,
    mode: Literal["minimal", "enriched"],
    athena_client: AthenaClient,
    athena_output_s3_uri: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build SQL for current dataset-version membership and return normalized rows.

    Modes:
    - minimal:
        returns only the fields needed to preserve membership/splits
    - enriched:
        joins to canonical imagery and returns the fields needed for
        split recomputation / rebalance flows
    """
    if mode not in {"minimal", "enriched"}:
        raise ValueError(f"Unsupported mode: {mode!r}")

    if not dataset_id or not str(dataset_id).strip():
        raise ValueError("dataset_id must be a non-empty string.")

    if type(version) is not int or version < 1:
        raise ValueError("version must be an integer >= 1.")

    membership_sql = build_dataset_membership_sql(iceberg_database_name=iceberg_database_name,
                                                    dataset_membership_table_name=dataset_membership_table_name,
                                                    canonical_imagery_table_name=canonical_imagery_table_name,
                                                    dataset_id=dataset_id,
                                                    version=version,
                                                    mode=mode)

    raw_rows = resolve_sql(athena_client=athena_client,
                           iceberg_database_name=iceberg_database_name,
                           athena_output_s3_uri=athena_output_s3_uri,
                           selection_sql=membership_sql)

    normalized_rows = [normalize_membership_row(row=row, mode=mode) for row in raw_rows]

    return membership_sql, normalized_rows

def normalize_membership_row(
    *,
    row: dict[str, Any],
    mode: Literal["minimal", "enriched"],
) -> dict[str, Any]:
    """
    Normalize Athena string-valued result cells into expected Python types and
    enforce the membership-row contract.
    """
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

    normalized: dict[str, Any] = {}

    for key, value in row.items():
        if value in ("", None):
            normalized[key] = None
            continue

        if key in int_fields:
            normalized[key] = int(value)
        elif key in float_fields:
            normalized[key] = float(value)
        elif key == "string_label_ids":
            normalized[key] = _normalize_array_field(value)
        else:
            normalized[key] = value

    _require_nonempty_string(normalized.get("dataset_id"), "dataset_id")
    _require_positive_int(normalized.get("version"), "version")
    _require_nonempty_string(normalized.get("image_id"), "image_id")
    _require_valid_split(normalized.get("split"))

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

def build_dataset_membership_sql(
    *,
    iceberg_database_name: str,
    dataset_membership_table_name: str,
    canonical_imagery_table_name: str,
    dataset_id: str,
    version: int,
    mode: Literal["minimal", "enriched"],
) -> str:
    dataset_id_sql = _sql_quote(dataset_id)

    if mode == "minimal":
        return f"""
SELECT
    m.dataset_id,
    m.version,
    m.image_id,
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

def _normalize_array_field(value: Any) -> list[str]:
    """
    Normalize Athena array output into a Python list[str].

    This is intentionally conservative. If your shared Athena row resolver
    already parses arrays into lists, this will preserve them. If it returns
    strings like '[a, b]', this will parse them.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        cleaned = []
        for p in parts:
            p = p.strip().strip('"').strip("'")
            if p:
                cleaned.append(p)
        return cleaned

    return [text]

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
    """
    Basic SQL string escaping for Athena string literals.
    """
    return "'" + value.replace("'", "''") + "'"