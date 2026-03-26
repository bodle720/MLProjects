import time
from typing import Any

def resolve_candidates(
    *,
    athena_client: Any,
    iceberg_database_name: str,
    athena_output_s3_uri: str,
    selection_sql: str,
) -> list[dict[str, Any]]:
    """
    Execute the selection SQL in Athena and return normalized candidate rows.
    """
    query_execution_id = start_athena_query(
        athena_client=athena_client,
        iceberg_database_name=iceberg_database_name,
        athena_output_s3_uri=athena_output_s3_uri,
        selection_sql=selection_sql,
    )
    wait_for_athena_query(
        athena_client=athena_client,
        query_execution_id=query_execution_id,
    )
    raw_rows = fetch_athena_results(
        athena_client=athena_client,
        query_execution_id=query_execution_id,
    )
    return [normalize_candidate_row(row) for row in raw_rows]

def start_athena_query(
    *,
    athena_client: Any,
    iceberg_database_name: str,
    athena_output_s3_uri: str,
    selection_sql: str,
) -> str:
    """
    Start an Athena query and return the QueryExecutionId.
    """
    response = athena_client.start_query_execution(
        QueryString=selection_sql,
        QueryExecutionContext={"Database": iceberg_database_name},
        ResultConfiguration={"OutputLocation": athena_output_s3_uri},
    )
    return response["QueryExecutionId"]

def wait_for_athena_query(
    *,
    athena_client: Any,
    query_execution_id: str,
    poll_interval_seconds: float = 1.0,
    timeout_seconds: int = 900,
) -> None:
    """
    Poll Athena until the query succeeds, fails, or times out.
    """
    start = time.time()

    while True:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = response["QueryExecution"]["Status"]["State"]

        if status == "SUCCEEDED":
            return

        if status in {"FAILED", "CANCELLED"}:
            reason = response["QueryExecution"]["Status"].get(
                "StateChangeReason",
                "Unknown Athena error.",
            )
            raise RuntimeError(
                f"Athena query {query_execution_id} ended with status {status}: {reason}"
            )

        if time.time() - start > timeout_seconds:
            try:
                athena_client.stop_query_execution(QueryExecutionId=query_execution_id)
            except Exception:
                pass

            raise TimeoutError(
                f"Athena query {query_execution_id} did not finish within "
                f"{timeout_seconds} seconds."
            )

        time.sleep(poll_interval_seconds)

def fetch_athena_results(
    *,
    athena_client: Any,
    query_execution_id: str,
) -> list[dict[str, Any]]:
    """
    Fetch all Athena result rows and return them as a list of dicts.
    Assumes the first row on the first page is the header row.
    """
    rows_out: list[dict[str, Any]] = []
    next_token: str | None = None
    column_names: list[str] | None = None
    is_first_page = True

    while True:
        kwargs: dict[str, Any] = {"QueryExecutionId": query_execution_id}
        if next_token:
            kwargs["NextToken"] = next_token

        response = athena_client.get_query_results(**kwargs)
        result_set = response["ResultSet"]
        rows = result_set.get("Rows", [])

        if is_first_page:
            if not rows:
                return []

            header_row = rows[0]
            column_names = [
                col.get("VarCharValue", "")
                for col in header_row.get("Data", [])
            ]
            data_rows = rows[1:]
            is_first_page = False
        else:
            data_rows = rows

        for row in data_rows:
            rows_out.append(
                athena_row_to_dict(
                    column_names=column_names or [],
                    row=row,
                )
            )

        next_token = response.get("NextToken")
        if not next_token:
            break

    return rows_out

def athena_row_to_dict(
    *,
    column_names: list[str],
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a single Athena row to a Python dict keyed by column name.
    Missing values become None.
    """
    data = row.get("Data", [])
    out: dict[str, Any] = {}

    for idx, col_name in enumerate(column_names):
        if idx >= len(data):
            out[col_name] = None
            continue

        cell = data[idx]
        out[col_name] = cell.get("VarCharValue")

    return out

def parse_athena_array_string(value: Any, *, field_name: str) -> list[str]:
    """
    Parse Athena's string representation of an array<string> into a Python list[str].

    Examples:
    - "[deer, fox]" -> ["deer", "fox"]
    - "[deer]" -> ["deer"]
    - "[]" -> []
    - None -> []

    Notes:
    - Athena GetQueryResults commonly returns arrays as bracketed strings.
    - This assumes simple string arrays with no embedded commas.
    - Order is preserved, and duplicates are removed.
    """
    if value is None:
        return []

    if isinstance(value, list):
        cleaned = [str(v).strip() for v in value if str(v).strip()]
        return list(dict.fromkeys(cleaned))

    if not isinstance(value, str):
        raise TypeError(
            f"Expected {field_name} to be str | list | None, got {type(value).__name__}"
        )

    text = value.strip()

    if text == "" or text == "[]":
        return []

    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"Unexpected Athena array format for {field_name}: {value!r}")

    inner = text[1:-1].strip()
    if not inner:
        return []

    parts = [part.strip() for part in inner.split(",")]
    cleaned = [p for p in parts if p]
    return list(dict.fromkeys(cleaned))

def parse_optional_string(value: Any) -> str | None:
    """
    Normalize Athena scalar string cells:
    - None stays None
    - blank strings become None
    - non-strings are stringified
    """
    if value is None:
        return None

    text = str(value).strip()
    return text or None

def parse_optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None or value == "":
        return None

    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid integer for {field_name}: {value!r}") from e

def parse_optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid float for {field_name}: {value!r}") from e

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