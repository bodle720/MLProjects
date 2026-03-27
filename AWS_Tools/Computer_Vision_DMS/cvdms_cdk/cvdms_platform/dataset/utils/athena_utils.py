import time
from typing import Any

##########################################################################
# Run SQL in Athena and return results.
##########################################################################

def resolve_sql(
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

    return raw_rows

##########################################################################
# Athena Execution helpers
##########################################################################

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

##########################################################################
# Parsing helpers for Athena
##########################################################################

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