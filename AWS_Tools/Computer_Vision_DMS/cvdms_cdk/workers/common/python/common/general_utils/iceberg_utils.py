import re
import math
import time
import random
from datetime import datetime
from decimal import Decimal
from typing import Union, Optional, Iterable, Any
from collections.abc import Iterable as AbcIterable

from common.general_utils.athena_utils import run_athena
from common.general_utils.table_schemas import TABLES, TableSchema

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

_ICEBERG_COMMIT_RETRY_MARKERS = (
    "ICEBERG_COMMIT_ERROR",
    "Failed to commit Iceberg update",
)

def _is_retryable_iceberg_commit_error(exc: Exception | str) -> bool:
    text = str(exc)
    return any(marker in text for marker in _ICEBERG_COMMIT_RETRY_MARKERS)

def _sleep_with_backoff(
    *,
    base_sleep_sec: float,
    attempt_index: int,
    jitter_sec: float,
) -> None:
    delay = base_sleep_sec * (2 ** attempt_index)
    if jitter_sec > 0:
        delay += random.uniform(0.0, jitter_sec)
    time.sleep(delay)

#################################################################################
# Perform chunked insert via WHERE NOT EXISTS clause.
#################################################################################

def build_insert_where_not_exists_sql(batch: list[dict],
                                    full_table: str,
                                    task_name: str,
                                    schema: TableSchema) -> str:
    """
    INSERT rows from VALUES only if the key does not already exist in the destination table.

    Important:
    - Requires schema.key_cols to be non-empty.
    - Requires key values to be non-null (and for your schemas, they're strings).
    - Dedupes within-batch by key to avoid duplicates inserted from the VALUES table itself.
    """
    if not batch:
        raise ValueError(f"{task_name} batch is empty")
    if not schema.key_cols:
        raise ValueError(f"{task_name} schema.key_cols is empty for insert-only helper")

    # Deduplicate by key_cols within this batch
    uniq: list[dict] = []
    seen = set()
    for r in batch:
        key_parts: list[str] = []
        for k in schema.key_cols:
            v = r.get(k)
            # For your usage, keys are strings; enforce non-empty
            if not isinstance(v, str) or not v.strip():
                raise RuntimeError(f"{task_name} insert-only({full_table}): missing/invalid key {k}={v!r}")
            key_parts.append(v.strip())
        t = tuple(key_parts)
        if t in seen:
            continue
        seen.add(t)
        uniq.append(r)

    cols = schema.cols
    values_clause: list[str] = []
    for r in uniq:
        values = [to_sql_value(schema, r, c, task_name) for c in cols]
        values_clause.append("(" + ", ".join(values) + ")")

    col_list = ", ".join(qident(c) for c in cols)
    select_list = ", ".join(f"v.{qident(c)}" for c in cols)

    # predicate: t.k1 = v.k1 AND t.k2 = v.k2 ...
    pred = " AND ".join(f"t.{qident(k)} = v.{qident(k)}" for k in schema.key_cols)

    return f"""
    INSERT INTO {full_table} ({col_list})
    SELECT {select_list}
    FROM (VALUES {", ".join(values_clause)}) AS v({col_list})
    WHERE NOT EXISTS (
        SELECT 1 FROM {full_table} t
        WHERE {pred}
    )
    """.strip()

def chunked_insert_where_not_exists(rows: Iterable[dict],
                                    task_name: str,
                                    iceberg_db_name: str,
                                    table_name: str,
                                    athena_workgroup: str,
                                    athena_output_s3: str,
                                    chunk_size: int = 200,
                                    allow_empty: bool = True,
                                    poll: Union[int, float] = 5,
                                    timeout: Union[int, float] = 1800,
                                    commit_retry_attempts: int = 4,
                                    commit_retry_base_sleep_sec: Union[int, float] = 2.0,
                                    commit_retry_jitter_sec: Union[int, float] = 0.5) -> tuple[bool, str]:
    """
    Insert-only ingest (no delete step). Idempotent by key via WHERE NOT EXISTS.

    This is intended for tables whose keyspace is NOT shard-owned (e.g. fingerprint tables),
    or append-only mapping tables where you never want destructive deletes.

    Retries are intentionally narrow: only Iceberg commit-conflict style failures are retried.
    """
    if not isinstance(chunk_size, int):
        return False, f"{task_name} chunk_size must be int, got {type(chunk_size).__name__}"
    if not isinstance(timeout, (int, float)):
        return False, f"{task_name} timeout must be int or float, got {type(timeout).__name__}"
    if not isinstance(poll, (int, float)):
        return False, f"{task_name} poll must be int or float, got {type(poll).__name__}"
    if not isinstance(commit_retry_attempts, int) or commit_retry_attempts < 1:
        return False, f"{task_name} commit_retry_attempts must be int >= 1, got {commit_retry_attempts!r}"
    if not isinstance(commit_retry_base_sleep_sec, (int, float)) or commit_retry_base_sleep_sec < 0:
        return False, f"{task_name} commit_retry_base_sleep_sec must be >= 0, got {commit_retry_base_sleep_sec!r}"
    if not isinstance(commit_retry_jitter_sec, (int, float)) or commit_retry_jitter_sec < 0:
        return False, f"{task_name} commit_retry_jitter_sec must be >= 0, got {commit_retry_jitter_sec!r}"
    if not (0 < chunk_size <= 1000):
        return False, f"{task_name} chunk_size must be 1..1000, got {chunk_size}"
    if rows is None:
        return False, "rows is None"
    if not isinstance(rows, AbcIterable):
        return False, f"{task_name} rows to insert is not iterable, got {type(rows).__name__}"

    schema = TABLES.get(table_name)
    if schema is None:
        return False, f"{task_name} Unknown table_name: {table_name}"
    if not schema.key_cols:
        return False, f"{task_name} insert-only requires schema.key_cols for table={table_name}"

    full_table = f"\"{iceberg_db_name}\".\"{table_name}\""

    batch: list[dict] = []
    chunk_counter = 1
    total_rows = 0
    saw_any = False

    def flush_one_batch(b: list[dict], chunk_no: int) -> tuple[bool, str]:
        sample = b[0] if b else {}
        sample_types = row_type_summary(sample, schema.cols)

        for attempt in range(commit_retry_attempts):
            try:
                insert_sql = build_insert_where_not_exists_sql(b, full_table, task_name, schema)
                run_athena(
                    insert_sql,
                    f"{task_name} INSERT_ONLY",
                    athena_output_s3,
                    athena_workgroup,
                    poll,
                    timeout,
                )
                return True, ""

            except Exception as e:
                retryable = _is_retryable_iceberg_commit_error(e)
                is_last_attempt = attempt >= (commit_retry_attempts - 1)

                if retryable and not is_last_attempt:
                    _sleep_with_backoff(
                        base_sleep_sec=float(commit_retry_base_sleep_sec),
                        attempt_index=attempt,
                        jitter_sec=float(commit_retry_jitter_sec),
                    )
                    continue

                retry_note = (
                    f" | commit_retry_attempt={attempt + 1}/{commit_retry_attempts}"
                    if retryable else ""
                )

                return False, (
                    f"{task_name} {e} | table={table_name} | chunk number={chunk_no} of chunks of size {chunk_size} "
                    f"| rows_seen_so_far={total_rows}{retry_note} | sample_row_types: {sample_types}"
                )

        return False, (
            f"{task_name} insert-only flush exhausted retries unexpectedly | table={table_name} "
            f"| chunk number={chunk_no} | rows_seen_so_far={total_rows} | sample_row_types: {sample_types}"
        )

    try:
        for r in rows:
            saw_any = True
            total_rows += 1
            if not isinstance(r, dict):
                return False, f"{task_name} row {total_rows} is not a dict, got {type(r).__name__}: {r!r}"

            batch.append(r)
            if len(batch) >= chunk_size:
                ok, err = flush_one_batch(batch, chunk_counter)
                if not ok:
                    return False, err
                batch = []
                chunk_counter += 1

        if batch:
            ok, err = flush_one_batch(batch, chunk_counter)
            if not ok:
                return False, err

    except Exception as e:
        return False, f"{task_name} chunked_insert_where_not_exists iteration failed after {total_rows} rows: {e}"

    if not saw_any:
        return (True, "") if allow_empty else (False, f"{task_name} empty iterator not allowed.")

    return True, ""

#################################################################################
# Perform chunked insert, simple insert call
#################################################################################

def build_insert_sql(batch: list[dict],
                     full_table: str,
                     task_name: str,
                     schema: TableSchema) -> str:
    cols = schema.cols

    values_clause: list[str] = []
    for r in batch:
        values = [to_sql_value(schema, r, c, task_name) for c in cols]
        values_clause.append("(" + ", ".join(values) + ")")

    col_list = ", ".join(qident(c) for c in cols)

    return f"INSERT INTO {full_table} ({col_list}) VALUES " + ", ".join(values_clause)

def build_delete_sql_by_keys(batch: list[dict],
                                             table: str,
                                             task_name: str,
                                             key_cols: list[str]) -> str:
    if not batch:
        raise ValueError(f"{task_name} batch is empty")
    if not key_cols:
        raise ValueError(f"{task_name} key_cols is empty")

    # Special-case upload_staging: partition-friendly delete (job_id partition)
    if set(key_cols) == {"job_id", "image_id"}:
        job_id = batch[0].get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise RuntimeError(f"{task_name} build delete sql: missing/invalid job_id in batch[0]")

        safe_job_id = escape_sql_string(job_id.strip())

        seen = set()
        uniq_ids: list[str] = []
        for r in batch:
            if r.get("job_id") != job_id:
                raise RuntimeError(f"{task_name} build delete sql: mixed job_id in batch")

            iid = r.get("image_id")
            if not isinstance(iid, str) or not iid.strip():
                raise RuntimeError(f"{task_name} build delete sql: missing/invalid image_id")

            iid_s = iid.strip()
            if iid_s not in seen:
                seen.add(iid_s)
                uniq_ids.append(iid_s)

        if not uniq_ids:
            # nothing to delete (shouldn't happen with the checks above, but keep it safe)
            return f"DELETE FROM {table} WHERE 1 = 0"

        in_list = ", ".join("'" + escape_sql_string(i) + "'" for i in uniq_ids)
        return f"DELETE FROM {table} WHERE job_id = '{safe_job_id}' AND image_id IN ({in_list})"

    # Single-key tables
    if len(key_cols) == 1:
        k = key_cols[0]
        seen = set()
        uniq_vals: list[str] = []
        for r in batch:
            v = r.get(k)
            if not isinstance(v, str) or not v.strip():
                raise RuntimeError(f"{task_name} delete({table}): missing/invalid {k}")

            v_s = v.strip()
            if v_s not in seen:
                seen.add(v_s)
                uniq_vals.append(v_s)

        if not uniq_vals:
            return f"DELETE FROM {table} WHERE 1 = 0"

        in_list = ", ".join("'" + escape_sql_string(v) + "'" for v in uniq_vals)
        return f"DELETE FROM {table} WHERE {k} IN ({in_list})"

    # Composite-key tables: (k1,k2,...) IN ((v1,v2,...), ...)
    tuples_sql_parts: list[str] = []
    seen = set()

    for r in batch:
        raw_key: list[str] = []
        parts: list[str] = []

        for k in key_cols:
            v = r.get(k)
            if not isinstance(v, str) or not v.strip():
                raise RuntimeError(f"{task_name} delete({table}): missing/invalid {k}")

            sv = v.strip()
            raw_key.append(sv)
            parts.append("'" + escape_sql_string(sv) + "'")

        raw_key_t = tuple(raw_key)
        if raw_key_t in seen:
            continue
        seen.add(raw_key_t)
        tuples_sql_parts.append("(" + ", ".join(parts) + ")")

    if not tuples_sql_parts:
        return f"DELETE FROM {table} WHERE 1 = 0"

    cols_sql = "(" + ", ".join(key_cols) + ")"
    tuples_sql = ", ".join(tuples_sql_parts)

    # Useful for image_labels partitioned by label_type.
    if "label_type" in key_cols:
        label_types = sorted({r["label_type"].strip() for r in batch if isinstance(r.get("label_type"), str) and r["label_type"].strip()})
        if label_types:
            lt_in = ", ".join("'" + escape_sql_string(x) + "'" for x in label_types)
            return f"DELETE FROM {table} WHERE label_type IN ({lt_in}) AND {cols_sql} IN ({tuples_sql})"

    return f"DELETE FROM {table} WHERE {cols_sql} IN ({tuples_sql})"

def chunked_insert(rows: Iterable[dict],
                   task_name: str,
                   iceberg_db_name: str,
                   table_name: str,
                   athena_workgroup: str,
                   athena_output_s3: str,
                   chunk_size: int = 200,
                   allow_empty: bool = True,
                   poll: Union[int, float] = 5,
                   timeout: Union[int, float] = 1800,
                   commit_retry_attempts: int = 4,
                   commit_retry_base_sleep_sec: Union[int, float] = 2.0,
                   commit_retry_jitter_sec: Union[int, float] = 0.5) -> tuple[bool, str]:

    if not isinstance(chunk_size, int):
        return False, f"chunk_size must be int, got {type(chunk_size).__name__}"
    if not isinstance(timeout, (int, float)):
        return False, f"timeout must be int or float, got {type(timeout).__name__}"
    if not isinstance(poll, (int, float)):
        return False, f"poll must be int or float, got {type(poll).__name__}"
    if not isinstance(commit_retry_attempts, int) or commit_retry_attempts < 1:
        return False, f"{task_name} commit_retry_attempts must be int >= 1, got {commit_retry_attempts!r}"
    if not isinstance(commit_retry_base_sleep_sec, (int, float)) or commit_retry_base_sleep_sec < 0:
        return False, f"{task_name} commit_retry_base_sleep_sec must be >= 0, got {commit_retry_base_sleep_sec!r}"
    if not isinstance(commit_retry_jitter_sec, (int, float)) or commit_retry_jitter_sec < 0:
        return False, f"{task_name} commit_retry_jitter_sec must be >= 0, got {commit_retry_jitter_sec!r}"
    if not (0 < chunk_size <= 1000):
        return False, f"chunk_size must be 1..1000, got {chunk_size}"
    if rows is None:
        return False, "rows is None"
    if not isinstance(rows, AbcIterable):
        return False, f"rows to insert is not iterable, got {type(rows).__name__}"

    schema = TABLES.get(table_name)
    if schema is None:
        return False, f"Unknown table_name: {table_name}"

    full_table = f"\"{iceberg_db_name}\".\"{table_name}\""

    batch: list[dict] = []
    chunk_counter = 1
    total_rows = 0
    saw_any = False

    def flush_one_batch(b: list[dict], chunk_no: int) -> tuple[bool, str]:
        sample = b[0] if b else {}
        sample_types = row_type_summary(sample, schema.cols)

        for attempt in range(commit_retry_attempts):
            try:
                delete_sql = build_delete_sql_by_keys(b, full_table, task_name, schema.key_cols)
                run_athena(
                    delete_sql,
                    f"{task_name} DELETE",
                    athena_output_s3,
                    athena_workgroup,
                    poll,
                    timeout,
                )

                insert_sql = build_insert_sql(b, full_table, task_name, schema)
                run_athena(
                    insert_sql,
                    f"{task_name} INSERT",
                    athena_output_s3,
                    athena_workgroup,
                    poll,
                    timeout,
                )

                return True, ""

            except Exception as e:
                retryable = _is_retryable_iceberg_commit_error(e)
                is_last_attempt = attempt >= (commit_retry_attempts - 1)

                if retryable and not is_last_attempt:
                    _sleep_with_backoff(
                        base_sleep_sec=float(commit_retry_base_sleep_sec),
                        attempt_index=attempt,
                        jitter_sec=float(commit_retry_jitter_sec),
                    )
                    continue

                retry_note = (
                    f" | commit_retry_attempt={attempt + 1}/{commit_retry_attempts}"
                    if retryable else ""
                )

                return False, (
                    f"{e} | table={table_name} | chunk number={chunk_no} of chunks of size {chunk_size} "
                    f"| rows_seen_so_far={total_rows}{retry_note} | sample_row_types: {sample_types}"
                )

        return False, (
            f"{task_name} flush exhausted retries unexpectedly | table={table_name} "
            f"| chunk number={chunk_no} | rows_seen_so_far={total_rows} | sample_row_types: {sample_types}"
        )

    try:
        for r in rows:
            saw_any = True
            total_rows += 1

            if not isinstance(r, dict):
                return False, f"row {total_rows} is not a dict, got {type(r).__name__}: {r!r}"

            batch.append(r)
            if len(batch) >= chunk_size:
                ok, err = flush_one_batch(batch, chunk_counter)
                if not ok:
                    return False, err
                batch = []
                chunk_counter += 1

        if batch:
            ok, err = flush_one_batch(batch, chunk_counter)
            if not ok:
                return False, err

    except Exception as e:
        return False, f"{task_name} chunked_insert iteration failed after {total_rows} rows: {e}"

    if not saw_any:
        return (True, "") if allow_empty else (False, f"{task_name} empty iterator not allowed.")

    return True, ""

#################################################################################
# Miscellaneous helpers.
#################################################################################

def qident(x: str) -> str:
    return '"' + x.replace('"','""') + '"'

def row_type_summary(row: dict, cols: list[str]) -> str:
    parts = []
    for c in cols:
        v = row.get(c)
        parts.append(f"{c}, value type={type(v).__name__}")
    return ", ".join(parts)

def validate_timestamp_str(s: str, task_name: str) -> None:
    # fast format check
    if not _TS_RE.match(s):
        raise ValueError(f"{task_name} Invalid timestamp format (expected YYYY-MM-DD HH:MM:SS): {s!r}")

    try:
        datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError as e:
        raise ValueError(f"{task_name} Invalid timestamp value: {s!r}") from e

def escape_sql_string(s: str) -> str:
    return s.replace("'", "''")

def coerce_int(v: Union[int, float, Decimal, str, None]) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None

    if isinstance(v, int):
        return v

    if isinstance(v, float):
        return int(v) if math.isfinite(v) and v.is_integer() else None

    if isinstance(v, Decimal):
        try:
            f = float(v)
        except (OverflowError, ValueError):
            return None
        return int(f) if math.isfinite(f) and f.is_integer() else None

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        return int(f) if math.isfinite(f) and f.is_integer() else None

    return None

def coerce_float(v: Union[int, float, Decimal, str, None]) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None

    if isinstance(v, (int, float, Decimal)):
        try:
            f = float(v)
        except (OverflowError, ValueError):
            return None
        return f if math.isfinite(f) else None

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            f = float(s)
        except ValueError:
            return None
        return f if math.isfinite(f) else None

    return None

def to_sql_value(schema,
                 r: dict,
                 col: str,
                 task_name: str) -> str:
    t = schema.types[col]
    v = r.get(col)

    if v is None or isinstance(v, bool):
        return "NULL"

    if t == "string":
        return "'" + escape_sql_string(str(v)) + "'"

    if t == "int":
        iv = coerce_int(v)
        return "NULL" if iv is None else str(iv)

    if t == "float":
        fv = coerce_float(v)
        return "NULL" if fv is None else str(fv)

    if t == "timestamp":
        if not isinstance(v, str) or not v.strip():
            return "NULL"
        s = v.strip()
        validate_timestamp_str(s, task_name)   # raises if bad
        return f"TIMESTAMP '{escape_sql_string(s)}'"

    if t == "array_string":
        if not isinstance(v, list):
            raise ValueError(
                f"{task_name} Column {col} expected array<string>, got {type(v).__name__}: {v!r}"
            )
        if len(v) == 0:
            return "CAST(ARRAY[] AS ARRAY(VARCHAR))"

        if any(x is None for x in v):
            raise ValueError(
                f"{task_name} List {v} contains a None value."
            )

        # Optional: filter/normalize elements
        items = []
        for x in v:
            items.append("'" + escape_sql_string(str(x)) + "'")

        if not items:
            return "CAST(ARRAY[] AS ARRAY(VARCHAR))"

        return "ARRAY[" + ", ".join(items) + "]"

    raise ValueError(f"{task_name} Unhandled SqlType {t} for col {col}")

def normalize_string_array(
    value: Any,
    *,
    field_name: str,
    require_nonempty: bool,
) -> list[str]:
    if value is None:
        values: list[str] = []
    elif isinstance(value, list):
        values = [str(v).strip() for v in value if str(v).strip()]
    else:
        raise TypeError(f"{field_name} must be list[str] | None, got {type(value).__name__}")

    # dedupe + deterministic sort
    values = sorted(set(values))

    if require_nonempty and not values:
        raise ValueError(f"{field_name} must be non-empty")

    return values

def require_nonempty_string(value: Any, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be None")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text