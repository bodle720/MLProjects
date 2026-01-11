import re
import math
from datetime import datetime
from decimal import Decimal
from typing import Union, Optional, Iterable
from collections.abc import Iterable as AbcIterable

from common.table_schemas import TABLES, TableSchema
from common.athena_utils import run_athena

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

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

def build_insert_sql(batch: list[dict],
                     full_table: str,
                     task_name: str,
                     schema: TableSchema) -> str:
    cols = schema.cols

    values_clause: list[str] = []
    for r in batch:
        values = [to_sql_value(schema, r, c, task_name) for c in cols]
        values_clause.append("(" + ", ".join(values) + ")")

    return f"INSERT INTO {full_table} ({', '.join(cols)}) VALUES " + ", ".join(values_clause)

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
            raise RuntimeError(f"{task_name} delete(upload_staging): missing/invalid job_id in batch[0]")

        safe_job_id = escape_sql_string(job_id.strip())

        seen = set()
        uniq_ids: list[str] = []
        for r in batch:
            if r.get("job_id") != job_id:
                raise RuntimeError(f"{task_name} delete(upload_staging): mixed job_id in batch")

            iid = r.get("image_id")
            if not isinstance(iid, str) or not iid.strip():
                raise RuntimeError(f"{task_name} delete(upload_staging): missing/invalid image_id")

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
                   poll: Union[int, float] = 5,
                   timeout: Union[int, float] = 1800) -> tuple[bool, str]:

    if not isinstance(chunk_size, int):
        return False, f"chunk_size must be int, got {type(chunk_size).__name__}"
    if not isinstance(timeout, (int, float)):
        return False, f"timeout must be int or float, got {type(timeout).__name__}"
    if not isinstance(poll, (int, float)):
        return False, f"poll must be int or float, got {type(poll).__name__}"
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
    saw_any = False  # <-- NEW

    def flush_one_batch(b: list[dict], chunk_no: int) -> tuple[bool, str]:
        try:
            delete_sql = build_delete_sql_by_keys(b, full_table, task_name, schema.key_cols)
            run_athena(delete_sql, f"{task_name} DELETE", athena_output_s3, athena_workgroup, poll, timeout)

            insert_sql = build_insert_sql(b, full_table, task_name, schema)
            run_athena(insert_sql, f"{task_name} INSERT", athena_output_s3, athena_workgroup, poll, timeout)

            return True, ""
        except Exception as e:
            sample = b[0] if b else {}
            sample_types = row_type_summary(sample, schema.cols)
            return False, (
                f"{e} | table={table_name} | chunk number={chunk_no} of chunks of size {chunk_size} "
                f"| rows_seen_so_far={total_rows} | sample_row_types: {sample_types}"
            )

    try:
        for r in rows:
            saw_any = True  # <-- NEW
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

        # <-- NEW: treat empty iterable as an error
        if not saw_any:
            return False, f"empty iterator of type: {type(rows).__name__}"

        # flush tail
        if batch:
            ok, err = flush_one_batch(batch, chunk_counter)
            if not ok:
                return False, err

    except Exception as e:
        return False, f"{task_name} chunked_insert iteration failed after {total_rows} rows: {e}"

    return True, ""


def delete_job_rows_from_table(job_id: str,
                                task_name: str,
                                iceberg_db_name: str,
                                table_name: str,
                                athena_output_s3: str,
                                athena_workgroup: str,
                                poll_interval: Union[int,float] = 5,
                                timeout_seconds: Union[int,float] = 1800,
                                run_compaction: bool = True) -> dict:
    """
    Delete all rows for a given job_id from an Iceberg table and optionally compact.
    Returns a dict with query ids and final states for DELETE and OPTIMIZE.
    """
    # Escape single quotes in job_id for SQL literal safety
    safe_job_id = job_id.replace("'", "''")
    full_table = f"\"{iceberg_db_name}\".\"{table_name}\""

    # 1) DELETE statement (Iceberg positional delete files)
    delete_sql = f"DELETE FROM {full_table} WHERE job_id = '{safe_job_id}'"
    delete_qid, delete_result = run_athena(delete_sql,
                                           f"{task_name} DELETE JOB ROWS",
                                           athena_output_s3,
                                           athena_workgroup,
                                           poll_interval,
                                           timeout_seconds)

    if delete_result["state"] != "SUCCEEDED":
        raise RuntimeError(f"{task_name} DELETE failed: {delete_result}")

    result = {
        "delete_query_id": delete_qid,
        "delete_state": delete_result["state"],
        "delete_resp": delete_result["response"]
    }

    # 2) Optional: compact / rewrite data for that partition to remove position deletes
    #    Use OPTIMIZE ... REWRITE DATA USING BIN_PACK WHERE job_id = '...'
    #    (WHERE may only reference partition columns; job_id is partitioned in your table)
    if run_compaction and delete_result["state"] == "SUCCEEDED":
        optimize_sql = f"OPTIMIZE {full_table} REWRITE DATA USING BIN_PACK WHERE job_id = '{safe_job_id}'"
        opt_qid, opt_result = run_athena(optimize_sql,
                                              f"{task_name} OPTIMIZE",
                                               athena_output_s3,
                                               athena_workgroup,
                                               poll_interval,
                                               timeout_seconds)
        result.update({
            "optimize_query_id": opt_qid,
            "optimize_result": opt_result
        })

    return result

def row_type_summary(row: dict, cols: list[str]) -> str:
    parts = []
    for c in cols:
        v = row.get(c)
        parts.append(f"{c}, value type={type(v).__name__}")
    return ", ".join(parts)