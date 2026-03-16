import time
import boto3
from typing import Union, TypedDict, Any, Dict, Optional, List

athena = boto3.client("athena")

class AthenaWaitResult(TypedDict):
    state: str
    response: Dict[str, Any]
    timed_out: bool

def wait_for_athena(query_execution_id: str,
                    task_name: str,
                    poll: Union[int, float] = 1.5,
                    timeout: Union[int, float] = 900) -> AthenaWaitResult:
    try:
        poll_s = float(poll)
        timeout_s = float(timeout)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{task_name} poll/timeout must be numbers: poll={poll!r} timeout={timeout!r}") from e

    if poll_s <= 0:
        raise ValueError(f"{task_name} poll must be > 0, got {poll_s}")
    if timeout_s <= 0:
        raise ValueError(f"{task_name} timeout must be > 0, got {timeout_s}")

    start = time.time()

    while True:
        try:
            resp = athena.get_query_execution(QueryExecutionId=query_execution_id)
            state = resp["QueryExecution"]["Status"]["State"]
        except Exception as e:
            raise RuntimeError(
                f"{task_name} wait_for_athena failed qid={query_execution_id}: {e}"
            ) from e

        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return {"state": state, "response": resp, "timed_out": False}

        elapsed = time.time() - start
        if elapsed > timeout_s:
            return {"state": state, "response": resp, "timed_out": True}

        # Sleep, but don't overshoot remaining time by too much
        remaining = timeout_s - elapsed
        time.sleep(min(poll_s, max(0.0, remaining)))

def athena_error_details(qid: str) -> str:
    try:
        qe = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
    except Exception as e:
        return f"failed_to_get_query_execution: {e}"

    st: Dict[str, Any] = qe.get("Status", {}) or {}
    reason = st.get("StateChangeReason") or "unknown"

    ae = st.get("AthenaError")
    if isinstance(ae, dict):
        msg = ae.get("ErrorMessage") or ""
        cat = ae.get("ErrorCategory")
        typ = ae.get("ErrorType")
        retryable = ae.get("Retryable")
        return f"{reason} | AthenaError(message={msg!r}, category={cat}, type={typ}, retryable={retryable})"

    return reason

def run_athena(sql: str,
               task_name: str,
               athena_output_s3: str,
               athena_workgroup: str,
               poll: Union[int, float] = 1.5,
               timeout: Union[int, float] = 900) -> tuple[str, AthenaWaitResult]:

    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": athena_output_s3},
        WorkGroup=athena_workgroup
    )["QueryExecutionId"]

    wait_res = wait_for_athena(qid, task_name, poll=poll, timeout=timeout)

    if wait_res["timed_out"]:
        post_stop_state = "unknown"
        try:
            athena.stop_query_execution(QueryExecutionId=qid)
            stop_attempt_msg = "stop_called"
        except Exception as e:
            stop_attempt_msg = f"stop_failed: {e}"

        try:
            post = athena.get_query_execution(QueryExecutionId=qid)
            post_stop_state = post["QueryExecution"]["Status"]["State"]
        except Exception:
            pass

        raise RuntimeError(
            f"{task_name} timed out qid={qid} pre_stop_state={wait_res['state']} "
            f"post_stop_state={post_stop_state} stop_attempt_msg={stop_attempt_msg}, sql preview: {sql[:200]}"
        )

    if wait_res["state"] != "SUCCEEDED":
        reason = athena_error_details(qid)
        raise RuntimeError(
            f"{task_name} failed qid={qid} state={wait_res['state']} reason={reason}, sql preview: {sql[:200]}"
        )

    return qid, wait_res

def athena_count_job_rows(job_id: str,
                           task_name: str,
                           db_name: str,
                           table_name: str,
                           athena_output_s3: str,
                           athena_workgroup: str = "primary",
                           poll: Union[int, float] = 2.0,
                           timeout: Union[int, float] = 600) -> int:
    """COUNT(*) from upload_staging WHERE job_id='<job_id>'."""

    if '"' in db_name or '"' in table_name:
        raise ValueError(f"{task_name} db_name/table_name must not contain quotes")

    safe_job_id = job_id.replace("'", "''")
    sql = (
        f"SELECT count(*) as cnt FROM \"{db_name}\".\"{table_name}\" "
        f"WHERE job_id = '{safe_job_id}'"
    )
    qid, _ = run_athena(sql,
                       task_name + " COUNT JOB ROWS",
                       athena_output_s3,
                       athena_workgroup,
                       poll,
                       timeout)

    count = athena_get_int_scalar(qid, task_name)  # row 1 col 0
    return count

def drop_table_if_exists(db_name: str,
                           table_name: str,
                           task_name: str,
                           athena_output_s3: str,
                           athena_workgroup: str = "primary",
                           poll: Union[int, float] = 2.0,
                           timeout: Union[int, float] = 600) -> str:
    if '"' in db_name or '"' in table_name:
        raise ValueError(f"{task_name} db_name/table_name must not contain quotes")

    sql = f"DROP TABLE IF EXISTS {db_name}.{table_name}" # # sql = f'DROP TABLE IF EXISTS "{db_name}"."{table_name}"'

    qid, _ = run_athena(sql,
                       task_name,
                       athena_output_s3,
                       athena_workgroup,
                       poll,
                       timeout)
    return qid

def athena_get_cell(qid: str,
                    task_name: str,
                    row_index: int = 1,
                    col_index: int = 0) -> Optional[str]:
    """
    Return a specific cell from Athena results as a string.
    Defaults to row 1/col 0 (first data row, first column).
    Returns None if missing or SQL NULL.
    """
    if row_index < 0 or col_index < 0:
        raise ValueError(f"{task_name} ATHENA row_index/col_index must be >= 0, got {row_index=}, {col_index=}")

    try:
        resp = athena.get_query_results(QueryExecutionId=qid, MaxResults=max(10, row_index + 1))
    except Exception as e:
        raise RuntimeError(f"{task_name} ATHENA get_query_results failed qid={qid}: {e}") from e

    rows = resp.get("ResultSet", {}).get("Rows", [])
    if len(rows) <= row_index:
        return None

    data = rows[row_index].get("Data", [])
    if len(data) <= col_index:
        return None

    return data[col_index].get("VarCharValue")

def athena_get_int_scalar(qid: str, task_name: str) -> int:
    s = athena_get_cell(qid, task_name, row_index=1, col_index=0)

    if s is None:
        raise Exception(f"{task_name} ATHENA get int scalar failed, athena get cell returned None")

    try:
        return int(s)
    except (TypeError, ValueError) as e:
        raise Exception(f"{task_name} ATHENA get int scalar failed, athena get cell returned invalid type: {type(s)}, returned = {s}, exception: {e}")

def athena_fetch_all_rows(qid: str) -> List[Dict[str, Optional[str]]]:
    """
    Returns list[dict] mapping column_name -> VarCharValue (strings).
    Note: Athena returns everything as strings here.
    """
    rows_out: List[Dict[str, Optional[str]]] = []
    next_token = None
    header: List[str] = []

    while True:
        kwargs = {"QueryExecutionId": qid}
        if next_token:
            kwargs["NextToken"] = next_token

        resp = athena.get_query_results(**kwargs)
        rs = resp.get("ResultSet", {})
        rows = rs.get("Rows", [])

        # first page contains header row
        if not header:
            if not rows:
                return []
            header = [c.get("VarCharValue", "") for c in rows[0].get("Data", [])]
            rows = rows[1:]  # drop header row

        for r in rows:
            data = r.get("Data", [])
            item = {}
            for i, col in enumerate(header):
                v = data[i].get("VarCharValue") if i < len(data) else None
                item[col] = v
            rows_out.append(item)

        next_token = resp.get("NextToken")
        if not next_token:
            break

    return rows_out
