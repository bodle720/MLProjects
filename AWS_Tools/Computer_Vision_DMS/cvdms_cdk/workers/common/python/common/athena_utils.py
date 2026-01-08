import time
import boto3
from typing import Union, Optional

athena = boto3.client("athena")

def wait_for_athena(query_execution_id,
                    poll=1.5,
                    timeout=900):
    start = time.time()
    while True:
        try:
            resp = athena.get_query_execution(QueryExecutionId=query_execution_id)
            state = resp["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                return {"state": state, "response": resp, "timed_out": False}
            if time.time() - start > timeout:
                return {"state": state, "response": resp, "timed_out": True}
            time.sleep(poll)
        except Exception as e:
            raise RuntimeError(f"Exception in wait_for_athena: {e}") from e

def athena_error_details(qid: str) -> str:
    qe = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
    st = qe.get("Status", {})
    ae = st.get("AthenaError")
    if ae:
        return f"{st.get('StateChangeReason','unknown')} | AthenaError={ae}"
    return st.get("StateChangeReason", "unknown")

def run_athena(sql: str,
               task_name: str,
               athena_output_s3: str,
               athena_workgroup: str,
               poll: Union[int, float],
               timeout: Union[int, float]) -> tuple[str, dict]:

    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": athena_output_s3},
        WorkGroup=athena_workgroup
    )["QueryExecutionId"]

    wait_res = wait_for_athena(qid, poll=poll, timeout=timeout)

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
            f"{task_name} timed out qid={qid} pre_stop_state={wait_res['state']} post_stop_state={post_stop_state} stop_attempt_msg={stop_attempt_msg}"
        )
    if wait_res["state"] != "SUCCEEDED":
        reason = athena_error_details(qid)
        raise RuntimeError(
            f"{task_name} failed qid={qid} state={wait_res['state']} timed_out={wait_res['timed_out']} reason={reason}"
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
    safe_job_id = job_id.replace("'", "''")
    sql = (
        f"SELECT count(*) as cnt FROM \"{db_name}\".\"{table_name}\" "
        f"WHERE job_id = '{safe_job_id}'"
    )
    qid, wait_res = run_athena(sql,
                               task_name,
                               athena_output_s3,
                               athena_workgroup,
                               poll,
                               timeout)

    out = athena.get_query_results(QueryExecutionId=qid)
    rows = out.get("ResultSet", {}).get("Rows", [])

    if len(rows) < 2 or not rows[1].get("Data"):
        return 0
    val = rows[1]["Data"][0].get("VarCharValue")

    try:
        return int(val)
    except (TypeError, ValueError):
        return 0

def drop_ctas_table_if_exists(db_name: str,
                               table_name: str,
                               task_name: str,
                               athena_output_s3: str,
                               athena_workgroup: str = "primary",
                               poll: int = 2.0,
                               timeout: int = 600) -> str:

    full_table_name = f"{db_name}.{table_name}"
    sql = f"DROP TABLE IF EXISTS {full_table_name}"

    qid, wait_res = run_athena(sql,
                               task_name,
                               athena_output_s3,
                               athena_workgroup,
                               poll,
                               timeout)
    return qid

def athena_get_scalar(qid: str) -> Optional[str]:
    """
    Return the first data cell (row 1, col 0) from an Athena query result as a string.
    - Returns None if there is no data row or no cell.
    """
    resp = athena.get_query_results(QueryExecutionId=qid, MaxResults=2)
    rows = resp.get("ResultSet", {}).get("Rows", [])
    if len(rows) < 2:
        return None

    data = rows[1].get("Data", [])
    if not data:
        return None

    return data[0].get("VarCharValue")