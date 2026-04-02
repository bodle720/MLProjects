from typing import Union

from common.general_utils.athena_utils import run_athena

def delete_job_rows_from_table(job_id: str,
                                task_name: str,
                                iceberg_db_name: str,
                                table_name: str,
                                athena_output_s3: str,
                                athena_workgroup: str,
                                poll_interval: Union[int,float] = 5,
                                timeout_seconds: Union[int,float] = 1800) -> dict:
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

    return result