from typing import Union

from common.general_utils.athena_utils import run_athena, athena_get_int_scalar

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