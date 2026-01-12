import logging
import time
from typing import Dict

import pandas as pd
from mypy_boto3_athena.client import AthenaClient

class LogClient:
    """
    High-level client to query logs.
    """
    def __init__(self,
                 *,
                 glue_db_name: str,
                 glue_table_name: str,
                 log_bucket_name: str,
                 athena_client: AthenaClient):

        self.glue_db_name = glue_db_name
        self.glue_table_name = glue_table_name
        self.log_bucket_name = log_bucket_name
        self.athena = athena_client

    def _run_query(self, query: str) -> Dict:
        resp = self.athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": self.glue_db_name},
            ResultConfiguration={
                "OutputLocation": f"s3://{self.log_bucket_name}/athena-results/"
            }
        )
        qid = resp["QueryExecutionId"]

        # poll until finished
        while True:
            status = self.athena.get_query_execution(QueryExecutionId=qid)
            state = status["QueryExecution"]["Status"]["State"]
            if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                break
            time.sleep(2)

        if state != "SUCCEEDED":
            raise RuntimeError(f"Athena query failed with state {state}")

        return self.athena.get_query_results(QueryExecutionId=qid)

    def get_logs_by_job_id(self, job_id: str) -> Dict:

        if not job_id:
            return {"error": "Job id is None"}

        try:
            # Step 1: repair partitions so new data is visible
            self._run_query(f"MSCK REPAIR TABLE {self.glue_table_name}")

            # Step 2: run the actual log query
            results = self._run_query(
                f"""
                SELECT *
                FROM {self.glue_table_name}
                WHERE job_id = '{job_id}'
                ORDER BY timestamp DESC
                """
            )

            # Extract column names
            columns = [col["Name"] for col in results["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]

            # Extract rows (skip the header row)
            rows = results["ResultSet"]["Rows"][1:]
            data = []
            for row in rows:
                values = []
                for cell in row.get("Data", []):
                    values.append(cell.get("VarCharValue"))
                data.append(values)

            # Build DataFrame if we have data
            df = pd.DataFrame(data, columns=columns) if data else None

            return {'logs_df': df, "error": None}

        except Exception as e:
            logging.error(f"Error querying Athena: {e}")
            return {"error": str(e)}