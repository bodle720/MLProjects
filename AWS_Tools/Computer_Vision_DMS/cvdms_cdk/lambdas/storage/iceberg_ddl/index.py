import boto3
import os
import time
from pathlib import Path

athena = boto3.client("athena")

ICEBERG_DATABASE_NAME = os.environ.get("ICEBERG_DATABASE_NAME")
S3_ATHENA_OUTPUT_URI = os.environ.get("S3_ATHENA_OUTPUT_URI")
ICEBERG_BUCKET_NAME = os.environ["ICEBERG_BUCKET_NAME"]

def run_query(query: str, context):
    """Submit a query to Athena and wait for completion."""
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": ICEBERG_DATABASE_NAME},
        ResultConfiguration={"OutputLocation": S3_ATHENA_OUTPUT_URI},
    )
    qid = response["QueryExecutionId"]

    st_time = time.time()
    max_wait_time = max(0, (context.get_remaining_time_in_millis() // 1000) - 5)
    while True and (time.time() - st_time) < max_wait_time:
        status = athena.get_query_execution(QueryExecutionId=qid)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(2)
    else:
        raise TimeoutError(f"Query execution failed due to timeout: waited {max_wait_time} seconds, perhaps raise the time limit?")

    if state in ["FAILED", "CANCELLED"]:
        reason = status["QueryExecution"]["Status"].get("StateChangeReason", "")
        raise RuntimeError(f"Query failed: {query}\nReason: {reason}")

def handler(event, context):
    # Ensure database exists
    run_query(f"CREATE DATABASE IF NOT EXISTS {ICEBERG_DATABASE_NAME}", context)

    # Load SQL file from the same directory
    sql_path = Path(__file__).parent / "tables.sql"
    with open(sql_path, "r") as f:
        sql_text = f.read()

    # Split on semicolons, strip whitespace
    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]

    # Replace placeholders with environment variables
    statements = [
        stmt.replace("${ICEBERG_BUCKET_NAME}", ICEBERG_BUCKET_NAME).replace("${ICEBERG_DATABASE_NAME}", ICEBERG_DATABASE_NAME)
        for stmt in statements
    ]

    for stmt in statements:
        print(f"Running: {stmt}")
        run_query(stmt, context)

    return {"status": "ok"}
