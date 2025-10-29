import boto3
import os
import time
from pathlib import Path

athena = boto3.client("athena")

DATABASE = os.environ.get("ICEBERG_DATABASE")
OUTPUT_LOCATION = os.environ.get("ATHENA_OUTPUT")
ICEBERG_BUCKET = os.environ["ICEBERG_BUCKET"]

def run_query(query: str):
    """Submit a query to Athena and wait for completion."""
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": DATABASE},
        ResultConfiguration={"OutputLocation": OUTPUT_LOCATION},
    )
    qid = response["QueryExecutionId"]

    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(2)

    if state != "SUCCEEDED":
        raise RuntimeError(f"Query failed: {query}")

    return qid


def handler(event, context):
    # Ensure database exists
    run_query(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")

    # Load SQL file from the same directory
    sql_path = Path(__file__).parent / "tables.sql"
    with open(sql_path, "r") as f:
        sql_text = f.read()

    # Split on semicolons, strip whitespace
    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]

    # Replace placeholders with environment variables
    statements = [
        stmt.replace("${ICEBERG_BUCKET}", ICEBERG_BUCKET).replace("${DATABASE}", DATABASE)
        for stmt in statements
    ]

    for stmt in statements:
        run_query(stmt)

    return {"status": "ok"}
