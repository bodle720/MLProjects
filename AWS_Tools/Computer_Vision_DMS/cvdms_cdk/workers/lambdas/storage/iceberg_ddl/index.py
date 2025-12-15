import os
import time
import json
import random
from pathlib import Path
from typing import Optional

import boto3

athena = boto3.client("athena")
glue = boto3.client("glue")

ICEBERG_DATABASE_NAME = os.environ.get("ICEBERG_DATABASE_NAME")
S3_ATHENA_OUTPUT_URI = os.environ.get("S3_ATHENA_OUTPUT_URI")
ICEBERG_BUCKET_NAME = os.environ["ICEBERG_BUCKET_NAME"]

# Configurable retry parameters
MAX_ATTEMPTS = 5
BASE_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 30.0
INTER_STATEMENT_DELAY = 1.5  # seconds

class AthenaError(RuntimeError):
    pass

def _remaining_seconds(context) -> float:
    return max(0.0, (context.get_remaining_time_in_millis() / 1000.0) - 2.0)

def start_query_with_retries(query: str, context, max_attempts: int = MAX_ATTEMPTS) -> str:
    """Start Athena query with retries on transient errors. Returns QueryExecutionId."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = athena.start_query_execution(
                QueryString=query,
                QueryExecutionContext={"Database": ICEBERG_DATABASE_NAME},
                ResultConfiguration={"OutputLocation": S3_ATHENA_OUTPUT_URI},
            )
            qid = resp["QueryExecutionId"]
            print(f"Started Athena query (attempt={attempt}): qid={qid}")
            print(f"Athena output URI (expected): {S3_ATHENA_OUTPUT_URI}/{qid}.csv")
            return qid
        except Exception as e:
            print(f"start_query_execution failed (attempt={attempt}): {e}")
            if attempt >= max_attempts or _remaining_seconds(context) < 5.0:
                raise AthenaError(f"start_query_execution failed after {attempt} attempts: {e}")
            backoff = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.random(), MAX_BACKOFF_SECONDS)
            time.sleep(backoff)

def wait_for_query(qid: str, context, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Wait for Athena query execution to finish, with retries for get_query_execution polling."""
    start = time.time()
    max_wait = _remaining_seconds(context)
    poll_delay = 2.0

    while True:
        try:
            status = athena.get_query_execution(QueryExecutionId=qid)
            state = status["QueryExecution"]["Status"]["State"]
            if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                print(f"Query {qid} finished with state {state}")
                return status
            if time.time() - start > max_wait:
                raise AthenaError(f"Timeout waiting for query {qid}. Waited {max_wait} seconds.")
            time.sleep(poll_delay)
            # small jitter on polling
            poll_delay = min(poll_delay * 1.2 + random.random() * 0.5, 10.0)
        except Exception as e:
            print(f"get_query_execution transient error for qid={qid}: {e}")
            # If remaining time is low or max attempts exhausted, fail
            if _remaining_seconds(context) < 5.0:
                raise AthenaError(f"get_query_execution failed and not enough remaining time: {e}")
            time.sleep(min(2 ** random.randint(0, 3), 5.0))

def run_athena_query(query: str, context):
    """Start and wait for an Athena query, raise on failure with details."""
    qid = start_query_with_retries(query, context)
    result = wait_for_query(qid, context)
    state = result["QueryExecution"]["Status"]["State"]

    if state != "SUCCEEDED":
        reason = result["QueryExecution"]["Status"].get("StateChangeReason", "<no reason>")
        raise AthenaError(f"Athena query {qid} failed with state={state}. Reason: {reason}. Query: {query}")
    return qid

def glue_table_exists_with_retry(database: str, table: str, context, attempts: int = 10) -> bool:
    """Validate table presence in Glue with retries (Glue can be eventually consistent)."""
    for attempt in range(1, attempts + 1):
        try:
            glue.get_table(DatabaseName=database, Name=table)
            print(f"Glue table found: {database}.{table} (attempt={attempt})")
            return True
        except glue.exceptions.EntityNotFoundException:
            print(f"Glue table not found yet: {database}.{table} (attempt={attempt})")
        except Exception as e:
            print(f"Glue get_table error for {database}.{table} (attempt={attempt}): {e}")

        # Respect remaining time
        if _remaining_seconds(context) < 5.0:
            break
        backoff = min(1.0 * attempt + random.random(), 8.0)
        time.sleep(backoff)
    return False

def safe_split_sql(sql_text: str) -> list:
    """
    Very small splitter that assumes semicolons are only used as statement terminators.
    Keep tables.sql simple: one statement per block and no semicolons inside strings.
    """
    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
    return statements

def handler(event, context):
    start_time = time.time()
    print(json.dumps({"msg": "DDL lambda start", "database": ICEBERG_DATABASE_NAME, "bucket": ICEBERG_BUCKET_NAME}))

    # 0. Create database (idempotent)
    db_q = f"CREATE DATABASE IF NOT EXISTS {ICEBERG_DATABASE_NAME}"
    try:
        run_athena_query(db_q, context)
    except Exception as e:
        print(f"Failed to create database: {e}")
        raise

    # 1. Load SQL file
    sql_path = Path(__file__).parent / "tables.sql"
    if not sql_path.exists():
        raise FileNotFoundError(f"tables.sql not found at {sql_path}")

    sql_text = sql_path.read_text()
    statements = safe_split_sql(sql_text)
    # 2. Replace placeholders
    statements = [
        stmt.replace("${ICEBERG_BUCKET_NAME}", ICEBERG_BUCKET_NAME).replace("${ICEBERG_DATABASE_NAME}", ICEBERG_DATABASE_NAME)
        for stmt in statements
    ]

    # 3. Execute statements with small delay, validation and retries
    for stmt in statements:
        # Skip empty or comment-only statements
        if not stmt or stmt.startswith("--"):
            continue

        print(f"Executing statement (truncated): {stmt[:320]}{'...' if len(stmt) > 320 else ''}")
        try:
            qid = run_athena_query(stmt, context)
            print(f"Athena query succeeded: qid={qid}")
        except Exception as e:
            print(f"Statement failed: {e}")
            raise

        # Add small delay between DDL statements to reduce race conditions
        time.sleep(INTER_STATEMENT_DELAY)

        # Attempt Glue validation for CREATE TABLE statements
        # Extract table name if statement contains "CREATE TABLE" (very simple parse)
        upper = stmt.upper()
        if "CREATE TABLE" in upper:
            # crude extraction: assume "CREATE TABLE IF NOT EXISTS db.table (" or "CREATE TABLE db.table ("
            try:
                after_create = stmt.split("CREATE TABLE", 1)[1]
                # remove IF NOT EXISTS
                after_create = after_create.replace("IF NOT EXISTS", "")
                table_ref = after_create.strip().split()[0]  # db.table
                if "." in table_ref:
                    db_name, table_name = table_ref.split(".", 1)
                    # strip any backticks or quotes
                    table_name = table_name.strip().strip("`").strip('"')
                    db_name = db_name.strip().strip("`").strip('"')
                    valid = glue_table_exists_with_retry(db_name, table_name, context)
                    if not valid:
                        print(f'Warning: table {table_name} has not been verified to exist in Glue yet.')
            except Exception as e:
                print(f"Glue validation parsing error for statement: {e}")
                # Not fatal in all cases, but surface; if you want strictness, raise here
                raise

    elapsed = time.time() - start_time
    print(json.dumps({"msg": "DDL lambda completed", "duration_seconds": elapsed}))
    return {"status": "ok", "duration_seconds": elapsed}