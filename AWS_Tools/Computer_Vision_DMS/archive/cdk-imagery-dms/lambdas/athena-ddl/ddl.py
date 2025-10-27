# -*- coding: utf-8 -*-

import boto3
import os
import time
import re

athena = boto3.client("athena")

def load_statements(bucket: str):
    """Load SQL from ddl_statements.sql and substitute bucket name."""
    here = os.path.dirname(__file__)
    sql_path = os.path.join(here, "ddl_statements.sql")
    with open(sql_path) as f:
        raw = f.read()

    # Replace placeholder with actual bucket
    raw = raw.replace("${DATALAKE_BUCKET}", bucket)

    # Split on semicolons, strip whitespace
    stmts = [s.strip() for s in re.split(r";\s*", raw) if s.strip()]
    return stmts

def run_query(sql: str, workgroup="primary"):
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": "default"},
        WorkGroup=workgroup
    )
    qid = resp["QueryExecutionId"]

    while True:
        result = athena.get_query_execution(QueryExecutionId=qid)
        state = result["QueryExecution"]["Status"]["State"]
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            return state
        time.sleep(2)

def handler(event, context):
    bucket = os.environ["DATALAKE_BUCKET"]
    ddls = load_statements(bucket)

    print("Running Athena DDL statements...")
    for stmt in ddls:
        preview = " ".join(stmt.split())[:100]
        print(f"Executing: {preview}...")
        state = run_query(stmt)
        if state != "SUCCEEDED":
            raise Exception(f"DDL failed: {stmt}")
    print("✅ All DDL statements executed successfully")
    return {"Status": "SUCCESS"}
