import os
import boto3

ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
TASK_NAME = "[DELETE_ICEBERG_DB]"

glue = boto3.client("glue")

def handler(event, context):
    # List and delete all tables
    paginator = glue.get_paginator("get_tables")
    for page in paginator.paginate(DatabaseName=ICEBERG_DATABASE_NAME):
        for table in page.get("TableList", []):
            tname = table["Name"]
            print(f"{TASK_NAME} Deleting table {tname}")
            glue.delete_table(DatabaseName=ICEBERG_DATABASE_NAME, Name=tname)

    # Delete the database itself
    try:
        glue.delete_database(Name=ICEBERG_DATABASE_NAME)
        print(f"{TASK_NAME} Deleted database {ICEBERG_DATABASE_NAME}")
    except glue.exceptions.EntityNotFoundException:
        print(f"{TASK_NAME} Database {ICEBERG_DATABASE_NAME} not found, nothing to delete")

    return {"status": "ok"}