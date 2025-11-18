import boto3
import os

glue = boto3.client("glue")

def handler(event, context):
    db_name = os.environ["GLUE_DATABASE_NAME"]

    # List and delete all tables
    paginator = glue.get_paginator("get_tables")
    for page in paginator.paginate(DatabaseName=db_name):
        for table in page.get("TableList", []):
            tname = table["Name"]
            print(f"Deleting table {tname}")
            glue.delete_table(DatabaseName=db_name, Name=tname)

    # Delete the database itself
    try:
        glue.delete_database(Name=db_name)
        print(f"Deleted database {db_name}")
    except glue.exceptions.EntityNotFoundException:
        print(f"Database {db_name} not found, nothing to delete")

    return {"status": "ok"}
