import os
import json
import logging

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

lambda_client = boto3.client("lambda")
DDL_FUNCTION_NAME = os.environ["DDL_FUNCTION_NAME"]

def handler(event, context):
    request_type = event.get("RequestType")
    logger.info("Received event: %s", json.dumps(event))

    if request_type == "Create":
        # invoke DDL lambda synchronously and fail the custom resource if it fails
        resp = lambda_client.invoke(
            FunctionName=DDL_FUNCTION_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps({"action": "run_ddl"})
        )
        payload = resp.get("Payload").read().decode("utf-8")
        status_code = resp.get("StatusCode")
        logger.info("DDL invoke status %s payload %s", status_code, payload)

        if status_code != 200:
            # raise to signal failure to CloudFormation
            raise Exception(f"DDL invocation failed: status {status_code} payload {payload}")

    elif request_type == "Delete":
        # optional cleanup; if nothing to do, just return success
        logger.info("Delete event received")
    else:
        logger.info("Update event received")

    return {"PhysicalResourceId": "IcebergDDLRun", "Data": {"message": "done"}}