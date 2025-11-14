import os
import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

lambda_client = boto3.client("lambda")
CLEANUP_FN = os.environ.get("CLEANUP_FUNCTION_NAME")

def handler(event, context):
    logger.info("Received event: %s", json.dumps(event))
    req_type = event.get("RequestType")

    # Run cleanup on Delete events
    if req_type == "Delete":
        resp = lambda_client.invoke(
            FunctionName=CLEANUP_FN,
            InvocationType="RequestResponse",
            Payload=json.dumps({"action": "cleanup"})
        )
        status_code = resp.get("StatusCode")
        payload = resp.get("Payload").read().decode("utf-8")
        logger.info("Cleanup invoke status=%s payload=%s", status_code, payload)
        if status_code != 200:
            raise Exception(f"Cleanup invocation failed: {status_code} {payload}")

    # Return success for Create/Update/Delete (or adapt as needed)
    return {"PhysicalResourceId": "GlueDatabaseCleanupResource", "Data": {"status": "ok"}}
