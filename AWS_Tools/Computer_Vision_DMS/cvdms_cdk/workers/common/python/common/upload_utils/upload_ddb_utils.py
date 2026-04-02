import time
import boto3
from typing import Any, Dict
from botocore.exceptions import ClientError

dynamodb = boto3.client("dynamodb")

DdbAttr = Dict[str, Any]
DdbItem = Dict[str, DdbAttr]

def batch_get_dynamodb_items(table_name: str,
                             keys: list[str],
                             ddb_batch_get_max: int,
                             task_name: str) -> Dict[str, DdbItem]:
    if not isinstance(keys, list):
        raise TypeError(f"{task_name} keys must be a list[str], got {type(keys).__name__}")

    keys = list(set(keys))

    if not (1 <= ddb_batch_get_max <= 100):
        raise ValueError(f"{task_name} ddb_batch_get_max must be between 1 and 100 inclusive, got {ddb_batch_get_max}")

    results = {}

    for i in range(0, len(keys), ddb_batch_get_max):
        chunk = keys[i:i + ddb_batch_get_max]
        request_keys = [{"sha256": {"S": k}} for k in chunk]
        request_items = {table_name: {"Keys": request_keys}}

        backoff = 1.0
        for attempt in range(30):
            try:
                resp = dynamodb.batch_get_item(RequestItems=request_items)

                for item in resp.get("Responses", {}).get(table_name, []):
                    sha = item.get("sha256", {}).get("S")
                    if sha:
                        results[sha] = item

                unprocessed = resp.get("UnprocessedKeys", {}).get(table_name, {}).get("Keys", [])

                if not unprocessed:
                    break

                request_items = {table_name: {"Keys": unprocessed}}
                time.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")

                retryable_codes = {
                    "ProvisionedThroughputExceededException",
                    "ThrottlingException",
                    "RequestLimitExceeded",
                    "InternalServerError",
                    "InternalServerException",
                }

                if code in retryable_codes:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 15.0)
                    continue

                raise

        else:
            raise RuntimeError(f"{task_name} DynamoDB batch_get_item exceeded retries for table {table_name}")

    return results