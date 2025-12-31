import json
from typing import Dict, Iterable, List

import boto3

s3 = boto3.client("s3")
athena = boto3.client("athena")

def _s3_read_json(bucket: str, key: str) -> Dict:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))

def _s3_read_jsonl(bucket: str, key: str) -> Iterable[Dict]:
    """Generator yielding parsed JSON objects from an S3 JSONL object."""
    resp = s3.get_object(Bucket=bucket, Key=key)
    for line in resp["Body"].iter_lines():
        if not line:
            continue
        yield json.loads(line.decode("utf-8"))

def _drop_ctas_table_if_exists(db_name: str,
                               table_name: str,
                               athena_output_s3: str,
                               athena_workgroup: str = "primary") -> str:

    full_table_name = f"{db_name}.{table_name}"
    sql = f"DROP TABLE IF EXISTS {full_table_name}"
    qid = athena.start_query_execution(
        QueryString=sql,
        ResultConfiguration={"OutputLocation": athena_output_s3},
        WorkGroup=athena_workgroup,
    )["QueryExecutionId"]
    return qid

def _read_all_processed_rows(bucket: str, jsonl_keys: List[str]):
    """Generator yielding all processed rows from the list of jsonl S3 keys."""
    for key in jsonl_keys:
        for row in _s3_read_jsonl(bucket, key):
            yield row

def _iter_rows_from_jsonl_keys(bucket: str, keys: List[str]) -> Iterable[Dict]:
    for key in keys:
        yield from _s3_read_jsonl(bucket, key)