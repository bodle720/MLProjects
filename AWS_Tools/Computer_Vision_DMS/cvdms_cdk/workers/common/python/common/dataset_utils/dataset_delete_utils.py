from typing import Any
import boto3
from boto3.dynamodb.conditions import Key

from common.general_utils.athena_utils import run_athena

s3_client = boto3.client("s3")
dynamodb_resource = boto3.resource("dynamodb")

_MEMBERSHIP_TABLE_BY_LABEL_TYPE: dict[str, str] = {
    "single-label": "single_label",
    "multi-label": "multi_label",
    "object-detection": "object_detection",
    "semantic-segmentation": "semantic_segmentation",
    "instance-segmentation": "instance_segmentation",
}

def delete_iceberg_membership(*,
                            iceberg_database_name: str,
                            athena_output_s3_uri: str,
                            athena_workgroup: str,
                            dataset_id: str,
                            dataset_label_type: str,
                            task_name: str) -> dict[str, Any]:
    """
    Delete all dataset membership rows for dataset_id from the single Iceberg
    membership table implied by dataset_label_type.

    Returns a small summary dict on success.
    Raises on Athena failure.
    """
    dataset_id = _require_nonempty_string(dataset_id, field_name="dataset_id")
    table_name = _get_membership_table_name(dataset_label_type=dataset_label_type)

    sql = build_delete_membership_sql(iceberg_database_name=iceberg_database_name,
                                        table_name=table_name,
                                        dataset_id=dataset_id)

    query_execution_id, _ = run_athena(sql,
                                        task_name,
                                        athena_output_s3_uri,
                                        athena_workgroup)

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "dataset_label_type": dataset_label_type,
        "table_name": table_name,
        "query_execution_id": query_execution_id,
        "delete_predicate": f"dataset_id = {dataset_id!r}",
    }

def build_delete_membership_sql(
    *,
    iceberg_database_name: str,
    table_name: str,
    dataset_id: str,
) -> str:
    dataset_id_sql = _sql_quote(dataset_id)

    return f"""
DELETE FROM {iceberg_database_name}.{table_name}
WHERE dataset_id = {dataset_id_sql}
""".strip() + "\n"

def delete_s3_artifacts(
    *,
    datasets_bucket_name: str,
    dataset_id: str,
) -> dict[str, Any]:
    """
    Delete all dataset artifacts under:

        s3://<datasets_bucket_name>/datasets/<dataset_id>/

    Returns a summary with object counts.
    Raises on S3 list/delete failure.
    """
    dataset_id = _require_nonempty_string(dataset_id, field_name="dataset_id")
    prefix = build_dataset_delete_prefix(dataset_id=dataset_id)

    paginator = s3_client.get_paginator("list_objects_v2")

    keys_to_delete: list[str] = []
    for page in paginator.paginate(Bucket=datasets_bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys_to_delete.append(key)

    deleted_count = 0

    for i in range(0, len(keys_to_delete), 1000):
        chunk = keys_to_delete[i:i + 1000]
        response = s3_client.delete_objects(
            Bucket=datasets_bucket_name,
            Delete={"Objects": [{"Key": key} for key in chunk]},
        )

        errors = response.get("Errors", [])
        if errors:
            raise RuntimeError(
                f"Failed deleting some S3 objects under prefix '{prefix}': {errors}"
            )

        deleted_count += len(response.get("Deleted", []))

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "bucket": datasets_bucket_name,
        "prefix": prefix,
        "deleted_object_count": deleted_count,
        "found_object_count": len(keys_to_delete),
    }

def build_dataset_delete_prefix(*, dataset_id: str) -> str:
    dataset_id = _require_nonempty_string(dataset_id, field_name="dataset_id")
    return f"datasets/{dataset_id}/"

def delete_ddb_rows(
    *,
    datasets_table_name: str,
    dataset_versions_table_name: str,
    dataset_id: str,
) -> dict[str, Any]:
    """
    Delete:
    - the single datasets table row for dataset_id
    - all dataset_versions rows for dataset_id

    Returns a summary with deleted version count.
    Raises on DynamoDB query/delete failure.
    """
    dataset_id = _require_nonempty_string(dataset_id, field_name="dataset_id")

    datasets_table = dynamodb_resource.Table(datasets_table_name)
    dataset_versions_table = dynamodb_resource.Table(dataset_versions_table_name)

    version_numbers: list[int] = []

    query_kwargs: dict[str, Any] = {
        "KeyConditionExpression": Key("dataset_id").eq(dataset_id),
        "ProjectionExpression": "#dataset_id, #version",
        "ExpressionAttributeNames": {
            "#dataset_id": "dataset_id",
            "#version": "version",
        },
    }

    while True:
        response = dataset_versions_table.query(**query_kwargs)
        items = response.get("Items", [])

        for item in items:
            version_value = item.get("version")
            if version_value is None:
                continue
            version_numbers.append(int(version_value))

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

        query_kwargs["ExclusiveStartKey"] = last_evaluated_key

    version_numbers = sorted(set(version_numbers))

    with dataset_versions_table.batch_writer() as batch:
        for version in version_numbers:
            batch.delete_item(
                Key={
                    "dataset_id": dataset_id,
                    "version": version,
                }
            )

    datasets_table.delete_item(
        Key={"dataset_id": dataset_id}
    )

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "deleted_dataset_row": True,
        "deleted_version_count": len(version_numbers),
        "deleted_versions": version_numbers,
    }

def _get_membership_table_name(*, dataset_label_type: str) -> str:
    try:
        return _MEMBERSHIP_TABLE_BY_LABEL_TYPE[dataset_label_type]
    except KeyError as e:
        raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type}") from e

def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be None")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text

def _sql_quote(value: str) -> str:
    return f"'{_sql_escape_literal(value)}'"

def _sql_escape_literal(value: str) -> str:
    return value.replace("'", "''")