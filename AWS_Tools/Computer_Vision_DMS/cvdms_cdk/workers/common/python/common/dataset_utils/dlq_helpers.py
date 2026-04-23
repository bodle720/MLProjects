import boto3
from botocore.exceptions import ClientError
from decimal import Decimal
from typing import Any

from common.general_utils.athena_utils import run_athena
from common.dataset_utils.dataset_delete_utils import (
    delete_iceberg_membership,
    delete_s3_artifacts,
    delete_ddb_rows,
)

s3_client = boto3.client("s3")
dynamodb_resource = boto3.resource("dynamodb")

_MEMBERSHIP_TABLE_BY_LABEL_TYPE: dict[str, str] = {
    "single-label": "single_label",
    "multi-label": "multi_label",
    "object-detection": "object_detection",
    "semantic-segmentation": "semantic_segmentation",
    "instance-segmentation": "instance_segmentation",
}


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    text = _optional_string(value)
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, Decimal):
        try:
            return int(value)
        except Exception:
            return None

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            f = float(text)
        except Exception:
            return None
        if not f.is_integer():
            return None
        return int(f)

    return None


def _require_positive_int(value: Any, *, field_name: str) -> int:
    out = _coerce_optional_int(value)
    if out is None or out < 1:
        raise ValueError(f"{field_name} must be an integer >= 1")
    return out


def _sql_escape_literal(value: str) -> str:
    return value.replace("'", "''")


def _sql_quote(value: str) -> str:
    return "'" + _sql_escape_literal(value) + "'"


def _get_membership_table_name(*, dataset_label_type: str | None) -> str:
    if not dataset_label_type:
        raise ValueError("dataset_label_type is required")
    try:
        return _MEMBERSHIP_TABLE_BY_LABEL_TYPE[dataset_label_type]
    except KeyError as e:
        raise ValueError(f"Unsupported dataset_label_type: {dataset_label_type!r}") from e


def _extract_versioned_context(dataset_context: Any) -> tuple[str, int, str | None]:
    if not isinstance(dataset_context, dict):
        raise ValueError("dataset_context must be an object")

    dataset_id = _require_nonempty_string(
        dataset_context.get("dataset_id"),
        field_name="dataset_context.dataset_id",
    )
    new_version = _require_positive_int(
        dataset_context.get("new_version"),
        field_name="dataset_context.new_version",
    )
    label_type = _optional_string(dataset_context.get("label_type"))
    return dataset_id, new_version, label_type


def _extract_delete_context(dataset_context: Any) -> tuple[str, str | None]:
    if not isinstance(dataset_context, dict):
        raise ValueError("dataset_context must be an object")

    dataset_id = _require_nonempty_string(
        dataset_context.get("dataset_id"),
        field_name="dataset_context.dataset_id",
    )
    label_type = _optional_string(dataset_context.get("label_type"))
    return dataset_id, label_type


def build_dataset_version_delete_prefix(*, dataset_id: str, version: int) -> str:
    dataset_id = _require_nonempty_string(dataset_id, field_name="dataset_id")
    if version < 1:
        raise ValueError(f"version must be >= 1, got {version}")
    return f"datasets/{dataset_id}/v{version}/"


def delete_dataset_version_s3_prefix(
    *,
    datasets_bucket_name: str,
    dataset_id: str,
    version: int,
) -> dict[str, Any]:
    """
    Delete only the version-specific S3 artifacts for a failed create/update:

      s3://<datasets_bucket>/datasets/<dataset_id>/v<version>/
    """
    prefix = build_dataset_version_delete_prefix(
        dataset_id=dataset_id,
        version=version,
    )

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
        "version": version,
        "bucket": datasets_bucket_name,
        "prefix": prefix,
        "found_object_count": len(keys_to_delete),
        "deleted_object_count": deleted_count,
    }


def build_delete_dataset_version_membership_sql(
    *,
    iceberg_database_name: str,
    table_name: str,
    dataset_id: str,
    version: int,
) -> str:
    full_table = f"\"{iceberg_database_name}\".\"{table_name}\""
    dataset_id_sql = _sql_quote(dataset_id)

    return (
        f"DELETE FROM {full_table} "
        f"WHERE dataset_id = {dataset_id_sql} AND version = {version}"
    )


def delete_dataset_version_iceberg_membership(
    *,
    iceberg_database_name: str,
    athena_output_s3_uri: str,
    athena_workgroup: str,
    dataset_id: str,
    version: int,
    dataset_label_type: str,
    task_name: str,
) -> dict[str, Any]:
    """
    Delete only the failed dataset-version membership slice from the single
    membership table implied by dataset_label_type.
    """
    dataset_id = _require_nonempty_string(dataset_id, field_name="dataset_id")
    version = _require_positive_int(version, field_name="version")
    table_name = _get_membership_table_name(dataset_label_type=dataset_label_type)

    sql = build_delete_dataset_version_membership_sql(
        iceberg_database_name=iceberg_database_name,
        table_name=table_name,
        dataset_id=dataset_id,
        version=version,
    )

    query_execution_id, _ = run_athena(
        sql,
        f"{task_name} DELETE_DATASET_VERSION_MEMBERSHIP",
        athena_output_s3_uri,
        athena_workgroup,
    )

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "version": version,
        "dataset_label_type": dataset_label_type,
        "table_name": table_name,
        "query_execution_id": query_execution_id,
        "delete_predicate": f"dataset_id={dataset_id!r} AND version={version}",
    }


def rollback_dataset_version_ddb(
    *,
    datasets_table_name: str,
    dataset_versions_table_name: str,
    dataset_id: str,
    new_version: int,
) -> dict[str, Any]:
    """
    Best-effort DDB rollback for failed create/update.

    Behavior:
    - failed create (new_version == 1):
        * delete dataset row only if latest_version == 1
        * delete version row only if safe (dataset row missing or latest_version == 1)

    - failed update (new_version > 1):
        * if dataset row currently points to new_version, revert latest_version to new_version - 1
        * delete version row for new_version if present

    The operations are intentionally idempotent / retry-friendly.
    """
    dataset_id = _require_nonempty_string(dataset_id, field_name="dataset_id")
    new_version = _require_positive_int(new_version, field_name="new_version")

    datasets_table = dynamodb_resource.Table(datasets_table_name)
    dataset_versions_table = dynamodb_resource.Table(dataset_versions_table_name)

    dataset_resp = datasets_table.get_item(
        Key={"dataset_id": dataset_id},
        ConsistentRead=True,
    )
    dataset_row = dataset_resp.get("Item")

    version_resp = dataset_versions_table.get_item(
        Key={"dataset_id": dataset_id, "version": new_version},
        ConsistentRead=True,
    )
    version_row = version_resp.get("Item")

    dataset_latest_version = None
    if dataset_row is not None:
        dataset_latest_version = _coerce_optional_int(dataset_row.get("latest_version"))

    summary: dict[str, Any] = {
        "ok": True,
        "dataset_id": dataset_id,
        "new_version": new_version,
        "dataset_row_exists_before": dataset_row is not None,
        "dataset_row_latest_version_before": dataset_latest_version,
        "version_row_exists_before": version_row is not None,
        "dataset_row_action": None,
        "version_row_action": None,
    }

    if new_version == 1:
        # Failed create rollback: safest public-truth cleanup is to remove the dataset row
        # when it clearly belongs to the failed create, then delete the version row.
        if dataset_row is None:
            summary["dataset_row_action"] = "missing_noop"
        elif dataset_latest_version == 1:
            datasets_table.delete_item(Key={"dataset_id": dataset_id})
            summary["dataset_row_action"] = "deleted"
        else:
            summary["dataset_row_action"] = "skipped_unexpected_latest_version"

        if dataset_row is None or dataset_latest_version == 1:
            if version_row is None:
                summary["version_row_action"] = "missing_noop"
            else:
                dataset_versions_table.delete_item(
                    Key={"dataset_id": dataset_id, "version": new_version}
                )
                summary["version_row_action"] = "deleted"
        else:
            summary["version_row_action"] = "skipped_unsafe_existing_dataset"

        return summary

    # Failed update rollback: revert dataset.latest_version first, then delete the new version row.
    previous_version = new_version - 1

    if dataset_row is None:
        summary["dataset_row_action"] = "missing_noop"
    elif dataset_latest_version == previous_version:
        summary["dataset_row_action"] = "already_reverted"
    elif dataset_latest_version == new_version:
        try:
            datasets_table.update_item(
                Key={"dataset_id": dataset_id},
                UpdateExpression="SET latest_version = :prev",
                ConditionExpression="attribute_exists(dataset_id) AND latest_version = :current",
                ExpressionAttributeValues={
                    ":prev": previous_version,
                    ":current": new_version,
                },
            )
            summary["dataset_row_action"] = "reverted_latest_version"
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "ConditionalCheckFailedException":
                summary["dataset_row_action"] = "conditional_noop"
            else:
                raise
    else:
        summary["dataset_row_action"] = "skipped_unexpected_latest_version"

    if version_row is None:
        summary["version_row_action"] = "missing_noop"
    else:
        dataset_versions_table.delete_item(
            Key={"dataset_id": dataset_id, "version": new_version}
        )
        summary["version_row_action"] = "deleted"

    return summary


def rollback_failed_create_or_update(
    *,
    task_name: str,
    task_type: str,
    dataset_context: Any,
    datasets_bucket_name: str,
    iceberg_database_name: str,
    athena_output_s3_uri: str,
    athena_workgroup: str,
    datasets_table_name: str,
    dataset_versions_table_name: str,
) -> dict[str, Any]:
    """
    Best-effort rollback for failed create/update.

    Tries all 3 durable cleanup steps and returns a summary:
    - delete dataset-version S3 prefix
    - delete dataset-version Iceberg membership
    - rollback DDB dataset/version rows

    The caller can inspect summary["ok"] and summary["errors"].
    """
    summary: dict[str, Any] = {
        "ok": False,
        "task_type": task_type,
        "dataset_id": None,
        "new_version": None,
        "label_type": None,
        "s3_result": None,
        "iceberg_result": None,
        "ddb_result": None,
        "errors": [],
    }

    if task_type not in {"create_dataset", "update_dataset"}:
        summary["errors"].append(f"Unsupported task_type for rollback: {task_type!r}")
        return summary

    try:
        dataset_id, new_version, label_type = _extract_versioned_context(dataset_context)
        summary["dataset_id"] = dataset_id
        summary["new_version"] = new_version
        summary["label_type"] = label_type
    except Exception as e:
        summary["errors"].append(f"Failed to parse dataset_context for rollback: {e}")
        return summary

    try:
        summary["s3_result"] = delete_dataset_version_s3_prefix(
            datasets_bucket_name=datasets_bucket_name,
            dataset_id=dataset_id,
            version=new_version,
        )
    except Exception as e:
        summary["errors"].append(f"S3 version-prefix rollback failed: {type(e).__name__}: {e}")

    if label_type:
        try:
            summary["iceberg_result"] = delete_dataset_version_iceberg_membership(
                iceberg_database_name=iceberg_database_name,
                athena_output_s3_uri=athena_output_s3_uri,
                athena_workgroup=athena_workgroup,
                dataset_id=dataset_id,
                version=new_version,
                dataset_label_type=label_type,
                task_name=task_name,
            )
        except Exception as e:
            summary["errors"].append(f"Iceberg version rollback failed: {type(e).__name__}: {e}")
    else:
        summary["errors"].append("dataset_context.label_type missing; cannot roll back Iceberg membership")

    try:
        summary["ddb_result"] = rollback_dataset_version_ddb(
            datasets_table_name=datasets_table_name,
            dataset_versions_table_name=dataset_versions_table_name,
            dataset_id=dataset_id,
            new_version=new_version,
        )
    except Exception as e:
        summary["errors"].append(f"DDB version rollback failed: {type(e).__name__}: {e}")

    summary["ok"] = len(summary["errors"]) == 0
    return summary


def finish_failed_delete(
    *,
    task_name: str,
    dataset_context: Any,
    datasets_bucket_name: str,
    iceberg_database_name: str,
    athena_output_s3_uri: str,
    athena_workgroup: str,
    datasets_table_name: str,
    dataset_versions_table_name: str,
) -> dict[str, Any]:
    """
    Best-effort completion for failed delete_dataset.

    This is NOT a rollback. Once delete has begun, preexisting data may already be gone,
    so the safest behavior is to try to finish deleting:
    - all dataset membership rows
    - all dataset S3 artifacts
    - all dataset DDB rows
    """
    summary: dict[str, Any] = {
        "ok": False,
        "task_type": "delete_dataset",
        "dataset_id": None,
        "label_type": None,
        "iceberg_result": None,
        "s3_result": None,
        "ddb_result": None,
        "errors": [],
    }

    try:
        dataset_id, label_type = _extract_delete_context(dataset_context)
        summary["dataset_id"] = dataset_id
        summary["label_type"] = label_type
    except Exception as e:
        summary["errors"].append(f"Failed to parse dataset_context for delete completion: {e}")
        return summary

    if label_type:
        try:
            summary["iceberg_result"] = delete_iceberg_membership(
                iceberg_database_name=iceberg_database_name,
                athena_output_s3_uri=athena_output_s3_uri,
                athena_workgroup=athena_workgroup,
                dataset_id=dataset_id,
                dataset_label_type=label_type,
                task_name=f"{task_name} COMPLETE_DELETE",
            )
        except Exception as e:
            summary["errors"].append(f"Iceberg delete completion failed: {type(e).__name__}: {e}")
    else:
        summary["errors"].append("dataset_context.label_type missing; cannot finish Iceberg deletion")

    try:
        summary["s3_result"] = delete_s3_artifacts(
            datasets_bucket_name=datasets_bucket_name,
            dataset_id=dataset_id,
        )
    except Exception as e:
        summary["errors"].append(f"S3 delete completion failed: {type(e).__name__}: {e}")

    try:
        summary["ddb_result"] = delete_ddb_rows(
            datasets_table_name=datasets_table_name,
            dataset_versions_table_name=dataset_versions_table_name,
            dataset_id=dataset_id,
        )
    except Exception as e:
        summary["errors"].append(f"DDB delete completion failed: {type(e).__name__}: {e}")

    summary["ok"] = len(summary["errors"]) == 0
    return summary