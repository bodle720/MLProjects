import os
from typing import Any

from common.general_utils.logging_utils import log
from common.general_utils.ddb_utils import update_job_status
from common.dataset_utils.dataset_get_info import get_dataset_info
from common.dataset_utils.dataset_delete_utils import (
    delete_iceberg_membership,
    delete_s3_artifacts,
    delete_ddb_rows,
)

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
DATASETS_TABLE_NAME = os.environ["DATASETS_TABLE_NAME"]
DATASET_VERSIONS_TABLE_NAME = os.environ["DATASET_VERSIONS_TABLE_NAME"]
DATASETS_BUCKET_NAME = os.environ["DATASETS_BUCKET_NAME"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[DATASET_DELETE]"

def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text

def _assert_request_shape(request: dict[str, Any]) -> str:
    dataset_id = _require_nonempty_string(
        request.get("dataset_id"),
        field_name="request.dataset_id",
    )
    return dataset_id

def handler(event, context):
    job_id = "unknown"
    user = "unknown"
    event_type = "DATASET_OP"

    try:
        job_id = _require_nonempty_string(event.get("job_id"), field_name="job_id")
        user = _require_nonempty_string(event.get("user"), field_name="user")
        event_type = _require_nonempty_string(event.get("event_type"), field_name="event_type")
        task_type = _require_nonempty_string(event.get("task_type"), field_name="task_type")
        submission_s3_uri = _require_nonempty_string(
            event.get("submission_s3_uri"),
            field_name="submission_s3_uri",
        )

        request = event.get("request")
        if not isinstance(request, dict):
            raise ValueError("request must be an object")

        if task_type != "delete_dataset":
            raise ValueError(f"{TASK_NAME} expected task_type=delete_dataset, got {task_type!r}")

        dataset_id = _assert_request_shape(request)

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Starting delete flow for dataset_id={dataset_id}",
            level="info",
        )

        update_job_status(
            job_id=job_id,
            status="IN_PROGRESS",
            job_table_name=JOB_TABLE_NAME,
            stream_name=LOG_FIREHOSE_STREAM_NAME,
            user=user,
            event_type=event_type,
        )

        result: dict[str, Any] = {
            "status": "ok",
            "job_id": job_id,
            "user": user,
            "event_type": event_type,
            "task_type": task_type,
            "submission_s3_uri": submission_s3_uri,
            "dataset_id": dataset_id,
            "dataset_id_exists": False,
            "label_type": None,
            "latest_version": None,
            "honor_source_splits": None,
            "deleted_iceberg_rows": False,
            "deleted_s3_artifacts": False,
            "deleted_ddb_records": False,
        }

        dataset_state = get_dataset_info(
            datasets_table_name=DATASETS_TABLE_NAME,
            dataset_versions_table_name=DATASET_VERSIONS_TABLE_NAME,
            dataset_id=dataset_id,
        )

        if not dataset_state["dataset_info"].get("exists"):
            raise ValueError(f"Dataset '{dataset_id}' does not exist.")

        dataset_meta = dataset_state["dataset_info"]
        dataset_label_type = dataset_meta.get("label_type")
        latest_version = dataset_meta.get("latest_version")
        honor_source_splits = dataset_meta.get("honor_source_splits")

        if not dataset_label_type:
            raise ValueError(
                f"Dataset '{dataset_id}' exists but is missing required field 'label_type'."
            )

        result["dataset_id_exists"] = True
        result["label_type"] = dataset_label_type
        result["latest_version"] = latest_version
        result["honor_source_splits"] = honor_source_splits

        iceberg_result = delete_iceberg_membership(
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            athena_output_s3_uri=ATHENA_OUTPUT_S3,
            athena_workgroup=ATHENA_WORKGROUP,
            dataset_id=dataset_id,
            dataset_label_type=dataset_label_type,
            task_name=TASK_NAME,
        )
        result["deleted_iceberg_rows"] = True
        result["iceberg_result"] = iceberg_result

        s3_result = delete_s3_artifacts(
            datasets_bucket_name=DATASETS_BUCKET_NAME,
            dataset_id=dataset_id,
        )
        result["deleted_s3_artifacts"] = True
        result["s3_result"] = s3_result

        ddb_result = delete_ddb_rows(
            datasets_table_name=DATASETS_TABLE_NAME,
            dataset_versions_table_name=DATASET_VERSIONS_TABLE_NAME,
            dataset_id=dataset_id,
        )
        result["deleted_ddb_records"] = True
        result["ddb_result"] = ddb_result

        update_job_status(
            job_id=job_id,
            status="COMPLETED",
            job_table_name=JOB_TABLE_NAME,
            stream_name=LOG_FIREHOSE_STREAM_NAME,
            user=user,
            event_type=event_type,
        )

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            (
                f"{TASK_NAME} Completed delete flow for dataset_id={dataset_id}, "
                f"label_type={dataset_label_type}, latest_version={latest_version}, "
                f"honor_source_splits={honor_source_splits}"
            ),
            level="info",
        )

        return result

    except Exception as e:
        error_message = f"{TASK_NAME} Failed: {type(e).__name__}: {e}"

        try:
            update_job_status(
                job_id=job_id,
                status="FAILED",
                job_table_name=JOB_TABLE_NAME,
                stream_name=LOG_FIREHOSE_STREAM_NAME,
                user=user,
                event_type=event_type,
                error_msg=error_message,
            )
        except Exception:
            pass

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            error_message,
            level="error",
        )
        raise