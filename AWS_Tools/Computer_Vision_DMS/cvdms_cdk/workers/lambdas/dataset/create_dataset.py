import os
from typing import Any

from common.general_utils.logging_utils import log
from common.general_utils.ddb_utils import update_job_status, dataset_exists
from common.dataset_utils.resolve_candidate_imagery import resolve_candidate_imagery
from common.dataset_utils.dataset_iceberg_utils import write_dataset_membership
from common.dataset_utils.dataset_s3_utils import write_s3_artifacts
from common.dataset_utils.dataset_ddb_utils import write_ddb_artifacts
from common.dataset_utils.split_strategies.stratified_v1 import stratified_v1

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
DATASETS_TABLE_NAME = os.environ["DATASETS_TABLE_NAME"]
DATASET_VERSIONS_TABLE_NAME = os.environ["DATASET_VERSIONS_TABLE_NAME"]
DATASETS_BUCKET_NAME = os.environ["DATASETS_BUCKET_NAME"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[DATASET_CREATE]"

VALID_LABEL_TYPES = {
    "single-label",
    "multi-label",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
}

def _assert_request_shape(request: dict[str, Any]) -> tuple[str, str, str, dict[str, Any], str]:
    dataset_id = _require_nonempty_string(request.get("dataset_id"), field_name="request.dataset_id")
    label_type = _require_nonempty_string(request.get("label_type"), field_name="request.label_type")
    description = _require_nonempty_string(request.get("description"), field_name="request.description")
    split_strategy_name = _require_nonempty_string(
        request.get("split_strategy_name"),
        field_name="request.split_strategy_name",
    )

    if label_type not in VALID_LABEL_TYPES:
        raise ValueError(f"Unsupported request.label_type: {label_type}")

    selection_config = request.get("selection_config")
    if not isinstance(selection_config, dict):
        raise ValueError("request.selection_config must be an object")

    return dataset_id, label_type, description, selection_config, split_strategy_name

def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text

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

        if task_type != "create_dataset":
            raise ValueError(f"{TASK_NAME} expected task_type=create_dataset, got {task_type!r}")

        dataset_id, label_type, description, selection_config, split_strategy_name = _assert_request_shape(request)

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Starting create flow for dataset_id={dataset_id}",
            level="info",
        )

        ok, reason = update_job_status(
            job_id=job_id,
            status="IN_PROGRESS",  # or COMPLETED / FAILED
            job_table_name=JOB_TABLE_NAME,
            stream_name=LOG_FIREHOSE_STREAM_NAME,
            user=user,
            event_type=event_type
        )

        # Authoritative server-side uniqueness check.
        if dataset_exists(dataset_id, DATASETS_TABLE_NAME):
            raise ValueError(f"Dataset '{dataset_id}' already exists.")

        # Resolve candidate imagery from canonical tables.
        selection_sql, candidates = resolve_candidate_imagery(
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            label_type=label_type,
            selection_config=selection_config,
            athena_output_s3_uri=ATHENA_OUTPUT_S3,
            athena_workgroup=ATHENA_WORKGROUP,
            task_name=TASK_NAME,
        )

        if not candidates:
            raise ValueError(f"Dataset '{dataset_id}' selection returned zero candidate rows.")

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Resolved {len(candidates)} candidate rows for dataset_id={dataset_id}",
            level="info",
        )

        # Split assignment
        if split_strategy_name != "stratified_v1":
            raise ValueError(f"Split strategy '{split_strategy_name}' not supported.")

        split_rows = stratified_v1(candidates=candidates)

        if not split_rows:
            raise ValueError(f"{TASK_NAME} stratified_v1 returned zero split rows for dataset_id={dataset_id}")

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Assigned splits for {len(split_rows)} rows",
            level="info",
        )

        # Iceberg membership rows
        membership_result = write_dataset_membership(
            task_name=TASK_NAME,
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            athena_output_s3_uri=ATHENA_OUTPUT_S3,
            athena_workgroup=ATHENA_WORKGROUP,
            dataset_id=dataset_id,
            version=1,
            dataset_label_type=label_type,
            split_rows=split_rows,
        )

        # S3 artifacts
        artifact_result = write_s3_artifacts(
            dataset_bucket_name=DATASETS_BUCKET_NAME,
            dataset_id=dataset_id,
            version=1,
            label_type=label_type,
            split_strategy_name=split_strategy_name,
            selection_sql=selection_sql,
            selection_config=selection_config,
            split_rows=split_rows,
        )

        # DDB dataset + version metadata
        ddb_result = write_ddb_artifacts(
            new_dataset=True,
            datasets_table_name=DATASETS_TABLE_NAME,
            dataset_versions_table_name=DATASET_VERSIONS_TABLE_NAME,
            dataset_id=dataset_id,
            new_version=1,
            label_type=label_type,
            description=description,
            split_strategy_name=split_strategy_name,
            created_by=user,
            operation="create",
            split_approach="initial",
            selection_config=selection_config,
            split_rows=split_rows,
            artifact_result=artifact_result,
        )

        ok, reason = update_job_status(
            job_id=job_id,
            status="COMPLETED",
            job_table_name=JOB_TABLE_NAME,
            stream_name=LOG_FIREHOSE_STREAM_NAME,
            user=user,
            event_type=event_type
        )

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            (
                f"{TASK_NAME} Completed create flow for dataset_id={dataset_id}, "
                f"version=1, candidates={len(candidates)}, membership_rows={membership_result['row_count']}"
            ),
            level="info",
        )

        return {
            "status": "ok",
            "job_id": job_id,
            "user": user,
            "event_type": event_type,
            "task_type": task_type,
            "submission_s3_uri": submission_s3_uri,
            "dataset_id": dataset_id,
            "version": 1,
            "label_type": label_type,
            "description": description,
            "split_strategy_name": split_strategy_name,
            "candidate_count": len(candidates),
            "membership_row_count": membership_result["row_count"],
            "membership_table_name": membership_result["table_name"],
            "artifact_result": artifact_result,
            "ddb_result": ddb_result,
        }

    except Exception as e:
        error_message = f"{TASK_NAME} Failed: {type(e).__name__}: {e}"

        try:
            ok, reason = update_job_status(
                job_id=job_id,
                status="FAILED",
                job_table_name=JOB_TABLE_NAME,
                stream_name=LOG_FIREHOSE_STREAM_NAME,
                user=user,
                event_type=event_type,
                error_msg=error_message
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