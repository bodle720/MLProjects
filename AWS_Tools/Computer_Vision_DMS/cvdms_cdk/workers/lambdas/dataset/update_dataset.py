import os
from typing import Any

from common.general_utils.logging_utils import log
from common.general_utils.ddb_utils import update_job_status
from common.dataset_utils.dataset_get_info import get_dataset_info
from common.dataset_utils.resolve_candidate_imagery import resolve_candidate_imagery
from common.dataset_utils.resolve_dataset_membership import resolve_dataset_membership
from common.dataset_utils.dataset_update_splits import update_dataset_splits
from common.dataset_utils.dataset_iceberg_utils import write_dataset_membership
from common.dataset_utils.dataset_s3_utils import write_s3_artifacts
from common.dataset_utils.dataset_ddb_utils import write_ddb_artifacts

JOB_TABLE_NAME = os.environ["JOB_TABLE_NAME"]
DATASETS_TABLE_NAME = os.environ["DATASETS_TABLE_NAME"]
DATASET_VERSIONS_TABLE_NAME = os.environ["DATASET_VERSIONS_TABLE_NAME"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[UPDATE_DATASET]"
CANONICAL_IMAGERY_TABLE_NAME = "canonical_imagery"

VALID_OPERATIONS = {"add", "remove"}
VALID_SPLIT_APPROACHES = {"maintain", "rebalance"}
VALID_LABEL_TYPES = {
    "single-label",
    "multi-label",
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
}
LABEL_TYPE_TO_MEMBERSHIP_TABLE = {
    "single-label": "single_label",
    "multi-label": "multi_label",
    "object-detection": "object_detection",
    "semantic-segmentation": "semantic_segmentation",
    "instance-segmentation": "instance_segmentation",
}

def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text

def _assert_request_shape(
    request: dict[str, Any],
) -> tuple[str, str, dict[str, Any], str, str, str | None]:
    dataset_id = _require_nonempty_string(
        request.get("dataset_id"),
        field_name="request.dataset_id",
    )
    operation = _require_nonempty_string(
        request.get("operation"),
        field_name="request.operation",
    )
    split_approach = _require_nonempty_string(
        request.get("split_approach"),
        field_name="request.split_approach",
    )

    if operation not in VALID_OPERATIONS:
        raise ValueError(f"Unsupported request.operation: {operation}")

    if split_approach not in VALID_SPLIT_APPROACHES:
        raise ValueError(f"Unsupported request.split_approach: {split_approach}")

    selection_config = request.get("selection_config")
    if not isinstance(selection_config, dict):
        raise ValueError("request.selection_config must be an object")

    split_strategy_name = request.get("split_strategy_name")
    if split_strategy_name is not None:
        split_strategy_name = _require_nonempty_string(
            split_strategy_name,
            field_name="request.split_strategy_name",
        )

    description = request.get("description")
    if description is not None:
        description = _require_nonempty_string(
            description,
            field_name="request.description",
        )

    return dataset_id, operation, selection_config, split_approach, split_strategy_name, description

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

        if task_type != "update_dataset":
            raise ValueError(f"{TASK_NAME} expected task_type=update_dataset, got {task_type!r}")


        dataset_id, operation, selection_config, split_approach, split_strategy_name, description = _assert_request_shape(request)

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Starting update flow for dataset_id={dataset_id}",
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

        # 1) Load existing dataset metadata/invariants
        dataset_info = get_dataset_info(
            datasets_table_name=DATASETS_TABLE_NAME,
            dataset_versions_table_name=DATASET_VERSIONS_TABLE_NAME,
            dataset_id=dataset_id,
        )

        if not dataset_info.get("exists"):
            raise ValueError(f"Dataset '{dataset_id}' does not exist.")

        latest_version = dataset_info["latest_version"]
        new_version = latest_version + 1
        label_type = dataset_info["label_type"]

        if label_type not in VALID_LABEL_TYPES:
            raise ValueError(f"Unsupported dataset label_type: {label_type!r}")

        if split_approach == "maintain":
            effective_split_strategy_name = dataset_info.get("latest_version_split_strategy")
            if not effective_split_strategy_name:
                raise ValueError(
                    f"{TASK_NAME} existing dataset missing latest_version_split_strategy "
                    f"for maintain update on dataset_id={dataset_id}"
                )
        else:
            if not split_strategy_name:
                raise ValueError(
                    "request.split_strategy_name is required when split_approach='rebalance'"
                )
            effective_split_strategy_name = split_strategy_name

        effective_description = (
            description
            if description is not None
            else dataset_info.get("latest_version_description")
        )
        if not effective_description:
            raise ValueError(
                f"{TASK_NAME} could not determine effective description for dataset_id={dataset_id}"
            )

        dataset_membership_table_name = LABEL_TYPE_TO_MEMBERSHIP_TABLE.get(label_type)
        if dataset_membership_table_name is None:
            raise ValueError(f"Unsupported dataset label_type: {label_type!r}")

        # 2) Resolve selected imagery rows for add/remove operation
        selection_sql_for_update, selected_imagery_rows = resolve_candidate_imagery(
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            label_type=label_type,
            selection_config=selection_config,
            athena_output_s3_uri=ATHENA_OUTPUT_S3,
            athena_workgroup=ATHENA_WORKGROUP,
            task_name=TASK_NAME,
        )

        if not selected_imagery_rows:
            raise ValueError(
                f"Selection returned zero candidate imagery rows to {operation} "
                f"to/from dataset {dataset_id}."
            )

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Resolved {len(selected_imagery_rows)} selected imagery rows",
            level="info",
        )

        # 3) Resolve current dataset membership rows
        membership_mode = "minimal" if split_approach == "maintain" else "enriched"

        membership_sql, current_rows = resolve_dataset_membership(
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            dataset_membership_table_name=dataset_membership_table_name,
            canonical_imagery_table_name=CANONICAL_IMAGERY_TABLE_NAME,
            dataset_id=dataset_id,
            version=latest_version,
            label_type=label_type,
            mode=membership_mode,
            athena_output_s3_uri=ATHENA_OUTPUT_S3,
            athena_workgroup=ATHENA_WORKGROUP,
            task_name=TASK_NAME,
        )

        if not current_rows:
            raise ValueError(f"Current dataset version for {dataset_id} contains zero rows.")

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Resolved {len(current_rows)} current membership rows",
            level="info",
        )

        # 4) Compute next-version rows and assign/preserve splits
        split_rows = update_dataset_splits(
            selected_imagery_rows=selected_imagery_rows,
            current_rows=current_rows,
            operation=operation,
            split_approach=split_approach,
            split_strategy_name=effective_split_strategy_name,
        )

        if not split_rows:
            raise ValueError(
                f"After {operation} operation, {dataset_id} had no rows for the newest updated version."
            )

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Computed {len(split_rows)} final split rows for version={new_version}",
            level="info",
        )

        # 5) Write new-version Iceberg membership rows
        membership_result = write_dataset_membership(
            task_name=TASK_NAME,
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            athena_output_s3_uri=ATHENA_OUTPUT_S3,
            athena_workgroup=ATHENA_WORKGROUP,
            dataset_id=dataset_id,
            version=new_version,
            dataset_label_type=label_type,
            split_rows=split_rows,
        )

        # 6) Write versioned S3 artifacts
        artifact_result = write_s3_artifacts(
            dataset_bucket_name=os.environ["DATASETS_BUCKET_NAME"],
            dataset_id=dataset_id,
            version=new_version,
            label_type=label_type,
            split_strategy_name=effective_split_strategy_name,
            selection_sql=selection_sql_for_update,
            selection_config=selection_config,
            split_rows=split_rows,
        )

        # 7) Write DDB version metadata and advance dataset latest_version
        ddb_result = write_ddb_artifacts(
            new_dataset=False,
            datasets_table_name=DATASETS_TABLE_NAME,
            dataset_versions_table_name=DATASET_VERSIONS_TABLE_NAME,
            dataset_id=dataset_id,
            new_version=new_version,
            label_type=label_type,
            description=effective_description,
            split_strategy_name=effective_split_strategy_name,
            created_by=user,
            operation=operation,
            split_approach=split_approach,
            selection_config=selection_config,
            split_rows=split_rows,
            artifact_result=artifact_result,
        )

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
                f"{TASK_NAME} Completed update flow for dataset_id={dataset_id}, "
                f"new_version={new_version}, operation={operation}, "
                f"selected_count={len(selected_imagery_rows)}, "
                f"prior_count={len(current_rows)}, final_count={membership_result['row_count']}"
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
            "new_version": new_version,
            "label_type": label_type,
            "description": effective_description,
            "effective_split_strategy_name": effective_split_strategy_name,
            "operation": operation,
            "split_approach": split_approach,
            f"candidate_imagery_count_to_{operation}": len(selected_imagery_rows),
            "preexisting_membership_count": len(current_rows),
            "final_membership_row_count": membership_result["row_count"],
            "membership_table_name": membership_result["table_name"],
            "artifact_result": artifact_result,
            "ddb_result": ddb_result,
        }

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