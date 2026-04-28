import os
from typing import Any
from copy import deepcopy

from common.general_utils.logging_utils import log
from common.dataset_utils.dataset_get_info import get_dataset_info
from common.dataset_utils.resolve_candidate_imagery import resolve_candidate_imagery
from common.dataset_utils.resolve_dataset_membership import resolve_dataset_membership
from common.dataset_utils.dataset_update_splits import update_dataset_splits
from common.dataset_utils.dataset_iceberg_utils import write_dataset_membership
from common.dataset_utils.dataset_s3_utils import write_s3_artifacts
from common.dataset_utils.dataset_ddb_utils import write_ddb_artifacts
from common.testing_utils.dataset_testing import maybe_fail

DATASETS_TABLE_NAME = os.environ["DATASETS_TABLE_NAME"]
DATASET_VERSIONS_TABLE_NAME = os.environ["DATASET_VERSIONS_TABLE_NAME"]
DATASETS_BUCKET_NAME = os.environ["DATASETS_BUCKET_NAME"]
ICEBERG_DATABASE_NAME = os.environ["ICEBERG_DATABASE_NAME"]
ATHENA_WORKGROUP = os.environ["ATHENA_WORKGROUP"]
ATHENA_OUTPUT_S3 = os.environ["ATHENA_OUTPUT_S3"]
LOG_FIREHOSE_STREAM_NAME = os.environ["LOG_FIREHOSE_STREAM_NAME"]

TASK_NAME = "[DATASET_UPDATE]"

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
SUPPORTED_SPLIT_STRATEGIES = {"stratified_v1"}

LABEL_TYPE_TO_MEMBERSHIP_TABLE = {
    "single-label": "single_label",
    "multi-label": "multi_label",
    "object-detection": "object_detection",
    "semantic-segmentation": "semantic_segmentation",
    "instance-segmentation": "instance_segmentation",
}

_VALID_SPLITS = {"train", "val", "test"}

def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text

def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

def _assert_request_shape(
    request: dict[str, Any],
) -> tuple[str, str, dict[str, Any], str, str | None, str | None]:
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
        if not isinstance(description, str):
            raise ValueError("request.description must be a string or null")
        description = description.strip() or None

    return (
        dataset_id,
        operation,
        selection_config,
        split_approach,
        split_strategy_name,
        description,
    )

def _normalize_supported_strategy_or_none(value: Any) -> str | None:
    text = _normalize_optional_string(value)
    if text is None:
        return None
    if text not in SUPPORTED_SPLIT_STRATEGIES:
        return None
    return text

def _resolve_effective_split_strategy_for_update(
    *,
    job_id: str,
    user: str,
    event_type: str,
    dataset_id: str,
    latest_meta: dict[str, Any],
    split_approach: str,
    requested_split_strategy_name: str | None,
    honor_source_splits: bool,
) -> tuple[str | None, str]:
    """
    Returns:
      (effective_split_strategy_name, effective_split_mode)

    Rules:
    - honor_source_splits=True:
        * rebalance forbidden
        * strategy is ignored for actual assignment
        * keep prior split_strategy_name only for metadata continuity if present
        * effective mode = honor_source_splits

    - honor_source_splits=False:
        * maintain:
            - inherit prior usable strategy if present
            - otherwise allow request to supply one
            - if neither exists, fail
        * rebalance:
            - explicit request strategy required
    """
    requested_strategy = _normalize_supported_strategy_or_none(requested_split_strategy_name)
    prior_strategy = _normalize_supported_strategy_or_none(latest_meta.get("split_strategy_name"))
    fallback_strategy = _normalize_supported_strategy_or_none(latest_meta.get("effective_split_mode"))

    if honor_source_splits:
        if split_approach == "rebalance":
            raise ValueError(
                "split_approach='rebalance' is not allowed when honor_source_splits=True."
            )

        if requested_split_strategy_name:
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                (
                    f"{TASK_NAME} honor_source_splits=True, so request.split_strategy_name="
                    f"{requested_split_strategy_name!r} is accepted but ignored for "
                    f"dataset_id={dataset_id}."
                ),
                level="info",
            )

        metadata_strategy = prior_strategy or fallback_strategy
        return metadata_strategy, "honor_source_splits"

    # honor_source_splits=False below
    if split_approach == "rebalance":
        if requested_strategy is None:
            raise ValueError(
                "request.split_strategy_name is required and must be supported when "
                "split_approach='rebalance'"
            )
        return requested_strategy, requested_strategy

    # maintain
    inherited_strategy = prior_strategy or fallback_strategy
    if inherited_strategy is not None:
        if requested_strategy and requested_strategy != inherited_strategy:
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                (
                    f"{TASK_NAME} split_approach='maintain' uses the dataset's existing "
                    f"split strategy {inherited_strategy!r}; "
                    f"request.split_strategy_name={requested_split_strategy_name!r} "
                    f"will be ignored for dataset_id={dataset_id}."
                ),
                level="info",
            )
        return inherited_strategy, inherited_strategy

    if requested_strategy is not None:
        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            (
                f"{TASK_NAME} latest version metadata did not contain a usable prior split strategy; "
                f"using request.split_strategy_name={requested_strategy!r} for "
                f"maintain update on dataset_id={dataset_id}."
            ),
            level="warning",
        )
        return requested_strategy, requested_strategy

    raise ValueError(
        f"{TASK_NAME} could not resolve a usable split strategy for maintain update on "
        f"dataset_id={dataset_id}; latest version metadata did not contain one and none "
        f"was supplied in the request."
    )

def _canonicalize_split_rows_or_raise(
    *,
    split_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    Enforce one canonical split row per image_id.

    Behavior:
    - exact duplicate row for same image_id -> collapse
    - conflicting row for same image_id -> raise

    Returns:
    - canonical rows sorted by image_id
    - duplicate_collapsed_count
    """
    canonical_by_image_id: dict[str, dict[str, Any]] = {}
    duplicate_collapsed_count = 0

    for row in split_rows:
        image_id = _require_nonempty_string(row.get("image_id"), field_name="split_rows[].image_id")
        split = _require_nonempty_string(row.get("split"), field_name="split_rows[].split")
        if split not in _VALID_SPLITS:
            raise ValueError(f"{TASK_NAME} Invalid split in split_rows for image_id={image_id}: {split!r}")

        normalized_row = {**row, "image_id": image_id, "split": split}

        existing = canonical_by_image_id.get(image_id)
        if existing is None:
            canonical_by_image_id[image_id] = normalized_row
            continue

        if existing == normalized_row:
            duplicate_collapsed_count += 1
            continue

        raise ValueError(
            f"{TASK_NAME} Conflicting duplicate split rows for image_id={image_id!r}. "
            f"First row={existing!r}; duplicate row={normalized_row!r}"
        )

    canonical_rows = sorted(
        canonical_by_image_id.values(),
        key=lambda r: r["image_id"],
    )

    return canonical_rows, duplicate_collapsed_count

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
            raise ValueError(
                f"{TASK_NAME} expected task_type=update_dataset, got {task_type!r}"
            )

        (
            dataset_id,
            operation,
            selection_config,
            split_approach,
            requested_split_strategy_name,
            description,
        ) = _assert_request_shape(request)

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            f"{TASK_NAME} Starting update flow for dataset_id={dataset_id}",
            level="info",
        )

        # 1) Load existing dataset metadata/invariants.
        dataset_state = get_dataset_info(
            datasets_table_name=DATASETS_TABLE_NAME,
            dataset_versions_table_name=DATASET_VERSIONS_TABLE_NAME,
            dataset_id=dataset_id,
        )

        if not dataset_state["dataset_info"].get("exists"):
            raise ValueError(f"Dataset '{dataset_id}' does not exist.")

        dataset_meta = dataset_state["dataset_info"]
        latest_meta = dataset_state["latest_version_info"]

        latest_version = dataset_meta["latest_version"]
        new_version = latest_version + 1
        label_type = dataset_meta["label_type"]
        honor_source_splits = dataset_meta["honor_source_splits"]

        dataset_allowed_classes = set(dataset_meta["allowed_classes"])
        requested_allowed_classes = set(selection_config["allowed_classes"])

        if not requested_allowed_classes.issubset(dataset_allowed_classes):
            raise ValueError(
                f"Requested update classes {sorted(requested_allowed_classes)} must be a subset of "
                f"{sorted(dataset_allowed_classes)}. To add classes, create a new dataset."
            )

        if label_type not in VALID_LABEL_TYPES:
            raise ValueError(f"Unsupported dataset label_type: {label_type!r}")

        effective_split_strategy_name, effective_split_mode = _resolve_effective_split_strategy_for_update(
            job_id=job_id,
            user=user,
            event_type=event_type,
            dataset_id=dataset_id,
            latest_meta=latest_meta,
            split_approach=split_approach,
            requested_split_strategy_name=requested_split_strategy_name,
            honor_source_splits=honor_source_splits,
        )

        effective_version_description = description

        dataset_membership_table_name = LABEL_TYPE_TO_MEMBERSHIP_TABLE.get(label_type)
        if dataset_membership_table_name is None:
            raise ValueError(f"Unsupported dataset label_type: {label_type!r}")

        # 2) Resolve selected imagery rows for add/remove operation.
        # For single-label updates, use dataset-wide allowed_classes for the SQL,
        # then filter back down to the requested subset after resolution.
        single_label_update_sc = deepcopy(selection_config)
        single_label_update_sc["allowed_classes"] = dataset_meta["allowed_classes"]

        selection_sql_for_update, selected_imagery_rows = resolve_candidate_imagery(
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            label_type=label_type,
            selection_config=selection_config if label_type != "single-label" else single_label_update_sc,
            athena_output_s3_uri=ATHENA_OUTPUT_S3,
            athena_workgroup=ATHENA_WORKGROUP,
            task_name=TASK_NAME,
        )

        if label_type == "single-label":
            selected_imagery_rows = [
                row
                for row in selected_imagery_rows
                if row["label"] in selection_config["allowed_classes"]
            ]

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

        # 3) Resolve current dataset membership rows.
        # Enriched mode is valid for both maintain and rebalance and simplifies downstream handling.
        membership_sql, current_rows = resolve_dataset_membership(
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            dataset_membership_table_name=dataset_membership_table_name,
            canonical_imagery_table_name=CANONICAL_IMAGERY_TABLE_NAME,
            dataset_id=dataset_id,
            version=latest_version,
            label_type=label_type,
            mode="enriched",
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

        # 4) Compute next-version rows and assign/preserve splits.
        split_rows, split_summary = update_dataset_splits(
            selected_imagery_rows=selected_imagery_rows,
            current_rows=current_rows,
            operation=operation,
            split_approach=split_approach,
            split_strategy_name=effective_split_strategy_name,
            honor_source_splits=honor_source_splits,
        )

        if not split_rows:
            raise ValueError(
                f"After {operation} operation, {dataset_id} had no rows for the newest updated version."
            )

        canonical_split_rows, duplicate_collapsed_count = _canonicalize_split_rows_or_raise(
            split_rows=split_rows,
        )

        if not canonical_split_rows:
            raise ValueError(
                f"{TASK_NAME} canonical split row set is empty for dataset_id={dataset_id}, version={new_version}"
            )

        if duplicate_collapsed_count > 0:
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                (
                    f"{TASK_NAME} Collapsed {duplicate_collapsed_count} exact duplicate split rows "
                    f"for dataset_id={dataset_id}, version={new_version}"
                ),
                level="warning",
            )

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            (
                f"{TASK_NAME} Computed {len(canonical_split_rows)} final split rows for version={new_version}, "
                f"effective_split_mode={split_summary['effective_split_mode']}, "
                f"excluded_unresolved={split_summary.get('excluded_unresolved_count', 0)}, "
                f"excluded_inconsistent={split_summary.get('excluded_inconsistent_count', 0)}"
            ),
            level="info",
        )

        # 5) Write versioned S3 artifacts first.
        artifact_result = write_s3_artifacts(
            dataset_bucket_name=DATASETS_BUCKET_NAME,
            dataset_id=dataset_id,
            version=new_version,
            label_type=label_type,
            split_strategy_name=effective_split_strategy_name,
            honor_source_splits=honor_source_splits,
            selection_sql=selection_sql_for_update,
            selection_config=selection_config,
            split_rows=canonical_split_rows,
        )

        # 6) Write new-version Iceberg membership rows.
        membership_result = write_dataset_membership(
            task_name=TASK_NAME,
            iceberg_database_name=ICEBERG_DATABASE_NAME,
            athena_output_s3_uri=ATHENA_OUTPUT_S3,
            athena_workgroup=ATHENA_WORKGROUP,
            dataset_id=dataset_id,
            version=new_version,
            dataset_label_type=label_type,
            split_rows=canonical_split_rows,
        )

        # 7) Write DDB version metadata and advance dataset latest_version.
        ddb_result = write_ddb_artifacts(
            new_dataset=False,
            datasets_table_name=DATASETS_TABLE_NAME,
            dataset_versions_table_name=DATASET_VERSIONS_TABLE_NAME,
            dataset_id=dataset_id,
            new_version=new_version,
            label_type=label_type,
            dataset_description=None,
            version_description=effective_version_description,
            split_strategy_name=effective_split_strategy_name,
            honor_source_splits=honor_source_splits,
            created_by=user,
            operation=operation,
            split_approach=split_approach,
            selection_config=selection_config,
            split_rows=canonical_split_rows,
            artifact_result=artifact_result,
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
                f"prior_count={len(current_rows)}, final_count={membership_result['row_count']}, "
                f"effective_split_mode={effective_split_mode}"
            ),
            level="info",
        )

        maybe_fail("update_fail")

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
            "description": effective_version_description,
            "honor_source_splits": honor_source_splits,
            "effective_split_strategy_name": effective_split_strategy_name,
            "effective_split_mode": effective_split_mode,
            "operation": operation,
            "split_approach": split_approach,
            f"candidate_imagery_count_to_{operation}": len(selected_imagery_rows),
            "preexisting_membership_count": len(current_rows),
            "canonical_split_row_count": len(canonical_split_rows),
            "duplicate_collapsed_count": duplicate_collapsed_count,
            "final_membership_row_count": membership_result["row_count"],
            "membership_table_name": membership_result["table_name"],
            "excluded_unresolved_count": split_summary.get("excluded_unresolved_count", 0),
            "excluded_inconsistent_count": split_summary.get("excluded_inconsistent_count", 0),
            "excluded_count": split_summary.get("excluded_count", 0),
            "artifact_result": artifact_result,
            "ddb_result": ddb_result,
        }

    except Exception as e:
        error_message = f"{TASK_NAME} Failed: {type(e).__name__}: {e}"

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            error_message,
            level="error",
        )
        raise