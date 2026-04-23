import os
from typing import Any

from common.general_utils.logging_utils import log
from common.general_utils.ddb_utils import dataset_exists
from common.dataset_utils.resolve_candidate_imagery import resolve_candidate_imagery
from common.dataset_utils.dataset_iceberg_utils import write_dataset_membership
from common.dataset_utils.dataset_s3_utils import write_s3_artifacts
from common.dataset_utils.dataset_ddb_utils import write_ddb_artifacts
from common.dataset_utils.split_strategies.stratified_v1 import stratified_v1
from common.testing_utils.dataset_testing import maybe_fail

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

_VALID_SPLITS = {"train", "val", "test"}
_VALID_SOURCE_SPLIT_STATUSES = {"resolved", "unresolved", "inconsistent"}


def _assert_request_shape(
    request: dict[str, Any],
) -> tuple[str, str, str | None, dict[str, Any], str | None, bool]:
    dataset_id = _require_nonempty_string(
        request.get("dataset_id"),
        field_name="request.dataset_id",
    )
    label_type = _require_nonempty_string(
        request.get("label_type"),
        field_name="request.label_type",
    )

    if label_type not in VALID_LABEL_TYPES:
        raise ValueError(f"Unsupported request.label_type: {label_type}")

    selection_config = request.get("selection_config")
    if not isinstance(selection_config, dict):
        raise ValueError("request.selection_config must be an object")

    description = request.get("description")
    if description is not None:
        if not isinstance(description, str):
            raise ValueError("request.description must be a string or null")
        description = description.strip() or None

    split_strategy_name = request.get("split_strategy_name")
    if split_strategy_name is not None:
        if not isinstance(split_strategy_name, str):
            raise ValueError("request.split_strategy_name must be a string or null")
        split_strategy_name = split_strategy_name.strip() or None

    honor_source_splits = request.get("honor_source_splits")
    if not isinstance(honor_source_splits, bool):
        raise ValueError("request.honor_source_splits must be a bool")

    return (
        dataset_id,
        label_type,
        description,
        selection_config,
        split_strategy_name,
        honor_source_splits,
    )


def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_honor_source_split_rows(
    *,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    split_rows: list[dict[str, Any]] = []
    resolved_count = 0
    unresolved_count = 0
    inconsistent_count = 0

    for row in candidates:
        status = _optional_string(row.get("source_split_status"))
        resolved_source_split = _optional_string(row.get("resolved_source_split"))

        if status not in _VALID_SOURCE_SPLIT_STATUSES:
            raise ValueError(
                f"{TASK_NAME} Invalid source_split_status={status!r} for image_id={row.get('image_id')!r}"
            )

        if status == "resolved":
            if resolved_source_split not in _VALID_SPLITS:
                raise ValueError(
                    f"{TASK_NAME} Resolved candidate missing valid resolved_source_split "
                    f"for image_id={row.get('image_id')!r}: {resolved_source_split!r}"
                )

            split_rows.append({**row, "split": resolved_source_split})
            resolved_count += 1
            continue

        if status == "unresolved":
            unresolved_count += 1
            continue

        if status == "inconsistent":
            inconsistent_count += 1
            continue

    summary = {
        "resolved_count": resolved_count,
        "unresolved_count": unresolved_count,
        "inconsistent_count": inconsistent_count,
        "excluded_count": unresolved_count + inconsistent_count,
    }

    return split_rows, summary


def _canonicalize_split_rows_or_raise(
    *,
    split_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    Enforce one canonical split row per image_id for this dataset version.

    Behavior:
    - exact duplicate row for same image_id -> collapse
    - conflicting row for same image_id -> raise

    Returns:
    - canonical_rows sorted deterministically by image_id
    - duplicate_collapsed_count
    """
    canonical_by_image_id: dict[str, dict[str, Any]] = {}
    duplicate_collapsed_count = 0

    for row in split_rows:
        image_id = _require_nonempty_string(row.get("image_id"), field_name="split_rows[].image_id")
        split = _require_nonempty_string(row.get("split"), field_name="split_rows[].split")
        if split not in _VALID_SPLITS:
            raise ValueError(f"{TASK_NAME} Invalid split in split_rows for image_id={image_id}: {split!r}")

        existing = canonical_by_image_id.get(image_id)
        if existing is None:
            canonical_by_image_id[image_id] = row
            continue

        if existing == row:
            duplicate_collapsed_count += 1
            continue

        raise ValueError(
            f"{TASK_NAME} Conflicting duplicate split rows for image_id={image_id!r}. "
            f"First row={existing!r}; duplicate row={row!r}"
        )

    canonical_rows = sorted(
        canonical_by_image_id.values(),
        key=lambda r: str(r["image_id"]).strip(),
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

        if task_type != "create_dataset":
            raise ValueError(f"{TASK_NAME} expected task_type=create_dataset, got {task_type!r}")

        (
            dataset_id,
            label_type,
            description,
            selection_config,
            split_strategy_name,
            honor_source_splits,
        ) = _assert_request_shape(request)

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            (
                f"{TASK_NAME} Starting create flow for dataset_id={dataset_id}, "
                f"label_type={label_type}, honor_source_splits={honor_source_splits}"
            ),
            level="info",
        )

        # Authoritative server-side uniqueness check.
        if dataset_exists(dataset_id, DATASETS_TABLE_NAME):
            raise ValueError(f"Dataset '{dataset_id}' already exists.")

        # Resolve candidate imagery from canonical/provenance tables.
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
        exclusion_summary = {
            "resolved_count": 0,
            "unresolved_count": 0,
            "inconsistent_count": 0,
            "excluded_count": 0,
        }

        if honor_source_splits:
            if split_strategy_name:
                log(
                    job_id,
                    user,
                    event_type,
                    LOG_FIREHOSE_STREAM_NAME,
                    (
                        f"{TASK_NAME} honor_source_splits=True, so split_strategy_name="
                        f"{split_strategy_name!r} is accepted but ignored. "
                        f"Splits will be assigned from image_source_membership.source_split."
                    ),
                    level="info",
                )

            split_rows, exclusion_summary = _build_honor_source_split_rows(
                candidates=candidates,
            )

            if not split_rows:
                raise ValueError(
                    f"{TASK_NAME} honor_source_splits=True but zero candidates had a resolved "
                    f"source split after excluding unresolved/inconsistent rows."
                )

            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                (
                    f"{TASK_NAME} Honored source splits for dataset_id={dataset_id}: "
                    f"kept={exclusion_summary['resolved_count']}, "
                    f"excluded_unresolved={exclusion_summary['unresolved_count']}, "
                    f"excluded_inconsistent={exclusion_summary['inconsistent_count']}"
                ),
                level="info",
            )

        else:
            if split_strategy_name != "stratified_v1":
                raise ValueError(
                    f"Split strategy {split_strategy_name!r} not supported when "
                    f"honor_source_splits=False. Expected 'stratified_v1'."
                )

            split_rows = stratified_v1(candidates=candidates)

            if not split_rows:
                raise ValueError(
                    f"{TASK_NAME} stratified_v1 returned zero split rows for dataset_id={dataset_id}"
                )

            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                f"{TASK_NAME} Assigned splits with stratified_v1 for {len(split_rows)} rows",
                level="info",
            )

        # Canonicalize once so Iceberg/S3/DDB all see the same exact row set.
        canonical_split_rows, duplicate_collapsed_count = _canonicalize_split_rows_or_raise(
            split_rows=split_rows,
        )

        if not canonical_split_rows:
            raise ValueError(f"{TASK_NAME} canonical split row set is empty for dataset_id={dataset_id}")

        if duplicate_collapsed_count > 0:
            log(
                job_id,
                user,
                event_type,
                LOG_FIREHOSE_STREAM_NAME,
                (
                    f"{TASK_NAME} Collapsed {duplicate_collapsed_count} exact duplicate split rows "
                    f"for dataset_id={dataset_id}"
                ),
                level="warning",
            )

        # Write S3 artifacts first: safer to leave orphaned version artifacts than
        # to leave membership rows if a later S3 write fails.
        artifact_result = write_s3_artifacts(
            dataset_bucket_name=DATASETS_BUCKET_NAME,
            dataset_id=dataset_id,
            version=1,
            label_type=label_type,
            split_strategy_name=split_strategy_name,
            honor_source_splits=honor_source_splits,
            selection_sql=selection_sql,
            selection_config=selection_config,
            split_rows=canonical_split_rows,
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
            split_rows=canonical_split_rows,
        )

        # DDB dataset + version metadata
        ddb_result = write_ddb_artifacts(
            new_dataset=True,
            datasets_table_name=DATASETS_TABLE_NAME,
            dataset_versions_table_name=DATASET_VERSIONS_TABLE_NAME,
            dataset_id=dataset_id,
            new_version=1,
            label_type=label_type,
            dataset_description=description,
            version_description=description,
            split_strategy_name=split_strategy_name,
            honor_source_splits=honor_source_splits,
            created_by=user,
            operation="create",
            split_approach="initial",
            selection_config=selection_config,
            split_rows=canonical_split_rows,
            artifact_result=artifact_result,
        )

        maybe_fail("create_fail")

        effective_split_mode = (
            "honor_source_splits"
            if honor_source_splits
            else split_strategy_name
        )

        log(
            job_id,
            user,
            event_type,
            LOG_FIREHOSE_STREAM_NAME,
            (
                f"{TASK_NAME} Completed create flow for dataset_id={dataset_id}, "
                f"version=1, candidates={len(candidates)}, "
                f"canonical_split_rows={len(canonical_split_rows)}, "
                f"membership_rows={membership_result['row_count']}, "
                f"effective_split_mode={effective_split_mode}"
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
            "honor_source_splits": honor_source_splits,
            "split_strategy_name": split_strategy_name,
            "effective_split_mode": effective_split_mode,
            "candidate_count": len(candidates),
            "canonical_split_row_count": len(canonical_split_rows),
            "duplicate_collapsed_count": duplicate_collapsed_count,
            "membership_row_count": membership_result["row_count"],
            "membership_table_name": membership_result["table_name"],
            "excluded_unresolved_count": exclusion_summary["unresolved_count"],
            "excluded_inconsistent_count": exclusion_summary["inconsistent_count"],
            "excluded_count": exclusion_summary["excluded_count"],
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