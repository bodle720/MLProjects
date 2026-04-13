from typing import Any, Literal

from common.dataset_utils.split_strategies.stratified_v1 import stratified_v1

Operation = Literal["add", "remove"]
SplitApproach = Literal["maintain", "rebalance"]

_STRUCTURED_TASK_TYPES = {
    "object-detection",
    "semantic-segmentation",
    "instance-segmentation",
}

_TASK_TYPE_TO_ID_FIELD = {
    "object-detection": "bbox_annotation_ids",
    "semantic-segmentation": "semantic_mask_ids",
    "instance-segmentation": "instance_annotation_ids",
}

_VALID_SPLITS = {"train", "val", "test"}
_VALID_SOURCE_SPLIT_STATUSES = {"resolved", "unresolved", "inconsistent"}

def update_dataset_splits(
    *,
    selected_imagery_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
    operation: Operation,
    split_approach: SplitApproach,
    split_strategy_name: str | None,
    honor_source_splits: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Compute the next dataset-version rows after applying an add/remove operation.

    Inputs:
    - selected_imagery_rows:
        output of resolve_candidate_imagery(...), already splitter-ready
    - current_rows:
        output of resolve_dataset_membership(...), which differs by task type:
          * single-label: label, no classes_present required in minimal mode
          * multi-label: labels, no classes_present required in minimal mode
          * structured tasks: payload + classes_present

    Semantics:
    - operation="add":
        next image universe = current ∪ selected, with overlap behavior:
          * single-label: keep current row unchanged
          * multi-label: enrich labels
          * structured tasks: enrich *_ids and classes_present

    - operation="remove":
        next image universe = current - selected

    - split_approach="maintain":
        preserve existing split for retained / merged existing rows
        assign splits only for truly new rows

    - split_approach="rebalance":
        recompute splits across the full next-version image universe

    Special honoring behavior:
    - honor_source_splits=True:
        * split_approach='rebalance' is forbidden
        * truly new rows are assigned from resolved_source_split
        * unresolved/inconsistent truly new rows are excluded
        * overlapping existing rows keep their current split regardless of selected-row
          source split status because they are already members of the dataset

    Output:
    - tuple of:
        1) final split_rows
        2) summary dict with split/exclusion metadata
    """
    _validate_operation(operation)
    _validate_split_approach(split_approach)

    if not isinstance(honor_source_splits, bool):
        raise ValueError("honor_source_splits must be a bool.")

    if not current_rows:
        raise ValueError("current_rows must not be empty.")

    if honor_source_splits and split_approach == "rebalance":
        raise ValueError(
            "split_approach='rebalance' is not allowed when honor_source_splits=True."
        )

    current_by_image_id = _index_rows_by_image_id(
        rows=current_rows,
        source_name="current_rows",
    )
    selected_by_image_id = _index_rows_by_image_id(
        rows=selected_imagery_rows,
        source_name="selected_imagery_rows",
    )

    current_ids = set(current_by_image_id.keys())
    selected_ids = set(selected_by_image_id.keys())

    if operation == "remove":
        retained_ids = current_ids - selected_ids
        retained_rows = [
            dict(current_by_image_id[image_id])
            for image_id in sorted(retained_ids)
        ]
        added_rows: list[dict[str, Any]] = []

    else:  # add
        retained_rows = []
        added_rows = []

        all_ids = sorted(current_ids | selected_ids)
        for image_id in all_ids:
            current_row = current_by_image_id.get(image_id)
            selected_row = selected_by_image_id.get(image_id)

            if current_row is not None and selected_row is not None:
                retained_rows.append(
                    _merge_overlapping_rows_for_add(
                        current_row=current_row,
                        selected_row=selected_row,
                    )
                )
            elif current_row is not None:
                retained_rows.append(dict(current_row))
            else:
                # truly new image
                added_rows.append(dict(selected_row))

    if len(retained_rows) + len(added_rows) == 0:
        raise ValueError("Update would produce an empty dataset version.")

    if split_approach == "maintain":
        final_rows, summary = _build_maintained_split_rows(
            retained_rows=retained_rows,
            added_rows=added_rows,
            split_strategy_name=split_strategy_name,
            honor_source_splits=honor_source_splits,
        )
    else:
        final_rows, summary = _build_rebalanced_split_rows(
            retained_rows=retained_rows,
            added_rows=added_rows,
            split_strategy_name=split_strategy_name,
            honor_source_splits=honor_source_splits,
        )

    summary.update(
        {
            "operation": operation,
            "split_approach": split_approach,
            "honor_source_splits": honor_source_splits,
            "retained_input_count": len(retained_rows),
            "added_input_count": len(added_rows),
            "selected_input_count": len(selected_imagery_rows),
            "current_input_count": len(current_rows),
            "final_row_count": len(final_rows),
        }
    )

    return final_rows, summary

def _build_maintained_split_rows(
    *,
    retained_rows: list[dict[str, Any]],
    added_rows: list[dict[str, Any]],
    split_strategy_name: str | None,
    honor_source_splits: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Preserve existing splits for retained rows. Only truly new rows are assigned.
    """
    _require_rows_have_existing_split(retained_rows)

    if honor_source_splits:
        assigned_new_rows, assign_summary = _assign_honored_source_splits(rows=added_rows)
        out = retained_rows + assigned_new_rows
        return _finalize_rows_for_output(out), {
            "effective_split_mode": "honor_source_splits",
            **assign_summary,
        }

    prepared_added_rows = _prepare_rows_for_splitter(added_rows)
    _require_rows_have_splitter_fields(prepared_added_rows)

    assigned_new_rows = _assign_splits(
        rows=prepared_added_rows,
        split_strategy_name=split_strategy_name,
    )

    out = retained_rows + assigned_new_rows
    return _finalize_rows_for_output(out), {
        "effective_split_mode": split_strategy_name,
        "assigned_new_rows_count": len(assigned_new_rows),
        "excluded_unresolved_count": 0,
        "excluded_inconsistent_count": 0,
        "excluded_count": 0,
    }

def _build_rebalanced_split_rows(
    *,
    retained_rows: list[dict[str, Any]],
    added_rows: list[dict[str, Any]],
    split_strategy_name: str | None,
    honor_source_splits: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Recompute splits across the full next-version image universe.

    Only valid when honor_source_splits=False.
    """
    if honor_source_splits:
        raise ValueError(
            "Rebalance is not allowed when honor_source_splits=True."
        )

    final_rows = retained_rows + added_rows
    prepared_rows = _prepare_rows_for_splitter(final_rows)
    _require_rows_have_splitter_fields(prepared_rows)

    assigned_rows = _assign_splits(
        rows=prepared_rows,
        split_strategy_name=split_strategy_name,
    )
    return _finalize_rows_for_output(assigned_rows), {
        "effective_split_mode": split_strategy_name,
        "assigned_new_rows_count": len(added_rows),
        "excluded_unresolved_count": 0,
        "excluded_inconsistent_count": 0,
        "excluded_count": 0,
    }

def _merge_overlapping_rows_for_add(
    *,
    current_row: dict[str, Any],
    selected_row: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge overlapping rows for operation='add'.

    Rules:
    - single-label:
        keep current row unchanged
    - multi-label:
        enrich labels (deduped)
    - structured tasks:
        enrich task-specific *_ids and classes_present (deduped)

    Existing split is always preserved on the current row for maintain flows.
    Rebalance flows will recompute split later anyway.
    """
    merged = dict(current_row)

    current_label_type = _require_nonempty_string(
        current_row.get("dataset_label_type"),
        field_name="dataset_label_type",
    )
    selected_label_type = _require_nonempty_string(
        selected_row.get("dataset_label_type"),
        field_name="dataset_label_type",
    )

    if current_label_type != selected_label_type:
        raise ValueError(
            f"Mismatched dataset_label_type for overlapping image_id "
            f"{current_row.get('image_id')!r}: current={current_label_type!r}, "
            f"selected={selected_label_type!r}"
        )

    if current_label_type == "single-label":
        # Keep existing membership row unchanged; do not enrich scalar label.
        return merged

    if current_label_type == "multi-label":
        current_labels = _normalize_nonempty_string_array(
            current_row.get("labels"),
            field_name="labels",
        )
        selected_labels = _normalize_nonempty_string_array(
            selected_row.get("labels"),
            field_name="labels",
        )
        merged_labels = _merge_string_arrays(current_labels, selected_labels)
        merged["labels"] = merged_labels
        merged["classes_present"] = list(merged_labels)
        return merged

    if current_label_type in _STRUCTURED_TASK_TYPES:
        id_field = _TASK_TYPE_TO_ID_FIELD[current_label_type]

        current_ids = _normalize_nonempty_string_array(
            current_row.get(id_field),
            field_name=id_field,
        )
        selected_ids = _normalize_nonempty_string_array(
            selected_row.get(id_field),
            field_name=id_field,
        )
        merged[id_field] = _merge_string_arrays(current_ids, selected_ids)

        current_classes = _normalize_nonempty_string_array(
            current_row.get("classes_present"),
            field_name="classes_present",
        )
        selected_classes = _normalize_nonempty_string_array(
            selected_row.get("classes_present"),
            field_name="classes_present",
        )
        merged["classes_present"] = _merge_string_arrays(
            current_classes,
            selected_classes,
        )
        return merged

    raise ValueError(f"Unsupported dataset_label_type: {current_label_type!r}")

def _assign_splits(
    *,
    rows: list[dict[str, Any]],
    split_strategy_name: str | None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    if split_strategy_name == "stratified_v1":
        assigned_rows = stratified_v1(candidates=rows)
        return _sort_rows_by_image_id([dict(row) for row in assigned_rows])

    raise ValueError(f"Split strategy {split_strategy_name!r} not supported.")

def _assign_honored_source_splits(
    *,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Assign splits from resolved_source_split for truly new rows.

    Rows with:
    - source_split_status='resolved' are kept and assigned
    - source_split_status='unresolved' are excluded
    - source_split_status='inconsistent' are excluded
    """
    assigned_rows: list[dict[str, Any]] = []
    unresolved_count = 0
    inconsistent_count = 0

    for row in rows:
        status = _optional_string(row.get("source_split_status"))
        resolved_source_split = _optional_string(row.get("resolved_source_split"))

        if status not in _VALID_SOURCE_SPLIT_STATUSES:
            raise ValueError(
                f"Invalid source_split_status={status!r} for image_id={row.get('image_id')!r}"
            )

        if status == "resolved":
            if resolved_source_split not in _VALID_SPLITS:
                raise ValueError(
                    f"Resolved row missing valid resolved_source_split for image_id="
                    f"{row.get('image_id')!r}: {resolved_source_split!r}"
                )
            assigned_rows.append({**row, "split": resolved_source_split})
            continue

        if status == "unresolved":
            unresolved_count += 1
            continue

        if status == "inconsistent":
            inconsistent_count += 1
            continue

    return _sort_rows_by_image_id(assigned_rows), {
        "assigned_new_rows_count": len(assigned_rows),
        "excluded_unresolved_count": unresolved_count,
        "excluded_inconsistent_count": inconsistent_count,
        "excluded_count": unresolved_count + inconsistent_count,
    }

def _prepare_rows_for_splitter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ensure every row is splitter-ready by synthesizing classes_present where needed.

    Rules:
    - single-label: classes_present = [label]
    - multi-label: classes_present = labels
    - structured tasks: use existing classes_present
    """
    out: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        copied = dict(row)

        dataset_label_type = _require_nonempty_string(
            copied.get("dataset_label_type"),
            field_name="dataset_label_type",
        )

        if dataset_label_type == "single-label":
            label = _require_nonempty_string(
                copied.get("label"),
                field_name="label",
            )
            copied["classes_present"] = [label]

        elif dataset_label_type == "multi-label":
            labels = _normalize_nonempty_string_array(
                copied.get("labels"),
                field_name="labels",
            )
            copied["labels"] = labels
            copied["classes_present"] = list(labels)

        elif dataset_label_type in _STRUCTURED_TASK_TYPES:
            classes_present = _normalize_nonempty_string_array(
                copied.get("classes_present"),
                field_name="classes_present",
            )
            copied["classes_present"] = classes_present

            id_field = _TASK_TYPE_TO_ID_FIELD[dataset_label_type]
            copied[id_field] = _normalize_nonempty_string_array(
                copied.get(id_field),
                field_name=id_field,
            )

        else:
            raise ValueError(
                f"Unsupported dataset_label_type on row {idx}: {dataset_label_type!r}"
            )

        out.append(copied)

    return out

def _finalize_rows_for_output(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize all final rows before returning them to downstream writers.

    This guarantees that:
    - split is present and valid on every output row
    - single-label and multi-label rows always carry classes_present
    - structured rows have normalized classes_present and *_ids
    """
    prepared = _prepare_rows_for_splitter(rows)
    _require_rows_have_existing_split(prepared)
    return _sort_rows_by_image_id(prepared)

def _index_rows_by_image_id(
    *,
    rows: list[dict[str, Any]],
    source_name: str,
) -> dict[str, dict[str, Any]]:
    """
    Build an image_id -> row mapping and fail fast on duplicates.
    """
    indexed: dict[str, dict[str, Any]] = {}

    for idx, row in enumerate(rows):
        image_id = _require_nonempty_string(row.get("image_id"), field_name="image_id")

        if image_id in indexed:
            raise ValueError(
                f"{source_name} contains duplicate image_id '{image_id}' at row {idx}."
            )

        indexed[image_id] = dict(row)

    return indexed

def _require_rows_have_existing_split(rows: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(rows):
        split = row.get("split")
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"Row {idx} is missing a valid split: {row!r}"
            )

def _require_rows_have_splitter_fields(rows: list[dict[str, Any]]) -> None:
    """
    stratified_v1 requires:
    - image_id
    - dataset_label_type
    - classes_present (non-empty list)
    """
    required_fields = {
        "image_id",
        "dataset_label_type",
        "classes_present",
    }

    for idx, row in enumerate(rows):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise ValueError(
                f"splitter row {idx} missing required fields {missing}: {row!r}"
            )

        classes_present = row.get("classes_present")
        if not isinstance(classes_present, list) or len(classes_present) == 0:
            raise ValueError(
                f"splitter row {idx} must have non-empty classes_present: {row!r}"
            )

def _sort_rows_by_image_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: _require_nonempty_string(
            row.get("image_id"),
            field_name="image_id",
        ),
    )

def _merge_string_arrays(left: list[str], right: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for item in list(left) + list(right):
        text = str(item).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)

    if not out:
        raise ValueError("Merged array must not be empty.")

    return out

def _normalize_nonempty_string_array(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a non-empty list[str].")

    out: list[str] = []
    seen: set[str] = set()

    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)

    if not out:
        raise ValueError(f"{field_name} must be a non-empty list[str].")

    return out

def _validate_operation(operation: str) -> None:
    if operation not in {"add", "remove"}:
        raise ValueError(f"Unsupported operation: {operation!r}")

def _validate_split_approach(split_approach: str) -> None:
    if split_approach not in {"maintain", "rebalance"}:
        raise ValueError(f"Unsupported split_approach: {split_approach!r}")

def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be None.")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty.")

    return text

def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None