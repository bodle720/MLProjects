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

def update_dataset_splits(*,
                            selected_imagery_rows: list[dict[str, Any]],
                            current_rows: list[dict[str, Any]],
                            operation: Operation,
                            split_approach: SplitApproach,
                            split_strategy_name: str,
                        ) -> list[dict[str, Any]]:
    """
    Compute the next dataset-version rows after applying an add/remove operation.

    Inputs:
    - selected_imagery_rows:
        output of resolve_candidate_imagery(...), already splitter-ready
    - current_rows:
        output of resolve_dataset_membership(...), which differs by task type:
          * single-label: label, no classes_present
          * multi-label: labels, no classes_present
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
    """
    _validate_operation(operation)
    _validate_split_approach(split_approach)

    if not current_rows:
        raise ValueError("current_rows must not be empty.")

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
        retained_rows = [dict(current_by_image_id[image_id]) for image_id in sorted(retained_ids)]
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
        return _build_maintained_split_rows(
            retained_rows=retained_rows,
            added_rows=added_rows,
            split_strategy_name=split_strategy_name,
        )

    return _build_rebalanced_split_rows(
        retained_rows=retained_rows,
        added_rows=added_rows,
        split_strategy_name=split_strategy_name,
    )

def _build_maintained_split_rows(
    *,
    retained_rows: list[dict[str, Any]],
    added_rows: list[dict[str, Any]],
    split_strategy_name: str,
) -> list[dict[str, Any]]:
    """
    Preserve existing splits for retained rows. Only truly new rows are assigned.
    """
    _require_rows_have_existing_split(retained_rows)

    prepared_added_rows = _prepare_rows_for_rebalance(added_rows)
    _require_rows_have_rebalance_fields(prepared_added_rows)

    assigned_new_rows = _assign_splits(
        rows=prepared_added_rows,
        split_strategy_name=split_strategy_name,
    )

    out = retained_rows + assigned_new_rows
    return _sort_rows_by_image_id(out)

def _build_rebalanced_split_rows(
    *,
    retained_rows: list[dict[str, Any]],
    added_rows: list[dict[str, Any]],
    split_strategy_name: str,
) -> list[dict[str, Any]]:
    """
    Recompute splits across the full next-version image universe.
    """
    final_rows = retained_rows + added_rows
    prepared_rows = _prepare_rows_for_rebalance(final_rows)
    _require_rows_have_rebalance_fields(prepared_rows)

    return _assign_splits(
        rows=prepared_rows,
        split_strategy_name=split_strategy_name,
    )

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

        # Helpful for rebalance; harmless if present in maintain output.
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
        merged["classes_present"] = _merge_string_arrays(current_classes, selected_classes)
        return merged

    raise ValueError(f"Unsupported dataset_label_type: {current_label_type!r}")

def _assign_splits(
    *,
    rows: list[dict[str, Any]],
    split_strategy_name: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    if split_strategy_name == "stratified_v1":
        assigned_rows = stratified_v1(candidates=rows)
        return _sort_rows_by_image_id([dict(row) for row in assigned_rows])

    raise ValueError(f"Split strategy '{split_strategy_name}' not supported.")

def _prepare_rows_for_rebalance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Retained row {idx} is missing a valid split: {row!r}"
            )

def _require_rows_have_rebalance_fields(rows: list[dict[str, Any]]) -> None:
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
                f"rebalance row {idx} missing required fields {missing}: {row!r}"
            )

        classes_present = row.get("classes_present")
        if not isinstance(classes_present, list) or len(classes_present) == 0:
            raise ValueError(
                f"rebalance row {idx} must have non-empty classes_present: {row!r}"
            )

def _sort_rows_by_image_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: _require_nonempty_string(row.get("image_id"), field_name="image_id"),
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