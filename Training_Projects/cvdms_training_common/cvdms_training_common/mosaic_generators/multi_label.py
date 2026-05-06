"""
Multi-label mosaic-generation utilities for CVDMS training projects.

This module contains task-specific logic for multi-label classification mosaics:

    - extract source_ref and labels from CVDMS manifest rows
    - validate multi-label rows
    - sort rows deterministically by label cardinality/signature
    - optionally group mosaics by:
        * none
        * cardinality
        * exact label signature
    - render/save mosaic sheets using mosaic_generators.common_utils

No text, labels, or image IDs are drawn onto the mosaic images themselves.
Labels are used only for ordering, grouping, folder layout, and output filenames.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from cvdms_training_common.mosaic_generators.common_utils import (
    ImageLoaderFn,
    MosaicConfig,
    MosaicItem,
    MosaicSheetResult,
    safe_filename_part,
    save_mosaic_sheets,
)

_ALLOWED_ORDER_STRATEGIES = {
    "cardinality_signature",
    "source_ref",
    "image_id",
    "random",
}

_ALLOWED_GROUP_MODES = {
    "none",
    "cardinality",
    "signature",
}

@dataclass(frozen=True)
class MultiLabelMosaicRecord:
    """
    Normalized multi-label manifest row used for mosaic generation.
    """

    source_ref: str
    labels: tuple[str, ...]
    image_id: str | None = None
    split: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, str) or not self.source_ref.strip():
            raise ValueError("source_ref must be a non-empty string")

        if not isinstance(self.labels, tuple):
            raise TypeError(f"labels must be a tuple[str, ...], got {type(self.labels).__name__}")

        for idx, label in enumerate(self.labels):
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"labels[{idx}] must be a non-empty string")

        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"labels must be unique, got {self.labels!r}")

        if self.image_id is not None and not str(self.image_id).strip():
            raise ValueError("image_id cannot be empty when provided")

        if self.split is not None and not str(self.split).strip():
            raise ValueError("split cannot be empty when provided")

        if not isinstance(self.raw, dict):
            raise TypeError(f"raw must be a dictionary, got {type(self.raw).__name__}")

    @property
    def cardinality(self) -> int:
        return len(self.labels)

    @property
    def label_signature(self) -> str:
        if not self.labels:
            return "no_labels"
        return "+".join(self.labels)

    @property
    def safe_label_signature(self) -> str:
        return safe_filename_part(self.label_signature)

    def to_mosaic_item(self) -> MosaicItem:
        return MosaicItem(
            source_ref=self.source_ref,
            metadata={
                "image_id": self.image_id,
                "split": self.split,
                "labels": list(self.labels),
                "label_cardinality": self.cardinality,
                "label_signature": self.label_signature,
            },
        )

    def sort_key_cardinality_signature(self) -> tuple[int, tuple[str, ...], str, str]:
        return (
            self.cardinality,
            self.labels,
            self.image_id or "",
            self.source_ref,
        )

    def sort_key_source_ref(self) -> tuple[str, str]:
        return self.source_ref, self.image_id or ""

    def sort_key_image_id(self) -> tuple[str, str]:
        return self.image_id or "", self.source_ref

@dataclass(frozen=True)
class MultiLabelMosaicResult:
    """
    Result summary for a multi-label mosaic generation run.
    """

    split: str
    output_dir: Path
    order_strategy: str
    group_mode: str
    total_items: int
    group_counts: dict[str, int]
    sheets: list[MosaicSheetResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "output_dir": str(self.output_dir),
            "order_strategy": self.order_strategy,
            "group_mode": self.group_mode,
            "total_items": self.total_items,
            "group_counts": dict(self.group_counts),
            "sheets": [sheet.to_dict() for sheet in self.sheets],
        }

def generate_multi_label_split_mosaics(
    *,
    rows: Sequence[Any],
    image_loader: ImageLoaderFn,
    output_dir: str | Path,
    split: str,
    config: MosaicConfig | None = None,
    class_to_idx: dict[str, int] | None = None,
    order_strategy: str = "cardinality_signature",
    group_mode: str = "none",
    group_by_cardinality: bool | None = None,
    random_seed: int = 42,
    max_items: int | None = None,
    allow_empty_labels: bool = False,
    validate_split: bool = True,
) -> MultiLabelMosaicResult:
    """
    Generate mosaic sheets for one multi-label split.

    Args:
        rows:
            Manifest rows. Each row may be a dict or an object with attributes
            such as source_ref, split, image_id, and raw.
        image_loader:
            Callable that loads PIL images from source_ref.
        output_dir:
            Base output directory. The split name is appended automatically.
        split:
            Split name, usually train/val/test.
        config:
            MosaicConfig. Defaults to 10x10 tiles of 128x128 pixels.
        class_to_idx:
            Optional class map used to validate labels.
        order_strategy:
            One of:
                cardinality_signature
                source_ref
                image_id
                random
        group_mode:
            One of:
                none
                cardinality
                signature
        group_by_cardinality:
            Backward-compatibility bridge. If provided and group_mode remains
            "none", True maps to "cardinality" and False maps to "none".
        random_seed:
            Seed used only when order_strategy="random".
        max_items:
            Optional cap after ordering input rows. Useful for previews.
        allow_empty_labels:
            If False, rows with no labels are rejected.
        validate_split:
            If True and a row has a split field, it must match the requested split.
    """
    resolved_config = config or MosaicConfig()
    normalized_split = _normalize_nonempty_string(split, "split")
    _validate_order_strategy(order_strategy)

    resolved_group_mode = _resolve_group_mode(
        group_mode=group_mode,
        group_by_cardinality=group_by_cardinality,
    )

    records = make_multi_label_records(
        rows=rows,
        split=normalized_split,
        class_to_idx=class_to_idx,
        allow_empty_labels=allow_empty_labels,
        validate_split=validate_split,
    )

    records = order_multi_label_records(
        records=records,
        order_strategy=order_strategy,
        random_seed=random_seed,
    )

    if max_items is not None:
        _validate_positive_int(max_items, "max_items")
        records = records[:max_items]

    if not records:
        raise ValueError(f"No records available for split {normalized_split!r}")

    split_output_dir = Path(output_dir) / safe_filename_part(normalized_split)
    all_sheets: list[MosaicSheetResult] = []
    group_counts: dict[str, int] = {}

    if resolved_group_mode == "none":
        group_counts["all"] = len(records)

        items = [record.to_mosaic_item() for record in records]
        prefix = f"{normalized_split}__order-{order_strategy}"

        sheets = save_mosaic_sheets(
            items=items,
            image_loader=image_loader,
            output_dir=split_output_dir,
            filename_prefix=prefix,
            config=resolved_config,
        )
        all_sheets.extend(sheets)

    elif resolved_group_mode == "cardinality":
        grouped = group_records_by_cardinality(records)

        for cardinality, group_records in grouped.items():
            group_name = f"card-{cardinality:02d}"
            group_counts[group_name] = len(group_records)

            items = [record.to_mosaic_item() for record in group_records]
            group_output_dir = split_output_dir / group_name
            prefix = (
                f"{normalized_split}"
                f"__{group_name}"
                f"__order-{order_strategy}"
            )

            sheets = save_mosaic_sheets(
                items=items,
                image_loader=image_loader,
                output_dir=group_output_dir,
                filename_prefix=prefix,
                config=resolved_config,
            )
            all_sheets.extend(sheets)

    elif resolved_group_mode == "signature":
        grouped = group_records_by_signature(records)

        for (cardinality, signature), group_records in grouped.items():
            card_name = f"card-{cardinality:02d}"
            sig_name = signature
            group_key = f"{card_name}/{sig_name}"
            group_counts[group_key] = len(group_records)

            items = [record.to_mosaic_item() for record in group_records]
            group_output_dir = split_output_dir / card_name
            prefix = (
                f"{normalized_split}"
                f"__{card_name}"
                f"__sig-{safe_filename_part(sig_name)}"
            )

            sheets = save_mosaic_sheets(
                items=items,
                image_loader=image_loader,
                output_dir=group_output_dir,
                filename_prefix=prefix,
                config=resolved_config,
            )
            all_sheets.extend(sheets)

    else:  # pragma: no cover
        raise ValueError(f"Unsupported group_mode={resolved_group_mode!r}")

    return MultiLabelMosaicResult(
        split=normalized_split,
        output_dir=split_output_dir,
        order_strategy=order_strategy,
        group_mode=resolved_group_mode,
        total_items=len(records),
        group_counts=group_counts,
        sheets=all_sheets,
    )

def make_multi_label_records(
    *,
    rows: Sequence[Any],
    split: str | None = None,
    class_to_idx: dict[str, int] | None = None,
    allow_empty_labels: bool = False,
    validate_split: bool = True,
) -> list[MultiLabelMosaicRecord]:
    """
    Convert manifest rows into normalized MultiLabelMosaicRecord objects.
    """
    records: list[MultiLabelMosaicRecord] = []

    for idx, row in enumerate(rows):
        record = multi_label_record_from_row(
            row=row,
            row_index=idx,
            split=split,
            class_to_idx=class_to_idx,
            allow_empty_labels=allow_empty_labels,
            validate_split=validate_split,
        )
        records.append(record)

    return records

def multi_label_record_from_row(
    *,
    row: Any,
    row_index: int,
    split: str | None = None,
    class_to_idx: dict[str, int] | None = None,
    allow_empty_labels: bool = False,
    validate_split: bool = True,
) -> MultiLabelMosaicRecord:
    """
    Normalize one manifest row into a MultiLabelMosaicRecord.
    """
    raw = _raw_dict_from_row(row)
    source_ref = _source_ref_from_row(row=row, raw=raw, row_index=row_index)
    image_id = _optional_string_from_row(row=row, raw=raw, keys=("image_id", "image-id"))
    row_split = _optional_string_from_row(row=row, raw=raw, keys=("split",))

    if split is not None and validate_split and row_split is not None:
        if row_split != split:
            raise ValueError(
                f"Row {row_index} split mismatch: expected {split!r}, got {row_split!r}"
            )

    labels = _labels_from_row(
        row=row,
        raw=raw,
        row_index=row_index,
        allow_empty_labels=allow_empty_labels,
    )

    if class_to_idx is not None:
        _validate_labels_in_class_map(
            labels=labels,
            class_to_idx=class_to_idx,
            row_index=row_index,
        )

    return MultiLabelMosaicRecord(
        source_ref=source_ref,
        labels=labels,
        image_id=image_id,
        split=row_split or split,
        raw=raw,
    )

def order_multi_label_records(
    *,
    records: Sequence[MultiLabelMosaicRecord],
    order_strategy: str = "cardinality_signature",
    random_seed: int = 42,
) -> list[MultiLabelMosaicRecord]:
    """
    Order multi-label records for visually interpretable mosaics.
    """
    _validate_order_strategy(order_strategy)
    ordered = list(records)

    if order_strategy == "cardinality_signature":
        return sorted(ordered, key=lambda record: record.sort_key_cardinality_signature())

    if order_strategy == "source_ref":
        return sorted(ordered, key=lambda record: record.sort_key_source_ref())

    if order_strategy == "image_id":
        return sorted(ordered, key=lambda record: record.sort_key_image_id())

    if order_strategy == "random":
        rng = random.Random(random_seed)
        rng.shuffle(ordered)
        return ordered

    raise ValueError(f"Unsupported order_strategy={order_strategy!r}")

def group_records_by_cardinality(
    records: Sequence[MultiLabelMosaicRecord],
) -> dict[int, list[MultiLabelMosaicRecord]]:
    """
    Group records by number of positive labels.
    """
    grouped: dict[int, list[MultiLabelMosaicRecord]] = {}

    for record in records:
        grouped.setdefault(record.cardinality, []).append(record)

    return {
        cardinality: grouped[cardinality]
        for cardinality in sorted(grouped)
    }

def group_records_by_signature(
    records: Sequence[MultiLabelMosaicRecord],
) -> dict[tuple[int, str], list[MultiLabelMosaicRecord]]:
    """
    Group records by exact label signature, sorted first by cardinality and then
    alphabetically by signature.
    """
    grouped: dict[tuple[int, str], list[MultiLabelMosaicRecord]] = {}

    for record in records:
        key = (record.cardinality, record.label_signature)
        grouped.setdefault(key, []).append(record)

    return {
        key: grouped[key]
        for key in sorted(grouped, key=lambda item: (item[0], item[1]))
    }

def count_by_cardinality(
    records: Sequence[MultiLabelMosaicRecord],
) -> dict[int, int]:
    """
    Count records by label cardinality.
    """
    counts: dict[int, int] = {}

    for record in records:
        counts[record.cardinality] = counts.get(record.cardinality, 0) + 1

    return {
        cardinality: counts[cardinality]
        for cardinality in sorted(counts)
    }

def count_by_label_signature(
    records: Sequence[MultiLabelMosaicRecord],
) -> dict[str, int]:
    """
    Count records by exact sorted label signature.
    """
    counts: dict[str, int] = {}

    for record in records:
        counts[record.label_signature] = counts.get(record.label_signature, 0) + 1

    return {
        signature: counts[signature]
        for signature in sorted(counts)
    }

def _raw_dict_from_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)

    raw = getattr(row, "raw", None)
    if isinstance(raw, dict):
        return dict(raw)

    payload: dict[str, Any] = {}

    for key in ("image_id", "source_ref", "split", "label_type", "labels"):
        if hasattr(row, key):
            payload[key] = getattr(row, key)

    return payload

def _source_ref_from_row(
    *,
    row: Any,
    raw: dict[str, Any],
    row_index: int,
) -> str:
    for key in ("source_ref", "source-ref"):
        value = raw.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    attr_value = getattr(row, "source_ref", None)
    if isinstance(attr_value, str) and attr_value.strip():
        return attr_value.strip()

    raise ValueError(f"Row {row_index} is missing non-empty source_ref/source-ref")

def _labels_from_row(
    *,
    row: Any,
    raw: dict[str, Any],
    row_index: int,
    allow_empty_labels: bool,
) -> tuple[str, ...]:
    value = raw.get("labels")

    if value is None and hasattr(row, "labels"):
        value = getattr(row, "labels")

    if value is None:
        value = raw.get("label_names")

    if value is None and hasattr(row, "label_names"):
        value = getattr(row, "label_names")

    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"Row {row_index} labels must be a list/tuple of strings, "
            f"got {type(value).__name__}"
        )

    labels: list[str] = []

    for label_index, label in enumerate(value):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"Row {row_index} labels[{label_index}] must be a non-empty string"
            )
        labels.append(label.strip())

    if not labels and not allow_empty_labels:
        raise ValueError(f"Row {row_index} has no labels")

    unique_sorted = tuple(sorted(set(labels)))

    if len(unique_sorted) != len(labels):
        raise ValueError(f"Row {row_index} has duplicate labels: {labels!r}")

    return unique_sorted

def _optional_string_from_row(
    *,
    row: Any,
    raw: dict[str, Any],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        value = raw.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

        attr_name = key.replace("-", "_")
        attr_value = getattr(row, attr_name, None)

        if isinstance(attr_value, str) and attr_value.strip():
            return attr_value.strip()

    return None

def _validate_labels_in_class_map(
    *,
    labels: tuple[str, ...],
    class_to_idx: dict[str, int],
    row_index: int,
) -> None:
    unknown = sorted(set(labels) - set(class_to_idx))

    if unknown:
        raise ValueError(
            f"Row {row_index} contains labels not present in class_to_idx: {unknown}"
        )

def _resolve_group_mode(
    *,
    group_mode: str,
    group_by_cardinality: bool | None,
) -> str:
    normalized = _normalize_nonempty_string(group_mode, "group_mode")
    _validate_group_mode(normalized)

    if group_by_cardinality is not None and normalized == "none":
        return "cardinality" if group_by_cardinality else "none"

    return normalized

def _validate_order_strategy(order_strategy: str) -> None:
    if order_strategy not in _ALLOWED_ORDER_STRATEGIES:
        raise ValueError(
            f"order_strategy must be one of {sorted(_ALLOWED_ORDER_STRATEGIES)}, "
            f"got {order_strategy!r}"
        )

def _validate_group_mode(group_mode: str) -> None:
    if group_mode not in _ALLOWED_GROUP_MODES:
        raise ValueError(
            f"group_mode must be one of {sorted(_ALLOWED_GROUP_MODES)}, "
            f"got {group_mode!r}"
        )

def _normalize_nonempty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text

def _validate_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {value!r}")

    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")