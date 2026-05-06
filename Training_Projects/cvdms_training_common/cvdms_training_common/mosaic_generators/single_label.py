"""
Single-label mosaic-generation utilities for CVDMS training projects.

This module contains task-specific logic for single-label classification mosaics:

    - extract source_ref and label from CVDMS manifest rows
    - validate single-label rows
    - sort rows deterministically
    - group mosaics by class (default) or optionally by split only
    - render/save mosaic sheets using mosaic_generators.common_utils

No text, labels, or image IDs are drawn onto the mosaic images themselves.
Labels are used only for grouping, ordering, and output filenames.
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
    "class_image_id",
    "class_source_ref",
    "image_id",
    "source_ref",
    "random",
}

@dataclass(frozen=True)
class SingleLabelMosaicRecord:
    """
    Normalized single-label manifest row used for mosaic generation.

    Args:
        source_ref:
            Image URI/path passed to the configured image loader.
        label:
            Single class label for this image.
        image_id:
            Optional CVDMS image_id.
        split:
            Optional dataset split.
        raw:
            Optional raw row payload for debugging or downstream use.
    """

    source_ref: str
    label: str
    image_id: str | None = None
    split: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, str) or not self.source_ref.strip():
            raise ValueError("source_ref must be a non-empty string")

        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")

        if self.image_id is not None and not str(self.image_id).strip():
            raise ValueError("image_id cannot be empty when provided")

        if self.split is not None and not str(self.split).strip():
            raise ValueError("split cannot be empty when provided")

        if not isinstance(self.raw, dict):
            raise TypeError(f"raw must be a dictionary, got {type(self.raw).__name__}")

    def to_mosaic_item(self) -> MosaicItem:
        return MosaicItem(
            source_ref=self.source_ref,
            metadata={
                "image_id": self.image_id,
                "split": self.split,
                "label": self.label,
            },
        )

    def sort_key_class_image_id(self) -> tuple[str, str, str]:
        return (
            self.label,
            self.image_id or "",
            self.source_ref,
        )

    def sort_key_class_source_ref(self) -> tuple[str, str, str]:
        return (
            self.label,
            self.source_ref,
            self.image_id or "",
        )

    def sort_key_image_id(self) -> tuple[str, str, str]:
        return (
            self.image_id or "",
            self.label,
            self.source_ref,
        )

    def sort_key_source_ref(self) -> tuple[str, str, str]:
        return (
            self.source_ref,
            self.label,
            self.image_id or "",
        )

@dataclass(frozen=True)
class SingleLabelMosaicResult:
    """
    Result summary for a single-label mosaic generation run.
    """

    split: str
    output_dir: Path
    order_strategy: str
    group_by_class: bool
    total_items: int
    group_counts: dict[str, int]
    sheets: list[MosaicSheetResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "output_dir": str(self.output_dir),
            "order_strategy": self.order_strategy,
            "group_by_class": self.group_by_class,
            "total_items": self.total_items,
            "group_counts": dict(self.group_counts),
            "sheets": [sheet.to_dict() for sheet in self.sheets],
        }

def generate_single_label_split_mosaics(
    *,
    rows: Sequence[Any],
    image_loader: ImageLoaderFn,
    output_dir: str | Path,
    split: str,
    config: MosaicConfig | None = None,
    class_to_idx: dict[str, int] | None = None,
    order_strategy: str = "class_image_id",
    group_by_class: bool = True,
    random_seed: int = 42,
    max_items: int | None = None,
    validate_split: bool = True,
) -> SingleLabelMosaicResult:
    """
    Generate mosaic sheets for one single-label split.

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
                class_image_id
                class_source_ref
                image_id
                source_ref
                random
        group_by_class:
            If True, write separate mosaic sets per class so no sheet mixes classes.
            If False, write a split-level mosaic that may mix classes.
        random_seed:
            Seed used only when order_strategy="random".
        max_items:
            Optional cap after ordering/grouping input rows. Useful for previews.
        validate_split:
            If True and a row has a split field, it must match the requested split.
    """
    resolved_config = config or MosaicConfig()
    normalized_split = _normalize_nonempty_string(split, "split")
    _validate_order_strategy(order_strategy)

    records = make_single_label_records(
        rows=rows,
        split=normalized_split,
        class_to_idx=class_to_idx,
        validate_split=validate_split,
    )

    records = order_single_label_records(
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

    if group_by_class:
        grouped = group_records_by_class(records)

        for class_name, group_records in grouped.items():
            group_counts[class_name] = len(group_records)

            items = [record.to_mosaic_item() for record in group_records]
            prefix = (
                f"{normalized_split}"
                f"__class-{safe_filename_part(class_name)}"
                f"__order-{order_strategy}"
            )

            sheets = save_mosaic_sheets(
                items=items,
                image_loader=image_loader,
                output_dir=split_output_dir,
                filename_prefix=prefix,
                config=resolved_config,
            )
            all_sheets.extend(sheets)
    else:
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

    return SingleLabelMosaicResult(
        split=normalized_split,
        output_dir=split_output_dir,
        order_strategy=order_strategy,
        group_by_class=group_by_class,
        total_items=len(records),
        group_counts=group_counts,
        sheets=all_sheets,
    )

def make_single_label_records(
    *,
    rows: Sequence[Any],
    split: str | None = None,
    class_to_idx: dict[str, int] | None = None,
    validate_split: bool = True,
) -> list[SingleLabelMosaicRecord]:
    """
    Convert manifest rows into normalized SingleLabelMosaicRecord objects.
    """
    records: list[SingleLabelMosaicRecord] = []

    for idx, row in enumerate(rows):
        record = single_label_record_from_row(
            row=row,
            row_index=idx,
            split=split,
            class_to_idx=class_to_idx,
            validate_split=validate_split,
        )
        records.append(record)

    return records

def single_label_record_from_row(
    *,
    row: Any,
    row_index: int,
    split: str | None = None,
    class_to_idx: dict[str, int] | None = None,
    validate_split: bool = True,
) -> SingleLabelMosaicRecord:
    """
    Normalize one manifest row into a SingleLabelMosaicRecord.
    """
    raw = _raw_dict_from_row(row)
    source_ref = _source_ref_from_row(row=row, raw=raw, row_index=row_index)
    image_id = _optional_string_from_row(row=row, raw=raw, keys=("image_id", "image-id"))
    row_split = _optional_string_from_row(row=row, raw=raw, keys=("split",))
    label = _label_from_row(row=row, raw=raw, row_index=row_index)

    if split is not None and validate_split and row_split is not None:
        if row_split != split:
            raise ValueError(
                f"Row {row_index} split mismatch: expected {split!r}, got {row_split!r}"
            )

    if class_to_idx is not None and label not in class_to_idx:
        raise ValueError(
            f"Row {row_index} label {label!r} is not present in class_to_idx"
        )

    return SingleLabelMosaicRecord(
        source_ref=source_ref,
        label=label,
        image_id=image_id,
        split=row_split or split,
        raw=raw,
    )

def order_single_label_records(
    *,
    records: Sequence[SingleLabelMosaicRecord],
    order_strategy: str = "class_image_id",
    random_seed: int = 42,
) -> list[SingleLabelMosaicRecord]:
    """
    Order single-label records deterministically.
    """
    _validate_order_strategy(order_strategy)
    ordered = list(records)

    if order_strategy == "class_image_id":
        return sorted(ordered, key=lambda record: record.sort_key_class_image_id())

    if order_strategy == "class_source_ref":
        return sorted(ordered, key=lambda record: record.sort_key_class_source_ref())

    if order_strategy == "image_id":
        return sorted(ordered, key=lambda record: record.sort_key_image_id())

    if order_strategy == "source_ref":
        return sorted(ordered, key=lambda record: record.sort_key_source_ref())

    if order_strategy == "random":
        rng = random.Random(random_seed)
        rng.shuffle(ordered)
        return ordered

    raise ValueError(f"Unsupported order_strategy={order_strategy!r}")

def group_records_by_class(
    records: Sequence[SingleLabelMosaicRecord],
) -> dict[str, list[SingleLabelMosaicRecord]]:
    """
    Group records by class label.
    """
    grouped: dict[str, list[SingleLabelMosaicRecord]] = {}

    for record in records:
        grouped.setdefault(record.label, []).append(record)

    return {
        class_name: grouped[class_name]
        for class_name in sorted(grouped)
    }

def count_by_class(
    records: Sequence[SingleLabelMosaicRecord],
) -> dict[str, int]:
    """
    Count records by class label.
    """
    counts: dict[str, int] = {}

    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1

    return {
        class_name: counts[class_name]
        for class_name in sorted(counts)
    }

def _raw_dict_from_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)

    raw = getattr(row, "raw", None)
    if isinstance(raw, dict):
        return dict(raw)

    payload: dict[str, Any] = {}

    for key in ("image_id", "source_ref", "split", "label_type", "label", "class_name"):
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

def _label_from_row(
    *,
    row: Any,
    raw: dict[str, Any],
    row_index: int,
) -> str:
    for key in ("label", "class_name", "class-name", "target"):
        value = raw.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

        attr_name = key.replace("-", "_")
        attr_value = getattr(row, attr_name, None)

        if isinstance(attr_value, str) and attr_value.strip():
            return attr_value.strip()

    labels_value = raw.get("labels")
    if labels_value is None and hasattr(row, "labels"):
        labels_value = getattr(row, "labels")

    if isinstance(labels_value, (list, tuple)):
        if len(labels_value) != 1:
            raise ValueError(
                f"Row {row_index} expected exactly one label in labels, got {labels_value!r}"
            )
        only_label = labels_value[0]
        if not isinstance(only_label, str) or not only_label.strip():
            raise ValueError(f"Row {row_index} labels[0] must be a non-empty string")
        return only_label.strip()

    raise ValueError(
        f"Row {row_index} is missing a valid single-label field "
        f"(expected one of label/class_name/class-name/target)"
    )

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

def _validate_order_strategy(order_strategy: str) -> None:
    if order_strategy not in _ALLOWED_ORDER_STRATEGIES:
        raise ValueError(
            f"order_strategy must be one of {sorted(_ALLOWED_ORDER_STRATEGIES)}, "
            f"got {order_strategy!r}"
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