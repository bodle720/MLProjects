"""
Object-detection mosaic-generation utilities for CVDMS training projects.

This module contains task-specific logic for object-detection mosaics:

    - extract source_ref and bounding boxes from CVDMS manifest rows
    - optionally load external annotation JSON payloads
    - validate object-detection rows
    - sort rows deterministically by box count, image_id, or source_ref
    - optionally group mosaics by class
    - render/save mosaic sheets with bounding boxes drawn on each tile

The default use case is split-level mosaics with boxes drawn. For a single-class
dataset such as Global Wheat Head Detection 2021, group_mode="none" is usually
the most useful setting.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, cast

from PIL import Image, ImageDraw, ImageFont, ImageOps

from cvdms_training_common.mosaic_generators.common_utils import (
    ImageLoaderFn,
    MosaicConfig,
    MosaicSheetResult,
    build_mosaic_filename,
    chunk_sequence,
    safe_filename_part,
)

LabelLoaderFn = Callable[[str], Any]
Color = tuple[int, int, int]

_ALLOWED_ORDER_STRATEGIES = {
    "box_count_desc",
    "box_count_asc",
    "class_box_count",
    "image_id",
    "source_ref",
    "random",
}

_ALLOWED_GROUP_MODES = {
    "none",
    "class",
}

@dataclass(frozen=True)
class ObjectDetectionBox:
    """
    One object-detection bounding box in pixel coordinates.

    Coordinates use the CVDMS object-detection convention:

        left, top, width, height

    Args:
        class_name:
            Class label for the object.
        left:
            Left x-coordinate in source-image pixels.
        top:
            Top y-coordinate in source-image pixels.
        width:
            Box width in source-image pixels.
        height:
            Box height in source-image pixels.
        raw:
            Optional raw annotation payload for debugging/downstream use.
    """

    class_name: str
    left: float
    top: float
    width: float
    height: float
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be a non-empty string")

        _validate_number(self.left, "left")
        _validate_number(self.top, "top")
        _validate_number(self.width, "width")
        _validate_number(self.height, "height")

        if self.left < 0:
            raise ValueError(f"left must be >= 0, got {self.left}")

        if self.top < 0:
            raise ValueError(f"top must be >= 0, got {self.top}")

        if self.width <= 0:
            raise ValueError(f"width must be > 0, got {self.width}")

        if self.height <= 0:
            raise ValueError(f"height must be > 0, got {self.height}")

        if not isinstance(self.raw, dict):
            raise TypeError(f"raw must be a dictionary, got {type(self.raw).__name__}")

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

@dataclass(frozen=True)
class ObjectDetectionMosaicRecord:
    """
    Normalized object-detection manifest row used for mosaic generation.
    """

    source_ref: str
    boxes: tuple[ObjectDetectionBox, ...]
    image_id: str | None = None
    split: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, str) or not self.source_ref.strip():
            raise ValueError("source_ref must be a non-empty string")

        if not isinstance(self.boxes, tuple):
            raise TypeError(f"boxes must be a tuple, got {type(self.boxes).__name__}")

        if not self.boxes:
            raise ValueError("boxes cannot be empty for an object-detection mosaic record")

        for idx, box in enumerate(self.boxes):
            if not isinstance(box, ObjectDetectionBox):
                raise TypeError(
                    f"boxes[{idx}] must be ObjectDetectionBox, got {type(box).__name__}"
                )

        if self.image_id is not None and not str(self.image_id).strip():
            raise ValueError("image_id cannot be empty when provided")

        if self.split is not None and not str(self.split).strip():
            raise ValueError("split cannot be empty when provided")

        if not isinstance(self.raw, dict):
            raise TypeError(f"raw must be a dictionary, got {type(self.raw).__name__}")

    @property
    def box_count(self) -> int:
        return len(self.boxes)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(sorted({box.class_name for box in self.boxes}))

    @property
    def primary_class_name(self) -> str:
        class_names = self.class_names
        if len(class_names) == 1:
            return class_names[0]
        return "mixed"

    def sort_key_box_count_desc(self) -> tuple[int, str, str]:
        return (
            -self.box_count,
            self.image_id or "",
            self.source_ref,
        )

    def sort_key_box_count_asc(self) -> tuple[int, str, str]:
        return (
            self.box_count,
            self.image_id or "",
            self.source_ref,
        )

    def sort_key_class_box_count(self) -> tuple[str, int, str, str]:
        return (
            self.primary_class_name,
            -self.box_count,
            self.image_id or "",
            self.source_ref,
        )

    def sort_key_image_id(self) -> tuple[str, str]:
        return self.image_id or "", self.source_ref

    def sort_key_source_ref(self) -> tuple[str, str]:
        return self.source_ref, self.image_id or ""

@dataclass(frozen=True)
class ObjectDetectionMosaicResult:
    """
    Result summary for an object-detection mosaic generation run.
    """

    split: str
    output_dir: Path
    order_strategy: str
    group_mode: str
    total_items: int
    total_boxes: int
    group_counts: dict[str, int]
    group_box_counts: dict[str, int]
    sheets: list[MosaicSheetResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "output_dir": str(self.output_dir),
            "order_strategy": self.order_strategy,
            "group_mode": self.group_mode,
            "total_items": self.total_items,
            "total_boxes": self.total_boxes,
            "group_counts": dict(self.group_counts),
            "group_box_counts": dict(self.group_box_counts),
            "sheets": [sheet.to_dict() for sheet in self.sheets],
        }

def generate_object_detection_split_mosaics(
    *,
    rows: Sequence[Any],
    image_loader: ImageLoaderFn,
    output_dir: str | Path,
    split: str,
    config: MosaicConfig | None = None,
    label_loader: LabelLoaderFn | None = None,
    class_to_idx: dict[str, int] | None = None,
    order_strategy: str = "box_count_desc",
    group_mode: str = "none",
    random_seed: int = 42,
    max_items: int | None = None,
    validate_split: bool = True,
    draw_labels: bool = False,
    draw_box_count: bool = True,
    box_color: Color = (255, 0, 0),
    box_width: int = 2,
) -> ObjectDetectionMosaicResult:
    """
    Generate object-detection mosaic sheets for one split.

    Args:
        rows:
            Manifest rows. Each row may be a dict or an object with attributes
            such as source_ref, split, image_id, labels, annotations, or raw.
        image_loader:
            Callable that loads PIL images from source_ref.
        output_dir:
            Base output directory. The split name is appended automatically.
        split:
            Split name, usually train/val/test.
        config:
            MosaicConfig. Defaults to 10x10 tiles of 128x128 pixels.
        label_loader:
            Optional callable that loads annotation JSON payloads from a URI/path.
            Use this when manifest rows store labels as external JSON references.
        class_to_idx:
            Optional class map used to validate box class names.
        order_strategy:
            One of:
                box_count_desc
                box_count_asc
                class_box_count
                image_id
                source_ref
                random
        group_mode:
            One of:
                none
                class
            For single-class wheat-head detection, group_mode="none" is usually best.
        random_seed:
            Seed used only when order_strategy="random".
        max_items:
            Optional cap after ordering input rows. Useful for previews.
        validate_split:
            If True and a row has a split field, it must match the requested split.
        draw_labels:
            If True, draw class names near boxes. Usually False is better for dense
            small-object datasets because labels can clutter the image.
        draw_box_count:
            If True, draw a small box-count label in the upper-left corner.
        box_color:
            RGB color for bounding boxes.
        box_width:
            Width of the drawn bounding boxes in pixels.
    """
    resolved_config = config or MosaicConfig()
    normalized_split = _normalize_nonempty_string(split, "split")
    _validate_order_strategy(order_strategy)
    _validate_group_mode(group_mode)
    _validate_color(box_color, "box_color")
    _validate_positive_int(box_width, "box_width")

    records = make_object_detection_records(
        rows=rows,
        split=normalized_split,
        label_loader=label_loader,
        class_to_idx=class_to_idx,
        validate_split=validate_split,
    )

    records = order_object_detection_records(
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
    group_box_counts: dict[str, int] = {}

    if group_mode == "none":
        group_counts["all"] = len(records)
        group_box_counts["all"] = sum(record.box_count for record in records)

        prefix = f"{normalized_split}__order-{order_strategy}"

        sheets = save_object_detection_mosaic_sheets(
            records=records,
            image_loader=image_loader,
            output_dir=split_output_dir,
            filename_prefix=prefix,
            config=resolved_config,
            draw_labels=draw_labels,
            draw_box_count=draw_box_count,
            box_color=box_color,
            box_width=box_width,
        )
        all_sheets.extend(sheets)

    elif group_mode == "class":
        grouped = group_records_by_class(records)

        for class_name, group_records in grouped.items():
            group_counts[class_name] = len(group_records)
            group_box_counts[class_name] = sum(record.box_count for record in group_records)

            group_output_dir = split_output_dir / safe_filename_part(class_name)
            prefix = (
                f"{normalized_split}"
                f"__class-{safe_filename_part(class_name)}"
                f"__order-{order_strategy}"
            )

            sheets = save_object_detection_mosaic_sheets(
                records=group_records,
                image_loader=image_loader,
                output_dir=group_output_dir,
                filename_prefix=prefix,
                config=resolved_config,
                draw_labels=draw_labels,
                draw_box_count=draw_box_count,
                box_color=box_color,
                box_width=box_width,
            )
            all_sheets.extend(sheets)

    else:  # pragma: no cover
        raise ValueError(f"Unsupported group_mode={group_mode!r}")

    return ObjectDetectionMosaicResult(
        split=normalized_split,
        output_dir=split_output_dir,
        order_strategy=order_strategy,
        group_mode=group_mode,
        total_items=len(records),
        total_boxes=sum(record.box_count for record in records),
        group_counts=group_counts,
        group_box_counts=group_box_counts,
        sheets=all_sheets,
    )

def make_object_detection_records(
    *,
    rows: Sequence[Any],
    split: str | None = None,
    label_loader: LabelLoaderFn | None = None,
    class_to_idx: dict[str, int] | None = None,
    validate_split: bool = True,
) -> list[ObjectDetectionMosaicRecord]:
    """
    Convert manifest rows into normalized ObjectDetectionMosaicRecord objects.
    """
    records: list[ObjectDetectionMosaicRecord] = []

    for idx, row in enumerate(rows):
        record = object_detection_record_from_row(
            row=row,
            row_index=idx,
            split=split,
            label_loader=label_loader,
            class_to_idx=class_to_idx,
            validate_split=validate_split,
        )
        records.append(record)

    return records

def object_detection_record_from_row(
    *,
    row: Any,
    row_index: int,
    split: str | None = None,
    label_loader: LabelLoaderFn | None = None,
    class_to_idx: dict[str, int] | None = None,
    validate_split: bool = True,
) -> ObjectDetectionMosaicRecord:
    """
    Normalize one manifest row into an ObjectDetectionMosaicRecord.
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

    boxes_payload = _boxes_payload_from_row(
        row=row,
        raw=raw,
        row_index=row_index,
        label_loader=label_loader,
    )

    boxes = tuple(
        _box_from_payload(
            payload=box_payload,
            row_index=row_index,
            box_index=box_index,
            class_to_idx=class_to_idx,
        )
        for box_index, box_payload in enumerate(boxes_payload)
    )

    if not boxes:
        raise ValueError(f"Row {row_index} has no object-detection boxes")

    return ObjectDetectionMosaicRecord(
        source_ref=source_ref,
        boxes=boxes,
        image_id=image_id,
        split=row_split or split,
        raw=raw,
    )

def order_object_detection_records(
    *,
    records: Sequence[ObjectDetectionMosaicRecord],
    order_strategy: str = "box_count_desc",
    random_seed: int = 42,
) -> list[ObjectDetectionMosaicRecord]:
    """
    Order object-detection records for visually useful mosaics.
    """
    _validate_order_strategy(order_strategy)
    ordered = list(records)

    if order_strategy == "box_count_desc":
        return sorted(ordered, key=lambda record: record.sort_key_box_count_desc())

    if order_strategy == "box_count_asc":
        return sorted(ordered, key=lambda record: record.sort_key_box_count_asc())

    if order_strategy == "class_box_count":
        return sorted(ordered, key=lambda record: record.sort_key_class_box_count())

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
    records: Sequence[ObjectDetectionMosaicRecord],
) -> dict[str, list[ObjectDetectionMosaicRecord]]:
    """
    Group records by their only class name.

    If a record contains multiple classes, it is grouped under "mixed".
    For Project 3 wheat-head detection, records should group under "wheat_head".
    """
    grouped: dict[str, list[ObjectDetectionMosaicRecord]] = {}

    for record in records:
        grouped.setdefault(record.primary_class_name, []).append(record)

    return {
        class_name: grouped[class_name]
        for class_name in sorted(grouped)
    }

def save_object_detection_mosaic_sheets(
    *,
    records: Sequence[ObjectDetectionMosaicRecord],
    image_loader: ImageLoaderFn,
    output_dir: str | Path,
    filename_prefix: str,
    config: MosaicConfig,
    draw_labels: bool = False,
    draw_box_count: bool = True,
    box_color: Color = (255, 0, 0),
    box_width: int = 2,
    start_sheet_index: int = 1,
) -> list[MosaicSheetResult]:
    """
    Render and save one or more bbox-aware mosaic sheets.
    """
    if not records:
        raise ValueError("records cannot be empty")

    _validate_positive_int(start_sheet_index, "start_sheet_index")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_prefix = safe_filename_part(filename_prefix)
    results: list[MosaicSheetResult] = []

    for offset, chunk in enumerate(chunk_sequence(records, config.tiles_per_sheet)):
        sheet_index = start_sheet_index + offset
        first_item_index = offset * config.tiles_per_sheet
        last_item_index = first_item_index + len(chunk) - 1

        mosaic = render_object_detection_mosaic_sheet(
            records=chunk,
            image_loader=image_loader,
            config=config,
            draw_labels=draw_labels,
            draw_box_count=draw_box_count,
            box_color=box_color,
            box_width=box_width,
        )

        filename = build_mosaic_filename(
            filename_prefix=safe_prefix,
            config=config,
            sheet_index=sheet_index,
        )
        destination = output_path / filename

        mosaic.save(destination, format=config.output_format)

        results.append(
            MosaicSheetResult(
                path=destination,
                sheet_index=sheet_index,
                item_count=len(chunk),
                first_item_index=first_item_index,
                last_item_index=last_item_index,
            )
        )

    return results

def render_object_detection_mosaic_sheet(
    *,
    records: Sequence[ObjectDetectionMosaicRecord],
    image_loader: ImageLoaderFn,
    config: MosaicConfig,
    draw_labels: bool = False,
    draw_box_count: bool = True,
    box_color: Color = (255, 0, 0),
    box_width: int = 2,
) -> Image.Image:
    """
    Render one object-detection mosaic sheet from up to config.tiles_per_sheet records.
    """
    if not records:
        raise ValueError("records cannot be empty")

    if len(records) > config.tiles_per_sheet:
        raise ValueError(
            f"records has {len(records)} entries, but config only allows "
            f"{config.tiles_per_sheet} tiles per sheet"
        )

    canvas = Image.new(
        "RGB",
        config.canvas_size,
        color=config.background_color,
    )

    for record_index, record in enumerate(records):
        row = record_index // config.cols
        col = record_index % config.cols
        x = col * config.tile_width
        y = row * config.tile_height

        tile = load_detection_record_as_tile(
            record=record,
            image_loader=image_loader,
            config=config,
            draw_labels=draw_labels,
            draw_box_count=draw_box_count,
            box_color=box_color,
            box_width=box_width,
        )
        canvas.paste(tile, (x, y))

    return canvas

def load_detection_record_as_tile(
    *,
    record: ObjectDetectionMosaicRecord,
    image_loader: ImageLoaderFn,
    config: MosaicConfig,
    draw_labels: bool = False,
    draw_box_count: bool = True,
    box_color: Color = (255, 0, 0),
    box_width: int = 2,
) -> Image.Image:
    """
    Load one detection record and convert it into a padded tile with boxes drawn.
    """
    try:
        image = image_loader(record.source_ref)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load image for mosaic record: {record.source_ref!r}"
        ) from exc

    try:
        return fit_detection_image_to_tile(
            image=image,
            boxes=record.boxes,
            tile_width=config.tile_width,
            tile_height=config.tile_height,
            background_color=config.background_color,
            image_mode=config.image_mode,
            draw_labels=draw_labels,
            draw_box_count=draw_box_count,
            box_color=box_color,
            box_width=box_width,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to render detection tile for: {record.source_ref!r}"
        ) from exc

def fit_detection_image_to_tile(
    *,
    image: Image.Image,
    boxes: Sequence[ObjectDetectionBox],
    tile_width: int,
    tile_height: int,
    background_color: Color = (0, 0, 0),
    image_mode: str | None = "RGB",
    draw_labels: bool = False,
    draw_box_count: bool = True,
    box_color: Color = (255, 0, 0),
    box_width: int = 2,
) -> Image.Image:
    """
    Resize an image to a padded tile and draw scaled object-detection boxes.

    The same scale/padding transform is applied to both the image and the boxes.
    """
    _validate_positive_int(tile_width, "tile_width")
    _validate_positive_int(tile_height, "tile_height")
    _validate_color(background_color, "background_color")
    _validate_color(box_color, "box_color")
    _validate_positive_int(box_width, "box_width")

    if image.width < 1 or image.height < 1:
        raise ValueError(f"Image dimensions must be positive, got {image.size}")

    working = ImageOps.exif_transpose(image)

    if image_mode is not None:
        working = working.convert(image_mode)

    scale = min(tile_width / working.width, tile_height / working.height)
    resized_width = max(1, int(round(working.width * scale)))
    resized_height = max(1, int(round(working.height * scale)))

    resized = working.resize(
        (resized_width, resized_height),
        resample=_resample_filter(),
    )

    tile = Image.new(
        "RGB",
        (tile_width, tile_height),
        color=background_color,
    )

    if resized.mode != "RGB":
        resized = resized.convert("RGB")

    paste_x = (tile_width - resized_width) // 2
    paste_y = (tile_height - resized_height) // 2
    tile.paste(resized, (paste_x, paste_y))

    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()

    for box in boxes:
        x1 = paste_x + box.left * scale
        y1 = paste_y + box.top * scale
        x2 = paste_x + box.right * scale
        y2 = paste_y + box.bottom * scale

        x1 = max(0, min(tile_width - 1, x1))
        y1 = max(0, min(tile_height - 1, y1))
        x2 = max(0, min(tile_width - 1, x2))
        y2 = max(0, min(tile_height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=box_color,
            width=box_width,
        )

        if draw_labels:
            _draw_text_with_background(
                draw=draw,
                xy=(int(x1), int(y1)),
                text=box.class_name,
                font=font,
                text_color=(255, 255, 255),
                background_color=(0, 0, 0),
            )

    if draw_box_count:
        _draw_text_with_background(
            draw=draw,
            xy=(2, 2),
            text=f"boxes: {len(boxes)}",
            font=font,
            text_color=(255, 255, 255),
            background_color=(0, 0, 0),
        )

    return tile

def _boxes_payload_from_row(
    *,
    row: Any,
    raw: dict[str, Any],
    row_index: int,
    label_loader: LabelLoaderFn | None,
) -> list[Any]:
    """
    Resolve a row's object-detection annotations.

    Supports either inline annotation payloads or external annotation references
    loaded through label_loader.
    """
    inline_keys = (
        "annotations",
        "objects",
        "bboxes",
        "boxes",
        "bbox_annotations",
        "bbox-annotations",
    )

    for key in inline_keys:
        value = raw.get(key)
        if value is not None:
            return _extract_boxes_from_payload(value, row_index=row_index)

        attr_name = key.replace("-", "_")
        if hasattr(row, attr_name):
            return _extract_boxes_from_payload(getattr(row, attr_name), row_index=row_index)

    labels_value = raw.get("labels")
    if labels_value is None and hasattr(row, "labels"):
        labels_value = getattr(row, "labels")

    if labels_value is not None:
        if isinstance(labels_value, str):
            if label_loader is None:
                raise ValueError(
                    f"Row {row_index} labels is an external reference, but label_loader is None"
                )
            loaded = label_loader(labels_value)
            return _extract_boxes_from_payload(loaded, row_index=row_index)

        return _extract_boxes_from_payload(labels_value, row_index=row_index)

    external_keys = (
        "label",
        "label_ref",
        "label-ref",
        "labels_ref",
        "labels-ref",
        "annotation_ref",
        "annotation-ref",
        "annotations_ref",
        "annotations-ref",
    )

    for key in external_keys:
        value = raw.get(key)

        if isinstance(value, str) and value.strip():
            if label_loader is None:
                raise ValueError(
                    f"Row {row_index} {key!r} is an external reference, "
                    "but label_loader is None"
                )
            loaded = label_loader(value.strip())
            return _extract_boxes_from_payload(loaded, row_index=row_index)

        attr_name = key.replace("-", "_")
        attr_value = getattr(row, attr_name, None)

        if isinstance(attr_value, str) and attr_value.strip():
            if label_loader is None:
                raise ValueError(
                    f"Row {row_index} {attr_name!r} is an external reference, "
                    "but label_loader is None"
                )
            loaded = label_loader(attr_value.strip())
            return _extract_boxes_from_payload(loaded, row_index=row_index)

    raise ValueError(
        f"Row {row_index} is missing object-detection annotations. Expected inline "
        "annotations/objects/bboxes/boxes/bbox_annotations or an external label reference."
    )

def _extract_boxes_from_payload(payload: Any, *, row_index: int) -> list[Any]:
    """
    Extract a list of raw boxes from a flexible annotation payload.
    """
    if isinstance(payload, list):
        return payload

    if isinstance(payload, tuple):
        return list(payload)

    if isinstance(payload, dict):
        for key in (
            "annotations",
            "objects",
            "bboxes",
            "boxes",
            "bbox_annotations",
            "bbox-annotations",
        ):
            value = payload.get(key)
            if isinstance(value, (list, tuple)):
                return list(value)

        if _looks_like_single_box(payload):
            return [payload]

    raise TypeError(
        f"Row {row_index} annotation payload must be a list of boxes or a dict "
        f"containing a box list, got {type(payload).__name__}"
    )

def _box_from_payload(
    *,
    payload: Any,
    row_index: int,
    box_index: int,
    class_to_idx: dict[str, int] | None,
) -> ObjectDetectionBox:
    if not isinstance(payload, dict):
        raise TypeError(
            f"Row {row_index} box {box_index} must be a dictionary, "
            f"got {type(payload).__name__}"
        )

    raw_box = dict(payload)
    class_name = _class_name_from_box(raw_box, row_index=row_index, box_index=box_index)

    if class_to_idx is not None and class_name not in class_to_idx:
        raise ValueError(
            f"Row {row_index} box {box_index} class {class_name!r} "
            "is not present in class_to_idx"
        )

    left, top, width, height = _coordinates_from_box(
        raw_box,
        row_index=row_index,
        box_index=box_index,
    )

    return ObjectDetectionBox(
        class_name=class_name,
        left=left,
        top=top,
        width=width,
        height=height,
        raw=raw_box,
    )

def _class_name_from_box(
    raw_box: dict[str, Any],
    *,
    row_index: int,
    box_index: int,
) -> str:
    for key in (
        "class_name",
        "class-name",
        "label",
        "class",
        "name",
        "category",
        "category_name",
        "category-name",
    ):
        value = raw_box.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    class_id = raw_box.get("class_id")
    if class_id is None:
        class_id = raw_box.get("class-id")

    if class_id is not None:
        return str(class_id).strip()

    raise ValueError(
        f"Row {row_index} box {box_index} is missing class_name/label/category"
    )

def _coordinates_from_box(
    raw_box: dict[str, Any],
    *,
    row_index: int,
    box_index: int,
) -> tuple[float, float, float, float]:
    """
    Extract left, top, width, height from common bbox conventions.
    """
    if _has_all_keys(raw_box, ("left", "top", "width", "height")):
        return (
            _float_from_mapping(raw_box, "left", row_index, box_index),
            _float_from_mapping(raw_box, "top", row_index, box_index),
            _float_from_mapping(raw_box, "width", row_index, box_index),
            _float_from_mapping(raw_box, "height", row_index, box_index),
        )

    if _has_all_keys(raw_box, ("x", "y", "width", "height")):
        return (
            _float_from_mapping(raw_box, "x", row_index, box_index),
            _float_from_mapping(raw_box, "y", row_index, box_index),
            _float_from_mapping(raw_box, "width", row_index, box_index),
            _float_from_mapping(raw_box, "height", row_index, box_index),
        )

    xyxy_key_sets = (
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    )

    for x1_key, y1_key, x2_key, y2_key in xyxy_key_sets:
        if _has_all_keys(raw_box, (x1_key, y1_key, x2_key, y2_key)):
            left = _float_from_mapping(raw_box, x1_key, row_index, box_index)
            top = _float_from_mapping(raw_box, y1_key, row_index, box_index)
            right = _float_from_mapping(raw_box, x2_key, row_index, box_index)
            bottom = _float_from_mapping(raw_box, y2_key, row_index, box_index)
            return left, top, right - left, bottom - top

    bbox = raw_box.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        left = _float_from_value(bbox[0], "bbox[0]", row_index, box_index)
        top = _float_from_value(bbox[1], "bbox[1]", row_index, box_index)
        width = _float_from_value(bbox[2], "bbox[2]", row_index, box_index)
        height = _float_from_value(bbox[3], "bbox[3]", row_index, box_index)
        return left, top, width, height

    raise ValueError(
        f"Row {row_index} box {box_index} is missing supported coordinates. "
        "Expected left/top/width/height, x/y/width/height, xmin/ymin/xmax/ymax, "
        "left/top/right/bottom, or bbox=[left, top, width, height]."
    )

def _raw_dict_from_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)

    raw = getattr(row, "raw", None)
    if isinstance(raw, dict):
        return dict(raw)

    payload: dict[str, Any] = {}

    for key in (
        "image_id",
        "source_ref",
        "split",
        "label_type",
        "label",
        "labels",
        "label_ref",
        "labels_ref",
        "annotations",
        "objects",
        "bboxes",
        "boxes",
        "bbox_annotations",
    ):
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

def _draw_text_with_background(
    *,
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    text_color: Color,
    background_color: Color,
) -> None:
    x, y = xy

    try:
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:  # pragma: no cover - older Pillow fallback
        text_width, text_height = draw.textsize(text, font=font)

    padding = 2
    draw.rectangle(
        [
            (x, y),
            (x + text_width + 2 * padding, y + text_height + 2 * padding),
        ],
        fill=background_color,
    )
    draw.text(
        (x + padding, y + padding),
        text,
        fill=text_color,
        font=font,
    )

def _looks_like_single_box(payload: dict[str, Any]) -> bool:
    has_class = any(
        key in payload
        for key in (
            "class_name",
            "class-name",
            "label",
            "class",
            "name",
            "category",
            "category_name",
            "category-name",
            "class_id",
            "class-id",
        )
    )

    has_ltrb = _has_all_keys(payload, ("left", "top", "width", "height"))
    has_xywh = _has_all_keys(payload, ("x", "y", "width", "height"))
    has_bbox = isinstance(payload.get("bbox"), (list, tuple)) and len(payload["bbox"]) == 4

    has_xyxy = any(
        _has_all_keys(payload, key_set)
        for key_set in (
            ("x_min", "y_min", "x_max", "y_max"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
        )
    )

    return has_class and (has_ltrb or has_xywh or has_bbox or has_xyxy)

def _has_all_keys(mapping: dict[str, Any], keys: Sequence[str]) -> bool:
    return all(key in mapping for key in keys)

def _float_from_mapping(
    mapping: dict[str, Any],
    key: str,
    row_index: int,
    box_index: int,
) -> float:
    return _float_from_value(mapping.get(key), key, row_index, box_index)

def _float_from_value(
    value: Any,
    field_name: str,
    row_index: int,
    box_index: int,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"Row {row_index} box {box_index} field {field_name!r} "
            f"must be numeric, got {value!r}"
        )

    return float(value)

def _resample_filter() -> int:
    try:
        return cast(int, Image.Resampling.LANCZOS)
    except AttributeError:  # pragma: no cover - older Pillow fallback
        return cast(int, Image.LANCZOS)

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

def _validate_number(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric, got {value!r}")

def _validate_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {value!r}")

    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")

def _validate_color(value: Any, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{field_name} must be an RGB tuple of length 3, got {value!r}")

    for idx, channel in enumerate(value):
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise TypeError(
                f"{field_name}[{idx}] must be an int in [0, 255], got {channel!r}"
            )

        if not 0 <= channel <= 255:
            raise ValueError(
                f"{field_name}[{idx}] must be in [0, 255], got {channel}"
            )