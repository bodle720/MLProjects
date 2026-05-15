"""
Generate object-detection mosaic sheets for Project 3.

Run from the project root:

    python training/data/generate_mosaics.py

Examples:

    python training/data/generate_mosaics.py --splits train val test
    python training/data/generate_mosaics.py --limit 100
    python training/data/generate_mosaics.py --rows 5 --cols 5 --tile-size 256
    python training/data/generate_mosaics.py --order-strategy box_count_desc
    python training/data/generate_mosaics.py --order-strategy random --random-seed 42
    python training/data/generate_mosaics.py --group-mode class

This script reads cached CVDMS manifests produced by:

    training/data/cache_dataset.py

Expected cached input layout:

    training/data/cached/
    ├── manifests/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── test.jsonl
    ├── images/
    └── labels/

Each cached manifest row should include:

    local_image_path
    local_label_paths

The script loads local label JSON files, injects their annotations into rows,
loads local images, and writes bbox-overlay mosaic PNGs.

This script intentionally does not read from S3.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from cvdms_training_common.mosaic_generators.common_utils import MosaicConfig
from cvdms_training_common.mosaic_generators.object_detection import (
    generate_object_detection_split_mosaics,
)

from helpers import (
    SPLITS,
    require_nonempty_string,
    require_positive_int,
)

_ORDER_STRATEGIES = (
    "box_count_desc",
    "box_count_asc",
    "class_box_count",
    "image_id",
    "source_ref",
    "random",
)

_GROUP_MODES = (
    "none",
    "class",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local object-detection mosaic sheets from cached CVDMS manifests."
    )
    parser.add_argument(
        "--metadata-path",
        default="training/data/original/manifests/metadata.json",
        help="Path to CVDMS metadata.json.",
    )
    parser.add_argument(
        "--cached-manifest-dir",
        default="training/data/cached/manifests",
        help="Directory containing cached train/val/test JSONL manifests.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/data/mosaics",
        help="Output directory for generated mosaic PNGs.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Dataset splits to render.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of tile rows per mosaic sheet.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of tile columns per mosaic sheet.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Square tile size in pixels. Sets both tile width and tile height.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=None,
        help="Optional tile width override in pixels.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=None,
        help="Optional tile height override in pixels.",
    )
    parser.add_argument(
        "--order-strategy",
        choices=list(_ORDER_STRATEGIES),
        default="box_count_desc",
        help="Ordering strategy for rows before tiling.",
    )
    parser.add_argument(
        "--group-mode",
        choices=list(_GROUP_MODES),
        default="none",
        help="Use 'none' for split-wide mosaics or 'class' for class-grouped mosaics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum images per split after ordering. Useful for previews.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used only when --order-strategy random is selected.",
    )
    parser.add_argument(
        "--draw-labels",
        action="store_true",
        help="Draw class names near boxes. Usually too cluttered for wheat-head mosaics.",
    )
    parser.add_argument(
        "--no-draw-box-count",
        action="store_true",
        help="Disable the small box-count label in each tile.",
    )
    parser.add_argument(
        "--box-width",
        type=int,
        default=2,
        help="Bounding-box line width in pixels.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path for a JSON summary. Defaults to <output-dir>/mosaic_summary.json.",
    )
    parser.add_argument(
        "--fail-on-unknown-class",
        action="store_true",
        help=(
            "Fail if a label JSON contains a class not present in metadata.class_to_idx. "
            "By default, unknown-class boxes are skipped and counted."
        ),
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    require_positive_int(args.rows, "--rows")
    require_positive_int(args.cols, "--cols")
    require_positive_int(args.tile_size, "--tile-size")
    require_positive_int(args.box_width, "--box-width")

    if args.tile_width is not None:
        require_positive_int(args.tile_width, "--tile-width")

    if args.tile_height is not None:
        require_positive_int(args.tile_height, "--tile-height")

    if args.limit is not None:
        require_positive_int(args.limit, "--limit")

    metadata_path = Path(args.metadata_path)
    cached_manifest_dir = Path(args.cached_manifest_dir)
    output_dir = Path(args.output_dir)

    metadata = read_json_object(metadata_path)
    validate_metadata(metadata)
    class_to_idx = require_class_to_idx(metadata)

    mosaic_config = MosaicConfig(
        rows=args.rows,
        cols=args.cols,
        tile_width=args.tile_width or args.tile_size,
        tile_height=args.tile_height or args.tile_size,
    )

    print("")
    print("CVDMS object-detection mosaic generation")
    print("=" * 80)
    print(f"metadata_path:        {metadata_path}")
    print(f"dataset_id:           {metadata.get('dataset_id')}")
    print(f"version:              {metadata.get('version')}")
    print(f"label_type:           {metadata.get('label_type')}")
    print(f"class_to_idx:         {class_to_idx}")
    print(f"cached_manifest_dir:  {cached_manifest_dir}")
    print(f"output_dir:           {output_dir}")
    print(f"splits:               {list(args.splits)}")
    print(f"grid:                 {mosaic_config.rows}x{mosaic_config.cols}")
    print(f"tile_size:            {mosaic_config.tile_width}x{mosaic_config.tile_height}")
    print(f"order_strategy:       {args.order_strategy}")
    print(f"group_mode:           {args.group_mode}")
    print(f"limit_per_split:      {args.limit}")
    print(f"draw_labels:          {args.draw_labels}")
    print(f"draw_box_count:       {not args.no_draw_box_count}")
    print(f"box_width:            {args.box_width}")

    all_results: list[dict[str, Any]] = []
    all_filter_summaries: dict[str, dict[str, int]] = {}

    for split in args.splits:
        manifest_path = cached_manifest_dir / f"{split}.jsonl"

        print("")
        print(f"Processing split: {split}")
        print("=" * 80)
        print(f"manifest_path: {manifest_path}")

        rows, filter_summary = read_mosaic_rows(
            manifest_path=manifest_path,
            split=split,
            class_to_idx=class_to_idx,
            fail_on_unknown_class=args.fail_on_unknown_class,
        )

        print(f"manifest rows:              {filter_summary['rows_seen']}")
        print(f"rows with usable boxes:     {len(rows)}")
        print(f"annotations seen:           {filter_summary['annotations_seen']}")
        print(f"annotations kept:           {filter_summary['annotations_kept']}")
        print(f"unknown-class boxes skipped:{filter_summary['annotations_skipped_unknown_class']}")

        if not rows:
            raise ValueError(f"No rows with usable annotations for split {split!r}")

        result = generate_object_detection_split_mosaics(
            rows=rows,
            image_loader=load_local_image,
            output_dir=output_dir,
            split=split,
            config=mosaic_config,
            class_to_idx=class_to_idx,
            order_strategy=args.order_strategy,
            group_mode=args.group_mode,
            random_seed=args.random_seed,
            max_items=args.limit,
            validate_split=True,
            draw_labels=args.draw_labels,
            draw_box_count=not args.no_draw_box_count,
            box_width=args.box_width,
        )

        result_dict = result.to_dict()
        all_results.append(result_dict)
        all_filter_summaries[split] = filter_summary

        print(f"rendered items: {result.total_items}")
        print(f"rendered boxes: {result.total_boxes}")
        print(f"saved sheets:   {len(result.sheets)}")
        print(f"split out dir:  {result.output_dir}")

    summary = {
        "metadata_path": str(metadata_path),
        "dataset_id": metadata.get("dataset_id"),
        "version": metadata.get("version"),
        "label_type": metadata.get("label_type"),
        "class_to_idx": class_to_idx,
        "cached_manifest_dir": str(cached_manifest_dir),
        "output_dir": str(output_dir),
        "grid": {
            "rows": mosaic_config.rows,
            "cols": mosaic_config.cols,
            "tile_width": mosaic_config.tile_width,
            "tile_height": mosaic_config.tile_height,
            "tiles_per_sheet": mosaic_config.tiles_per_sheet,
        },
        "order_strategy": args.order_strategy,
        "group_mode": args.group_mode,
        "limit_per_split": args.limit,
        "draw_labels": args.draw_labels,
        "draw_box_count": not args.no_draw_box_count,
        "box_width": args.box_width,
        "filter_summaries": all_filter_summaries,
        "splits": all_results,
    }

    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "mosaic_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("")
    print("Mosaic generation complete.")
    print(f"Summary: {summary_path}")

def read_mosaic_rows(
    *,
    manifest_path: Path,
    split: str,
    class_to_idx: dict[str, int],
    fail_on_unknown_class: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    summary = {
        "rows_seen": 0,
        "rows_kept": 0,
        "rows_skipped_no_usable_boxes": 0,
        "annotations_seen": 0,
        "annotations_kept": 0,
        "annotations_skipped_unknown_class": 0,
    }

    for line_number, row in iter_jsonl_local(manifest_path):
        summary["rows_seen"] += 1

        row_split = require_nonempty_string(
            row.get("split"),
            f"{manifest_path}:{line_number}:split",
        )

        if row_split != split:
            raise ValueError(
                f"Split mismatch in {manifest_path}:{line_number}: "
                f"expected {split!r}, got {row_split!r}"
            )

        local_image_path = require_nonempty_string(
            row.get("local_image_path"),
            f"{manifest_path}:{line_number}:local_image_path",
        )

        local_label_paths = local_label_paths_from_row(
            row=row,
            manifest_path=manifest_path,
            line_number=line_number,
        )

        annotations: list[dict[str, Any]] = []

        for label_path_text in local_label_paths:
            label_path = Path(label_path_text)
            label_payload = read_json_object(label_path)
            label_annotations = annotations_from_label_payload(
                payload=label_payload,
                label_path=label_path,
            )

            for annotation in label_annotations:
                summary["annotations_seen"] += 1
                class_name = require_nonempty_string(
                    annotation.get("class_name"),
                    f"{label_path}:annotation.class_name",
                )

                if class_name not in class_to_idx:
                    summary["annotations_skipped_unknown_class"] += 1

                    if fail_on_unknown_class:
                        raise ValueError(
                            f"Unknown class {class_name!r} in {label_path}. "
                            f"Allowed classes: {sorted(class_to_idx)}"
                        )

                    continue

                annotations.append(dict(annotation))
                summary["annotations_kept"] += 1

        if not annotations:
            summary["rows_skipped_no_usable_boxes"] += 1
            continue

        mosaic_row = dict(row)

        # The object-detection mosaic generator expects source_ref to be what
        # the image_loader can open. For this local-only script, use the cached
        # local image path while preserving the original S3 source_ref.
        mosaic_row["original_source_ref"] = row.get("source_ref")
        mosaic_row["source_ref"] = local_image_path
        mosaic_row["annotations"] = annotations

        rows.append(mosaic_row)
        summary["rows_kept"] += 1

    return rows, summary

def local_label_paths_from_row(
    *,
    row: dict[str, Any],
    manifest_path: Path,
    line_number: int,
) -> list[str]:
    value = row.get("local_label_paths")

    if not isinstance(value, list):
        raise TypeError(
            f"local_label_paths must be a list at {manifest_path}:{line_number}, "
            f"got {type(value).__name__}"
        )

    if not value:
        raise ValueError(f"local_label_paths cannot be empty at {manifest_path}:{line_number}")

    paths: list[str] = []

    for idx, item in enumerate(value):
        path = require_nonempty_string(
            item,
            f"{manifest_path}:{line_number}:local_label_paths[{idx}]",
        )
        paths.append(path)

    return paths

def annotations_from_label_payload(
    *,
    payload: dict[str, Any],
    label_path: Path,
) -> list[dict[str, Any]]:
    value = payload.get("annotations")

    if not isinstance(value, list):
        raise TypeError(
            f"Label JSON must contain annotations list at {label_path}, "
            f"got {type(value).__name__}"
        )

    annotations: list[dict[str, Any]] = []

    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(
                f"Annotation {idx} in {label_path} must be a dictionary, "
                f"got {type(item).__name__}"
            )

        annotations.append(dict(item))

    return annotations

def load_local_image(path_text: str) -> Image.Image:
    path = Path(path_text)

    if not path.exists():
        raise FileNotFoundError(f"Local image does not exist: {path}")

    return Image.open(path)

def iter_jsonl_local(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()

            if not text:
                continue

            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc

            if not isinstance(row, dict):
                raise TypeError(
                    f"Manifest row must be a JSON object in {path} at line {line_number}, "
                    f"got {type(row).__name__}"
                )

            yield line_number, row

def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object from {path}, got {type(payload).__name__}")

    return payload

def validate_metadata(metadata: dict[str, Any]) -> None:
    label_type = metadata.get("label_type")
    if label_type != "object-detection":
        raise ValueError(f"Expected metadata.label_type='object-detection', got {label_type!r}")

    dataset_id = metadata.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("metadata.dataset_id must be a non-empty string")

    if "version" not in metadata:
        raise ValueError("metadata.version is required")

def require_class_to_idx(metadata: dict[str, Any]) -> dict[str, int]:
    value = metadata.get("class_to_idx")

    if not isinstance(value, dict):
        raise TypeError(
            f"metadata.class_to_idx must be a dictionary, got {type(value).__name__}"
        )

    out: dict[str, int] = {}

    for key, raw_idx in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"class_to_idx contains invalid class name: {key!r}")

        if isinstance(raw_idx, bool):
            raise TypeError(f"class_to_idx[{key!r}] must be an int, got {raw_idx!r}")

        idx = int(raw_idx)
        if idx < 0:
            raise ValueError(f"class_to_idx[{key!r}] must be >= 0, got {idx}")

        out[key] = idx

    if not out:
        raise ValueError("class_to_idx cannot be empty")

    return out

if __name__ == "__main__":
    main()