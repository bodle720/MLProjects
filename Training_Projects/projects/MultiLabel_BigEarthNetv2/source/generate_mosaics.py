"""
Generate image mosaic sheets for the CVDMS BigEarthNet v2 multi-label project.

Run from the project root:

    python source/generate_mosaics.py --config config.yaml

Examples:

    python source/generate_mosaics.py --config config.yaml --splits train val test
    python source/generate_mosaics.py --config config.yaml --limit 300
    python source/generate_mosaics.py --config config.yaml --rows 10 --cols 10 --tile-size 128
    python source/generate_mosaics.py --config config.yaml --group-mode cardinality
    python source/generate_mosaics.py --config config.yaml --group-mode signature

This script is manifest-driven, not folder-structure-driven. It reads the CVDMS
metadata and manifests, orders multi-label rows deterministically, loads images
through the configured image loader, and writes mosaic PNG files.

No image IDs, labels, or text are drawn onto the mosaic sheets.

Grouping modes:

    none:
        one ordered mosaic set per split

    cardinality:
        one folder per split/cardinality, with mixed label signatures in each
        cardinality group

    signature:
        one folder per split/cardinality, with separate mosaic files for each
        exact label signature
"""

import argparse
import json
from pathlib import Path
from typing import Any

import boto3
import yaml

from cvdms_training_common.mosaic_generators.common_utils import MosaicConfig
from cvdms_training_common.mosaic_generators.multi_label import (
    generate_multi_label_split_mosaics,
)

from helpers import (
    SPLITS,
    build_project_image_loader,
    iter_jsonl_s3,
    read_json_from_s3,
    require_dict,
    require_nonempty_string,
    require_positive_int,
    resolve_manifest_uris,
)

_ORDER_STRATEGIES = ("cardinality_signature", "source_ref", "image_id", "random")
_GROUP_MODES = ("none", "cardinality", "signature")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manifest-driven multi-label image mosaics."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to project config YAML file.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Dataset splits to render.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to logging.output_dir/../mosaics or outputs/mosaics.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of tile rows per mosaic sheet.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=10,
        help="Number of tile columns per mosaic sheet.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
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
        default="cardinality_signature",
        help="Ordering strategy for rows before tiling.",
    )
    parser.add_argument(
        "--group-mode",
        choices=list(_GROUP_MODES),
        default="none",
        help=(
            "Mosaic grouping mode. Use 'none' for split-wide mosaics, "
            "'cardinality' for one group per label count, or 'signature' "
            "for one mosaic set per exact label combination."
        ),
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
        "--summary-json",
        default=None,
        help="Optional path for a JSON summary. Defaults to <output-dir>/mosaic_summary.json.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    require_positive_int(args.rows, "--rows")
    require_positive_int(args.cols, "--cols")
    require_positive_int(args.tile_size, "--tile-size")

    if args.tile_width is not None:
        require_positive_int(args.tile_width, "--tile-width")

    if args.tile_height is not None:
        require_positive_int(args.tile_height, "--tile-height")

    if args.limit is not None:
        require_positive_int(args.limit, "--limit")

    config = load_config(args.config)

    data_config = require_dict(config.get("data"), "data")
    logging_config = config.get("logging") or {}
    if logging_config is not None and not isinstance(logging_config, dict):
        raise TypeError(
            f"logging must be a dictionary when provided, got {type(logging_config).__name__}"
        )

    aws_config = config.get("aws") or {}
    if not isinstance(aws_config, dict):
        raise TypeError(f"aws must be a dictionary when provided, got {type(aws_config).__name__}")

    metadata_uri = require_nonempty_string(config.get("metadata_uri"), "metadata_uri")
    output_dir = resolve_output_dir(
        output_dir_arg=args.output_dir,
        logging_config=logging_config,
    )

    s3_client = make_s3_client(
        profile_name=aws_config.get("profile_name"),
        region_name=aws_config.get("region_name"),
    )

    image_loader = build_project_image_loader(
        data_config=data_config,
        s3_client=s3_client,
    )

    metadata = read_json_from_s3(metadata_uri, s3_client=s3_client)
    validate_metadata(metadata)

    manifest_uris = resolve_manifest_uris(metadata)
    class_to_idx = require_class_to_idx(metadata)

    mosaic_config = MosaicConfig(
        rows=args.rows,
        cols=args.cols,
        tile_width=args.tile_width or args.tile_size,
        tile_height=args.tile_height or args.tile_size,
    )

    print("")
    print("CVDMS multi-label mosaic generation")
    print("=" * 80)
    print(f"metadata_uri:   {metadata_uri}")
    print(f"dataset_id:     {metadata.get('dataset_id')}")
    print(f"version:        {metadata.get('version')}")
    print(f"label_type:     {metadata.get('label_type')}")
    print(f"num_classes:    {len(class_to_idx)}")
    print(f"output_dir:     {output_dir}")
    print(f"splits:         {list(args.splits)}")
    print(f"grid:           {mosaic_config.rows}x{mosaic_config.cols}")
    print(f"tile_size:      {mosaic_config.tile_width}x{mosaic_config.tile_height}")
    print(f"order_strategy: {args.order_strategy}")
    print(f"group_mode:     {args.group_mode}")
    print(f"limit_per_split:{args.limit}")

    all_results: list[dict[str, Any]] = []

    for split in args.splits:
        manifest_uri = manifest_uris[split]

        print("")
        print(f"Processing split: {split}")
        print("=" * 80)
        print(f"manifest_uri: {manifest_uri}")

        rows = read_manifest_rows(
            manifest_uri=manifest_uri,
            s3_client=s3_client,
        )

        print(f"manifest rows: {len(rows)}")

        result = generate_multi_label_split_mosaics(
            rows=rows,
            image_loader=image_loader,
            output_dir=output_dir,
            split=split,
            config=mosaic_config,
            class_to_idx=class_to_idx,
            order_strategy=args.order_strategy,
            group_mode=args.group_mode,
            random_seed=args.random_seed,
            max_items=args.limit,
            allow_empty_labels=False,
            validate_split=True,
        )

        result_dict = result.to_dict()
        all_results.append(result_dict)

        print(f"rendered items: {result.total_items}")
        print(f"saved sheets:   {len(result.sheets)}")
        print(f"split out dir:  {result.output_dir}")

    summary = {
        "metadata_uri": metadata_uri,
        "dataset_id": metadata.get("dataset_id"),
        "version": metadata.get("version"),
        "label_type": metadata.get("label_type"),
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
        "splits": all_results,
    }

    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "mosaic_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("")
    print("Mosaic generation complete.")
    print(f"Summary: {summary_path}")

def read_manifest_rows(
    *,
    manifest_uri: str,
    s3_client,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for _, row in iter_jsonl_s3(manifest_uri, s3_client=s3_client):
        rows.append(row)

    return rows

def validate_metadata(metadata: dict[str, Any]) -> None:
    label_type = metadata.get("label_type")
    if label_type != "multi-label":
        raise ValueError(f"Expected metadata.label_type='multi-label', got {label_type!r}")

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

    if len(out) < 2:
        raise ValueError(f"class_to_idx must contain at least 2 classes, got {len(out)}")

    return out

def resolve_output_dir(
    *,
    output_dir_arg: str | None,
    logging_config: dict[str, Any],
) -> Path:
    if output_dir_arg is not None:
        return Path(require_nonempty_string(output_dir_arg, "--output-dir"))

    configured_output_dir = logging_config.get("output_dir")

    if isinstance(configured_output_dir, str) and configured_output_dir.strip():
        return Path(configured_output_dir).parent / "mosaics"

    return Path("outputs") / "mosaics"

def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise TypeError(f"Config must parse to a dictionary, got {type(payload).__name__}")

    return payload

def make_s3_client(
    *,
    profile_name: str | None,
    region_name: str | None,
):
    if profile_name:
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
    else:
        session = boto3.Session(region_name=region_name)

    return session.client("s3")

if __name__ == "__main__":
    main()