"""
Generate image mosaic sheets for the CVDMS EuroSAT single-label project.

Run from the project root:

    python source/generate_mosaics.py --config config.yaml

Examples:

    python source/generate_mosaics.py --config config.yaml --splits train val test
    python source/generate_mosaics.py --config config.yaml --splits train --limit 100
    python source/generate_mosaics.py --config config.yaml --rows 10 --cols 10 --tile-size 128
    python source/generate_mosaics.py --config config.yaml --no-group-by-class

This script is manifest-driven, not folder-structure-driven. It reads the CVDMS
metadata and manifests, groups single-label rows by split and class, loads images
through the configured image loader, and writes mosaic PNG files.

No image IDs, labels, or text are drawn onto the mosaic sheets.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import boto3
import yaml
from botocore.exceptions import ClientError

from cvdms_training_common.image_loading import LocalMirrorImageLoader, S3ImageLoader
from cvdms_training_common.mosaic_generators.common_utils import MosaicConfig
from cvdms_training_common.mosaic_generators.single_label import (
    generate_single_label_split_mosaics,
)
from cvdms_training_common.s3_io import parse_s3_uri

_SPLITS = ("train", "val", "test")
_ORDER_STRATEGIES = ("class_image_id", "class_source_ref", "image_id", "source_ref", "random")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manifest-driven single-label image mosaics."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to project config YAML file.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(_SPLITS),
        choices=list(_SPLITS),
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
        default="class_image_id",
        help="Ordering strategy for rows before tiling.",
    )
    parser.add_argument(
        "--no-group-by-class",
        action="store_true",
        help="Disable class-pure mosaics and create split-level mosaics instead.",
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
    if not isinstance(logging_config, dict):
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

    group_by_class = not args.no_group_by_class

    print("")
    print("CVDMS single-label mosaic generation")
    print("=" * 80)
    print(f"metadata_uri:     {metadata_uri}")
    print(f"dataset_id:       {metadata.get('dataset_id')}")
    print(f"version:          {metadata.get('version')}")
    print(f"label_type:       {metadata.get('label_type')}")
    print(f"num_classes:      {len(class_to_idx)}")
    print(f"output_dir:       {output_dir}")
    print(f"splits:           {list(args.splits)}")
    print(f"grid:             {mosaic_config.rows}x{mosaic_config.cols}")
    print(f"tile_size:        {mosaic_config.tile_width}x{mosaic_config.tile_height}")
    print(f"order_strategy:   {args.order_strategy}")
    print(f"group_by_class:   {group_by_class}")
    print(f"limit_per_split:  {args.limit}")

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

        result = generate_single_label_split_mosaics(
            rows=rows,
            image_loader=image_loader,
            output_dir=output_dir,
            split=split,
            config=mosaic_config,
            class_to_idx=class_to_idx,
            order_strategy=args.order_strategy,
            group_by_class=group_by_class,
            random_seed=args.random_seed,
            max_items=args.limit,
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
        "group_by_class": group_by_class,
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

def build_project_image_loader(
    *,
    data_config: dict[str, Any],
    s3_client,
):
    """
    Build the project image loader.

    Defaults to S3ImageLoader because Project 1 originally reads CVDMS images
    directly from S3. If data.image_loader.mode='local_mirror' is added later,
    this can use LocalMirrorImageLoader without changing the mosaic code.
    """
    loader_config = data_config.get("image_loader") or {}

    if not isinstance(loader_config, dict):
        raise TypeError(
            f"data.image_loader must be a dictionary, got {type(loader_config).__name__}"
        )

    mode = str(loader_config.get("mode", "s3")).strip().lower()

    if mode == "s3":
        return S3ImageLoader(s3_client=s3_client)

    if mode == "local_mirror":
        cache_dir = require_nonempty_string(
            loader_config.get("cache_dir"),
            "data.image_loader.cache_dir",
        )
        return LocalMirrorImageLoader(local_root=cache_dir)

    raise ValueError(
        "data.image_loader.mode must be one of {'s3', 'local_mirror'}, "
        f"got {mode!r}"
    )

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
    if label_type != "single-label":
        raise ValueError(f"Expected metadata.label_type='single-label', got {label_type!r}")

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

def read_json_from_s3(uri: str, *, s3_client) -> dict[str, Any]:
    data = read_s3_bytes(uri, s3_client=s3_client)

    try:
        payload = json.loads(data.decode("utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"S3 object is not valid JSON: {uri}") from exc

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object from {uri}, got {type(payload).__name__}")

    return payload

def iter_jsonl_s3(uri: str, *, s3_client) -> Iterable[tuple[int, dict[str, Any]]]:
    data = read_s3_bytes(uri, s3_client=s3_client)

    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), start=1):
        text = line.strip()

        if not text:
            continue

        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {uri} at line {line_number}") from exc

        if not isinstance(row, dict):
            raise TypeError(
                f"Manifest row must be a JSON object in {uri} at line {line_number}, "
                f"got {type(row).__name__}"
            )

        yield line_number, row

def read_s3_bytes(uri: str, *, s3_client) -> bytes:
    parsed = parse_s3_uri(uri)

    try:
        response = s3_client.get_object(
            Bucket=parsed.bucket,
            Key=parsed.key,
        )
        return response["Body"].read()
    except ClientError as exc:
        raise RuntimeError(f"Failed to read S3 object: {uri}") from exc

def resolve_manifest_uris(metadata: dict[str, Any]) -> dict[str, str]:
    """
    Resolve train/val/test manifest URIs from CVDMS metadata.json.

    This accepts the same artifact layouts used by Project 2, including nested
    metadata["artifacts"] structures.
    """
    resolved: dict[str, str] = {}

    preferred_roots = [
        metadata.get("manifest_uris"),
        metadata.get("manifest_s3_uris"),
        metadata.get("split_manifest_uris"),
        metadata.get("split_manifests"),
        metadata.get("manifests"),
        metadata.get("splits"),
        metadata.get("artifacts"),
        metadata,
    ]

    for root in preferred_roots:
        collect_manifest_uris_recursive(root, resolved)

        if all(split in resolved for split in _SPLITS):
            return {split: resolved[split] for split in _SPLITS}

    available_keys = sorted(str(key) for key in metadata.keys())
    artifacts = metadata.get("artifacts")
    artifact_keys = sorted(str(key) for key in artifacts.keys()) if isinstance(artifacts, dict) else None

    raise ValueError(
        "Could not resolve train/val/test manifest URIs from metadata.json. "
        f"Available top-level keys: {available_keys}. "
        f"Artifact keys: {artifact_keys}"
    )

def collect_manifest_uris_recursive(value: Any, resolved: dict[str, str]) -> None:
    if all(split in resolved for split in _SPLITS):
        return

    if isinstance(value, dict):
        collect_manifest_uris_from_current_dict(value, resolved)

        for child in value.values():
            collect_manifest_uris_recursive(child, resolved)

            if all(split in resolved for split in _SPLITS):
                return

    elif isinstance(value, list):
        for child in value:
            collect_manifest_uris_recursive(child, resolved)

            if all(split in resolved for split in _SPLITS):
                return

def collect_manifest_uris_from_current_dict(
    value: dict[str, Any],
    resolved: dict[str, str],
) -> None:
    for split in _SPLITS:
        if split in resolved:
            continue

        item = value.get(split)
        uri = manifest_uri_from_item(item)
        if uri is not None:
            resolved[split] = uri
            continue

        for key in (
            f"{split}_manifest_uri",
            f"{split}_manifest_s3_uri",
            f"{split}_manifest",
            f"{split}_jsonl_uri",
            f"{split}_jsonl_s3_uri",
            f"{split}_uri",
            f"{split}_s3_uri",
        ):
            uri = manifest_uri_from_item(value.get(key))
            if uri is not None:
                resolved[split] = uri
                break

    split_value = value.get("split")
    if isinstance(split_value, str):
        split = split_value.strip().lower()

        if split in _SPLITS and split not in resolved:
            uri = manifest_uri_from_item(value)
            if uri is not None:
                resolved[split] = uri

    name_value = value.get("name")
    if isinstance(name_value, str):
        split = name_value.strip().lower()

        if split in _SPLITS and split not in resolved:
            uri = manifest_uri_from_item(value)
            if uri is not None:
                resolved[split] = uri

def manifest_uri_from_item(item: Any) -> str | None:
    if isinstance(item, str):
        text = item.strip()
        if looks_like_manifest_uri(text):
            return text
        return None

    if isinstance(item, dict):
        for key in (
            "uri",
            "s3_uri",
            "manifest_uri",
            "manifest_s3_uri",
            "jsonl_uri",
            "jsonl_s3_uri",
            "path",
            "s3_path",
        ):
            value = item.get(key)

            if isinstance(value, str):
                text = value.strip()
                if looks_like_manifest_uri(text):
                    return text

    return None

def looks_like_manifest_uri(value: str) -> bool:
    text = value.strip()

    if not text.startswith("s3://"):
        return False

    lowered = text.lower()

    return (
        "manifest" in lowered
        or lowered.endswith(".jsonl")
        or "/manifests/" in lowered
    )

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

def require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dictionary, got {type(value).__name__}")

    return value

def require_nonempty_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text

def require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {value!r}")

    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")

    return value

if __name__ == "__main__":
    main()