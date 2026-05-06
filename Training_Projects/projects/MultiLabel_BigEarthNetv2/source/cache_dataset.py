"""
Cache CVDMS dataset images locally for the BigEarthNet v2 multi-label project.

Run from the project root:

    python source/cache_dataset.py --config config.yaml

Useful test commands:

    python source/cache_dataset.py --config config.yaml --dry-run --limit 5
    python source/cache_dataset.py --config config.yaml --limit 20
    python source/cache_dataset.py --config config.yaml --verify-only
    python source/cache_dataset.py --config config.yaml

This script reads the CVDMS dataset metadata and split manifests from S3,
collects each manifest row's source_ref, and downloads the referenced images
into the configured local mirror directory.

The local mirror layout intentionally matches LocalMirrorImageLoader:

    source_ref:
        s3://bucket/canonical/images/bigearthnetv2/images/training/img.png

    local file:
        <cache_dir>/canonical/images/bigearthnetv2/images/training/img.png

After this script succeeds, train.py and inspect_dataset.py can use:

    data.image_loader.mode: local_mirror

without repeatedly reading image bytes from S3 during training.
"""

import argparse
import os
from pathlib import Path
from typing import Any

import boto3
import yaml

from cvdms_training_common.s3_io import parse_s3_uri

from helpers import (
    require_dict,
    require_nonempty_string,
    require_positive_int,
    iter_jsonl_s3,
    read_json_from_s3,
    resolve_manifest_uris,
    SPLITS
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CVDMS manifest images into a local mirror cache."
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
        help="Dataset splits to cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and report image URIs without downloading.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that expected cache files exist. Do not download.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process, useful for testing.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Progress print frequency.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided")

    print_every = require_positive_int(args.print_every, "--print-every")

    config = load_config(args.config)
    data_config = require_dict(config.get("data"), "data")
    aws_config = config.get("aws") or {}

    metadata_uri = require_nonempty_string(config.get("metadata_uri"), "metadata_uri")
    cache_dir = get_cache_dir(data_config)

    s3_client = make_s3_client(
        profile_name=aws_config.get("profile_name"),
        region_name=aws_config.get("region_name"),
    )

    print("")
    print("CVDMS local image cache")
    print("=" * 80)
    print(f"metadata_uri: {metadata_uri}")
    print(f"cache_dir:    {cache_dir}")
    print(f"splits:       {list(args.splits)}")
    print(f"force:        {args.force}")
    print(f"dry_run:      {args.dry_run}")
    print(f"verify_only:  {args.verify_only}")

    metadata = read_json_from_s3(metadata_uri, s3_client=s3_client)
    manifest_uris = resolve_manifest_uris(metadata)

    selected_manifest_uris = {
        split: manifest_uris[split]
        for split in args.splits
    }

    print("")
    print("Manifest URIs")
    print("=" * 80)
    for split, uri in selected_manifest_uris.items():
        print(f"{split}: {uri}")

    source_refs = collect_source_refs(
        manifest_uris=selected_manifest_uris,
        s3_client=s3_client,
    )

    if args.limit is not None:
        source_refs = source_refs[:args.limit]

    print("")
    print("Image references")
    print("=" * 80)
    print(f"unique source_ref count: {len(source_refs)}")

    if not source_refs:
        raise ValueError("No source_ref values found in selected manifests")

    if args.dry_run:
        print("")
        print("Dry run complete. No files downloaded.")
        preview_source_refs(
            source_refs=source_refs,
            cache_dir=cache_dir,
        )
        return

    cache_dir.mkdir(parents=True, exist_ok=True)

    result = cache_images(
        source_refs=source_refs,
        cache_dir=cache_dir,
        s3_client=s3_client,
        force=args.force,
        verify_only=args.verify_only,
        print_every=print_every,
    )

    print("")
    print("Cache summary")
    print("=" * 80)
    print(f"expected:     {result['expected']}")
    print(f"downloaded:   {result['downloaded']}")
    print(f"skipped:      {result['skipped']}")
    print(f"verified:     {result['verified']}")
    print(f"missing:      {result['missing']}")
    print(f"failed:       {result['failed']}")

    if result["missing"] or result["failed"]:
        raise RuntimeError(
            "Cache did not complete cleanly. "
            f"missing={result['missing']} failed={result['failed']}"
        )

    print("")
    print("Dataset image cache completed successfully.")

def get_cache_dir(data_config: dict[str, Any]) -> Path:
    image_loader_config = data_config.get("image_loader")

    if not isinstance(image_loader_config, dict):
        raise TypeError(
            "data.image_loader must be a dictionary with mode='local_mirror' "
            "and cache_dir set"
        )

    mode = str(image_loader_config.get("mode", "")).strip().lower()
    if mode != "local_mirror":
        raise ValueError(
            "cache_dataset.py is intended for data.image_loader.mode='local_mirror', "
            f"got {mode!r}"
        )

    cache_dir = require_nonempty_string(
        image_loader_config.get("cache_dir"),
        "data.image_loader.cache_dir",
    )

    return Path(cache_dir)

def collect_source_refs(
    *,
    manifest_uris: dict[str, str],
    s3_client,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    for split, manifest_uri in manifest_uris.items():
        split_count = 0

        for line_number, row in iter_jsonl_s3(manifest_uri, s3_client=s3_client):
            source_ref = source_ref_from_row(
                row=row,
                manifest_uri=manifest_uri,
                line_number=line_number,
            )

            if source_ref not in seen:
                seen.add(source_ref)
                ordered.append(source_ref)

            split_count += 1

        print(f"{split}: read {split_count} manifest row(s)")

    return ordered

def source_ref_from_row(
    *,
    row: dict[str, Any],
    manifest_uri: str,
    line_number: int,
) -> str:
    value = row.get("source_ref")
    if value is None:
        value = row.get("source-ref")

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Manifest row missing non-empty source_ref/source-ref at "
            f"{manifest_uri}:{line_number}"
        )

    source_ref = value.strip()

    if not source_ref.startswith("s3://"):
        raise ValueError(
            f"Expected source_ref to be an S3 URI at {manifest_uri}:{line_number}, "
            f"got {source_ref!r}"
        )

    return source_ref

def cache_images(
    *,
    source_refs: list[str],
    cache_dir: Path,
    s3_client,
    force: bool,
    verify_only: bool,
    print_every: int,
) -> dict[str, int]:
    result = {
        "expected": len(source_refs),
        "downloaded": 0,
        "skipped": 0,
        "verified": 0,
        "missing": 0,
        "failed": 0,
    }

    for index, source_ref in enumerate(source_refs, start=1):
        parsed = parse_s3_uri(source_ref)
        local_path = cache_dir / parsed.key

        try:
            if local_path.exists() and not force:
                result["skipped"] += 1
                result["verified"] += 1
            elif verify_only:
                result["missing"] += 1
                print(f"[missing] {source_ref} -> {local_path}")
            else:
                download_s3_object(
                    bucket=parsed.bucket,
                    key=parsed.key,
                    destination=local_path,
                    s3_client=s3_client,
                )
                result["downloaded"] += 1
                result["verified"] += 1

        except Exception as exc:
            result["failed"] += 1
            print(f"[failed] {source_ref} -> {local_path}")
            print(f"         {type(exc).__name__}: {exc}")

        if index == 1 or index % print_every == 0 or index == len(source_refs):
            print(
                " | ".join(
                    [
                        f"processed={index}/{len(source_refs)}",
                        f"downloaded={result['downloaded']}",
                        f"skipped={result['skipped']}",
                        f"missing={result['missing']}",
                        f"failed={result['failed']}",
                    ]
                )
            )

    return result

def download_s3_object(
    *,
    bucket: str,
    key: str,
    destination: Path,
    s3_client,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path = destination.with_name(destination.name + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    try:
        s3_client.download_file(
            Bucket=bucket,
            Key=key,
            Filename=str(temp_path),
        )
        os.replace(temp_path, destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

def preview_source_refs(
    *,
    source_refs: list[str],
    cache_dir: Path,
    max_items: int = 5,
) -> None:
    print("")
    print("Preview")
    print("=" * 80)

    for source_ref in source_refs[:max_items]:
        parsed = parse_s3_uri(source_ref)
        print(source_ref)
        print(f"  -> {cache_dir / parsed.key}")

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