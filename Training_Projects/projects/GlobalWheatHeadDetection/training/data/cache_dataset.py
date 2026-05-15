"""
Cache CVDMS object-detection dataset images and bbox label JSONs locally.

Run from the Project 3 root:

    python training/data/cache_dataset.py ^
      --bbox-label-prefix s3://YOUR_FILE_BUCKET/canonical/labels/bounding-boxes/ ^
      --profile-name your_profile ^
      --region-name us-east-1

Useful test commands:

    python training/data/cache_dataset.py --bbox-label-prefix s3://YOUR_FILE_BUCKET/canonical/labels/bounding-boxes/ --dry-run --limit 5
    python training/data/cache_dataset.py --bbox-label-prefix s3://YOUR_FILE_BUCKET/canonical/labels/bounding-boxes/ --limit 20
    python training/data/cache_dataset.py --bbox-label-prefix s3://YOUR_FILE_BUCKET/canonical/labels/bounding-boxes/ --verify-only

Expected input layout:

    training/data/original/manifests/
    ├── metadata.json
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl

Expected output layout:

    training/data/cached/
    ├── images/
    │   └── <bucket>/<mirrored-s3-key>
    ├── labels/
    │   └── <bucket>/canonical/labels/bounding-boxes/<bbox_annotation_id>.json
    ├── manifests/
    │   ├── train.jsonl
    │   ├── val.jsonl
    │   └── test.jsonl
    └── cache_report.json

This script intentionally does not convert to YOLO format. It creates a faithful
local cache of CVDMS images and CVDMS bbox label JSON artifacts. Later scripts
can use the cached manifests for mosaics and CVDMS-to-YOLO conversion.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3

from cvdms_training_common.s3_io import parse_s3_uri

from helpers import (
    SPLITS,
    require_nonempty_string,
    require_positive_int,
)

@dataclass(frozen=True)
class CachedRow:
    split: str
    row: dict[str, Any]
    source_ref: str
    local_image_path: Path
    bbox_label_uris: list[str]
    local_label_paths: list[Path]

    def to_manifest_row(self) -> dict[str, Any]:
        payload = dict(self.row)
        payload["local_image_path"] = str(self.local_image_path)
        payload["bbox_label_uris"] = list(self.bbox_label_uris)
        payload["local_label_paths"] = [str(path) for path in self.local_label_paths]
        return payload

@dataclass
class CacheSummary:
    expected_images: int = 0
    expected_labels: int = 0
    downloaded_images: int = 0
    downloaded_labels: int = 0
    skipped_images: int = 0
    skipped_labels: int = 0
    verified_images: int = 0
    verified_labels: int = 0
    missing_images: int = 0
    missing_labels: int = 0
    failed_images: int = 0
    failed_labels: int = 0
    rows_by_split: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_images": self.expected_images,
            "expected_labels": self.expected_labels,
            "downloaded_images": self.downloaded_images,
            "downloaded_labels": self.downloaded_labels,
            "skipped_images": self.skipped_images,
            "skipped_labels": self.skipped_labels,
            "verified_images": self.verified_images,
            "verified_labels": self.verified_labels,
            "missing_images": self.missing_images,
            "missing_labels": self.missing_labels,
            "failed_images": self.failed_images,
            "failed_labels": self.failed_labels,
            "rows_by_split": dict(self.rows_by_split),
        }

    @property
    def has_errors(self) -> bool:
        return any(
            value > 0
            for value in (
                self.missing_images,
                self.missing_labels,
                self.failed_images,
                self.failed_labels,
            )
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache CVDMS object-detection images and bbox label JSONs locally."
    )
    parser.add_argument(
        "--manifest-dir",
        default="training/data/original/manifests",
        help="Directory containing metadata.json and train/val/test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/data/cached",
        help="Output directory for cached images, labels, manifests, and cache_report.json.",
    )
    parser.add_argument(
        "--bbox-label-prefix",
        required=True,
        help=(
            "S3 prefix containing canonical bbox label JSON files, for example "
            "s3://bucket/canonical/labels/bounding-boxes/"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Dataset splits to cache.",
    )
    parser.add_argument(
        "--profile-name",
        default=None,
        help="Optional AWS profile name. Omit to use the default credential chain.",
    )
    parser.add_argument(
        "--region-name",
        default=None,
        help="Optional AWS region name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and report cache targets without downloading or writing outputs.",
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
        help="Optional maximum number of manifest rows to process across selected splits.",
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
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    bbox_label_prefix = require_nonempty_string(
        args.bbox_label_prefix,
        "--bbox-label-prefix",
    ).rstrip("/")

    s3_client = make_s3_client(
        profile_name=args.profile_name,
        region_name=args.region_name,
    )

    print("")
    print("CVDMS object-detection local cache")
    print("=" * 80)
    print(f"manifest_dir:      {manifest_dir}")
    print(f"output_dir:        {output_dir}")
    print(f"bbox_label_prefix: {bbox_label_prefix}")
    print(f"splits:            {list(args.splits)}")
    print(f"profile_name:      {args.profile_name}")
    print(f"region_name:       {args.region_name}")
    print(f"force:             {args.force}")
    print(f"dry_run:           {args.dry_run}")
    print(f"verify_only:       {args.verify_only}")

    metadata = read_metadata(manifest_dir / "metadata.json")
    validate_metadata(metadata)

    cached_rows = collect_cached_rows(
        manifest_dir=manifest_dir,
        output_dir=output_dir,
        bbox_label_prefix=bbox_label_prefix,
        splits=list(args.splits),
        limit=args.limit,
    )

    if not cached_rows:
        raise ValueError("No manifest rows found for selected splits")

    summary = build_initial_summary(cached_rows)

    print("")
    print("Collected cache targets")
    print("=" * 80)
    print(f"manifest rows:       {len(cached_rows)}")
    print(f"unique image files:  {summary.expected_images}")
    print(f"unique label files:  {summary.expected_labels}")
    for split in SPLITS:
        if split in summary.rows_by_split:
            print(f"{split}: {summary.rows_by_split[split]} row(s)")

    if args.dry_run:
        preview_cached_rows(cached_rows)
        print("")
        print("Dry run complete. No files downloaded or written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    image_items, label_items = collect_unique_download_items(cached_rows)

    cache_uri_items(
        items=image_items,
        force=args.force,
        verify_only=args.verify_only,
        s3_client=s3_client,
        kind="image",
        summary=summary,
        print_every=print_every,
    )

    cache_uri_items(
        items=label_items,
        force=args.force,
        verify_only=args.verify_only,
        s3_client=s3_client,
        kind="label",
        summary=summary,
        print_every=print_every,
    )

    write_cached_manifests(
        cached_rows=cached_rows,
        output_dir=output_dir / "manifests",
    )

    write_cache_report(
        path=output_dir / "cache_report.json",
        metadata=metadata,
        args=args,
        bbox_label_prefix=bbox_label_prefix,
        summary=summary,
    )

    print("")
    print("Cache summary")
    print("=" * 80)
    print(f"expected_images:    {summary.expected_images}")
    print(f"downloaded_images:  {summary.downloaded_images}")
    print(f"skipped_images:     {summary.skipped_images}")
    print(f"verified_images:    {summary.verified_images}")
    print(f"missing_images:     {summary.missing_images}")
    print(f"failed_images:      {summary.failed_images}")
    print(f"expected_labels:    {summary.expected_labels}")
    print(f"downloaded_labels:  {summary.downloaded_labels}")
    print(f"skipped_labels:     {summary.skipped_labels}")
    print(f"verified_labels:    {summary.verified_labels}")
    print(f"missing_labels:     {summary.missing_labels}")
    print(f"failed_labels:      {summary.failed_labels}")

    if summary.has_errors:
        raise RuntimeError(
            "Cache did not complete cleanly. "
            f"missing_images={summary.missing_images} "
            f"missing_labels={summary.missing_labels} "
            f"failed_images={summary.failed_images} "
            f"failed_labels={summary.failed_labels}"
        )

    print("")
    print("Dataset cache completed successfully.")

def read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"metadata.json does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata.json is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise TypeError(f"metadata.json must contain a JSON object, got {type(payload).__name__}")

    return payload

def validate_metadata(metadata: dict[str, Any]) -> None:
    dataset_id = metadata.get("dataset_id")
    version = metadata.get("version")
    label_type = metadata.get("label_type")
    class_to_idx = metadata.get("class_to_idx")

    print("")
    print("Metadata")
    print("=" * 80)
    print(f"dataset_id:  {dataset_id}")
    print(f"version:     {version}")
    print(f"label_type:  {label_type}")
    print(f"class_to_idx:{class_to_idx}")

    if label_type != "object-detection":
        raise ValueError(f"Expected metadata.label_type='object-detection', got {label_type!r}")

    if not isinstance(class_to_idx, dict) or not class_to_idx:
        raise ValueError("metadata.class_to_idx must be a non-empty dictionary")

def collect_cached_rows(
    *,
    manifest_dir: Path,
    output_dir: Path,
    bbox_label_prefix: str,
    splits: list[str],
    limit: int | None,
) -> list[CachedRow]:
    cached_rows: list[CachedRow] = []

    for split in splits:
        manifest_path = manifest_dir / f"{split}.jsonl"
        split_count = 0

        for line_number, row in iter_jsonl_local(manifest_path):
            if limit is not None and len(cached_rows) >= limit:
                return cached_rows

            cached_row = cached_row_from_manifest_row(
                row=row,
                manifest_path=manifest_path,
                line_number=line_number,
                expected_split=split,
                output_dir=output_dir,
                bbox_label_prefix=bbox_label_prefix,
            )
            cached_rows.append(cached_row)
            split_count += 1

        print(f"{split}: read {split_count} manifest row(s)")

    return cached_rows

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

def cached_row_from_manifest_row(
    *,
    row: dict[str, Any],
    manifest_path: Path,
    line_number: int,
    expected_split: str,
    output_dir: Path,
    bbox_label_prefix: str,
) -> CachedRow:
    row_split = require_nonempty_string(
        row.get("split"),
        f"{manifest_path}:{line_number}:split",
    )

    if row_split != expected_split:
        raise ValueError(
            f"Split mismatch in {manifest_path}:{line_number}: "
            f"expected {expected_split!r}, got {row_split!r}"
        )

    label_type = require_nonempty_string(
        row.get("label_type"),
        f"{manifest_path}:{line_number}:label_type",
    )

    if label_type != "object-detection":
        raise ValueError(
            f"Expected label_type='object-detection' in {manifest_path}:{line_number}, "
            f"got {label_type!r}"
        )

    source_ref = source_ref_from_row(
        row=row,
        manifest_path=manifest_path,
        line_number=line_number,
    )
    bbox_annotation_ids = bbox_annotation_ids_from_row(
        row=row,
        manifest_path=manifest_path,
        line_number=line_number,
    )

    bbox_label_uris = [
        build_bbox_label_uri(
            bbox_label_prefix=bbox_label_prefix,
            annotation_id=annotation_id,
        )
        for annotation_id in bbox_annotation_ids
    ]

    return CachedRow(
        split=row_split,
        row=dict(row),
        source_ref=source_ref,
        local_image_path=local_image_cache_path(
            source_ref=source_ref,
            output_dir=output_dir,
            split=row_split,
        ),
        bbox_label_uris=bbox_label_uris,
        local_label_paths=[
            local_label_cache_path(
                annotation_id=annotation_id,
                output_dir=output_dir,
                split=row_split,
            )
            for annotation_id in bbox_annotation_ids
        ],
    )

def source_ref_from_row(
    *,
    row: dict[str, Any],
    manifest_path: Path,
    line_number: int,
) -> str:
    value = row.get("source_ref")
    if value is None:
        value = row.get("source-ref")

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Manifest row missing non-empty source_ref/source-ref at "
            f"{manifest_path}:{line_number}"
        )

    source_ref = value.strip()

    if not source_ref.startswith("s3://"):
        raise ValueError(
            f"Expected source_ref to be an S3 URI at {manifest_path}:{line_number}, "
            f"got {source_ref!r}"
        )

    return source_ref

def bbox_annotation_ids_from_row(
    *,
    row: dict[str, Any],
    manifest_path: Path,
    line_number: int,
) -> list[str]:
    value = row.get("bbox_annotation_ids")

    if not isinstance(value, list):
        raise TypeError(
            f"bbox_annotation_ids must be a list at {manifest_path}:{line_number}, "
            f"got {type(value).__name__}"
        )

    if not value:
        raise ValueError(f"bbox_annotation_ids cannot be empty at {manifest_path}:{line_number}")

    annotation_ids: list[str] = []

    for idx, item in enumerate(value):
        annotation_id = require_nonempty_string(
            item,
            f"{manifest_path}:{line_number}:bbox_annotation_ids[{idx}]",
        )
        annotation_ids.append(annotation_id)

    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError(
            f"Duplicate bbox_annotation_ids in one row at {manifest_path}:{line_number}: "
            f"{annotation_ids}"
        )

    return annotation_ids

def build_bbox_label_uri(
    *,
    bbox_label_prefix: str,
    annotation_id: str,
) -> str:
    clean_prefix = bbox_label_prefix.rstrip("/")
    clean_id = annotation_id.strip()

    if clean_id.endswith(".json"):
        filename = clean_id
    else:
        filename = f"{clean_id}.json"

    return f"{clean_prefix}/{filename}"

def local_image_cache_path(
    *,
    source_ref: str,
    output_dir: Path,
    split: str,
) -> Path:
    parsed = parse_s3_uri(source_ref)
    filename = Path(parsed.key).name

    if not filename:
        raise ValueError(f"Could not derive image filename from source_ref: {source_ref}")

    return output_dir / "images" / split / filename

def local_label_cache_path(
    *,
    annotation_id: str,
    output_dir: Path,
    split: str,
) -> Path:
    clean_id = annotation_id.strip()

    if clean_id.endswith(".json"):
        filename = clean_id
    else:
        filename = f"{clean_id}.json"

    return output_dir / "labels" / split / filename

def build_initial_summary(cached_rows: list[CachedRow]) -> CacheSummary:
    image_paths = {row.local_image_path for row in cached_rows}
    label_paths = {
        label_path
        for row in cached_rows
        for label_path in row.local_label_paths
    }

    summary = CacheSummary(
        expected_images=len(image_paths),
        expected_labels=len(label_paths),
    )

    for row in cached_rows:
        summary.rows_by_split[row.split] = summary.rows_by_split.get(row.split, 0) + 1

    return summary

def collect_unique_download_items(
    cached_rows: list[CachedRow],
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    images_by_path: dict[Path, str] = {}
    labels_by_path: dict[Path, str] = {}

    for row in cached_rows:
        images_by_path.setdefault(row.local_image_path, row.source_ref)

        for label_uri, label_path in zip(row.bbox_label_uris, row.local_label_paths):
            labels_by_path.setdefault(label_path, label_uri)

    image_items = [
        (uri, path)
        for path, uri in sorted(images_by_path.items(), key=lambda item: str(item[0]))
    ]
    label_items = [
        (uri, path)
        for path, uri in sorted(labels_by_path.items(), key=lambda item: str(item[0]))
    ]

    return image_items, label_items

def cache_uri_items(
    *,
    items: list[tuple[str, Path]],
    force: bool,
    verify_only: bool,
    s3_client,
    kind: str,
    summary: CacheSummary,
    print_every: int,
) -> None:
    print("")
    print(f"Caching {kind}s")
    print("=" * 80)

    for index, (uri, local_path) in enumerate(items, start=1):
        parsed = parse_s3_uri(uri)

        try:
            if local_path.exists() and not force:
                increment_summary(summary, kind, "skipped")
                increment_summary(summary, kind, "verified")
            elif verify_only:
                increment_summary(summary, kind, "missing")
                print(f"[missing] {uri} -> {local_path}")
            else:
                download_s3_object(
                    bucket=parsed.bucket,
                    key=parsed.key,
                    destination=local_path,
                    s3_client=s3_client,
                )
                increment_summary(summary, kind, "downloaded")
                increment_summary(summary, kind, "verified")

        except Exception as exc:
            increment_summary(summary, kind, "failed")
            print(f"[failed] {uri} -> {local_path}")
            print(f"         {type(exc).__name__}: {exc}")

        if index == 1 or index % print_every == 0 or index == len(items):
            print_cache_progress(
                index=index,
                total=len(items),
                kind=kind,
                summary=summary,
            )

def increment_summary(summary: CacheSummary, kind: str, field: str) -> None:
    if kind == "image":
        attr = f"{field}_images"
    elif kind == "label":
        attr = f"{field}_labels"
    else:
        raise ValueError(f"Unsupported kind={kind!r}")

    setattr(summary, attr, getattr(summary, attr) + 1)

def print_cache_progress(
    *,
    index: int,
    total: int,
    kind: str,
    summary: CacheSummary,
) -> None:
    if kind == "image":
        downloaded = summary.downloaded_images
        skipped = summary.skipped_images
        missing = summary.missing_images
        failed = summary.failed_images
    elif kind == "label":
        downloaded = summary.downloaded_labels
        skipped = summary.skipped_labels
        missing = summary.missing_labels
        failed = summary.failed_labels
    else:
        raise ValueError(f"Unsupported kind={kind!r}")

    print(
        " | ".join(
            [
                f"{kind}s_processed={index}/{total}",
                f"downloaded={downloaded}",
                f"skipped={skipped}",
                f"missing={missing}",
                f"failed={failed}",
            ]
        )
    )

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

def write_cached_manifests(
    *,
    cached_rows: list[CachedRow],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[CachedRow]] = {}

    for row in cached_rows:
        rows_by_split.setdefault(row.split, []).append(row)

    for split in SPLITS:
        rows = rows_by_split.get(split)
        if not rows:
            continue

        destination = output_dir / f"{split}.jsonl"

        with destination.open("w", encoding="utf-8") as file:
            for cached_row in rows:
                file.write(json.dumps(cached_row.to_manifest_row(), sort_keys=True))
                file.write("\n")

        print(f"Wrote cached manifest: {destination}")

def write_cache_report(
    *,
    path: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    bbox_label_prefix: str,
    summary: CacheSummary,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_id": metadata.get("dataset_id"),
        "version": metadata.get("version"),
        "label_type": metadata.get("label_type"),
        "class_to_idx": metadata.get("class_to_idx"),
        "split_counts": metadata.get("split_counts"),
        "manifest_dir": str(Path(args.manifest_dir)),
        "output_dir": str(Path(args.output_dir)),
        "bbox_label_prefix": bbox_label_prefix,
        "splits": list(args.splits),
        "force": bool(args.force),
        "verify_only": bool(args.verify_only),
        "limit": args.limit,
        "summary": summary.to_dict(),
    }

    path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Wrote cache report: {path}")

def preview_cached_rows(
    cached_rows: list[CachedRow],
    max_items: int = 5,
) -> None:
    print("")
    print("Preview")
    print("=" * 80)

    for cached_row in cached_rows[:max_items]:
        print(f"split: {cached_row.split}")
        print(f"image: {cached_row.source_ref}")
        print(f"  -> {cached_row.local_image_path}")

        for label_uri, label_path in zip(cached_row.bbox_label_uris, cached_row.local_label_paths):
            print(f"label: {label_uri}")
            print(f"  -> {label_path}")

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