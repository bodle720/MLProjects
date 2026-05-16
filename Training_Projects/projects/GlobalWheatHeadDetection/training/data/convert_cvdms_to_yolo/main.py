# To run from the project root:
# python -m training.data.convert_cvdms_to_yolo.main --overwrite

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = PROJECT_ROOT / "training"
DATA_ROOT = TRAINING_ROOT / "data"

ORIGINAL_MANIFEST_DIR = DATA_ROOT / "original" / "manifests"
CACHED_DATA_ROOT = DATA_ROOT / "cached"
CACHED_MANIFEST_DIR = CACHED_DATA_ROOT / "manifests"
CACHED_IMAGES_DIR = CACHED_DATA_ROOT / "images"
CACHED_LABELS_DIR = CACHED_DATA_ROOT / "labels"

CONVERTED_YOLO_ROOT = DATA_ROOT / "yolo"

try:
    from .conversion_helpers.conversion_report import (
        summarize_split_stats,
        write_conversion_report,
    )
    from .conversion_helpers.cvdms_labels import validate_cached_labels_exist_for_rows
    from .conversion_helpers.cvdms_manifest import (
        flatten_manifest_rows,
        load_manifest_bundle,
        summarize_manifest_bundle,
    )
    from .conversion_helpers.splits import (
        build_preserved_yolo_split_rows,
        summarize_split_rows,
    )
    from .conversion_helpers.yolo_writer import (
        split_stats_to_dict,
        validate_cached_images_exist_for_rows,
        write_yolo_dataset,
    )
except ImportError:
    from conversion_helpers.conversion_report import (
        summarize_split_stats,
        write_conversion_report,
    )
    from conversion_helpers.cvdms_labels import validate_cached_labels_exist_for_rows
    from conversion_helpers.cvdms_manifest import (
        flatten_manifest_rows,
        load_manifest_bundle,
        summarize_manifest_bundle,
    )
    from conversion_helpers.splits import (
        build_preserved_yolo_split_rows,
        summarize_split_rows,
    )
    from conversion_helpers.yolo_writer import (
        split_stats_to_dict,
        validate_cached_images_exist_for_rows,
        write_yolo_dataset,
    )


def resolve_project_path(path_value: str | None) -> Path | None:
    if path_value is None:
        return None

    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def build_default_output_dir(dataset_id: str, version: int) -> Path:
    return CONVERTED_YOLO_ROOT / f"{dataset_id}-v{version}"


def validate_cached_inputs(
    all_rows: list[dict[str, Any]],
    project_root: Path,
    images_cache_dir: Path | None,
    labels_cache_dir: Path | None,
    prefer_local_paths: bool,
) -> dict[str, Any]:
    image_validation = validate_cached_images_exist_for_rows(
        rows=all_rows,
        project_root=project_root,
        images_cache_dir=images_cache_dir,
        prefer_local_paths=prefer_local_paths,
    )

    if image_validation["missing_cached_images"] > 0:
        examples = image_validation["missing_cached_image_examples"]
        raise FileNotFoundError(
            "Missing cached CVDMS image files. "
            f"Missing count: {image_validation['missing_cached_images']}. "
            f"Examples: {examples}"
        )

    label_validation = validate_cached_labels_exist_for_rows(
        rows=all_rows,
        project_root=project_root,
        labels_cache_dir=labels_cache_dir,
        prefer_local_paths=prefer_local_paths,
    )

    if label_validation["missing_label_files"] > 0:
        examples = label_validation["missing_label_file_examples"]
        raise FileNotFoundError(
            "Missing cached CVDMS label JSON files. "
            f"Missing count: {label_validation['missing_label_files']}. "
            f"Examples: {examples}"
        )

    return {
        "project_root": str(project_root),
        "images_cache_dir": str(images_cache_dir) if images_cache_dir else None,
        "labels_cache_dir": str(labels_cache_dir) if labels_cache_dir else None,
        "prefer_local_paths": prefer_local_paths,
        "image_validation": image_validation,
        "label_validation": label_validation,
    }


def build_config_summary(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "project_root": str(PROJECT_ROOT),
        "training_root": str(TRAINING_ROOT),
        "data_root": str(DATA_ROOT),
        "metadata_path": str(resolve_project_path(args.metadata_path)),
        "manifest_dir": str(resolve_project_path(args.manifest_dir)),
        "images_cache_dir": str(resolve_project_path(args.images_cache_dir)) if args.images_cache_dir else None,
        "labels_cache_dir": str(resolve_project_path(args.labels_cache_dir)) if args.labels_cache_dir else None,
        "output_dir": str(output_dir),
        "overwrite": args.overwrite,
        "copy_mode": args.copy_mode,
        "preserve_cvdms_splits": True,
        "prefer_local_paths": not args.no_prefer_local_paths,
        "require_local_paths": not args.no_require_local_paths,
        "clip_to_image": not args.no_clip_to_image,
        "keep_images_with_no_boxes": args.keep_images_with_no_boxes,
        "max_failure_examples": args.max_failure_examples,
    }


def build_warnings(split_stats: dict[str, Any]) -> list[str]:
    warnings = []

    for split_name, stats in split_stats.items():
        skipped_no_boxes = stats.get("images_skipped_no_kept_boxes", 0)
        missing_images = stats.get("images_skipped_missing_cached_image", 0)
        dropped_class_boxes = stats.get("boxes_dropped_class_not_selected", 0)
        dropped_invalid_boxes = stats.get("boxes_dropped_invalid", 0)
        empty_label_files = stats.get("empty_label_files_written", 0)
        failures = len(stats.get("failures", []))

        if skipped_no_boxes:
            warnings.append(
                f"{split_name}: skipped {skipped_no_boxes} images because no selected-class boxes remained."
            )

        if missing_images:
            warnings.append(
                f"{split_name}: skipped {missing_images} images because cached image files were missing."
            )

        if dropped_class_boxes:
            warnings.append(
                f"{split_name}: dropped {dropped_class_boxes} boxes outside the selected class set."
            )

        if dropped_invalid_boxes:
            warnings.append(
                f"{split_name}: dropped {dropped_invalid_boxes} invalid boxes."
            )

        if empty_label_files:
            warnings.append(
                f"{split_name}: wrote {empty_label_files} empty YOLO label files."
            )

        if failures:
            warnings.append(
                f"{split_name}: recorded {failures} failure examples in the conversion report."
            )

    return warnings


def log_summary(title: str, data: dict[str, Any]) -> None:
    logging.info("")
    logging.info("=== %s ===", title)

    for key, value in data.items():
        logging.info("%s: %s", key, value)


def run_conversion(args: argparse.Namespace) -> int:
    metadata_path = resolve_project_path(args.metadata_path)
    manifest_dir = resolve_project_path(args.manifest_dir)
    images_cache_dir = resolve_project_path(args.images_cache_dir) if args.images_cache_dir else None
    labels_cache_dir = resolve_project_path(args.labels_cache_dir) if args.labels_cache_dir else None

    if metadata_path is None or manifest_dir is None:
        raise ValueError("Metadata path and manifest directory are required.")

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Training root: %s", TRAINING_ROOT)
    logging.info("Data root: %s", DATA_ROOT)
    logging.info("Metadata path: %s", metadata_path)
    logging.info("Cached manifest directory: %s", manifest_dir)
    logging.info("Converted YOLO root: %s", CONVERTED_YOLO_ROOT)

    bundle = load_manifest_bundle(
        metadata_path=metadata_path,
        manifest_dir=manifest_dir,
        require_local_paths=not args.no_require_local_paths,
        allow_missing_test=args.allow_missing_test,
    )
    metadata_summary = summarize_manifest_bundle(bundle)

    output_dir = resolve_project_path(args.output_dir)
    if output_dir is None:
        output_dir = build_default_output_dir(
            dataset_id=bundle.metadata.dataset_id,
            version=bundle.metadata.version,
        )

    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = f"{bundle.metadata.dataset_id}-v{bundle.metadata.version}"

    log_summary("CVDMS metadata and manifests", metadata_summary)

    all_manifest_rows = flatten_manifest_rows(bundle)
    cache_summary = validate_cached_inputs(
        all_rows=all_manifest_rows,
        project_root=PROJECT_ROOT,
        images_cache_dir=images_cache_dir,
        labels_cache_dir=labels_cache_dir,
        prefer_local_paths=not args.no_prefer_local_paths,
    )
    log_summary("Cache validation", cache_summary)

    split_rows = build_preserved_yolo_split_rows(
        cvdms_train_rows=bundle.train_rows,
        cvdms_val_rows=bundle.val_rows,
        cvdms_test_rows=bundle.test_rows,
    )
    split_summary = summarize_split_rows(split_rows)
    log_summary("YOLO split summary", split_summary)

    logging.info("")
    logging.info("Writing YOLO dataset to: %s", output_dir)

    write_result = write_yolo_dataset(
        split_rows=split_rows,
        output_dir=output_dir,
        class_to_id=bundle.metadata.class_to_id,
        dataset_name=dataset_name,
        project_root=PROJECT_ROOT,
        images_cache_dir=images_cache_dir,
        labels_cache_dir=labels_cache_dir,
        prefer_local_paths=not args.no_prefer_local_paths,
        overwrite=args.overwrite,
        copy_mode=args.copy_mode,
        keep_images_with_no_boxes=args.keep_images_with_no_boxes,
        clip_to_image=not args.no_clip_to_image,
        max_failure_examples=args.max_failure_examples,
        report_extra={
            "metadata_summary": metadata_summary,
            "split_summary": split_summary,
            "cache_summary": cache_summary,
        },
    )

    split_stats = {
        split_name: split_stats_to_dict(stats)
        for split_name, stats in write_result.split_stats.items()
    }
    compact_stats = summarize_split_stats(split_stats)
    log_summary("YOLO write summary", compact_stats)

    detailed_report_path = output_dir / "conversion_report_detailed.json"
    write_conversion_report(
        report_path=detailed_report_path,
        dataset_name=dataset_name,
        output_dir=output_dir,
        dataset_yaml_path=write_result.dataset_yaml_path,
        metadata_summary=metadata_summary,
        split_summary=split_summary,
        split_stats=split_stats,
        config_summary=build_config_summary(args, output_dir),
        warnings=build_warnings(split_stats),
        extra={
            "cache_summary": cache_summary,
            "compact_write_summary": compact_stats,
            "basic_report_path": str(write_result.report_path),
        },
    )

    logging.info("")
    logging.info("Dataset YAML:    %s", write_result.dataset_yaml_path)
    logging.info("Basic report:    %s", write_result.report_path)
    logging.info("Detailed report: %s", detailed_report_path)

    if args.fail_on_recorded_failures:
        num_failures = compact_stats.get("failures", 0)
        missing_images = compact_stats.get("images_skipped_missing_cached_image", 0)

        if num_failures > 0 or missing_images > 0:
            raise RuntimeError(
                "Conversion completed with recorded failures. "
                f"failures={num_failures}, missing_images={missing_images}"
            )

    logging.info("")
    logging.info("Conversion completed successfully.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert cached CVDMS Global Wheat Head Detection object-detection "
            "manifests/images/labels into Ultralytics YOLO format."
        )
    )

    parser.add_argument(
        "--metadata-path",
        default="training/data/original/manifests/metadata.json",
        help="Path to the original CVDMS metadata.json.",
    )
    parser.add_argument(
        "--manifest-dir",
        default="training/data/cached/manifests",
        help=(
            "Directory containing cached train.jsonl, val.jsonl, and test.jsonl. "
            "These rows should include local_image_path and local_label_paths."
        ),
    )
    parser.add_argument(
        "--images-cache-dir",
        default="training/data/cached/images",
        help=(
            "Fallback cached image directory. Usually not needed when cached manifests "
            "include local_image_path."
        ),
    )
    parser.add_argument(
        "--labels-cache-dir",
        default="training/data/cached/labels",
        help=(
            "Fallback cached label directory. Usually not needed when cached manifests "
            "include local_label_paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "YOLO output directory. Defaults to "
            "training/converted_cvdms_to_yolo/<dataset_id>-v<version>."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional name to store in reports. Defaults to <dataset_id>-v<version>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help=(
            "How to place cached images into the YOLO dataset. "
            "hardlink saves disk space when supported, otherwise falls back to copy."
        ),
    )
    parser.add_argument(
        "--no-prefer-local-paths",
        action="store_true",
        help=(
            "Do not prefer local_image_path/local_label_paths from cached manifest rows. "
            "Use cache-directory fallback resolution instead."
        ),
    )
    parser.add_argument(
        "--no-require-local-paths",
        action="store_true",
        help=(
            "Allow manifest rows without local_image_path/local_label_paths. "
            "Useful only for fallback/debug conversions."
        ),
    )
    parser.add_argument(
        "--allow-missing-test",
        action="store_true",
        help="Allow test.jsonl to be missing or absent.",
    )
    parser.add_argument(
        "--no-clip-to-image",
        action="store_true",
        help="Disable clipping boxes to image boundaries before YOLO conversion.",
    )
    parser.add_argument(
        "--keep-images-with-no-boxes",
        action="store_true",
        help=(
            "Write images and empty YOLO label files even when no selected-class boxes remain. "
            "Default is to skip those images."
        ),
    )
    parser.add_argument(
        "--max-failure-examples",
        type=int,
        default=100,
        help="Maximum failure examples to keep per split in reports.",
    )
    parser.add_argument(
        "--fail-on-recorded-failures",
        action="store_true",
        help="Exit with an error if conversion records failures or missing cached images.",
    )

    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()

    try:
        return run_conversion(args)
    except Exception as exc:
        logging.exception("CVDMS-to-YOLO conversion failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())