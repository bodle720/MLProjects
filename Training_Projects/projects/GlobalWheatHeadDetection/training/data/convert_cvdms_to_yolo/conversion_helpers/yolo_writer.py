import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .bbox_utils import (
    cvdms_annotation_to_box,
    cvdms_box_to_yolo_box,
    format_yolo_label_row,
)
from .conversion_report import to_jsonable
from .cvdms_labels import load_annotations_for_manifest_row
from .splits import YoloSplitRows


VALID_COPY_MODES = {"copy", "hardlink"}


@dataclass
class SplitWriteStats:
    rows_seen: int = 0
    images_written: int = 0
    images_copied: int = 0
    images_hardlinked: int = 0
    images_skipped_no_kept_boxes: int = 0
    images_skipped_missing_cached_image: int = 0
    labels_written: int = 0
    empty_label_files_written: int = 0
    boxes_seen: int = 0
    boxes_written: int = 0
    boxes_dropped_class_not_selected: int = 0
    boxes_dropped_invalid: int = 0
    dropped_class_counts: dict[str, int] = field(default_factory=dict)
    written_class_counts: dict[str, int] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class YoloDatasetWriteResult:
    output_dir: Path
    dataset_yaml_path: Path
    report_path: Path
    split_stats: dict[str, SplitWriteStats]


def resolve_project_path(path_value: str, project_root: Path | None = None) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    if project_root is None:
        return path

    return project_root / path


def require_manifest_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)

    if not isinstance(value, str) or not value.strip():
        image_id = row.get("image_id", "<unknown>")
        raise ValueError(f"Manifest row for image {image_id!r} missing valid {key!r}")

    return value.strip()


def get_image_extension_from_source_ref(source_ref: str) -> str:
    suffix = Path(source_ref).suffix.lower()

    if suffix:
        return suffix

    return ".jpg"


def resolve_cached_image_path_from_row(
    manifest_row: dict[str, Any],
    project_root: Path | None = None,
) -> Path:
    local_image_path = manifest_row.get("local_image_path")

    if not isinstance(local_image_path, str) or not local_image_path.strip():
        image_id = manifest_row.get("image_id", "<unknown>")
        raise ValueError(f"Manifest row for image {image_id!r} has no local_image_path")

    return resolve_project_path(
        path_value=local_image_path.strip(),
        project_root=project_root,
    )


def resolve_cached_image_path_from_cache_dir(
    images_cache_dir: Path,
    manifest_row: dict[str, Any],
) -> Path:
    image_id = require_manifest_string(manifest_row, "image_id")
    source_ref = require_manifest_string(manifest_row, "source_ref")
    split = manifest_row.get("split")
    extension = get_image_extension_from_source_ref(source_ref)

    if isinstance(split, str) and split.strip():
        split_candidate = images_cache_dir / split.strip() / f"{image_id}{extension}"
        if split_candidate.exists():
            return split_candidate

    return images_cache_dir / f"{image_id}{extension}"


def resolve_cached_image_path(
    manifest_row: dict[str, Any],
    project_root: Path | None = None,
    images_cache_dir: Path | None = None,
    prefer_local_paths: bool = True,
) -> Path:
    if prefer_local_paths and "local_image_path" in manifest_row:
        return resolve_cached_image_path_from_row(
            manifest_row=manifest_row,
            project_root=project_root,
        )

    if images_cache_dir is None:
        raise ValueError(
            "images_cache_dir is required when manifest row does not provide local_image_path"
        )

    return resolve_cached_image_path_from_cache_dir(
        images_cache_dir=images_cache_dir,
        manifest_row=manifest_row,
    )


def get_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size

    if width <= 0 or height <= 0:
        raise ValueError(f"Image has invalid dimensions {width}x{height}: {image_path}")

    return width, height


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"YOLO output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )

        shutil.rmtree(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def place_image_in_yolo_split(
    source_image_path: Path,
    output_dir: Path,
    split: str,
    image_id: str,
    copy_mode: str = "copy",
) -> tuple[Path, str]:
    if copy_mode not in VALID_COPY_MODES:
        raise ValueError(f"copy_mode must be one of {sorted(VALID_COPY_MODES)}, got {copy_mode!r}")

    extension = source_image_path.suffix.lower() or ".jpg"
    destination_path = output_dir / "images" / split / f"{image_id}{extension}"
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        destination_path.unlink()

    if copy_mode == "hardlink":
        try:
            os.link(source_image_path, destination_path)
            return destination_path, "hardlink"
        except OSError:
            shutil.copy2(source_image_path, destination_path)
            return destination_path, "copy"

    shutil.copy2(source_image_path, destination_path)
    return destination_path, "copy"


def write_label_file(
    output_dir: Path,
    split: str,
    image_id: str,
    label_rows: list[str],
) -> Path:
    label_path = output_dir / "labels" / split / f"{image_id}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)

    text = "\n".join(label_rows)
    if text:
        text += "\n"

    label_path.write_text(text, encoding="utf-8")
    return label_path


def record_failure(
    stats: SplitWriteStats,
    failure: dict[str, Any],
    max_failure_examples: int,
) -> None:
    if len(stats.failures) < max_failure_examples:
        stats.failures.append(failure)


def convert_annotations_to_yolo_rows(
    annotations: list[dict[str, Any]],
    class_to_id: dict[str, int],
    image_width: int,
    image_height: int,
    stats: SplitWriteStats,
    image_id: str,
    clip_to_image: bool,
    max_failure_examples: int,
) -> list[str]:
    yolo_rows = []

    for annotation_index, annotation in enumerate(annotations):
        stats.boxes_seen += 1

        class_name = annotation.get("class_name")
        if not isinstance(class_name, str) or not class_name.strip():
            stats.boxes_dropped_invalid += 1
            record_failure(
                stats=stats,
                failure={
                    "image_id": image_id,
                    "annotation_index": annotation_index,
                    "reason": "invalid_class_name",
                    "annotation": annotation,
                },
                max_failure_examples=max_failure_examples,
            )
            continue

        class_name = class_name.strip()

        if class_name not in class_to_id:
            stats.boxes_dropped_class_not_selected += 1
            stats.dropped_class_counts[class_name] = stats.dropped_class_counts.get(class_name, 0) + 1
            continue

        try:
            cvdms_box = cvdms_annotation_to_box(annotation)
            yolo_box = cvdms_box_to_yolo_box(
                box=cvdms_box,
                image_width=image_width,
                image_height=image_height,
                clip_to_image=clip_to_image,
            )

            if yolo_box is None:
                stats.boxes_dropped_invalid += 1
                record_failure(
                    stats=stats,
                    failure={
                        "image_id": image_id,
                        "annotation_index": annotation_index,
                        "class_name": class_name,
                        "reason": "bbox_invalid_after_clipping",
                        "annotation": annotation,
                    },
                    max_failure_examples=max_failure_examples,
                )
                continue

            class_id = class_to_id[class_name]
            yolo_row = format_yolo_label_row(class_id, yolo_box)
            yolo_rows.append(yolo_row)

            stats.boxes_written += 1
            stats.written_class_counts[class_name] = stats.written_class_counts.get(class_name, 0) + 1

        except ValueError as exc:
            stats.boxes_dropped_invalid += 1
            record_failure(
                stats=stats,
                failure={
                    "image_id": image_id,
                    "annotation_index": annotation_index,
                    "class_name": class_name,
                    "reason": "invalid_bbox",
                    "error": str(exc),
                    "annotation": annotation,
                },
                max_failure_examples=max_failure_examples,
            )

    return yolo_rows


def write_split(
    split_name: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    class_to_id: dict[str, int],
    project_root: Path | None = None,
    images_cache_dir: Path | None = None,
    labels_cache_dir: Path | None = None,
    prefer_local_paths: bool = True,
    copy_mode: str = "copy",
    keep_images_with_no_boxes: bool = False,
    clip_to_image: bool = True,
    max_failure_examples: int = 100,
) -> SplitWriteStats:
    stats = SplitWriteStats(rows_seen=len(rows))

    for row in rows:
        image_id = require_manifest_string(row, "image_id")

        try:
            cached_image_path = resolve_cached_image_path(
                manifest_row=row,
                project_root=project_root,
                images_cache_dir=images_cache_dir,
                prefer_local_paths=prefer_local_paths,
            )
        except Exception as exc:
            stats.images_skipped_missing_cached_image += 1
            record_failure(
                stats=stats,
                failure={
                    "image_id": image_id,
                    "reason": "image_path_resolution_failed",
                    "error": str(exc),
                },
                max_failure_examples=max_failure_examples,
            )
            continue

        if not cached_image_path.exists():
            stats.images_skipped_missing_cached_image += 1
            record_failure(
                stats=stats,
                failure={
                    "image_id": image_id,
                    "reason": "missing_cached_image",
                    "path": str(cached_image_path),
                },
                max_failure_examples=max_failure_examples,
            )
            continue

        try:
            image_width, image_height = get_image_size(cached_image_path)
            annotations = load_annotations_for_manifest_row(
                manifest_row=row,
                project_root=project_root,
                labels_cache_dir=labels_cache_dir,
                prefer_local_paths=prefer_local_paths,
            )
            yolo_label_rows = convert_annotations_to_yolo_rows(
                annotations=annotations,
                class_to_id=class_to_id,
                image_width=image_width,
                image_height=image_height,
                stats=stats,
                image_id=image_id,
                clip_to_image=clip_to_image,
                max_failure_examples=max_failure_examples,
            )

            if not yolo_label_rows and not keep_images_with_no_boxes:
                stats.images_skipped_no_kept_boxes += 1
                continue

            _, placement_mode = place_image_in_yolo_split(
                source_image_path=cached_image_path,
                output_dir=output_dir,
                split=split_name,
                image_id=image_id,
                copy_mode=copy_mode,
            )
            write_label_file(
                output_dir=output_dir,
                split=split_name,
                image_id=image_id,
                label_rows=yolo_label_rows,
            )

            stats.images_written += 1
            stats.labels_written += 1

            if placement_mode == "hardlink":
                stats.images_hardlinked += 1
            else:
                stats.images_copied += 1

            if not yolo_label_rows:
                stats.empty_label_files_written += 1

        except Exception as exc:
            record_failure(
                stats=stats,
                failure={
                    "image_id": image_id,
                    "reason": "row_conversion_failed",
                    "error": str(exc),
                },
                max_failure_examples=max_failure_examples,
            )

    sort_stats(stats)
    return stats


def sort_stats(stats: SplitWriteStats) -> None:
    stats.dropped_class_counts = dict(sorted(stats.dropped_class_counts.items()))
    stats.written_class_counts = dict(sorted(stats.written_class_counts.items()))


def write_dataset_yaml(
    output_dir: Path,
    class_to_id: dict[str, int],
    dataset_name: str,
) -> Path:
    names = {
        class_id: class_name
        for class_name, class_id in sorted(class_to_id.items(), key=lambda item: item[1])
    }

    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }

    yaml_path = output_dir / "dataset.yaml"

    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(f"# YOLO dataset exported from CVDMS dataset: {dataset_name}\n")
        yaml.safe_dump(data, f, sort_keys=False)

    return yaml_path


def split_stats_to_dict(stats: SplitWriteStats) -> dict[str, Any]:
    return {
        "rows_seen": stats.rows_seen,
        "images_written": stats.images_written,
        "images_copied": stats.images_copied,
        "images_hardlinked": stats.images_hardlinked,
        "images_skipped_no_kept_boxes": stats.images_skipped_no_kept_boxes,
        "images_skipped_missing_cached_image": stats.images_skipped_missing_cached_image,
        "labels_written": stats.labels_written,
        "empty_label_files_written": stats.empty_label_files_written,
        "boxes_seen": stats.boxes_seen,
        "boxes_written": stats.boxes_written,
        "boxes_dropped_class_not_selected": stats.boxes_dropped_class_not_selected,
        "boxes_dropped_invalid": stats.boxes_dropped_invalid,
        "dropped_class_counts": stats.dropped_class_counts,
        "written_class_counts": stats.written_class_counts,
        "failures": stats.failures,
    }


def write_basic_conversion_report(
    output_dir: Path,
    dataset_name: str,
    split_stats: dict[str, SplitWriteStats],
    extra: dict[str, Any] | None = None,
) -> Path:
    report = {
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "splits": {
            split_name: split_stats_to_dict(stats)
            for split_name, stats in split_stats.items()
        },
    }

    if extra:
        report["extra"] = extra

    report_path = output_dir / "conversion_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, indent=2, sort_keys=True)

    return report_path


def validate_cached_images_exist_for_rows(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
    images_cache_dir: Path | None = None,
    prefer_local_paths: bool = True,
    max_missing_examples: int = 20,
) -> dict[str, Any]:
    missing_paths = []
    rows_with_local_image_path = 0
    rows_using_cache_dir_fallback = 0

    for row in rows:
        if prefer_local_paths and "local_image_path" in row:
            rows_with_local_image_path += 1
        else:
            rows_using_cache_dir_fallback += 1

        image_path = resolve_cached_image_path(
            manifest_row=row,
            project_root=project_root,
            images_cache_dir=images_cache_dir,
            prefer_local_paths=prefer_local_paths,
        )

        if not image_path.exists():
            missing_paths.append(str(image_path))

    return {
        "total_image_refs": len(rows),
        "missing_cached_images": len(missing_paths),
        "missing_cached_image_examples": missing_paths[:max_missing_examples],
        "rows_with_local_image_path": rows_with_local_image_path,
        "rows_using_cache_dir_fallback": rows_using_cache_dir_fallback,
    }


def write_yolo_dataset(
    split_rows: YoloSplitRows,
    output_dir: Path,
    class_to_id: dict[str, int],
    dataset_name: str,
    project_root: Path | None = None,
    images_cache_dir: Path | None = None,
    labels_cache_dir: Path | None = None,
    prefer_local_paths: bool = True,
    overwrite: bool = False,
    copy_mode: str = "copy",
    keep_images_with_no_boxes: bool = False,
    clip_to_image: bool = True,
    max_failure_examples: int = 100,
    report_extra: dict[str, Any] | None = None,
) -> YoloDatasetWriteResult:
    """
    Write a cached CVDMS object-detection dataset into Ultralytics YOLO format.

    Project 3 normal input:
      - split_rows.train / split_rows.val / split_rows.test
      - each row has local_image_path
      - each row has local_label_paths

    Output:
      - images/train, images/val, images/test
      - labels/train, labels/val, labels/test
      - dataset.yaml
      - conversion_report.json
    """
    reset_output_dir(output_dir, overwrite=overwrite)

    split_stats = {
        "train": write_split(
            split_name="train",
            rows=split_rows.train,
            output_dir=output_dir,
            class_to_id=class_to_id,
            project_root=project_root,
            images_cache_dir=images_cache_dir,
            labels_cache_dir=labels_cache_dir,
            prefer_local_paths=prefer_local_paths,
            copy_mode=copy_mode,
            keep_images_with_no_boxes=keep_images_with_no_boxes,
            clip_to_image=clip_to_image,
            max_failure_examples=max_failure_examples,
        ),
        "val": write_split(
            split_name="val",
            rows=split_rows.val,
            output_dir=output_dir,
            class_to_id=class_to_id,
            project_root=project_root,
            images_cache_dir=images_cache_dir,
            labels_cache_dir=labels_cache_dir,
            prefer_local_paths=prefer_local_paths,
            copy_mode=copy_mode,
            keep_images_with_no_boxes=keep_images_with_no_boxes,
            clip_to_image=clip_to_image,
            max_failure_examples=max_failure_examples,
        ),
        "test": write_split(
            split_name="test",
            rows=split_rows.test,
            output_dir=output_dir,
            class_to_id=class_to_id,
            project_root=project_root,
            images_cache_dir=images_cache_dir,
            labels_cache_dir=labels_cache_dir,
            prefer_local_paths=prefer_local_paths,
            copy_mode=copy_mode,
            keep_images_with_no_boxes=keep_images_with_no_boxes,
            clip_to_image=clip_to_image,
            max_failure_examples=max_failure_examples,
        ),
    }

    dataset_yaml_path = write_dataset_yaml(
        output_dir=output_dir,
        class_to_id=class_to_id,
        dataset_name=dataset_name,
    )
    report_path = write_basic_conversion_report(
        output_dir=output_dir,
        dataset_name=dataset_name,
        split_stats=split_stats,
        extra=report_extra,
    )

    return YoloDatasetWriteResult(
        output_dir=output_dir,
        dataset_yaml_path=dataset_yaml_path,
        report_path=report_path,
        split_stats=split_stats,
    )