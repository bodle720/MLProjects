import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CvdmsLabelFile:
    bbox_annotation_id: str | None
    path: Path
    annotations: list[dict[str, Any]]


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def resolve_project_path(path_value: str, project_root: Path | None = None) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    if project_root is None:
        return path

    return project_root / path


def resolve_cached_label_path(
    labels_cache_dir: Path,
    bbox_annotation_id: str,
    split: str | None = None,
) -> Path:
    if not isinstance(bbox_annotation_id, str) or not bbox_annotation_id.strip():
        raise ValueError(f"Invalid bbox_annotation_id: {bbox_annotation_id!r}")

    clean_id = bbox_annotation_id.strip()

    if split:
        return labels_cache_dir / split / f"{clean_id}.json"

    return labels_cache_dir / f"{clean_id}.json"


def read_label_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"CVDMS label JSON does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f, parse_constant=reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid CVDMS label JSON file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in label file: {path}")

    return data


def load_cvdms_label_file_from_path(
    label_path: Path,
    bbox_annotation_id: str | None = None,
) -> CvdmsLabelFile:
    raw_label = read_label_json(label_path)
    annotations = parse_annotations(raw_label, label_path)

    return CvdmsLabelFile(
        bbox_annotation_id=bbox_annotation_id.strip() if isinstance(bbox_annotation_id, str) else None,
        path=label_path,
        annotations=annotations,
    )


def load_cvdms_label_file(
    labels_cache_dir: Path,
    bbox_annotation_id: str,
    split: str | None = None,
) -> CvdmsLabelFile:
    label_path = resolve_cached_label_path(
        labels_cache_dir=labels_cache_dir,
        bbox_annotation_id=bbox_annotation_id,
        split=split,
    )

    return load_cvdms_label_file_from_path(
        label_path=label_path,
        bbox_annotation_id=bbox_annotation_id,
    )


def parse_annotations(
    raw_label: dict[str, Any],
    label_path: Path,
) -> list[dict[str, Any]]:
    annotations = raw_label.get("annotations")

    if not isinstance(annotations, list):
        raise ValueError(f"Label file missing annotations list: {label_path}")

    parsed_annotations = []

    for idx, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError(
                f"Expected annotation object in {label_path} at annotations[{idx}]"
            )

        parsed_annotations.append(validate_annotation(annotation, label_path, idx))

    return parsed_annotations


def validate_annotation(
    annotation: dict[str, Any],
    label_path: Path,
    annotation_index: int,
) -> dict[str, Any]:
    class_name = annotation.get("class_name")

    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError(
            f"Invalid class_name in {label_path} at annotations[{annotation_index}]"
        )

    required_bbox_fields = ["top", "left", "height", "width"]

    for field_name in required_bbox_fields:
        if field_name not in annotation:
            raise ValueError(
                f"Missing bbox field {field_name!r} in {label_path} "
                f"at annotations[{annotation_index}]"
            )

    cleaned = dict(annotation)
    cleaned["class_name"] = class_name.strip()

    return cleaned


def get_local_label_paths_from_manifest_row(
    manifest_row: dict[str, Any],
    project_root: Path | None = None,
) -> list[Path]:
    local_label_paths = manifest_row.get("local_label_paths")

    if not isinstance(local_label_paths, list) or not local_label_paths:
        image_id = manifest_row.get("image_id", "<unknown>")
        raise ValueError(f"Manifest row for image {image_id!r} has no local_label_paths")

    resolved_paths = []

    for idx, path_value in enumerate(local_label_paths):
        if not isinstance(path_value, str) or not path_value.strip():
            image_id = manifest_row.get("image_id", "<unknown>")
            raise ValueError(
                f"Manifest row for image {image_id!r} has invalid "
                f"local_label_paths[{idx}]: {path_value!r}"
            )

        resolved_paths.append(
            resolve_project_path(
                path_value=path_value.strip(),
                project_root=project_root,
            )
        )

    return resolved_paths


def load_annotations_from_local_label_paths(
    manifest_row: dict[str, Any],
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    local_label_paths = get_local_label_paths_from_manifest_row(
        manifest_row=manifest_row,
        project_root=project_root,
    )
    bbox_annotation_ids = manifest_row.get("bbox_annotation_ids", [])

    all_annotations = []

    for idx, label_path in enumerate(local_label_paths):
        bbox_annotation_id = None

        if isinstance(bbox_annotation_ids, list) and idx < len(bbox_annotation_ids):
            possible_id = bbox_annotation_ids[idx]
            if isinstance(possible_id, str) and possible_id.strip():
                bbox_annotation_id = possible_id.strip()

        label_file = load_cvdms_label_file_from_path(
            label_path=label_path,
            bbox_annotation_id=bbox_annotation_id,
        )
        all_annotations.extend(label_file.annotations)

    return all_annotations


def load_annotations_from_bbox_ids(
    labels_cache_dir: Path,
    manifest_row: dict[str, Any],
) -> list[dict[str, Any]]:
    bbox_annotation_ids = manifest_row.get("bbox_annotation_ids")
    split = manifest_row.get("split")

    if not isinstance(bbox_annotation_ids, list) or not bbox_annotation_ids:
        image_id = manifest_row.get("image_id", "<unknown>")
        raise ValueError(f"Manifest row for image {image_id!r} has no bbox_annotation_ids")

    if not isinstance(split, str) or not split.strip():
        split = None

    all_annotations = []

    for bbox_annotation_id in bbox_annotation_ids:
        label_file = load_cvdms_label_file(
            labels_cache_dir=labels_cache_dir,
            bbox_annotation_id=bbox_annotation_id,
            split=split,
        )
        all_annotations.extend(label_file.annotations)

    return all_annotations


def load_annotations_for_manifest_row(
    manifest_row: dict[str, Any],
    project_root: Path | None = None,
    labels_cache_dir: Path | None = None,
    prefer_local_paths: bool = True,
) -> list[dict[str, Any]]:
    if prefer_local_paths and "local_label_paths" in manifest_row:
        return load_annotations_from_local_label_paths(
            manifest_row=manifest_row,
            project_root=project_root,
        )

    if labels_cache_dir is None:
        raise ValueError(
            "labels_cache_dir is required when manifest row does not provide local_label_paths"
        )

    return load_annotations_from_bbox_ids(
        labels_cache_dir=labels_cache_dir,
        manifest_row=manifest_row,
    )


def filter_annotations_to_classes(
    annotations: list[dict[str, Any]],
    allowed_classes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept_annotations = []
    dropped_counts_by_class: dict[str, int] = {}

    for annotation in annotations:
        class_name = annotation["class_name"]

        if class_name in allowed_classes:
            kept_annotations.append(annotation)
        else:
            dropped_counts_by_class[class_name] = dropped_counts_by_class.get(class_name, 0) + 1

    return kept_annotations, dropped_counts_by_class


def summarize_annotations(
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    counts_by_class: dict[str, int] = {}

    for annotation in annotations:
        class_name = annotation.get("class_name", "<missing>")
        counts_by_class[class_name] = counts_by_class.get(class_name, 0) + 1

    return {
        "num_annotations": len(annotations),
        "counts_by_class": dict(sorted(counts_by_class.items())),
    }


def validate_cached_labels_exist_for_rows(
    rows: list[dict[str, Any]],
    project_root: Path | None = None,
    labels_cache_dir: Path | None = None,
    prefer_local_paths: bool = True,
    max_missing_examples: int = 20,
) -> dict[str, Any]:
    total_label_refs = 0
    unique_label_refs = set()
    missing_paths = []
    rows_with_local_label_paths = 0
    rows_using_bbox_id_fallback = 0

    for row in rows:
        if prefer_local_paths and "local_label_paths" in row:
            label_paths = get_local_label_paths_from_manifest_row(
                manifest_row=row,
                project_root=project_root,
            )
            rows_with_local_label_paths += 1

            for label_path in label_paths:
                total_label_refs += 1
                unique_label_refs.add(str(label_path))

                if not label_path.exists():
                    missing_paths.append(str(label_path))

            continue

        if labels_cache_dir is None:
            image_id = row.get("image_id", "<unknown>")
            raise ValueError(
                f"Manifest row for image {image_id!r} does not provide local_label_paths "
                "and labels_cache_dir was not provided"
            )

        rows_using_bbox_id_fallback += 1
        bbox_annotation_ids = row.get("bbox_annotation_ids", [])
        split = row.get("split")

        if not isinstance(bbox_annotation_ids, list):
            image_id = row.get("image_id", "<unknown>")
            raise ValueError(f"Manifest row for image {image_id!r} has invalid bbox_annotation_ids")

        if not isinstance(split, str) or not split.strip():
            split = None

        for bbox_annotation_id in bbox_annotation_ids:
            total_label_refs += 1
            unique_label_refs.add(bbox_annotation_id)

            label_path = resolve_cached_label_path(
                labels_cache_dir=labels_cache_dir,
                bbox_annotation_id=bbox_annotation_id,
                split=split,
            )

            if not label_path.exists():
                missing_paths.append(str(label_path))

    return {
        "total_label_refs": total_label_refs,
        "unique_label_refs": len(unique_label_refs),
        "missing_label_files": len(missing_paths),
        "missing_label_file_examples": missing_paths[:max_missing_examples],
        "rows_with_local_label_paths": rows_with_local_label_paths,
        "rows_using_bbox_id_fallback": rows_using_bbox_id_fallback,
    }