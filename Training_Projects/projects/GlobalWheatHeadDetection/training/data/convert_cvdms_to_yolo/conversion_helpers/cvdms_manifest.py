import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OBJECT_DETECTION_LABEL_TYPE = "object-detection"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CvdmsDatasetMetadata:
    dataset_id: str
    version: int
    label_type: str
    effective_classes: list[str]
    class_to_id: dict[str, int]
    split_counts: dict[str, int]


@dataclass(frozen=True)
class CvdmsManifestBundle:
    metadata: CvdmsDatasetMetadata
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]


def reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f, parse_constant=reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {path}")

    return data


def read_jsonl_file(path: Path, allow_missing: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"JSONL file does not exist: {path}")

    rows = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                row = json.loads(stripped, parse_constant=reject_json_constant)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_number}")

            rows.append(row)

    return rows


def parse_metadata(metadata_path: Path) -> CvdmsDatasetMetadata:
    raw_metadata = read_json_file(metadata_path)

    dataset_id = require_string(raw_metadata, "dataset_id", metadata_path)
    label_type = require_string(raw_metadata, "label_type", metadata_path)
    version = require_int(raw_metadata, "version", metadata_path)

    if label_type != OBJECT_DETECTION_LABEL_TYPE:
        raise ValueError(
            f"Expected metadata label_type={OBJECT_DETECTION_LABEL_TYPE!r}, "
            f"got {label_type!r}"
        )

    effective_classes = parse_effective_classes(raw_metadata, metadata_path)
    class_to_id = parse_class_to_id(raw_metadata, effective_classes, metadata_path)

    split_counts = raw_metadata.get("split_counts", {})
    if not isinstance(split_counts, dict):
        raise ValueError(f"metadata split_counts must be an object in {metadata_path}")

    return CvdmsDatasetMetadata(
        dataset_id=dataset_id,
        version=version,
        label_type=label_type,
        effective_classes=effective_classes,
        class_to_id=class_to_id,
        split_counts=split_counts,
    )


def require_string(data: dict[str, Any], key: str, source: Path) -> str:
    value = data.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid string field {key!r} in {source}")

    return value.strip()


def require_int(data: dict[str, Any], key: str, source: Path) -> int:
    value = data.get(key)

    if not isinstance(value, int):
        raise ValueError(f"Missing or invalid integer field {key!r} in {source}")

    return value


def parse_effective_classes(
    raw_metadata: dict[str, Any],
    metadata_path: Path,
) -> list[str]:
    effective_classes = raw_metadata.get("effective_classes")

    if not isinstance(effective_classes, list) or not effective_classes:
        raise ValueError(f"metadata effective_classes must be a non-empty list in {metadata_path}")

    parsed_classes = []

    for idx, class_name in enumerate(effective_classes):
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError(
                f"metadata effective_classes[{idx}] must be a non-empty string "
                f"in {metadata_path}"
            )

        parsed_classes.append(class_name.strip())

    duplicate_classes = sorted({
        class_name
        for class_name in parsed_classes
        if parsed_classes.count(class_name) > 1
    })

    if duplicate_classes:
        raise ValueError(f"Duplicate effective_classes in {metadata_path}: {duplicate_classes}")

    return parsed_classes


def parse_class_to_id(
    raw_metadata: dict[str, Any],
    effective_classes: list[str],
    metadata_path: Path,
) -> dict[str, int]:
    raw_class_to_idx = raw_metadata.get("class_to_idx")

    if not isinstance(raw_class_to_idx, dict):
        raise ValueError(f"metadata class_to_idx must be an object in {metadata_path}")

    class_to_id = {}

    for class_name in effective_classes:
        raw_idx = raw_class_to_idx.get(class_name)

        if not isinstance(raw_idx, int):
            raise ValueError(
                f"metadata class_to_idx missing integer id for class {class_name!r} "
                f"in {metadata_path}"
            )

        class_to_id[class_name] = raw_idx

    expected_ids = set(range(len(effective_classes)))
    actual_ids = set(class_to_id.values())

    if actual_ids != expected_ids:
        raise ValueError(
            f"metadata class_to_idx IDs must be contiguous 0..{len(effective_classes) - 1}. "
            f"Got: {sorted(actual_ids)}"
        )

    id_to_class = {
        class_id: class_name
        for class_name, class_id in class_to_id.items()
    }

    if len(id_to_class) != len(class_to_id):
        raise ValueError(f"metadata class_to_idx has duplicate class IDs in {metadata_path}")

    return class_to_id


def require_manifest_string(
    row: dict[str, Any],
    key: str,
    source_name: str,
    row_number: int,
) -> str:
    value = row.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source_name} row {row_number} missing valid {key!r}")

    return value.strip()


def require_string_list(
    row: dict[str, Any],
    key: str,
    source_name: str,
    row_number: int,
    allow_empty: bool = False,
) -> list[str]:
    value = row.get(key)

    if not isinstance(value, list):
        raise ValueError(f"{source_name} row {row_number} {key!r} must be a list")

    if not value and not allow_empty:
        raise ValueError(f"{source_name} row {row_number} has empty {key!r}")

    cleaned_values = []

    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"{source_name} row {row_number} {key}[{idx}] "
                "must be a non-empty string"
            )

        cleaned_values.append(item.strip())

    return cleaned_values


def validate_manifest_row(
    row: dict[str, Any],
    expected_split: str,
    source_name: str,
    row_number: int,
    require_local_paths: bool = True,
) -> dict[str, Any]:
    image_id = require_manifest_string(row, "image_id", source_name, row_number)
    source_ref = require_manifest_string(row, "source_ref", source_name, row_number)

    if not source_ref.startswith("s3://"):
        raise ValueError(
            f"{source_name} row {row_number} source_ref must be an s3:// URI, "
            f"got {source_ref!r}"
        )

    label_type = row.get("label_type")
    if label_type != OBJECT_DETECTION_LABEL_TYPE:
        raise ValueError(
            f"{source_name} row {row_number} expected label_type "
            f"{OBJECT_DETECTION_LABEL_TYPE!r}, got {label_type!r}"
        )

    split = row.get("split")
    if split != expected_split:
        raise ValueError(
            f"{source_name} row {row_number} expected split {expected_split!r}, "
            f"got {split!r}"
        )

    bbox_annotation_ids = require_string_list(
        row=row,
        key="bbox_annotation_ids",
        source_name=source_name,
        row_number=row_number,
    )

    cleaned_row = dict(row)
    cleaned_row["image_id"] = image_id
    cleaned_row["source_ref"] = source_ref
    cleaned_row["label_type"] = label_type
    cleaned_row["split"] = split
    cleaned_row["bbox_annotation_ids"] = bbox_annotation_ids

    if require_local_paths:
        local_image_path = require_manifest_string(
            row=row,
            key="local_image_path",
            source_name=source_name,
            row_number=row_number,
        )
        local_label_paths = require_string_list(
            row=row,
            key="local_label_paths",
            source_name=source_name,
            row_number=row_number,
        )

        cleaned_row["local_image_path"] = local_image_path
        cleaned_row["local_label_paths"] = local_label_paths

    elif "local_image_path" in row:
        local_image_path = row.get("local_image_path")
        if isinstance(local_image_path, str) and local_image_path.strip():
            cleaned_row["local_image_path"] = local_image_path.strip()

    if not require_local_paths and "local_label_paths" in row:
        local_label_paths = row.get("local_label_paths")
        if isinstance(local_label_paths, list):
            cleaned_row["local_label_paths"] = [
                item.strip()
                for item in local_label_paths
                if isinstance(item, str) and item.strip()
            ]

    return cleaned_row


def validate_manifest_rows(
    rows: list[dict[str, Any]],
    expected_split: str,
    source_name: str,
    require_local_paths: bool = True,
) -> list[dict[str, Any]]:
    validated_rows = []
    seen_image_ids = set()

    for idx, row in enumerate(rows, start=1):
        validated_row = validate_manifest_row(
            row=row,
            expected_split=expected_split,
            source_name=source_name,
            row_number=idx,
            require_local_paths=require_local_paths,
        )

        image_id = validated_row["image_id"]
        if image_id in seen_image_ids:
            raise ValueError(f"{source_name} contains duplicate image_id: {image_id}")

        seen_image_ids.add(image_id)
        validated_rows.append(validated_row)

    return validated_rows


def load_manifest_bundle(
    metadata_path: Path,
    manifest_dir: Path,
    require_local_paths: bool = True,
    allow_missing_test: bool = False,
) -> CvdmsManifestBundle:
    metadata = parse_metadata(metadata_path)

    train_path = manifest_dir / "train.jsonl"
    val_path = manifest_dir / "val.jsonl"
    test_path = manifest_dir / "test.jsonl"

    train_rows = validate_manifest_rows(
        rows=read_jsonl_file(train_path),
        expected_split="train",
        source_name=str(train_path),
        require_local_paths=require_local_paths,
    )
    val_rows = validate_manifest_rows(
        rows=read_jsonl_file(val_path),
        expected_split="val",
        source_name=str(val_path),
        require_local_paths=require_local_paths,
    )
    test_rows = validate_manifest_rows(
        rows=read_jsonl_file(test_path, allow_missing=allow_missing_test),
        expected_split="test",
        source_name=str(test_path),
        require_local_paths=require_local_paths,
    )

    return CvdmsManifestBundle(
        metadata=metadata,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
    )


def flatten_manifest_rows(bundle: CvdmsManifestBundle) -> list[dict[str, Any]]:
    rows = []
    rows.extend(bundle.train_rows)
    rows.extend(bundle.val_rows)
    rows.extend(bundle.test_rows)
    return rows


def count_rows_with_field(rows: list[dict[str, Any]], field_name: str) -> int:
    return sum(1 for row in rows if field_name in row)


def summarize_manifest_bundle(bundle: CvdmsManifestBundle) -> dict[str, Any]:
    all_rows = flatten_manifest_rows(bundle)

    return {
        "dataset_id": bundle.metadata.dataset_id,
        "version": bundle.metadata.version,
        "label_type": bundle.metadata.label_type,
        "num_classes": len(bundle.metadata.effective_classes),
        "classes": bundle.metadata.effective_classes,
        "class_to_id": bundle.metadata.class_to_id,
        "metadata_split_counts": bundle.metadata.split_counts,
        "manifest_row_counts": {
            "train": len(bundle.train_rows),
            "val": len(bundle.val_rows),
            "test": len(bundle.test_rows),
        },
        "local_path_fields": {
            "rows_with_local_image_path": count_rows_with_field(all_rows, "local_image_path"),
            "rows_with_local_label_paths": count_rows_with_field(all_rows, "local_label_paths"),
        },
    }