import csv
import re
from pathlib import Path
from typing import Any

from .common import (
    BootstrapConfig,
    BootstrapFailure,
    BootstrapResult,
    deterministic_sample,
    download_http_file,
    ensure_dir,
    extract_zip,
    make_object_detection_row,
    s3_key_join,
    upload_file_to_s3,
)


DATASET_NAME = "Global Wheat Head Dataset 2021"
DATASET_SLUG = "global_wheat_head_2021"
TASK_NAME = "object-detection"
CLASS_NAME = "wheat_head"

VALID_SPLITS = {"train", "val", "test"}

GWHD_2021_ZIP_URL = "https://zenodo.org/records/5092309/files/gwhd_2021.zip?download=1"
GWHD_2021_ZIP_FILENAME = "gwhd_2021.zip"
GWHD_2021_ZIP_MD5 = "22b4b542c9ae7e056d7fcdeae9ecaed5"

EXPECTED_IMAGE_EXTENSION = ".png"
EXPECTED_IMAGE_SIZE = 1024


def bootstrap_wheat_head_2021(config: BootstrapConfig, s3_client: Any) -> BootstrapResult:
    split = require_split(config.split)

    reuse_stats: dict[str, Any] = {
        "reuse_from_run_dir": str(config.reuse_from_run_dir) if config.reuse_from_run_dir else None,
        "reused_archive": False,
        "reused_extracted": False,
    }

    archive_path = resolve_archive(
        config=config,
        reuse_stats=reuse_stats,
    )
    extracted_root = resolve_extracted_root(
        config=config,
        archive_path=archive_path,
        reuse_stats=reuse_stats,
    )

    split_csv_path = find_split_csv(extracted_root, split)
    images_dir = find_images_dir(extracted_root)

    if split_csv_path is None:
        raise RuntimeError(f"Could not locate competition_{split}.csv under: {extracted_root}")
    if images_dir is None:
        raise RuntimeError(f"Could not locate images directory under: {extracted_root}")

    candidates, parse_stats = build_object_detection_candidates(
        split_csv_path=split_csv_path,
        images_dir=images_dir,
    )
    selected = deterministic_sample(candidates, config.max_items, config.sample_seed)

    manifest_rows: list[dict[str, Any]] = []
    failures: list[BootstrapFailure] = []

    total = len(selected)
    for idx, record in enumerate(selected, start=1):
        image_name = record["image_name"]
        image_path = record["image_path"]

        if idx == 1 or idx % 100 == 0 or idx == total:
            print(
                f"[bootstrap] On {idx} out of {total}. "
                f"image_name={image_name}, split={split}, file={image_path.name}"
            )

        try:
            image_s3_key = s3_key_join(
                config.s3_prefix,
                DATASET_SLUG,
                TASK_NAME,
                "images",
                split,
                image_path.name,
            )
            source_ref = upload_file_to_s3(
                s3_client=s3_client,
                local_path=image_path,
                bucket=config.bucket,
                key=image_s3_key,
            )

            manifest_rows.append(
                make_object_detection_row(
                    source_ref=source_ref,
                    annotations=record["annotations"],
                )
            )

        except Exception as exc:  # noqa: BLE001
            failures.append(
                BootstrapFailure(
                    dataset_item_id=str(image_name),
                    reason=str(exc),
                    context={
                        "split": split,
                        "image_path": str(image_path),
                        "domain": record.get("domain"),
                    },
                )
            )

    stats = {
        "upstream_dataset": DATASET_NAME,
        "dataset_slug": DATASET_SLUG,
        "task": TASK_NAME,
        "class_name": CLASS_NAME,
        "split": split,
        "archive_path": str(archive_path),
        "extracted_root": str(extracted_root),
        "split_csv_path": str(split_csv_path),
        "images_dir": str(images_dir),
        "discovered_positive_image_count": len(candidates),
        "selected_count": len(selected),
        "uploaded_manifest_row_count": len(manifest_rows),
        "failure_count": len(failures),
        "requested_max_items": config.max_items,
        "sample_seed": config.sample_seed,
        "expected_image_extension": EXPECTED_IMAGE_EXTENSION,
        "expected_image_size": EXPECTED_IMAGE_SIZE,
        **parse_stats,
        **reuse_stats,
    }

    return BootstrapResult(
        manifest_rows=manifest_rows,
        failures=failures,
        stats=stats,
    )


def require_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in VALID_SPLITS:
        valid = ", ".join(sorted(VALID_SPLITS))
        raise ValueError(f"split must be one of: {valid}")
    return normalized


def old_dirs_from_run(reuse_from_run_dir: Path | None) -> tuple[Path | None, Path | None]:
    if reuse_from_run_dir is None:
        return None, None

    old_work_dir = reuse_from_run_dir / "_work"
    return old_work_dir / "downloads", old_work_dir / "extracted"


def resolve_archive(
    *,
    config: BootstrapConfig,
    reuse_stats: dict[str, Any],
) -> Path:
    download_dir = config.work_dir / "downloads"
    ensure_dir(download_dir)

    old_download_dir, _ = old_dirs_from_run(config.reuse_from_run_dir)

    if old_download_dir is not None:
        old_archive = old_download_dir / GWHD_2021_ZIP_FILENAME
        if old_archive.is_file():
            reuse_stats["reused_archive"] = True
            print(f"[reuse] using prior archive: {old_archive}")
            return old_archive

    current_archive = download_dir / GWHD_2021_ZIP_FILENAME
    return download_http_file(
        GWHD_2021_ZIP_URL,
        current_archive,
        expected_md5=GWHD_2021_ZIP_MD5,
    )


def resolve_extracted_root(
    *,
    config: BootstrapConfig,
    archive_path: Path,
    reuse_stats: dict[str, Any],
) -> Path:
    extract_dir = config.work_dir / "extracted"
    ensure_dir(extract_dir)

    _, old_extract_dir = old_dirs_from_run(config.reuse_from_run_dir)

    if old_extract_dir is not None and looks_like_extracted_dataset(old_extract_dir):
        reuse_stats["reused_extracted"] = True
        print(f"[reuse] using prior extracted dataset: {old_extract_dir}")
        return old_extract_dir

    if looks_like_extracted_dataset(extract_dir):
        return extract_dir

    print(f"[extract] extracting archive to: {extract_dir}")
    extract_zip(archive_path, extract_dir)

    if not looks_like_extracted_dataset(extract_dir):
        raise RuntimeError(
            f"Archive was extracted, but expected GWHD 2021 files were not found under: {extract_dir}"
        )

    return extract_dir


def looks_like_extracted_dataset(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False

    return (
        find_images_dir(root) is not None
        and find_split_csv(root, "train") is not None
        and find_split_csv(root, "val") is not None
        and find_split_csv(root, "test") is not None
    )


def find_split_csv(root: Path, split: str) -> Path | None:
    filename = f"competition_{split}.csv"

    direct = root / filename
    if direct.is_file():
        return direct

    for candidate in root.rglob(filename):
        if candidate.is_file():
            return candidate

    return None


def find_images_dir(root: Path) -> Path | None:
    direct = root / "images"
    if looks_like_images_dir(direct):
        return direct

    candidates = []
    for candidate in root.rglob("images"):
        if looks_like_images_dir(candidate):
            candidates.append(candidate)

    if candidates:
        candidates.sort(key=lambda p: len(str(p)))
        return candidates[0]

    return None


def looks_like_images_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    try:
        return any(
            child.is_file() and child.suffix.lower() == EXPECTED_IMAGE_EXTENSION
            for child in path.iterdir()
        )
    except OSError:
        return False


def build_object_detection_candidates(
    *,
    split_csv_path: Path,
    images_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_stem_to_path, duplicate_image_stem_count = build_image_stem_map(images_dir)

    candidates: list[dict[str, Any]] = []
    rows_seen = 0
    no_box_rows = 0
    missing_image_rows = 0
    rows_with_no_valid_boxes = 0
    invalid_box_count = 0
    valid_box_count = 0

    with split_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required_columns = {"image_name", "BoxesString", "domain"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{split_csv_path} is missing required columns: {sorted(missing_columns)}"
            )

        for row in reader:
            rows_seen += 1

            image_name = require_nonempty_csv_value(row, "image_name")
            boxes_string = require_nonempty_csv_value(row, "BoxesString")
            domain = require_nonempty_csv_value(row, "domain")

            image_path = image_stem_to_path.get(image_name)
            if image_path is None:
                missing_image_rows += 1
                continue

            if boxes_string.strip().lower() == "no_box":
                no_box_rows += 1
                continue

            annotations, box_stats = parse_boxes_string(boxes_string)
            invalid_box_count += box_stats["invalid_box_count"]
            valid_box_count += box_stats["valid_box_count"]

            if not annotations:
                rows_with_no_valid_boxes += 1
                continue

            candidates.append(
                {
                    "image_name": image_name,
                    "image_path": image_path,
                    "domain": domain,
                    "annotations": annotations,
                    "annotation_count": len(annotations),
                }
            )

    stats = {
        "csv_rows_seen": rows_seen,
        "csv_no_box_rows_skipped": no_box_rows,
        "csv_missing_image_rows_skipped": missing_image_rows,
        "csv_rows_with_no_valid_boxes_skipped": rows_with_no_valid_boxes,
        "valid_box_count": valid_box_count,
        "invalid_box_count": invalid_box_count,
        "image_file_count": len(image_stem_to_path),
        "duplicate_image_stem_count": duplicate_image_stem_count,
    }

    return candidates, stats


def build_image_stem_map(images_dir: Path) -> tuple[dict[str, Path], int]:
    image_stem_to_path: dict[str, Path] = {}
    duplicate_count = 0

    for image_path in sorted(images_dir.rglob(f"*{EXPECTED_IMAGE_EXTENSION}")):
        if not image_path.is_file():
            continue

        stem = image_path.stem
        if stem in image_stem_to_path:
            duplicate_count += 1
            continue

        image_stem_to_path[stem] = image_path

    return image_stem_to_path, duplicate_count


def require_nonempty_csv_value(row: dict[str, Any], column_name: str) -> str:
    value = row.get(column_name)
    if not isinstance(value, str):
        raise ValueError(f"CSV column {column_name!r} must be a string")

    value = value.strip()
    if not value:
        raise ValueError(f"CSV column {column_name!r} cannot be empty")

    return value


def parse_boxes_string(boxes_string: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw = boxes_string.strip()

    box_chunks = re.findall(r"\[([^\]]+)\]", raw)
    if not box_chunks:
        box_chunks = [chunk.strip() for chunk in raw.split(";") if chunk.strip()]

    annotations: list[dict[str, Any]] = []
    invalid_box_count = 0

    for chunk in box_chunks:
        parsed = parse_box_chunk(chunk)
        if parsed is None:
            invalid_box_count += 1
            continue

        x_min, y_min, x_max, y_max = parsed
        annotation = xyxy_to_cvdms_annotation(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

        if annotation is None:
            invalid_box_count += 1
            continue

        annotations.append(annotation)

    stats = {
        "valid_box_count": len(annotations),
        "invalid_box_count": invalid_box_count,
    }

    return annotations, stats


def parse_box_chunk(chunk: str) -> tuple[float, float, float, float] | None:
    cleaned = chunk.strip().strip("[]").strip()
    parts = [part for part in re.split(r"[\s,]+", cleaned) if part]

    if len(parts) != 4:
        return None

    try:
        x_min, y_min, x_max, y_max = [float(part) for part in parts]
    except ValueError:
        return None

    return x_min, y_min, x_max, y_max


def xyxy_to_cvdms_annotation(
    *,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> dict[str, Any] | None:
    if x_min < 0 or y_min < 0:
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    if x_max > EXPECTED_IMAGE_SIZE or y_max > EXPECTED_IMAGE_SIZE:
        return None

    return {
        "class_name": CLASS_NAME,
        "top": y_min,
        "left": x_min,
        "height": y_max - y_min,
        "width": x_max - x_min,
    }