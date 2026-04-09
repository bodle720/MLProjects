import json
from pathlib import Path
from typing import Any

from dataset_bootstrap.dataset_helpers.common import (
    BootstrapConfig,
    download_http_file,
    ensure_dir,
    extract_zip,
)

# Object Det vars
COCO_SPLIT = "train2017"
COCO_TRAIN2017_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_ANNOTATIONS_TRAINVAL2017_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Semantic Seg vars
COCO_STUFFTHINGMAPS_URL = (
    "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/"
    "stuffthingmaps_trainval2017.zip"
)
COCO_STUFF_LABELS_URL = "https://raw.githubusercontent.com/nightrome/cocostuff/master/labels.txt"

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def old_dirs_from_run(reuse_from_run_dir: Path | None) -> tuple[Path | None, Path | None]:
    if reuse_from_run_dir is None:
        return None, None
    old_work_dir = reuse_from_run_dir / "_work"
    return old_work_dir / "downloads", old_work_dir / "extracted"

def find_train_images_dir(root: Path) -> Path | None:
    if not root.exists():
        return None

    direct = root / COCO_SPLIT
    if direct.is_dir():
        return direct

    for candidate in root.rglob(COCO_SPLIT):
        if candidate.is_dir():
            return candidate

    return None

def find_file_named(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None

    direct = root / filename
    if direct.is_file():
        return direct

    for candidate in root.rglob(filename):
        if candidate.is_file():
            return candidate

    return None

def resolve_train_images_dir(
    *,
    config: BootstrapConfig,
    reuse_stats: dict[str, Any],
) -> Path:
    download_dir = config.work_dir / "downloads"
    extract_dir = config.work_dir / "extracted"

    ensure_dir(download_dir)
    ensure_dir(extract_dir)

    old_download_dir, old_extract_dir = old_dirs_from_run(config.reuse_from_run_dir)

    if old_extract_dir is not None:
        old_images_dir = find_train_images_dir(old_extract_dir)
        if old_images_dir is not None:
            reuse_stats["reused_train_images"] = True
            return old_images_dir

    current_images_dir = find_train_images_dir(extract_dir)
    if current_images_dir is not None:
        return current_images_dir

    train_zip_path = None
    if old_download_dir is not None:
        old_train_zip = old_download_dir / "train2017.zip"
        if old_train_zip.is_file():
            train_zip_path = old_train_zip
            reuse_stats["reused_train_images"] = True

    if train_zip_path is None:
        train_zip_path = download_http_file(
            COCO_TRAIN2017_URL,
            download_dir / "train2017.zip",
        )

    images_extract_root = extract_dir / "images"
    ensure_dir(images_extract_root)

    if find_train_images_dir(images_extract_root) is None:
        extract_zip(train_zip_path, images_extract_root)

    images_dir = find_train_images_dir(images_extract_root)
    if images_dir is None:
        raise RuntimeError("Could not locate COCO train2017 image directory after extraction")

    return images_dir

def resolve_instances_json(
    *,
    config: BootstrapConfig,
    reuse_stats: dict[str, Any],
) -> Path:
    download_dir = config.work_dir / "downloads"
    extract_dir = config.work_dir / "extracted"

    ensure_dir(download_dir)
    ensure_dir(extract_dir)

    old_download_dir, old_extract_dir = old_dirs_from_run(config.reuse_from_run_dir)

    if old_extract_dir is not None:
        old_instances_json = find_file_named(old_extract_dir, "instances_train2017.json")
        if old_instances_json is not None:
            reuse_stats["reused_instances_json"] = True
            return old_instances_json

    current_instances_json = find_file_named(extract_dir, "instances_train2017.json")
    if current_instances_json is not None:
        return current_instances_json

    ann_zip_path = None
    if old_download_dir is not None:
        old_ann_zip = old_download_dir / "annotations_trainval2017.zip"
        if old_ann_zip.is_file():
            ann_zip_path = old_ann_zip
            reuse_stats["reused_annotations_zip"] = True

    if ann_zip_path is None:
        ann_zip_path = download_http_file(
            COCO_ANNOTATIONS_TRAINVAL2017_URL,
            download_dir / "annotations_trainval2017.zip",
        )

    ann_extract_root = extract_dir / "annotations"
    ensure_dir(ann_extract_root)

    if find_file_named(ann_extract_root, "instances_train2017.json") is None:
        extract_zip(ann_zip_path, ann_extract_root)

    instances_json = find_file_named(ann_extract_root, "instances_train2017.json")
    if instances_json is None:
        raise RuntimeError("Could not locate instances_train2017.json after extraction")

    return instances_json

def find_stuffthingmaps_train_dir(root: Path) -> Path | None:
    if not root.exists():
        return None

    direct = root / COCO_SPLIT
    if (
        direct.is_dir()
        and "stuffthingmaps" in str(direct).lower()
        and any(p.is_file() and p.suffix.lower() == ".png" for p in direct.iterdir())
    ):
        return direct

    for candidate in root.rglob(COCO_SPLIT):
        if not candidate.is_dir():
            continue
        if "stuffthingmaps" not in str(candidate).lower():
            continue
        if any(p.is_file() and p.suffix.lower() == ".png" for p in candidate.iterdir()):
            return candidate

    return None

def resolve_stuffthingmaps_train_dir(
    *,
    config: BootstrapConfig,
    reuse_stats: dict[str, Any],
) -> Path:
    download_dir = config.work_dir / "downloads"
    extract_dir = config.work_dir / "extracted"

    ensure_dir(download_dir)
    ensure_dir(extract_dir)

    old_download_dir, old_extract_dir = old_dirs_from_run(config.reuse_from_run_dir)

    if old_extract_dir is not None:
        old_stuffthingmaps_train_dir = find_stuffthingmaps_train_dir(old_extract_dir)
        if old_stuffthingmaps_train_dir is not None:
            reuse_stats["reused_stuffthingmaps_dir"] = True
            return old_stuffthingmaps_train_dir

    current_stuffthingmaps_train_dir = find_stuffthingmaps_train_dir(extract_dir)
    if current_stuffthingmaps_train_dir is not None:
        return current_stuffthingmaps_train_dir

    stuff_zip_path = None
    if old_download_dir is not None:
        old_stuff_zip = old_download_dir / "stuffthingmaps_trainval2017.zip"
        if old_stuff_zip.is_file():
            stuff_zip_path = old_stuff_zip
            reuse_stats["reused_stuffthingmaps_zip"] = True

    if stuff_zip_path is None:
        stuff_zip_path = download_http_file(
            COCO_STUFFTHINGMAPS_URL,
            download_dir / "stuffthingmaps_trainval2017.zip",
        )

    stuff_extract_root = extract_dir / "stuffthingmaps"
    ensure_dir(stuff_extract_root)

    if find_stuffthingmaps_train_dir(stuff_extract_root) is None:
        extract_zip(stuff_zip_path, stuff_extract_root)

    stuffthingmaps_train_dir = find_stuffthingmaps_train_dir(stuff_extract_root)
    if stuffthingmaps_train_dir is None:
        raise RuntimeError("Could not locate COCO-Stuff train2017 masks after extraction")

    return stuffthingmaps_train_dir

def resolve_cocostuff_labels_path(
    *,
    config: BootstrapConfig,
    reuse_stats: dict[str, Any],
) -> Path:
    download_dir = config.work_dir / "downloads"
    ensure_dir(download_dir)

    old_download_dir, _ = old_dirs_from_run(config.reuse_from_run_dir)

    if old_download_dir is not None:
        old_labels = old_download_dir / "cocostuff_labels.txt"
        if old_labels.is_file():
            reuse_stats["reused_stuff_labels"] = True
            return old_labels

    return download_http_file(
        COCO_STUFF_LABELS_URL,
        download_dir / "cocostuff_labels.txt",
    )

def resolve_stuff_mask_path(
    *,
    stuffthingmaps_train_dir: Path,
    image_filename: str,
) -> Path | None:
    stem = Path(image_filename).stem
    candidates = [
        stuffthingmaps_train_dir / f"{stem}.png",
        stuffthingmaps_train_dir / f"{stem}_labelTrainIds.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None

def color_for_index(idx: int) -> tuple[int, int, int]:
    r = (37 * idx + 23) % 256
    g = (73 * idx + 47) % 256
    b = (109 * idx + 89) % 256

    rgb = (
        1 if r == 0 else r,
        1 if g == 0 else g,
        1 if b == 0 else b,
    )
    if rgb == (0, 0, 0):
        return 255, 0, 0
    return rgb

def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)