import json
from pathlib import Path
from typing import Any

from dataset_bootstrap.dataset_helpers.common import (
    BootstrapConfig,
    download_http_file,
    ensure_dir,
    extract_zip,
)

COCO_ANNOTATIONS_TRAINVAL2017_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

COCO_STUFFTHINGMAPS_URL = (
    "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/"
    "stuffthingmaps_trainval2017.zip"
)

COCO_STUFF_LABELS_URL = (
    "https://raw.githubusercontent.com/nightrome/cocostuff/master/labels.txt"
)

def require_coco_split(config: BootstrapConfig) -> str:
    if config.split not in {"train", "val"}:
        raise ValueError("COCO bootstrap requires config.split to be 'train' or 'val'")
    return config.split

def coco_split_dirname(split: str) -> str:
    if split == "train":
        return "train2017"
    if split == "val":
        return "val2017"
    raise ValueError(f"Unsupported COCO split: {split}")

def coco_images_zip_filename(split: str) -> str:
    return f"{coco_split_dirname(split)}.zip"

def coco_images_url(split: str) -> str:
    return f"http://images.cocodataset.org/zips/{coco_split_dirname(split)}.zip"

def coco_instances_filename(split: str) -> str:
    return f"instances_{coco_split_dirname(split)}.json"

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def old_dirs_from_run(reuse_from_run_dir: Path | None) -> tuple[Path | None, Path | None]:
    if reuse_from_run_dir is None:
        return None, None
    old_work_dir = reuse_from_run_dir / "_work"
    return old_work_dir / "downloads", old_work_dir / "extracted"

def find_images_dir(root: Path, split: str) -> Path | None:
    if not root.exists():
        return None

    split_dirname = coco_split_dirname(split)

    direct = root / split_dirname
    if _looks_like_coco_images_dir(direct):
        return direct

    for candidate in root.rglob(split_dirname):
        if _looks_like_coco_images_dir(candidate):
            return candidate

    return None

def _looks_like_coco_images_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    return any(
        child.is_file() and child.suffix.lower() in {".jpg", ".jpeg"}
        for child in path.iterdir()
    )

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

def resolve_images_dir(
    *,
    config: BootstrapConfig,
    reuse_stats: dict[str, Any],
) -> Path:
    split = require_coco_split(config)
    split_dirname = coco_split_dirname(split)
    zip_name = coco_images_zip_filename(split)
    url = coco_images_url(split)

    download_dir = config.work_dir / "downloads"
    extract_dir = config.work_dir / "extracted"

    ensure_dir(download_dir)
    ensure_dir(extract_dir)

    old_download_dir, old_extract_dir = old_dirs_from_run(config.reuse_from_run_dir)

    if old_extract_dir is not None:
        old_images_dir = find_images_dir(old_extract_dir, split)
        if old_images_dir is not None:
            reuse_stats["reused_images"] = True
            return old_images_dir

    current_images_dir = find_images_dir(extract_dir, split)
    if current_images_dir is not None:
        return current_images_dir

    image_zip_path = None
    if old_download_dir is not None:
        old_zip = old_download_dir / zip_name
        if old_zip.is_file():
            image_zip_path = old_zip
            reuse_stats["reused_images_zip"] = True

    if image_zip_path is None:
        image_zip_path = download_http_file(
            url,
            download_dir / zip_name,
        )

    images_extract_root = extract_dir / "images"
    ensure_dir(images_extract_root)

    if find_images_dir(images_extract_root, split) is None:
        extract_zip(image_zip_path, images_extract_root)

    images_dir = find_images_dir(images_extract_root, split)
    if images_dir is None:
        raise RuntimeError(
            f"Could not locate COCO {split_dirname} image directory after extraction"
        )

    return images_dir

def resolve_instances_json(
    *,
    config: BootstrapConfig,
    reuse_stats: dict[str, Any],
) -> Path:
    split = require_coco_split(config)
    instances_filename = coco_instances_filename(split)

    download_dir = config.work_dir / "downloads"
    extract_dir = config.work_dir / "extracted"

    ensure_dir(download_dir)
    ensure_dir(extract_dir)

    old_download_dir, old_extract_dir = old_dirs_from_run(config.reuse_from_run_dir)

    if old_extract_dir is not None:
        old_instances_json = find_file_named(old_extract_dir, instances_filename)
        if old_instances_json is not None:
            reuse_stats["reused_instances_json"] = True
            return old_instances_json

    current_instances_json = find_file_named(extract_dir, instances_filename)
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

    if find_file_named(ann_extract_root, instances_filename) is None:
        extract_zip(ann_zip_path, ann_extract_root)

    instances_json = find_file_named(ann_extract_root, instances_filename)
    if instances_json is None:
        raise RuntimeError(f"Could not locate {instances_filename} after extraction")

    return instances_json

def find_stuffthingmaps_dir(root: Path, split: str) -> Path | None:
    if not root.exists():
        return None

    split_dirname = coco_split_dirname(split)

    direct = root / split_dirname
    if (
        direct.is_dir()
        and "stuffthingmaps" in str(direct).lower()
        and any(p.is_file() and p.suffix.lower() == ".png" for p in direct.iterdir())
    ):
        return direct

    for candidate in root.rglob(split_dirname):
        if not candidate.is_dir():
            continue
        if "stuffthingmaps" not in str(candidate).lower():
            continue
        if any(p.is_file() and p.suffix.lower() == ".png" for p in candidate.iterdir()):
            return candidate

    return None

def resolve_stuffthingmaps_dir(
    *,
    config: BootstrapConfig,
    reuse_stats: dict[str, Any],
) -> Path:
    split = require_coco_split(config)
    split_dirname = coco_split_dirname(split)

    download_dir = config.work_dir / "downloads"
    extract_dir = config.work_dir / "extracted"

    ensure_dir(download_dir)
    ensure_dir(extract_dir)

    old_download_dir, old_extract_dir = old_dirs_from_run(config.reuse_from_run_dir)

    if old_extract_dir is not None:
        old_stuffthingmaps_dir = find_stuffthingmaps_dir(old_extract_dir, split)
        if old_stuffthingmaps_dir is not None:
            reuse_stats["reused_stuffthingmaps_dir"] = True
            return old_stuffthingmaps_dir

    current_stuffthingmaps_dir = find_stuffthingmaps_dir(extract_dir, split)
    if current_stuffthingmaps_dir is not None:
        return current_stuffthingmaps_dir

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

    if find_stuffthingmaps_dir(stuff_extract_root, split) is None:
        extract_zip(stuff_zip_path, stuff_extract_root)

    stuffthingmaps_dir = find_stuffthingmaps_dir(stuff_extract_root, split)
    if stuffthingmaps_dir is None:
        raise RuntimeError(
            f"Could not locate COCO-Stuff {split_dirname} masks after extraction"
        )

    return stuffthingmaps_dir

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
    stuffthingmaps_dir: Path,
    image_filename: str,
) -> Path | None:
    stem = Path(image_filename).stem
    candidates = [
        stuffthingmaps_dir / f"{stem}.png",
        stuffthingmaps_dir / f"{stem}_labelTrainIds.png",
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

    return rgb

def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)