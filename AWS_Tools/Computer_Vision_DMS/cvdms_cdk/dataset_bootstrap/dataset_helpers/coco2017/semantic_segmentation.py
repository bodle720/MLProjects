import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataset_bootstrap.dataset_helpers.common import (
    BootstrapConfig,
    BootstrapFailure,
    BootstrapResult,
    deterministic_sample,
    make_semantic_segmentation_row,
    s3_key_join,
    upload_bytes_to_s3,
    upload_file_to_s3,
)
from dataset_bootstrap.dataset_helpers.coco2017.common import (
    color_for_index,
    require_coco_split,
    resolve_cocostuff_labels_path,
    resolve_images_dir,
    resolve_stuff_mask_path,
    resolve_stuffthingmaps_dir,
    rgb_to_hex,
)

from common.general_utils.class_normalizer import canonicalize_class_name

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

def coco_semantic_segmentation(config: BootstrapConfig, s3_client: Any) -> BootstrapResult:
    split = require_coco_split(config)

    reuse_stats: dict[str, Any] = {
        "reuse_from_run_dir": str(config.reuse_from_run_dir) if config.reuse_from_run_dir else None,
        "reused_images": False,
        "reused_images_zip": False,
        "reused_stuffthingmaps_zip": False,
        "reused_stuffthingmaps_dir": False,
        "reused_stuff_labels": False,
    }

    images_dir = resolve_images_dir(
        config=config,
        reuse_stats=reuse_stats,
    )
    stuffthingmaps_dir = resolve_stuffthingmaps_dir(
        config=config,
        reuse_stats=reuse_stats,
    )
    labels_path = resolve_cocostuff_labels_path(
        config=config,
        reuse_stats=reuse_stats,
    )

    id_to_name = _load_cocostuff_labels(labels_path)

    candidates = _build_semantic_candidates(
        images_dir=images_dir,
        stuffthingmaps_dir=stuffthingmaps_dir,
    )
    selected = deterministic_sample(candidates, config.max_items, config.sample_seed)

    manifest_rows: list[dict[str, Any]] = []
    failures: list[BootstrapFailure] = []

    total = len(selected)
    for idx, record in enumerate(selected, start=1):
        image_id = record["image_id"]
        image_path = record["image_path"]
        mask_path = record["mask_path"]

        if idx % 100 == 0 or idx == 1:
            print(
                f"On {idx} out of {total}. image_id = {image_id}, "
                f"split = {split}, file = {image_path.name}"
            )

        try:
            image_s3_key = s3_key_join(
                config.s3_prefix,
                "coco2017",
                "semantic-segmentation",
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

            rgb_mask_bytes, color_map = _build_semantic_rgb_mask_and_color_map(
                mask_path=mask_path,
                id_to_name=id_to_name,
            )

            mask_s3_key = s3_key_join(
                config.s3_prefix,
                "coco2017",
                "semantic-segmentation",
                "masks",
                split,
                f"{image_path.stem}.png",
            )
            mask_ref = upload_bytes_to_s3(
                s3_client=s3_client,
                payload=rgb_mask_bytes,
                bucket=config.bucket,
                key=mask_s3_key,
                content_type="image/png",
            )

            manifest_rows.append(
                make_semantic_segmentation_row(
                    source_ref=source_ref,
                    mask_ref=mask_ref,
                    color_map=color_map,
                )
            )

        except Exception as exc:  # noqa: BLE001
            failures.append(
                BootstrapFailure(
                    dataset_item_id=str(image_id),
                    reason=str(exc),
                    context={
                        "split": split,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                    },
                )
            )

    stats = {
        "upstream_dataset": "COCO 2017 + COCO-Stuff",
        "task": "semantic-segmentation",
        "split": split,
        "images_dir": str(images_dir),
        "stuffthingmaps_dir": str(stuffthingmaps_dir),
        "labels_path": str(labels_path),
        "discovered_count": len(candidates),
        "selected_count": len(selected),
        **reuse_stats,
    }

    return BootstrapResult(
        manifest_rows=manifest_rows,
        failures=failures,
        stats=stats,
    )

def _build_semantic_candidates(
    *,
    images_dir: Path,
    stuffthingmaps_dir: Path,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        mask_path = resolve_stuff_mask_path(
            stuffthingmaps_dir=stuffthingmaps_dir,
            image_filename=image_path.name,
        )
        if mask_path is None or not mask_path.is_file():
            continue

        candidates.append(
            {
                "image_id": image_path.stem,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )

    return candidates

def _load_cocostuff_labels(labels_path: Path) -> dict[int, str]:
    """
    labels.txt is treated as line-indexed.
    We skip class 0/unlabeled and any background-like names because the output
    manifest contract wants an explicit background entry supplied separately.
    """
    out: dict[int, str] = {}

    raw_lines = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
    for idx, raw_name in enumerate(raw_lines):
        if raw_name == "":
            continue

        name = canonicalize_class_name(
            raw_name,
            field_name=f"cocostuff_labels[{idx}]",
            allow_background=True,
        )

        if name in {"bg", "background", "unlabeled"}:
            continue

        out[idx] = name

    return out

def _build_semantic_rgb_mask_and_color_map(
    *,
    mask_path: Path,
    id_to_name: dict[int, str],
) -> tuple[bytes, dict[str, list[str]]]:
    label_map = np.array(Image.open(mask_path))
    if label_map.ndim != 2:
        raise ValueError(f"Expected 2D indexed COCO-Stuff mask, got shape={label_map.shape}")

    rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    color_map: dict[str, list[str]] = {
        "background": ["#000000"],
    }

    unique_ids = sorted(int(x) for x in np.unique(label_map).tolist())

    non_background_ids = [label_id for label_id in unique_ids if label_id not in {0, 255}]
    if len(non_background_ids) > 255:
        raise ValueError(
            f"Semantic mask has too many non-background classes for uint8 indexed output: "
            f"{len(non_background_ids)} > 255 ({mask_path})"
        )

    for label_id in unique_ids:
        if label_id in {0, 255}:
            continue

        class_name = id_to_name.get(label_id)
        if not class_name:
            raise ValueError(f"Unknown COCO-Stuff label id {label_id} in mask: {mask_path}")

        color = color_for_index(label_id)
        hex_color = rgb_to_hex(color)

        rgb[label_map == label_id] = color
        color_map[class_name] = [hex_color]

    if len(color_map) == 1:
        raise ValueError(f"Semantic mask has no non-background classes: {mask_path}")

    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue(), color_map