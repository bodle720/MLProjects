from pathlib import Path
from typing import Any

from dataset_bootstrap.dataset_helpers.common import (
    BootstrapConfig,
    BootstrapFailure,
    BootstrapResult,
    deterministic_sample,
    make_object_detection_row,
    s3_key_join,
    upload_file_to_s3,
)
from dataset_bootstrap.dataset_helpers.coco.common import (
    COCO_SPLIT,
    load_json,
    resolve_instances_json,
    resolve_train_images_dir,
)

def coco_object_detection(config: BootstrapConfig, s3_client: Any) -> BootstrapResult:
    reuse_stats: dict[str, Any] = {
        "reuse_from_run_dir": str(config.reuse_from_run_dir) if config.reuse_from_run_dir else None,
        "reused_train_images": False,
        "reused_train_images_zip": False,
        "reused_annotations_zip": False,
        "reused_instances_json": False,
    }

    images_dir = resolve_train_images_dir(
        config=config,
        reuse_stats=reuse_stats,
    )
    instances_json_path = resolve_instances_json(
        config=config,
        reuse_stats=reuse_stats,
    )

    candidates = _build_object_detection_candidates(
        instances_json_path=instances_json_path,
        images_dir=images_dir,
    )
    selected = deterministic_sample(candidates, config.max_items, config.sample_seed)

    manifest_rows: list[dict[str, Any]] = []
    failures: list[BootstrapFailure] = []

    total = len(selected)
    for idx, record in enumerate(selected, start=1):
        image_id = record["image_id"]
        image_path = record["image_path"]

        if idx % 100 == 0 or idx == 1:
            print(f"On {idx} out of {total}. image_id = {image_id}, file = {image_path.name}")

        try:
            image_s3_key = s3_key_join(
                config.s3_prefix,
                "coco",
                "object-detection",
                "images",
                COCO_SPLIT,
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
                    dataset_item_id=str(image_id),
                    reason=str(exc),
                    context={
                        "image_path": str(image_path),
                    },
                )
            )

    stats = {
        "upstream_dataset": "COCO 2017",
        "task": "object-detection",
        "split": COCO_SPLIT,
        "instances_json_path": str(instances_json_path),
        "images_dir": str(images_dir),
        "discovered_count": len(candidates),
        "selected_count": len(selected),
        **reuse_stats,
    }

    return BootstrapResult(
        manifest_rows=manifest_rows,
        failures=failures,
        stats=stats,
    )

def _build_object_detection_candidates(
    *,
    instances_json_path: Path,
    images_dir: Path,
) -> list[dict[str, Any]]:
    data = load_json(instances_json_path)

    category_id_to_name = {
        int(cat["id"]): cat["name"]
        for cat in data["categories"]
    }

    image_id_to_image = {
        int(img["id"]): img
        for img in data["images"]
    }

    anns_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        anns_by_image_id.setdefault(image_id, []).append(ann)

    candidates: list[dict[str, Any]] = []

    for image_id, image_info in image_id_to_image.items():
        image_path = images_dir / image_info["file_name"]
        if not image_path.is_file():
            continue

        anns = anns_by_image_id.get(image_id, [])
        normalized_anns = _normalize_coco_detection_annotations(
            anns=anns,
            category_id_to_name=category_id_to_name,
        )
        if not normalized_anns:
            continue

        candidates.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "annotations": normalized_anns,
            }
        )

    return candidates

def _normalize_coco_detection_annotations(
    *,
    anns: list[dict[str, Any]],
    category_id_to_name: dict[int, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for ann in anns:
        if int(ann.get("iscrowd", 0)) == 1:
            continue

        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        left, top, width, height = bbox

        try:
            left = float(left)
            top = float(top)
            width = float(width)
            height = float(height)
        except (TypeError, ValueError):
            continue

        if width <= 0 or height <= 0:
            continue

        category_id = int(ann["category_id"])
        class_name = category_id_to_name.get(category_id)
        if not class_name:
            continue

        out.append(
            {
                "class_name": class_name,
                "top": top,
                "left": left,
                "height": height,
                "width": width,
            }
        )

    return out