import base64
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from dataset_bootstrap.dataset_helpers.common import (
    BootstrapConfig,
    BootstrapFailure,
    BootstrapResult,
    deterministic_sample,
    make_instance_segmentation_row,
    s3_key_join,
    upload_bytes_to_s3,
    upload_file_to_s3,
)
from dataset_bootstrap.dataset_helpers.coco.common import (
    COCO_SPLIT,
    color_for_index,
    load_json,
    resolve_instances_json,
    resolve_train_images_dir,
    rgb_to_hex,
)

from common.general_utils.class_normalizer import canonicalize_class_name

def coco_instance_segmentation(config: BootstrapConfig, s3_client: Any) -> BootstrapResult:
    reuse_stats: dict[str, Any] = {
        "reuse_from_run_dir": str(config.reuse_from_run_dir) if config.reuse_from_run_dir else None,
        "reused_train_images": False,
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

    candidates = _build_instance_candidates(
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
                "instance-segmentation",
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

            worker_payload = _build_instance_worker_response(
                width=record["width"],
                height=record["height"],
                instances=record["instances"],
            )

            worker_s3_key = s3_key_join(
                config.s3_prefix,
                "coco",
                "instance-segmentation",
                "worker-responses",
                COCO_SPLIT,
                f"{image_path.stem}.json",
            )
            worker_response_ref = upload_bytes_to_s3(
                s3_client=s3_client,
                payload=json.dumps(worker_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                bucket=config.bucket,
                key=worker_s3_key,
                content_type="application/json",
            )

            manifest_rows.append(
                make_instance_segmentation_row(
                    source_ref=source_ref,
                    worker_response_ref=worker_response_ref,
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
        "task": "instance-segmentation",
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

def _build_instance_candidates(
    *,
    instances_json_path: Path,
    images_dir: Path,
) -> list[dict[str, Any]]:
    data = load_json(instances_json_path)

    category_id_to_name = {
        int(cat["id"]): canonicalize_class_name(
            cat["name"],
            field_name=f"categories[{cat['id']}].name",
        )
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
        normalized_instances = _normalize_coco_instance_annotations(
            anns=anns,
            category_id_to_name=category_id_to_name,
        )
        if not normalized_instances:
            continue

        candidates.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "width": int(image_info["width"]),
                "height": int(image_info["height"]),
                "instances": normalized_instances,
            }
        )

    return candidates

def _normalize_coco_instance_annotations(
    *,
    anns: list[dict[str, Any]],
    category_id_to_name: dict[int, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for ann in anns:
        # Skip crowd masks / RLE for now; keep this bootstrapper polygon-only.
        if int(ann.get("iscrowd", 0)) == 1:
            continue

        segmentation = ann.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            continue

        polygons: list[list[float]] = []
        for poly in segmentation:
            if not isinstance(poly, list):
                continue
            if len(poly) < 6 or len(poly) % 2 != 0:
                continue

            try:
                poly_floats = [float(x) for x in poly]
            except (TypeError, ValueError):
                continue

            polygons.append(poly_floats)

        if not polygons:
            continue

        category_id = int(ann["category_id"])
        class_name = category_id_to_name.get(category_id)
        if not class_name:
            continue

        try:
            area = float(ann.get("area", 0.0))
        except (TypeError, ValueError):
            area = 0.0

        out.append(
            {
                "class_name": class_name,
                "area": area,
                "polygons": polygons,
            }
        )

    return out

def _build_instance_worker_response(
    *,
    width: int,
    height: int,
    instances: list[dict[str, Any]],
) -> dict[str, Any]:
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Draw larger objects first so smaller objects remain visible on top.
    sorted_instances = sorted(instances, key=lambda x: x.get("area", 0.0), reverse=True)

    worker_instances: list[dict[str, Any]] = []
    for idx, inst in enumerate(sorted_instances, start=1):
        rgb = color_for_index(idx)
        hex_color = rgb_to_hex(rgb)

        for poly in inst["polygons"]:
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(pts, fill=rgb)

        worker_instances.append(
            {
                "color": hex_color,
                "label": inst["class_name"],
            }
        )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    now = _utc_now_z()
    submitted = _utc_now_z(offset_seconds=30)

    return {
        "answers": [
            {
                "acceptanceTime": now,
                "answerContent": {
                    "annotatedResult": {
                        "inputImageProperties": {
                            "height": height,
                            "width": width,
                        },
                        "instances": worker_instances,
                        "labeledImage": {
                            "pngImageData": png_b64,
                        },
                    }
                },
                "submissionTime": submitted,
                "timeSpentInSeconds": 30.0,
                "workerId": "bootstrap.coco.instance_segmentation",
                "workerMetadata": {
                    "identityData": {
                        "identityProviderType": "bootstrap",
                        "issuer": "dataset_bootstrap",
                        "sub": "coco-instance-segmentation",
                    }
                },
            }
        ]
    }

def _utc_now_z(offset_seconds: int = 0) -> str:
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")