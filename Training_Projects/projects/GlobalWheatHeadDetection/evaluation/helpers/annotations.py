from pathlib import Path
from typing import Any

from PIL import Image


def image_path_to_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)

    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")

    return image_path.with_suffix(".txt")


def get_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def load_ground_truth_boxes(
    image_path: Path,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    label_path = image_path_to_label_path(image_path)

    if not label_path.exists():
        return []

    width, height = get_image_size(image_path)
    boxes: list[dict[str, Any]] = []

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()

            if not raw:
                continue

            parts = raw.split()

            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height

            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2
            x_max = x_center + box_width / 2
            y_max = y_center + box_height / 2

            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, str(class_id)),
                    "bbox_xyxy": [x_min, y_min, x_max, y_max],
                    "label_path": str(label_path),
                }
            )

    return boxes


def extract_predictions_from_result(result: Any, class_names: dict[int, str]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []

    if result.boxes is None:
        return predictions

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()

    for bbox, score, class_id in zip(xyxy, conf, cls):
        class_id_int = int(class_id)
        predictions.append(
            {
                "class_id": class_id_int,
                "class_name": class_names.get(class_id_int, str(class_id_int)),
                "confidence": float(score),
                "bbox_xyxy": [float(value) for value in bbox.tolist()],
            }
        )

    return predictions