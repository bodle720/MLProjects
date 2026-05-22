from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


RED = (255, 0, 0)
GREEN = (0, 180, 0)
BLUE = (0, 90, 255)


def draw_evaluation_overlay(
    image_path: Path,
    ground_truth: list[dict[str, Any]],
    matched_predictions: list[dict[str, Any]],
    unmatched_predictions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        line_width = max(2, int(min(image.size) / 300))

        for gt_box in ground_truth:
            draw_box(
                draw=draw,
                bbox=gt_box["bbox_xyxy"],
                color=RED,
                width=line_width,
                label=f"GT {gt_box['class_name']}",
            )

        for prediction in unmatched_predictions:
            draw_box(
                draw=draw,
                bbox=prediction["bbox_xyxy"],
                color=BLUE,
                width=line_width,
                label=f"P {prediction['confidence']:.2f}",
            )

        for prediction in matched_predictions:
            draw_box(
                draw=draw,
                bbox=prediction["bbox_xyxy"],
                color=GREEN,
                width=line_width,
                label=f"M {prediction['confidence']:.2f}",
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=95)


def draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: list[float],
    color: tuple[int, int, int],
    width: int,
    label: str | None = None,
) -> None:
    x_min, y_min, x_max, y_max = bbox
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

    if label:
        font = ImageFont.load_default()
        text_position = (x_min, max(0, y_min - 12))
        draw.text(text_position, label, fill=color, font=font)