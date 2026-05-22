import json
from pathlib import Path
from typing import Any


def save_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_text(path: str | Path, content: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(content)


def build_markdown_summary(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    speed = summary["speed_ms_per_image"]
    settings = summary["settings"]

    return f"""# Evaluation Summary

## Run

| Field | Value |
|---|---|
| Split | `{settings["split"]}` |
| Model URI | `{settings["model_uri"]}` |
| MLflow tracking URI | `{settings["mlflow_tracking_uri"]}` |
| Metric confidence | `{settings["metric_conf"]}` |
| Visual confidence | `{settings["visual_conf"]}` |
| Image size | `{settings["imgsz"]}` |
| NMS IoU | `{settings["iou"]}` |
| Max detections | `{settings["max_det"]}` |

## Metrics

| Metric | Value |
|---|---:|
| Precision | {metrics["box_precision"]:.6f} |
| Recall | {metrics["box_recall"]:.6f} |
| mAP50 | {metrics["box_map50"]:.6f} |
| mAP75 | {metrics["box_map75"]:.6f} |
| mAP50-95 | {metrics["box_map50_95"]:.6f} |

## Speed

| Stage | ms / image |
|---|---:|
| Preprocess | {speed["preprocess"]:.3f} |
| Inference | {speed["inference"]:.3f} |
| Loss | {speed["loss"]:.3f} |
| Postprocess | {speed["postprocess"]:.3f} |
| Total pipeline | {speed["total_pipeline"]:.3f} |

## Visualizations

Visualization sample count: `{settings["visualize_sample"]}`  
Visualization strategy: `{settings["visualize_strategy"]}`

Colors:
- Red: ground-truth box
- Green: matched prediction
- Blue: unmatched prediction
"""