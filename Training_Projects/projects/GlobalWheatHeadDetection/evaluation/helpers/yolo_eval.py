import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from evaluation.config import EvalConfig
from evaluation.helpers.paths import resolve_project_path


def load_yolo_model(weights_path: str | Path) -> YOLO:
    return YOLO(str(weights_path))


def run_full_split_eval(
    model: YOLO,
    config: EvalConfig,
    run_dir: Path,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    metrics = model.val(
        data=str(resolve_project_path(config.data_yaml)),
        split=config.split,
        conf=config.metric_conf,
        iou=config.iou,
        imgsz=config.imgsz,
        max_det=config.max_det,
        batch=config.batch,
        workers=config.workers,
        device=config.device,
        project=str(run_dir),
        name="yolo_eval",
        exist_ok=True,
        plots=config.plots,
        rect=config.rect,
        verbose=True,
    )

    runtime_seconds = time.perf_counter() - start_time
    metric_payload = extract_metric_payload(metrics)

    metric_payload["eval_runtime_seconds"] = runtime_seconds
    metric_payload["eval_output_dir"] = str(Path(metrics.save_dir))

    return metric_payload


def extract_metric_payload(metrics: Any) -> dict[str, Any]:
    box = metrics.box
    speed = getattr(metrics, "speed", {}) or {}

    preprocess = float(speed.get("preprocess", 0.0))
    inference = float(speed.get("inference", 0.0))
    loss = float(speed.get("loss", 0.0))
    postprocess = float(speed.get("postprocess", 0.0))

    return {
        "metrics": {
            "box_precision": to_float(getattr(box, "mp", None)),
            "box_recall": to_float(getattr(box, "mr", None)),
            "box_map50": to_float(getattr(box, "map50", None)),
            "box_map75": to_float(getattr(box, "map75", None)),
            "box_map50_95": to_float(getattr(box, "map", None)),
        },
        "speed_ms_per_image": {
            "preprocess": preprocess,
            "inference": inference,
            "loss": loss,
            "postprocess": postprocess,
            "total_pipeline": preprocess + inference + loss + postprocess,
        },
    }


def to_float(value: Any) -> float:
    if value is None:
        return 0.0

    if hasattr(value, "mean"):
        return float(value.mean())

    return float(value)