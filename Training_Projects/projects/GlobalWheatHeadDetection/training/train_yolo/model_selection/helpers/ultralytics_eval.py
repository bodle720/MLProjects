import time
from pathlib import Path

from ultralytics import YOLO

from helpers import sweep_settings as settings


def _to_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_getattr(obj, attr: str):
    try:
        return getattr(obj, attr)
    except Exception:
        return None


def _count_parameters(yolo_model: YOLO) -> int | None:
    model = _safe_getattr(yolo_model, "model")
    if model is None:
        return None

    try:
        return sum(param.numel() for param in model.parameters())
    except Exception:
        return None


def _count_trainable_parameters(yolo_model: YOLO) -> int | None:
    model = _safe_getattr(yolo_model, "model")
    if model is None:
        return None

    try:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    except Exception:
        return None


def _get_model_stride(yolo_model: YOLO) -> int | None:
    model = _safe_getattr(yolo_model, "model")
    if model is None:
        return None

    stride = _safe_getattr(model, "stride")
    if stride is None:
        return None

    try:
        if hasattr(stride, "max"):
            return int(stride.max())
        return int(max(stride))
    except Exception:
        return None


def _try_get_flops_gflops(yolo_model: YOLO, imgsz: int) -> float | None:
    model = _safe_getattr(yolo_model, "model")
    if model is None:
        return None

    # Ultralytics internals may expose different helpers across versions.
    # Keep this best-effort and never fail the sweep because FLOPs are unavailable.
    try:
        from ultralytics.utils.torch_utils import get_flops

        flops = get_flops(model, imgsz=imgsz)
        return _to_float(flops)
    except Exception:
        pass

    try:
        from ultralytics.utils.torch_utils import get_flops_with_torch_profiler

        flops = get_flops_with_torch_profiler(model, imgsz=imgsz)
        return _to_float(flops)
    except Exception:
        return None


def _extract_box_metrics(metrics) -> dict:
    box = _safe_getattr(metrics, "box")

    if box is None:
        return {
            "box_precision": None,
            "box_recall": None,
            "box_map50": None,
            "box_map50_95": None,
            "box_map75": None,
        }

    return {
        "box_precision": _to_float(_safe_getattr(box, "mp")),
        "box_recall": _to_float(_safe_getattr(box, "mr")),
        "box_map50": _to_float(_safe_getattr(box, "map50")),
        "box_map50_95": _to_float(_safe_getattr(box, "map")),
        "box_map75": _to_float(_safe_getattr(box, "map75")),
    }


def _extract_speed_metrics(metrics) -> dict:
    speed = _safe_getattr(metrics, "speed")

    if not isinstance(speed, dict):
        return {
            "speed_preprocess_ms_per_image": None,
            "speed_inference_ms_per_image": None,
            "speed_loss_ms_per_image": None,
            "speed_postprocess_ms_per_image": None,
        }

    return {
        "speed_preprocess_ms_per_image": _to_float(speed.get("preprocess")),
        "speed_inference_ms_per_image": _to_float(speed.get("inference")),
        "speed_loss_ms_per_image": _to_float(speed.get("loss")),
        "speed_postprocess_ms_per_image": _to_float(speed.get("postprocess")),
    }


def _extract_results_dir(metrics) -> str | None:
    save_dir = _safe_getattr(metrics, "save_dir")
    if save_dir is None:
        return None

    return str(save_dir)


def build_eval_project_dir(output_dir: Path) -> Path:
    project_dir = output_dir / settings.ULTRALYTICS_VAL_RUNS_DIRNAME
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def build_eval_run_name(
    run_name: str,
    imgsz: int,
    iou: float,
    max_det: int,
) -> str:
    safe_iou = str(iou).replace(".", "p")
    return f"{run_name}_img{imgsz}_iou{safe_iou}_maxdet{max_det}"


def load_yolo_model(checkpoint_path: str | Path) -> YOLO:
    return YOLO(str(checkpoint_path))


def get_static_model_metadata(yolo_model: YOLO, imgsz: int) -> dict:
    params = _count_parameters(yolo_model)
    trainable_params = _count_trainable_parameters(yolo_model)
    flops_gflops = _try_get_flops_gflops(yolo_model, imgsz=imgsz)

    return {
        "params": params,
        "params_millions": params / 1_000_000 if params is not None else None,
        "trainable_params": trainable_params,
        "trainable_params_millions": (
            trainable_params / 1_000_000 if trainable_params is not None else None
        ),
        "flops_gflops": flops_gflops,
        "model_stride": _get_model_stride(yolo_model),
    }


def run_validation(
    yolo_model: YOLO,
    data_yaml: Path,
    split: str,
    imgsz: int,
    iou: float,
    max_det: int,
    output_dir: Path,
    run_name: str,
):
    eval_project_dir = build_eval_project_dir(output_dir)
    eval_run_name = build_eval_run_name(
        run_name=run_name,
        imgsz=imgsz,
        iou=iou,
        max_det=max_det,
    )

    started_at = time.perf_counter()

    metrics = yolo_model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=settings.CONF,
        iou=iou,
        max_det=max_det,
        device=settings.DEVICE,
        batch=settings.BATCH,
        workers=settings.WORKERS,
        plots=settings.PLOTS,
        rect=settings.RECT,
        project=str(eval_project_dir),
        name=eval_run_name,
        exist_ok=True,
        verbose=False,
    )

    elapsed_seconds = time.perf_counter() - started_at

    return metrics, elapsed_seconds


def evaluate_candidate_config(
    candidate: dict,
    data_yaml: Path,
    split: str,
    imgsz: int,
    iou: float,
    max_det: int,
    output_dir: Path,
) -> dict:
    checkpoint_path = Path(candidate["best_pt_local_path"])
    yolo_model = load_yolo_model(checkpoint_path)

    static_model_metadata = get_static_model_metadata(yolo_model, imgsz=imgsz)

    metrics, elapsed_seconds = run_validation(
        yolo_model=yolo_model,
        data_yaml=data_yaml,
        split=split,
        imgsz=imgsz,
        iou=iou,
        max_det=max_det,
        output_dir=output_dir,
        run_name=candidate["run_name"],
    )

    result = {
        "run_id": candidate.get("run_id"),
        "run_name": candidate.get("run_name"),
        "model_family": candidate.get("model_family"),
        "model_size": candidate.get("model_size"),
        "is_lightweight_candidate": candidate.get("is_lightweight_candidate"),
        "training_imgsz": candidate.get("training_imgsz"),
        "training_epochs": candidate.get("training_epochs"),
        "training_batch": candidate.get("training_batch"),
        "training_workers": candidate.get("training_workers"),
        "logged_eval_best_val_box_map50_95": candidate.get("logged_eval_best_val_box_map50_95"),
        "logged_eval_best_val_box_map50": candidate.get("logged_eval_best_val_box_map50"),
        "logged_eval_best_test_box_map50_95": candidate.get("logged_eval_best_test_box_map50_95"),
        "logged_eval_best_test_box_map50": candidate.get("logged_eval_best_test_box_map50"),
        "best_pt_artifact_path": candidate.get("best_pt_artifact_path"),
        "best_pt_local_path": candidate.get("best_pt_local_path"),
        "model_file_size_mb": candidate.get("model_file_size_mb"),
        "split": split,
        "imgsz": imgsz,
        "iou": iou,
        "max_det": max_det,
        "conf": settings.CONF,
        "device": settings.DEVICE,
        "batch": settings.BATCH,
        "workers": settings.WORKERS,
        "eval_runtime_seconds": elapsed_seconds,
        "eval_output_dir": _extract_results_dir(metrics),
    }

    result.update(static_model_metadata)
    result.update(_extract_box_metrics(metrics))
    result.update(_extract_speed_metrics(metrics))

    return result


def evaluate_candidate_grid(
    candidate: dict,
    data_yaml: Path,
    split: str,
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    records = []
    failures = []

    for imgsz in settings.IMG_SIZES:
        for iou in settings.IOU_VALUES:
            for max_det in settings.MAX_DET_VALUES:
                try:
                    record = evaluate_candidate_config(
                        candidate=candidate,
                        data_yaml=data_yaml,
                        split=split,
                        imgsz=imgsz,
                        iou=iou,
                        max_det=max_det,
                        output_dir=output_dir,
                    )
                    records.append(record)
                except Exception as exc:
                    failures.append({
                        "run_id": candidate.get("run_id"),
                        "run_name": candidate.get("run_name"),
                        "best_pt_local_path": candidate.get("best_pt_local_path"),
                        "split": split,
                        "imgsz": imgsz,
                        "iou": iou,
                        "max_det": max_det,
                        "error": repr(exc),
                    })

    return records, failures


def evaluate_all_candidate_grids(
    candidates: list[dict],
    data_yaml: Path,
    split: str,
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    all_records = []
    all_failures = []

    for candidate in candidates:
        records, failures = evaluate_candidate_grid(
            candidate=candidate,
            data_yaml=data_yaml,
            split=split,
            output_dir=output_dir,
        )
        all_records.extend(records)
        all_failures.extend(failures)

    return all_records, all_failures