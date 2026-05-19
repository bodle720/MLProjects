import json
import logging
import shutil
from pathlib import Path
from typing import Any

from ultralytics import YOLO, settings

from .config_loader import remove_none_values


def configure_ultralytics(config: dict[str, Any]) -> None:
    mlflow_enabled = bool(config.get("mlflow", {}).get("enabled", False))
    tensorboard_enabled = bool(config.get("tensorboard", {}).get("enabled", True))
    runtime_cfg = config["runtime"]
    model_cfg = config.get("model", {})

    update_values = {
        "mlflow": mlflow_enabled,
        "tensorboard": tensorboard_enabled,
        "runs_dir": runtime_cfg["run_root_dir"],
    }

    weights_dir = model_cfg.get("weights_dir_resolved")
    if weights_dir:
        update_values["weights_dir"] = str(weights_dir)

    settings.update(update_values)

def get_cuda_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "torch_available": False,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "error": str(exc),
        }

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None

    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_device_name": device_name,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "error": None,
    }

def resolve_device(device_value: Any) -> Any:
    if device_value is None:
        return None

    if isinstance(device_value, str):
        normalized = device_value.strip().lower()

        if normalized == "auto":
            cuda_info = get_cuda_info()
            if cuda_info["cuda_available"]:
                return 0
            return "cpu"

        if normalized in {"gpu", "cuda"}:
            return 0

        if normalized == "cpu":
            return "cpu"

        if normalized.isdigit():
            return int(normalized)

    return device_value

def ensure_device_runtime_fields(config: dict[str, Any]) -> Any:
    runtime_cfg = config.setdefault("runtime", {})
    training_cfg = config["training"]

    if "resolved_device" in runtime_cfg:
        return runtime_cfg["resolved_device"]

    requested_device = training_cfg.get("device", "auto")
    resolved_device = resolve_device(requested_device)
    cuda_info = get_cuda_info()

    runtime_cfg["requested_device"] = requested_device
    runtime_cfg["resolved_device"] = resolved_device
    runtime_cfg["cuda_info"] = cuda_info

    return resolved_device

def copy_weights_if_needed(source_path: Path, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    source_resolved = source_path.resolve()
    destination_resolved = destination_path.resolve()

    if source_resolved == destination_resolved:
        return destination_path

    shutil.copy2(source_path, destination_path)
    return destination_path

def try_download_with_ultralytics_downloads(weights_name: str, destination_path: Path) -> Path | None:
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except Exception as exc:
        logging.info("Ultralytics attempt_download_asset is unavailable: %s", exc)
        return None

    try:
        downloaded_path = Path(attempt_download_asset(weights_name))
    except Exception as exc:
        logging.info("Ultralytics attempt_download_asset failed for %s: %s", weights_name, exc)
        return None

    if downloaded_path.exists() and downloaded_path.is_file():
        return copy_weights_if_needed(downloaded_path, destination_path)

    return None

def get_candidate_weight_paths(model: YOLO, weights_name: str) -> list[Path]:
    candidates = []

    ckpt_path = getattr(model, "ckpt_path", None)
    if ckpt_path:
        candidates.append(Path(ckpt_path))

    inner_model = getattr(model, "model", None)
    pt_path = getattr(inner_model, "pt_path", None)
    if pt_path:
        candidates.append(Path(pt_path))

    candidates.extend(
        [
            Path(weights_name),
            Path.cwd() / weights_name,
            Path.home() / ".cache" / "ultralytics" / weights_name,
        ]
    )

    unique_candidates = []
    seen = set()

    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        unique_candidates.append(candidate)

    return unique_candidates

def try_download_with_yolo_constructor(weights_name: str, destination_path: Path) -> Path | None:
    try:
        model = YOLO(weights_name)
    except Exception as exc:
        logging.info("YOLO constructor download/load failed for %s: %s", weights_name, exc)
        return None

    for candidate in get_candidate_weight_paths(model, weights_name):
        if candidate.exists() and candidate.is_file():
            return copy_weights_if_needed(candidate, destination_path)

    if destination_path.exists() and destination_path.is_file():
        return destination_path

    return None

def resolve_weights_for_training(config: dict[str, Any]) -> Path:
    model_cfg = config["model"]

    weights_resolved = Path(model_cfg["weights_resolved"])
    weights_requested = str(model_cfg.get("weights_requested", model_cfg.get("weights", "")))
    weights_name = str(model_cfg.get("weights_name", weights_resolved.name))
    weights_dir = Path(model_cfg.get("weights_dir_resolved", weights_resolved.parent))
    auto_download_weights = bool(model_cfg.get("auto_download_weights", False))
    is_bare_filename = bool(model_cfg.get("weights_is_bare_filename", False))

    if weights_resolved.exists() and weights_resolved.is_file():
        model_cfg["weights_resolved"] = str(weights_resolved)
        model_cfg["weights_source"] = "existing_local_file"
        return weights_resolved

    can_auto_download = (
        auto_download_weights
        and is_bare_filename
        and weights_name.lower().endswith(".pt")
    )

    if not can_auto_download:
        raise FileNotFoundError(
            "YOLO weights file does not exist and auto-download is not enabled for "
            f"this value. Requested weights: {weights_requested}. "
            f"Resolved path: {weights_resolved}"
        )

    weights_dir.mkdir(parents=True, exist_ok=True)
    destination_path = weights_dir / weights_name

    logging.info("YOLO weights not found locally.")
    logging.info("Requested weights: %s", weights_requested)
    logging.info("Downloading/caching weights to: %s", destination_path)

    downloaded_path = try_download_with_ultralytics_downloads(
        weights_name=weights_name,
        destination_path=destination_path,
    )

    if downloaded_path is None:
        downloaded_path = try_download_with_yolo_constructor(
            weights_name=weights_name,
            destination_path=destination_path,
        )

    if downloaded_path is None or not downloaded_path.exists():
        raise FileNotFoundError(
            "Could not auto-download/cache YOLO weights. "
            f"Requested weights: {weights_requested}. "
            f"Expected cache path: {destination_path}"
        )

    model_cfg["weights_resolved"] = str(downloaded_path)
    model_cfg["weights_source"] = "auto_downloaded_or_cached"
    return downloaded_path

def build_train_args(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    runtime_cfg = config["runtime"]

    resolved_device = ensure_device_runtime_fields(config)

    required_args = {
        "data": dataset_cfg["dataset_yaml_resolved"],
        "epochs": training_cfg["epochs"],
        "imgsz": training_cfg["imgsz"],
        "batch": training_cfg["batch"],
        "device": resolved_device,
        "workers": training_cfg["workers"],
        "project": runtime_cfg["experiment_dir"],
        "name": runtime_cfg["resolved_run_name"],
        "exist_ok": True,
        "patience": training_cfg.get("patience", 20),
        "seed": training_cfg.get("seed", 42),
        "deterministic": training_cfg.get("deterministic", True),
        "plots": training_cfg.get("plots", True),
        "cache": training_cfg.get("cache", False),
        "optimizer": training_cfg.get("optimizer", "auto"),
    }

    optional_args = {
        "freeze": training_cfg.get("freeze"),
        "lr0": training_cfg.get("lr0"),
        "lrf": training_cfg.get("lrf"),
        "momentum": training_cfg.get("momentum"),
        "weight_decay": training_cfg.get("weight_decay"),
        "warmup_epochs": training_cfg.get("warmup_epochs"),
        "warmup_momentum": training_cfg.get("warmup_momentum"),
        "warmup_bias_lr": training_cfg.get("warmup_bias_lr"),
        "cos_lr": training_cfg.get("cos_lr"),
        "amp": training_cfg.get("amp"),
        "degrees": training_cfg.get("degrees"),
        "translate": training_cfg.get("translate"),
        "scale": training_cfg.get("scale"),
        "shear": training_cfg.get("shear"),
        "perspective": training_cfg.get("perspective"),
        "fliplr": training_cfg.get("fliplr"),
        "flipud": training_cfg.get("flipud"),
        "mosaic": training_cfg.get("mosaic"),
        "mixup": training_cfg.get("mixup"),
        "copy_paste": training_cfg.get("copy_paste"),
        "close_mosaic": training_cfg.get("close_mosaic"),
        "hsv_h": training_cfg.get("hsv_h"),
        "hsv_s": training_cfg.get("hsv_s"),
        "hsv_v": training_cfg.get("hsv_v"),
    }

    return {
        **required_args,
        **remove_none_values(optional_args),
    }

def get_eval_value(
    eval_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    key: str,
) -> Any:
    value = eval_cfg.get(key)

    if value is not None:
        return value

    return training_cfg[key]

def get_eval_device(
    eval_cfg: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    device_value = eval_cfg.get("device")

    if device_value is not None:
        return resolve_device(device_value)

    return ensure_device_runtime_fields(config)

def iter_enabled_eval_configs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    evaluation_cfg = config.get("evaluation", {})

    if not isinstance(evaluation_cfg, dict):
        return []

    explicit_eval_names = ["val", "test"]
    found_explicit = any(isinstance(evaluation_cfg.get(name), dict) for name in explicit_eval_names)

    if found_explicit:
        enabled = []

        for eval_name in explicit_eval_names:
            eval_cfg = evaluation_cfg.get(eval_name)

            if not isinstance(eval_cfg, dict):
                continue

            if eval_cfg.get("enabled", True):
                enabled.append((eval_name, eval_cfg))

        return enabled

    if evaluation_cfg.get("enabled", True):
        eval_split = evaluation_cfg.get("split", "test")
        eval_name = str(eval_split)
        return [(eval_name, evaluation_cfg)]

    return []

def build_eval_args(
    config: dict[str, Any],
    run_dir: Path,
    eval_name: str,
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    eval_split = eval_cfg.get("split", eval_name)

    required_args = {
        "data": dataset_cfg["dataset_yaml_resolved"],
        "split": eval_split,
        "imgsz": get_eval_value(eval_cfg, training_cfg, "imgsz"),
        "batch": get_eval_value(eval_cfg, training_cfg, "batch"),
        "device": get_eval_device(eval_cfg, config),
        "workers": get_eval_value(eval_cfg, training_cfg, "workers"),
        "project": str(run_dir),
        "name": f"eval_best_{eval_name}",
        "exist_ok": True,
        "plots": eval_cfg.get("plots", True),
    }

    optional_args = {
        "conf": eval_cfg.get("conf"),
        "iou": eval_cfg.get("iou"),
        "max_det": eval_cfg.get("max_det"),
        "save_json": eval_cfg.get("save_json"),
        "save_txt": eval_cfg.get("save_txt"),
        "save_conf": eval_cfg.get("save_conf"),
    }

    return {
        **required_args,
        **remove_none_values(optional_args),
    }

def normalize_metric_name(metric_name: str) -> str:
    normalized = metric_name.strip()

    replacements = {
        "metrics/precision(B)": "box_precision",
        "metrics/recall(B)": "box_recall",
        "metrics/mAP50(B)": "box_map50",
        "metrics/mAP50-95(B)": "box_map50_95",
        "fitness": "fitness",
    }

    if normalized in replacements:
        return replacements[normalized]

    normalized = normalized.replace("metrics/", "")
    normalized = normalized.replace("(B)", "_box")
    normalized = normalized.replace("mAP", "map")
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace("/", "_")
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("(", "")
    normalized = normalized.replace(")", "")
    normalized = normalized.lower()

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")

def extract_results_dict(results: Any) -> dict[str, float]:
    if results is None:
        return {}

    raw = getattr(results, "results_dict", None)

    if raw is None:
        return {}

    metrics = {}

    for key, value in raw.items():
        try:
            metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    return metrics

def extract_normalized_results_dict(results: Any) -> dict[str, float]:
    raw_metrics = extract_results_dict(results)
    normalized_metrics = {}

    for key, value in raw_metrics.items():
        normalized_metrics[normalize_metric_name(key)] = value

    return normalized_metrics

def validate_expected_artifacts(run_dir: Path) -> dict[str, str]:
    best_model_path = run_dir / "weights" / "best.pt"
    last_model_path = run_dir / "weights" / "last.pt"
    results_csv_path = run_dir / "results.csv"

    if not best_model_path.exists():
        raise FileNotFoundError(f"Expected best model not found: {best_model_path}")

    if not last_model_path.exists():
        raise FileNotFoundError(f"Expected last model not found: {last_model_path}")

    if not results_csv_path.exists():
        logging.warning("Expected Ultralytics results.csv not found: %s", results_csv_path)

    return {
        "run_dir": str(run_dir),
        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
        "results_csv_path": str(results_csv_path),
    }

def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)

def copy_config_snapshot(config_path: Path, run_dir: Path) -> Path:
    destination = run_dir / "config_snapshot.yaml"
    shutil.copy2(config_path, destination)
    return destination

def log_summary_to_tensorboard(
    run_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    tensorboard_cfg = config.get("tensorboard", {})

    if not tensorboard_cfg.get("enabled", True):
        return

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        logging.warning("Could not import TensorBoard SummaryWriter: %s", exc)
        return

    writer = SummaryWriter(log_dir=str(run_dir))

    try:
        if tensorboard_cfg.get("log_config_text", True):
            add_training_config_text(writer, config)

        if tensorboard_cfg.get("log_eval_metrics", True):
            add_eval_metrics(writer, config, summary)

    finally:
        writer.flush()
        writer.close()

def add_training_config_text(writer, config: dict[str, Any]) -> None:
    training_cfg = config.get("training", {})
    runtime_cfg = config.get("runtime", {})
    mlflow_cfg = config.get("mlflow", {})
    model_cfg = config.get("model", {})
    cuda_info = runtime_cfg.get("cuda_info", {})

    tracked_keys = [
        "run_name",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "workers",
        "freeze",
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "cos_lr",
        "amp",
        "mosaic",
        "mixup",
        "close_mosaic",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
        "copy_paste",
        "hsv_h",
        "hsv_s",
        "hsv_v",
    ]

    lines = [
        "# YOLO training run",
        "",
        f"- Experiment: `{mlflow_cfg.get('experiment_name')}`",
        f"- Base run name: `{runtime_cfg.get('base_run_name')}`",
        f"- Resolved run name: `{runtime_cfg.get('resolved_run_name')}`",
        f"- Run directory: `{runtime_cfg.get('run_dir')}`",
        f"- Requested weights: `{model_cfg.get('weights_requested', model_cfg.get('weights'))}`",
        f"- Resolved weights: `{model_cfg.get('weights_resolved')}`",
        f"- Weights source: `{model_cfg.get('weights_source')}`",
        f"- Requested device: `{runtime_cfg.get('requested_device')}`",
        f"- Resolved device: `{runtime_cfg.get('resolved_device')}`",
        f"- CUDA available: `{cuda_info.get('cuda_available')}`",
        f"- CUDA device name: `{cuda_info.get('cuda_device_name')}`",
        "",
        "| key | value |",
        "| --- | --- |",
    ]

    for key in tracked_keys:
        lines.append(f"| {key} | {training_cfg.get(key)} |")

    writer.add_text("config/training", "\n".join(lines), 0)

def add_eval_metrics(writer, config: dict[str, Any], summary: dict[str, Any]) -> None:
    training_cfg = config.get("training", {})
    evaluations = summary.get("evaluations", {})
    final_step = int(training_cfg.get("epochs", 0))

    for eval_name, eval_summary in evaluations.items():
        metrics = eval_summary.get("metrics", {})

        for key, value in metrics.items():
            try:
                writer.add_scalar(f"eval_best_{eval_name}/{key}", float(value), final_step)
            except (TypeError, ValueError):
                continue

def run_best_model_evaluations(
    config: dict[str, Any],
    best_model_path: str,
    run_dir: Path,
) -> dict[str, Any]:
    evaluations = {}
    eval_configs = iter_enabled_eval_configs(config)

    if not eval_configs:
        logging.info("No explicit best-model evaluations are enabled.")
        return evaluations

    best_model = YOLO(best_model_path)

    for eval_name, eval_cfg in eval_configs:
        eval_args = build_eval_args(
            config=config,
            run_dir=run_dir,
            eval_name=eval_name,
            eval_cfg=eval_cfg,
        )

        logging.info("Evaluating best model.")
        logging.info("Evaluation name: %s", eval_name)
        logging.info("Evaluation split: %s", eval_args["split"])
        logging.info("Evaluation output name: %s", eval_args["name"])
        logging.info("Evaluation device: %s", eval_args["device"])

        eval_results = best_model.val(**eval_args)
        raw_metrics = extract_results_dict(eval_results)
        normalized_metrics = extract_normalized_results_dict(eval_results)
        eval_save_dir = getattr(eval_results, "save_dir", None)

        evaluations[eval_name] = {
            "enabled": True,
            "split": eval_args["split"],
            "metrics": normalized_metrics,
            "raw_metrics": raw_metrics,
            "save_dir": str(eval_save_dir) if eval_save_dir else None,
            "eval_args": eval_args,
        }

    return evaluations

def run_yolo_training(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    configure_ultralytics(config)

    model_cfg = config["model"]
    runtime_cfg = config["runtime"]

    weights_path = resolve_weights_for_training(config)
    run_dir = Path(runtime_cfg["run_dir"])
    train_args = build_train_args(config)

    logging.info("Starting YOLO training run.")
    logging.info("Experiment: %s", runtime_cfg["experiment_name"])
    logging.info("Base run name: %s", runtime_cfg["base_run_name"])
    logging.info("Resolved run name: %s", runtime_cfg["resolved_run_name"])
    logging.info("Run directory: %s", run_dir)
    logging.info("Requested weights: %s", model_cfg.get("weights_requested", model_cfg.get("weights")))
    logging.info("Resolved weights: %s", weights_path)
    logging.info("Weights source: %s", model_cfg.get("weights_source"))
    logging.info("Dataset: %s", train_args["data"])
    logging.info("Requested device: %s", runtime_cfg.get("requested_device"))
    logging.info("Resolved device: %s", runtime_cfg.get("resolved_device"))
    logging.info("CUDA available: %s", runtime_cfg.get("cuda_info", {}).get("cuda_available"))
    logging.info("CUDA device: %s", runtime_cfg.get("cuda_info", {}).get("cuda_device_name"))

    model = YOLO(str(weights_path))
    model.train(**train_args)

    artifacts = validate_expected_artifacts(run_dir)
    config_snapshot_path = copy_config_snapshot(config_path, run_dir)

    evaluations = run_best_model_evaluations(
        config=config,
        best_model_path=artifacts["best_model_path"],
        run_dir=run_dir,
    )

    summary_path = run_dir / "training_run_summary.json"

    summary = {
        "runtime": runtime_cfg,
        "dataset": {
            "name": config.get("dataset", {}).get("name"),
            "dataset_yaml": config.get("dataset", {}).get("dataset_yaml_resolved"),
            "conversion_report": config.get("dataset", {}).get("conversion_report_resolved"),
            "source_system": config.get("dataset", {}).get("source_system"),
            "cvdms_dataset_id": config.get("dataset", {}).get("cvdms_dataset_id"),
            "cvdms_dataset_version": config.get("dataset", {}).get("cvdms_dataset_version"),
            "label_type": config.get("dataset", {}).get("label_type"),
            "notes": config.get("dataset", {}).get("notes", {}),
        },
        "model": {
            "weights_requested": model_cfg.get("weights_requested", model_cfg.get("weights")),
            "weights_resolved": str(weights_path),
            "weights_source": model_cfg.get("weights_source"),
            "weights_dir": model_cfg.get("weights_dir_resolved"),
            "auto_download_weights": model_cfg.get("auto_download_weights"),
            "family": model_cfg.get("family"),
            "size": model_cfg.get("size"),
        },
        "train_args": train_args,
        "evaluations": evaluations,
        "artifacts": {
            **artifacts,
            "config_snapshot_path": str(config_snapshot_path),
            "summary_path": str(summary_path),
        },
    }

    write_json(summary_path, summary)

    log_summary_to_tensorboard(
        run_dir=run_dir,
        config=config,
        summary=summary,
    )

    logging.info("Training run directory: %s", run_dir)
    logging.info("Best model: %s", artifacts["best_model_path"])
    logging.info("Last model: %s", artifacts["last_model_path"])
    logging.info("Summary: %s", summary_path)

    return summary