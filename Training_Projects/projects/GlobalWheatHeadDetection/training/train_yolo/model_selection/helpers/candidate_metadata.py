import re
from pathlib import Path

from helpers import sweep_settings as settings


def _get_nested(data: dict | None, keys: list[str]):
    if not isinstance(data, dict):
        return None

    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current


def _first_not_none(values):
    for value in values:
        if value is not None:
            return value

    return None


def _to_int(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None

        try:
            return int(float(cleaned))
        except ValueError:
            return None

    return None


def _to_float(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None

    return None


def _normalize_model_size(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip().lower()
    if not cleaned:
        return None

    aliases = {
        "nano": "n",
        "small": "s",
        "medium": "m",
        "large": "l",
        "xlarge": "x",
        "extra_large": "x",
        "extra-large": "x",
    }

    return aliases.get(cleaned, cleaned)


def _extract_from_model_token(text: str | None) -> tuple[str | None, str | None]:
    if not text:
        return None, None

    text = str(text).lower().replace("\\", "/")
    stem = Path(text).stem.lower()

    # Examples:
    # yolo11n.pt
    # yolo11s
    # baseline_002_yolo11s_e30_img640_b16_w4
    # yolo26n.pt, yolov8s.pt, etc.
    patterns = [
        r"(yolo\d+)([nsmmlx])",
        r"(yolov\d+)([nsmmlx])",
    ]

    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return match.group(1), _normalize_model_size(match.group(2))

    return None, None


def _extract_imgsz_from_text(text: str | None) -> int | None:
    if not text:
        return None

    match = re.search(r"(?:img|imgsz|image_size)(\d{3,4})", str(text).lower())
    if match:
        return _to_int(match.group(1))

    return None


def _extract_epochs_from_text(text: str | None) -> int | None:
    if not text:
        return None

    match = re.search(r"(?:^|_)e(\d+)(?:_|$)", str(text).lower())
    if match:
        return _to_int(match.group(1))

    match = re.search(r"epochs?(\d+)", str(text).lower())
    if match:
        return _to_int(match.group(1))

    return None


def _extract_batch_from_text(text: str | None) -> int | None:
    if not text:
        return None

    match = re.search(r"(?:^|_)b(\d+)(?:_|$)", str(text).lower())
    if match:
        return _to_int(match.group(1))

    match = re.search(r"batch(\d+)", str(text).lower())
    if match:
        return _to_int(match.group(1))

    return None


def _extract_workers_from_text(text: str | None) -> int | None:
    if not text:
        return None

    match = re.search(r"(?:^|_)w(\d+)(?:_|$)", str(text).lower())
    if match:
        return _to_int(match.group(1))

    match = re.search(r"workers?(\d+)", str(text).lower())
    if match:
        return _to_int(match.group(1))

    return None


def _candidate_text_sources(candidate: dict) -> list[str]:
    metadata = candidate.get("metadata", {})
    args_yaml = metadata.get("args_yaml") or {}
    config_snapshot = metadata.get("config_snapshot") or {}

    sources = [
        candidate.get("run_name"),
        candidate.get("best_pt_artifact_path"),
        candidate.get("best_pt_local_path"),
        args_yaml.get("model"),
        args_yaml.get("name"),
        _get_nested(config_snapshot, ["model", "weights"]),
        _get_nested(config_snapshot, ["model", "family"]),
        _get_nested(config_snapshot, ["training", "run_name"]),
    ]

    return [str(source) for source in sources if source is not None]


def infer_model_family_and_size(candidate: dict) -> tuple[str | None, str | None]:
    metadata = candidate.get("metadata", {})
    args_yaml = metadata.get("args_yaml") or {}
    config_snapshot = metadata.get("config_snapshot") or {}
    mlflow_params = candidate.get("mlflow_params", {}) or {}

    family_candidates = [
        mlflow_params.get("model.family"),
        mlflow_params.get("family"),
        _get_nested(config_snapshot, ["model", "family"]),
    ]

    size_candidates = [
        mlflow_params.get("model.size"),
        mlflow_params.get("size"),
        _get_nested(config_snapshot, ["model", "size"]),
    ]

    family = _first_not_none(family_candidates)
    size = _normalize_model_size(_first_not_none(size_candidates))

    if family is not None and size is not None:
        return str(family).lower(), size

    for source in _candidate_text_sources(candidate):
        parsed_family, parsed_size = _extract_from_model_token(source)

        if family is None and parsed_family is not None:
            family = parsed_family

        if size is None and parsed_size is not None:
            size = parsed_size

        if family is not None and size is not None:
            break

    return str(family).lower() if family is not None else None, size


def infer_training_imgsz(candidate: dict) -> int | None:
    metadata = candidate.get("metadata", {})
    args_yaml = metadata.get("args_yaml") or {}
    config_snapshot = metadata.get("config_snapshot") or {}
    training_summary = metadata.get("training_summary") or {}
    mlflow_params = candidate.get("mlflow_params", {}) or {}

    value = _first_not_none([
        mlflow_params.get("imgsz"),
        mlflow_params.get("training.imgsz"),
        mlflow_params.get("train.imgsz"),
        args_yaml.get("imgsz"),
        _get_nested(config_snapshot, ["training", "imgsz"]),
        _get_nested(training_summary, ["training", "imgsz"]),
        _get_nested(training_summary, ["config", "training", "imgsz"]),
    ])

    parsed = _to_int(value)
    if parsed is not None:
        return parsed

    for source in _candidate_text_sources(candidate):
        parsed = _extract_imgsz_from_text(source)
        if parsed is not None:
            return parsed

    return None


def infer_training_epochs(candidate: dict) -> int | None:
    metadata = candidate.get("metadata", {})
    args_yaml = metadata.get("args_yaml") or {}
    config_snapshot = metadata.get("config_snapshot") or {}
    training_summary = metadata.get("training_summary") or {}
    mlflow_params = candidate.get("mlflow_params", {}) or {}

    value = _first_not_none([
        mlflow_params.get("epochs"),
        mlflow_params.get("training.epochs"),
        args_yaml.get("epochs"),
        _get_nested(config_snapshot, ["training", "epochs"]),
        _get_nested(training_summary, ["training", "epochs"]),
        _get_nested(training_summary, ["config", "training", "epochs"]),
    ])

    parsed = _to_int(value)
    if parsed is not None:
        return parsed

    for source in _candidate_text_sources(candidate):
        parsed = _extract_epochs_from_text(source)
        if parsed is not None:
            return parsed

    return None


def infer_training_batch(candidate: dict) -> int | None:
    metadata = candidate.get("metadata", {})
    args_yaml = metadata.get("args_yaml") or {}
    config_snapshot = metadata.get("config_snapshot") or {}
    training_summary = metadata.get("training_summary") or {}
    mlflow_params = candidate.get("mlflow_params", {}) or {}

    value = _first_not_none([
        mlflow_params.get("batch"),
        mlflow_params.get("training.batch"),
        args_yaml.get("batch"),
        _get_nested(config_snapshot, ["training", "batch"]),
        _get_nested(training_summary, ["training", "batch"]),
        _get_nested(training_summary, ["config", "training", "batch"]),
    ])

    parsed = _to_int(value)
    if parsed is not None:
        return parsed

    for source in _candidate_text_sources(candidate):
        parsed = _extract_batch_from_text(source)
        if parsed is not None:
            return parsed

    return None


def infer_training_workers(candidate: dict) -> int | None:
    metadata = candidate.get("metadata", {})
    args_yaml = metadata.get("args_yaml") or {}
    config_snapshot = metadata.get("config_snapshot") or {}
    training_summary = metadata.get("training_summary") or {}
    mlflow_params = candidate.get("mlflow_params", {}) or {}

    value = _first_not_none([
        mlflow_params.get("workers"),
        mlflow_params.get("training.workers"),
        args_yaml.get("workers"),
        _get_nested(config_snapshot, ["training", "workers"]),
        _get_nested(training_summary, ["training", "workers"]),
        _get_nested(training_summary, ["config", "training", "workers"]),
    ])

    parsed = _to_int(value)
    if parsed is not None:
        return parsed

    for source in _candidate_text_sources(candidate):
        parsed = _extract_workers_from_text(source)
        if parsed is not None:
            return parsed

    return None


def infer_training_metrics(candidate: dict) -> dict:
    mlflow_metrics = candidate.get("mlflow_metrics", {}) or {}

    return {
        "logged_eval_best_val_box_map50_95": _to_float(
            _first_not_none([
                mlflow_metrics.get("eval_best_val.box_map50_95"),
                mlflow_metrics.get("eval_best_val.box.map"),
                mlflow_metrics.get("metrics/mAP50-95B"),
                mlflow_metrics.get("metrics/mAP50-95(B)"),
            ])
        ),
        "logged_eval_best_val_box_map50": _to_float(
            _first_not_none([
                mlflow_metrics.get("eval_best_val.box_map50"),
                mlflow_metrics.get("eval_best_val.box.map50"),
                mlflow_metrics.get("metrics/mAP50B"),
                mlflow_metrics.get("metrics/mAP50(B)"),
            ])
        ),
        "logged_eval_best_test_box_map50_95": _to_float(
            _first_not_none([
                mlflow_metrics.get("eval_best_test.box_map50_95"),
                mlflow_metrics.get("eval_best_test.box.map"),
            ])
        ),
        "logged_eval_best_test_box_map50": _to_float(
            _first_not_none([
                mlflow_metrics.get("eval_best_test.box_map50"),
                mlflow_metrics.get("eval_best_test.box.map50"),
            ])
        ),
    }


def infer_is_lightweight_candidate(model_size: str | None) -> bool:
    normalized = _normalize_model_size(model_size)
    return normalized in settings.LIGHTWEIGHT_MODEL_SIZES


def enrich_candidate_metadata(candidate: dict) -> dict:
    enriched = dict(candidate)

    model_family, model_size = infer_model_family_and_size(candidate)
    training_metrics = infer_training_metrics(candidate)

    enriched.update({
        "model_family": model_family,
        "model_size": model_size,
        "training_imgsz": infer_training_imgsz(candidate),
        "training_epochs": infer_training_epochs(candidate),
        "training_batch": infer_training_batch(candidate),
        "training_workers": infer_training_workers(candidate),
        "is_lightweight_candidate": infer_is_lightweight_candidate(model_size),
    })

    enriched.update(training_metrics)

    return enriched


def enrich_all_candidate_metadata(candidates: list[dict]) -> list[dict]:
    return [enrich_candidate_metadata(candidate) for candidate in candidates]