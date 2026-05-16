import copy
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Expected top-level YAML mapping in {config_path}")

    return config

def resolve_project_path(project_root: Path, path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None

    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path

def get_required_section(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = config.get(section_name)

    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid config section: {section_name}")

    return section

def get_required_value(section: dict[str, Any], key: str, section_name: str) -> Any:
    value = section.get(key)

    if value is None:
        raise ValueError(f"Missing required config value: {section_name}.{key}")

    return value

def flatten_dict(
    data: dict[str, Any],
    prefix: str = "",
    max_value_length: int = 240,
) -> dict[str, str]:
    flattened = {}

    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flattened.update(
                flatten_dict(
                    value,
                    prefix=full_key,
                    max_value_length=max_value_length,
                )
            )
        elif isinstance(value, (list, tuple)):
            text = json.dumps(value, default=str)
            flattened[full_key] = text[:max_value_length]
        else:
            text = str(value)
            flattened[full_key] = text[:max_value_length]

    return flattened

def remove_none_values(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}

def is_bare_filename(path_value: str | Path) -> bool:
    path = Path(path_value)
    return not path.is_absolute() and path.parent == Path(".")

def resolve_model_weights_config(
    model_cfg: dict[str, Any],
    project_root: Path,
) -> None:
    weights_value = str(get_required_value(model_cfg, "weights", "model"))
    weights_path = Path(weights_value)

    weights_dir_value = model_cfg.get("weights_dir", "training/train_yolo/weights")
    weights_dir_resolved = resolve_project_path(project_root, weights_dir_value)

    if weights_dir_resolved is None:
        raise ValueError("model.weights_dir could not be resolved")

    auto_download_weights = bool(model_cfg.get("auto_download_weights", False))
    bare_filename = is_bare_filename(weights_value)

    if weights_path.is_absolute():
        weights_resolved = weights_path
    elif bare_filename:
        cached_weights_path = weights_dir_resolved / weights_path.name
        project_root_candidate = project_root / weights_path

        if cached_weights_path.exists():
            weights_resolved = cached_weights_path
        elif project_root_candidate.exists():
            weights_resolved = project_root_candidate
        else:
            weights_resolved = cached_weights_path
    else:
        weights_resolved = project_root / weights_path

    model_cfg["weights_requested"] = weights_value
    model_cfg["weights_name"] = weights_path.name
    model_cfg["weights_is_bare_filename"] = bare_filename
    model_cfg["weights_dir_resolved"] = str(weights_dir_resolved)
    model_cfg["weights_resolved"] = str(weights_resolved)
    model_cfg["auto_download_weights"] = auto_download_weights

def build_resolved_config(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    resolved = copy.deepcopy(config)

    dataset_cfg = get_required_section(resolved, "dataset")
    model_cfg = get_required_section(resolved, "model")
    paths_cfg = get_required_section(resolved, "paths")

    dataset_cfg["dataset_yaml_resolved"] = str(
        resolve_project_path(
            project_root,
            get_required_value(dataset_cfg, "dataset_yaml", "dataset"),
        )
    )

    conversion_report = dataset_cfg.get("conversion_report")
    dataset_cfg["conversion_report_resolved"] = (
        str(resolve_project_path(project_root, conversion_report))
        if conversion_report
        else None
    )

    resolve_model_weights_config(
        model_cfg=model_cfg,
        project_root=project_root,
    )

    paths_cfg["run_root_dir_resolved"] = str(
        resolve_project_path(
            project_root,
            get_required_value(paths_cfg, "run_root_dir", "paths"),
        )
    )

    return resolved

def validate_device_value(value: Any, field_name: str) -> None:
    if value is None:
        return

    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field_name} must be >= 0 when given as an integer GPU index")
        return

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"auto", "cpu", "cuda", "gpu"}:
            return

        if normalized.isdigit():
            return

    raise ValueError(
        f"{field_name} must be one of: auto, cpu, cuda, gpu, 0, 1, ...; got {value!r}"
    )

def validate_numeric_if_present(value: Any, field_name: str) -> None:
    if value is None:
        return

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return

    raise ValueError(f"{field_name} must be numeric or null; got {value!r}")

def validate_bool_if_present(value: Any, field_name: str) -> None:
    if value is None:
        return

    if isinstance(value, bool):
        return

    raise ValueError(f"{field_name} must be true, false, or null; got {value!r}")

def validate_model_section(model_cfg: dict[str, Any]) -> None:
    weights_path = Path(get_required_value(model_cfg, "weights_resolved", "model"))
    weights_requested = str(get_required_value(model_cfg, "weights_requested", "model"))
    weights_name = str(get_required_value(model_cfg, "weights_name", "model"))
    weights_dir = Path(get_required_value(model_cfg, "weights_dir_resolved", "model"))
    auto_download_weights = bool(model_cfg.get("auto_download_weights", False))
    is_bare = bool(model_cfg.get("weights_is_bare_filename", False))

    validate_bool_if_present(
        model_cfg.get("auto_download_weights"),
        "model.auto_download_weights",
    )

    if weights_path.exists():
        return

    can_auto_download = (
        auto_download_weights
        and is_bare
        and weights_name.lower().endswith(".pt")
    )

    if can_auto_download:
        weights_dir.mkdir(parents=True, exist_ok=True)
        return

    raise FileNotFoundError(
        "YOLO weights file does not exist and cannot be auto-downloaded with the "
        "current config. Either provide an existing local path or use a bare "
        f"Ultralytics weight name such as yolo11n.pt with auto_download_weights=true. "
        f"Requested weights: {weights_requested}. Resolved path: {weights_path}"
    )

def validate_training_section(training_cfg: dict[str, Any]) -> None:
    required_keys = ["run_name", "epochs", "imgsz", "batch", "device", "workers"]

    for key in required_keys:
        if key not in training_cfg:
            raise ValueError(f"training.{key} is required")

    if not training_cfg.get("run_name"):
        raise ValueError("training.run_name is required")

    for key in ["epochs", "imgsz", "batch", "workers"]:
        value = training_cfg.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"training.{key} must be a non-negative integer")

    if training_cfg["epochs"] <= 0:
        raise ValueError("training.epochs must be > 0")

    if training_cfg["imgsz"] <= 0:
        raise ValueError("training.imgsz must be > 0")

    if training_cfg["batch"] <= 0:
        raise ValueError("training.batch must be > 0")

    validate_device_value(training_cfg.get("device"), "training.device")

    validate_bool_if_present(training_cfg.get("deterministic"), "training.deterministic")
    validate_bool_if_present(training_cfg.get("plots"), "training.plots")
    validate_bool_if_present(training_cfg.get("cos_lr"), "training.cos_lr")
    validate_bool_if_present(training_cfg.get("amp"), "training.amp")

    numeric_or_null_keys = [
        "freeze",
        "patience",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
        "mosaic",
        "mixup",
        "close_mosaic",
    ]

    for key in numeric_or_null_keys:
        validate_numeric_if_present(training_cfg.get(key), f"training.{key}")

def validate_eval_block(eval_name: str, eval_cfg: dict[str, Any]) -> None:
    allowed_keys = {
        "enabled",
        "split",
        "plots",
        "imgsz",
        "batch",
        "device",
        "workers",
        "conf",
        "iou",
        "max_det",
        "save_json",
        "save_txt",
        "save_conf",
    }

    unknown_keys = sorted(set(eval_cfg) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"evaluation.{eval_name} contains unknown keys: {unknown_keys}")

    validate_bool_if_present(eval_cfg.get("enabled"), f"evaluation.{eval_name}.enabled")
    validate_bool_if_present(eval_cfg.get("plots"), f"evaluation.{eval_name}.plots")
    validate_bool_if_present(eval_cfg.get("save_json"), f"evaluation.{eval_name}.save_json")
    validate_bool_if_present(eval_cfg.get("save_txt"), f"evaluation.{eval_name}.save_txt")
    validate_bool_if_present(eval_cfg.get("save_conf"), f"evaluation.{eval_name}.save_conf")

    split = eval_cfg.get("split", eval_name)
    if not isinstance(split, str) or not split.strip():
        raise ValueError(f"evaluation.{eval_name}.split must be a non-empty string")

    validate_device_value(eval_cfg.get("device"), f"evaluation.{eval_name}.device")

    for key in ["imgsz", "batch", "workers", "conf", "iou", "max_det"]:
        validate_numeric_if_present(eval_cfg.get(key), f"evaluation.{eval_name}.{key}")

def validate_evaluation_section(config: dict[str, Any]) -> None:
    evaluation_cfg = config.get("evaluation", {})

    if evaluation_cfg is None:
        return

    if not isinstance(evaluation_cfg, dict):
        raise ValueError("evaluation must be a mapping if provided")

    nested_eval_names = [
        name
        for name in ("val", "test")
        if isinstance(evaluation_cfg.get(name), dict)
    ]

    if nested_eval_names:
        for eval_name in nested_eval_names:
            validate_eval_block(eval_name, evaluation_cfg[eval_name])
        return

    validate_eval_block("single", evaluation_cfg)

def validate_training_inputs(config: dict[str, Any]) -> None:
    dataset_cfg = get_required_section(config, "dataset")
    model_cfg = get_required_section(config, "model")
    paths_cfg = get_required_section(config, "paths")
    training_cfg = get_required_section(config, "training")

    dataset_yaml = Path(get_required_value(dataset_cfg, "dataset_yaml_resolved", "dataset"))
    run_root_dir = Path(get_required_value(paths_cfg, "run_root_dir_resolved", "paths"))

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"YOLO dataset YAML does not exist: {dataset_yaml}")

    validate_model_section(model_cfg)
    validate_training_section(training_cfg)
    validate_evaluation_section(config)

    run_root_dir.mkdir(parents=True, exist_ok=True)

    conversion_report = dataset_cfg.get("conversion_report_resolved")
    if conversion_report and not Path(conversion_report).exists():
        raise FileNotFoundError(
            f"Configured conversion report does not exist: {conversion_report}"
        )

    mlflow_cfg = config.get("mlflow", {})
    if mlflow_cfg.get("enabled", False):
        if not mlflow_cfg.get("tracking_uri"):
            raise ValueError("mlflow.tracking_uri is required when mlflow.enabled=true")
        if not mlflow_cfg.get("experiment_name"):
            raise ValueError("mlflow.experiment_name is required when mlflow.enabled=true")

def load_training_config(config_path: Path, project_root: Path) -> dict[str, Any]:
    raw_config = load_yaml_config(config_path)
    resolved_config = build_resolved_config(raw_config, project_root)
    validate_training_inputs(resolved_config)
    return resolved_config