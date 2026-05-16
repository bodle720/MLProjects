import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from .config_loader import flatten_dict
from deployment.model_runtime.ultralytics_pyfunc import UltralyticsYoloPyfuncModel

def mlflow_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("mlflow", {}).get("enabled", False))

def configure_mlflow_environment(config: dict[str, Any]) -> None:
    mlflow_cfg = config.get("mlflow", {})
    runtime_cfg = config.get("runtime", {})

    if not mlflow_enabled(config):
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
        os.environ.pop("MLFLOW_RUN", None)
        os.environ.pop("MLFLOW_KEEP_RUN_ACTIVE", None)
        return

    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name")
    resolved_run_name = runtime_cfg.get("resolved_run_name")

    if not tracking_uri:
        raise ValueError("mlflow.tracking_uri is required when mlflow.enabled=true")

    if not experiment_name:
        raise ValueError("mlflow.experiment_name is required when mlflow.enabled=true")

    if not resolved_run_name:
        raise ValueError("runtime.resolved_run_name is required when mlflow.enabled=true")

    os.environ["MLFLOW_TRACKING_URI"] = str(tracking_uri)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = str(experiment_name)
    os.environ["MLFLOW_RUN"] = str(resolved_run_name)

    # Keep the run open after Ultralytics training so this wrapper can add
    # explicit val/test best.pt metrics, best.pt artifacts, summaries, and config snapshots.
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "true"

    mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(str(experiment_name))

@contextmanager
def managed_mlflow_run(
    config: dict[str, Any],
    config_path: Path,
) -> Iterator[str | None]:
    if not mlflow_enabled(config):
        yield None
        return

    configure_mlflow_environment(config)

    runtime_cfg = config["runtime"]
    resolved_run_name = runtime_cfg["resolved_run_name"]

    log_system_metrics = configure_system_metrics(config)

    active_run = start_mlflow_run_compat(
        run_name=resolved_run_name,
        log_system_metrics=log_system_metrics,
    )

    run_id = active_run.info.run_id

    try:
        log_initial_mlflow_metadata(config=config, config_path=config_path)
        yield run_id
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()

def start_mlflow_run_compat(
    run_name: str,
    log_system_metrics: bool,
):
    try:
        return mlflow.start_run(
            run_name=run_name,
            log_system_metrics=log_system_metrics,
        )
    except TypeError:
        return mlflow.start_run(run_name=run_name)

def log_initial_mlflow_metadata(config: dict[str, Any], config_path: Path) -> None:
    mlflow_cfg = config.get("mlflow", {})
    runtime_cfg = config.get("runtime", {})
    cuda_info = runtime_cfg.get("cuda_info", {})
    tags = mlflow_cfg.get("tags", {})

    if isinstance(tags, dict):
        mlflow.set_tags({str(key): str(value) for key, value in tags.items()})

    mlflow.set_tag("run_name", str(runtime_cfg.get("resolved_run_name", "")))
    mlflow.set_tag("base_run_name", str(runtime_cfg.get("base_run_name", "")))
    mlflow.set_tag("experiment_name", str(runtime_cfg.get("experiment_name", "")))
    mlflow.set_tag("experiment_dir", str(runtime_cfg.get("experiment_dir", "")))
    mlflow.set_tag("run_dir", str(runtime_cfg.get("run_dir", "")))

    mlflow.set_tag("project_name", str(config.get("project", {}).get("name", "")))
    mlflow.set_tag("project_display_name", str(config.get("project", {}).get("display_name", "")))
    mlflow.set_tag("dataset_name", str(config.get("dataset", {}).get("name", "")))
    mlflow.set_tag("cvdms_dataset_id", str(config.get("dataset", {}).get("cvdms_dataset_id", "")))
    mlflow.set_tag("cvdms_dataset_version", str(config.get("dataset", {}).get("cvdms_dataset_version", "")))
    mlflow.set_tag("label_type", str(config.get("dataset", {}).get("label_type", "")))
    mlflow.set_tag("model_family", str(config.get("model", {}).get("family", "")))
    mlflow.set_tag("model_size", str(config.get("model", {}).get("size", "")))
    mlflow.set_tag("registered_model_name", str(mlflow_cfg.get("registered_model_name", "")))
    mlflow.set_tag("champion_alias", str(mlflow_cfg.get("champion_alias", "")))
    mlflow.set_tag("candidate_alias", str(mlflow_cfg.get("candidate_alias", "")))

    mlflow.set_tag("requested_device", str(runtime_cfg.get("requested_device", "")))
    mlflow.set_tag("resolved_device", str(runtime_cfg.get("resolved_device", "")))
    mlflow.set_tag("cuda_available", str(cuda_info.get("cuda_available", "")))
    mlflow.set_tag("cuda_device_name", str(cuda_info.get("cuda_device_name", "")))
    mlflow.set_tag("torch_version", str(cuda_info.get("torch_version", "")))
    mlflow.set_tag("torch_cuda_version", str(cuda_info.get("torch_cuda_version", "")))

    system_metrics_cfg = mlflow_cfg.get("system_metrics", {})

    if isinstance(system_metrics_cfg, dict):
        mlflow.set_tag(
            "system_metrics_enabled",
            str(system_metrics_cfg.get("enabled", False)),
        )
        mlflow.set_tag(
            "system_metrics_sampling_interval_seconds",
            str(system_metrics_cfg.get("sampling_interval_seconds", "")),
        )
        mlflow.set_tag(
            "system_metrics_samples_before_logging",
            str(system_metrics_cfg.get("samples_before_logging", "")),
        )

    flattened = flatten_dict(config, prefix="cfg")
    safe_params = {}

    for key, value in flattened.items():
        safe_key = key[:250]
        safe_params[safe_key] = str(value)[:240]

    mlflow.log_params(safe_params)
    mlflow.log_artifact(str(config_path), artifact_path="config")

def log_json_artifact(data: dict[str, Any], artifact_name: str, artifact_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / artifact_name

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, default=str)

        mlflow.log_artifact(str(path), artifact_path=artifact_path)

def log_file_if_exists(path: Path, artifact_path: str) -> None:
    if path.exists() and path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

def log_directory_if_exists(path: Path, artifact_path: str) -> None:
    if path.exists() and path.is_dir():
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)

def clean_mlflow_metric_name(metric_name: str) -> str:
    cleaned = metric_name.strip()
    cleaned = cleaned.replace("(", "")
    cleaned = cleaned.replace(")", "")
    cleaned = cleaned.replace(" ", "_")
    cleaned = cleaned.replace("/", "_")
    cleaned = cleaned.replace("-", "_")

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    return cleaned.strip("_")

def log_final_metrics(metrics: dict[str, float], prefix: str) -> None:
    if not metrics:
        return

    safe_metrics = {}

    for key, value in metrics.items():
        if isinstance(value, bool):
            continue

        try:
            metric_value = float(value)
        except (TypeError, ValueError):
            continue

        safe_key = clean_mlflow_metric_name(f"{prefix}.{key}")
        safe_metrics[safe_key] = metric_value

    if safe_metrics:
        mlflow.log_metrics(safe_metrics)

def log_evaluation_artifacts(run_dir: Path, summary: dict[str, Any]) -> None:
    evaluations = summary.get("evaluations", {})

    if not isinstance(evaluations, dict):
        return

    for eval_name, eval_summary in evaluations.items():
        save_dir = eval_summary.get("save_dir")
        if save_dir:
            log_directory_if_exists(
                Path(save_dir),
                artifact_path=f"evaluations/eval_best_{eval_name}",
            )

    for eval_dir in sorted(run_dir.glob("eval_best_*")):
        log_directory_if_exists(
            eval_dir,
            artifact_path=f"evaluations/{eval_dir.name}",
        )

def log_final_training_artifacts(
    config: dict[str, Any],
    run_dir: Path,
    summary: dict[str, Any],
) -> None:
    if not mlflow_enabled(config):
        return

    log_json_artifact(
        data=summary,
        artifact_name="training_run_summary.json",
        artifact_path="summary",
    )

    best_path = Path(summary["artifacts"]["best_model_path"])
    last_path = Path(summary["artifacts"]["last_model_path"])
    summary_path = Path(summary["artifacts"]["summary_path"])
    config_snapshot_path = Path(summary["artifacts"]["config_snapshot_path"])

    # Explicit stable model artifact locations for later inspection.
    log_file_if_exists(best_path, artifact_path="model/best")
    log_file_if_exists(last_path, artifact_path="model/last")

    # Real MLflow Model for later registration/promotion.
    log_best_model_as_pyfunc(
        config=config,
        best_model_path=best_path,
    )

    log_file_if_exists(summary_path, artifact_path="summary")
    log_file_if_exists(config_snapshot_path, artifact_path="config")

    for path in [
        run_dir / "results.csv",
        run_dir / "args.yaml",
        run_dir / "hyp.yaml",
    ]:
        log_file_if_exists(path, artifact_path="ultralytics")

    dataset_yaml = config.get("dataset", {}).get("dataset_yaml_resolved")
    if dataset_yaml:
        log_file_if_exists(Path(dataset_yaml), artifact_path="dataset")

    conversion_report = config.get("dataset", {}).get("conversion_report_resolved")
    if conversion_report:
        log_file_if_exists(Path(conversion_report), artifact_path="dataset")

    log_evaluation_artifacts(run_dir=run_dir, summary=summary)

    for suffix in ("*.png", "*.jpg", "*.yaml", "*.csv"):
        for path in run_dir.glob(suffix):
            log_file_if_exists(path, artifact_path="ultralytics/run_root")

def configure_system_metrics(config: dict[str, Any]) -> bool:
    mlflow_cfg = config.get("mlflow", {})
    system_metrics_cfg = mlflow_cfg.get("system_metrics", {})

    if not isinstance(system_metrics_cfg, dict):
        return False

    enabled = bool(system_metrics_cfg.get("enabled", False))

    if not enabled:
        try:
            mlflow.disable_system_metrics_logging()
        except AttributeError:
            pass
        return False

    try:
        mlflow.enable_system_metrics_logging()
    except AttributeError:
        pass

    sampling_interval = system_metrics_cfg.get("sampling_interval_seconds")
    samples_before_logging = system_metrics_cfg.get("samples_before_logging")
    node_id = system_metrics_cfg.get("node_id")

    try:
        from mlflow import system_metrics

        if sampling_interval is not None:
            system_metrics.set_system_metrics_sampling_interval(int(sampling_interval))

        if samples_before_logging is not None:
            system_metrics.set_system_metrics_samples_before_logging(
                int(samples_before_logging)
            )

        if node_id:
            system_metrics.set_system_metrics_node_id(str(node_id))

    except Exception:
        pass

    return True

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def build_yolo_pyfunc_signature() -> ModelSignature:
    """Build an explicit MLflow signature for the YOLO pyfunc wrapper.

    The pyfunc model expects a pandas DataFrame with at least:

        image_path

    It returns:

        image_path
        detections_json

    Optional inference controls such as conf/iou/imgsz/device are supported by
    the wrapper as input DataFrame columns, but they are intentionally omitted
    from the strict signature so the minimal serving contract stays simple.
    """

    input_schema = Schema(
        [
            ColSpec("string", "image_path"),
        ]
    )

    output_schema = Schema(
        [
            ColSpec("string", "image_path"),
            ColSpec("string", "detections_json"),
        ]
    )

    return ModelSignature(
        inputs=input_schema,
        outputs=output_schema,
    )

def log_best_model_as_pyfunc(
    config: dict[str, Any],
    best_model_path: Path,
) -> None:
    """Log best.pt as a real MLflow pyfunc model.

    This creates a model artifact such as:

        runs:/<run_id>/best_yolo_model

    Later, the promotion script can register that model and assign aliases such as:

        models:/GlobalWheatHeadDetector@champion
    """

    if not mlflow_enabled(config):
        return

    if not best_model_path.exists():
        raise FileNotFoundError(f"Cannot log missing best model: {best_model_path}")

    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    runtime_cfg = config.get("runtime", {})
    mlflow_cfg = config.get("mlflow", {})

    metadata = {
        "project_name": config.get("project", {}).get("name"),
        "project_display_name": config.get("project", {}).get("display_name"),
        "dataset_name": dataset_cfg.get("name"),
        "cvdms_dataset_id": dataset_cfg.get("cvdms_dataset_id"),
        "cvdms_dataset_version": dataset_cfg.get("cvdms_dataset_version"),
        "label_type": dataset_cfg.get("label_type"),
        "model_family": model_cfg.get("family"),
        "model_size": model_cfg.get("size"),
        "registered_model_name": mlflow_cfg.get("registered_model_name"),
        "champion_alias": mlflow_cfg.get("champion_alias"),
        "candidate_alias": mlflow_cfg.get("candidate_alias"),
        "base_run_name": runtime_cfg.get("base_run_name"),
        "resolved_run_name": runtime_cfg.get("resolved_run_name"),
        "best_model_artifact": "best.pt",
    }

    pip_requirements = [
        "mlflow",
        "ultralytics",
        "torch",
        "torchvision",
        "pandas",
        "pillow",
        "pyyaml",
    ]

    project_root = get_project_root()
    deployment_code_path = project_root / "deployment"

    signature = build_yolo_pyfunc_signature()

    model_info = log_pyfunc_model_compat(
        model_name="best_yolo_model",
        python_model=UltralyticsYoloPyfuncModel(),
        artifacts={
            "weights": str(best_model_path),
        },
        pip_requirements=pip_requirements,
        metadata=metadata,
        code_paths=[
            str(deployment_code_path),
        ],
        signature=signature,
    )

    model_uri = getattr(model_info, "model_uri", None)
    if model_uri is None:
        active_run = mlflow.active_run()
        if active_run is not None:
            model_uri = f"runs:/{active_run.info.run_id}/best_yolo_model"

    mlflow.set_tag("best_mlflow_model_name", "best_yolo_model")
    if model_uri:
        mlflow.set_tag("best_mlflow_model_uri", model_uri)
    mlflow.set_tag("best_weights_artifact_path", "model/best/best.pt")

def log_pyfunc_model_compat(
    model_name: str,
    python_model: Any,
    artifacts: dict[str, str],
    pip_requirements: list[str],
    metadata: dict[str, Any],
    code_paths: list[str] | None = None,
    signature: ModelSignature | None = None,
):
    try:
        return mlflow.pyfunc.log_model(
            name=model_name,
            python_model=python_model,
            artifacts=artifacts,
            pip_requirements=pip_requirements,
            metadata=metadata,
            code_paths=code_paths,
            signature=signature,
        )
    except TypeError:
        return mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=python_model,
            artifacts=artifacts,
            pip_requirements=pip_requirements,
            metadata=metadata,
            code_paths=code_paths,
            signature=signature,
        )