# To run:
#  python training/train_yolo/model_selection/select_best_candidate.py --sweep-dir _model_select/sweeps/global_wheat_head_detection/20260519_201048_val --data-yaml training/data/yolo/global-wheat-head-2021-v1/dataset.yaml

import argparse
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from helpers import sweep_settings as settings


# ---------------------------------------------------------------------
# Selection behavior
# ---------------------------------------------------------------------

SELECTION_SOURCE_FILENAME = "best_overall_val.json"
SELECTION_NAME = "primary"
REGISTERED_MODEL_NAME = "GlobalWheatHeadDetector"
MODEL_ALIAS = "champion"
MODEL_ARTIFACT_PATH = "final_model"

# Test evaluation uses the validation-selected inference settings, but it
# should keep a low confidence threshold so AP/mAP can be computed properly.
TEST_SPLIT = "test"
EVAL_CONF = 0.001
EVAL_BATCH = 4
EVAL_WORKERS = 0
EVAL_DEVICE = 0
EVAL_PLOTS = False

# Serving defaults are intentionally different from AP/mAP eval defaults.
# conf=0.001 is useful for metric computation, but too noisy for a demo API.
SERVING_CONF = 0.25
SERVING_DEVICE = None

SELECTED_OUTPUT_ROOT = PROJECT_ROOT / "_model_select" / "selected"
PYFUNC_WRAPPER_PATH = PROJECT_ROOT / "deployment" / "model_runtime" / "ultralytics_pyfunc.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package the validation-selected YOLO checkpoint as a fresh "
            "deployment-ready MLflow pyfunc model, run final test evaluation, "
            "and assign the champion registry alias."
        )
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=None,
        help=(
            "Path to the validation sweep output directory. If omitted, the "
            "script tries to resolve the latest sweep directory automatically."
        ),
    )
    parser.add_argument(
        "--data-yaml",
        required=True,
        type=Path,
        help="Path to the Ultralytics dataset.yaml file containing the test split.",
    )
    return parser.parse_args()


def resolve_sweep_dir(sweep_dir: Path | None) -> Path:
    if sweep_dir is not None:
        resolved = sweep_dir.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Sweep directory does not exist: {resolved}")
        return resolved

    candidates = list((PROJECT_ROOT / "_model_select" / "sweeps").rglob("latest_sweep_dir.txt"))
    if candidates:
        latest_file = max(candidates, key=lambda path: path.stat().st_mtime)
        raw_path = latest_file.read_text(encoding="utf-8").strip()
        resolved = Path(raw_path).expanduser().resolve()
        if resolved.exists():
            return resolved

    sweep_dirs = [
        path for path in (PROJECT_ROOT / "_model_select" / "sweeps").rglob("*_val")
        if path.is_dir()
    ]
    if sweep_dirs:
        return max(sweep_dirs, key=lambda path: path.stat().st_mtime).resolve()

    raise FileNotFoundError(
        "Could not resolve latest sweep directory. Pass --sweep-dir explicitly."
    )


def validate_data_yaml(data_yaml: Path) -> Path:
    resolved = data_yaml.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Dataset YAML path is not a file: {resolved}")
    return resolved


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, indent=2)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {"-", "_", " ", ".", "/"}:
            cleaned.append("_")

    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")

    return slug.strip("_") or "unnamed"


def load_selected_candidate(sweep_dir: Path) -> dict:
    candidate_path = sweep_dir / SELECTION_SOURCE_FILENAME
    if not candidate_path.exists():
        raise FileNotFoundError(f"Could not find selected candidate file: {candidate_path}")

    candidate = read_json(candidate_path)
    weights_path = Path(candidate["best_pt_local_path"]).expanduser().resolve()

    if not weights_path.exists():
        raise FileNotFoundError(
            "Selected best.pt file does not exist. "
            f"Path from {SELECTION_SOURCE_FILENAME}: {weights_path}"
        )

    candidate["best_pt_local_path"] = str(weights_path)
    return candidate


def load_sweep_config(sweep_dir: Path) -> dict:
    sweep_config_path = sweep_dir / "sweep_config.json"
    if not sweep_config_path.exists():
        return {}
    return read_json(sweep_config_path)


def create_selection_output_dir(candidate: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id_short = str(candidate.get("run_id", "selected"))[:8]

    output_dir = SELECTED_OUTPUT_ROOT / f"{timestamp}_{SELECTION_NAME}_{run_id_short}"
    output_dir.mkdir(parents=True, exist_ok=False)

    latest_path = SELECTED_OUTPUT_ROOT / "latest_selected_dir.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(output_dir), encoding="utf-8")

    return output_dir


def build_eval_config(candidate: dict) -> dict:
    return {
        "split": TEST_SPLIT,
        "conf": EVAL_CONF,
        "iou": float(candidate["iou"]),
        "imgsz": int(candidate["imgsz"]),
        "max_det": int(candidate["max_det"]),
        "batch": EVAL_BATCH,
        "workers": EVAL_WORKERS,
        "device": EVAL_DEVICE,
        "plots": EVAL_PLOTS,
    }


def build_serving_config(candidate: dict) -> dict:
    return {
        "conf": SERVING_CONF,
        "iou": float(candidate["iou"]),
        "imgsz": int(candidate["imgsz"]),
        "max_det": int(candidate["max_det"]),
        "device": "0" if SERVING_DEVICE is None else str(SERVING_DEVICE),
    }

def package_selected_checkpoint(candidate: dict, output_dir: Path) -> Path:
    source_path = Path(candidate["best_pt_local_path"])
    package_weights_dir = output_dir / "weights"
    package_weights_dir.mkdir(parents=True, exist_ok=True)

    packaged_path = package_weights_dir / "best.pt"
    shutil.copy2(source_path, packaged_path)

    return packaged_path


def run_test_evaluation(
    weights_path: Path,
    candidate: dict,
    data_yaml: Path,
    eval_config: dict,
    output_dir: Path,
) -> dict:
    from ultralytics import YOLO

    eval_runs_dir = output_dir / "test"
    eval_runs_dir.mkdir(parents=True, exist_ok=True)

    run_id_short = str(candidate.get("run_id", "selected"))[:8]
    eval_run_name = (
        f"{run_id_short}"
        f"_i{eval_config['imgsz']}"
        f"_u{str(eval_config['iou']).replace('.', 'p')}"
        f"_m{eval_config['max_det']}"
    )

    model = YOLO(str(weights_path))

    val_kwargs = {
        "data": str(data_yaml),
        "split": eval_config["split"],
        "conf": eval_config["conf"],
        "iou": eval_config["iou"],
        "imgsz": eval_config["imgsz"],
        "max_det": eval_config["max_det"],
        "batch": eval_config["batch"],
        "workers": eval_config["workers"],
        "plots": eval_config["plots"],
        "project": str(eval_runs_dir),
        "name": eval_run_name,
        "exist_ok": True,
        "verbose": False,
    }

    if eval_config["device"] is not None:
        val_kwargs["device"] = eval_config["device"]

    start_time = time.perf_counter()
    metrics = model.val(**val_kwargs)
    elapsed_seconds = time.perf_counter() - start_time

    result = extract_ultralytics_metrics(metrics)
    result.update(
        {
            "split": eval_config["split"],
            "conf": eval_config["conf"],
            "iou": eval_config["iou"],
            "imgsz": eval_config["imgsz"],
            "max_det": eval_config["max_det"],
            "batch": eval_config["batch"],
            "workers": eval_config["workers"],
            "device": eval_config["device"],
            "eval_runtime_seconds": elapsed_seconds,
            "eval_output_dir": str(eval_runs_dir / eval_run_name),
        }
    )

    return result


def extract_ultralytics_metrics(metrics: Any) -> dict:
    box = getattr(metrics, "box", None)

    result = {
        "box_precision": get_float_attr(box, "mp"),
        "box_recall": get_float_attr(box, "mr"),
        "box_map50": get_float_attr(box, "map50"),
        "box_map75": get_float_attr(box, "map75"),
        "box_map50_95": get_float_attr(box, "map"),
    }

    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        preprocess = to_float_or_none(speed.get("preprocess"))
        inference = to_float_or_none(speed.get("inference"))
        loss = to_float_or_none(speed.get("loss"))
        postprocess = to_float_or_none(speed.get("postprocess"))

        total_pipeline = None
        if preprocess is not None and inference is not None and postprocess is not None:
            total_pipeline = preprocess + inference + postprocess

        result.update(
            {
                "speed_preprocess_ms_per_image": preprocess,
                "speed_inference_ms_per_image": inference,
                "speed_loss_ms_per_image": loss,
                "speed_postprocess_ms_per_image": postprocess,
                "speed_total_inference_pipeline_ms_per_image": total_pipeline,
            }
        )
    else:
        result.update(
            {
                "speed_preprocess_ms_per_image": None,
                "speed_inference_ms_per_image": None,
                "speed_loss_ms_per_image": None,
                "speed_postprocess_ms_per_image": None,
                "speed_total_inference_pipeline_ms_per_image": None,
            }
        )

    return result


def get_float_attr(obj: Any, attr_name: str) -> float | None:
    if obj is None:
        return None

    value = getattr(obj, attr_name, None)
    return to_float_or_none(value)


def to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_pyfunc_model_class():
    if not PYFUNC_WRAPPER_PATH.exists():
        raise FileNotFoundError(f"Could not find pyfunc wrapper: {PYFUNC_WRAPPER_PATH}")

    spec = importlib.util.spec_from_file_location(
        "selected_ultralytics_pyfunc",
        PYFUNC_WRAPPER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pyfunc wrapper spec from: {PYFUNC_WRAPPER_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.UltralyticsYoloPyfuncModel


def log_selected_model_to_mlflow(
    candidate: dict,
    sweep_config: dict,
    output_dir: Path,
    packaged_weights_path: Path,
    eval_config: dict,
    serving_config: dict,
    test_metrics: dict,
) -> dict:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    experiment_name = sweep_config.get("experiment_name") or candidate.get("experiment_name")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    run_name = f"select_{SELECTION_NAME}_{slugify(candidate.get('run_name', 'model'))}"

    UltralyticsYoloPyfuncModel = load_pyfunc_model_class()
    client = MlflowClient()

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        log_selection_params(candidate, eval_config, serving_config)
        log_metric_group("selected_val", candidate)
        log_metric_group("final_test", test_metrics)

        mlflow.set_tag("stage", "model_selection")
        mlflow.set_tag("selection_name", SELECTION_NAME)
        mlflow.set_tag("selection_source", SELECTION_SOURCE_FILENAME)
        mlflow.set_tag("selected_training_run_id", candidate.get("run_id"))
        mlflow.set_tag("selected_training_run_name", candidate.get("run_name"))
        mlflow.set_tag("registered_model_name", REGISTERED_MODEL_NAME)
        mlflow.set_tag("model_alias", MODEL_ALIAS)

        mlflow.log_artifacts(str(output_dir), artifact_path="selection_package")

        input_example = pd.DataFrame(
            [
                {
                    "image_path": "example.png",
                }
            ]
        )

        output_example = pd.DataFrame(
            [
                {
                    "image_path": "example.png",
                    "detections_json": "[]",
                }
            ]
        )

        params_example = {
            "conf": serving_config["conf"],
            "iou": serving_config["iou"],
            "imgsz": serving_config["imgsz"],
            "max_det": serving_config["max_det"],
            "device": serving_config["device"],
        }

        signature = infer_signature(
            model_input=input_example,
            model_output=output_example,
            params=params_example,
        )

        model_info = mlflow.pyfunc.log_model(
            artifact_path=MODEL_ARTIFACT_PATH,
            python_model=UltralyticsYoloPyfuncModel(),
            artifacts={"weights": str(packaged_weights_path)},
            code_paths=[str(PYFUNC_WRAPPER_PATH)],
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        model_version = resolve_registered_model_version(
            client=client,
            model_info=model_info,
            registered_model_name=REGISTERED_MODEL_NAME,
            run_id=run_id,
        )

        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=MODEL_ALIAS,
            version=str(model_version),
        )

        mlflow_result = {
            "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
            "mlflow_run_id": run_id,
            "mlflow_run_name": run_name,
            "logged_model_uri": model_info.model_uri,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "registered_model_version": str(model_version),
            "registered_model_alias": MODEL_ALIAS,
        }

        write_json(output_dir / "mlflow_registration.json", mlflow_result)

    return mlflow_result


def log_selection_params(candidate: dict, eval_config: dict, serving_config: dict) -> None:
    params = {
        "selection_name": SELECTION_NAME,
        "selection_source": SELECTION_SOURCE_FILENAME,
        "selected_run_name": candidate.get("run_name"),
        "selected_run_id": candidate.get("run_id"),
        "selected_model_family": candidate.get("model_family"),
        "selected_model_size": candidate.get("model_size"),
        "selected_training_imgsz": candidate.get("training_imgsz"),
        "selected_training_epochs": candidate.get("training_epochs"),
        "selected_val_imgsz": candidate.get("imgsz"),
        "selected_val_iou": candidate.get("iou"),
        "selected_val_max_det": candidate.get("max_det"),
        "eval_conf": eval_config["conf"],
        "eval_iou": eval_config["iou"],
        "eval_imgsz": eval_config["imgsz"],
        "eval_max_det": eval_config["max_det"],
        "serving_conf": serving_config["conf"],
        "serving_iou": serving_config["iou"],
        "serving_imgsz": serving_config["imgsz"],
        "serving_max_det": serving_config["max_det"],
        "serving_device": serving_config["device"],
    }

    mlflow.log_params({key: value for key, value in params.items() if value is not None})


def log_metric_group(prefix: str, record: dict) -> None:
    metric_keys = [
        "box_precision",
        "box_recall",
        "box_map50",
        "box_map75",
        "box_map50_95",
        "speed_preprocess_ms_per_image",
        "speed_inference_ms_per_image",
        "speed_loss_ms_per_image",
        "speed_postprocess_ms_per_image",
        "speed_total_inference_pipeline_ms_per_image",
        "eval_runtime_seconds",
        "flops_gflops",
        "params_millions",
        "model_file_size_mb",
    ]

    for key in metric_keys:
        value = to_float_or_none(record.get(key))
        if value is not None:
            mlflow.log_metric(f"{prefix}.{key}", value)


def resolve_registered_model_version(
    client: MlflowClient,
    model_info: Any,
    registered_model_name: str,
    run_id: str,
) -> str:
    direct_version = getattr(model_info, "registered_model_version", None)
    if direct_version:
        return str(direct_version)

    versions = client.search_model_versions(f"name='{registered_model_name}'")
    matching_versions = [
        version for version in versions
        if getattr(version, "run_id", None) == run_id
    ]

    if not matching_versions:
        raise RuntimeError(
            "Could not find registered model version for the current run. "
            f"Registered model: {registered_model_name}, run_id: {run_id}"
        )

    latest_version = max(matching_versions, key=lambda version: int(version.version))
    return str(latest_version.version)


def write_selection_package(
    output_dir: Path,
    sweep_dir: Path,
    data_yaml: Path,
    candidate: dict,
    sweep_config: dict,
    packaged_weights_path: Path,
    eval_config: dict,
    serving_config: dict,
    test_metrics: dict,
) -> None:
    selection_summary = {
        "selection_name": SELECTION_NAME,
        "selection_source_file": SELECTION_SOURCE_FILENAME,
        "sweep_dir": str(sweep_dir),
        "data_yaml": str(data_yaml),
        "selected_training_run_name": candidate.get("run_name"),
        "selected_training_run_id": candidate.get("run_id"),
        "selected_weights_path": str(packaged_weights_path),
        "selection_rule": "Highest validation box_map50_95 from inference-config sweep.",
        "selected_by_split": "val",
        "reported_final_split": TEST_SPLIT,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "registered_model_alias": MODEL_ALIAS,
    }

    write_json(output_dir / "selected_candidate_val.json", candidate)
    write_json(output_dir / "sweep_config.json", sweep_config)
    write_json(output_dir / "eval_config.json", eval_config)
    write_json(output_dir / "serving_config.json", serving_config)
    write_json(output_dir / "test_metrics.json", test_metrics)
    write_json(output_dir / "selection_summary.json", selection_summary)


def print_summary(
    output_dir: Path,
    candidate: dict,
    eval_config: dict,
    serving_config: dict,
    test_metrics: dict,
    mlflow_result: dict,
) -> None:
    print()
    print("Selected model packaged and registered")
    print("--------------------------------------")
    print(f"Selected run:       {candidate.get('run_name')}")
    print(f"Training run id:    {candidate.get('run_id')}")
    print(f"Validation mAP50-95:{candidate.get('box_map50_95')}")
    print(f"Test mAP50-95:      {test_metrics.get('box_map50_95')}")
    print(f"Test mAP50:         {test_metrics.get('box_map50')}")
    print(f"Test precision:     {test_metrics.get('box_precision')}")
    print(f"Test recall:        {test_metrics.get('box_recall')}")
    print(f"Eval config:        imgsz={eval_config['imgsz']}, iou={eval_config['iou']}, max_det={eval_config['max_det']}")
    print(
        f"Serving config:     conf={serving_config['conf']}, "
        f"imgsz={serving_config['imgsz']}, "
        f"iou={serving_config['iou']}, "
        f"max_det={serving_config['max_det']}, "
        f"device={serving_config['device']}"
    )
    print(f"Registered model:   {mlflow_result['registered_model_name']}")
    print(f"Version:            {mlflow_result['registered_model_version']}")
    print(f"Alias:              {mlflow_result['registered_model_alias']}")
    print(f"Output dir:         {output_dir}")
    print()


def run_selection(sweep_dir: Path | None, data_yaml: Path) -> Path:
    sweep_dir = resolve_sweep_dir(sweep_dir)
    data_yaml = validate_data_yaml(data_yaml)

    candidate = load_selected_candidate(sweep_dir)
    sweep_config = load_sweep_config(sweep_dir)

    output_dir = create_selection_output_dir(candidate)
    packaged_weights_path = package_selected_checkpoint(candidate, output_dir)

    eval_config = build_eval_config(candidate)
    serving_config = build_serving_config(candidate)

    print()
    print("Finalizing validation-selected model")
    print("------------------------------------")
    print(f"Sweep dir:          {sweep_dir}")
    print(f"Data YAML:          {data_yaml}")
    print(f"Selected run:       {candidate.get('run_name')}")
    print(f"Selected checkpoint:{packaged_weights_path}")
    print(f"Eval split:         {TEST_SPLIT}")
    print(f"Eval config:        imgsz={eval_config['imgsz']}, iou={eval_config['iou']}, max_det={eval_config['max_det']}")
    print()

    print("Running final held-out test evaluation...")
    test_metrics = run_test_evaluation(
        weights_path=packaged_weights_path,
        candidate=candidate,
        data_yaml=data_yaml,
        eval_config=eval_config,
        output_dir=output_dir,
    )

    write_selection_package(
        output_dir=output_dir,
        sweep_dir=sweep_dir,
        data_yaml=data_yaml,
        candidate=candidate,
        sweep_config=sweep_config,
        packaged_weights_path=packaged_weights_path,
        eval_config=eval_config,
        serving_config=serving_config,
        test_metrics=test_metrics,
    )

    print("Logging fresh deployment pyfunc model to MLflow...")
    mlflow_result = log_selected_model_to_mlflow(
        candidate=candidate,
        sweep_config=sweep_config,
        output_dir=output_dir,
        packaged_weights_path=packaged_weights_path,
        eval_config=eval_config,
        serving_config=serving_config,
        test_metrics=test_metrics,
    )

    print_summary(
        output_dir=output_dir,
        candidate=candidate,
        eval_config=eval_config,
        serving_config=serving_config,
        test_metrics=test_metrics,
        mlflow_result=mlflow_result,
    )

    return output_dir


def main() -> None:
    args = parse_args()
    run_selection(
        sweep_dir=args.sweep_dir,
        data_yaml=args.data_yaml,
    )


if __name__ == "__main__":
    main()