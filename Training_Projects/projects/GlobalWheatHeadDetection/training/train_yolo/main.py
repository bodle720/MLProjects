# To run from the project root:
#
#   cd MLProjects/Training_Project/projects/GlobalWheatHeadDetection
#
# Start MLflow server:
#
#   mlflow server --host 127.0.0.1 --port 5000
#
# Dry run:
#
#   python -m training.train_yolo.main --dry-run
#
# Actual training run:
#
#   python -m training.train_yolo.main
#
# Optional custom config:
#
#   python -m training.train_yolo.main --config training/config.yaml
#
# Open TensorBoard for an entire experiment:
#
#   tensorboard --logdir="training/train_yolo/run_dirs/global-wheat-head-detection"
#
# Open TensorBoard for one specific run:
#
#   tensorboard --logdir="training/train_yolo/run_dirs/global-wheat-head-detection/baseline_001_yolo11n_e50_img640_b8"
#
# Metric naming note:
#
#   train/box_loss, train/cls_loss, train/dfl_loss
#       Training split losses logged by Ultralytics.
#
#   val/box_loss, val/cls_loss, val/dfl_loss
#       Validation split losses logged during training.
#
#   metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
#       Ultralytics validation metrics calculated during training.
#
#   eval_best_val.box_map50_95
#       Explicit post-training evaluation of best.pt on the validation split.
#       Use this for model selection / champion promotion.
#
#   eval_best_test.box_map50_95
#       Explicit post-training evaluation of best.pt on the held-out test split.
#       Use this for final reporting.

import argparse
import logging
import sys
from pathlib import Path

try:
    from .helpers.config_loader import load_training_config
    from .helpers.mlflow_helpers import (
        log_final_metrics,
        log_final_training_artifacts,
        managed_mlflow_run,
        mlflow_enabled,
    )
    from .helpers.run_context import prepare_run_context
    from .helpers.yolo_training import build_train_args, run_yolo_training
except ImportError:
    from helpers.config_loader import load_training_config
    from helpers.mlflow_helpers import (
        log_final_metrics,
        log_final_training_artifacts,
        managed_mlflow_run,
        mlflow_enabled,
    )
    from helpers.run_context import prepare_run_context
    from helpers.yolo_training import build_train_args, run_yolo_training


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "training" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a YOLO object detector using the Project 3 "
            "CVDMS-exported Global Wheat Head Detection YOLO dataset."
        )
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to training YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load config, resolve run/device settings, print YOLO train args, "
            "then exit without reserving a run directory or training."
        ),
    )

    return parser.parse_args()

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

def resolve_config_path(config_arg: str) -> Path:
    config_path = Path(config_arg)

    if config_path.is_absolute():
        return config_path

    return PROJECT_ROOT / config_path

def log_run_context(config: dict, run_context) -> None:
    runtime_cfg = config.get("runtime", {})
    cuda_info = runtime_cfg.get("cuda_info", {})

    logging.info("Experiment name: %s", run_context.experiment_name)
    logging.info("Experiment directory: %s", run_context.experiment_dir)
    logging.info("Base run name: %s", run_context.base_run_name)
    logging.info("Resolved run name: %s", run_context.resolved_run_name)
    logging.info("Run directory: %s", run_context.run_dir)

    if "resolved_device" in runtime_cfg:
        logging.info("Requested device: %s", runtime_cfg.get("requested_device"))
        logging.info("Resolved device: %s", runtime_cfg.get("resolved_device"))
        logging.info("CUDA available: %s", cuda_info.get("cuda_available"))
        logging.info("CUDA device count: %s", cuda_info.get("cuda_device_count"))
        logging.info("CUDA device name: %s", cuda_info.get("cuda_device_name"))
        logging.info("Torch version: %s", cuda_info.get("torch_version"))
        logging.info("Torch CUDA version: %s", cuda_info.get("torch_cuda_version"))

def log_dataset_plan(config: dict) -> None:
    dataset_cfg = config.get("dataset", {})

    logging.info("Dataset name: %s", dataset_cfg.get("name"))
    logging.info("Dataset YAML: %s", dataset_cfg.get("dataset_yaml_resolved"))
    logging.info("Conversion report: %s", dataset_cfg.get("conversion_report_resolved"))
    logging.info("Source system: %s", dataset_cfg.get("source_system"))
    logging.info("CVDMS dataset ID: %s", dataset_cfg.get("cvdms_dataset_id"))
    logging.info("CVDMS dataset version: %s", dataset_cfg.get("cvdms_dataset_version"))
    logging.info("Label type: %s", dataset_cfg.get("label_type"))

def log_model_plan(config: dict) -> None:
    model_cfg = config.get("model", {})

    logging.info("Requested weights: %s", model_cfg.get("weights_requested", model_cfg.get("weights")))
    logging.info("Resolved weights: %s", model_cfg.get("weights_resolved"))
    logging.info("Weights directory: %s", model_cfg.get("weights_dir_resolved"))
    logging.info("Auto-download weights: %s", model_cfg.get("auto_download_weights"))
    logging.info("Model family: %s", model_cfg.get("family"))
    logging.info("Model size: %s", model_cfg.get("size"))

def log_train_args(train_args: dict) -> None:
    logging.info("YOLO train args:")
    for key, value in train_args.items():
        logging.info("  %s: %s", key, value)

def log_evaluation_plan(config: dict) -> None:
    evaluation_cfg = config.get("evaluation", {})

    if not isinstance(evaluation_cfg, dict):
        logging.info("No evaluation config found.")
        return

    logging.info("Configured best-model evaluations:")

    found_any = False
    for eval_name in ("val", "test"):
        eval_cfg = evaluation_cfg.get(eval_name)

        if not isinstance(eval_cfg, dict):
            continue

        found_any = True
        enabled = eval_cfg.get("enabled", True)
        split = eval_cfg.get("split", eval_name)
        max_det = eval_cfg.get("max_det")
        logging.info(
            "  %s: enabled=%s split=%s max_det=%s",
            eval_name,
            enabled,
            split,
            max_det,
        )

    if not found_any:
        enabled = evaluation_cfg.get("enabled", True)
        split = evaluation_cfg.get("split", "test")
        max_det = evaluation_cfg.get("max_det")
        logging.info(
            "  legacy_single_eval: enabled=%s split=%s max_det=%s",
            enabled,
            split,
            max_det,
        )

def log_all_evaluation_metrics(summary: dict) -> None:
    evaluations = summary.get("evaluations", {})

    if not evaluations:
        logging.info("No explicit evaluation metrics found to log to MLflow.")
        return

    for eval_name, eval_summary in evaluations.items():
        split = eval_summary.get("split", eval_name)
        metrics = eval_summary.get("metrics", {})

        if not metrics:
            logging.info(
                "No metrics found for evaluation '%s' on split '%s'.",
                eval_name,
                split,
            )
            continue

        prefix = f"eval_best_{eval_name}"
        logging.info(
            "Logging MLflow metrics for evaluation '%s' on split '%s' with prefix '%s'.",
            eval_name,
            split,
            prefix,
        )

        log_final_metrics(
            metrics=metrics,
            prefix=prefix,
        )

def run(args: argparse.Namespace) -> int:
    config_path = resolve_config_path(args.config)

    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Training config: %s", config_path)

    config = load_training_config(
        config_path=config_path,
        project_root=PROJECT_ROOT,
    )

    run_context = prepare_run_context(
        config=config,
        reserve=not args.dry_run,
    )
    config["runtime"] = run_context.to_dict()

    train_args = build_train_args(config)

    log_run_context(config, run_context)
    log_dataset_plan(config)
    log_model_plan(config)
    log_evaluation_plan(config)

    if args.dry_run:
        logging.info("Dry run enabled. No run directory was reserved and no training was started.")
        log_train_args(train_args)
        return 0

    if mlflow_enabled(config):
        logging.info("MLflow enabled.")
        logging.info("Tracking URI: %s", config["mlflow"]["tracking_uri"])
        logging.info("Experiment: %s", config["mlflow"]["experiment_name"])
        logging.info("Run name: %s", config["runtime"]["resolved_run_name"])
    else:
        logging.info("MLflow disabled.")

    with managed_mlflow_run(config=config, config_path=config_path) as run_id:
        if run_id:
            logging.info("MLflow run ID: %s", run_id)

        summary = run_yolo_training(
            config=config,
            config_path=config_path,
        )

        if mlflow_enabled(config):
            log_all_evaluation_metrics(summary)

            run_dir = Path(summary["artifacts"]["run_dir"])
            log_final_training_artifacts(
                config=config,
                run_dir=run_dir,
                summary=summary,
            )

    logging.info("YOLO training workflow completed successfully.")
    return 0

def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        return run(args)
    except Exception as exc:
        logging.exception("YOLO training workflow failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())