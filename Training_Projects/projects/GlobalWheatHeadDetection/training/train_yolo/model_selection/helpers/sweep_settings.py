from pathlib import Path

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# This file lives at:
# training/train_yolo/model_selection/helpers/sweep_settings.py
MODEL_SELECTION_ROOT = Path(__file__).resolve().parents[1]
TRAIN_YOLO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Keep this short because Windows path length can break artifact downloads.
OUTPUT_ROOT = PROJECT_ROOT / "_model_select" / "sweeps"

DOWNLOADED_ARTIFACTS_DIRNAME = "artifacts"
ULTRALYTICS_VAL_RUNS_DIRNAME = "val_runs"

# ---------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# ---------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------

# This is a validation-only model-selection sweep. Test should not be used here.
SPLIT = "val"

# This is the primary model-selection metric.
# Ultralytics metrics.box.map is mAP50-95.
SELECTION_METRIC = "box_map50_95"

# Inference-time image-size search.
# 512 is included as a low-cost deployment option.
# 640 is the main training/evaluation baseline.
# 768 is the high-detail option that may help small dense objects but is thermally heavier.
# IMG_SIZES = [512, 640, 768]
#
# # NMS IoU threshold search.
# IOU_VALUES = [0.70, 0.80, 0.85, 0.90]
#
# # Maximum detections per image.
# MAX_DET_VALUES = [300, 500, 1000]

IMG_SIZES = [640]
IOU_VALUES = [0.85]
MAX_DET_VALUES = [500]

# ---------------------------------------------------------------------
# Ultralytics validation settings
# ---------------------------------------------------------------------

# Keep conf low for mAP evaluation so the precision-recall curve is not artificially clipped.
CONF = 0.001

# Local project default. Use 0 for the first CUDA GPU.
# Change this to "cpu" only when you intentionally want CPU validation.
DEVICE = 0

# Batch/workers for validation. These are not training params.
BATCH = 16
WORKERS = 4

# Keep plots off for the grid sweep to avoid producing many heavy artifacts.
PLOTS = False

# Use rectangular validation when possible, matching common Ultralytics val behavior.
RECT = True

# ---------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------

# Finished runs only. Failed/killed/running runs should not be candidates.
VALID_RUN_STATUSES = {"FINISHED"}

# Try these common MLflow artifact paths for best checkpoints.
# The helper code will also recursively search artifacts as a fallback.
BEST_PT_ARTIFACT_CANDIDATES = [
    "weights/best.pt",
    "best.pt",
    "artifacts/weights/best.pt",
    "model/best.pt",
]

# These are useful for metadata only, not required for the sweep to run.
ARGS_YAML_ARTIFACT_CANDIDATES = [
    "args.yaml",
    "weights/args.yaml",
    "artifacts/args.yaml",
]

CONFIG_SNAPSHOT_ARTIFACT_CANDIDATES = [
    "config_snapshot.yaml",
    "artifacts/config_snapshot.yaml",
]

TRAINING_SUMMARY_ARTIFACT_CANDIDATES = [
    "training_run_summary.json",
    "artifacts/training_run_summary.json",
]

# ---------------------------------------------------------------------
# Lightweight / Pareto labeling
# ---------------------------------------------------------------------

# Treat nano models as lightweight candidates.
# This lets YOLO11n remain visible even if it does not win absolute mAP.
LIGHTWEIGHT_MODEL_SIZES = {"n", "nano"}

# These columns are interpreted as "higher is better" for Pareto ranking.
PARETO_HIGHER_IS_BETTER = [
    "box_map50_95",
    "box_map50",
]

# These columns are interpreted as "lower is better" for Pareto ranking.
# Missing values are ignored by the helper logic.
PARETO_LOWER_IS_BETTER = [
    "model_file_size_mb",
    "params_millions",
    "flops_gflops",
    "eval_runtime_seconds",
]

# ---------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------

SWEEP_CONFIG_FILENAME = "sweep_config.json"
CANDIDATE_RUNS_FILENAME = "candidate_runs.json"
DISCOVERY_FAILURES_FILENAME = "discovery_failures.json"
EVAL_FAILURES_FILENAME = "eval_failures.json"

SWEEP_CSV_TEMPLATE = "inference_config_sweep_{split}.csv"
SWEEP_JSON_TEMPLATE = "inference_config_sweep_{split}.json"

BEST_OVERALL_TEMPLATE = "best_overall_{split}.json"
BEST_LIGHTWEIGHT_TEMPLATE = "best_lightweight_{split}.json"
PARETO_CANDIDATES_TEMPLATE = "pareto_candidates_{split}.json"

LATEST_SWEEP_POINTER_FILENAME = "latest_sweep_dir.txt"