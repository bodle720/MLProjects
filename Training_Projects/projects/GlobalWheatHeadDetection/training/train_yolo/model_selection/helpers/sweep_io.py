import csv
import json
import re
from datetime import datetime
from pathlib import Path

from helpers import sweep_settings as settings


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "experiment"


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_sweep_output_dir(experiment_name: str, split: str) -> Path:
    experiment_slug = slugify(experiment_name)
    timestamp = make_timestamp()

    output_dir = settings.OUTPUT_ROOT / experiment_slug / f"{timestamp}_{split}"
    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / settings.DOWNLOADED_ARTIFACTS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (output_dir / settings.ULTRALYTICS_VAL_RUNS_DIRNAME).mkdir(parents=True, exist_ok=True)

    latest_pointer = settings.OUTPUT_ROOT / experiment_slug / settings.LATEST_SWEEP_POINTER_FILENAME
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(str(output_dir.resolve()), encoding="utf-8")

    return output_dir


def get_downloaded_artifacts_dir(output_dir: Path) -> Path:
    path = output_dir / settings.DOWNLOADED_ARTIFACTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_ultralytics_val_runs_dir(output_dir: Path) -> Path:
    path = output_dir / settings.ULTRALYTICS_VAL_RUNS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_records_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for record in records for key in record.keys()})

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_sweep_config(
    output_dir: Path,
    experiment_name: str,
    data_yaml: Path,
    split: str,
) -> None:
    config = {
        "experiment_name": experiment_name,
        "data_yaml": str(data_yaml),
        "split": split,
        "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
        "selection_metric": settings.SELECTION_METRIC,
        "img_sizes": settings.IMG_SIZES,
        "iou_values": settings.IOU_VALUES,
        "max_det_values": settings.MAX_DET_VALUES,
        "conf": settings.CONF,
        "device": settings.DEVICE,
        "batch": settings.BATCH,
        "workers": settings.WORKERS,
        "plots": settings.PLOTS,
        "rect": settings.RECT,
    }
    write_json(config, output_dir / settings.SWEEP_CONFIG_FILENAME)


def write_sweep_results(output_dir: Path, split: str, records: list[dict]) -> None:
    csv_path = output_dir / settings.SWEEP_CSV_TEMPLATE.format(split=split)
    json_path = output_dir / settings.SWEEP_JSON_TEMPLATE.format(split=split)

    write_records_csv(records, csv_path)
    write_json(records, json_path)


def write_discovery_outputs(
    output_dir: Path,
    candidate_runs: list[dict],
    discovery_failures: list[dict],
) -> None:
    write_json(candidate_runs, output_dir / settings.CANDIDATE_RUNS_FILENAME)
    write_json(discovery_failures, output_dir / settings.DISCOVERY_FAILURES_FILENAME)


def write_eval_failures(output_dir: Path, eval_failures: list[dict]) -> None:
    write_json(eval_failures, output_dir / settings.EVAL_FAILURES_FILENAME)


def write_ranking_outputs(
    output_dir: Path,
    split: str,
    best_overall: dict | None,
    best_lightweight: dict | None,
    pareto_candidates: list[dict],
) -> None:
    write_json(
        best_overall,
        output_dir / settings.BEST_OVERALL_TEMPLATE.format(split=split),
    )
    write_json(
        best_lightweight,
        output_dir / settings.BEST_LIGHTWEIGHT_TEMPLATE.format(split=split),
    )
    write_json(
        pareto_candidates,
        output_dir / settings.PARETO_CANDIDATES_TEMPLATE.format(split=split),
    )