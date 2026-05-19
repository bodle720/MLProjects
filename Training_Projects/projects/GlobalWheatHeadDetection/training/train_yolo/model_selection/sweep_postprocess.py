# To run:
# python training/train_yolo/model_selection/sweep_postprocess.py --experiment-name global-wheat-head-detection --data-yaml training/data/yolo/global-wheat-head-2021-v1/dataset.yaml

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from helpers import artifacts
from helpers import candidate_metadata
from helpers import mlflow_runs
from helpers import ranking
from helpers import sweep_io
from helpers import sweep_settings as settings
from helpers import ultralytics_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a validation-only inference-configuration sweep across all "
            "candidate best.pt checkpoints in an MLflow experiment."
        )
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="MLflow experiment name containing completed YOLO training runs.",
    )
    parser.add_argument(
        "--data-yaml",
        required=True,
        type=Path,
        help="Path to the Ultralytics dataset.yaml file to use for validation.",
    )

    return parser.parse_args()


def validate_inputs(data_yaml: Path) -> Path:
    resolved_data_yaml = data_yaml.expanduser().resolve()

    if not resolved_data_yaml.exists():
        raise FileNotFoundError(f"Could not find dataset YAML: {resolved_data_yaml}")

    if not resolved_data_yaml.is_file():
        raise ValueError(f"Dataset YAML path is not a file: {resolved_data_yaml}")

    return resolved_data_yaml


def print_header(experiment_name: str, data_yaml: Path, output_dir: Path) -> None:
    print()
    print("YOLO inference-configuration sweep")
    print("----------------------------------")
    print(f"Experiment:       {experiment_name}")
    print(f"MLflow URI:       {settings.MLFLOW_TRACKING_URI}")
    print(f"Dataset YAML:     {data_yaml}")
    print(f"Split:            {settings.SPLIT}")
    print(f"Selection metric: {settings.SELECTION_METRIC}")
    print(f"Image sizes:      {settings.IMG_SIZES}")
    print(f"IoU values:       {settings.IOU_VALUES}")
    print(f"Max det values:   {settings.MAX_DET_VALUES}")
    print(f"Output dir:       {output_dir}")
    print()


def print_candidate_summary(candidates: list[dict], discovery_failures: list[dict]) -> None:
    print(f"Candidate runs with best.pt: {len(candidates)}")
    print(f"Discovery failures/skips:    {len(discovery_failures)}")

    if not candidates:
        print()
        print("No candidate best.pt checkpoints found.")
        return

    print()
    print("Candidates:")
    for candidate in candidates:
        print(
            "  - "
            f"{candidate.get('run_name')} "
            f"(size={candidate.get('model_size')}, "
            f"family={candidate.get('model_family')}, "
            f"train_imgsz={candidate.get('training_imgsz')}, "
            f"lightweight={candidate.get('is_lightweight_candidate')})"
        )
    print()


def print_ranking_summary(
    best_overall: dict | None,
    best_lightweight: dict | None,
    pareto_candidates: list[dict],
) -> None:
    print()
    print("Validation sweep summary")
    print("------------------------")

    if best_overall is None:
        print("Best overall:      None")
    else:
        print(
            "Best overall:      "
            f"{best_overall.get('run_name')} | "
            f"imgsz={best_overall.get('imgsz')} | "
            f"iou={best_overall.get('iou')} | "
            f"max_det={best_overall.get('max_det')} | "
            f"{settings.SELECTION_METRIC}={best_overall.get(settings.SELECTION_METRIC)}"
        )

    if best_lightweight is None:
        print("Best lightweight:  None")
    else:
        print(
            "Best lightweight:  "
            f"{best_lightweight.get('run_name')} | "
            f"imgsz={best_lightweight.get('imgsz')} | "
            f"iou={best_lightweight.get('iou')} | "
            f"max_det={best_lightweight.get('max_det')} | "
            f"{settings.SELECTION_METRIC}={best_lightweight.get(settings.SELECTION_METRIC)}"
        )

    print(f"Pareto candidates: {len(pareto_candidates)}")
    print()


def run_sweep(experiment_name: str, data_yaml: Path) -> Path:
    data_yaml = validate_inputs(data_yaml)

    output_dir = sweep_io.create_sweep_output_dir(
        experiment_name=experiment_name,
        split=settings.SPLIT,
    )

    print_header(
        experiment_name=experiment_name,
        data_yaml=data_yaml,
        output_dir=output_dir,
    )

    sweep_io.write_sweep_config(
        output_dir=output_dir,
        experiment_name=experiment_name,
        data_yaml=data_yaml,
        split=settings.SPLIT,
    )

    client = mlflow_runs.build_mlflow_client()

    print("Discovering finished MLflow runs...")
    run_summaries = mlflow_runs.list_finished_run_summaries(
        client=client,
        experiment_name=experiment_name,
    )
    print(f"Finished runs found: {len(run_summaries)}")

    download_root = sweep_io.get_downloaded_artifacts_dir(output_dir)

    print("Locating and downloading candidate best.pt artifacts...")
    candidates, discovery_failures = artifacts.prepare_all_candidate_artifacts(
        client=client,
        run_summaries=run_summaries,
        download_root=download_root,
    )

    candidates = candidate_metadata.enrich_all_candidate_metadata(candidates)

    sweep_io.write_discovery_outputs(
        output_dir=output_dir,
        candidate_runs=candidates,
        discovery_failures=discovery_failures,
    )

    print_candidate_summary(
        candidates=candidates,
        discovery_failures=discovery_failures,
    )

    if not candidates:
        print(f"Outputs saved to: {output_dir}")
        return output_dir

    total_evals = (
        len(candidates)
        * len(settings.IMG_SIZES)
        * len(settings.IOU_VALUES)
        * len(settings.MAX_DET_VALUES)
    )

    print(f"Running validation sweep with {total_evals} total evaluations...")
    print("This can take a while depending on model count and image sizes.")
    print()

    records, eval_failures = ultralytics_eval.evaluate_all_candidate_grids(
        candidates=candidates,
        data_yaml=data_yaml,
        split=settings.SPLIT,
        output_dir=output_dir,
    )

    print(f"Successful evaluations: {len(records)}")
    print(f"Evaluation failures:    {len(eval_failures)}")

    ranking_summary = ranking.summarize_rankings(records)

    ranked_records = ranking_summary["ranked_records"]
    best_overall = ranking_summary["best_overall"]
    best_lightweight = ranking_summary["best_lightweight"]
    pareto_candidates = ranking_summary["pareto_candidates"]

    sweep_io.write_sweep_results(
        output_dir=output_dir,
        split=settings.SPLIT,
        records=ranked_records,
    )
    sweep_io.write_eval_failures(
        output_dir=output_dir,
        eval_failures=eval_failures,
    )
    sweep_io.write_ranking_outputs(
        output_dir=output_dir,
        split=settings.SPLIT,
        best_overall=best_overall,
        best_lightweight=best_lightweight,
        pareto_candidates=pareto_candidates,
    )

    print_ranking_summary(
        best_overall=best_overall,
        best_lightweight=best_lightweight,
        pareto_candidates=pareto_candidates,
    )

    print(f"Outputs saved to: {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()

    run_sweep(
        experiment_name=args.experiment_name,
        data_yaml=args.data_yaml,
    )


if __name__ == "__main__":
    main()