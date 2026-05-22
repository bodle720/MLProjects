# Example run:
# python -m evaluation.main --visualize-sample 10

import argparse
from datetime import datetime, timezone
from pathlib import Path

from evaluation.config import EvalConfig
from evaluation.helpers.annotations import (
    extract_predictions_from_result,
    load_ground_truth_boxes,
)
from evaluation.helpers.dataset import (
    get_class_names,
    get_split_image_paths,
    load_dataset_yaml,
)
from evaluation.helpers.matching import match_predictions_to_ground_truth
from evaluation.helpers.mlflow_model import download_mlflow_model_artifact
from evaluation.helpers.paths import create_run_dir, resolve_project_path
from evaluation.helpers.report_io import build_markdown_summary, save_json, save_text
from evaluation.helpers.visual_selection import select_visualization_images
from evaluation.helpers.visualization import draw_evaluation_overlay
from evaluation.helpers.yolo_eval import load_yolo_model, run_full_split_eval


def parse_args() -> argparse.Namespace:
    defaults = EvalConfig()

    parser = argparse.ArgumentParser(
        description="Run full-split YOLO evaluation and save visual examples."
    )

    parser.add_argument("--data-yaml", default=defaults.data_yaml)
    parser.add_argument("--split", default=defaults.split, choices=["train", "val", "test"])
    parser.add_argument("--mlflow-tracking-uri", default=defaults.mlflow_tracking_uri)
    parser.add_argument("--model-uri", default=defaults.model_uri)
    parser.add_argument("--output-root", default=defaults.output_root)

    parser.add_argument("--metric-conf", type=float, default=defaults.metric_conf)
    parser.add_argument("--visual-conf", type=float, default=defaults.visual_conf)
    parser.add_argument("--iou", type=float, default=defaults.iou)
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--max-det", type=int, default=defaults.max_det)
    parser.add_argument("--batch", type=int, default=defaults.batch)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--device", default=defaults.device)

    parser.add_argument("--visualize-sample", type=int, default=defaults.visualize_sample)
    parser.add_argument(
        "--visualize-strategy",
        default=defaults.visualize_strategy,
        choices=["random", "first", "highest_conf", "most_boxes"],
    )
    parser.add_argument("--visualize-seed", type=int, default=defaults.visualize_seed)
    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=defaults.match_iou_threshold,
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        data_yaml=args.data_yaml,
        split=args.split,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        model_uri=args.model_uri,
        output_root=args.output_root,
        metric_conf=args.metric_conf,
        visual_conf=args.visual_conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        visualize_sample=args.visualize_sample,
        visualize_strategy=args.visualize_strategy,
        visualize_seed=args.visualize_seed,
        match_iou_threshold=args.match_iou_threshold,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    run_dir = create_run_dir(
        output_root=config.output_root,
        split=config.split,
    )

    save_json(run_dir / "settings.json", config.to_dict())

    mlflow_info = download_mlflow_model_artifact(
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        model_uri=config.model_uri,
        run_dir=run_dir,
    )
    save_json(run_dir / "mlflow_model_info.json", mlflow_info)

    model = load_yolo_model(mlflow_info["weights_path"])

    eval_payload = run_full_split_eval(
        model=model,
        config=config,
        run_dir=run_dir,
    )

    dataset_payload = load_dataset_yaml(config.data_yaml)
    class_names = get_class_names(dataset_payload)
    image_paths = get_split_image_paths(dataset_payload, split=config.split)

    selected_images = select_visualization_images(
        image_paths=image_paths,
        config=config,
        model=model,
    )

    visualization_payload = generate_visualizations(
        model=model,
        image_paths=selected_images,
        class_names=class_names,
        config=config,
        run_dir=run_dir,
    )

    summary = {
        "run_metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
        },
        "settings": config.to_dict(),
        "mlflow_model": mlflow_info,
        "dataset": {
            "data_yaml": str(resolve_project_path(config.data_yaml)),
            "split": config.split,
            "image_count": len(image_paths),
            "class_names": class_names,
        },
        "metrics": eval_payload["metrics"],
        "speed_ms_per_image": eval_payload["speed_ms_per_image"],
        "eval_runtime_seconds": eval_payload["eval_runtime_seconds"],
        "eval_output_dir": eval_payload["eval_output_dir"],
        "visualizations": visualization_payload,
    }

    save_json(run_dir / "eval_summary.json", summary)
    save_text(run_dir / "eval_summary.md", build_markdown_summary(summary))

    print(f"Saved evaluation run to: {run_dir}")


def generate_visualizations(
    model,
    image_paths: list[Path],
    class_names: dict[int, str],
    config: EvalConfig,
    run_dir: Path,
) -> dict:
    visual_dir = run_dir / "visualizations"
    records = []

    for idx, image_path in enumerate(image_paths, start=1):
        result = model.predict(
            source=str(image_path),
            conf=config.visual_conf,
            iou=config.iou,
            imgsz=config.imgsz,
            max_det=config.max_det,
            device=config.device,
            verbose=False,
        )[0]

        ground_truth = load_ground_truth_boxes(
            image_path=image_path,
            class_names=class_names,
        )
        predictions = extract_predictions_from_result(
            result=result,
            class_names=class_names,
        )
        matching = match_predictions_to_ground_truth(
            predictions=predictions,
            ground_truth=ground_truth,
            match_iou_threshold=config.match_iou_threshold,
        )

        output_path = visual_dir / f"{idx:03d}_{image_path.stem}.jpg"

        draw_evaluation_overlay(
            image_path=image_path,
            ground_truth=ground_truth,
            matched_predictions=matching["matched_predictions"],
            unmatched_predictions=matching["unmatched_predictions"],
            output_path=output_path,
        )

        records.append(
            {
                "index": idx,
                "image_path": str(image_path),
                "output_path": str(output_path),
                "ground_truth_count": len(ground_truth),
                "prediction_count": len(predictions),
                "matched_prediction_count": matching["matched_count"],
                "unmatched_prediction_count": matching["unmatched_prediction_count"],
                "missed_ground_truth_count": matching["missed_ground_truth_count"],
            }
        )

    payload = {
        "visualize_sample": config.visualize_sample,
        "visualize_strategy": config.visualize_strategy,
        "visual_conf": config.visual_conf,
        "match_iou_threshold": config.match_iou_threshold,
        "records": records,
    }

    save_json(run_dir / "visualizations_summary.json", payload)
    return payload


if __name__ == "__main__":
    main()