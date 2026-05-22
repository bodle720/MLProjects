import random
from pathlib import Path
from typing import Any

from evaluation.config import EvalConfig, VISUALIZE_STRATEGIES
from evaluation.helpers.annotations import image_path_to_label_path


def select_visualization_images(
    image_paths: list[Path],
    config: EvalConfig,
    model: Any,
) -> list[Path]:
    sample_count = min(config.visualize_sample, len(image_paths))

    if sample_count <= 0:
        return []

    if config.visualize_strategy not in VISUALIZE_STRATEGIES:
        raise ValueError(
            f"Unsupported visualize_strategy '{config.visualize_strategy}'. "
            f"Expected one of: {sorted(VISUALIZE_STRATEGIES)}"
        )

    if config.visualize_strategy == "first":
        return image_paths[:sample_count]

    if config.visualize_strategy == "random":
        rng = random.Random(config.visualize_seed)
        return sorted(rng.sample(image_paths, sample_count))

    if config.visualize_strategy == "most_boxes":
        return select_most_boxes(image_paths=image_paths, sample_count=sample_count)

    if config.visualize_strategy == "highest_conf":
        return select_highest_conf(
            image_paths=image_paths,
            sample_count=sample_count,
            config=config,
            model=model,
        )

    raise ValueError(f"Unsupported visualize_strategy: {config.visualize_strategy}")


def select_most_boxes(image_paths: list[Path], sample_count: int) -> list[Path]:
    scored = [
        (count_yolo_labels(image_path_to_label_path(path)), path)
        for path in image_paths
    ]

    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return [path for _, path in scored[:sample_count]]


def count_yolo_labels(label_path: Path) -> int:
    if not label_path.exists():
        return 0

    with label_path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def select_highest_conf(
    image_paths: list[Path],
    sample_count: int,
    config: EvalConfig,
    model: Any,
) -> list[Path]:
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=config.visual_conf,
        iou=config.iou,
        imgsz=config.imgsz,
        max_det=config.max_det,
        device=config.device,
        stream=True,
        verbose=False,
    )

    scored: list[tuple[float, Path]] = []

    for result in results:
        image_path = Path(result.path)
        score = 0.0

        if result.boxes is not None and len(result.boxes) > 0:
            score = float(result.boxes.conf.max().item())

        scored.append((score, image_path))

    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return [path for _, path in scored[:sample_count]]