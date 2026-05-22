from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VISUALIZE_STRATEGIES = {"random", "first", "highest_conf", "most_boxes"}


@dataclass
class EvalConfig:
    data_yaml: str = "training/data/yolo/global-wheat-head-2021-v1/dataset.yaml"
    split: str = "test"

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    model_uri: str = "models:/GlobalWheatHeadDetector@champion"

    output_root: str = "evaluation/outputs"

    metric_conf: float = 0.001
    visual_conf: float = 0.25
    iou: float = 0.8
    imgsz: int = 640
    max_det: int = 1000
    batch: int = 4
    workers: int = 0
    device: str = "0"
    rect: bool = True
    plots: bool = True

    visualize_sample: int = 0
    visualize_strategy: str = "random"
    visualize_seed: int = 42
    match_iou_threshold: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)