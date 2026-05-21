from dataclasses import dataclass
import os


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppConfig:
    app_name: str = os.getenv(
        "APP_NAME",
        "Global Wheat Head Detection API",
    )
    app_version: str = os.getenv("APP_VERSION", "0.1.0")

    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    model_uri: str = os.getenv(
        "MODEL_URI",
        "models:/GlobalWheatHeadDetector@champion",
    )

    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.8"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "640"))
    max_det: int = int(os.getenv("MAX_DET", "1000"))
    device: str = os.getenv("INFERENCE_DEVICE", "cpu")

    temp_upload_dir: str = os.getenv("TEMP_UPLOAD_DIR", "app/tmp/uploads")

    enable_inference_logging: bool = _get_bool_env(
        "ENABLE_INFERENCE_LOGGING",
        True,
    )
    inference_log_path: str = os.getenv(
        "INFERENCE_LOG_PATH",
        "app/logs/inference_log.jsonl",
    )


def get_config() -> AppConfig:
    return AppConfig()