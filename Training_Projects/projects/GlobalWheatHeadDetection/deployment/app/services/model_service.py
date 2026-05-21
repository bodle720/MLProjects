import logging
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd

from app.config import AppConfig

logger = logging.getLogger(__name__)


class ModelServiceError(RuntimeError):
    pass


@dataclass
class ModelService:
    config: AppConfig
    model: Any | None = None

    def load_model(self) -> None:
        if self.model is not None:
            logger.info("MLflow model is already loaded: %s", self.config.model_uri)
            return

        logger.info("Setting MLflow tracking URI: %s", self.config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

        try:
            logger.info("Loading MLflow model: %s", self.config.model_uri)
            self.model = mlflow.pyfunc.load_model(self.config.model_uri)
        except Exception as exc:
            raise ModelServiceError(
                f"Failed to load MLflow model '{self.config.model_uri}'."
            ) from exc

        logger.info("Loaded MLflow model: %s", self.config.model_uri)

    def predict(
        self,
        input_df: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if self.model is None:
            raise ModelServiceError("Model has not been loaded yet.")

        try:
            raw_output = self.model.predict(input_df, params=params)
        except Exception as exc:
            logger.exception("Model prediction failed.")
            raise ModelServiceError("Model prediction failed.") from exc

        return normalize_prediction_output(raw_output)

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_model_type(self) -> str | None:
        if self.model is None:
            return None

        return type(self.model).__name__

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_loaded": self.is_loaded(),
            "model_uri": self.config.model_uri,
            "mlflow_tracking_uri": self.config.mlflow_tracking_uri,
            "model_type": self.get_model_type(),
            "default_confidence_threshold": self.config.confidence_threshold,
            "default_iou_threshold": self.config.iou_threshold,
            "default_image_size": self.config.image_size,
            "default_max_det": self.config.max_det,
            "default_device": self.config.device,
        }


def build_model_service(config: AppConfig) -> ModelService:
    return ModelService(config=config)


def normalize_prediction_output(raw_output: Any) -> pd.DataFrame:
    if isinstance(raw_output, pd.DataFrame):
        return raw_output

    if isinstance(raw_output, list):
        return pd.DataFrame(raw_output)

    if isinstance(raw_output, dict):
        return pd.DataFrame([raw_output])

    raise ModelServiceError(
        f"Unsupported model output type: {type(raw_output).__name__}"
    )