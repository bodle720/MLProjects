from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall API health status.")
    app_name: str = Field(..., description="Application name.")
    app_version: str = Field(..., description="Application version.")


class ModelInfoResponse(BaseModel):
    model_loaded: bool = Field(..., description="Whether the MLflow model is currently loaded.")
    model_uri: str = Field(..., description="MLflow model URI used by the app.")
    mlflow_tracking_uri: str = Field(..., description="MLflow tracking server URI.")
    model_type: str | None = Field(
        default=None,
        description="Runtime type name of the loaded model object.",
    )
    default_confidence_threshold: float = Field(
        ...,
        description="Default confidence threshold used by the API.",
    )
    default_iou_threshold: float = Field(
        ...,
        description="Default IoU threshold used by the API.",
    )
    default_image_size: int = Field(
        ...,
        description="Default image size passed to the model.",
    )
    default_max_det: int = Field(
        ...,
        description="Default maximum number of detections returned by the model.",
    )
    default_device: str = Field(
        ...,
        description="Default inference device requested by the API.",
    )


class BoundingBoxXYXY(BaseModel):
    x_min: float = Field(..., description="Left coordinate of the bounding box.")
    y_min: float = Field(..., description="Top coordinate of the bounding box.")
    x_max: float = Field(..., description="Right coordinate of the bounding box.")
    y_max: float = Field(..., description="Bottom coordinate of the bounding box.")


class BoundingBoxXYWH(BaseModel):
    x_center: float = Field(..., description="Center x-coordinate of the bounding box.")
    y_center: float = Field(..., description="Center y-coordinate of the bounding box.")
    width: float = Field(..., description="Bounding box width.")
    height: float = Field(..., description="Bounding box height.")


class Detection(BaseModel):
    class_id: int | None = Field(default=None, description="Predicted class index.")
    class_name: str | None = Field(default=None, description="Predicted class name.")
    confidence: float = Field(..., description="Detection confidence score.")
    bbox_xyxy: BoundingBoxXYXY | None = Field(
        default=None,
        description="Bounding box in x_min, y_min, x_max, y_max format.",
    )
    bbox_xywh: BoundingBoxXYWH | None = Field(
        default=None,
        description="Bounding box in x_center, y_center, width, height format.",
    )


class InferenceSettings(BaseModel):
    confidence_threshold: float = Field(..., description="Confidence threshold used for prediction.")
    iou_threshold: float = Field(..., description="IoU threshold used for prediction.")
    image_size: int = Field(..., description="Image size passed to the model.")
    max_det: int = Field(..., description="Maximum number of detections returned by the model.")
    device: str = Field(..., description="Inference device requested by the API.")


class PredictionTiming(BaseModel):
    upload_read_validate_save_ms: float = Field(
        ...,
        description="Time spent reading, validating, and saving the uploaded image.",
    )
    model_inference_ms: float = Field(
        ...,
        description="Time spent inside the MLflow model prediction call.",
    )
    prediction_parse_ms: float = Field(
        ...,
        description="Time spent parsing model output into API detections.",
    )
    total_request_ms: float = Field(
        ...,
        description="End-to-end request latency measured by the API.",
    )


class PredictionResponse(BaseModel):
    filename: str = Field(..., description="Original uploaded filename.")
    content_type: str | None = Field(default=None, description="Uploaded file content type.")
    image_width: int | None = Field(default=None, description="Uploaded image width in pixels.")
    image_height: int | None = Field(default=None, description="Uploaded image height in pixels.")
    detection_count: int = Field(..., description="Number of detections returned.")
    detections: list[Detection] = Field(..., description="Object detections.")
    inference_settings: InferenceSettings = Field(..., description="Inference settings used.")
    timing_ms: PredictionTiming = Field(..., description="Measured API timing breakdown.")
    latency_ms: float = Field(
        ...,
        description="End-to-end request latency in milliseconds. Same as timing_ms.total_request_ms.",
    )


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message.")


class RawPredictionDebugResponse(BaseModel):
    raw_output: list[dict[str, Any]] = Field(
        ...,
        description="Raw model output records. Intended only for debugging.",
    )
    inference_settings: InferenceSettings = Field(..., description="Inference settings used.")
    timing_ms: PredictionTiming = Field(..., description="Measured API timing breakdown.")