import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, UploadFile

from app.config import AppConfig
from app.schemas import (
    BoundingBoxXYWH,
    BoundingBoxXYXY,
    Detection,
    InferenceSettings,
    PredictionResponse,
    PredictionTiming,
    RawPredictionDebugResponse,
)
from app.services.image_io import SavedUpload, delete_file_safely, save_upload_to_temp_file
from app.services.model_service import ModelService, ModelServiceError

logger = logging.getLogger(__name__)


@dataclass
class PredictionService:
    config: AppConfig
    model_service: ModelService

    async def predict_upload(self, upload_file: UploadFile) -> PredictionResponse:
        total_start = time.perf_counter()
        saved_upload: SavedUpload | None = None

        try:
            upload_start = time.perf_counter()
            saved_upload = await save_upload_to_temp_file(
                upload_file=upload_file,
                temp_upload_dir=self.config.temp_upload_dir,
            )
            upload_ms = elapsed_ms(upload_start)

            input_df = self._build_input_dataframe(saved_upload)
            params = self._build_prediction_params()

            inference_start = time.perf_counter()
            output_df = self.model_service.predict(input_df=input_df, params=params)
            inference_ms = elapsed_ms(inference_start)

            parse_start = time.perf_counter()
            detections = parse_detections_from_output(output_df)
            parse_ms = elapsed_ms(parse_start)

            total_ms = elapsed_ms(total_start)
            timing = build_prediction_timing(
                upload_ms=upload_ms,
                inference_ms=inference_ms,
                parse_ms=parse_ms,
                total_ms=total_ms,
            )
            inference_settings = self._build_inference_settings()

            response = PredictionResponse(
                filename=saved_upload.original_filename,
                content_type=saved_upload.content_type,
                image_width=saved_upload.image_width,
                image_height=saved_upload.image_height,
                detection_count=len(detections),
                detections=detections,
                inference_settings=inference_settings,
                timing_ms=timing,
                latency_ms=timing.total_request_ms,
            )

            self._log_successful_prediction(
                saved_upload=saved_upload,
                response=response,
            )

            return response
        except ModelServiceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            if saved_upload is not None:
                delete_file_safely(saved_upload.path)

    async def predict_upload_debug(
        self,
        upload_file: UploadFile,
    ) -> RawPredictionDebugResponse:
        total_start = time.perf_counter()
        saved_upload: SavedUpload | None = None

        try:
            upload_start = time.perf_counter()
            saved_upload = await save_upload_to_temp_file(
                upload_file=upload_file,
                temp_upload_dir=self.config.temp_upload_dir,
            )
            upload_ms = elapsed_ms(upload_start)

            input_df = self._build_input_dataframe(saved_upload)
            params = self._build_prediction_params()

            inference_start = time.perf_counter()
            output_df = self.model_service.predict(input_df=input_df, params=params)
            inference_ms = elapsed_ms(inference_start)

            parse_start = time.perf_counter()
            _ = output_df.to_dict(orient="records")
            parse_ms = elapsed_ms(parse_start)

            timing = build_prediction_timing(
                upload_ms=upload_ms,
                inference_ms=inference_ms,
                parse_ms=parse_ms,
                total_ms=elapsed_ms(total_start),
            )

            return RawPredictionDebugResponse(
                raw_output=output_df.to_dict(orient="records"),
                inference_settings=self._build_inference_settings(),
                timing_ms=timing,
            )
        except ModelServiceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            if saved_upload is not None:
                delete_file_safely(saved_upload.path)

    def _build_input_dataframe(self, saved_upload: SavedUpload) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "image_path": str(saved_upload.path),
                }
            ]
        )

    def _build_prediction_params(self) -> dict[str, Any]:
        return {
            "conf": self.config.confidence_threshold,
            "iou": self.config.iou_threshold,
            "imgsz": self.config.image_size,
            "max_det": self.config.max_det,
            "device": self.config.device,
        }

    def _build_inference_settings(self) -> InferenceSettings:
        return InferenceSettings(
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            image_size=self.config.image_size,
            max_det=self.config.max_det,
            device=self.config.device,
        )

    def _log_successful_prediction(
        self,
        saved_upload: SavedUpload,
        response: PredictionResponse,
    ) -> None:
        if not self.config.enable_inference_logging:
            return

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "filename": response.filename,
            "content_type": response.content_type,
            "image_width": response.image_width,
            "image_height": response.image_height,
            "image_format": saved_upload.image_format,
            "detection_count": response.detection_count,
            "inference_settings": response.inference_settings.model_dump(),
            "timing_ms": response.timing_ms.model_dump(),
            "latency_ms": response.latency_ms,
            "model_uri": self.config.model_uri,
        }

        try:
            append_jsonl_record(
                path=self.config.inference_log_path,
                record=record,
            )
        except OSError:
            logger.exception(
                "Failed to write inference log record to %s.",
                self.config.inference_log_path,
            )


def build_prediction_service(
    config: AppConfig,
    model_service: ModelService,
) -> PredictionService:
    return PredictionService(
        config=config,
        model_service=model_service,
    )


def elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000.0


def build_prediction_timing(
    upload_ms: float,
    inference_ms: float,
    parse_ms: float,
    total_ms: float,
) -> PredictionTiming:
    return PredictionTiming(
        upload_read_validate_save_ms=round(upload_ms, 3),
        model_inference_ms=round(inference_ms, 3),
        prediction_parse_ms=round(parse_ms, 3),
        total_request_ms=round(total_ms, 3),
    )


def append_jsonl_record(path: str, record: dict[str, Any]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def parse_detections_from_output(output_df: pd.DataFrame) -> list[Detection]:
    if output_df.empty:
        return []

    first_row = output_df.iloc[0].to_dict()

    if "detections_json" in first_row:
        detections_payload = parse_json_payload(first_row["detections_json"])
        return normalize_detection_list(detections_payload)

    if "detections" in first_row:
        detections_payload = parse_json_payload(first_row["detections"])
        return normalize_detection_list(detections_payload)

    return normalize_detection_list([first_row])


def parse_json_payload(value: Any) -> Any:
    if value is None:
        return []

    if isinstance(value, str):
        if not value.strip():
            return []

        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Model returned invalid detections JSON.",
            ) from exc

    return value


def normalize_detection_list(payload: Any) -> list[Detection]:
    if payload is None:
        return []

    if isinstance(payload, dict):
        if "detections" in payload:
            payload = payload["detections"]
        else:
            payload = [payload]

    if not isinstance(payload, list):
        raise HTTPException(
            status_code=500,
            detail="Model returned detections in an unsupported format.",
        )

    return [normalize_detection(item) for item in payload]


def normalize_detection(item: Any) -> Detection:
    if not isinstance(item, dict):
        raise HTTPException(
            status_code=500,
            detail="Model returned a detection item in an unsupported format.",
        )

    bbox_xyxy = build_bbox_xyxy(item)
    bbox_xywh = build_bbox_xywh(item)
    confidence = item.get("confidence", item.get("conf", item.get("score")))

    if confidence is None:
        raise HTTPException(
            status_code=500,
            detail="Model returned a detection without a confidence score.",
        )

    return Detection(
        class_id=coerce_optional_int(
            item.get("class_id", item.get("class", item.get("cls"))),
        ),
        class_name=coerce_optional_str(
            item.get("class_name", item.get("name", item.get("label"))),
        ),
        confidence=float(confidence),
        bbox_xyxy=bbox_xyxy,
        bbox_xywh=bbox_xywh,
    )


def build_bbox_xyxy(item: dict[str, Any]) -> BoundingBoxXYXY | None:
    bbox = item.get("bbox_xyxy", item.get("xyxy"))

    if bbox is None and "bbox" in item:
        bbox = item["bbox"]

    if bbox is None:
        keys = ("x_min", "y_min", "x_max", "y_max")
        if all(key in item for key in keys):
            bbox = [item[key] for key in keys]

    if bbox is None:
        return None

    if isinstance(bbox, dict):
        return BoundingBoxXYXY(
            x_min=float(bbox.get("x_min", bbox.get("xmin"))),
            y_min=float(bbox.get("y_min", bbox.get("ymin"))),
            x_max=float(bbox.get("x_max", bbox.get("xmax"))),
            y_max=float(bbox.get("y_max", bbox.get("ymax"))),
        )

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return BoundingBoxXYXY(
            x_min=float(bbox[0]),
            y_min=float(bbox[1]),
            x_max=float(bbox[2]),
            y_max=float(bbox[3]),
        )

    raise HTTPException(
        status_code=500,
        detail="Model returned bbox_xyxy in an unsupported format.",
    )


def build_bbox_xywh(item: dict[str, Any]) -> BoundingBoxXYWH | None:
    bbox = item.get("bbox_xywh", item.get("xywh"))

    if bbox is None:
        keys = ("x_center", "y_center", "width", "height")
        if all(key in item for key in keys):
            bbox = [item[key] for key in keys]

    if bbox is None:
        return None

    if isinstance(bbox, dict):
        return BoundingBoxXYWH(
            x_center=float(bbox.get("x_center", bbox.get("x"))),
            y_center=float(bbox.get("y_center", bbox.get("y"))),
            width=float(bbox.get("width", bbox.get("w"))),
            height=float(bbox.get("height", bbox.get("h"))),
        )

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return BoundingBoxXYWH(
            x_center=float(bbox[0]),
            y_center=float(bbox[1]),
            width=float(bbox[2]),
            height=float(bbox[3]),
        )

    raise HTTPException(
        status_code=500,
        detail="Model returned bbox_xywh in an unsupported format.",
    )


def coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None

    return int(value)


def coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None

    return str(value)