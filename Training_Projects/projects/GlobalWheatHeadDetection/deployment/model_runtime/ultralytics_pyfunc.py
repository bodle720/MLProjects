import json
from pathlib import Path
from typing import Any

import mlflow.pyfunc


class UltralyticsYoloPyfuncModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper around an Ultralytics YOLO .pt model.

    Expected input:
        A pandas DataFrame with at least:

            image_path

        Optional columns:

            conf
            iou
            imgsz
            max_det
            device

    Optional pyfunc params:

            conf
            iou
            imgsz
            max_det
            device

    Output:
        A pandas DataFrame with one row per input image:

            image_path
            detections_json

        detections_json is a JSON string containing a list of detections.

    In prediction: If both a DataFrame column and params provide the same key, params wins
    because it is applied after row values.
    """

    PREDICT_KWARGS = ["conf", "iou", "imgsz", "max_det", "device"]

    def load_context(self, context: Any) -> None:
        from ultralytics import YOLO

        weights_path = self._resolve_artifact_path(context.artifacts["weights"])
        self.model = YOLO(str(weights_path))

    def predict(self, context, model_input, params=None):
        import pandas as pd

        rows = self._normalize_input(model_input)
        predictions = []

        for row in rows:
            image_path = row["image_path"]

            predict_kwargs = {
                "source": image_path,
                "verbose": False,
            }

            for key in self.PREDICT_KWARGS:
                value = row.get(key)
                if self._has_value(value):
                    predict_kwargs[key] = self._coerce_predict_arg(key, value)

            if params:
                for key in self.PREDICT_KWARGS:
                    value = params.get(key)
                    if self._has_value(value):
                        predict_kwargs[key] = self._coerce_predict_arg(key, value)

            results = self.model.predict(**predict_kwargs)
            detections = []

            for result in results:
                detections.extend(self._result_to_detections(result))

            predictions.append(
                {
                    "image_path": image_path,
                    "detections_json": json.dumps(detections),
                }
            )

        return pd.DataFrame(predictions)

    @staticmethod
    def _resolve_artifact_path(path_value: str | Path) -> Path:
        raw_path = str(path_value)
        normalized_path = Path(raw_path.replace("\\", "/"))

        if normalized_path.exists():
            return normalized_path

        raise FileNotFoundError(
            "Model artifact path does not exist after path normalization. "
            f"Original path: {raw_path}. "
            f"Normalized path: {normalized_path}"
        )

    @staticmethod
    def _normalize_input(model_input: Any) -> list[dict[str, Any]]:
        if hasattr(model_input, "to_dict"):
            records = model_input.to_dict(orient="records")
        elif isinstance(model_input, list):
            records = model_input
        else:
            raise TypeError(
                "Expected model_input to be a pandas DataFrame or list of records."
            )

        normalized = []

        for record in records:
            if isinstance(record, str):
                normalized.append({"image_path": record})
                continue

            if not isinstance(record, dict):
                raise TypeError(
                    f"Expected input record to be dict or str, got {type(record)}"
                )

            image_path = record.get("image_path")
            if not image_path:
                raise ValueError("Each input record must include image_path")

            normalized.append(
                {
                    "image_path": str(Path(str(image_path).replace("\\", "/"))),
                    "conf": record.get("conf"),
                    "iou": record.get("iou"),
                    "imgsz": record.get("imgsz"),
                    "max_det": record.get("max_det"),
                    "device": record.get("device"),
                }
            )

        return normalized

    @staticmethod
    def _has_value(value: Any) -> bool:
        if value is None:
            return False

        try:
            import pandas as pd

            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass

        return True

    @staticmethod
    def _coerce_predict_arg(key: str, value: Any) -> Any:
        if key in {"conf", "iou"}:
            return float(value)

        if key in {"imgsz", "max_det"}:
            return int(value)

        if key == "device":
            if isinstance(value, str):
                stripped = value.strip()
                try:
                    return int(stripped)
                except ValueError:
                    return stripped

            if isinstance(value, float) and value.is_integer():
                return int(value)

            return value

        return value

    @staticmethod
    def _result_to_detections(result: Any) -> list[dict[str, Any]]:
        detections = []
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)

        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
        xywh = boxes.xywh.cpu().tolist() if boxes.xywh is not None else []
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        classes = boxes.cls.cpu().tolist() if boxes.cls is not None else []

        for idx, class_value in enumerate(classes):
            class_id = int(class_value)
            class_name = names.get(class_id, str(class_id))

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confs[idx]) if idx < len(confs) else None,
                    "bbox_xyxy": xyxy[idx] if idx < len(xyxy) else None,
                    "bbox_xywh": xywh[idx] if idx < len(xywh) else None,
                }
            )

        return detections