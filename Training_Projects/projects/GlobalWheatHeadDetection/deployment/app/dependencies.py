from typing import Any

from fastapi import HTTPException, Request

from app.config import AppConfig
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService


def get_app_config(request: Request) -> AppConfig:
    return _get_state_value(
        request=request,
        name="config",
        expected_type=AppConfig,
    )


def get_model_service(request: Request) -> ModelService:
    return _get_state_value(
        request=request,
        name="model_service",
        expected_type=ModelService,
    )


def get_prediction_service(request: Request) -> PredictionService:
    return _get_state_value(
        request=request,
        name="prediction_service",
        expected_type=PredictionService,
    )


def _get_state_value(
    request: Request,
    name: str,
    expected_type: type[Any],
) -> Any:
    value = getattr(request.app.state, name, None)

    if value is None:
        raise HTTPException(
            status_code=503,
            detail=f"Application dependency is not available: {name}",
        )

    if not isinstance(value, expected_type):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Application dependency '{name}' has unexpected type: "
                f"{type(value).__name__}"
            ),
        )

    return value