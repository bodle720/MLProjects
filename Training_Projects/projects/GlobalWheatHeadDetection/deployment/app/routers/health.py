from fastapi import APIRouter, Depends

from app.config import AppConfig
from app.dependencies import get_app_config
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
)
def get_health(
    config: AppConfig = Depends(get_app_config),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=config.app_name,
        app_version=config.app_version,
    )


@router.get(
    "/",
    response_model=HealthResponse,
    include_in_schema=False,
)
def get_root_health(
    config: AppConfig = Depends(get_app_config),
) -> HealthResponse:
    return get_health(config=config)