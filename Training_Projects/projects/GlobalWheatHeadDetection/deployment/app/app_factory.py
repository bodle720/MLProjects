import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.config import AppConfig, get_config
from app.routers import health, model, prediction
from app.services.model_service import ModelServiceError, build_model_service
from app.services.prediction_service import build_prediction_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting FastAPI app.")

    try:
        app.state.model_service.load_model()
    except ModelServiceError:
        logger.exception("Failed to load MLflow model during application startup.")
        raise

    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown complete.")


def create_app(config: AppConfig | None = None) -> FastAPI:
    configure_logging()

    resolved_config = config or get_config()
    model_service = build_model_service(config=resolved_config)
    prediction_service = build_prediction_service(
        config=resolved_config,
        model_service=model_service,
    )

    app = FastAPI(
        title=resolved_config.app_name,
        version=resolved_config.app_version,
        description=(
            "FastAPI serving app for the Global Wheat Head Detection YOLO model "
            "loaded from the MLflow Model Registry champion alias."
        ),
        lifespan=lifespan,
    )

    app.state.config = resolved_config
    app.state.model_service = model_service
    app.state.prediction_service = prediction_service

    include_routers(app)

    return app


def include_routers(app: FastAPI) -> None:
    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(prediction.router)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )