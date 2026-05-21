from fastapi import APIRouter, Depends, File, UploadFile

from app.dependencies import get_prediction_service
from app.schemas import ErrorResponse, PredictionResponse, RawPredictionDebugResponse
from app.services.prediction_service import PredictionService

router = APIRouter(
    prefix="/predict",
    tags=["prediction"],
)


@router.post(
    "",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Run wheat head detection on an uploaded image",
)
async def predict_image(
    file: UploadFile = File(..., description="Image file to run inference on."),
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    return await prediction_service.predict_upload(upload_file=file)


@router.post(
    "/debug",
    response_model=RawPredictionDebugResponse,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Run wheat head detection and return raw model output",
)
async def predict_image_debug(
    file: UploadFile = File(..., description="Image file to run inference on."),
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> RawPredictionDebugResponse:
    return await prediction_service.predict_upload_debug(upload_file=file)