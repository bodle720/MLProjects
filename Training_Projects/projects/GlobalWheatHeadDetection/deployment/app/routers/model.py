from fastapi import APIRouter, Depends

from app.dependencies import get_model_service
from app.schemas import ModelInfoResponse
from app.services.model_service import ModelService

router = APIRouter(
    prefix="/model",
    tags=["model"],
)


@router.get(
    "/info",
    response_model=ModelInfoResponse,
    summary="Get loaded model information",
)
def get_model_info(
    model_service: ModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    return ModelInfoResponse(**model_service.get_model_info())