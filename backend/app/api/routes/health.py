"""Health check."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...config import Settings
from ...schemas import HealthResponse
from ...services.model_service import ModelService
from ..deps import get_app_settings, get_model_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health(
    model_service: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Report service status and whether the brain-age model is loaded."""
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        model_loaded=model_service.is_available,
    )
