"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import Request

from ..config import Settings, get_settings
from ..services.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    """Return the model service created at startup.

    Falls back to an unavailable service if the app was constructed without the
    lifespan (e.g. in unit tests that override this dependency).
    """
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        return ModelService(model=None)
    return service


def get_app_settings() -> Settings:
    return get_settings()
