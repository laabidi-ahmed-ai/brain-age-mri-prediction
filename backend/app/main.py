"""FastAPI application factory + entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .logging_config import configure_logging, get_logger
from .api.routes import analysis, health, reports
from .services.model_service import ModelService

logger = get_logger("brain.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the brain-age model exactly once at startup; store it on app.state."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("startup service=%s loading_model=1", settings.service_name)
    app.state.model_service = ModelService.load_from_checkpoint(settings.checkpoint_path)
    logger.info("startup_complete model_loaded=%s", app.state.model_service.is_available)
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="brAIn — Axis 4 brain-age API", version="1.0.0", lifespan=lifespan)

    if settings.cors_allow_all_origins:
        allow_origins = ["*"]
    else:
        allow_origins = settings.allowed_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log request completion + duration. Never logs body/query (no PII)."""
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "request method=%s path=%s status=%d duration_ms=%.0f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    app.include_router(health.router)
    app.include_router(analysis.router)
    app.include_router(reports.router)
    return app


app = create_app()
