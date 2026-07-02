"""Analysis endpoints.

- ``POST /api/axis4-brain-aging/analyze/`` runs the real EfficientNet-B0 brain-age
  model (or returns mock output only in explicit demo mode).
- ``POST /api/{axis_id}/analyze/`` returns deterministic mock demonstrations for the
  other axes, preserving the frontend's behavior when a backend is configured.
"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from ...config import Settings
from ...logging_config import get_logger
from ...ml import inference
from ...schemas import AnalyzeMetadata
from ...services.mock_axes import MOCK_AXES, build_mock_result
from ...services.model_service import ModelService
from ...utils import new_case_id
from ..deps import get_app_settings, get_model_service

logger = get_logger("brain.analyze")

router = APIRouter()

AXIS4_ID = "axis4-brain-aging"
AXIS4_ACCEPTED_EXTENSIONS = (
    ".nii", ".nii.gz", ".png", ".jpg", ".jpeg", ".webp", ".zip", ".hdr", ".img",
)
GENERIC_DISCLAIMER = (
    "Decision-support tool. Not a medical diagnosis. Clinical correlation required."
)


def _parse_metadata(metadata_raw: str) -> tuple[dict, AnalyzeMetadata]:
    """Parse + validate the metadata field. 400 on bad JSON, 422 on bad values."""
    try:
        meta = json.loads(metadata_raw or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
    if not isinstance(meta, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    try:
        model = AnalyzeMetadata.model_validate(meta)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"invalid metadata: {exc.errors()[0]['msg']}") from exc
    return meta, model


def _extension_ok(filename: str) -> bool:
    name = (filename or "").lower()
    return any(name.endswith(ext) for ext in AXIS4_ACCEPTED_EXTENSIONS)


def _finalize(result: dict, *, axis_id: str, model_loaded: bool) -> dict:
    """Stamp the standard envelope fields the frontend expects."""
    result.setdefault("axisId", axis_id)
    result["caseId"] = new_case_id()
    result.setdefault("disclaimer", GENERIC_DISCLAIMER)
    result["modelLoaded"] = model_loaded
    return result


@router.post("/api/axis4-brain-aging/analyze/")
async def analyze_axis4(
    file: UploadFile | None = File(default=None),
    file_analyze_pair: UploadFile | None = File(default=None),
    metadata: str = Form(default="{}"),
    model_service: ModelService = Depends(get_model_service),
    settings: Settings = Depends(get_app_settings),
) -> dict:
    """Brain-age analysis. Real inference unless explicit demo mode is requested."""
    meta, meta_model = _parse_metadata(metadata)

    # Explicit demo mode → mock, no model required.
    if meta_model.demo:
        result = inference.mock_result(meta)
        logger.info("analyze axis=%s mode=demo status=200", AXIS4_ID)
        return _finalize(result, axis_id=AXIS4_ID, model_loaded=model_service.is_available)

    # Real inference path.
    if file is None:
        raise HTTPException(status_code=400, detail="file is required (or set metadata.demo=true)")
    if not _extension_ok(file.filename or ""):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type for {AXIS4_ID}. "
                f"Accepted: {', '.join(AXIS4_ACCEPTED_EXTENSIONS)}"
            ),
        )
    if not model_service.is_available:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable — checkpoint not loaded. Retry once the model is installed, or use demo mode.",
        )

    raw = await file.read()
    _reject_if_too_large(raw, settings)

    pair_bytes = None
    pair_name = ""
    if file_analyze_pair is not None:
        pair_bytes = await file_analyze_pair.read()
        _reject_if_too_large(pair_bytes, settings)
        pair_name = file_analyze_pair.filename or ""

    started = time.perf_counter()
    try:
        result = await run_in_threadpool(
            model_service.run_inference,
            raw,
            file.filename or "",
            meta,
            pair_bytes=pair_bytes,
            pair_name=pair_name,
        )
    except inference.PreprocessingError as exc:
        raise HTTPException(status_code=400, detail=f"Could not read input: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — do not leak internals/PII
        logger.error("analyze axis=%s mode=real status=500 error_type=%s", AXIS4_ID, type(exc).__name__)
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    inference_ms = (time.perf_counter() - started) * 1000
    logger.info("analyze axis=%s mode=real status=200 inference_ms=%.0f", AXIS4_ID, inference_ms)
    return _finalize(result, axis_id=AXIS4_ID, model_loaded=True)


@router.post("/api/{axis_id}/analyze/")
async def analyze_mock_axis(
    axis_id: str,
    file: UploadFile | None = File(default=None),
    metadata: str = Form(default="{}"),
) -> dict:
    """Mock demonstration for the non-Axis-4 axes (no real model)."""
    if axis_id not in MOCK_AXES:
        raise HTTPException(status_code=404, detail=f"Unknown axis: {axis_id}")
    result = build_mock_result(axis_id)
    logger.info("analyze axis=%s mode=mock status=200", axis_id)
    return _finalize(result, axis_id=axis_id, model_loaded=False)


def _reject_if_too_large(payload: bytes, settings: Settings) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Upload too large (max {settings.max_upload_mb} MB).",
        )
