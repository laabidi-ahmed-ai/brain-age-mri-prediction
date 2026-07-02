"""Pydantic request/response schemas.

The `/analyze/` response is intentionally kept as a free-form dict (not a strict
model) so the exact `AnalysisResult` contract the React frontend expects is
preserved field-for-field without accidental narrowing.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool


class AnalyzeMetadata(BaseModel):
    """Parsed `metadata` JSON sent as a multipart field by the frontend.

    All fields optional; unknown keys ignored. Patient fields are accepted but
    never logged.
    """

    model_config = {"extra": "ignore"}

    demo: bool = False
    age: Optional[float] = None
    sex: Optional[str] = None
    patientId: Optional[str] = None
    notes: Optional[str] = None
    patientEmail: Optional[str] = None


class EmailPatient(BaseModel):
    model_config = {"extra": "ignore"}

    id: Optional[str] = None
    age: Optional[float] = None
    sex: Optional[str] = None


class SendReportRequest(BaseModel):
    """Body for POST /api/send-report-email/ (mirrors the frontend payload).

    ``to`` is a plain string validated in the route (returns 400 on a bad
    address), matching the previous backend behavior.
    """

    to: str
    axis_id: str = ""
    axis_title: str = "Analysis"
    patient: EmailPatient = Field(default_factory=EmailPatient)
    result: dict[str, Any]


class SendReportResponse(BaseModel):
    ok: bool
    followUpNote: bool
