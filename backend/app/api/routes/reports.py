"""Patient-report email endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...config import Settings
from ...logging_config import get_logger
from ...schemas import SendReportRequest, SendReportResponse
from ...services import email_service
from ..deps import get_app_settings

logger = get_logger("brain.reports")

router = APIRouter()


@router.post("/api/send-report-email/", response_model=SendReportResponse, tags=["reports"])
def send_report_email(
    body: SendReportRequest,
    settings: Settings = Depends(get_app_settings),
) -> SendReportResponse:
    """Email an HTML summary of an analysis to the patient (SMTP or console mode)."""
    to = body.to.strip()
    if not email_service.is_valid_email(to):
        raise HTTPException(status_code=400, detail="Invalid email address.")

    patient_label = (body.patient.id or "").strip() or "Patient"

    try:
        include_note = email_service.send_patient_report(
            settings,
            to=to,
            axis_id=body.axis_id,
            axis_title=body.axis_title or body.axis_id or "Analysis",
            patient_label=patient_label,
            result=body.result,
        )
    except email_service.EmailSendError as exc:
        raise HTTPException(status_code=502, detail="Could not send email. Check SMTP settings.") from exc
    except Exception as exc:  # noqa: BLE001 — never leak internals/PII to the client
        logger.error("send_report status=500 error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Email handler error.") from exc

    logger.info("send_report status=200 follow_up=%s", include_note)
    return SendReportResponse(ok=True, followUpNote=include_note)
