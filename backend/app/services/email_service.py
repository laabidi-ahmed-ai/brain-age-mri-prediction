"""Send patient-report emails via SMTP, or print nothing in console mode.

Configuration is entirely environment-driven (see ``app.config``). This module
never logs recipient addresses, patient ids, or message content — only
coarse, non-identifying facts (delivery mode, byte count).
"""
from __future__ import annotations

import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from ..config import Settings
from ..logging_config import get_logger
from .report_email import build_report_email_html, should_include_follow_up_note

logger = get_logger("brain.email")

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class EmailSendError(Exception):
    """Raised when SMTP delivery fails (route maps to 502)."""


def is_valid_email(address: str) -> bool:
    return bool(_EMAIL_RE.match(address or ""))


def send_patient_report(
    settings: Settings,
    *,
    to: str,
    axis_id: str,
    axis_title: str,
    patient_label: str,
    result: dict,
) -> bool:
    """Render and deliver the report email. Returns whether the follow-up note applies.

    In console mode (default) nothing is delivered and no patient data is logged.
    """
    include_note = should_include_follow_up_note(axis_id, result)
    html_body = build_report_email_html(
        axis_title=axis_title,
        patient_label=patient_label,
        result=result,
        include_follow_up_note=include_note,
        axis_id=axis_id,
    )
    case_id = str(result.get("caseId") or "report")
    subject = f"[brAIn] Analysis summary — {axis_title} ({case_id})"

    if not settings.use_smtp:
        logger.info("email delivery=console_skipped bytes=%d", len(html_body))
        return include_note

    _send_smtp(settings, to=to, subject=subject, html_body=html_body)
    logger.info("email delivery=sent bytes=%d", len(html_body))
    return include_note


def _send_smtp(settings: Settings, *, to: str, subject: str, html_body: str) -> None:
    """Deliver one HTML email over SMTP. Raises ``EmailSendError`` on failure."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.default_from_email
    msg["To"] = to
    msg.attach(MIMEText("This message contains HTML. Use an HTML-capable mail client.", "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(settings.email_host, settings.email_port, timeout=20) as server:
            if settings.email_use_tls:
                server.starttls()
            if settings.email_host_user:
                server.login(settings.email_host_user, settings.email_host_password)
            server.sendmail(settings.default_from_email, [to], msg.as_string())
    except Exception as exc:  # noqa: BLE001 — surface transport errors without leaking PII
        logger.error("email delivery=failed error_type=%s", type(exc).__name__)
        raise EmailSendError("Could not send email via SMTP.") from exc
