"""Environment-driven application settings (pydantic-settings)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent / "ml" / "checkpoints" / "best_ref_b_dropout03_lr5e5.pth"
)


class Settings(BaseSettings):
    """All configuration comes from environment variables / backend/.env."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_name: str = "brAIn backend"

    # Model
    checkpoint_path: Path = DEFAULT_CHECKPOINT

    # Uploads
    max_upload_mb: int = 100

    # CORS
    cors_allow_all_origins: bool = True
    cors_allowed_origins: str = ""  # comma-separated, used when allow_all is False

    # Email / SMTP. Leave email_host empty (or email_backend=console) for console mode.
    email_backend: str = "console"  # "console" | "smtp"
    email_host: str = ""
    email_port: int = 587
    email_use_tls: bool = True
    email_host_user: str = ""
    email_host_password: str = ""
    default_from_email: str = "brAIn <noreply@localhost>"

    # Logging
    log_level: str = "INFO"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def use_smtp(self) -> bool:
        return self.email_backend.lower() == "smtp" and bool(self.email_host)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
