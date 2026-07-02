"""Shared pytest fixtures for the FastAPI backend.

No test here requires the real trained checkpoint or sends real email:
- ``fake_brain_age_model`` is a randomly-initialised EfficientNet-B0 (same
  architecture, untrained weights) — enough to exercise the full forward +
  Grad-CAM code path without the 16 MB checkpoint file.
- ``test_settings`` forces the console email backend, so no SMTP connection
  is ever attempted.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.api import deps  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.ml.architecture import build_brain_age_efficientnet  # noqa: E402
from app.services.model_service import ModelService  # noqa: E402


def make_nii_bytes(shape=(24, 24, 12), gz: bool = False) -> bytes:
    """Build a small valid NIfTI file in memory (optionally gzip-compressed)."""
    vol = (np.random.rand(*shape) * 200).astype(np.float32)
    raw = nib.Nifti1Image(vol, affine=np.eye(4)).to_bytes()
    return gzip.compress(raw) if gz else raw


@pytest.fixture(scope="session")
def fake_brain_age_model():
    """Untrained EfficientNet-B0 — same architecture as the real checkpoint."""
    model = build_brain_age_efficientnet()
    model.eval()
    return model


@pytest.fixture
def test_settings() -> Settings:
    """Deterministic, hermetic settings: console email, permissive CORS."""
    return Settings(email_backend="console", email_host="", cors_allow_all_origins=True)


@pytest.fixture
def client_with_model(fake_brain_age_model, test_settings):
    """TestClient where the model service reports a loaded model."""
    app = create_app()
    app.dependency_overrides[deps.get_model_service] = lambda: ModelService(model=fake_brain_age_model)
    app.dependency_overrides[deps.get_app_settings] = lambda: test_settings
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def client_without_model(test_settings):
    """TestClient where the model service reports no model available."""
    app = create_app()
    app.dependency_overrides[deps.get_model_service] = lambda: ModelService(model=None)
    app.dependency_overrides[deps.get_app_settings] = lambda: test_settings
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
