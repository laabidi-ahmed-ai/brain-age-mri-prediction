"""FastAPI TestClient tests for the health, analysis, and reports endpoints.

None of these tests require the real trained checkpoint (a randomly-initialised
EfficientNet-B0 of the same architecture stands in — see conftest.py) and none
send real email (console backend forced via test_settings).
"""
from __future__ import annotations

import tempfile

from app.ml import preprocess

from .conftest import make_nii_bytes


# --- health -----------------------------------------------------------------


def test_health_model_available(client_with_model):
    res = client_with_model.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_model_unavailable(client_without_model):
    res = client_without_model.get("/health")
    assert res.status_code == 200
    assert res.json()["model_loaded"] is False


# --- axis4: demo mode ---------------------------------------------------------


def test_axis4_demo_mode_does_not_require_model(client_without_model):
    """Explicit demo mode returns mock output even when no model is loaded."""
    res = client_without_model.post(
        "/api/axis4-brain-aging/analyze/", data={"metadata": '{"demo": true}'}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["axisId"] == "axis4-brain-aging"
    assert body["modelLoaded"] is False
    assert "predictedClass" in body
    assert "caseId" in body


# --- axis4: model availability gating -----------------------------------------


def test_axis4_real_inference_returns_503_when_model_unavailable(client_without_model):
    res = client_without_model.post(
        "/api/axis4-brain-aging/analyze/",
        data={"metadata": "{}"},
        files={"file": ("scan.nii", make_nii_bytes(), "application/octet-stream")},
    )
    assert res.status_code == 503


def test_axis4_successful_prediction_with_mocked_model(client_with_model):
    """End-to-end real-inference path using the fake (untrained) EfficientNet-B0."""
    res = client_with_model.post(
        "/api/axis4-brain-aging/analyze/",
        data={"metadata": '{"age": 55}'},
        files={"file": ("scan.nii", make_nii_bytes(), "application/octet-stream")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["modelLoaded"] is True
    assert body["axisId"] == "axis4-brain-aging"
    assert "gradCamDataUrl" in body and body["gradCamDataUrl"].startswith("data:image/png;base64,")
    assert isinstance(body["regions"], list) and len(body["regions"]) == 3
    assert "Brain–chronology Δ" in [m["label"] for m in body["metrics"]]


def test_axis4_valid_nii_gz_upload(client_with_model):
    res = client_with_model.post(
        "/api/axis4-brain-aging/analyze/",
        data={"metadata": "{}"},
        files={"file": ("scan.nii.gz", make_nii_bytes(gz=True), "application/gzip")},
    )
    assert res.status_code == 200
    assert res.json()["modelLoaded"] is True


# --- axis4: validation / error handling ---------------------------------------


def test_axis4_missing_upload_returns_400(client_with_model):
    res = client_with_model.post("/api/axis4-brain-aging/analyze/", data={"metadata": "{}"})
    assert res.status_code == 400


def test_axis4_invalid_metadata_json_returns_400(client_with_model):
    res = client_with_model.post(
        "/api/axis4-brain-aging/analyze/",
        data={"metadata": "not json"},
        files={"file": ("scan.nii", make_nii_bytes(), "application/octet-stream")},
    )
    assert res.status_code == 400


def test_axis4_invalid_metadata_value_returns_422(client_with_model):
    res = client_with_model.post(
        "/api/axis4-brain-aging/analyze/",
        data={"metadata": '{"age": "not-a-number"}'},
        files={"file": ("scan.nii", make_nii_bytes(), "application/octet-stream")},
    )
    assert res.status_code == 422


def test_axis4_unsupported_extension_returns_400(client_with_model):
    res = client_with_model.post(
        "/api/axis4-brain-aging/analyze/",
        data={"metadata": "{}"},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 400


def test_axis4_malformed_scan_returns_400(client_with_model):
    """A file with a valid extension but unreadable content is a 400, not a 500."""
    res = client_with_model.post(
        "/api/axis4-brain-aging/analyze/",
        data={"metadata": "{}"},
        files={"file": ("scan.nii", b"not actually a nifti file" * 10, "application/octet-stream")},
    )
    assert res.status_code == 400
    assert "Could not read input" in res.json()["detail"]


def test_axis4_cleans_up_temp_dir_through_full_api_path(client_with_model, monkeypatch):
    """Windows temp-dir cleanup verified end-to-end through the real HTTP route."""
    created_dirs: list[str] = []
    real_temporary_directory = tempfile.TemporaryDirectory

    class RecordingTemporaryDirectory(real_temporary_directory):
        def __enter__(self):
            path = super().__enter__()
            created_dirs.append(path)
            return path

    monkeypatch.setattr(preprocess.tempfile, "TemporaryDirectory", RecordingTemporaryDirectory)

    res = client_with_model.post(
        "/api/axis4-brain-aging/analyze/",
        data={"metadata": "{}"},
        files={"file": ("scan.nii", make_nii_bytes(), "application/octet-stream")},
    )

    assert res.status_code == 200
    assert len(created_dirs) == 1
    import os

    assert not os.path.exists(created_dirs[0])


# --- other axes: mock demonstrations ------------------------------------------


def test_other_axis_returns_mock_result(client_with_model):
    res = client_with_model.post(
        "/api/axis1-alzheimer-dementia/analyze/", data={"metadata": "{}"}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["axisId"] == "axis1-alzheimer-dementia"
    assert body["modelLoaded"] is False
    assert "predictedClass" in body


def test_unknown_axis_returns_404(client_with_model):
    res = client_with_model.post("/api/axis99-unknown/analyze/", data={"metadata": "{}"})
    assert res.status_code == 404


# --- reports / email -----------------------------------------------------------


def test_send_report_invalid_email_returns_400(client_with_model):
    res = client_with_model.post(
        "/api/send-report-email/",
        json={"to": "not-an-email", "axis_id": "axis4-brain-aging", "result": {}},
    )
    assert res.status_code == 400


def test_send_report_valid_email_console_mode(client_with_model):
    """Console mode: no SMTP connection is made; the call still succeeds."""
    res = client_with_model.post(
        "/api/send-report-email/",
        json={
            "to": "patient@example.com",
            "axis_id": "axis1-alzheimer-dementia",
            "axis_title": "Alzheimer & Dementias",
            "patient": {"id": "P-001"},
            "result": {"caseId": "BRN-TEST", "metrics": []},
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["followUpNote"] is False
