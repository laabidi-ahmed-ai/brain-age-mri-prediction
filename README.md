# brAIn — Brain-Age Estimation from MRI (AI Decision Support)

An end-to-end deep-learning application that predicts a subject's **brain age**
directly from an MRI scan, quantifies the **gap** between that estimate and the
patient's real (chronological) age, and explains _where_ in the brain the model
is looking with **Grad-CAM** — all inside a clinician-facing decision-support UI.

Built around a **real, trained EfficientNet-B0 model** (transfer learning on the
OASIS dataset, **validation Subject MAE ≈ 4.34 years**), served through an
end-to-end React + FastAPI research prototype.

> **University group project — Esprit School of Engineering.** _brAIn_ is a
> team-built neurology decision-support platform. This README documents **my
> individual contribution: Axis 4 — Brain-Age Estimation** — its model,
> preprocessing pipeline, Grad-CAM explainability, and API. Teammates own the
> other six axes.

---

## What it does

Upload a brain MRI → the model returns a **predicted brain age**, an approximate
error band, and — if you enter the patient's real age — the **brain–chronology
gap (Δ)**, a biomarker linked in the literature to neurodegenerative risk. A
**Grad-CAM heatmap** overlays the regions that drove the prediction, and the
result can be exported as a **decision-support report** or emailed to the patient
as a plain-language summary.

|                    |                                                                            |
| ------------------ | -------------------------------------------------------------------------- |
| **Task**           | Regress chronological age from brain MRI slices                            |
| **Model**          | EfficientNet-B0, transfer learning + controlled fine-tuning                |
| **Dataset**        | OASIS (2D axial slices from 3D volumes)                                    |
| **Best metric**    | Validation **Subject MAE ≈ 4.34 years**                                    |
| **Explainability** | Grad-CAM overlay + coarse anterior/central/posterior saliency-mass summary |
| **Stack**          | React + TanStack Start (Vite) · FastAPI (Python) · PyTorch                 |

---

## Key features

- **Real inference — when the checkpoint is installed.** When the trained
  EfficientNet-B0 checkpoint (`best_ref_b_dropout03_lr5e5.pth`) is present and
  loads successfully, the API runs real inference on every upload and reports
  `modelLoaded: true` — see
  [`backend/app/ml/inference.py`](backend/app/ml/inference.py).
  The model is loaded once at startup (FastAPI lifespan). If the checkpoint is
  missing or fails to load, the app stays up in demo-only mode: explicit demo
  requests still return labelled mock output, while a real (non-demo) request
  gets an honest `503 Model unavailable` instead of a fake prediction.
- **Multi-format MRI ingestion.** Accepts NIfTI (`.nii` / `.nii.gz`), PNG/JPEG
  slices, and **OASIS Analyze 7.5 raw** `.hdr` + `.img` pairs — either zipped
  together or supplied through two upload slots (order-independent).
- **Brain-age gap.** Enter the patient's real age and the app reports the
  Δ (predicted − chronological) against an approximate MAE band.
- **Grad-CAM explainability.** Saliency overlay on the preprocessed slice,
  summarized as coarse anterior / central / posterior Grad-CAM mass — not
  validated anatomical region measurements.
- **Clinical report + patient email.** One-click decision-support report, and a
  patient-friendly HTML email that adds a plain-language follow-up note when the
  age gap is large (> ~12 years).
- **Honest by design.** Every output is labelled a research prototype — not a
  medical diagnosis.

<img width="1024" height="486" alt="brain_axis4_upload_results" src="https://github.com/user-attachments/assets/fe07c5eb-7024-4ccd-8759-78f495f6d563" />
<img width="1024" height="486" alt="brain_axis4_gradcam_ui" src="https://github.com/user-attachments/assets/3d3644a1-f1f6-4c0b-9a30-57d7e9928695" />


_Grad-CAM heatmap over the preprocessed MRI slice, with coarse anterior/central/posterior Grad-CAM mass summaries._

---

## From research to product

The model was developed in a two-stage experimental workflow:

1. **Architecture screening** — EfficientNet-B0 vs ResNet18/34 vs DenseNet121 on
   a small subset; EfficientNet-B0 selected as the best accuracy/efficiency
   trade-off.
2. **Refinement** — training-strategy and hyperparameter tuning on EfficientNet-B0,
   improving validation Subject MAE from **4.59 → 4.3391**.

| Configuration                                       | Best val. Subject MAE (years) |
| --------------------------------------------------- | ----------------------------: |
| EfficientNet-B0, full fine-tune, LR 1e-4 (baseline) |                        4.4696 |
| EfficientNet-B0, dropout 0.3                        |                        4.5125 |
| **EfficientNet-B0, dropout 0.3, LR 5e-5 (final)**   |                    **4.3391** |

Loss: `SmoothL1Loss` · Primary metric: **Subject MAE** (predictions aggregated
per subject before scoring).

---

## Architecture

```
.
├── src/                        # Frontend — React + TanStack Start (Vite)
│   ├── pages/Axis4BrainAgingPage.tsx   # the brain-age UI
│   └── lib/                    # API client + result shaping
└── backend/                    # Backend — FastAPI
    ├── app/
    │   ├── main.py              # app factory, lifespan model loading, CORS, request logging
    │   ├── config.py            # environment-driven settings (pydantic-settings)
    │   ├── schemas.py           # Pydantic request models
    │   ├── logging_config.py    # structured logging setup
    │   ├── utils.py             # case ids, deterministic demo signal/timeline
    │   ├── api/routes/
    │   │   ├── health.py        # GET /health
    │   │   ├── analysis.py      # POST /api/axis4-brain-aging/analyze/ + mock axes
    │   │   └── reports.py       # POST /api/send-report-email/
    │   ├── services/
    │   │   ├── model_service.py # owns the loaded model; runs inference off the event loop
    │   │   ├── email_service.py # SMTP / console delivery, no PII logging
    │   │   ├── report_email.py  # HTML report renderer + follow-up-note heuristic
    │   │   └── mock_axes.py     # deterministic mock AnalysisResults for axes 1,2,3,5,6,7
    │   └── ml/                  # ★ the end-to-end brain-age inference module (my contribution)
    │       ├── architecture.py  # EfficientNet-B0 regression head (weights=None)
    │       ├── checkpoint.py    # loads the local .pth state dict — no network access
    │       ├── preprocess.py    # NIfTI / Analyze / image → 224×224 tensor
    │       ├── gradcam.py       # Grad-CAM computation + coarse band pooling
    │       └── inference.py     # tensor → brain age → AnalysisResult dict
    ├── tests/                   # pytest + FastAPI TestClient suite
    ├── requirements.txt
    └── .env.example
```

The app is organised as a multi-axis neurology platform, with each team member
owning one axis. **Axis 4 — Brain Aging is my contribution** and the
implemented end-to-end AI module (real model, real preprocessing, real
explainability). The remaining axes (1–3, 5–7), owned by teammates, are
UI-complete demonstration stubs that return illustrative data, showing how the
platform scales to other modalities.

### Request flow (Axis 4, real inference)

```
React (fetch multipart) → FastAPI route → metadata/extension/size validation
  → preprocessing (NIfTI/Analyze/image → 224×224 tensor)
  → EfficientNet-B0 forward pass → Grad-CAM
  → AnalysisResult JSON → React
```

Demo mode (`metadata.demo=true`) short-circuits straight to a labelled mock
result and never touches the model.

---

## Run locally

### Frontend

```bash
npm install
npm run dev
```

`npm run dev` prints the local URL (often `:5173` or `:8080`). To make the UI
call the real model instead of the in-browser demo data, create `.env` with:

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

### Backend (serves the real model)

**Linux / macOS:**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Windows (PowerShell):**

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The trained checkpoint ships with the repo at
`backend/app/ml/checkpoints/best_ref_b_dropout03_lr5e5.pth`, so real inference
works immediately after cloning — no manual download step. The model loads
once at startup; watch the console for `model_load ok` (or
`model_load skipped/failed`, in which case the API stays up in demo-only mode).

Interactive API docs (Swagger UI) are auto-generated at `http://127.0.0.1:8000/docs`.

### Environment variables

All optional — see [`backend/.env.example`](backend/.env.example). Copy it to
`backend/.env` and adjust as needed:

| Variable                                                                                                    | Default   | Meaning                                                                 |
| ----------------------------------------------------------------------------------------------------------- | --------- | ----------------------------------------------------------------------- |
| `CORS_ALLOW_ALL_ORIGINS`                                                                                    | `True`    | Set `False` + `CORS_ALLOWED_ORIGINS` for anything beyond a laptop demo. |
| `MAX_UPLOAD_MB`                                                                                             | `100`     | Reject uploads larger than this.                                        |
| `LOG_LEVEL`                                                                                                 | `INFO`    | Structured log verbosity.                                               |
| `EMAIL_BACKEND`                                                                                             | `console` | `console` (print only) or `smtp` (real delivery).                       |
| `EMAIL_HOST`, `EMAIL_PORT`, `EMAIL_USE_TLS`, `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD`, `DEFAULT_FROM_EMAIL` | —         | SMTP settings, used only when `EMAIL_BACKEND=smtp`.                     |

### Tests

pytest + FastAPI `TestClient` cover NIfTI preprocessing (format handling,
malformed input, Windows temp-dir cleanup) and the full API (health, model
availability, validation errors, demo mode, mock axes, email). No test
requires the real checkpoint or sends real email — a randomly-initialised
EfficientNet-B0 of the same architecture stands in for inference tests.

```bash
cd backend
pip install -r requirements.txt   # includes pytest + httpx
python -m pytest tests/ -v
```

---

## Backend API

**Endpoint:** `POST /api/axis4-brain-aging/analyze/` — `multipart/form-data`

| Field               | Meaning                                                                         |
| ------------------- | ------------------------------------------------------------------------------- |
| `file`              | Primary scan (see accepted formats below).                                      |
| `file_analyze_pair` | _(optional)_ the second half of an Analyze `.hdr`+`.img` pair.                  |
| `metadata`          | JSON string, e.g. `{"age":68,"sex":"F"}` (or `{"demo":true}` to skip the file). |

**Accepted scan formats:** NIfTI (`.nii`, `.nii.gz`) · 2D slices
(`.png`, `.jpg`, `.jpeg`, `.webp`) · OASIS Analyze 7.5 raw (`.hdr` + `.img`,
zipped as matching `basename.hdr`/`basename.img`, or via both upload slots).

**Response** — a JSON `AnalysisResult` consumed directly by the frontend:

| Field                                                          | Meaning                                                                                                                                  |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `predictedClass`                                               | e.g. `"Predicted brain age: 54.1 years"`                                                                                                 |
| `metrics[]`                                                    | Predicted brain age, typical error (MAE), approximate range, and — if `age` was supplied — chronological age and **brain–chronology Δ**. |
| `confidence[]`                                                 | Soft three-way display keyed off the size of the age gap (illustrative, not calibrated probabilities).                                   |
| `summary`                                                      | Point estimate + approximate band + Δ, in plain language.                                                                                |
| `gradCamDataUrl`                                               | Base64 PNG of the Grad-CAM overlay.                                                                                                      |
| `regions[]`                                                    | Anterior / central / posterior Grad-CAM mass summary (coarse bands, not validated anatomical regions).                                   |
| `caseId`, `axisId`, `generatedAt`, `disclaimer`, `modelLoaded` | Envelope fields.                                                                                                                         |

Smoke test:

```bash
# real scan
curl -X POST http://localhost:8000/api/axis4-brain-aging/analyze/ \
  -F file=@sample.nii \
  -F 'metadata={"age":55,"sex":"F"}'

# demo mode (no file needed)
curl -X POST http://localhost:8000/api/axis4-brain-aging/analyze/ \
  -F 'metadata={"demo":true}'
```

`"modelLoaded": true` confirms the checkpoint was found and is serving real
predictions. Error responses: `400` (bad JSON, missing file, unsupported
extension, unreadable scan), `422` (invalid metadata value), `503` (real
inference requested but the model isn't loaded — demo mode still works),
`500` (unexpected failure).

---

## Patient email (optional)

The Axis 4 page can email an HTML summary of the analysis to the patient via
`POST /api/send-report-email/`. When the brain–chronology gap exceeds ~12 years,
the email automatically includes a reassuring, plain-language note recommending
the patient mention the result to their doctor.

By default the API uses the **console email backend** — nothing is delivered;
the full HTML is printed to the terminal running `uvicorn` (ideal for demos).
For real delivery, copy `backend/.env.example` to `backend/.env` and set:

```
EMAIL_BACKEND=smtp
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=youraccount@gmail.com
EMAIL_HOST_PASSWORD=your_app_password
DEFAULT_FROM_EMAIL=brAIn Demo <youraccount@gmail.com>
```

For Gmail, create an **App password** (Google Account → Security → 2-Step
Verification → App passwords).

---

## Tech stack

**Frontend:** React, TanStack Start, Vite, TypeScript
**Backend:** Python, FastAPI, Uvicorn, Pydantic
**ML:** PyTorch, torchvision, NumPy, OpenCV, nibabel (Grad-CAM, NIfTI/Analyze I/O)

---

## Current engineering status

**Implemented:**

- React interface
- FastAPI REST API
- PyTorch checkpoint inference
- MRI preprocessing
- Grad-CAM
- Health endpoint
- Environment-based configuration
- Structured logging (startup, model load, request completion, inference duration, errors — never patient data)
- NIfTI preprocessing tests
- Automated API test suite (FastAPI TestClient: health, model availability, validation, demo mode, mock axes, email)

**Planned:**

- Docker
- CI/CD
- Live deployment
- Model artifact/version management
- Frontend automated tests
- Case/result persistence (the previous Django version best-effort saved each analysis to SQLite; this was not carried over — see Limitations)

---

## Limitations & disclaimer

This is a **research prototype** (student project). It is trained on a single
dataset family (OASIS), uses 2D slice-based modelling, and is **not FDA/CE-marked
and not a substitute for clinical diagnosis or radiology review**. Grad-CAM
explanations are qualitative. External validation and broader demographic
coverage would be required before any clinical use.

**Migration note (Django → FastAPI):** the previous Django backend
best-effort-persisted each analysis (case id, metadata, result) to a local
SQLite database. That persistence layer was not part of the required FastAPI
structure and has not been carried over — results are returned to the
frontend but no longer saved server-side. Everything else (inference,
preprocessing, Grad-CAM, the API contract, and patient email) is preserved.
