# brAIn — Brain-Age Estimation from MRI (AI Decision Support)

An end-to-end deep-learning application that predicts a subject's **brain age**
directly from an MRI scan, quantifies the **gap** between that estimate and the
patient's real (chronological) age, and explains *where* in the brain the model
is looking with **Grad-CAM** — all inside a clinician-facing decision-support UI.

Built around a **real, trained EfficientNet-B0 model** (transfer learning on the
OASIS dataset, **validation Subject MAE ≈ 4.34 years**), served through a
production-style React + Django stack.

> **University group project — Esprit School of Engineering.** *brAIn* is a
> team-built neurology decision-support platform. This README documents **my
> individual contribution: Axis 4 — Brain-Age Estimation** — its model,
> preprocessing pipeline, Grad-CAM explainability, and API. Teammates own the
> other six axes.

<!-- 📸 Insert screenshot here: prediction result view -->
![Brain-age prediction result](docs/screenshots/axis4-prediction.png)
*Predicted brain age with an approximate confidence band and the brain–chronology gap (Δ).*

---

## What it does

Upload a brain MRI → the model returns a **predicted brain age**, an approximate
error band, and — if you enter the patient's real age — the **brain–chronology
gap (Δ)**, a biomarker linked in the literature to neurodegenerative risk. A
**Grad-CAM heatmap** overlays the regions that drove the prediction, and the
result can be exported as a **decision-support report** or emailed to the patient
as a plain-language summary.

| | |
|---|---|
| **Task** | Regress chronological age from brain MRI slices |
| **Model** | EfficientNet-B0, transfer learning + controlled fine-tuning |
| **Dataset** | OASIS (2D axial slices from 3D volumes) |
| **Best metric** | Validation **Subject MAE ≈ 4.34 years** |
| **Explainability** | Grad-CAM overlay + per-region contributions |
| **Stack** | React + TanStack Start (Vite) · Django REST Framework · PyTorch |

---

## Key features

- **Real inference, not a mock.** A trained EfficientNet-B0 checkpoint
  (`best_ref_b_dropout03_lr5e5.pth`) runs on every upload — see
  [`backend/axis4_brain_aging/ml/inference.py`](backend/axis4_brain_aging/ml/inference.py).
- **Multi-format MRI ingestion.** Accepts NIfTI (`.nii` / `.nii.gz`), PNG/JPEG
  slices, and **OASIS Analyze 7.5 raw** `.hdr` + `.img` pairs — either zipped
  together or supplied through two upload slots (order-independent).
- **Brain-age gap.** Enter the patient's real age and the app reports the
  Δ (predicted − chronological) against an approximate MAE band.
- **Grad-CAM explainability.** Saliency overlay on the preprocessed slice plus
  anterior / central / posterior region-contribution breakdown.
- **Clinical report + patient email.** One-click decision-support report, and a
  patient-friendly HTML email that adds a plain-language follow-up note when the
  age gap is large (> ~12 years).
- **Honest by design.** Every output is labelled a research prototype — not a
  medical diagnosis.

<!-- 📸 Insert screenshot here: Grad-CAM explainability + region contributions -->
![Grad-CAM explainability](docs/screenshots/axis4-gradcam.png)
*Grad-CAM heatmap over the preprocessed MRI slice, with per-region contribution scores.*

---

## From research to product

The model was developed in a two-stage experimental workflow:

1. **Architecture screening** — EfficientNet-B0 vs ResNet18/34 vs DenseNet121 on
   a small subset; EfficientNet-B0 selected as the best accuracy/efficiency
   trade-off.
2. **Refinement** — training-strategy and hyperparameter tuning on EfficientNet-B0,
   improving validation Subject MAE from **4.59 → 4.3391**.

| Configuration | Best val. Subject MAE (years) |
|---|---:|
| EfficientNet-B0, full fine-tune, LR 1e-4 (baseline) | 4.4696 |
| EfficientNet-B0, dropout 0.3 | 4.5125 |
| **EfficientNet-B0, dropout 0.3, LR 5e-5 (final)** | **4.3391** |

Loss: `SmoothL1Loss` · Primary metric: **Subject MAE** (predictions aggregated
per subject before scoring).

---

## Architecture

```
.
├── src/                        # Frontend — React + TanStack Start (Vite)
│   ├── pages/Axis4BrainAgingPage.tsx   # the brain-age UI
│   └── lib/                    # API client + result shaping
└── backend/                    # Backend — Django REST Framework
    └── axis4_brain_aging/      # ★ the production brain-age module
        ├── ml/
        │   ├── architecture.py # EfficientNet-B0 regression head
        │   ├── checkpoint.py   # loads the .pth state_dict
        │   ├── preprocess.py   # NIfTI / Analyze / image → 224×224 tensor
        │   ├── gradcam.py      # Grad-CAM computation + PNG overlay
        │   └── inference.py    # predict(): tensor → brain age → result dict
        ├── explain/            # Grad-CAM heatmap → region contributions
        └── views.py            # AnalyzeView (+ optional Analyze-pair upload slot)
```

The app is organised as a multi-axis neurology platform, with each team member
owning one axis. **Axis 4 — Brain Aging is my contribution** and the fully
realised, production-grade AI module (real model, real preprocessing, real
explainability). The remaining axes (1–3, 5–7), owned by teammates, are
UI-complete demonstration stubs that return illustrative data, showing how the
platform scales to other modalities.

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

```bash
cd backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Place the trained checkpoint at
`backend/axis4_brain_aging/ml/checkpoints/best_ref_b_dropout03_lr5e5.pth`.

> On Windows you can start both servers with one command from the repo root:
> `.\dev-all.ps1` (opens Django in a second window, Vite in the current one).

---

## Backend API

**Endpoint:** `POST /api/axis4-brain-aging/analyze/` — `multipart/form-data`

| Field | Meaning |
|-------|---------|
| `file` | Primary scan (see accepted formats below). |
| `file_analyze_pair` | *(optional)* the second half of an Analyze `.hdr`+`.img` pair. |
| `metadata` | JSON string, e.g. `{"age":68,"sex":"F"}` (or `{"demo":true}` to skip the file). |

**Accepted scan formats:** NIfTI (`.nii`, `.nii.gz`) · 2D slices
(`.png`, `.jpg`, `.jpeg`, `.webp`) · OASIS Analyze 7.5 raw (`.hdr` + `.img`,
zipped as matching `basename.hdr`/`basename.img`, or via both upload slots).

**Response** — a JSON `AnalysisResult` consumed directly by the frontend:

| Field | Meaning |
|-------|---------|
| `predictedClass` | e.g. `"Predicted brain age: 54.1 years"` |
| `metrics[]` | Predicted brain age, typical error (MAE), approximate range, and — if `age` was supplied — chronological age and **brain–chronology Δ**. |
| `confidence[]` | Soft three-way display keyed off the size of the age gap (illustrative, not calibrated probabilities). |
| `summary` | Point estimate + approximate band + Δ, in plain language. |
| `gradCamDataUrl` | Base64 PNG of the Grad-CAM overlay. |
| `regions[]` | Anterior / central / posterior contribution scores. |
| `caseId`, `axisId`, `generatedAt`, `disclaimer`, `modelLoaded` | Envelope fields. |

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
predictions.

---

## Patient email (optional)

The Axis 4 page can email an HTML summary of the analysis to the patient via
`POST /api/send-report-email/`. When the brain–chronology gap exceeds ~12 years,
the email automatically includes a reassuring, plain-language note recommending
the patient mention the result to their doctor.

By default Django uses the **console email backend** — nothing is delivered; the
full message is printed in the terminal running `runserver` (ideal for demos).
For real delivery, copy `backend/.env.example` to `backend/.env` and set SMTP
values (loaded automatically via `python-dotenv`):

```
DJANGO_EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=youraccount@gmail.com
EMAIL_HOST_PASSWORD=your_app_password
EMAIL_USE_TLS=1
DEFAULT_FROM_EMAIL=brAIn Demo <youraccount@gmail.com>
```

For Gmail, create an **App password** (Google Account → Security → 2-Step
Verification → App passwords).

---

## Tech stack

**Frontend:** React, TanStack Start, Vite, TypeScript
**Backend:** Python, Django, Django REST Framework
**ML:** PyTorch, torchvision, NumPy, OpenCV, nibabel (Grad-CAM, NIfTI/Analyze I/O)

---

## Limitations & disclaimer

This is a **research prototype** (student project). It is trained on a single
dataset family (OASIS), uses 2D slice-based modelling, and is **not FDA/CE-marked
and not a substitute for clinical diagnosis or radiology review**. Grad-CAM
explanations are qualitative. External validation and broader demographic
coverage would be required before any clinical use.
