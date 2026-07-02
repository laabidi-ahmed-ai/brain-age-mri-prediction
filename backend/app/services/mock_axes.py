"""Mock demonstration payloads for the non-Axis-4 axes.

Axis 4 is the only real ML module. The frontend still POSTs to the other axes'
``/analyze/`` endpoints whenever ``VITE_API_BASE_URL`` is set, so these endpoints
return the same deterministic mock ``AnalysisResult`` shapes the previous backend
produced — preserving navigation and demonstrations without any real model.
"""
from __future__ import annotations

from typing import Callable

from ..utils import make_signal, make_timeline

_DISCLAIMER = (
    "Decision-support output. Not a standalone diagnosis — to be interpreted by a qualified clinician."
)


def _regions(rows: list[tuple[str, str, float]]) -> list[dict]:
    return [{"region": r, "side": s, "contribution": c} for r, s, c in rows]


def _metrics(rows: list[tuple[str, str]]) -> list[dict]:
    return [{"label": k, "value": v} for k, v in rows]


def _confidence(classes: list[str], probs: list[float]) -> list[dict]:
    return [{"label": c, "value": float(p)} for c, p in zip(classes, probs)]


def _axis1() -> dict:
    classes = ["Alzheimer's disease", "Vascular dementia", "Frontotemporal dementia", "Other"]
    probs = [0.62, 0.1267, 0.1267, 0.1267]
    return {
        "predictedClass": classes[0],
        "topConfidence": probs[0],
        "confidence": _confidence(classes, probs),
        "summary": "Atrophy pattern (medial temporal + posterior cingulate) most consistent with Alzheimer's disease.",
        "regions": _regions([
            ("Hippocampus", "L", 0.34), ("Hippocampus", "R", 0.31),
            ("Entorhinal cortex", "L", 0.22), ("Entorhinal cortex", "R", 0.2),
            ("Posterior cingulate", "B", 0.18), ("Precuneus", "B", 0.15),
        ]),
        "metrics": _metrics([
            ("Cortical thickness gap", "-0.42 mm"), ("Hippocampal volume %", "-18%"), ("MMSE-equiv", "21/30"),
        ]),
        "signal": make_signal(n=80, seed=21),
    }


def _axis2() -> dict:
    classes = ["Parkinson's disease", "MSA", "PSP", "Other atypical"]
    probs = [0.62, 0.1267, 0.1267, 0.1267]
    return {
        "predictedClass": classes[0],
        "topConfidence": probs[0],
        "confidence": _confidence(classes, probs),
        "summary": "Imaging signature compatible with idiopathic Parkinson's; atypical features below threshold.",
        "regions": _regions([
            ("Substantia nigra", "B", 0.38), ("Putamen", "L", 0.21), ("Putamen", "R", 0.19),
            ("Midbrain (SCP)", "B", 0.24), ("Pons", "B", 0.12),
        ]),
        "metrics": _metrics([
            ("MR Parkinsonism Index", "12.4"), ("SCP width", "2.1 mm"), ("Putaminal asymmetry", "8%"),
        ]),
        "signal": make_signal(n=80, seed=21),
    }


def _axis3() -> dict:
    classes = ["No cerebellar involvement", "Mild cerebellar involvement", "Marked cerebellar involvement"]
    probs = [0.62, 0.19, 0.19]
    return {
        "predictedClass": classes[0],
        "topConfidence": probs[0],
        "confidence": _confidence(classes, probs),
        "summary": "Mild cerebellar involvement detected in posterior lobules; clinical correlation advised.",
        "regions": _regions([
            ("Cerebellar lobule VI", "L", 0.27), ("Cerebellar lobule VI", "R", 0.25),
            ("Crus I", "L", 0.22), ("Crus II", "R", 0.18), ("Vermis", "B", 0.14),
        ]),
        "metrics": _metrics([
            ("Cerebellar volume %", "-6.4%"), ("SARA-equiv", "9/40"), ("Vermis atrophy", "Mild"),
        ]),
        "signal": make_signal(n=80, seed=21),
    }


def _axis5() -> dict:
    classes = ["Typical connectivity", "Compensatory pattern", "Disrupted connectivity"]
    probs = [0.62, 0.19, 0.19]
    return {
        "predictedClass": classes[0],
        "topConfidence": probs[0],
        "confidence": _confidence(classes, probs),
        "summary": "Compensatory frontoparietal recruitment suggests hidden cognitive effort.",
        "regions": _regions([
            ("Default mode network", "B", 0.3), ("Frontoparietal network", "L", 0.24),
            ("Salience network", "R", 0.2), ("DLPFC", "L", 0.16),
        ]),
        "metrics": _metrics([
            ("Network efficiency", "0.71"), ("DMN-FPN coupling", "+0.34"), ("Effort index", "High"),
        ]),
        "network": {
            "nodes": [
                {"id": "DMN", "label": "Default Mode"},
                {"id": "FPN", "label": "Frontoparietal"},
                {"id": "SAL", "label": "Salience"},
                {"id": "DAN", "label": "Dorsal Attention"},
            ],
            "edges": [
                {"source": "DMN", "target": "FPN", "weight": 0.71},
                {"source": "FPN", "target": "SAL", "weight": 0.58},
                {"source": "SAL", "target": "DAN", "weight": 0.44},
                {"source": "DMN", "target": "DAN", "weight": 0.22},
            ],
        },
        "signal": make_signal(n=120, seed=11),
    }


def _axis6() -> dict:
    classes = ["No anomaly", "Mild gait anomaly", "Tremor detected", "Postural instability"]
    probs = [0.62, 0.1267, 0.1267, 0.1267]
    return {
        "predictedClass": classes[0],
        "topConfidence": probs[0],
        "confidence": _confidence(classes, probs),
        "summary": "Resting tremor (~5 Hz) and mild gait asymmetry detected.",
        "regions": _regions([
            ("Right upper limb", "R", 0.32), ("Left lower limb", "L", 0.21), ("Trunk sway", "B", 0.18),
        ]),
        "metrics": _metrics([
            ("Stride variability", "12.4%"), ("Tremor freq", "5.2 Hz"), ("Postural sway", "Moderate"),
        ]),
        "timeline": make_timeline([
            (3, "Gait initiation hesitation", "moderate"),
            (11, "Tremor onset (right hand)", "high"),
            (22, "Postural sway", "moderate"),
        ]),
    }


def _axis7() -> dict:
    classes = ["Stable network", "Mild instability", "Vulnerable / pre-ictal pattern"]
    probs = [0.62, 0.19, 0.19]
    return {
        "predictedClass": classes[0],
        "topConfidence": probs[0],
        "confidence": _confidence(classes, probs),
        "summary": "Left-temporal instability windows suggest heightened epilepsy vulnerability.",
        "regions": _regions([
            ("Temporal channel T7", "L", 0.34), ("Temporal channel T8", "R", 0.21), ("Frontal Fp1", "L", 0.16),
        ]),
        "metrics": _metrics([
            ("Instability score", "0.72"), ("Spike rate", "4.1/min"), ("Network entropy", "1.81"),
        ]),
        "signal": make_signal(n=160, seed=42),
        "timeline": make_timeline([
            (12, "Spike train", "moderate"),
            (47, "Instability window", "high"),
            (98, "Quiet period", "low"),
        ]),
    }


MOCK_AXES: dict[str, Callable[[], dict]] = {
    "axis1-alzheimer-dementia": _axis1,
    "axis2-parkinson-atypical": _axis2,
    "axis3-cerebellar-dysfunction": _axis3,
    "axis5-functional-connectivity": _axis5,
    "axis6-neuromotor-video": _axis6,
    "axis7-epilepsy-network": _axis7,
}


def build_mock_result(axis_id: str) -> dict:
    """Return the mock ``AnalysisResult`` core for a non-Axis-4 axis (with disclaimer)."""
    payload = MOCK_AXES[axis_id]()
    payload.setdefault("disclaimer", _DISCLAIMER)
    return payload
