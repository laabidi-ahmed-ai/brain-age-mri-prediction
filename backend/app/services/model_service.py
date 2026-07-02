"""Model service: owns the loaded brain-age model and runs inference.

Constructed once at application startup (see ``app.main`` lifespan) and stored on
``app.state``. Inference is CPU/GPU-bound and synchronous; routes run it in a
threadpool so the async event loop is never blocked.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from ..logging_config import get_logger
from ..ml import inference
from ..ml.checkpoint import load_brain_age_model

logger = get_logger("brain.model")


class ModelService:
    """Wraps the (optional) loaded model and exposes availability + inference."""

    def __init__(self, model: Optional[Any] = None):
        self._model = model

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: Path) -> "ModelService":
        """Try to load the checkpoint; on any failure return an unavailable service.

        Never raises — a missing/invalid checkpoint degrades to demo-only mode
        rather than crashing startup.
        """
        started = time.perf_counter()
        if not Path(checkpoint_path).exists():
            logger.warning("model_load skipped reason=checkpoint_missing path=%s", checkpoint_path)
            return cls(model=None)
        try:
            model = load_brain_age_model(checkpoint_path)
        except Exception as exc:  # noqa: BLE001 — keep the API up without the model
            logger.error("model_load failed error_type=%s", type(exc).__name__)
            return cls(model=None)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("model_load ok duration_ms=%.0f", elapsed_ms)
        return cls(model=model)

    def run_inference(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: dict,
        *,
        pair_bytes: Optional[bytes] = None,
        pair_name: str = "",
    ) -> dict:
        """Real inference. Raises ``inference.PreprocessingError`` on bad input."""
        if self._model is None:
            raise RuntimeError("run_inference called with no model loaded")
        return inference.run_inference(
            self._model,
            file_bytes,
            filename,
            metadata,
            pair_bytes=pair_bytes,
            pair_name=pair_name,
        )
