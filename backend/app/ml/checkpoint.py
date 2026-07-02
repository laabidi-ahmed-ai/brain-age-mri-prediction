"""Load the trained brain-age model from a local checkpoint (no network access)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .architecture import build_brain_age_efficientnet


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_brain_age_model(checkpoint_path: Path) -> nn.Module:
    """Build EfficientNet-B0 (weights=None) and load the complete local state dict.

    Raises if the file is missing or the state dict does not match — callers
    decide how to surface that (the model service treats it as "unavailable").
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_brain_age_efficientnet()
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch without weights_only kwarg
        state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(pick_device())
    model.eval()
    return model
