"""EfficientNet-B0 regression head matching `best_ref_b_dropout03_lr5e5.pth` (dropout 0.3 + linear).

The backbone is built with ``weights=None``: the complete trained state dict is
loaded from the local checkpoint, so no ImageNet weights are downloaded at
startup.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_brain_age_efficientnet() -> nn.Module:
    """EfficientNet-B0 backbone (no pretrained download) with the refinement-B head."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(in_features, 1),
    )
    return model
