from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower().strip()
    if name == "vgg16":
        model = models.vgg16(weights=None)
    elif name == "vgg16_bn":
        model = models.vgg16_bn(weights=None)
    else:
        raise ValueError(f"Unsupported model.name: {name}")

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, int(num_classes))
    return model
