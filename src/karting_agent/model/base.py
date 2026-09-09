"""Temporal image-classification model factory."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

Architecture = Literal["mobilenet_v3_small", "resnet18"]


class BinaryTemporalClassifier(nn.Module):
    """Wrap a torchvision classifier and return one binary logit per sample."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs).squeeze(-1)


def _expand_input_conv(conv: nn.Conv2d, frame_stack: int) -> nn.Conv2d:
    if frame_stack < 1:
        raise ValueError("frame_stack must be >= 1")
    if conv.in_channels != 3:
        raise ValueError("expected a 3-channel pretrained input convolution")

    expanded = nn.Conv2d(
        in_channels=3 * frame_stack,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        expanded.weight.copy_(
            conv.weight.repeat(1, frame_stack, 1, 1) / frame_stack
        )
        if conv.bias is not None and expanded.bias is not None:
            expanded.bias.copy_(conv.bias)

    return expanded


def build_model(
    architecture: Architecture = "mobilenet_v3_small",
    *,
    frame_stack: int = 3,
    pretrained: bool = True,
) -> BinaryTemporalClassifier:
    """Build a binary classifier with ``3 * frame_stack`` input channels."""
    if architecture == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        first_conv = backbone.features[0][0]
        backbone.features[0][0] = _expand_input_conv(first_conv, frame_stack)
        last_linear = backbone.classifier[-1]
        backbone.classifier[-1] = nn.Linear(last_linear.in_features, 1)
        return BinaryTemporalClassifier(backbone)

    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.conv1 = _expand_input_conv(backbone.conv1, frame_stack)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        return BinaryTemporalClassifier(backbone)

    raise ValueError(f"unsupported architecture: {architecture}")


@torch.inference_mode()
def predict_probability(
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Return PRESS probabilities for one batch."""
    return torch.sigmoid(model(inputs))
