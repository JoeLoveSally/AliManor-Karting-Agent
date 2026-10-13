"""Matched-architecture control-input ablations for frozen Tiny RGB experiments.

All variants use the same CNN+GRU weights/parameter count. Only the
*input tensor* differs: full uses [state, age, frame_dt]; vision_only
uses [0, 0, frame_dt]; state_only uses [state, 0, frame_dt]. This is a
training/evaluation helper, not a live ADB policy or control scheduler.
"""
from __future__ import annotations

import torch
from torch import nn

from karting_agent.model.tiny_temporal_policy import TinyTemporalPolicy

CONTROL_MASKS = {
    "full": (1.0, 1.0, 1.0),
    "vision_only": (0.0, 0.0, 1.0),
    "state_only": (1.0, 0.0, 1.0),
}


class TinyControlAblation(nn.Module):
    """Frozen variant selection with the same trainable architecture."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in CONTROL_MASKS:
            raise ValueError(f"unsupported Tiny control ablation: {variant}")
        self.variant = variant
        self.policy = TinyTemporalPolicy(image_channels=3)
        # Nonpersistent: controls behavior, not state_dict shape/weight count.
        self.register_buffer(
            "control_mask",
            torch.tensor(CONTROL_MASKS[variant], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, images: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        if control.ndim != 3 or control.shape[-1] != 3:
            raise ValueError("control must have shape [B,T,3]")
        return self.policy(images, control * self.control_mask.to(dtype=control.dtype))
