"""Small experimental RGB or RGB+HSV temporal PRESS-state predictor.

No runtime/ADB dependencies. The model scores expert state at a future
timestamp, NOT an H100/H200/H300 monotone event-time distribution.
"""
from __future__ import annotations

import torch
from torch import nn


class TinyTemporalPolicy(nn.Module):
    """Per-frame shared CNN + GRU; no pretrained weights."""

    def __init__(self, *, image_channels: int = 3,
                 control_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        if image_channels not in (3, 4) or control_dim != 3 or hidden_dim < 1:
            raise ValueError("expected RGB/RGB+HSV, three causal control features")
        self.image_channels = image_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.gru = nn.GRU(32 + control_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, images: torch.Tensor,
                control_features: torch.Tensor) -> torch.Tensor:
        """images [B,T,C,H,W], control_features [B,T,3] -> logits [B]."""
        if images.ndim != 5 or images.shape[2] != self.image_channels:
            raise ValueError("images must be [B,T,C,H,W] with expected channels")
        if (control_features.ndim != 3 or
                control_features.shape[:2] != images.shape[:2] or
                control_features.shape[2] != 3):
            raise ValueError("control_features must be [B,T,3]")
        batch, steps, channels, height, width = images.shape
        frame_features = self.encoder(
            images.reshape(batch * steps, channels, height, width)
        ).reshape(batch, steps, 32)
        sequence = torch.cat((frame_features, control_features), dim=-1)
        sequence_features, _ = self.gru(sequence)
        return self.output(sequence_features[:, -1, :]).squeeze(1)
