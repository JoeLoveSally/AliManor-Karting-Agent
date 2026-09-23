"""Frozen V4-C2 control path with a zero-initialized action-history correction."""

from __future__ import annotations

import torch
from torch import nn


class ActionHistoryResidualPolicy(nn.Module):
    """Train only the residual; at initialization logits equal the base model."""

    def __init__(self, base: nn.Module, *, history_width: int, hidden_dim: int = 64):
        super().__init__()
        if history_width < 1 or hidden_dim < 1:
            raise ValueError("invalid residual dimensions")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        visual_dim = int(base.visual_feature_dim)
        state_dim = int(base.state_embedding.embedding_dim)
        horizon_count = int(base.horizon_count)
        self.residual = nn.Sequential(
            nn.Linear(visual_dim + state_dim + history_width, hidden_dim),
            nn.Hardswish(),
            nn.Linear(hidden_dim, horizon_count),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.history_width = history_width

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()  # Retain the baseline's BatchNorm and dropout behavior.
        return self

    def forward(self, images, current_pressed, action_history):
        if action_history.ndim != 2 or action_history.shape[1] != self.history_width:
            raise ValueError("action history has incorrect batch shape or feature width")
        with torch.no_grad():
            feature = self.base.visual_encoder(images)
            state_feature = self.base.state_embedding(current_pressed.reshape(-1).long())
            baseline = self.base.switch_head(torch.cat((feature, state_feature), dim=1))
        conditioned = torch.cat((feature, state_feature, action_history), dim=1)
        corrected = baseline + self.residual(conditioned)
        return corrected, baseline
