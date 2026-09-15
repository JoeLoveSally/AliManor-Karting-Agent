"""v4-C1: preserve v3 control while adding road-axis auxiliary supervision."""

from __future__ import annotations

import torch
from torch import nn

from karting_agent.model.base import Architecture, build_model


class AxisSupervisedStateConditionedPolicy(nn.Module):
    """v3 KEEP/SWITCH policy plus an auxiliary axial-orientation head."""

    def __init__(
        self,
        architecture: Architecture = "mobilenet_v3_small",
        *,
        frame_stack: int = 5,
        pretrained: bool = True,
        horizon_count: int = 3,
        visual_feature_dim: int = 128,
        state_embedding_dim: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.horizon_count = horizon_count
        self.visual_feature_dim = visual_feature_dim
        self.visual_encoder = build_model(
            architecture,
            frame_stack=frame_stack,
            pretrained=pretrained,
            output_dim=visual_feature_dim,
        )
        self.state_embedding = nn.Embedding(2, state_embedding_dim)
        self.switch_head = nn.Sequential(
            nn.Linear(visual_feature_dim + state_embedding_dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, horizon_count),
        )
        self.future_action_head = nn.Linear(visual_feature_dim, horizon_count)
        self.axis_head = nn.Linear(visual_feature_dim, 2)

    def encode_visual(self, inputs: torch.Tensor) -> torch.Tensor:
        visual_features = self.visual_encoder(inputs)
        if visual_features.ndim != 2:
            raise ValueError(
                "visual encoder must return BxF features, got "
                f"{tuple(visual_features.shape)}"
            )
        return visual_features

    def forward_from_features(
        self,
        visual_features: torch.Tensor,
        current_pressed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = current_pressed.reshape(-1).to(dtype=torch.long)
        if states.shape[0] != visual_features.shape[0]:
            raise ValueError("current_pressed batch size does not match visual input")
        if torch.any((states < 0) | (states > 1)):
            raise ValueError("current_pressed values must be 0 or 1")

        state_features = self.state_embedding(states)
        conditioned = torch.cat((visual_features, state_features), dim=1)
        switch_logits = self.switch_head(conditioned)
        future_action_logits = self.future_action_head(visual_features)
        axis_vector = self.axis_head(visual_features)
        return switch_logits, future_action_logits, axis_vector

    def forward(
        self,
        inputs: torch.Tensor,
        current_pressed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_from_features(self.encode_visual(inputs), current_pressed)


def build_axis_supervised_model(
    architecture: Architecture = "mobilenet_v3_small",
    *,
    frame_stack: int = 5,
    pretrained: bool = True,
    horizon_count: int = 3,
    visual_feature_dim: int = 128,
    state_embedding_dim: int = 8,
    hidden_dim: int = 128,
) -> AxisSupervisedStateConditionedPolicy:
    return AxisSupervisedStateConditionedPolicy(
        architecture,
        frame_stack=frame_stack,
        pretrained=pretrained,
        horizon_count=horizon_count,
        visual_feature_dim=visual_feature_dim,
        state_embedding_dim=state_embedding_dim,
        hidden_dim=hidden_dim,
    )
