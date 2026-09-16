"""v4-C2: v3 control plus kart-relative auxiliary supervision."""

from __future__ import annotations

import torch
from torch import nn

from karting_agent.model.base import Architecture, build_model


class KartRelativeStateConditionedPolicy(nn.Module):
    """Keep the v3 control path and add visual-only kart-relative heads."""

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
        if horizon_count < 1:
            raise ValueError("horizon_count must be >= 1")
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
        self.lateral_head = nn.Linear(visual_feature_dim, 1)
        self.heading_error_head = nn.Linear(visual_feature_dim, 2)
        self.edge_risk_head = nn.Linear(visual_feature_dim, 1)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = current_pressed.reshape(-1).to(dtype=torch.long)
        if states.shape[0] != visual_features.shape[0]:
            raise ValueError("current_pressed batch size does not match visual input")
        if torch.any((states < 0) | (states > 1)):
            raise ValueError("current_pressed values must be 0 or 1")

        state_features = self.state_embedding(states)
        conditioned = torch.cat((visual_features, state_features), dim=1)
        switch_logits = self.switch_head(conditioned)
        future_action_logits = self.future_action_head(visual_features)
        lateral = self.lateral_head(visual_features).squeeze(1)
        heading_vector = self.heading_error_head(visual_features)
        edge_risk_logit = self.edge_risk_head(visual_features).squeeze(1)
        return (
            switch_logits,
            future_action_logits,
            lateral,
            heading_vector,
            edge_risk_logit,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        current_pressed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward_from_features(self.encode_visual(inputs), current_pressed)


def build_kart_relative_model(
    architecture: Architecture = "mobilenet_v3_small",
    *,
    frame_stack: int = 5,
    pretrained: bool = True,
    horizon_count: int = 3,
    visual_feature_dim: int = 128,
    state_embedding_dim: int = 8,
    hidden_dim: int = 128,
) -> KartRelativeStateConditionedPolicy:
    return KartRelativeStateConditionedPolicy(
        architecture,
        frame_stack=frame_stack,
        pretrained=pretrained,
        horizon_count=horizon_count,
        visual_feature_dim=visual_feature_dim,
        state_embedding_dim=state_embedding_dim,
        hidden_dim=hidden_dim,
    )
