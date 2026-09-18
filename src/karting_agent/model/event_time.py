"""V4-C4 policy with current-action and first-transition-time heads."""

from __future__ import annotations

import torch
from torch import nn

from karting_agent.model.base import Architecture, build_model


class EventTimePolicy(nn.Module):
    """Predict desired action now and the next expert transition time."""

    def __init__(
        self,
        architecture: Architecture = "mobilenet_v3_small",
        *,
        frame_stack: int = 5,
        pretrained: bool = True,
        event_time_classes: int = 7,
        visual_feature_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if event_time_classes < 2:
            raise ValueError("event_time_classes must be >= 2")
        if visual_feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature dimensions must be >= 1")

        self.event_time_classes = event_time_classes
        self.visual_feature_dim = visual_feature_dim
        self.visual_encoder = build_model(
            architecture,
            frame_stack=frame_stack,
            pretrained=pretrained,
            output_dim=visual_feature_dim,
        )
        self.current_action_head = nn.Linear(visual_feature_dim, 1)
        self.event_time_head = nn.Sequential(
            nn.Linear(visual_feature_dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, event_time_classes),
        )
        self.lateral_head = nn.Linear(visual_feature_dim, 1)
        self.heading_error_head = nn.Linear(visual_feature_dim, 2)
        self.edge_risk_head = nn.Linear(visual_feature_dim, 1)

    def encode_visual(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.visual_encoder(inputs)
        if features.ndim != 2:
            raise ValueError(
                "visual encoder must return BxF features, got "
                f"{tuple(features.shape)}"
            )
        return features

    def forward_from_features(
        self,
        features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        current_action_logit = self.current_action_head(features).squeeze(1)
        event_time_logits = self.event_time_head(features)
        lateral = self.lateral_head(features).squeeze(1)
        heading_vector = self.heading_error_head(features)
        edge_risk_logit = self.edge_risk_head(features).squeeze(1)
        return (
            current_action_logit,
            event_time_logits,
            lateral,
            heading_vector,
            edge_risk_logit,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return self.forward_from_features(self.encode_visual(inputs))


def build_event_time_model(
    architecture: Architecture = "mobilenet_v3_small",
    *,
    frame_stack: int = 5,
    pretrained: bool = True,
    event_time_classes: int = 7,
    visual_feature_dim: int = 128,
    hidden_dim: int = 128,
) -> EventTimePolicy:
    return EventTimePolicy(
        architecture,
        frame_stack=frame_stack,
        pretrained=pretrained,
        event_time_classes=event_time_classes,
        visual_feature_dim=visual_feature_dim,
        hidden_dim=hidden_dim,
    )
