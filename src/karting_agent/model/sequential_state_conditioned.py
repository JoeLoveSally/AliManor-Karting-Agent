"""Per-frame CNN + GRU state-conditioned transition policy used by v4-A."""

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


class SingleFrameEncoder(nn.Module):
    """Encode one RGB frame into a compact visual feature vector."""

    def __init__(
        self,
        architecture: Architecture,
        *,
        pretrained: bool,
        output_dim: int,
    ) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("output_dim must be >= 1")

        if architecture == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            self.features = backbone.features
            self.pool = backbone.avgpool
            in_features = backbone.classifier[0].in_features
        elif architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.pool = nn.Identity()
            in_features = backbone.fc.in_features
        else:
            raise ValueError(f"unsupported architecture: {architecture}")

        self.projection = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_features, output_dim),
            nn.Hardswish(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(
                "single-frame encoder expects Bx3xHxW input, got "
                f"{tuple(inputs.shape)}"
            )
        features = self.features(inputs)
        features = self.pool(features)
        return self.projection(features)


class SequentialStateConditionedPolicy(nn.Module):
    """Encode frames independently, model time with GRU, then predict KEEP/SWITCH."""

    def __init__(
        self,
        architecture: Architecture = "mobilenet_v3_small",
        *,
        pretrained: bool = True,
        frame_stack: int = 5,
        horizon_count: int = 3,
        visual_feature_dim: int = 128,
        gru_hidden_dim: int = 128,
        gru_layers: int = 1,
        state_embedding_dim: int = 8,
        policy_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        if horizon_count < 1:
            raise ValueError("horizon_count must be >= 1")
        if visual_feature_dim < 1:
            raise ValueError("visual_feature_dim must be >= 1")
        if gru_hidden_dim < 1:
            raise ValueError("gru_hidden_dim must be >= 1")
        if gru_layers < 1:
            raise ValueError("gru_layers must be >= 1")
        if state_embedding_dim < 1:
            raise ValueError("state_embedding_dim must be >= 1")
        if policy_hidden_dim < 1:
            raise ValueError("policy_hidden_dim must be >= 1")

        self.frame_stack = frame_stack
        self.horizon_count = horizon_count
        self.visual_feature_dim = visual_feature_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_layers = gru_layers
        self.state_embedding_dim = state_embedding_dim

        self.visual_encoder = SingleFrameEncoder(
            architecture,
            pretrained=pretrained,
            output_dim=visual_feature_dim,
        )
        self.temporal_model = nn.GRU(
            input_size=visual_feature_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.state_embedding = nn.Embedding(2, state_embedding_dim)
        self.switch_head = nn.Sequential(
            nn.Linear(gru_hidden_dim + state_embedding_dim, policy_hidden_dim),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(policy_hidden_dim, horizon_count),
        )
        self.future_action_head = nn.Linear(gru_hidden_dim, horizon_count)

    def encode_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return ordered per-frame visual features as BxTxF."""
        if inputs.ndim != 5:
            raise ValueError(
                "v4-A expects BxTx3xHxW input, got " f"{tuple(inputs.shape)}"
            )
        batch, time, channels, height, width = inputs.shape
        if time != self.frame_stack:
            raise ValueError(
                f"expected {self.frame_stack} frames, got temporal length {time}"
            )
        if channels != 3:
            raise ValueError("v4-A expects RGB frames with 3 channels")

        flattened = inputs.reshape(batch * time, channels, height, width)
        encoded = self.visual_encoder(flattened)
        return encoded.reshape(batch, time, self.visual_feature_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        current_pressed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.encode_sequence(inputs)
        temporal_output, _ = self.temporal_model(sequence)
        temporal_features = temporal_output[:, -1, :]

        states = current_pressed.reshape(-1).to(dtype=torch.long)
        if states.shape[0] != temporal_features.shape[0]:
            raise ValueError("current_pressed batch size does not match visual input")
        if torch.any((states < 0) | (states > 1)):
            raise ValueError("current_pressed values must be 0 or 1")

        state_features = self.state_embedding(states)
        conditioned = torch.cat((temporal_features, state_features), dim=1)
        switch_logits = self.switch_head(conditioned)
        future_action_logits = self.future_action_head(temporal_features)
        return switch_logits, future_action_logits


def build_sequential_state_conditioned_model(
    architecture: Architecture = "mobilenet_v3_small",
    *,
    pretrained: bool = True,
    frame_stack: int = 5,
    horizon_count: int = 3,
    visual_feature_dim: int = 128,
    gru_hidden_dim: int = 128,
    gru_layers: int = 1,
    state_embedding_dim: int = 8,
    policy_hidden_dim: int = 128,
) -> SequentialStateConditionedPolicy:
    return SequentialStateConditionedPolicy(
        architecture,
        pretrained=pretrained,
        frame_stack=frame_stack,
        horizon_count=horizon_count,
        visual_feature_dim=visual_feature_dim,
        gru_hidden_dim=gru_hidden_dim,
        gru_layers=gru_layers,
        state_embedding_dim=state_embedding_dim,
        policy_hidden_dim=policy_hidden_dim,
    )
