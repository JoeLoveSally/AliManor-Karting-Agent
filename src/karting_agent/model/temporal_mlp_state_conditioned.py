"""Per-frame CNN + ordered temporal MLP policy used by v4-A2."""

from __future__ import annotations

import torch
from torch import nn

from karting_agent.model.base import Architecture
from karting_agent.model.sequential_state_conditioned import SingleFrameEncoder


class TemporalMlpStateConditionedPolicy(nn.Module):
    """Keep the time axis explicit, then fuse ordered frame features with an MLP.

    v4-A2 deliberately removes recurrence from v4-A. Each frame is still encoded
    independently by one shared CNN, but the ordered feature sequence is flattened
    only after per-frame encoding. Temporal position is therefore preserved by the
    fixed concatenation order without introducing a recurrent hidden state.
    """

    def __init__(
        self,
        architecture: Architecture = "mobilenet_v3_small",
        *,
        pretrained: bool = True,
        frame_stack: int = 5,
        horizon_count: int = 3,
        visual_feature_dim: int = 128,
        temporal_hidden_dim: int = 128,
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
        if temporal_hidden_dim < 1:
            raise ValueError("temporal_hidden_dim must be >= 1")
        if state_embedding_dim < 1:
            raise ValueError("state_embedding_dim must be >= 1")
        if policy_hidden_dim < 1:
            raise ValueError("policy_hidden_dim must be >= 1")

        self.frame_stack = frame_stack
        self.horizon_count = horizon_count
        self.visual_feature_dim = visual_feature_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.state_embedding_dim = state_embedding_dim

        self.visual_encoder = SingleFrameEncoder(
            architecture,
            pretrained=pretrained,
            output_dim=visual_feature_dim,
        )
        self.temporal_fusion = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(frame_stack * visual_feature_dim, temporal_hidden_dim),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
        )
        self.state_embedding = nn.Embedding(2, state_embedding_dim)
        self.switch_head = nn.Sequential(
            nn.Linear(temporal_hidden_dim + state_embedding_dim, policy_hidden_dim),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(policy_hidden_dim, horizon_count),
        )
        self.future_action_head = nn.Linear(temporal_hidden_dim, horizon_count)

    def encode_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return ordered per-frame visual features as BxTxF."""
        if inputs.ndim != 5:
            raise ValueError(
                "v4-A2 expects BxTx3xHxW input, got " f"{tuple(inputs.shape)}"
            )
        batch, time, channels, height, width = inputs.shape
        if time != self.frame_stack:
            raise ValueError(
                f"expected {self.frame_stack} frames, got temporal length {time}"
            )
        if channels != 3:
            raise ValueError("v4-A2 expects RGB frames with 3 channels")

        flattened = inputs.reshape(batch * time, channels, height, width)
        encoded = self.visual_encoder(flattened)
        return encoded.reshape(batch, time, self.visual_feature_dim)

    def forward_from_features(
        self,
        sequence: torch.Tensor,
        current_pressed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict from precomputed ordered per-frame features."""
        if sequence.ndim != 3:
            raise ValueError(
                "feature sequence must be BxTxF, got " f"{tuple(sequence.shape)}"
            )
        if sequence.shape[1] != self.frame_stack:
            raise ValueError(
                f"expected {self.frame_stack} feature steps, got {sequence.shape[1]}"
            )
        if sequence.shape[2] != self.visual_feature_dim:
            raise ValueError(
                f"expected feature dim {self.visual_feature_dim}, got {sequence.shape[2]}"
            )

        temporal_features = self.temporal_fusion(sequence)
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

    def forward(
        self,
        inputs: torch.Tensor,
        current_pressed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.encode_sequence(inputs)
        return self.forward_from_features(sequence, current_pressed)


def build_temporal_mlp_state_conditioned_model(
    architecture: Architecture = "mobilenet_v3_small",
    *,
    pretrained: bool = True,
    frame_stack: int = 5,
    horizon_count: int = 3,
    visual_feature_dim: int = 128,
    temporal_hidden_dim: int = 128,
    state_embedding_dim: int = 8,
    policy_hidden_dim: int = 128,
) -> TemporalMlpStateConditionedPolicy:
    return TemporalMlpStateConditionedPolicy(
        architecture,
        pretrained=pretrained,
        frame_stack=frame_stack,
        horizon_count=horizon_count,
        visual_feature_dim=visual_feature_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        state_embedding_dim=state_embedding_dim,
        policy_hidden_dim=policy_hidden_dim,
    )
