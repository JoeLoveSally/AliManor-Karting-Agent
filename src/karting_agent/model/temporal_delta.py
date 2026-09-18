"""Temporal input representations for efficient early-fusion policies."""

from __future__ import annotations

from typing import Literal

import numpy as np


TemporalInputRepresentation = Literal[
    "raw_rgb_stack",
    "current_rgb_plus_adjacent_deltas",
]

SUPPORTED_TEMPORAL_INPUT_REPRESENTATIONS: tuple[TemporalInputRepresentation, ...] = (
    "raw_rgb_stack",
    "current_rgb_plus_adjacent_deltas",
)


def validate_temporal_input_representation(value: str) -> TemporalInputRepresentation:
    if value not in SUPPORTED_TEMPORAL_INPUT_REPRESENTATIONS:
        expected = ", ".join(SUPPORTED_TEMPORAL_INPUT_REPRESENTATIONS)
        raise ValueError(
            f"unsupported temporal input representation {value!r}; "
            f"expected one of: {expected}"
        )
    return value  # type: ignore[return-value]


def current_rgb_plus_adjacent_deltas_numpy(
    inputs: np.ndarray,
    *,
    frame_stack: int,
) -> np.ndarray:
    """Convert normalized RGB history into current RGB plus ordered frame deltas."""

    if frame_stack < 2:
        raise ValueError("frame_stack must be >= 2 for temporal deltas")
    if inputs.ndim != 3 or inputs.shape[0] != 3 * frame_stack:
        raise ValueError(
            "inputs must have shape (3 * frame_stack, H, W); "
            f"got {inputs.shape}"
        )
    frames = inputs.reshape(frame_stack, 3, *inputs.shape[1:])
    current = frames[-1]
    deltas = frames[1:] - frames[:-1]
    transformed = np.concatenate(
        (current, deltas.reshape(-1, *inputs.shape[1:])),
        axis=0,
    )
    return np.ascontiguousarray(transformed, dtype=inputs.dtype)


def current_rgb_plus_adjacent_deltas_torch(
    inputs,
    *,
    frame_stack: int,
):
    """Torch equivalent of the NumPy temporal-delta transform."""

    if frame_stack < 2:
        raise ValueError("frame_stack must be >= 2 for temporal deltas")
    if inputs.ndim != 4 or inputs.shape[1] != 3 * frame_stack:
        raise ValueError(
            "inputs must have shape (B, 3 * frame_stack, H, W); "
            f"got {tuple(inputs.shape)}"
        )
    import torch

    frames = inputs.reshape(
        inputs.shape[0],
        frame_stack,
        3,
        inputs.shape[2],
        inputs.shape[3],
    )
    current = frames[:, -1]
    deltas = frames[:, 1:] - frames[:, :-1]
    return torch.cat(
        (
            current,
            deltas.reshape(
                inputs.shape[0],
                -1,
                inputs.shape[2],
                inputs.shape[3],
            ),
        ),
        dim=1,
    )


def transform_temporal_input_numpy(
    inputs: np.ndarray,
    *,
    frame_stack: int,
    representation: str,
) -> np.ndarray:
    resolved = validate_temporal_input_representation(representation)
    if resolved == "raw_rgb_stack":
        return inputs
    return current_rgb_plus_adjacent_deltas_numpy(
        inputs,
        frame_stack=frame_stack,
    )


def transform_temporal_input_torch(
    inputs,
    *,
    frame_stack: int,
    representation: str,
):
    resolved = validate_temporal_input_representation(representation)
    if resolved == "raw_rgb_stack":
        return inputs
    return current_rgb_plus_adjacent_deltas_torch(
        inputs,
        frame_stack=frame_stack,
    )


def initialize_current_rgb_delta_first_conv(
    model,
    *,
    architecture: str,
    frame_stack: int,
) -> None:
    """Initialize current RGB with pretrained weights and delta groups at zero."""

    if frame_stack < 2:
        raise ValueError("frame_stack must be >= 2 for temporal deltas")

    visual_encoder = getattr(model, "visual_encoder", None)
    backbone = getattr(visual_encoder, "backbone", None)
    if backbone is None:
        raise ValueError("model does not expose visual_encoder.backbone")

    if architecture == "mobilenet_v3_small":
        conv = backbone.features[0][0]
    elif architecture == "resnet18":
        conv = backbone.conv1
    else:
        raise ValueError(
            "temporal-delta initialization supports mobilenet_v3_small "
            "and resnet18"
        )

    if int(conv.in_channels) != 3 * frame_stack:
        raise ValueError(
            "first convolution input channels do not match frame_stack"
        )

    import torch

    with torch.no_grad():
        rgb_weights = conv.weight[:, :3].clone() * frame_stack
        conv.weight.zero_()
        conv.weight[:, :3].copy_(rgb_weights)
