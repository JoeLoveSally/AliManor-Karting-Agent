from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from karting_agent.model.temporal_delta import (
    current_rgb_plus_adjacent_deltas_numpy,
    current_rgb_plus_adjacent_deltas_torch,
    transform_temporal_input_numpy,
    validate_temporal_input_representation,
)


def _stack(values: list[float]) -> np.ndarray:
    frames = [
        np.full((3, 2, 2), value, dtype=np.float32)
        for value in values
    ]
    return np.concatenate(frames, axis=0)


def test_current_rgb_plus_adjacent_deltas_numpy() -> None:
    inputs = _stack([1.0, 2.0, 4.0, 7.0, 11.0])

    transformed = current_rgb_plus_adjacent_deltas_numpy(
        inputs,
        frame_stack=5,
    )
    groups = transformed.reshape(5, 3, 2, 2)

    assert np.allclose(groups[0], 11.0)
    assert np.allclose(groups[1], 1.0)
    assert np.allclose(groups[2], 2.0)
    assert np.allclose(groups[3], 3.0)
    assert np.allclose(groups[4], 4.0)


def test_numpy_and_torch_temporal_delta_transforms_match() -> None:
    import torch

    inputs = _stack([0.0, 0.5, 0.25, 1.0, 0.75])
    expected = current_rgb_plus_adjacent_deltas_numpy(
        inputs,
        frame_stack=5,
    )

    actual = current_rgb_plus_adjacent_deltas_torch(
        torch.from_numpy(inputs).unsqueeze(0),
        frame_stack=5,
    )[0].numpy()

    assert np.allclose(actual, expected)


def test_raw_temporal_representation_is_identity() -> None:
    inputs = _stack([0.0, 1.0, 2.0, 3.0, 4.0])

    transformed = transform_temporal_input_numpy(
        inputs,
        frame_stack=5,
        representation="raw_rgb_stack",
    )

    assert transformed is inputs


def test_temporal_delta_requires_valid_channel_count() -> None:
    with pytest.raises(ValueError, match="3 \* frame_stack"):
        current_rgb_plus_adjacent_deltas_numpy(
            np.zeros((12, 2, 2), dtype=np.float32),
            frame_stack=5,
        )


def test_validate_temporal_input_representation_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported temporal input representation"):
        validate_temporal_input_representation("bad")
