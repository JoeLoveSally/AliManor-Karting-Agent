from pathlib import Path

import numpy as np
import pytest

from karting_agent.train.dataset import DatasetSample
from karting_agent.train.state_conditioned_dataset import (
    SequentialStateConditionedVideoDataset,
)


def make_sample() -> DatasetSample:
    return DatasetSample(
        video="test.mp4",
        input_frame_indices=(0, 1, 2, 3, 4),
        input_timestamps_ms=(0.0, 50.0, 100.0, 150.0, 200.0),
        target_frame_index=5,
        target_timestamp_ms=300.0,
        target_pressed=True,
        transition_distance_ms=0.0,
        near_transition=True,
        near_short_correction=False,
        target_frame_indices=(5, 6, 7),
        target_timestamps_ms=(300.0, 400.0, 500.0),
        target_pressed_by_horizon=(True, False, False),
        current_pressed=False,
    )


def test_v4a_dataset_keeps_time_axis_explicit(tmp_path: Path) -> None:
    class FakeConditionedDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, object]:
            assert index == 0
            stacked = np.arange(15 * 4 * 6, dtype=np.float32).reshape(15, 4, 6)
            return {
                "input": stacked,
                "current_pressed": np.int64(0),
                "switch_target": np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
                "future_action_target": np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
            }

        def close(self) -> None:
            pass

    dataset = SequentialStateConditionedVideoDataset(
        [make_sample()],
        frame_stack=5,
        project_root=tmp_path,
    )
    dataset.base.close()
    dataset.base = FakeConditionedDataset()  # type: ignore[assignment]
    try:
        item = dataset[0]
        sequence = item["input"]
        assert isinstance(sequence, np.ndarray)
        assert sequence.shape == (5, 3, 4, 6)
        np.testing.assert_array_equal(sequence[0], np.arange(3 * 4 * 6).reshape(3, 4, 6))
    finally:
        dataset.close()


def test_v4a_model_encodes_each_frame_before_gru() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from karting_agent.model.sequential_state_conditioned import (
        build_sequential_state_conditioned_model,
    )

    torch.manual_seed(7)
    model = build_sequential_state_conditioned_model(
        "mobilenet_v3_small",
        pretrained=False,
        frame_stack=5,
        horizon_count=3,
        visual_feature_dim=16,
        gru_hidden_dim=12,
        state_embedding_dim=4,
        policy_hidden_dim=8,
    ).eval()
    inputs = torch.randn(2, 5, 3, 64, 64)
    captured_gru_inputs = []

    def capture_gru_input(_module, args) -> None:
        captured_gru_inputs.append(args[0].detach().clone())

    hook = model.temporal_model.register_forward_pre_hook(capture_gru_input)
    try:
        with torch.inference_mode():
            features = model.encode_sequence(inputs)
            reversed_features = model.encode_sequence(inputs.flip(1))
            switch_logits, future_logits = model(inputs, torch.tensor([0, 1]))
            reversed_switch, reversed_future = model(
                inputs.flip(1), torch.tensor([0, 1])
            )
    finally:
        hook.remove()

    assert features.shape == (2, 5, 16)
    assert torch.allclose(reversed_features, features.flip(1), atol=1e-6, rtol=1e-5)
    assert switch_logits.shape == (2, 3)
    assert future_logits.shape == (2, 3)
    assert reversed_switch.shape == (2, 3)
    assert reversed_future.shape == (2, 3)

    # The test is structural: verify that the GRU receives the ordered feature
    # sequence and that reversing input frames reverses that explicit time axis.
    # Random, untrained logits are not required to differ numerically.
    assert len(captured_gru_inputs) == 2
    assert torch.allclose(captured_gru_inputs[0], features, atol=1e-6, rtol=1e-5)
    assert torch.allclose(
        captured_gru_inputs[1], reversed_features, atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        captured_gru_inputs[1], captured_gru_inputs[0].flip(1), atol=1e-6, rtol=1e-5
    )
