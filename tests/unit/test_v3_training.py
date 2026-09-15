from pathlib import Path

import numpy as np
import pytest

from karting_agent.train.dataset import DatasetConfig, DatasetSample, build_samples
from karting_agent.train.labels.touch_marker import (
    ActionTimeline,
    segments_to_events,
    states_to_segments,
)
from karting_agent.train.state_conditioned_dataset import StateConditionedVideoDataset


def make_timeline(states: list[bool], fps: float = 100.0) -> ActionTimeline:
    segments = states_to_segments(states, fps)
    return ActionTimeline(
        source="test.mp4",
        fps=fps,
        frame_count=len(states),
        duration_ms=len(states) * 1000 / fps,
        events=segments_to_events(segments),
        segments=segments,
        cleaned_frames=0,
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
        current_pressed=True,
    )


def test_v3_samples_record_current_action() -> None:
    states = [False] * 50 + [True] * 35 + [False] * 20 + [True] * 45
    timeline = make_timeline(states)
    config = DatasetConfig(
        sample_fps=10,
        frame_stack=5,
        history_ms=200,
        frame_interval_ms=50,
        prediction_horizon_ms=100,
        prediction_horizons_ms=(100, 200, 300),
        transition_window_ms=100,
    )

    samples = build_samples(timeline, "test.mp4", config)
    sample = next(
        item for item in samples if abs(item.input_timestamps_ms[-1] - 700.0) < 1e-6
    )

    assert sample.current_pressed is True
    assert sample.target_pressed_by_horizon == (True, False, False)


def test_counterfactual_dataset_emits_both_conditioning_states(tmp_path: Path) -> None:
    dataset = StateConditionedVideoDataset(
        [make_sample()],
        project_root=tmp_path,
        counterfactual_states=True,
    )
    try:
        assert len(dataset) == 2
        assert dataset._resolve_index(0) == (0, False)
        assert dataset._resolve_index(1) == (0, True)
    finally:
        dataset.close()


def test_counterfactual_switch_targets_are_complements(tmp_path: Path) -> None:
    class FakeBase:
        def __getitem__(self, index: int) -> dict[str, object]:
            assert index == 0
            return {
                "input": np.zeros((15, 224, 224), dtype=np.float32),
                "target": np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
            }

        def close(self) -> None:
            pass

    dataset = StateConditionedVideoDataset(
        [make_sample()],
        project_root=tmp_path,
        counterfactual_states=True,
    )
    dataset.base.close()
    dataset.base = FakeBase()  # type: ignore[assignment]
    try:
        conditioned_release = dataset[0]
        conditioned_press = dataset[1]

        assert conditioned_release["current_pressed"] == 0
        assert conditioned_press["current_pressed"] == 1
        np.testing.assert_array_equal(
            conditioned_release["switch_target"],
            np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            conditioned_press["switch_target"],
            np.asarray((0.0, 1.0, 1.0), dtype=np.float32),
        )
    finally:
        dataset.close()


def test_v3_model_outputs_switch_and_auxiliary_heads() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from karting_agent.model.state_conditioned import build_state_conditioned_model

    model = build_state_conditioned_model(
        "mobilenet_v3_small",
        frame_stack=5,
        pretrained=False,
        horizon_count=3,
        visual_feature_dim=32,
        state_embedding_dim=4,
        hidden_dim=16,
    )
    switch_logits, future_logits = model(
        torch.zeros(2, 15, 224, 224),
        torch.tensor([0, 1]),
    )

    assert switch_logits.shape == (2, 3)
    assert future_logits.shape == (2, 3)
