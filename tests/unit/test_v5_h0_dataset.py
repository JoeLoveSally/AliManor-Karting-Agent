from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from karting_agent.train.dataset import DatasetConfig, DatasetSample, build_samples
from karting_agent.train.labels.touch_marker import (
    ActionTimeline,
    segments_to_events,
    states_to_segments,
)
from karting_agent.train import state_conditioned_dataset as state_dataset


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


def make_config(*, dense: bool) -> DatasetConfig:
    horizons = (
        (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
        if dense
        else (100.0, 200.0, 300.0)
    )
    return DatasetConfig(
        sample_fps=30.0,
        frame_stack=5,
        history_ms=200.0,
        frame_interval_ms=50.0,
        prediction_horizon_ms=horizons[0],
        prediction_horizons_ms=horizons,
        sampling_horizons_ms=(100.0, 200.0, 300.0) if dense else (),
        transition_window_ms=200.0,
        short_correction_min_ms=100.0,
        short_correction_max_ms=300.0,
    )


def test_dense_h0_targets_share_the_current_observation_frame() -> None:
    states = [False] * 50 + [True] * 20 + [False] * 15 + [True] * 115
    samples = build_samples(make_timeline(states), "test.mp4", make_config(dense=True))

    sample = samples[4]
    assert sample.target_timestamps_ms[0] == pytest.approx(
        sample.input_timestamps_ms[-1]
    )
    assert sample.target_frame_indices[0] == sample.input_frame_indices[-1]
    assert sample.target_pressed_by_horizon[0] is sample.current_pressed
    assert len(sample.target_pressed_by_horizon) == 7


def test_dense_horizons_preserve_v4_sampling_flags_and_shared_targets() -> None:
    # Include both ordinary transitions and a 150 ms RELEASE correction.
    states = [False] * 50 + [True] * 30 + [False] * 15 + [True] * 105
    timeline = make_timeline(states)
    baseline = build_samples(timeline, "test.mp4", make_config(dense=False))
    dense = build_samples(timeline, "test.mp4", make_config(dense=True))

    assert len(dense) == len(baseline)
    dense_horizons = make_config(dense=True).target_horizons_ms
    shared_indices = tuple(
        dense_horizons.index(value) for value in (100.0, 200.0, 300.0)
    )

    for old, new in zip(baseline, dense, strict=True):
        assert new.input_frame_indices == old.input_frame_indices
        assert new.input_timestamps_ms == old.input_timestamps_ms
        assert new.current_pressed is old.current_pressed
        assert new.near_transition is old.near_transition
        assert new.near_short_correction is old.near_short_correction
        assert tuple(new.target_pressed_by_horizon[index] for index in shared_indices) == (
            old.target_pressed_by_horizon
        )
        assert tuple(new.near_transition_by_horizon[index] for index in shared_indices) == (
            old.near_transition_by_horizon
        )
        assert tuple(
            new.near_short_correction_by_horizon[index] for index in shared_indices
        ) == old.near_short_correction_by_horizon


def test_counterfactual_h0_switch_targets_are_exact_complements(monkeypatch) -> None:
    class FakeTemporalVideoDataset:
        def __init__(self, samples, **kwargs) -> None:
            del samples, kwargs

        def __getitem__(self, index: int) -> dict[str, object]:
            del index
            return {
                "input": np.zeros((15, 8, 8), dtype=np.float32),
                "target": np.asarray([0, 0, 1, 1, 1, 1, 1], dtype=np.float32),
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        state_dataset,
        "TemporalVideoDataset",
        FakeTemporalVideoDataset,
    )
    sample = DatasetSample(
        video="test.mp4",
        input_frame_indices=(0, 5, 10, 15, 20),
        input_timestamps_ms=(0.0, 50.0, 100.0, 150.0, 200.0),
        target_frame_index=20,
        target_timestamp_ms=200.0,
        target_pressed=False,
        transition_distance_ms=None,
        near_transition=False,
        near_short_correction=False,
        target_frame_indices=(20, 25, 30, 35, 40, 45, 50),
        target_timestamps_ms=(200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0),
        target_pressed_by_horizon=(False, False, True, True, True, True, True),
        current_pressed=False,
    )
    dataset = state_dataset.StateConditionedVideoDataset(
        [sample],
        project_root=Path("."),
        counterfactual_states=True,
    )

    conditioned_release = dataset[0]
    conditioned_press = dataset[1]

    assert conditioned_release["current_pressed"] == 0
    assert conditioned_press["current_pressed"] == 1
    np.testing.assert_array_equal(
        conditioned_release["switch_target"],
        np.asarray([0, 0, 1, 1, 1, 1, 1], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        conditioned_press["switch_target"],
        np.asarray([1, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    )
    assert conditioned_release["switch_target"][0] == 0.0
    assert conditioned_press["switch_target"][0] == 1.0


def test_sampling_horizons_must_be_prediction_horizons() -> None:
    config = make_config(dense=True)
    invalid = DatasetConfig(
        **{
            **config.__dict__,
            "sampling_horizons_ms": (75.0, 100.0, 200.0, 300.0),
        }
    )

    with pytest.raises(ValueError, match="sampling horizons must be prediction horizons"):
        invalid.validate()
