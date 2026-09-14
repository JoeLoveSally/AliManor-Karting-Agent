import numpy as np
import pytest

from karting_agent.train.dataset import DatasetConfig, build_samples
from karting_agent.train.labels.touch_marker import (
    ActionTimeline,
    segments_to_events,
    states_to_segments,
)
from karting_agent.vision.preprocess import PreprocessConfig, prepare_frame


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


def test_v2_builds_five_frame_multi_horizon_sample() -> None:
    # RELEASE [0,500), PRESS [500,850), RELEASE [850,1050), PRESS afterwards.
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

    assert sample.input_timestamps_ms == (500.0, 550.0, 600.0, 650.0, 700.0)
    assert sample.input_frame_indices == (50, 55, 60, 65, 70)
    assert sample.target_timestamps_ms == (800.0, 900.0, 1000.0)
    assert sample.target_pressed_by_horizon == (True, False, False)
    assert sample.target_pressed is True
    assert sample.near_transition


def test_preprocess_applies_additional_fixed_masks() -> None:
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    config = PreprocessConfig(
        input_width=100,
        input_height=100,
        mask_touch_area=False,
        mask_rois=((0.0, 0.0, 0.25, 0.25), (0.75, 0.0, 1.0, 0.25)),
    )

    prepared = prepare_frame(frame, config)

    assert np.all(prepared[:25, :25] == 0)
    assert np.all(prepared[:25, 75:] == 0)
    assert np.all(prepared[30:70, 30:70] == 255)


def test_model_supports_five_frames_and_three_horizons() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from karting_agent.model.base import build_model

    model = build_model(
        "mobilenet_v3_small",
        frame_stack=5,
        pretrained=False,
        output_dim=3,
    )
    logits = model(torch.zeros(2, 15, 224, 224))
    assert logits.shape == (2, 3)
