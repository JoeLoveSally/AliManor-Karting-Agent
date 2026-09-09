from karting_agent.train.dataset import DatasetConfig, build_samples
from karting_agent.train.labels.touch_marker import (
    ActionTimeline,
    clean_isolated_one_frame_glitches,
    segments_to_events,
    states_to_segments,
)


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


def test_clean_only_isolated_one_frame() -> None:
    cleaned, changed = clean_isolated_one_frame_glitches(
        [False, False, True, False, False]
    )
    assert cleaned == [False] * 5
    assert changed == 1

    cleaned, changed = clean_isolated_one_frame_glitches(
        [False, False, True, True, False]
    )
    assert cleaned == [False, False, True, True, False]
    assert changed == 0


def test_build_samples_uses_future_target_and_temporal_stack() -> None:
    # 100 Hz = 10 ms/frame. RELEASE until 500ms, PRESS 500-700ms,
    # short RELEASE 700-850ms, then PRESS again.
    states = [False] * 50 + [True] * 20 + [False] * 15 + [True] * 35
    timeline = make_timeline(states)
    config = DatasetConfig(
        sample_fps=10,
        frame_stack=3,
        history_ms=100,
        frame_interval_ms=50,
        prediction_horizon_ms=100,
        transition_window_ms=100,
    )

    samples = build_samples(timeline, "test.mp4", config)
    sample = next(
        sample for sample in samples if abs(sample.target_timestamp_ms - 800) < 1e-6
    )

    assert sample.target_pressed is False
    assert sample.input_timestamps_ms == (600.0, 650.0, 700.0)
    assert sample.input_frame_indices == (60, 65, 70)
    assert sample.near_transition
    assert sample.near_short_correction
