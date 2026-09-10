from karting_agent.train.baselines import persistence_points, persistence_prediction
from karting_agent.train.dataset import DatasetSample
from karting_agent.train.labels.touch_marker import ActionEvent


def _sample(current_ms: float, target_ms: float, target_pressed: bool) -> DatasetSample:
    return DatasetSample(
        video="video.mp4",
        input_frame_indices=(0, 1, 2),
        input_timestamps_ms=(current_ms - 100.0, current_ms - 50.0, current_ms),
        target_frame_index=3,
        target_timestamp_ms=target_ms,
        target_pressed=target_pressed,
        transition_distance_ms=None,
        near_transition=False,
        near_short_correction=False,
    )


def test_persistence_prediction_copies_action_at_last_input_timestamp() -> None:
    events = [
        ActionEvent(frame_index=0, timestamp_ms=0.0, pressed=True),
        ActionEvent(frame_index=15, timestamp_ms=150.0, pressed=False),
    ]

    before_transition = _sample(100.0, 200.0, False)
    after_transition = _sample(200.0, 300.0, False)

    assert persistence_prediction(before_transition, events) is True
    assert persistence_prediction(after_transition, events) is False

    points = persistence_points([before_transition, after_transition], events)
    assert [point.timestamp_ms for point in points] == [200.0, 300.0]
    assert [point.probability for point in points] == [1.0, 0.0]
