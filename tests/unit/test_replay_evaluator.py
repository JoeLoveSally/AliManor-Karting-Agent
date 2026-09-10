import pytest

from karting_agent.train.replay_evaluator import replay_points


def test_replay_points_support_target_and_observation_timelines() -> None:
    steps = [
        {
            "observation_timestamp_ms": 100.0,
            "prediction_target_timestamp_ms": 200.0,
            "pressed": False,
        },
        {
            "observation_timestamp_ms": 133.0,
            "prediction_target_timestamp_ms": 233.0,
            "pressed": True,
        },
    ]

    target = replay_points(steps, video="video.mp4", timeline="target")
    observation = replay_points(steps, video="video.mp4", timeline="observation")

    assert [point.timestamp_ms for point in target] == [200.0, 233.0]
    assert [point.timestamp_ms for point in observation] == [100.0, 133.0]
    assert [point.probability for point in target] == [0.0, 1.0]


def test_replay_points_reject_non_monotonic_timestamps() -> None:
    steps = [
        {
            "observation_timestamp_ms": 100.0,
            "prediction_target_timestamp_ms": 200.0,
            "pressed": False,
        },
        {
            "observation_timestamp_ms": 100.0,
            "prediction_target_timestamp_ms": 200.0,
            "pressed": True,
        },
    ]

    with pytest.raises(ValueError):
        replay_points(steps, video="video.mp4", timeline="target")
