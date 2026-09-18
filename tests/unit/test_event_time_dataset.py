from __future__ import annotations

import numpy as np
import pytest

from karting_agent.train.event_time_dataset import event_time_class


def test_event_time_class_maps_each_50ms_bin() -> None:
    transitions = np.asarray([25.0, 75.0, 125.0, 175.0, 225.0, 275.0])

    assert event_time_class(
        transitions,
        0.0,
        max_horizon_ms=300.0,
        bin_ms=50.0,
    ) == (0, 25.0)
    assert event_time_class(
        transitions,
        25.0,
        max_horizon_ms=300.0,
        bin_ms=50.0,
    ) == (1, 50.0)
    assert event_time_class(
        transitions,
        75.0,
        max_horizon_ms=300.0,
        bin_ms=50.0,
    ) == (2, 50.0)


def test_event_time_class_uses_no_event_class_beyond_horizon() -> None:
    transitions = np.asarray([350.0])

    target_class, delay_ms = event_time_class(
        transitions,
        0.0,
        max_horizon_ms=300.0,
        bin_ms=50.0,
    )

    assert target_class == 6
    assert delay_ms == pytest.approx(350.0)


def test_event_time_class_uses_no_event_class_without_future_transition() -> None:
    target_class, delay_ms = event_time_class(
        np.asarray([], dtype=np.float64),
        100.0,
        max_horizon_ms=300.0,
        bin_ms=50.0,
    )

    assert target_class == 6
    assert delay_ms is None


def test_event_time_class_requires_divisible_horizon() -> None:
    with pytest.raises(ValueError, match="divisible"):
        event_time_class(
            np.asarray([100.0]),
            0.0,
            max_horizon_ms=300.0,
            bin_ms=70.0,
        )
