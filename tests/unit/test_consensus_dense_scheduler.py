from __future__ import annotations

import pytest

from karting_agent.runtime.consensus_dense_scheduler import (
    ConsensusDenseConfig,
    ConsensusDenseHorizonScheduler,
)


def make_scheduler() -> ConsensusDenseHorizonScheduler:
    return ConsensusDenseHorizonScheduler(
        ConsensusDenseConfig(
            horizons_ms=(0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0),
            threshold=0.6,
            min_state_hold_ms=100.0,
            min_consensus_heads=3,
            consensus_window_ms=60.0,
        )
    )


def test_single_future_crossing_does_not_arm() -> None:
    scheduler = make_scheduler()
    first = scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4),
        current_pressed=False,
    )
    second = scheduler.update(
        timestamp_ms=50.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8),
        current_pressed=False,
    )

    assert first.reason == "hold"
    assert second.reason == "hold"
    assert scheduler.pending_due_ms is None


def test_three_agreeing_future_heads_arm_consensus_deadline() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4),
        current_pressed=False,
    )
    scheduler.update(
        timestamp_ms=50.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8),
        current_pressed=False,
    )
    armed = scheduler.update(
        timestamp_ms=100.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.8, 0.9, 0.9),
        current_pressed=False,
    )

    assert armed.reason == "consensus_armed"
    assert armed.pending_due_ms is not None
    assert len(armed.consensus_heads_ms) >= 3
    assert armed.consensus_due_spread_ms is not None
    assert armed.consensus_due_spread_ms <= 60.0


def test_disagreeing_forecasts_do_not_arm() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4),
        current_pressed=False,
    )
    scheduler.update(
        timestamp_ms=100.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8),
        current_pressed=False,
    )
    decision = scheduler.update(
        timestamp_ms=300.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.8, 0.9, 0.9),
        current_pressed=False,
    )

    assert decision.reason == "hold"
    assert scheduler.pending_due_ms is None


def test_consensus_cancels_when_support_disappears() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4),
        current_pressed=False,
    )
    scheduler.update(
        timestamp_ms=50.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8),
        current_pressed=False,
    )
    armed = scheduler.update(
        timestamp_ms=100.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.8, 0.9, 0.9),
        current_pressed=False,
    )
    assert armed.reason == "consensus_armed"

    cancelled = scheduler.update(
        timestamp_ms=125.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2),
        current_pressed=False,
    )
    assert cancelled.reason == "consensus_cancelled"
    assert scheduler.pending_due_ms is None


def test_h0_still_switches_immediately() -> None:
    scheduler = make_scheduler()
    decision = scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        current_pressed=False,
    )

    assert decision.switch
    assert decision.reason == "primary"


def test_pending_executes_on_deadline() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4),
        current_pressed=False,
    )
    scheduler.update(
        timestamp_ms=50.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8),
        current_pressed=False,
    )
    armed = scheduler.update(
        timestamp_ms=100.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.8, 0.9, 0.9),
        current_pressed=False,
    )
    assert armed.pending_due_ms is not None

    execution = scheduler.execute_pending_if_due(
        timestamp_ms=armed.pending_due_ms,
        current_pressed=False,
    )

    assert execution is not None
    assert execution.due_at_ms == pytest.approx(armed.pending_due_ms)
