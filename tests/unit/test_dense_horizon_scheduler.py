from __future__ import annotations

import pytest

from karting_agent.runtime.dense_horizon_scheduler import (
    DenseHorizonSchedulerConfig,
    DenseHorizonSwitchScheduler,
)


def make_scheduler() -> DenseHorizonSwitchScheduler:
    return DenseHorizonSwitchScheduler(
        DenseHorizonSchedulerConfig(
            horizons_ms=(0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0),
            threshold=0.6,
            min_state_hold_ms=100.0,
        )
    )


def test_future_head_rising_crossing_arms_fixed_deadline() -> None:
    scheduler = make_scheduler()

    first = scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4),
        current_pressed=False,
    )
    armed = scheduler.update(
        timestamp_ms=50.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8),
        current_pressed=False,
    )

    assert first.reason == "hold"
    assert armed.reason == "pending_armed"
    # H300 crosses 0.6 halfway between t=0 and t=50 -> crossing t=25, due=325.
    assert armed.pending_due_ms == pytest.approx(325.0)
    assert armed.crossing_horizon_ms == pytest.approx(300.0)


def test_later_nearer_head_confirmation_does_not_push_deadline_later() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4),
        current_pressed=False,
    )
    armed = scheduler.update(
        timestamp_ms=50.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8),
        current_pressed=False,
    )
    confirmed = scheduler.update(
        timestamp_ms=100.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.9),
        current_pressed=False,
    )

    assert armed.pending_due_ms == pytest.approx(325.0)
    assert confirmed.reason == "pending_wait"
    assert confirmed.pending_due_ms == pytest.approx(325.0)


def test_pending_warning_cancels_if_all_future_heads_fall_below_threshold() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4),
        current_pressed=False,
    )
    scheduler.update(
        timestamp_ms=50.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8),
        current_pressed=False,
    )
    cancelled = scheduler.update(
        timestamp_ms=100.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2),
        current_pressed=False,
    )

    assert cancelled.reason == "pending_cancelled"
    assert scheduler.pending_due_ms is None


def test_pending_executes_on_exact_deadline() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4),
        current_pressed=False,
    )
    scheduler.update(
        timestamp_ms=50.0,
        probabilities=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.8),
        current_pressed=False,
    )

    execution = scheduler.execute_pending_if_due(
        timestamp_ms=325.0,
        current_pressed=False,
    )

    assert execution is not None
    assert execution.due_at_ms == pytest.approx(325.0)
    assert execution.source_horizon_ms == pytest.approx(300.0)
    assert scheduler.pending_due_ms is None


def test_h0_switches_immediately_outside_hold() -> None:
    scheduler = make_scheduler()
    decision = scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
        current_pressed=False,
    )

    assert decision.switch
    assert decision.reason == "primary"


def test_h0_inside_hold_arms_hold_expiry() -> None:
    scheduler = make_scheduler()
    first = scheduler.update(
        timestamp_ms=0.0,
        probabilities=(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
        current_pressed=False,
    )
    assert first.switch

    blocked = scheduler.update(
        timestamp_ms=50.0,
        probabilities=(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9),
        current_pressed=True,
    )

    assert not blocked.switch
    assert blocked.reason == "pending_armed"
    assert blocked.pending_due_ms == pytest.approx(100.0)
