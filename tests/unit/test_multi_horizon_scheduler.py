from __future__ import annotations

import pytest

from karting_agent.runtime.multi_horizon_scheduler import (
    MultiHorizonSchedulerConfig,
    MultiHorizonSwitchScheduler,
)


def make_scheduler() -> MultiHorizonSwitchScheduler:
    return MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
        )
    )


def test_primary_control_horizon_switches_immediately() -> None:
    scheduler = make_scheduler()

    decision = scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.1, 0.7, 0.9),
        current_pressed=False,
    )

    assert decision.switch is True
    assert decision.reason == "primary"
    assert scheduler.pending_due_ms is None
    assert scheduler.last_switch_ms == pytest.approx(1000.0)


def test_far_horizon_arms_interpolated_pending_switch() -> None:
    scheduler = make_scheduler()

    decision = scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.006, 0.045, 0.769),
        current_pressed=False,
    )

    assert decision.switch is False
    assert decision.reason == "pending_armed"
    assert decision.crossing_horizon_ms == pytest.approx(276.6574586)
    assert decision.pending_delay_ms == pytest.approx(76.6574586)
    assert decision.pending_due_ms == pytest.approx(1076.6574586)


def test_pending_advance_shifts_only_scheduled_delay() -> None:
    scheduler = MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
            pending_advance_ms=40.0,
        )
    )

    decision = scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.006, 0.045, 0.769),
        current_pressed=False,
    )

    assert decision.switch is False
    assert decision.reason == "pending_armed"
    assert decision.crossing_horizon_ms == pytest.approx(276.6574586)
    assert decision.pending_delay_ms == pytest.approx(36.6574586)
    assert decision.pending_due_ms == pytest.approx(1036.6574586)


def test_pending_advance_clamps_effective_delay_to_zero() -> None:
    scheduler = MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
            pending_advance_ms=100.0,
        )
    )

    armed = scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.1, 0.59, 0.9),
        current_pressed=False,
    )
    executed = scheduler.update(
        timestamp_ms=1001.0,
        probabilities=(0.1, 0.59, 0.9),
        current_pressed=False,
    )

    assert armed.reason == "pending_armed"
    assert armed.pending_delay_ms == pytest.approx(0.0)
    assert armed.pending_due_ms == pytest.approx(1000.0)
    assert executed.switch is True
    assert executed.reason == "pending_execute"


def test_pending_advance_must_fit_horizon_span() -> None:
    config = MultiHorizonSchedulerConfig(
        horizons_ms=(100.0, 200.0, 300.0),
        control_horizon_ms=200.0,
        anticipation_horizon_ms=300.0,
        pending_advance_ms=100.1,
    )

    with pytest.raises(ValueError, match="pending_advance_ms"):
        config.validate()


def test_pending_requires_warning_to_persist_until_due() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.006, 0.045, 0.769),
        current_pressed=False,
    )

    waiting = scheduler.update(
        timestamp_ms=1050.0,
        probabilities=(0.02, 0.20, 0.80),
        current_pressed=False,
    )
    executed = scheduler.update(
        timestamp_ms=1080.0,
        probabilities=(0.03, 0.30, 0.90),
        current_pressed=False,
    )

    assert waiting.switch is False
    assert waiting.reason == "pending_wait"
    assert executed.switch is True
    assert executed.reason == "pending_execute"
    assert scheduler.pending_due_ms is None
    assert scheduler.last_switch_ms == pytest.approx(1080.0)


def test_pending_deadline_can_be_consumed_without_new_model_observation() -> None:
    scheduler = make_scheduler()
    armed = scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.1, 0.55, 0.80),
        current_pressed=False,
    )
    assert armed.pending_due_ms == pytest.approx(1020.0)

    early = scheduler.execute_pending_if_due(
        timestamp_ms=1019.9,
        current_pressed=False,
    )
    executed = scheduler.execute_pending_if_due(
        timestamp_ms=1020.0,
        current_pressed=False,
    )

    assert early is None
    assert scheduler.pending_due_ms is None
    assert executed is not None
    assert executed.state_before is False
    assert executed.due_at_ms == pytest.approx(1020.0)
    assert scheduler.last_switch_ms == pytest.approx(1020.0)


def test_pending_deadline_rejects_stale_state() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.1, 0.55, 0.80),
        current_pressed=False,
    )

    executed = scheduler.execute_pending_if_due(
        timestamp_ms=1020.0,
        current_pressed=True,
    )

    assert executed is None
    assert scheduler.pending_due_ms is None


def test_pending_is_cancelled_when_far_warning_disappears() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.006, 0.045, 0.769),
        current_pressed=False,
    )

    decision = scheduler.update(
        timestamp_ms=1040.0,
        probabilities=(0.01, 0.05, 0.40),
        current_pressed=False,
    )

    assert decision.switch is False
    assert decision.reason == "pending_cancelled"
    assert scheduler.pending_due_ms is None


def test_non_monotonic_horizons_do_not_arm_pending_switch() -> None:
    scheduler = make_scheduler()

    decision = scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.8, 0.2, 0.9),
        current_pressed=False,
    )

    assert decision.switch is False
    assert decision.reason == "hold"
    assert scheduler.pending_due_ms is None


def test_primary_signal_overrides_pending_delay() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.006, 0.045, 0.769),
        current_pressed=False,
    )

    decision = scheduler.update(
        timestamp_ms=1030.0,
        probabilities=(0.2, 0.65, 0.95),
        current_pressed=False,
    )

    assert decision.switch is True
    assert decision.reason == "primary"
    assert scheduler.pending_due_ms is None


def test_minimum_state_hold_suppresses_immediate_primary_reversal() -> None:
    scheduler = make_scheduler()
    first = scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.1, 0.7, 0.9),
        current_pressed=False,
    )
    assert first.switch is True

    blocked = scheduler.update(
        timestamp_ms=1099.0,
        probabilities=(0.95, 0.90, 0.80),
        current_pressed=True,
    )
    eligible = scheduler.update(
        timestamp_ms=1100.0,
        probabilities=(0.95, 0.90, 0.80),
        current_pressed=True,
    )

    assert blocked.switch is False
    assert blocked.reason == "min_hold"
    assert eligible.switch is True
    assert eligible.reason == "primary"


def test_minimum_state_hold_does_not_arm_pending_transition() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.1, 0.7, 0.9),
        current_pressed=False,
    )

    blocked = scheduler.update(
        timestamp_ms=1050.0,
        probabilities=(0.1, 0.2, 0.8),
        current_pressed=True,
    )

    assert blocked.switch is False
    assert blocked.reason == "min_hold"
    assert scheduler.pending_due_ms is None


def test_minimum_hold_can_arm_pending_without_early_execution() -> None:
    scheduler = MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
            min_state_hold_ms=100.0,
            arm_pending_during_min_hold=True,
        )
    )
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.1, 0.7, 0.9),
        current_pressed=False,
    )

    armed = scheduler.update(
        timestamp_ms=1050.0,
        probabilities=(0.1, 0.59, 0.9),
        current_pressed=True,
    )

    assert armed.switch is False
    assert armed.reason == "pending_armed"
    assert armed.pending_due_ms == pytest.approx(1100.0)
    assert armed.pending_delay_ms == pytest.approx(50.0)
    assert scheduler.execute_pending_if_due(
        timestamp_ms=1099.0,
        current_pressed=True,
    ) is None

    executed = scheduler.execute_pending_if_due(
        timestamp_ms=1100.0,
        current_pressed=True,
    )
    assert executed is not None
    assert executed.due_at_ms == pytest.approx(1100.0)
    assert scheduler.last_switch_ms == pytest.approx(1100.0)


def test_minimum_hold_preserves_later_natural_pending_deadline() -> None:
    scheduler = MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
            min_state_hold_ms=100.0,
            arm_pending_during_min_hold=True,
        )
    )
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.1, 0.7, 0.9),
        current_pressed=False,
    )

    armed = scheduler.update(
        timestamp_ms=1050.0,
        probabilities=(0.1, 0.2, 0.8),
        current_pressed=True,
    )

    assert armed.reason == "pending_armed"
    assert armed.pending_due_ms == pytest.approx(1116.6666667)
    assert armed.pending_delay_ms == pytest.approx(66.6666667)


def test_minimum_hold_pending_is_cancelled_when_warning_disappears() -> None:
    scheduler = MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
            min_state_hold_ms=100.0,
            arm_pending_during_min_hold=True,
        )
    )
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.1, 0.7, 0.9),
        current_pressed=False,
    )
    scheduler.update(
        timestamp_ms=1050.0,
        probabilities=(0.1, 0.2, 0.8),
        current_pressed=True,
    )

    cancelled = scheduler.update(
        timestamp_ms=1070.0,
        probabilities=(0.1, 0.1, 0.4),
        current_pressed=True,
    )

    assert cancelled.switch is False
    assert cancelled.reason == "pending_cancelled"
    assert scheduler.pending_due_ms is None


def test_zero_minimum_state_hold_preserves_immediate_reversal() -> None:
    scheduler = MultiHorizonSwitchScheduler(
        MultiHorizonSchedulerConfig(
            horizons_ms=(100.0, 200.0, 300.0),
            control_horizon_ms=200.0,
            anticipation_horizon_ms=300.0,
            threshold=0.6,
            min_state_hold_ms=0.0,
        )
    )
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.1, 0.7, 0.9),
        current_pressed=False,
    )

    reversal = scheduler.update(
        timestamp_ms=1050.0,
        probabilities=(0.95, 0.90, 0.80),
        current_pressed=True,
    )

    assert reversal.switch is True
    assert reversal.reason == "primary"


def test_state_change_invalidates_old_pending_switch() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.006, 0.045, 0.769),
        current_pressed=False,
    )

    decision = scheduler.update(
        timestamp_ms=1030.0,
        probabilities=(0.9, 0.3, 0.2),
        current_pressed=True,
    )

    assert decision.switch is False
    assert decision.reason == "hold"
    assert scheduler.pending_due_ms is None


def test_scheduler_rejects_non_increasing_timestamps() -> None:
    scheduler = make_scheduler()
    scheduler.update(
        timestamp_ms=1000.0,
        probabilities=(0.0, 0.0, 0.0),
        current_pressed=False,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        scheduler.update(
            timestamp_ms=1000.0,
            probabilities=(0.0, 0.0, 0.0),
            current_pressed=False,
        )
