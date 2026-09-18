from __future__ import annotations

import pytest

from karting_agent.runtime.event_time_policy import (
    EventTimePolicyConfig,
    EventTimePolicyDecoder,
)


def make_decoder() -> EventTimePolicyDecoder:
    return EventTimePolicyDecoder(
        EventTimePolicyConfig(
            bin_ms=50.0,
            event_bins=6,
            no_event_class=6,
            action_threshold=0.5,
            min_state_hold_ms=100.0,
        )
    )


def test_current_action_mismatch_switches_immediately() -> None:
    decoder = make_decoder()

    decision = decoder.update(
        timestamp_ms=1000.0,
        current_pressed=False,
        current_action_probability=0.9,
        event_class=6,
    )

    assert decision.switch is True
    assert decision.state_after is True
    assert decision.reason == "action_mismatch"
    assert decoder.last_switch_ms == pytest.approx(1000.0)


def test_event_time_arms_bin_midpoint_deadline() -> None:
    decoder = make_decoder()

    decision = decoder.update(
        timestamp_ms=1000.0,
        current_pressed=False,
        current_action_probability=0.1,
        event_class=2,
    )

    assert decision.switch is False
    assert decision.reason == "event_armed"
    assert decision.pending_due_ms == pytest.approx(1125.0)


def test_event_time_refreshes_pending_deadline() -> None:
    decoder = make_decoder()
    decoder.update(
        timestamp_ms=1000.0,
        current_pressed=False,
        current_action_probability=0.1,
        event_class=2,
    )

    refreshed = decoder.update(
        timestamp_ms=1030.0,
        current_pressed=False,
        current_action_probability=0.1,
        event_class=1,
    )

    assert refreshed.reason == "event_refreshed"
    assert refreshed.pending_due_ms == pytest.approx(1105.0)


def test_no_event_cancels_pending_prediction() -> None:
    decoder = make_decoder()
    decoder.update(
        timestamp_ms=1000.0,
        current_pressed=False,
        current_action_probability=0.1,
        event_class=2,
    )

    cancelled = decoder.update(
        timestamp_ms=1030.0,
        current_pressed=False,
        current_action_probability=0.1,
        event_class=6,
    )

    assert cancelled.reason == "no_event"
    assert decoder.pending_due_ms is None


def test_pending_executes_at_fixed_due_time() -> None:
    decoder = make_decoder()
    decoder.update(
        timestamp_ms=1000.0,
        current_pressed=False,
        current_action_probability=0.1,
        event_class=0,
    )

    early = decoder.execute_pending_if_due(
        timestamp_ms=1024.9,
        current_pressed=False,
    )
    executed = decoder.execute_pending_if_due(
        timestamp_ms=1025.0,
        current_pressed=False,
    )

    assert early is None
    assert executed is not None
    assert executed.due_at_ms == pytest.approx(1025.0)
    assert executed.state_before is False
    assert executed.state_after is True
    assert decoder.last_switch_ms == pytest.approx(1025.0)


def test_minimum_hold_defers_action_mismatch_to_hold_expiry() -> None:
    decoder = make_decoder()
    first = decoder.update(
        timestamp_ms=1000.0,
        current_pressed=False,
        current_action_probability=0.9,
        event_class=6,
    )
    assert first.switch is True

    blocked = decoder.update(
        timestamp_ms=1050.0,
        current_pressed=True,
        current_action_probability=0.1,
        event_class=6,
    )

    assert blocked.switch is False
    assert blocked.reason == "action_hold"
    assert blocked.pending_due_ms == pytest.approx(1100.0)

    executed = decoder.execute_pending_if_due(
        timestamp_ms=1100.0,
        current_pressed=True,
    )
    assert executed is not None
    assert executed.state_after is False


def test_event_deadline_is_clamped_to_hold_expiry() -> None:
    decoder = make_decoder()
    decoder.update(
        timestamp_ms=1000.0,
        current_pressed=False,
        current_action_probability=0.9,
        event_class=6,
    )

    armed = decoder.update(
        timestamp_ms=1040.0,
        current_pressed=True,
        current_action_probability=0.9,
        event_class=0,
    )

    assert armed.reason == "event_armed"
    assert armed.pending_due_ms == pytest.approx(1100.0)
