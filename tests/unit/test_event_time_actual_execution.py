from __future__ import annotations

import pytest

from karting_agent.runtime.event_time_policy import EventTimePolicyConfig, EventTimePolicyDecoder


def _held_decoder() -> EventTimePolicyDecoder:
    decoder = EventTimePolicyDecoder(EventTimePolicyConfig())
    first = decoder.update(
        timestamp_ms=200.0,
        current_pressed=False,
        current_action_probability=0.99,
        event_class=6,
    )
    assert first.switch
    blocked = decoder.update(
        timestamp_ms=233.333,
        current_pressed=True,
        current_action_probability=0.01,
        event_class=6,
    )
    assert blocked.reason == "action_hold"
    assert blocked.pending_due_ms == pytest.approx(300.0)
    return decoder


def test_offline_pending_keeps_legacy_deadline_timestamp() -> None:
    decoder = _held_decoder()
    executed = decoder.execute_pending_if_due(timestamp_ms=333.333, current_pressed=True)
    assert executed is not None
    assert executed.due_at_ms == pytest.approx(300.0)
    assert decoder.last_switch_ms == pytest.approx(300.0)


def test_runtime_pending_uses_actual_observation_as_hold_anchor() -> None:
    decoder = _held_decoder()
    executed = decoder.execute_pending_if_due(
        timestamp_ms=333.333,
        current_pressed=True,
        execution_timestamp_ms=333.333,
    )
    assert executed is not None
    assert executed.due_at_ms == pytest.approx(300.0)
    assert decoder.last_switch_ms == pytest.approx(333.333)
    blocked = decoder.update(
        timestamp_ms=366.667,
        current_pressed=False,
        current_action_probability=0.99,
        event_class=6,
    )
    assert blocked.reason == "action_hold"
    assert blocked.pending_due_ms == pytest.approx(433.333)


def test_runtime_rejects_fictional_or_future_execution_time() -> None:
    decoder = _held_decoder()
    with pytest.raises(ValueError, match="execution_timestamp_ms"):
        decoder.execute_pending_if_due(
            timestamp_ms=333.333, current_pressed=True, execution_timestamp_ms=299.0
        )
    assert decoder.pending_due_ms == pytest.approx(300.0)
    with pytest.raises(ValueError, match="execution_timestamp_ms"):
        decoder.execute_pending_if_due(
            timestamp_ms=333.333, current_pressed=True, execution_timestamp_ms=350.0
        )
    assert decoder.pending_due_ms == pytest.approx(300.0)
