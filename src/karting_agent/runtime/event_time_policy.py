"""Deterministic decoder for V4-C4 current-action and event-time outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


EventTimeReason = Literal[
    "hold",
    "action_mismatch",
    "action_hold",
    "event_armed",
    "event_refreshed",
    "no_event",
]


@dataclass(frozen=True)
class EventTimePolicyConfig:
    bin_ms: float = 50.0
    event_bins: int = 6
    no_event_class: int = 6
    action_threshold: float = 0.5
    min_state_hold_ms: float = 100.0

    def validate(self) -> None:
        if not math.isfinite(self.bin_ms) or self.bin_ms <= 0:
            raise ValueError("bin_ms must be finite and > 0")
        if self.event_bins < 1:
            raise ValueError("event_bins must be >= 1")
        if self.no_event_class != self.event_bins:
            raise ValueError("no_event_class must equal event_bins")
        if not 0.0 < self.action_threshold < 1.0:
            raise ValueError("action_threshold must be in (0, 1)")
        if (
            not math.isfinite(self.min_state_hold_ms)
            or self.min_state_hold_ms < 0
        ):
            raise ValueError("min_state_hold_ms must be finite and >= 0")


@dataclass(frozen=True)
class EventTimeDecision:
    timestamp_ms: float
    switch: bool
    state_before: bool
    state_after: bool
    reason: EventTimeReason
    desired_pressed: bool
    event_class: int
    pending_due_ms: float | None = None


@dataclass(frozen=True)
class PendingEventExecution:
    due_at_ms: float
    state_before: bool
    state_after: bool


@dataclass(frozen=True)
class _PendingEvent:
    state_before: bool
    state_after: bool
    due_at_ms: float


class EventTimePolicyDecoder:
    """Receding-horizon decoder with deterministic minimum-state hold."""

    def __init__(self, config: EventTimePolicyConfig) -> None:
        config.validate()
        self.config = config
        self._pending: _PendingEvent | None = None
        self._last_update_ms: float | None = None
        self._last_switch_ms: float | None = None

    @property
    def pending_due_ms(self) -> float | None:
        return None if self._pending is None else self._pending.due_at_ms

    @property
    def last_switch_ms(self) -> float | None:
        return self._last_switch_ms

    def reset(self) -> None:
        self._pending = None
        self._last_update_ms = None
        self._last_switch_ms = None

    def _hold_expiry_ms(self) -> float | None:
        if self._last_switch_ms is None:
            return None
        return self._last_switch_ms + self.config.min_state_hold_ms

    def _inside_hold(self, timestamp_ms: float) -> bool:
        expiry = self._hold_expiry_ms()
        return expiry is not None and timestamp_ms + 1e-6 < expiry

    def execute_pending_if_due(
        self,
        *,
        timestamp_ms: float,
        current_pressed: bool,
    ) -> PendingEventExecution | None:
        timestamp_ms = float(timestamp_ms)
        if not math.isfinite(timestamp_ms):
            raise ValueError("timestamp_ms must be finite")
        current_pressed = bool(current_pressed)
        pending = self._pending
        if pending is None:
            return None
        if pending.state_before != current_pressed:
            self._pending = None
            return None
        if timestamp_ms + 1e-6 < pending.due_at_ms:
            return None

        self._pending = None
        self._last_switch_ms = pending.due_at_ms
        return PendingEventExecution(
            due_at_ms=pending.due_at_ms,
            state_before=pending.state_before,
            state_after=pending.state_after,
        )

    def update(
        self,
        *,
        timestamp_ms: float,
        current_pressed: bool,
        current_action_probability: float,
        event_class: int,
    ) -> EventTimeDecision:
        timestamp_ms = float(timestamp_ms)
        if not math.isfinite(timestamp_ms):
            raise ValueError("timestamp_ms must be finite")
        if (
            self._last_update_ms is not None
            and timestamp_ms <= self._last_update_ms
        ):
            raise ValueError("decoder timestamps must be strictly increasing")
        self._last_update_ms = timestamp_ms

        current_pressed = bool(current_pressed)
        probability = float(current_action_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("current_action_probability must be in [0, 1]")
        event_class = int(event_class)
        if not 0 <= event_class <= self.config.no_event_class:
            raise ValueError("event_class is out of range")

        desired_pressed = probability >= self.config.action_threshold
        if self._pending is not None and self._pending.state_before != current_pressed:
            self._pending = None

        if desired_pressed != current_pressed:
            if self._inside_hold(timestamp_ms):
                expiry = self._hold_expiry_ms()
                assert expiry is not None
                self._pending = _PendingEvent(
                    state_before=current_pressed,
                    state_after=desired_pressed,
                    due_at_ms=expiry,
                )
                return EventTimeDecision(
                    timestamp_ms=timestamp_ms,
                    switch=False,
                    state_before=current_pressed,
                    state_after=current_pressed,
                    reason="action_hold",
                    desired_pressed=desired_pressed,
                    event_class=event_class,
                    pending_due_ms=expiry,
                )

            self._pending = None
            self._last_switch_ms = timestamp_ms
            return EventTimeDecision(
                timestamp_ms=timestamp_ms,
                switch=True,
                state_before=current_pressed,
                state_after=desired_pressed,
                reason="action_mismatch",
                desired_pressed=desired_pressed,
                event_class=event_class,
            )

        if event_class == self.config.no_event_class:
            self._pending = None
            return EventTimeDecision(
                timestamp_ms=timestamp_ms,
                switch=False,
                state_before=current_pressed,
                state_after=current_pressed,
                reason="no_event",
                desired_pressed=desired_pressed,
                event_class=event_class,
            )

        due_at_ms = timestamp_ms + (event_class + 0.5) * self.config.bin_ms
        expiry = self._hold_expiry_ms()
        if expiry is not None:
            due_at_ms = max(due_at_ms, expiry)

        reason: EventTimeReason = (
            "event_refreshed" if self._pending is not None else "event_armed"
        )
        self._pending = _PendingEvent(
            state_before=current_pressed,
            state_after=not current_pressed,
            due_at_ms=due_at_ms,
        )
        return EventTimeDecision(
            timestamp_ms=timestamp_ms,
            switch=False,
            state_before=current_pressed,
            state_after=current_pressed,
            reason=reason,
            desired_pressed=desired_pressed,
            event_class=event_class,
            pending_due_ms=due_at_ms,
        )
