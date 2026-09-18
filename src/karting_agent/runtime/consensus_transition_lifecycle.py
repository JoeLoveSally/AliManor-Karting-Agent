"""Lifecycle guard for future-scheduled state transitions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


LifecycleStatus = Literal["idle", "waiting", "confirmed", "expired"]


@dataclass(frozen=True)
class ProvisionalTransition:
    previous_pressed: bool
    current_pressed: bool
    executed_at_ms: float
    confirmation_deadline_ms: float


class ConsensusTransitionLifecycle:
    """Hold a future-scheduled transition until H0 catches up.

    The confirmation window is derived from the evidence lead of the consumed
    future event, with the existing minimum state hold as a lower bound. No new
    tuning parameter is introduced.
    """

    def __init__(self, *, threshold: float, min_state_hold_ms: float) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if not math.isfinite(min_state_hold_ms) or min_state_hold_ms < 0.0:
            raise ValueError("min_state_hold_ms must be finite and >= 0")
        self.threshold = float(threshold)
        self.min_state_hold_ms = float(min_state_hold_ms)
        self._pending: ProvisionalTransition | None = None

    @property
    def pending(self) -> ProvisionalTransition | None:
        return self._pending

    @property
    def active(self) -> bool:
        return self._pending is not None

    def clear(self) -> None:
        self._pending = None

    def begin(
        self,
        *,
        executed_at_ms: float,
        previous_pressed: bool,
        current_pressed: bool,
        evidence_lead_ms: float,
    ) -> ProvisionalTransition:
        executed_at_ms = float(executed_at_ms)
        evidence_lead_ms = float(evidence_lead_ms)
        if not math.isfinite(executed_at_ms):
            raise ValueError("executed_at_ms must be finite")
        if not math.isfinite(evidence_lead_ms) or evidence_lead_ms < 0.0:
            raise ValueError("evidence_lead_ms must be finite and >= 0")
        if bool(previous_pressed) == bool(current_pressed):
            raise ValueError("provisional transition must change state")

        confirmation_window_ms = max(
            self.min_state_hold_ms,
            evidence_lead_ms,
        )
        self._pending = ProvisionalTransition(
            previous_pressed=bool(previous_pressed),
            current_pressed=bool(current_pressed),
            executed_at_ms=executed_at_ms,
            confirmation_deadline_ms=executed_at_ms + confirmation_window_ms,
        )
        return self._pending

    def observe(
        self,
        *,
        timestamp_ms: float,
        previous_state_h0_probability: float,
    ) -> LifecycleStatus:
        timestamp_ms = float(timestamp_ms)
        probability = float(previous_state_h0_probability)
        if not math.isfinite(timestamp_ms):
            raise ValueError("timestamp_ms must be finite")
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("previous_state_h0_probability must be in [0, 1]")

        pending = self._pending
        if pending is None:
            return "idle"
        if timestamp_ms < pending.executed_at_ms - 1e-6:
            raise ValueError("timestamp_ms precedes provisional execution")

        if probability >= self.threshold:
            self._pending = None
            return "confirmed"
        if timestamp_ms >= pending.confirmation_deadline_ms - 1e-6:
            self._pending = None
            return "expired"
        return "waiting"
