"""Decode multi-horizon switch probabilities into timed state transitions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence


SchedulerReason = Literal[
    "hold",
    "min_hold",
    "primary",
    "short_overdue",
    "pending_armed",
    "pending_wait",
    "pending_execute",
    "pending_cancelled",
]


@dataclass(frozen=True)
class MultiHorizonSchedulerConfig:
    """Configuration for anticipation-assisted transition timing.

    ``control_horizon_ms`` remains the deployed execution horizon. A farther
    ``anticipation_horizon_ms`` may arm a pending transition when probabilities
    rise monotonically with horizon. The pending delay is estimated by linearly
    interpolating where the switch probability crosses ``threshold`` between the
    control and anticipation horizons, then subtracting the control horizon.

    ``pending_advance_ms`` shifts only anticipation-scheduled transitions earlier
    than that interpolated delay. The effective pending delay is clamped to zero,
    so this never schedules a transition before the warning was observed.

    ``min_state_hold_ms`` rejects immediate reversals after an executed switch.
    The default 100 ms matches the lower bound of the dataset's explicitly
    retained short-correction segments, so it removes scheduler-created
    sub-100 ms chatter without intentionally suppressing the 100--300 ms
    corrections that are part of the training/evaluation target.

    When ``arm_pending_during_min_hold`` is enabled, the hold remains an execution
    gate rather than an observation blind spot: monotonic anticipation warnings
    may arm and remain pending during the hold, but their deadline is clamped to
    the hold expiry so no reversal can execute before ``min_state_hold_ms``.

    ``execute_short_horizon_overdue`` handles the complementary non-monotonic
    pattern where a horizon shorter than the deployed control horizon already
    predicts a switch while the control horizon does not. Under the deployed
    control-horizon semantics that means a short correction lies before the
    controller's nominal target time, so once the minimum hold has expired the
    next switch is treated as overdue and executed immediately. This remains
    model-only runtime logic; no analytic visual teacher is used online.
    """

    horizons_ms: tuple[float, ...]
    control_horizon_ms: float = 200.0
    anticipation_horizon_ms: float = 300.0
    threshold: float = 0.6
    monotonic_tolerance: float = 1e-6
    min_state_hold_ms: float = 100.0
    pending_advance_ms: float = 0.0
    arm_pending_during_min_hold: bool = False
    execute_short_horizon_overdue: bool = False

    def validate(self) -> None:
        if len(self.horizons_ms) < 2:
            raise ValueError("at least two prediction horizons are required")
        if any(not math.isfinite(value) or value < 0 for value in self.horizons_ms):
            raise ValueError("prediction horizons must be finite and >= 0")
        if tuple(sorted(self.horizons_ms)) != self.horizons_ms:
            raise ValueError("prediction horizons must be sorted")
        if len(set(self.horizons_ms)) != len(self.horizons_ms):
            raise ValueError("prediction horizons must be unique")
        if self.control_horizon_ms not in self.horizons_ms:
            raise ValueError("control_horizon_ms must be present in horizons_ms")
        if self.anticipation_horizon_ms not in self.horizons_ms:
            raise ValueError("anticipation_horizon_ms must be present in horizons_ms")
        if self.anticipation_horizon_ms <= self.control_horizon_ms:
            raise ValueError("anticipation horizon must be greater than control horizon")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if not math.isfinite(self.monotonic_tolerance) or self.monotonic_tolerance < 0:
            raise ValueError("monotonic_tolerance must be finite and >= 0")
        if not math.isfinite(self.min_state_hold_ms) or self.min_state_hold_ms < 0:
            raise ValueError("min_state_hold_ms must be finite and >= 0")
        if not math.isfinite(self.pending_advance_ms) or self.pending_advance_ms < 0:
            raise ValueError("pending_advance_ms must be finite and >= 0")
        anticipation_span_ms = (
            self.anticipation_horizon_ms - self.control_horizon_ms
        )
        if self.pending_advance_ms > anticipation_span_ms + 1e-6:
            raise ValueError(
                "pending_advance_ms must not exceed the anticipation/control horizon span"
            )


@dataclass(frozen=True)
class SchedulerDecision:
    timestamp_ms: float
    switch: bool
    reason: SchedulerReason
    current_pressed: bool
    probabilities: tuple[float, ...]
    control_probability: float
    anticipation_probability: float
    crossing_horizon_ms: float | None = None
    pending_due_ms: float | None = None
    pending_delay_ms: float | None = None


@dataclass(frozen=True)
class PendingExecution:
    """Metadata returned when a previously armed pending switch reaches its deadline."""

    state_before: bool
    armed_at_ms: float
    due_at_ms: float
    crossing_horizon_ms: float
    delay_ms: float


@dataclass(frozen=True)
class _PendingSwitch:
    state_before: bool
    armed_at_ms: float
    due_at_ms: float
    crossing_horizon_ms: float
    delay_ms: float


class MultiHorizonSwitchScheduler:
    """Use learned multi-horizon state forecasts to time the next switch."""

    def __init__(self, config: MultiHorizonSchedulerConfig) -> None:
        config.validate()
        self.config = config
        self._control_index = config.horizons_ms.index(config.control_horizon_ms)
        self._anticipation_index = config.horizons_ms.index(
            config.anticipation_horizon_ms
        )
        self._pending: _PendingSwitch | None = None
        self._last_timestamp_ms: float | None = None
        self._last_switch_ms: float | None = None

    @property
    def pending_due_ms(self) -> float | None:
        return None if self._pending is None else self._pending.due_at_ms

    @property
    def last_switch_ms(self) -> float | None:
        return self._last_switch_ms

    def reset(self) -> None:
        self._pending = None
        self._last_timestamp_ms = None
        self._last_switch_ms = None

    def execute_pending_if_due(
        self,
        *,
        timestamp_ms: float,
        current_pressed: bool,
    ) -> PendingExecution | None:
        """Consume an armed pending transition once its fixed deadline is reached.

        This path performs no model inference. It is intended for a runtime timer
        that has already kept the warning alive through normal scheduler updates.
        A state mismatch invalidates the pending transition instead of executing it.
        """

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
        if self._inside_min_hold(timestamp_ms):
            return None
        if timestamp_ms + 1e-6 < pending.due_at_ms:
            return None

        self._pending = None
        self._last_switch_ms = timestamp_ms
        return PendingExecution(
            state_before=pending.state_before,
            armed_at_ms=pending.armed_at_ms,
            due_at_ms=pending.due_at_ms,
            crossing_horizon_ms=pending.crossing_horizon_ms,
            delay_ms=pending.delay_ms,
        )

    def _validated_probabilities(self, values: Sequence[float]) -> tuple[float, ...]:
        probabilities = tuple(float(value) for value in values)
        if len(probabilities) != len(self.config.horizons_ms):
            raise ValueError("probability width does not match prediction horizons")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in probabilities
        ):
            raise ValueError("switch probabilities must be finite and in [0, 1]")
        return probabilities

    def _monotonic_to_anticipation(self, probabilities: tuple[float, ...]) -> bool:
        values = probabilities[: self._anticipation_index + 1]
        tolerance = self.config.monotonic_tolerance
        return all(
            left <= right + tolerance for left, right in zip(values, values[1:])
        )

    def _short_horizon_overdue(self, probabilities: tuple[float, ...]) -> bool:
        if not self.config.execute_short_horizon_overdue:
            return False
        threshold = self.config.threshold
        return any(
            probability >= threshold
            for probability in probabilities[: self._control_index]
        )

    def _candidate(
        self,
        probabilities: tuple[float, ...],
    ) -> tuple[float, float] | None:
        control = probabilities[self._control_index]
        anticipation = probabilities[self._anticipation_index]
        threshold = self.config.threshold
        if not (control < threshold <= anticipation):
            return None
        if not self._monotonic_to_anticipation(probabilities):
            return None

        denominator = anticipation - control
        if denominator <= 0.0:
            return None
        span_ms = (
            self.config.anticipation_horizon_ms - self.config.control_horizon_ms
        )
        fraction = (threshold - control) / denominator
        fraction = min(1.0, max(0.0, fraction))
        crossing_horizon_ms = self.config.control_horizon_ms + fraction * span_ms
        raw_delay_ms = crossing_horizon_ms - self.config.control_horizon_ms
        delay_ms = max(0.0, raw_delay_ms - self.config.pending_advance_ms)
        return crossing_horizon_ms, delay_ms

    def _decision(
        self,
        *,
        timestamp_ms: float,
        switch: bool,
        reason: SchedulerReason,
        current_pressed: bool,
        probabilities: tuple[float, ...],
        crossing_horizon_ms: float | None = None,
        pending_due_ms: float | None = None,
        pending_delay_ms: float | None = None,
    ) -> SchedulerDecision:
        return SchedulerDecision(
            timestamp_ms=timestamp_ms,
            switch=switch,
            reason=reason,
            current_pressed=current_pressed,
            probabilities=probabilities,
            control_probability=probabilities[self._control_index],
            anticipation_probability=probabilities[self._anticipation_index],
            crossing_horizon_ms=crossing_horizon_ms,
            pending_due_ms=pending_due_ms,
            pending_delay_ms=pending_delay_ms,
        )

    def _switch_decision(
        self,
        *,
        timestamp_ms: float,
        reason: Literal["primary", "short_overdue", "pending_execute"],
        current_pressed: bool,
        probabilities: tuple[float, ...],
        crossing_horizon_ms: float | None = None,
        pending_due_ms: float | None = None,
        pending_delay_ms: float | None = None,
    ) -> SchedulerDecision:
        self._pending = None
        self._last_switch_ms = timestamp_ms
        return self._decision(
            timestamp_ms=timestamp_ms,
            switch=True,
            reason=reason,
            current_pressed=current_pressed,
            probabilities=probabilities,
            crossing_horizon_ms=crossing_horizon_ms,
            pending_due_ms=pending_due_ms,
            pending_delay_ms=pending_delay_ms,
        )

    def _inside_min_hold(self, timestamp_ms: float) -> bool:
        if self._last_switch_ms is None:
            return False
        return (
            timestamp_ms - self._last_switch_ms + 1e-6
            < self.config.min_state_hold_ms
        )

    def _hold_expiry_ms(self) -> float | None:
        if self._last_switch_ms is None:
            return None
        return self._last_switch_ms + self.config.min_state_hold_ms

    def update(
        self,
        *,
        timestamp_ms: float,
        probabilities: Sequence[float],
        current_pressed: bool,
    ) -> SchedulerDecision:
        timestamp_ms = float(timestamp_ms)
        if not math.isfinite(timestamp_ms):
            raise ValueError("timestamp_ms must be finite")
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms <= self._last_timestamp_ms
        ):
            raise ValueError("scheduler timestamps must be strictly increasing")
        self._last_timestamp_ms = timestamp_ms
        probabilities = self._validated_probabilities(probabilities)
        current_pressed = bool(current_pressed)

        # Any externally changed state invalidates an old pending transition.
        if self._pending is not None and self._pending.state_before != current_pressed:
            self._pending = None

        inside_min_hold = self._inside_min_hold(timestamp_ms)
        if inside_min_hold and not self.config.arm_pending_during_min_hold:
            self._pending = None
            return self._decision(
                timestamp_ms=timestamp_ms,
                switch=False,
                reason="min_hold",
                current_pressed=current_pressed,
                probabilities=probabilities,
            )

        control = probabilities[self._control_index]
        short_overdue = self._short_horizon_overdue(probabilities)
        if not inside_min_hold and control >= self.config.threshold:
            return self._switch_decision(
                timestamp_ms=timestamp_ms,
                reason="primary",
                current_pressed=current_pressed,
                probabilities=probabilities,
            )
        if not inside_min_hold and control < self.config.threshold and short_overdue:
            return self._switch_decision(
                timestamp_ms=timestamp_ms,
                reason="short_overdue",
                current_pressed=current_pressed,
                probabilities=probabilities,
            )

        candidate = self._candidate(probabilities)
        if self._pending is not None:
            pending = self._pending
            warning_valid = (
                candidate is not None
                or control >= self.config.threshold
                or short_overdue
            )
            if not warning_valid:
                self._pending = None
                return self._decision(
                    timestamp_ms=timestamp_ms,
                    switch=False,
                    reason="pending_cancelled",
                    current_pressed=current_pressed,
                    probabilities=probabilities,
                    crossing_horizon_ms=pending.crossing_horizon_ms,
                    pending_due_ms=pending.due_at_ms,
                    pending_delay_ms=pending.delay_ms,
                )
            if not inside_min_hold and timestamp_ms + 1e-6 >= pending.due_at_ms:
                return self._switch_decision(
                    timestamp_ms=timestamp_ms,
                    reason="pending_execute",
                    current_pressed=current_pressed,
                    probabilities=probabilities,
                    crossing_horizon_ms=pending.crossing_horizon_ms,
                    pending_due_ms=pending.due_at_ms,
                    pending_delay_ms=pending.delay_ms,
                )
            return self._decision(
                timestamp_ms=timestamp_ms,
                switch=False,
                reason="pending_wait",
                current_pressed=current_pressed,
                probabilities=probabilities,
                crossing_horizon_ms=pending.crossing_horizon_ms,
                pending_due_ms=pending.due_at_ms,
                pending_delay_ms=pending.delay_ms,
            )

        if candidate is None:
            return self._decision(
                timestamp_ms=timestamp_ms,
                switch=False,
                reason="min_hold" if inside_min_hold else "hold",
                current_pressed=current_pressed,
                probabilities=probabilities,
            )

        crossing_horizon_ms, delay_ms = candidate
        due_at_ms = timestamp_ms + delay_ms
        if inside_min_hold:
            hold_expiry_ms = self._hold_expiry_ms()
            assert hold_expiry_ms is not None
            due_at_ms = max(due_at_ms, hold_expiry_ms)
            delay_ms = due_at_ms - timestamp_ms
        self._pending = _PendingSwitch(
            state_before=current_pressed,
            armed_at_ms=timestamp_ms,
            due_at_ms=due_at_ms,
            crossing_horizon_ms=crossing_horizon_ms,
            delay_ms=delay_ms,
        )
        return self._decision(
            timestamp_ms=timestamp_ms,
            switch=False,
            reason="pending_armed",
            current_pressed=current_pressed,
            probabilities=probabilities,
            crossing_horizon_ms=crossing_horizon_ms,
            pending_due_ms=due_at_ms,
            pending_delay_ms=delay_ms,
        )
