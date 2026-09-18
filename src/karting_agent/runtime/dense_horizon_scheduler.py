"""Schedule V5 dense-horizon switch forecasts on an H0-anchored timeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Literal, Sequence


DenseSchedulerReason = Literal[
    "hold",
    "min_hold",
    "primary",
    "pending_armed",
    "pending_wait",
    "pending_execute",
    "pending_cancelled",
]


@dataclass(frozen=True)
class DenseHorizonSchedulerConfig:
    """Configuration for H0-anchored dense future transition scheduling.

    Horizon 0 is the immediate control head. Future heads are not executed
    immediately. A rising threshold crossing through time for future horizon h
    estimates the expert transition time as crossing_time + h.

    The first persistent future warning arms a fixed deadline. Later warnings
    may confirm it but never push it later.
    """

    horizons_ms: tuple[float, ...]
    threshold: float = 0.6
    min_state_hold_ms: float = 100.0

    @property
    def control_horizon_ms(self) -> float:
        return 0.0

    def validate(self) -> None:
        if len(self.horizons_ms) < 2:
            raise ValueError("at least H0 and one future horizon are required")
        if self.horizons_ms[0] != 0.0:
            raise ValueError("dense scheduler requires H0 as the first horizon")
        if tuple(sorted(self.horizons_ms)) != self.horizons_ms:
            raise ValueError("prediction horizons must be sorted")
        if len(set(self.horizons_ms)) != len(self.horizons_ms):
            raise ValueError("prediction horizons must be unique")
        if any(not math.isfinite(value) or value < 0.0 for value in self.horizons_ms):
            raise ValueError("prediction horizons must be finite and >= 0")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if not math.isfinite(self.min_state_hold_ms) or self.min_state_hold_ms < 0.0:
            raise ValueError("min_state_hold_ms must be finite and >= 0")


@dataclass(frozen=True)
class DenseSchedulerDecision:
    timestamp_ms: float
    switch: bool
    reason: DenseSchedulerReason
    current_pressed: bool
    probabilities: tuple[float, ...]
    control_probability: float
    anticipation_probability: float
    crossing_horizon_ms: float | None = None
    pending_due_ms: float | None = None
    pending_delay_ms: float | None = None


@dataclass(frozen=True)
class DensePendingExecution:
    state_before: bool
    armed_at_ms: float
    due_at_ms: float
    source_horizon_ms: float
    crossing_at_ms: float
    delay_ms: float


@dataclass(frozen=True)
class _PendingSwitch:
    state_before: bool
    armed_at_ms: float
    due_at_ms: float
    source_horizon_ms: float
    crossing_at_ms: float


class DenseHorizonSwitchScheduler:
    """Use H0 immediately and future-head temporal crossings as fixed deadlines."""

    def __init__(self, config: DenseHorizonSchedulerConfig) -> None:
        config.validate()
        self.config = config
        self._pending: _PendingSwitch | None = None
        self._last_switch_ms: float | None = None
        self._last_timestamp_ms: float | None = None
        self._history_state: bool | None = None
        self._previous_probabilities: tuple[float, ...] | None = None
        self._previous_probability_timestamp_ms: float | None = None

    @property
    def pending_due_ms(self) -> float | None:
        return None if self._pending is None else self._pending.due_at_ms

    @property
    def last_switch_ms(self) -> float | None:
        return self._last_switch_ms

    def reset(self) -> None:
        self._pending = None
        self._last_switch_ms = None
        self._last_timestamp_ms = None
        self._history_state = None
        self._previous_probabilities = None
        self._previous_probability_timestamp_ms = None

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

    def _inside_min_hold(self, timestamp_ms: float) -> bool:
        return (
            self._last_switch_ms is not None
            and timestamp_ms < self._last_switch_ms + self.config.min_state_hold_ms - 1e-6
        )

    def _hold_expiry_ms(self) -> float | None:
        if self._last_switch_ms is None:
            return None
        return self._last_switch_ms + self.config.min_state_hold_ms

    def _clear_probability_history(self, current_pressed: bool) -> None:
        self._history_state = current_pressed
        self._previous_probabilities = None
        self._previous_probability_timestamp_ms = None

    def _rising_crossings(
        self,
        *,
        timestamp_ms: float,
        probabilities: tuple[float, ...],
    ) -> list[tuple[float, float, float]]:
        """Return (due_ms, horizon_ms, crossing_at_ms) for newly crossed future heads."""

        threshold = self.config.threshold
        previous = self._previous_probabilities
        previous_timestamp = self._previous_probability_timestamp_ms
        candidates: list[tuple[float, float, float]] = []

        for index, horizon_ms in enumerate(self.config.horizons_ms[1:], start=1):
            current_probability = probabilities[index]
            if previous is None or previous_timestamp is None:
                if current_probability >= threshold:
                    crossing_at_ms = timestamp_ms
                else:
                    continue
            else:
                previous_probability = previous[index]
                if not (previous_probability < threshold <= current_probability):
                    continue
                denominator = current_probability - previous_probability
                if denominator <= 0.0:
                    crossing_at_ms = timestamp_ms
                else:
                    fraction = (threshold - previous_probability) / denominator
                    fraction = min(1.0, max(0.0, fraction))
                    crossing_at_ms = previous_timestamp + fraction * (
                        timestamp_ms - previous_timestamp
                    )
            candidates.append(
                (crossing_at_ms + horizon_ms, horizon_ms, crossing_at_ms)
            )

        return candidates

    def _remember(
        self,
        *,
        timestamp_ms: float,
        probabilities: tuple[float, ...],
        current_pressed: bool,
    ) -> None:
        self._history_state = current_pressed
        self._previous_probabilities = probabilities
        self._previous_probability_timestamp_ms = timestamp_ms
        self._last_timestamp_ms = timestamp_ms

    def _decision(
        self,
        *,
        timestamp_ms: float,
        switch: bool,
        reason: DenseSchedulerReason,
        current_pressed: bool,
        probabilities: tuple[float, ...],
        source_horizon_ms: float | None = None,
        pending_due_ms: float | None = None,
    ) -> DenseSchedulerDecision:
        pending_delay_ms = (
            None if pending_due_ms is None else max(0.0, pending_due_ms - timestamp_ms)
        )
        return DenseSchedulerDecision(
            timestamp_ms=timestamp_ms,
            switch=switch,
            reason=reason,
            current_pressed=current_pressed,
            probabilities=probabilities,
            control_probability=probabilities[0],
            anticipation_probability=max(probabilities[1:]),
            crossing_horizon_ms=source_horizon_ms,
            pending_due_ms=pending_due_ms,
            pending_delay_ms=pending_delay_ms,
        )

    def _switch_decision(
        self,
        *,
        timestamp_ms: float,
        reason: Literal["primary", "pending_execute"],
        current_pressed: bool,
        probabilities: tuple[float, ...],
        source_horizon_ms: float | None = None,
    ) -> DenseSchedulerDecision:
        self._pending = None
        self._last_switch_ms = timestamp_ms
        self._previous_probabilities = None
        self._previous_probability_timestamp_ms = None
        self._history_state = None
        self._last_timestamp_ms = timestamp_ms
        return self._decision(
            timestamp_ms=timestamp_ms,
            switch=True,
            reason=reason,
            current_pressed=current_pressed,
            probabilities=probabilities,
            source_horizon_ms=source_horizon_ms,
        )

    def execute_pending_if_due(
        self,
        *,
        timestamp_ms: float,
        current_pressed: bool,
    ) -> DensePendingExecution | None:
        timestamp_ms = float(timestamp_ms)
        if not math.isfinite(timestamp_ms):
            raise ValueError("timestamp_ms must be finite")
        pending = self._pending
        if pending is None:
            return None
        if pending.state_before != bool(current_pressed):
            self._pending = None
            return None
        if self._inside_min_hold(timestamp_ms):
            return None
        if timestamp_ms + 1e-6 < pending.due_at_ms:
            return None

        self._pending = None
        self._last_switch_ms = timestamp_ms
        self._previous_probabilities = None
        self._previous_probability_timestamp_ms = None
        self._history_state = None
        return DensePendingExecution(
            state_before=pending.state_before,
            armed_at_ms=pending.armed_at_ms,
            due_at_ms=pending.due_at_ms,
            source_horizon_ms=pending.source_horizon_ms,
            crossing_at_ms=pending.crossing_at_ms,
            delay_ms=max(0.0, pending.due_at_ms - pending.armed_at_ms),
        )

    def update(
        self,
        *,
        timestamp_ms: float,
        probabilities: Sequence[float],
        current_pressed: bool,
    ) -> DenseSchedulerDecision:
        timestamp_ms = float(timestamp_ms)
        if not math.isfinite(timestamp_ms):
            raise ValueError("timestamp_ms must be finite")
        if self._last_timestamp_ms is not None and timestamp_ms < self._last_timestamp_ms - 1e-6:
            raise ValueError("timestamps must be nondecreasing")

        current_pressed = bool(current_pressed)
        values = self._validated_probabilities(probabilities)
        if self._history_state is not None and self._history_state != current_pressed:
            self._clear_probability_history(current_pressed)
        if self._pending is not None and self._pending.state_before != current_pressed:
            self._pending = None

        h0 = values[0]
        if h0 >= self.config.threshold:
            if not self._inside_min_hold(timestamp_ms):
                return self._switch_decision(
                    timestamp_ms=timestamp_ms,
                    reason="primary",
                    current_pressed=current_pressed,
                    probabilities=values,
                )

            hold_expiry = self._hold_expiry_ms()
            assert hold_expiry is not None
            if self._pending is None:
                self._pending = _PendingSwitch(
                    state_before=current_pressed,
                    armed_at_ms=timestamp_ms,
                    due_at_ms=hold_expiry,
                    source_horizon_ms=0.0,
                    crossing_at_ms=timestamp_ms,
                )
                reason: DenseSchedulerReason = "pending_armed"
            else:
                reason = "pending_wait"
            self._remember(
                timestamp_ms=timestamp_ms,
                probabilities=values,
                current_pressed=current_pressed,
            )
            return self._decision(
                timestamp_ms=timestamp_ms,
                switch=False,
                reason=reason,
                current_pressed=current_pressed,
                probabilities=values,
                source_horizon_ms=0.0,
                pending_due_ms=self._pending.due_at_ms,
            )

        new_crossings = self._rising_crossings(
            timestamp_ms=timestamp_ms,
            probabilities=values,
        )
        any_future_warning = any(
            probability >= self.config.threshold for probability in values[1:]
        )

        if self._pending is None and new_crossings:
            due_values = [item[0] for item in new_crossings]
            representative_due = float(median(due_values))
            representative = min(
                new_crossings,
                key=lambda item: abs(item[0] - representative_due),
            )
            due_at_ms, source_horizon_ms, crossing_at_ms = representative
            hold_expiry = self._hold_expiry_ms()
            if hold_expiry is not None:
                due_at_ms = max(due_at_ms, hold_expiry)
            self._pending = _PendingSwitch(
                state_before=current_pressed,
                armed_at_ms=timestamp_ms,
                due_at_ms=due_at_ms,
                source_horizon_ms=source_horizon_ms,
                crossing_at_ms=crossing_at_ms,
            )
            reason = "pending_armed"
        elif self._pending is not None and not any_future_warning:
            old_pending = self._pending
            self._pending = None
            self._remember(
                timestamp_ms=timestamp_ms,
                probabilities=values,
                current_pressed=current_pressed,
            )
            return self._decision(
                timestamp_ms=timestamp_ms,
                switch=False,
                reason="pending_cancelled",
                current_pressed=current_pressed,
                probabilities=values,
                source_horizon_ms=old_pending.source_horizon_ms,
                pending_due_ms=old_pending.due_at_ms,
            )
        elif self._pending is not None:
            reason = "pending_wait"
        else:
            reason = "min_hold" if self._inside_min_hold(timestamp_ms) else "hold"

        pending = self._pending
        if (
            pending is not None
            and timestamp_ms + 1e-6 >= pending.due_at_ms
            and not self._inside_min_hold(timestamp_ms)
        ):
            return self._switch_decision(
                timestamp_ms=timestamp_ms,
                reason="pending_execute",
                current_pressed=current_pressed,
                probabilities=values,
                source_horizon_ms=pending.source_horizon_ms,
            )

        self._remember(
            timestamp_ms=timestamp_ms,
            probabilities=values,
            current_pressed=current_pressed,
        )
        return self._decision(
            timestamp_ms=timestamp_ms,
            switch=False,
            reason=reason,
            current_pressed=current_pressed,
            probabilities=values,
            source_horizon_ms=(
                None if pending is None else pending.source_horizon_ms
            ),
            pending_due_ms=None if pending is None else pending.due_at_ms,
        )
