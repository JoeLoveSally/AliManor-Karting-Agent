"""Consensus scheduler for V5 dense future switch horizons."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Literal, Sequence


ConsensusReason = Literal[
    "hold",
    "min_hold",
    "primary",
    "consensus_armed",
    "consensus_wait",
    "consensus_cancelled",
    "consensus_execute",
]


@dataclass(frozen=True)
class ConsensusDenseConfig:
    horizons_ms: tuple[float, ...]
    threshold: float = 0.6
    min_state_hold_ms: float = 100.0
    min_consensus_heads: int = 3
    consensus_window_ms: float = 60.0

    @property
    def control_horizon_ms(self) -> float:
        return 0.0

    def validate(self) -> None:
        if len(self.horizons_ms) < 4:
            raise ValueError(\n                "consensus scheduler requires H0 plus at least three future heads"\n            )
        if self.horizons_ms[0] != 0.0:
            raise ValueError("consensus scheduler requires H0 as the first horizon")
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
        if self.min_consensus_heads < 2:
            raise ValueError("min_consensus_heads must be >= 2")
        if self.min_consensus_heads > len(self.horizons_ms) - 1:
            raise ValueError("min_consensus_heads exceeds number of future heads")
        if (\n            not math.isfinite(self.consensus_window_ms)\n            or self.consensus_window_ms < 0.0\n        ):
            raise ValueError("consensus_window_ms must be finite and >= 0")


@dataclass(frozen=True)
class ConsensusDecision:
    timestamp_ms: float
    switch: bool
    reason: ConsensusReason
    current_pressed: bool
    probabilities: tuple[float, ...]
    control_probability: float
    pending_due_ms: float | None = None
    pending_delay_ms: float | None = None
    consensus_heads_ms: tuple[float, ...] = ()
    consensus_due_spread_ms: float | None = None


@dataclass(frozen=True)
class ConsensusPendingExecution:
    state_before: bool
    armed_at_ms: float
    due_at_ms: float
    support_heads_ms: tuple[float, ...]
    due_spread_ms: float
    delay_ms: float


@dataclass(frozen=True)
class _Forecast:
    horizon_ms: float
    crossing_at_ms: float
    due_at_ms: float


@dataclass(frozen=True)
class _Pending:
    state_before: bool
    armed_at_ms: float
    due_at_ms: float
    support_heads_ms: tuple[float, ...]
    due_spread_ms: float


class ConsensusDenseHorizonScheduler:
    """Arm a future switch only when several horizons agree on transition time."""

    def __init__(self, config: ConsensusDenseConfig) -> None:
        config.validate()
        self.config = config
        self._pending: _Pending | None = None
        self._last_switch_ms: float | None = None
        self._last_timestamp_ms: float | None = None
        self._history_state: bool | None = None
        self._previous_probabilities: tuple[float, ...] | None = None
        self._previous_timestamp_ms: float | None = None
        self._forecasts: dict[float, _Forecast] = {}

    @property
    def pending_due_ms(self) -> float | None:
        return None if self._pending is None else self._pending.due_at_ms

    def reset(self) -> None:
        self._pending = None
        self._last_switch_ms = None
        self._last_timestamp_ms = None
        self._history_state = None
        self._previous_probabilities = None
        self._previous_timestamp_ms = None
        self._forecasts.clear()

    def _inside_min_hold(self, timestamp_ms: float) -> bool:
        return (
            self._last_switch_ms is not None
            and timestamp_ms\n            < self._last_switch_ms + self.config.min_state_hold_ms - 1e-6
        )

    def _hold_expiry_ms(self) -> float | None:
        if self._last_switch_ms is None:
            return None
        return self._last_switch_ms + self.config.min_state_hold_ms

    def _clear_state_history(self, current_pressed: bool) -> None:
        self._pending = None
        self._history_state = current_pressed
        self._previous_probabilities = None
        self._previous_timestamp_ms = None
        self._forecasts.clear()

    def _validate_probabilities(self, values: Sequence[float]) -> tuple[float, ...]:
        probabilities = tuple(float(value) for value in values)
        if len(probabilities) != len(self.config.horizons_ms):
            raise ValueError("probability width does not match prediction horizons")
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in probabilities
        ):
            raise ValueError("switch probabilities must be finite and in [0, 1]")
        return probabilities

    def _refresh_forecasts(
        self,
        *,
        timestamp_ms: float,
        probabilities: tuple[float, ...],
    ) -> None:
        threshold = self.config.threshold
        previous = self._previous_probabilities
        previous_timestamp = self._previous_timestamp_ms

        for index, horizon_ms in enumerate(self.config.horizons_ms[1:], start=1):
            current_probability = probabilities[index]
            if current_probability < threshold:
                self._forecasts.pop(horizon_ms, None)
                continue

            if horizon_ms in self._forecasts:
                continue

            if previous is None or previous_timestamp is None:
                crossing_at_ms = timestamp_ms
            else:
                previous_probability = previous[index]
                if previous_probability >= threshold:
                    crossing_at_ms = timestamp_ms
                else:
                    denominator = current_probability - previous_probability
                    if denominator <= 0.0:
                        crossing_at_ms = timestamp_ms
                    else:
                        fraction = (threshold - previous_probability) / denominator
                        fraction = min(1.0, max(0.0, fraction))
                        crossing_at_ms = previous_timestamp + fraction * (
                            timestamp_ms - previous_timestamp
                        )

            self._forecasts[horizon_ms] = _Forecast(
                horizon_ms=horizon_ms,
                crossing_at_ms=crossing_at_ms,
                due_at_ms=crossing_at_ms + horizon_ms,
            )

        # Forecasts that are far in the past cannot describe the next transition.
        stale_before = timestamp_ms - self.config.consensus_window_ms
        self._forecasts = {
            horizon: forecast
            for horizon, forecast in self._forecasts.items()
            if forecast.due_at_ms >= stale_before
        }

    def _best_consensus(self) -> tuple[float, tuple[float, ...], float] | None:
        forecasts = sorted(self._forecasts.values(), key=lambda item: item.due_at_ms)
        if len(forecasts) < self.config.min_consensus_heads:
            return None

        best: tuple[int, float, float, tuple[_Forecast, ...]] | None = None
        for start in range(len(forecasts)):
            cluster: list[_Forecast] = []
            for forecast in forecasts[start:]:
                if not cluster:
                    cluster.append(forecast)
                    continue
                if (\n                    forecast.due_at_ms - cluster[0].due_at_ms\n                    <= self.config.consensus_window_ms + 1e-6\n                ):
                    cluster.append(forecast)
                else:
                    break
            if len(cluster) < self.config.min_consensus_heads:
                continue
            due_values = [item.due_at_ms for item in cluster]
            spread = max(due_values) - min(due_values)
            med = float(median(due_values))
            score = (len(cluster), -spread, -med, tuple(cluster))
            if best is None or score[:3] > best[:3]:
                best = score

        if best is None:
            return None
        cluster = best[3]
        due_values = [item.due_at_ms for item in cluster]
        due_at_ms = float(median(due_values))
        spread_ms = max(due_values) - min(due_values)
        heads = tuple(item.horizon_ms for item in cluster)
        return due_at_ms, heads, spread_ms

    def _remember(
        self,
        *,
        timestamp_ms: float,
        probabilities: tuple[float, ...],
        current_pressed: bool,
    ) -> None:
        self._history_state = current_pressed
        self._previous_probabilities = probabilities
        self._previous_timestamp_ms = timestamp_ms
        self._last_timestamp_ms = timestamp_ms

    def _decision(
        self,
        *,
        timestamp_ms: float,
        switch: bool,
        reason: ConsensusReason,
        current_pressed: bool,
        probabilities: tuple[float, ...],
        pending_due_ms: float | None = None,
        heads: tuple[float, ...] = (),
        spread_ms: float | None = None,
    ) -> ConsensusDecision:
        return ConsensusDecision(
            timestamp_ms=timestamp_ms,
            switch=switch,
            reason=reason,
            current_pressed=current_pressed,
            probabilities=probabilities,
            control_probability=probabilities[0],
            pending_due_ms=pending_due_ms,
            pending_delay_ms=(
                None\n                if pending_due_ms is None\n                else max(0.0, pending_due_ms - timestamp_ms)
            ),
            consensus_heads_ms=heads,
            consensus_due_spread_ms=spread_ms,
        )

    def _switch_now(
        self,
        *,
        timestamp_ms: float,
        reason: Literal["primary", "consensus_execute"],
        current_pressed: bool,
        probabilities: tuple[float, ...],
        heads: tuple[float, ...] = (),
        spread_ms: float | None = None,
    ) -> ConsensusDecision:
        self._pending = None
        self._last_switch_ms = timestamp_ms
        self._history_state = None
        self._previous_probabilities = None
        self._previous_timestamp_ms = None
        self._forecasts.clear()
        self._last_timestamp_ms = timestamp_ms
        return self._decision(
            timestamp_ms=timestamp_ms,
            switch=True,
            reason=reason,
            current_pressed=current_pressed,
            probabilities=probabilities,
            heads=heads,
            spread_ms=spread_ms,
        )

    def execute_pending_if_due(
        self,
        *,
        timestamp_ms: float,
        current_pressed: bool,
    ) -> ConsensusPendingExecution | None:
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
        self._history_state = None
        self._previous_probabilities = None
        self._previous_timestamp_ms = None
        self._forecasts.clear()
        return ConsensusPendingExecution(
            state_before=pending.state_before,
            armed_at_ms=pending.armed_at_ms,
            due_at_ms=pending.due_at_ms,
            support_heads_ms=pending.support_heads_ms,
            due_spread_ms=pending.due_spread_ms,
            delay_ms=max(0.0, pending.due_at_ms - pending.armed_at_ms),
        )

    def update(
        self,
        *,
        timestamp_ms: float,
        probabilities: Sequence[float],
        current_pressed: bool,
    ) -> ConsensusDecision:
        timestamp_ms = float(timestamp_ms)
        if not math.isfinite(timestamp_ms):
            raise ValueError("timestamp_ms must be finite")
        if (\n            self._last_timestamp_ms is not None\n            and timestamp_ms < self._last_timestamp_ms - 1e-6\n        ):
            raise ValueError("timestamps must be nondecreasing")

        current_pressed = bool(current_pressed)
        values = self._validate_probabilities(probabilities)
        if self._history_state is not None and self._history_state != current_pressed:
            self._clear_state_history(current_pressed)

        if values[0] >= self.config.threshold:
            if not self._inside_min_hold(timestamp_ms):
                return self._switch_now(
                    timestamp_ms=timestamp_ms,
                    reason="primary",
                    current_pressed=current_pressed,
                    probabilities=values,
                )
            expiry = self._hold_expiry_ms()
            assert expiry is not None
            if self._pending is None:
                self._pending = _Pending(
                    state_before=current_pressed,
                    armed_at_ms=timestamp_ms,
                    due_at_ms=expiry,
                    support_heads_ms=(0.0,),
                    due_spread_ms=0.0,
                )
                reason: ConsensusReason = "consensus_armed"
            else:
                reason = "consensus_wait"
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
                pending_due_ms=self._pending.due_at_ms,
                heads=self._pending.support_heads_ms,
                spread_ms=self._pending.due_spread_ms,
            )

        self._refresh_forecasts(timestamp_ms=timestamp_ms, probabilities=values)
        consensus = self._best_consensus()

        if self._pending is None and consensus is not None:
            due_at_ms, heads, spread_ms = consensus
            expiry = self._hold_expiry_ms()
            if expiry is not None:
                due_at_ms = max(due_at_ms, expiry)
            self._pending = _Pending(
                state_before=current_pressed,
                armed_at_ms=timestamp_ms,
                due_at_ms=due_at_ms,
                support_heads_ms=heads,
                due_spread_ms=spread_ms,
            )
            reason = "consensus_armed"
        elif self._pending is not None:
            consensus_matches_pending = (
                consensus is not None
                and abs(consensus[0] - self._pending.due_at_ms)
                <= self.config.consensus_window_ms + 1e-6
            )
            if not consensus_matches_pending:
                old = self._pending
                self._pending = None
                self._remember(
                    timestamp_ms=timestamp_ms,
                    probabilities=values,
                    current_pressed=current_pressed,
                )
                return self._decision(
                    timestamp_ms=timestamp_ms,
                    switch=False,
                    reason="consensus_cancelled",
                    current_pressed=current_pressed,
                    probabilities=values,
                    pending_due_ms=old.due_at_ms,
                    heads=old.support_heads_ms,
                    spread_ms=old.due_spread_ms,
                )
            reason = "consensus_wait"
        else:
            reason = "min_hold" if self._inside_min_hold(timestamp_ms) else "hold"

        pending = self._pending
        if (
            pending is not None
            and timestamp_ms + 1e-6 >= pending.due_at_ms
            and not self._inside_min_hold(timestamp_ms)
        ):
            return self._switch_now(
                timestamp_ms=timestamp_ms,
                reason="consensus_execute",
                current_pressed=current_pressed,
                probabilities=values,
                heads=pending.support_heads_ms,
                spread_ms=pending.due_spread_ms,
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
            pending_due_ms=None if pending is None else pending.due_at_ms,
            heads=() if pending is None else pending.support_heads_ms,
            spread_ms=None if pending is None else pending.due_spread_ms,
        )
