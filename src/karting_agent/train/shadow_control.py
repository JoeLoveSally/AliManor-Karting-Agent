"""Deterministic, read-only timing simulator for expert-video policy diagnosis.

It never sends controls to a game. The video is a fixed expert trajectory,
so simulated control histories cannot change later observed game pixels.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import heapq
import math

import numpy as np


@dataclass(frozen=True)
class AppliedAction:
    time_ms: float
    pressed: bool


@dataclass(frozen=True)
class Proposal:
    observe_ms: float
    target_ms: float
    ready_ms: float
    execute_ms: float
    pressed: bool
    probability: float


class ShadowController:
    """Causal t+H action queue and recorded executed-action feedback.

    Proposals take effect at max(target_time, inference_ready_time).
    Calls to advance_to() apply only due decisions. In particular, a queued
    future action is *never* included in the observed control-history features.
    """

    def __init__(self, *, initial_pressed: bool, horizon_ms: float = 100.0,
                 latency_ms: float = 0.0, threshold: float = 0.5) -> None:
        if (not math.isfinite(horizon_ms) or horizon_ms <= 0 or
                not math.isfinite(latency_ms) or latency_ms < 0 or
                not 0 < threshold < 1):
            raise ValueError("invalid horizon/latency/threshold")
        self.horizon_ms = float(horizon_ms)
        self.latency_ms = float(latency_ms)
        self.threshold = float(threshold)
        self.now_ms = 0.0
        self.events: list[AppliedAction] = [AppliedAction(0.0, bool(initial_pressed))]
        self.proposals: list[Proposal] = []
        self._due: list[tuple[float, int, Proposal]] = []
        self._seq = 0
        self.redundant = 0

    @property
    def pressed(self) -> bool:
        return self.events[-1].pressed

    def advance_to(self, time_ms: float) -> None:
        if not math.isfinite(time_ms) or time_ms < self.now_ms - 1e-6:
            raise ValueError("shadow clock must advance monotonically")
        while self._due and self._due[0][0] <= time_ms + 1e-6:
            due, _, proposal = heapq.heappop(self._due)
            if proposal.pressed == self.pressed:
                self.redundant += 1
            else:
                self.events.append(AppliedAction(due, proposal.pressed))
        self.now_ms = float(time_ms)

    def control_features(self, input_timestamps_ms) -> np.ndarray:
        """[T,3]: executed state, switch age / 500ms, frame dt / 200ms.

        Assumes advance_to(observation_t) happened before the call. No oracle
        timeline is consulted, except the initial boolean used for bootstrapping.
        """
        stamps = tuple(float(x) for x in input_timestamps_ms)
        if (not stamps or any(not math.isfinite(x) or x < 0 or
                              x > self.now_ms + 1e-6 for x in stamps) or
                any(b < a for a, b in zip(stamps, stamps[1:]))):
            raise ValueError("invalid historical input timestamps")
        times = [event.time_ms for event in self.events]
        result = np.zeros((len(stamps), 3), dtype=np.float32)
        for i, t in enumerate(stamps):
            idx = bisect_right(times, t + 1e-7) - 1
            event = self.events[idx]
            result[i, 0] = float(event.pressed)
            result[i, 1] = min(max(t - event.time_ms, 0.0), 500.0) / 500.0
            result[i, 2] = min(max(t - stamps[i-1], 0.0), 200.0) / 200.0 if i else 0.0
        return result

    def propose(self, *, observe_ms: float, target_ms: float,
                probability: float) -> Proposal:
        if (not math.isfinite(observe_ms) or
                abs(observe_ms - self.now_ms) > 1e-4 or
                not math.isfinite(target_ms) or
                abs(target_ms - observe_ms - self.horizon_ms) > 1e-3 or
                not math.isfinite(probability) or not 0.0 <= probability <= 1.0):
            raise ValueError("invalid, premature or inconsistent proposal")
        ready = observe_ms + self.latency_ms
        due = max(target_ms, ready)
        proposal = Proposal(observe_ms, target_ms, ready, due,
                            bool(probability >= self.threshold), float(probability))
        if self.proposals and due < self.proposals[-1].execute_ms - 1e-6:
            raise ValueError("nonmonotonic execution deadlines")
        self._seq += 1
        heapq.heappush(self._due, (due, self._seq, proposal))
        self.proposals.append(proposal)
        return proposal

    def state_at(self, time_ms: float) -> bool:
        """Only query *already simulated* past, never a pending/future state."""
        if not math.isfinite(time_ms) or time_ms < 0 or time_ms > self.now_ms + 1e-6:
            raise ValueError("cannot inspect future shadow state")
        times = [event.time_ms for event in self.events]
        return self.events[bisect_right(times, time_ms + 1e-7)-1].pressed

    def flush(self) -> None:
        if self._due:
            self.advance_to(max(self.now_ms, max(entry[0] for entry in self._due)))
