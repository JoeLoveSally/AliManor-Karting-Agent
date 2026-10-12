"""Training-only, reproducible causal action-history robustness augmentation.

A single sample's history is replaced with a plausible constant held state.
Do not use in validation/test or interpret as a physically realized trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class HistoryCorruptionConfig:
    probability: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be finite and between 0 and 1")


class HistoryCorruptor:
    """Replace training feedback, keeping [state, switch age, dt] coherent.

    With fixed per-sample probability, draw a constant held state (0/1) and a
    nonnegative initial action age. Reconstruct age through observed causal
    time differences, preserving channel 2 and the supervision label. Unlike
    independent bit flips, this cannot fabricate a stream of impossible short
    PRESS/RELEASE transitions within a five-frame history window.
    """

    def __init__(self, config: HistoryCorruptionConfig, *, seed: int = 9242):
        config.validate()
        self.config = config
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def __call__(self, control: torch.Tensor) -> torch.Tensor:
        if control.ndim != 3 or control.shape[-1] != 3:
            raise ValueError("expected [batch, timesteps, 3] control features")
        if control.device.type != "cpu":
            raise ValueError("augment causal history on CPU before device transfer")
        if not torch.isfinite(control).all():
            raise ValueError("nonfinite control features")
        if not torch.all((control[:, :, 2] >= 0) & (control[:, :, 2] <= 1)):
            raise ValueError("invalid causal timestep feature")
        if self.config.probability == 0:
            return control

        result = control.clone()
        batch, steps, _ = result.shape
        selected = torch.rand(batch, generator=self.generator) < self.config.probability
        held = torch.randint(0, 2, (batch,), generator=self.generator).float()
        initial_age = torch.rand(batch, generator=self.generator) * 500.0
        elapsed = torch.cumsum(control[:, :, 2] * 200.0, dim=1)
        ages = ((initial_age[:, None] + elapsed).clamp(max=500.0) / 500.0)
        result[selected, :, 0] = held[selected, None]
        result[selected, :, 1] = ages[selected]
        return result
