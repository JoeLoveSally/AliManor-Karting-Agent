"""Training-sample selection and weighting utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from karting_agent.train.dataset import DatasetSample


@dataclass(frozen=True)
class SamplingConfig:
    """Relative sampling weights for temporal training samples."""

    stable_weight: float = 1.0
    transition_weight: float = 2.0
    short_correction_weight: float = 3.0
    replacement: bool = True

    def validate(self) -> None:
        if self.stable_weight <= 0:
            raise ValueError("stable_weight must be > 0")
        if self.transition_weight < self.stable_weight:
            raise ValueError("transition_weight must be >= stable_weight")
        if self.short_correction_weight < self.transition_weight:
            raise ValueError(
                "short_correction_weight must be >= transition_weight"
            )


def load_samples(path: Path) -> list[DatasetSample]:
    """Load temporal samples from a JSONL manifest."""
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                samples.append(
                    DatasetSample(
                        video=str(raw["video"]),
                        input_frame_indices=tuple(raw["input_frame_indices"]),
                        input_timestamps_ms=tuple(raw["input_timestamps_ms"]),
                        target_frame_index=int(raw["target_frame_index"]),
                        target_timestamp_ms=float(raw["target_timestamp_ms"]),
                        target_pressed=bool(raw["target_pressed"]),
                        transition_distance_ms=(
                            None
                            if raw["transition_distance_ms"] is None
                            else float(raw["transition_distance_ms"])
                        ),
                        near_transition=bool(raw["near_transition"]),
                        near_short_correction=bool(
                            raw["near_short_correction"]
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid dataset sample at {path}:{line_number}"
                ) from exc
    return samples


def select_samples_by_videos(
    samples: Iterable[DatasetSample],
    videos: Iterable[str],
) -> list[DatasetSample]:
    """Return samples whose video field exactly matches one of ``videos``."""
    allowed = set(videos)
    return [sample for sample in samples if sample.video in allowed]


def sample_weight(
    sample: DatasetSample,
    config: SamplingConfig,
) -> float:
    """Return one relative sampling weight.

    Short-correction samples take precedence over generic transition samples.
    """
    config.validate()
    if sample.near_short_correction:
        return config.short_correction_weight
    if sample.near_transition:
        return config.transition_weight
    return config.stable_weight


def build_sample_weights(
    samples: Sequence[DatasetSample],
    config: SamplingConfig,
) -> list[float]:
    """Build weights aligned one-to-one with ``samples``."""
    config.validate()
    return [sample_weight(sample, config) for sample in samples]


def sampling_summary(
    samples: Sequence[DatasetSample],
    config: SamplingConfig,
) -> dict[str, float | int]:
    """Summarize raw and weighted sample composition."""
    config.validate()
    total = len(samples)
    stable = sum(
        not sample.near_transition and not sample.near_short_correction
        for sample in samples
    )
    short = sum(sample.near_short_correction for sample in samples)
    transition = total - stable - short

    weighted_stable = stable * config.stable_weight
    weighted_transition = transition * config.transition_weight
    weighted_short = short * config.short_correction_weight
    weighted_total = weighted_stable + weighted_transition + weighted_short

    def share(value: float, denominator: float) -> float:
        return value / denominator if denominator else 0.0

    return {
        "samples": total,
        "stable_samples": stable,
        "transition_samples": transition,
        "short_correction_samples": short,
        "stable_raw_share": share(stable, total),
        "transition_raw_share": share(transition, total),
        "short_correction_raw_share": share(short, total),
        "stable_weighted_share": share(weighted_stable, weighted_total),
        "transition_weighted_share": share(weighted_transition, weighted_total),
        "short_correction_weighted_share": share(weighted_short, weighted_total),
    }


def make_weighted_sampler(
    samples: Sequence[DatasetSample],
    config: SamplingConfig,
    *,
    generator=None,
):
    """Create a PyTorch ``WeightedRandomSampler`` for training.

    PyTorch is imported lazily so dataset tooling does not require the train
    optional dependency.
    """
    config.validate()
    if not samples:
        raise ValueError("cannot create a sampler for an empty dataset")

    try:
        from torch.utils.data import WeightedRandomSampler
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: pip install -e ".[train]"'
        ) from exc

    return WeightedRandomSampler(
        weights=build_sample_weights(samples, config),
        num_samples=len(samples),
        replacement=config.replacement,
        generator=generator,
    )
