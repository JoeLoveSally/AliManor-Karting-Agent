"""Training-sample selection, splitting, weighting, and epoch utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from karting_agent.train.dataset import DatasetSample
from karting_agent.train.evaluator import BinaryMetricAccumulator


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


@dataclass(frozen=True)
class TrainLoopConfig:
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")


@dataclass(frozen=True)
class VideoSplit:
    """Video-level train/validation/test partition."""

    name: str
    strategy: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        groups = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        for name, videos in groups.items():
            if not videos:
                raise ValueError(f"{name} split must not be empty")
            if len(videos) != len(set(videos)):
                raise ValueError(f"{name} split contains duplicate videos")

        names = tuple(groups)
        for index, left_name in enumerate(names):
            left = set(groups[left_name])
            for right_name in names[index + 1 :]:
                overlap = left.intersection(groups[right_name])
                if overlap:
                    joined = ", ".join(sorted(overlap))
                    raise ValueError(
                        f"{left_name}/{right_name} split overlap: {joined}"
                    )

    @property
    def all_videos(self) -> tuple[str, ...]:
        return self.train + self.validation + self.test


def _load_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def load_train_loop_config(path: Path) -> TrainLoopConfig:
    raw = _load_yaml(path).get("train", {})
    config = TrainLoopConfig(
        batch_size=int(raw.get("batch_size", 64)),
        epochs=int(raw.get("epochs", 30)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        num_workers=int(raw.get("num_workers", 0)),
        seed=int(raw.get("seed", 42)),
    )
    config.validate()
    return config


def load_sampling_config(path: Path) -> SamplingConfig:
    raw = _load_yaml(path).get("sampling", {})
    config = SamplingConfig(
        stable_weight=float(raw.get("stable_weight", 1.0)),
        transition_weight=float(raw.get("transition_weight", 2.0)),
        short_correction_weight=float(raw.get("short_correction_weight", 3.0)),
        replacement=bool(raw.get("replacement", True)),
    )
    config.validate()
    return config


def load_video_split(path: Path) -> VideoSplit:
    raw = _load_yaml(path).get("split")
    if not isinstance(raw, dict):
        raise ValueError(f"missing split mapping in config: {path}")

    config = VideoSplit(
        name=str(raw.get("name", "unnamed")),
        strategy=str(raw.get("strategy", "video_holdout")),
        train=tuple(str(item) for item in raw.get("train", ())),
        validation=tuple(str(item) for item in raw.get("validation", ())),
        test=tuple(str(item) for item in raw.get("test", ())),
    )
    config.validate()
    return config


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


def partition_samples(
    samples: Sequence[DatasetSample],
    split: VideoSplit,
) -> dict[str, list[DatasetSample]]:
    """Partition samples by complete videos and reject missing/unassigned videos."""
    split.validate()
    available = {sample.video for sample in samples}
    configured = set(split.all_videos)

    missing = configured - available
    if missing:
        raise ValueError(
            "split references videos absent from samples: "
            + ", ".join(sorted(missing))
        )

    unassigned = available - configured
    if unassigned:
        raise ValueError(
            "samples contain videos not assigned to a split: "
            + ", ".join(sorted(unassigned))
        )

    return {
        "train": select_samples_by_videos(samples, split.train),
        "validation": select_samples_by_videos(samples, split.validation),
        "test": select_samples_by_videos(samples, split.test),
    }


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


def run_epoch(
    model,
    data_loader,
    device,
    *,
    optimizer=None,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, object]:
    """Run one train or evaluation epoch and return loss plus subset metrics."""
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: pip install -e ".[train]"'
        ) from exc

    training = optimizer is not None
    model.train(training)

    all_metrics = BinaryMetricAccumulator(threshold)
    transition_metrics = BinaryMetricAccumulator(threshold)
    short_metrics = BinaryMetricAccumulator(threshold)
    total_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = batch["input"].to(device=device, dtype=torch.float32)
            targets = batch["target"].to(device=device, dtype=torch.float32)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = int(targets.numel())
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size

            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            target_values = targets.detach().cpu().numpy()
            transition_mask = (
                batch["near_transition"].detach().cpu().numpy().astype(bool)
            )
            short_mask = (
                batch["near_short_correction"].detach().cpu().numpy().astype(bool)
            )

            all_metrics.update(probabilities, target_values)
            transition_metrics.update(
                probabilities[transition_mask],
                target_values[transition_mask],
            )
            short_metrics.update(
                probabilities[short_mask],
                target_values[short_mask],
            )

    if total_samples == 0:
        raise ValueError("data loader produced no samples")

    return {
        "loss": total_loss / total_samples,
        "all": all_metrics.result().to_dict(),
        "transition": transition_metrics.result().to_dict(),
        "short_correction": short_metrics.result().to_dict(),
    }
