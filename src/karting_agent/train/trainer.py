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
            raise ValueError("short_correction_weight must be >= transition_weight")


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
                    raise ValueError(
                        f"{left_name}/{right_name} split overlap: "
                        + ", ".join(sorted(overlap))
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


def _optional_float_tuple(raw: dict, key: str) -> tuple[float, ...]:
    return tuple(float(item) for item in raw.get(key, ()))


def _optional_int_tuple(raw: dict, key: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.get(key, ()))


def _optional_bool_tuple(raw: dict, key: str) -> tuple[bool, ...]:
    return tuple(bool(item) for item in raw.get(key, ()))


def load_samples(path: Path) -> list[DatasetSample]:
    """Load old single-horizon or new multi-horizon JSONL samples."""
    samples: list[DatasetSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                distance = raw["transition_distance_ms"]
                samples.append(
                    DatasetSample(
                        video=str(raw["video"]),
                        input_frame_indices=tuple(
                            int(value) for value in raw["input_frame_indices"]
                        ),
                        input_timestamps_ms=tuple(
                            float(value) for value in raw["input_timestamps_ms"]
                        ),
                        target_frame_index=int(raw["target_frame_index"]),
                        target_timestamp_ms=float(raw["target_timestamp_ms"]),
                        target_pressed=bool(raw["target_pressed"]),
                        transition_distance_ms=(
                            None if distance is None else float(distance)
                        ),
                        near_transition=bool(raw["near_transition"]),
                        near_short_correction=bool(raw["near_short_correction"]),
                        target_frame_indices=_optional_int_tuple(
                            raw, "target_frame_indices"
                        ),
                        target_timestamps_ms=_optional_float_tuple(
                            raw, "target_timestamps_ms"
                        ),
                        target_pressed_by_horizon=_optional_bool_tuple(
                            raw, "target_pressed_by_horizon"
                        ),
                        transition_distance_ms_by_horizon=tuple(
                            None if value is None else float(value)
                            for value in raw.get(
                                "transition_distance_ms_by_horizon", ()
                            )
                        ),
                        near_transition_by_horizon=_optional_bool_tuple(
                            raw, "near_transition_by_horizon"
                        ),
                        near_short_correction_by_horizon=_optional_bool_tuple(
                            raw, "near_short_correction_by_horizon"
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
    allowed = set(videos)
    return [sample for sample in samples if sample.video in allowed]


def partition_samples(
    samples: Sequence[DatasetSample],
    split: VideoSplit,
) -> dict[str, list[DatasetSample]]:
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


def sample_weight(sample: DatasetSample, config: SamplingConfig) -> float:
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
    config.validate()
    return [sample_weight(sample, config) for sample in samples]


def sampling_summary(
    samples: Sequence[DatasetSample],
    config: SamplingConfig,
) -> dict[str, float | int]:
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
    primary_output_index: int = 0,
    max_batches: int | None = None,
) -> dict[str, object]:
    """Run one epoch and report the selected control head plus every output."""
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: pip install -e ".[train]"'
        ) from exc

    if primary_output_index < 0:
        raise ValueError("primary_output_index must be >= 0")

    training = optimizer is not None
    model.train(training)
    all_metrics = BinaryMetricAccumulator(threshold)
    transition_metrics = BinaryMetricAccumulator(threshold)
    short_metrics = BinaryMetricAccumulator(threshold)
    output_metrics: list[BinaryMetricAccumulator] | None = None
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
            if logits.ndim == 1:
                if primary_output_index != 0:
                    raise ValueError("single-output model only supports primary index 0")
                if targets.ndim != 1:
                    targets = targets.reshape(-1)
                primary_logits = logits
                primary_targets = targets
                output_count = 1
            elif logits.ndim == 2:
                if targets.ndim != 2 or targets.shape != logits.shape:
                    raise ValueError(
                        "multi-horizon target/logit shape mismatch: "
                        f"{targets.shape} vs {logits.shape}"
                    )
                output_count = logits.shape[1]
                if primary_output_index >= output_count:
                    raise ValueError(
                        f"primary_output_index {primary_output_index} >= {output_count}"
                    )
                primary_logits = logits[:, primary_output_index]
                primary_targets = targets[:, primary_output_index]
            else:
                raise ValueError(
                    f"unsupported model output shape: {tuple(logits.shape)}"
                )

            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = int(inputs.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size

            primary_probabilities = (
                torch.sigmoid(primary_logits).detach().cpu().numpy()
            )
            primary_values = primary_targets.detach().cpu().numpy()
            transition_mask = (
                batch["near_transition"].detach().cpu().numpy().astype(bool)
            )
            short_mask = (
                batch["near_short_correction"].detach().cpu().numpy().astype(bool)
            )
            all_metrics.update(primary_probabilities, primary_values)
            transition_metrics.update(
                primary_probabilities[transition_mask],
                primary_values[transition_mask],
            )
            short_metrics.update(
                primary_probabilities[short_mask],
                primary_values[short_mask],
            )

            if output_metrics is None:
                output_metrics = [
                    BinaryMetricAccumulator(threshold) for _ in range(output_count)
                ]
            if logits.ndim == 1:
                output_metrics[0].update(primary_probabilities, primary_values)
            else:
                probabilities = torch.sigmoid(logits).detach().cpu().numpy()
                values = targets.detach().cpu().numpy()
                for output_index, metric in enumerate(output_metrics):
                    metric.update(
                        probabilities[:, output_index],
                        values[:, output_index],
                    )

    if total_samples == 0:
        raise ValueError("data loader produced no samples")
    assert output_metrics is not None
    return {
        "loss": total_loss / total_samples,
        "primary_output_index": primary_output_index,
        "all": all_metrics.result().to_dict(),
        "transition": transition_metrics.result().to_dict(),
        "short_correction": short_metrics.result().to_dict(),
        "outputs": [metric.result().to_dict() for metric in output_metrics],
    }
