"""State-conditioned KEEP/SWITCH datasets for v3 and sequential v4-A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from karting_agent.train.dataset import DatasetSample
from karting_agent.train.video_dataset import TemporalVideoDataset
from karting_agent.vision.preprocess import PreprocessConfig


def _optional_float_tuple(raw: dict[str, object], key: str) -> tuple[float, ...]:
    values = raw.get(key, ())
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{key} must be a sequence")
    return tuple(float(item) for item in values)


def _optional_int_tuple(raw: dict[str, object], key: str) -> tuple[int, ...]:
    values = raw.get(key, ())
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{key} must be a sequence")
    return tuple(int(item) for item in values)


def _optional_bool_tuple(raw: dict[str, object], key: str) -> tuple[bool, ...]:
    values = raw.get(key, ())
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{key} must be a sequence")
    return tuple(bool(item) for item in values)


def load_v3_samples(path: Path) -> list[DatasetSample]:
    """Load a v3-compatible manifest and require action at observation time."""
    samples: list[DatasetSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("sample must be a mapping")
                if "current_pressed" not in raw:
                    raise KeyError("current_pressed")
                distance = raw["transition_distance_ms"]
                current_pressed = raw["current_pressed"]
                if not isinstance(current_pressed, bool):
                    raise TypeError("current_pressed must be bool")
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
                        current_pressed=current_pressed,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid v3 dataset sample at {path}:{line_number}"
                ) from exc
    return samples


class StateConditionedVideoDataset:
    """Pair visual history with an explicit physical action state.

    When ``counterfactual_states`` is true, every visual sample is emitted twice:
    once conditioned on RELEASE and once on PRESS. The primary target is whether
    that supplied state must be flipped to match each future-action horizon.
    """

    def __init__(
        self,
        samples: Sequence[DatasetSample],
        *,
        project_root: Path,
        preprocess_config: PreprocessConfig = PreprocessConfig(),
        cache_root: Path | None = None,
        require_cache: bool = False,
        counterfactual_states: bool = False,
    ) -> None:
        self.samples = list(samples)
        if any(sample.current_pressed is None for sample in self.samples):
            raise ValueError("v3 samples must contain current_pressed")
        self.counterfactual_states = counterfactual_states
        self.base = TemporalVideoDataset(
            self.samples,
            project_root=project_root,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=require_cache,
        )

    def __len__(self) -> int:
        multiplier = 2 if self.counterfactual_states else 1
        return len(self.samples) * multiplier

    def _resolve_index(self, index: int) -> tuple[int, bool]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        if self.counterfactual_states:
            sample_index = index // 2
            conditioned_pressed = bool(index % 2)
        else:
            sample_index = index
            recorded = self.samples[sample_index].current_pressed
            assert recorded is not None
            conditioned_pressed = bool(recorded)
        return sample_index, conditioned_pressed

    @staticmethod
    def _horizon_mask(values: tuple[bool, ...], fallback: bool, width: int) -> np.ndarray:
        if values:
            if len(values) != width:
                raise ValueError("per-horizon metadata width mismatch")
            return np.asarray(values, dtype=np.bool_)
        return np.full(width, fallback, dtype=np.bool_)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_index, conditioned_pressed = self._resolve_index(index)
        sample = self.samples[sample_index]
        item = self.base[sample_index]

        future_action = np.asarray(item["target"], dtype=np.float32).reshape(-1)
        if future_action.size < 1:
            raise ValueError("future-action target must not be empty")
        switch_target = np.not_equal(
            future_action >= 0.5,
            conditioned_pressed,
        ).astype(np.float32)
        width = int(future_action.size)

        return {
            "input": item["input"],
            "current_pressed": np.int64(1 if conditioned_pressed else 0),
            "switch_target": switch_target,
            "future_action_target": future_action,
            "video": sample.video,
            "target_timestamp_ms": np.float32(sample.target_timestamp_ms),
            "near_transition_by_horizon": self._horizon_mask(
                sample.near_transition_by_horizon,
                sample.near_transition,
                width,
            ),
            "near_short_correction_by_horizon": self._horizon_mask(
                sample.near_short_correction_by_horizon,
                sample.near_short_correction,
                width,
            ),
        }

    def close(self) -> None:
        self.base.close()

    def __getstate__(self) -> dict[str, object]:
        return self.__dict__.copy()

    def __del__(self) -> None:
        self.close()


class SequentialStateConditionedVideoDataset:
    """Expose the same v3 targets while keeping the RGB time axis explicit.

    The existing frame cache remains reusable: the base dataset returns the
    normalized temporal input as ``(3*T, H, W)`` and this adapter only reshapes
    it to ``(T, 3, H, W)``. No spatial preprocessing or labels are changed.
    """

    def __init__(
        self,
        samples: Sequence[DatasetSample],
        *,
        frame_stack: int,
        project_root: Path,
        preprocess_config: PreprocessConfig = PreprocessConfig(),
        cache_root: Path | None = None,
        require_cache: bool = False,
        counterfactual_states: bool = False,
    ) -> None:
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        self.frame_stack = frame_stack
        self.base = StateConditionedVideoDataset(
            samples,
            project_root=project_root,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=require_cache,
            counterfactual_states=counterfactual_states,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.base[index]
        stacked = np.asarray(item["input"], dtype=np.float32)
        if stacked.ndim != 3:
            raise ValueError(
                f"stacked temporal input must be CxHxW, got {stacked.shape}"
            )
        channels, height, width = stacked.shape
        expected_channels = self.frame_stack * 3
        if channels != expected_channels:
            raise ValueError(
                f"expected {expected_channels} channels for {self.frame_stack} frames, "
                f"got {channels}"
            )
        result = dict(item)
        result["input"] = stacked.reshape(self.frame_stack, 3, height, width)
        return result

    def close(self) -> None:
        self.base.close()

    def __getstate__(self) -> dict[str, object]:
        return self.__dict__.copy()

    def __del__(self) -> None:
        self.close()
