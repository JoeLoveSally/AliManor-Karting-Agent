"""Video-backed temporal dataset with optional memory-mapped frame cache."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from karting_agent.train.dataset import DatasetSample
from karting_agent.train.frame_cache import cache_paths, read_cache_metadata
from karting_agent.vision.preprocess import (
    PreprocessConfig,
    stack_frames,
    stack_prepared_frames,
)


class TemporalVideoDataset:
    def __init__(
        self,
        samples: Sequence[DatasetSample],
        *,
        project_root: Path,
        preprocess_config: PreprocessConfig = PreprocessConfig(),
        cache_root: Path | None = None,
        require_cache: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.project_root = Path(project_root).resolve()
        self.preprocess_config = preprocess_config
        self.preprocess_config.validate()
        self.cache_root = None if cache_root is None else Path(cache_root).resolve()
        self.require_cache = require_cache
        if self.require_cache and self.cache_root is None:
            raise ValueError("require_cache=True requires cache_root")

        self._captures: dict[Path, cv2.VideoCapture] = {}
        self._cache_arrays: dict[str, np.ndarray] = {}
        self._cache_rows: dict[str, dict[int, int]] = {}
        self._cache_missing: set[str] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def _video_path(self, video: str) -> Path:
        path = Path(video)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _capture(self, path: Path) -> cv2.VideoCapture:
        capture = self._captures.get(path)
        if capture is not None and capture.isOpened():
            return capture
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"cannot open video: {path}")
        self._captures[path] = capture
        return capture

    def _load_cache(self, video: str) -> tuple[np.ndarray, dict[int, int]] | None:
        if self.cache_root is None or video in self._cache_missing:
            return None

        cached = self._cache_arrays.get(video)
        rows = self._cache_rows.get(video)
        if cached is not None and rows is not None:
            return cached, rows

        frames_path, index_path = cache_paths(self.cache_root, video)
        frames_exists = frames_path.is_file()
        index_exists = index_path.is_file()
        if not frames_exists and not index_exists:
            if self.require_cache:
                raise RuntimeError(
                    f"frame cache missing for {video}; run scripts/build_frame_cache.py"
                )
            self._cache_missing.add(video)
            return None
        if frames_exists != index_exists:
            raise RuntimeError(f"incomplete frame cache for {video}: {frames_path.parent}")

        try:
            metadata = read_cache_metadata(index_path)
            metadata.validate_for(video, self.preprocess_config)
            cached = np.load(frames_path, mmap_mode="r")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"invalid frame cache for {video}") from exc

        expected_shape = (
            len(metadata.frame_indices),
            self.preprocess_config.input_height,
            self.preprocess_config.input_width,
            3,
        )
        if cached.dtype != np.uint8 or cached.shape != expected_shape:
            raise RuntimeError(
                f"frame cache array mismatch for {video}: "
                f"shape={cached.shape}, dtype={cached.dtype}"
            )

        rows = {frame_index: row for row, frame_index in enumerate(metadata.frame_indices)}
        self._cache_arrays[video] = cached
        self._cache_rows[video] = rows
        return cached, rows

    def _read_cached_frames(
        self,
        video: str,
        frame_indices: Sequence[int],
    ) -> list[np.ndarray] | None:
        loaded = self._load_cache(video)
        if loaded is None:
            return None

        cached, rows = loaded
        missing = [int(index) for index in frame_indices if int(index) not in rows]
        if missing:
            raise RuntimeError(
                f"frame cache for {video} is stale; missing frames {missing}. "
                "Rebuild it with scripts/build_frame_cache.py --force"
            )
        return [cached[rows[int(index)]] for index in frame_indices]

    def _read_video_frames(
        self,
        video: str,
        frame_indices: Sequence[int],
    ) -> list[np.ndarray]:
        if not frame_indices:
            raise ValueError("frame_indices must not be empty")
        if any(index < 0 for index in frame_indices):
            raise ValueError("frame indices must be non-negative")

        path = self._video_path(video)
        capture = self._capture(path)
        requested = tuple(int(index) for index in frame_indices)
        start = min(requested)
        end = max(requested)
        wanted = set(requested)

        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        decoded: dict[int, np.ndarray] = {}
        for frame_index in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to decode frame {frame_index} from {path}")
            if frame_index in wanted:
                decoded[frame_index] = frame

        missing = [index for index in requested if index not in decoded]
        if missing:
            raise RuntimeError(f"missing frames {missing} while reading {path}")
        return [decoded[index] for index in requested]

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        cached_frames = self._read_cached_frames(sample.video, sample.input_frame_indices)
        if cached_frames is not None:
            model_input = stack_prepared_frames(cached_frames, self.preprocess_config)
        else:
            frames = self._read_video_frames(sample.video, sample.input_frame_indices)
            model_input = stack_frames(frames, self.preprocess_config)

        return {
            "input": model_input,
            "target": np.float32(1.0 if sample.target_pressed else 0.0),
            "video": sample.video,
            "target_timestamp_ms": np.float32(sample.target_timestamp_ms),
            "near_transition": sample.near_transition,
            "near_short_correction": sample.near_short_correction,
        }

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
        self._cache_arrays.clear()
        self._cache_rows.clear()
        self._cache_missing.clear()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_captures"] = {}
        state["_cache_arrays"] = {}
        state["_cache_rows"] = {}
        state["_cache_missing"] = set()
        return state

    def __del__(self) -> None:
        self.close()
