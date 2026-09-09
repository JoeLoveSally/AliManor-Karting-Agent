"""Lazy video-backed dataset for temporal model training."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from karting_agent.train.dataset import DatasetSample
from karting_agent.vision.preprocess import PreprocessConfig, stack_frames


class TemporalVideoDataset:
    """Read temporal frame stacks from source videos on demand.

    VideoCapture handles are opened lazily and cached per process. ``__getstate__``
    drops open handles when a DataLoader worker serializes the dataset.
    """

    def __init__(
        self,
        samples: Sequence[DatasetSample],
        *,
        project_root: Path,
        preprocess_config: PreprocessConfig = PreprocessConfig(),
    ) -> None:
        self.samples = list(samples)
        self.project_root = Path(project_root).resolve()
        self.preprocess_config = preprocess_config
        self.preprocess_config.validate()
        self._captures: dict[Path, cv2.VideoCapture] = {}

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

    def _read_frames(
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
                raise RuntimeError(
                    f"failed to decode frame {frame_index} from {path}"
                )
            if frame_index in wanted:
                decoded[frame_index] = frame

        missing = [index for index in requested if index not in decoded]
        if missing:
            raise RuntimeError(
                f"missing frames {missing} while reading {path}"
            )
        return [decoded[index] for index in requested]

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        frames = self._read_frames(sample.video, sample.input_frame_indices)
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

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_captures"] = {}
        return state

    def __del__(self) -> None:
        self.close()
