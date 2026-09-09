"""Memory-mapped frame cache for temporal training samples."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import cv2
import numpy as np

from karting_agent.train.dataset import DatasetSample
from karting_agent.vision.preprocess import PreprocessConfig, prepare_frame

CACHE_VERSION = 1
FRAMES_FILENAME = "frames.npy"
INDEX_FILENAME = "index.json"


@dataclass(frozen=True)
class FrameCacheMetadata:
    version: int
    video: str
    frame_indices: tuple[int, ...]
    input_width: int
    input_height: int
    mask_touch_area: bool
    touch_roi: tuple[float, float, float, float]
    color_space: str = "RGB"
    dtype: str = "uint8"

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "FrameCacheMetadata":
        return cls(
            version=int(raw["version"]),
            video=str(raw["video"]),
            frame_indices=tuple(int(value) for value in raw["frame_indices"]),
            input_width=int(raw["input_width"]),
            input_height=int(raw["input_height"]),
            mask_touch_area=bool(raw["mask_touch_area"]),
            touch_roi=tuple(float(value) for value in raw["touch_roi"]),
            color_space=str(raw.get("color_space", "RGB")),
            dtype=str(raw.get("dtype", "uint8")),
        )

    def validate_for(self, video: str, config: PreprocessConfig) -> None:
        if self.version != CACHE_VERSION:
            raise ValueError(
                f"unsupported frame cache version {self.version}; expected {CACHE_VERSION}"
            )
        if self.video != video:
            raise ValueError(f"frame cache video mismatch: {self.video} != {video}")
        if self.input_width != config.input_width or self.input_height != config.input_height:
            raise ValueError("frame cache input size does not match preprocess config")
        if self.mask_touch_area != config.mask_touch_area:
            raise ValueError("frame cache touch-mask setting does not match preprocess config")
        if len(self.touch_roi) != 4 or any(
            abs(left - right) > 1e-9
            for left, right in zip(self.touch_roi, config.touch_roi, strict=True)
        ):
            raise ValueError("frame cache touch ROI does not match preprocess config")
        if self.color_space != "RGB" or self.dtype != "uint8":
            raise ValueError("unsupported frame cache representation")


def frame_cache_root_from_config(
    raw_config: dict[str, object],
    project_root: Path,
) -> Path | None:
    section = raw_config.get("frame_cache", {})
    if not isinstance(section, dict):
        raise ValueError("frame_cache config must be a mapping")
    if not bool(section.get("enabled", True)):
        return None

    configured = Path(str(section.get("root", "data/processed/frame_cache")))
    if not configured.is_absolute():
        configured = Path(project_root) / configured
    return configured.resolve()


def cache_dir_for_video(cache_root: Path, video: str) -> Path:
    return Path(cache_root) / Path(video).stem


def cache_paths(cache_root: Path, video: str) -> tuple[Path, Path]:
    directory = cache_dir_for_video(cache_root, video)
    return directory / FRAMES_FILENAME, directory / INDEX_FILENAME


def required_frame_indices(samples: Sequence[DatasetSample]) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, set[int]] = defaultdict(set)
    for sample in samples:
        grouped[sample.video].update(int(index) for index in sample.input_frame_indices)
    return {
        video: tuple(sorted(indices))
        for video, indices in sorted(grouped.items())
    }


def read_cache_metadata(path: Path) -> FrameCacheMetadata:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"frame cache metadata must be an object: {path}")
    return FrameCacheMetadata.from_dict(raw)


def cache_is_complete(
    cache_root: Path,
    video: str,
    frame_indices: Sequence[int],
    config: PreprocessConfig,
) -> bool:
    frames_path, index_path = cache_paths(cache_root, video)
    if not frames_path.is_file() or not index_path.is_file():
        return False

    try:
        metadata = read_cache_metadata(index_path)
        metadata.validate_for(video, config)
        expected_indices = tuple(int(index) for index in frame_indices)
        if metadata.frame_indices != expected_indices:
            return False
        frames = np.load(frames_path, mmap_mode="r")
        expected_shape = (
            len(expected_indices),
            config.input_height,
            config.input_width,
            3,
        )
        return frames.dtype == np.uint8 and frames.shape == expected_shape
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _resolve_video_path(video: str, project_root: Path) -> Path:
    path = Path(video)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def build_video_cache(
    video: str,
    frame_indices: Sequence[int],
    *,
    project_root: Path,
    cache_root: Path,
    preprocess_config: PreprocessConfig,
    force: bool = False,
) -> dict[str, object]:
    preprocess_config.validate()
    indices = tuple(sorted({int(index) for index in frame_indices}))
    if not indices:
        raise ValueError(f"no frame indices requested for {video}")
    if indices[0] < 0:
        raise ValueError("frame indices must be non-negative")

    directory = cache_dir_for_video(cache_root, video)
    directory.mkdir(parents=True, exist_ok=True)
    frames_path, index_path = cache_paths(cache_root, video)

    if not force and cache_is_complete(cache_root, video, indices, preprocess_config):
        return {
            "video": video,
            "frames": len(indices),
            "status": "cached",
            "path": str(frames_path),
        }

    source_path = _resolve_video_path(video, project_root)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open video: {source_path}")

    tmp_frames = directory / f".{FRAMES_FILENAME}.tmp"
    tmp_index = directory / f".{INDEX_FILENAME}.tmp"
    for tmp_path in (tmp_frames, tmp_index):
        if tmp_path.exists():
            tmp_path.unlink()

    mapped = np.lib.format.open_memmap(
        tmp_frames,
        mode="w+",
        dtype=np.uint8,
        shape=(
            len(indices),
            preprocess_config.input_height,
            preprocess_config.input_width,
            3,
        ),
    )

    wanted_position = 0
    max_index = indices[-1]
    try:
        for frame_index in range(max_index + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"failed to decode frame {frame_index} from {source_path}"
                )
            if frame_index != indices[wanted_position]:
                continue

            mapped[wanted_position] = prepare_frame(frame, preprocess_config)
            wanted_position += 1
            if wanted_position == len(indices):
                break
    finally:
        capture.release()

    if wanted_position != len(indices):
        raise RuntimeError(
            f"cached {wanted_position}/{len(indices)} requested frames from {source_path}"
        )

    mapped.flush()
    del mapped

    metadata = FrameCacheMetadata(
        version=CACHE_VERSION,
        video=video,
        frame_indices=indices,
        input_width=preprocess_config.input_width,
        input_height=preprocess_config.input_height,
        mask_touch_area=preprocess_config.mask_touch_area,
        touch_roi=preprocess_config.touch_roi,
    )
    tmp_index.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    os.replace(tmp_frames, frames_path)
    os.replace(tmp_index, index_path)

    return {
        "video": video,
        "frames": len(indices),
        "status": "built",
        "path": str(frames_path),
    }


def build_frame_cache(
    samples: Sequence[DatasetSample],
    *,
    project_root: Path,
    cache_root: Path,
    preprocess_config: PreprocessConfig,
    force: bool = False,
    progress: Callable[[int, int, str, int], None] | None = None,
) -> dict[str, object]:
    grouped = required_frame_indices(samples)
    results: list[dict[str, object]] = []
    total = len(grouped)

    for position, (video, indices) in enumerate(grouped.items(), start=1):
        if progress:
            progress(position, total, video, len(indices))
        results.append(
            build_video_cache(
                video,
                indices,
                project_root=project_root,
                cache_root=cache_root,
                preprocess_config=preprocess_config,
                force=force,
            )
        )

    return {
        "videos": len(results),
        "frames": sum(int(item["frames"]) for item in results),
        "built": sum(item["status"] == "built" for item in results),
        "cached": sum(item["status"] == "cached" for item in results),
        "items": results,
    }
