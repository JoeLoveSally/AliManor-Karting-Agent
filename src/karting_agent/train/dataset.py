"""Build temporal training sample manifests from labeled gameplay videos."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from karting_agent.train.labels.touch_marker import (
    ActionEvent,
    ActionSegment,
    ActionTimeline,
    action_at_timestamp,
    build_action_timeline,
)


@dataclass(frozen=True)
class DatasetConfig:
    sample_fps: float = 30.0
    frame_stack: int = 3
    history_ms: float = 100.0
    frame_interval_ms: float = 50.0
    prediction_horizon_ms: float = 100.0
    transition_window_ms: float = 200.0
    short_correction_min_ms: float = 100.0
    short_correction_max_ms: float = 300.0
    clean_isolated_one_frame_glitches: bool = True

    def validate(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be > 0")
        if self.frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        if self.frame_interval_ms < 0 or self.prediction_horizon_ms < 0:
            raise ValueError("time intervals must be >= 0")
        expected_history = (self.frame_stack - 1) * self.frame_interval_ms
        if abs(expected_history - self.history_ms) > 1e-6:
            raise ValueError(
                "history_ms must equal (frame_stack - 1) * frame_interval_ms"
            )
        if not (0 <= self.short_correction_min_ms <= self.short_correction_max_ms):
            raise ValueError("invalid short-correction duration range")

    @property
    def frame_offsets_ms(self) -> tuple[float, ...]:
        return tuple(
            -(self.frame_stack - 1 - index) * self.frame_interval_ms
            for index in range(self.frame_stack)
        )


@dataclass(frozen=True)
class DatasetSample:
    video: str
    input_frame_indices: tuple[int, ...]
    input_timestamps_ms: tuple[float, ...]
    target_frame_index: int
    target_timestamp_ms: float
    target_pressed: bool
    transition_distance_ms: float | None
    near_transition: bool
    near_short_correction: bool


def _nearest_frame_index(timestamp_ms: float, fps: float, frame_count: int) -> int:
    return min(frame_count - 1, max(0, round(timestamp_ms * fps / 1000.0)))


def _nearest_transition_distance(
    timestamp_ms: float, events: Sequence[ActionEvent]
) -> float | None:
    transitions = [event.timestamp_ms for event in events[1:]]
    if not transitions:
        return None

    index = bisect_left(transitions, timestamp_ms)
    candidates = []
    if index < len(transitions):
        candidates.append(abs(transitions[index] - timestamp_ms))
    if index > 0:
        candidates.append(abs(transitions[index - 1] - timestamp_ms))
    return min(candidates)


def _short_corrections(
    segments: Sequence[ActionSegment], config: DatasetConfig
) -> tuple[ActionSegment, ...]:
    return tuple(
        segment
        for segment in segments
        if not segment.pressed
        and config.short_correction_min_ms <= segment.duration_ms <= config.short_correction_max_ms
    )


def _near_short_correction(
    timestamp_ms: float,
    corrections: Sequence[ActionSegment],
    window_ms: float,
) -> bool:
    return any(
        segment.start_ms - window_ms <= timestamp_ms <= segment.end_ms + window_ms
        for segment in corrections
    )


def build_samples(
    timeline: ActionTimeline,
    video_name: str,
    config: DatasetConfig,
) -> list[DatasetSample]:
    config.validate()
    start_ms = config.history_ms
    end_ms = timeline.duration_ms - config.prediction_horizon_ms
    if end_ms < start_ms:
        return []

    step_ms = 1000.0 / config.sample_fps
    corrections = _short_corrections(timeline.segments, config)
    samples: list[DatasetSample] = []
    sample_index = 0

    while True:
        current_ms = start_ms + sample_index * step_ms
        if current_ms > end_ms + 1e-6:
            break

        input_timestamps = tuple(
            current_ms + offset for offset in config.frame_offsets_ms
        )
        input_indices = tuple(
            _nearest_frame_index(timestamp_ms, timeline.fps, timeline.frame_count)
            for timestamp_ms in input_timestamps
        )
        target_ms = current_ms + config.prediction_horizon_ms
        distance = _nearest_transition_distance(target_ms, timeline.events)

        samples.append(
            DatasetSample(
                video=video_name,
                input_frame_indices=input_indices,
                input_timestamps_ms=input_timestamps,
                target_frame_index=_nearest_frame_index(
                    target_ms, timeline.fps, timeline.frame_count
                ),
                target_timestamp_ms=target_ms,
                target_pressed=action_at_timestamp(timeline.events, target_ms),
                transition_distance_ms=distance,
                near_transition=(
                    distance is not None and distance <= config.transition_window_ms
                ),
                near_short_correction=_near_short_correction(
                    target_ms, corrections, config.transition_window_ms
                ),
            )
        )
        sample_index += 1

    return samples


def build_dataset(
    video_paths: Iterable[Path],
    output_dir: Path,
    config: DatasetConfig,
    *,
    project_root: Path | None = None,
    progress: Callable[[int, int, Path], None] | None = None,
) -> dict[str, object]:
    config.validate()
    videos = sorted(Path(path).resolve() for path in video_paths)

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for stale_label in labels_dir.glob("*.json"):
        stale_label.unlink()

    samples_path = output_dir / "samples.jsonl"
    root = project_root.resolve() if project_root else None
    video_summaries: list[dict[str, object]] = []
    total_samples = 0
    total_cleaned_frames = 0

    with samples_path.open("w", encoding="utf-8") as sample_file:
        for index, video_path in enumerate(videos, start=1):
            if progress:
                progress(index, len(videos), video_path)

            timeline = build_action_timeline(
                video_path,
                clean_one_frame=config.clean_isolated_one_frame_glitches,
            )
            if root:
                try:
                    video_name = str(video_path.relative_to(root))
                except ValueError:
                    video_name = str(video_path)
            else:
                video_name = str(video_path)

            samples = build_samples(timeline, video_name, config)
            for sample in samples:
                sample_file.write(
                    json.dumps(asdict(sample), separators=(",", ":")) + "\n"
                )

            label_payload = {
                "video": video_name,
                "fps": timeline.fps,
                "frame_count": timeline.frame_count,
                "duration_ms": timeline.duration_ms,
                "cleaned_frames": timeline.cleaned_frames,
                "events": [asdict(event) for event in timeline.events],
            }
            (labels_dir / f"{video_path.stem}.json").write_text(
                json.dumps(label_payload, indent=2), encoding="utf-8"
            )

            short_corrections = _short_corrections(timeline.segments, config)
            video_summaries.append(
                {
                    "video": video_name,
                    "fps": timeline.fps,
                    "frames": timeline.frame_count,
                    "duration_ms": timeline.duration_ms,
                    "events": len(timeline.events),
                    "cleaned_frames": timeline.cleaned_frames,
                    "short_corrections": len(short_corrections),
                    "samples": len(samples),
                    "near_transition_samples": sum(
                        sample.near_transition for sample in samples
                    ),
                    "near_short_correction_samples": sum(
                        sample.near_short_correction for sample in samples
                    ),
                }
            )
            total_samples += len(samples)
            total_cleaned_frames += timeline.cleaned_frames

    manifest = {
        "config": asdict(config),
        "videos": video_summaries,
        "summary": {
            "videos": len(video_summaries),
            "samples": total_samples,
            "cleaned_frames": total_cleaned_frames,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
