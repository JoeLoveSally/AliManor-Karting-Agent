#!/usr/bin/env python3
"""Inspect touch-action timing in recorded karting videos."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.train.labels.touch_marker import detect_touch_marker


@dataclass(frozen=True)
class ActionSegment:
    pressed: bool
    start_frame: int
    end_frame: int
    start_ms: float
    end_ms: float
    duration_ms: float
    complete: bool


@dataclass(frozen=True)
class VideoResult:
    path: str
    fps: float
    frames: int
    duration_ms: float
    pressed_ratio: float
    segments: list[ActionSegment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect touch-marker PRESS/RELEASE episodes at native video FPS and "
            "report high-frequency action durations."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[ROOT / "data" / "raw"],
        help="MP4 file(s) or directories. Defaults to data/raw.",
    )
    parser.add_argument(
        "--short-ms",
        type=float,
        default=300.0,
        help="Threshold used to list short action episodes (default: 300 ms).",
    )
    parser.add_argument(
        "--list-short",
        action="store_true",
        help="Print every complete action segment at or below --short-ms.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for the full per-video segment report as JSON.",
    )
    return parser.parse_args()


def collect_videos(paths: Iterable[Path]) -> list[Path]:
    videos: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            videos.extend(sorted(path.glob("*.mp4")))
        elif path.is_file() and path.suffix.lower() == ".mp4":
            videos.append(path)
        else:
            raise FileNotFoundError(f"not an MP4 file or directory: {path}")

    # Preserve deterministic order while removing duplicate paths.
    unique: dict[Path, None] = {}
    for video in videos:
        unique[video.resolve()] = None
    result = list(unique)
    if not result:
        raise FileNotFoundError("no MP4 videos found")
    return result


def states_to_segments(states: list[bool], fps: float) -> list[ActionSegment]:
    if not states:
        return []

    segments: list[ActionSegment] = []
    start = 0
    current = states[0]
    frame_ms = 1000.0 / fps

    for index in range(1, len(states)):
        if states[index] == current:
            continue
        segments.append(
            ActionSegment(
                pressed=current,
                start_frame=start,
                end_frame=index,
                start_ms=start * frame_ms,
                end_ms=index * frame_ms,
                duration_ms=(index - start) * frame_ms,
                complete=start > 0 and index < len(states),
            )
        )
        start = index
        current = states[index]

    segments.append(
        ActionSegment(
            pressed=current,
            start_frame=start,
            end_frame=len(states),
            start_ms=start * frame_ms,
            end_ms=len(states) * frame_ms,
            duration_ms=(len(states) - start) * frame_ms,
            complete=False,
        )
    )
    return segments


def inspect_video(path: Path) -> VideoResult:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid FPS for video: {path}")

    states: list[bool] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            states.append(detect_touch_marker(frame) is not None)
    finally:
        capture.release()

    if not states:
        raise RuntimeError(f"video contains no readable frames: {path}")

    segments = states_to_segments(states, fps)
    return VideoResult(
        path=str(path),
        fps=fps,
        frames=len(states),
        duration_ms=len(states) * 1000.0 / fps,
        pressed_ratio=sum(states) / len(states),
        segments=segments,
    )


def duration_bins(values: list[float]) -> dict[str, int]:
    bins = {"<100": 0, "100-200": 0, "200-300": 0, ">=300": 0}
    for value in values:
        if value < 100:
            bins["<100"] += 1
        elif value < 200:
            bins["100-200"] += 1
        elif value < 300:
            bins["200-300"] += 1
        else:
            bins[">=300"] += 1
    return bins


def summarize_durations(values: list[float]) -> str:
    if not values:
        return "count=0"
    array = np.asarray(values, dtype=np.float64)
    bins = duration_bins(values)
    return (
        f"count={len(values)}, min={array.min():.0f} ms, "
        f"p50={np.percentile(array, 50):.0f} ms, "
        f"p95={np.percentile(array, 95):.0f} ms, "
        f"bins(<100/100-200/200-300/>=300)="
        f"{bins['<100']}/{bins['100-200']}/{bins['200-300']}/{bins['>=300']}"
    )


def complete_durations(result: VideoResult, pressed: bool) -> list[float]:
    return [
        segment.duration_ms
        for segment in result.segments
        if segment.complete and segment.pressed == pressed
    ]


def print_result(result: VideoResult, short_ms: float, list_short: bool) -> None:
    press = complete_durations(result, True)
    release = complete_durations(result, False)
    short_release = [value for value in release if value <= short_ms]

    print(f"\n{Path(result.path).name}")
    print(
        f"  {result.fps:.2f} FPS | {result.frames} frames | "
        f"{result.duration_ms / 1000:.2f} s | pressed={result.pressed_ratio:.1%}"
    )
    print(f"  PRESS   {summarize_durations(press)}")
    print(f"  RELEASE {summarize_durations(release)}")
    print(
        f"  P-R-P correction gaps <= {short_ms:.0f} ms: "
        f"{len(short_release)}/{len(release)}"
    )

    if list_short:
        for segment in result.segments:
            if not segment.complete or segment.duration_ms > short_ms:
                continue
            state = "PRESS" if segment.pressed else "RELEASE"
            print(
                f"    {state:<7} {segment.start_ms / 1000:8.3f}s -> "
                f"{segment.end_ms / 1000:8.3f}s  {segment.duration_ms:6.1f} ms"
            )


def print_aggregate(results: list[VideoResult], short_ms: float) -> None:
    press = [value for result in results for value in complete_durations(result, True)]
    release = [value for result in results for value in complete_durations(result, False)]
    short_release = [value for value in release if value <= short_ms]

    total_frames = sum(result.frames for result in results)
    pressed_frames = sum(round(result.frames * result.pressed_ratio) for result in results)
    total_ms = sum(result.duration_ms for result in results)

    print("\nAggregate")
    print(
        f"  videos={len(results)} | duration={total_ms / 1000:.2f} s | "
        f"pressed={pressed_frames / total_frames:.1%}"
    )
    print(f"  PRESS   {summarize_durations(press)}")
    print(f"  RELEASE {summarize_durations(release)}")
    if release:
        print(
            f"  P-R-P correction gaps <= {short_ms:.0f} ms: "
            f"{len(short_release)}/{len(release)} ({len(short_release) / len(release):.1%})"
        )


def write_json(path: Path, results: list[VideoResult], short_ms: float) -> None:
    payload = {
        "short_ms": short_ms,
        "videos": [
            {
                **{key: value for key, value in asdict(result).items() if key != "segments"},
                "segments": [asdict(segment) for segment in result.segments],
            }
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    videos = collect_videos(args.paths)

    results: list[VideoResult] = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] inspecting {video} ...", flush=True)
        result = inspect_video(video)
        results.append(result)
        print_result(result, args.short_ms, args.list_short)

    print_aggregate(results, args.short_ms)

    if args.json_out:
        write_json(args.json_out, results, args.short_ms)
        print(f"\nWrote JSON report: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
