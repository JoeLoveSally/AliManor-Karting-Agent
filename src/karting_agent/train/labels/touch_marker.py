"""Touch-marker detection and action-timeline utilities."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class TouchMarker:
    """Detected Android touch marker in image coordinates."""

    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class ActionEvent:
    frame_index: int
    timestamp_ms: float
    pressed: bool


@dataclass(frozen=True)
class ActionSegment:
    start_frame: int
    end_frame: int
    start_ms: float
    end_ms: float
    pressed: bool

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class ActionTimeline:
    source: str
    fps: float
    frame_count: int
    duration_ms: float
    events: tuple[ActionEvent, ...]
    segments: tuple[ActionSegment, ...]
    cleaned_frames: int


_REFERENCE_WIDTH = 720.0
_MIN_RADIUS = 8.0
_MAX_RADIUS = 18.0


def detect_touch_marker(frame: np.ndarray) -> TouchMarker | None:
    """Return the touch marker if it is visible in a gameplay frame."""

    if frame is None or frame.ndim != 3:
        raise ValueError("frame must be a BGR image with shape HxWxC")

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("frame must have non-zero width and height")

    # The marker is semi-transparent, so color thresholding is unreliable when
    # it crosses dark road pixels. Restrict Hough detection to its control area.
    x0 = int(width * 0.78)
    x1 = int(width * 0.98)
    y0 = int(height * 0.82)
    y1 = int(height * 0.98)
    roi = frame[y0:y1, x0:x1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)

    scale = width / _REFERENCE_WIDTH
    min_radius = max(1, round(_MIN_RADIUS * scale))
    max_radius = max(min_radius + 1, round(_MAX_RADIUS * scale))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1, round(30 * scale)),
        param1=80,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    expected_x = width * 0.88
    expected_y = height * 0.93
    candidates = []
    for center_x, center_y, radius in circles[0]:
        global_x = float(center_x + x0)
        global_y = float(center_y + y0)
        distance_sq = (global_x - expected_x) ** 2 + (global_y - expected_y) ** 2
        candidates.append((distance_sq, global_x, global_y, float(radius)))

    _, center_x, center_y, radius = min(candidates, key=lambda item: item[0])
    return TouchMarker(center_x=center_x, center_y=center_y, radius=radius)


def extract_touch_states(video_path: Path) -> tuple[float, list[bool]]:
    """Detect the raw PRESS state of every native video frame."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"invalid FPS for video: {video_path}")

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
        raise RuntimeError(f"video contains no readable frames: {video_path}")
    return fps, states


def clean_isolated_one_frame_glitches(
    states: Sequence[bool],
) -> tuple[list[bool], int]:
    """Merge isolated one-frame inversions without smoothing real short actions."""

    cleaned = list(states)
    changed_total = 0
    while len(cleaned) >= 3:
        replacements = [
            index
            for index in range(1, len(cleaned) - 1)
            if cleaned[index - 1] == cleaned[index + 1] != cleaned[index]
        ]
        if not replacements:
            break
        for index in replacements:
            cleaned[index] = cleaned[index - 1]
        changed_total += len(replacements)
    return cleaned, changed_total


def states_to_segments(
    states: Sequence[bool], fps: float
) -> tuple[ActionSegment, ...]:
    if not states:
        return ()

    boundaries = [0]
    boundaries.extend(
        index for index in range(1, len(states)) if states[index] != states[index - 1]
    )
    boundaries.append(len(states))
    frame_ms = 1000.0 / fps

    return tuple(
        ActionSegment(
            start_frame=start,
            end_frame=end,
            start_ms=start * frame_ms,
            end_ms=end * frame_ms,
            pressed=bool(states[start]),
        )
        for start, end in zip(boundaries, boundaries[1:])
    )


def segments_to_events(
    segments: Sequence[ActionSegment],
) -> tuple[ActionEvent, ...]:
    return tuple(
        ActionEvent(
            frame_index=segment.start_frame,
            timestamp_ms=segment.start_ms,
            pressed=segment.pressed,
        )
        for segment in segments
    )


def build_action_timeline(
    video_path: Path, *, clean_one_frame: bool = True
) -> ActionTimeline:
    fps, raw_states = extract_touch_states(video_path)
    states = raw_states
    cleaned_frames = 0
    if clean_one_frame:
        states, cleaned_frames = clean_isolated_one_frame_glitches(raw_states)

    segments = states_to_segments(states, fps)
    events = segments_to_events(segments)
    return ActionTimeline(
        source=str(video_path),
        fps=fps,
        frame_count=len(states),
        duration_ms=len(states) * 1000.0 / fps,
        events=events,
        segments=segments,
        cleaned_frames=cleaned_frames,
    )


def action_at_timestamp(
    events: Sequence[ActionEvent], timestamp_ms: float
) -> bool:
    if not events:
        raise ValueError("events must not be empty")

    timestamps = [event.timestamp_ms for event in events]
    index = bisect_right(timestamps, timestamp_ms) - 1
    return events[max(index, 0)].pressed
