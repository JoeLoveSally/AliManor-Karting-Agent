"""Touch-marker detection for recorded gameplay labels."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TouchMarker:
    """Detected Android touch marker in image coordinates."""

    center_x: float
    center_y: float
    radius: float


# The recordings use a 720 px-wide phone capture where the touch indicator is
# roughly a 12-14 px radius circle near the lower-right control area. Scale the
# radius with frame width so the detector is not tied to one encoded size.
_REFERENCE_WIDTH = 720.0
_MIN_RADIUS = 8.0
_MAX_RADIUS = 18.0


def detect_touch_marker(frame: np.ndarray) -> TouchMarker | None:
    """Return the touch marker if it is visible in a gameplay frame.

    Detection intentionally has no temporal debounce or minimum-duration rule.
    Short PRESS/RELEASE episodes are meaningful training signals in this game,
    so temporal cleaning belongs to later dataset analysis, not raw detection.
    """

    if frame is None or frame.ndim != 3:
        raise ValueError("frame must be a BGR image with shape HxWxC")

    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("frame must have non-zero width and height")

    # Restrict Hough detection to the lower-right touch-control area. The
    # indicator is semi-transparent, so color thresholding alone is unreliable
    # when it crosses dark road pixels; its circular edge is much more stable.
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

    # Hough can occasionally return more than one candidate. Prefer the one
    # closest to the expected lower-right control area.
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
