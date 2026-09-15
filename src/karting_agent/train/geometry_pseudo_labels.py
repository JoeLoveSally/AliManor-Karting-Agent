"""Offline pseudo-label helpers for structured visual supervision.

The v4 geometry teacher is intentionally an offline diagnostic/labeling tool. It
must never become part of the deployed control path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


HsvTriplet = tuple[int, int, int]
HsvRange = tuple[HsvTriplet, HsvTriplet]


@dataclass(frozen=True)
class RoadMaskConfig:
    """Theme-aware HSV road-mask configuration.

    ``hsv_lower``/``hsv_upper`` are kept for backward compatibility with the
    first blue-track POC. When ``hsv_ranges`` is non-empty, all configured
    ranges are OR-ed before morphology and connected-component filtering.
    """

    hsv_lower: HsvTriplet = (103, 90, 50)
    hsv_upper: HsvTriplet = (130, 255, 220)
    hsv_ranges: tuple[HsvRange, ...] = ()
    close_kernel: int = 9
    open_kernel: int = 3
    min_component_area_fraction: float = 0.01
    max_components: int = 4

    def ranges(self) -> tuple[HsvRange, ...]:
        return self.hsv_ranges or ((self.hsv_lower, self.hsv_upper),)

    def validate(self) -> None:
        if self.close_kernel < 1 or self.open_kernel < 1:
            raise ValueError("morphology kernels must be >= 1")
        if not 0.0 <= self.min_component_area_fraction <= 1.0:
            raise ValueError("min_component_area_fraction must be in [0, 1]")
        if self.max_components < 1:
            raise ValueError("max_components must be >= 1")
        ranges = self.ranges()
        if not ranges:
            raise ValueError("at least one HSV range is required")
        for lower, upper in ranges:
            if len(lower) != 3 or len(upper) != 3:
                raise ValueError("HSV lower/upper bounds must have 3 values")
            if any(value < 0 for value in lower):
                raise ValueError("HSV lower values must be >= 0")
            if lower[0] > 179 or upper[0] > 179:
                raise ValueError("OpenCV HSV hue must be <= 179")
            if any(value > 255 for value in lower[1:] + upper[1:]):
                raise ValueError("HSV saturation/value must be <= 255")
            if any(low > high for low, high in zip(lower, upper)):
                raise ValueError("HSV lower bound must not exceed upper bound")


@dataclass(frozen=True)
class RectilinearGeometryConfig:
    """Line/corner diagnostics for piecewise-straight kart tracks."""

    canny_low: int = 40
    canny_high: int = 120
    hough_threshold: int = 24
    min_line_length_fraction: float = 0.10
    max_line_gap_fraction: float = 0.025
    axis_tolerance_deg: float = 10.0
    min_axis_separation_deg: float = 25.0
    corner_extension_fraction: float = 0.08
    anchor_x_norm: float = 0.5
    anchor_y_norm: float = 0.58

    def validate(self) -> None:
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
        if self.hough_threshold < 1:
            raise ValueError("hough_threshold must be >= 1")
        if not 0.0 < self.min_line_length_fraction <= 1.0:
            raise ValueError("min_line_length_fraction must be in (0, 1]")
        if not 0.0 <= self.max_line_gap_fraction <= 1.0:
            raise ValueError("max_line_gap_fraction must be in [0, 1]")
        if not 0.0 < self.axis_tolerance_deg < 90.0:
            raise ValueError("axis_tolerance_deg must be in (0, 90)")
        if not 0.0 < self.min_axis_separation_deg < 90.0:
            raise ValueError("min_axis_separation_deg must be in (0, 90)")
        if not 0.0 <= self.corner_extension_fraction <= 1.0:
            raise ValueError("corner_extension_fraction must be in [0, 1]")
        if not 0.0 <= self.anchor_x_norm <= 1.0:
            raise ValueError("anchor_x_norm must be in [0, 1]")
        if not 0.0 <= self.anchor_y_norm <= 1.0:
            raise ValueError("anchor_y_norm must be in [0, 1]")


@dataclass(frozen=True)
class LineSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    angle_deg: float
    length: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "angle_deg": self.angle_deg,
            "length": self.length,
        }


@dataclass(frozen=True)
class RectilinearGeometry:
    """Theme-independent geometry distilled from a road mask.

    The teacher deliberately does not claim to know *next turn direction* yet.
    That requires a reliable travel/kart-heading estimate. For now it reports
    visible straight axes and the nearest visible axis intersection relative to
    a fixed diagnostic anchor.
    """

    geometry_class: str
    primary_angle_deg: float | None
    secondary_angle_deg: float | None
    primary_support: float
    secondary_support: float
    straight_confidence: float
    corner_score: float
    corner_visible: bool
    corner_x_norm: float | None
    corner_y_norm: float | None
    corner_distance_norm: float | None
    segments: tuple[LineSegment, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "geometry_class": self.geometry_class,
            "primary_angle_deg": self.primary_angle_deg,
            "secondary_angle_deg": self.secondary_angle_deg,
            "primary_support": self.primary_support,
            "secondary_support": self.secondary_support,
            "straight_confidence": self.straight_confidence,
            "corner_score": self.corner_score,
            "corner_visible": self.corner_visible,
            "corner_x_norm": self.corner_x_norm,
            "corner_y_norm": self.corner_y_norm,
            "corner_distance_norm": self.corner_distance_norm,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def _odd_kernel(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def extract_road_mask(frame_bgr: np.ndarray, config: RoadMaskConfig) -> np.ndarray:
    """Return a binary uint8 road mask from one or more theme HSV ranges."""

    config.validate()
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape HxWx3")

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for lower, upper in config.ranges():
        selected = cv2.inRange(
            hsv,
            np.asarray(lower, dtype=np.uint8),
            np.asarray(upper, dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, selected)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel(config.close_kernel), _odd_kernel(config.close_kernel)),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_odd_kernel(config.open_kernel), _odd_kernel(config.open_kernel)),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if component_count <= 1:
        return np.zeros(mask.shape, dtype=np.uint8)

    image_area = mask.shape[0] * mask.shape[1]
    min_area = image_area * config.min_component_area_fraction
    candidate_labels = [
        label
        for label in range(1, component_count)
        if stats[label, cv2.CC_STAT_AREA] >= min_area
    ]
    if not candidate_labels:
        return np.zeros(mask.shape, dtype=np.uint8)

    candidate_labels.sort(
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True
    )
    kept = candidate_labels[: config.max_components]
    return np.where(np.isin(labels, kept), 255, 0).astype(np.uint8)


def row_centerline(
    mask: np.ndarray,
    *,
    row_step: int = 16,
    min_run_width: int = 12,
) -> list[tuple[int, float]]:
    """Legacy diagnostic center points retained for comparison only."""

    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    if row_step < 1 or min_run_width < 1:
        raise ValueError("row_step and min_run_width must be >= 1")

    points: list[tuple[int, float]] = []
    width = mask.shape[1]
    for y in range(0, mask.shape[0], row_step):
        xs = np.flatnonzero(mask[y] > 0)
        if xs.size == 0:
            continue

        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [xs.size - 1]))
        runs = [(int(xs[start]), int(xs[end])) for start, end in zip(starts, ends)]
        left, right = max(runs, key=lambda run: run[1] - run[0] + 1)
        if right - left + 1 < min_run_width:
            continue
        center_x = (left + right) * 0.5
        points.append((y, center_x / max(1, width - 1)))
    return points


def _angle_distance_deg(left: float, right: float) -> float:
    raw = abs(left - right) % 180.0
    return min(raw, 180.0 - raw)


def _weighted_axis_angle(
    segments: list[LineSegment], seed_angle: float, tolerance_deg: float
) -> tuple[float, float, list[LineSegment]]:
    members = [
        segment
        for segment in segments
        if _angle_distance_deg(segment.angle_deg, seed_angle) <= tolerance_deg
    ]
    if not members:
        return seed_angle, 0.0, []

    # Angles are axial (0° == 180°), so average after doubling the angle.
    sin_sum = sum(
        math.sin(math.radians(2.0 * segment.angle_deg)) * segment.length
        for segment in members
    )
    cos_sum = sum(
        math.cos(math.radians(2.0 * segment.angle_deg)) * segment.length
        for segment in members
    )
    angle = (0.5 * math.degrees(math.atan2(sin_sum, cos_sum))) % 180.0
    support = sum(segment.length for segment in members)
    return angle, support, members


def detect_line_segments(
    mask: np.ndarray, config: RectilinearGeometryConfig
) -> list[LineSegment]:
    """Detect long road-edge segments from a binary road mask."""

    config.validate()
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")
    height, width = mask.shape
    diag = math.hypot(width, height)
    edges = cv2.Canny(mask, config.canny_low, config.canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=config.hough_threshold,
        minLineLength=max(8, int(round(diag * config.min_line_length_fraction))),
        maxLineGap=max(1, int(round(diag * config.max_line_gap_fraction))),
    )
    if lines is None:
        return []

    # OpenCV commonly returns (N, 1, 4), but some builds/bindings return
    # (N, 4). Normalize both layouts before iterating.
    normalized_lines = np.asarray(lines).reshape(-1, 4)
    segments: list[LineSegment] = []
    for raw in normalized_lines:
        x1, y1, x2, y2 = (int(value) for value in raw)
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 0:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        segments.append(LineSegment(x1, y1, x2, y2, angle, length))
    return segments


def _point_near_segment_extent(
    x: float,
    y: float,
    segment: LineSegment,
    extension: float,
) -> bool:
    return (
        min(segment.x1, segment.x2) - extension
        <= x
        <= max(segment.x1, segment.x2) + extension
        and min(segment.y1, segment.y2) - extension
        <= y
        <= max(segment.y1, segment.y2) + extension
    )


def _line_intersection(
    first: LineSegment, second: LineSegment
) -> tuple[float, float] | None:
    x1, y1, x2, y2 = first.x1, first.y1, first.x2, first.y2
    x3, y3, x4, y4 = second.x1, second.y1, second.x2, second.y2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None
    cross1 = x1 * y2 - y1 * x2
    cross2 = x3 * y4 - y3 * x4
    x = (cross1 * (x3 - x4) - (x1 - x2) * cross2) / denominator
    y = (cross1 * (y3 - y4) - (y1 - y2) * cross2) / denominator
    return float(x), float(y)


def estimate_rectilinear_geometry(
    mask: np.ndarray,
    config: RectilinearGeometryConfig | None = None,
) -> RectilinearGeometry:
    """Estimate visible piecewise-straight axes and nearest visible corner.

    This is a feasibility diagnostic, not a final label definition. A reliable
    *next corner left/right* label is intentionally deferred until kart/travel
    direction can be estimated.
    """

    config = config or RectilinearGeometryConfig()
    config.validate()
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")

    segments = detect_line_segments(mask, config)
    if not segments:
        return RectilinearGeometry(
            geometry_class="unknown",
            primary_angle_deg=None,
            secondary_angle_deg=None,
            primary_support=0.0,
            secondary_support=0.0,
            straight_confidence=0.0,
            corner_score=0.0,
            corner_visible=False,
            corner_x_norm=None,
            corner_y_norm=None,
            corner_distance_norm=None,
            segments=(),
        )

    seed_candidates = sorted(segments, key=lambda segment: segment.length, reverse=True)
    best_primary: tuple[float, float, list[LineSegment]] | None = None
    for seed in seed_candidates:
        candidate = _weighted_axis_angle(
            segments, seed.angle_deg, config.axis_tolerance_deg
        )
        if best_primary is None or candidate[1] > best_primary[1]:
            best_primary = candidate
    assert best_primary is not None
    primary_angle, primary_support, primary_members = best_primary

    secondary_pool = [
        segment
        for segment in segments
        if _angle_distance_deg(segment.angle_deg, primary_angle)
        >= config.min_axis_separation_deg
    ]
    best_secondary: tuple[float, float, list[LineSegment]] | None = None
    for seed in sorted(secondary_pool, key=lambda segment: segment.length, reverse=True):
        candidate = _weighted_axis_angle(
            secondary_pool, seed.angle_deg, config.axis_tolerance_deg
        )
        if best_secondary is None or candidate[1] > best_secondary[1]:
            best_secondary = candidate

    if best_secondary is None:
        secondary_angle = None
        secondary_support = 0.0
        secondary_members: list[LineSegment] = []
    else:
        secondary_angle, secondary_support, secondary_members = best_secondary

    total_support = max(1e-6, sum(segment.length for segment in segments))
    straight_confidence = min(1.0, primary_support / total_support)
    corner_score = min(1.0, secondary_support / total_support)

    height, width = mask.shape
    diag = max(1.0, math.hypot(width, height))
    extension = diag * config.corner_extension_fraction
    anchor_x = config.anchor_x_norm * max(1, width - 1)
    anchor_y = config.anchor_y_norm * max(1, height - 1)
    intersections: list[tuple[float, float, float]] = []
    for primary in primary_members:
        for secondary in secondary_members:
            point = _line_intersection(primary, secondary)
            if point is None:
                continue
            x, y = point
            if not (-extension <= x <= width - 1 + extension):
                continue
            if not (-extension <= y <= height - 1 + extension):
                continue
            if not _point_near_segment_extent(x, y, primary, extension):
                continue
            if not _point_near_segment_extent(x, y, secondary, extension):
                continue
            distance = math.hypot(x - anchor_x, y - anchor_y) / diag
            intersections.append((distance, x, y))

    if intersections:
        corner_distance, corner_x, corner_y = min(intersections, key=lambda item: item[0])
        corner_visible = True
        corner_x_norm = float(np.clip(corner_x / max(1, width - 1), 0.0, 1.0))
        corner_y_norm = float(np.clip(corner_y / max(1, height - 1), 0.0, 1.0))
        corner_distance_norm = float(corner_distance)
        geometry_class = "corner_visible"
    else:
        corner_visible = False
        corner_x_norm = None
        corner_y_norm = None
        corner_distance_norm = None
        geometry_class = "straight" if secondary_support == 0.0 else "mixed_axes"

    return RectilinearGeometry(
        geometry_class=geometry_class,
        primary_angle_deg=float(primary_angle),
        secondary_angle_deg=(float(secondary_angle) if secondary_angle is not None else None),
        primary_support=float(primary_support),
        secondary_support=float(secondary_support),
        straight_confidence=float(straight_confidence),
        corner_score=float(corner_score),
        corner_visible=corner_visible,
        corner_x_norm=corner_x_norm,
        corner_y_norm=corner_y_norm,
        corner_distance_norm=corner_distance_norm,
        segments=tuple(segments),
    )


def road_mask_metrics(
    mask: np.ndarray,
    centerline: list[tuple[int, float]] | None = None,
    geometry: RectilinearGeometry | None = None,
) -> dict[str, float]:
    """Return cheap diagnostics for pseudo-label inspection."""

    area_fraction = float(np.mean(mask > 0))
    metrics = {"road_area_fraction": area_fraction}
    if centerline is not None:
        possible_rows = max(1, int(np.ceil(mask.shape[0] / 16.0)))
        metrics["centerline_valid_fraction"] = min(
            1.0, len(centerline) / possible_rows
        )
    if geometry is not None:
        metrics.update(
            {
                "straight_confidence": geometry.straight_confidence,
                "corner_score": geometry.corner_score,
                "corner_visible": 1.0 if geometry.corner_visible else 0.0,
                "geometry_usable": 1.0
                if geometry.primary_angle_deg is not None
                else 0.0,
            }
        )
    return metrics


def overlay_road_geometry(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    centerline: list[tuple[int, float]] | None = None,
    geometry: RectilinearGeometry | None = None,
    geometry_config: RectilinearGeometryConfig | None = None,
) -> np.ndarray:
    """Render road mask plus rectilinear diagnostics for human inspection."""

    overlay = frame_bgr.copy()
    tint = np.zeros_like(frame_bgr)
    tint[..., 1] = 255
    selected = mask > 0
    overlay[selected] = (
        0.65 * overlay[selected].astype(np.float32)
        + 0.35 * tint[selected].astype(np.float32)
    ).astype(np.uint8)

    if centerline is not None:
        width = frame_bgr.shape[1]
        for y, x_norm in centerline:
            x = int(round(x_norm * max(1, width - 1)))
            cv2.circle(overlay, (x, int(y)), 2, (0, 255, 255), -1)

    if geometry is None:
        return overlay

    primary = geometry.primary_angle_deg
    secondary = geometry.secondary_angle_deg
    config = geometry_config or RectilinearGeometryConfig()
    for segment in geometry.segments:
        if primary is not None and _angle_distance_deg(segment.angle_deg, primary) <= config.axis_tolerance_deg:
            color = (0, 255, 255)  # primary axis: yellow
        elif secondary is not None and _angle_distance_deg(segment.angle_deg, secondary) <= config.axis_tolerance_deg:
            color = (255, 0, 255)  # secondary axis: magenta
        else:
            color = (255, 180, 0)
        cv2.line(
            overlay,
            (segment.x1, segment.y1),
            (segment.x2, segment.y2),
            color,
            2,
            cv2.LINE_AA,
        )

    height, width = frame_bgr.shape[:2]
    anchor = (
        int(round(config.anchor_x_norm * max(1, width - 1))),
        int(round(config.anchor_y_norm * max(1, height - 1))),
    )
    cv2.drawMarker(
        overlay,
        anchor,
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=2,
    )
    if geometry.corner_visible:
        assert geometry.corner_x_norm is not None
        assert geometry.corner_y_norm is not None
        corner = (
            int(round(geometry.corner_x_norm * max(1, width - 1))),
            int(round(geometry.corner_y_norm * max(1, height - 1))),
        )
        cv2.circle(overlay, corner, 7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(overlay, anchor, corner, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay
