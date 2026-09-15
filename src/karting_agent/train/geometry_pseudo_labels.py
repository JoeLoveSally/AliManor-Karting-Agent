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
    """Theme-aware HSV road-mask configuration."""

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
    """Edge-pair and corner diagnostics for piecewise-straight kart tracks."""

    canny_low: int = 40
    canny_high: int = 120
    hough_threshold: int = 24
    min_line_length_fraction: float = 0.16
    max_line_gap_fraction: float = 0.025
    axis_tolerance_deg: float = 10.0
    min_axis_separation_deg: float = 25.0
    min_road_width_fraction: float = 0.025
    max_road_width_fraction: float = 0.22
    min_axis_overlap_fraction: float = 0.06
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
        if not 0.0 < self.min_road_width_fraction < self.max_road_width_fraction <= 1.0:
            raise ValueError(
                "road width fractions must satisfy 0 < min < max <= 1"
            )
        if not 0.0 < self.min_axis_overlap_fraction <= 1.0:
            raise ValueError("min_axis_overlap_fraction must be in (0, 1]")
        if not 0.0 <= self.corner_extension_fraction <= 1.0:
            raise ValueError("corner_extension_fraction must be in [0, 1]")
        if not 0.0 <= self.anchor_x_norm <= 1.0:
            raise ValueError("anchor_x_norm must be in [0, 1]")
        if not 0.0 <= self.anchor_y_norm <= 1.0:
            raise ValueError("anchor_y_norm must be in [0, 1]")


@dataclass(frozen=True)
class LineSegment:
    """One Hough road-edge segment."""

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
class CenterAxisSegment:
    """Road center axis inferred from two approximately parallel road edges."""

    x1: float
    y1: float
    x2: float
    y2: float
    angle_deg: float
    length: float
    width_px: float
    pair_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "angle_deg": self.angle_deg,
            "length": self.length,
            "width_px": self.width_px,
            "pair_score": self.pair_score,
        }


@dataclass(frozen=True)
class RectilinearGeometry:
    """Theme-independent geometry distilled from a road candidate mask.

    Hough lines are treated as road *edges*. Center axes are inferred only when
    two compatible, overlapping, parallel edge segments can be paired. Corner
    diagnostics are intersections between center axes, never raw edge lines.
    """

    geometry_class: str
    primary_angle_deg: float | None
    secondary_angle_deg: float | None
    primary_support: float
    secondary_support: float
    straight_confidence: float
    corner_score: float
    primary_center_axis: CenterAxisSegment | None
    secondary_center_axis: CenterAxisSegment | None
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
            "center_axis_usable": self.primary_center_axis is not None,
            "primary_center_axis": (
                self.primary_center_axis.to_dict()
                if self.primary_center_axis is not None
                else None
            ),
            "secondary_center_axis": (
                self.secondary_center_axis.to_dict()
                if self.secondary_center_axis is not None
                else None
            ),
            "corner_visible": self.corner_visible,
            "corner_x_norm": self.corner_x_norm,
            "corner_y_norm": self.corner_y_norm,
            "corner_distance_norm": self.corner_distance_norm,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def _odd_kernel(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def extract_road_mask(frame_bgr: np.ndarray, config: RoadMaskConfig) -> np.ndarray:
    """Return a binary uint8 road candidate mask from multiple HSV ranges."""

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
    """Legacy row-center diagnostic retained for historical comparison."""

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


def _project_segment(
    segment: LineSegment,
    tangent: tuple[float, float],
    normal: tuple[float, float],
) -> tuple[float, float, float]:
    tx, ty = tangent
    nx, ny = normal
    t1 = segment.x1 * tx + segment.y1 * ty
    t2 = segment.x2 * tx + segment.y2 * ty
    n1 = segment.x1 * nx + segment.y1 * ny
    n2 = segment.x2 * nx + segment.y2 * ny
    return min(t1, t2), max(t1, t2), 0.5 * (n1 + n2)


def _best_center_axis(
    members: list[LineSegment],
    axis_angle_deg: float,
    *,
    diag: float,
    config: RectilinearGeometryConfig,
) -> CenterAxisSegment | None:
    """Pair parallel road edges and return the strongest center-axis candidate."""

    if len(members) < 2:
        return None
    theta = math.radians(axis_angle_deg)
    tangent = (math.cos(theta), math.sin(theta))
    normal = (-math.sin(theta), math.cos(theta))
    min_width = diag * config.min_road_width_fraction
    max_width = diag * config.max_road_width_fraction
    min_overlap = diag * config.min_axis_overlap_fraction

    best: CenterAxisSegment | None = None
    for first_index, first in enumerate(members):
        first_start, first_end, first_offset = _project_segment(
            first, tangent, normal
        )
        for second in members[first_index + 1 :]:
            if _angle_distance_deg(first.angle_deg, second.angle_deg) > config.axis_tolerance_deg:
                continue
            second_start, second_end, second_offset = _project_segment(
                second, tangent, normal
            )
            overlap_start = max(first_start, second_start)
            overlap_end = min(first_end, second_end)
            overlap = overlap_end - overlap_start
            if overlap < min_overlap:
                continue
            road_width = abs(first_offset - second_offset)
            if not min_width <= road_width <= max_width:
                continue

            center_offset = 0.5 * (first_offset + second_offset)
            tx, ty = tangent
            nx, ny = normal
            x1 = overlap_start * tx + center_offset * nx
            y1 = overlap_start * ty + center_offset * ny
            x2 = overlap_end * tx + center_offset * nx
            y2 = overlap_end * ty + center_offset * ny

            # Longer overlap is more trustworthy. Mildly prefer narrower pairs
            # because unrelated parallel track pieces are more likely far apart.
            pair_score = overlap / max(1.0, road_width)
            candidate = CenterAxisSegment(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                angle_deg=float(axis_angle_deg),
                length=float(overlap),
                width_px=float(road_width),
                pair_score=float(pair_score),
            )
            if best is None or candidate.pair_score > best.pair_score:
                best = candidate
    return best


def _axis_intersection(
    first: CenterAxisSegment,
    second: CenterAxisSegment,
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


def _point_near_axis_extent(
    x: float,
    y: float,
    axis: CenterAxisSegment,
    extension: float,
) -> bool:
    length = max(axis.length, 1e-6)
    tx = (axis.x2 - axis.x1) / length
    ty = (axis.y2 - axis.y1) / length
    projection = (x - axis.x1) * tx + (y - axis.y1) * ty
    return -extension <= projection <= length + extension


def _unknown_geometry() -> RectilinearGeometry:
    return RectilinearGeometry(
        geometry_class="unknown",
        primary_angle_deg=None,
        secondary_angle_deg=None,
        primary_support=0.0,
        secondary_support=0.0,
        straight_confidence=0.0,
        corner_score=0.0,
        primary_center_axis=None,
        secondary_center_axis=None,
        corner_visible=False,
        corner_x_norm=None,
        corner_y_norm=None,
        corner_distance_norm=None,
        segments=(),
    )


def estimate_rectilinear_geometry(
    mask: np.ndarray,
    config: RectilinearGeometryConfig | None = None,
) -> RectilinearGeometry:
    """Estimate road center axes and their nearest visible intersection."""

    config = config or RectilinearGeometryConfig()
    config.validate()
    if mask.ndim != 2:
        raise ValueError("mask must be HxW")

    segments = detect_line_segments(mask, config)
    if not segments:
        return _unknown_geometry()

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
    primary_center = _best_center_axis(
        primary_members,
        primary_angle,
        diag=diag,
        config=config,
    )
    secondary_center = (
        _best_center_axis(
            secondary_members,
            secondary_angle,
            diag=diag,
            config=config,
        )
        if secondary_angle is not None
        else None
    )

    corner_visible = False
    corner_x_norm = None
    corner_y_norm = None
    corner_distance_norm = None
    if primary_center is not None and secondary_center is not None:
        point = _axis_intersection(primary_center, secondary_center)
        if point is not None:
            x, y = point
            extension = diag * config.corner_extension_fraction
            in_frame = (
                -extension <= x <= width - 1 + extension
                and -extension <= y <= height - 1 + extension
            )
            if (
                in_frame
                and _point_near_axis_extent(x, y, primary_center, extension)
                and _point_near_axis_extent(x, y, secondary_center, extension)
            ):
                anchor_x = config.anchor_x_norm * max(1, width - 1)
                anchor_y = config.anchor_y_norm * max(1, height - 1)
                corner_visible = True
                corner_x_norm = float(np.clip(x / max(1, width - 1), 0.0, 1.0))
                corner_y_norm = float(np.clip(y / max(1, height - 1), 0.0, 1.0))
                corner_distance_norm = float(
                    math.hypot(x - anchor_x, y - anchor_y) / diag
                )

    if corner_visible:
        geometry_class = "corner_visible"
    elif primary_center is not None:
        geometry_class = "straight" if secondary_center is None else "mixed_center_axes"
    else:
        geometry_class = "edge_only"

    return RectilinearGeometry(
        geometry_class=geometry_class,
        primary_angle_deg=(
            primary_center.angle_deg if primary_center is not None else float(primary_angle)
        ),
        secondary_angle_deg=(
            secondary_center.angle_deg
            if secondary_center is not None
            else (float(secondary_angle) if secondary_angle is not None else None)
        ),
        primary_support=float(primary_support),
        secondary_support=float(secondary_support),
        straight_confidence=float(straight_confidence),
        corner_score=float(corner_score),
        primary_center_axis=primary_center,
        secondary_center_axis=secondary_center,
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
                "center_axis_usable": 1.0
                if geometry.primary_center_axis is not None
                else 0.0,
                "secondary_center_axis_usable": 1.0
                if geometry.secondary_center_axis is not None
                else 0.0,
            }
        )
    return metrics


def _draw_center_axis(
    overlay: np.ndarray,
    axis: CenterAxisSegment,
    color: tuple[int, int, int],
) -> None:
    cv2.line(
        overlay,
        (int(round(axis.x1)), int(round(axis.y1))),
        (int(round(axis.x2)), int(round(axis.y2))),
        color,
        4,
        cv2.LINE_AA,
    )


def overlay_road_geometry(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    centerline: list[tuple[int, float]] | None = None,
    geometry: RectilinearGeometry | None = None,
    geometry_config: RectilinearGeometryConfig | None = None,
) -> np.ndarray:
    """Render candidate road mask, edge evidence, center axes, and corner."""

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

    config = geometry_config or RectilinearGeometryConfig()
    primary = geometry.primary_angle_deg
    secondary = geometry.secondary_angle_deg
    for segment in geometry.segments:
        if primary is not None and _angle_distance_deg(
            segment.angle_deg, primary
        ) <= config.axis_tolerance_deg:
            color = (0, 190, 190)
        elif secondary is not None and _angle_distance_deg(
            segment.angle_deg, secondary
        ) <= config.axis_tolerance_deg:
            color = (180, 0, 180)
        else:
            color = (120, 120, 120)
        cv2.line(
            overlay,
            (segment.x1, segment.y1),
            (segment.x2, segment.y2),
            color,
            1,
            cv2.LINE_AA,
        )

    if geometry.primary_center_axis is not None:
        _draw_center_axis(overlay, geometry.primary_center_axis, (255, 255, 0))
    if geometry.secondary_center_axis is not None:
        _draw_center_axis(overlay, geometry.secondary_center_axis, (255, 0, 255))

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
