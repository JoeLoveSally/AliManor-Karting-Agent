"""Offline kart-pose and kart-relative road diagnostics for v4.

This module is a teacher/debugging component only. It must not participate in
runtime control. The first POC intentionally uses simple, auditable color and
geometry cues so failure modes are visible before any labels are used for
training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from karting_agent.train.geometry_pseudo_labels import RectilinearGeometry


HsvTriplet = tuple[int, int, int]
HsvRange = tuple[HsvTriplet, HsvTriplet]


@dataclass(frozen=True)
class KartPoseConfig:
    """Warm-color kart detector and axial-heading estimator configuration."""

    warm_hsv_ranges: tuple[HsvRange, ...] = (
        ((0, 120, 90), (12, 255, 255)),
        ((13, 110, 100), (35, 255, 255)),
        ((170, 120, 90), (179, 255, 255)),
    )
    red_hsv_ranges: tuple[HsvRange, ...] = (
        ((0, 120, 90), (12, 255, 255)),
        ((170, 120, 90), (179, 255, 255)),
    )
    search_x_min_norm: float = 0.18
    search_x_max_norm: float = 0.92
    search_y_min_norm: float = 0.28
    search_y_max_norm: float = 0.82
    close_kernel: int = 5
    min_component_area_fraction: float = 0.00015
    max_component_area_fraction: float = 0.01
    min_red_fraction: float = 0.15
    anchor_x_norm: float = 0.55
    anchor_y_norm: float = 0.50
    anchor_distance_weight: float = 6.0
    join_distance_fraction: float = 0.075
    min_heading_quality: float = 0.35

    def validate(self) -> None:
        if not self.warm_hsv_ranges or not self.red_hsv_ranges:
            raise ValueError("kart HSV ranges must not be empty")
        if not 0.0 <= self.search_x_min_norm < self.search_x_max_norm <= 1.0:
            raise ValueError("kart search x range must satisfy 0 <= min < max <= 1")
        if not 0.0 <= self.search_y_min_norm < self.search_y_max_norm <= 1.0:
            raise ValueError("kart search y range must satisfy 0 <= min < max <= 1")
        if self.close_kernel < 1:
            raise ValueError("close_kernel must be >= 1")
        if not (
            0.0
            < self.min_component_area_fraction
            < self.max_component_area_fraction
            <= 1.0
        ):
            raise ValueError("kart component area fractions must satisfy 0 < min < max <= 1")
        if not 0.0 <= self.min_red_fraction <= 1.0:
            raise ValueError("min_red_fraction must be in [0,1]")
        if not 0.0 <= self.anchor_x_norm <= 1.0 or not 0.0 <= self.anchor_y_norm <= 1.0:
            raise ValueError("kart anchor must be normalized to [0,1]")
        if self.anchor_distance_weight < 0:
            raise ValueError("anchor_distance_weight must be >= 0")
        if not 0.0 <= self.join_distance_fraction <= 1.0:
            raise ValueError("join_distance_fraction must be in [0,1]")
        if not 0.0 <= self.min_heading_quality <= 1.0:
            raise ValueError("min_heading_quality must be in [0,1]")


@dataclass(frozen=True)
class KartRoadRelationConfig:
    """Local road cross-section configuration around the detected kart."""

    tangent_offsets_fraction: tuple[float, ...] = (-0.05, 0.05)
    max_normal_distance_fraction: float = 0.30
    min_road_width_fraction: float = 0.025
    max_road_width_fraction: float = 0.35

    def validate(self) -> None:
        if not self.tangent_offsets_fraction:
            raise ValueError("at least one tangent cross-section offset is required")
        if any(abs(value) > 1.0 for value in self.tangent_offsets_fraction):
            raise ValueError("tangent offsets must be fractions in [-1,1]")
        if not 0.0 < self.max_normal_distance_fraction <= 1.0:
            raise ValueError("max_normal_distance_fraction must be in (0,1]")
        if not 0.0 < self.min_road_width_fraction < self.max_road_width_fraction <= 1.0:
            raise ValueError("road width fractions must satisfy 0 < min < max <= 1")


@dataclass(frozen=True)
class KartPose:
    center_x: float
    center_y: float
    center_x_norm: float
    center_y_norm: float
    heading_angle_deg: float | None
    heading_quality: float
    area_fraction: float
    red_fraction: float
    anchor_distance_norm: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int

    @property
    def heading_usable(self) -> bool:
        return self.heading_angle_deg is not None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "center_x_norm": self.center_x_norm,
            "center_y_norm": self.center_y_norm,
            "heading_angle_deg": self.heading_angle_deg,
            "heading_quality": self.heading_quality,
            "heading_usable": self.heading_usable,
            "area_fraction": self.area_fraction,
            "red_fraction": self.red_fraction,
            "anchor_distance_norm": self.anchor_distance_norm,
            "bbox_x": self.bbox_x,
            "bbox_y": self.bbox_y,
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
        }


@dataclass(frozen=True)
class KartRoadRelation:
    road_angle_deg: float
    heading_error_deg: float | None
    abs_heading_error_deg: float | None
    local_road_center_x: float | None
    local_road_center_y: float | None
    local_road_width_px: float | None
    lateral_offset_px: float | None
    lateral_offset_norm: float | None
    inside_road: bool | None
    valid_cross_sections: int
    cross_section_count: int
    confidence: float

    @property
    def local_road_usable(self) -> bool:
        return self.lateral_offset_norm is not None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "road_angle_deg": self.road_angle_deg,
            "heading_error_deg": self.heading_error_deg,
            "abs_heading_error_deg": self.abs_heading_error_deg,
            "local_road_center_x": self.local_road_center_x,
            "local_road_center_y": self.local_road_center_y,
            "local_road_width_px": self.local_road_width_px,
            "lateral_offset_px": self.lateral_offset_px,
            "lateral_offset_norm": self.lateral_offset_norm,
            "inside_road": self.inside_road,
            "local_road_usable": self.local_road_usable,
            "valid_cross_sections": self.valid_cross_sections,
            "cross_section_count": self.cross_section_count,
            "confidence": self.confidence,
        }


def _validate_hsv_ranges(ranges: Sequence[HsvRange]) -> None:
    for lower, upper in ranges:
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("HSV lower/upper bounds must contain 3 values")
        if lower[0] < 0 or upper[0] > 179:
            raise ValueError("OpenCV HSV hue must be in [0,179]")
        if any(value < 0 or value > 255 for value in lower[1:] + upper[1:]):
            raise ValueError("HSV saturation/value must be in [0,255]")
        if any(low > high for low, high in zip(lower, upper)):
            raise ValueError("HSV lower bound must not exceed upper bound")


def _hsv_union(hsv: np.ndarray, ranges: Sequence[HsvRange]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                hsv,
                np.asarray(lower, dtype=np.uint8),
                np.asarray(upper, dtype=np.uint8),
            ),
        )
    return mask


def extract_kart_color_masks(
    frame_bgr: np.ndarray,
    config: KartPoseConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return warm kart-candidate mask and a stricter red-support mask."""

    config = config or KartPoseConfig()
    config.validate()
    _validate_hsv_ranges(config.warm_hsv_ranges)
    _validate_hsv_ranges(config.red_hsv_ranges)
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape HxWx3")

    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    warm = _hsv_union(hsv, config.warm_hsv_ranges)
    red = _hsv_union(hsv, config.red_hsv_ranges)

    roi = np.zeros((height, width), dtype=np.uint8)
    x0 = int(round(config.search_x_min_norm * width))
    x1 = int(round(config.search_x_max_norm * width))
    y0 = int(round(config.search_y_min_norm * height))
    y1 = int(round(config.search_y_max_norm * height))
    roi[y0:y1, x0:x1] = 255
    warm = cv2.bitwise_and(warm, roi)
    red = cv2.bitwise_and(red, roi)

    kernel_size = config.close_kernel if config.close_kernel % 2 else config.close_kernel + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, kernel)
    return warm, red


def estimate_kart_pose(
    frame_bgr: np.ndarray,
    config: KartPoseConfig | None = None,
) -> tuple[KartPose | None, np.ndarray]:
    """Estimate kart center and axial heading from its red/orange chassis pixels.

    The detector intentionally ignores the mostly white chicken body. Warm kart
    pieces are first filtered by red support, then nearby pieces are merged so
    PCA operates on the chassis rather than one disconnected decorative part.
    The heading is axial (modulo 180 degrees); front/back is not inferred here.
    """

    config = config or KartPoseConfig()
    config.validate()
    warm, red = extract_kart_color_masks(frame_bgr, config)
    height, width = warm.shape
    image_area = max(1, height * width)
    diag = max(1.0, math.hypot(width, height))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(warm, 8)
    anchor = np.asarray(
        [config.anchor_x_norm * width, config.anchor_y_norm * height], dtype=np.float64
    )
    candidates: list[tuple[float, int, int, np.ndarray, float]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_fraction = area / image_area
        if not config.min_component_area_fraction <= area_fraction <= config.max_component_area_fraction:
            continue
        component = labels == label
        red_fraction = float(np.count_nonzero((red > 0) & component) / max(1, area))
        if red_fraction < config.min_red_fraction:
            continue
        center = centroids[label].astype(np.float64)
        anchor_distance = float(np.linalg.norm(center - anchor) / diag)
        score = area * (0.5 + red_fraction) / (
            1.0 + config.anchor_distance_weight * anchor_distance
        )
        candidates.append((score, label, area, center, red_fraction))

    if not candidates:
        return None, warm

    _, _, _, seed_center, _ = max(candidates, key=lambda item: item[0])
    join_distance = diag * config.join_distance_fraction
    joined_labels = [
        label
        for _, label, _, center, _ in candidates
        if float(np.linalg.norm(center - seed_center)) <= join_distance
    ]
    selected = np.isin(labels, joined_labels)
    ys, xs = np.nonzero(selected)
    if xs.size < 3:
        return None, warm

    xy = np.column_stack((xs, ys)).astype(np.float64)
    center = xy.mean(axis=0)
    centered = xy - center
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_index = int(np.argmax(eigenvalues))
    major = float(max(eigenvalues[major_index], 0.0))
    minor = float(max(eigenvalues[1 - major_index], 0.0))
    heading_quality = 0.0 if major <= 1e-9 else float(np.clip(1.0 - minor / major, 0.0, 1.0))
    heading_angle: float | None = None
    if heading_quality >= config.min_heading_quality:
        vector = eigenvectors[:, major_index]
        heading_angle = float(math.degrees(math.atan2(vector[1], vector[0])) % 180.0)

    selected_area = int(xs.size)
    red_fraction = float(np.count_nonzero((red > 0) & selected) / max(1, selected_area))
    anchor_distance = float(np.linalg.norm(center - anchor) / diag)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    pose = KartPose(
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_x_norm=float(center[0] / max(1, width - 1)),
        center_y_norm=float(center[1] / max(1, height - 1)),
        heading_angle_deg=heading_angle,
        heading_quality=heading_quality,
        area_fraction=float(selected_area / image_area),
        red_fraction=red_fraction,
        anchor_distance_norm=anchor_distance,
        bbox_x=x_min,
        bbox_y=y_min,
        bbox_width=x_max - x_min + 1,
        bbox_height=y_max - y_min + 1,
    )
    return pose, warm


def signed_axial_difference_deg(angle: float, reference: float) -> float:
    """Return signed minimum axial difference in [-90, 90)."""

    return float((angle - reference + 90.0) % 180.0 - 90.0)


def _nearest_road_run(
    mask: np.ndarray,
    base: np.ndarray,
    normal: np.ndarray,
    *,
    max_distance: int,
    min_width: float,
    max_width: float,
) -> tuple[float, float] | None:
    """Return (center_offset_px, width_px) for the nearest plausible road run."""

    height, width = mask.shape
    offsets = np.arange(-max_distance, max_distance + 1, dtype=np.int32)
    points = base[None, :] + offsets[:, None] * normal[None, :]
    xs = np.rint(points[:, 0]).astype(np.int32)
    ys = np.rint(points[:, 1]).astype(np.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    occupied = np.zeros(offsets.shape, dtype=bool)
    occupied[valid] = mask[ys[valid], xs[valid]] > 0
    indices = np.flatnonzero(occupied)
    if indices.size == 0:
        return None

    breaks = np.where(np.diff(indices) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [indices.size - 1]))
    candidates: list[tuple[float, float, float]] = []
    for start, end in zip(starts, ends):
        first = float(offsets[indices[start]])
        last = float(offsets[indices[end]])
        road_width = last - first + 1.0
        if not min_width <= road_width <= max_width:
            continue
        distance = 0.0 if first <= 0.0 <= last else min(abs(first), abs(last))
        center = 0.5 * (first + last)
        candidates.append((distance, -road_width, center))
    if not candidates:
        return None

    _, negative_width, center = min(candidates)
    return center, -negative_width


def estimate_kart_road_relation(
    road_mask: np.ndarray,
    geometry: RectilinearGeometry,
    pose: KartPose,
    config: KartRoadRelationConfig | None = None,
) -> KartRoadRelation | None:
    """Estimate road-relative heading and lateral displacement near the kart.

    Lateral offset is obtained from local cross-sections perpendicular to the
    dominant road axis. Cross-sections are shifted slightly forward/backward
    along the road tangent so the kart sprite itself does not create a large
    hole in the road mask.
    """

    config = config or KartRoadRelationConfig()
    config.validate()
    if road_mask.ndim != 2:
        raise ValueError("road_mask must be HxW")
    if geometry.primary_angle_deg is None:
        return None

    road_angle = float(geometry.primary_angle_deg)
    heading_error = (
        signed_axial_difference_deg(pose.heading_angle_deg, road_angle)
        if pose.heading_angle_deg is not None
        else None
    )
    abs_heading_error = abs(heading_error) if heading_error is not None else None

    height, width = road_mask.shape
    diag = max(1.0, math.hypot(width, height))
    theta = math.radians(road_angle)
    tangent = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
    normal = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    kart_center = np.asarray([pose.center_x, pose.center_y], dtype=np.float64)
    max_distance = max(1, int(round(diag * config.max_normal_distance_fraction)))
    min_width = diag * config.min_road_width_fraction
    max_width = diag * config.max_road_width_fraction

    cross_sections: list[tuple[float, float]] = []
    for tangent_fraction in config.tangent_offsets_fraction:
        base = kart_center + tangent * (diag * tangent_fraction)
        candidate = _nearest_road_run(
            road_mask,
            base,
            normal,
            max_distance=max_distance,
            min_width=min_width,
            max_width=max_width,
        )
        if candidate is not None:
            cross_sections.append(candidate)

    local_center_x: float | None = None
    local_center_y: float | None = None
    local_width: float | None = None
    lateral_px: float | None = None
    lateral_norm: float | None = None
    inside_road: bool | None = None
    if cross_sections:
        center_offset = float(np.median([item[0] for item in cross_sections]))
        local_width = float(np.median([item[1] for item in cross_sections]))
        local_center = kart_center + normal * center_offset
        local_center_x = float(local_center[0])
        local_center_y = float(local_center[1])
        lateral_px = float(-center_offset)
        lateral_norm = float(lateral_px / max(1.0, 0.5 * local_width))
        inside_road = abs(lateral_norm) <= 1.0

    valid_fraction = len(cross_sections) / len(config.tangent_offsets_fraction)
    confidence = float(
        np.clip(
            pose.heading_quality
            * geometry.straight_confidence
            * max(0.0, 1.0 - geometry.corner_score)
            * valid_fraction,
            0.0,
            1.0,
        )
    )
    return KartRoadRelation(
        road_angle_deg=road_angle,
        heading_error_deg=heading_error,
        abs_heading_error_deg=abs_heading_error,
        local_road_center_x=local_center_x,
        local_road_center_y=local_center_y,
        local_road_width_px=local_width,
        lateral_offset_px=lateral_px,
        lateral_offset_norm=lateral_norm,
        inside_road=inside_road,
        valid_cross_sections=len(cross_sections),
        cross_section_count=len(config.tangent_offsets_fraction),
        confidence=confidence,
    )


def overlay_kart_relative_geometry(
    frame_bgr: np.ndarray,
    kart_mask: np.ndarray,
    pose: KartPose | None,
    relation: KartRoadRelation | None,
) -> np.ndarray:
    """Render kart mask, center/heading and local road-relative diagnostics."""

    overlay = frame_bgr.copy()
    if kart_mask.shape != frame_bgr.shape[:2]:
        raise ValueError("kart_mask shape must match frame")

    selected = kart_mask > 0
    if np.any(selected):
        tint = np.zeros_like(frame_bgr)
        tint[..., 2] = 255
        tint[..., 1] = 120
        overlay[selected] = (
            0.55 * overlay[selected].astype(np.float32)
            + 0.45 * tint[selected].astype(np.float32)
        ).astype(np.uint8)

    if pose is None:
        return overlay

    center = (int(round(pose.center_x)), int(round(pose.center_y)))
    cv2.circle(overlay, center, 5, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(
        overlay,
        (pose.bbox_x, pose.bbox_y),
        (pose.bbox_x + pose.bbox_width - 1, pose.bbox_y + pose.bbox_height - 1),
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )
    if pose.heading_angle_deg is not None:
        theta = math.radians(pose.heading_angle_deg)
        half_length = 45.0
        delta = np.asarray([math.cos(theta), math.sin(theta)]) * half_length
        start = (int(round(pose.center_x - delta[0])), int(round(pose.center_y - delta[1])))
        end = (int(round(pose.center_x + delta[0])), int(round(pose.center_y + delta[1])))
        cv2.line(overlay, start, end, (255, 0, 255), 3, cv2.LINE_AA)

    if relation is not None and relation.local_road_center_x is not None:
        assert relation.local_road_center_y is not None
        road_center = (
            int(round(relation.local_road_center_x)),
            int(round(relation.local_road_center_y)),
        )
        color = (0, 220, 0) if relation.inside_road else (0, 0, 255)
        cv2.drawMarker(
            overlay,
            road_center,
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )
        cv2.line(overlay, road_center, center, color, 2, cv2.LINE_AA)
        theta = math.radians(relation.road_angle_deg)
        delta = np.asarray([math.cos(theta), math.sin(theta)]) * 55.0
        start = (
            int(round(relation.local_road_center_x - delta[0])),
            int(round(relation.local_road_center_y - delta[1])),
        )
        end = (
            int(round(relation.local_road_center_x + delta[0])),
            int(round(relation.local_road_center_y + delta[1])),
        )
        cv2.line(overlay, start, end, (255, 255, 0), 2, cv2.LINE_AA)

    return overlay
