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

from karting_agent.train.geometry_pseudo_labels import LineSegment, RectilinearGeometry


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
        if not (
            0.0 <= self.anchor_x_norm <= 1.0
            and 0.0 <= self.anchor_y_norm <= 1.0
        ):
            raise ValueError("kart anchor must be normalized to [0,1]")
        if self.anchor_distance_weight < 0:
            raise ValueError("anchor_distance_weight must be >= 0")
        if not 0.0 <= self.join_distance_fraction <= 1.0:
            raise ValueError("join_distance_fraction must be in [0,1]")
        if not 0.0 <= self.min_heading_quality <= 1.0:
            raise ValueError("min_heading_quality must be in [0,1]")


@dataclass(frozen=True)
class KartRoadRelationConfig:
    """Local road-axis and cross-section configuration around the kart."""

    tangent_offsets_fraction: tuple[float, ...] = (-0.05, 0.05)
    max_normal_distance_fraction: float = 0.30
    min_road_width_fraction: float = 0.025
    max_road_width_fraction: float = 0.35
    local_axis_tolerance_deg: float = 10.0
    local_axis_radius_fraction: float = 0.30

    def validate(self) -> None:
        if not self.tangent_offsets_fraction:
            raise ValueError("at least one tangent cross-section offset is required")
        if any(abs(value) > 1.0 for value in self.tangent_offsets_fraction):
            raise ValueError("tangent offsets must be fractions in [-1,1]")
        if not 0.0 < self.max_normal_distance_fraction <= 1.0:
            raise ValueError("max_normal_distance_fraction must be in (0,1]")
        if not (
            0.0
            < self.min_road_width_fraction
            < self.max_road_width_fraction
            <= 1.0
        ):
            raise ValueError("road width fractions must satisfy 0 < min < max <= 1")
        if not 0.0 < self.local_axis_tolerance_deg < 90.0:
            raise ValueError("local_axis_tolerance_deg must be in (0,90)")
        if not 0.0 < self.local_axis_radius_fraction <= 1.0:
            raise ValueError("local_axis_radius_fraction must be in (0,1]")


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
    road_support_fraction: float
    heading_error_deg: float | None
    abs_heading_error_deg: float | None
    local_road_center_x: float
    local_road_center_y: float
    local_road_width_px: float
    lateral_offset_px: float
    lateral_offset_norm: float
    inside_road: bool
    valid_cross_sections: int
    cross_section_count: int
    confidence: float

    @property
    def local_road_usable(self) -> bool:
        return True

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "road_angle_deg": self.road_angle_deg,
            "road_support_fraction": self.road_support_fraction,
            "heading_error_deg": self.heading_error_deg,
            "abs_heading_error_deg": self.abs_heading_error_deg,
            "local_road_center_x": self.local_road_center_x,
            "local_road_center_y": self.local_road_center_y,
            "local_road_width_px": self.local_road_width_px,
            "lateral_offset_px": self.lateral_offset_px,
            "lateral_offset_norm": self.lateral_offset_norm,
            "inside_road": self.inside_road,
            "local_road_usable": True,
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

    kernel_size = (
        config.close_kernel if config.close_kernel % 2 else config.close_kernel + 1
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, kernel)
    return warm, red


def estimate_kart_pose(
    frame_bgr: np.ndarray,
    config: KartPoseConfig | None = None,
) -> tuple[KartPose | None, np.ndarray]:
    """Estimate kart center and axial heading from red/orange chassis pixels."""

    config = config or KartPoseConfig()
    config.validate()
    warm, red = extract_kart_color_masks(frame_bgr, config)
    height, width = warm.shape
    image_area = max(1, height * width)
    diag = max(1.0, math.hypot(width, height))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(warm, 8)
    anchor = np.asarray(
        [config.anchor_x_norm * width, config.anchor_y_norm * height],
        dtype=np.float64,
    )
    candidates: list[tuple[float, int, int, np.ndarray, float]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_fraction = area / image_area
        if not (
            config.min_component_area_fraction
            <= area_fraction
            <= config.max_component_area_fraction
        ):
            continue
        component = labels == label
        red_fraction = float(
            np.count_nonzero((red > 0) & component) / max(1, area)
        )
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
    heading_quality = (
        0.0
        if major <= 1e-9
        else float(np.clip(1.0 - minor / major, 0.0, 1.0))
    )
    heading_angle: float | None = None
    if heading_quality >= config.min_heading_quality:
        vector = eigenvectors[:, major_index]
        heading_angle = float(
            math.degrees(math.atan2(vector[1], vector[0])) % 180.0
        )

    selected_area = int(xs.size)
    red_fraction = float(
        np.count_nonzero((red > 0) & selected) / max(1, selected_area)
    )
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


def _angle_distance_deg(left: float, right: float) -> float:
    return abs(signed_axial_difference_deg(left, right))


def _point_segment_distance(
    x: float,
    y: float,
    segment: LineSegment,
) -> float:
    start = np.asarray([segment.x1, segment.y1], dtype=np.float64)
    end = np.asarray([segment.x2, segment.y2], dtype=np.float64)
    point = np.asarray([x, y], dtype=np.float64)
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-9:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    nearest = start + fraction * delta
    return float(np.linalg.norm(point - nearest))


def _local_axis_candidates(
    geometry: RectilinearGeometry,
    pose: KartPose,
    *,
    diag: float,
    config: KartRoadRelationConfig,
) -> list[tuple[float, float]]:
    """Return distinct local road-axis candidates as (angle, support_fraction)."""

    if not geometry.segments:
        return []
    radius = max(1.0, diag * config.local_axis_radius_fraction)
    weighted: list[tuple[LineSegment, float]] = []
    total_length = max(1e-6, sum(segment.length for segment in geometry.segments))
    for segment in geometry.segments:
        distance = _point_segment_distance(pose.center_x, pose.center_y, segment)
        proximity = 1.0 / (1.0 + distance / radius)
        weighted.append((segment, segment.length * proximity))

    raw: list[tuple[float, float]] = []
    for seed, _ in weighted:
        members = [
            item
            for item in weighted
            if _angle_distance_deg(item[0].angle_deg, seed.angle_deg)
            <= config.local_axis_tolerance_deg
        ]
        if not members:
            continue
        sin_sum = sum(
            math.sin(math.radians(2.0 * segment.angle_deg)) * weight
            for segment, weight in members
        )
        cos_sum = sum(
            math.cos(math.radians(2.0 * segment.angle_deg)) * weight
            for segment, weight in members
        )
        angle = float(
            (0.5 * math.degrees(math.atan2(sin_sum, cos_sum))) % 180.0
        )
        support_fraction = float(
            np.clip(sum(weight for _, weight in members) / total_length, 0.0, 1.0)
        )
        raw.append((angle, support_fraction))

    distinct: list[tuple[float, float]] = []
    for angle, support in sorted(raw, key=lambda item: item[1], reverse=True):
        if all(
            _angle_distance_deg(angle, existing_angle)
            > config.local_axis_tolerance_deg
            for existing_angle, _ in distinct
        ):
            distinct.append((angle, support))
    return distinct


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
        distance = (
            0.0 if first <= 0.0 <= last else min(abs(first), abs(last))
        )
        center = 0.5 * (first + last)
        candidates.append((distance, -road_width, center))
    if not candidates:
        return None

    _, negative_width, center = min(candidates)
    return center, -negative_width


def _cross_sections_for_angle(
    road_mask: np.ndarray,
    pose: KartPose,
    road_angle: float,
    *,
    config: KartRoadRelationConfig,
) -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray, float]:
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
    return cross_sections, tangent, normal, diag


def estimate_kart_road_relation(
    road_mask: np.ndarray,
    geometry: RectilinearGeometry,
    pose: KartPose,
    config: KartRoadRelationConfig | None = None,
) -> KartRoadRelation | None:
    """Estimate the locally relevant road axis and kart-relative state.

    Global dominant orientation is insufficient near an intersection because a
    long but distant road segment can win the Hough-support vote. This function
    therefore evaluates multiple Hough orientation clusters around the kart.
    A candidate is preferred when local normal cross-sections form a plausible
    corridor near the kart, its axis is compatible with kart heading, and it has
    stronger nearby edge support. If no candidate yields a plausible local road
    cross-section, the relation fails closed instead of emitting a label.
    """

    config = config or KartRoadRelationConfig()
    config.validate()
    if road_mask.ndim != 2:
        raise ValueError("road_mask must be HxW")
    if not geometry.segments:
        return None

    height, width = road_mask.shape
    diag = max(1.0, math.hypot(width, height))
    candidates = _local_axis_candidates(
        geometry,
        pose,
        diag=diag,
        config=config,
    )
    if not candidates:
        return None

    selected: tuple[
        float,
        float,
        list[tuple[float, float]],
        np.ndarray,
        float,
        float,
    ] | None = None
    for road_angle, support_fraction in candidates:
        cross_sections, _, normal, _ = _cross_sections_for_angle(
            road_mask,
            pose,
            road_angle,
            config=config,
        )
        if not cross_sections:
            continue
        center_offset = float(np.median([item[0] for item in cross_sections]))
        road_width = float(np.median([item[1] for item in cross_sections]))
        lateral_norm = float(-center_offset / max(1.0, 0.5 * road_width))
        heading_error = (
            abs(signed_axial_difference_deg(pose.heading_angle_deg, road_angle))
            if pose.heading_angle_deg is not None
            else 0.0
        )

        # The score is intentionally simple and auditable. Lateral proximity is
        # primary, heading compatibility breaks ambiguous intersection cases,
        # while nearby edge support and two valid probes provide small bonuses.
        score = (
            abs(lateral_norm)
            + 0.65 * (heading_error / 90.0)
            - 0.25 * support_fraction
            - 0.10 * len(cross_sections)
        )
        candidate = (
            score,
            road_angle,
            cross_sections,
            normal,
            support_fraction,
            road_width,
        )
        if selected is None or candidate[0] < selected[0]:
            selected = candidate

    if selected is None:
        return None

    _, road_angle, cross_sections, normal, support_fraction, road_width = selected
    center_offset = float(np.median([item[0] for item in cross_sections]))
    kart_center = np.asarray([pose.center_x, pose.center_y], dtype=np.float64)
    local_center = kart_center + normal * center_offset
    lateral_px = float(-center_offset)
    lateral_norm = float(lateral_px / max(1.0, 0.5 * road_width))
    heading_error = (
        signed_axial_difference_deg(pose.heading_angle_deg, road_angle)
        if pose.heading_angle_deg is not None
        else None
    )
    valid_fraction = len(cross_sections) / len(config.tangent_offsets_fraction)
    corner_factor = max(0.25, 1.0 - geometry.corner_score)
    confidence = float(
        np.clip(
            pose.heading_quality
            * support_fraction
            * valid_fraction
            * corner_factor,
            0.0,
            1.0,
        )
    )
    return KartRoadRelation(
        road_angle_deg=float(road_angle),
        road_support_fraction=float(support_fraction),
        heading_error_deg=heading_error,
        abs_heading_error_deg=(abs(heading_error) if heading_error is not None else None),
        local_road_center_x=float(local_center[0]),
        local_road_center_y=float(local_center[1]),
        local_road_width_px=float(road_width),
        lateral_offset_px=lateral_px,
        lateral_offset_norm=lateral_norm,
        inside_road=abs(lateral_norm) <= 1.0,
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
        (
            pose.bbox_x + pose.bbox_width - 1,
            pose.bbox_y + pose.bbox_height - 1,
        ),
        (0, 165, 255),
        1,
        cv2.LINE_AA,
    )
    if pose.heading_angle_deg is not None:
        theta = math.radians(pose.heading_angle_deg)
        half_length = 45.0
        delta = np.asarray([math.cos(theta), math.sin(theta)]) * half_length
        start = (
            int(round(pose.center_x - delta[0])),
            int(round(pose.center_y - delta[1])),
        )
        end = (
            int(round(pose.center_x + delta[0])),
            int(round(pose.center_y + delta[1])),
        )
        cv2.line(overlay, start, end, (255, 0, 255), 3, cv2.LINE_AA)

    if relation is not None:
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
