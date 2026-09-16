"""Temporally consistent offline kart-road relation teacher for v4-C2 follow-up.

This module is teacher/debugging only. Runtime control remains model-only.
The tracker wraps the existing single-frame kart/road geometry primitives and
adds continuity plus hysteresis so a crossing/next-segment candidate cannot
replace the current corridor on one noisy frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from karting_agent.train.geometry_pseudo_labels import RectilinearGeometry
from karting_agent.train.kart_pose_pseudo_labels import (
    KartPose,
    KartRoadRelation,
    KartRoadRelationConfig,
    _cross_sections_for_angle,
    _local_axis_candidates,
    signed_axial_difference_deg,
)


@dataclass(frozen=True)
class TemporalRoadTrackerConfig:
    """Continuity and hysteresis parameters for the offline road teacher."""

    continuity_weight: float = 0.75
    switch_angle_deg: float = 25.0
    switch_confirm_samples: int = 3
    switch_score_margin: float = 0.05
    reset_gap_samples: int = 3
    stable_angle_alpha: float = 0.25

    def validate(self) -> None:
        if self.continuity_weight < 0.0:
            raise ValueError("continuity_weight must be >= 0")
        if not 0.0 < self.switch_angle_deg < 90.0:
            raise ValueError("switch_angle_deg must be in (0,90)")
        if self.switch_confirm_samples < 1:
            raise ValueError("switch_confirm_samples must be >= 1")
        if self.switch_score_margin < 0.0:
            raise ValueError("switch_score_margin must be >= 0")
        if self.reset_gap_samples < 1:
            raise ValueError("reset_gap_samples must be >= 1")
        if not 0.0 < self.stable_angle_alpha <= 1.0:
            raise ValueError("stable_angle_alpha must be in (0,1]")


@dataclass(frozen=True)
class RoadRelationCandidate:
    """One locally plausible corridor and the v4-C2 single-frame score."""

    relation: KartRoadRelation
    base_score: float


@dataclass(frozen=True)
class TemporalKartRoadResult:
    """Selected relation plus auditable tracker state for one sample."""

    relation: KartRoadRelation | None
    mode: str
    stable_angle_deg: float | None
    pending_angle_deg: float | None
    pending_count: int
    candidate_angles_deg: tuple[float, ...]


def _axial_distance_deg(left: float, right: float) -> float:
    return abs(signed_axial_difference_deg(left, right))


def _axial_blend_deg(previous: float, current: float, alpha: float) -> float:
    """Blend two 180-degree-periodic axial orientations."""

    prev = math.radians(2.0 * previous)
    curr = math.radians(2.0 * current)
    x = (1.0 - alpha) * math.cos(prev) + alpha * math.cos(curr)
    y = (1.0 - alpha) * math.sin(prev) + alpha * math.sin(curr)
    if abs(x) + abs(y) <= 1e-12:
        return float(current % 180.0)
    return float((0.5 * math.degrees(math.atan2(y, x))) % 180.0)


def road_relation_candidates(
    road_mask: np.ndarray,
    geometry: RectilinearGeometry,
    pose: KartPose,
    config: KartRoadRelationConfig | None = None,
) -> list[RoadRelationCandidate]:
    """Return every plausible local corridor using the v4-C2 base score.

    This intentionally mirrors ``estimate_kart_road_relation`` rather than
    changing its single-frame semantics. Temporal selection is layered on top,
    making the teacher-v2 experiment isolated and reversible.
    """

    config = config or KartRoadRelationConfig()
    config.validate()
    if road_mask.ndim != 2:
        raise ValueError("road_mask must be HxW")
    if not geometry.segments:
        return []

    height, width = road_mask.shape
    diag = max(1.0, math.hypot(width, height))
    axes = _local_axis_candidates(geometry, pose, diag=diag, config=config)
    candidates: list[RoadRelationCandidate] = []
    for road_angle, support_fraction in axes:
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
        heading_abs = (
            abs(signed_axial_difference_deg(pose.heading_angle_deg, road_angle))
            if pose.heading_angle_deg is not None
            else 0.0
        )
        base_score = (
            abs(lateral_norm)
            + 0.65 * (heading_abs / 90.0)
            - 0.25 * support_fraction
            - 0.10 * len(cross_sections)
        )

        kart_center = np.asarray([pose.center_x, pose.center_y], dtype=np.float64)
        local_center = kart_center + normal * center_offset
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
        relation = KartRoadRelation(
            road_angle_deg=float(road_angle),
            road_support_fraction=float(support_fraction),
            heading_error_deg=heading_error,
            abs_heading_error_deg=(
                abs(heading_error) if heading_error is not None else None
            ),
            local_road_center_x=float(local_center[0]),
            local_road_center_y=float(local_center[1]),
            local_road_width_px=float(road_width),
            lateral_offset_px=float(-center_offset),
            lateral_offset_norm=lateral_norm,
            inside_road=abs(lateral_norm) <= 1.0,
            valid_cross_sections=len(cross_sections),
            cross_section_count=len(config.tangent_offsets_fraction),
            confidence=confidence,
        )
        candidates.append(RoadRelationCandidate(relation, float(base_score)))
    return candidates


class TemporalKartRoadTracker:
    """Track one local road corridor across sequential teacher samples."""

    def __init__(self, config: TemporalRoadTrackerConfig | None = None) -> None:
        self.config = config or TemporalRoadTrackerConfig()
        self.config.validate()
        self.stable_angle_deg: float | None = None
        self.pending_angle_deg: float | None = None
        self.pending_count = 0
        self.gap_count = 0

    def reset(self) -> None:
        self.stable_angle_deg = None
        self.pending_angle_deg = None
        self.pending_count = 0
        self.gap_count = 0

    def miss(self) -> TemporalKartRoadResult:
        self.gap_count += 1
        if self.gap_count >= self.config.reset_gap_samples:
            self.reset()
        return self._result(None, "missing", ())

    def _result(
        self,
        relation: KartRoadRelation | None,
        mode: str,
        candidate_angles: tuple[float, ...],
    ) -> TemporalKartRoadResult:
        return TemporalKartRoadResult(
            relation=relation,
            mode=mode,
            stable_angle_deg=self.stable_angle_deg,
            pending_angle_deg=self.pending_angle_deg,
            pending_count=self.pending_count,
            candidate_angles_deg=candidate_angles,
        )

    def update_candidates(
        self,
        candidates: list[RoadRelationCandidate],
    ) -> TemporalKartRoadResult:
        """Select from precomputed candidates; exposed for deterministic tests."""

        if not candidates:
            return self.miss()
        self.gap_count = 0
        candidate_angles = tuple(item.relation.road_angle_deg for item in candidates)
        raw_best = min(candidates, key=lambda item: item.base_score)

        if self.stable_angle_deg is None:
            self.stable_angle_deg = raw_best.relation.road_angle_deg
            self.pending_angle_deg = None
            self.pending_count = 0
            return self._result(raw_best.relation, "initialize", candidate_angles)

        stable_angle = self.stable_angle_deg
        compatible = [
            item
            for item in candidates
            if _axial_distance_deg(item.relation.road_angle_deg, stable_angle)
            <= self.config.switch_angle_deg
        ]
        current = (
            min(
                compatible,
                key=lambda item: item.base_score
                + self.config.continuity_weight
                * (
                    _axial_distance_deg(item.relation.road_angle_deg, stable_angle)
                    / 90.0
                ),
            )
            if compatible
            else None
        )

        challenger = (
            _axial_distance_deg(raw_best.relation.road_angle_deg, stable_angle)
            > self.config.switch_angle_deg
        )
        if not challenger:
            selected = current or raw_best
            self.stable_angle_deg = _axial_blend_deg(
                stable_angle,
                selected.relation.road_angle_deg,
                self.config.stable_angle_alpha,
            )
            self.pending_angle_deg = None
            self.pending_count = 0
            return self._result(selected.relation, "stable", candidate_angles)

        compelling = (
            current is None
            or raw_best.base_score + self.config.switch_score_margin < current.base_score
        )
        if not compelling:
            assert current is not None
            self.stable_angle_deg = _axial_blend_deg(
                stable_angle,
                current.relation.road_angle_deg,
                self.config.stable_angle_alpha,
            )
            self.pending_angle_deg = None
            self.pending_count = 0
            return self._result(current.relation, "hold_margin", candidate_angles)

        challenger_angle = raw_best.relation.road_angle_deg
        if (
            self.pending_angle_deg is not None
            and _axial_distance_deg(challenger_angle, self.pending_angle_deg)
            <= self.config.switch_angle_deg
        ):
            self.pending_angle_deg = _axial_blend_deg(
                self.pending_angle_deg,
                challenger_angle,
                self.config.stable_angle_alpha,
            )
            self.pending_count += 1
        else:
            self.pending_angle_deg = challenger_angle
            self.pending_count = 1

        if self.pending_count >= self.config.switch_confirm_samples:
            self.stable_angle_deg = challenger_angle
            self.pending_angle_deg = None
            self.pending_count = 0
            return self._result(raw_best.relation, "switch_confirmed", candidate_angles)

        # Keep emitting the old corridor while it is still observable. If it is
        # temporarily absent, fail closed: retain pending evidence for possible
        # switch confirmation but do not emit an unconfirmed challenger label.
        relation = current.relation if current is not None else None
        mode = "hold_pending" if current is not None else "pending_no_current"
        return self._result(relation, mode, candidate_angles)

    def update(
        self,
        road_mask: np.ndarray,
        geometry: RectilinearGeometry,
        pose: KartPose | None,
        relation_config: KartRoadRelationConfig | None = None,
    ) -> TemporalKartRoadResult:
        if pose is None or pose.heading_angle_deg is None:
            return self.miss()
        candidates = road_relation_candidates(
            road_mask,
            geometry,
            pose,
            relation_config,
        )
        return self.update_candidates(candidates)
