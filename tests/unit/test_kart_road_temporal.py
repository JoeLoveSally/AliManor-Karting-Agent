from __future__ import annotations

import pytest

from karting_agent.train.kart_pose_pseudo_labels import KartRoadRelation
from karting_agent.train.kart_road_temporal import (
    RoadRelationCandidate,
    TemporalKartRoadTracker,
    TemporalRoadTrackerConfig,
)


def _candidate(angle: float, score: float) -> RoadRelationCandidate:
    relation = KartRoadRelation(
        road_angle_deg=angle,
        road_support_fraction=0.5,
        heading_error_deg=0.0,
        abs_heading_error_deg=0.0,
        local_road_center_x=0.0,
        local_road_center_y=0.0,
        local_road_width_px=100.0,
        lateral_offset_px=0.0,
        lateral_offset_norm=0.0,
        inside_road=True,
        valid_cross_sections=2,
        cross_section_count=2,
        confidence=0.5,
    )
    return RoadRelationCandidate(relation=relation, base_score=score)


def _tracker() -> TemporalKartRoadTracker:
    return TemporalKartRoadTracker(
        TemporalRoadTrackerConfig(
            continuity_weight=0.75,
            switch_angle_deg=25.0,
            switch_confirm_samples=3,
            switch_score_margin=0.05,
            reset_gap_samples=3,
            stable_angle_alpha=0.25,
        )
    )


def test_tracker_ignores_one_frame_crossing_challenger() -> None:
    tracker = _tracker()
    first = tracker.update_candidates([_candidate(130.0, 0.20)])
    assert first.mode == "initialize"

    challenged = tracker.update_candidates(
        [_candidate(130.0, 0.30), _candidate(25.0, 0.10)]
    )
    assert challenged.mode == "hold_pending"
    assert challenged.relation is not None
    assert challenged.relation.road_angle_deg == pytest.approx(130.0)
    assert challenged.pending_count == 1

    recovered = tracker.update_candidates([_candidate(130.0, 0.20)])
    assert recovered.mode == "stable"
    assert recovered.pending_count == 0
    assert recovered.relation is not None
    assert recovered.relation.road_angle_deg == pytest.approx(130.0)


def test_tracker_switches_after_confirmed_challenger_sequence() -> None:
    tracker = _tracker()
    tracker.update_candidates([_candidate(130.0, 0.20)])

    for expected_count in (1, 2):
        result = tracker.update_candidates(
            [_candidate(130.0, 0.35), _candidate(25.0, 0.10)]
        )
        assert result.mode == "hold_pending"
        assert result.pending_count == expected_count
        assert result.relation is not None
        assert result.relation.road_angle_deg == pytest.approx(130.0)

    switched = tracker.update_candidates(
        [_candidate(130.0, 0.35), _candidate(25.0, 0.10)]
    )
    assert switched.mode == "switch_confirmed"
    assert switched.pending_count == 0
    assert switched.stable_angle_deg == pytest.approx(25.0)
    assert switched.relation is not None
    assert switched.relation.road_angle_deg == pytest.approx(25.0)


def test_tracker_requires_score_margin_before_starting_switch() -> None:
    tracker = _tracker()
    tracker.update_candidates([_candidate(130.0, 0.20)])

    result = tracker.update_candidates(
        [_candidate(130.0, 0.20), _candidate(25.0, 0.17)]
    )
    assert result.mode == "hold_margin"
    assert result.pending_count == 0
    assert result.relation is not None
    assert result.relation.road_angle_deg == pytest.approx(130.0)


def test_tracker_resets_after_missing_samples() -> None:
    tracker = _tracker()
    tracker.update_candidates([_candidate(130.0, 0.20)])

    tracker.miss()
    tracker.miss()
    result = tracker.miss()
    assert result.mode == "missing"
    assert result.stable_angle_deg is None

    restarted = tracker.update_candidates([_candidate(25.0, 0.20)])
    assert restarted.mode == "initialize"
    assert restarted.stable_angle_deg == pytest.approx(25.0)
