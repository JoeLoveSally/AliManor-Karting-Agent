"""Read-only diagnostics for frozen tiny-policy predictions on expert videos.

Event matching MUST come from the repository's sequence evaluator: this module
only projects matched/missed/spurious events onto a timeline for investigation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

MODES = ("rgb", "rgb_hsv")
COLORS = {"rgb": (40, 140, 255), "rgb_hsv": (30, 180, 80)}  # BGR


def event_diagnostics(evaluation) -> dict:
    """Describe exact evaluator matches without rematching by proximity."""
    matched_expected = {match.expected: match for match in evaluation.matches}
    matched_predicted = {match.predicted for match in evaluation.matches}
    expected = [
        {
            "time_ms": float(event.timestamp_ms),
            "pressed": bool(event.pressed),
            "status": "matched" if event in matched_expected else "missed",
            "predicted_ms": (
                float(matched_expected[event].predicted.timestamp_ms)
                if event in matched_expected else None
            ),
            "error_ms": (
                float(matched_expected[event].error_ms)
                if event in matched_expected else None
            ),
        }
        for event in evaluation.expected_transitions
    ]
    predicted = [
        {"time_ms": float(event.timestamp_ms), "pressed": bool(event.pressed),
         "status": "matched" if event in matched_predicted else "false_positive"}
        for event in evaluation.predicted_transitions
    ]
    if len(evaluation.release_segments) != len(evaluation.release_detected):
        raise ValueError("release segment/detection length mismatch")
    short = []
    for segment, detected in zip(evaluation.release_segments, evaluation.release_detected):
        if not 100.0 <= segment.duration_ms <= 300.0:
            continue
        start_ok = segment.start_transition in matched_expected
        end_ok = segment.end_transition in matched_expected
        if bool(detected) != (start_ok and end_ok):
            raise ValueError("release detection inconsistent with exact event matches")
        short.append({
            "start_ms": float(segment.start_ms),
            "end_ms": float(segment.end_ms),
            "duration_ms": float(segment.duration_ms),
            "detected": bool(detected),
            "release_onset_matched": start_ok,
            "press_onset_matched": end_ok,
        })
    return {"expected": expected, "predicted": predicted, "short_releases": short}


def build_focus_cases(mode_diagnostics: Mapping[str, dict], *, maximum: int = 16) -> list[dict]:
    """Prioritize any missed short RELEASE, then unmatched predicted switches."""
    if maximum < 1:
        raise ValueError("maximum must be >= 1")
    segments = {}
    for mode, report in mode_diagnostics.items():
        if mode not in MODES:
            raise ValueError("unknown model mode")
        for item in report["short_releases"]:
            key = (item["start_ms"], item["end_ms"])
            record = segments.setdefault(key, {"kind": "short_release", "time_ms": item["start_ms"],
                                               "end_ms": item["end_ms"], "missing_models": []})
            if not item["detected"]:
                record["missing_models"].append(mode)
    missing = [r for r in segments.values() if r["missing_models"]]
    missing.sort(key=lambda r: (-len(r["missing_models"]), r["time_ms"]))
    extras = []
    for mode, report in mode_diagnostics.items():
        for item in report["predicted"]:
            if item["status"] == "false_positive":
                extras.append({"kind": "extra_switch", "time_ms": item["time_ms"],
                               "model": mode, "pressed": item["pressed"]})
    extras.sort(key=lambda r: (r["time_ms"], r["model"]))
    return (missing + extras)[:maximum]


def probabilities_at_points(points_by_mode: Mapping[str, Sequence]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Enforce the same target timestamps across visual models."""
    if set(points_by_mode) != set(MODES):
        raise ValueError("exactly rgb and rgb_hsv are required")
    base = points_by_mode["rgb"]
    if not base:
        raise ValueError("empty visual timeline")
    times = np.asarray([float(p.timestamp_ms) for p in base], dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("target timestamps must increase strictly")
    values = {}
    for mode in MODES:
        pts = points_by_mode[mode]
        their_times = [float(p.timestamp_ms) for p in pts]
        if (len(pts) != len(times) or
                not np.allclose(their_times, times, atol=1e-5, rtol=0)):
            raise ValueError("RGB variants have mismatched target timestamps")
        video = base[0].video
        if any(p.video != video for p in pts):
            raise ValueError("mixed video timeline")
        values[mode] = np.asarray([float(p.probability) for p in pts], np.float64)
        if not np.all(np.isfinite(values[mode])) or np.any((values[mode] < 0) | (values[mode] > 1)):
            raise ValueError("invalid predicted probability")
    return times, values


def draw_probability_chart(
    times_ms: np.ndarray, probabilities: Mapping[str, np.ndarray],
    gt_events: Sequence, *, start_ms: float, end_ms: float,
    width: int = 1600, height: int = 290,
) -> np.ndarray:
    """Pure OpenCV plot: original timestamps, threshold and expert state."""
    if not start_ms < end_ms or width < 480 or height < 190:
        raise ValueError("invalid chart extent")
    canvas = np.full((height, width, 3), 255, np.uint8)
    left, right, top, bottom = 75, width - 25, 32, height - 48
    cv2.rectangle(canvas, (left, top), (right, bottom), (180, 180, 180), 1)
    def tx(t):
        return int(round(left + (float(t)-start_ms)/(end_ms-start_ms)*(right-left)))
    def py(p):
        return int(round(bottom - float(p)*(bottom-top)))
    cv2.line(canvas, (left, py(0.5)), (right, py(0.5)), (170,170,170), 1)
    cv2.putText(canvas,"p=0.5",(6,py(.5)+5),cv2.FONT_HERSHEY_SIMPLEX,.42,(75,75,75),1)
    # Ground-truth state is an actual step sequence, never interpolated.
    events = sorted(gt_events, key=lambda e: e.timestamp_ms)
    state = bool(events[0].pressed) if events else False
    for item in events:
        if item.timestamp_ms <= start_ms:
            state = bool(item.pressed)
    prev_t = start_ms
    for item in events:
        t = float(item.timestamp_ms)
        if not start_ms < t < end_ms:
            continue
        cv2.line(canvas,(tx(prev_t),py(.93 if state else .07)),
                 (tx(t),py(.93 if state else .07)), (120,120,120), 2)
        cv2.line(canvas,(tx(t),py(.93)),(tx(t),py(.07)),(220,220,220),1)
        state = bool(item.pressed)
        prev_t = t
    cv2.line(canvas,(tx(prev_t),py(.93 if state else .07)),
             (tx(end_ms),py(.93 if state else .07)),(120,120,120),2)
    for mode in MODES:
        vals = probabilities[mode]
        indices = np.flatnonzero((times_ms >= start_ms) & (times_ms <= end_ms))
        if indices.size > 1:
            pts = np.asarray([(tx(times_ms[i]),py(vals[i])) for i in indices], np.int32)
            cv2.polylines(canvas,[pts],False,COLORS[mode],2,cv2.LINE_AA)
    cv2.putText(canvas,f"{start_ms/1000:.2f}-{end_ms/1000:.2f}s",
                (left,bottom+24),cv2.FONT_HERSHEY_SIMPLEX,.48,(40,40,40),1)
    cv2.putText(canvas,"RGB",(left,19),cv2.FONT_HERSHEY_SIMPLEX,.5,COLORS["rgb"],2)
    cv2.putText(canvas,"RGB+HSV",(left+100,19),cv2.FONT_HERSHEY_SIMPLEX,.5,COLORS["rgb_hsv"],2)
    cv2.putText(canvas,"GT step (grey)",(left+260,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(90,90,90),1)
    return canvas


def render_timeline(times_ms, values, events, diagnostics, *, width=2200):
    """A full-video timeline with all missed GT and false positives flagged."""
    plot = draw_probability_chart(
        times_ms, values, events, start_ms=float(times_ms[0]),
        end_ms=float(times_ms[-1]), width=width, height=430,
    )
    start,end = float(times_ms[0]),float(times_ms[-1])
    def x(t):
        return int(round(75+(t-start)/(end-start)*(width-100)))
    for mode in MODES:
        y = 332 if mode == "rgb" else 357
        cv2.putText(plot,mode,(5,y+4),cv2.FONT_HERSHEY_SIMPLEX,.45,COLORS[mode],1)
        for item in diagnostics[mode]["expected"]:
            if item["status"] == "missed":
                cv2.circle(plot,(x(item["time_ms"]),y),4,(0,0,220),-1)
        for item in diagnostics[mode]["predicted"]:
            if item["status"] == "false_positive":
                cv2.line(plot,(x(item["time_ms"])-4,y-5),
                         (x(item["time_ms"])+4,y+5),(220,70,10),2)
                cv2.line(plot,(x(item["time_ms"])-4,y+5),
                         (x(item["time_ms"])+4,y-5),(220,70,10),2)
    cv2.putText(plot,"red dot = missed GT; blue X = extra predicted switch",
                (75,402),cv2.FONT_HERSHEY_SIMPLEX,.53,(75,75,75),1)
    return plot


def render_focus_contact(times_ms, values, events, cases, *, limit=16):
    """Compact zoom-in of short-release misses and extra switches."""
    shown = list(cases[:limit])
    if not shown:
        return np.full((220,1200,3),255,np.uint8)
    card_width = 1200
    rows = []
    for case in shown:
        center = case["time_ms"] if case["kind"] == "extra_switch" else (
            case["time_ms"]+case["end_ms"])/2
        start,end = center-600,center+600
        chart = draw_probability_chart(times_ms,values,events,
                                       start_ms=start,end_ms=end,width=card_width,height=240)
        if case["kind"] == "short_release":
            label = (f"SHORT RELEASE {case['time_ms']:.0f}-{case['end_ms']:.0f}ms "
                     f"miss: {','.join(case['missing_models'])}")
        else:
            label = (f"EXTRA {case['model']} {'PRESS' if case['pressed'] else 'RELEASE'} "
                     f"at {case['time_ms']:.0f}ms")
        card = np.full((275,card_width,3),255,np.uint8)
        cv2.putText(card,label[:110],(10,23),cv2.FONT_HERSHEY_SIMPLEX,.57,(20,20,20),2)
        card[35:] = chart
        rows.append(card)
    return np.concatenate(rows,axis=0)
