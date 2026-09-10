#!/usr/bin/env python3
"""Evaluate a replay artifact against native-FPS action labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.train.replay_evaluator import replay_points
from karting_agent.train.sequence_evaluator import (
    ReleaseSegment,
    Transition,
    evaluate_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Replay Runtime output with recorded action labels."
    )
    parser.add_argument("replay", type=Path)
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    parser.add_argument(
        "--timeline",
        choices=("target", "observation", "both"),
        default="both",
        help=(
            "target evaluates future-action prediction semantics; observation "
            "evaluates when Runtime emitted the decision before hardware latency."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_label_data(video: str) -> tuple[list[Transition], list[ReleaseSegment]]:
    path = ROOT / "data" / "processed" / "labels" / f"{Path(video).stem}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"label file has no events: {path}")

    events: list[tuple[float, bool]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            raise ValueError(f"invalid event in {path}")
        events.append((float(raw_event["timestamp_ms"]), bool(raw_event["pressed"])))

    transitions = [
        Transition(video=video, timestamp_ms=timestamp_ms, pressed=pressed)
        for timestamp_ms, pressed in events[1:]
    ]
    releases = [
        ReleaseSegment(video=video, start_ms=start_ms, end_ms=end_ms)
        for (start_ms, pressed), (end_ms, next_pressed) in zip(events, events[1:])
        if not pressed and next_pressed
    ]
    return transitions, releases


def print_summary(name: str, summary: dict[str, object]) -> None:
    transition = summary["transition"]["all"]
    press = summary["onset_timing_ms"]["press"]
    release = summary["onset_timing_ms"]["release"]
    short = summary["release_segment_recall"]["short_100_300ms"]
    print(
        f"{name}: "
        f"transition_f1={transition['f1']:.3f}, "
        f"precision={transition['precision']:.3f}, "
        f"recall={transition['recall']:.3f}, "
        f"predicted={transition['predicted']}, "
        f"matched={transition['matched']}/{transition['ground_truth']}, "
        f"press_mean={press['mean_error_ms']:.1f}ms, "
        f"press_mae={press['mae_ms']:.1f}ms, "
        f"release_mean={release['mean_error_ms']:.1f}ms, "
        f"release_mae={release['mae_ms']:.1f}ms, "
        f"short_recall={short['recall']:.3f} "
        f"({short['detected']}/{short['segments']})",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.tolerance_ms < 0:
        raise ValueError("--tolerance-ms must be >= 0")

    replay_path = args.replay.resolve()
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("replay root must be a mapping")
    replay = payload.get("replay")
    if not isinstance(replay, dict):
        raise ValueError("replay artifact must contain a replay mapping")
    raw_steps = replay.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("replay artifact has no steps")
    if not all(isinstance(step, dict) for step in raw_steps):
        raise ValueError("replay steps must be mappings")

    video = str(replay.get("video", ""))
    if not video:
        raise ValueError("replay artifact is missing video path")
    transitions, releases = load_label_data(video)

    timelines = (
        ("target", "observation") if args.timeline == "both" else (args.timeline,)
    )
    results: dict[str, object] = {}
    print(
        f"Replay: {replay_path}\n"
        f"Video: {video}\n"
        f"Tolerance: {args.tolerance_ms:.1f}ms",
        flush=True,
    )
    for timeline in timelines:
        points = replay_points(raw_steps, video=video, timeline=timeline)
        evaluation = evaluate_sequence(
            points,
            transitions,
            releases,
            threshold=0.5,
            tolerance_ms=args.tolerance_ms,
        )
        summary = evaluation.summary()
        results[timeline] = summary
        print_summary(timeline, summary)

    output = (
        args.output.resolve()
        if args.output is not None
        else replay_path.with_name(
            f"{replay_path.stem}.evaluation_t{args.tolerance_ms:g}.json"
        )
    )
    output.write_text(
        json.dumps(
            {
                "replay": str(replay_path),
                "video": video,
                "tolerance_ms": args.tolerance_ms,
                "timelines": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Evaluation: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
