#!/usr/bin/env python3
"""Compare two recorded live multi-horizon runs around one transition.

The analyzer aligns both runs by executed transition ordinal, compares the
recorded H100/H200/H300 trajectories on a common relative-time grid, and emits
a side-by-side video-frame montage. It never replays counterfactual control
after the runs diverge; all probabilities are the values actually observed in
each recorded closed loop.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.train.live_run_comparison import (  # noqa: E402
    aligned_snapshots,
    extract_transitions,
)


DEFAULT_OFFSETS_MS = (
    0.0,
    50.0,
    100.0,
    150.0,
    200.0,
    250.0,
    300.0,
    350.0,
    400.0,
    500.0,
    600.0,
    700.0,
    800.0,
    900.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align two recorded multi-horizon runs by transition ordinal and "
            "compare their local probability/visual trajectories."
        )
    )
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--reference-recording", type=Path, default=None)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--candidate-recording", type=Path, default=None)
    parser.add_argument(
        "--anchor-transition",
        type=int,
        default=5,
        help="1-based executed transition ordinal used as t=0.",
    )
    parser.add_argument(
        "--offsets-ms",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_OFFSETS_MS),
        help="Comma-separated relative offsets sampled around the anchor.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tile-width", type=int, default=180)
    return parser.parse_args()


def _load_run(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run root must be a mapping: {path}")
    runtime = payload.get("runtime")
    recording = payload.get("recording")
    steps = payload.get("steps")
    if not isinstance(runtime, dict):
        raise ValueError(f"run has no runtime metadata: {path}")
    if not isinstance(recording, dict):
        raise ValueError(f"run has no recording metadata: {path}")
    if not bool(recording.get("frame_mapping_valid", False)):
        raise ValueError(f"run recording frame mapping is invalid: {path}")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"run contains no runtime steps: {path}")
    return payload


def _resolve_recording(
    run: dict[str, object],
    override: Path | None,
) -> Path:
    if override is not None:
        return override.resolve()
    recording = run["recording"]
    assert isinstance(recording, dict)
    raw_path = recording.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("recording path is missing from run metadata")
    return Path(raw_path).resolve()


def _parse_offsets(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--offsets-ms must contain at least one value")
    if tuple(sorted(values)) != values:
        raise ValueError("--offsets-ms must be sorted")
    if len(set(values)) != len(values):
        raise ValueError("--offsets-ms must be unique")
    return values


def _decode_frames(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"recording not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open recording: {path}")

    frames: dict[int, np.ndarray] = {}
    max_index = max(indices)
    try:
        index = 0
        while index <= max_index:
            ok, frame = capture.read()
            if not ok:
                break
            if index in indices:
                frames[index] = frame
            index += 1
    finally:
        capture.release()

    missing = sorted(indices - set(frames))
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise RuntimeError(
            f"{path.name} is missing {len(missing)} requested frame(s): {preview}"
        )
    return frames


def _runtime_horizons(run: dict[str, object]) -> tuple[float, ...]:
    runtime = run["runtime"]
    assert isinstance(runtime, dict)
    raw = runtime.get("prediction_horizons_ms")
    if not isinstance(raw, list) or not raw:
        raise ValueError("prediction_horizons_ms missing from run")
    return tuple(float(value) for value in raw)


def _control_horizon(run: dict[str, object]) -> float:
    runtime = run["runtime"]
    assert isinstance(runtime, dict)
    raw = runtime.get("prediction_horizon_ms")
    if raw is None:
        raise ValueError("prediction_horizon_ms missing from run")
    return float(raw)


def _transition_table(
    reference_transitions,
    candidate_transitions,
    *,
    anchor_ordinal: int,
) -> list[dict[str, object]]:
    count = min(
        len(reference_transitions),
        len(candidate_transitions),
        anchor_ordinal + 3,
    )
    rows: list[dict[str, object]] = []
    ref_anchor = reference_transitions[anchor_ordinal - 1].timestamp_ms
    cand_anchor = candidate_transitions[anchor_ordinal - 1].timestamp_ms

    for index in range(count):
        ref = reference_transitions[index]
        cand = candidate_transitions[index]
        rows.append(
            {
                "ordinal": index + 1,
                "reference_action": ref.action,
                "candidate_action": cand.action,
                "reference_relative_ms": ref.timestamp_ms - ref_anchor,
                "candidate_relative_ms": cand.timestamp_ms - cand_anchor,
                "relative_delta_ms": (
                    cand.timestamp_ms
                    - cand_anchor
                    - (ref.timestamp_ms - ref_anchor)
                ),
                "reference_reason": ref.reason,
                "candidate_reason": cand.reason,
                "reference_frame": ref.source_frame,
                "candidate_frame": cand.source_frame,
            }
        )
    return rows


def _snapshot_row(
    label: str,
    snapshot: dict[str, object],
    horizons: tuple[float, ...],
) -> dict[str, object]:
    row: dict[str, object] = {
        "run": label,
        "requested_offset_ms": snapshot["requested_offset_ms"],
        "actual_offset_ms": snapshot["actual_offset_ms"],
        "source_frame": snapshot["source_frame"],
        "action": snapshot["action"],
        "pressed": snapshot["pressed"],
        "reason": snapshot["reason"],
    }
    probabilities = snapshot["probabilities"]
    assert isinstance(probabilities, list)
    for horizon, probability in zip(horizons, probabilities):
        row[f"h{horizon:g}"] = probability
    return row


def _resize_tile(frame: np.ndarray, width: int) -> np.ndarray:
    height = int(round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _draw_tile_label(
    frame: np.ndarray,
    *,
    run_label: str,
    snapshot: dict[str, object],
    horizons: tuple[float, ...],
) -> np.ndarray:
    result = frame.copy()
    lines = [
        (
            f"{run_label} {float(snapshot['actual_offset_ms']):+.0f}ms "
            f"f{int(snapshot['source_frame'])}"
        ),
        (
            f"{snapshot['action']} "
            f"{'P' if bool(snapshot['pressed']) else 'R'} "
            f"{snapshot['reason']}"
        ),
    ]
    probabilities = snapshot["probabilities"]
    assert isinstance(probabilities, list)
    lines.append(
        " ".join(
            f"h{horizon:g}={float(probability):.2f}"
            for horizon, probability in zip(horizons, probabilities)
        )
    )
    for line_index, line in enumerate(lines):
        y = 18 + line_index * 18
        cv2.putText(
            result,
            line,
            (5, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            line,
            (5, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return result


def _build_montage(
    *,
    reference_frames: dict[int, np.ndarray],
    candidate_frames: dict[int, np.ndarray],
    reference_snapshots: list[dict[str, object]],
    candidate_snapshots: list[dict[str, object]],
    horizons: tuple[float, ...],
    tile_width: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for run_label, frames, snapshots in (
        ("REF", reference_frames, reference_snapshots),
        ("CUR", candidate_frames, candidate_snapshots),
    ):
        tiles: list[np.ndarray] = []
        for snapshot in snapshots:
            source_frame = int(snapshot["source_frame"])
            tile = _resize_tile(frames[source_frame], tile_width)
            tiles.append(
                _draw_tile_label(
                    tile,
                    run_label=run_label,
                    snapshot=snapshot,
                    horizons=horizons,
                )
            )
        rows.append(cv2.hconcat(tiles))
    return cv2.vconcat(rows)


def main() -> int:
    args = parse_args()
    if args.anchor_transition < 1:
        raise ValueError("--anchor-transition must be >= 1")
    if args.tile_width < 80:
        raise ValueError("--tile-width must be >= 80")
    offsets = _parse_offsets(args.offsets_ms)

    reference_path = args.reference_run.resolve()
    candidate_path = args.candidate_run.resolve()
    reference = _load_run(reference_path)
    candidate = _load_run(candidate_path)

    reference_horizons = _runtime_horizons(reference)
    candidate_horizons = _runtime_horizons(candidate)
    if reference_horizons != candidate_horizons:
        raise ValueError(
            "run horizon layouts differ: "
            f"{reference_horizons} vs {candidate_horizons}"
        )
    if _control_horizon(reference) != _control_horizon(candidate):
        raise ValueError(
            "run control horizons differ: "
            f"{_control_horizon(reference):g} vs "
            f"{_control_horizon(candidate):g}"
        )

    reference_steps = reference["steps"]
    candidate_steps = candidate["steps"]
    assert isinstance(reference_steps, list)
    assert isinstance(candidate_steps, list)

    reference_transitions = extract_transitions(reference_steps)
    candidate_transitions = extract_transitions(candidate_steps)
    if len(reference_transitions) < args.anchor_transition:
        raise ValueError("reference run does not contain anchor transition")
    if len(candidate_transitions) < args.anchor_transition:
        raise ValueError("candidate run does not contain anchor transition")

    ref_anchor = reference_transitions[args.anchor_transition - 1]
    cand_anchor = candidate_transitions[args.anchor_transition - 1]
    if ref_anchor.action != cand_anchor.action:
        raise ValueError(
            "anchor actions differ: "
            f"reference={ref_anchor.action} candidate={cand_anchor.action}"
        )

    reference_snapshots = aligned_snapshots(
        reference_steps,
        anchor_timestamp_ms=ref_anchor.timestamp_ms,
        offsets_ms=offsets,
    )
    candidate_snapshots = aligned_snapshots(
        candidate_steps,
        anchor_timestamp_ms=cand_anchor.timestamp_ms,
        offsets_ms=offsets,
    )
    transition_rows = _transition_table(
        reference_transitions,
        candidate_transitions,
        anchor_ordinal=args.anchor_transition,
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else candidate_path.parent
        / (
            f"compare_{reference_path.stem}_vs_{candidate_path.stem}"
            f"_transition{args.anchor_transition}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_recording = _resolve_recording(
        reference,
        args.reference_recording,
    )
    candidate_recording = _resolve_recording(
        candidate,
        args.candidate_recording,
    )
    reference_indices = {
        int(snapshot["source_frame"]) for snapshot in reference_snapshots
    }
    candidate_indices = {
        int(snapshot["source_frame"]) for snapshot in candidate_snapshots
    }
    reference_frames = _decode_frames(reference_recording, reference_indices)
    candidate_frames = _decode_frames(candidate_recording, candidate_indices)

    montage = _build_montage(
        reference_frames=reference_frames,
        candidate_frames=candidate_frames,
        reference_snapshots=reference_snapshots,
        candidate_snapshots=candidate_snapshots,
        horizons=reference_horizons,
        tile_width=args.tile_width,
    )
    montage_path = output_dir / "aligned_montage.jpg"
    if not cv2.imwrite(str(montage_path), montage):
        raise RuntimeError(f"failed to write montage: {montage_path}")

    csv_path = output_dir / "aligned_probabilities.csv"
    csv_rows = [
        _snapshot_row("reference", snapshot, reference_horizons)
        for snapshot in reference_snapshots
    ] + [
        _snapshot_row("candidate", snapshot, candidate_horizons)
        for snapshot in candidate_snapshots
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = {
        "reference_run": str(reference_path),
        "candidate_run": str(candidate_path),
        "reference_recording": str(reference_recording),
        "candidate_recording": str(candidate_recording),
        "horizons_ms": list(reference_horizons),
        "control_horizon_ms": _control_horizon(reference),
        "anchor_transition": args.anchor_transition,
        "anchor_action": ref_anchor.action,
        "reference_anchor": {
            "timestamp_ms": ref_anchor.timestamp_ms,
            "source_frame": ref_anchor.source_frame,
            "reason": ref_anchor.reason,
        },
        "candidate_anchor": {
            "timestamp_ms": cand_anchor.timestamp_ms,
            "source_frame": cand_anchor.source_frame,
            "reason": cand_anchor.reason,
        },
        "transition_alignment": transition_rows,
        "reference_snapshots": reference_snapshots,
        "candidate_snapshots": candidate_snapshots,
    }
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"reference={reference_path.name} candidate={candidate_path.name} "
        f"control_horizon={_control_horizon(reference):g}ms",
        flush=True,
    )
    print(
        f"anchor transition #{args.anchor_transition}: {ref_anchor.action} "
        f"ref=f{ref_anchor.source_frame}@{ref_anchor.timestamp_ms:.1f}ms "
        f"candidate=f{cand_anchor.source_frame}@{cand_anchor.timestamp_ms:.1f}ms",
        flush=True,
    )
    print("transition alignment relative to anchor:", flush=True)
    for row in transition_rows:
        print(
            f"  #{int(row['ordinal']):02d} "
            f"{row['reference_action']:<7} "
            f"ref={float(row['reference_relative_ms']):+8.1f}ms "
            f"cur={float(row['candidate_relative_ms']):+8.1f}ms "
            f"delta={float(row['relative_delta_ms']):+7.1f}ms",
            flush=True,
        )

    print("aligned probability snapshots:", flush=True)
    header = " offset | " + " ".join(
        f"h{horizon:g}".rjust(8) for horizon in reference_horizons
    )
    print(header + " | " + " ".join(
        f"h{horizon:g}".rjust(8) for horizon in reference_horizons
    ), flush=True)
    print(" " * 8 + "reference".center(8 * len(reference_horizons))
          + "   candidate", flush=True)
    for ref_snapshot, cand_snapshot in zip(
        reference_snapshots,
        candidate_snapshots,
    ):
        ref_probs = ref_snapshot["probabilities"]
        cand_probs = cand_snapshot["probabilities"]
        assert isinstance(ref_probs, list)
        assert isinstance(cand_probs, list)
        ref_values = " ".join(f"{float(value):8.3f}" for value in ref_probs)
        cand_values = " ".join(f"{float(value):8.3f}" for value in cand_probs)
        print(
            f"{float(ref_snapshot['requested_offset_ms']):+7.0f} | "
            f"{ref_values} | {cand_values}",
            flush=True,
        )

    next_index = args.anchor_transition
    if (
        next_index < len(reference_transitions)
        and next_index < len(candidate_transitions)
    ):
        ref_next = reference_transitions[next_index]
        cand_next = candidate_transitions[next_index]
        print(
            "next transition after anchor: "
            f"reference={ref_next.action}@"
            f"{ref_next.timestamp_ms - ref_anchor.timestamp_ms:.1f}ms "
            f"candidate={cand_next.action}@"
            f"{cand_next.timestamp_ms - cand_anchor.timestamp_ms:.1f}ms "
            f"delay_delta="
            f"{(cand_next.timestamp_ms - cand_anchor.timestamp_ms) - (ref_next.timestamp_ms - ref_anchor.timestamp_ms):+.1f}ms",
            flush=True,
        )

    print(f"JSON: {json_path}", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"Montage: {montage_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
