#!/usr/bin/env python3
"""Compare temporal motion signatures in aligned live model input stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v5_live_horizons as live  # noqa: E402

from karting_agent.model.state_conditioned_runner import (  # noqa: E402
    StateConditionedModelRunner,
)
from karting_agent.train.live_run_comparison import (  # noqa: E402
    extract_transitions,
    nearest_step,
)
from karting_agent.vision.preprocess import stack_frames  # noqa: E402


DEFAULT_OFFSETS_MS = (0.0, 50.0, 100.0, 150.0, 200.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare order-sensitive temporal delta signatures between two "
            "aligned live runs."
        )
    )
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--reference-recording", type=Path, default=None)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--candidate-recording", type=Path, default=None)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--anchor-transition", type=int, default=5)
    parser.add_argument(
        "--offsets-ms",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_OFFSETS_MS),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _parse_offsets(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--offsets-ms must contain at least one value")
    if tuple(sorted(values)) != values:
        raise ValueError("--offsets-ms must be sorted")
    if len(set(values)) != len(values):
        raise ValueError("--offsets-ms must be unique")
    return values


def _steps(run: dict[str, object]) -> list[dict[str, object]]:
    raw = run.get("steps")
    if not isinstance(raw, list) or not raw:
        raise ValueError("run contains no runtime steps")
    if not all(isinstance(item, dict) for item in raw):
        raise ValueError("runtime steps must be mappings")
    return raw  # type: ignore[return-value]


def _required_indices(steps: list[dict[str, object]]) -> set[int]:
    indices: set[int] = set()
    for step in steps:
        raw = step.get("input_frame_indices")
        if not isinstance(raw, list) or not raw:
            raise ValueError("runtime step is missing input_frame_indices")
        indices.update(int(value) for value in raw)
    return indices


def _stack_input(
    step: dict[str, object],
    frames: dict[int, np.ndarray],
    model: StateConditionedModelRunner,
) -> np.ndarray:
    raw_indices = step.get("input_frame_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("runtime step is missing input_frame_indices")
    indices = tuple(int(value) for value in raw_indices)
    return stack_frames(
        [frames[index] for index in indices],
        model.spec.preprocess_config,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.astype(np.float64, copy=False).reshape(-1)
    right_flat = right.astype(np.float64, copy=False).reshape(-1)
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left_flat, right_flat) / (left_norm * right_norm))


def _temporal_signature(
    stacked: np.ndarray,
    *,
    frame_stack: int,
) -> tuple[np.ndarray, tuple[float, ...]]:
    if stacked.shape[0] != 3 * frame_stack:
        raise ValueError("stacked input channel count does not match frame_stack")
    frames = stacked.reshape(frame_stack, 3, *stacked.shape[1:])
    deltas = np.diff(frames.astype(np.float64, copy=False), axis=0)
    energies = tuple(
        float(np.mean(np.abs(delta)))
        for delta in deltas
    )
    return deltas, energies


def _source_frame(step: dict[str, object]) -> int:
    raw = step.get("input_frame_indices")
    if not isinstance(raw, list) or not raw:
        raise ValueError("runtime step is missing input_frame_indices")
    return int(raw[-1])


def _probability(step: dict[str, object], index: int) -> float:
    raw = step.get("probabilities")
    if not isinstance(raw, list) or index >= len(raw):
        raise ValueError("runtime step is missing probabilities")
    return float(raw[index])


def main() -> int:
    args = parse_args()
    if args.anchor_transition < 1:
        raise ValueError("--anchor-transition must be >= 1")
    offsets = _parse_offsets(args.offsets_ms)

    reference_path = args.reference_run.resolve()
    candidate_path = args.candidate_run.resolve()
    reference = live._load_run(reference_path)
    candidate = live._load_run(candidate_path)
    reference_steps = _steps(reference)
    candidate_steps = _steps(candidate)

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
            f"anchor actions differ: {ref_anchor.action} vs {cand_anchor.action}"
        )

    model = StateConditionedModelRunner(
        args.model.resolve(),
        metadata_path=args.metadata.resolve(),
        device="cpu",
    )
    horizons = tuple(float(value) for value in model.spec.prediction_horizons_ms)
    control_index = horizons.index(float(model.spec.control_horizon_ms))

    reference_recording = live._recording_path(
        reference,
        args.reference_recording,
    )
    candidate_recording = live._recording_path(
        candidate,
        args.candidate_recording,
    )
    reference_frames = live._decode_required_frames(
        reference_recording,
        _required_indices(reference_steps),
    )
    candidate_frames = live._decode_required_frames(
        candidate_recording,
        _required_indices(candidate_steps),
    )

    rows: list[dict[str, Any]] = []
    for requested_offset_ms in offsets:
        ref_index, ref_step = nearest_step(
            reference_steps,
            ref_anchor.timestamp_ms + requested_offset_ms,
        )
        cand_index, cand_step = nearest_step(
            candidate_steps,
            cand_anchor.timestamp_ms + requested_offset_ms,
        )

        ref_input = _stack_input(ref_step, reference_frames, model)
        cand_input = _stack_input(cand_step, candidate_frames, model)
        ref_deltas, ref_energies = _temporal_signature(
            ref_input,
            frame_stack=model.spec.frame_stack,
        )
        cand_deltas, cand_energies = _temporal_signature(
            cand_input,
            frame_stack=model.spec.frame_stack,
        )
        ref_current = ref_input[-3:]
        cand_current = cand_input[-3:]

        row = {
            "requested_offset_ms": requested_offset_ms,
            "reference_source_frame": _source_frame(ref_step),
            "candidate_source_frame": _source_frame(cand_step),
            "stack_cosine": _cosine(ref_input, cand_input),
            "current_frame_cosine": _cosine(ref_current, cand_current),
            "temporal_delta_cosine": _cosine(ref_deltas, cand_deltas),
            "reference_delta_energy": ref_energies,
            "candidate_delta_energy": cand_energies,
            "reference_mean_delta_energy": float(np.mean(ref_energies)),
            "candidate_mean_delta_energy": float(np.mean(cand_energies)),
            "delta_energy_ratio": (
                float(np.mean(cand_energies)) / float(np.mean(ref_energies))
                if float(np.mean(ref_energies)) > 1e-12
                else None
            ),
            "reference_h200": _probability(ref_step, control_index),
            "candidate_h200": _probability(cand_step, control_index),
        }
        rows.append(row)

    print(
        f"reference={reference_path.name} candidate={candidate_path.name} "
        f"anchor=#{args.anchor_transition} {ref_anchor.action}",
        flush=True,
    )
    print(
        " offset | stack_cos current_cos delta_cos | "
        "ref_motion cur_motion ratio | ref_h200 cur_h200",
        flush=True,
    )
    for row in rows:
        ratio = row["delta_energy_ratio"]
        ratio_text = "NA" if ratio is None else f"{float(ratio):5.2f}"
        print(
            f"{float(row['requested_offset_ms']):+7.0f} | "
            f"{float(row['stack_cosine']):9.4f} "
            f"{float(row['current_frame_cosine']):11.4f} "
            f"{float(row['temporal_delta_cosine']):9.4f} | "
            f"{float(row['reference_mean_delta_energy']):10.4f} "
            f"{float(row['candidate_mean_delta_energy']):10.4f} "
            f"{ratio_text:>5} | "
            f"{float(row['reference_h200']):8.3f} "
            f"{float(row['candidate_h200']):8.3f}",
            flush=True,
        )

    print("per-gap temporal delta energy:", flush=True)
    for row in rows:
        ref_values = row["reference_delta_energy"]
        cand_values = row["candidate_delta_energy"]
        assert isinstance(ref_values, tuple)
        assert isinstance(cand_values, tuple)
        print(
            f"  {float(row['requested_offset_ms']):+7.0f}ms "
            f"ref=[{','.join(f'{float(value):.4f}' for value in ref_values)}] "
            f"cur=[{','.join(f'{float(value):.4f}' for value in cand_values)}]",
            flush=True,
        )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else candidate_path.with_name(
            candidate_path.stem + "_temporal_motion_divergence.json"
        )
    )
    output_path.write_text(
        json.dumps(
            {
                "reference_run": str(reference_path),
                "candidate_run": str(candidate_path),
                "model": str(args.model.resolve()),
                "metadata": str(args.metadata.resolve()),
                "anchor_transition": args.anchor_transition,
                "anchor_action": ref_anchor.action,
                "horizons_ms": list(horizons),
                "control_horizon_ms": model.spec.control_horizon_ms,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
