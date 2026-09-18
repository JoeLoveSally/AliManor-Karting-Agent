#!/usr/bin/env python3
"""Compare model visual features between aligned live runs before control diverges.

This analyzer uses the deployed V4 model's visual encoder itself. For each
candidate step after an aligned transition, it compares:

1. the reference step at the same relative time;
2. the most similar reference visual feature while both runs still share the
   same physical action state.

This distinguishes a simple visual-phase shift from a genuine representation
or decision-boundary divergence.
"""

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


DEFAULT_OFFSETS_MS = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deployed visual features between two live runs around an "
            "aligned transition."
        )
    )
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--reference-recording", type=Path, default=None)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--candidate-recording", type=Path, default=None)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--anchor-transition", type=int, default=5)
    parser.add_argument(
        "--offsets-ms",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_OFFSETS_MS),
    )
    parser.add_argument(
        "--search-start-ms",
        type=float,
        default=0.0,
        help="Reference search range start relative to the anchor.",
    )
    parser.add_argument(
        "--search-end-ms",
        type=float,
        default=None,
        help=(
            "Reference search range end relative to the anchor. By default it "
            "stops just before the next reference transition."
        ),
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


def _step_input(
    step: dict[str, object],
    frames: dict[int, np.ndarray],
    model: StateConditionedModelRunner,
) -> np.ndarray:
    raw_indices = step.get("input_frame_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("runtime step is missing input_frame_indices")
    frame_indices = tuple(int(value) for value in raw_indices)
    return stack_frames(
        [frames[index] for index in frame_indices],
        model.spec.preprocess_config,
    )


def _visual_feature(
    model: StateConditionedModelRunner,
    inputs: np.ndarray,
) -> np.ndarray:
    tensor = model._tensor(inputs)
    with model._torch.inference_mode():
        feature = model.model.visual_encoder(tensor)[0]
    values = feature.detach().cpu().numpy().astype(np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise ValueError("visual feature has zero norm")
    return values / norm


def _input_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.astype(np.float64, copy=False).reshape(-1)
    right_flat = right.astype(np.float64, copy=False).reshape(-1)
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(np.dot(left_flat, right_flat) / (left_norm * right_norm))


def _probabilities(step: dict[str, object]) -> tuple[float, ...]:
    raw = step.get("probabilities")
    if not isinstance(raw, list) or not raw:
        raise ValueError("runtime step is missing probabilities")
    return tuple(float(value) for value in raw)


def _source_frame(step: dict[str, object]) -> int:
    raw = step.get("input_frame_indices")
    if not isinstance(raw, list) or not raw:
        raise ValueError("runtime step is missing input_frame_indices")
    return int(raw[-1])


def _required_indices(steps: list[dict[str, object]]) -> set[int]:
    indices: set[int] = set()
    for step in steps:
        raw = step.get("input_frame_indices")
        if not isinstance(raw, list) or not raw:
            raise ValueError("runtime step is missing input_frame_indices")
        indices.update(int(value) for value in raw)
    return indices


def _feature_bank(
    *,
    model: StateConditionedModelRunner,
    steps: list[dict[str, object]],
    frames: dict[int, np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    features: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    for step in steps:
        stacked = _step_input(step, frames, model)
        inputs.append(stacked)
        features.append(_visual_feature(model, stacked))
    return features, inputs


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
    if len(reference_transitions) < args.anchor_transition + 1:
        raise ValueError("reference run needs an anchor and following transition")
    if len(candidate_transitions) < args.anchor_transition:
        raise ValueError("candidate run does not contain anchor transition")

    ref_anchor = reference_transitions[args.anchor_transition - 1]
    cand_anchor = candidate_transitions[args.anchor_transition - 1]
    if ref_anchor.action != cand_anchor.action:
        raise ValueError(
            f"anchor actions differ: {ref_anchor.action} vs {cand_anchor.action}"
        )

    default_search_end_ms = (
        reference_transitions[args.anchor_transition].timestamp_ms
        - ref_anchor.timestamp_ms
        - 1e-3
    )
    search_end_ms = (
        default_search_end_ms
        if args.search_end_ms is None
        else float(args.search_end_ms)
    )
    if search_end_ms <= args.search_start_ms:
        raise ValueError("visual search range must be non-empty")

    model = StateConditionedModelRunner(
        args.model.resolve(),
        metadata_path=args.metadata.resolve(),
        device=args.device,
    )

    reference_recording = live._recording_path(
        reference,
        args.reference_recording,
    )
    candidate_recording = live._recording_path(
        candidate,
        args.candidate_recording,
    )

    reference_indices = _required_indices(reference_steps)
    candidate_indices = _required_indices(candidate_steps)
    reference_frames = live._decode_required_frames(
        reference_recording,
        reference_indices,
    )
    candidate_frames = live._decode_required_frames(
        candidate_recording,
        candidate_indices,
    )

    reference_features, reference_inputs = _feature_bank(
        model=model,
        steps=reference_steps,
        frames=reference_frames,
    )
    candidate_features, candidate_inputs = _feature_bank(
        model=model,
        steps=candidate_steps,
        frames=candidate_frames,
    )

    search_indices = [
        index
        for index, step in enumerate(reference_steps)
        if args.search_start_ms
        <= float(step["observation_timestamp_ms"]) - ref_anchor.timestamp_ms
        <= search_end_ms
    ]
    if not search_indices:
        raise ValueError("no reference steps fall inside visual search range")

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

        cand_feature = candidate_features[cand_index]
        same_feature_cosine = float(
            np.dot(reference_features[ref_index], cand_feature)
        )
        same_input_cosine = _input_cosine(
            reference_inputs[ref_index],
            candidate_inputs[cand_index],
        )

        best_index = max(
            search_indices,
            key=lambda index: float(
                np.dot(reference_features[index], cand_feature)
            ),
        )
        best_step = reference_steps[best_index]
        best_feature_cosine = float(
            np.dot(reference_features[best_index], cand_feature)
        )
        best_input_cosine = _input_cosine(
            reference_inputs[best_index],
            candidate_inputs[cand_index],
        )

        ref_relative_ms = (
            float(ref_step["observation_timestamp_ms"]) - ref_anchor.timestamp_ms
        )
        cand_relative_ms = (
            float(cand_step["observation_timestamp_ms"]) - cand_anchor.timestamp_ms
        )
        best_relative_ms = (
            float(best_step["observation_timestamp_ms"]) - ref_anchor.timestamp_ms
        )

        rows.append(
            {
                "requested_offset_ms": requested_offset_ms,
                "reference_relative_ms": ref_relative_ms,
                "candidate_relative_ms": cand_relative_ms,
                "reference_source_frame": _source_frame(ref_step),
                "candidate_source_frame": _source_frame(cand_step),
                "same_time_feature_cosine": same_feature_cosine,
                "same_time_input_cosine": same_input_cosine,
                "reference_probabilities": _probabilities(ref_step),
                "candidate_probabilities": _probabilities(cand_step),
                "best_reference_relative_ms": best_relative_ms,
                "best_reference_source_frame": _source_frame(best_step),
                "best_feature_cosine": best_feature_cosine,
                "best_input_cosine": best_input_cosine,
                "visual_phase_shift_ms": (
                    best_relative_ms - cand_relative_ms
                ),
                "best_reference_probabilities": _probabilities(best_step),
            }
        )

    print(
        f"reference={reference_path.name} candidate={candidate_path.name} "
        f"anchor=#{args.anchor_transition} {ref_anchor.action}",
        flush=True,
    )
    print(
        f"shared-state search range: "
        f"{args.search_start_ms:.1f}..{search_end_ms:.1f}ms",
        flush=True,
    )
    print(
        " offset | same_feat same_input | best_ref phase_shift best_feat "
        "best_input | ref_h200 cur_h200 best_h200",
        flush=True,
    )
    horizons = tuple(float(value) for value in model.spec.prediction_horizons_ms)
    control_index = horizons.index(float(model.spec.control_horizon_ms))
    for row in rows:
        ref_probs = row["reference_probabilities"]
        cand_probs = row["candidate_probabilities"]
        best_probs = row["best_reference_probabilities"]
        assert isinstance(ref_probs, tuple)
        assert isinstance(cand_probs, tuple)
        assert isinstance(best_probs, tuple)
        print(
            f"{float(row['requested_offset_ms']):+7.0f} | "
            f"{float(row['same_time_feature_cosine']):9.4f} "
            f"{float(row['same_time_input_cosine']):10.4f} | "
            f"{float(row['best_reference_relative_ms']):+8.1f} "
            f"{float(row['visual_phase_shift_ms']):+11.1f} "
            f"{float(row['best_feature_cosine']):9.4f} "
            f"{float(row['best_input_cosine']):10.4f} | "
            f"{ref_probs[control_index]:8.3f} "
            f"{cand_probs[control_index]:8.3f} "
            f"{best_probs[control_index]:9.3f}",
            flush=True,
        )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else candidate_path.with_name(
            candidate_path.stem + "_visual_phase_divergence.json"
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
                "reference_anchor_timestamp_ms": ref_anchor.timestamp_ms,
                "candidate_anchor_timestamp_ms": cand_anchor.timestamp_ms,
                "search_start_ms": args.search_start_ms,
                "search_end_ms": search_end_ms,
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
