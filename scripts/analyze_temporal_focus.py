#!/usr/bin/env python3
"""Probe whether a temporal transition policy actually depends on motion/history order."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_model_focus import (  # noqa: E402
    load_diagnostic_runner,
    load_run,
    pre_action_pressed,
    predict_probability,
    read_video_frames,
    selected_steps,
)
from karting_agent.vision.preprocess import prepare_frame, stack_prepared_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the original temporal stack with order/history-preserving "
            "counterfactuals. This is a diagnostic probe, not causal attribution."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--source-frame",
        type=int,
        action="append",
        default=None,
        help=(
            "Analyze a specific latest temporal source-frame index. May be "
            "repeated. By default all state-changing steps are analyzed."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def predict(runner, frames: list[np.ndarray], *, current_pressed: bool) -> float:
    inputs = stack_prepared_frames(frames, runner.spec.preprocess_config)
    return predict_probability(
        runner,
        inputs,
        current_pressed=current_pressed,
    )


def temporal_probe(
    runner,
    prepared_frames: list[np.ndarray],
    *,
    current_pressed: bool,
) -> dict[str, object]:
    if len(prepared_frames) < 2:
        raise ValueError("temporal probe requires at least two frames")

    baseline = predict(
        runner,
        prepared_frames,
        current_pressed=current_pressed,
    )
    latest = prepared_frames[-1]
    oldest = prepared_frames[0]

    variants = {
        "repeat_latest": [latest for _ in prepared_frames],
        "repeat_oldest": [oldest for _ in prepared_frames],
        "reverse": list(reversed(prepared_frames)),
    }
    variant_results: dict[str, dict[str, float | bool]] = {}
    for name, frames in variants.items():
        probability = predict(
            runner,
            frames,
            current_pressed=current_pressed,
        )
        variant_results[name] = {
            "probability": probability,
            "delta_from_baseline": baseline - probability,
            "binary_flip_at_0_5": (baseline >= 0.5) != (probability >= 0.5),
        }

    slice_replacement: list[dict[str, object]] = []
    for index in range(len(prepared_frames)):
        replacement_index = index - 1 if index > 0 else 1
        frames = list(prepared_frames)
        frames[index] = prepared_frames[replacement_index]
        probability = predict(
            runner,
            frames,
            current_pressed=current_pressed,
        )
        slice_replacement.append(
            {
                "index": index,
                "offset_ms": runner.spec.frame_offsets_ms[index],
                "replacement_index": replacement_index,
                "replacement_offset_ms": runner.spec.frame_offsets_ms[replacement_index],
                "probability": probability,
                "delta_from_baseline": baseline - probability,
                "binary_flip_at_0_5": (baseline >= 0.5) != (probability >= 0.5),
            }
        )

    return {
        "baseline_probability": baseline,
        "variants": variant_results,
        "slice_replacement": slice_replacement,
    }


def summarize(values: list[float]) -> dict[str, float]:
    absolute = [abs(value) for value in values]
    if not values:
        return {
            "mean_delta": 0.0,
            "mean_abs_delta": 0.0,
            "median_abs_delta": 0.0,
            "max_abs_delta": 0.0,
        }
    return {
        "mean_delta": float(statistics.fmean(values)),
        "mean_abs_delta": float(statistics.fmean(absolute)),
        "median_abs_delta": float(statistics.median(absolute)),
        "max_abs_delta": float(max(absolute)),
    }


def aggregate_report(steps: list[dict[str, object]]) -> dict[str, object]:
    variant_deltas: dict[str, list[float]] = defaultdict(list)
    variant_flips: dict[str, int] = defaultdict(int)
    slice_deltas: dict[float, list[float]] = defaultdict(list)
    slice_flips: dict[float, int] = defaultdict(int)

    for step in steps:
        probe = step["temporal_probe"]
        for name, variant in probe["variants"].items():
            variant_deltas[name].append(float(variant["delta_from_baseline"]))
            variant_flips[name] += int(bool(variant["binary_flip_at_0_5"]))
        for item in probe["slice_replacement"]:
            offset = float(item["offset_ms"])
            slice_deltas[offset].append(float(item["delta_from_baseline"]))
            slice_flips[offset] += int(bool(item["binary_flip_at_0_5"]))

    count = len(steps)
    variants = {
        name: {
            **summarize(values),
            "binary_flip_rate_at_0_5": variant_flips[name] / count if count else 0.0,
        }
        for name, values in variant_deltas.items()
    }
    slices = {
        f"{offset:g}ms": {
            "offset_ms": offset,
            **summarize(values),
            "binary_flip_rate_at_0_5": slice_flips[offset] / count if count else 0.0,
        }
        for offset, values in sorted(slice_deltas.items())
    }
    return {"steps": count, "variants": variants, "slice_replacement": slices}


def main() -> int:
    args = parse_args()
    runner = load_diagnostic_runner(
        args.model,
        metadata_path=args.metadata,
        device=args.device,
    )
    run_path = args.run_json.resolve()
    run = load_run(run_path)
    steps = selected_steps(run, args.source_frame)

    all_indices: list[int] = []
    for step in steps:
        indices = step.get("input_frame_indices")
        if not isinstance(indices, list) or len(indices) != runner.spec.frame_stack:
            raise ValueError("RuntimeStep frame indices do not match model frame_stack")
        all_indices.extend(int(index) for index in indices)
    source_frames = read_video_frames(args.video.resolve(), all_indices)

    report_steps: list[dict[str, object]] = []
    for step in steps:
        indices = [int(index) for index in step["input_frame_indices"]]
        prepared = [
            prepare_frame(source_frames[index], runner.spec.preprocess_config)
            for index in indices
        ]
        current_pressed = pre_action_pressed(step)
        probe = temporal_probe(
            runner,
            prepared,
            current_pressed=current_pressed,
        )
        saved_probability = float(step["probability"])
        baseline = float(probe["baseline_probability"])
        mismatch = abs(saved_probability - baseline)
        action = str(step.get("action", "?"))
        report_steps.append(
            {
                "source_frame": indices[-1],
                "input_frame_indices": indices,
                "frame_offsets_ms": list(runner.spec.frame_offsets_ms),
                "observation_timestamp_ms": float(step["observation_timestamp_ms"]),
                "action": action,
                "current_pressed": current_pressed,
                "pressed_after_action": bool(step["pressed"]),
                "saved_probability": saved_probability,
                "recomputed_probability": baseline,
                "absolute_probability_mismatch": mismatch,
                "temporal_probe": probe,
            }
        )
        repeat_latest = probe["variants"]["repeat_latest"]
        reverse = probe["variants"]["reverse"]
        state_label = "P" if current_pressed else "R"
        print(
            f"src={indices[-1]} state={state_label} action={action:<7} "
            f"p={baseline:.3f} diff={mismatch:.3f} "
            f"repeat_latest={repeat_latest['probability']:.3f} "
            f"reverse={reverse['probability']:.3f}",
            flush=True,
        )

    summary = aggregate_report(report_steps)
    output = (
        args.output.resolve()
        if args.output is not None
        else ROOT / "artifacts" / "analysis" / f"temporal_{run_path.stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "model": str(Path(args.model).resolve()),
                "metadata": str(Path(args.metadata).resolve()) if args.metadata else None,
                "video": str(args.video.resolve()),
                "run_json": str(run_path),
                "frame_offsets_ms": list(runner.spec.frame_offsets_ms),
                "summary": summary,
                "steps": report_steps,
                "note": (
                    "Counterfactual temporal probes measure sensitivity to history/order. "
                    "They preserve the runtime control state for state-conditioned models. "
                    "They are diagnostic perturbations, not causal feature attribution."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nAggregate temporal sensitivity:", flush=True)
    for name, item in summary["variants"].items():
        print(
            f"  {name}: mean_abs_delta={item['mean_abs_delta']:.3f}, "
            f"flip_rate={item['binary_flip_rate_at_0_5']:.3f}",
            flush=True,
        )
    print("Adjacent-frame replacement by temporal slice:", flush=True)
    for name, item in summary["slice_replacement"].items():
        print(
            f"  {name}: mean_abs_delta={item['mean_abs_delta']:.3f}, "
            f"flip_rate={item['binary_flip_rate_at_0_5']:.3f}",
            flush=True,
        )
    print(f"Temporal report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
