#!/usr/bin/env python3
"""Diagnose V4-C4 action-only replay without changing its control policy.

This script uses the same checkpoint, preprocessing, decoder, transition matcher,
and expert-visual replay as evaluate_event_time_policy.py. Ground truth is read
only for diagnostics, never for decisions after replay initialization.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_event_time_policy import (  # noqa: E402
    _artifact_dir,
    _load_label_data,
    _load_yaml,
    _points_and_reasons,
    _select_device,
)
from karting_agent.model.event_time import build_event_time_model  # noqa: E402
from karting_agent.model.temporal_delta import (  # noqa: E402
    transform_temporal_input_torch,
    validate_temporal_input_representation,
)
from karting_agent.runtime.event_time_policy import (  # noqa: E402
    EventTimePolicyConfig,
    EventTimePolicyDecoder,
)
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    SequencePoint,
    evaluate_sequence,
    transitions_from_points,
)
from karting_agent.train.state_conditioned_dataset import (  # noqa: E402
    StateConditionedVideoDataset,
    load_v3_samples,
)
from karting_agent.train.trainer import (  # noqa: E402
    load_train_loop_config,
    load_video_split,
    partition_samples,
)
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the unmatched transitions and pending origins of V4-C4."
    )
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "train_v4c4_event_time.yaml",
    )
    parser.add_argument(
        "--samples", type=Path,
        default=ROOT / "data/processed/v4c3_temporal_delta_dense/samples.jsonl",
    )
    parser.add_argument(
        "--labels-dir", type=Path,
        default=ROOT / "data/processed/v4c3_temporal_delta_dense/labels",
    )
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--action-threshold", type=float, default=0.5)
    parser.add_argument("--min-state-hold-ms", type=float, default=100.0)
    parser.add_argument("--tolerance-ms", type=float, default=100.0)
    parser.add_argument("--window-ms", type=float, default=110.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _replay_with_trace(
    *, video, samples, indices, probabilities, config,
) -> tuple[list[SequencePoint], list[dict[str, object]], list[dict[str, object]]]:
    ordered = sorted(indices, key=lambda i: samples[i].input_timestamps_ms[-1])
    first = samples[ordered[0]]
    state = bool(first.current_pressed)
    points = [
        SequencePoint(
            video=video,
            timestamp_ms=float(first.input_timestamps_ms[-1]) - 1e-3,
            probability=float(state),
        )
    ]
    decoder = EventTimePolicyDecoder(config)
    observations: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    pending_origin: dict[str, object] | None = None

    for index in ordered:
        timestamp = float(samples[index].input_timestamps_ms[-1])
        probability = float(probabilities[index])
        observed = {
            "timestamp_ms": timestamp,
            "action_probability": probability,
            "expert_pressed_at_observation": bool(samples[index].current_pressed),
        }
        observations.append(observed)

        pending = decoder.execute_pending_if_due(
            timestamp_ms=timestamp, current_pressed=state,
        )
        if pending is not None:
            state = pending.state_after
            points.append(
                SequencePoint(
                    video=video,
                    timestamp_ms=float(pending.due_at_ms),
                    probability=float(state),
                )
            )
            events.append({
                "video": video,
                "timestamp_ms": float(pending.due_at_ms),
                "pressed": bool(state),
                "reason": "pending_execute",
                "observed_at_execution_check_ms": timestamp,
                "action_probability_at_execution_check": probability,
                "pending_origin": pending_origin,
            })
            pending_origin = None

        decision = decoder.update(
            timestamp_ms=timestamp,
            current_pressed=state,
            current_action_probability=probability,
            event_class=config.no_event_class,
        )
        if decision.reason == "action_hold":
            pending_origin = {
                "timestamp_ms": timestamp,
                "action_probability": probability,
                "due_at_ms": decision.pending_due_ms,
                "desired_pressed": bool(decision.desired_pressed),
            }
        elif decision.reason != "hold":
            pending_origin = None

        if decision.switch:
            state = bool(decision.state_after)
            points.append(
                SequencePoint(
                    video=video, timestamp_ms=timestamp, probability=float(state),
                )
            )
            events.append({
                "video": video,
                "timestamp_ms": timestamp,
                "pressed": state,
                "reason": decision.reason,
                "action_probability_at_decision": probability,
            })

    last_ms = float(samples[ordered[-1]].input_timestamps_ms[-1])
    if points[-1].timestamp_ms < last_ms - 1e-6:
        points.append(
            SequencePoint(video=video, timestamp_ms=last_ms, probability=float(state))
        )
    return points, events, observations


def _nearest_transition(events, timestamp: float, pressed: bool):
    same = [event for event in events if event.pressed == pressed]
    return min(same, key=lambda event: abs(event.timestamp_ms - timestamp)) if same else None


def _nearby_observations(observations, timestamp, window):
    return [
        observation for observation in observations
        if abs(float(observation["timestamp_ms"]) - timestamp) <= window
    ]


def _audit_video(
    *, video, samples, indices, probabilities, config, label_data,
    tolerance_ms: float, window_ms: float,
):
    points, events, observations = _replay_with_trace(
        video=video, samples=samples, indices=indices,
        probabilities=probabilities, config=config,
    )
    # Detect any diagnostic-replay drift relative to the existing evaluator.
    reference_points, reference_reasons = _points_and_reasons(
        video=video,
        samples=samples,
        indices=indices,
        action_probabilities=probabilities,
        event_classes=np.full(len(samples), config.no_event_class, dtype=np.int64),
        decoder_config=config,
        use_event_time=False,
    )
    if points != reference_points:
        raise AssertionError(f"diagnostic replay differs from production replay: {video}")
    predicted = transitions_from_points(points, threshold=0.5)
    if len(predicted) != len(events):
        raise AssertionError(f"transition trace width mismatch: {video}")
    for predicted_event, row in zip(predicted, events):
        if (predicted_event.timestamp_ms != row["timestamp_ms"]
                or predicted_event.pressed != row["pressed"]):
            raise AssertionError(f"transition trace content mismatch: {video}")

    expected, releases = label_data
    evaluation = evaluate_sequence(
        points, expected, releases, threshold=0.5, tolerance_ms=tolerance_ms,
    )
    matched_predicted = {match.predicted for match in evaluation.matches}
    matched_expected = {match.expected for match in evaluation.matches}
    expected_by_predicted = {match.predicted: match for match in evaluation.matches}
    annotated: list[dict[str, object]] = []
    for i, (transition, row) in enumerate(zip(predicted, events)):
        timestamp = transition.timestamp_ms
        match = expected_by_predicted.get(transition)
        nearest = _nearest_transition(
            evaluation.expected_transitions, timestamp, transition.pressed,
        )
        previous_ms = predicted[i - 1].timestamp_ms if i else None
        next_ms = predicted[i + 1].timestamp_ms if i + 1 < len(predicted) else None
        history = _nearby_observations(observations, timestamp, window_ms)
        annotated.append({
            **row,
            "matched": transition in matched_predicted,
            "matched_gt_ms": None if match is None else match.expected.timestamp_ms,
            "timing_error_ms": None if match is None else match.error_ms,
            "nearest_same_direction_gt_ms": (
                None if nearest is None else nearest.timestamp_ms
            ),
            "nearest_same_direction_gt_delta_ms": (
                None if nearest is None else timestamp - nearest.timestamp_ms
            ),
            "previous_pred_interval_ms": (
                None if previous_ms is None else timestamp - previous_ms
            ),
            "next_pred_interval_ms": (
                None if next_ms is None else next_ms - timestamp
            ),
            "nearby_observations": history,
        })
    misses = []
    for event in evaluation.expected_transitions:
        if event in matched_expected:
            continue
        nearest = _nearest_transition(predicted, event.timestamp_ms, event.pressed)
        misses.append({
            "video": video,
            "timestamp_ms": event.timestamp_ms,
            "pressed": event.pressed,
            "nearest_predicted_ms": (
                None if nearest is None else nearest.timestamp_ms
            ),
            "nearest_predicted_delta_ms": (
                None if nearest is None else nearest.timestamp_ms - event.timestamp_ms
            ),
            "nearby_observations": _nearby_observations(
                observations, event.timestamp_ms, window_ms,
            ),
        })
    summary = evaluation.summary()["transition"]["all"]
    return annotated, misses, summary, dict(reference_reasons)


def main() -> int:
    args = parse_args()
    if not 0.0 < args.action_threshold < 1.0:
        raise ValueError("action threshold must be in (0,1)")
    if args.min_state_hold_ms < 0 or args.tolerance_ms < 0 or args.window_ms < 0:
        raise ValueError("hold, tolerance, and diagnostic window must be >= 0")

    import torch
    from torch.utils.data import DataLoader

    config_path = args.config.resolve()
    raw = _load_yaml(config_path)
    loop = load_train_loop_config(config_path)
    preprocess = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)
    artifact = _artifact_dir(raw)
    model_path = args.model.resolve() if args.model else artifact / "model.pt"
    metadata_path = args.metadata.resolve() if args.metadata else artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_family") != "event_time_v4c4":
        raise ValueError("metadata is not a V4-C4 event-time artifact")
    device = _select_device(torch, args.device)
    model = build_event_time_model(
        str(metadata["architecture"]),
        frame_stack=int(metadata["frame_stack"]),
        pretrained=False,
        event_time_classes=int(metadata["event_time_classes"]),
        visual_feature_dim=int(metadata["visual_feature_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    representation = validate_temporal_input_representation(
        str(metadata["input_representation"])
    )
    all_samples = load_v3_samples(args.samples.resolve())
    samples = partition_samples(all_samples, load_video_split(config_path))[args.split]
    dataset = StateConditionedVideoDataset(
        samples, project_root=ROOT, preprocess_config=preprocess,
        cache_root=cache_root, require_cache=args.require_cache,
        counterfactual_states=False,
    )
    num_workers = loop.num_workers if args.num_workers is None else args.num_workers
    loader = DataLoader(
        dataset, batch_size=loop.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    batches = []
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = transform_temporal_input_torch(
                    batch["input"].to(device=device, dtype=torch.float32),
                    frame_stack=int(metadata["frame_stack"]), representation=representation,
                )
                action_logits, _, _, _, _ = model(inputs)
                batches.append(torch.sigmoid(action_logits).cpu().numpy())
    finally:
        dataset.close()
    probabilities = np.concatenate(batches, axis=0)
    if probabilities.shape != (len(samples),):
        raise RuntimeError("action prediction shape mismatch")

    config = EventTimePolicyConfig(
        bin_ms=float(metadata["event_time_bin_ms"]),
        event_bins=int(metadata["event_time_classes"]) - 1,
        no_event_class=int(metadata["no_event_class"]),
        action_threshold=args.action_threshold,
        min_state_hold_ms=args.min_state_hold_ms,
    )
    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_video[sample.video].append(index)
    annotated = []
    misses = []
    reasons: Counter[str] = Counter()
    matched_count = gt_count = predicted_count = 0
    for video, indices in indices_by_video.items():
        video_events, video_misses, metrics, video_reasons = _audit_video(
            video=video, samples=samples, indices=indices,
            probabilities=probabilities, config=config,
            label_data=_load_label_data(args.labels_dir.resolve(), video),
            tolerance_ms=args.tolerance_ms, window_ms=args.window_ms,
        )
        annotated.extend(video_events)
        misses.extend(video_misses)
        reasons.update(video_reasons)
        matched_count += int(metrics["matched"])
        gt_count += int(metrics["ground_truth"])
        predicted_count += int(metrics["predicted"])

    if matched_count + len(misses) != gt_count:
        raise AssertionError("unmatched GT accounting mismatch")
    if matched_count + sum(not row["matched"] for row in annotated) != predicted_count:
        raise AssertionError("unmatched prediction accounting mismatch")
    output = args.output.resolve() if args.output else (
        artifact / "evaluation" / f"{args.split}_action_only_transition_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "model": str(model_path),
        "action_threshold": args.action_threshold,
        "min_state_hold_ms": args.min_state_hold_ms,
        "tolerance_ms": args.tolerance_ms,
        "matched": matched_count,
        "ground_truth": gt_count,
        "predicted": predicted_count,
        "reasons": dict(reasons),
        "predictions": annotated,
        "unmatched_gt": misses,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    false_predictions = [row for row in annotated if not row["matched"]]
    pending_predictions = [
        row for row in annotated if row["reason"] == "pending_execute"
    ]
    large_errors = sorted(
        [row for row in annotated if row["timing_error_ms"] is not None],
        key=lambda row: abs(float(row["timing_error_ms"])), reverse=True,
    )[:5]
    print(
        f"split={args.split} matched={matched_count}/{gt_count} "
        f"pred={predicted_count} unmatched_pred={len(false_predictions)} "
        f"unmatched_gt={len(misses)} pending_exec={len(pending_predictions)}"
    )
    for label, rows in (
        ("UNMATCHED PREDICTIONS", false_predictions),
        ("UNMATCHED GT", misses),
        ("PENDING EXECUTIONS", pending_predictions),
        ("LARGEST MATCHED TIMING ERRORS", large_errors),
    ):
        print(f"\n{label} ({len(rows)}):")
        for row in rows:
            nearby = row["nearby_observations"]
            compact = [(round(float(obs["timestamp_ms"]), 1),
                        round(float(obs["action_probability"]), 3))
                       for obs in nearby]
            print(
                f"  {Path(str(row['video'])).stem} "
                f"t={float(row['timestamp_ms']):.1f} "
                f"state={'PRESS' if row['pressed'] else 'RELEASE'} "
                f"reason={row.get('reason', 'ground_truth')} "
                f"error={row.get('timing_error_ms')} "
                f"nearest_gt_delta={row.get('nearest_same_direction_gt_delta_ms')} "
                f"nearest_pred_delta={row.get('nearest_predicted_delta_ms')} "
                f"prev_dt={row.get('previous_pred_interval_ms')} "
                f"next_dt={row.get('next_pred_interval_ms')} "
                f"origin={row.get('pending_origin')} "
                f"probabilities={compact}"
            )
    print(f"\nOutput: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
