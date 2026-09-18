#!/usr/bin/env python3
"""Compare fixed-horizon H200 replay with anticipation-assisted scheduling."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.model.kart_relative_supervised import (  # noqa: E402
    build_kart_relative_model,
)
from karting_agent.model.temporal_delta import (  # noqa: E402
    transform_temporal_input_torch,
    validate_temporal_input_representation,
)
from karting_agent.runtime.multi_horizon_scheduler import (  # noqa: E402
    MultiHorizonSchedulerConfig,
    MultiHorizonSwitchScheduler,
)
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.labels.touch_marker import ActionEvent  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    ReleaseSegment,
    SequencePoint,
    Transition,
    combine_evaluations,
    evaluate_sequence,
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


@dataclass(frozen=True)
class ReplayDecision:
    video: str
    observation_timestamp_ms: float
    state_before: bool
    switched: bool
    state_after: bool
    probability: float
    reason: str
    probabilities: tuple[float, ...]
    pending_delay_ms: float | None = None
    pending_due_ms: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the fixed H200 controller with an H300-assisted pending "
            "scheduler on held-out expert visuals."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_v4c2_temporal_v2.yaml",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "v3" / "samples.jsonl",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "v3" / "labels",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--control-horizon-ms", type=float, default=200.0)
    parser.add_argument("--anticipation-horizon-ms", type=float, default=300.0)
    parser.add_argument("--pending-advance-ms", type=float, default=0.0)
    parser.add_argument(
        "--arm-pending-during-min-hold",
        action="store_true",
        help=(
            "Allow monotonic anticipation warnings to arm during the minimum state "
            "hold while clamping execution to the hold expiry."
        ),
    )
    parser.add_argument(
        "--execute-short-horizon-overdue",
        action="store_true",
        help=(
            "Treat a shorter-than-control horizon crossing with a low control "
            "probability as an overdue short correction once the state hold expires."
        ),
    )
    parser.add_argument(
        "--reserve-bounded-reversal-during-min-hold",
        action="store_true",
        help=(
            "Remember a near-horizon reversal pattern observed during minimum "
            "state hold and execute it when the hold expires."
        ),
    )
    parser.add_argument("--tolerance-ms", type=float, default=None)
    return parser.parse_args()


def load_yaml_mapping(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return raw


def select_device(torch, requested: str | None):
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def artifact_dir(raw: dict[str, object]) -> Path:
    artifact = raw.get("artifact", {})
    if not isinstance(artifact, dict):
        raise ValueError("artifact config must be a mapping")
    return ROOT / "artifacts" / "models" / str(artifact["name"])


def evaluation_tolerance(raw: dict[str, object], override: float | None) -> float:
    if override is not None:
        value = float(override)
    else:
        evaluation = raw.get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise ValueError("evaluation config must be a mapping")
        value = float(evaluation.get("transition_tolerance_ms", 100.0))
    if value < 0:
        raise ValueError("transition tolerance must be >= 0")
    return value


def load_label_data(
    labels_dir: Path,
    video: str,
) -> tuple[list[Transition], list[ReleaseSegment]]:
    path = labels_dir / f"{Path(video).stem}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"label file has no events: {path}")
    events = [
        ActionEvent(
            frame_index=int(raw["frame_index"]),
            timestamp_ms=float(raw["timestamp_ms"]),
            pressed=bool(raw["pressed"]),
        )
        for raw in raw_events
    ]
    transitions = [
        Transition(video=video, timestamp_ms=event.timestamp_ms, pressed=event.pressed)
        for event in events[1:]
    ]
    releases = [
        ReleaseSegment(video=video, start_ms=current.timestamp_ms, end_ms=nxt.timestamp_ms)
        for current, nxt in zip(events, events[1:])
        if not current.pressed and nxt.pressed
    ]
    return transitions, releases


def simulate_fixed_horizon(
    video: str,
    samples,
    sample_indices: list[int],
    switch_if_release: np.ndarray,
    switch_if_press: np.ndarray,
    *,
    horizon_index: int,
    threshold: float,
) -> list[ReplayDecision]:
    ordered = sorted(
        sample_indices,
        key=lambda index: float(samples[index].input_timestamps_ms[-1]),
    )
    if not ordered:
        return []
    initial = samples[ordered[0]].current_pressed
    if initial is None:
        raise ValueError("sample is missing current_pressed")
    state = bool(initial)
    decisions: list[ReplayDecision] = []
    for index in ordered:
        observation_ms = float(samples[index].input_timestamps_ms[-1])
        all_probabilities = switch_if_press[index] if state else switch_if_release[index]
        probability = float(all_probabilities[horizon_index])
        before = state
        switched = probability >= threshold
        if switched:
            state = not state
        decisions.append(
            ReplayDecision(
                video=video,
                observation_timestamp_ms=observation_ms,
                state_before=before,
                switched=switched,
                state_after=state,
                probability=probability,
                reason="primary" if switched else "hold",
                probabilities=tuple(float(value) for value in all_probabilities),
            )
        )
    return decisions


def simulate_fixed_horizon_min_hold(
    video: str,
    samples,
    sample_indices: list[int],
    switch_if_release: np.ndarray,
    switch_if_press: np.ndarray,
    *,
    horizon_index: int,
    threshold: float,
    min_state_hold_ms: float,
) -> list[ReplayDecision]:
    """Replay one horizon with only a deterministic minimum state hold."""

    ordered = sorted(
        sample_indices,
        key=lambda index: float(samples[index].input_timestamps_ms[-1]),
    )
    if not ordered:
        return []
    initial = samples[ordered[0]].current_pressed
    if initial is None:
        raise ValueError("sample is missing current_pressed")
    state = bool(initial)
    last_switch_ms: float | None = None
    decisions: list[ReplayDecision] = []

    for index in ordered:
        observation_ms = float(samples[index].input_timestamps_ms[-1])
        all_probabilities = (
            switch_if_press[index] if state else switch_if_release[index]
        )
        probability = float(all_probabilities[horizon_index])
        before = state
        inside_hold = (
            last_switch_ms is not None
            and observation_ms - last_switch_ms < min_state_hold_ms - 1e-6
        )
        switched = probability >= threshold and not inside_hold
        if switched:
            state = not state
            last_switch_ms = observation_ms
        decisions.append(
            ReplayDecision(
                video=video,
                observation_timestamp_ms=observation_ms,
                state_before=before,
                switched=switched,
                state_after=state,
                probability=probability,
                reason=(
                    "primary"
                    if switched
                    else ("min_hold" if inside_hold else "hold")
                ),
                probabilities=tuple(
                    float(value) for value in all_probabilities
                ),
            )
        )
    return decisions


def simulate_scheduler(
    video: str,
    samples,
    sample_indices: list[int],
    switch_if_release: np.ndarray,
    switch_if_press: np.ndarray,
    *,
    scheduler_config: MultiHorizonSchedulerConfig,
) -> list[ReplayDecision]:
    ordered = sorted(
        sample_indices,
        key=lambda index: float(samples[index].input_timestamps_ms[-1]),
    )
    if not ordered:
        return []
    initial = samples[ordered[0]].current_pressed
    if initial is None:
        raise ValueError("sample is missing current_pressed")
    state = bool(initial)
    scheduler = MultiHorizonSwitchScheduler(scheduler_config)
    control_index = scheduler_config.horizons_ms.index(
        scheduler_config.control_horizon_ms
    )
    decisions: list[ReplayDecision] = []
    for index in ordered:
        observation_ms = float(samples[index].input_timestamps_ms[-1])
        all_probabilities = switch_if_press[index] if state else switch_if_release[index]
        result = scheduler.update(
            timestamp_ms=observation_ms,
            probabilities=all_probabilities,
            current_pressed=state,
        )
        before = state
        if result.switch:
            state = not state
        decisions.append(
            ReplayDecision(
                video=video,
                observation_timestamp_ms=observation_ms,
                state_before=before,
                switched=result.switch,
                state_after=state,
                probability=float(all_probabilities[control_index]),
                reason=result.reason,
                probabilities=result.probabilities,
                pending_delay_ms=result.pending_delay_ms,
                pending_due_ms=result.pending_due_ms,
            )
        )
    return decisions


def points_from_decisions(
    decisions: list[ReplayDecision],
    *,
    shift_ms: float,
) -> list[SequencePoint]:
    if not decisions:
        return []
    first = decisions[0]
    points = [
        SequencePoint(
            video=first.video,
            timestamp_ms=first.observation_timestamp_ms + shift_ms - 1e-3,
            probability=1.0 if first.state_before else 0.0,
        )
    ]
    points.extend(
        SequencePoint(
            video=decision.video,
            timestamp_ms=decision.observation_timestamp_ms + shift_ms,
            probability=1.0 if decision.state_after else 0.0,
        )
        for decision in decisions
    )
    return points


def chatter_summary(decisions: list[ReplayDecision]) -> dict[str, int]:
    timestamps = np.asarray(
        [
            decision.observation_timestamp_ms
            for decision in decisions
            if decision.switched
        ],
        dtype=np.float64,
    )
    intervals = np.diff(timestamps)
    return {
        "transitions": int(timestamps.size),
        "lt_100ms": int((intervals < 100.0).sum()) if intervals.size else 0,
        "lt_200ms": int((intervals < 200.0).sum()) if intervals.size else 0,
    }


def replay_summary(
    decisions_by_video: dict[str, list[ReplayDecision]],
    label_data: dict[str, tuple[list[Transition], list[ReleaseSegment]]],
    *,
    control_horizon_ms: float,
    tolerance_ms: float,
) -> dict[str, object]:
    target_evaluations = []
    observation_evaluations = []
    chatter = {"transitions": 0, "lt_100ms": 0, "lt_200ms": 0}
    for video, decisions in decisions_by_video.items():
        transitions, releases = label_data[video]
        target_evaluations.append(
            evaluate_sequence(
                points_from_decisions(decisions, shift_ms=control_horizon_ms),
                transitions,
                releases,
                threshold=0.5,
                tolerance_ms=tolerance_ms,
            )
        )
        observation_evaluations.append(
            evaluate_sequence(
                points_from_decisions(decisions, shift_ms=0.0),
                transitions,
                releases,
                threshold=0.5,
                tolerance_ms=tolerance_ms,
            )
        )
        local = chatter_summary(decisions)
        for key in chatter:
            chatter[key] += local[key]
    return {
        "target_timeline": combine_evaluations(target_evaluations).summary(),
        "observation_timeline": combine_evaluations(observation_evaluations).summary(),
        "chatter": chatter,
    }


def scheduler_diagnostics(
    decisions_by_video: dict[str, list[ReplayDecision]],
) -> dict[str, object]:
    decisions = [decision for items in decisions_by_video.values() for decision in items]
    reasons = Counter(decision.reason for decision in decisions)
    delays = np.asarray(
        [
            decision.pending_delay_ms
            for decision in decisions
            if decision.reason == "pending_armed"
            and decision.pending_delay_ms is not None
        ],
        dtype=np.float64,
    )
    hold_reversal_delays = np.asarray(
        [
            decision.pending_delay_ms
            for decision in decisions
            if decision.reason == "hold_reversal_armed"
            and decision.pending_delay_ms is not None
        ],
        dtype=np.float64,
    )

    def summarize(values: np.ndarray) -> dict[str, float | int]:
        return {
            "count": int(values.size),
            "mean": float(values.mean()) if values.size else 0.0,
            "p50": float(np.percentile(values, 50)) if values.size else 0.0,
            "p95": float(np.percentile(values, 95)) if values.size else 0.0,
            "min": float(values.min()) if values.size else 0.0,
            "max": float(values.max()) if values.size else 0.0,
        }

    return {
        "reasons": dict(sorted(reasons.items())),
        "armed_delay_ms": summarize(delays),
        "hold_reversal_delay_ms": summarize(hold_reversal_delays),
    }


def hold_reversal_event_diagnostics(
    decisions_by_video: dict[str, list[ReplayDecision]],
    label_data: dict[str, tuple[list[Transition], list[ReleaseSegment]]],
    *,
    control_horizon_ms: float,
    tolerance_ms: float,
) -> list[dict[str, object]]:
    """Describe each bounded hold-reversal execution against nearby ground truth."""

    rows: list[dict[str, object]] = []
    for video, decisions in decisions_by_video.items():
        transitions, _ = label_data[video]
        armed: ReplayDecision | None = None
        for decision in decisions:
            if decision.reason == "hold_reversal_armed":
                armed = decision
                continue
            if decision.reason != "hold_reversal_execute":
                continue

            target_timestamp_ms = (
                decision.observation_timestamp_ms + control_horizon_ms
            )
            same_state = [
                transition
                for transition in transitions
                if transition.pressed == decision.state_after
            ]
            nearest = (
                min(
                    same_state,
                    key=lambda transition: abs(
                        transition.timestamp_ms - target_timestamp_ms
                    ),
                )
                if same_state
                else None
            )
            delta_ms = (
                target_timestamp_ms - nearest.timestamp_ms
                if nearest is not None
                else None
            )
            rows.append(
                {
                    "video": video,
                    "armed_at_ms": (
                        None
                        if armed is None
                        else armed.observation_timestamp_ms
                    ),
                    "executed_at_ms": decision.observation_timestamp_ms,
                    "target_timestamp_ms": target_timestamp_ms,
                    "state_after": decision.state_after,
                    "arm_probabilities": (
                        None if armed is None else armed.probabilities
                    ),
                    "execute_probabilities": decision.probabilities,
                    "nearest_gt_timestamp_ms": (
                        None if nearest is None else nearest.timestamp_ms
                    ),
                    "target_delta_ms": delta_ms,
                    "within_tolerance": (
                        False
                        if delta_ms is None
                        else abs(delta_ms) <= tolerance_ms
                    ),
                }
            )
            armed = None
    return rows


def print_replay_summary(name: str, summary: dict[str, object]) -> None:
    target = summary["target_timeline"]
    observation = summary["observation_timeline"]
    chatter = summary["chatter"]
    target_transition = target["transition"]["all"]
    target_short = target["release_segment_recall"]["short_100_300ms"]
    observation_transition = observation["transition"]["all"]
    press_timing = target["onset_timing_ms"]["press"]
    release_timing = target["onset_timing_ms"]["release"]
    print(
        f"{name}: target_f1={target_transition['f1']:.3f} "
        f"matched={target_transition['matched']}/{target_transition['ground_truth']} "
        f"pred={target_transition['predicted']} "
        f"short_recall={target_short['recall']:.3f} "
        f"press_mae={press_timing['mae_ms']:.1f}ms "
        f"release_mae={release_timing['mae_ms']:.1f}ms "
        f"obs_f1={observation_transition['f1']:.3f} "
        f"chatter_lt100={chatter['lt_100ms']} "
        f"lt200={chatter['lt_200ms']}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: python -m pip install -e ".[train]"'
        ) from exc

    config_path = args.config.resolve()
    raw = load_yaml_mapping(config_path)
    split_config = load_video_split(config_path)
    loop_config = load_train_loop_config(config_path)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)
    tolerance_ms = evaluation_tolerance(raw, args.tolerance_ms)
    artifact = artifact_dir(raw)
    model_path = args.model.resolve() if args.model else artifact / "model.pt"
    metadata_path = args.metadata.resolve() if args.metadata else artifact / "metadata.json"
    if args.output:
        output_path = args.output.resolve()
    else:
        suffix_parts: list[str] = []
        if abs(float(args.pending_advance_ms)) >= 1e-9:
            suffix_parts.append(f"advance_{float(args.pending_advance_ms):g}ms")
        if args.arm_pending_during_min_hold:
            suffix_parts.append("hold_arm")
        if args.execute_short_horizon_overdue:
            suffix_parts.append("short_overdue")
        if args.reserve_bounded_reversal_during_min_hold:
            suffix_parts.append("hold_reversal")
        suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)
        output_path = (
            artifact
            / "evaluation"
            / f"{args.split}_multi_horizon_scheduler{suffix}.json"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_family = str(metadata.get("model_family", ""))
    if model_family not in {
        "state_conditioned_kart_relative_v4c2",
        "state_conditioned_kart_relative_v4c3_delta",
    }:
        raise ValueError(
            "scheduler evaluator expects a v4-C2/C3 kart-relative artifact"
        )
    input_representation = validate_temporal_input_representation(
        str(metadata.get("input_representation", "raw_rgb_stack"))
    )
    horizons = tuple(float(value) for value in metadata["prediction_horizons_ms"])
    scheduler_config = MultiHorizonSchedulerConfig(
        horizons_ms=horizons,
        control_horizon_ms=float(args.control_horizon_ms),
        anticipation_horizon_ms=float(args.anticipation_horizon_ms),
        threshold=float(args.threshold),
        pending_advance_ms=float(args.pending_advance_ms),
        arm_pending_during_min_hold=bool(args.arm_pending_during_min_hold),
        execute_short_horizon_overdue=bool(args.execute_short_horizon_overdue),
        reserve_bounded_reversal_during_min_hold=bool(
            args.reserve_bounded_reversal_during_min_hold
        ),
    )
    scheduler_config.validate()
    control_index = horizons.index(scheduler_config.control_horizon_ms)

    model = build_kart_relative_model(
        str(metadata["architecture"]),
        frame_stack=int(metadata["frame_stack"]),
        pretrained=False,
        horizon_count=len(horizons),
        visual_feature_dim=int(metadata["visual_feature_dim"]),
        state_embedding_dim=int(metadata["state_embedding_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
    )
    device = select_device(torch, args.device)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    all_samples = load_v3_samples(args.samples.resolve())
    samples = partition_samples(all_samples, split_config)[args.split]
    dataset = StateConditionedVideoDataset(
        samples,
        project_root=ROOT,
        preprocess_config=preprocess_config,
        cache_root=cache_root,
        require_cache=args.require_cache,
        counterfactual_states=False,
    )
    num_workers = loop_config.num_workers if args.num_workers is None else args.num_workers
    loader_kwargs = {
        "batch_size": loop_config.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    release_batches: list[np.ndarray] = []
    press_batches: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["input"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                inputs = transform_temporal_input_torch(
                    inputs,
                    frame_stack=int(metadata["frame_stack"]),
                    representation=input_representation,
                )
                features = model.encode_visual(inputs)
                batch_size = int(inputs.shape[0])
                release_states = torch.zeros(batch_size, device=device, dtype=torch.long)
                press_states = torch.ones(batch_size, device=device, dtype=torch.long)
                release_logits = model.forward_from_features(
                    features, release_states
                )[0]
                press_logits = model.forward_from_features(features, press_states)[0]
                release_batches.append(torch.sigmoid(release_logits).cpu().numpy())
                press_batches.append(torch.sigmoid(press_logits).cpu().numpy())
    finally:
        dataset.close()

    switch_if_release = np.concatenate(release_batches, axis=0)
    switch_if_press = np.concatenate(press_batches, axis=0)
    expected_shape = (len(samples), len(horizons))
    if switch_if_release.shape != expected_shape or switch_if_press.shape != expected_shape:
        raise RuntimeError("switch prediction shape mismatch")

    indices_by_video: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        indices_by_video[sample.video].append(index)
    labels_dir = args.labels_dir.resolve()
    label_data = {
        video: load_label_data(labels_dir, video) for video in indices_by_video
    }

    baseline_by_video: dict[str, list[ReplayDecision]] = {}
    baseline_min_hold_by_video: dict[str, list[ReplayDecision]] = {}
    scheduler_control_by_video: dict[str, list[ReplayDecision]] = {}
    scheduler_by_video: dict[str, list[ReplayDecision]] = {}
    for video, indices in indices_by_video.items():
        baseline_by_video[video] = simulate_fixed_horizon(
            video,
            samples,
            indices,
            switch_if_release,
            switch_if_press,
            horizon_index=control_index,
            threshold=args.threshold,
        )
        baseline_min_hold_by_video[video] = simulate_fixed_horizon_min_hold(
            video,
            samples,
            indices,
            switch_if_release,
            switch_if_press,
            horizon_index=control_index,
            threshold=args.threshold,
            min_state_hold_ms=scheduler_config.min_state_hold_ms,
        )
        if scheduler_config.reserve_bounded_reversal_during_min_hold:
            scheduler_control_by_video[video] = simulate_scheduler(
                video,
                samples,
                indices,
                switch_if_release,
                switch_if_press,
                scheduler_config=replace(
                    scheduler_config,
                    reserve_bounded_reversal_during_min_hold=False,
                ),
            )
        scheduler_by_video[video] = simulate_scheduler(
            video,
            samples,
            indices,
            switch_if_release,
            switch_if_press,
            scheduler_config=scheduler_config,
        )

    baseline_summary = replay_summary(
        baseline_by_video,
        label_data,
        control_horizon_ms=scheduler_config.control_horizon_ms,
        tolerance_ms=tolerance_ms,
    )
    baseline_min_hold_summary = replay_summary(
        baseline_min_hold_by_video,
        label_data,
        control_horizon_ms=scheduler_config.control_horizon_ms,
        tolerance_ms=tolerance_ms,
    )
    scheduler_control_summary = (
        replay_summary(
            scheduler_control_by_video,
            label_data,
            control_horizon_ms=scheduler_config.control_horizon_ms,
            tolerance_ms=tolerance_ms,
        )
        if scheduler_control_by_video
        else None
    )
    scheduler_summary = replay_summary(
        scheduler_by_video,
        label_data,
        control_horizon_ms=scheduler_config.control_horizon_ms,
        tolerance_ms=tolerance_ms,
    )
    diagnostics = scheduler_diagnostics(scheduler_by_video)
    hold_reversal_events = hold_reversal_event_diagnostics(
        scheduler_by_video,
        label_data,
        control_horizon_ms=scheduler_config.control_horizon_ms,
        tolerance_ms=tolerance_ms,
    )

    print(
        f"split={args.split} samples={len(samples)} device={device} "
        f"threshold={args.threshold:.2f} "
        f"control_h={scheduler_config.control_horizon_ms:g}ms "
        f"anticipation_h={scheduler_config.anticipation_horizon_ms:g}ms "
        f"pending_advance={scheduler_config.pending_advance_ms:g}ms "
        f"hold_arm={scheduler_config.arm_pending_during_min_hold} "
        f"short_overdue={scheduler_config.execute_short_horizon_overdue} "
        f"hold_reversal="
        f"{scheduler_config.reserve_bounded_reversal_during_min_hold}",
        flush=True,
    )
    print_replay_summary("baseline_h200", baseline_summary)
    print_replay_summary(
        "baseline_h200_min_hold",
        baseline_min_hold_summary,
    )
    if scheduler_control_summary is not None:
        print_replay_summary("scheduler_control", scheduler_control_summary)
    print_replay_summary("scheduler", scheduler_summary)
    print(
        "scheduler events: "
        + ", ".join(
            f"{key}={value}" for key, value in diagnostics["reasons"].items()
        ),
        flush=True,
    )
    delays = diagnostics["armed_delay_ms"]
    print(
        f"armed delay: n={delays['count']} mean={delays['mean']:.1f}ms "
        f"p50={delays['p50']:.1f}ms p95={delays['p95']:.1f}ms "
        f"range={delays['min']:.1f}..{delays['max']:.1f}ms",
        flush=True,
    )
    reversal_delays = diagnostics["hold_reversal_delay_ms"]
    print(
        "hold reversal delay: "
        f"n={reversal_delays['count']} "
        f"mean={reversal_delays['mean']:.1f}ms "
        f"p50={reversal_delays['p50']:.1f}ms "
        f"p95={reversal_delays['p95']:.1f}ms "
        f"range={reversal_delays['min']:.1f}.."
        f"{reversal_delays['max']:.1f}ms",
        flush=True,
    )
    if hold_reversal_events:
        print("hold reversal executions:", flush=True)
        for row in hold_reversal_events:
            arm_probs = row["arm_probabilities"]
            assert isinstance(arm_probs, tuple)
            delta_ms = row["target_delta_ms"]
            delta_text = "n/a" if delta_ms is None else f"{float(delta_ms):+.1f}ms"
            print(
                f"  {row['video']} "
                f"arm={float(row['armed_at_ms']):.1f}ms "
                f"exec={float(row['executed_at_ms']):.1f}ms "
                f"state={'PRESS' if row['state_after'] else 'RELEASE'} "
                f"arm_probs=["
                + ",".join(f"{float(value):.3f}" for value in arm_probs)
                + "] "
                f"nearest_gt_delta={delta_text} "
                f"within_tol={row['within_tolerance']}",
                flush=True,
            )

    payload = {
        "split": args.split,
        "samples": len(samples),
        "model": str(model_path),
        "metadata": str(metadata_path),
        "scheduler_config": asdict(scheduler_config),
        "model_family": model_family,
        "input_representation": input_representation,
        "transition_tolerance_ms": tolerance_ms,
        "baseline_h200": baseline_summary,
        "baseline_h200_min_hold": baseline_min_hold_summary,
        "scheduler_control": scheduler_control_summary,
        "scheduler": scheduler_summary,
        "scheduler_diagnostics": diagnostics,
        "hold_reversal_events": hold_reversal_events,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
