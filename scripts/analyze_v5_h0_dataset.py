#!/usr/bin/env python3
"""Validate the V5-H0 dense-horizon dataset before training.

This is a structural/data diagnostic only. It verifies that the proposed dense
0..300 ms targets keep the existing observation grid and v4-C2 sampling flags
fixed, then summarizes the additional H0/50 ms supervision on train or
validation data. The test split is intentionally unavailable here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.train.dataset import DatasetSample  # noqa: E402
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.trainer import (  # noqa: E402
    load_video_split,
    partition_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the V5-H0 dataset scaffold.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_v5_h0.yaml",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "data" / "processed" / "v3" / "samples.jsonl",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "data" / "processed" / "v5_h0" / "samples.jsonl",
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sample_key(sample: DatasetSample) -> tuple[str, tuple[int, ...]]:
    return sample.video, sample.input_frame_indices


def infer_horizons(sample: DatasetSample) -> tuple[float, ...]:
    observation_ms = float(sample.input_timestamps_ms[-1])
    timestamps = sample.target_timestamps_ms or (sample.target_timestamp_ms,)
    return tuple(round(float(value) - observation_ms, 6) for value in timestamps)


def index_samples(
    samples: list[DatasetSample],
) -> dict[tuple[str, tuple[int, ...]], DatasetSample]:
    result: dict[tuple[str, tuple[int, ...]], DatasetSample] = {}
    for sample in samples:
        key = sample_key(sample)
        if key in result:
            raise ValueError(f"duplicate observation key: {key}")
        result[key] = sample
    return result


def state_tuple(sample: DatasetSample) -> tuple[bool, ...]:
    return sample.target_pressed_by_horizon or (sample.target_pressed,)


def bool_tuple(values: tuple[bool, ...], fallback: bool) -> tuple[bool, ...]:
    return values or (fallback,)


def main() -> int:
    args = parse_args()
    split = load_video_split(args.config.resolve())

    baseline_all = load_v3_samples(args.baseline.resolve())
    candidate_all = load_v3_samples(args.candidate.resolve())
    baseline = partition_samples(baseline_all, split)[args.split]
    candidate = partition_samples(candidate_all, split)[args.split]
    if not baseline or not candidate:
        raise ValueError("selected split has no samples")

    baseline_horizons = infer_horizons(baseline[0])
    candidate_horizons = infer_horizons(candidate[0])
    expected_candidate = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
    if candidate_horizons != expected_candidate:
        raise ValueError(
            f"candidate horizons must be {expected_candidate}, got {candidate_horizons}"
        )

    baseline_by_key = index_samples(baseline)
    candidate_by_key = index_samples(candidate)
    baseline_keys = set(baseline_by_key)
    candidate_keys = set(candidate_by_key)
    common_keys = baseline_keys & candidate_keys

    shared_horizons = tuple(
        horizon for horizon in baseline_horizons if horizon in candidate_horizons
    )
    baseline_shared_indices = tuple(
        baseline_horizons.index(horizon) for horizon in shared_horizons
    )
    candidate_shared_indices = tuple(
        candidate_horizons.index(horizon) for horizon in shared_horizons
    )

    current_state_mismatch = 0
    input_timestamp_mismatch = 0
    shared_target_mismatch = 0
    shared_transition_mask_mismatch = 0
    shared_short_mask_mismatch = 0
    sampling_flag_mismatch = 0
    h0_state_mismatch = 0
    h0_frame_overlap = 0
    h0_counterfactual_positive = 0
    h0_counterfactual_total = 0
    adjacent_changes = Counter()
    relative_patterns = Counter()
    multi_transition_samples = 0

    for key in sorted(common_keys):
        old = baseline_by_key[key]
        new = candidate_by_key[key]
        if old.current_pressed is not new.current_pressed:
            current_state_mismatch += 1
        if old.input_timestamps_ms != new.input_timestamps_ms:
            input_timestamp_mismatch += 1

        old_states = state_tuple(old)
        new_states = state_tuple(new)
        old_transition = bool_tuple(
            old.near_transition_by_horizon, old.near_transition
        )
        new_transition = bool_tuple(
            new.near_transition_by_horizon, new.near_transition
        )
        old_short = bool_tuple(
            old.near_short_correction_by_horizon, old.near_short_correction
        )
        new_short = bool_tuple(
            new.near_short_correction_by_horizon, new.near_short_correction
        )

        if tuple(old_states[index] for index in baseline_shared_indices) != tuple(
            new_states[index] for index in candidate_shared_indices
        ):
            shared_target_mismatch += 1
        if tuple(old_transition[index] for index in baseline_shared_indices) != tuple(
            new_transition[index] for index in candidate_shared_indices
        ):
            shared_transition_mask_mismatch += 1
        if tuple(old_short[index] for index in baseline_shared_indices) != tuple(
            new_short[index] for index in candidate_shared_indices
        ):
            shared_short_mask_mismatch += 1
        if (
            old.near_transition != new.near_transition
            or old.near_short_correction != new.near_short_correction
        ):
            sampling_flag_mismatch += 1

        h0_state = bool(new_states[0])
        if new.current_pressed is None or h0_state != bool(new.current_pressed):
            h0_state_mismatch += 1
        if new.target_frame_indices and (
            new.target_frame_indices[0] == new.input_frame_indices[-1]
        ):
            h0_frame_overlap += 1

        # Counterfactual training emits each visual sample once with RELEASE and
        # once with PRESS. Exactly one supplied state differs from the expert H0
        # action, so H0 switch supervision is balanced by construction.
        for conditioned_pressed in (False, True):
            h0_counterfactual_positive += int(h0_state != conditioned_pressed)
            h0_counterfactual_total += 1

        transition_count = 0
        for left_index in range(len(candidate_horizons) - 1):
            right_index = left_index + 1
            changed = bool(new_states[left_index]) != bool(new_states[right_index])
            if changed:
                transition_count += 1
                label = (
                    f"{candidate_horizons[left_index]:g}->"
                    f"{candidate_horizons[right_index]:g}ms"
                )
                adjacent_changes[label] += 1
        if transition_count >= 2:
            multi_transition_samples += 1

        h0 = bool(new_states[0])
        pattern = "".join("1" if bool(value) != h0 else "0" for value in new_states)
        relative_patterns[pattern] += 1

    sample_count = len(candidate)
    counterfactual_share = (
        h0_counterfactual_positive / h0_counterfactual_total
        if h0_counterfactual_total
        else 0.0
    )
    payload = {
        "split": args.split,
        "baseline_samples": len(baseline),
        "candidate_samples": len(candidate),
        "common_samples": len(common_keys),
        "missing_from_candidate": len(baseline_keys - candidate_keys),
        "new_candidate_observations": len(candidate_keys - baseline_keys),
        "baseline_horizons_ms": list(baseline_horizons),
        "candidate_horizons_ms": list(candidate_horizons),
        "shared_horizons_ms": list(shared_horizons),
        "alignment": {
            "current_state_mismatch": current_state_mismatch,
            "input_timestamp_mismatch": input_timestamp_mismatch,
            "shared_target_mismatch": shared_target_mismatch,
            "shared_transition_mask_mismatch": shared_transition_mask_mismatch,
            "shared_short_mask_mismatch": shared_short_mask_mismatch,
            "sampling_flag_mismatch": sampling_flag_mismatch,
        },
        "h0": {
            "recorded_state_mismatch": h0_state_mismatch,
            "target_frame_equals_current_input": h0_frame_overlap,
            "target_frame_overlap_share": h0_frame_overlap / sample_count,
            "counterfactual_switch_positive_share": counterfactual_share,
        },
        "adjacent_action_changes": dict(sorted(adjacent_changes.items())),
        "multi_transition_samples": multi_transition_samples,
        "multi_transition_share": multi_transition_samples / sample_count,
        "relative_to_h0_patterns": dict(relative_patterns.most_common()),
    }

    print(
        f"split={args.split} baseline={len(baseline)} candidate={len(candidate)} "
        f"common={len(common_keys)} horizons={candidate_horizons}"
    )
    print(
        "alignment: "
        f"missing={payload['missing_from_candidate']} "
        f"new={payload['new_candidate_observations']} "
        f"state={current_state_mismatch} timestamps={input_timestamp_mismatch} "
        f"shared_target={shared_target_mismatch} "
        f"transition_mask={shared_transition_mask_mismatch} "
        f"short_mask={shared_short_mask_mismatch} "
        f"sampling_flags={sampling_flag_mismatch}"
    )
    print(
        "h0: "
        f"recorded_state_mismatch={h0_state_mismatch} "
        f"frame_overlap={h0_frame_overlap}/{sample_count} "
        f"counterfactual_positive_share={counterfactual_share:.3f}"
    )
    print("adjacent action changes:")
    for label, count in sorted(adjacent_changes.items()):
        print(f"  {label}: n={count} share={count / sample_count:.4f}")
    print(
        f"multi-transition within 0..300ms: n={multi_transition_samples} "
        f"share={multi_transition_samples / sample_count:.4f}"
    )
    print("top relative-to-H0 patterns:")
    for pattern, count in relative_patterns.most_common(12):
        print(f"  {pattern}: n={count} share={count / sample_count:.4f}")

    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Output: {output}")

    structural_errors = sum(
        (
            len(baseline_keys - candidate_keys),
            len(candidate_keys - baseline_keys),
            current_state_mismatch,
            input_timestamp_mismatch,
            shared_target_mismatch,
            shared_transition_mask_mismatch,
            shared_short_mask_mismatch,
            sampling_flag_mismatch,
            h0_state_mismatch,
        )
    )
    return 0 if structural_errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
