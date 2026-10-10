"""Fixed-video policy replay with expert-vs-simulated executed-action inputs.

Offline counterfactual *action history*, not counterfactual vehicle imagery.
Results cannot estimate true closed-loop driving success or recovery.
"""
from __future__ import annotations

import math
from time import perf_counter
from typing import Callable, Mapping

import numpy as np
import torch

from .shadow_control import ShadowController


def replay_fixed_video(
    model: torch.nn.Module,
    samples: list,
    *,
    frames_for_sample: Callable,
    expert_events: tuple,
    expert_features: Callable,
    latency_ms: float = 0.0,
    teacher_probabilities: Mapping[tuple[str, float], float] | None = None,
) -> dict:
    """Collect paired frozen-policy predictions with no future control leakage.

    samples must be ordered at ~30 Hz; visual frames are the same expert video
    for both paths. Future targets are only used for labelling after inference.
    Optional teacher_probabilities come from the full-validation original
    DataLoader with its original batch size, not from per-frame paired batches.
    Inference does NOT execute game controls; scheduling is simulated.
    """
    if not samples or not expert_events:
        raise ValueError("no samples or expert initialization")
    if abs(float(expert_events[0].timestamp_ms)) > 1e-5:
        raise ValueError("expert initial event must be at t=0")
    ordered = sorted(samples, key=lambda row: float(row.target_timestamp_ms))
    video = ordered[0].video
    if any(row.video != video for row in ordered):
        raise ValueError("one video per replay, no cross-video state carryover")
    controller = ShadowController(initial_pressed=bool(expert_events[0].pressed),
                                  horizon_ms=100.0, latency_ms=latency_ms)
    model.eval()
    rows = []
    inference_ms = []
    previous_observation = -1.0
    with torch.inference_mode():
        for sample in ordered:
            stamps = tuple(float(t) for t in sample.input_timestamps_ms)
            observation = stamps[-1]
            target = float(sample.target_timestamp_ms)
            if (observation <= previous_observation or
                    not math.isclose(target-observation, 100.0, abs_tol=1e-3)):
                raise ValueError("out-of-order samples or unexpected target horizon")
            previous_observation = observation
            controller.advance_to(observation)
            feedback = controller.control_features(stamps)
            # Same pixels for both paths: only past *executed* control varies.
            images = np.asarray(frames_for_sample(sample),dtype=np.float32)
            if images.ndim != 4 or images.shape[0] != len(stamps):
                raise ValueError("visual feature shape mismatch")
            if teacher_probabilities is None:
                oracle_features = expert_features(stamps, expert_events)
                inputs = torch.from_numpy(np.stack([images,images]))
                controls = torch.from_numpy(np.stack([oracle_features,feedback]))
            else:
                # Full Validation teacher reference was precomputed with
                # the ORIGINAL unshuffled batch=16 DataLoader (possibly
                # spanning video boundaries). Running a paired batch=2 here
                # changes numerical kernels and can flip threshold crossings.
                key = (video, target)
                if key not in teacher_probabilities:
                    raise ValueError(f"missing canonical teacher sample: {key}")
                inputs = torch.from_numpy(images[None])
                controls = torch.from_numpy(feedback[None])
            start = perf_counter()
            logits = model(inputs,controls)
            inference_ms.append(1000.0*(perf_counter()-start))
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            required = (2,) if teacher_probabilities is None else (1,)
            if probabilities.shape != required or not np.all(np.isfinite(probabilities)):
                raise ValueError("invalid model probability outputs")
            if teacher_probabilities is None:
                expert_probability, shadow_probability = map(float,probabilities)
            else:
                expert_probability = float(teacher_probabilities[(video,target)])
                shadow_probability = float(probabilities[0])
                if not 0.0 <= expert_probability <= 1.0 or not math.isfinite(expert_probability):
                    raise ValueError("invalid canonical teacher probability")
            proposal = controller.propose(observe_ms=observation,target_ms=target,
                                           probability=shadow_probability)
            rows.append({
                "video":video,
                "observation_ms":observation,
                "target_ms":target,
                "expert_future_pressed":bool(sample.target_pressed),
                "expert_current_pressed":bool(sample.current_pressed),
                "simulated_current_pressed":bool(feedback[-1,0] >= 0.5),
                "teacher_probability":expert_probability,
                "shadow_probability":shadow_probability,
                "shadow_desired_pressed":proposal.pressed,
                "simulated_execution_ms":proposal.execute_ms,
                "inference_ready_ms":proposal.ready_ms,
                "shadow_current_action_age_ms":float(feedback[-1,1]*500),
                "per_sample_forward_ms":inference_ms[-1],
            })
    controller.flush()
    for row in rows:
        row["simulated_executed_at_target"] = controller.state_at(row["target_ms"])
        row["executed_matches_expert_at_target"] = (
            row["simulated_executed_at_target"] == row["expert_future_pressed"])
    return {
        "video": video, "rows":rows,
        "simulated_events": [
            {"time_ms":event.time_ms,"pressed":event.pressed}
            for event in controller.events
        ],
        "proposals":len(controller.proposals),
        "redundant_proposals":controller.redundant,
        "late_deadlines":sum(p.execute_ms>p.target_ms+1e-6 for p in controller.proposals),
        "inference_wall_ms_p50":float(np.median(inference_ms)),
        "inference_wall_ms_p95":float(np.percentile(inference_ms,95)),
        "first_observation_ms":rows[0]["observation_ms"],
        "first_simulated_divergence_observation_ms":next(
            (row["observation_ms"] for row in rows
             if row["expert_current_pressed"]!=row["simulated_current_pressed"]),None),
        "simulated_observation_disagreement_fraction":float(np.mean([
            row["expert_current_pressed"]!=row["simulated_current_pressed"]
            for row in rows
        ])),
        "executed_target_accuracy_proxy":float(np.mean([
            row["executed_matches_expert_at_target"] for row in rows
        ])),
    }
