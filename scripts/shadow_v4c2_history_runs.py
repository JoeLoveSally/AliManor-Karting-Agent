#!/usr/bin/env python3
"""Compare H200 *signals* on recorded original-policy frames; never drive ADB.

Recorded MP4 PTS is synthetic: decode by zero-based source frame indices and use
JSON timestamps. Decisions after the first signal difference remain on the
original policy's observed trajectory, NOT a candidate closed-loop replay.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def recorded_events(run):
    """Command timeline: a step's action occurs AFTER its model observation."""
    events = []
    for i, step in enumerate(run["steps"]):
        if step["action"] in ("PRESS", "RELEASE"):
            state = step["action"] == "PRESS"
            if bool(step["pressed"]) != state:
                raise ValueError(f"step {i}: action disagrees with post-action pressed")
            events.append((float(step["observation_timestamp_ms"]), 1, i, state))
    for i, event in enumerate(run.get("deadline_events", [])):
        if event["action"] in ("PRESS", "RELEASE"):
            state = event["action"] == "PRESS"
            if bool(event["pressed"]) != state:
                raise ValueError(f"deadline event {i}: inconsistent pressed")
            events.append((float(event["timestamp_ms"]), 0, i, state))
    events.sort()
    initial = bool(run["runtime"].get("initial_pressed", False))
    previous = initial
    times = []
    transitions = []
    for timestamp, priority, index, state in events:
        if times and timestamp <= times[-1]:
            raise ValueError(f"same-time or backwards command events at {timestamp}ms")
        if state == previous:
            raise ValueError(f"non-alternating command events at {timestamp}ms, source {priority}:{index}")
        times.append(timestamp)
        transitions.append((timestamp, state))
        previous = state
    return initial, times, transitions


def preflight(run):
    """Validate factual input state and log mapping without guessing past actions."""
    recording = run.get("recording") or {}
    if not recording.get("frame_mapping_valid") or recording.get("dropped_frames") != 0:
        raise ValueError("MP4/source-frame mapping not validated or recording has missing frames")
    if recording.get("timing") != "synthetic_cfr_stream_copy":
        raise ValueError("unknown MP4 timing semantics")
    if run.get("input_mode") != "video" or not run.get("steps"):
        raise ValueError("expected nonempty video run")
    count = int(recording["frame_count"])
    decoded_times = run["capture"]["decoded_frame_timestamps_ms"]
    if len(decoded_times) != count:
        raise ValueError(f"JSON decoded-frame count {len(decoded_times)} != recording {count}")
    initial, times, transitions = recorded_events(run)
    previous_observation = float("-inf")
    for i, step in enumerate(run["steps"]):
        timestamp = float(step["observation_timestamp_ms"])
        if timestamp <= previous_observation:
            raise ValueError(f"step {i}: observations not strictly increasing")
        previous_observation = timestamp
        indices = step["input_frame_indices"]
        frame_times = step["input_frame_timestamps_ms"]
        if len(indices) != len(frame_times) or not indices:
            raise ValueError(f"step {i}: frame indices/timestamps mismatch")
        if any(not isinstance(index, int) or index < 0 or index >= count for index in indices):
            raise ValueError(f"step {i}: source frame outside MP4")
        for index, logged_time in zip(indices, frame_times):
            if abs(float(decoded_times[index]) - float(logged_time)) > 0.05:
                raise ValueError(f"step {i}: frame timestamp doesn't match decoded-frame index {index}")
        if max(frame_times) > timestamp + 1e-5:
            raise ValueError(f"step {i}: future input frame")
        before = not bool(step["pressed"]) if step["action"] in ("PRESS", "RELEASE") else bool(step["pressed"])
        # Exclude the current observation's action; the logged timer timestamp
        # is its nominal deadline, not an independently measured command time.
        cut = bisect_left(times, timestamp)
        reconstructed = transitions[cut - 1][1] if cut else initial
        if before != reconstructed:
            raise ValueError(f"step {i}: inferred pre-state {int(reconstructed)} != observed pre-state {int(before)}; timer timing/order ambiguous")
    return initial, times, transitions, count


def frames_by_index(mp4: Path, needed: set[int], expected_count: int, resolution):
    """Decode sequentially, so a random seek cannot silently change H.264 indices."""
    from karting_agent.vision.preprocess import prepare_frame
    capture = cv2.VideoCapture(str(mp4))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open MP4: {mp4}")
    frames = {}
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count in needed:
                if [frame.shape[1], frame.shape[0]] != resolution:
                    raise ValueError(f"MP4 resolution mismatch at source frame {count}")
                frames[count] = frame
            count += 1
    finally:
        capture.release()
    if count != expected_count or set(frames) != needed:
        raise ValueError(f"MP4 decoded {count} vs logged {expected_count} frames; missing {len(needed - set(frames))} requested frames")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = ROOT / "artifacts/models/mobilenet_v3_small_v4c2_temporal_v2"
    adapter = ROOT / "artifacts/models/mobilenet_v3_small_v4c2_history_residual_v1"
    parser.add_argument("runs", nargs="+", type=Path, help="matched JSON files; each MP4 must have the same stem")
    parser.add_argument("--base-model", type=Path, default=baseline / "model.pt")
    parser.add_argument("--metadata", type=Path, default=baseline / "metadata_h200.json")
    parser.add_argument("--adapter", type=Path, default=adapter / "history_adapter.pt")
    parser.add_argument("--adapter-metadata", type=Path, default=adapter / "history_adapter_metadata.json")
    parser.add_argument("--base-threshold", type=float, default=0.60)
    parser.add_argument("--adapter-threshold", type=float, default=0.76)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=adapter / "recorded_runs_shadow.json")
    args = parser.parse_args()
    if not (0 < args.base_threshold < 1 and 0 < args.adapter_threshold < 1):
        raise ValueError("invalid threshold")
    if args.output.exists():
        raise FileExistsError(f"report exists: {args.output}")

    import torch
    from karting_agent.model.action_history import encode_action_history
    from karting_agent.model.action_history_runner import ActionHistoryAdapterRunner
    from karting_agent.vision.preprocess import stack_frames

    model = ActionHistoryAdapterRunner(
        base_model_path=args.base_model, base_metadata_path=args.metadata,
        adapter_path=args.adapter, adapter_metadata_path=args.adapter_metadata, device=args.device)
    spec = model.baseline.spec
    if spec.control_horizon_ms != 200:
        raise ValueError("this audit requires H200 control metadata")
    model.model.eval()
    report = {"comparison": "H200 threshold signals, NOT scheduler actions or counterfactual car trajectory",
              "base_threshold": args.base_threshold, "adapter_threshold": args.adapter_threshold,
              "runs": {}}
    for json_path in args.runs:
        run = json.loads(json_path.read_text(encoding="utf-8"))
        initial, times, transitions, frame_count = preflight(run)
        if list(run["runtime"]["frame_offsets_ms"]) != list(spec.frame_offsets_ms):
            raise ValueError(f"{json_path}: frame offsets differ from base model")
        if list(run["runtime"]["prediction_horizons_ms"]) != list(spec.prediction_horizons_ms):
            raise ValueError(f"{json_path}: horizon metadata mismatch")
        if len(run["steps"][0]["input_frame_indices"]) != spec.frame_stack:
            raise ValueError(f"{json_path}: frame stack mismatch")
        mp4 = json_path.with_suffix(".mp4")
        if not mp4.is_file():
            raise FileNotFoundError(f"matching MP4 missing: {mp4}")
        needed = {int(index) for step in run["steps"] for index in step["input_frame_indices"]}
        frames = frames_by_index(mp4, needed, frame_count, run["recording"]["resolution"])
        rows = []
        for i, step in enumerate(run["steps"]):
            observed = float(step["observation_timestamp_ms"])
            cut = bisect_left(times, observed)
            before = not bool(step["pressed"]) if step["action"] in ("PRESS", "RELEASE") else bool(step["pressed"])
            frame_times = [float(t) for t in step["input_frame_timestamps_ms"]]
            history = encode_action_history(frame_times, observed, initial_pressed=initial,
                                            transitions=transitions[:cut], current_pressed=before,
                                            max_age_ms=model.max_age_ms)
            indices = [int(idx) for idx in step["input_frame_indices"]]
            inputs = stack_frames([frames[idx] for idx in indices], spec.preprocess_config)
            tensor = model.baseline._tensor(inputs)
            state = torch.tensor([int(before)], device=model.baseline.device)
            history_tensor = torch.from_numpy(history).unsqueeze(0).to(model.baseline.device)
            with torch.inference_mode():
                new_logits, base_logits = model.model(tensor, state, history_tensor)
                base = float(torch.sigmoid(base_logits)[0, spec.control_output_index])
                new = float(torch.sigmoid(new_logits)[0, spec.control_output_index])
            logged = float(step["probabilities"][spec.control_output_index])
            old_signal = base >= args.base_threshold
            new_signal = new >= args.adapter_threshold
            age_ms = observed - times[cut - 1] if cut else None
            rows.append({"step": i, "t_ms": observed,
                         "frame_indices": indices, "state_before": "PRESS" if before else "RELEASE",
                         "recorded_action": step["action"], "scheduler_reason": step.get("scheduler_reason"),
                         "logged_h200": round(logged, 6), "replayed_h200": round(base, 6),
                         "adapter_h200": round(new, 6), "original_signal": old_signal,
                         "adapter_signal": new_signal, "signal_disagrees": old_signal != new_signal,
                         "base_log_difference": round(abs(base - logged), 6),
                         "ms_since_logged_command": age_ms,
                         "mixed_frame_command_history": bool((history[:-2:3] != float(before)).any())})
        del frames
        errors = np.array([row["base_log_difference"] for row in rows])
        # Replay fidelity must be independently established, not inferred from
        # a valid MP4 frame mapping alone. Divergent runs remain diagnostic-only.
        mismatched = int((errors > 0.05).sum())
        fidelity = {"median_abs_h200_diff": float(np.median(errors)),
                    "p95_abs_h200_diff": float(np.percentile(errors, 95)),
                    "max_abs_h200_diff": float(errors.max()),
                    "over_0_05": mismatched, "fraction_over_0_05": mismatched / len(rows)}
        eligible = mismatched == 0
        disagreements = [row for row in rows if row["signal_disagrees"]]
        first = disagreements[0] if disagreements else None
        run_report = {"source_json": str(json_path), "source_mp4": str(mp4),
                      "frame_count": frame_count, "steps": len(rows),
                      "events": len(transitions), "replay_fidelity": fidelity,
                      "signal_analysis_valid": eligible,
                      "signal_disagreements": len(disagreements) if eligible else None,
                      "first_signal_disagreement": first if eligible else None,
                      "first_disagreement_context": (rows[max(0, first["step"] - 3):first["step"] + 4]
                                                     if eligible and first else []),
                      "disagreements_before_first": 0 if eligible else None,
                      "warning": ("Base replay differs from recorded H200; do not interpret candidate disagreements."
                                  if not eligible else
                                  "Signal-only; recorded pixels/state remain from the original controller after any difference.")}
        report["runs"][json_path.stem] = run_report
        print(json.dumps({"run": json_path.stem, "steps": len(rows), "fidelity": fidelity,
                          "valid": eligible, "disagreements": run_report["signal_disagreements"],
                          "first": run_report["first_signal_disagreement"]}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved read-only recorded-run signal audit: {args.output}")


if __name__ == "__main__":
    main()
