#!/usr/bin/env python3
"""Read-only offline evaluation of a frozen Tiny live Shadow recording.

Without checkpoints: analyze real touch events, model logs and frame metadata.
With checkpoints: compare real/virtual causal control history and touch masking.
Never sends Android controls, trains, or changes checkpoint files.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import replace
import json
from pathlib import Path
import statistics


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def touch_timeline(rows, origin_ms, shift_ms=0.0):
    """Map host-received getevent time into the decoder's host clock.

    Shift is diagnostic, NOT a calibrated camera/encoder timing correction.
    """
    result = []
    current = False
    transitions = [r for r in rows if r.get("kind") == "touch_transition"]
    if not transitions:
        raise ValueError("No physical touch transitions; cannot evaluate empty labels")
    previous = float("-inf")
    for row in transitions:
        t = float(row["host_received_perf_counter_ms"]) - origin_ms + shift_ms
        if t <= previous:
            raise ValueError("touch host-received timestamps not strictly increasing")
        previous = t
        pressed = bool(row["pressed"])
        if t <= 0:
            current = pressed
            continue
        if not result:
            result.append((0.0, current))
        if pressed != result[-1][1]:
            result.append((t, pressed))
    if not result:
        result.append((0.0, current))
    return result


def timeline_state(events, t):
    times = [x[0] for x in events]
    return bool(events[max(0, bisect_right(times, t) - 1)][1])


def control_features(stamps, events):
    """Causal [pressed, transition age / 500, frame delta / 200]."""
    import numpy as np
    times = [x[0] for x in events]
    array = np.zeros((len(stamps), 3), dtype=np.float32)
    for i, t in enumerate(stamps):
        if t < 0 or (i and t < stamps[i - 1]):
            raise ValueError("noncausal image timestamps")
        at, pressed = events[max(0, bisect_right(times, t + 1e-7) - 1)]
        array[i, 0] = float(pressed)
        array[i, 1] = min(max(t - at, 0), 500) / 500
        array[i, 2] = min(max(t - stamps[i - 1], 0), 200) / 200 if i else 0
    return array


def state_scores(points, truth):
    """State prediction is scored at t+100ms, not observation time."""
    if not points:
        return {"samples": 0}
    tp = fp = tn = fn = 0
    for t, p in points:
        gt = timeline_state(truth, t)
        if p and gt:
            tp += 1
        elif p:
            fp += 1
        elif gt:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "samples": len(points), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / len(points),
        "press_precision": precision, "press_recall": recall,
        "press_f1": (2 * precision * recall / (precision + recall)
                     if precision is not None and recall is not None
                     and precision + recall else None),
    }


def transitions_from_points(points):
    result = []
    last = None
    for t, pressed in sorted(points):
        if last is not None and pressed != last:
            result.append((float(t), bool(pressed)))
        last = pressed
    return result


def window(events, start, end):
    return [(t, p) for t, p in events if start <= t < end]


def match_events(pred, truth, *, pressed, tol_ms=150.0):
    """One-to-one nearest matching, only for same-polarity transitions."""
    p = [x for x in pred if x[1] == pressed]
    gt = [x for x in truth if x[1] == pressed]
    pairs = sorted(
        (abs(a[0] - b[0]), i, j, a[0] - b[0])
        for i, a in enumerate(p) for j, b in enumerate(gt)
        if abs(a[0] - b[0]) <= tol_ms
    )
    used_p, used_gt, deltas = set(), set(), []
    for _, i, j, delta in pairs:
        if i not in used_p and j not in used_gt:
            used_p.add(i)
            used_gt.add(j)
            deltas.append(delta)
    return {
        "actual": len(gt), "predicted": len(p), "matched": len(deltas),
        "precision": len(deltas) / len(p) if p else None,
        "recall": len(deltas) / len(gt) if gt else None,
        "signed_lag_median_ms": statistics.median(deltas) if deltas else None,
        "absolute_error_median_ms": (
            statistics.median(map(abs, deltas)) if deltas else None
        ),
    }


def physical_press_spans(touch_rows, origin_ms, start, end, shift_ms=0.0):
    """Pair via host time, measure duration using the Android kernel clock."""
    transitions = [r for r in touch_rows if r.get("kind") == "touch_transition"]
    pressed = None
    result = []
    for row in transitions:
        if row["pressed"]:
            pressed = row
        elif pressed is not None:
            host_start = pressed["host_received_perf_counter_ms"] - origin_ms + shift_ms
            host_end = row["host_received_perf_counter_ms"] - origin_ms + shift_ms
            duration = row["device_timestamp_ms"] - pressed["device_timestamp_ms"]
            if start <= host_start and host_end < end and duration >= 0:
                result.append((host_start, host_end, duration))
            pressed = None
    return result


def paired_duration_drift(model_events, human_events, start, end,
                          tol_ms=150.0, physical_spans=None):
    """Compare hold lengths; a constant cross-clock shift cancels out."""
    def episodes(events):
        return [
            (events[i][0], events[i + 1][0])
            for i in range(len(events) - 1)
            if events[i][1] and not events[i + 1][1]
            and start <= events[i][0] and events[i + 1][0] < end
        ]
    pred = episodes(model_events)
    actual = (physical_spans if physical_spans is not None
              else [(a, b, b - a) for a, b in episodes(human_events)])
    candidates = sorted(
        (abs(p[0] - a[0]), i, j)
        for i, p in enumerate(pred) for j, a in enumerate(actual)
        if abs(p[0] - a[0]) <= tol_ms
    )
    used_p, used_actual, delta = set(), set(), []
    for _, i, j in candidates:
        if i in used_p or j in used_actual:
            continue
        used_p.add(i)
        used_actual.add(j)
        delta.append((pred[i][1] - pred[i][0]) - actual[j][2])
    return {
        "pairs_matched_by_press_onset": len(delta),
        "pred_episodes": len(pred), "human_episodes": len(actual),
        "duration_difference_median_ms": statistics.median(delta) if delta else None,
        "extra_holding_fraction": (
            sum(d > 0 for d in delta) / len(delta) if delta else None
        ),
        "note": ("Android kernel duration vs model duration; pairing by "
                 "uncalibrated host PRESS arrival, may mispair chattering"),
    }


def frame_consistency(summary, frames, pred):
    indices = [int(row["frame_index"]) for row in frames]
    if indices != list(range(len(indices))):
        raise ValueError("frame indices must be contiguous and zero-based")
    clocks = [float(x["host_decoded_perf_counter_ms"]) for x in frames]
    if any(b <= a for a, b in zip(clocks, clocks[1:])):
        raise ValueError("nonmonotonic frame timestamps")
    for row in pred:
        if not 0 <= row["frame_index"] < len(frames):
            raise ValueError("model frame index missing in decoder sidecar")
        timestamp = frames[row["frame_index"]]["host_decoded_ms"]
        if abs(row["observation_ms"] - timestamp) > .01:
            raise ValueError("prediction timestamp does not match frame index")
    if summary["recording"]["recorded_frames"] != len(frames):
        raise ValueError("recording frame count != frame timestamp count")
    return {
        "frames": len(frames), "prediction_rows": len(pred),
        "frame_metadata_consistent": True,
        "ffmpeg_recording_error": summary["recording"].get("recording_error"),
        "summary_mapping_flag": summary["recording"].get("frame_mapping_valid"),
    }


def estimate_logged(pred, summary, truth, start, end, physical_spans=None):
    data = {}
    human_transitions = window(truth[1:], start, end)
    for mode in ("rgb", "rgb_hsv"):
        rows = [r for r in pred
                if start <= r["models"][mode]["target_ms"] < end]
        points = [
            (r["models"][mode]["target_ms"],
             r["models"][mode]["probability"] >= .5)
            for r in rows
        ]
        events = [
            (float(x["ms"]), bool(x["pressed"]))
            for x in summary["summary"]["virtual_events"][mode]
        ][1:]
        visible = window(events, start, end)
        data[mode] = {
            "state_at_target": state_scores(points, truth),
            "virtual_press": match_events(visible, human_transitions, pressed=True),
            "virtual_release": match_events(visible, human_transitions, pressed=False),
            "paired_hold_duration": paired_duration_drift(
                events, truth, start, end, physical_spans=physical_spans
            ),
        }
    persistent = [
        (r["models"]["rgb"]["target_ms"],
         timeline_state(truth, r["observation_ms"]))
        for r in pred if start <= r["models"]["rgb"]["target_ms"] < end
    ]
    data["expert_state_persistence_reference"] = state_scores(persistent, truth)
    return data


def replay_models(video_path, frames, pred, summary, checkpoint_dir, config_path,
                  road_config_path, batch_size, cpu_threads, truth):
    import cv2
    import numpy as np
    import torch
    import yaml
    from experiment_tiny_temporal_policy import prepare_video_feature
    from inspect_geometry_pseudo_labels import load_config
    from karting_agent.vision.preprocess import preprocess_config_from_mapping
    from run_tiny_live_shadow import load_frozen_models

    torch.set_num_threads(cpu_threads)
    models, provenance = load_frozen_models(checkpoint_dir, config_path)
    for mode in models:
        expected = summary["models"][mode]["weights_sha256"]
        if provenance[mode]["weights_sha256"] != expected:
            raise ValueError("checkpoint differs from recorded Shadow model")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    original = preprocess_config_from_mapping(config)
    wide = replace(original, touch_roi=(.68, .76, 1.0, 1.0))
    road, _, _ = load_config(road_config_path)

    wanted = {int(i) for row in pred for i in row["input_indices"]}
    features = {"original": {}, "wide": {}}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("recorded video cannot be opened")
    count = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if count in wanted:
                features["original"][count] = prepare_video_feature(
                    image, preprocess=original, road_config=road
                )
                features["wide"][count] = prepare_video_feature(
                    image, preprocess=wide, road_config=road
                )
            count += 1
    finally:
        capture.release()
    if count != len(frames) or any(len(group) != len(wanted)
                                   for group in features.values()):
        raise ValueError("MP4 decoded-frame indexing mismatch; replay unsafe")

    predictions = {}
    for mask, lookup in features.items():
        for history in ("teacher", "recorded_virtual"):
            for mode, model in models.items():
                virtual = [
                    (float(x["ms"]), bool(x["pressed"]))
                    for x in summary["summary"]["virtual_events"][mode]
                ]
                events = truth if history == "teacher" else virtual
                probs = []
                for k in range(0, len(pred), batch_size):
                    block = pred[k:k + batch_size]
                    channels = 3 if mode == "rgb" else 4
                    images = np.stack([
                        np.stack([lookup[i][:channels] for i in row["input_indices"]])
                        for row in block
                    ]).astype(np.float32) / 255.0
                    controls = np.stack([
                        control_features(row["input_timestamp_ms"], events)
                        for row in block
                    ])
                    with torch.inference_mode():
                        batch = torch.sigmoid(
                            model(torch.from_numpy(images), torch.from_numpy(controls))
                        ).cpu().numpy()
                    probs.extend(map(float, batch))
                predictions[f"{mode}_{history}_{mask}"] = probs

    # Replaying the original virtual input should reproduce real-time logits.
    # If it does not, MP4 mapping or preprocessing is not trustworthy.
    parity = {}
    for mode in ("rgb", "rgb_hsv"):
        calculated = np.asarray(
            predictions[f"{mode}_recorded_virtual_original"]
        )
        original_prob = np.asarray([
            row["models"][mode]["probability"] for row in pred
        ])
        delta = np.abs(calculated - original_prob)
        parity[mode] = {
            "p50_abs_prob_diff": float(np.percentile(delta, 50)),
            "p95_abs_prob_diff": float(np.percentile(delta, 95)),
            "max_abs_prob_diff": float(delta.max()),
            "passes_0p03_p95": bool(np.percentile(delta, 95) <= .03),
        }
    return predictions, parity, count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-stem", type=Path, required=True,
                    help="Path prefix of five paired files, without suffix")
    ap.add_argument("--checkpoint-dir", type=Path, default=None,
                    help="Frozen RGB/HSV weights; enables four-way mask/history replay")
    ap.add_argument("--output-dir", type=Path, required=True)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--config", type=Path,
                    default=root / "configs/train_v4c2_temporal_v2.yaml")
    ap.add_argument("--road-config", type=Path,
                    default=root / "configs/geometry_pseudo_labels.yaml")
    ap.add_argument("--race-start-ms", type=float, default=0.0)
    ap.add_argument("--race-end-ms", type=float, default=None)
    ap.add_argument("--touch-shift-ms", type=float, default=0.0,
                    help="Exploratory touch/video offset; not a calibration")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--cpu-threads", type=int, default=4)
    args = ap.parse_args()
    if not args.output_dir.is_dir() or args.batch_size < 1 or args.cpu_threads < 1:
        ap.error("output-dir must exist, batch and cpu threads must be positive")
    stem = args.run_stem.with_suffix("") if args.run_stem.suffix == ".mp4" else args.run_stem

    summary = json.loads(Path(str(stem) + "_summary.json").read_text(encoding="utf-8"))
    predictions = [
        row for row in read_jsonl(Path(str(stem) + ".jsonl"))
        if row.get("kind") == "prediction"
    ]
    frames = read_jsonl(Path(str(stem) + "_frames.jsonl"))
    touches = read_jsonl(Path(str(stem) + "_touch.jsonl"))
    if summary.get("touch_events", {}).get("transitions", 0) < 1:
        raise ValueError("Missing recorded physical touch transitions")
    consistency = frame_consistency(summary, frames, predictions)
    origin = (frames[0]["host_decoded_perf_counter_ms"]
              - frames[0]["host_decoded_ms"])
    end = (args.race_end_ms if args.race_end_ms is not None
           else frames[-1]["host_decoded_ms"])
    start = args.race_start_ms
    if not 0 <= start < end <= frames[-1]["host_decoded_ms"] + 1:
        ap.error("invalid race time range")

    truth = touch_timeline(touches, origin, args.touch_shift_ms)
    physical = physical_press_spans(
        touches, origin, start, end, args.touch_shift_ms
    )
    durations = [span[2] for span in physical]
    result = {
        "kind": "offline_tiny_live_shadow_diagnostics",
        "source_run": str(stem), "read_only": True,
        "window_ms": [start, end], "touch_shift_ms": args.touch_shift_ms,
        "clock_status": ("UNCALIBRATED: getevent host arrival vs video decoder "
                         "availability; encoder/transport latency remains unknown"),
        "video_frame_consistency": consistency,
        "touch": {
            "transitions_in_window": len(window(truth[1:], start, end)),
            "complete_press_spans": len(physical),
            "press_duration_ms_median": (
                statistics.median(durations) if durations else None
            ),
        },
        "recorded_virtual": estimate_logged(
            predictions, summary, truth, start, end, physical
        ),
        "shift_sensitivity": {}, "ablations": None,
    }
    for shift in (-150, -100, -50, 0, 50, 100, 150):
        candidate = touch_timeline(touches, origin, args.touch_shift_ms + shift)
        shifted_physical = physical_press_spans(
            touches, origin, start, end, args.touch_shift_ms + shift
        )
        result["shift_sensitivity"][str(shift)] = estimate_logged(
            predictions, summary, candidate, start, end, shifted_physical
        )

    report_path = args.output_dir / (stem.name + "_offline_eval_summary.json")
    prediction_path = args.output_dir / (stem.name + "_offline_eval_predictions.jsonl")
    if report_path.exists() or (args.checkpoint_dir is not None
                                and prediction_path.exists()):
        raise FileExistsError("Refusing to overwrite offline evaluation")
    if args.checkpoint_dir is not None:
        if not args.checkpoint_dir.is_dir():
            ap.error("checkpoint-dir does not exist")
        computed, parity, decoded = replay_models(
            Path(str(stem) + ".mp4"), frames, predictions, summary,
            args.checkpoint_dir, args.config, args.road_config,
            args.batch_size, args.cpu_threads, truth
        )
        comparisons = {}
        truth_events = window(truth[1:], start, end)
        for name, probabilities in computed.items():
            points = [
                (row["models"]["rgb"]["target_ms"], probability >= .5)
                for row, probability in zip(predictions, probabilities)
                if start <= row["models"]["rgb"]["target_ms"] < end
            ]
            changes = window(transitions_from_points(points), start, end)
            comparisons[name] = {
                "state_at_target": state_scores(points, truth),
                "press_transitions": match_events(
                    changes, truth_events, pressed=True
                ),
                "release_transitions": match_events(
                    changes, truth_events, pressed=False
                ),
            }
        result["ablations"] = {
            "checkpoint_provenance_checked": True,
            "video_decoded_frames": decoded,
            "wide_touch_roi": [.68, .76, 1., 1.],
            "recorded_live_probability_replay_parity": parity,
            "interpretation_valid_only_if_parity_passed": all(
                x["passes_0p03_p95"] for x in parity.values()
            ),
            "comparisons": comparisons,
            "note": ("Mask comparison holds the recorded virtual history fixed, "
                     "not a counterfactual closed-loop trajectory. Teacher control "
                     "uses uncalibrated physical-touch host-arrival timestamps."),
        }
        with prediction_path.open("x", encoding="utf-8") as stream:
            for i, row in enumerate(predictions):
                target = row["models"]["rgb"]["target_ms"]
                stream.write(json.dumps({
                    "frame_index": row["frame_index"], "target_ms": target,
                    "true_pressed_host_uncalibrated": timeline_state(truth, target),
                    "probabilities": {key: values[i]
                                      for key, values in computed.items()},
                }, ensure_ascii=False) + "\n")

    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Evaluation:", report_path)
    if args.checkpoint_dir is not None:
        print("Predictions:", prediction_path)
    print("Physical spans:", result["touch"])
    for mode in ("rgb", "rgb_hsv"):
        detail = result["recorded_virtual"][mode]
        print(mode, "target accuracy",
              round(detail["state_at_target"]["accuracy"], 4),
              "PRESS matched", detail["virtual_press"]["matched"],
              "/", detail["virtual_press"]["actual"],
              "RELEASE matched", detail["virtual_release"]["matched"],
              "/", detail["virtual_release"]["actual"])
    if result["ablations"] is not None:
        print("Replay parity:",
              result["ablations"]["recorded_live_probability_replay_parity"])


if __name__ == "__main__":
    main()
