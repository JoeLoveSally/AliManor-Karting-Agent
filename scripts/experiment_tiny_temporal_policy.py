#!/usr/bin/env python3
"""Expert-only, offline RGB vs RGB+HSV tiny temporal policy experiment.

Uses the existing V4-C2 sample manifest, native-fps expert action timeline
and video-level split. Never sends game controls, runs ADB or edits Armed.
All output paths must be explicitly supplied and already exist as directories.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from audit_road_occupancy_poc import normalize_game_frame  # noqa: E402
from karting_agent.model.tiny_temporal_policy import TinyTemporalPolicy  # noqa: E402
from karting_agent.train.geometry_pseudo_labels import extract_road_mask  # noqa: E402
from karting_agent.train.labels.touch_marker import ActionEvent  # noqa: E402
from karting_agent.train.sequence_evaluator import (  # noqa: E402
    SequencePoint, Transition, ReleaseSegment, combine_evaluations,
    evaluate_sequence,
)
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.trainer import (  # noqa: E402
    load_video_split, load_sampling_config, build_sample_weights,
    partition_samples,
)
from karting_agent.vision.preprocess import (  # noqa: E402
    mask_area, preprocess_config_from_mapping,
)

SIZE = 96
HORIZON_MS = 100.0


def label_events(labels_dir: Path, video: str) -> tuple[ActionEvent, ...]:
    path = labels_dir / f"{Path(video).stem}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = tuple(
        ActionEvent(
            frame_index=int(v["frame_index"]),
            timestamp_ms=float(v["timestamp_ms"]),
            pressed=bool(v["pressed"]),
        )
        for v in raw["events"]
    )
    if not events or abs(events[0].timestamp_ms) > 1e-4:
        raise ValueError(f"missing initial action event at t=0: {path}")
    if any(a.timestamp_ms >= b.timestamp_ms for a,b in zip(events, events[1:])):
        raise ValueError(f"nonmonotonic action events in {path}")
    return events


def observed_state_features(
    input_timestamps_ms, events: tuple[ActionEvent, ...],
) -> np.ndarray:
    """Per-frame [actual_pressed, last_action_age, dt]; all causal at t."""
    if not events:
        raise ValueError("missing actual expert control history")
    event_times = [event.timestamp_ms for event in events]
    stamps = tuple(float(t) for t in input_timestamps_ms)
    if not stamps or any(t < 0 for t in stamps):
        raise ValueError("invalid input timestamps")
    if any(b < a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("input timestamps must not go backwards")
    result = np.zeros((len(stamps), 3), dtype=np.float32)
    for i, t in enumerate(stamps):
        idx = bisect_right(event_times, t + 1e-7) - 1
        if idx < 0:
            raise ValueError("input precedes first expert action event")
        result[i, 0] = float(events[idx].pressed)
        result[i, 1] = min(max(t - events[idx].timestamp_ms, 0.0), 500.0) / 500.0
        result[i, 2] = min(max(t - stamps[i-1], 0.0), 200.0) / 200.0 if i else 0.0
    return result


def validate_sample(sample, events: tuple[ActionEvent, ...],
                    *, horizon_ms: float = HORIZON_MS) -> bool:
    """Verify no future video or control metadata sneaks into model input."""
    stamps = tuple(float(t) for t in sample.input_timestamps_ms)
    frames = tuple(int(i) for i in sample.input_frame_indices)
    if len(stamps) != 5 or len(frames) != 5 or stamps[-1] < 0:
        raise ValueError("expected five input frames at nonnegative times")
    future_times = tuple(float(t) for t in sample.target_timestamps_ms)
    future_states = tuple(bool(x) for x in sample.target_pressed_by_horizon)
    if not future_times or len(future_times) != len(future_states):
        raise ValueError("missing native multi-horizon future action labels")
    if (abs(future_times[0] - sample.target_timestamp_ms) > 1e-3
            or abs(future_times[0] - stamps[-1] - horizon_ms) > 1e-3):
        raise ValueError("inconsistent observation/target horizon")
    if future_times[0] <= stamps[-1]:
        raise ValueError("future target must follow final observed frame")
    if future_states[0] != bool(sample.target_pressed):
        raise ValueError("future target disagreement")
    state = bool(observed_state_features((stamps[-1],), events)[0, 0])
    if state != bool(sample.current_pressed):
        raise ValueError("actual expert action state disagrees with manifest")
    return future_states[0]


def prepare_video_feature(frame: np.ndarray, *,
                          preprocess, road_config,
                          size: int = SIZE) -> np.ndarray:
    """Shared four-channel UINT8 cache, both variants reuse identical RGB."""
    canonical = normalize_game_frame(frame, target_size=(360, 800))
    safe = canonical
    if preprocess.mask_touch_area:
        safe = mask_area(safe, preprocess.touch_roi)
    for roi in preprocess.mask_rois:
        safe = mask_area(safe, roi)
    rgb = cv2.cvtColor(
        cv2.resize(safe, (size, size), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    hsv = cv2.resize(
        extract_road_mask(safe, road_config), (size, size),
        interpolation=cv2.INTER_AREA,
    )
    return np.concatenate((rgb.transpose(2, 0, 1), hsv[None]), axis=0)


def cache_required_frames(samples, *, preprocess, road_config) -> dict[str, dict[int, np.ndarray]]:
    """Decode only requested source-frame indices; one sequential pass/video."""
    requested: dict[str, set[int]] = defaultdict(set)
    for sample in samples:
        requested[sample.video].update(map(int, sample.input_frame_indices))
    features: dict[str, dict[int, np.ndarray]] = {}
    for video, indices in sorted(requested.items()):
        path = ROOT / video
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open expert video: {path}")
        wanted = set(indices)
        if min(wanted) < 0:
            raise ValueError(f"negative requested frame index: {video}")
        result: dict[int, np.ndarray] = {}
        index = 0
        try:
            while len(result) < len(wanted) and index <= max(wanted):
                ok, frame = capture.read()
                if not ok:
                    break
                if index in wanted:
                    result[index] = prepare_video_feature(
                        frame, preprocess=preprocess, road_config=road_config,
                    )
                index += 1
        finally:
            capture.release()
        if len(result) != len(wanted):
            missing = wanted - set(result)
            raise RuntimeError(f"incomplete video decode {video}: {sorted(missing)[:10]}")
        features[video] = result
        print(f"cached {video}: {len(result)} input frames", flush=True)
    return features


class TemporalExpertDataset(Dataset):
    def __init__(self, samples, features, event_map,
                 *, mode: str, horizon_ms: float = HORIZON_MS) -> None:
        if mode not in ("rgb", "rgb_hsv"):
            raise ValueError("mode must be rgb or rgb_hsv")
        self.samples = list(samples)
        self.mode = mode
        self.features = features
        self.event_map = event_map
        self.horizon_ms = horizon_ms
        self.labels = [
            float(validate_sample(s, event_map[s.video], horizon_ms=horizon_ms))
            for s in self.samples
        ]
        self.controls = [
            observed_state_features(s.input_timestamps_ms, event_map[s.video])
            for s in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        channels = 3 if self.mode == "rgb" else 4
        images = np.stack([
            self.features[sample.video][int(i)][:channels]
            for i in sample.input_frame_indices
        ], axis=0).astype(np.float32) / 255.0
        return (
            torch.from_numpy(images),
            torch.from_numpy(self.controls[index]),
            torch.tensor(self.labels[index], dtype=torch.float32),
        )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    count = 0
    correct = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, control, target in loader:
            images,control,target = (
                images.to(device), control.to(device), target.to(device)
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images, control)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            if training:
                loss.backward()
                optimizer.step()
            batch_count = int(target.numel())
            count += batch_count
            loss_sum += float(loss.item()) * batch_count
            correct += int(((logits >= 0.0) == (target >= 0.5)).sum().item())
    if count == 0:
        raise ValueError("no dataset samples")
    return {"bce": loss_sum/count, "accuracy": correct/count, "samples": count}


def evaluate_sequences(model, dataset, *, batch_size: int, device,
                       event_map, split_videos) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    values = []
    with torch.inference_mode():
        for images, control, target in loader:
            probs = torch.sigmoid(model(
                images.to(device), control.to(device)
            )).cpu().numpy()
            values.extend(map(float, probs))
    by_video: dict[str, list[tuple[object,float]]] = defaultdict(list)
    for sample, prob in zip(dataset.samples, values):
        by_video[sample.video].append((sample, prob))
    evaluations = []
    baseline_evals = []
    for video in split_videos:
        rows = sorted(by_video.get(video, []), key=lambda x:x[0].target_timestamp_ms)
        if not rows:
            raise ValueError(f"evaluation video without sample: {video}")
        actual = event_map[video]
        transitions = [
            Transition(video, event.timestamp_ms, event.pressed)
            for event in actual[1:]
        ]
        releases = [
            ReleaseSegment(video, a.timestamp_ms, b.timestamp_ms)
            for a, b in zip(actual, actual[1:])
            if not a.pressed and b.pressed
        ]
        # Predictions represent action at t+100ms, so sequence points
        # MUST use the target timestamp, NOT the observation timestamp.
        prediction_points = [
            SequencePoint(video, s.target_timestamp_ms, p) for s,p in rows
        ]
        persistence_points = [
            SequencePoint(video, s.target_timestamp_ms, float(s.current_pressed))
            for s,_ in rows
        ]
        evaluations.append(evaluate_sequence(
            prediction_points, transitions, releases,
            threshold=0.5, tolerance_ms=100.0,
        ))
        baseline_evals.append(evaluate_sequence(
            persistence_points, transitions, releases,
            threshold=0.5, tolerance_ms=100.0,
        ))
    return {
        "candidate": combine_evaluations(evaluations).summary(),
        "persistence": combine_evaluations(baseline_evals).summary(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=ROOT/"configs/train_v4c2_temporal_v2.yaml")
    ap.add_argument("--samples", type=Path,
                    default=ROOT/"data/processed/v3/samples.jsonl")
    ap.add_argument("--labels-dir", type=Path,
                    default=ROOT/"data/processed/v3/labels")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--modes", nargs="+", choices=("rgb","rgb_hsv"),
                    default=("rgb","rgb_hsv"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", choices=("cpu","cuda"), default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="short functional test; metrics NOT comparative")
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        ap.error("epochs and batch-size must be >=1")
    if len(set(args.modes)) != len(args.modes):
        ap.error("duplicate modes")
    if not args.output_dir.is_dir():
        ap.error(f"--output-dir must already exist: {args.output_dir}")
    for mode in args.modes:
        for name in (f"tiny_policy_{mode}.pt", f"tiny_policy_{mode}.json"):
            if (args.output_dir/name).exists():
                ap.error(f"refusing to overwrite: {args.output_dir/name}")

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if tuple(raw["dataset"]["prediction_horizons_ms"]) != (100,200,300):
        ap.error("expected V4-C2 100/200/300ms source labels")
    if int(raw["model"]["frame_stack"]) != 5:
        ap.error("expected 5-frame V4-C2 input")
    if not bool(raw["preprocess"]["mask_touch_area"]):
        ap.error("touch ROI masking is required to prevent label leakage")
    split = load_video_split(args.config)
    all_samples = load_v3_samples(args.samples)
    partitions = partition_samples(all_samples, split)
    if args.smoke:
        partitions = {
            name: rows[:64] if name=="validation" else rows[:128]
            for name, rows in partitions.items() if name != "test"
        }
    else:
        partitions = {k:v for k,v in partitions.items() if k in ("train","validation")}
    selected = partitions["train"] + partitions["validation"]
    videos = {s.video for s in selected}
    events = {video: label_events(args.labels_dir,video) for video in videos}
    preprocess = preprocess_config_from_mapping(raw)
    road_cfg, _, _ = load_road_config(ROOT/"configs/geometry_pseudo_labels.yaml")
    # Validate every target and causal state before expensive decoding.
    for sample in selected:
        validate_sample(sample, events[sample.video])
    print(f"preflight: train={len(partitions['train'])}, "
          f"val={len(partitions['validation'])}, videos={len(videos)}, "
          f"smoke={args.smoke}", flush=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache = cache_required_frames(
        selected, preprocess=preprocess, road_config=road_cfg
    )
    reports = []
    for mode in args.modes:
        # Equal initial seeds and the same cached RGB for a paired comparison.
        torch.manual_seed(42)
        channels = 3 if mode=="rgb" else 4
        model = TinyTemporalPolicy(image_channels=channels).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=0.0001
        )
        train_ds = TemporalExpertDataset(
            partitions["train"],cache,events,mode=mode
        )
        val_ds = TemporalExpertDataset(
            partitions["validation"],cache,events,mode=mode
        )
        sample_weights = build_sample_weights(
            partitions["train"],load_sampling_config(args.config)
        )
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights),
            replacement=True, generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            train_ds,batch_size=args.batch_size,sampler=sampler,num_workers=0
        )
        val_loader = DataLoader(
            val_ds,batch_size=args.batch_size,shuffle=False,num_workers=0
        )
        best_loss = float("inf")
        best_state = None
        history = []
        for epoch in range(1, 2 if args.smoke else args.epochs+1):
            train = run_epoch(model,train_loader,device,optimizer)
            validation = run_epoch(model,val_loader,device)
            history.append({"epoch":epoch,"train":train,"validation":validation})
            print(f"{mode} epoch={epoch} train_bce={train['bce']:.4f} "
                  f"val_bce={validation['bce']:.4f} "
                  f"val_acc={validation['accuracy']:.4f}",flush=True)
            if validation["bce"] < best_loss:
                best_loss = validation["bce"]
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name,tensor in model.state_dict().items()
                }
        assert best_state is not None
        model.load_state_dict(best_state)
        sequence = evaluate_sequences(
            model,val_ds,batch_size=args.batch_size,device=device,
            event_map=events,
            split_videos=(
                [v for v in split.validation if v in videos]
                if not args.smoke else [partitions["validation"][0].video]
            ),
        )
        result = {
            "model":"tiny_cnn_gru", "mode":mode,
            "model_not_deployable":True, "expert_teacher_forced":True,
            "smoke_unrepresentative":args.smoke,
            "split":split.name, "train_videos":list(split.train),
            "validation_videos":list(split.validation),
            "test_videos_not_evaluated":list(split.test),
            "target":"expert_PRESS_at_t_plus_100ms",
            "input":"five frames with prior/executed action-state features",
            "visual_size":[SIZE,SIZE],
            "touch_masking":True,
            "selection":"lowest validation BCE at fixed 0.5 threshold",
            "history":history,
            "validation_sequence":sequence,
        }
        torch.save(best_state,args.output_dir/f"tiny_policy_{mode}.pt")
        (args.output_dir/f"tiny_policy_{mode}.json").write_text(
            json.dumps(result,ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8",
        )
        reports.append(result)
        print(json.dumps({
            "mode":mode, "best_val_bce":best_loss,
            "transition_f1":sequence["candidate"]["transition"]["all"]["f1"],
            "persistence_transition_f1":sequence["persistence"]["transition"]["all"]["f1"],
            "short_release_recall":sequence["candidate"]["release_segment_recall"]["short_100_300ms"]["recall"],
        }),flush=True)
    print("Saved only experimental checkpoints and reports to",args.output_dir)


if __name__=="__main__":
    main()
