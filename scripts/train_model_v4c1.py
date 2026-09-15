#!/usr/bin/env python3
"""Train v4-C1: v3 control path + high-confidence road-axis supervision."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_model_v3 import (  # noqa: E402
    control_horizon,
    false_positive_rate,
    load_raw_config,
    make_loader,
    select_device,
    target_horizons,
)

from karting_agent.model.axis_supervised import build_axis_supervised_model  # noqa: E402
from karting_agent.train.axis_labels import (  # noqa: E402
    AxisSupervisedVideoDataset,
    load_axis_pseudo_labels,
)
from karting_agent.train.evaluator import BinaryMetricAccumulator  # noqa: E402
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.state_conditioned_dataset import load_v3_samples  # noqa: E402
from karting_agent.train.trainer import (  # noqa: E402
    build_sample_weights,
    load_sampling_config,
    load_train_loop_config,
    load_video_split,
    partition_samples,
)
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v4-C1 road-axis-supervised policy.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "train_v4c1.yaml"
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "v3" / "samples.jsonl",
    )
    parser.add_argument("--axis-labels", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def weighted_axis_loss(axis_vector, axis_target, axis_weight):
    import torch
    from torch.nn import functional as F

    prediction = F.normalize(axis_vector, dim=1, eps=1e-6)
    target = F.normalize(axis_target, dim=1, eps=1e-6)
    per_sample = 1.0 - torch.sum(prediction * target, dim=1)
    weights = axis_weight.reshape(-1).to(dtype=per_sample.dtype)
    denominator = weights.sum()
    if float(denominator.detach().item()) <= 0.0:
        return per_sample.sum() * 0.0, prediction, weights
    return (per_sample * weights).sum() / denominator, prediction, weights


def run_epoch(
    model,
    data_loader,
    device,
    *,
    auxiliary_action_weight: float,
    axis_weight_multiplier: float,
    primary_output_index: int,
    optimizer=None,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, object]:
    import torch
    from torch.nn import functional as F

    training = optimizer is not None
    model.train(training)
    switch_metrics: list[BinaryMetricAccumulator] | None = None
    future_metrics: list[BinaryMetricAccumulator] | None = None
    primary_transition = BinaryMetricAccumulator(threshold)
    primary_short = BinaryMetricAccumulator(threshold)
    total_loss = 0.0
    total_switch_loss = 0.0
    total_future_loss = 0.0
    total_axis_loss = 0.0
    total_samples = 0
    axis_angle_weighted_sum = 0.0
    axis_weight_sum = 0.0
    axis_labeled_samples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = batch["input"].to(device=device, dtype=torch.float32)
            current_pressed = batch["current_pressed"].to(device=device)
            switch_targets = batch["switch_target"].to(device=device, dtype=torch.float32)
            future_targets = batch["future_action_target"].to(device=device, dtype=torch.float32)
            axis_targets = batch["axis_target"].to(device=device, dtype=torch.float32)
            axis_weights = batch["axis_weight"].to(device=device, dtype=torch.float32)
            if training:
                optimizer.zero_grad(set_to_none=True)

            switch_logits, future_logits, axis_vector = model(inputs, current_pressed)
            switch_loss = F.binary_cross_entropy_with_logits(switch_logits, switch_targets)
            future_loss = F.binary_cross_entropy_with_logits(future_logits, future_targets)
            axis_loss, axis_prediction, normalized_axis_weights = weighted_axis_loss(
                axis_vector, axis_targets, axis_weights
            )
            loss = (
                switch_loss
                + auxiliary_action_weight * future_loss
                + axis_weight_multiplier * axis_loss
            )
            if training:
                loss.backward()
                optimizer.step()

            batch_size = int(inputs.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_switch_loss += float(switch_loss.detach().item()) * batch_size
            total_future_loss += float(future_loss.detach().item()) * batch_size
            total_axis_loss += float(axis_loss.detach().item()) * batch_size
            total_samples += batch_size

            target_unit = F.normalize(axis_targets, dim=1, eps=1e-6)
            dots = torch.sum(axis_prediction * target_unit, dim=1).clamp(-1.0, 1.0)
            # axial vector encodes 2*theta, so orientation error is half the
            # angular error between encoded unit vectors.
            angle_error_deg = 0.5 * torch.rad2deg(torch.acos(dots))
            axis_angle_weighted_sum += float(
                (angle_error_deg * normalized_axis_weights).sum().detach().item()
            )
            axis_weight_sum += float(normalized_axis_weights.sum().detach().item())
            axis_labeled_samples += int((normalized_axis_weights > 0).sum().item())

            switch_probabilities = torch.sigmoid(switch_logits).detach().cpu().numpy()
            future_probabilities = torch.sigmoid(future_logits).detach().cpu().numpy()
            switch_values = switch_targets.detach().cpu().numpy()
            future_values = future_targets.detach().cpu().numpy()
            output_count = int(switch_logits.shape[1])
            if switch_metrics is None:
                switch_metrics = [BinaryMetricAccumulator(threshold) for _ in range(output_count)]
                future_metrics = [BinaryMetricAccumulator(threshold) for _ in range(output_count)]
            assert future_metrics is not None
            for output_index in range(output_count):
                switch_metrics[output_index].update(
                    switch_probabilities[:, output_index], switch_values[:, output_index]
                )
                future_metrics[output_index].update(
                    future_probabilities[:, output_index], future_values[:, output_index]
                )

            transition_masks = batch["near_transition_by_horizon"].detach().cpu().numpy().astype(bool)
            short_masks = batch["near_short_correction_by_horizon"].detach().cpu().numpy().astype(bool)
            primary_probabilities = switch_probabilities[:, primary_output_index]
            primary_values = switch_values[:, primary_output_index]
            primary_transition.update(
                primary_probabilities[transition_masks[:, primary_output_index]],
                primary_values[transition_masks[:, primary_output_index]],
            )
            primary_short.update(
                primary_probabilities[short_masks[:, primary_output_index]],
                primary_values[short_masks[:, primary_output_index]],
            )

    if total_samples == 0 or switch_metrics is None or future_metrics is None:
        raise ValueError("data loader produced no samples")
    switch_results = [metric.result().to_dict() for metric in switch_metrics]
    future_results = [metric.result().to_dict() for metric in future_metrics]
    primary_metric = switch_metrics[primary_output_index]
    return {
        "loss": total_loss / total_samples,
        "switch_loss": total_switch_loss / total_samples,
        "future_action_loss": total_future_loss / total_samples,
        "axis_loss": total_axis_loss / total_samples,
        "axis_labeled_samples": axis_labeled_samples,
        "axis_weight_sum": axis_weight_sum,
        "axis_mean_error_deg": axis_angle_weighted_sum / axis_weight_sum if axis_weight_sum else 0.0,
        "primary_switch": switch_results[primary_output_index],
        "primary_switch_false_positive_rate": false_positive_rate(primary_metric),
        "primary_transition": primary_transition.result().to_dict(),
        "primary_short_correction": primary_short.result().to_dict(),
        "switch_outputs": switch_results,
        "future_action_outputs": future_results,
    }


def print_summary(prefix: str, summary: dict[str, object], horizons, primary_index: int) -> None:
    primary = summary["primary_switch"]
    transition = summary["primary_transition"]
    short = summary["primary_short_correction"]
    switch_outputs = summary["switch_outputs"]
    future_outputs = summary["future_action_outputs"]
    switch_bits = ", ".join(
        f"sw{h:g}_f1={metrics['f1']:.3f}"
        for h, metrics in zip(horizons, switch_outputs, strict=False)
    )
    future_bits = ", ".join(
        f"act{h:g}_f1={metrics['f1']:.3f}"
        for h, metrics in zip(horizons, future_outputs, strict=False)
    )
    print(
        f"{prefix}: loss={summary['loss']:.4f}, switch_loss={summary['switch_loss']:.4f}, "
        f"aux_loss={summary['future_action_loss']:.4f}, axis_loss={summary['axis_loss']:.4f}, "
        f"axis_err={summary['axis_mean_error_deg']:.2f}deg, axis_n={summary['axis_labeled_samples']}, "
        f"switch_h{horizons[primary_index]:g}_p={primary['precision']:.3f}, "
        f"r={primary['recall']:.3f}, f1={primary['f1']:.3f}, "
        f"transition_f1={transition['f1']:.3f}, short_f1={short['f1']:.3f}, "
        f"{switch_bits}, {future_bits}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    config_path = args.config.resolve()
    raw = load_raw_config(config_path)
    horizons = target_horizons(raw)
    selected_control_horizon = control_horizon(raw, horizons)
    primary_index = horizons.index(selected_control_horizon)
    loop_config = load_train_loop_config(config_path)
    sampling_config = load_sampling_config(config_path)
    split = load_video_split(config_path)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)

    model_config = raw.get("model", {})
    dataset_config = raw.get("dataset", {})
    loss_config = raw.get("loss", {})
    axis_config = raw.get("axis_supervision", {})
    if not all(isinstance(value, dict) for value in (model_config, dataset_config, loss_config, axis_config)):
        raise ValueError("model/dataset/loss/axis_supervision configs must be mappings")

    architecture = str(model_config.get("architecture", "mobilenet_v3_small"))
    frame_stack = int(model_config.get("frame_stack", 5))
    pretrained = bool(model_config.get("pretrained", True)) and not args.no_pretrained
    visual_feature_dim = int(model_config.get("visual_feature_dim", 128))
    state_embedding_dim = int(model_config.get("state_embedding_dim", 8))
    hidden_dim = int(model_config.get("hidden_dim", 128))
    counterfactual_train = bool(dataset_config.get("counterfactual_train_states", True))
    auxiliary_action_weight = float(loss_config.get("auxiliary_action_weight", 0.5))
    axis_weight_multiplier = float(loss_config.get("axis_weight", 0.1))
    axis_labels_path = (
        args.axis_labels.resolve()
        if args.axis_labels
        else (ROOT / str(axis_config.get("labels", "data/processed/v4c1/axis_labels.jsonl"))).resolve()
    )
    axis_labels = load_axis_pseudo_labels(axis_labels_path)

    num_workers = loop_config.num_workers if args.num_workers is None else args.num_workers
    samples = load_v3_samples(args.samples.resolve())
    partitions = partition_samples(samples, split)

    random.seed(loop_config.seed)
    np.random.seed(loop_config.seed)
    torch.manual_seed(loop_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(loop_config.seed)
    device = select_device(torch)
    print(f"Device: {device}", flush=True)
    print(
        f"V4-C1: v3_control=True axis_weight={axis_weight_multiplier:g} "
        f"horizons={horizons} control_horizon={selected_control_horizon:g}ms "
        f"counterfactual_train={counterfactual_train}",
        flush=True,
    )
    print(f"Axis labels: {axis_labels_path} ({len(axis_labels)} unique frames)", flush=True)

    datasets = {
        name: AxisSupervisedVideoDataset(
            partitions[name],
            axis_labels=axis_labels,
            project_root=ROOT,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=args.require_cache,
            counterfactual_states=(counterfactual_train if name == "train" else False),
        )
        for name in ("train", "validation", "test")
    }

    generator = torch.Generator().manual_seed(loop_config.seed)
    base_weights = build_sample_weights(partitions["train"], sampling_config)
    train_weights = (
        np.repeat(np.asarray(base_weights, dtype=np.float64), 2)
        if counterfactual_train
        else np.asarray(base_weights, dtype=np.float64)
    )
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=sampling_config.replacement,
        generator=generator,
    )
    pin_memory = device.type == "cuda"
    loaders = {
        "train": make_loader(
            DataLoader,
            datasets["train"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
        ),
        "validation": make_loader(
            DataLoader,
            datasets["validation"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": make_loader(
            DataLoader,
            datasets["test"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    model = build_axis_supervised_model(
        architecture,
        frame_stack=frame_stack,
        pretrained=pretrained,
        horizon_count=len(horizons),
        visual_feature_dim=visual_feature_dim,
        state_embedding_dim=state_embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=loop_config.learning_rate, weight_decay=loop_config.weight_decay
    )

    try:
        if args.smoke:
            train_summary = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                auxiliary_action_weight=auxiliary_action_weight,
                axis_weight_multiplier=axis_weight_multiplier,
                primary_output_index=primary_index,
                max_batches=1,
            )
            validation_summary = run_epoch(
                model,
                loaders["validation"],
                device,
                auxiliary_action_weight=auxiliary_action_weight,
                axis_weight_multiplier=axis_weight_multiplier,
                primary_output_index=primary_index,
                max_batches=1,
            )
            print_summary("smoke/train", train_summary, horizons, primary_index)
            print_summary("smoke/validation", validation_summary, horizons, primary_index)
            print("V4-C1 smoke test passed.", flush=True)
            return 0

        artifact = raw.get("artifact", {})
        if not isinstance(artifact, dict):
            raise ValueError("artifact config must be a mapping")
        artifact_dir = ROOT / "artifacts" / "models" / str(artifact.get("name", "mobilenet_v3_small_v4c1"))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "model.pt"
        best_val_switch_loss = float("inf")
        best_epoch = 0
        history = []
        for epoch in range(1, loop_config.epochs + 1):
            started = time.perf_counter()
            train_summary = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                auxiliary_action_weight=auxiliary_action_weight,
                axis_weight_multiplier=axis_weight_multiplier,
                primary_output_index=primary_index,
            )
            validation_summary = run_epoch(
                model,
                loaders["validation"],
                device,
                auxiliary_action_weight=auxiliary_action_weight,
                axis_weight_multiplier=axis_weight_multiplier,
                primary_output_index=primary_index,
            )
            elapsed = time.perf_counter() - started
            print_summary(f"epoch {epoch:02d}/train", train_summary, horizons, primary_index)
            print_summary(f"epoch {epoch:02d}/validation", validation_summary, horizons, primary_index)
            print(f"epoch {epoch:02d}/time: {elapsed:.1f}s", flush=True)
            history.append(
                {"epoch": epoch, "elapsed_seconds": elapsed, "train": train_summary, "validation": validation_summary}
            )
            val_switch_loss = float(validation_summary["switch_loss"])
            if val_switch_loss < best_val_switch_loss:
                best_val_switch_loss = val_switch_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)

        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        test_summary = run_epoch(
            model,
            loaders["test"],
            device,
            auxiliary_action_weight=auxiliary_action_weight,
            axis_weight_multiplier=axis_weight_multiplier,
            primary_output_index=primary_index,
        )
        print_summary("test", test_summary, horizons, primary_index)
        metadata = {
            "model_family": "state_conditioned_axis_v4c1",
            "architecture": architecture,
            "frame_stack": frame_stack,
            "prediction_horizons_ms": list(horizons),
            "control_horizon_ms": selected_control_horizon,
            "visual_feature_dim": visual_feature_dim,
            "state_embedding_dim": state_embedding_dim,
            "hidden_dim": hidden_dim,
            "auxiliary_action_weight": auxiliary_action_weight,
            "axis_weight": axis_weight_multiplier,
            "axis_encoding": "cos_2theta_sin_2theta",
            "axis_labels": str(axis_labels_path),
            "best_epoch": best_epoch,
            "best_validation_switch_loss": best_val_switch_loss,
            "test": test_summary,
            "config": raw,
        }
        (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (artifact_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Artifact: {artifact_dir}", flush=True)
        return 0
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
