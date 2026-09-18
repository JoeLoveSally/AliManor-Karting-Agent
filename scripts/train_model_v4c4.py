#!/usr/bin/env python3
"""Train V4-C4: temporal-delta current-action plus first-event-time policy."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_model_v3 import make_loader, select_device  # noqa: E402
from train_model_v4c1 import should_stop_early, write_json_atomic  # noqa: E402
from train_model_v4c2 import (  # noqa: E402
    weighted_edge_risk_loss,
    weighted_heading_loss,
    weighted_lateral_loss,
)

from karting_agent.model.event_time import build_event_time_model  # noqa: E402
from karting_agent.model.temporal_delta import (  # noqa: E402
    initialize_current_rgb_delta_first_conv,
    transform_temporal_input_torch,
    validate_temporal_input_representation,
)
from karting_agent.train.event_time_dataset import (  # noqa: E402
    EventTimeKartRelativeVideoDataset,
)
from karting_agent.train.evaluator import BinaryMetricAccumulator  # noqa: E402
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.kart_relative_labels import (  # noqa: E402
    load_kart_relative_pseudo_labels,
)
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
    parser = argparse.ArgumentParser(description="Train V4-C4 event-time policy.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_v4c4_event_time.yaml",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed"
            / "v4c3_temporal_delta_dense"
            / "samples.jsonl"
        ),
    )
    parser.add_argument("--artifact-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    return parser.parse_args()


def load_raw_config(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    return raw


def _event_metrics(
    confusion: np.ndarray,
    timing_errors: list[float],
    *,
    no_event_class: int,
) -> dict[str, object]:
    classes = int(confusion.shape[0])
    per_class_f1: list[float] = []
    for index in range(classes):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum()) - tp
        fn = int(confusion[index, :].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class_f1.append(f1)

    target_event = int(confusion[:no_event_class, :].sum())
    predicted_event = int(confusion[:, :no_event_class].sum())
    matched_event = int(confusion[:no_event_class, :no_event_class].sum())
    event_precision = (
        matched_event / predicted_event if predicted_event else 0.0
    )
    event_recall = matched_event / target_event if target_event else 0.0
    event_f1 = (
        2.0 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )

    errors = np.asarray(timing_errors, dtype=np.float64)
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum())
        if confusion.sum()
        else 0.0,
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "per_class_f1": per_class_f1,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "timing_matched": int(errors.size),
        "timing_bias_ms": float(errors.mean()) if errors.size else 0.0,
        "timing_mae_ms": float(np.abs(errors).mean()) if errors.size else 0.0,
        "timing_p95_abs_ms": (
            float(np.percentile(np.abs(errors), 95)) if errors.size else 0.0
        ),
        "confusion": confusion.tolist(),
    }


def run_epoch(
    model,
    data_loader,
    device,
    *,
    frame_stack: int,
    input_representation: str,
    action_weight: float,
    event_time_weight: float,
    lateral_weight: float,
    heading_weight: float,
    edge_risk_weight: float,
    event_time_classes: int,
    no_event_class: int,
    bin_ms: float,
    optimizer=None,
    max_batches: int | None = None,
) -> dict[str, object]:
    import torch
    from torch.nn import functional as F

    training = optimizer is not None
    model.train(training)
    action_metric = BinaryMetricAccumulator(0.5)
    edge_metric = BinaryMetricAccumulator(0.5)
    confusion = np.zeros(
        (event_time_classes, event_time_classes),
        dtype=np.int64,
    )
    timing_errors: list[float] = []
    totals = {
        "loss": 0.0,
        "primary_loss": 0.0,
        "current_action_loss": 0.0,
        "event_time_loss": 0.0,
        "lateral_loss": 0.0,
        "heading_loss": 0.0,
        "edge_risk_loss": 0.0,
    }
    total_samples = 0
    relation_weight_sum = 0.0
    lateral_abs_error_sum = 0.0
    heading_angle_weighted_sum = 0.0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = batch["input"].to(device=device, dtype=torch.float32)
            inputs = transform_temporal_input_torch(
                inputs,
                frame_stack=frame_stack,
                representation=input_representation,
            )
            action_target = batch["current_action_target"].to(
                device=device,
                dtype=torch.float32,
            )
            event_target = batch["event_time_target"].to(
                device=device,
                dtype=torch.long,
            )
            event_delay = batch["event_time_delay_ms"].to(
                device=device,
                dtype=torch.float32,
            )
            relation_weights = batch["relation_weight"].to(
                device=device,
                dtype=torch.float32,
            )
            lateral_target = batch["lateral_target"].to(
                device=device,
                dtype=torch.float32,
            )
            heading_target = batch["heading_error_target"].to(
                device=device,
                dtype=torch.float32,
            )
            edge_target = batch["edge_risk_target"].to(
                device=device,
                dtype=torch.float32,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            (
                action_logit,
                event_logits,
                lateral_prediction,
                heading_prediction,
                edge_logit,
            ) = model(inputs)
            action_loss = F.binary_cross_entropy_with_logits(
                action_logit,
                action_target,
            )
            event_loss = F.cross_entropy(event_logits, event_target)
            lateral_loss = weighted_lateral_loss(
                lateral_prediction,
                lateral_target,
                relation_weights,
            )
            heading_loss, heading_unit = weighted_heading_loss(
                heading_prediction,
                heading_target,
                relation_weights,
            )
            edge_loss = weighted_edge_risk_loss(
                edge_logit,
                edge_target,
                relation_weights,
            )
            primary_loss = (
                action_weight * action_loss
                + event_time_weight * event_loss
            )
            loss = (
                primary_loss
                + lateral_weight * lateral_loss
                + heading_weight * heading_loss
                + edge_risk_weight * edge_loss
            )
            if training:
                loss.backward()
                optimizer.step()

            batch_size = int(inputs.shape[0])
            total_samples += batch_size
            for name, value in (
                ("loss", loss),
                ("primary_loss", primary_loss),
                ("current_action_loss", action_loss),
                ("event_time_loss", event_loss),
                ("lateral_loss", lateral_loss),
                ("heading_loss", heading_loss),
                ("edge_risk_loss", edge_loss),
            ):
                totals[name] += float(value.detach().item()) * batch_size

            action_prob = torch.sigmoid(action_logit).detach().cpu().numpy()
            action_metric.update(
                action_prob,
                action_target.detach().cpu().numpy(),
            )
            predicted_class = event_logits.argmax(dim=1)
            expected_np = event_target.detach().cpu().numpy()
            predicted_np = predicted_class.detach().cpu().numpy()
            np.add.at(confusion, (expected_np, predicted_np), 1)

            delay_np = event_delay.detach().cpu().numpy()
            both_event = np.logical_and(
                expected_np != no_event_class,
                predicted_np != no_event_class,
            )
            if np.any(both_event):
                midpoint_ms = (predicted_np[both_event] + 0.5) * bin_ms
                timing_errors.extend(
                    (midpoint_ms - delay_np[both_event]).tolist()
                )

            weights = relation_weights.reshape(-1)
            relation_weight_sum += float(weights.sum().detach().item())
            lateral_abs_error_sum += float(
                (
                    torch.abs(
                        lateral_prediction.reshape(-1)
                        - lateral_target.reshape(-1)
                    )
                    * weights
                )
                .sum()
                .detach()
                .item()
            )
            heading_target_unit = F.normalize(
                heading_target,
                dim=1,
                eps=1e-6,
            )
            dots = (
                (heading_unit * heading_target_unit)
                .sum(dim=1)
                .clamp(-1.0, 1.0)
            )
            heading_errors = 0.5 * torch.rad2deg(torch.acos(dots))
            heading_angle_weighted_sum += float(
                (heading_errors * weights).sum().detach().item()
            )

            labeled = weights.detach().cpu().numpy() > 0
            if np.any(labeled):
                edge_metric.update(
                    torch.sigmoid(edge_logit).detach().cpu().numpy()[labeled],
                    edge_target.detach().cpu().numpy()[labeled],
                )

    if total_samples == 0:
        raise ValueError("data loader produced no samples")
    result = {name: value / total_samples for name, value in totals.items()}
    action = action_metric.result().to_dict()
    event = _event_metrics(
        confusion,
        timing_errors,
        no_event_class=no_event_class,
    )
    result.update(
        {
            "current_action": action,
            "event_time": event,
            "lateral_mae": (
                lateral_abs_error_sum / relation_weight_sum
                if relation_weight_sum
                else 0.0
            ),
            "heading_mean_error_deg": (
                heading_angle_weighted_sum / relation_weight_sum
                if relation_weight_sum
                else 0.0
            ),
            "edge_risk": edge_metric.result().to_dict(),
        }
    )
    return result


def print_summary(prefix: str, summary: dict[str, object]) -> None:
    action = summary["current_action"]
    event = summary["event_time"]
    edge = summary["edge_risk"]
    print(
        f"{prefix}: loss={summary['loss']:.4f} "
        f"primary_loss={summary['primary_loss']:.4f} "
        f"action_loss={summary['current_action_loss']:.4f} "
        f"event_loss={summary['event_time_loss']:.4f} "
        f"action_f1={action['f1']:.3f} "
        f"event_acc={event['accuracy']:.3f} "
        f"event_macro_f1={event['macro_f1']:.3f} "
        f"event_presence_f1={event['event_f1']:.3f} "
        f"event_timing_mae={event['timing_mae_ms']:.1f}ms "
        f"event_timing_bias={event['timing_bias_ms']:+.1f}ms "
        f"lat_mae={summary['lateral_mae']:.3f}w "
        f"head_err={summary['heading_mean_error_deg']:.1f}deg "
        f"risk_f1={edge['f1']:.3f}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    config_path = args.config.resolve()
    raw = load_raw_config(config_path)
    model_config = raw.get("model", {})
    train_config = raw.get("train", {})
    loss_config = raw.get("loss", {})
    supervision_config = raw.get("kart_relative_supervision", {})
    event_config = raw.get("event_time", {})
    artifact_config = raw.get("artifact", {})
    if not all(
        isinstance(value, dict)
        for value in (
            model_config,
            train_config,
            loss_config,
            supervision_config,
            event_config,
            artifact_config,
        )
    ):
        raise ValueError("V4-C4 config sections must be mappings")

    loop = load_train_loop_config(config_path)
    sampling = load_sampling_config(config_path)
    split = load_video_split(config_path)
    preprocess = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)

    architecture = str(model_config.get("architecture", "mobilenet_v3_small"))
    frame_stack = int(model_config.get("frame_stack", 5))
    visual_feature_dim = int(model_config.get("visual_feature_dim", 128))
    hidden_dim = int(model_config.get("hidden_dim", 128))
    pretrained = bool(model_config.get("pretrained", True)) and not args.no_pretrained
    input_representation = validate_temporal_input_representation(
        str(
            model_config.get(
                "input_representation",
                "current_rgb_plus_adjacent_deltas",
            )
        )
    )
    max_horizon_ms = float(event_config.get("max_horizon_ms", 300.0))
    bin_ms = float(event_config.get("bin_ms", 50.0))
    bin_count = int(round(max_horizon_ms / bin_ms))
    if bin_count < 1 or abs(bin_count * bin_ms - max_horizon_ms) > 1e-6:
        raise ValueError("event_time max_horizon_ms must be divisible by bin_ms")
    event_time_classes = bin_count + 1
    no_event_class = bin_count

    weights = {
        "action_weight": float(loss_config.get("current_action_weight", 1.0)),
        "event_time_weight": float(loss_config.get("event_time_weight", 1.0)),
        "lateral_weight": float(loss_config.get("lateral_weight", 0.03)),
        "heading_weight": float(loss_config.get("heading_weight", 0.03)),
        "edge_risk_weight": float(loss_config.get("edge_risk_weight", 0.01)),
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError("loss weights must be >= 0")

    relation_path = (
        ROOT
        / str(
            supervision_config.get(
                "labels",
                "data/processed/v4c2_temporal_v2/kart_relative_labels.jsonl",
            )
        )
    ).resolve()
    labels_dir = (
        ROOT
        / str(
            event_config.get(
                "labels_dir",
                "data/processed/v4c3_temporal_delta_dense/labels",
            )
        )
    ).resolve()
    relation_labels = load_kart_relative_pseudo_labels(relation_path)
    samples = load_v3_samples(args.samples.resolve())
    partitions = partition_samples(samples, split)

    random.seed(loop.seed)
    np.random.seed(loop.seed)
    torch.manual_seed(loop.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(loop.seed)
    device = select_device(torch)
    print(f"Device: {device}", flush=True)
    print(
        f"V4-C4 event-time: classes={event_time_classes} "
        f"bins={bin_count} bin_ms={bin_ms:g} max_horizon={max_horizon_ms:g}ms "
        f"input_representation={input_representation}",
        flush=True,
    )

    dataset_names = (
        ("train", "validation")
        if args.skip_test
        else ("train", "validation", "test")
    )
    datasets = {
        name: EventTimeKartRelativeVideoDataset(
            partitions[name],
            relation_labels=relation_labels,
            labels_dir=labels_dir,
            project_root=ROOT,
            max_horizon_ms=max_horizon_ms,
            bin_ms=bin_ms,
            preprocess_config=preprocess,
            cache_root=cache_root,
            require_cache=args.require_cache,
        )
        for name in dataset_names
    }

    base_weights = build_sample_weights(partitions["train"], sampling)
    generator = torch.Generator().manual_seed(loop.seed)
    sampler = WeightedRandomSampler(
        base_weights,
        num_samples=len(base_weights),
        replacement=sampling.replacement,
        generator=generator,
    )
    num_workers = loop.num_workers if args.num_workers is None else args.num_workers
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    loaders = {
        "train": make_loader(
            DataLoader,
            datasets["train"],
            batch_size=loop.batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            sampler=sampler,
        ),
        "validation": make_loader(
            DataLoader,
            datasets["validation"],
            batch_size=loop.batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        ),
    }
    if not args.skip_test:
        loaders["test"] = make_loader(
            DataLoader,
            datasets["test"],
            batch_size=loop.batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    model = build_event_time_model(
        architecture,
        frame_stack=frame_stack,
        pretrained=pretrained,
        event_time_classes=event_time_classes,
        visual_feature_dim=visual_feature_dim,
        hidden_dim=hidden_dim,
    )
    if (
        input_representation == "current_rgb_plus_adjacent_deltas"
        and pretrained
    ):
        initialize_current_rgb_delta_first_conv(
            model,
            architecture=architecture,
            frame_stack=frame_stack,
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=loop.learning_rate,
        weight_decay=loop.weight_decay,
    )

    kwargs = {
        "frame_stack": frame_stack,
        "input_representation": input_representation,
        "event_time_classes": event_time_classes,
        "no_event_class": no_event_class,
        "bin_ms": bin_ms,
        **weights,
    }
    if args.smoke:
        train_summary = run_epoch(
            model,
            loaders["train"],
            device,
            optimizer=optimizer,
            max_batches=1,
            **kwargs,
        )
        validation_summary = run_epoch(
            model,
            loaders["validation"],
            device,
            max_batches=1,
            **kwargs,
        )
        print_summary("smoke/train", train_summary)
        print_summary("smoke/validation", validation_summary)
        for dataset in datasets.values():
            dataset.close()
        print("V4-C4 event-time smoke test passed.", flush=True)
        return 0

    max_epochs = loop.epochs if args.epochs is None else int(args.epochs)
    patience = (
        int(train_config.get("early_stopping_patience", 4))
        if args.early_stopping_patience is None
        else int(args.early_stopping_patience)
    )
    artifact_name = args.artifact_name or str(
        artifact_config.get("name", "mobilenet_v3_small_v4c4_event_time")
    )
    artifact_dir = ROOT / "artifacts" / "models" / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    history_path = artifact_dir / "history.json"
    state_path = artifact_dir / "training_state.json"

    best_loss = float("inf")
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, object]] = []
    try:
        for epoch in range(1, max_epochs + 1):
            started = time.perf_counter()
            train_summary = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                **kwargs,
            )
            validation_summary = run_epoch(
                model,
                loaders["validation"],
                device,
                **kwargs,
            )
            elapsed = time.perf_counter() - started
            print_summary(f"epoch {epoch:02d}/train", train_summary)
            print_summary(f"epoch {epoch:02d}/validation", validation_summary)
            print(f"epoch {epoch:02d}/time: {elapsed:.1f}s", flush=True)

            validation_loss = float(validation_summary["primary_loss"])
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                without_improvement = 0
                torch.save(model.state_dict(), model_path)
            else:
                without_improvement += 1
            history.append(
                {
                    "epoch": epoch,
                    "elapsed_seconds": elapsed,
                    "train": train_summary,
                    "validation": validation_summary,
                    "best_epoch": best_epoch,
                    "best_validation_primary_loss": best_loss,
                }
            )
            write_json_atomic(history_path, history)
            write_json_atomic(
                state_path,
                {
                    "status": "running",
                    "last_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_primary_loss": best_loss,
                    "epochs_without_improvement": without_improvement,
                },
            )
            if should_stop_early(without_improvement, patience):
                print(
                    f"Early stopping: best_epoch={best_epoch} "
                    f"best_primary_loss={best_loss:.6f}",
                    flush=True,
                )
                break

        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        test_summary = None
        if not args.skip_test:
            test_summary = run_epoch(
                model,
                loaders["test"],
                device,
                **kwargs,
            )
            print_summary("test", test_summary)

        write_json_atomic(
            artifact_dir / "metadata.json",
            {
                "model_family": "event_time_v4c4",
                "architecture": architecture,
                "frame_stack": frame_stack,
                "visual_feature_dim": visual_feature_dim,
                "hidden_dim": hidden_dim,
                "input_representation": input_representation,
                "event_time_bin_ms": bin_ms,
                "event_time_max_horizon_ms": max_horizon_ms,
                "event_time_classes": event_time_classes,
                "no_event_class": no_event_class,
                "best_epoch": best_epoch,
                "best_validation_primary_loss": best_loss,
                "loss_weights": weights,
                "config": str(config_path),
                "test_evaluated": not args.skip_test,
                "test": test_summary,
            },
        )
        write_json_atomic(
            state_path,
            {
                "status": "complete",
                "last_epoch": history[-1]["epoch"],
                "best_epoch": best_epoch,
                "best_validation_primary_loss": best_loss,
            },
        )
    finally:
        for dataset in datasets.values():
            dataset.close()

    print(f"Artifact: {artifact_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
