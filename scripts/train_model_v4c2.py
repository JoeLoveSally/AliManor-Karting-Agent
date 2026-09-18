#!/usr/bin/env python3
"""Train v4-C2: v3 control plus kart-relative auxiliary supervision."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_model_v3 import (  # noqa: E402
    control_horizon,
    false_positive_rate,
    load_raw_config,
    make_loader,
    select_device,
    target_horizons,
)
from train_model_v4c1 import should_stop_early, write_json_atomic  # noqa: E402

from karting_agent.model.kart_relative_supervised import (  # noqa: E402
    build_kart_relative_model,
)
from karting_agent.model.temporal_delta import (  # noqa: E402
    initialize_current_rgb_delta_first_conv,
    transform_temporal_input_torch,
    validate_temporal_input_representation,
)
from karting_agent.train.evaluator import BinaryMetricAccumulator  # noqa: E402
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.kart_relative_labels import (  # noqa: E402
    KartRelativeSupervisedVideoDataset,
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
    parser = argparse.ArgumentParser(description="Train v4-C2 kart-relative policy.")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "train_v4c2.yaml"
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "v3" / "samples.jsonl",
    )
    parser.add_argument("--relation-labels", type=Path, default=None)
    parser.add_argument("--artifact-name", type=str, default=None)
    parser.add_argument("--lateral-weight", type=float, default=None)
    parser.add_argument("--heading-weight", type=float, default=None)
    parser.add_argument("--edge-risk-weight", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Train and validate without loading or evaluating the frozen test split.",
    )
    return parser.parse_args()


def _weighted_mean(per_sample, weights):
    denominator = weights.sum()
    if float(denominator.detach().item()) <= 0.0:
        return per_sample.sum() * 0.0
    return (per_sample * weights).sum() / denominator


def weighted_lateral_loss(prediction, target, weights):
    from torch.nn import functional as F

    per_sample = F.smooth_l1_loss(
        prediction.reshape(-1), target.reshape(-1), reduction="none"
    )
    weights = weights.reshape(-1).to(dtype=per_sample.dtype)
    return _weighted_mean(per_sample, weights)


def weighted_heading_loss(prediction, target, weights):
    from torch.nn import functional as F

    prediction_unit = F.normalize(prediction, dim=1, eps=1e-6)
    target_unit = F.normalize(target, dim=1, eps=1e-6)
    per_sample = 1.0 - (prediction_unit * target_unit).sum(dim=1)
    weights = weights.reshape(-1).to(dtype=per_sample.dtype)
    return _weighted_mean(per_sample, weights), prediction_unit


def weighted_edge_risk_loss(logit, target, weights):
    from torch.nn import functional as F

    per_sample = F.binary_cross_entropy_with_logits(
        logit.reshape(-1), target.reshape(-1), reduction="none"
    )
    weights = weights.reshape(-1).to(dtype=per_sample.dtype)
    return _weighted_mean(per_sample, weights)


def run_epoch(
    model,
    data_loader,
    device,
    *,
    auxiliary_action_weight: float,
    lateral_weight_multiplier: float,
    heading_weight_multiplier: float,
    edge_risk_weight_multiplier: float,
    primary_output_index: int,
    input_representation: str = "raw_rgb_stack",
    frame_stack: int = 5,
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
    edge_metric = BinaryMetricAccumulator(0.5)

    total_samples = 0
    totals = {
        "loss": 0.0,
        "switch_loss": 0.0,
        "future_action_loss": 0.0,
        "lateral_loss": 0.0,
        "heading_loss": 0.0,
        "edge_risk_loss": 0.0,
    }
    relation_weight_sum = 0.0
    relation_labeled_samples = 0
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
            current_pressed = batch["current_pressed"].to(device=device)
            switch_targets = batch["switch_target"].to(
                device=device, dtype=torch.float32
            )
            future_targets = batch["future_action_target"].to(
                device=device, dtype=torch.float32
            )
            relation_weights = batch["relation_weight"].to(
                device=device, dtype=torch.float32
            )
            lateral_targets = batch["lateral_target"].to(
                device=device, dtype=torch.float32
            )
            heading_targets = batch["heading_error_target"].to(
                device=device, dtype=torch.float32
            )
            edge_targets = batch["edge_risk_target"].to(
                device=device, dtype=torch.float32
            )
            if training:
                optimizer.zero_grad(set_to_none=True)

            (
                switch_logits,
                future_logits,
                lateral_prediction,
                heading_prediction,
                edge_logit,
            ) = model(inputs, current_pressed)
            switch_loss = F.binary_cross_entropy_with_logits(
                switch_logits, switch_targets
            )
            future_loss = F.binary_cross_entropy_with_logits(
                future_logits, future_targets
            )
            lateral_loss = weighted_lateral_loss(
                lateral_prediction, lateral_targets, relation_weights
            )
            heading_loss, heading_unit = weighted_heading_loss(
                heading_prediction, heading_targets, relation_weights
            )
            edge_loss = weighted_edge_risk_loss(
                edge_logit, edge_targets, relation_weights
            )
            loss = (
                switch_loss
                + auxiliary_action_weight * future_loss
                + lateral_weight_multiplier * lateral_loss
                + heading_weight_multiplier * heading_loss
                + edge_risk_weight_multiplier * edge_loss
            )
            if training:
                loss.backward()
                optimizer.step()

            batch_size = int(inputs.shape[0])
            total_samples += batch_size
            for name, value in (
                ("loss", loss),
                ("switch_loss", switch_loss),
                ("future_action_loss", future_loss),
                ("lateral_loss", lateral_loss),
                ("heading_loss", heading_loss),
                ("edge_risk_loss", edge_loss),
            ):
                totals[name] += float(value.detach().item()) * batch_size

            weights = relation_weights.reshape(-1)
            relation_weight_sum += float(weights.sum().detach().item())
            relation_labeled_samples += int((weights > 0).sum().item())
            lateral_abs_error_sum += float(
                (
                    torch.abs(
                        lateral_prediction.reshape(-1) - lateral_targets.reshape(-1)
                    )
                    * weights
                )
                .sum()
                .detach()
                .item()
            )
            heading_target_unit = F.normalize(heading_targets, dim=1, eps=1e-6)
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
                    edge_targets.detach().cpu().numpy()[labeled],
                )

            switch_probabilities = (
                torch.sigmoid(switch_logits).detach().cpu().numpy()
            )
            future_probabilities = (
                torch.sigmoid(future_logits).detach().cpu().numpy()
            )
            switch_values = switch_targets.detach().cpu().numpy()
            future_values = future_targets.detach().cpu().numpy()
            output_count = int(switch_logits.shape[1])
            if switch_metrics is None:
                switch_metrics = [
                    BinaryMetricAccumulator(threshold) for _ in range(output_count)
                ]
                future_metrics = [
                    BinaryMetricAccumulator(threshold) for _ in range(output_count)
                ]
            assert future_metrics is not None
            for output_index in range(output_count):
                switch_metrics[output_index].update(
                    switch_probabilities[:, output_index],
                    switch_values[:, output_index],
                )
                future_metrics[output_index].update(
                    future_probabilities[:, output_index],
                    future_values[:, output_index],
                )

            transition_masks = (
                batch["near_transition_by_horizon"]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
            short_masks = (
                batch["near_short_correction_by_horizon"]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )
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
    result = {name: value / total_samples for name, value in totals.items()}
    result.update(
        {
            "relation_labeled_samples": relation_labeled_samples,
            "relation_weight_sum": relation_weight_sum,
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
            "primary_switch": switch_results[primary_output_index],
            "primary_switch_false_positive_rate": false_positive_rate(primary_metric),
            "primary_transition": primary_transition.result().to_dict(),
            "primary_short_correction": primary_short.result().to_dict(),
            "switch_outputs": switch_results,
            "future_action_outputs": future_results,
        }
    )
    return result


def print_summary(
    prefix: str,
    summary: dict[str, object],
    horizons,
    primary_index: int,
) -> None:
    primary = summary["primary_switch"]
    transition = summary["primary_transition"]
    short = summary["primary_short_correction"]
    edge = summary["edge_risk"]
    print(
        f"{prefix}: loss={summary['loss']:.4f}, "
        f"switch_loss={summary['switch_loss']:.4f}, "
        f"aux_loss={summary['future_action_loss']:.4f}, "
        f"lat_loss={summary['lateral_loss']:.4f}, "
        f"head_loss={summary['heading_loss']:.4f}, "
        f"risk_loss={summary['edge_risk_loss']:.4f}, "
        f"lat_mae={summary['lateral_mae']:.3f}w, "
        f"head_err={summary['heading_mean_error_deg']:.1f}deg, "
        f"risk_f1={edge['f1']:.3f}, "
        f"relation_n={summary['relation_labeled_samples']}, "
        f"switch_h{horizons[primary_index]:g}_p={primary['precision']:.3f}, "
        f"r={primary['recall']:.3f}, f1={primary['f1']:.3f}, "
        f"transition_f1={transition['f1']:.3f}, "
        f"short_f1={short['f1']:.3f}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    config_path = args.config.resolve()
    raw = load_raw_config(config_path)
    horizons = target_horizons(raw)
    control_ms = control_horizon(raw, horizons)
    primary_index = horizons.index(control_ms)
    loop_config = load_train_loop_config(config_path)
    sampling_config = load_sampling_config(config_path)
    split = load_video_split(config_path)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)

    model_config = raw.get("model", {})
    train_config = raw.get("train", {})
    dataset_config = raw.get("dataset", {})
    loss_config = raw.get("loss", {})
    supervision_config = raw.get("kart_relative_supervision", {})
    if not all(
        isinstance(value, dict)
        for value in (
            model_config,
            train_config,
            dataset_config,
            loss_config,
            supervision_config,
        )
    ):
        raise ValueError("v4-C2 config sections must be mappings")

    model_family = str(
        model_config.get("family", "state_conditioned_kart_relative_v4c2")
    )
    if model_family not in {
        "state_conditioned_kart_relative_v4c2",
        "state_conditioned_kart_relative_v4c3_delta",
    }:
        raise ValueError(f"unsupported v4-C2/C3 model family: {model_family}")
    input_representation = validate_temporal_input_representation(
        str(model_config.get("input_representation", "raw_rgb_stack"))
    )
    if (
        model_family == "state_conditioned_kart_relative_v4c3_delta"
        and input_representation != "current_rgb_plus_adjacent_deltas"
    ):
        raise ValueError(
            "state_conditioned_kart_relative_v4c3_delta requires "
            "current_rgb_plus_adjacent_deltas"
        )

    architecture = str(model_config.get("architecture", "mobilenet_v3_small"))
    frame_stack = int(model_config.get("frame_stack", 5))
    pretrained = bool(model_config.get("pretrained", True)) and not args.no_pretrained
    visual_feature_dim = int(model_config.get("visual_feature_dim", 128))
    state_embedding_dim = int(model_config.get("state_embedding_dim", 8))
    hidden_dim = int(model_config.get("hidden_dim", 128))
    counterfactual_train = bool(
        dataset_config.get("counterfactual_train_states", True)
    )
    auxiliary_action_weight = float(loss_config.get("auxiliary_action_weight", 0.5))
    lateral_weight = (
        float(loss_config.get("lateral_weight", 0.03))
        if args.lateral_weight is None
        else args.lateral_weight
    )
    heading_weight = (
        float(loss_config.get("heading_weight", 0.03))
        if args.heading_weight is None
        else args.heading_weight
    )
    edge_risk_weight = (
        float(loss_config.get("edge_risk_weight", 0.01))
        if args.edge_risk_weight is None
        else args.edge_risk_weight
    )
    weights = (
        auxiliary_action_weight,
        lateral_weight,
        heading_weight,
        edge_risk_weight,
    )
    if any(value < 0 for value in weights):
        raise ValueError("loss weights must be >= 0")

    max_epochs = loop_config.epochs if args.epochs is None else int(args.epochs)
    patience = (
        int(train_config.get("early_stopping_patience", 0))
        if args.early_stopping_patience is None
        else args.early_stopping_patience
    )
    min_delta = (
        float(train_config.get("early_stopping_min_delta", 0.0))
        if args.early_stopping_min_delta is None
        else args.early_stopping_min_delta
    )
    if max_epochs < 1 or patience < 0 or min_delta < 0:
        raise ValueError("invalid epoch/early-stopping configuration")
    num_workers = (
        loop_config.num_workers if args.num_workers is None else args.num_workers
    )
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    relation_labels_path = (
        args.relation_labels.resolve()
        if args.relation_labels
        else (
            ROOT
            / str(
                supervision_config.get(
                    "labels",
                    "data/processed/v4c2/kart_relative_labels.jsonl",
                )
            )
        ).resolve()
    )
    relation_labels = load_kart_relative_pseudo_labels(relation_labels_path)
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
        f"{model_family}: lateral_weight={lateral_weight:g} "
        f"heading_weight={heading_weight:g} "
        f"edge_risk_weight={edge_risk_weight:g} horizons={horizons} "
        f"control_horizon={control_ms:g}ms "
        f"counterfactual_train={counterfactual_train} "
        f"input_representation={input_representation}",
        flush=True,
    )
    print(
        f"Relation labels: {relation_labels_path} "
        f"({len(relation_labels)} unique frames)",
        flush=True,
    )

    dataset_names = (
        ("train", "validation")
        if args.skip_test
        else ("train", "validation", "test")
    )
    datasets = {
        name: KartRelativeSupervisedVideoDataset(
            partitions[name],
            relation_labels=relation_labels,
            project_root=ROOT,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=args.require_cache,
            counterfactual_states=(
                counterfactual_train if name == "train" else False
            ),
        )
        for name in dataset_names
    }
    base_weights = build_sample_weights(partitions["train"], sampling_config)
    train_weights = (
        [weight for weight in base_weights for _ in range(2)]
        if counterfactual_train
        else base_weights
    )
    generator = torch.Generator().manual_seed(loop_config.seed)
    sampler = WeightedRandomSampler(
        train_weights,
        num_samples=len(train_weights),
        replacement=sampling_config.replacement,
        generator=generator,
    )
    loaders = {
        "train": make_loader(
            DataLoader,
            datasets["train"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            sampler=sampler,
        ),
        "validation": make_loader(
            DataLoader,
            datasets["validation"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        ),
    }
    if not args.skip_test:
        loaders["test"] = make_loader(
            DataLoader,
            datasets["test"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    model = build_kart_relative_model(
        architecture,
        frame_stack=frame_stack,
        pretrained=pretrained,
        horizon_count=len(horizons),
        visual_feature_dim=visual_feature_dim,
        state_embedding_dim=state_embedding_dim,
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
        lr=loop_config.learning_rate,
        weight_decay=loop_config.weight_decay,
    )

    kwargs = {
        "auxiliary_action_weight": auxiliary_action_weight,
        "lateral_weight_multiplier": lateral_weight,
        "heading_weight_multiplier": heading_weight,
        "edge_risk_weight_multiplier": edge_risk_weight,
        "primary_output_index": primary_index,
        "input_representation": input_representation,
        "frame_stack": frame_stack,
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
        print_summary("smoke/train", train_summary, horizons, primary_index)
        print_summary(
            "smoke/validation",
            validation_summary,
            horizons,
            primary_index,
        )
        for dataset in datasets.values():
            dataset.close()
        print("V4-C2 smoke test passed.", flush=True)
        return 0

    artifact_config = raw.get("artifact", {})
    if not isinstance(artifact_config, dict):
        raise ValueError("artifact config must be a mapping")
    artifact_name = args.artifact_name or str(
        artifact_config.get("name", "mobilenet_v3_small_v4c2")
    )
    artifact_dir = ROOT / "artifacts" / "models" / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pt"
    history_path = artifact_dir / "history.json"
    training_state_path = artifact_dir / "training_state.json"

    best_loss = float("inf")
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, object]] = []
    try:
        for epoch in range(1, max_epochs + 1):
            start = time.perf_counter()
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
            elapsed = time.perf_counter() - start
            print_summary(
                f"epoch {epoch:02d}/train", train_summary, horizons, primary_index
            )
            print_summary(
                f"epoch {epoch:02d}/validation",
                validation_summary,
                horizons,
                primary_index,
            )
            print(f"epoch {epoch:02d}/time: {elapsed:.1f}s", flush=True)

            validation_switch_loss = float(validation_summary["switch_loss"])
            improved = validation_switch_loss < best_loss - min_delta
            if improved:
                best_loss = validation_switch_loss
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
                    "best_validation_switch_loss": best_loss,
                }
            )
            write_json_atomic(history_path, history)
            write_json_atomic(
                training_state_path,
                {
                    "status": "running",
                    "last_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_switch_loss": best_loss,
                    "epochs_without_improvement": without_improvement,
                },
            )
            if should_stop_early(without_improvement, patience):
                print(
                    f"Early stopping: best_epoch={best_epoch} "
                    f"best_validation_switch_loss={best_loss:.6f} "
                    f"patience={patience}",
                    flush=True,
                )
                break

        try:
            state_dict = torch.load(
                model_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        test_summary = None
        if not args.skip_test:
            test_summary = run_epoch(model, loaders["test"], device, **kwargs)
            print_summary("test", test_summary, horizons, primary_index)
        metadata = {
            "model_family": model_family,
            "architecture": architecture,
            "frame_stack": frame_stack,
            "visual_feature_dim": visual_feature_dim,
            "state_embedding_dim": state_embedding_dim,
            "hidden_dim": hidden_dim,
            "prediction_horizons_ms": list(horizons),
            "control_horizon_ms": control_ms,
            "input_representation": input_representation,
            "counterfactual_train_states": counterfactual_train,
            "relation_labels": str(relation_labels_path),
            "loss_weights": {
                "auxiliary_action_weight": auxiliary_action_weight,
                "lateral_weight": lateral_weight,
                "heading_weight": heading_weight,
                "edge_risk_weight": edge_risk_weight,
            },
            "best_epoch": best_epoch,
            "best_validation_switch_loss": best_loss,
            "test_evaluated": not args.skip_test,
            "test": test_summary,
            "config": str(config_path),
        }
        write_json_atomic(artifact_dir / "metadata.json", metadata)
        write_json_atomic(
            training_state_path,
            {
                "status": "complete",
                "last_epoch": history[-1]["epoch"],
                "best_epoch": best_epoch,
                "best_validation_switch_loss": best_loss,
            },
        )
    finally:
        for dataset in datasets.values():
            dataset.close()

    print(f"Artifact: {artifact_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
