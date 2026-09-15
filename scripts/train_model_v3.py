#!/usr/bin/env python3
"""Train the v3 state-conditioned KEEP/SWITCH policy."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from karting_agent.model.state_conditioned import build_state_conditioned_model  # noqa: E402
from karting_agent.train.evaluator import BinaryMetricAccumulator  # noqa: E402
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.state_conditioned_dataset import (  # noqa: E402
    StateConditionedVideoDataset,
    load_v3_samples,
)
from karting_agent.train.trainer import (  # noqa: E402
    build_sample_weights,
    load_sampling_config,
    load_train_loop_config,
    load_video_split,
    partition_samples,
)
from karting_agent.vision.preprocess import preprocess_config_from_mapping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the v3 state-conditioned transition policy."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train_v3.yaml",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "v3" / "samples.jsonl",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def load_raw_config(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("train config root must be a mapping")
    return raw


def select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def target_horizons(raw: dict[str, object]) -> tuple[float, ...]:
    dataset = raw.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("dataset config must be a mapping")
    values = dataset.get("prediction_horizons_ms")
    if not isinstance(values, list) or not values:
        raise ValueError("v3 requires dataset.prediction_horizons_ms")
    horizons = tuple(float(value) for value in values)
    if tuple(sorted(horizons)) != horizons or len(set(horizons)) != len(horizons):
        raise ValueError("prediction_horizons_ms must be sorted and unique")
    return horizons


def control_horizon(raw: dict[str, object], horizons: tuple[float, ...]) -> float:
    dataset = raw.get("dataset", {})
    assert isinstance(dataset, dict)
    value = float(dataset.get("control_horizon_ms", horizons[0]))
    if value not in horizons:
        raise ValueError("control_horizon_ms must be one of prediction_horizons_ms")
    return value


def false_positive_rate(metric: BinaryMetricAccumulator) -> float:
    target_negative = metric.samples - metric.target_positive
    return metric.false_positive / target_negative if target_negative else 0.0


def run_epoch_v3(
    model,
    data_loader,
    device,
    *,
    auxiliary_action_weight: float,
    primary_output_index: int,
    optimizer=None,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> dict[str, object]:
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: python -m pip install -e ".[train]"'
        ) from exc

    if auxiliary_action_weight < 0:
        raise ValueError("auxiliary_action_weight must be >= 0")
    if primary_output_index < 0:
        raise ValueError("primary_output_index must be >= 0")

    training = optimizer is not None
    model.train(training)
    switch_metrics: list[BinaryMetricAccumulator] | None = None
    future_metrics: list[BinaryMetricAccumulator] | None = None
    primary_transition = BinaryMetricAccumulator(threshold)
    primary_short = BinaryMetricAccumulator(threshold)
    total_loss = 0.0
    total_switch_loss = 0.0
    total_future_loss = 0.0
    total_samples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = batch["input"].to(device=device, dtype=torch.float32)
            current_pressed = batch["current_pressed"].to(device=device)
            switch_targets = batch["switch_target"].to(
                device=device, dtype=torch.float32
            )
            future_targets = batch["future_action_target"].to(
                device=device, dtype=torch.float32
            )
            if training:
                optimizer.zero_grad(set_to_none=True)

            switch_logits, future_logits = model(inputs, current_pressed)
            if switch_logits.ndim != 2 or future_logits.ndim != 2:
                raise ValueError("v3 model outputs must be BxH tensors")
            if switch_logits.shape != switch_targets.shape:
                raise ValueError(
                    "switch target/logit shape mismatch: "
                    f"{switch_targets.shape} vs {switch_logits.shape}"
                )
            if future_logits.shape != future_targets.shape:
                raise ValueError(
                    "future-action target/logit shape mismatch: "
                    f"{future_targets.shape} vs {future_logits.shape}"
                )
            output_count = int(switch_logits.shape[1])
            if primary_output_index >= output_count:
                raise ValueError("primary_output_index exceeds v3 horizon count")

            switch_loss = F.binary_cross_entropy_with_logits(
                switch_logits, switch_targets
            )
            future_loss = F.binary_cross_entropy_with_logits(
                future_logits, future_targets
            )
            loss = switch_loss + auxiliary_action_weight * future_loss
            if training:
                loss.backward()
                optimizer.step()

            batch_size = int(inputs.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_switch_loss += float(switch_loss.detach().item()) * batch_size
            total_future_loss += float(future_loss.detach().item()) * batch_size
            total_samples += batch_size

            switch_probabilities = torch.sigmoid(switch_logits).detach().cpu().numpy()
            future_probabilities = torch.sigmoid(future_logits).detach().cpu().numpy()
            switch_values = switch_targets.detach().cpu().numpy()
            future_values = future_targets.detach().cpu().numpy()

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
                batch["near_transition_by_horizon"].detach().cpu().numpy().astype(bool)
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
    return {
        "loss": total_loss / total_samples,
        "switch_loss": total_switch_loss / total_samples,
        "future_action_loss": total_future_loss / total_samples,
        "primary_output_index": primary_output_index,
        "primary_switch": switch_results[primary_output_index],
        "primary_switch_false_positive_rate": false_positive_rate(primary_metric),
        "primary_transition": primary_transition.result().to_dict(),
        "primary_short_correction": primary_short.result().to_dict(),
        "switch_outputs": switch_results,
        "future_action_outputs": future_results,
    }


def print_summary(
    prefix: str,
    summary: dict[str, object],
    horizons: tuple[float, ...],
    primary_index: int,
) -> None:
    primary = summary["primary_switch"]
    transition = summary["primary_transition"]
    short = summary["primary_short_correction"]
    switch_outputs = summary["switch_outputs"]
    future_outputs = summary["future_action_outputs"]
    primary_horizon = horizons[primary_index]
    switch_bits = ", ".join(
        f"sw{horizon:g}_f1={metrics['f1']:.3f}"
        for horizon, metrics in zip(horizons, switch_outputs, strict=False)
    )
    future_bits = ", ".join(
        f"act{horizon:g}_f1={metrics['f1']:.3f}"
        for horizon, metrics in zip(horizons, future_outputs, strict=False)
    )
    print(
        f"{prefix}: loss={summary['loss']:.4f}, "
        f"switch_loss={summary['switch_loss']:.4f}, "
        f"aux_loss={summary['future_action_loss']:.4f}, "
        f"switch_h{primary_horizon:g}_p={primary['precision']:.3f}, "
        f"r={primary['recall']:.3f}, f1={primary['f1']:.3f}, "
        f"target_pos={primary['target_positive_rate']:.3f}, "
        f"pred_pos={primary['predicted_positive_rate']:.3f}, "
        f"stable_fpr={summary['primary_switch_false_positive_rate']:.3f}, "
        f"transition_f1={transition['f1']:.3f}, short_f1={short['f1']:.3f}, "
        f"{switch_bits}, {future_bits}",
        flush=True,
    )


def make_loader(
    DataLoader,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    sampler=None,
):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": False,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def main() -> int:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: python -m pip install -e ".[train]"'
        ) from exc

    raw = load_raw_config(args.config)
    horizons = target_horizons(raw)
    selected_control_horizon = control_horizon(raw, horizons)
    primary_index = horizons.index(selected_control_horizon)
    loop_config = load_train_loop_config(args.config)
    sampling_config = load_sampling_config(args.config)
    split = load_video_split(args.config)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)

    model_config = raw.get("model", {})
    dataset_config = raw.get("dataset", {})
    loss_config = raw.get("loss", {})
    if not isinstance(model_config, dict) or not isinstance(dataset_config, dict):
        raise ValueError("model and dataset config must be mappings")
    if not isinstance(loss_config, dict):
        raise ValueError("loss config must be a mapping")

    architecture = str(model_config.get("architecture", "mobilenet_v3_small"))
    frame_stack = int(model_config.get("frame_stack", 5))
    history_ms = float(dataset_config.get("history_ms", 200))
    pretrained = bool(model_config.get("pretrained", True)) and not args.no_pretrained
    visual_feature_dim = int(model_config.get("visual_feature_dim", 128))
    state_embedding_dim = int(model_config.get("state_embedding_dim", 8))
    hidden_dim = int(model_config.get("hidden_dim", 128))
    counterfactual_train = bool(
        dataset_config.get("counterfactual_train_states", True)
    )
    auxiliary_action_weight = float(loss_config.get("auxiliary_action_weight", 0.5))
    if auxiliary_action_weight < 0:
        raise ValueError("loss.auxiliary_action_weight must be >= 0")

    num_workers = loop_config.num_workers if args.num_workers is None else args.num_workers
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    samples = load_v3_samples(args.samples)
    if not samples:
        raise ValueError("sample manifest is empty")
    target_widths = {len(sample.target_states) for sample in samples}
    if target_widths != {len(horizons)}:
        raise ValueError(
            f"sample target widths {sorted(target_widths)} do not match horizons {horizons}"
        )
    partitions = partition_samples(samples, split)

    random.seed(loop_config.seed)
    np.random.seed(loop_config.seed)
    torch.manual_seed(loop_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(loop_config.seed)

    device = select_device(torch)
    print(f"Device: {device}", flush=True)
    print(
        f"V3: history={history_ms:g}ms frame_stack={frame_stack} "
        f"horizons={horizons} control_horizon={selected_control_horizon:g}ms "
        f"counterfactual_train={counterfactual_train}",
        flush=True,
    )
    print(
        f"Split {split.name}: train={len(partitions['train'])}, "
        f"validation={len(partitions['validation'])}, "
        f"test={len(partitions['test'])}",
        flush=True,
    )
    print(
        "Frame cache: "
        + (
            f"{cache_root} (required={args.require_cache})"
            if cache_root is not None
            else "disabled"
        ),
        flush=True,
    )

    datasets = {
        "train": StateConditionedVideoDataset(
            partitions["train"],
            project_root=ROOT,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=args.require_cache,
            counterfactual_states=counterfactual_train,
        ),
        "validation": StateConditionedVideoDataset(
            partitions["validation"],
            project_root=ROOT,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=args.require_cache,
            counterfactual_states=False,
        ),
        "test": StateConditionedVideoDataset(
            partitions["test"],
            project_root=ROOT,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=args.require_cache,
            counterfactual_states=False,
        ),
    }

    generator = torch.Generator().manual_seed(loop_config.seed)
    base_weights = build_sample_weights(partitions["train"], sampling_config)
    if counterfactual_train:
        train_weights = np.repeat(np.asarray(base_weights, dtype=np.float64), 2)
    else:
        train_weights = np.asarray(base_weights, dtype=np.float64)
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

    model = build_state_conditioned_model(
        architecture,
        frame_stack=frame_stack,
        pretrained=pretrained,
        horizon_count=len(horizons),
        visual_feature_dim=visual_feature_dim,
        state_embedding_dim=state_embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=loop_config.learning_rate,
        weight_decay=loop_config.weight_decay,
    )

    try:
        if args.smoke:
            train_summary = run_epoch_v3(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                auxiliary_action_weight=auxiliary_action_weight,
                primary_output_index=primary_index,
                max_batches=1,
            )
            validation_summary = run_epoch_v3(
                model,
                loaders["validation"],
                device,
                auxiliary_action_weight=auxiliary_action_weight,
                primary_output_index=primary_index,
                max_batches=1,
            )
            print_summary(
                "smoke/train", train_summary, horizons, primary_index
            )
            print_summary(
                "smoke/validation", validation_summary, horizons, primary_index
            )
            print("V3 smoke test passed.", flush=True)
            return 0

        artifact_config = raw.get("artifact", {})
        if not isinstance(artifact_config, dict):
            raise ValueError("artifact config must be a mapping")
        artifact_name = str(
            artifact_config.get("name", "mobilenet_v3_small_v3")
        )
        artifact_dir = ROOT / "artifacts" / "models" / artifact_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "model.pt"

        best_val_switch_loss = float("inf")
        best_epoch = 0
        history: list[dict[str, object]] = []
        for epoch in range(1, loop_config.epochs + 1):
            started = time.perf_counter()
            train_summary = run_epoch_v3(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                auxiliary_action_weight=auxiliary_action_weight,
                primary_output_index=primary_index,
            )
            validation_summary = run_epoch_v3(
                model,
                loaders["validation"],
                device,
                auxiliary_action_weight=auxiliary_action_weight,
                primary_output_index=primary_index,
            )
            elapsed = time.perf_counter() - started
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
            history.append(
                {
                    "epoch": epoch,
                    "elapsed_seconds": elapsed,
                    "train": train_summary,
                    "validation": validation_summary,
                }
            )
            val_switch_loss = float(validation_summary["switch_loss"])
            if val_switch_loss < best_val_switch_loss:
                best_val_switch_loss = val_switch_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)

        try:
            state_dict = torch.load(
                model_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        test_summary = run_epoch_v3(
            model,
            loaders["test"],
            device,
            auxiliary_action_weight=auxiliary_action_weight,
            primary_output_index=primary_index,
        )
        print_summary("test", test_summary, horizons, primary_index)

        metadata = {
            "model_family": "state_conditioned_transition_v3",
            "architecture": architecture,
            "frame_stack": frame_stack,
            "prediction_horizons_ms": list(horizons),
            "control_horizon_ms": selected_control_horizon,
            "visual_feature_dim": visual_feature_dim,
            "state_embedding_dim": state_embedding_dim,
            "hidden_dim": hidden_dim,
            "auxiliary_action_weight": auxiliary_action_weight,
            "counterfactual_train_states": counterfactual_train,
            "input_size": int(model_config.get("input_size", 224)),
            "pretrained": pretrained,
            "split": split.name,
            "best_epoch": best_epoch,
            "best_validation_switch_loss": best_val_switch_loss,
            "test": test_summary,
            "config": raw,
        }
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        (artifact_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(f"Artifact: {artifact_dir}", flush=True)
        return 0
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    raise SystemExit(main())
