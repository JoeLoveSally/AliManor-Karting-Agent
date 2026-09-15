#!/usr/bin/env python3
"""Train v4-A2: per-frame shared CNN -> ordered temporal MLP -> KEEP/SWITCH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the exact v4-A training/evaluation loop so the temporal fusion module is
# the only intended model-level change in this ablation.
from train_model_v4a import (  # noqa: E402
    control_horizon,
    load_raw_config,
    make_loader,
    print_summary,
    run_epoch,
    select_device,
    target_horizons,
)

from karting_agent.model.temporal_mlp_state_conditioned import (  # noqa: E402
    build_temporal_mlp_state_conditioned_model,
)
from karting_agent.train.frame_cache import frame_cache_root_from_config  # noqa: E402
from karting_agent.train.state_conditioned_dataset import (  # noqa: E402
    SequentialStateConditionedVideoDataset,
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
        description=(
            "Train v4-A2 with explicit per-frame CNN features and ordered MLP fusion."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "train_v4a2.yaml"
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "v3" / "samples.jsonl",
        help="v4-A2 intentionally reuses the exact v3/v4-A 200ms manifest.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required; install with: python -m pip install -e ".[train]"'
        ) from exc

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
    if not isinstance(model_config, dict) or not isinstance(dataset_config, dict):
        raise ValueError("model and dataset config must be mappings")
    if not isinstance(loss_config, dict):
        raise ValueError("loss config must be a mapping")

    architecture = str(model_config.get("architecture", "mobilenet_v3_small"))
    frame_stack = int(model_config.get("frame_stack", 5))
    history_ms = float(dataset_config.get("history_ms", 200))
    pretrained = bool(model_config.get("pretrained", True)) and not args.no_pretrained
    visual_feature_dim = int(model_config.get("visual_feature_dim", 128))
    temporal_hidden_dim = int(model_config.get("temporal_hidden_dim", 128))
    state_embedding_dim = int(model_config.get("state_embedding_dim", 8))
    policy_hidden_dim = int(model_config.get("policy_hidden_dim", 128))
    counterfactual_train = bool(
        dataset_config.get("counterfactual_train_states", True)
    )
    auxiliary_action_weight = float(loss_config.get("auxiliary_action_weight", 0.5))
    if auxiliary_action_weight < 0:
        raise ValueError("loss.auxiliary_action_weight must be >= 0")

    num_workers = loop_config.num_workers if args.num_workers is None else args.num_workers
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    samples = load_v3_samples(args.samples.resolve())
    if not samples:
        raise ValueError("sample manifest is empty")
    frame_widths = {len(sample.input_frame_indices) for sample in samples}
    if frame_widths != {frame_stack}:
        raise ValueError(
            f"sample frame widths {sorted(frame_widths)} do not match frame_stack={frame_stack}"
        )
    target_widths = {len(sample.target_states) for sample in samples}
    if target_widths != {len(horizons)}:
        raise ValueError(
            f"sample target widths {sorted(target_widths)} do not match horizons={horizons}"
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
        f"V4-A2: explicit_time_axis=True temporal_fusion=mlp "
        f"history={history_ms:g}ms frame_stack={frame_stack} horizons={horizons} "
        f"control_horizon={selected_control_horizon:g}ms "
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
        + (f"{cache_root} (required={args.require_cache})" if cache_root else "disabled"),
        flush=True,
    )

    datasets = {
        name: SequentialStateConditionedVideoDataset(
            partitions[name],
            frame_stack=frame_stack,
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

    model = build_temporal_mlp_state_conditioned_model(
        architecture,
        pretrained=pretrained,
        frame_stack=frame_stack,
        horizon_count=len(horizons),
        visual_feature_dim=visual_feature_dim,
        temporal_hidden_dim=temporal_hidden_dim,
        state_embedding_dim=state_embedding_dim,
        policy_hidden_dim=policy_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=loop_config.learning_rate,
        weight_decay=loop_config.weight_decay,
    )

    try:
        if args.smoke:
            train_summary = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                auxiliary_action_weight=auxiliary_action_weight,
                primary_output_index=primary_index,
                max_batches=1,
            )
            validation_summary = run_epoch(
                model,
                loaders["validation"],
                device,
                auxiliary_action_weight=auxiliary_action_weight,
                primary_output_index=primary_index,
                max_batches=1,
            )
            print_summary("smoke/train", train_summary, horizons, primary_index)
            print_summary(
                "smoke/validation", validation_summary, horizons, primary_index
            )
            print("V4-A2 smoke test passed.", flush=True)
            return 0

        artifact_config = raw.get("artifact", {})
        if not isinstance(artifact_config, dict):
            raise ValueError("artifact config must be a mapping")
        artifact_name = str(artifact_config.get("name", "mobilenet_v3_small_v4a2"))
        artifact_dir = ROOT / "artifacts" / "models" / artifact_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "model.pt"

        best_val_switch_loss = float("inf")
        best_epoch = 0
        history: list[dict[str, object]] = []
        for epoch in range(1, loop_config.epochs + 1):
            started = time.perf_counter()
            train_summary = run_epoch(
                model,
                loaders["train"],
                device,
                optimizer=optimizer,
                auxiliary_action_weight=auxiliary_action_weight,
                primary_output_index=primary_index,
            )
            validation_summary = run_epoch(
                model,
                loaders["validation"],
                device,
                auxiliary_action_weight=auxiliary_action_weight,
                primary_output_index=primary_index,
            )
            elapsed = time.perf_counter() - started
            print_summary(f"epoch {epoch:02d}/train", train_summary, horizons, primary_index)
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
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        test_summary = run_epoch(
            model,
            loaders["test"],
            device,
            auxiliary_action_weight=auxiliary_action_weight,
            primary_output_index=primary_index,
        )
        print_summary("test", test_summary, horizons, primary_index)

        metadata = {
            "model_family": "temporal_mlp_state_conditioned_v4a2",
            "architecture": architecture,
            "frame_stack": frame_stack,
            "explicit_time_axis": True,
            "temporal_fusion": "ordered_concat_mlp",
            "prediction_horizons_ms": list(horizons),
            "control_horizon_ms": selected_control_horizon,
            "visual_feature_dim": visual_feature_dim,
            "temporal_hidden_dim": temporal_hidden_dim,
            "state_embedding_dim": state_embedding_dim,
            "policy_hidden_dim": policy_hidden_dim,
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
