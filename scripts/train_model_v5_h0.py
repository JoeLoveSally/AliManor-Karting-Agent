#!/usr/bin/env python3
"""Train V5-H0 with counterfactual validation and no test-set evaluation.

V5-H0 keeps the v4-C2 architecture and auxiliary supervision, but uses dense
0..300 ms action targets with H0 as the deployed control horizon. Checkpoint
selection is based on counterfactual validation switch loss so the H0 target is
non-degenerate: every validation visual is evaluated once conditioned on
RELEASE and once on PRESS. The frozen test split is intentionally never loaded
or evaluated by this script.
"""

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
    load_raw_config,
    make_loader,
    select_device,
    target_horizons,
)
from train_model_v4c1 import should_stop_early, write_json_atomic  # noqa: E402
from train_model_v4c2 import print_summary, run_epoch  # noqa: E402

from karting_agent.model.kart_relative_supervised import (  # noqa: E402
    build_kart_relative_model,
)
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

EXPECTED_HORIZONS_MS = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V5-H0 with counterfactual validation."
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "train_v5_h0.yaml"
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "processed" / "v5_h0" / "samples.jsonl",
    )
    parser.add_argument("--relation-labels", type=Path, default=None)
    parser.add_argument("--artifact-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def _require_mapping(raw: dict[str, object], key: str) -> dict[str, object]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} config must be a mapping")
    return value


def validate_v5_config(raw: dict[str, object]) -> tuple[float, ...]:
    horizons = target_horizons(raw)
    if horizons != EXPECTED_HORIZONS_MS:
        raise ValueError(
            f"V5-H0 requires horizons={EXPECTED_HORIZONS_MS}, got {horizons}"
        )
    if control_horizon(raw, horizons) != 0.0:
        raise ValueError("V5-H0 requires control_horizon_ms=0")
    dataset_config = _require_mapping(raw, "dataset")
    if not bool(dataset_config.get("counterfactual_train_states", True)):
        raise ValueError("V5-H0 requires counterfactual_train_states=true")
    return horizons


def main() -> int:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    config_path = args.config.resolve()
    raw = load_raw_config(config_path)
    horizons = validate_v5_config(raw)
    primary_index = 0
    loop_config = load_train_loop_config(config_path)
    sampling_config = load_sampling_config(config_path)
    split = load_video_split(config_path)
    preprocess_config = preprocess_config_from_mapping(raw)
    cache_root = frame_cache_root_from_config(raw, ROOT)

    model_config = _require_mapping(raw, "model")
    train_config = _require_mapping(raw, "train")
    loss_config = _require_mapping(raw, "loss")
    supervision_config = _require_mapping(raw, "kart_relative_supervision")

    architecture = str(model_config.get("architecture", "mobilenet_v3_small"))
    frame_stack = int(model_config.get("frame_stack", 5))
    pretrained = bool(model_config.get("pretrained", True)) and not args.no_pretrained
    visual_feature_dim = int(model_config.get("visual_feature_dim", 128))
    state_embedding_dim = int(model_config.get("state_embedding_dim", 8))
    hidden_dim = int(model_config.get("hidden_dim", 128))

    auxiliary_action_weight = float(loss_config.get("auxiliary_action_weight", 0.5))
    lateral_weight = float(loss_config.get("lateral_weight", 0.03))
    heading_weight = float(loss_config.get("heading_weight", 0.03))
    edge_risk_weight = float(loss_config.get("edge_risk_weight", 0.01))
    if any(
        value < 0
        for value in (
            auxiliary_action_weight,
            lateral_weight,
            heading_weight,
            edge_risk_weight,
        )
    ):
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
    num_workers = (
        loop_config.num_workers if args.num_workers is None else int(args.num_workers)
    )
    if max_epochs < 1 or patience < 0 or min_delta < 0 or num_workers < 0:
        raise ValueError("invalid training-loop configuration")

    relation_labels_path = (
        args.relation_labels.resolve()
        if args.relation_labels
        else (ROOT / str(supervision_config["labels"])).resolve()
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
        "V5-H0: "
        f"horizons={horizons} control_horizon=0ms "
        "train_states=counterfactual validation_states=counterfactual "
        "checkpoint=validation_counterfactual.switch_loss test_split=untouched",
        flush=True,
    )
    print(
        f"Relation labels: {relation_labels_path} "
        f"({len(relation_labels)} unique frames)",
        flush=True,
    )

    datasets = {
        "train": KartRelativeSupervisedVideoDataset(
            partitions["train"],
            relation_labels=relation_labels,
            project_root=ROOT,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=args.require_cache,
            counterfactual_states=True,
        ),
        "validation_counterfactual": KartRelativeSupervisedVideoDataset(
            partitions["validation"],
            relation_labels=relation_labels,
            project_root=ROOT,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=args.require_cache,
            counterfactual_states=True,
        ),
        "validation_expert": KartRelativeSupervisedVideoDataset(
            partitions["validation"],
            relation_labels=relation_labels,
            project_root=ROOT,
            preprocess_config=preprocess_config,
            cache_root=cache_root,
            require_cache=args.require_cache,
            counterfactual_states=False,
        ),
    }

    base_weights = build_sample_weights(partitions["train"], sampling_config)
    train_weights = [weight for weight in base_weights for _ in range(2)]
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
        "validation_counterfactual": make_loader(
            DataLoader,
            datasets["validation_counterfactual"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        ),
        "validation_expert": make_loader(
            DataLoader,
            datasets["validation_expert"],
            batch_size=loop_config.batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        ),
    }

    model = build_kart_relative_model(
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

    epoch_kwargs = {
        "auxiliary_action_weight": auxiliary_action_weight,
        "lateral_weight_multiplier": lateral_weight,
        "heading_weight_multiplier": heading_weight,
        "edge_risk_weight_multiplier": edge_risk_weight,
        "primary_output_index": primary_index,
    }

    if args.smoke:
        train_summary = run_epoch(
            model,
            loaders["train"],
            device,
            optimizer=optimizer,
            max_batches=1,
            **epoch_kwargs,
        )
        validation_cf_summary = run_epoch(
            model,
            loaders["validation_counterfactual"],
            device,
            max_batches=1,
            **epoch_kwargs,
        )
        validation_expert_summary = run_epoch(
            model,
            loaders["validation_expert"],
            device,
            max_batches=1,
            **epoch_kwargs,
        )
        print_summary("smoke/train", train_summary, horizons, primary_index)
        print_summary(
            "smoke/validation-counterfactual",
            validation_cf_summary,
            horizons,
            primary_index,
        )
        print_summary(
            "smoke/validation-expert",
            validation_expert_summary,
            horizons,
            primary_index,
        )
        primary_cf = validation_cf_summary["primary_switch"]
        print(
            "smoke/checkpoint: "
            f"switch_loss={validation_cf_summary['switch_loss']:.4f} "
            f"h0_target_positive={primary_cf['target_positive']}/"
            f"{primary_cf['samples']}",
            flush=True,
        )
        for dataset in datasets.values():
            dataset.close()
        print("V5-H0 smoke test passed; frozen test split was not evaluated.", flush=True)
        return 0

    artifact_config = _require_mapping(raw, "artifact")
    artifact_name = args.artifact_name or str(
        artifact_config.get("name", "mobilenet_v3_small_v5_h0")
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
                **epoch_kwargs,
            )
            validation_summary = run_epoch(
                model,
                loaders["validation_counterfactual"],
                device,
                **epoch_kwargs,
            )
            elapsed = time.perf_counter() - start

            print_summary(
                f"epoch {epoch:02d}/train", train_summary, horizons, primary_index
            )
            print_summary(
                f"epoch {epoch:02d}/validation-counterfactual",
                validation_summary,
                horizons,
                primary_index,
            )
            print(f"epoch {epoch:02d}/time: {elapsed:.1f}s", flush=True)

            checkpoint_loss = float(validation_summary["switch_loss"])
            improved = checkpoint_loss < best_loss - min_delta
            if improved:
                best_loss = checkpoint_loss
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
                    "validation_counterfactual": validation_summary,
                    "best_epoch": best_epoch,
                    "best_validation_counterfactual_switch_loss": best_loss,
                }
            )
            write_json_atomic(history_path, history)
            write_json_atomic(
                training_state_path,
                {
                    "status": "running",
                    "last_epoch": epoch,
                    "best_epoch": best_epoch,
                    "checkpoint_metric": "validation_counterfactual.switch_loss",
                    "best_validation_counterfactual_switch_loss": best_loss,
                    "epochs_without_improvement": without_improvement,
                    "test_evaluated": False,
                },
            )

            if should_stop_early(without_improvement, patience):
                print(
                    f"Early stopping: best_epoch={best_epoch} "
                    f"best_validation_counterfactual_switch_loss={best_loss:.6f} "
                    f"patience={patience}",
                    flush=True,
                )
                break

        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        final_validation_cf = run_epoch(
            model,
            loaders["validation_counterfactual"],
            device,
            **epoch_kwargs,
        )
        final_validation_expert = run_epoch(
            model,
            loaders["validation_expert"],
            device,
            **epoch_kwargs,
        )
        print_summary(
            "best/validation-counterfactual",
            final_validation_cf,
            horizons,
            primary_index,
        )
        print_summary(
            "best/validation-expert",
            final_validation_expert,
            horizons,
            primary_index,
        )

        metadata = {
            "model_family": "state_conditioned_kart_relative_v5_h0",
            "architecture": architecture,
            "frame_stack": frame_stack,
            "visual_feature_dim": visual_feature_dim,
            "state_embedding_dim": state_embedding_dim,
            "hidden_dim": hidden_dim,
            "prediction_horizons_ms": list(horizons),
            "control_horizon_ms": 0.0,
            "counterfactual_train_states": True,
            "counterfactual_validation_states": True,
            "checkpoint_metric": "validation_counterfactual.switch_loss",
            "relation_labels": str(relation_labels_path),
            "loss_weights": {
                "auxiliary_action_weight": auxiliary_action_weight,
                "lateral_weight": lateral_weight,
                "heading_weight": heading_weight,
                "edge_risk_weight": edge_risk_weight,
            },
            "best_epoch": best_epoch,
            "best_validation_counterfactual_switch_loss": best_loss,
            "validation_counterfactual": final_validation_cf,
            "validation_expert": final_validation_expert,
            "test_evaluated": False,
            "config": str(config_path),
        }
        write_json_atomic(artifact_dir / "metadata.json", metadata)
        write_json_atomic(
            training_state_path,
            {
                "status": "complete",
                "last_epoch": history[-1]["epoch"],
                "best_epoch": best_epoch,
                "checkpoint_metric": "validation_counterfactual.switch_loss",
                "best_validation_counterfactual_switch_loss": best_loss,
                "test_evaluated": False,
            },
        )
    finally:
        for dataset in datasets.values():
            dataset.close()

    print(f"Artifact: {artifact_dir}", flush=True)
    print("Frozen test split was not evaluated.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
