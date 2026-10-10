"""Immutable metadata checks for independent expert-video Test evaluation."""
from __future__ import annotations


def require_frozen_checkpoint(meta: dict, *, mode: str, split) -> None:
    """Reject an incompatible or potentially test-exposed experiment artifact.

    This validates declarations in training metadata; it cannot cryptographically
    prove the history of a source file or weights saved by an untrusted process.
    """
    if not isinstance(meta, dict):
        raise ValueError("checkpoint report must be a JSON object")
    if mode not in ("action_only", "rgb", "rgb_hsv"):
        raise ValueError("unsupported evaluation mode")
    checks = (
        (meta.get("model") == "tiny_cnn_gru", "unsupported model family"),
        (meta.get("mode") == mode, "checkpoint mode mismatch"),
        (meta.get("split") == split.name, "split name mismatch"),
        (meta.get("train_videos") == list(split.train), "train split mismatch"),
        (meta.get("validation_videos") == list(split.validation),
         "validation split mismatch"),
        (meta.get("test_videos_not_evaluated") == list(split.test),
         "test holdout provenance mismatch"),
        (meta.get("smoke_unrepresentative") is False,
         "smoke checkpoint is not eligible for test"),
        (meta.get("model_not_deployable") is True,
         "missing offline experimental checkpoint marker"),
        (meta.get("expert_teacher_forced") is True,
         "checkpoint must come from expert-supervised training"),
        (meta.get("target") == "expert_PRESS_at_t_plus_100ms",
         "unexpected target horizon"),
        (meta.get("visual_size") == [96, 96], "unexpected visual size"),
        (meta.get("touch_masking") is True, "touch mask must be enabled"),
        (meta.get("selection") == "lowest validation BCE at fixed 0.5 threshold",
         "checkpoint selection protocol mismatch"),
        (bool(meta.get("history")), "missing validation training history"),
    )
    for valid, message in checks:
        if not valid:
            raise ValueError(message)
    expected_ablation = mode == "action_only"
    if bool(meta.get("visual_ablation", False)) != expected_ablation:
        raise ValueError("visual ablation flag mismatch")
    if bool(meta.get("image_channels_are_constant_zero", False)) != expected_ablation:
        raise ValueError("image input ablation flag mismatch")
    if set(split.test) & (set(split.train) | set(split.validation)):
        raise ValueError("test videos overlap previously used videos")
