"""CPU-only tests for frozen-checkpoint test protocol."""
from types import SimpleNamespace
import copy
import pytest
from karting_agent.train.heldout_guard import require_frozen_checkpoint

SPLIT = SimpleNamespace(
    name="v4c2_temporal_v2", train=("trainA.mp4", "trainB.mp4"),
    validation=("valA.mp4",), test=("testA.mp4", "testB.mp4"),
)


def report(mode="rgb"):
    return {
        "model": "tiny_cnn_gru", "mode": mode, "split": SPLIT.name,
        "train_videos": list(SPLIT.train),
        "validation_videos": list(SPLIT.validation),
        "test_videos_not_evaluated": list(SPLIT.test),
        "smoke_unrepresentative": False,
        "model_not_deployable": True, "expert_teacher_forced": True,
        "target": "expert_PRESS_at_t_plus_100ms", "visual_size": [96, 96],
        "touch_masking": True,
        "selection": "lowest validation BCE at fixed 0.5 threshold",
        "history": [{"epoch": 1}],
        "visual_ablation": mode == "action_only",
        "image_channels_are_constant_zero": mode == "action_only",
    }


@pytest.mark.parametrize("mode", ["rgb", "rgb_hsv", "action_only"])
def test_accepts_each_frozen_model(mode):
    require_frozen_checkpoint(report(mode), mode=mode, split=SPLIT)


@pytest.mark.parametrize("key,new,why", [
    ("smoke_unrepresentative", True, "smoke"),
    ("train_videos", ["changed.mp4"], "train split"),
    ("validation_videos", [], "validation split"),
    ("test_videos_not_evaluated", [], "test holdout"),
    ("selection", "test best epoch", "selection"),
    ("target", "t+200ms", "horizon"),
    ("visual_size", [224,224], "visual size"),
    ("touch_masking", False, "touch mask"),
    ("mode", "action_only", "mode"),
    ("expert_teacher_forced", False, "expert-supervised"),
    ("history", [], "history"),
])
def test_rejects_changed_provenance(key,new,why):
    item = report()
    item[key] = new
    with pytest.raises(ValueError, match=why):
        require_frozen_checkpoint(item, mode="rgb", split=SPLIT)


def test_rejects_missing_action_only_flag():
    item = report("action_only")
    del item["visual_ablation"]
    with pytest.raises(ValueError, match="visual ablation"):
        require_frozen_checkpoint(item, mode="action_only", split=SPLIT)


def test_rejects_ablation_false_reported_as_true():
    item = report("rgb")
    item["image_channels_are_constant_zero"] = True
    with pytest.raises(ValueError, match="image input ablation"):
        require_frozen_checkpoint(item, mode="rgb", split=SPLIT)


def test_rejects_overlapping_test_split_even_with_matching_metadata():
    invalid = copy.copy(SPLIT)
    invalid.test = ("testA.mp4", "trainA.mp4")
    item = report()
    item["test_videos_not_evaluated"] = list(invalid.test)
    with pytest.raises(ValueError, match="overlap"):
        require_frozen_checkpoint(item, mode="rgb", split=invalid)
