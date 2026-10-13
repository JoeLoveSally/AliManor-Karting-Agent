"""CPU-only contract tests for road-mask benchmark labels and metrics."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/road_segmentation_benchmark.py"
spec = importlib.util.spec_from_file_location("_road_segmentation_benchmark_test", SCRIPT)
assert spec and spec.loader
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def test_polygon_labels_distinguish_road_background_and_ignore():
    polygons = [
        (1, [(2,2),(13,2),(13,13),(2,13)]),
        (255,[(6,6),(9,6),(9,9),(6,9)]),
        (0,[(2,2),(4,2),(4,4),(2,4)]),
    ]
    mask = benchmark.polygon_labels((20,20), polygons, touch_roi=(.8,.8,1,1))
    assert mask[5,5] == 1
    assert mask[7,7] == 255
    assert mask[3,3] == 0
    assert mask[17,17] == 255
    assert mask[1,1] == 0


def test_invalid_polygon_does_not_produce_partial_gt():
    with pytest.raises(ValueError,match="invalid polygon"):
        benchmark.polygon_labels((20,20), [(1, [(0,0),(2,2)])])


def test_ignore_pixels_are_completely_removed_from_confusion():
    gt=np.array([[1, 0, 255], [1, 0, 255]],dtype=np.uint8)
    pred=np.array([[255,255,255], [0,0,0]],dtype=np.uint8)
    c=benchmark.confusion(pred,gt)
    assert c=={"tp":1,"fp":1,"fn":1,"tn":1}
    m=benchmark.metrics(c)
    assert m["iou"]==pytest.approx(1/3)
    assert m["background_fpr"]==pytest.approx(.5)


def test_near_kart_scoping_changes_metrics():
    gt=np.zeros((100,100),dtype=np.uint8)
    gt[40:60,40:60]=1
    pred=np.zeros((100,100),dtype=np.uint8)
    pred[40:60,40:60]=255
    pred[0:20,0:20]=255  # False positives far from kart
    full=benchmark.confusion(pred,gt)
    near=benchmark.confusion(
        pred,gt,benchmark.centered_region(gt.shape,(50,50),35,35)
    )
    assert full["fp"]==400
    assert near["fp"]==0
    assert near["tp"]==400


def test_empty_positive_union_is_null_not_perfect_score():
    m=benchmark.metrics({"tp":0,"fp":0,"fn":0,"tn":100})
    assert m["iou"] is None
    assert m["recall"] is None
    assert m["background_fpr"]==0


def test_label_ui_is_ignored_even_if_road_polygon_covers_it():
    polygons=[(1,[(0,0),(99,0),(99,99),(0,99)])]
    mask=benchmark.polygon_labels((100,100),polygons)
    assert mask[90,90]==255
    assert mask[20,20]==1


def test_sample_name_prefix_is_stable_for_split_match():
    assert benchmark.sample_name(
        Path("video_20260130_173426.mp4"),240
    )=="roadbm_video_20260130_173426_f000240.png"
    m=benchmark.IMAGE_RE.fullmatch("roadbm_video_20260130_173426_f000240.png")
    assert m.group(1)=="video_20260130_173426"
    assert m.group(2)=="000240"


def test_centered_region_clips_to_boundaries():
    region=benchmark.centered_region((50,100),(0,0),40,40)
    assert region.shape==(50,100)
    assert region[0,0]
    assert not region[49,99]


def test_original_v4c2_split_has_no_overlap():
    path=Path(__file__).resolve().parents[2]/"configs/train_v4c2_temporal_v2.yaml"
    mapping=benchmark.split_map(path)
    assert mapping["video_20260130_172959"]=="train"
    assert mapping["video_20260130_173545"]=="validation"
    assert mapping["video_20260130_173426"]=="test"
    assert len(mapping)==15


def test_evaluation_skips_unlabeled_and_explicit_test_split(tmp_path):
    import cv2
    image=np.zeros((800,360,3),dtype=np.uint8)
    file=tmp_path/"roadbm_video_20260130_173426_f000240.png"
    assert cv2.imwrite(str(file),image)
    conf=Path(__file__).resolve().parents[2]/"configs/train_v4c2_temporal_v2.yaml"
    road=Path(__file__).resolve().parents[2]/"configs/geometry_pseudo_labels.yaml"
    # No test evaluation unless explicitly opted in by the caller.
    result=benchmark.evaluate(
        tmp_path,train_config=conf,geometry_config=road,
        splits=("train","validation"),
    )
    assert result["sample_count"]==0


def test_annotation_fixed_bitmap_is_smaller_than_source():
    assert benchmark.annotation_view_size(360, 800) == (288, 640)
    assert benchmark.annotation_view_size(360, 800, 720) == (324, 720)
    assert benchmark.annotation_view_size(360, 800, 1200) == (360, 800)


def test_annotation_click_coordinates_remain_in_source_image():
    convert = benchmark.annotation_source_point
    assert convert(
        144, 320, source_width=360, source_height=800,
        view_width=288, view_height=640,
    ) == (180, 400)
    assert convert(
        287, 639, source_width=360, source_height=800,
        view_width=288, view_height=640,
    ) == (358, 798)
    assert convert(
        300, -10, source_width=360, source_height=800,
        view_width=288, view_height=640,
    ) == (359, 0)


def test_annotation_view_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        benchmark.annotation_view_size(360, 800, 0)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        benchmark.annotation_source_point(
            1, 1, source_width=360, source_height=800,
            view_width=0, view_height=640,
        )
