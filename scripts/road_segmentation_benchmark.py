#!/usr/bin/env python3
"""Road/background/ignore benchmark: export -> manual polygons -> offline evaluation.

This is a *diagnostic* script. It never trains a policy, invokes ADB, or
modifies the control runtime. Ground-truth masks use 0=BACKGROUND, 1=ROAD,
255=IGNORE. Expert touch-marker pixels are excluded from evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for parent in (ROOT / "src", ROOT / "scripts"):
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))

from inspect_geometry_pseudo_labels import load_config as load_road_config  # noqa: E402
from inspect_kart_relative_geometry import load_kart_configs  # noqa: E402
from audit_road_occupancy_poc import (  # noqa: E402
    normalize_game_frame, roi_redaction_mask, select_kart_local_component,
)
from karting_agent.train.geometry_pseudo_labels import extract_road_mask  # noqa: E402
from karting_agent.train.kart_pose_pseudo_labels import estimate_kart_pose  # noqa: E402

IMAGE_RE = re.compile(r"roadbm_(.+)_f(\d{6})\.png")
LABELS = {"background": 0, "road": 1, "ignore": 255}
TOUCH_ROI = (0.78, 0.82, 0.98, 0.98)


def sample_name(video: Path, frame_index: int) -> str:
    if frame_index < 0:
        raise ValueError("negative frame index")
    return f"roadbm_{video.stem}_f{frame_index:06d}.png"


def export_frames(video: Path, indices: list[int], output_dir: Path,
                  canonical_size=(360, 800)) -> list[Path]:
    """Sequentially decode requested frames; never seek by unreliable CFR PTS."""
    wanted = sorted(set(map(int, indices)))
    if not wanted or wanted[0] < 0:
        raise ValueError("frame indices must be nonempty and >= 0")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output directory must exist: {output_dir}")
    destinations = [output_dir / sample_name(video, i) for i in wanted]
    existing = [str(p) for p in destinations if p.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace images: {existing[:3]}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    written = []
    count = 0
    current = 0
    try:
        while current <= wanted[-1] and count < len(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            if current == wanted[count]:
                normalized = normalize_game_frame(frame, target_size=canonical_size)
                target = destinations[count]
                if not cv2.imwrite(str(target), normalized):
                    raise RuntimeError(f"could not write {target}")
                written.append(target)
                count += 1
            current += 1
    finally:
        cap.release()
    if count != len(wanted):
        raise ValueError(f"video ended before frames {wanted[count:]} were decoded")
    return written


def polygon_labels(shape: tuple[int, int],
                   polygons: list[tuple[int, list[tuple[int, int]]]],
                   touch_roi=TOUCH_ROI) -> np.ndarray:
    h, w = shape
    label = np.zeros((h, w), dtype=np.uint8)
    for value, vertices in polygons:
        if value not in (0, 1, 255) or len(vertices) < 3:
            raise ValueError("invalid polygon/label")
        cv2.fillPoly(label, [np.asarray(vertices, dtype=np.int32)], int(value))
    # UI cannot contribute to any metric, irrespective of human drawing order.
    redaction = roi_redaction_mask(label.shape, touch_roi)
    label[redaction > 0] = 255
    return label


def _annotate_one(image_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    target = image_path.with_name(image_path.stem + "_gt.png")
    if target.exists():
        print(f"Already labeled, skip: {target}")
        return True
    polygons: list[tuple[int, list[tuple[int, int]]]] = []
    points: list[tuple[int, int]] = []
    current_label = 1
    h, w = image.shape[:2]
    win = "RoadBM polygon labeling"
    # AUTOSIZE preserves a strict mouse-pixel == label-pixel mapping.
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((max(0, min(w-1, int(x))),
                           max(0, min(h-1, int(y)))))

    cv2.setMouseCallback(win, mouse)
    try:
        while True:
            preview = image.copy()
            label = polygon_labels((h,w), polygons)
            tint = np.zeros_like(preview)
            tint[label == 1] = (50, 185, 40)
            tint[label == 255] = (110, 110, 110)
            marked = label != 0
            preview[marked] = cv2.addWeighted(
                preview[marked], .48, tint[marked], .52, 0
            )
            if points:
                poly = np.asarray(points, dtype=np.int32)
                cv2.polylines(preview, [poly], False, (0, 215, 255), 2)
                for px, py in points:
                    cv2.circle(preview, (px,py), 3, (0, 215, 255), -1)
            cv2.rectangle(preview, (0,0), (w,45), (0,0,0), -1)
            cv2.putText(preview, image_path.name[-31:], (5,15),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1)
            cv2.putText(preview, "R road / I ignore / B bg / Enter close",
                        (5,29), cv2.FONT_HERSHEY_SIMPLEX, .34, (255,255,255), 1)
            cv2.putText(preview, "Z back / U undo / C clear / S save / Q quit",
                        (5,42), cv2.FONT_HERSHEY_SIMPLEX, .34, (255,255,255), 1)
            cv2.imshow(win, preview)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("r"),ord("i"),ord("b")):
                current_label = {ord("r"):1,ord("i"):255,ord("b"):0}[key]
            elif key in (10,13):
                if len(points) >= 3:
                    polygons.append((current_label, list(points)))
                    points.clear()
            elif key == ord("z") and points:
                points.pop()
            elif key == ord("u"):
                if points: points.clear()
                elif polygons: polygons.pop()
            elif key == ord("c"):
                points.clear()
            elif key == ord("s"):
                if points:
                    print("Close current polygon with Enter before saving")
                    continue
                if not any(value == 1 for value, _ in polygons):
                    print("At least one ROAD polygon is required")
                    continue
                mask = polygon_labels((h,w), polygons)
                if not cv2.imwrite(str(target), mask):
                    raise RuntimeError(f"could not write ground truth: {target}")
                print(f"Saved: {target}")
                return True
            elif key in (ord("q"), 27):
                return False
    finally:
        cv2.destroyWindow(win)


def confusion(pred: np.ndarray, truth: np.ndarray,
              region: np.ndarray | None = None) -> dict:
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("pred/truth mask sizes differ")
    valid = (truth != 255)
    if region is not None:
        if region.shape != truth.shape: raise ValueError("region shape differs")
        valid &= region.astype(bool)
    p = (pred > 0) & valid
    t = (truth == 1) & valid
    return {
        "tp": int(np.count_nonzero(p & t)),
        "fp": int(np.count_nonzero(p & ~t)),
        "fn": int(np.count_nonzero(~p & t)),
        "tn": int(np.count_nonzero(valid & ~p & ~t)),
    }


def metrics(count: dict) -> dict:
    tp, fp, fn, tn = (count[k] for k in ("tp","fp","fn","tn"))
    def ratio(n, d):
        return n/d if d else None
    return {
        **count, "pixels": tp+fp+fn+tn,
        "iou": ratio(tp, tp+fp+fn),
        "precision": ratio(tp, tp+fp),
        "recall": ratio(tp, tp+fn),
        "background_fpr": ratio(fp, fp+tn),
    }


def centered_region(shape: tuple[int,int], xy: tuple[float,float],
                    crop_width:int,crop_height:int) -> np.ndarray:
    h,w=shape
    cx,cy=xy
    x0=max(0,int(round(cx-crop_width/2)))
    x1=min(w,int(round(cx+crop_width/2)))
    y0=max(0,int(round(cy-crop_height/2)))
    y1=min(h,int(round(cy+crop_height/2)))
    region=np.zeros((h,w),dtype=bool)
    region[y0:y1,x0:x1]=True
    return region


def split_map(config_path:Path) -> dict[str,str]:
    raw=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out={}
    for split in ("train","validation","test"):
        for name in raw["split"][split]:
            stem=Path(name).stem
            if stem in out: raise ValueError(f"duplicate split video: {stem}")
            out[stem]=split
    return out


def evaluate(image_dir:Path, *, train_config:Path, geometry_config:Path,
             splits:tuple[str,...]) -> dict:
    mapping=split_map(train_config)
    road_cfg,_,_=load_road_config(geometry_config)
    kart_cfg,_=load_kart_configs(geometry_config)
    totals=defaultdict(lambda: defaultdict(lambda: {k:0 for k in ("tp","fp","fn","tn")}))
    rows=[]
    for file in sorted(image_dir.glob("roadbm_*_f??????.png")):
        found=IMAGE_RE.fullmatch(file.name)
        if found is None: continue
        stem,frame=found.group(1),int(found.group(2))
        split=mapping.get(stem)
        if split not in splits: continue
        gt_path=file.with_name(file.stem+"_gt.png")
        if not gt_path.is_file():
            print(f"Unlabeled, skipping: {file.name}")
            continue
        image=cv2.imread(str(file))
        gt=cv2.imread(str(gt_path),cv2.IMREAD_UNCHANGED)
        if image is None or gt is None or gt.ndim != 2 or image.shape[:2] != gt.shape:
            raise ValueError(f"invalid image or ground truth: {file.name}")
        if not np.isin(gt, [0,1,255]).all():
            raise ValueError(f"unexpected GT label values: {gt_path}")
        raw=extract_road_mask(image,road_cfg)
        pose,_=estimate_kart_pose(image,kart_cfg)
        xy=(pose.center_x,pose.center_y) if pose is not None else None
        selected,_=select_kart_local_component(raw,xy)
        gt=gt.copy()
        gt[roi_redaction_mask(gt.shape,TOUCH_ROI)>0]=255
        regions={"all":np.ones(gt.shape,dtype=bool)}
        if xy is not None:
            regions["local"]=centered_region(gt.shape,xy,240,240)
            regions["wide"]=centered_region(gt.shape,xy,360,480)
        sample={"video":stem,"frame":frame,"split":split,
                "kart_found":xy is not None,"metrics":{}}
        for method,pred in (("hsv_raw",raw),("kart_component",selected)):
            for region_name,region in regions.items():
                key=f"{method}:{region_name}"
                n=confusion(pred,gt,region)
                sample["metrics"][key]=metrics(n)
                for scope in (split,"all_evaluated"):
                    record=totals[scope][key]
                    for k,v in n.items(): record[k]+=v
        rows.append(sample)
    return {
        "sample_count":len(rows),
        "splits_evaluated":list(splits),
        "summary":{sp:{key:metrics(count) for key,count in vals.items()}
                   for sp,vals in totals.items()},
        "samples":rows,
        "warning": (
            "GT is polygon annotated and must be reviewed for boundary/occlusion "
            "errors. Unlabeled images are skipped. Test split must not be used "
            "to tune HSV or select a model."
        ),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="command",required=True)
    ex=sub.add_parser("extract",help="export canonical expert frames on Spark")
    ex.add_argument("--video",type=Path,required=True)
    ex.add_argument("--frames",type=int,nargs="+",required=True)
    ex.add_argument("--output-dir",type=Path,required=True)
    ann=sub.add_parser("annotate",help="OpenCV WSLg polygon annotation GUI")
    ann.add_argument("--image-dir",type=Path,required=True)
    ann.add_argument("--max-images",type=int,default=None)
    ev=sub.add_parser("evaluate",help="compare GT with existing HSV teacher")
    ev.add_argument("--image-dir",type=Path,required=True)
    ev.add_argument("--train-config",type=Path,default=ROOT/"configs/train_v4c2_temporal_v2.yaml")
    ev.add_argument("--geometry-config",type=Path,default=ROOT/"configs/geometry_pseudo_labels.yaml")
    ev.add_argument("--splits",nargs="+",choices=("train","validation","test"),
                    default=["train","validation"])
    ev.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.command=="extract":
        print(json.dumps([str(p) for p in export_frames(
            args.video,args.frames,args.output_dir)],indent=2))
    elif args.command=="annotate":
        if args.max_images is not None and args.max_images < 1:
            ap.error("max image count must be positive")
        images=sorted(args.image_dir.glob("roadbm_*_f??????.png"))
        for path in images[:args.max_images]:
            if not _annotate_one(path):
                break
    else:
        if args.output.exists(): raise FileExistsError(args.output)
        report=evaluate(args.image_dir,train_config=args.train_config,
                        geometry_config=args.geometry_config,
                        splits=tuple(args.splits))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:
            json.dump(report,stream,indent=2,ensure_ascii=False)
            stream.write("\n")
        print(json.dumps({"sample_count":report["sample_count"],
                          "summary":report["summary"]},ensure_ascii=False))


if __name__=="__main__":
    main()
