# Road Segmentation Benchmark — first 20 independent human labels

## Scope and file placement

- No code, checkpoints or executors in V4-C2/Armed are modified.
- Permanent user-facing exports and results live in **/home/jianqiao/downloads**
  on WSL. Spark's **/tmp/roadbm_export** is an ephemeral staging directory
  used solely because the expert MP4s currently reside on Spark.
- Source and test code remain tracked in the experimental Git branch.
- Canonical canvas is **360×800**, consistent with the ADB occupancy audit.
- Use the frozen `configs/train_v4c2_temporal_v2.yaml` video split; NEVER
  tune a threshold/model using test frames.

## Stage 1 — export four frames per video

Sync the script to Spark, then run from the Spark repository root:

```bash
mkdir -p /tmp/roadbm_export

for name in \
  video_20260130_172959 \
  video_20260130_174257 \
  video_20260130_173545 \
  video_20260130_173426 \
  video_20260130_173835
do
  python scripts/road_segmentation_benchmark.py extract \
    --video "data/raw/${name}.mp4" \
    --frames 240 480 720 1080 \
    --output-dir /tmp/roadbm_export
done
```

Copy ONLY image exports from Spark to the designated WSL location:

```bash
rsync -av grg@10.1.48.26:/tmp/roadbm_export/ /home/jianqiao/downloads/
```

These frames are a workflow smoke test, **not** a representative statistical
benchmark. If some lack bends, replace/add samples selected independently
of the HSV predictions. Do not remove frames after seeing model errors.

## Stage 2 — independent manual labels, in WSL with OpenCV GUI

Run using a local Python environment that has `cv2`, `numpy`, `yaml`:

```bash
python scripts/road_segmentation_benchmark.py annotate \
  --image-dir /home/jianqiao/downloads
```

Mouse/keyboard: press **R** then left-click vertices around visible drivable
ROAD. Press **Enter** to complete each polygon. Press **I** for occluded/unclear
areas (kart, smoke, UI); Enter to close. Press **B** and draw if you need to
erase an incorrect road region as BACKGROUND. **Z** removes the last vertex;
**U** cancels the current polygon or undoes a completed polygon; **C**
clears unfinished vertices; **S** saves `*_gt.png`; **Q** exits safely.
Default unpainted region is BACKGROUND. Draw all disjoint road sections.

Every GT is an uncompressed uint8 PNG with values:
- 0: certain BACKGROUND
- 1: certain ROAD
- 255: IGNORE, masked out of evaluation

The expert touch-marker ROI is excluded regardless of polygon order.
Do **not** use HSV masks to draw ground truth: label from the original RGB
frame alone. Do not treat black tire marks as physical road gaps. Uncertain
car/smoke pixels are IGNORE, not automatically ROAD. If the WSL GUI is
unavailable, stop and use another polygon annotator only with explicit
conversion to the same 0/1/255 PNG contract.

## Stage 3 — evaluate TRAIN and VALIDATION first

From WSL repository worktree:

```bash
python scripts/road_segmentation_benchmark.py evaluate \
  --image-dir /home/jianqiao/downloads \
  --splits train validation \
  --output /home/jianqiao/downloads/roadbm_hsv_trainval_v1.json
```

Reports original HSV and near-kart connected-component predictions separately,
over complete image, kart-centred 240×240 local and 360×480 wide windows.

- Road IoU=TP/(TP+FP+FN)
- Precision=TP/(TP+FP)
- Recall=TP/(TP+FN)
- Background FPR=FP/(FP+TN)

IGNORE never counts in confusion; empty denominators become `null` rather
than an invented perfect score. Missing frames are skipped and must be counted
before drawing conclusions. If local/wide metrics are absent, inspect the
kart detector on that frame.

Use validation examples to select a *single* next representation candidate.
After freezing every rule, separately run:

```bash
python scripts/road_segmentation_benchmark.py evaluate \
  --image-dir /home/jianqiao/downloads \
  --splits test \
  --output /home/jianqiao/downloads/roadbm_hsv_test_v1.json
```

No universal IoU threshold is specified before observing label quality and
failure cases. Review severe false-positive backgrounds and bend boundaries
first; only then decide between constrained HSV and a tiny segmentation net.

## After benchmark

If road masks provide trustworthy near-kart and lookahead information, build
a temporally dense **expert-only** occupancy/action dataset with the original
video/action timestamps, protecting the expert touch-marker from leakage.
Frozen V4-C2 remains the closed-loop baseline. Then evaluate persistence,
single-frame model and CNN+GRU for PRESS/RELEASE event timing and short
corrections — no simulator or Armed changes until evidence warrants them.
