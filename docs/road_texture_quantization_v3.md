# Read-only Road Texture Quantization v3

## Why this diagnostic exists

The kart track is built from regular, straight-sided sections, and the kart is
typically close to its current track section. However the available expert
videos do NOT all use an identical checkerboard texture:

- blue theme: broad triangular/diagonal paving motifs;
- teal theme: fine grid;
- other themes: larger tile or polygonal motifs.

Consequently this POC does not use fixed global tile colors, a single checker
template, the earlier Hough **centerline** teacher, or a hand-annotated dataset.

This is NOT a semantic segmentation model or a finished policy.

## Reused project components

- original frame normalization: 360x800 pixels;
- existing KartPose color tracker (for a location anchor only);
- original HSV-v2 road mask (side-by-side **baseline**, not a v3 input);
- expert touch-marker exclusion ROI `(0.78, 0.82, 0.98, 0.98)`;
- original raw MP4 frame indices; nominal FPS when there is no paired ADB JSON.

## The v3 candidate

1. LAB-lightness local contrast at two spatial scales, smoothed over local
   neighborhoods: **texture evidence** from road tiles/grids, not a binary mask.
2. Hough line directions whose midpoints lie within 190px of the kart but
   outside the inner 55px chassis area. At most two orientation peaks, with a
   *soft* pixel-wise alignment map: **line evidence**, not a fitted polygon.
3. Candidate = smoothed 75% texture and 25% edge alignment. Connected-component
   selection only among regions supported in an annulus **65–135px** from the
   detected kart. It does not hallucinate other distant road sections.
4. Mask the central 55px kart disk as **unobserved**, and touch-marker ROI
   plus 30px guard as **unobserved**. A missing kart or weak nearby evidence
   means candidate invalid and every occupancy pixel unverified.

These numerical choices are fixed **exploratory defaults**. They are not
validated or trained weights. The mixture/threshold intentionally remains
unchanged across expert themes; do not tune it on held-out test frames.

## Output

Each video produces four flat, prefixed files under `--output-dir`:

- `<video>_v3_contact.jpg`: eight real-frame comparison columns:
  `REAL RGB / HSV-v2 / TEXTURE / LINE EVIDENCE / V3 CANDIDATE /
   32x32 LOCAL / 32x32 WIDE / LOCAL KNOWN`.
- `<video>_v3_features.npz`: `grids[T,8,32,32]`, source frame indices,
  nominal source timestamps and candidate-valid flags.
- `<video>_v3_metadata.jsonl`: per-frame evidence, orientation peaks and seed
  status.
- `<video>_v3_summary.json`: aggregate diagnostic coverage, **not accuracy**.

Channel order:

    local_candidate, local_texture, local_geometry, local_known,
    wide_candidate, wide_texture, wide_geometry, wide_known

Candidate values are heuristic scores, NOT calibrated ROAD probabilities.
Known means only that the frame region is observed/not explicitly occluded.
Smoke touching the road edge, fake perspective junctions and distant same-color
structures are still open problems.

## Where to save outputs

All files that the user needs to view or upload ultimately go **directly to**
`/home/jianqiao/downloads` in WSL. If the source videos are on Spark, the
Spark-only `/tmp/road_texture_v3_export` is disposable transport staging
(no new persistent artifact directory inside the repository).

### 1. Sync tracked code from WSL to Spark

    cd /home/jianqiao/workspace/AliManor-Karting-Agent-v4c2-history
    git pull --ff-only origin exp/v4c2-action-history-residual
    rspark ./src/karting_agent/train/road_texture_poc.py \
            ./scripts/audit_road_texture_poc.py \
            ./tests/unit/test_road_texture_poc.py

### 2. Verify tests, then generate five themes on Spark

    cd /home/grg/Workspace/AliManor-Karting-Agent
    python -m pytest -q tests/unit/test_road_texture_poc.py
    mkdir -p /tmp/road_texture_v3_export
    python scripts/audit_road_texture_poc.py \
      data/raw/video_20260130_172959.mp4 \
      data/raw/video_20260130_174257.mp4 \
      data/raw/video_20260130_173545.mp4 \
      data/raw/video_20260130_173426.mp4 \
      data/raw/video_20260130_173835.mp4 \
      --frame-start 240 \
      --sample-every-frames 120 \
      --max-samples 12 \
      --touch-roi 0.78 0.82 0.98 0.98 \
      --output-dir /tmp/road_texture_v3_export

### 3. Sync flat results to WSL

    rsync -av \
      grg@10.1.48.26:/tmp/road_texture_v3_export/ \
      /home/jianqiao/downloads/

Review the five `*_v3_contact.jpg` files and the valid/invalid counts.
Never infer IoU without independent ground truth. If this representation
retains obvious current road and upcoming turns while suppressing far-away
decorations on multiple themes, proceed to a *temporally dense*, expert-only
visual data prototype before trying a small sequence policy.

No files in baseline V4-C2/V4-C4 or Armed are changed.
