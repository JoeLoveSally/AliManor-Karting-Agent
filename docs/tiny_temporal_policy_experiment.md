# Minimal temporal policy experiment: RGB vs RGB+HSV

## Scope

This is an **offline, read-only, expert-supervised** comparison. It is NOT an
Armed/controller change and none of its checkpoints should be loaded into the
current runtime. Original V4-C2 and its scheduler remain frozen.

The experiment uses:
- `configs/train_v4c2_temporal_v2.yaml`, preserving **11 train / 2 validation /
  2 test** complete-video splits;
- `data/processed/v3/samples.jsonl`, with exactly 5 input frames and
  `target_pressed_by_horizon[0]` at **t+100ms**;
- `data/processed/v3/labels/<video>.json`, the cleaned native-FPS
  PRESS/RELEASE event timeline (both current-action features and GT sequence
  transition timestamps);
- the existing expert-video path and HSV-v2 teacher, not generated pseudo-labels.

The full script will **not decode or evaluate test videos**. It intentionally
does not choose a threshold on the test split. Do not interpret smokes as metrics.

## Experimental variants

| Variant | Input per frame | Additional controls | Model |
| --- | --- | --- | --- |
| A: `rgb` | masked RGB `[3,96,96]` | expert state at observation, elapsed since last expert action, observed frame delta | CNN32 + GRU64 |
| B: `rgb_hsv` | identical masked RGB plus normalized HSV-v2 mask `[4,96,96]` | identical | CNN32 + GRU64 |

Both variants share **exact source frames** and expert temporal labels, no
separate sampling process. HSV feature is an unverified coarse image feature;
it is **not** used as semantic ground truth or a supervision label.

The same V4-C2 touch ROI and other HUD ROIs are masked on the source BGR
frame **before** generating *either* RGB or HSV features. HSV features
cannot include visible expert-action touch markers.

Use explicit finite differences from the existing sample timestamps. Action
age is computed from events at or before the frame's observation time and
clipped to 500ms. This is teacher-forced action history. Feeding the recorded
future action into the current model is prohibited.

The supervised output is a SINGLE Bernoulli logit for absolute expert PRESS
state at `t+100ms`. No XOR horizon interpolation, pending timer or synthetic
counterfactual dynamics is evaluated.

## Evaluation

Best checkpoint selected on **validation BCE only**, no threshold sweep.
For each validation video, concatenate future PRESS probability at its
**target_timestamp_ms** (not its input observation timestamp); run the existing
`evaluate_sequence` at fixed threshold 0.5 and tolerance ±100ms. The same
timestamps are used for the expert-state persistence baseline.

Report:
- validation BCE and sample accuracy;
- transition Precision/Recall/F1, PRESS and RELEASE onset timing;
- native-label short 100–300ms RELEASE-segment recall;
- action(t) -> action(t+100ms) persistence on exactly the same samples.

This remains **expert trajectory / teacher-forced offline** evaluation.
A winning result does not demonstrate recovery from unseen off-road states,
closed-loop stability or safety.

## Use in the existing WSL + Spark setup

All user-facing outputs must be copied to **/home/jianqiao/downloads** on WSL.
Only temporary staging under **/tmp** is used on Spark. Do not create
additional persistent WSL experiment directories.

**WSL:**

    cd /home/jianqiao/workspace/AliManor-Karting-Agent-v4c2-history
    git pull --ff-only origin exp/v4c2-action-history-residual

    rspark ./src/karting_agent/model/tiny_temporal_policy.py \
           ./scripts/experiment_tiny_temporal_policy.py \
           ./tests/unit/test_tiny_temporal_policy_experiment.py

**Spark:**

    cd /home/grg/Workspace/AliManor-Karting-Agent

    python -m pytest -q tests/unit/test_tiny_temporal_policy_experiment.py

    mkdir -p /tmp/tiny_policy_export

    python scripts/experiment_tiny_temporal_policy.py \
      --preflight-only --output-dir /tmp/tiny_policy_export

    python scripts/experiment_tiny_temporal_policy.py \
      --smoke --epochs 1 --output-dir /tmp/tiny_policy_export

Do not launch a full training run until the unit tests, preflight and smoke
all pass. Smoke trains on only 128 train + 64 validation samples, saving
`tiny_policy_smoke_*.pt/json`. These metrics are not comparative.

**After successful smoke, Spark full experiment:**

    python scripts/experiment_tiny_temporal_policy.py \
      --epochs 8 --batch-size 16 --device auto \
      --cuda-memory-fraction 0.03 --min-cuda-free-mib 2048 \
      --cpu-threads 4 \
      --output-dir /tmp/tiny_policy_export

A co-resident vLLM-Omni process may leave insufficient CUDA memory even to
create a new GPU context (the prior experiment failed on model.to(device),
before its first batch). Device selection NOW happens before expert video
decoding. Auto mode examines nvidia-smi without initializing CUDA; if fewer
than 2048 MiB are free, CUDA is unavailable, or context setup fails, it
falls back to CPU automatically. The actual device and budget reason are
printed at startup and saved in each result JSON.

--cuda-memory-fraction 0.03 limits PyTorch's caching allocator to 3% of
the GPU-reported total memory; it does NOT reserve GPU space, prevent
vLLM allocations, or cap all CUDA-driver overhead. On DGX Spark this can
also compete with vLLM for unified CPU/GPU memory. Reduce batch size further
if CUDA runs out of memory while training. Selecting --device cpu avoids
allocating tensors on GPU entirely but still uses host/unified system RAM.
Use --device cuda for a strict GPU-only attempt (fail fast if unavailable)
or combine with --allow-cpu-fallback.

This saves `tiny_policy_rgb.pt/json` and
`tiny_policy_rgb_hsv.pt/json`. Feature frames are cached transiently in
process RAM (not written as arbitrary files); total preprocessing may take
multiple minutes. Requires sufficient Spark RAM and GPU memory for batch 64.
Use a smaller `--batch-size` if needed; note that changes the experiment
conditions and should be applied to both modes.

**WSL:**

    rsync -av \
      grg@10.1.48.26:/tmp/tiny_policy_export/ \
      /home/jianqiao/downloads/

Only four final filenames and optional smoke files are written directly to
the requested downloads location. Reports are JSON, weights are PyTorch
state dictionaries; no runtime integration is built in this experiment.
Existing output filenames are **never overwritten**.

## Decision rule

Use Validation only to decide if either tiny model improves transition F1,
onset timing and short-release recall against persistence; compare with the
frozen V4-C2 baseline using the exact same horizons and evaluation protocol
before claiming a genuine win. If RGB+HSV does not improve events, do not
force the segmentation feature into the next model.

If no variant recovers action-event timing, inspect whether the
absolute-future-action target is sufficient before introducing a first-event
hazard head. Do not tune scheduler/pending timer or try Armed yet.
