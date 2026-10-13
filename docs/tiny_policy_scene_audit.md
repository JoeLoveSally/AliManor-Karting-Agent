# Frozen Test visual scenes — read-only review

The preceding timeline audit found two error types that should not be confused:
- **Event timing slightly outside the ±100ms matching tolerance:** On the
  held-out `173426` video every unmatched GT event for RGB / RGB+HSV has
  a same-direction model event about 105–123ms away. These are late switches,
  not an absence of any model response.
- **Actual extra predicted switches:** On `173835` both policies have
  additional short probability dips, especially RGB near 5.5s and 32s.

These are descriptive Test diagnostics only. Never choose a new event threshold,
alter training, choose a new checkpoint or tune any scheduler with Test.

## Scene evidence needed

The probability curves alone cannot distinguish road geometry ambiguity,
vehicle occlusion, visual-theme artifacts, or an unstable model output. Review
original video frames around each distinct incident. No manual road labeling.

The read-only script `scripts/audit_tiny_policy_scenes.py` takes the FROZEN
`tiny_policy_timeline_audit.json` and both CSV traces. It verifies the frozen
audit protocol, CSV column presence, sample count and increasing target times,
then chooses distinct event windows:

- all missed 100–300ms GT release segments first, shared misses first
- additional false predicted switches, isolated ones before near-GT switches
- deduplicates predicted switch bursts closer than 260ms; max 8 windows/video

For each window it retrieves **RAW SOURCE** frames at t−200ms, t, t+200ms
with nearest-frame indexing, and places each triplet next to the exact
frozen-audit event description. The frames shown are deliberately unmasked:
the original model actually saw HUD/touch-masked RGB or RGB+HSV at 96×96.
These screenshots are scene evidence, **not** exact model-input pixels.

Expected case selection on current uploaded audit: 5 in `173426` and 8 in
`173835`. No GPU, no models, no online execution, no changes to controls,
labels, prediction horizon, event matches, or Test reports.

## Run

**WSL repository:**

```bash
cd /home/jianqiao/workspace/AliManor-Karting-Agent-v4c2-history
git pull --ff-only origin exp/v4c2-action-history-residual
rspark \
  ./scripts/audit_tiny_policy_scenes.py \
  ./tests/unit/test_tiny_policy_scenes.py
```

**Spark:**

```bash
cd /home/grg/Workspace/AliManor-Karting-Agent

python -m pytest -q tests/unit/test_tiny_policy_scenes.py

python scripts/audit_tiny_policy_scenes.py \
  --audit /tmp/tiny_policy_export/tiny_policy_timeline_audit.json \
  --trace-dir /tmp/tiny_policy_export \
  --video-root /home/grg/Workspace/AliManor-Karting-Agent \
  --output-dir /tmp/tiny_policy_export
```

Expected output files (flat directory, no persistent WSL subfolder):

- `tiny_scenes_video_20260130_173426.jpg`
- `tiny_scenes_video_20260130_173835.jpg`

Existing output files are never overwritten.

**WSL result transfer:**

```bash
rsync -av \
  --include='tiny_scenes_*.jpg' \
  --exclude='*' \
  grg@10.1.48.26:/tmp/tiny_policy_export/ \
  /home/jianqiao/downloads/
```

Please review the two images alongside the earlier
`tiny_focus_*.jpg` time/decision curves. Classify each scene as observable
road-bend cue, motion/position ambiguity, HUD/occlusion, or unidentifiable
without a video clip; do not claim a causal explanation from the curves alone.

## Verification

A local CPU environment ran six synthetic-video tests covering event
deduplication, ranking, frame seek at actual FPS, JPEG output, CSV provenance
and refusal to overwrite. Also verified both uploaded original Test traces
against the frozen diagnostic JSON. Spark source MP4 files are not available
here: original video playback must be verified on Spark.
