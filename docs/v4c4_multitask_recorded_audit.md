# V4-C4 recorded Armed run: all-head diagnosis (no ADB)

After the first Armed test failed at the first bend, the existing recorded-input
replay audit reproduced **234/234 current-action probabilities exactly**. This
establishes replay consistency for recorded pixels; it does not establish the
age of the game frames or the correctness of model predictions.

The browser-stamp latency experiment produced no valid stamps because its
pixel-diagnostic capture was not displaying the expected binary bars. The
Settings/HOME change-response median (approximately 553 ms) includes Android
UI-command and page-rendering overhead and must not be read as video latency.
Do not retrigger the same game or browser probe to run this audit.

## Run using the previous Armed JSON, MP4 and ORIGINAL checkpoint

From the repository root in WSL:

```bash
python -m pytest tests/unit/test_v4c4_multitask_recorded.py -q

python scripts/diagnose_v4c4_multitask_recorded.py \
  --run-json artifacts/adb_runs/v4c4_armed_01.json \
  --video artifacts/adb_runs/v4c4_armed_01.mp4 \
  --device cpu \
  --output artifacts/adb_runs/v4c4_armed_01_multitask.json
```

The script runs **no ADB commands, no Android touch, and no gameplay**. It
uses the exact five recorded frame indices and training-compatible preprocessing
from `diagnose_v4c4_recorded_input.py`. It loads the existing checkpoint, then
runs current-action and auxiliary heads in one forward pass, without changing
`EventTimeActionRunner.predict_action` or production policy. The original
`model.pt` and sibling `metadata.json` must still match the failed Armed run.

The result JSON includes all original model-step frame indices and five-head
outputs in `rows`; `inspection_frames` is a short representative selection for
frames near 320, 376, 390, 400, 410, 420, 430, 440 and 444; `inspection_windows`
summarizes index ranges only. The window names do **not** assert which frames
are on/off-road. For that, visually inspect the corresponding old debug MP4
or previous montage. MP4 presentation timestamps are synthetic and cannot be
used for actual screen-capture timing.

## What to look for

1. `action_replay_passed` must be true. A failure to reproduce the originally
   logged action probability (maximum absolute error >0.05 or a 0.5-threshold
   disagreement) means the model/checkpoint, preprocessing or recorded inputs
   no longer match; do not interpret auxiliary-head differences until resolved.
2. Check the old recording around the first bend, then correlate visual lateral
   displacement and heading with `lateral_raw`, `heading_cos2_raw`,
   `heading_sin2_raw`, `heading_vector_norm`, and `edge_risk_probability` at the
   SAME source frame indices. Compare on-track and early-drift states before
   the UI enters a failure overlay. Numeric changes alone are not ground truth.
3. `lateral_raw` is an unconstrained regression intended to estimate lateral
   offset divided by half-road width; its sign and magnitude are not validated
   on this live visual theme. `heading_cos2_raw`, `heading_sin2_raw` are
   regression outputs for `(cos(2*error), sin(2*error))`. The derived half-angle
   is modulo 180 degrees; the raw vector norm need not be one. Do not assume
   that a number crossing zero uniquely defines a corrective direction.
   `edge_risk_probability` is a sigmoid output whose training label was
   `abs(lateral_offset)>=0.70`; it is NOT calibrated as an actual probability
   of crash on the phone. Event-time bins concern expert transitions and are
   not a reliable timer for correcting an off-expert course.
4. If state heads are visually inconsistent before the failure overlay, treat
   them as untrusted on these live frames; collecting real off-expert states and
   checking live visual-domain shift is a training/data problem. If they track
   geometry qualitatively while current-action stays PRESS, that supports
   exploring a **separately validated** state-aware closed-loop controller,
   not immediately enabling a forced-RELEASE rule.

This is a *single-trial diagnostic*, not evidence of policy success or a
substitute for independently measuring video-frame age and actual control
response. Keep the pull request in draft until a controlled closed-loop test
with suitable freshness safeguards succeeds.
