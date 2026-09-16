# Kart-Relative Geometry Teacher and v4-C2

## 1. Why this exists

The latest closed-loop experiments show that road orientation alone is not the missing state for the second-bend failure.

The road-axis auxiliary task was learnable, but did not improve control. The `λ_axis=0` control ablation was then tested in the real ADB closed loop at validation-selected threshold `0.70`.

Runtime performance was healthy:

```text
input fps      ≈ 55.6
control fps    ≈ 29.2
inference mean ≈ 6.7 ms
inference p95  ≈ 9.2 ms
dropped frames = 1
```

The control timeline was:

```text
PRESS   source frame 399
RELEASE source frame 516
PRESS   source frame 595
```

After frame 516, the policy remained strongly in KEEP-RELEASE while the kart drifted out. The useful pre-failure diagnostic window is approximately frame `532–548`; later fully off-road/failure frames must not be treated as synthetic recovery demonstrations.

The missing state is better described as:

```text
kart center
kart heading
local road orientation
→ lateral offset
→ heading error
```

Analytic CV remains an offline teacher/debugger only. Runtime deployment stays neural.

## 2. Teacher implementation

Implemented in:

```text
src/karting_agent/train/kart_pose_pseudo_labels.py
scripts/inspect_kart_relative_geometry.py
configs/geometry_pseudo_labels.yaml
```

The kart detector uses red/orange chassis evidence inside a gameplay ROI. Nearby chassis pieces are merged, the centroid becomes the kart center, and PCA provides an axial heading. The mostly-white chicken body is intentionally not the primary pose cue because drift smoke/effects are also white.

The road relation does not blindly reuse the globally dominant Hough axis. Around intersections, a long remote road segment can dominate global support. The teacher therefore clusters nearby Hough edge evidence, evaluates candidate local axes around the kart, probes cross-sections normal to each candidate, and selects a plausible local corridor using lateral proximity, kart-heading compatibility, local support, and valid-probe count.

For a selected corridor:

```text
lateral_offset_norm = lateral_offset_px / (road_width / 2)
```

so `0` is local center, `|1|` is approximately an edge, and `>1` is outside the inferred corridor.

Heading error is the signed axial difference between kart and road axes, wrapped to `[-90°, 90°)`.

## 3. Critical-run audit result

The known failed closed-loop run was audited densely over frames `500–570`, sampling every 4 frames:

```text
previews          = 18
kart detected     = 1.000
heading usable    = 1.000
relation usable   = 0.833
offroad           = 0.133
mean |heading err|= 39.2°
mean |lateral|    = 0.49 road-half-widths
```

The overlay audit shows that kart center and axial heading remain trackable through the pre-failure region, while kart-relative geometry degrades before the visible failure. This is enough to proceed to a full expert-video label gate, but it is not enough to assume the teacher generalizes across all 15 visual themes.

## 4. v4-C2 pseudo-labels

Implemented in:

```text
scripts/build_kart_relative_pseudo_labels.py
src/karting_agent/train/kart_relative_labels.py
```

Every v3 observation frame receives one JSONL row. Invalid or low-confidence teacher outputs are retained with `weight=0`, so dataset alignment stays explicit.

The supervised targets are:

### 4.1 Lateral offset

```text
lateral_target = clip(lateral_offset_norm, -1.25, +1.25)
```

Training uses confidence-weighted Smooth-L1 loss.

### 4.2 Axial heading error

A heading error is axial. `+90°` and `-90°` describe the same perpendicular axis, so direct scalar regression or ordinary `(cos e, sin e)` would introduce the wrong periodicity. v4-C2 encodes:

```text
(cos 2e, sin 2e)
```

and trains with confidence-weighted cosine loss.

### 4.3 Edge-risk auxiliary

A pure `offroad` head would be nearly degenerate on expert demonstrations because true off-road positives should be rare. v4-C2 instead uses a pre-failure edge-risk target:

```text
edge_risk = |lateral_offset_norm| >= 0.70
```

This is a diagnostic/representation auxiliary, not a hand-written runtime safety rule.

The current label gate is:

```yaml
kart_relative_label:
  min_confidence: 0.05
  lateral_clip_abs: 1.25
  edge_risk_threshold: 0.70
```

These thresholds are provisional until the 15-video label distribution and sparse visual audit are reviewed.

## 5. v4-C2 model

Implemented in:

```text
src/karting_agent/model/kart_relative_supervised.py
scripts/train_model_v4c2.py
scripts/evaluate_model_v4c2.py
configs/train_v4c2.yaml
```

The v3 control path is unchanged:

```text
5 RGB frames → 15 channels → MobileNetV3-Small → visual feature
                                             ├→ future-action head
                                             ├→ lateral head
                                             ├→ axial heading-error head
                                             └→ edge-risk head
visual feature + current PRESS/RELEASE → KEEP/SWITCH head
```

All kart-relative heads are training auxiliaries from the shared visual feature. They are not fed as analytic runtime inputs.

Initial loss:

```text
L = L_switch
  + 0.50 * L_future_action
  + 0.03 * L_lateral
  + 0.03 * L_heading
  + 0.01 * L_edge_risk
```

The weights are deliberately conservative because v4-C1 demonstrated that a learnable auxiliary objective can still damage control when it competes too strongly with the primary representation.

Checkpoint selection and early stopping continue to monitor **validation switch loss**, not geometry loss.

## 6. Evaluation

`evaluate_model_v4c2.py` preserves the existing static and stateful control evaluation and additionally reports:

```text
weighted lateral MAE
weighted axial heading-error degrees
edge-risk F1
```

The first requirement is that the auxiliary quantities are actually learnable on held-out videos. The control requirement remains that stateful behavior must not regress relative to the v3-family baseline.

Do not select a control threshold from test. Select it on validation, then report test once using that threshold.

## 7. Required gate before training

The next step is not Spark training. First build labels on all 19,844 v3 observation frames and inspect per-video coverage:

```text
pose rate
heading rate
relation rate
accepted rate
edge-risk positive rate
true off-road rate
mean teacher weight
mean |lateral offset|
mean |heading error|
```

Pay special attention to held-out videos:

```text
validation: 173545, 174012
test:       173426, 173835
```

Reject or revise the teacher if a visual theme collapses, if local road selection is systematically wrong, or if edge-risk has essentially no positive examples.

Then run a sparse visual overlay audit across all 15 expert videos. Only after both quantitative and visual gates pass should v4-C2 be trained.

## 8. Covariate shift remains independent

Kart-relative supervision may improve causal representation, but it does not solve expert-only behavior-cloning distribution shift. If closed-loop deviations remain after v4-C2, valid pre-failure correction data must be collected iteratively. Do not manufacture recovery labels from states in which the kart is already irreversibly off track.

## 9. Current status

```text
road orientation teacher       audited; learnable; not control-helpful
kart center/heading teacher     critical-run audit passed; 15-video gate pending
local kart-road relation        critical-run audit passed; 15-video gate pending
v4-C2 code path                 implemented; local tests pending
v4-C2 full training             blocked on label + visual gate
corner / next-corner semantics postponed
runtime geometry controller     explicitly out of scope
```
