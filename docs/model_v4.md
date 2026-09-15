# Model v4: Structured Model-Driven Policy

## 1. Goal

v4 keeps the deployed controller fully model-driven while separating two questions that v1-v3 mixed together:

1. how temporal information should be represented;
2. whether visual supervision should force the model to understand causal track/kart geometry instead of relying on shortcuts.

Traditional computer vision is allowed only as an offline pseudo-label teacher and shadow debugger. It never sends runtime control actions.

## 2. v3 reference

v3 remains the best control baseline at the end of the temporal ablation.

```text
5 RGB frames
→ concatenate as 15 channels
→ MobileNetV3-Small
→ visual feature
+ current PRESS / RELEASE embedding
→ KEEP / SWITCH
```

The expanded first convolution is initialized by repeating pretrained RGB weights and dividing by `frame_stack`, but it is trainable. Therefore v3 is only equivalent to temporal averaging at initialization, not after training.

Previous temporal counterfactual analysis confirmed that the trained model uses frame order:

```text
repeat_latest flip rate = 0.351
reverse       flip rate = 0.622
```

Relevant v3 stateful reference at threshold `0.60`:

```text
target transition F1 = 0.918
matched              = 78 / 88
predicted            = 82
short recall         = 0.778
chatter <100ms       = 1
chatter <200ms       = 9
observation F1       = 0.753
```

## 3. Controlled temporal ablation

The v3 → v4-A → v4-A2 comparison kept data, split, 200ms/5-frame history, preprocessing, targets, state conditioning, sampling, optimizer and checkpoint semantics fixed.

### v4-A: shared per-frame CNN + GRU

Best useful stateful result:

```text
target transition F1 = 0.892
matched              = 74 / 88
short recall         = 0.722
observation F1       = 0.554
```

GRU was smoother but missed useful short corrections.

### v4-A2: shared per-frame CNN + ordered temporal MLP

Best target-timeline point was threshold `0.70`:

```text
target transition F1 = 0.907
matched              = 78 / 88
short recall         = 0.722
observation F1       = 0.744
chatter <200ms       = 4
```

It recovered some performance versus GRU but still did not beat v3, especially on short-correction recall.

### Temporal conclusion

```text
v3 15-channel temporal CNN         → best overall control baseline
v4-A per-frame CNN + GRU           → worse, overly smooth
v4-A2 per-frame CNN + temporal MLP → closer, still no meaningful win
```

Therefore temporal architecture search is paused:

- do not implement v4-A/v4-A2 ADB runtime;
- do not continue with 400/800ms GRU history now;
- do not add TCN/attention simply to continue architecture search.

The next experiment returns to the v3 control formulation and changes visual supervision instead.

## 4. Track geometry model: piecewise straight, not continuous curvature

The recorded game tracks are not ordinary continuously curved racing roads. Their useful structure is better described as:

```text
straight segment
→ discrete corner
→ straight segment
→ discrete corner
```

Screen projection may make the world axes appear diagonal, but the topology remains piecewise rectilinear. The curved visual trajectory during drifting is primarily vehicle motion, not the road centerline itself.

Therefore v4-C must **not** treat continuous centerline curvature as the primary geometry target.

The geometry priorities are now:

```text
1. drivable-road / road-region representation
2. visible local road center-axis direction
3. visible center-axis corner presence and proximity
4. later: travel-relative next-corner direction
5. later: kart center / heading
6. later: lateral offset and heading error relative to current road axis
```

Continuous curvature is deferred unless later evidence shows it adds value.

## 5. Why kart-relative state matters

The important closed-loop failure observed in v3 was not simply "the model cannot see a bend".

During the critical second bend, after RELEASE the kart progressively moved/rotated toward the road edge while the policy continued to KEEP RELEASE. This is naturally described by state variables such as:

```text
current road axis
+ kart lateral offset
+ kart heading error
+ distance to a relevant corner
+ current PRESS / RELEASE state
```

That motivates structured visual supervision. It does **not** imply the deployed controller should become a hand-written geometric controller.

## 6. v4-C target architecture

Keep the best v3 temporal/control path and add auxiliary geometry tasks:

```text
15-channel v3 visual history
        ↓
MobileNetV3 backbone
        ↓
shared visual feature
   ┌─────────────┬──────────────────┬───────────────────┐
   ↓             ↓                  ↓                   ↓
KEEP/SWITCH   future action      road/axis heads     later kart-pose heads
   head          head               │                   │
   │                              training-only structured supervision
current action
   │
learned policy
```

The deployment path stays neural. Analytic CV is used only to create labels and diagnose failures.

## 7. Geometry teacher: feasibility stage

The first road-mask POC used one blue HSV range. The 15-video audit showed that this assumption was too narrow: several dark/blue/cyan/green/neutral themes produced effectively zero mask coverage while others worked.

The revised teacher uses two stages.

### 7.1 Multi-theme road candidate mask

The POC ORs broad HSV bands covering:

- blue;
- cyan / teal;
- green;
- dark neutral road themes.

Morphology and connected-component filtering are then applied. This remains a teacher candidate rather than ground truth.

### 7.2 Edge evidence → road center axis

A key audit result was that Hough lines extracted from `Canny(mask)` naturally lie on **road boundaries**, not road centers. That behavior is correct and useful as intermediate evidence, but raw edge lines must not become model targets.

The current teacher therefore uses:

```text
road candidate mask
→ Canny boundary edges
→ Hough long edge segments
→ cluster approximately parallel edges by orientation
→ pair overlapping parallel edges with plausible road width
→ take the midpoint between each edge pair
→ inferred road center axis
```

For one straight corridor:

```text
road edge A  ─────────────────────

             ===== center axis ====

road edge B  ─────────────────────
```

Pair acceptance is constrained by:

```text
minimum/maximum road width
minimum longitudinal overlap
orientation tolerance
```

all normalized by image size where appropriate.

This explicitly separates three concepts:

```text
Hough segment      = road-edge evidence
center axis        = inferred training/diagnostic geometry
corner             = intersection of inferred center axes
```

A raw edge intersection is no longer called a road corner.

### 7.3 Center-axis corner diagnostic

When both a primary and secondary road center axis can be inferred:

```text
primary center axis
        ×
secondary center axis
        ↓
visible center-axis intersection
```

The intersection is accepted only when it lies within the frame (with a small extension allowance) and near the finite extents of both inferred axes.

Per frame the POC reports:

```text
geometry_class:
  unknown / edge_only / straight / mixed_center_axes / corner_visible

primary_angle_deg
secondary_angle_deg
primary_center_axis
secondary_center_axis
straight_confidence
corner_score
corner_visible
corner_x_norm / corner_y_norm
corner_distance_norm
```

The overlay now uses:

```text
green        = road candidate mask
thin yellow  = primary road-edge evidence
thin purple  = secondary road-edge evidence
thick cyan   = inferred primary road center axis
thick magenta= inferred secondary road center axis
red circle   = center-axis intersection
white X      = diagnostic anchor
```

The center-axis lines, not the thin edge lines, are the important audit output.

## 8. Important semantic limit: visible corner is not next corner

The current teacher deliberately does **not** output:

```text
next corner = left/right
```

or claim that the white diagnostic anchor is the kart center.

A road mask alone does not provide travel direction. A visible center-axis intersection can still be ahead, behind, or unrelated to the vehicle's current motion.

Reliable travel-relative labels require an additional teacher for at least one of:

```text
kart center + heading
or
kart motion vector from adjacent frames
```

Only after that exists can we define:

```text
current road direction relative to travel
next corner distance
next corner direction
lateral offset
heading error
```

without inventing semantics that the teacher cannot support.

## 9. Current geometry audit outputs

Run:

```bash
python scripts/inspect_geometry_pseudo_labels.py \
  data/raw/video_*.mp4 \
  --max-previews-per-video 12
```

Outputs:

```text
artifacts/geometry_pseudo_labels/
├── overview_contact_sheet.jpg
├── index.json
└── <video>/
    ├── contact_sheet.jpg
    ├── frame_XXXXXX.jpg
    ├── frame_XXXXXX_mask.png
    └── summary.json
```

Console summary now distinguishes:

```text
edge_axis
center_axis
secondary_axis
corner
```

so high Hough coverage cannot be mistaken for high center-axis-label coverage.

`index.json` also records that these labels are not yet available:

```text
next_corner_direction
kart_center
kart_heading
lateral_offset
heading_error
```

## 10. Gate before model training

Do **not** generate full pseudo-labels or train v4-C until the center-axis teacher passes audit.

Inspect all 15 video themes for:

1. road candidate coverage on straight segments;
2. background/UI false positives;
3. stability through patterned road themes;
4. paired center axis lying between the two real road boundaries;
5. plausible inferred road width;
6. secondary center-axis detection around actual corners;
7. false edge pairings between unrelated parallel road fragments;
8. center-axis corner intersections near real topology changes;
9. center-axis usability rate by video/theme.

A label family only enters training if its teacher quality is high enough to be more informative than noisy.

## 11. Planned supervision stages

Do not add every geometry target at once.

### v4-C1: center-axis orientation supervision

The preferred first target is now the **inferred road center-axis orientation**, not raw segmentation and not raw Hough-edge angle.

Conceptually:

```text
L = L_switch
  + λ_action * L_future_action
  + λ_axis * L_center_axis
```

Axial orientation should avoid angle-wrap ambiguity by predicting:

```text
(cos 2θ, sin 2θ)
```

rather than direct degrees.

Road segmentation may still be retained later as a weaker auxiliary task if its audit quality justifies it, but it is no longer the default first geometry loss.

### v4-C2: visible center-axis corner state

Only if center-axis corner labels are reliable:

```text
corner visible / not visible
corner location or normalized proximity
```

Do not call this `next_corner` until travel direction is known.

### v4-C3: kart-relative geometry

Once kart position/heading supervision is reliable:

```text
kart center
kart heading
current-road-axis relation
lateral offset
heading error
travel-relative next-corner distance/direction
```

These targets are more directly tied to the second-bend failure than continuous road curvature.

## 12. Geometry loss policy

Every auxiliary term is a hyperparameter, not a convention:

```text
L = L_switch
  + λ_action * L_future_action
  + λ_axis * L_center_axis
  + λ_corner * L_corner
  + optional λ_road * L_road
  + later λ_pose * L_pose
```

Targets are added incrementally so causal attribution remains possible.

A geometry head is retained only if:

- its held-out geometry metric is meaningful;
- it does not degrade v3 stateful control metrics;
- ideally it improves closed-loop behavior after the offline gate.

## 13. Covariate shift remains independent

Structured perception does not solve expert-only behavior-cloning covariate shift.

Even with better road/kart features, a model-driven closed loop can create observations absent from the expert data after an earlier timing error. Later iterations still need valid pre-failure deviation/recovery examples through iterative behavior cloning or a DAgger-like process.

Do not create impossible recovery labels after the kart is already irreversibly off track.

## 14. Shadow teacher

Keep the analytic teacher after training for offline post-run diagnosis:

```text
recorded MP4
   ├── learned structured visual prediction
   ├── analytic center-axis teacher geometry
   └── policy KEEP/SWITCH output
```

Failure classification:

```text
teacher plausible, learned geometry wrong
→ perception/representation problem

geometry plausible, policy wrong
→ control problem

both plausible, observation off expert distribution
→ data/covariate-shift problem
```

The teacher never controls the kart.

## 15. Current implementation order

```text
1. run center-axis teacher on all 15 videos
2. audit overview and suspicious per-video sheets
3. quantify edge-axis / center-axis / corner coverage by theme
4. tune edge-pair thresholds only if visual failure modes are systematic
5. decide whether center-axis orientation is trustworthy enough for v4-C1
6. only then implement full pseudo-label generation
7. first model experiment: v3 + center-axis orientation auxiliary head
8. stateful offline comparison against v3 reference
9. only if offline gate passes, run ADB closed-loop A/B
10. add kart/travel-direction teacher before travel-relative corner labels
11. preserve failure/deviation runs for later DAgger-like iteration
```

## 16. What v4 is not

v4 is not a hand-written geometric controller.

v4 is not based on the claim that v3 lacked temporal information.

v4 is not committed to GRU or explicit per-frame encoding when the evidence does not support them.

v4 is not based on continuous road curvature when the game geometry is mostly piecewise straight.

The project remains model-driven: analytic CV creates training supervision and diagnostics; deployment remains neural perception plus learned control.
