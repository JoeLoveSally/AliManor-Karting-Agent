# Kart-Relative Geometry Teacher POC

## 1. Why this exists

The latest closed-loop experiments show that road orientation alone is not the missing state for the second-bend failure.

The road-axis auxiliary task was learnable, but it did not improve control:

```text
v4-C1 λ_axis=0.10
axis error ≈ 26.2°
stateful target F1 @0.60 = 0.906

v4-C1 λ_axis=0.03
axis error ≈ 27.5°
stateful target F1 @0.60 = 0.849

v4-C1 λ_axis=0
axis head untrained
validation selected threshold = 0.70
```

The `λ_axis=0` checkpoint was then tested in the real ADB closed loop at threshold `0.70`.

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

After the RELEASE at frame 516, the policy remained strongly in KEEP-RELEASE while the kart drifted out of the road. The useful pre-failure diagnostic window is approximately:

```text
frame 532 → frame 548
```

By roughly frame 560 the kart is already outside the useful correction regime. Do not create synthetic "recovery" supervision from the later failure state.

This motivates a different structured state:

```text
kart center
kart heading
road orientation
→ lateral offset
→ heading error
```

The first step is an **offline teacher/audit only**. No new model head is approved until these labels are visually audited.

## 2. Scope

The POC is implemented in:

```text
src/karting_agent/train/kart_pose_pseudo_labels.py
scripts/inspect_kart_relative_geometry.py
configs/geometry_pseudo_labels.yaml
```

It is deliberately not used by runtime control.

## 3. Kart center teacher

The player kart has a stable red/orange chassis in the current recordings. The teacher therefore:

```text
BGR frame
→ HSV warm-color mask
→ gameplay-region ROI
→ connected components
→ reject components without enough red support
→ choose the plausible component near the gameplay anchor
→ merge nearby chassis pieces
→ centroid = kart center
```

The white chicken body is not used as the primary detector because white smoke/effects are common during drifting.

The minimum red-support rule is also important for rejecting failure-dialog coins/buttons, which are warm-colored but not red chassis objects.

This is a dataset-specific weak teacher and must be audited if kart skins change.

## 4. Kart heading teacher

Heading is estimated from PCA of the merged red/orange chassis pixels.

The POC predicts an **axial heading**:

```text
θ == θ + 180°
```

It does not yet decide which end of the kart is the front. That is sufficient for an initial alignment-error diagnostic against the axial road orientation.

`heading_quality` is derived from PCA anisotropy. Approximately circular/noisy components are marked as heading-unusable rather than forced into a direction.

## 5. Local road relation

Global road center-axis pairing had low coverage, so lateral offset does not depend on the old global center-axis teacher.

Instead, once a kart center and dominant road angle exist, the POC probes local cross-sections normal to the road direction:

```text
                 road tangent
                      →

road edge  =============================

          cross section      |
                             |
                    kart  ●  |
                             |

road edge  =============================
```

The probes are shifted slightly forward/backward along the road tangent so the kart sprite itself does not split the road mask.

For each valid cross-section:

```text
road run midpoint → local road center
road run width    → local road width
```

The reported lateral offset is:

```text
lateral_offset_norm = lateral_offset_px / (road_width / 2)
```

Interpretation:

```text
0      = local center
|1|    = road edge
> 1    = outside the inferred road corridor
```

The sign is an image-axis convention from the canonical road normal; it is diagnostic for now and not yet a control semantic such as "left/right of travel".

## 6. Heading error

The POC reports the signed minimum axial difference:

```text
heading_error = kart_heading - road_axis
wrapped to [-90°, 90°)
```

Again, this is image-axis geometry. A future travel-relative left/right semantic would require directed travel orientation.

## 7. Confidence

`KartRoadRelation.confidence` combines:

```text
kart heading quality
× road straight confidence
× (1 - road corner score)
× valid local cross-section fraction
```

This value is for audit/ranking only. No threshold has been approved for training labels.

## 8. Critical-run audit

After syncing the code to the PC, inspect the known failed closed-loop run densely around the failure:

```bash
python scripts/inspect_kart_relative_geometry.py \
  artifacts/adb_runs/adb_20260916T003536Z.mp4 \
  --run-json artifacts/adb_runs/adb_20260916T003536Z.json \
  --frame-start 500 \
  --frame-end 570 \
  --sample-every-frames 4 \
  --max-previews-per-video 40
```

Output:

```text
artifacts/kart_relative_geometry/
└── adb_20260916T003536Z/
    ├── contact_sheet.jpg
    ├── summary.json
    └── frame_*.jpg
```

Each overlay shows:

```text
yellow circle   = detected kart center
magenta axis    = kart axial heading
cyan cross/axis = local inferred road center/orientation
green/red link  = kart-to-road-center offset (inside/outside)
policy text     = nearest p_switch/action/state from the run JSON
```

## 9. Audit gate

Do not train a kart-relative auxiliary head yet.

First inspect the critical window and then a sparse sample across all 15 expert recordings. Required qualitative checks:

```text
1. kart center stays on the chassis through straight and drift frames;
2. heading axis follows the chassis rather than smoke/skid marks;
3. failure UI is rejected rather than detected as a kart;
4. local road center lies between the relevant road boundaries;
5. lateral offset grows before visible off-road failure;
6. heading error changes coherently through the drift;
7. low-confidence/corner frames fail closed rather than producing arbitrary labels.
```

If these checks pass, the next model experiment should supervise **kart-relative state**, not add another road-only objective.

## 10. Training status

Current status:

```text
road orientation teacher          audited / trainable but not control-helpful
kart center teacher               POC / audit required
kart axial heading teacher        POC / audit required
local lateral offset              POC / audit required
road-relative heading error       POC / audit required
corner / next-corner semantics    not ready
runtime geometry controller       explicitly out of scope
```

Deployment remains a learned neural policy. The analytic teacher exists only to create/audit structured supervision and diagnose failures.
