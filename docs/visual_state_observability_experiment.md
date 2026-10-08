# Visual-State Observability Gate (read-only)

## Hypothesis

The game's two-state control pattern may be learnable with a small model. Before
replacing the BC policy, determine whether CV-derived kart/road state is valid,
stable and sufficient at the recoverable failure boundary. This experiment
never emits or executes control actions.

Existing KartPose, road-mask / rectilinear geometry and TemporalKartRoadTracker
are reused. No new neural weights are trained.

## Observations and limits

| State | Source | Limitation |
| --- | --- | --- |
| Kart center and heading | red/orange chassis detector | Heading is axial (180-degree periodic), not travel direction |
| Lateral offset | selected local road corridor | Sign belongs to the selected axis normal; not a globally consistent steering direction |
| Heading error | kart and road axial difference | Not a signed vehicle yaw |
| Lateral / heading change rates | consecutive valid outputs | Screen-space proxies, NOT calibrated physical velocities |
| Validity | relation confidence and temporal continuity gate | Threshold is not calibrated classification correctness |
| Visible corner intersection | rectilinear geometry teacher | Intersection is not necessarily the next corner |
| Next corner direction / distance | unknown | Intentionally null until verified |

Finite-difference rates require consecutive valid samples with a gap at most
250 ms and a road-axis jump at most 15 degrees. The confidence starting
threshold of 0.05 comes from the pre-existing pseudo-label gate; it does not
prove perceptual correctness.

## Sampling

First inspect the three existing closed-loop ADB recordings and two
held-out expert videos, sampling contiguous gameplay windows at fixed frame
intervals. Use failure-prevention frames, not only obviously offroad frames.

Recorded ADB MP4s use synthetic-CFR timing. The matching JSON's decoded source
frame timestamps must be used, not MP4 FPS. For expert videos without paired
JSON, OpenCV container PTS / FPS is approximate.

Example on Spark:

    python scripts/audit_visual_state_observability.py \
      artifacts/adb_runs/v4c2_baseline_repro_01.mp4 \
      --run-json artifacts/adb_runs/v4c2_baseline_repro_01.json \
      --frame-start 425 --frame-end 965 \
      --sample-every-frames 3 --max-samples 181 \
      --preview-every-samples 5 \
      --output-dir artifacts/analysis/state_observability_baseline_01

Outputs per video: measurements.jsonl, previews, contact_sheet.jpg,
manual_review.csv and summary.json. Files are created under the selected
unique output directory; existing results are not overwritten.

## Review and advancement gates

1. Visually inspect KartPose, selected corridor, lateral sign, heading axis
   and axis-intersection overlay. Enter human judgment in manual_review.csv.
2. Report valid coverage separately for normal gameplay and recoverable
   pre-failure windows; aggregate coverage does not prove accuracy.
3. Inspect abrupt rate spikes, missing corridors and tracker transitions.
   Reject misleading derivatives at axis reassignments or long gaps.
4. Compare visually similar states that require different expert actions.
   If ambiguity remains, test action age, motion trend, or corner phase.
5. Do not assume an intersection means an upcoming turn. Do not use
   unreviewed pseudo-labels as real-world/physics ground truth.
6. If valid state remains reliably observable in the critical recovery window,
   trial a tiny state-conditioned KEEP/SWITCH decision baseline and compare
   closed-loop behaviour against frozen V4-C2. Otherwise repair quantization
   or keep learned visual features as the main representation.

This diagnostic never modifies the ADB/Armed runner or physical executor.
