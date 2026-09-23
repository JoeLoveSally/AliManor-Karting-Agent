# V4-C2: post-switch transients and recovery experiment

## Baseline and evidence

Keep the 2026-09-17 `ec5b58f` runtime and the existing `mobilenet_v3_small_v4c2_temporal_v2/model.pt` + `metadata_h200.json` checkpoint as the baseline. The historical `adb_20260917T060240Z` run reached about 14.11 s on the game's timer; subsequent runs `v4c2_baseline_repro_01` (current branch runtime) and `v4c2_0917_code_repro_01` (`ec5b58f` runtime) failed around game timers 7.22 s and 7.15 s, respectively. The later 15-second process duration includes inference on the failure overlay and is not driving time. These are three separate closed-loop trajectories, not three counterfactual trajectories on identical images.

Do not infer a precise model-weight hash from the historical launch command; it identifies the path, not the file bytes at the historical time.

## What must NOT be changed blindly

In the 9/17 run, the FIRST PRESS is followed within about 3 ms by H100=0.98/H200=0.81 even though the next logged RELEASE is approximately 1.73 seconds later. The other two runs exhibit the same near-immediate high short-horizon switch probabilities after ordinary, sustained actions. Thus high H100/H200 just after a switch is **not** a validated short corrective pulse and is not evidence for always reserving a reversal at minimum-hold expiry. In particular, do not globally enable `reserve_bounded_reversal_during_min_hold` or `execute_short_horizon_overdue` as a purported recovery fix.

The 200-ms visual stack spans multiple commanded action states following a switch, while the model conditions its switch head only on the *current* pressed state. This mismatch is a plausible source of transient predictions, not an established causal explanation. A recorded command timestamp is not the time when the game actually rendered or responded to that command.

## Audit implementation

`scripts/audit_v4c2_action_history.py` merges regular and timer-driven PRESS/RELEASE events, reconstructs the **commanded** action state at each logged input-frame timestamp, and reports: (i) stacks spanning multiple commanded states, (ii) simultaneous H100/H200 warnings during the first 100 ms after a switch, and (iii) such warnings that were NOT followed by a logged reversal in the next 500 ms. An action executed after a step's observation is excluded from that observation's input frames. It never changes controls or loads model weights.

Run the script from the `exp/v4c2-action-history-audit` branch, pointing at the three JSON files in `artifacts/adb_runs`, or use the explicit paths under your local artifact directory. Only the two new reproduction files may reside at the default paths; locate the 9/17 historical JSON separately.

Three already supplied runs gave the following audit results:

| Run | Steps | Logged transitions | Stacks mixing commanded states | H100 & H200 high within 100 ms of a switch | Of these, no logged reversal within 500 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9/17 long run | 470 | 16 | 100 | 11 | 8 |
| Current-runtime reproduction | 451 | 7 | 43 | 5 | 2 |
| Old-runtime reproduction | 455 | 11 | 65 | 8 | 1 |

The final column describes **what the original policy did**, not whether a hypothetical reversal would have been good or bad. These counts use raw command timestamps and include no independent ground-truth optimal actions.

## Next experimental candidate (not enabled)

1. Preserve model weights and the old H200/H300 controller. Before any active override, define a visually grounded off-track-risk signal and establish that it discriminates *pre-failure* frames from normal tight turns on **all three** existing recordings. A simple HSV road-pixel occupancy threshold is not sufficient: on the successful run it can be low during normal bends and on the failing runs it becomes near zero only after the car has already visibly left the road.
2. The active candidate, if a sufficiently early and selective risk signal exists, would combine that signal with a persistent, state-aware model prediction and a bounded pulse with its own cooldown. Require explicit evidence for which button state steers toward safety; a low road-pixel fraction alone cannot determine PRESS versus RELEASE.
3. If no visual signal meets those tests, leave the original controller untouched and prototype an additional per-frame action-history / time-since-transition conditioning feature on **the V4-C2 model**, trained from existing timestamped human data. Because it changes model inputs, this would be a new checkpoint with its own evaluation; it must not be described as an unchanged V4-C2 weight replay.
4. Evaluate the first divergence from the unchanged baseline offline. Only a separate, clearly named Armed run can establish whether any candidate improves closed-loop progress. Never use the reference video's later frames as if the altered controller had actually visited them.

This branch adds only a read-only audit and tests. It is not an active recovery controller or a claim of improved performance.
