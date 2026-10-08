#!/usr/bin/env python3
"""Read-only attribution of early V4-C2 pending switches from saved prefix reports.

Diagnoses the H200/H300 threshold-crossing interpolation and H100->H200->H300
monotonicity gate at the *first logged pending-plan change*. Hybrid probabilities
are mathematical interventions, NOT calibrated model outputs, deployable model
configurations, or actual closed-loop control trajectories.

No torch, OpenCV, video decode, ADB, or write to model/runner code.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def warning_gate(probabilities, threshold: float) -> dict:
    if len(probabilities) != 3:
        raise ValueError("V4-C2 analysis requires H100/H200/H300")
    a, b, c = map(float, probabilities)
    if not 0.0 < threshold < 1.0 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in (a, b, c)):
        raise ValueError("invalid threshold/probabilities")
    monotonic = a <= b + 1e-6 and b <= c + 1e-6
    crosses = b < threshold <= c
    # Raw interpolation intentionally omits monotonicity gating, minimum hold,
    # pending_advance_ms and event-loop scheduling.
    raw_delay = (
        100.0 * (threshold - b) / (c - b)
        if crosses and c > b else None
    )
    return {
        "h100_gt_h200": a > b + 1e-6,
        "h200_gt_h300": b > c + 1e-6,
        "monotonic": monotonic,
        "h300_at_least_threshold": c >= threshold,
        "threshold_crosses_h200_h300": crosses,
        "anticipation_gate_eligible": monotonic and crosses,
        "ungated_interpolated_delay_ms": raw_delay,
    }


def analyze_first_plan(run_name: str, key: str, entry: dict) -> dict:
    if entry.get("candidate_horizons") != "full":
        raise ValueError(f"{run_name}/{key} is not full residual variant")
    first = entry.get("first_pending_plan_change")
    if not first:
        return {"run": run_name, "variant": key, "first_pending_plan_change": None}
    base = tuple(map(float, first["logged_probabilities"]))
    cand = tuple(map(float, first["candidate_probabilities"]))
    tb = float(entry["baseline_threshold_all_horizons"])
    tc = float(entry["candidate_threshold_all_horizons"])

    # The two hybrids hold the other horizon fixed. They expose interpolation
    # sensitivity and gating violations, not runnable model predictions.
    vectors = {
        "baseline": (base, tb),
        "full_candidate": (cand, tc),
        "h200_only_synthetic": ((base[0], cand[1], base[2]), tc),
        "h300_only_synthetic": ((base[0], base[1], cand[2]), tc),
    }
    gates = {name: warning_gate(p, thr) for name, (p, thr) in vectors.items()}

    # For physically consistent intervention attribution, separate the
    # calibration-independent 0.60 question from the proposed 0.76 threshold:
    # a higher threshold itself changes the candidate before any model effect.
    matched_threshold = tb
    sensitivity = {
        "baseline": warning_gate(base, matched_threshold),
        "full_candidate": warning_gate(cand, matched_threshold),
        "h200_only_synthetic": warning_gate(
            (base[0], cand[1], base[2]), matched_threshold),
        "h300_only_synthetic": warning_gate(
            (base[0], base[1], cand[2]), matched_threshold),
    }
    base_raw = sensitivity["baseline"]["ungated_interpolated_delay_ms"]
    for name, part in sensitivity.items():
        raw = part["ungated_interpolated_delay_ms"]
        part["delay_change_from_baseline_ms"] = (
            raw - base_raw if raw is not None and base_raw is not None else None
        )

    return {
        "run": run_name,
        "variant": key,
        "step": first["step"],
        "observation_ms": first["observation_ms"],
        "baseline_threshold": tb,
        "candidate_threshold": tc,
        "baseline_probabilities": base,
        "adapter_probabilities": cand,
        "reference_due_ms": first.get("reference_due_ms"),
        "candidate_due_ms": first.get("candidate_due_ms"),
        "actual_scheduler_gate": gates,
        "fixed_baseline_threshold_sensitivity": sensitivity,
        "first_difference": entry.get("first_difference"),
        "status": entry.get("status"),
        "warning": (
            "Analytic first-plan attribution only. Ungated delays intentionally "
            "ignore H100 monotonicity, min-hold, pending advance, callback order, "
            "and later observations. Synthetic hybrid cannot be used as a model."
        ),
    }


def analyze_reports(documents: list[dict]) -> dict:
    rows = []
    for report in documents:
        runs = report.get("runs")
        if not isinstance(runs, dict):
            raise ValueError("expected candidate-prefix report with runs mapping")
        for name, entries in runs.items():
            for key, entry in entries.items():
                if entry.get("candidate_horizons") == "full":
                    rows.append(analyze_first_plan(name, key, entry))
    return {"analysis": "first-pending analytic attribution; NO counterfactual trajectory",
            "cases": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    result = analyze_reports(docs)
    for row in result["cases"]:
        if "first_pending_plan_change" in row:
            print(json.dumps(row, ensure_ascii=False))
            continue
        gates = row["fixed_baseline_threshold_sensitivity"]
        print(json.dumps({
            "run": row["run"], "variant": row["variant"],
            "step": row["step"], "status": row["status"],
            "reference_due_ms": row["reference_due_ms"],
            "candidate_due_ms": row["candidate_due_ms"],
            "base_raw_delay_ms": gates["baseline"]["ungated_interpolated_delay_ms"],
            "full_raw_delay_delta_ms": gates["full_candidate"]["delay_change_from_baseline_ms"],
            "h200_raw_delay_delta_ms": gates["h200_only_synthetic"]["delay_change_from_baseline_ms"],
            "h300_raw_delay_delta_ms": gates["h300_only_synthetic"]["delay_change_from_baseline_ms"],
            "h200_only_monotone": gates["h200_only_synthetic"]["monotonic"],
            "full_gate_eligible": row["actual_scheduler_gate"]["full_candidate"]["anticipation_gate_eligible"],
        }, ensure_ascii=False), flush=True)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"report exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
