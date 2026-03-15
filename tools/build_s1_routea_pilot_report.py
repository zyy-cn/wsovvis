#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARM_ORDER = [
    "A0_baseline_subset",
    "A1_highconf_subset",
    "A2_lowconf_subset",
    "C0_dupselect_control",
    "C1_cardinality_control",
]

PRIMARY_METRICS = [
    "mean_best_iou",
    "recall_at_0.5",
    "fragmentation_per_gt_instance",
]

DIAGNOSTIC_METRICS = [
    "recall_at_0.7",
    "short_track_ratio",
    "broken_track_ratio",
    "average_track_duration",
    "temporal_inconsistency",
    "unmatched_gt_ratio",
    "over_segmentation_tendency",
    "under_coverage_tendency",
    "predicted_tracks_total",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _arm_root(pilot_root: Path, arm_name: str) -> Path:
    return pilot_root / "arms" / arm_name


def _stop_rule(base: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    base_refined = base["refined"]
    current_refined = current["refined"]
    base_frag = float(base_refined["fragmentation_per_gt_instance"])
    current_frag = float(current_refined["fragmentation_per_gt_instance"])
    frag_improvement_rel = 0.0 if base_frag <= 0.0 else (base_frag - current_frag) / base_frag
    mean_delta = float(current_refined["mean_best_iou"]) - float(base_refined["mean_best_iou"])
    recall_delta = float(current_refined["recall_at_0.5"]) - float(base_refined["recall_at_0.5"])

    improves_mean = mean_delta >= 0.01
    improves_recall = recall_delta >= 0.01
    if improves_mean and improves_recall:
        other_metric_ok = True
    elif improves_mean:
        other_metric_ok = recall_delta >= -0.01
    elif improves_recall:
        other_metric_ok = mean_delta >= -0.01
    else:
        other_metric_ok = False

    coverage_diag_bad = (
        float(current_refined["over_segmentation_tendency"]) > float(base_refined["over_segmentation_tendency"])
        and float(current_refined["under_coverage_tendency"]) > float(base_refined["under_coverage_tendency"])
    )
    predicted_tracks_ok = float(current_refined["predicted_tracks_total"]) >= 0.8 * float(base_refined["predicted_tracks_total"])
    promising = bool(
        frag_improvement_rel >= 0.10
        and (improves_mean or improves_recall)
        and other_metric_ok
        and not coverage_diag_bad
        and predicted_tracks_ok
    )
    return {
        "fragmentation_improvement_rel": float(frag_improvement_rel),
        "mean_best_iou_delta_vs_a0": float(mean_delta),
        "recall_at_0.5_delta_vs_a0": float(recall_delta),
        "coverage_diag_bad": bool(coverage_diag_bad),
        "predicted_tracks_ok": bool(predicted_tracks_ok),
        "promising": promising,
    }


def _render_table(raw_summary: Dict[str, Any], arm_summaries: Dict[str, Dict[str, Any]], stop_rules: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "| arm | mean_best_iou | recall_at_0.5 | fragmentation_per_gt_instance | predicted_tracks_total | over_segmentation_tendency | unmatched_gt_ratio | stop_rule |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| raw_pseudo_tube | {float(raw_summary['mean_best_iou']):.6f} | {float(raw_summary['recall_at_0.5']):.6f} | "
            f"{float(raw_summary['fragmentation_per_gt_instance']):.6f} | {float(raw_summary['predicted_tracks_total']):.0f} | "
            f"{float(raw_summary['over_segmentation_tendency']):.6f} | {float(raw_summary['unmatched_gt_ratio']):.6f} | context |"
        ),
    ]
    for arm_name in ARM_ORDER:
        summary = arm_summaries[arm_name]["refined"]
        stop_value = "baseline" if arm_name == "A0_baseline_subset" else ("promising" if stop_rules[arm_name]["promising"] else "not_promising")
        lines.append(
            f"| {arm_name} | {float(summary['mean_best_iou']):.6f} | {float(summary['recall_at_0.5']):.6f} | "
            f"{float(summary['fragmentation_per_gt_instance']):.6f} | {float(summary['predicted_tracks_total']):.0f} | "
            f"{float(summary['over_segmentation_tendency']):.6f} | {float(summary['unmatched_gt_ratio']):.6f} | {stop_value} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the five-arm Route A pilot results and apply the frozen stop rule.")
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pilot_root = args.pilot_root.resolve()
    arm_summaries: Dict[str, Dict[str, Any]] = {}
    arm_extras: Dict[str, Dict[str, Any]] = {}
    for arm_name in ARM_ORDER:
        arm_root = _arm_root(pilot_root, arm_name)
        arm_summaries[arm_name] = _load_json(arm_root / "eval" / "summary.json")
        extra_path = arm_root / "arm_extra_summary.json"
        arm_extras[arm_name] = _load_json(extra_path) if extra_path.exists() else {}

    baseline = arm_summaries["A0_baseline_subset"]
    raw_summary = baseline["raw"]
    stop_rules: Dict[str, Dict[str, Any]] = {"A0_baseline_subset": {"promising": False}}
    promising_arms: List[str] = []
    for arm_name in ARM_ORDER:
        if arm_name == "A0_baseline_subset":
            continue
        stop_rules[arm_name] = _stop_rule(baseline, arm_summaries[arm_name])
        if stop_rules[arm_name]["promising"]:
            promising_arms.append(arm_name)

    ranked_candidates = sorted(
        (arm_name for arm_name in ARM_ORDER if arm_name != "A0_baseline_subset"),
        key=lambda arm_name: (
            not stop_rules[arm_name]["promising"],
            -stop_rules[arm_name]["fragmentation_improvement_rel"],
            -(stop_rules[arm_name]["mean_best_iou_delta_vs_a0"] + stop_rules[arm_name]["recall_at_0.5_delta_vs_a0"]),
            arm_name,
        ),
    )
    best_arm = ranked_candidates[0] if ranked_candidates else "A0_baseline_subset"
    overall_status = "full_scale_routea_justified" if promising_arms else "full_scale_routea_not_justified"

    payload = {
        "route": "Route A",
        "mode": "diagnostic_five_arm_pilot",
        "baseline_arm": "A0_baseline_subset",
        "arm_order": ARM_ORDER,
        "raw_context": raw_summary,
        "arm_summaries": arm_summaries,
        "arm_extra_summaries": arm_extras,
        "stop_rule_results": stop_rules,
        "pilot_judgment": {
            "overall_status": overall_status,
            "best_arm": best_arm,
            "promising_arms": promising_arms,
            "stop_rule_applied_exactly": True,
        },
        "selected_artifacts": {
            "worked_example_json": str(_arm_root(pilot_root, best_arm) / "eval" / "paired_worked_example.json"),
            "worked_example_svg": str(_arm_root(pilot_root, best_arm) / "eval" / "paired_worked_example.svg"),
            "failure_example_json": str(_arm_root(pilot_root, best_arm) / "eval" / "failure_case_example.json"),
            "failure_example_svg": str(_arm_root(pilot_root, best_arm) / "eval" / "failure_case_example.svg"),
        },
        "metric_surface": {
            "primary_metrics": PRIMARY_METRICS,
            "diagnostic_metrics": DIAGNOSTIC_METRICS,
        },
    }
    _dump_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_table(raw_summary, arm_summaries, stop_rules), encoding="utf-8")
    print(json.dumps(payload["pilot_judgment"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
