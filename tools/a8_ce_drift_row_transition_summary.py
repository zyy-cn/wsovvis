#!/usr/bin/env python3
"""Compact reporting utility for A8 CE drift-onset audit outputs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args(argv)

    audit_dir = args.audit_dir
    out_dir = args.output_dir or audit_dir / "compact_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_csv = audit_dir / "epoch_level_drift_table.csv"
    trans_csv = audit_dir / "row_transition_events.csv"
    hub_csv = audit_dir / "hub_onset_table.csv"
    if not epoch_csv.exists():
        raise SystemExit(f"Missing {epoch_csv}")

    epochs = read_csv(epoch_csv)
    transitions = read_csv(trans_csv) if trans_csv.exists() else []
    hubs = read_csv(hub_csv) if hub_csv.exists() else []

    # Best/worst points.
    best_micro = max(epochs, key=lambda r: as_float(r.get("micro_top1"), -1)) if epochs else None
    best_macro = max(epochs, key=lambda r: as_float(r.get("macro_rank1"), -1)) if epochs else None
    best_rank = min(epochs, key=lambda r: as_float(r.get("mean_normalized_gt_rank"), 1e9)) if epochs else None
    min_wrong = min(epochs, key=lambda r: as_int(r.get("total_wrong_rows"), 10**9)) if epochs else None

    # Onset by first local regression in micro or correct-to-wrong.
    onset = None
    prev = None
    for r in epochs:
        if prev is not None:
            micro_drop = as_float(r.get("micro_top1")) < as_float(prev.get("micro_top1"))
            c2w_rise = as_int(r.get("correct_to_wrong_vs_baseline")) > as_int(prev.get("correct_to_wrong_vs_baseline"))
            if micro_drop or c2w_rise:
                onset = {
                    "epoch": as_int(r.get("epoch")),
                    "prev_epoch": as_int(prev.get("epoch")),
                    "micro_drop": micro_drop,
                    "correct_to_wrong_rise": c2w_rise,
                }
                break
        prev = r

    # Transition groups.
    transition_counter = Counter(r.get("transition", "") for r in transitions)
    c2w_by_gt = Counter()
    c2w_by_top1 = Counter()
    c2w_pseudo_bad = 0
    for r in transitions:
        if r.get("transition") == "correct_to_wrong":
            c2w_by_gt[(r.get("gt_raw_id", ""), r.get("gt_class_name", ""))] += 1
            c2w_by_top1[(r.get("epoch_top1_raw_id", ""), r.get("epoch_top1_class_name", ""))] += 1
            if str(r.get("pseudo_matches_gt", "")) == "0" and str(r.get("top1_matches_pseudo", "")) == "1":
                c2w_pseudo_bad += 1

    # Hub peak table.
    hub_peak = {}
    for r in hubs:
        hid = r.get("hub_raw_id", "")
        n = as_int(r.get("absorbed_wrong_rows"))
        old = hub_peak.get(hid)
        if old is None or n > old["absorbed_wrong_rows"]:
            hub_peak[hid] = {
                "hub_raw_id": hid,
                "peak_epoch": as_int(r.get("epoch")),
                "absorbed_wrong_rows": n,
                "source_class_count": as_int(r.get("source_class_count")),
                "mean_wrong_abs_gap": as_float(r.get("mean_wrong_abs_gap")),
                "is_tracked_hub": as_int(r.get("is_tracked_hub")),
            }
    hub_peak_rows = sorted(hub_peak.values(), key=lambda x: -x["absorbed_wrong_rows"])[: args.topk]
    write_csv(out_dir / "top_hub_peaks.csv", hub_peak_rows)

    top_c2w_gt = [
        {"gt_raw_id": k[0], "gt_class_name": k[1], "correct_to_wrong_rows": v}
        for k, v in c2w_by_gt.most_common(args.topk)
    ]
    top_c2w_top1 = [
        {"wrong_top1_raw_id": k[0], "wrong_top1_class_name": k[1], "correct_to_wrong_rows": v}
        for k, v in c2w_by_top1.most_common(args.topk)
    ]
    write_csv(out_dir / "top_correct_to_wrong_gt_classes.csv", top_c2w_gt)
    write_csv(out_dir / "top_correct_to_wrong_absorbers.csv", top_c2w_top1)

    summary = {
        "status": "PASS",
        "audit_dir": str(audit_dir),
        "best_micro": best_micro,
        "best_macro": best_macro,
        "best_mean_normalized_gt_rank": best_rank,
        "min_total_wrong": min_wrong,
        "first_drift_onset_candidate": onset,
        "transition_counter": dict(transition_counter),
        "correct_to_wrong_pseudo_bad_proxy": c2w_pseudo_bad,
        "outputs": {
            "top_hub_peaks": str(out_dir / "top_hub_peaks.csv"),
            "top_correct_to_wrong_gt_classes": str(out_dir / "top_correct_to_wrong_gt_classes.csv"),
            "top_correct_to_wrong_absorbers": str(out_dir / "top_correct_to_wrong_absorbers.csv"),
        },
    }
    write_json(out_dir / "drift_row_transition_compact_summary.json", summary)

    md = []
    md.append("# A8 CE Drift Row Transition Compact Summary")
    md.append("")
    if onset:
        md.append(f"- first_drift_onset_candidate: epoch {onset['epoch']} vs {onset['prev_epoch']} ")
    else:
        md.append("- first_drift_onset_candidate: none detected by local regression rule")
    for label, row in [
        ("best_micro", best_micro),
        ("best_macro", best_macro),
        ("best_mean_normalized_gt_rank", best_rank),
        ("min_total_wrong", min_wrong),
    ]:
        if row:
            md.append(
                f"- {label}: epoch={row.get('epoch')}, micro={row.get('micro_top1')}, "
                f"macro={row.get('macro_rank1')}, mean_norm_rank={row.get('mean_normalized_gt_rank')}, "
                f"total_wrong={row.get('total_wrong_rows')}, large_i01={row.get('large_iter0_plus_iter1')}, middle_i01={row.get('middle_iter0_plus_iter1')}"
            )
    md.append("")
    md.append("## Transition counts")
    for k, v in transition_counter.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append(f"- correct_to_wrong_pseudo_bad_proxy: {c2w_pseudo_bad}")
    (out_dir / "DRIFT_ROW_TRANSITION_COMPACT_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("WROTE", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
