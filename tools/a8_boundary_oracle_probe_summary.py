#!/usr/bin/env python3
"""Summarize A8 Boundary-Oracle Probe runs against CE-5ep baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_eval(summary: Dict[str, Any]) -> Dict[str, Any]:
    if "eval_summary" in summary:
        return summary.get("eval_summary", {})
    if "eval_after" in summary:
        return summary.get("eval_after", {})
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_validation_json", required=True)
    ap.add_argument("--baseline_join_summary", required=True)
    ap.add_argument("--run", action="append", default=[], help="name:validation_json:join_summary")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_val = load_json(Path(args.baseline_validation_json).resolve())
    base_eval = pick_eval(base_val)
    base_join = load_json(Path(args.baseline_join_summary).resolve())

    rows: List[Dict[str, Any]] = []
    base_row = {
        "name": "CE5_baseline",
        "micro_top1": base_eval.get("micro_top1"),
        "macro_rank1": base_eval.get("macro_rank1"),
        "mean_normalized_gt_rank": base_eval.get("mean_normalized_gt_rank"),
        "total_wrong_rows": base_join.get("total_wrong_rows"),
        "large_iter0_plus_iter1": base_join.get("large_iter0_plus_iter1"),
        "middle_iter0_plus_iter1": base_join.get("middle_iter0_plus_iter1"),
        "small_iter0_plus_iter1": base_join.get("small_iter0_plus_iter1"),
        "iter0_plus_iter1_rate": base_join.get("iter0_plus_iter1_rate"),
    }
    rows.append(base_row)

    comparisons = []
    for spec in args.run:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise RuntimeError(f"--run must be name:validation_json:join_summary, got {spec!r}")
        name, val_p, join_p = parts
        val = load_json(Path(val_p).resolve())
        ev = pick_eval(val)
        join = load_json(Path(join_p).resolve())
        row = {
            "name": name,
            "micro_top1": ev.get("micro_top1"),
            "macro_rank1": ev.get("macro_rank1"),
            "mean_normalized_gt_rank": ev.get("mean_normalized_gt_rank"),
            "total_wrong_rows": join.get("total_wrong_rows"),
            "large_iter0_plus_iter1": join.get("large_iter0_plus_iter1"),
            "middle_iter0_plus_iter1": join.get("middle_iter0_plus_iter1"),
            "small_iter0_plus_iter1": join.get("small_iter0_plus_iter1"),
            "iter0_plus_iter1_rate": join.get("iter0_plus_iter1_rate"),
        }
        rows.append(row)
        comp = {"name": name}
        for k, v in row.items():
            if k == "name":
                continue
            try:
                comp[f"delta_{k}"] = float(v) - float(base_row.get(k))
            except Exception:
                comp[f"delta_{k}"] = None
        comparisons.append(comp)

    payload = {
        "status": "PASS",
        "baseline": base_row,
        "runs": rows[1:],
        "comparisons_vs_baseline": comparisons,
        "interpretation_guide": {
            "probe_success": "large/middle iter0+1 drop materially without catastrophic total wrong increase",
            "probe_failure": "oracle boundary training cannot reduce large/middle iter0+1, suggesting representation/prototype limits",
            "note": "This is an oracle diagnostic using train-side GT; it is not a clean method result.",
        },
    }
    (out_dir / "boundary_oracle_probe_comparison_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = ["# A8 Boundary-Oracle Probe Comparison", ""]
    md.append("| run | micro_top1 | macro_rank1 | mean_norm_rank | total_wrong | large_i01 | middle_i01 | small_i01 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r['name']} | {r.get('micro_top1')} | {r.get('macro_rank1')} | {r.get('mean_normalized_gt_rank')} | "
            f"{r.get('total_wrong_rows')} | {r.get('large_iter0_plus_iter1')} | {r.get('middle_iter0_plus_iter1')} | {r.get('small_iter0_plus_iter1')} |"
        )
    md.append("")
    md.append("This is an oracle diagnostic, not a clean method result.")
    (out_dir / "BOUNDARY_ORACLE_PROBE_COMPARISON.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
