#!/usr/bin/env python3
"""Summarize GT-clean base overfit capacity results.

This is a lightweight report builder. It does not train and does not read large
prediction files. It aggregates:
  * oracle supervised overfit stage_summary/pipeline summary;
  * optional weak baseline/nohub final attribution compare CSVs;
  * optional per-arm train summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _pick(obj: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for p in path:
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _oracle_row(name: str, root: Path) -> Dict[str, Any]:
    pipe = _read_json(root / "train" / "pipeline_train_summary.json")
    stage = _read_json(root / "train" / "prealign" / "stage_summary.json")
    initial = pipe.get("initial_eval") or stage.get("initial_eval") or {}
    final = pipe.get("final_eval") or stage.get("final_eval") or {}
    return {
        "setting": name,
        "source": str(root),
        "status": pipe.get("status", "MISSING"),
        "protocol": pipe.get("protocol", stage.get("protocol", "")),
        "candidate_scope": stage.get("eval_candidate_scope", ""),
        "gt_count": final.get("evaluated_gt_count", ""),
        "initial_top1": initial.get("gt_top1_hit_rate", ""),
        "final_top1": final.get("gt_top1_hit_rate", ""),
        "delta_top1": _as_float(final.get("gt_top1_hit_rate")) - _as_float(initial.get("gt_top1_hit_rate")),
        "initial_rank": initial.get("mean_normalized_gt_rank", ""),
        "final_rank": final.get("mean_normalized_gt_rank", ""),
        "delta_rank": _as_float(final.get("mean_normalized_gt_rank")) - _as_float(initial.get("mean_normalized_gt_rank")),
        "final_top5": final.get("gt_top5_hit_rate", ""),
        "final_top10": final.get("gt_top10_hit_rate", ""),
        "final_loss": final.get("loss_mean", ""),
        "temperature": final.get("temperature", ""),
        "global_step": pipe.get("stages", {}).get("prealign", {}).get("global_step", stage.get("global_step", "")),
    }


def _weak_rows_from_compare(compare_dir: Path) -> List[Dict[str, Any]]:
    rows = _read_csv(compare_dir / "summary_by_run.csv")
    out: List[Dict[str, Any]] = []
    for r in rows:
        if str(r.get("group_name", "overall")) not in {"overall", ""}:
            continue
        out.append({
            "setting": r.get("checkpoint", r.get("run", "")),
            "source": str(compare_dir),
            "status": "PASS_FROM_COMPARE" if rows else "MISSING",
            "protocol": "weak_full_y",
            "candidate_scope": "compare_defined",
            "gt_count": r.get("gt_count", ""),
            "initial_top1": "",
            "final_top1": r.get("gt_top1_hit_rate", ""),
            "delta_top1": "",
            "initial_rank": "",
            "final_rank": r.get("mean_normalized_gt_rank", ""),
            "delta_rank": "",
            "final_top5": r.get("gt_top5_hit_rate", ""),
            "final_top10": r.get("gt_top10_hit_rate", ""),
            "final_loss": "",
            "temperature": "",
            "global_step": "",
        })
    return out


def _diagnosis(rows: Sequence[Mapping[str, Any]]) -> str:
    oracle = None
    for r in rows:
        if "oracle" in str(r.get("protocol", "")) or "oracle" in str(r.get("setting", "")).lower():
            oracle = r
            break
    if oracle is None:
        return "NO_ORACLE_ROW__CANNOT_JUDGE_CAPACITY"
    top1 = _as_float(oracle.get("final_top1"))
    rank = _as_float(oracle.get("final_rank"), 1.0)
    if top1 >= 0.70 or rank <= 0.10:
        return "ORACLE_BASE_OVERFIT_STRONG__WEAK_ASSIGNMENT_OR_PROPOSAL_NEXT"
    if top1 >= 0.30 or rank <= 0.30:
        return "ORACLE_BASE_OVERFIT_PARTIAL__CAPACITY_PRESENT_BUT_WEAK"
    return "ORACLE_BASE_OVERFIT_WEAK__PROJECTOR_TEXT_FEATURE_OBJECTIVE_BOTTLENECK"


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize GT-clean base overfit capacity audit.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--oracle_root", default="")
    p.add_argument("--weak_compare_dir", default="")
    p.add_argument("--extra_row", action="append", default=[], help="Optional name=path to another pipeline root.")
    args = p.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    rows: List[Dict[str, Any]] = []
    if str(args.oracle_root).strip():
        rows.append(_oracle_row("oracle_supervised_gt_class", Path(args.oracle_root).expanduser().resolve()))
    for item in args.extra_row or []:
        if "=" in str(item):
            name, path = str(item).split("=", 1)
        else:
            path = str(item)
            name = Path(path).name
        rows.append(_oracle_row(name, Path(path).expanduser().resolve()))
    if str(args.weak_compare_dir).strip():
        rows.extend(_weak_rows_from_compare(Path(args.weak_compare_dir).expanduser().resolve()))

    summary = {
        "status": "PASS" if rows else "NO_ROWS",
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "diagnosis": _diagnosis(rows),
        "rows": rows,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "base_overfit_capacity_summary.csv", rows)
    md_lines = [
        "# GT-clean Base Overfit Capacity Audit",
        "",
        f"Status: `{summary['status']}`",
        f"Diagnosis: `{summary['diagnosis']}`",
        "",
        "Core interpretation:",
        "- If oracle supervised GT-class overfit is weak, fix projector/text/feature/objective before novel transfer.",
        "- If oracle is strong but weak full-Y is weak, the weak assignment objective is the bottleneck.",
        "- If GT weak is strong but VC/val is weak, proposal/inference noise is the bottleneck.",
        "",
        "Core outputs:",
        "- summary.json",
        "- base_overfit_capacity_summary.csv",
    ]
    (output_dir / "GT_CLEAN_BASE_OVERFIT_TAKEOVER.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
