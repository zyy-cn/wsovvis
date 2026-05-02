#!/usr/bin/env python3
"""Summarize GT-clean oracle vs weak full-Y overfit arms."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def _summary_path(root: Path) -> Path:
    p = root / "train" / "pipeline_train_summary.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing pipeline summary: {p}")
    return p


def _oracle_row(setting: str, root: Path) -> Dict[str, Any]:
    obj = _read_json(_summary_path(root))
    final_eval = dict(obj.get("final_eval", {}))
    initial_eval = dict(obj.get("initial_eval", {}))
    if not final_eval:
        stages = obj.get("stages", {}) if isinstance(obj.get("stages"), Mapping) else {}
        pre = stages.get("prealign", {}) if isinstance(stages.get("prealign"), Mapping) else {}
        final_eval = dict(pre.get("final_eval", {}))
        initial_eval = dict(pre.get("initial_eval", {}))
    return {
        "setting": setting,
        "source": str(root),
        "status": obj.get("status", ""),
        "protocol": obj.get("protocol", "oracle_supervised_gt_class"),
        "eval_scope": final_eval.get("candidate_scope", ""),
        "gt_count": final_eval.get("evaluated_gt_count", ""),
        "initial_top1": initial_eval.get("gt_top1_hit_rate", ""),
        "final_top1": final_eval.get("gt_top1_hit_rate", ""),
        "delta_top1": _delta(final_eval.get("gt_top1_hit_rate"), initial_eval.get("gt_top1_hit_rate")),
        "initial_rank": initial_eval.get("mean_normalized_gt_rank", ""),
        "final_rank": final_eval.get("mean_normalized_gt_rank", ""),
        "delta_rank": _delta(final_eval.get("mean_normalized_gt_rank"), initial_eval.get("mean_normalized_gt_rank")),
        "final_top5": final_eval.get("gt_top5_hit_rate", ""),
        "final_top10": final_eval.get("gt_top10_hit_rate", ""),
        "final_loss": final_eval.get("loss_mean", ""),
        "temperature": final_eval.get("temperature", ""),
        "global_step": obj.get("stages", {}).get("prealign", {}).get("global_step", "") if isinstance(obj.get("stages"), Mapping) else "",
    }


def _weak_rows(setting: str, root: Path) -> List[Dict[str, Any]]:
    obj = _read_json(_summary_path(root))
    initial_by = obj.get("initial_eval_by_scope", {}) if isinstance(obj.get("initial_eval_by_scope"), Mapping) else {}
    final_by = obj.get("final_eval_by_scope", {}) if isinstance(obj.get("final_eval_by_scope"), Mapping) else {}
    out: List[Dict[str, Any]] = []
    for scope, final_eval_obj in final_by.items():
        final_eval = dict(final_eval_obj) if isinstance(final_eval_obj, Mapping) else {}
        initial_eval = dict(initial_by.get(scope, {})) if isinstance(initial_by.get(scope, {}), Mapping) else {}
        out.append({
            "setting": setting,
            "source": str(root),
            "status": obj.get("status", ""),
            "protocol": obj.get("protocol", ""),
            "positive_scope": obj.get("stages", {}).get("prealign", {}).get("positive_scope", "") if isinstance(obj.get("stages"), Mapping) else "",
            "denominator_scope": obj.get("stages", {}).get("prealign", {}).get("denominator_scope", "") if isinstance(obj.get("stages"), Mapping) else "",
            "eval_scope": str(scope),
            "gt_count": final_eval.get("evaluated_gt_count", ""),
            "initial_top1": initial_eval.get("gt_top1_hit_rate", ""),
            "final_top1": final_eval.get("gt_top1_hit_rate", ""),
            "delta_top1": _delta(final_eval.get("gt_top1_hit_rate"), initial_eval.get("gt_top1_hit_rate")),
            "initial_rank": initial_eval.get("mean_normalized_gt_rank", ""),
            "final_rank": final_eval.get("mean_normalized_gt_rank", ""),
            "delta_rank": _delta(final_eval.get("mean_normalized_gt_rank"), initial_eval.get("mean_normalized_gt_rank")),
            "final_top5": final_eval.get("gt_top5_hit_rate", ""),
            "final_top10": final_eval.get("gt_top10_hit_rate", ""),
            "final_loss": final_eval.get("loss_mean", ""),
            "temperature": final_eval.get("temperature", ""),
            "global_step": obj.get("stages", {}).get("prealign", {}).get("global_step", "") if isinstance(obj.get("stages"), Mapping) else "",
        })
    return out


def _delta(a: Any, b: Any) -> Any:
    try:
        return float(a) - float(b)
    except Exception:
        return ""


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _find(rows: Sequence[Mapping[str, Any]], setting: str, scope: str) -> Dict[str, Any]:
    for r in rows:
        if str(r.get("setting")) == str(setting) and str(r.get("eval_scope")) == str(scope):
            return dict(r)
    return {}


def _diagnose(rows: Sequence[Mapping[str, Any]]) -> str:
    oracle_base = _find(rows, "oracle_base_vocab", "base_vocab") or _find(rows, "oracle_supervised_gt_class", "base_vocab")
    weak_nohub_base = _find(rows, "weak_fullY_nohub", "base_vocab")
    weak_nohub_clip = _find(rows, "weak_fullY_nohub", "clip_y_base")
    oracle_top1 = _as_float(oracle_base.get("final_top1"))
    weak_base_top1 = _as_float(weak_nohub_base.get("final_top1"))
    weak_clip_top1 = _as_float(weak_nohub_clip.get("final_top1"))
    if oracle_top1 >= 0.70 and weak_base_top1 < 0.50:
        return "ORACLE_STRONG__WEAK_FULLY_ASSIGNMENT_FAILS_TO_RELEASE_BASE_CAPACITY"
    if oracle_top1 >= 0.70 and weak_base_top1 >= 0.50 and weak_clip_top1 >= 0.70:
        return "ORACLE_AND_GT_WEAK_STRONG__PROPOSAL_OR_VAL_TRANSFER_NEXT"
    if oracle_top1 < 0.70:
        return "ORACLE_WEAK__CAPACITY_BOTTLENECK"
    return "MIXED__INSPECT_WEAK_OBJECTIVE_AND_ASSIGNMENT_CONCENTRATION"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize GT-clean weak full-Y overfit capacity audit.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--oracle_base_vocab_root", default="")
    p.add_argument("--oracle_clip_y_root", default="")
    p.add_argument("--weak_baseline_root", default="")
    p.add_argument("--weak_nohub_root", default="")
    p.add_argument("--extra_row", action="append", default=[], help="name=/path/to/root; weak roots emit all eval scopes")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if args.oracle_base_vocab_root:
        rows.append(_oracle_row("oracle_base_vocab", Path(args.oracle_base_vocab_root).expanduser()))
    if args.oracle_clip_y_root:
        rows.append(_oracle_row("oracle_clip_y", Path(args.oracle_clip_y_root).expanduser()))
    if args.weak_baseline_root:
        rows.extend(_weak_rows("weak_fullY_baseline", Path(args.weak_baseline_root).expanduser()))
    if args.weak_nohub_root:
        rows.extend(_weak_rows("weak_fullY_nohub", Path(args.weak_nohub_root).expanduser()))
    for item in args.extra_row or []:
        if "=" not in str(item):
            raise ValueError(f"--extra_row must be name=/path, got: {item}")
        name, path = str(item).split("=", 1)
        root = Path(path).expanduser()
        obj = _read_json(_summary_path(root))
        if "final_eval_by_scope" in obj:
            rows.extend(_weak_rows(name, root))
        else:
            rows.append(_oracle_row(name, root))

    diagnosis = _diagnose(rows)
    _write_csv(out / "gt_clean_weak_fully_overfit_summary.csv", rows)
    summary = {
        "status": "PASS",
        "output_dir": str(out),
        "row_count": len(rows),
        "diagnosis": diagnosis,
        "rows": rows,
    }
    _write_json(out / "summary.json", summary)
    md = [
        "# GT-clean Weak Full-Y Overfit Audit",
        "",
        "Status: `PASS`",
        f"Diagnosis: `{diagnosis}`",
        "",
        "Core interpretation:",
        "- Oracle strong + weak weak means the weak assignment objective is the bottleneck.",
        "- Oracle strong + weak strong but VC/val weak means proposal or inference transfer is next.",
        "- This audit uses GT labels only for clean denominator/evaluation; weak training does not use instance GT targets.",
        "",
        "Core outputs:",
        "- summary.json",
        "- gt_clean_weak_fully_overfit_summary.csv",
        "",
    ]
    (out / "GT_CLEAN_WEAK_FULLY_OVERFIT_TAKEOVER.md").write_text("\n".join(md), encoding="utf-8")
    print(str(out / "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
