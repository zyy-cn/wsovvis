#!/usr/bin/env python3
"""Residual-gated training result audit reducer.

Read-only post-training comparator. It joins:
- class_training_policy.csv from Phase-1 pseudo-pool planner;
- a new by-class rank/scorer CSV produced by a future GT-clean pilot.

It does not train. It is included now so manual execution has a fixed validation
gate once a training pilot writes by-class results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RAW_ID_KEYS = ["raw_id", "raw_category_id", "category_id", "class_raw_id", "gt_raw_id"]
BEFORE_KEYS = ["D_gt_rank1_rate", "before_gt_rank1_rate", "baseline_gt_rank1_rate", "gt_rank1_rate_before"]
AFTER_KEYS = ["after_gt_rank1_rate", "new_gt_rank1_rate", "gt_rank1_rate", "rank1_rate", "gt_top1_hit_rate"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "": return default
        v=float(x)
        if math.isnan(v) or math.isinf(v): return default
        return v
    except Exception:
        return default


def raw_id(row: Dict[str, Any]) -> str:
    for k in RAW_ID_KEYS:
        v=row.get(k, "")
        if str(v).strip():
            try: return str(int(float(str(v).strip())))
            except Exception: return str(v).strip()
    return ""


def pick(row: Dict[str, Any], keys: Sequence[str], default: str="") -> str:
    for k in keys:
        if k in row and str(row[k]).strip() != "": return str(row[k])
    return default


def by_raw(rows: List[Dict[str,str]]) -> Dict[str, Dict[str,str]]:
    out={}
    for r in rows:
        rid=raw_id(r)
        if rid and rid not in out: out[rid]=r
    return out


def bucket(delta: float, before: float, after: float) -> str:
    if after >= 0.5 and before < 0.5: return "crossed_rank1_ge_0.5"
    if before < 0.1 and after >= 0.1: return "recovered_from_collapse"
    if delta >= 0.10: return "improved_large"
    if delta >= 0.03: return "improved_small"
    if delta <= -0.03: return "degraded"
    return "flat"


def summarize(rows: List[Dict[str,Any]]) -> Dict[str,Any]:
    vals=[fnum(r.get("delta_rank1")) for r in rows]
    before=[fnum(r.get("before_rank1")) for r in rows]
    after=[fnum(r.get("after_rank1")) for r in rows]
    out={"count":len(rows)}
    if rows:
        out.update({
            "mean_delta_rank1_macro": sum(vals)/len(vals),
            "before_rank1_ge_0.5_count": sum(v>=0.5 for v in before),
            "after_rank1_ge_0.5_count": sum(v>=0.5 for v in after),
            "before_rank1_lt_0.1_count": sum(v<0.1 for v in before),
            "after_rank1_lt_0.1_count": sum(v<0.1 for v in after),
            "improved_count": sum(v>0.0 for v in vals),
            "degraded_count": sum(v<0.0 for v in vals),
            "bucket_counts": dict(Counter(r.get("delta_bucket","") for r in rows)),
        })
    return out


def fail_missing_input(out_dir: Path, status: str, missing_path: str, message: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"status": status, "missing_path": missing_path, "message": message}
    write_json(out_dir / "training_audit_summary.json", summary)
    (out_dir / "RESIDUAL_GATED_TRAINING_AUDIT_TAKEOVER.md").write_text(
        f"# Residual-Gated Training Audit\n\n- status: {status}\n- missing_path: `{missing_path}`\n- message: {message}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--policy_csv", required=True, help="class_training_policy.csv from pseudo-pool planner")
    ap.add_argument("--after_by_class_csv", required=True, help="Future training output by-class rank CSV")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_rank1_ge_0_5_gain", type=int, default=20)
    ap.add_argument("--min_collapse_reduction", type=int, default=20)
    args=ap.parse_args()
    out_dir=Path(args.out_dir)
    policy_path = Path(args.policy_csv)
    after_path = Path(args.after_by_class_csv)
    if not policy_path.is_file():
        fail_missing_input(out_dir, "FAIL_MISSING_POLICY_CSV", str(policy_path), "Run residual_gated_pseudo_pool first, or pass a valid --policy_csv.")
        raise SystemExit(2)
    if not after_path.is_file():
        fail_missing_input(out_dir, "FAIL_MISSING_AFTER_BY_CLASS_CSV", str(after_path), "This audit is only for after a future training run; pass the produced by-class rank summary CSV.")
        raise SystemExit(2)
    policy=read_csv(policy_path)
    after=read_csv(after_path)
    after_by=by_raw(after)
    joined=[]
    for p in policy:
        rid=raw_id(p)
        if not rid: continue
        a=after_by.get(rid)
        before=fnum(pick(p, BEFORE_KEYS, p.get("D_gt_rank1_rate", "0")))
        if not a:
            after_v=before
            has_after=False
        else:
            after_v=fnum(pick(a, AFTER_KEYS, ""), before)
            has_after=True
        delta=after_v-before
        joined.append({
            "raw_id": rid,
            "class_name": p.get("class_name", a.get("class_name", "") if a else ""),
            "policy": p.get("policy", ""),
            "certificate_type": p.get("certificate_type", ""),
            "risk_flags": p.get("risk_flags", ""),
            "before_rank1": before,
            "after_rank1": after_v,
            "delta_rank1": delta,
            "delta_bucket": bucket(delta, before, after_v),
            "has_after_result": has_after,
        })
    write_csv(out_dir/"residual_gated_training_delta_by_class.csv", joined)
    by_policy={k:summarize(v) for k,v in defaultdict(list, {}).items()}
    tmp=defaultdict(list)
    for r in joined: tmp[r.get("policy","")].append(r)
    by_policy={k:summarize(v) for k,v in sorted(tmp.items())}
    total=summarize(joined)
    gain=(total.get("after_rank1_ge_0.5_count",0)-total.get("before_rank1_ge_0.5_count",0))
    collapse_reduction=(total.get("before_rank1_lt_0.1_count",0)-total.get("after_rank1_lt_0.1_count",0))
    gate_checks=[
        {"name":"rank1_ge_0.5_gain", "status":"PASS" if gain>=args.min_rank1_ge_0_5_gain else "WARN", "gain":gain, "threshold":args.min_rank1_ge_0_5_gain},
        {"name":"collapse_reduction", "status":"PASS" if collapse_reduction>=args.min_collapse_reduction else "WARN", "gain":collapse_reduction, "threshold":args.min_collapse_reduction},
    ]
    summary={"status":"PASS_WITH_WARNINGS" if any(c['status']=='WARN' for c in gate_checks) else "PASS", "total":total, "by_policy":by_policy, "gate_checks":gate_checks}
    write_json(out_dir/"training_audit_summary.json", summary)
    md=["# Residual-Gated Training Audit", "", f"- status: {summary['status']}", f"- rank1>=0.5 gain: {gain}", f"- collapse reduction: {collapse_reduction}", "", "## By policy", "```json", json.dumps(by_policy, ensure_ascii=False, indent=2), "```"]
    (out_dir/"RESIDUAL_GATED_TRAINING_AUDIT_TAKEOVER.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
