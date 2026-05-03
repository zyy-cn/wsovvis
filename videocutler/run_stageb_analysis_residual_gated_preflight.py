#!/usr/bin/env python3
"""Residual-gated assignment Phase-0 preflight reducer.

Read-only, standard-library only, CPU-light.

Purpose
-------
Validate that the current experiment assets are sufficient before any training:
1) iterative residual label identifiability table;
2) label-resolvable vs GT-carrier scorer table;
3) assignment oracle gap audit table.

It emits compact JSON/CSV/MD artifacts and a gate verdict. It does not train,
load checkpoints, run GPU kernels, or modify existing artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TRUE_SET = {"1", "true", "yes", "y", "pass"}
RAW_ID_KEYS = ["raw_id", "raw_category_id", "category_id", "class_raw_id", "gt_raw_id"]
RANK1_KEYS = ["D_gt_rank1_rate", "gt_rank1_rate", "gt_top1_hit_rate", "rank1_rate", "top1_rate"]
TOP5_KEYS = ["D_gt_top5_rate", "gt_top5_rate", "gt_top5_hit_rate", "top5_rate"]
SUPPORT_KEYS = ["D_support_weight", "gt_count", "candidate_contains_gt_count", "row_count", "count", "instance_count"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def inum(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(round(float(x)))
    except Exception:
        return default


def truth(x: Any) -> bool:
    return str(x).strip().lower() in TRUE_SET


def raw_id(row: Dict[str, Any]) -> str:
    for k in RAW_ID_KEYS:
        v = row.get(k, "")
        if str(v).strip() != "":
            try:
                return str(int(float(str(v).strip())))
            except Exception:
                return str(v).strip()
    return ""


def first_col(row: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return str(row[k])
    return default


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def median(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def candidate_paths(run_root: Path, dataset: str) -> Dict[str, List[Path]]:
    # Keep this list deterministic and shallow: no full recursive scans by default.
    g8_root = run_root
    for p in run_root.parents:
        if p.name == "G8_inference_and_eval":
            g8_root = p
            break
    return {
        "label_csv": [
            run_root / "analysis" / "iterative_residual_label_identifiability" / dataset / "per_class_iterative_residual_identifiability.csv",
            g8_root / "sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427" / "analysis" / "iterative_residual_label_identifiability" / dataset / "per_class_iterative_residual_identifiability.csv",
        ],
        "scorer_join_csv": [
            run_root / "analysis" / "label_resolvable_vs_gtcarrier_scorer" / dataset / "label_resolvable_vs_gtcarrier_scorer_by_class.csv",
            g8_root / "sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427" / "analysis" / "label_resolvable_vs_gtcarrier_scorer" / dataset / "label_resolvable_vs_gtcarrier_scorer_by_class.csv",
        ],
        "oracle_gap_csv": [
            run_root / "analysis" / "assignment_oracle_gap_audit" / dataset / "base_vocab" / "per_class_oracle_gap.csv",
            g8_root / "analysis" / "assignment_oracle_gap_audit" / dataset / "base_vocab" / "per_class_oracle_gap.csv",
        ],
        "oracle_run_summary_csv": [
            run_root / "analysis" / "assignment_oracle_gap_audit" / dataset / "base_vocab" / "run_summary.csv",
            g8_root / "analysis" / "assignment_oracle_gap_audit" / dataset / "base_vocab" / "run_summary.csv",
        ],
    }


def resolve_path(explicit: str, candidates: Sequence[Path], name: str, optional: bool = False) -> Tuple[Optional[Path], Dict[str, Any]]:
    tried: List[str] = []
    if explicit:
        p = Path(explicit)
        tried.append(str(p))
        if p.is_file():
            return p, {"name": name, "status": "FOUND", "path": str(p), "tried": tried}
        return None, {"name": name, "status": "MISSING", "path": str(p), "tried": tried, "optional": optional}
    for p in candidates:
        tried.append(str(p))
        if p.is_file():
            return p, {"name": name, "status": "FOUND", "path": str(p), "tried": tried}
    return None, {"name": name, "status": "MISSING", "tried": tried, "optional": optional}


def filter_label_variant(rows: List[Dict[str, str]], variant: str) -> List[Dict[str, str]]:
    vals = {r.get("variant", "") for r in rows if r.get("variant", "")}
    if vals and variant in vals:
        rows = [r for r in rows if r.get("variant", "") == variant]
    elif vals and variant not in vals:
        # Use all rows only when no variant column exists; otherwise this is unsafe.
        return []
    out = []
    seen = set()
    for r in rows:
        rid = raw_id(r)
        if not rid or rid in seen:
            continue
        split = r.get("split_type", r.get("target_split", "base"))
        if split and split != "base":
            continue
        seen.add(rid)
        out.append(r)
    return out


def filter_scorer_variant(rows: List[Dict[str, str]], variant: str) -> List[Dict[str, str]]:
    """Keep the requested variant if present, then deduplicate one row per raw id.

    Older scorer-join tables may contain both strict/person-aware variants, which
    doubles class counts (e.g. 1006 resolved rows instead of 503).  If no variant
    column exists, this still deduplicates by raw id for backwards compatibility.
    """
    vals = {r.get("variant", "") for r in rows if r.get("variant", "")}
    if vals and variant in vals:
        rows = [r for r in rows if r.get("variant", "") == variant]
    elif vals and variant not in vals:
        return []
    out = []
    seen = set()
    for r in rows:
        rid = raw_id(r)
        if not rid or rid in seen:
            continue
        split = r.get("split_type", r.get("target_split", r.get("split", "base")))
        if split and split != "base":
            continue
        seen.add(rid)
        out.append(r)
    return out


def summarize_label(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    clip_col = "clip_count" if rows and "clip_count" in rows[0] else "base_clip_count"
    resolved_col = "resolved" if rows and "resolved" in rows[0] else "is_resolved"
    cert_col = "certificate_type" if rows and "certificate_type" in rows[0] else "certificate_type_norm"
    total = len(rows)
    absent = [r for r in rows if inum(r.get(clip_col)) <= 0]
    observed = [r for r in rows if inum(r.get(clip_col)) > 0]
    resolved = [r for r in observed if truth(r.get(resolved_col, r.get("is_identifiable", "")))]
    unresolved = [r for r in observed if r not in resolved]
    by_cert = Counter(r.get(cert_col, "unknown") or "unknown" for r in resolved)
    clip_break = Counter(inum(r.get(clip_col)) for r in rows)
    return {
        "total_base_rows": total,
        "train_absent_count": len(absent),
        "train_observed_count": len(observed),
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "resolved_rate_among_train_observed": len(resolved) / len(observed) if observed else None,
        "certificate_counts_resolved": dict(by_cert),
        "clip_count_breakdown": dict(sorted(clip_break.items())),
        "columns_used": {"clip_count": clip_col, "resolved": resolved_col, "certificate": cert_col},
    }


def summarize_scorer(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"rows": 0}
    rank1_col = next((k for k in RANK1_KEYS if k in rows[0]), None)
    top5_col = next((k for k in TOP5_KEYS if k in rows[0]), None)
    support_col = next((k for k in SUPPORT_KEYS if k in rows[0]), None)
    resolved_col = "resolved_bool" if "resolved_bool" in rows[0] else "resolved"
    vals = []
    supports = []
    resolved_rows = []
    for r in rows:
        if resolved_col in r and not truth(r.get(resolved_col)):
            continue
        if rank1_col is None:
            continue
        vals.append(fnum(r.get(rank1_col)))
        supports.append(max(0.0, fnum(r.get(support_col), 1.0)) if support_col else 1.0)
        resolved_rows.append(r)
    total_w = sum(supports)
    out: Dict[str, Any] = {
        "rows": len(rows),
        "resolved_scored_rows": len(vals),
        "rank1_col": rank1_col,
        "top5_col": top5_col,
        "support_col": support_col,
    }
    if vals:
        out.update({
            "macro_mean_rank1": mean(vals),
            "macro_median_rank1": median(vals),
            "rank1_ge_0.5_count": sum(v >= 0.5 for v in vals),
            "rank1_lt_0.1_count": sum(v < 0.1 for v in vals),
            "rank1_ge_0.5_rate": sum(v >= 0.5 for v in vals) / len(vals),
            "rank1_lt_0.1_rate": sum(v < 0.1 for v in vals) / len(vals),
        })
        if total_w > 0:
            out["micro_weighted_rank1"] = sum(v * w for v, w in zip(vals, supports)) / total_w
    return out


def summarize_oracle_gap(rows: List[Dict[str, str]], run_summary_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_class_rows": len(rows)}
    if rows:
        for col in ["oracle_top1_rate", "weak_base_top1_rate", "weak_nohub_top1_rate", "weak_base_other_positive_confusion_rate", "weak_nohub_other_positive_confusion_rate"]:
            vals = [fnum(r.get(col), default=float("nan")) for r in rows if r.get(col, "") != ""]
            vals = [v for v in vals if not math.isnan(v)]
            if vals:
                out[f"macro_mean_{col}"] = mean(vals)
                out[f"macro_median_{col}"] = median(vals)
        sentinel = {}
        for name in ["shoe", "baseball_cap", "trousers", "watch", "rearview_mirror"]:
            hit = next((r for r in rows if r.get("class_name") == name), None)
            if hit:
                sentinel[name] = {k: hit.get(k, "") for k in ["raw_id", "gt_count", "oracle_top1_rate", "weak_base_top1_rate", "weak_nohub_top1_rate", "nohub_rescued_rate", "weak_base_other_positive_confusion_rate", "weak_nohub_other_positive_confusion_rate"]}
        out["sentinel_classes"] = sentinel
    if run_summary_rows:
        out["run_summary_rows"] = run_summary_rows[:10]
    return out


def approx_check(name: str, got: Optional[float], expected: Optional[float], tol: float, hard: bool = True) -> Dict[str, Any]:
    if expected is None:
        return {"name": name, "status": "SKIP", "got": got}
    ok = got is not None and abs(float(got) - float(expected)) <= tol
    return {"name": name, "status": "PASS" if ok else ("FAIL" if hard else "WARN"), "got": got, "expected": expected, "tol": tol, "hard": hard}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--variant", default="person_aware")
    ap.add_argument("--label_csv", default="")
    ap.add_argument("--scorer_join_csv", default="")
    ap.add_argument("--oracle_gap_csv", default="")
    ap.add_argument("--oracle_run_summary_csv", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--relaxed", action="store_true", help="Do not hard-fail expected-number mismatch; still report it.")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "analysis" / "residual_gated_phase01_preflight" / args.dataset_name
    cands = candidate_paths(run_root, args.dataset_name)
    label_path, inv_label = resolve_path(args.label_csv, cands["label_csv"], "label_csv")
    scorer_path, inv_scorer = resolve_path(args.scorer_join_csv, cands["scorer_join_csv"], "scorer_join_csv")
    oracle_path, inv_oracle = resolve_path(args.oracle_gap_csv, cands["oracle_gap_csv"], "oracle_gap_csv", optional=True)
    run_summary_path, inv_run = resolve_path(args.oracle_run_summary_csv, cands["oracle_run_summary_csv"], "oracle_run_summary_csv", optional=True)

    inventory = {
        "status": "PASS",
        "run_root": str(run_root),
        "dataset_name": args.dataset_name,
        "variant": args.variant,
        "artifacts": [inv_label, inv_scorer, inv_oracle, inv_run],
    }
    hard_missing = [x for x in inventory["artifacts"] if x["status"] != "FOUND" and not x.get("optional")]
    if hard_missing:
        inventory["status"] = "FAIL_MISSING_REQUIRED_ARTIFACT"

    label_summary: Dict[str, Any] = {}
    scorer_summary: Dict[str, Any] = {}
    oracle_summary: Dict[str, Any] = {}
    gate_checks: List[Dict[str, Any]] = []

    if label_path:
        label_rows = filter_label_variant(read_csv(label_path), args.variant)
        label_summary = summarize_label(label_rows)
        write_csv(out_dir / "label_rows_filtered.csv", label_rows)
        gate_checks.extend([
            approx_check("train_observed_base_count", label_summary.get("train_observed_count"), 525, 0.5, not args.relaxed),
            approx_check("label_resolved_count", label_summary.get("resolved_count"), 503, 0.5, not args.relaxed),
            approx_check("label_unresolved_count", label_summary.get("unresolved_count"), 22, 0.5, not args.relaxed),
        ])
    if scorer_path:
        scorer_rows = filter_scorer_variant(read_csv(scorer_path), args.variant)
        scorer_summary = summarize_scorer(scorer_rows)
        gate_checks.extend([
            approx_check("rank1_ge_0.5_count", scorer_summary.get("rank1_ge_0.5_count"), 176, 1.5, not args.relaxed),
            approx_check("rank1_lt_0.1_count", scorer_summary.get("rank1_lt_0.1_count"), 243, 1.5, not args.relaxed),
        ])
    if oracle_path:
        run_rows = read_csv(run_summary_path) if run_summary_path else []
        oracle_summary = summarize_oracle_gap(read_csv(oracle_path), run_rows)
        # These are macro class summaries; row-level old claims are in run_summary. Do not hard-fail on macro mismatch.

    hard_fails = [c for c in gate_checks if c["status"] == "FAIL" and c.get("hard")]
    final_status = inventory["status"]
    if final_status == "PASS" and hard_fails:
        final_status = "FAIL_EXPECTED_NUMBER_MISMATCH"

    summary = {
        "status": final_status,
        "inventory_status": inventory["status"],
        "label_summary": label_summary,
        "scorer_summary": scorer_summary,
        "oracle_gap_summary": oracle_summary,
        "gate_checks": gate_checks,
        "interpretation": {
            "purpose": "Preflight-only: verify existing assets and known baseline numbers before residual-gated pseudo-pool construction.",
            "does_not_train": True,
            "gpu_required": False,
            "rowwise_large_matrix_compute": False,
        },
    }
    write_json(out_dir / "asset_inventory.json", inventory)
    write_json(out_dir / "baseline_numbers.json", summary)

    rows = []
    for k, v in label_summary.items():
        if isinstance(v, (int, float, str)) or v is None:
            rows.append({"section": "label", "metric": k, "value": v})
    for k, v in scorer_summary.items():
        if isinstance(v, (int, float, str)) or v is None:
            rows.append({"section": "scorer", "metric": k, "value": v})
    write_csv(out_dir / "class_bucket_summary.csv", rows)

    md = [
        "# Residual-Gated Phase-01 Preflight",
        "",
        f"- status: {final_status}",
        f"- run_root: `{run_root}`",
        f"- dataset: `{args.dataset_name}`",
        f"- variant: `{args.variant}`",
        "",
        "## Key numbers",
        f"- train_observed_base: {label_summary.get('train_observed_count')}",
        f"- label_resolved: {label_summary.get('resolved_count')}",
        f"- label_unresolved: {label_summary.get('unresolved_count')}",
        f"- scorer rank1>=0.5: {scorer_summary.get('rank1_ge_0.5_count')}",
        f"- scorer rank1<0.1: {scorer_summary.get('rank1_lt_0.1_count')}",
        "",
        "## Gate checks",
    ]
    for c in gate_checks:
        md.append(f"- {c['name']}: {c['status']} got={c.get('got')} expected={c.get('expected')} tol={c.get('tol')}")
    md.extend(["", "## Sentinel classes", "```json", json.dumps(oracle_summary.get("sentinel_classes", {}), ensure_ascii=False, indent=2), "```"])
    (out_dir / "RESIDUAL_GATED_PHASE01_PREFLIGHT_TAKEOVER.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if final_status.startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
