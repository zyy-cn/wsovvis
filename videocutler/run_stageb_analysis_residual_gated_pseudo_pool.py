#!/usr/bin/env python3
"""Residual-gated Phase-1 pseudo-pool planner.

Read-only planner, not a trainer.

It merges three existing evidence tables:
1) iterative residual label identifiability;
2) label-resolvable vs GT-carrier scorer;
3) assignment oracle gap per-class / optional row examples.

It outputs a conservative class-level training policy:
- strong_hard_ce_candidate
- weak_soft_ce_candidate
- prototype_calibration_candidate
- deferred_unresolved_or_unsafe
- train_absent

It also emits optional seed row examples if nohub top-row CSVs are available.
The script intentionally does not create checkpoint weights and does not modify training code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

TRUE_SET = {"1", "true", "yes", "y", "pass"}
RAW_ID_KEYS = ["raw_id", "raw_category_id", "category_id", "class_raw_id", "gt_raw_id"]
RANK1_KEYS = ["D_gt_rank1_rate", "gt_rank1_rate", "gt_top1_hit_rate", "rank1_rate", "top1_rate"]
TOP5_KEYS = ["D_gt_top5_rate", "gt_top5_rate", "gt_top5_hit_rate", "top5_rate"]
SUPPORT_KEYS = ["D_support_weight", "gt_count", "candidate_contains_gt_count", "row_count", "count", "instance_count"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


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


def pick(row: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        if k in row and str(row[k]).strip() != "":
            return str(row[k])
    return default


def candidate_paths(run_root: Path, dataset: str) -> Dict[str, List[Path]]:
    g8_root = run_root
    for p in run_root.parents:
        if p.name == "G8_inference_and_eval":
            g8_root = p
            break
    base_gap = g8_root / "analysis" / "assignment_oracle_gap_audit" / dataset / "base_vocab"
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
            base_gap / "per_class_oracle_gap.csv",
        ],
        "row_seed_csvs": [
            base_gap / "nohub_rescued_rows_top.csv",
            base_gap / "nohub_rank_improved_rows_top.csv",
            base_gap / "nohub_broken_rows_top.csv",
            base_gap / "nohub_rank_degraded_rows_top.csv",
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


def filter_label(rows: List[Dict[str, str]], variant: str) -> List[Dict[str, str]]:
    vals = {r.get("variant", "") for r in rows if r.get("variant", "")}
    if vals:
        if variant not in vals:
            return []
        rows = [r for r in rows if r.get("variant", "") == variant]
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


def by_raw(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out = {}
    for r in rows:
        rid = raw_id(r)
        if rid and rid not in out:
            out[rid] = r
    return out


def get_rank1(row: Optional[Dict[str, str]]) -> Optional[float]:
    if not row:
        return None
    for k in RANK1_KEYS:
        if k in row and row[k] != "":
            return fnum(row[k])
    return None


def get_top5(row: Optional[Dict[str, str]]) -> Optional[float]:
    if not row:
        return None
    for k in TOP5_KEYS:
        if k in row and row[k] != "":
            return fnum(row[k])
    return None


def support_weight(row: Optional[Dict[str, str]]) -> int:
    if not row:
        return 0
    for k in SUPPORT_KEYS:
        if k in row and row[k] != "":
            return max(0, inum(row[k]))
    return 0


def label_fields(row: Dict[str, str]) -> Dict[str, Any]:
    clip_col = "clip_count" if "clip_count" in row else "base_clip_count"
    resolved_col = "resolved" if "resolved" in row else "is_resolved"
    cert_col = "certificate_type" if "certificate_type" in row else "certificate_type_norm"
    return {
        "clip_count": inum(row.get(clip_col)),
        "instance_count": inum(row.get("instance_count")),
        "resolved": truth(row.get(resolved_col, row.get("is_identifiable", ""))),
        "certificate_type": row.get(cert_col, "unknown") or "unknown",
        "resolved_at_iteration": row.get("resolved_at_iteration", row.get("iteration", "")),
    }


def risk_and_policy(
    *,
    label: Dict[str, Any],
    scorer: Optional[Dict[str, str]],
    gap: Optional[Dict[str, str]],
    strong_rank1: float,
    weak_rank1: float,
    collapsed_rank1: float,
    oracle_min: float,
    other_positive_high: float,
) -> Dict[str, Any]:
    clip_count = int(label["clip_count"])
    resolved = bool(label["resolved"])
    cert = str(label["certificate_type"])
    d_rank1 = get_rank1(scorer)
    d_top5 = get_top5(scorer)
    d_support = support_weight(scorer)
    oracle_top1 = fnum(gap.get("oracle_top1_rate"), default=None) if gap else None
    weak_base = fnum(gap.get("weak_base_top1_rate"), default=None) if gap else None
    weak_nohub = fnum(gap.get("weak_nohub_top1_rate"), default=None) if gap else None
    rescue = fnum(gap.get("nohub_rescued_rate"), default=0.0) if gap else 0.0
    other_pos = fnum(gap.get("weak_nohub_other_positive_confusion_rate", gap.get("weak_base_other_positive_confusion_rate", "")), default=0.0) if gap else 0.0

    flags = []
    if clip_count <= 0:
        flags.append("train_absent")
    if 0 < clip_count < 3:
        flags.append("low_support_clip_lt_3")
    if "person" in cert:
        flags.append("person_conditioned")
    if other_pos >= other_positive_high:
        flags.append("high_other_positive_confusion")
    if d_rank1 is not None and d_rank1 < collapsed_rank1:
        flags.append("scorer_collapsed")
    if oracle_top1 is not None and oracle_top1 >= oracle_min and (weak_nohub is not None and weak_nohub < collapsed_rank1):
        flags.append("oracle_capacity_but_weak_collapsed")

    # Conservative policy: never hard-CE collapsed or train-absent/unresolved classes.
    reason = []
    if clip_count <= 0:
        policy = "train_absent_no_current_label_evidence"
        reason.append("clip_count=0")
    elif not resolved:
        policy = "deferred_unresolved_label_context"
        reason.append("label_not_resolved")
    elif d_rank1 is not None and d_rank1 >= strong_rank1 and (oracle_top1 is None or oracle_top1 >= oracle_min):
        if other_pos >= other_positive_high and rescue <= 0:
            policy = "weak_soft_ce_candidate"
            reason.append("strong_scorer_but_high_other_positive_confusion")
        else:
            policy = "strong_hard_ce_candidate"
            reason.append("resolved_and_rank1_strong")
    elif d_rank1 is not None and d_rank1 < collapsed_rank1:
        if oracle_top1 is not None and oracle_top1 >= oracle_min:
            policy = "prototype_calibration_candidate"
            reason.append("oracle_capacity_but_current_scorer_collapsed")
        else:
            policy = "deferred_unsafe_oracle_weak"
            reason.append("scorer_collapsed_and_oracle_not_strong")
    elif d_rank1 is not None and d_rank1 >= weak_rank1:
        policy = "weak_soft_ce_candidate"
        reason.append("resolved_and_rank1_weak")
    elif d_top5 is not None and d_top5 >= 0.5:
        policy = "weak_soft_ce_candidate"
        reason.append("resolved_and_top5_support")
    else:
        policy = "prototype_calibration_candidate"
        reason.append("resolved_but_needs_prototype_or_more_evidence")

    return {
        "policy": policy,
        "reason": ";".join(reason),
        "risk_flags": ";".join(flags),
        "D_gt_rank1_rate": d_rank1 if d_rank1 is not None else "",
        "D_gt_top5_rate": d_top5 if d_top5 is not None else "",
        "D_support_weight": d_support,
        "oracle_top1_rate": oracle_top1 if oracle_top1 is not None else "",
        "weak_base_top1_rate": weak_base if weak_base is not None else "",
        "weak_nohub_top1_rate": weak_nohub if weak_nohub is not None else "",
        "nohub_rescued_rate": rescue,
        "other_positive_confusion_rate": other_pos,
    }


def merge_rows(label_rows: List[Dict[str, str]], scorer_rows: List[Dict[str, str]], gap_rows: List[Dict[str, str]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    scorer_by = by_raw(scorer_rows)
    gap_by = by_raw(gap_rows)
    out = []
    for lr in label_rows:
        rid = raw_id(lr)
        lf = label_fields(lr)
        sr = scorer_by.get(rid)
        gr = gap_by.get(rid)
        pol = risk_and_policy(
            label=lf,
            scorer=sr,
            gap=gr,
            strong_rank1=args.strong_rank1,
            weak_rank1=args.weak_rank1,
            collapsed_rank1=args.collapsed_rank1,
            oracle_min=args.oracle_min,
            other_positive_high=args.other_positive_high,
        )
        out.append({
            "raw_id": rid,
            "class_name": lr.get("class_name", gr.get("class_name", "") if gr else ""),
            **lf,
            **pol,
        })
    return out


def read_seed_rows(paths: List[Path], policy_by_raw: Dict[str, Dict[str, Any]], max_rows_per_file: int) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        if not p.is_file():
            continue
        tag = p.stem
        with p.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            for idx, r in enumerate(rdr):
                if idx >= max_rows_per_file:
                    break
                rid = r.get("gt_raw_id") or r.get("raw_id") or raw_id(r)
                try:
                    rid = str(int(float(rid)))
                except Exception:
                    rid = str(rid)
                pol = policy_by_raw.get(rid, {})
                rows.append({
                    "seed_source": tag,
                    "candidate_scope": r.get("candidate_scope", ""),
                    "clip_id": r.get("clip_id", ""),
                    "trajectory_id": r.get("trajectory_id", ""),
                    "gt_raw_id": rid,
                    "gt_class_name": r.get("gt_class_name", pol.get("class_name", "")),
                    "weak_base_top1_class_name": r.get("weak_base_top1_class_name", ""),
                    "weak_nohub_top1_class_name": r.get("weak_nohub_top1_class_name", ""),
                    "oracle_top1_class_name": r.get("oracle_top1_class_name", ""),
                    "weak_base_error_type": r.get("weak_base_error_type", ""),
                    "weak_nohub_error_type": r.get("weak_nohub_error_type", ""),
                    "clip_y_size": r.get("clip_y_size", ""),
                    "recommended_class_policy": pol.get("policy", "missing_class_policy"),
                    "recommended_class_risk_flags": pol.get("risk_flags", ""),
                })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--variant", default="person_aware")
    ap.add_argument("--label_csv", default="")
    ap.add_argument("--scorer_join_csv", default="")
    ap.add_argument("--oracle_gap_csv", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--strong_rank1", type=float, default=0.50)
    ap.add_argument("--weak_rank1", type=float, default=0.25)
    ap.add_argument("--collapsed_rank1", type=float, default=0.10)
    ap.add_argument("--oracle_min", type=float, default=0.50)
    ap.add_argument("--other_positive_high", type=float, default=0.90)
    ap.add_argument("--max_seed_rows_per_file", type=int, default=200)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "analysis" / "residual_gated_pseudo_pool" / args.dataset_name
    cands = candidate_paths(run_root, args.dataset_name)
    label_path, inv_label = resolve_path(args.label_csv, cands["label_csv"], "label_csv")
    scorer_path, inv_scorer = resolve_path(args.scorer_join_csv, cands["scorer_join_csv"], "scorer_join_csv")
    gap_path, inv_gap = resolve_path(args.oracle_gap_csv, cands["oracle_gap_csv"], "oracle_gap_csv")
    inventory = {"artifacts": [inv_label, inv_scorer, inv_gap]}
    missing = [x for x in inventory["artifacts"] if x["status"] != "FOUND"]
    if missing:
        write_json(out_dir / "pseudo_pool_summary.json", {"status": "FAIL_MISSING_REQUIRED_ARTIFACT", "inventory": inventory})
        print(json.dumps({"status": "FAIL_MISSING_REQUIRED_ARTIFACT", "inventory": inventory}, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    label_rows = filter_label(read_csv(label_path), args.variant)  # type: ignore[arg-type]
    scorer_rows = read_csv(scorer_path)  # type: ignore[arg-type]
    gap_rows = read_csv(gap_path)  # type: ignore[arg-type]
    rows = merge_rows(label_rows, scorer_rows, gap_rows, args)
    policy_by_raw = {str(r["raw_id"]): r for r in rows}

    write_csv(out_dir / "class_training_policy.csv", rows)
    # Alias with old planned name: this is class-level, not row-level.
    write_csv(out_dir / "pseudo_pool_by_class.csv", rows)
    risky = [r for r in rows if "scorer_collapsed" in str(r.get("risk_flags", "")) or "oracle_capacity_but_weak_collapsed" in str(r.get("risk_flags", "")) or r.get("policy", "").startswith("deferred")]
    write_csv(out_dir / "risky_classes.csv", risky)

    seed_paths = cands["row_seed_csvs"]
    seed_rows = read_seed_rows(seed_paths, policy_by_raw, args.max_seed_rows_per_file)
    write_csv(out_dir / "pseudo_label_pool_seed_rows.csv", seed_rows)

    counts = Counter(r["policy"] for r in rows)
    cert_counts = Counter(r.get("certificate_type", "unknown") for r in rows)
    hard_ce_collapsed = [r for r in rows if r["policy"] == "strong_hard_ce_candidate" and fnum(r.get("D_gt_rank1_rate"), 1.0) < args.collapsed_rank1]
    sentinel = {}
    for name in ["shoe", "baseball_cap", "trousers", "watch", "rearview_mirror"]:
        hit = next((r for r in rows if r.get("class_name") == name), None)
        if hit:
            sentinel[name] = hit

    gate_checks = [
        {"name": "no_hard_ce_for_collapsed_classes", "status": "PASS" if not hard_ce_collapsed else "FAIL", "bad_count": len(hard_ce_collapsed)},
        {"name": "has_nonempty_policy_rows", "status": "PASS" if rows else "FAIL", "count": len(rows)},
        {"name": "has_seed_rows_if_available", "status": "PASS" if seed_rows else "WARN", "count": len(seed_rows)},
    ]
    final_status = "PASS" if all(c["status"] != "FAIL" for c in gate_checks) else "FAIL_GATE"
    summary = {
        "status": final_status,
        "run_root": str(run_root),
        "dataset_name": args.dataset_name,
        "variant": args.variant,
        "thresholds": {
            "strong_rank1": args.strong_rank1,
            "weak_rank1": args.weak_rank1,
            "collapsed_rank1": args.collapsed_rank1,
            "oracle_min": args.oracle_min,
            "other_positive_high": args.other_positive_high,
        },
        "inventory": inventory,
        "class_count": len(rows),
        "policy_counts": dict(counts),
        "certificate_counts": dict(cert_counts),
        "risky_class_count": len(risky),
        "seed_row_count": len(seed_rows),
        "sentinel_classes": sentinel,
        "gate_checks": gate_checks,
        "interpretation": {
            "hard_ce_only_for": "resolved classes with strong current scorer evidence; collapsed classes are blocked from hard CE.",
            "prototype_candidates": "resolved classes with oracle capacity but current scorer collapse or weak evidence.",
            "deferred": "train-absent, unresolved, or unsafe classes.",
            "gpu_required": False,
            "rowwise_large_matrix_compute": False,
        },
    }
    write_json(out_dir / "pseudo_pool_summary.json", summary)

    # Training recipe: deterministic but conservative defaults.
    recipe = {
        "class_balanced_sampling": {
            "sample_class_first": True,
            "max_samples_per_class_per_epoch_default": 64,
            "hub_class_cap_multiplier": 0.25,
            "low_support_repeat_cap": 8,
            "do_not_hard_ce_policies": ["prototype_calibration_candidate", "deferred_unresolved_label_context", "train_absent_no_current_label_evidence", "deferred_unsafe_oracle_weak"],
        },
        "loss_recommendation": {
            "strong_hard_ce_candidate": "hard_ce_allowed",
            "weak_soft_ce_candidate": "soft_ce_only",
            "prototype_calibration_candidate": "prototype_calibration_or_soft_contrastive_only",
            "deferred": "no_training_signal_in_phase02",
            "ordinary_infonce": "blocked_unless_positive_set_aware_safe_negatives_are_used",
        },
    }
    write_json(out_dir / "training_recipe_recommendation.json", recipe)

    md = [
        "# Residual-Gated Pseudo Pool",
        "",
        f"- status: {final_status}",
        f"- class_count: {len(rows)}",
        f"- seed_row_count: {len(seed_rows)}",
        "",
        "## Policy counts",
    ]
    for k, v in sorted(counts.items()):
        md.append(f"- {k}: {v}")
    md.extend(["", "## Sentinel classes", "```json", json.dumps(sentinel, ensure_ascii=False, indent=2, default=str), "```", "", "## Gate checks"])
    for c in gate_checks:
        md.append(f"- {c['name']}: {c['status']} { {k:v for k,v in c.items() if k not in {'name','status'}} }")
    (out_dir / "RESIDUAL_GATED_PSEUDO_POOL_TAKEOVER.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if final_status.startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
