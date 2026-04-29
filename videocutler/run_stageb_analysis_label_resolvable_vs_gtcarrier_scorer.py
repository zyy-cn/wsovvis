#!/usr/bin/env python3
"""Join label-only iterative residual identifiability with GT-carrier full-Y scorer audit.

Read-only reducer. It does not train, infer, or modify existing artifacts.

Main fixes vs ad-hoc reducer:
- filters exactly one iterative-residual variant (default: person_aware), avoiding 2x duplicate class rows;
- robustly discovers D-scorer column names, including top-k columns;
- reports macro class stats and micro weighted stats using gt_count-like support columns;
- emits compact CSV/JSON/MD summaries and flags missing columns explicitly.
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
FALSE_SET = {"0", "false", "no", "n", "fail"}

RAW_ID_KEYS = [
    "raw_id",
    "raw_category_id",
    "category_raw_id",
    "category_id",
    "class_raw_id",
    "gt_raw_id",
]

RANK1_KEYS = [
    "gt_rank1_rate",
    "gt_top1_hit_rate",
    "gt_top1_rate",
    "top1_hit_rate",
    "top1_rate",
    "rank1_rate",
    "gt_rank1_hit_rate",
]
TOP5_KEYS = [
    "gt_top5_rate",
    "gt_top5_hit_rate",
    "gt_top5_contains_rate",
    "top5_hit_rate",
    "top5_rate",
    "top5_contains_gt_rate",
    "gt_in_top5_rate",
]
TOP20_KEYS = [
    "gt_top20_rate",
    "gt_top20_hit_rate",
    "gt_top20_contains_rate",
    "top20_hit_rate",
    "top20_rate",
    "top20_contains_gt_rate",
    "gt_in_top20_rate",
]
MEAN_RANK_KEYS = [
    "mean_gt_rank",
    "gt_mean_rank",
    "mean_text_gt_rank_full_vocab",
    "mean_final_gt_rank",
    "mean_rank",
]
MEAN_NORM_RANK_KEYS = [
    "mean_normalized_gt_rank",
    "mean_final_gt_normalized_rank",
    "gt_mean_normalized_rank",
]
MARGIN_KEYS = [
    "mean_gt_margin",
    "mean_gt_margin_vs_best_non_gt",
    "mean_text_margin_gt_vs_person",
    "mean_margin",
]
SUPPORT_KEYS = [
    "gt_count",
    "candidate_contains_gt_count",
    "class_row_count",
    "row_count",
    "count",
    "instance_count",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


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
        return int(round(float(x)))
    except Exception:
        return default


def truth(x: Any) -> bool:
    s = str(x).strip().lower()
    if s in TRUE_SET:
        return True
    if s in FALSE_SET:
        return False
    return False


def raw_id(row: Dict[str, Any]) -> Optional[str]:
    for k in RAW_ID_KEYS:
        if k in row and str(row[k]).strip() != "":
            try:
                return str(int(float(str(row[k]).strip())))
            except Exception:
                return str(row[k]).strip()
    return None


def find_col(headers: Sequence[str], candidates: Sequence[str], fuzzy_terms: Sequence[str] = ()) -> Optional[str]:
    hset = {h: h for h in headers}
    lower = {h.lower(): h for h in headers}
    for c in candidates:
        if c in hset:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    if fuzzy_terms:
        terms = [t.lower() for t in fuzzy_terms]
        for h in headers:
            hl = h.lower()
            if all(t in hl for t in terms):
                return h
    return None


def get_first(row: Dict[str, str], keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    return default


def recognition_bucket(rate: float) -> str:
    if rate >= 0.75:
        return "strongly_recognized_ge_0.75"
    if rate >= 0.50:
        return "recognized_ge_0.50"
    if rate >= 0.25:
        return "weakly_recognized_ge_0.25"
    if rate >= 0.10:
        return "mostly_failed_ge_0.10"
    return "collapsed_lt_0.10"


def summarize(rows: List[Dict[str, Any]], rank1_col: str = "D_gt_rank1_rate", top5_col: str = "D_gt_top5_rate", support_col: str = "D_support_weight") -> Dict[str, Any]:
    scored = [r for r in rows if r.get("join_has_d_scorer")]
    vals1 = [fnum(r.get(rank1_col)) for r in scored]
    vals5 = [fnum(r.get(top5_col), default=float("nan")) for r in scored]
    vals5_valid = [v for v in vals5 if not math.isnan(v)]
    supports = [max(0.0, fnum(r.get(support_col), default=1.0)) for r in scored]

    out: Dict[str, Any] = {"count": len(rows), "scored_count": len(scored)}
    if not scored:
        return out

    total_w = sum(supports) if supports else 0.0
    out.update({
        "mean_gt_rank1_rate_macro": sum(vals1) / len(vals1),
        "median_gt_rank1_rate_macro": statistics.median(vals1),
        "recognized_ge_0.50_count": sum(v >= 0.5 for v in vals1),
        "recognized_ge_0.50_rate_macro": sum(v >= 0.5 for v in vals1) / len(vals1),
        "collapsed_lt_0.10_count": sum(v < 0.1 for v in vals1),
        "collapsed_lt_0.10_rate_macro": sum(v < 0.1 for v in vals1) / len(vals1),
        "bucket_counts": dict(Counter(recognition_bucket(v) for v in vals1)),
    })
    if vals5_valid:
        out["mean_gt_top5_rate_macro"] = sum(vals5_valid) / len(vals5_valid)
        out["median_gt_top5_rate_macro"] = statistics.median(vals5_valid)
    else:
        out["mean_gt_top5_rate_macro"] = None
        out["median_gt_top5_rate_macro"] = None

    if total_w > 0:
        out["support_weight_total"] = total_w
        out["mean_gt_rank1_rate_micro_weighted"] = sum(v * w for v, w in zip(vals1, supports)) / total_w
        if vals5_valid and len(vals5_valid) == len(scored):
            out["mean_gt_top5_rate_micro_weighted"] = sum(v * w for v, w in zip(vals5, supports)) / total_w
    return out


def detect_variant_values(rows: List[Dict[str, str]]) -> List[str]:
    vals = sorted({r.get("variant", "") for r in rows if r.get("variant", "")})
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Join iterative residual label identifiability with GT-carrier scorer audit.")
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--label_csv", default="", help="per_class_iterative_residual_identifiability.csv")
    ap.add_argument("--d_csv", default="", help="D_fully_gtcarrier_latent_by_class.csv; if omitted, search under --search_root")
    ap.add_argument("--search_root", default="", help="Root to search D csv; defaults to run_root ancestor G8_inference_and_eval if possible")
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--variant", default="person_aware", help="Variant to keep: person_aware or strict_anchor")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--allow_missing_top5", action="store_true")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    label_csv = Path(args.label_csv) if args.label_csv else run_root / "analysis" / "iterative_residual_label_identifiability" / args.dataset_name / "per_class_iterative_residual_identifiability.csv"

    if args.d_csv:
        d_csv = Path(args.d_csv)
    else:
        search_root = Path(args.search_root) if args.search_root else run_root.parent
        candidates = list(search_root.rglob("D_fully_gtcarrier_latent_by_class.csv"))
        if not candidates:
            raise FileNotFoundError(f"Cannot find D_fully_gtcarrier_latent_by_class.csv under {search_root}; pass --d_csv")
        # Prefer oracle_clean_data_ablation and gtcarrier paths if present.
        candidates.sort(key=lambda p: ("oracle_clean_data_ablation" not in str(p), "gtcarrier" not in str(p), str(p)))
        d_csv = candidates[0]

    out_dir = Path(args.out_dir) if args.out_dir else run_root / "analysis" / "label_resolvable_vs_gtcarrier_scorer" / args.dataset_name / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    label_rows_all = read_csv(label_csv)
    d_rows = read_csv(d_csv)
    if not label_rows_all:
        raise RuntimeError(f"No label rows in {label_csv}")
    if not d_rows:
        raise RuntimeError(f"No D scorer rows in {d_csv}")

    variants = detect_variant_values(label_rows_all)
    if variants:
        label_rows = [r for r in label_rows_all if r.get("variant") == args.variant]
    else:
        label_rows = label_rows_all
    if not label_rows:
        raise RuntimeError(f"No rows after variant filter {args.variant!r}; available={variants}")

    d_headers = list(d_rows[0].keys())
    rank1_col = find_col(d_headers, RANK1_KEYS, fuzzy_terms=("rank1",)) or find_col(d_headers, RANK1_KEYS, fuzzy_terms=("top1",))
    top5_col = find_col(d_headers, TOP5_KEYS, fuzzy_terms=("top5",))
    top20_col = find_col(d_headers, TOP20_KEYS, fuzzy_terms=("top20",))
    mean_rank_col = find_col(d_headers, MEAN_RANK_KEYS, fuzzy_terms=("mean", "rank"))
    mean_norm_rank_col = find_col(d_headers, MEAN_NORM_RANK_KEYS, fuzzy_terms=("normalized", "rank"))
    margin_col = find_col(d_headers, MARGIN_KEYS, fuzzy_terms=("margin",))
    support_col = find_col(d_headers, SUPPORT_KEYS)

    if not rank1_col:
        raise RuntimeError(f"Could not identify rank1 column in D csv. Headers={d_headers}")
    if not top5_col and not args.allow_missing_top5:
        # Do not hard-fail because some D by-class csv may omit top5. But flag clearly.
        pass

    d_by_id: Dict[str, Dict[str, str]] = {}
    d_duplicates = Counter()
    for r in d_rows:
        rid = raw_id(r)
        if rid:
            if rid in d_by_id:
                d_duplicates[rid] += 1
            d_by_id[rid] = r

    joined: List[Dict[str, Any]] = []
    duplicate_ids = Counter()
    seen_label_ids = set()
    for r in label_rows:
        rid = raw_id(r)
        if not rid:
            continue
        split_type = r.get("split_type", r.get("target_split", "base")) or "base"
        if split_type != "base":
            continue
        clip_count = inum(get_first(r, ["clip_count", "base_clip_count"], "0"))
        # Train-observed only; absent official-base classes are not the target universe here.
        if clip_count <= 0:
            continue
        if rid in seen_label_ids:
            duplicate_ids[rid] += 1
            continue
        seen_label_ids.add(rid)

        d = d_by_id.get(rid)
        resolved = truth(get_first(r, ["resolved", "is_resolved"], ""))
        cert = get_first(r, ["certificate_type", "certificate", "final_bucket"], "unknown")
        out = dict(r)
        out["variant_kept"] = args.variant
        out["raw_id_norm"] = rid
        out["resolved_bool"] = str(resolved).lower()
        out["certificate_type_norm"] = cert
        out["join_has_d_scorer"] = bool(d)
        if d:
            out["D_rank1_column"] = rank1_col
            out["D_gt_rank1_rate"] = fnum(d.get(rank1_col))
            if top5_col:
                out["D_top5_column"] = top5_col
                out["D_gt_top5_rate"] = fnum(d.get(top5_col))
            else:
                out["D_top5_column"] = ""
                out["D_gt_top5_rate"] = ""
            if top20_col:
                out["D_top20_column"] = top20_col
                out["D_gt_top20_rate"] = fnum(d.get(top20_col))
            if mean_rank_col:
                out["D_mean_rank_column"] = mean_rank_col
                out["D_mean_gt_rank"] = fnum(d.get(mean_rank_col))
            if mean_norm_rank_col:
                out["D_mean_norm_rank_column"] = mean_norm_rank_col
                out["D_mean_normalized_gt_rank"] = fnum(d.get(mean_norm_rank_col))
            if margin_col:
                out["D_margin_column"] = margin_col
                out["D_mean_gt_margin"] = fnum(d.get(margin_col))
            if support_col:
                out["D_support_column"] = support_col
                out["D_support_weight"] = max(1.0, fnum(d.get(support_col), default=1.0))
            else:
                out["D_support_column"] = ""
                out["D_support_weight"] = 1.0
        else:
            out["D_gt_rank1_rate"] = ""
            out["D_gt_top5_rate"] = ""
            out["D_support_weight"] = ""
        joined.append(out)

    resolved_rows = [r for r in joined if truth(r.get("resolved_bool"))]
    unresolved_rows = [r for r in joined if not truth(r.get("resolved_bool"))]

    by_cert: Dict[str, Dict[str, Any]] = {}
    cert_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in resolved_rows:
        cert_groups[str(r.get("certificate_type_norm", "unknown"))].append(r)
    for cert, rows in sorted(cert_groups.items()):
        by_cert[cert] = summarize(rows)

    # Compare against expected counts from this run if available.
    expected_summary_path = label_csv.parent / "summary.json"
    expected: Dict[str, Any] = {}
    if expected_summary_path.exists():
        try:
            expected = json.loads(expected_summary_path.read_text(encoding="utf-8"))
        except Exception:
            expected = {}

    summary: Dict[str, Any] = {
        "status": "PASS",
        "run_root": str(run_root),
        "dataset_name": args.dataset_name,
        "variant": args.variant,
        "available_variants_in_label_csv": variants,
        "label_csv": str(label_csv),
        "d_csv": str(d_csv),
        "output_dir": str(out_dir),
        "d_detected_columns": {
            "rank1": rank1_col,
            "top5": top5_col,
            "top20": top20_col,
            "mean_rank": mean_rank_col,
            "mean_normalized_rank": mean_norm_rank_col,
            "margin": margin_col,
            "support_weight": support_col,
        },
        "warnings": [],
        "label_rows_total_before_variant_filter": len(label_rows_all),
        "label_rows_after_variant_filter": len(label_rows),
        "train_observed_base_joined_count": len(joined),
        "label_resolved_count": len(resolved_rows),
        "label_unresolved_count": len(unresolved_rows),
        "label_resolved_with_d_scorer_count": sum(bool(r.get("join_has_d_scorer")) for r in resolved_rows),
        "label_unresolved_with_d_scorer_count": sum(bool(r.get("join_has_d_scorer")) for r in unresolved_rows),
        "duplicate_label_ids_skipped": dict(duplicate_ids),
        "duplicate_d_ids_seen": dict(d_duplicates),
        "resolved_summary": summarize(resolved_rows),
        "unresolved_summary": summarize(unresolved_rows),
        "resolved_by_certificate": by_cert,
    }
    if expected:
        summary["source_iterative_summary_counts"] = {
            "official_base_count": expected.get("official_base_count"),
            "train_observed_base_count": expected.get("train_observed_base_count"),
            "train_absent_base_count": expected.get("train_absent_base_count"),
            "variant_resolved_total_count": expected.get("variants", {}).get(args.variant, {}).get("resolved_total_count"),
            "variant_unresolved_count": expected.get("variants", {}).get(args.variant, {}).get("unresolved_count"),
        }
        exp_res = expected.get("variants", {}).get(args.variant, {}).get("resolved_total_count")
        exp_unres = expected.get("variants", {}).get(args.variant, {}).get("unresolved_count")
        if exp_res is not None and exp_res != len(resolved_rows):
            summary["warnings"].append(f"resolved count differs from iterative summary: expected {exp_res}, got {len(resolved_rows)}")
        if exp_unres is not None and exp_unres != len(unresolved_rows):
            summary["warnings"].append(f"unresolved count differs from iterative summary: expected {exp_unres}, got {len(unresolved_rows)}")
    if not top5_col:
        summary["warnings"].append("top5 column was not detected; D_gt_top5_rate omitted instead of defaulting to 0")

    write_csv(out_dir / "label_resolvable_vs_gtcarrier_scorer_by_class.csv", joined)

    cert_summary_rows = []
    for cert, stats in by_cert.items():
        row = {"certificate_type": cert}
        row.update(stats)
        cert_summary_rows.append(row)
    write_csv(out_dir / "summary_by_certificate.csv", cert_summary_rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md: List[str] = []
    md.append("# Label-resolvable vs GT-carrier Scorer Audit")
    md.append("")
    md.append(f"- status: {summary['status']}")
    md.append(f"- variant: `{args.variant}`")
    md.append(f"- train_observed_base_joined_count: {len(joined)}")
    md.append(f"- label_resolved_count: {len(resolved_rows)}")
    md.append(f"- label_unresolved_count: {len(unresolved_rows)}")
    md.append(f"- D rank1 column: `{rank1_col}`")
    md.append(f"- D top5 column: `{top5_col or 'MISSING'}`")
    md.append("")
    rs = summary["resolved_summary"]
    md.append("## Resolved classes")
    md.append(f"- macro mean gt_rank1_rate: {rs.get('mean_gt_rank1_rate_macro')}")
    md.append(f"- macro median gt_rank1_rate: {rs.get('median_gt_rank1_rate_macro')}")
    md.append(f"- macro mean gt_top5_rate: {rs.get('mean_gt_top5_rate_macro')}")
    md.append(f"- recognized@0.5: {rs.get('recognized_ge_0.50_count')} / {rs.get('scored_count')} = {rs.get('recognized_ge_0.50_rate_macro')}")
    md.append(f"- collapsed<0.1: {rs.get('collapsed_lt_0.10_count')} / {rs.get('scored_count')} = {rs.get('collapsed_lt_0.10_rate_macro')}")
    if "mean_gt_rank1_rate_micro_weighted" in rs:
        md.append(f"- micro-weighted mean gt_rank1_rate: {rs.get('mean_gt_rank1_rate_micro_weighted')}")
    md.append("")
    md.append("## By certificate")
    for cert, stats in by_cert.items():
        md.append(f"- `{cert}`: count={stats.get('count')}, mean_rank1={stats.get('mean_gt_rank1_rate_macro')}, recognized@0.5={stats.get('recognized_ge_0.50_rate_macro')}, collapsed<0.1={stats.get('collapsed_lt_0.10_rate_macro')}")
    if summary["warnings"]:
        md.append("")
        md.append("## Warnings")
        for w in summary["warnings"]:
            md.append(f"- {w}")
    (out_dir / "LABEL_RESOLVABLE_VS_GTCARRIER_SCORER_TAKEOVER.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
