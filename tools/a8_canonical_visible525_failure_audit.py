#!/usr/bin/env python3
"""Canonical Visible-525 failure audit.

This tool audits the canonical metric surface:
  target in train-visible 525, candidate in train-visible 525, GT trajectory source.

It consumes the per-row CSV emitted by tools/a8_visible525_candidate_rankk_audit.py,
so it never redefines the candidate set or target set.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


K_LIST = (1, 5, 10, 20, 50)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(str(k))
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(str(x)))
    except Exception:
        return default


def to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(str(x))
    except Exception:
        return default


def truthy(x: Any) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y"}


def mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def safe_median(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(median(vals)) if vals else None


def count_bin(n: int) -> str:
    if n <= 1:
        return "01_1"
    if n <= 5:
        return "02_2_5"
    if n <= 20:
        return "03_6_20"
    if n <= 50:
        return "04_21_50"
    if n <= 100:
        return "05_51_100"
    if n <= 200:
        return "06_101_200"
    return "07_gt200"


def rank_bin(rank: int) -> str:
    if rank <= 1:
        return "01_top1"
    if rank == 2:
        return "02_rank2_near_miss"
    if rank <= 5:
        return "03_rank3_5"
    if rank <= 10:
        return "04_rank6_10"
    if rank <= 20:
        return "05_rank11_20"
    if rank <= 50:
        return "06_rank21_50"
    return "07_rank_gt50"


def margin_bin(m: Optional[float]) -> str:
    if m is None or not math.isfinite(float(m)):
        return "missing_margin"
    m = float(m)
    if m >= 0:
        return "01_gt_top1_margin_ge0"
    if m >= -0.05:
        return "02_small_negative_0_to_-0.05"
    if m >= -0.10:
        return "03_negative_-0.05_to_-0.10"
    if m >= -0.25:
        return "04_negative_-0.10_to_-0.25"
    if m >= -0.50:
        return "05_negative_-0.25_to_-0.50"
    return "06_large_negative_lt_-0.50"


def normalize_rows(raw_rows: Sequence[Mapping[str, str]], *, source_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        gt = to_int(r.get("gt_raw_id"))
        rank = to_int(r.get("restricted_rank") or r.get("rank") or r.get("gt_rank"))
        if gt is None or rank is None:
            continue
        gt_name = str(r.get("gt_name") or r.get("gt_class_name") or r.get("class_name") or "")
        top1_raw = to_int(r.get("top1_raw_id"), default=-1)
        top1_name = str(r.get("top1_name") or "")
        margin = to_float(r.get("margin_gt_minus_top1") or r.get("margin") or r.get("gt_minus_top1_score"))
        gt_score = to_float(r.get("gt_score"))
        top1_score = to_float(r.get("top1_score"))
        rows.append({
            "source_name": source_name,
            "dataset_name": r.get("dataset_name", ""),
            "clip_id": r.get("clip_id", ""),
            "trajectory_id": r.get("trajectory_id", ""),
            "video_id": r.get("video_id", ""),
            "gt_raw_id": int(gt),
            "gt_name": gt_name,
            "candidate_scope": r.get("candidate_scope", ""),
            "candidate_count": to_int(r.get("candidate_count"), default=None),
            "rank": int(rank),
            "top1_raw_id": int(top1_raw) if top1_raw is not None else -1,
            "top1_name": top1_name,
            "margin": margin,
            "gt_score": gt_score,
            "top1_score": top1_score,
            "is_top1": int(rank <= 1),
            "is_failure": int(rank > 1),
            "rank_bin": rank_bin(int(rank)),
            "margin_bin": margin_bin(margin),
        })
    return rows


def summarize_overall(rows: Sequence[Mapping[str, Any]], *, source_name: str) -> Dict[str, Any]:
    n = len(rows)
    ranks = [int(r["rank"]) for r in rows]
    if not rows:
        return {"source_name": source_name, "count": 0}
    failures = [r for r in rows if int(r["rank"]) > 1]
    return {
        "source_name": source_name,
        "count": n,
        "class_count": len({int(r["gt_raw_id"]) for r in rows}),
        **{f"rank@{k}": sum(int(x) <= k for x in ranks) / n for k in K_LIST},
        "failure_rate": len(failures) / n,
        "mean_rank": mean([float(x) for x in ranks]),
        "median_rank": safe_median([float(x) for x in ranks]),
        "mean_margin": mean([r.get("margin") for r in rows]),
        "failure_mean_margin": mean([r.get("margin") for r in failures]),
        "failure_count": len(failures),
    }


def summarize_by_key(rows: Sequence[Mapping[str, Any]], key: str, *, source_name: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key, ""))].append(r)
    out: List[Dict[str, Any]] = []
    for g in sorted(groups):
        rs = groups[g]
        n = len(rs)
        ranks = [int(r["rank"]) for r in rs]
        failures = [r for r in rs if int(r["rank"]) > 1]
        out.append({
            "source_name": source_name,
            key: g,
            "row_count": n,
            "failure_count": len(failures),
            "failure_rate": len(failures) / n if n else 0,
            **{f"rank@{k}": sum(x <= k for x in ranks) / n if n else 0 for k in K_LIST},
            "mean_rank": mean([float(x) for x in ranks]),
            "median_rank": safe_median([float(x) for x in ranks]),
            "mean_margin": mean([r.get("margin") for r in rs]),
            "failure_mean_margin": mean([r.get("margin") for r in failures]),
        })
    return out


def top_suppressors(rows: Sequence[Mapping[str, Any]], *, source_name: str, topn: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    failures = [r for r in rows if int(r["rank"]) > 1]
    overall_counter: Counter[Tuple[int, str]] = Counter()
    pair_counter: Counter[Tuple[int, str, int, str]] = Counter()
    pair_margin: Dict[Tuple[int, str, int, str], List[float]] = defaultdict(list)
    pair_rank: Dict[Tuple[int, str, int, str], List[int]] = defaultdict(list)

    for r in failures:
        top = (int(r.get("top1_raw_id", -1)), str(r.get("top1_name", "")))
        gt = (int(r["gt_raw_id"]), str(r.get("gt_name", "")))
        overall_counter[top] += 1
        key = (gt[0], gt[1], top[0], top[1])
        pair_counter[key] += 1
        if r.get("margin") is not None:
            pair_margin[key].append(float(r["margin"]))
        pair_rank[key].append(int(r["rank"]))

    overall_rows: List[Dict[str, Any]] = []
    denom = len(failures) or 1
    for (top_id, top_name), c in overall_counter.most_common(topn):
        overall_rows.append({
            "source_name": source_name,
            "top1_raw_id": top_id,
            "top1_name": top_name,
            "wrong_count": c,
            "wrong_rate_within_failures": c / denom,
        })

    pair_rows: List[Dict[str, Any]] = []
    for key, c in pair_counter.most_common(topn):
        gt_id, gt_name, top_id, top_name = key
        pair_rows.append({
            "source_name": source_name,
            "gt_raw_id": gt_id,
            "gt_name": gt_name,
            "top1_raw_id": top_id,
            "top1_name": top_name,
            "wrong_count": c,
            "mean_margin": mean(pair_margin.get(key, [])),
            "mean_rank": mean([float(x) for x in pair_rank.get(key, [])]),
        })
    return overall_rows, pair_rows


def per_class_summary(rows: Sequence[Mapping[str, Any]], *, source_name: str, topn_per_class: int = 5) -> List[Dict[str, Any]]:
    groups: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[int(r["gt_raw_id"])].append(r)
    out: List[Dict[str, Any]] = []
    for cid in sorted(groups):
        rs = groups[cid]
        n = len(rs)
        ranks = [int(r["rank"]) for r in rs]
        failures = [r for r in rs if int(r["rank"]) > 1]
        sup = Counter((int(r.get("top1_raw_id", -1)), str(r.get("top1_name", ""))) for r in failures)
        sup_str = " | ".join(f"{sid}:{name}:{cnt}" for (sid, name), cnt in sup.most_common(topn_per_class))
        out.append({
            "source_name": source_name,
            "gt_raw_id": cid,
            "gt_name": str(rs[0].get("gt_name", "")),
            "count": n,
            "count_bin": count_bin(n),
            "failure_count": len(failures),
            "failure_rate": len(failures) / n if n else 0,
            **{f"rank@{k}": sum(x <= k for x in ranks) / n if n else 0 for k in K_LIST},
            "mean_rank": mean([float(x) for x in ranks]),
            "median_rank": safe_median([float(x) for x in ranks]),
            "mean_margin": mean([r.get("margin") for r in rs]),
            "failure_mean_margin": mean([r.get("margin") for r in failures]),
            "top_suppressors": sup_str,
        })
    out.sort(key=lambda r: (float(r["rank@1"]), -int(r["count"]), int(r["gt_raw_id"])))
    return out


def count_bin_summary(class_rows: Sequence[Mapping[str, Any]], *, source_name: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in class_rows:
        groups[str(r["count_bin"])].append(r)
    out: List[Dict[str, Any]] = []
    for b in sorted(groups):
        rs = groups[b]
        total_rows = sum(int(r["count"]) for r in rs)
        weighted_rank1 = sum(float(r["rank@1"]) * int(r["count"]) for r in rs) / total_rows if total_rows else 0
        weighted_rank5 = sum(float(r["rank@5"]) * int(r["count"]) for r in rs) / total_rows if total_rows else 0
        weighted_rank50 = sum(float(r["rank@50"]) * int(r["count"]) for r in rs) / total_rows if total_rows else 0
        out.append({
            "source_name": source_name,
            "count_bin": b,
            "class_count": len(rs),
            "row_count": total_rows,
            "unweighted_rank@1": mean([float(r["rank@1"]) for r in rs]),
            "row_weighted_rank@1": weighted_rank1,
            "row_weighted_rank@5": weighted_rank5,
            "row_weighted_rank@50": weighted_rank50,
            "unweighted_failure_rate": mean([float(r["failure_rate"]) for r in rs]),
        })
    return out


def compare_class_summaries(primary: Sequence[Mapping[str, Any]], compare: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    p = {int(r["gt_raw_id"]): r for r in primary}
    q = {int(r["gt_raw_id"]): r for r in compare}
    rows: List[Dict[str, Any]] = []
    for cid in sorted(set(p) & set(q)):
        a = p[cid]
        b = q[cid]
        rows.append({
            "gt_raw_id": cid,
            "gt_name": a.get("gt_name") or b.get("gt_name"),
            "primary_count": a.get("count"),
            "compare_count": b.get("count"),
            "primary_rank@1": a.get("rank@1"),
            "compare_rank@1": b.get("rank@1"),
            "rank@1_drop_primary_minus_compare": float(a.get("rank@1", 0)) - float(b.get("rank@1", 0)),
            "primary_rank@5": a.get("rank@5"),
            "compare_rank@5": b.get("rank@5"),
            "primary_rank@50": a.get("rank@50"),
            "compare_rank@50": b.get("rank@50"),
            "primary_top_suppressors": a.get("top_suppressors"),
            "compare_top_suppressors": b.get("top_suppressors"),
        })
    rows.sort(key=lambda r: (-float(r["rank@1_drop_primary_minus_compare"]), -int(r["compare_count"] or 0)))
    return rows


def read_matched_proxy(path: Path) -> List[Dict[str, Any]]:
    if not path or not path.is_file():
        return []
    rows = read_csv(path)
    counts: Dict[int, Counter[str]] = defaultdict(Counter)
    names: Dict[int, str] = {}
    for r in rows:
        for field, key in (("matched_raw_id", "matched_raw_id_rows"), ("audit_gt_raw_id", "audit_gt_raw_id_rows"), ("gt_raw_id", "gt_raw_id_rows")):
            cid = to_int(r.get(field))
            if cid is not None:
                counts[int(cid)][key] += 1
                if cid not in names:
                    names[cid] = str(r.get("gt_name") or r.get("matched_name") or r.get("audit_gt_name") or "")
    out: List[Dict[str, Any]] = []
    for cid in sorted(counts):
        c = counts[cid]
        out.append({
            "raw_id": cid,
            "name": names.get(cid, ""),
            "matched_raw_id_rows": c.get("matched_raw_id_rows", 0),
            "audit_gt_raw_id_rows": c.get("audit_gt_raw_id_rows", 0),
            "gt_raw_id_rows": c.get("gt_raw_id_rows", 0),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_row_csv", required=True, help="Canonical visible525 per-row CSV for the primary split/run.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--source_name", default="primary")
    ap.add_argument("--compare_per_row_csv", default="", help="Optional canonical visible525 per-row CSV to compare, e.g. val.")
    ap.add_argument("--compare_source_name", default="compare")
    ap.add_argument("--matched_pairs_csv", default="", help="Optional static matched_pairs_csv exposure proxy.")
    ap.add_argument("--topn", type=int, default=100)
    args = ap.parse_args()

    per_row_csv = Path(args.per_row_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not per_row_csv.is_file():
        raise FileNotFoundError(per_row_csv)

    primary_rows = normalize_rows(read_csv(per_row_csv), source_name=args.source_name)
    if not primary_rows:
        raise RuntimeError(f"No valid canonical rows loaded from {per_row_csv}")

    overall_rows = [summarize_overall(primary_rows, source_name=args.source_name)]
    margin_rows = summarize_by_key(primary_rows, "margin_bin", source_name=args.source_name)
    rank_bin_rows = summarize_by_key(primary_rows, "rank_bin", source_name=args.source_name)
    class_rows = per_class_summary(primary_rows, source_name=args.source_name)
    count_bin_rows = count_bin_summary(class_rows, source_name=args.source_name)
    sup_overall_rows, sup_pair_rows = top_suppressors(primary_rows, source_name=args.source_name, topn=args.topn)

    compare_class_rows: List[Dict[str, Any]] = []
    compare_overall_rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []
    if args.compare_per_row_csv:
        compare_path = Path(args.compare_per_row_csv).expanduser().resolve()
        if not compare_path.is_file():
            raise FileNotFoundError(compare_path)
        compare_rows = normalize_rows(read_csv(compare_path), source_name=args.compare_source_name)
        compare_overall_rows = [summarize_overall(compare_rows, source_name=args.compare_source_name)]
        compare_class_summary = per_class_summary(compare_rows, source_name=args.compare_source_name)
        compare_class_rows = compare_class_summaries(class_rows, compare_class_summary)
        overall_rows.extend(compare_overall_rows)

    matched_proxy_rows: List[Dict[str, Any]] = []
    if args.matched_pairs_csv:
        matched_proxy_rows = read_matched_proxy(Path(args.matched_pairs_csv).expanduser().resolve())

    write_csv(output_dir / "canonical_visible525_overall_summary.csv", overall_rows)
    write_csv(output_dir / "failure_margin_bins.csv", margin_rows)
    write_csv(output_dir / "failure_rank_bins.csv", rank_bin_rows)
    write_csv(output_dir / "failure_by_gt_class.csv", class_rows)
    write_csv(output_dir / "failure_by_count_bin.csv", count_bin_rows)
    write_csv(output_dir / "failure_top_suppressors_overall.csv", sup_overall_rows)
    write_csv(output_dir / "failure_top_suppressor_pairs.csv", sup_pair_rows)
    if compare_class_rows:
        write_csv(output_dir / "train_val_or_primary_compare_by_class.csv", compare_class_rows)
    if matched_proxy_rows:
        write_csv(output_dir / "matched_pairs_exposure_proxy_by_class.csv", matched_proxy_rows)

    payload = {
        "status": "PASS",
        "definition": "canonical visible525 failure audit: target=train-visible 525, candidate=train-visible 525, source=GT trajectory per-row CSV",
        "primary_per_row_csv": str(per_row_csv),
        "primary_source_name": args.source_name,
        "compare_per_row_csv": str(Path(args.compare_per_row_csv).expanduser().resolve()) if args.compare_per_row_csv else "",
        "compare_source_name": args.compare_source_name if args.compare_per_row_csv else "",
        "matched_pairs_csv": str(Path(args.matched_pairs_csv).expanduser().resolve()) if args.matched_pairs_csv else "",
        "overall_summary": overall_rows,
        "primary_failure_margin_bin_summary": margin_rows,
        "primary_failure_rank_bin_summary": rank_bin_rows,
        "primary_top_suppressors": sup_overall_rows[:20],
        "primary_top_suppressor_pairs": sup_pair_rows[:30],
        "artifacts": {
            "overall_summary_csv": str(output_dir / "canonical_visible525_overall_summary.csv"),
            "failure_margin_bins_csv": str(output_dir / "failure_margin_bins.csv"),
            "failure_rank_bins_csv": str(output_dir / "failure_rank_bins.csv"),
            "failure_by_gt_class_csv": str(output_dir / "failure_by_gt_class.csv"),
            "failure_by_count_bin_csv": str(output_dir / "failure_by_count_bin.csv"),
            "failure_top_suppressors_overall_csv": str(output_dir / "failure_top_suppressors_overall.csv"),
            "failure_top_suppressor_pairs_csv": str(output_dir / "failure_top_suppressor_pairs.csv"),
            "compare_by_class_csv": str(output_dir / "train_val_or_primary_compare_by_class.csv") if compare_class_rows else "",
            "matched_pairs_exposure_proxy_by_class_csv": str(output_dir / "matched_pairs_exposure_proxy_by_class.csv") if matched_proxy_rows else "",
        },
    }
    write_json(output_dir / "canonical_visible525_failure_audit_summary.json", payload)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
