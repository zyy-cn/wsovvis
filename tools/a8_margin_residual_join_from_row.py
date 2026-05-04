#!/usr/bin/env python3
"""Join a true-margin row audit with residual-peeling identifiability.

This is a standalone audit utility. It does not train or modify artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        if not fieldnames:
            fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def inum(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def bval(x: Any) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y"}


def detect(row: Dict[str, str], candidates: Iterable[str], default: str = "") -> str:
    for c in candidates:
        v = row.get(c)
        if v is not None and str(v) != "":
            return str(v)
    return default


def build_residual_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_csv(path):
        rid = str(row.get("raw_category_id") or row.get("raw_id") or row.get("category_id") or "").strip()
        if not rid:
            continue
        old = out.get(rid)
        if old is None or "person" in str(row.get("variant") or "").lower():
            out[rid] = row
    return out


def residual_bucket(raw_id: str, lookup: Dict[str, Dict[str, Any]]) -> str:
    row = lookup.get(str(raw_id), {})
    if not bval(row.get("resolved")):
        return "unresolved"
    try:
        it = int(float(row.get("resolved_at_iteration")))
    except Exception:
        return "resolved_unknown_iteration"
    if it == 0:
        return "iter0_initial_anchor"
    if it == 1:
        return "iter1_first_peeling"
    if it >= 2:
        return "iter2plus_late_chain"
    return "resolved_unknown_iteration"


def margin_bucket(gap: float, small_th: float, large_th: float) -> str:
    if gap <= small_th:
        return f"small_le_{small_th:g}".replace(".", "p")
    if gap >= large_th:
        return f"large_ge_{large_th:g}".replace(".", "p")
    return f"middle_{small_th:g}_to_{large_th:g}".replace(".", "p")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--true_margin_row_csv", required=True)
    ap.add_argument("--residual_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--small_threshold", type=float, default=1.0)
    ap.add_argument("--large_threshold", type=float, default=3.0)
    args = ap.parse_args()

    row_csv = Path(args.true_margin_row_csv).resolve()
    resid_csv = Path(args.residual_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    residual_lookup = build_residual_lookup(resid_csv)
    rows = read_csv(row_csv)
    if not rows:
        raise RuntimeError(f"empty true margin rows: {row_csv}")

    bucket_count = Counter()
    bucket_gap_sum = Counter()
    absorber_count = Counter()
    absorber_source = defaultdict(set)
    pair_count = Counter()
    wrong_rows: List[Dict[str, Any]] = []

    for row in rows:
        gt = detect(row, ["gt_raw_id", "target_raw_id", "raw_id"])
        top1 = detect(row, ["top1_raw_id", "wrong_top1_raw_id", "pred_raw_id"])
        hit = inum(detect(row, ["top1_hit", "gt_top1_hit", "is_top1_gt"], "0"), 0)
        gap = fnum(detect(row, ["wrong_abs_gap", "margin_abs_gap"], "0"), 0.0)
        if hit == 1 or gap <= 0 or not gt or not top1:
            continue
        rb = residual_bucket(gt, residual_lookup)
        mb = margin_bucket(gap, float(args.small_threshold), float(args.large_threshold))
        out = dict(row)
        out["residual_bucket"] = rb
        out["margin_bucket"] = mb
        wrong_rows.append(out)
        bucket_count[(mb, rb)] += 1
        bucket_gap_sum[(mb, rb)] += gap
        absorber_count[(mb, rb, top1)] += 1
        absorber_source[(mb, rb, top1)].add(gt)
        gtn = detect(row, ["gt_class_name", "class_name", "target_class_name"])
        pair_count[(mb, rb, gt, gtn, top1)] += 1

    write_csv(out_dir / "wrong_rows_with_margin_and_residual.csv", wrong_rows)

    summary_rows = []
    for (mb, rb), n in sorted(bucket_count.items()):
        summary_rows.append({
            "margin_bucket": mb,
            "residual_bucket": rb,
            "wrong_rows": n,
            "mean_wrong_abs_gap": bucket_gap_sum[(mb, rb)] / max(n, 1),
        })
    write_csv(out_dir / "margin_bucket_x_residual_bucket.csv", summary_rows)

    pair_rows = []
    for (mb, rb, gt, gtn, pred), n in pair_count.items():
        pair_rows.append({
            "margin_bucket": mb,
            "residual_bucket": rb,
            "gt_raw_id": gt,
            "gt_class_name": gtn,
            "wrong_top1_raw_id": pred,
            "wrong_rows": n,
        })
    pair_rows.sort(key=lambda x: (x["margin_bucket"], x["residual_bucket"], -int(x["wrong_rows"])))
    write_csv(out_dir / "top_pairs_by_margin_residual.csv", pair_rows)

    absorber_rows = []
    for (mb, rb, pred), n in absorber_count.items():
        absorber_rows.append({
            "margin_bucket": mb,
            "residual_bucket": rb,
            "wrong_top1_raw_id": pred,
            "absorbed_wrong_rows": n,
            "source_class_count": len(absorber_source[(mb, rb, pred)]),
        })
    absorber_rows.sort(key=lambda x: (x["margin_bucket"], x["residual_bucket"], -int(x["absorbed_wrong_rows"])))
    write_csv(out_dir / "absorber_by_margin_residual.csv", absorber_rows)

    def get(mb: str, rb: str) -> int:
        return int(bucket_count.get((mb, rb), 0))

    small_name = margin_bucket(0.0, float(args.small_threshold), float(args.large_threshold))
    mid_name = margin_bucket((float(args.small_threshold) + float(args.large_threshold)) / 2, float(args.small_threshold), float(args.large_threshold))
    large_name = margin_bucket(float(args.large_threshold), float(args.small_threshold), float(args.large_threshold))

    compact = {
        "status": "PASS",
        "true_margin_row_csv": str(row_csv),
        "residual_csv": str(resid_csv),
        "total_wrong_rows": len(wrong_rows),
        "small_bucket_name": small_name,
        "middle_bucket_name": mid_name,
        "large_bucket_name": large_name,
        "small_total": sum(n for (mb, _), n in bucket_count.items() if mb == small_name),
        "middle_total": sum(n for (mb, _), n in bucket_count.items() if mb == mid_name),
        "large_total": sum(n for (mb, _), n in bucket_count.items() if mb == large_name),
        "small_iter0_plus_iter1": get(small_name, "iter0_initial_anchor") + get(small_name, "iter1_first_peeling"),
        "middle_iter0_plus_iter1": get(mid_name, "iter0_initial_anchor") + get(mid_name, "iter1_first_peeling"),
        "large_iter0_plus_iter1": get(large_name, "iter0_initial_anchor") + get(large_name, "iter1_first_peeling"),
        "iter2plus_total": sum(n for (_, rb), n in bucket_count.items() if rb == "iter2plus_late_chain"),
        "unresolved_total": sum(n for (_, rb), n in bucket_count.items() if rb == "unresolved"),
        "full_table": summary_rows,
    }
    compact["iter0_plus_iter1_total"] = int(compact["small_iter0_plus_iter1"] + compact["middle_iter0_plus_iter1"] + compact["large_iter0_plus_iter1"])
    compact["iter0_plus_iter1_rate"] = compact["iter0_plus_iter1_total"] / max(int(compact["total_wrong_rows"]), 1)
    (out_dir / "margin_bucket_iter01_summary.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
