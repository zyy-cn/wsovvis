#!/usr/bin/env python3
"""Build diagnostic A8 Boundary-Oracle Probe matched pairs.

This tool is read-only with respect to existing experiment artifacts. It creates a
new matched-pairs CSV for a diagnostic oracle boundary probe:

- select CE-5ep wrong rows with wrong_abs_gap >= threshold
- keep only residual-peeling iteration 0/1 classes by default
- intersect with the existing Hungarian matched-pairs CSV
- copy the original matched-pairs rows, but replace the pseudo-label column with
  the GT raw id, and store the original pseudo label + wrong top1 as audit fields

The generated CSV can be fed into the existing
videocutler/run_stageb_train_residual_gated_hungarian_matched.py as an oracle
CE boundary probe. This is diagnostic only and must not be reported as a clean
method.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
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
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def first_present(row: Dict[str, str], names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value) != "":
            return str(value)
    return default


def detect_column(rows: List[Dict[str, str]], candidates: Iterable[str], *, required: bool, label: str) -> Optional[str]:
    fields = set(rows[0].keys()) if rows else set()
    for cand in candidates:
        if cand in fields:
            return cand
    if required:
        raise RuntimeError(f"Could not detect {label}; available columns={sorted(fields)}; candidates={list(candidates)}")
    return None


def row_key(row: Dict[str, str], *, clip_col: Optional[str], tid_col: Optional[str]) -> Optional[Tuple[str, str]]:
    tid = str(row.get(tid_col or "", "")).strip() if tid_col else ""
    clip = str(row.get(clip_col or "", "")).strip() if clip_col else ""
    if tid and clip:
        return (clip, tid)
    if tid:
        return ("", tid)
    return None


def build_residual_lookup(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_csv(path):
        rid = str(row.get("raw_category_id") or row.get("raw_id") or row.get("category_id") or "").strip()
        if not rid:
            continue
        old = out.get(rid)
        # Prefer person-aware variant when duplicated.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--true_margin_row_csv", required=True)
    ap.add_argument("--residual_csv", required=True)
    ap.add_argument("--matched_pairs_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_wrong_abs_gap", type=float, default=1.0)
    ap.add_argument("--include_residual_buckets", default="iter0_initial_anchor,iter1_first_peeling")
    ap.add_argument("--target_raw_id_column", default="", help="Override target pseudo-label column in matched_pairs_csv.")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap after deterministic sorting; 0 means no cap.")
    args = ap.parse_args()

    true_margin_path = Path(args.true_margin_row_csv).resolve()
    residual_path = Path(args.residual_csv).resolve()
    matched_path = Path(args.matched_pairs_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tm_rows = read_csv(true_margin_path)
    matched_rows = read_csv(matched_path)
    if not tm_rows:
        raise RuntimeError(f"empty true margin row csv: {true_margin_path}")
    if not matched_rows:
        raise RuntimeError(f"empty matched pairs csv: {matched_path}")

    tm_clip_col = detect_column(tm_rows, ["clip_id", "video_clip_id", "clip"], required=False, label="true-margin clip id")
    tm_tid_col = detect_column(tm_rows, ["trajectory_id", "track_id", "tid", "carrier_id"], required=True, label="true-margin trajectory id")
    mp_clip_col = detect_column(matched_rows, ["clip_id", "video_clip_id", "clip"], required=False, label="matched-pairs clip id")
    mp_tid_col = detect_column(matched_rows, ["trajectory_id", "track_id", "tid", "carrier_id"], required=True, label="matched-pairs trajectory id")

    if args.target_raw_id_column:
        target_col = args.target_raw_id_column
        if target_col not in matched_rows[0]:
            raise RuntimeError(f"requested target column {target_col!r} not in matched CSV columns: {list(matched_rows[0].keys())}")
    else:
        target_col = detect_column(
            matched_rows,
            [
                "pseudo_raw_id", "assigned_raw_id", "matched_raw_id", "target_raw_id",
                "raw_category_id", "raw_id", "category_id", "class_raw_id", "matched_category_id",
            ],
            required=True,
            label="matched-pairs pseudo label raw id",
        ) or ""

    gt_col = detect_column(tm_rows, ["gt_raw_id", "target_raw_id", "raw_id"], required=True, label="true-margin gt raw id")
    top1_col = detect_column(tm_rows, ["top1_raw_id", "wrong_top1_raw_id", "pred_raw_id"], required=True, label="true-margin top1 raw id")
    hit_col = detect_column(tm_rows, ["top1_hit", "gt_top1_hit", "is_top1_gt"], required=False, label="true-margin top1 hit")
    gap_col = detect_column(tm_rows, ["wrong_abs_gap", "margin_abs_gap"], required=True, label="true-margin wrong gap")
    class_col = detect_column(tm_rows, ["gt_class_name", "class_name", "target_class_name"], required=False, label="true-margin class name")

    residual_lookup = build_residual_lookup(residual_path)
    include_buckets = {x.strip() for x in str(args.include_residual_buckets).split(",") if x.strip()}

    selected_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    selection_counters = Counter()
    for row in tm_rows:
        key = row_key(row, clip_col=tm_clip_col, tid_col=tm_tid_col)
        if key is None:
            selection_counters["no_key"] += 1
            continue
        hit = inum(row.get(hit_col or ""), default=0) if hit_col else 0
        gap = fnum(row.get(gap_col), default=0.0)
        gt = str(row.get(gt_col, "")).strip()
        top1 = str(row.get(top1_col, "")).strip()
        if not gt or not top1:
            selection_counters["missing_gt_or_top1"] += 1
            continue
        if hit == 1 or gap <= 0:
            selection_counters["correct_or_nonwrong"] += 1
            continue
        if gap < float(args.min_wrong_abs_gap):
            selection_counters["below_gap_threshold"] += 1
            continue
        bucket = residual_bucket(gt, residual_lookup)
        if bucket not in include_buckets:
            selection_counters[f"excluded_bucket:{bucket}"] += 1
            continue
        enriched = dict(row)
        enriched["oracle_gt_raw_id"] = gt
        enriched["oracle_negative_raw_id"] = top1
        enriched["oracle_wrong_abs_gap"] = gap
        enriched["oracle_residual_bucket"] = bucket
        enriched["oracle_gt_class_name"] = row.get(class_col or "", "")
        # If duplicates exist, keep the harder one.
        old = selected_by_key.get(key)
        if old is None or gap > fnum(old.get("oracle_wrong_abs_gap"), default=0.0):
            selected_by_key[key] = enriched
        selection_counters["selected_before_matched_intersection"] += 1

    selected_keys = set(selected_by_key.keys())
    output_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    matched_intersection_counters = Counter()

    for mrow in matched_rows:
        key = row_key(mrow, clip_col=mp_clip_col, tid_col=mp_tid_col)
        if key not in selected_keys:
            continue
        sel = selected_by_key[key]
        out = dict(mrow)
        original_target = out.get(target_col, "")
        out[target_col] = str(sel["oracle_gt_raw_id"])
        out["oracle_boundary_enabled"] = "1"
        out["oracle_boundary_original_target_raw_id"] = original_target
        out["oracle_boundary_positive_raw_id"] = str(sel["oracle_gt_raw_id"])
        out["oracle_boundary_negative_raw_id"] = str(sel["oracle_negative_raw_id"])
        out["oracle_boundary_wrong_abs_gap"] = str(sel["oracle_wrong_abs_gap"])
        out["oracle_boundary_residual_bucket"] = str(sel["oracle_residual_bucket"])
        out["oracle_boundary_gt_class_name"] = str(sel.get("oracle_gt_class_name", ""))
        output_rows.append(out)
        selected_rows.append(sel)
        matched_intersection_counters["matched_selected"] += 1

    if int(args.max_rows) > 0 and len(output_rows) > int(args.max_rows):
        # Deterministic hard-first cap.
        rows_with_sel = list(zip(output_rows, selected_rows))
        rows_with_sel.sort(key=lambda p: (-fnum(p[1].get("oracle_wrong_abs_gap"), 0.0), str(p[0].get(mp_clip_col or "", "")), str(p[0].get(mp_tid_col or "", ""))))
        rows_with_sel = rows_with_sel[: int(args.max_rows)]
        output_rows = [p[0] for p in rows_with_sel]
        selected_rows = [p[1] for p in rows_with_sel]
        matched_intersection_counters["capped_to_max_rows"] = int(args.max_rows)

    if not output_rows:
        raise RuntimeError(
            "No rows selected after intersecting true-margin errors with matched_pairs_csv. "
            "Check trajectory/clip id columns or lower --min_wrong_abs_gap."
        )

    # Preserve original matched-pair header first, append oracle columns.
    fieldnames = list(matched_rows[0].keys())
    for col in [
        "oracle_boundary_enabled",
        "oracle_boundary_original_target_raw_id",
        "oracle_boundary_positive_raw_id",
        "oracle_boundary_negative_raw_id",
        "oracle_boundary_wrong_abs_gap",
        "oracle_boundary_residual_bucket",
        "oracle_boundary_gt_class_name",
    ]:
        if col not in fieldnames:
            fieldnames.append(col)

    matched_out = out_dir / "oracle_boundary_matched_pairs.csv"
    selected_out = out_dir / "oracle_boundary_selected_true_margin_rows.csv"
    write_csv(matched_out, output_rows, fieldnames=fieldnames)
    write_csv(selected_out, selected_rows)

    gap_values = [fnum(r.get("oracle_boundary_wrong_abs_gap"), 0.0) for r in output_rows]
    manifest = {
        "status": "PASS",
        "true_margin_row_csv": str(true_margin_path),
        "residual_csv": str(residual_path),
        "matched_pairs_csv": str(matched_path),
        "output_matched_pairs_csv": str(matched_out),
        "output_selected_true_margin_rows_csv": str(selected_out),
        "target_raw_id_column": target_col,
        "min_wrong_abs_gap": float(args.min_wrong_abs_gap),
        "include_residual_buckets": sorted(include_buckets),
        "selection_counters": dict(selection_counters),
        "matched_intersection_counters": dict(matched_intersection_counters),
        "selected_after_matched_intersection": len(output_rows),
        "gap_mean": sum(gap_values) / max(len(gap_values), 1),
        "gap_min": min(gap_values) if gap_values else None,
        "gap_max": max(gap_values) if gap_values else None,
        "diagnostic_semantics": {
            "uses_train_side_gt_for_positive": True,
            "uses_wrong_top1_as_oracle_negative_metadata": True,
            "clean_method_candidate": False,
            "purpose": "diagnose whether explicit GT boundary supervision can repair middle/large iter0/1 errors",
        },
    }
    (out_dir / "oracle_boundary_probe_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
