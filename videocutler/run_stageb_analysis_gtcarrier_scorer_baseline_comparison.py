#!/usr/bin/env python3
"""Read-only scorer baseline comparison for GT-carrier full-Y latent assignment.

This script is intentionally conservative:
- It always records the existing V2-B/prealign D result as the reference backend.
- It runs an additional raw/direct cosine backend only when both GT-carrier vectors
  and mapped-text prototype vectors can be loaded with verified matching dimensions.
- It does not train, infer masks, update checkpoints, or modify model state.
- It does not hard-code any semantic class as a training prior; top labels are only
  data-driven audit outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def _as_int(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def _as_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _norm_key(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _safe_path(base: Optional[Path], p: Any) -> Optional[Path]:
    if not p:
        return None
    pp = Path(str(p))
    if pp.is_absolute():
        return pp
    if base is not None:
        return base / pp
    return pp


def _candidate_raw_id(record: Dict[str, Any]) -> Optional[int]:
    for k in (
        "raw_category_id",
        "raw_id",
        "category_id",
        "gt_raw_id",
        "matched_gt_raw_id",
        "gt_category_id",
        "class_raw_id",
        "id",
    ):
        v = _as_int(record.get(k))
        if v is not None:
            return v
    # Common text-prototype records sometimes carry names like {"label": {"raw_id": ...}}
    for k in ("label", "category", "class"):
        v = record.get(k)
        if isinstance(v, dict):
            rid = _candidate_raw_id(v)
            if rid is not None:
                return rid
    return None


def _vector_from_inline(record: Dict[str, Any]) -> Optional[np.ndarray]:
    for k in (
        "vector",
        "feature",
        "features",
        "embedding",
        "emb",
        "z",
        "z_raw",
        "z_norm",
        "prototype",
        "text_feature",
        "mapped_text_feature",
        "mapped_text_prototype",
        "visual_feature",
        "carrier",
    ):
        v = record.get(k)
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            return np.asarray(v, dtype=np.float32)
    return None


def _load_np_array(path: Path, key: Optional[str] = None, index: Optional[int] = None) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
        else:
            npz = np.load(path, mmap_mode="r")
            if key is None:
                if len(npz.files) == 1:
                    key = npz.files[0]
                elif "arr_0" in npz.files:
                    key = "arr_0"
                elif "vectors" in npz.files:
                    key = "vectors"
                elif "features" in npz.files:
                    key = "features"
                else:
                    key = npz.files[0]
            arr = npz[key]
        if index is not None and getattr(arr, "ndim", 0) >= 2:
            arr = arr[index]
        return np.asarray(arr, dtype=np.float32)
    except Exception:
        return None


def _vector_from_locator(record: Dict[str, Any], asset_root: Optional[Path]) -> Optional[np.ndarray]:
    inline = _vector_from_inline(record)
    if inline is not None:
        return inline

    # Nested locator dicts.
    locator_keys = (
        "vector_locator",
        "feature_locator",
        "z_locator",
        "locator",
        "embedding_locator",
        "payload_locator",
    )
    for lk in locator_keys:
        loc = record.get(lk)
        if not isinstance(loc, dict):
            continue
        p = loc.get("path") or loc.get("file") or loc.get("npz") or loc.get("npy") or loc.get("payload_path")
        key = loc.get("key") or loc.get("array_key") or loc.get("name") or loc.get("field")
        index = _as_int(loc.get("index") or loc.get("row") or loc.get("offset"))
        pp = _safe_path(asset_root, p)
        if pp is not None:
            arr = _load_np_array(pp, key=key, index=index)
            if arr is not None:
                return arr

    # Flat path fields.
    for pk in (
        "vector_path",
        "feature_path",
        "z_path",
        "npy_path",
        "npz_path",
        "payload_path",
        "file_path",
    ):
        p = record.get(pk)
        if not p:
            continue
        key = record.get("array_key") or record.get("key")
        index = _as_int(record.get("index") or record.get("row") or record.get("offset") or record.get("vector_index"))
        pp = _safe_path(asset_root, p)
        if pp is not None:
            arr = _load_np_array(pp, key=key, index=index)
            if arr is not None:
                return arr
    return None


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if not math.isfinite(n) or n <= 0:
        return x
    return x / n


def _load_full_y(full_y_path: Path) -> Dict[str, List[int]]:
    obj = _read_json(full_y_path)
    records = obj.get("records", []) if isinstance(obj, dict) else []
    out: Dict[str, List[int]] = {}
    for r in records:
        clip = r.get("clip_id")
        if clip is None:
            continue
        vals = r.get("full_y_raw_ids") or r.get("raw_ids") or r.get("labels") or []
        clean = sorted({int(x) for x in vals if _as_int(x) is not None})
        out[_norm_key(clip)] = clean
    return out


def _load_identity(binding_path: Path) -> Dict[str, Dict[str, Any]]:
    by_traj: Dict[str, Dict[str, Any]] = {}
    by_index: Dict[str, Dict[str, Any]] = {}
    for r in _iter_jsonl(binding_path) or []:
        tid = r.get("trajectory_id") or r.get("carrier_id")
        if tid is not None:
            by_traj[_norm_key(tid)] = r
        idx = r.get("carrier_row_index")
        if idx is not None:
            by_index[_norm_key(idx)] = r
    return {"by_traj": by_traj, "by_index": by_index}


def _load_text_vectors(text_records_path: Path, asset_root: Optional[Path]) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    vecs: Dict[int, np.ndarray] = {}
    sampled_keys: List[List[str]] = []
    fail = 0
    rows = 0
    for r in _iter_jsonl(text_records_path) or []:
        rows += 1
        if len(sampled_keys) < 5:
            sampled_keys.append(sorted(r.keys()))
        rid = _candidate_raw_id(r)
        if rid is None:
            fail += 1
            continue
        vec = _vector_from_locator(r, asset_root)
        if vec is None:
            fail += 1
            continue
        vecs[rid] = _l2norm(vec)
    dims = sorted({int(v.shape[0]) for v in vecs.values() if v is not None and v.ndim == 1})
    return vecs, {
        "path": str(text_records_path),
        "exists": text_records_path.exists(),
        "rows": rows,
        "loaded_vectors": len(vecs),
        "failed_rows": fail,
        "dims": dims,
        "sampled_key_sets": sampled_keys,
    }


def _find_text_records(run_root_v2b: Path, explicit: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend([
        run_root_v2b / "text_bank" / "text_prototype_records.jsonl",
        run_root_v2b / "text_bank" / "text_records.jsonl",
        run_root_v2b / "text_prototype_records.jsonl",
    ])
    for p in candidates:
        if p.exists():
            return p
    return None


def _baseline_metrics_from_summary(summary: Dict[str, Any], backend_name: str) -> Dict[str, Any]:
    return {
        "backend": backend_name,
        "status": "REFERENCE_EXISTING",
        "gt_rank1_rate": summary.get("gt_rank1_rate"),
        "gt_top5_rate": summary.get("gt_top5_rate"),
        "gt_top20_rate": summary.get("gt_top20_rate"),
        "mean_gt_rank": summary.get("mean_gt_rank"),
        "median_gt_rank": summary.get("median_gt_rank"),
        "mean_normalized_gt_rank": summary.get("mean_normalized_gt_rank"),
        "mean_gt_margin_vs_best_non_gt": summary.get("mean_gt_margin_vs_best_non_gt"),
        "positive_gt_margin_rate": summary.get("positive_gt_margin_rate"),
        "top1_wrong_rate": summary.get("top1_wrong_rate"),
        "wrong_large_negative_margin_rate": summary.get("wrong_large_negative_margin_rate"),
        "wrong_near_tie_margin_rate": summary.get("wrong_near_tie_margin_rate"),
        "hub_like_top1_concentration": (summary.get("top1_top_label") or {}).get("top1_rate_over_evaluable")
        or summary.get("hub_like_top1_concentration"),
        "wrong_top1_top_raw_id": (summary.get("wrong_top1_top_label") or {}).get("raw_category_id"),
        "wrong_top1_top_rate_over_wrong": (summary.get("wrong_top1_top_label") or {}).get("wrong_top1_rate_over_wrong"),
        "source": "existing_D_full_failure_summary",
    }


def _score_raw_direct(
    *,
    gt_carrier_path: Path,
    gt_identity_path: Path,
    full_y: Dict[str, List[int]],
    text_vecs: Dict[int, np.ndarray],
    carrier_asset_root: Optional[Path],
    output_rows_path: Optional[Path],
    topk: int,
    max_rows: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    identity = _load_identity(gt_identity_path)
    text_dim_set = sorted({int(v.shape[0]) for v in text_vecs.values() if v is not None})

    ranks: List[int] = []
    margins: List[float] = []
    candidate_counts: List[int] = []
    wrong_counter: Counter[int] = Counter()
    top1_counter: Counter[int] = Counter()
    confusion_counter: Counter[Tuple[int, int]] = Counter()
    by_class: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "n": 0,
        "rank1": 0,
        "top5": 0,
        "top20": 0,
        "rank_sum": 0.0,
        "norm_rank_sum": 0.0,
        "margin_sum": 0.0,
        "wrong_top1": Counter(),
    })
    candidate_size_bins: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, "rank1": 0, "wrong": 0, "margin_sum": 0.0})

    total = 0
    with_label = 0
    in_scope = 0
    evaluable = 0
    carrier_fail = 0
    missing_text = 0
    malformed = 0
    dim_mismatch = 0

    out_f = None
    if output_rows_path is not None:
        output_rows_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = output_rows_path.open("w", encoding="utf-8")

    try:
        for row_idx, carrier in enumerate(_iter_jsonl(gt_carrier_path) or []):
            total += 1
            if max_rows is not None and total > max_rows:
                break
            tid = carrier.get("trajectory_id") or carrier.get("carrier_id")
            ident = None
            if tid is not None:
                ident = identity["by_traj"].get(_norm_key(tid))
            if ident is None:
                ident = identity["by_index"].get(_norm_key(row_idx))
            if not ident:
                malformed += 1
                continue
            gt = _as_int(ident.get("raw_category_id") or ident.get("raw_id") or ident.get("gt_raw_id"))
            clip = ident.get("clip_id") if ident.get("clip_id") is not None else carrier.get("clip_id")
            if gt is None or clip is None:
                malformed += 1
                continue
            labels = full_y.get(_norm_key(clip))
            if labels is None:
                continue
            with_label += 1
            if gt not in labels:
                continue
            in_scope += 1
            candidate_ids = [rid for rid in labels if rid in text_vecs]
            if gt not in candidate_ids:
                missing_text += 1
                continue
            z = _vector_from_locator(carrier, carrier_asset_root)
            if z is None:
                carrier_fail += 1
                continue
            z = _l2norm(z)
            if text_dim_set and int(z.shape[0]) not in text_dim_set:
                dim_mismatch += 1
                continue
            mat = np.stack([text_vecs[rid] for rid in candidate_ids], axis=0)
            scores = mat @ z.reshape(-1)
            order = np.argsort(-scores)
            ranked_ids = [candidate_ids[int(i)] for i in order]
            gt_pos = ranked_ids.index(gt)
            rank = gt_pos + 1
            top1 = ranked_ids[0]
            score_gt = float(scores[candidate_ids.index(gt)])
            if top1 == gt and len(order) > 1:
                best_non_gt_score = float(scores[int(order[1])])
            elif top1 != gt:
                best_non_gt_score = float(scores[int(order[0])])
            else:
                best_non_gt_score = float("nan")
            margin = score_gt - best_non_gt_score if math.isfinite(best_non_gt_score) else float("nan")

            evaluable += 1
            ranks.append(rank)
            if math.isfinite(margin):
                margins.append(margin)
            candidate_counts.append(len(candidate_ids))
            top1_counter[int(top1)] += 1
            if top1 != gt:
                wrong_counter[int(top1)] += 1
                confusion_counter[(int(gt), int(top1))] += 1

            bc = by_class[int(gt)]
            bc["n"] += 1
            bc["rank1"] += int(rank == 1)
            bc["top5"] += int(rank <= 5)
            bc["top20"] += int(rank <= 20)
            bc["rank_sum"] += rank
            bc["norm_rank_sum"] += (rank - 1) / max(len(candidate_ids) - 1, 1)
            if math.isfinite(margin):
                bc["margin_sum"] += margin
            if top1 != gt:
                bc["wrong_top1"][int(top1)] += 1

            cn = len(candidate_ids)
            if cn <= 3:
                cb = "01_<=3"
            elif cn <= 5:
                cb = "02_4-5"
            elif cn <= 10:
                cb = "03_6-10"
            elif cn <= 20:
                cb = "04_11-20"
            else:
                cb = "05_>20"
            co = candidate_size_bins[cb]
            co["n"] += 1
            co["rank1"] += int(rank == 1)
            co["wrong"] += int(rank != 1)
            if math.isfinite(margin):
                co["margin_sum"] += margin

            if out_f is not None:
                compact = {
                    "row_index": row_idx,
                    "trajectory_id": tid,
                    "clip_id": clip,
                    "gt_raw_id": gt,
                    "candidate_label_count": len(candidate_ids),
                    "top1_raw_id": int(top1),
                    "gt_rank": rank,
                    "score_gt": score_gt,
                    "score_top1": float(scores[int(order[0])]),
                    "margin_gt_vs_best_non_gt": margin,
                    "topk_raw_ids": [int(x) for x in ranked_ids[:topk]],
                    "topk_scores": [float(scores[int(i)]) for i in order[:topk]],
                }
                out_f.write(json.dumps(compact, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()

    if evaluable == 0:
        return {
            "backend": "raw_direct_text_carrier_cosine",
            "status": "FAIL_NO_EVALUABLE_ROWS",
            "row_count_total": total,
            "row_count_with_label_record": with_label,
            "row_count_in_scope": in_scope,
            "latent_evaluable_row_count": 0,
            "carrier_load_fail_count": carrier_fail,
            "candidate_missing_vocab_count": missing_text,
            "malformed_count": malformed,
            "dim_mismatch_count": dim_mismatch,
        }, [], [], []

    rank_arr = np.asarray(ranks, dtype=np.float64)
    margin_arr = np.asarray(margins, dtype=np.float64) if margins else np.asarray([], dtype=np.float64)
    top1_label, top1_count = top1_counter.most_common(1)[0] if top1_counter else (None, 0)
    wrong_label, wrong_count = wrong_counter.most_common(1)[0] if wrong_counter else (None, 0)
    wrong_total = sum(wrong_counter.values())
    rank1_count = int(np.sum(rank_arr == 1))
    top5_count = int(np.sum(rank_arr <= 5))
    top20_count = int(np.sum(rank_arr <= 20))
    wrong_margins = margin_arr[margin_arr < 0] if margin_arr.size else np.asarray([], dtype=np.float64)

    metrics = {
        "backend": "raw_direct_text_carrier_cosine",
        "status": "PASS_SCORER",
        "row_count_total": total,
        "row_count_with_label_record": with_label,
        "row_count_in_scope": in_scope,
        "latent_evaluable_row_count": evaluable,
        "candidate_contains_gt_rate": 1.0,
        "gt_rank1_rate": rank1_count / evaluable,
        "gt_top5_rate": top5_count / evaluable,
        "gt_top20_rate": top20_count / evaluable,
        "top1_wrong_rate": 1.0 - (rank1_count / evaluable),
        "mean_gt_rank": float(np.mean(rank_arr)),
        "median_gt_rank": float(np.median(rank_arr)),
        "mean_normalized_gt_rank": float(np.mean([(r - 1) / max(c - 1, 1) for r, c in zip(ranks, candidate_counts)])),
        "mean_gt_margin_vs_best_non_gt": float(np.mean(margin_arr)) if margin_arr.size else None,
        "positive_gt_margin_rate": float(np.mean(margin_arr > 0)) if margin_arr.size else None,
        "wrong_large_negative_margin_rate": float(np.mean(wrong_margins < -0.2)) if wrong_margins.size else None,
        "wrong_near_tie_margin_rate": float(np.mean((wrong_margins >= -0.05) & (wrong_margins < 0))) if wrong_margins.size else None,
        "hub_like_top1_concentration": top1_count / evaluable if evaluable else None,
        "wrong_top1_top_raw_id": wrong_label,
        "wrong_top1_top_rate_over_wrong": wrong_count / wrong_total if wrong_total else None,
        "carrier_load_fail_count": carrier_fail,
        "candidate_missing_vocab_count": missing_text,
        "malformed_count": malformed,
        "dim_mismatch_count": dim_mismatch,
        "top1_top_label": {"raw_category_id": top1_label, "count": top1_count, "rate_over_evaluable": top1_count / evaluable},
        "wrong_top1_top_label": {"raw_category_id": wrong_label, "count": wrong_count, "rate_over_wrong": wrong_count / wrong_total if wrong_total else None},
    }

    wrong_rows = [
        {"raw_category_id": rid, "wrong_top1_count": cnt, "wrong_top1_rate_over_wrong": cnt / wrong_total if wrong_total else None}
        for rid, cnt in wrong_counter.most_common(50)
    ]
    conf_rows = [
        {"gt_raw_id": gt, "wrong_top1_raw_id": pred, "count": cnt, "rate_over_wrong": cnt / wrong_total if wrong_total else None}
        for (gt, pred), cnt in confusion_counter.most_common(200)
    ]
    class_rows: List[Dict[str, Any]] = []
    for rid, o in by_class.items():
        n = int(o["n"])
        most_wrong, most_wrong_count = o["wrong_top1"].most_common(1)[0] if o["wrong_top1"] else (None, 0)
        class_rows.append({
            "raw_category_id": rid,
            "row_count": n,
            "gt_rank1_rate": o["rank1"] / n,
            "gt_top5_rate": o["top5"] / n,
            "gt_top20_rate": o["top20"] / n,
            "mean_gt_rank": o["rank_sum"] / n,
            "mean_normalized_gt_rank": o["norm_rank_sum"] / n,
            "mean_margin": o["margin_sum"] / n,
            "most_common_wrong_top1_raw_id": most_wrong,
            "most_common_wrong_top1_count": most_wrong_count,
        })
    class_rows.sort(key=lambda r: (r["gt_rank1_rate"], -r["row_count"]))
    return metrics, wrong_rows, conf_rows, class_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root_v2b", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--full_y_path", required=True)
    ap.add_argument("--gt_carrier_path", required=True)
    ap.add_argument("--gt_identity_path", required=True)
    ap.add_argument("--text_records_path", default=None)
    ap.add_argument("--carrier_asset_root", default="/home/zyy/code/wsovvis_asserts")
    ap.add_argument("--text_asset_root", default=None)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--write_raw_rows", action="store_true")
    args = ap.parse_args()

    run_root_v2b = Path(args.run_root_v2b)
    output_root = Path(args.output_root)
    out = output_root / "scorer_baseline_comparison"
    out.mkdir(parents=True, exist_ok=True)

    d_summary_path = output_root / "d_failure_decomposition_full" / "D_full_failure_summary.json"
    if not d_summary_path.exists():
        d_summary_path = output_root / "gtcarrier_latent_assignment" / "D_fully_gtcarrier_latent_summary.json"
    d_summary = _read_json(d_summary_path)
    if not d_summary:
        print(f"Missing D summary: {d_summary_path}", file=sys.stderr)
        sys.exit(2)

    readiness: Dict[str, Any] = {
        "status": "PASS",
        "run_root_v2b": str(run_root_v2b),
        "output_root": str(output_root),
        "reference_d_summary_path": str(d_summary_path),
        "backends": {},
    }

    comparison: List[Dict[str, Any]] = []
    v2b_metrics = _baseline_metrics_from_summary(d_summary, "v2b_prealign_checkpoint")
    comparison.append(v2b_metrics)
    readiness["backends"]["v2b_prealign_checkpoint"] = {
        "status": "REFERENCE_EXISTING",
        "reason": "Existing D full row-level summary is used as the V2-B baseline.",
        "summary_path": str(d_summary_path),
    }

    # Check epoch0/initial checkpoints only for readiness; do not score without a verified adapter.
    ckpt_dir = run_root_v2b / "train" / "prealign" / "checkpoints"
    initial_candidates = []
    if ckpt_dir.exists():
        for p in ckpt_dir.glob("*.pth"):
            name = p.name.lower()
            if any(tok in name for tok in ("epoch0", "epoch_0", "initial", "init", "pretrain", "step0")):
                initial_candidates.append(str(p))
    readiness["backends"]["initial_or_epoch0_checkpoint"] = {
        "status": "PARTIAL_NO_SAFE_SCORING_ADAPTER" if initial_candidates else "NOT_AVAILABLE",
        "candidate_paths": initial_candidates,
        "reason": "This overlay does not invent checkpoint scorer semantics. It only reports availability.",
    }

    # Raw direct cosine backend.
    text_path = _find_text_records(run_root_v2b, Path(args.text_records_path) if args.text_records_path else None)
    text_asset_root = Path(args.text_asset_root) if args.text_asset_root else (text_path.parent if text_path else None)
    if text_path is None:
        readiness["backends"]["raw_direct_text_carrier_cosine"] = {
            "status": "NOT_AVAILABLE",
            "reason": "No text prototype records path found.",
        }
    else:
        text_vecs, text_meta = _load_text_vectors(text_path, text_asset_root)
        readiness["backends"]["raw_direct_text_carrier_cosine"] = {
            "status": "READINESS_CHECKED",
            "text_meta": text_meta,
            "text_asset_root": str(text_asset_root) if text_asset_root else None,
        }
        if not text_vecs:
            readiness["backends"]["raw_direct_text_carrier_cosine"]["status"] = "NOT_AVAILABLE"
            readiness["backends"]["raw_direct_text_carrier_cosine"]["reason"] = "Text records found but no vectors could be loaded."
        else:
            full_y = _load_full_y(Path(args.full_y_path))
            raw_rows_path = out / "scorer_baseline_rows_raw_direct_text_carrier_cosine.jsonl" if args.write_raw_rows else None
            metrics, wrong_rows, conf_rows, class_rows = _score_raw_direct(
                gt_carrier_path=Path(args.gt_carrier_path),
                gt_identity_path=Path(args.gt_identity_path),
                full_y=full_y,
                text_vecs=text_vecs,
                carrier_asset_root=Path(args.carrier_asset_root) if args.carrier_asset_root else None,
                output_rows_path=raw_rows_path,
                topk=args.topk,
                max_rows=args.max_rows if args.max_rows > 0 else None,
            )
            comparison.append(metrics)
            readiness["backends"]["raw_direct_text_carrier_cosine"].update({
                "status": metrics.get("status"),
                "metrics_summary": {k: metrics.get(k) for k in (
                    "gt_rank1_rate", "gt_top5_rate", "mean_gt_margin_vs_best_non_gt",
                    "wrong_large_negative_margin_rate", "hub_like_top1_concentration",
                    "latent_evaluable_row_count", "carrier_load_fail_count", "dim_mismatch_count",
                )},
                "rows_path": str(raw_rows_path) if raw_rows_path else None,
            })
            _write_csv(out / "raw_direct_wrong_top1_concentration.csv", wrong_rows)
            _write_csv(out / "raw_direct_confusion_pairs.csv", conf_rows)
            _write_csv(out / "raw_direct_by_class.csv", class_rows)

    # Delta fields against V2-B.
    v2b_rank1 = _as_float(v2b_metrics.get("gt_rank1_rate"))
    v2b_margin = _as_float(v2b_metrics.get("mean_gt_margin_vs_best_non_gt"))
    v2b_hub = _as_float(v2b_metrics.get("hub_like_top1_concentration"))
    v2b_large = _as_float(v2b_metrics.get("wrong_large_negative_margin_rate"))
    for r in comparison:
        if r["backend"] == "v2b_prealign_checkpoint":
            r["delta_rank1_vs_v2b"] = 0.0
            r["delta_margin_vs_v2b"] = 0.0
            r["delta_hub_concentration_vs_v2b"] = 0.0
            r["delta_large_wrong_margin_vs_v2b"] = 0.0
        else:
            r_rank1 = _as_float(r.get("gt_rank1_rate"))
            r_margin = _as_float(r.get("mean_gt_margin_vs_best_non_gt"))
            r_hub = _as_float(r.get("hub_like_top1_concentration"))
            r_large = _as_float(r.get("wrong_large_negative_margin_rate"))
            r["delta_rank1_vs_v2b"] = (r_rank1 - v2b_rank1) if r_rank1 is not None and v2b_rank1 is not None else None
            r["delta_margin_vs_v2b"] = (r_margin - v2b_margin) if r_margin is not None and v2b_margin is not None else None
            r["delta_hub_concentration_vs_v2b"] = (r_hub - v2b_hub) if r_hub is not None and v2b_hub is not None else None
            r["delta_large_wrong_margin_vs_v2b"] = (r_large - v2b_large) if r_large is not None and v2b_large is not None else None

    _write_json(out / "scorer_backend_readiness.json", readiness)
    _write_csv(out / "scorer_baseline_comparison.csv", comparison)

    summary = {
        "status": "PASS",
        "audit_name": "gtcarrier_scorer_baseline_comparison",
        "reference_backend": "v2b_prealign_checkpoint",
        "output_root": str(out),
        "readiness": readiness,
        "comparison": comparison,
        "interpretation": {
            "rule": "If raw/direct is available and V2-B improves rank1 with lower hub concentration, training helps; if V2-B is worse, prealign may amplify shortcuts; if raw/direct is also poor, global trajectory-text evidence is insufficient.",
            "guardrail": "Unavailable raw/untrained backends are reported as unavailable rather than guessed.",
        },
    }
    _write_json(out / "scorer_baseline_comparison_summary.json", summary)

    md: List[str] = []
    md.append("# GT-Carrier Scorer Baseline Comparison")
    md.append("")
    md.append("Status: PASS")
    md.append("")
    md.append("Scope:")
    md.append("- Read-only scorer baseline readiness/comparison on D = full-Y + GT carrier.")
    md.append("- V2-B prealign checkpoint is used as the existing reference result.")
    md.append("- Raw/direct cosine is executed only if vectors are verifiably loadable and dimension-compatible.")
    md.append("- No training, inference, or checkpoint writes.")
    md.append("")
    md.append("## Backend readiness")
    for name, meta in readiness["backends"].items():
        md.append(f"- {name}: `{meta.get('status')}`")
        if meta.get("reason"):
            md.append(f"  - reason: {meta.get('reason')}")
    md.append("")
    md.append("## Comparison")
    for r in comparison:
        md.append(f"### {r.get('backend')}")
        for k in (
            "status", "gt_rank1_rate", "gt_top5_rate", "mean_gt_margin_vs_best_non_gt",
            "wrong_large_negative_margin_rate", "hub_like_top1_concentration", "delta_rank1_vs_v2b",
            "delta_hub_concentration_vs_v2b",
        ):
            md.append(f"- {k}: `{r.get(k)}`")
        md.append("")
    md.append("## Outputs")
    md.append(f"- readiness: `{out / 'scorer_backend_readiness.json'}`")
    md.append(f"- comparison_csv: `{out / 'scorer_baseline_comparison.csv'}`")
    md.append(f"- summary: `{out / 'scorer_baseline_comparison_summary.json'}`")
    md.append("")
    (out / "SCORER_BASELINE_COMPARISON_TAKEOVER.md").write_text("\n".join(md), encoding="utf-8")

    # Also update TAKEOVER if requested by caller? We do not overwrite codex/control here by default.
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(out),
        "backends": {k: v.get("status") for k, v in readiness["backends"].items()},
        "comparison": [
            {"backend": r.get("backend"), "status": r.get("status"), "gt_rank1_rate": r.get("gt_rank1_rate"), "delta_rank1_vs_v2b": r.get("delta_rank1_vs_v2b")}
            for r in comparison
        ],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
