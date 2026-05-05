#!/usr/bin/env python3
"""
A8 semantic-boundary bottleneck audit.

Read-only reducer for existing A8 rank@K per-row artifacts. It does not train,
modify checkpoints, or write control-plane files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

PERSON_RAW_ID_DEFAULT = "773"


def _norm_id(x: Any) -> str:
    try:
        if x is None:
            return ""
        s = str(x).strip()
        if s == "":
            return ""
        return str(int(float(s)))
    except Exception:
        return str(x).strip()


def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _to_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f"}:
        return False
    return None


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = "") -> Any:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return default


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
        fieldnames = fields or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mean(xs: Sequence[float], default: Optional[float] = None) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not vals:
        return default
    return float(sum(vals) / len(vals))


def _median(xs: Sequence[float], default: Optional[float] = None) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not vals:
        return default
    return float(statistics.median(vals))


class RankRow(dict):
    pass


def _parse_rank_rows(raw_rows: Sequence[Mapping[str, Any]], source_tag: str) -> List[RankRow]:
    rank_names = ["restricted_rank", "gt_rank", "rank", "rank_of_gt", "gt_rank_in_candidates", "visible525_gt_rank"]
    gt_id_names = ["gt_raw_id", "raw_id", "category_id", "gt_class_raw_id", "matched_gt_raw_id_canonical"]
    gt_name_names = ["gt_name", "gt_class_name", "class_name", "category_name"]
    top1_id_names = ["top1_raw_id", "pred_raw_id", "top1_class_raw_id", "top1_id"]
    top1_name_names = ["top1_name", "pred_name", "top1_class_name"]
    top2_id_names = ["top2_raw_id", "top2_id", "second_raw_id", "second_class_raw_id"]
    top2_name_names = ["top2_name", "top2_class_name", "second_name"]
    gt_score_names = ["gt_score", "score_gt", "gt_logit"]
    top1_score_names = ["top1_score", "pred_score", "top1_logit"]
    top2_score_names = ["top2_score", "top2_logit", "second_score"]
    margin_names = ["margin_gt_minus_top1", "gt_vs_top1_margin", "gt_margin", "margin", "gt_vs_pred_margin"]

    out: List[RankRow] = []
    for row in raw_rows:
        gt = _norm_id(_first(row, gt_id_names))
        if not gt:
            continue
        rank = _to_float(_first(row, rank_names, None))
        if rank is None:
            raise RuntimeError(f"Cannot find rank column for {source_tag}. Header={list(row.keys())}")
        cand = int(_to_float(_first(row, ["candidate_count"], 0), 0) or 0)
        if cand <= 0:
            cand = 525 if "visible" in source_tag.lower() else 641
        gt_score = _to_float(_first(row, gt_score_names, None))
        top1_score = _to_float(_first(row, top1_score_names, None))
        top2_score = _to_float(_first(row, top2_score_names, None))
        margin = _to_float(_first(row, margin_names, None))
        if margin is None and gt_score is not None and top1_score is not None:
            margin = gt_score - top1_score
        top1 = _norm_id(_first(row, top1_id_names, ""))
        top2 = _norm_id(_first(row, top2_id_names, ""))
        top1_vs_top2_margin = None
        if top1_score is not None and top2_score is not None:
            top1_vs_top2_margin = top1_score - top2_score
        gt_vs_top2_margin = None
        if gt_score is not None and top2_score is not None:
            gt_vs_top2_margin = gt_score - top2_score
        out.append(RankRow({
            "source_tag": source_tag,
            "dataset_name": _first(row, ["dataset_name"], ""),
            "trajectory_id": str(_first(row, ["trajectory_id", "row_id", "carrier_id"], "")),
            "video_id": str(_first(row, ["video_id"], "")),
            "clip_id": str(_first(row, ["clip_id", "video_id"], "")),
            "gt_raw_id": gt,
            "gt_name": str(_first(row, gt_name_names, "")),
            "top1_raw_id": top1,
            "top1_name": str(_first(row, top1_name_names, "")),
            "top2_raw_id": top2,
            "top2_name": str(_first(row, top2_name_names, "")),
            "rank": int(rank),
            "candidate_count": cand,
            "gt_score": gt_score,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "gt_vs_top1_margin": margin,
            "top1_vs_top2_margin": top1_vs_top2_margin,
            "gt_vs_top2_margin": gt_vs_top2_margin,
            "is_gt_top1": bool(rank <= 1),
            "is_gt_top5": bool(rank <= 5),
            "is_gt_top10": bool(rank <= 10),
            "is_gt_top20": bool(rank <= 20),
            "is_gt_top50": bool(rank <= 50),
            "has_top2_score": top2_score is not None,
        }))
    return out


def _class_metrics(rows: Sequence[RankRow], *, near_tie_abs_margin: float, large_negative_margin: float) -> Dict[str, Dict[str, Any]]:
    by: Dict[str, List[RankRow]] = defaultdict(list)
    for r in rows:
        by[r["gt_raw_id"]].append(r)
    metrics: Dict[str, Dict[str, Any]] = {}
    for rid, items in by.items():
        n = len(items)
        ranks = [int(r["rank"]) for r in items]
        errors = [r for r in items if int(r["rank"]) > 1]
        margins = [r["gt_vs_top1_margin"] for r in items if r.get("gt_vs_top1_margin") is not None]
        error_margins = [r["gt_vs_top1_margin"] for r in errors if r.get("gt_vs_top1_margin") is not None]
        near_tie_errors = [r for r in errors if r.get("gt_vs_top1_margin") is not None and abs(float(r["gt_vs_top1_margin"])) < near_tie_abs_margin]
        large_neg_errors = [r for r in errors if r.get("gt_vs_top1_margin") is not None and float(r["gt_vs_top1_margin"]) < large_negative_margin]
        top1_vs_top2_error_margins = [r["top1_vs_top2_margin"] for r in errors if r.get("top1_vs_top2_margin") is not None]
        suppressors = Counter(str(r.get("top1_raw_id") or "NA") for r in errors)
        metrics[rid] = {
            "raw_id": rid,
            "class_name": next((str(r.get("gt_name") or "") for r in items if r.get("gt_name")), ""),
            "count": n,
            "rank@1": sum(x <= 1 for x in ranks) / n,
            "rank@5": sum(x <= 5 for x in ranks) / n,
            "rank@10": sum(x <= 10 for x in ranks) / n,
            "rank@20": sum(x <= 20 for x in ranks) / n,
            "rank@50": sum(x <= 50 for x in ranks) / n,
            "mean_rank": _mean([float(x) for x in ranks]),
            "median_rank": _median([float(x) for x in ranks]),
            "mean_normalized_gt_rank": _mean([(float(x) - 1.0) / max(int(items[i]["candidate_count"]) - 1, 1) for i, x in enumerate(ranks)]),
            "error_count": len(errors),
            "error_rate": len(errors) / n,
            "mean_gt_vs_top1_margin": _mean([float(x) for x in margins], ""),
            "median_gt_vs_top1_margin": _median([float(x) for x in margins], ""),
            "mean_error_gt_vs_top1_margin": _mean([float(x) for x in error_margins], ""),
            "median_error_gt_vs_top1_margin": _median([float(x) for x in error_margins], ""),
            "near_tie_error_count": len(near_tie_errors),
            "near_tie_error_rate": len(near_tie_errors) / max(len(errors), 1),
            "large_negative_error_count": len(large_neg_errors),
            "large_negative_error_rate": len(large_neg_errors) / max(len(errors), 1),
            "mean_top1_vs_top2_margin_on_errors": _mean([float(x) for x in top1_vs_top2_error_margins], ""),
            "top_suppressor_raw_id": suppressors.most_common(1)[0][0] if suppressors else "",
            "top_suppressor_count": suppressors.most_common(1)[0][1] if suppressors else 0,
        }
    return metrics


def _support_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    if count <= 50:
        return "11-50"
    if count <= 200:
        return "51-200"
    return ">200"


def _quadrant(train_rank1: Optional[float], val_rank1: Optional[float]) -> str:
    if train_rank1 is None or val_rank1 is None:
        return "missing_train_or_val"
    if train_rank1 >= 0.8 and val_rank1 >= 0.7:
        return "learned_stable"
    if train_rank1 < 0.5 and val_rank1 < 0.5:
        return "underlearned"
    if train_rank1 >= 0.8 and val_rank1 < 0.5:
        return "overfit_context_fail"
    if train_rank1 < 0.5 and val_rank1 >= 0.5:
        return "train_weak_val_ok"
    return "mid_uncertain"


def _macro_mean(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    vals = []
    for r in rows:
        v = _to_float(r.get(key))
        if v is not None:
            vals.append(v)
    if not vals:
        return ""
    return str(sum(vals) / len(vals))


def _per_class_join(train_m: Dict[str, Dict[str, Any]], val_m: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rid in sorted(set(train_m) | set(val_m), key=lambda x: int(x) if x.isdigit() else x):
        tr = train_m.get(rid, {})
        va = val_m.get(rid, {})
        train_count = int(tr.get("count", 0) or 0)
        val_count = int(va.get("count", 0) or 0)
        train_rank1 = _to_float(tr.get("rank@1"))
        val_rank1 = _to_float(va.get("rank@1"))
        rows.append({
            "raw_id": rid,
            "class_name": tr.get("class_name") or va.get("class_name") or "",
            "support_bucket": _support_bucket(train_count),
            "quadrant": _quadrant(train_rank1, val_rank1),
            "train_count": train_count,
            "val_count": val_count,
            "train_rank@1": train_rank1 if train_rank1 is not None else "",
            "val_rank@1": val_rank1 if val_rank1 is not None else "",
            "train_rank@5": tr.get("rank@5", ""),
            "val_rank@5": va.get("rank@5", ""),
            "train_mean_rank": tr.get("mean_rank", ""),
            "val_mean_rank": va.get("mean_rank", ""),
            "train_val_rank1_gap": (train_rank1 - val_rank1) if train_rank1 is not None and val_rank1 is not None else "",
            "train_val_rank5_gap": (_to_float(tr.get("rank@5")) - _to_float(va.get("rank@5"))) if _to_float(tr.get("rank@5")) is not None and _to_float(va.get("rank@5")) is not None else "",
            "val_error_count": va.get("error_count", ""),
            "val_large_negative_error_count": va.get("large_negative_error_count", ""),
            "val_large_negative_error_rate": va.get("large_negative_error_rate", ""),
            "val_near_tie_error_count": va.get("near_tie_error_count", ""),
            "val_near_tie_error_rate": va.get("near_tie_error_rate", ""),
            "val_mean_error_gt_vs_top1_margin": va.get("mean_error_gt_vs_top1_margin", ""),
            "val_top_suppressor_raw_id": va.get("top_suppressor_raw_id", ""),
            "val_top_suppressor_count": va.get("top_suppressor_count", ""),
        })
    return rows


def _support_bucket_summary(join_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in join_rows:
        by[str(r.get("support_bucket", ""))].append(r)
    order = ["0", "1-2", "3-5", "6-10", "11-50", "51-200", ">200"]
    rows: List[Dict[str, Any]] = []
    for b in order:
        items = by.get(b, [])
        if not items:
            continue
        rows.append({
            "support_bucket": b,
            "class_count": len(items),
            "total_train_rows": sum(int(r.get("train_count", 0) or 0) for r in items),
            "total_val_rows": sum(int(r.get("val_count", 0) or 0) for r in items),
            "train_rank@1_macro": _macro_mean(items, "train_rank@1"),
            "val_rank@1_macro": _macro_mean(items, "val_rank@1"),
            "val_rank@5_macro": _macro_mean(items, "val_rank@5"),
            "mean_train_val_gap": _macro_mean(items, "train_val_rank1_gap"),
            "val_large_negative_error_rate_macro": _macro_mean(items, "val_large_negative_error_rate"),
            "val_near_tie_error_rate_macro": _macro_mean(items, "val_near_tie_error_rate"),
        })
    return rows


def _quadrant_summary(join_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    counts = Counter(str(r.get("quadrant", "")) for r in join_rows)
    summary = [{"quadrant": k, "class_count": v} for k, v in counts.most_common()]
    return summary, list(join_rows)


def _margin_outputs(rows: Sequence[RankRow], person_raw_id: str, near_tie_abs_margin: float, large_negative_margin: float, confident_top12_margin: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    err_rows: List[Dict[str, Any]] = []
    per_class_counter: Dict[str, Counter] = defaultdict(Counter)
    top2_available = any(r.get("top1_vs_top2_margin") is not None for r in rows)
    for r in rows:
        if int(r["rank"]) <= 1:
            continue
        margin = r.get("gt_vs_top1_margin")
        top12 = r.get("top1_vs_top2_margin")
        near = bool(margin is not None and abs(float(margin)) < near_tie_abs_margin)
        large_neg = bool(margin is not None and float(margin) < large_negative_margin)
        confident_wrong = bool(large_neg and top12 is not None and float(top12) > confident_top12_margin)
        hub_confident_wrong = bool(confident_wrong and str(r.get("top1_raw_id")) == person_raw_id)
        d = {
            "trajectory_id": r.get("trajectory_id"),
            "clip_id": r.get("clip_id"),
            "gt_raw_id": r.get("gt_raw_id"),
            "gt_name": r.get("gt_name"),
            "top1_raw_id": r.get("top1_raw_id"),
            "top1_name": r.get("top1_name"),
            "top2_raw_id": r.get("top2_raw_id"),
            "top2_name": r.get("top2_name"),
            "rank": r.get("rank"),
            "gt_score": r.get("gt_score"),
            "top1_score": r.get("top1_score"),
            "top2_score": r.get("top2_score"),
            "gt_vs_top1_margin": margin,
            "top1_vs_top2_margin": top12,
            "gt_vs_top2_margin": r.get("gt_vs_top2_margin"),
            "is_near_tie_top1_error": near,
            "is_large_negative_error": large_neg,
            "is_confident_wrong_error": confident_wrong,
            "is_hub_confident_wrong": hub_confident_wrong,
            "top2_score_available": top2_available,
        }
        err_rows.append(d)
        c = per_class_counter[str(r.get("gt_raw_id"))]
        c["error_count"] += 1
        c["near_tie_error_count"] += int(near)
        c["large_negative_error_count"] += int(large_neg)
        c["confident_wrong_error_count"] += int(confident_wrong)
        c["person_suppressed_count"] += int(str(r.get("top1_raw_id")) == person_raw_id)
    per_class = []
    for rid, c in sorted(per_class_counter.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        n = c["error_count"]
        per_class.append({
            "gt_raw_id": rid,
            "error_count": n,
            "near_tie_error_count": c["near_tie_error_count"],
            "near_tie_error_rate": c["near_tie_error_count"] / max(n, 1),
            "large_negative_error_count": c["large_negative_error_count"],
            "large_negative_error_rate": c["large_negative_error_count"] / max(n, 1),
            "confident_wrong_error_count": c["confident_wrong_error_count"],
            "confident_wrong_error_rate": c["confident_wrong_error_count"] / max(n, 1),
            "person_suppressed_count": c["person_suppressed_count"],
            "person_suppressed_rate": c["person_suppressed_count"] / max(n, 1),
        })
    return err_rows, per_class


def _suppressor_outputs(rows: Sequence[RankRow], person_raw_id: str, large_negative_margin: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    pair: Dict[Tuple[str, str], List[RankRow]] = defaultdict(list)
    cls: Dict[str, List[RankRow]] = defaultdict(list)
    total_errors = 0
    person_errors = 0
    for r in rows:
        if int(r["rank"]) <= 1:
            continue
        total_errors += 1
        gt = str(r.get("gt_raw_id"))
        top1 = str(r.get("top1_raw_id") or "NA")
        pair[(gt, top1)].append(r)
        cls[gt].append(r)
        if top1 == person_raw_id:
            person_errors += 1
    pair_rows: List[Dict[str, Any]] = []
    for (gt, top1), items in sorted(pair.items(), key=lambda kv: len(kv[1]), reverse=True):
        margins = [float(r["gt_vs_top1_margin"]) for r in items if r.get("gt_vs_top1_margin") is not None]
        pair_rows.append({
            "gt_raw_id": gt,
            "gt_name": next((r.get("gt_name") for r in items if r.get("gt_name")), ""),
            "top1_raw_id": top1,
            "top1_name": next((r.get("top1_name") for r in items if r.get("top1_name")), ""),
            "count": len(items),
            "mean_gt_vs_top1_margin": _mean(margins, ""),
            "large_negative_count": sum(1 for m in margins if m < large_negative_margin),
            "is_person_suppressor": top1 == person_raw_id,
        })
    per_class: List[Dict[str, Any]] = []
    for gt, items in sorted(cls.items(), key=lambda kv: len(kv[1]), reverse=True):
        suppressors = Counter(str(r.get("top1_raw_id") or "NA") for r in items)
        top = suppressors.most_common(1)[0] if suppressors else ("", 0)
        per_class.append({
            "gt_raw_id": gt,
            "gt_name": next((r.get("gt_name") for r in items if r.get("gt_name")), ""),
            "error_count": len(items),
            "top_suppressor_raw_id": top[0],
            "top_suppressor_count": top[1],
            "person_suppressed_count": suppressors.get(person_raw_id, 0),
            "person_suppressed_rate": suppressors.get(person_raw_id, 0) / max(len(items), 1),
        })
    summary = {
        "total_error_rows": total_errors,
        "person_suppressed_rows": person_errors,
        "person_suppression_rate": person_errors / max(total_errors, 1),
        "person_raw_id": person_raw_id,
    }
    return pair_rows, per_class, summary


def _base641_drop_outputs(val_visible: Sequence[RankRow], val_base641: Sequence[RankRow], visible_set: set[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    def key(r: RankRow) -> Tuple[str, str]:
        tid = str(r.get("trajectory_id") or "")
        if tid:
            return (tid, str(r.get("gt_raw_id")))
        return (str(r.get("video_id")) + ":" + str(r.get("clip_id")), str(r.get("gt_raw_id")))

    vmap = {key(r): r for r in val_visible}
    bmap = {key(r): r for r in val_base641}
    detail: List[Dict[str, Any]] = []
    counts = Counter()
    for k in sorted(set(vmap) | set(bmap)):
        v = vmap.get(k)
        b = bmap.get(k)
        if v is not None and b is not None:
            v_ok = int(v["rank"]) <= 1
            b_ok = int(b["rank"]) <= 1
            if v_ok and b_ok:
                category = "same_visible_target_correct_both"
            elif v_ok and not b_ok:
                if str(b.get("top1_raw_id")) not in visible_set:
                    category = "same_visible_target_525_correct_641_wrong_new_candidate"
                else:
                    category = "same_visible_target_525_correct_641_wrong_visible_candidate"
            elif (not v_ok) and b_ok:
                category = "same_visible_target_525_wrong_641_correct"
            else:
                category = "same_visible_target_wrong_both"
        elif b is not None and v is None:
            category = "new_base_target_only_in_641"
        else:
            category = "visible_row_missing_in_641"
        counts[category] += 1
        row = b or v
        detail.append({
            "category": category,
            "trajectory_id": row.get("trajectory_id") if row else "",
            "gt_raw_id": row.get("gt_raw_id") if row else "",
            "gt_name": row.get("gt_name") if row else "",
            "rank_525": v.get("rank") if v else "",
            "top1_525": v.get("top1_raw_id") if v else "",
            "rank_641": b.get("rank") if b else "",
            "top1_641": b.get("top1_raw_id") if b else "",
            "top1_641_is_visible525": str(b.get("top1_raw_id")) in visible_set if b else "",
        })
    summary = [{"category": k, "count": v} for k, v in counts.most_common()]
    return detail, summary, dict(counts)


def _epoch_overfit_outputs(dj3_m: Dict[str, Dict[str, Any]], dj4_m: Dict[str, Dict[str, Any]], dj5_m: Dict[str, Dict[str, Any]], support_lookup: Mapping[str, str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    all_ids = set(dj3_m) | set(dj4_m) | set(dj5_m)
    for rid in sorted(all_ids, key=lambda x: int(x) if x.isdigit() else x):
        r3 = _to_float(dj3_m.get(rid, {}).get("rank@1"))
        r4 = _to_float(dj4_m.get(rid, {}).get("rank@1"))
        r5 = _to_float(dj5_m.get(rid, {}).get("rank@1"))
        m3 = _to_float(dj3_m.get(rid, {}).get("mean_rank"))
        m4 = _to_float(dj4_m.get(rid, {}).get("mean_rank"))
        m5 = _to_float(dj5_m.get(rid, {}).get("mean_rank"))
        rows.append({
            "raw_id": rid,
            "class_name": dj3_m.get(rid, {}).get("class_name") or dj4_m.get(rid, {}).get("class_name") or dj5_m.get(rid, {}).get("class_name") or "",
            "support_bucket": support_lookup.get(rid, ""),
            "dj3_val_rank@1": r3 if r3 is not None else "",
            "dj4_val_rank@1": r4 if r4 is not None else "",
            "dj5_val_rank@1": r5 if r5 is not None else "",
            "dj4_minus_dj3_rank@1": (r4 - r3) if r4 is not None and r3 is not None else "",
            "dj5_minus_dj3_rank@1": (r5 - r3) if r5 is not None and r3 is not None else "",
            "dj3_mean_rank": m3 if m3 is not None else "",
            "dj4_mean_rank": m4 if m4 is not None else "",
            "dj5_mean_rank": m5 if m5 is not None else "",
            "dj4_minus_dj3_mean_rank": (m4 - m3) if m4 is not None and m3 is not None else "",
            "dj5_minus_dj3_mean_rank": (m5 - m3) if m5 is not None and m3 is not None else "",
        })
    summary: List[Dict[str, Any]] = []
    for bucket in ["1-2", "3-5", "6-10", "11-50", "51-200", ">200", ""]:
        items = [r for r in rows if r.get("support_bucket") == bucket]
        if not items:
            continue
        summary.append({
            "support_bucket": bucket or "unknown",
            "class_count": len(items),
            "mean_dj4_minus_dj3_rank@1": _macro_mean(items, "dj4_minus_dj3_rank@1"),
            "mean_dj5_minus_dj3_rank@1": _macro_mean(items, "dj5_minus_dj3_rank@1"),
            "mean_dj4_minus_dj3_mean_rank": _macro_mean(items, "dj4_minus_dj3_mean_rank"),
            "mean_dj5_minus_dj3_mean_rank": _macro_mean(items, "dj5_minus_dj3_mean_rank"),
            "degraded_dj4_class_count": sum(1 for r in items if _to_float(r.get("dj4_minus_dj3_rank@1"), 0) is not None and float(r.get("dj4_minus_dj3_rank@1") or 0) < 0),
            "degraded_dj5_class_count": sum(1 for r in items if _to_float(r.get("dj5_minus_dj3_rank@1"), 0) is not None and float(r.get("dj5_minus_dj3_rank@1") or 0) < 0),
        })
    return rows, summary


def _takeover_text(payload: Mapping[str, Any]) -> str:
    lines = ["# A8 D-J3 Semantic Boundary Bottleneck Audit TAKEOVER", ""]
    lines.append(f"- status: {payload.get('status')}")
    lines.append(f"- output_root: {payload.get('output_root')}")
    lines.append(f"- top2_score_available: {payload.get('top2_score_available')}")
    if not payload.get("top2_score_available"):
        lines.append("- top1_vs_top2_margin_unavailable_in_current_per_row_exports: true")
    lines.append("")
    lines.append("## Headline")
    h = payload.get("headline", {})
    for k, v in h.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Bottleneck interpretation")
    for item in payload.get("interpretation", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Key artifacts")
    for name, path in payload.get("artifacts", {}).items():
        lines.append(f"- {name}: {path}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This audit is read-only: no training, checkpoint mutation, or control-plane writeback.")
    lines.append("- Margin thresholds are heuristic and can be swept later.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--output_root", default="")
    ap.add_argument("--train_visible_dj3_per_row", default="")
    ap.add_argument("--val_visible_dj3_per_row", default="")
    ap.add_argument("--val_visible_dj4_per_row", default="")
    ap.add_argument("--val_visible_dj5_per_row", default="")
    ap.add_argument("--train_base641_dj3_per_row", default="")
    ap.add_argument("--val_base641_dj3_per_row", default="")
    ap.add_argument("--person_raw_id", default=PERSON_RAW_ID_DEFAULT)
    ap.add_argument("--near_tie_abs_margin", type=float, default=0.02)
    ap.add_argument("--large_negative_margin", type=float, default=-0.1)
    ap.add_argument("--confident_top12_margin", type=float, default=0.05)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    out = Path(args.output_root).resolve() if args.output_root else run_root / "analysis/a8_dj3_semantic_boundary_bottleneck_audit"
    out.mkdir(parents=True, exist_ok=True)

    train_visible_p = Path(args.train_visible_dj3_per_row) if args.train_visible_dj3_per_row else run_root / "outputs/a8_joint_train_time_dynamic_hungarian/lvvis_train_base/D-J3_pre1_dyn1_ep10/analysis/canonical_visible525_lvvis_train_base/visible525_candidate_rankk_per_row.csv"
    val_visible_p = Path(args.val_visible_dj3_per_row) if args.val_visible_dj3_per_row else run_root / "analysis/a8_visible525_candidate_rankk_audit/lvvis_val/D-J3_train_time_dynamic_ep10_val_target525_candidate525/visible525_candidate_rankk_per_row.csv"
    val_dj4_p = Path(args.val_visible_dj4_per_row) if args.val_visible_dj4_per_row else run_root / "analysis/a8_visible525_candidate_rankk_audit/lvvis_val/D-J4_pre1_dyn1_ep50_val_target525_candidate525/visible525_candidate_rankk_per_row.csv"
    val_dj5_p = Path(args.val_visible_dj5_per_row) if args.val_visible_dj5_per_row else run_root / "analysis/a8_visible525_candidate_rankk_audit/lvvis_val/D-J5_pre1_dyn1_ep100_val_target525_candidate525/visible525_candidate_rankk_per_row.csv"
    train_641_p = Path(args.train_base641_dj3_per_row) if args.train_base641_dj3_per_row else run_root / "analysis/a8_base641_candidate_rankk_audit/lvvis_train_base/D-J3_pre1_dyn1_ep10_base641/base641_candidate_rankk_per_row.csv"
    val_641_p = Path(args.val_base641_dj3_per_row) if args.val_base641_dj3_per_row else run_root / "analysis/a8_base641_candidate_rankk_audit/lvvis_val/D-J3_pre1_dyn1_ep10_base641/base641_candidate_rankk_per_row.csv"

    input_paths = {
        "train_visible_dj3_per_row": train_visible_p,
        "val_visible_dj3_per_row": val_visible_p,
        "val_visible_dj4_per_row": val_dj4_p,
        "val_visible_dj5_per_row": val_dj5_p,
        "train_base641_dj3_per_row": train_641_p,
        "val_base641_dj3_per_row": val_641_p,
    }
    missing = {k: str(p) for k, p in input_paths.items() if not p.exists()}
    if missing:
        _write_json(out / "missing_inputs.json", missing)
        raise SystemExit(f"MISSING inputs: {missing}")

    train_visible = _parse_rank_rows(_read_csv(train_visible_p), "train_visible525_dj3")
    val_visible = _parse_rank_rows(_read_csv(val_visible_p), "val_visible525_dj3")
    val_dj4 = _parse_rank_rows(_read_csv(val_dj4_p), "val_visible525_dj4")
    val_dj5 = _parse_rank_rows(_read_csv(val_dj5_p), "val_visible525_dj5")
    train_641 = _parse_rank_rows(_read_csv(train_641_p), "train_base641_dj3")
    val_641 = _parse_rank_rows(_read_csv(val_641_p), "val_base641_dj3")

    train_m = _class_metrics(train_visible, near_tie_abs_margin=args.near_tie_abs_margin, large_negative_margin=args.large_negative_margin)
    val_m = _class_metrics(val_visible, near_tie_abs_margin=args.near_tie_abs_margin, large_negative_margin=args.large_negative_margin)
    dj4_m = _class_metrics(val_dj4, near_tie_abs_margin=args.near_tie_abs_margin, large_negative_margin=args.large_negative_margin)
    dj5_m = _class_metrics(val_dj5, near_tie_abs_margin=args.near_tie_abs_margin, large_negative_margin=args.large_negative_margin)

    join_rows = _per_class_join(train_m, val_m)
    support_rows = _support_bucket_summary(join_rows)
    quadrant_summary, quadrant_list = _quadrant_summary(join_rows)
    margin_errors, margin_per_class = _margin_outputs(val_visible, args.person_raw_id, args.near_tie_abs_margin, args.large_negative_margin, args.confident_top12_margin)
    suppressor_pairs, per_class_suppressor, suppressor_summary = _suppressor_outputs(val_visible, args.person_raw_id, args.large_negative_margin)
    visible_set = set(train_m.keys())
    base641_detail, base641_summary_rows, base641_counts = _base641_drop_outputs(val_visible, val_641, visible_set)
    support_lookup = {str(r["raw_id"]): str(r["support_bucket"]) for r in join_rows}
    epoch_delta_rows, epoch_delta_summary = _epoch_overfit_outputs(val_m, dj4_m, dj5_m, support_lookup)

    # Write CSV artifacts.
    artifacts = {
        "per_class_train_val_525_join": out / "per_class_train_val_525_join.csv",
        "support_bucket_summary": out / "support_bucket_summary.csv",
        "four_quadrant_class_summary": out / "four_quadrant_class_summary.csv",
        "four_quadrant_class_list": out / "four_quadrant_class_list.csv",
        "margin_error_decomposition": out / "margin_error_decomposition.csv",
        "per_class_margin_summary": out / "per_class_margin_summary.csv",
        "top_suppressor_pair_summary": out / "top_suppressor_pair_summary.csv",
        "person_hub_suppression_summary": out / "person_hub_suppression_summary.json",
        "per_class_top_suppressor": out / "per_class_top_suppressor.csv",
        "base641_drop_decomposition": out / "base641_drop_decomposition.csv",
        "base641_drop_summary": out / "base641_drop_summary.csv",
        "epoch_overfit_per_class_delta": out / "epoch_overfit_per_class_delta.csv",
        "epoch_overfit_delta_summary": out / "epoch_overfit_delta_summary.csv",
    }
    _write_csv(artifacts["per_class_train_val_525_join"], join_rows)
    _write_csv(artifacts["support_bucket_summary"], support_rows)
    _write_csv(artifacts["four_quadrant_class_summary"], quadrant_summary)
    _write_csv(artifacts["four_quadrant_class_list"], quadrant_list)
    _write_csv(artifacts["margin_error_decomposition"], margin_errors)
    _write_csv(artifacts["per_class_margin_summary"], margin_per_class)
    _write_csv(artifacts["top_suppressor_pair_summary"], suppressor_pairs)
    _write_json(artifacts["person_hub_suppression_summary"], suppressor_summary)
    _write_csv(artifacts["per_class_top_suppressor"], per_class_suppressor)
    _write_csv(artifacts["base641_drop_decomposition"], base641_detail)
    _write_csv(artifacts["base641_drop_summary"], base641_summary_rows)
    _write_csv(artifacts["epoch_overfit_per_class_delta"], epoch_delta_rows)
    _write_csv(artifacts["epoch_overfit_delta_summary"], epoch_delta_summary)

    qcounts = Counter(str(r.get("quadrant")) for r in join_rows)
    top2_available = any(r.get("top1_vs_top2_margin") is not None for r in val_visible)
    low_buckets = {"1-2", "3-5", "6-10"}
    low_items = [r for r in join_rows if r.get("support_bucket") in low_buckets]
    high_items = [r for r in join_rows if r.get("support_bucket") in {"51-200", ">200"}]
    low_val_rank1 = _macro_mean(low_items, "val_rank@1")
    high_val_rank1 = _macro_mean(high_items, "val_rank@1")
    large_negative_total = sum(1 for r in margin_errors if str(r.get("is_large_negative_error")) == "True")
    near_tie_total = sum(1 for r in margin_errors if str(r.get("is_near_tie_top1_error")) == "True")

    interpretation: List[str] = []
    interpretation.append("train/val 525 class-level boundary is evaluated by per-class macro buckets, not only micro rank@1.")
    if low_val_rank1 != "" and high_val_rank1 != "" and float(low_val_rank1) + 0.15 < float(high_val_rank1):
        interpretation.append("low-support classes are substantially weaker on val; long-tail support is likely an important bottleneck.")
    else:
        interpretation.append("low-support weakness is not sufficient alone; inspect hub/context/similar-class suppressors.")
    if suppressor_summary.get("person_suppression_rate", 0) > 0.1:
        interpretation.append("person/raw_id=773 remains a non-trivial val suppressor.")
    else:
        interpretation.append("person/raw_id=773 is not the dominant aggregate suppressor under current thresholds.")
    if base641_counts.get("same_visible_target_525_correct_641_wrong_new_candidate", 0) > 0:
        interpretation.append("part of 525->641 drop is caused by new official-base candidates stealing top1 from visible525 targets.")
    if base641_counts.get("new_base_target_only_in_641", 0) > 0:
        interpretation.append("part of 525->641 drop comes from base target rows outside the visible525 audit target set.")
    if not top2_available:
        interpretation.append("top1/top2 margin is unavailable in current per-row exports; this audit uses gt-vs-top1 margin and flags the missing top2 score explicitly.")

    payload = {
        "status": "PASS",
        "output_root": str(out),
        "inputs": {k: str(v) for k, v in input_paths.items()},
        "top2_score_available": top2_available,
        "thresholds": {
            "near_tie_abs_margin": args.near_tie_abs_margin,
            "large_negative_margin": args.large_negative_margin,
            "confident_top12_margin": args.confident_top12_margin,
            "person_raw_id": args.person_raw_id,
        },
        "headline": {
            "class_join_count": len(join_rows),
            "learned_stable_count": qcounts.get("learned_stable", 0),
            "underlearned_count": qcounts.get("underlearned", 0),
            "overfit_context_fail_count": qcounts.get("overfit_context_fail", 0),
            "low_support_val_rank1_macro": low_val_rank1,
            "high_support_val_rank1_macro": high_val_rank1,
            "val_error_rows": len(margin_errors),
            "large_negative_error_count": large_negative_total,
            "near_tie_error_count": near_tie_total,
            "person_suppression_rate": suppressor_summary.get("person_suppression_rate"),
            "base641_new_candidate_interference_count": base641_counts.get("same_visible_target_525_correct_641_wrong_new_candidate", 0),
            "base641_new_target_only_count": base641_counts.get("new_base_target_only_in_641", 0),
        },
        "interpretation": interpretation,
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }
    _write_json(out / "semantic_boundary_bottleneck_summary.json", payload)
    (out / "A8_DJ3_SEMANTIC_BOUNDARY_BOTTLENECK_TAKEOVER.md").write_text(_takeover_text(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
