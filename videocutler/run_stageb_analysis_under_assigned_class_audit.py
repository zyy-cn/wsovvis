#!/usr/bin/env python3
"""Read-only under-assigned class audit for GT-fullY clean nohub training.

This script does not train, does not touch checkpoints, and does not run mAP.
It analyzes class demand-side statistics from nohub responsibility records and
joins them with existing class-level GT attribution diagnostics for post-hoc
interpretation only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ----------------------------- basic IO helpers -----------------------------


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return repr(v)
    return v


def _safe_int_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k, "")) for k in fields})


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _pick(row: Mapping[str, Any], names: Sequence[str], default: str = "") -> str:
    for name in names:
        if name and name in row and row[name] not in (None, ""):
            return str(row[name])
    return default


# ----------------------------- stats helpers --------------------------------


def _rankdata(vals: Sequence[float]) -> List[float]:
    pairs = sorted((v, i) for i, v in enumerate(vals))
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for _, idx in pairs[i:j]:
            ranks[idx] = avg
        i = j
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _auc_score(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if len(scores) != len(labels) or len(scores) < 2:
        return None
    pos = sum(1 for y in labels if y)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = _rankdata(scores)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


# ---------------------------- metadata readers ------------------------------


def _infer_fields(rows: Sequence[Mapping[str, str]]) -> Dict[str, str]:
    if not rows:
        return {}
    cols = set(rows[0].keys())

    def first(cands: Sequence[str]) -> str:
        for c in cands:
            if c in cols:
                return c
        return ""

    return {
        "run": first(["checkpoint", "run", "protocol"]),
        "raw_id": first(["raw_id", "class_raw_id", "gt_raw_id", "category_id", "class_id"]),
        "class_name": first(["class_name", "gt_class_name", "name"]),
        "gt_count": first(["gt_count", "row_count", "count"]),
        "top1": first(["gt_top1_hit_rate", "top1_hit_rate", "top1_is_gt_mean"]),
        "norm_rank": first(["mean_normalized_gt_rank", "normalized_gt_rank_mean"]),
        "rank_mean": first(["gt_rank_mean", "mean_gt_rank"]),
        "candidate_size": first(["candidate_size_mean"]),
        "certificate_family": first(["certificate_family"]),
        "certificate_type": first(["certificate_type"]),
        "resolved_round": first(["resolved_round"]),
        "base_group": first(["base_group", "base_observed_unobserved"]),
        "person_conditioned": first(["person_conditioned"]),
    }


def _load_class_metadata(compare_dir: Optional[Path], schedule_csv: Optional[Path], run_name: str) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}

    def update_from_row(raw_id: str, row: Mapping[str, Any], fields: Mapping[str, str]) -> None:
        if not raw_id:
            return
        dst = meta.setdefault(raw_id, {"raw_id": raw_id})
        mapping = {
            "class_name": fields.get("class_name", ""),
            "gt_count": fields.get("gt_count", ""),
            "gt_top1_hit_rate": fields.get("top1", ""),
            "mean_normalized_gt_rank": fields.get("norm_rank", ""),
            "gt_rank_mean": fields.get("rank_mean", ""),
            "candidate_size_mean": fields.get("candidate_size", ""),
            "certificate_family": fields.get("certificate_family", ""),
            "certificate_type": fields.get("certificate_type", ""),
            "resolved_round": fields.get("resolved_round", ""),
            "base_group": fields.get("base_group", ""),
            "person_conditioned": fields.get("person_conditioned", ""),
        }
        for out_key, in_key in mapping.items():
            if in_key and row.get(in_key, "") not in (None, ""):
                dst[out_key] = row.get(in_key)

    if compare_dir:
        attr_rows = _read_csv(compare_dir / "per_class_attribution.csv")
        fields = _infer_fields(attr_rows)
        run_col = fields.get("run") or "checkpoint"
        raw_col = fields.get("raw_id") or "raw_id"
        for row in attr_rows:
            if str(row.get(run_col, "")) != run_name:
                continue
            update_from_row(_safe_int_str(row.get(raw_col)), row, fields)

    if schedule_csv and schedule_csv.exists():
        sched = _read_csv(schedule_csv)
        if sched:
            fields = _infer_fields(sched)
            raw_col = fields.get("raw_id") or "raw_id"
            for row in sched:
                raw_id = _safe_int_str(row.get(raw_col))
                if not raw_id:
                    continue
                dst = meta.setdefault(raw_id, {"raw_id": raw_id})
                for k in ["class_name", "certificate_family", "certificate_type", "resolved_round", "base_group", "person_conditioned"]:
                    fk = fields.get(k, "")
                    if fk and row.get(fk, "") not in (None, "") and not dst.get(k):
                        dst[k] = row.get(fk)
    return meta


def _load_nohub_delta(compare_dir: Optional[Path], nohub_run: str) -> Dict[str, Dict[str, Any]]:
    """Load nohub-vs-baseline class deltas. Prefer emitted delta table; fallback compute from attribution table."""
    if not compare_dir:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    delta_rows = _read_csv(compare_dir / "per_class_delta_vs_baseline.csv")
    if delta_rows:
        fields = _infer_fields(delta_rows)
        run_col = fields.get("run") or "run"
        raw_col = fields.get("raw_id") or "raw_id"
        for row in delta_rows:
            if str(row.get(run_col, "")) != nohub_run:
                continue
            raw_id = _safe_int_str(row.get(raw_col))
            if not raw_id:
                continue
            out[raw_id] = {
                "delta_gt_top1_hit_rate": _num(row.get("delta_gt_top1_hit_rate")),
                "delta_mean_normalized_gt_rank": _num(row.get("delta_mean_normalized_gt_rank")),
                "delta_gt_rank_mean": _num(row.get("delta_gt_rank_mean")),
            }
        if out:
            return out

    attr_rows = _read_csv(compare_dir / "per_class_attribution.csv")
    if not attr_rows:
        return {}
    fields = _infer_fields(attr_rows)
    run_col = fields.get("run") or "checkpoint"
    raw_col = fields.get("raw_id") or "raw_id"
    top1_col = fields.get("top1") or "gt_top1_hit_rate"
    rank_col = fields.get("norm_rank") or "mean_normalized_gt_rank"
    gt_rank_col = fields.get("rank_mean") or "gt_rank_mean"
    idx: Dict[Tuple[str, str], Mapping[str, str]] = {}
    for row in attr_rows:
        idx[(_safe_int_str(row.get(raw_col)), str(row.get(run_col, "")))] = row
    raw_ids = sorted({k[0] for k in idx.keys() if k[0]})
    for raw_id in raw_ids:
        base = idx.get((raw_id, "baseline_full_y"))
        nr = idx.get((raw_id, nohub_run))
        if not base or not nr:
            continue
        out[raw_id] = {
            "delta_gt_top1_hit_rate": _num(nr.get(top1_col)) - _num(base.get(top1_col)),
            "delta_mean_normalized_gt_rank": _num(nr.get(rank_col)) - _num(base.get(rank_col)),
            "delta_gt_rank_mean": _num(nr.get(gt_rank_col)) - _num(base.get(gt_rank_col)),
        }
    return out


# ----------------------- responsibility aggregation -------------------------


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if isinstance(obj, Mapping):
                yield obj


def _responsibility_dict(rec: Mapping[str, Any], preferred: str) -> Mapping[str, Any]:
    if preferred in rec and isinstance(rec[preferred], Mapping):
        return rec[preferred]
    for k in ["r_final", "responsibility", "responsibilities", "r_init"]:
        if k in rec and isinstance(rec[k], Mapping):
            return rec[k]
    return {}


def _candidate_ids(rec: Mapping[str, Any], r: Mapping[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in ["candidate_ids_known", "candidate_ids", "known_candidate_ids", "Y_base", "y_base"]:
        val = rec.get(key)
        if isinstance(val, list):
            ids.extend(_safe_int_str(x) for x in val if _safe_int_str(x))
            break
    if not ids:
        for k in r.keys():
            kk = _safe_int_str(k)
            if kk and kk != "unknown":
                ids.append(kk)
    # preserve order, dedup
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _aggregate_responsibilities(path: Path, preferred_resp: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    stats: DefaultDict[str, Dict[str, float]] = defaultdict(lambda: {
        "candidate_support": 0.0,
        "responsibility_mass": 0.0,
        "top1_count": 0.0,
        "r_positive_count": 0.0,
        "max_prob_sum": 0.0,
    })
    total_rows = 0
    numeric_mass_rows = 0
    empty_candidate_rows = 0
    for rec in _iter_jsonl(path):
        total_rows += 1
        r = _responsibility_dict(rec, preferred_resp)
        cand = _candidate_ids(rec, r)
        if not cand:
            empty_candidate_rows += 1
            continue
        for cid in cand:
            stats[cid]["candidate_support"] += 1.0
        numeric_items: List[Tuple[str, float]] = []
        for k, v in r.items():
            cid = _safe_int_str(k)
            if not cid or cid == "unknown":
                continue
            val = _num(v)
            numeric_items.append((cid, val))
            if val > 0:
                stats[cid]["r_positive_count"] += 1.0
                stats[cid]["responsibility_mass"] += val
        if numeric_items:
            numeric_mass_rows += 1
            top_c, top_v = max(numeric_items, key=lambda kv: kv[1])
            stats[top_c]["top1_count"] += 1.0
            stats[top_c]["max_prob_sum"] += top_v
    meta = {
        "responsibility_records": str(path),
        "total_rows": total_rows,
        "numeric_mass_rows": numeric_mass_rows,
        "empty_candidate_rows": empty_candidate_rows,
        "class_count_from_responsibility": len(stats),
    }
    return dict(stats), meta


# ------------------------------ table builders ------------------------------


def _compute_class_table(
    resp_stats: Mapping[str, Mapping[str, float]],
    metadata: Mapping[str, Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, Any]],
    mass_floor: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    raw_ids = sorted(set(resp_stats) | set(metadata) | set(deltas), key=lambda x: int(x) if str(x).isdigit() else str(x))
    for raw_id in raw_ids:
        rs = resp_stats.get(raw_id, {})
        md = metadata.get(raw_id, {})
        de = deltas.get(raw_id, {})
        support = _num(rs.get("candidate_support"))
        mass = _num(rs.get("responsibility_mass"))
        top1 = _num(rs.get("top1_count"))
        mass_per_support = mass / support if support > 0 else 0.0
        top1_per_support = top1 / support if support > 0 else 0.0
        support_mass_gap = max(0.0, support - mass)
        support_mass_ratio = support / max(mass, mass_floor) if support > 0 else 0.0
        under_top1_ratio = support / max(top1, mass_floor) if support > 0 else 0.0
        low_mass_per_support = 1.0 - mass_per_support if support > 0 else 0.0
        # Conservative hybrid: demand deficit must have both support and poor mass/top1 assignment.
        hybrid = math.log1p(support) * (0.5 * low_mass_per_support + 0.5 * (1.0 - top1_per_support if support > 0 else 0.0))
        row: Dict[str, Any] = {
            "raw_id": raw_id,
            "class_name": md.get("class_name", ""),
            "candidate_support": support,
            "responsibility_mass": mass,
            "top1_count": top1,
            "r_positive_count": _num(rs.get("r_positive_count")),
            "mean_responsibility_per_support": mass_per_support,
            "top1_per_support": top1_per_support,
            "support_mass_gap": support_mass_gap,
            "support_mass_ratio": support_mass_ratio,
            "under_top1_ratio": under_top1_ratio,
            "low_mass_per_support": low_mass_per_support,
            "hybrid_under_assignment_score": hybrid,
            "gt_count": _num(md.get("gt_count")),
            "gt_top1_hit_rate": _num(md.get("gt_top1_hit_rate")),
            "mean_normalized_gt_rank": _num(md.get("mean_normalized_gt_rank")),
            "delta_gt_top1_hit_rate": _num(de.get("delta_gt_top1_hit_rate")),
            "delta_mean_normalized_gt_rank": _num(de.get("delta_mean_normalized_gt_rank")),
            "delta_gt_rank_mean": _num(de.get("delta_gt_rank_mean")),
            "is_nohub_degraded_top1": 1 if _num(de.get("delta_gt_top1_hit_rate")) < 0 else 0,
            "is_nohub_degraded_rank": 1 if _num(de.get("delta_mean_normalized_gt_rank")) > 0 else 0,
            "is_nohub_degraded_either": 1 if (_num(de.get("delta_gt_top1_hit_rate")) < 0 or _num(de.get("delta_mean_normalized_gt_rank")) > 0) else 0,
            "certificate_family": md.get("certificate_family", ""),
            "certificate_type": md.get("certificate_type", ""),
            "resolved_round": md.get("resolved_round", ""),
            "base_group": md.get("base_group", ""),
            "person_conditioned": md.get("person_conditioned", ""),
            "is_anchor_conditioned": 1 if str(md.get("certificate_family", "")) == "anchor_conditioned" else 0,
        }
        rows.append(row)
    return rows


METRICS = [
    ("support_mass_gap", True),
    ("support_mass_ratio", True),
    ("under_top1_ratio", True),
    ("low_mass_per_support", True),
    ("hybrid_under_assignment_score", True),
    ("negative_mean_responsibility_per_support", True),
]


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == "negative_mean_responsibility_per_support":
        return -_num(row.get("mean_responsibility_per_support"))
    return _num(row.get(metric))


def _top_by_metric(rows: Sequence[Mapping[str, Any]], metric: str, k: int, min_support: float = 0.0) -> List[Dict[str, Any]]:
    filt = [r for r in rows if _num(r.get("candidate_support")) >= min_support]
    ranked = sorted(filt, key=lambda r: _metric_value(r, metric), reverse=True)
    out: List[Dict[str, Any]] = []
    for rank, r in enumerate(ranked[:k], start=1):
        rr = dict(r)
        rr["metric"] = metric
        rr["metric_rank"] = rank
        rr["metric_value"] = _metric_value(r, metric)
        out.append(rr)
    return out


def _metric_quality(rows: Sequence[Mapping[str, Any]], metric: str, top_k: int, support_threshold: float) -> Dict[str, Any]:
    top = _top_by_metric(rows, metric, top_k, min_support=0.0)
    top_sf = _top_by_metric(rows, metric, top_k, min_support=support_threshold)
    def frac(pred) -> float:
        if not top:
            return 0.0
        return sum(1 for r in top if pred(r)) / len(top)
    def frac_sf(pred) -> float:
        if not top_sf:
            return 0.0
        return sum(1 for r in top_sf if pred(r)) / len(top_sf)

    xs = [_metric_value(r, metric) for r in rows]
    y_top1_bad = [-_num(r.get("delta_gt_top1_hit_rate")) for r in rows]
    y_rank_bad = [_num(r.get("delta_mean_normalized_gt_rank")) for r in rows]
    labels_degraded = [int(_num(r.get("is_nohub_degraded_either")) > 0) for r in rows]
    return {
        "metric": metric,
        "top_k": top_k,
        "support_threshold_for_filtered_top": support_threshold,
        "topk_low_support_lt5_rate": frac(lambda r: _num(r.get("candidate_support")) < 5),
        "topk_low_support_lt10_rate": frac(lambda r: _num(r.get("candidate_support")) < 10),
        "topk_anchor_rate": frac(lambda r: _num(r.get("is_anchor_conditioned")) > 0),
        "topk_nohub_degraded_either_rate": frac(lambda r: _num(r.get("is_nohub_degraded_either")) > 0),
        "topk_nohub_degraded_top1_rate": frac(lambda r: _num(r.get("is_nohub_degraded_top1")) > 0),
        "topk_nohub_degraded_rank_rate": frac(lambda r: _num(r.get("is_nohub_degraded_rank")) > 0),
        "support_filtered_topk_anchor_rate": frac_sf(lambda r: _num(r.get("is_anchor_conditioned")) > 0),
        "support_filtered_topk_nohub_degraded_either_rate": frac_sf(lambda r: _num(r.get("is_nohub_degraded_either")) > 0),
        "pearson_metric_vs_negative_delta_top1": _pearson(xs, y_top1_bad),
        "spearman_metric_vs_negative_delta_top1": _spearman(xs, y_top1_bad),
        "pearson_metric_vs_delta_rank_bad": _pearson(xs, y_rank_bad),
        "spearman_metric_vs_delta_rank_bad": _spearman(xs, y_rank_bad),
        "auc_metric_predict_degraded_either": _auc_score(xs, labels_degraded),
    }


def _build_overlap(rows: Sequence[Mapping[str, Any]], metric: str, top_k: int, support_threshold: float, anchor_only: bool) -> List[Dict[str, Any]]:
    top = _top_by_metric(rows, metric, top_k, min_support=support_threshold)
    out: List[Dict[str, Any]] = []
    for r in top:
        if anchor_only and _num(r.get("is_anchor_conditioned")) <= 0:
            continue
        if _num(r.get("is_nohub_degraded_either")) <= 0:
            continue
        out.append({
            "metric": metric,
            "metric_rank": r.get("metric_rank"),
            "raw_id": r.get("raw_id"),
            "class_name": r.get("class_name"),
            "candidate_support": r.get("candidate_support"),
            "responsibility_mass": r.get("responsibility_mass"),
            "top1_count": r.get("top1_count"),
            "metric_value": r.get("metric_value"),
            "certificate_family": r.get("certificate_family"),
            "certificate_type": r.get("certificate_type"),
            "resolved_round": r.get("resolved_round"),
            "gt_count": r.get("gt_count"),
            "delta_gt_top1_hit_rate": r.get("delta_gt_top1_hit_rate"),
            "delta_mean_normalized_gt_rank": r.get("delta_mean_normalized_gt_rank"),
            "is_nohub_degraded_top1": r.get("is_nohub_degraded_top1"),
            "is_nohub_degraded_rank": r.get("is_nohub_degraded_rank"),
        })
    return out


def _correlation_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for metric, _ in METRICS:
        xs = [_metric_value(r, metric) for r in rows]
        targets = {
            "negative_delta_top1": [-_num(r.get("delta_gt_top1_hit_rate")) for r in rows],
            "delta_rank_bad": [_num(r.get("delta_mean_normalized_gt_rank")) for r in rows],
            "gt_top1_hit_rate": [_num(r.get("gt_top1_hit_rate")) for r in rows],
            "mean_normalized_gt_rank": [_num(r.get("mean_normalized_gt_rank")) for r in rows],
        }
        for target, ys in targets.items():
            out.append({
                "metric": metric,
                "target": target,
                "n": len(rows),
                "pearson": _pearson(xs, ys),
                "spearman": _spearman(xs, ys),
            })
    return out


# ---------------------------------- main -------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only under-assigned class audit for GT-fullY clean nohub.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--responsibility_records", required=True, help="Path to nohub train/prealign/responsibility_records.jsonl")
    ap.add_argument("--compare_dir", default="", help="Attribution compare directory containing per_class_attribution.csv and per_class_delta_vs_baseline.csv")
    ap.add_argument("--schedule_csv", default="")
    ap.add_argument("--nohub_run", default="soft_e2e_nohub")
    ap.add_argument("--responsibility_field", default="r_final")
    ap.add_argument("--mass_floor", type=float, default=20.0)
    ap.add_argument("--support_threshold", type=float, default=20.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--min_class_gt_count", type=float, default=0.0, help="Only affects metric-quality/correlation diagnostic subset, not metric definition.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    resp_path = Path(args.responsibility_records)
    compare_dir = Path(args.compare_dir) if args.compare_dir else None
    sched = Path(args.schedule_csv) if args.schedule_csv else None

    if not resp_path.exists():
        raise FileNotFoundError(str(resp_path))
    if compare_dir and not compare_dir.exists():
        raise FileNotFoundError(str(compare_dir))

    resp_stats, resp_meta = _aggregate_responsibilities(resp_path, args.responsibility_field)
    metadata = _load_class_metadata(compare_dir, sched, args.nohub_run)
    deltas = _load_nohub_delta(compare_dir, args.nohub_run)
    table = _compute_class_table(resp_stats, metadata, deltas, mass_floor=args.mass_floor)

    diagnostic_rows = [r for r in table if _num(r.get("gt_count")) >= args.min_class_gt_count]
    if not diagnostic_rows:
        diagnostic_rows = table

    fields = [
        "raw_id", "class_name", "candidate_support", "responsibility_mass", "top1_count", "r_positive_count",
        "mean_responsibility_per_support", "top1_per_support", "support_mass_gap", "support_mass_ratio",
        "under_top1_ratio", "low_mass_per_support", "hybrid_under_assignment_score",
        "gt_count", "gt_top1_hit_rate", "mean_normalized_gt_rank", "delta_gt_top1_hit_rate",
        "delta_mean_normalized_gt_rank", "delta_gt_rank_mean", "is_nohub_degraded_top1", "is_nohub_degraded_rank",
        "is_nohub_degraded_either", "certificate_family", "certificate_type", "resolved_round", "base_group",
        "person_conditioned", "is_anchor_conditioned",
    ]
    table_sorted = sorted(table, key=lambda r: _num(r.get("candidate_support")), reverse=True)
    _write_csv(out_dir / "under_assigned_class_table.csv", table_sorted, fields)

    quality = [_metric_quality(diagnostic_rows, metric, args.top_k, args.support_threshold) for metric, _ in METRICS]
    q_fields = [
        "metric", "top_k", "support_threshold_for_filtered_top", "topk_low_support_lt5_rate", "topk_low_support_lt10_rate",
        "topk_anchor_rate", "topk_nohub_degraded_either_rate", "topk_nohub_degraded_top1_rate", "topk_nohub_degraded_rank_rate",
        "support_filtered_topk_anchor_rate", "support_filtered_topk_nohub_degraded_either_rate",
        "pearson_metric_vs_negative_delta_top1", "spearman_metric_vs_negative_delta_top1",
        "pearson_metric_vs_delta_rank_bad", "spearman_metric_vs_delta_rank_bad", "auc_metric_predict_degraded_either",
    ]
    _write_csv(out_dir / "under_assignment_metric_quality.csv", quality, q_fields)

    top_rows: List[Dict[str, Any]] = []
    for metric, _ in METRICS:
        top_rows.extend(_top_by_metric(table, metric, args.top_k, min_support=0.0))
        for r in _top_by_metric(table, metric, args.top_k, min_support=args.support_threshold):
            rr = dict(r)
            rr["metric"] = metric + f"_support_ge_{int(args.support_threshold)}"
            top_rows.append(rr)
    top_fields = ["metric", "metric_rank", "metric_value"] + fields
    _write_csv(out_dir / "top50_under_assigned_by_metric.csv", top_rows, top_fields)

    overlap_anchor: List[Dict[str, Any]] = []
    overlap_degraded: List[Dict[str, Any]] = []
    for metric, _ in METRICS:
        overlap_anchor.extend(_build_overlap(table, metric, args.top_k, args.support_threshold, anchor_only=True))
        overlap_degraded.extend(_build_overlap(table, metric, args.top_k, args.support_threshold, anchor_only=False))
    overlap_fields = [
        "metric", "metric_rank", "raw_id", "class_name", "candidate_support", "responsibility_mass", "top1_count",
        "metric_value", "certificate_family", "certificate_type", "resolved_round", "gt_count", "delta_gt_top1_hit_rate",
        "delta_mean_normalized_gt_rank", "is_nohub_degraded_top1", "is_nohub_degraded_rank",
    ]
    _write_csv(out_dir / "anchor_degraded_overlap.csv", overlap_anchor, overlap_fields)
    _write_csv(out_dir / "nohub_degraded_overlap.csv", overlap_degraded, overlap_fields)

    corr = _correlation_rows(diagnostic_rows)
    _write_csv(out_dir / "pressure_effect_correlation.csv", corr, ["metric", "target", "n", "pearson", "spearman"])

    # A compact recommendation: prefer support-filtered metric with low contamination and high degraded/anchor overlap.
    def quality_score(q: Mapping[str, Any]) -> float:
        return (
            2.0 * _num(q.get("support_filtered_topk_nohub_degraded_either_rate"))
            + 1.5 * _num(q.get("support_filtered_topk_anchor_rate"))
            + 0.5 * max(0.0, _num(q.get("auc_metric_predict_degraded_either")) - 0.5)
            - 2.0 * _num(q.get("topk_low_support_lt10_rate"))
        )
    best = max(quality, key=quality_score) if quality else {}
    interpretation = "INCONCLUSIVE"
    if best:
        if quality_score(best) > 0.5 and _num(best.get("topk_low_support_lt10_rate")) < 0.5:
            interpretation = "UNDER_ASSIGNED_SIGNAL_PROMISING_FOR_REPLAY"
        elif _num(best.get("topk_low_support_lt10_rate")) >= 0.5:
            interpretation = "UNDER_ASSIGNED_SIGNAL_LOW_SUPPORT_CONTAMINATED"
        else:
            interpretation = "UNDER_ASSIGNED_SIGNAL_WEAK_OR_MIXED"

    summary = {
        "status": "PASS",
        "output_dir": str(out_dir),
        "responsibility_meta": resp_meta,
        "compare_dir": str(compare_dir) if compare_dir else "",
        "schedule_csv": str(sched) if sched else "",
        "nohub_run": args.nohub_run,
        "class_count": len(table),
        "diagnostic_class_count": len(diagnostic_rows),
        "min_class_gt_count_for_diagnostics": args.min_class_gt_count,
        "support_threshold": args.support_threshold,
        "mass_floor": args.mass_floor,
        "top_k": args.top_k,
        "best_metric": best.get("metric", ""),
        "best_metric_quality_score": quality_score(best) if best else None,
        "interpretation": interpretation,
        "outputs": {
            "under_assigned_class_table": str(out_dir / "under_assigned_class_table.csv"),
            "under_assignment_metric_quality": str(out_dir / "under_assignment_metric_quality.csv"),
            "top50_under_assigned_by_metric": str(out_dir / "top50_under_assigned_by_metric.csv"),
            "anchor_degraded_overlap": str(out_dir / "anchor_degraded_overlap.csv"),
            "nohub_degraded_overlap": str(out_dir / "nohub_degraded_overlap.csv"),
            "pressure_effect_correlation": str(out_dir / "pressure_effect_correlation.csv"),
        },
    }
    _write_json(out_dir / "summary.json", summary)

    takeover = f"""# Under-assigned Class Audit Takeover

Status: `PASS`

Output: `{out_dir}`

## Scope

Read-only GT-fullY clean nohub demand-side audit. No training, no checkpoint modification, no mAP, no VideoCutLER/Y′/extra.
GT attribution metrics are used only for post-hoc diagnosis and are not used to define under-assignment metrics.

## Inputs

- responsibility_records: `{resp_path}`
- compare_dir: `{compare_dir or ''}`
- schedule_csv: `{sched or ''}`
- nohub_run: `{args.nohub_run}`

## Key findings

- class_count: `{len(table)}`
- diagnostic_class_count: `{len(diagnostic_rows)}`
- best_metric: `{summary['best_metric']}`
- interpretation: `{interpretation}`

## Core outputs

- summary.json
- under_assigned_class_table.csv
- under_assignment_metric_quality.csv
- top50_under_assigned_by_metric.csv
- anchor_degraded_overlap.csv
- nohub_degraded_overlap.csv
- pressure_effect_correlation.csv

## Required follow-up

Proceed to demand-floor replay only if the top under-assigned classes overlap with anchor/nohub-degraded classes without being dominated by low-support noise.
"""
    (out_dir / "UNDER_ASSIGNED_CLASS_AUDIT_TAKEOVER.md").write_text(takeover, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
