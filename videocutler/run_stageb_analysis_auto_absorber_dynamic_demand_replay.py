#!/usr/bin/env python3
"""Read-only support-aware auto-absorber dynamic-demand feasibility replay.

This script does NOT train, does NOT modify checkpoints, and does NOT use GT counts
or GT correctness to define the absorber metric. GT attribution outputs are used only
for post-hoc risk/benefit correlation and group diagnosis.

It consumes the GT-fullY clean E2E nohub absorber logging output plus the clean
attribution compare outputs, then simulates conservative class-level downweight
settings based on an observable absorber metric:

    support_floor_absorber = responsibility_mass_ema / max(label_support_ema, floor)

The simulation is intentionally class-level and diagnostic: it estimates exposure,
risk, and opportunity before implementing any training-time dynamic demand.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


KNOWN_CLASSES_DEFAULT = [
    "person",
    "car_(automobile)",
    "wheel",
    "ball",
    "curtain",
    "short_pants",
    "dress",
    "jacket",
    "necktie",
    "dog",
    "horse",
    "knife",
    "shirt",
    "trousers",
    "shoe",
    "hat",
    "watch",
]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _f(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        val = row.get(key, default)
        if val is None or val == "":
            return default
        return float(val)
    except Exception:
        return default


def _s(row: Mapping[str, Any], key: str, default: str = "") -> str:
    val = row.get(key, default)
    if val is None:
        return default
    return str(val)


def _percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _zscore_map(values_by_key: Mapping[str, float]) -> Dict[str, float]:
    vals = [v for v in values_by_key.values() if math.isfinite(v)]
    if not vals:
        return {k: 0.0 for k in values_by_key}
    mean = statistics.fmean(vals)
    stdev = statistics.pstdev(vals)
    if stdev <= 1e-12:
        return {k: 0.0 for k in values_by_key}
    return {k: (v - mean) / stdev for k, v in values_by_key.items()}


def _minmax_norm(values_by_key: Mapping[str, float]) -> Dict[str, float]:
    vals = [v for v in values_by_key.values() if math.isfinite(v)]
    if not vals:
        return {k: 0.0 for k in values_by_key}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {k: 0.0 for k in values_by_key}
    return {k: max(0.0, min(1.0, (v - lo) / (hi - lo))) for k, v in values_by_key.items()}


def _rank_norm(values_by_key: Mapping[str, float], descending: bool = True) -> Dict[str, float]:
    items = sorted(values_by_key.items(), key=lambda kv: kv[1], reverse=descending)
    n = len(items)
    if n <= 1:
        return {k: 1.0 for k in values_by_key}
    out: Dict[str, float] = {}
    for idx, (k, _v) in enumerate(items):
        # best item gets 1.0, worst gets 0.0
        out[k] = 1.0 - idx / (n - 1)
    return out


def _canonical_id(raw_id: Any) -> str:
    return str(raw_id).strip()


def _load_per_class(per_class_path: Path, baseline_name: str, target_names: Sequence[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    rows = _read_csv(per_class_path)
    by: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        rid = _canonical_id(r.get("raw_id"))
        ck = _s(r, "checkpoint")
        if not rid or not ck:
            continue
        by[rid][ck] = dict(r)
    # Ensure only classes with baseline and at least one target are kept by caller as needed.
    return by


def _build_class_table(
    absorber_rows: Sequence[Mapping[str, str]],
    per_class_by_id: Mapping[str, Mapping[str, Mapping[str, Any]]],
    baseline_name: str,
    target_name: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in absorber_rows:
        rid = _canonical_id(a.get("raw_id"))
        if not rid:
            continue
        pc = per_class_by_id.get(rid, {})
        b = pc.get(baseline_name)
        t = pc.get(target_name)
        # The absorber file may contain classes not present in comparable attribution rows.
        if b is None or t is None:
            continue
        support = _f(a, "label_support_ema")
        mass = _f(a, "responsibility_mass_ema")
        top1_mass = _f(a, "top1_count_ema")
        raw_abs = _f(a, "absorber_score", mass / max(support, 1e-12) if support > 0 else 0.0)
        rec: Dict[str, Any] = {
            "raw_id": rid,
            "class_name": _s(a, "class_name") or _s(t, "class_name") or _s(b, "class_name") or rid,
            "label_support_ema": support,
            "responsibility_mass_ema": mass,
            "top1_count_ema": top1_mass,
            "raw_absorber_score": raw_abs,
            "top1_absorb_score": _f(a, "top1_absorb_score"),
            "baseline_gt_count": _f(b, "gt_count"),
            "target_gt_count": _f(t, "gt_count"),
            "baseline_top1": _f(b, "gt_top1_hit_rate"),
            "target_top1": _f(t, "gt_top1_hit_rate"),
            "delta_top1": _f(t, "gt_top1_hit_rate") - _f(b, "gt_top1_hit_rate"),
            "baseline_mean_normalized_gt_rank": _f(b, "mean_normalized_gt_rank"),
            "target_mean_normalized_gt_rank": _f(t, "mean_normalized_gt_rank"),
            "delta_mean_normalized_gt_rank": _f(t, "mean_normalized_gt_rank") - _f(b, "mean_normalized_gt_rank"),
            "baseline_gt_rank_mean": _f(b, "gt_rank_mean"),
            "target_gt_rank_mean": _f(t, "gt_rank_mean"),
            "delta_gt_rank_mean": _f(t, "gt_rank_mean") - _f(b, "gt_rank_mean"),
            "certificate_family": _s(t, "certificate_family") or _s(b, "certificate_family"),
            "certificate_type": _s(t, "certificate_type") or _s(b, "certificate_type"),
            "resolved_round": _s(t, "resolved_round") or _s(b, "resolved_round"),
            "base_group": _s(t, "base_group") or _s(b, "base_group"),
            "person_conditioned": _s(t, "person_conditioned") or _s(b, "person_conditioned"),
        }
        out.append(rec)
    return out


def _add_metric_columns(rows: List[Dict[str, Any]], floors: Sequence[float]) -> None:
    by_id = {r["raw_id"]: r for r in rows}
    raw_abs = {rid: float(r["raw_absorber_score"]) for rid, r in by_id.items()}
    mass = {rid: float(r["responsibility_mass_ema"]) for rid, r in by_id.items()}
    top1 = {rid: float(r["top1_count_ema"]) for rid, r in by_id.items()}
    support = {rid: float(r["label_support_ema"]) for rid, r in by_id.items()}

    # clipped raw absorber for hybrid stability
    p95 = _percentile(list(raw_abs.values()), 0.95)
    clipped_abs = {rid: min(v, p95) for rid, v in raw_abs.items()}

    z_mass = _zscore_map({rid: math.log1p(max(0.0, v)) for rid, v in mass.items()})
    z_top1 = _zscore_map({rid: math.log1p(max(0.0, v)) for rid, v in top1.items()})
    z_support = _zscore_map({rid: math.log1p(max(0.0, v)) for rid, v in support.items()})
    z_abs_clip = _zscore_map(clipped_abs)

    rn_mass = _rank_norm(mass)
    rn_top1 = _rank_norm(top1)
    # support>=10 absorber, classes below threshold receive 0 for this component.
    sf10 = {rid: (raw_abs[rid] if support[rid] >= 10.0 else 0.0) for rid in by_id}
    rn_sf10 = _rank_norm(sf10)

    for r in rows:
        rid = r["raw_id"]
        sup = float(r["label_support_ema"])
        resp = float(r["responsibility_mass_ema"])
        r["metric_raw_absorber_score"] = float(r["raw_absorber_score"])
        r["metric_mass_rank"] = resp
        r["metric_top1_mass_rank"] = float(r["top1_count_ema"])
        r["metric_log_support_weighted_absorber"] = float(r["raw_absorber_score"]) * math.log1p(max(0.0, sup))
        r["metric_hybrid_absorber_score"] = z_mass[rid] + z_top1[rid] + z_support[rid] + z_abs_clip[rid]
        r["metric_high_mass_high_ratio_score"] = rn_mass[rid] + rn_top1[rid] + rn_sf10[rid]
        for fl in floors:
            key = f"metric_support_floor_absorber_score_floor_{int(fl)}"
            r[key] = resp / max(sup, fl)
        for th in (5, 10, 20, 50):
            key = f"metric_support_filtered_absorber_score_ge_{th}"
            r[key] = float(r["raw_absorber_score"]) if sup >= th else 0.0


def _metric_names(floors: Sequence[float]) -> List[str]:
    names = [
        "metric_raw_absorber_score",
        "metric_mass_rank",
        "metric_top1_mass_rank",
        "metric_log_support_weighted_absorber",
        "metric_hybrid_absorber_score",
        "metric_high_mass_high_ratio_score",
    ]
    names += [f"metric_support_floor_absorber_score_floor_{int(fl)}" for fl in floors]
    names += [f"metric_support_filtered_absorber_score_ge_{th}" for th in (5, 10, 20, 50)]
    return names


def _normalize_for_downweight(rows: Sequence[Mapping[str, Any]], metric: str, clip_q: float = 0.95) -> Dict[str, float]:
    vals = {str(r["raw_id"]): float(r.get(metric, 0.0) or 0.0) for r in rows}
    clip = _percentile(list(vals.values()), clip_q)
    if clip <= 1e-12:
        return {rid: 0.0 for rid in vals}
    return {rid: max(0.0, min(1.0, v / clip)) for rid, v in vals.items()}


def _summarize_metric(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    known_classes: Sequence[str],
    high_mass_top50_ids: set,
    improved_top50_ids: set,
    degraded_top50_ids: set,
) -> Dict[str, Any]:
    ranked = sorted(rows, key=lambda r: float(r.get(metric, 0.0) or 0.0), reverse=True)
    top50 = ranked[:50]
    top100 = ranked[:100]
    top50_ids = {str(r["raw_id"]) for r in top50}
    top100_names = {str(r.get("class_name", "")) for r in top100}
    low5 = sum(1 for r in top50 if float(r.get("label_support_ema", 0.0) or 0.0) < 5.0) / max(1, len(top50))
    low10 = sum(1 for r in top50 if float(r.get("label_support_ema", 0.0) or 0.0) < 10.0) / max(1, len(top50))
    known_top50 = [c for c in known_classes if c in {str(r.get("class_name", "")) for r in top50}]
    known_top100 = [c for c in known_classes if c in top100_names]
    overlap_high_mass = len(top50_ids & high_mass_top50_ids)
    overlap_improved = len(top50_ids & improved_top50_ids)
    overlap_degraded = len(top50_ids & degraded_top50_ids)
    # Heuristic quality score used only for metric selection, not for training.
    quality = (
        overlap_high_mass
        + 2.0 * len(known_top50)
        + 1.0 * len(known_top100)
        - 50.0 * low5
        - 25.0 * low10
    )
    return {
        "metric": metric.replace("metric_", ""),
        "quality_score": quality,
        "top50_low_support_lt5_rate": low5,
        "top50_low_support_lt10_rate": low10,
        "known_classes_top50_count": len(known_top50),
        "known_classes_top100_count": len(known_top100),
        "known_classes_top50": ";".join(known_top50),
        "known_classes_top100": ";".join(known_top100),
        "overlap_high_mass_top50": overlap_high_mass,
        "overlap_improved_top50": overlap_improved,
        "overlap_degraded_top50": overlap_degraded,
    }


def _aggregate_group_exposure(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    alpha: float,
    min_class_weight: float,
    group_name: str,
    norm: Mapping[str, float],
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(group_name, "unknown") or "unknown")].append(r)
    out: List[Dict[str, Any]] = []
    for gval, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        gt = sum(float(r.get("target_gt_count", 0.0) or 0.0) for r in rs)
        if gt <= 0:
            continue
        weight_sum = 0.0
        pressure_sum = 0.0
        risk = 0.0
        opportunity = 0.0
        delta_top1_weighted = 0.0
        delta_rank_weighted = 0.0
        for r in rs:
            rid = str(r["raw_id"])
            cnt = float(r.get("target_gt_count", 0.0) or 0.0)
            pressure = norm.get(rid, 0.0)
            cw = max(min_class_weight, 1.0 - alpha * pressure)
            down = 1.0 - cw
            dt = float(r.get("delta_top1", 0.0) or 0.0)
            dr = float(r.get("delta_mean_normalized_gt_rank", 0.0) or 0.0)
            weight_sum += cnt * cw
            pressure_sum += cnt * pressure
            # Risk: suppressing classes that nohub already improved.
            risk += cnt * down * max(0.0, dt)
            # Opportunity: suppressing classes that nohub degraded.
            opportunity += cnt * down * max(0.0, -dt)
            delta_top1_weighted += cnt * dt
            delta_rank_weighted += cnt * dr
        out.append(
            {
                "metric": metric.replace("metric_", ""),
                "alpha": alpha,
                "min_class_weight": min_class_weight,
                "group_name": group_name,
                "group_value": gval,
                "class_count": len(rs),
                "gt_count": gt,
                "mean_normalized_pressure": pressure_sum / gt,
                "mean_class_weight": weight_sum / gt,
                "baseline_to_nohub_delta_top1_weighted": delta_top1_weighted / gt,
                "baseline_to_nohub_delta_rank_weighted": delta_rank_weighted / gt,
                "suppression_risk_positive_delta": risk,
                "suppression_opportunity_negative_delta": opportunity,
                "opportunity_minus_risk": opportunity - risk,
            }
        )
    return out


def _simulate_settings(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    alphas: Sequence[float],
    min_weights: Sequence[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    norm = _normalize_for_downweight(rows, metric)
    summary_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        for mw in min_weights:
            class_weights = []
            risk = 0.0
            opportunity = 0.0
            weighted_pressure = 0.0
            gt_total = 0.0
            down_gt = 0.0
            for r in rows:
                rid = str(r["raw_id"])
                cnt = float(r.get("target_gt_count", 0.0) or 0.0)
                pressure = norm.get(rid, 0.0)
                cw = max(mw, 1.0 - alpha * pressure)
                down = 1.0 - cw
                dt = float(r.get("delta_top1", 0.0) or 0.0)
                class_weights.append(cw)
                gt_total += cnt
                weighted_pressure += cnt * pressure
                down_gt += cnt * down
                risk += cnt * down * max(0.0, dt)
                opportunity += cnt * down * max(0.0, -dt)
            summary_rows.append(
                {
                    "metric": metric.replace("metric_", ""),
                    "alpha": alpha,
                    "min_class_weight": mw,
                    "class_count": len(rows),
                    "gt_count": gt_total,
                    "mean_class_weight": statistics.fmean(class_weights) if class_weights else 1.0,
                    "min_actual_class_weight": min(class_weights) if class_weights else 1.0,
                    "gt_weighted_pressure_mean": weighted_pressure / max(gt_total, 1e-12),
                    "gt_weighted_downweight_mean": down_gt / max(gt_total, 1e-12),
                    "suppression_risk_positive_delta": risk,
                    "suppression_opportunity_negative_delta": opportunity,
                    "opportunity_minus_risk": opportunity - risk,
                }
            )
            for g in ("certificate_family", "certificate_type", "resolved_round", "base_group", "person_conditioned"):
                group_rows.extend(_aggregate_group_exposure(rows, metric, alpha, mw, g, norm))
    return summary_rows, group_rows


def _write_markdown(
    path: Path,
    summary: Mapping[str, Any],
    best_metric: str,
    best_setting: Mapping[str, Any],
    metric_quality: Sequence[Mapping[str, Any]],
    top_recommended: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Auto Absorber Dynamic Demand Replay Takeover")
    lines.append("")
    lines.append(f"Status: `{summary.get('status')}`")
    lines.append(f"Output: `{summary.get('output_dir')}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Read-only class-level feasibility replay. No training, no checkpoint modification, no VideoCutLER/Y′/extra/mAP.")
    lines.append("The absorber metric uses only observable training statistics; GT attribution deltas are used only for post-hoc risk analysis.")
    lines.append("")
    lines.append("## Recommended metric")
    lines.append("")
    lines.append(f"Recommended absorber metric: `{best_metric.replace('metric_', '')}`")
    lines.append("")
    lines.append("## Recommended weak setting")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dict(best_setting), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Top metric quality rows")
    lines.append("")
    lines.append("| metric | quality | low<5 | low<10 | known@50 | known@100 | high-mass overlap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in metric_quality[:8]:
        lines.append(
            f"| {r.get('metric')} | {float(r.get('quality_score', 0)):.3f} | "
            f"{float(r.get('top50_low_support_lt5_rate', 0)):.3f} | "
            f"{float(r.get('top50_low_support_lt10_rate', 0)):.3f} | "
            f"{r.get('known_classes_top50_count')} | {r.get('known_classes_top100_count')} | {r.get('overlap_high_mass_top50')} |"
        )
    lines.append("")
    lines.append("## Top classes under recommended metric")
    lines.append("")
    lines.append("| rank | raw_id | class | support | mass | top1_mass | metric | delta_top1 |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|")
    for idx, r in enumerate(top_recommended[:20], start=1):
        metric_name = best_metric
        lines.append(
            f"| {idx} | {r.get('raw_id')} | {r.get('class_name')} | "
            f"{float(r.get('label_support_ema', 0)):.3f} | {float(r.get('responsibility_mass_ema', 0)):.3f} | "
            f"{float(r.get('top1_count_ema', 0)):.3f} | {float(r.get(metric_name, 0)):.3f} | "
            f"{float(r.get('delta_top1', 0)):.4f} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(str(summary.get("decision", "")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--absorber_csv", type=Path, required=True)
    ap.add_argument("--compare_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--baseline", default="baseline_full_y")
    ap.add_argument("--target", default="soft_e2e_nohub")
    ap.add_argument("--support_floors", default="5,10,20")
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--min_class_weights", default="0.7,0.8,0.9")
    ap.add_argument("--recommended_metric", default="support_floor_absorber_score_floor_20")
    ap.add_argument("--known_classes", default=",".join(KNOWN_CLASSES_DEFAULT))
    args = ap.parse_args(argv)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    floors = [float(x) for x in args.support_floors.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    min_weights = [float(x) for x in args.min_class_weights.split(",") if x.strip()]
    known_classes = [x.strip() for x in args.known_classes.split(",") if x.strip()]

    absorber_rows = _read_csv(args.absorber_csv)
    per_class_path = args.compare_dir / "per_class_attribution.csv"
    per_class_by = _load_per_class(per_class_path, args.baseline, [args.target])
    rows = _build_class_table(absorber_rows, per_class_by, args.baseline, args.target)
    if not rows:
        raise SystemExit("No paired absorber/per-class rows found. Check --baseline/--target and input schemas.")

    _add_metric_columns(rows, floors)
    metric_names = _metric_names(floors)

    high_mass_top50_ids = {str(r["raw_id"]) for r in sorted(rows, key=lambda r: float(r.get("responsibility_mass_ema", 0.0) or 0.0), reverse=True)[:50]}
    improved_top50_ids = {str(r["raw_id"]) for r in sorted(rows, key=lambda r: float(r.get("delta_top1", 0.0) or 0.0), reverse=True)[:50]}
    degraded_top50_ids = {str(r["raw_id"]) for r in sorted(rows, key=lambda r: float(r.get("delta_top1", 0.0) or 0.0))[:50]}

    metric_quality = [
        _summarize_metric(rows, m, known_classes, high_mass_top50_ids, improved_top50_ids, degraded_top50_ids)
        for m in metric_names
    ]
    metric_quality = sorted(metric_quality, key=lambda r: float(r.get("quality_score", 0.0) or 0.0), reverse=True)

    rec_metric = "metric_" + args.recommended_metric if not args.recommended_metric.startswith("metric_") else args.recommended_metric
    if rec_metric not in metric_names:
        # fall back to best metric if the requested one is not present.
        rec_metric = "metric_" + str(metric_quality[0]["metric"])
    rec_ranked = sorted(rows, key=lambda r: float(r.get(rec_metric, 0.0) or 0.0), reverse=True)

    all_summary_rows: List[Dict[str, Any]] = []
    all_group_rows: List[Dict[str, Any]] = []
    for metric in [rec_metric]:
        sr, gr = _simulate_settings(rows, metric, alphas, min_weights)
        all_summary_rows.extend(sr)
        all_group_rows.extend(gr)

    # Choose a conservative setting: prefer alpha=0.05/min_weight=0.9 if present; otherwise best opportunity-risk.
    preferred = [r for r in all_summary_rows if abs(float(r["alpha"]) - 0.05) < 1e-9 and abs(float(r["min_class_weight"]) - 0.9) < 1e-9]
    best_setting = preferred[0] if preferred else sorted(all_summary_rows, key=lambda r: float(r["opportunity_minus_risk"]), reverse=True)[0]

    class_fields = [
        "raw_id", "class_name", "label_support_ema", "responsibility_mass_ema", "top1_count_ema",
        "raw_absorber_score", "top1_absorb_score", "baseline_gt_count", "target_gt_count",
        "baseline_top1", "target_top1", "delta_top1", "baseline_mean_normalized_gt_rank",
        "target_mean_normalized_gt_rank", "delta_mean_normalized_gt_rank", "certificate_family",
        "certificate_type", "resolved_round", "base_group", "person_conditioned",
    ] + metric_names
    _write_csv(out / "class_absorber_replay_table.csv", rows, class_fields)

    _write_csv(
        out / "metric_quality_summary.csv",
        metric_quality,
        [
            "metric", "quality_score", "top50_low_support_lt5_rate", "top50_low_support_lt10_rate",
            "known_classes_top50_count", "known_classes_top100_count", "known_classes_top50",
            "known_classes_top100", "overlap_high_mass_top50", "overlap_improved_top50", "overlap_degraded_top50",
        ],
    )

    top50_rows: List[Dict[str, Any]] = []
    for m in metric_names:
        for idx, r in enumerate(sorted(rows, key=lambda rr: float(rr.get(m, 0.0) or 0.0), reverse=True)[:50], start=1):
            top50_rows.append({
                "metric": m.replace("metric_", ""),
                "rank": idx,
                "raw_id": r.get("raw_id"),
                "class_name": r.get("class_name"),
                "label_support_ema": r.get("label_support_ema"),
                "responsibility_mass_ema": r.get("responsibility_mass_ema"),
                "top1_count_ema": r.get("top1_count_ema"),
                "metric_value": r.get(m),
                "delta_top1": r.get("delta_top1"),
                "delta_mean_normalized_gt_rank": r.get("delta_mean_normalized_gt_rank"),
                "certificate_family": r.get("certificate_family"),
            })
    _write_csv(
        out / "top50_by_metric.csv",
        top50_rows,
        ["metric", "rank", "raw_id", "class_name", "label_support_ema", "responsibility_mass_ema", "top1_count_ema", "metric_value", "delta_top1", "delta_mean_normalized_gt_rank", "certificate_family"],
    )

    known_rank_rows: List[Dict[str, Any]] = []
    for m in metric_names:
        ranked = sorted(rows, key=lambda rr: float(rr.get(m, 0.0) or 0.0), reverse=True)
        for k in known_classes:
            for idx, r in enumerate(ranked, start=1):
                if str(r.get("class_name")) == k:
                    known_rank_rows.append({
                        "metric": m.replace("metric_", ""),
                        "class_name": k,
                        "rank": idx,
                        "raw_id": r.get("raw_id"),
                        "label_support_ema": r.get("label_support_ema"),
                        "responsibility_mass_ema": r.get("responsibility_mass_ema"),
                        "top1_count_ema": r.get("top1_count_ema"),
                        "metric_value": r.get(m),
                    })
                    break
    _write_csv(out / "selected_known_class_ranks.csv", known_rank_rows, ["metric", "class_name", "rank", "raw_id", "label_support_ema", "responsibility_mass_ema", "top1_count_ema", "metric_value"])

    low_support_rows = [
        {
            "metric": r["metric"],
            "top50_low_support_lt5_rate": r["top50_low_support_lt5_rate"],
            "top50_low_support_lt10_rate": r["top50_low_support_lt10_rate"],
        }
        for r in metric_quality
    ]
    _write_csv(out / "low_support_contamination_by_metric.csv", low_support_rows, ["metric", "top50_low_support_lt5_rate", "top50_low_support_lt10_rate"])

    overlap_rows = [
        {
            "metric": r["metric"],
            "overlap_high_mass_top50": r["overlap_high_mass_top50"],
            "overlap_improved_top50": r["overlap_improved_top50"],
            "overlap_degraded_top50": r["overlap_degraded_top50"],
            "known_classes_top50": r["known_classes_top50"],
            "known_classes_top100": r["known_classes_top100"],
        }
        for r in metric_quality
    ]
    _write_csv(out / "metric_overlap_summary.csv", overlap_rows, ["metric", "overlap_high_mass_top50", "overlap_improved_top50", "overlap_degraded_top50", "known_classes_top50", "known_classes_top100"])

    _write_csv(
        out / "summary_by_alpha.csv",
        all_summary_rows,
        ["metric", "alpha", "min_class_weight", "class_count", "gt_count", "mean_class_weight", "min_actual_class_weight", "gt_weighted_pressure_mean", "gt_weighted_downweight_mean", "suppression_risk_positive_delta", "suppression_opportunity_negative_delta", "opportunity_minus_risk"],
    )
    _write_csv(
        out / "summary_by_group.csv",
        all_group_rows,
        ["metric", "alpha", "min_class_weight", "group_name", "group_value", "class_count", "gt_count", "mean_normalized_pressure", "mean_class_weight", "baseline_to_nohub_delta_top1_weighted", "baseline_to_nohub_delta_rank_weighted", "suppression_risk_positive_delta", "suppression_opportunity_negative_delta", "opportunity_minus_risk"],
    )
    # Alias requested by plan.
    _write_csv(
        out / "summary_delta_vs_nohub.csv",
        all_group_rows,
        ["metric", "alpha", "min_class_weight", "group_name", "group_value", "class_count", "gt_count", "mean_normalized_pressure", "mean_class_weight", "suppression_risk_positive_delta", "suppression_opportunity_negative_delta", "opportunity_minus_risk"],
    )

    # Top classes by opportunity/risk under the conservative setting.
    norm = _normalize_for_downweight(rows, rec_metric)
    alpha = float(best_setting["alpha"])
    mw = float(best_setting["min_class_weight"])
    scored_classes: List[Dict[str, Any]] = []
    for r in rows:
        pressure = norm[str(r["raw_id"])]
        cw = max(mw, 1.0 - alpha * pressure)
        down = 1.0 - cw
        cnt = float(r.get("target_gt_count", 0.0) or 0.0)
        dt = float(r.get("delta_top1", 0.0) or 0.0)
        rr = dict(r)
        rr.update({
            "recommended_metric": rec_metric.replace("metric_", ""),
            "normalized_pressure": pressure,
            "class_weight": cw,
            "downweight": down,
            "opportunity_score": cnt * down * max(0.0, -dt),
            "risk_score": cnt * down * max(0.0, dt),
        })
        scored_classes.append(rr)
    improved = sorted(scored_classes, key=lambda r: float(r.get("opportunity_score", 0.0)), reverse=True)[:20]
    degraded = sorted(scored_classes, key=lambda r: float(r.get("risk_score", 0.0)), reverse=True)[:20]
    top_fields = ["raw_id", "class_name", "target_gt_count", "label_support_ema", "responsibility_mass_ema", "top1_count_ema", "recommended_metric", "normalized_pressure", "class_weight", "downweight", "delta_top1", "delta_mean_normalized_gt_rank", "opportunity_score", "risk_score", "certificate_family", "certificate_type", "resolved_round"]
    _write_csv(out / "top20_improved_classes.csv", improved, top_fields)
    _write_csv(out / "top20_degraded_classes.csv", degraded, top_fields)

    best_metric_name = rec_metric.replace("metric_", "")
    best_quality_row = next((r for r in metric_quality if r["metric"] == best_metric_name), metric_quality[0])
    low5 = float(best_quality_row["top50_low_support_lt5_rate"])
    low10 = float(best_quality_row["top50_low_support_lt10_rate"])
    # Conservative decision.
    if low5 <= 0.05 and low10 <= 0.10 and int(best_quality_row["known_classes_top100_count"]) >= 10:
        decision = "B. auto absorber dynamic demand is promising, but only as a weak, support-aware, stop-gradient class-weight replay/training candidate. Start with alpha=0.05 and min_class_weight=0.9."
    elif low5 > 0.25 or low10 > 0.35:
        decision = "C. absorber statistics are too noisy; do not use dynamic demand yet."
    else:
        decision = "E. mixed / inconclusive; keep row-level nohub soft routing and run additional support-aware diagnostics before training."

    summary: Dict[str, Any] = {
        "status": "PASS",
        "output_dir": str(out),
        "absorber_csv": str(args.absorber_csv),
        "compare_dir": str(args.compare_dir),
        "baseline": args.baseline,
        "target": args.target,
        "paired_class_count": len(rows),
        "recommended_metric": best_metric_name,
        "recommended_setting": dict(best_setting),
        "best_metric_quality": best_quality_row,
        "decision": decision,
        "outputs": {
            "summary_json": str(out / "summary.json"),
            "metric_quality_summary_csv": str(out / "metric_quality_summary.csv"),
            "top50_by_metric_csv": str(out / "top50_by_metric.csv"),
            "selected_known_class_ranks_csv": str(out / "selected_known_class_ranks.csv"),
            "summary_by_alpha_csv": str(out / "summary_by_alpha.csv"),
            "summary_by_group_csv": str(out / "summary_by_group.csv"),
            "top20_improved_classes_csv": str(out / "top20_improved_classes.csv"),
            "top20_degraded_classes_csv": str(out / "top20_degraded_classes.csv"),
            "takeover_md": str(out / "AUTO_ABSORBER_DYNAMIC_DEMAND_REPLAY_TAKEOVER.md"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rec_md = out / "recommended_absorber_metric.md"
    rec_lines = [
        "# Recommended absorber metric",
        "",
        f"Recommended metric: `{best_metric_name}`",
        "",
        "Recommended conservative training candidate if approved:",
        "",
        "```text",
        f"metric = {best_metric_name}",
        f"alpha = {best_setting.get('alpha')}",
        f"min_class_weight = {best_setting.get('min_class_weight')}",
        "warmup_epochs >= 3",
        "stop_gradient = true",
        "```",
        "",
        "This recommendation is based on observable statistics only. GT attribution deltas were used only for risk analysis.",
        "",
        f"Decision: {decision}",
    ]
    rec_md.write_text("\n".join(rec_lines) + "\n", encoding="utf-8")

    _write_markdown(
        out / "AUTO_ABSORBER_DYNAMIC_DEMAND_REPLAY_TAKEOVER.md",
        summary,
        rec_metric,
        best_setting,
        metric_quality,
        rec_ranked,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
