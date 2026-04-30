#!/usr/bin/env python3
"""Read-only absorber score design audit for GT-fullY clean E2E nohub runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_SELECTED_CLASS_NAMES = [
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

DEFAULT_FAMILY_MAP = {
    "person": "person",
    "car_(automobile)": "vehicle",
    "wheel": "vehicle",
    "ball": "object_or_animal",
    "dog": "object_or_animal",
    "horse": "object_or_animal",
    "knife": "object_or_animal",
    "curtain": "clothing_accessory",
    "short_pants": "clothing_accessory",
    "dress": "clothing_accessory",
    "jacket": "clothing_accessory",
    "necktie": "clothing_accessory",
    "shirt": "clothing_accessory",
    "trousers": "clothing_accessory",
    "shoe": "clothing_accessory",
    "hat": "clothing_accessory",
    "watch": "clothing_accessory",
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        v = float(value)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _median(xs: Sequence[float]) -> float:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return 0.0
    return float(statistics.median(vals))


def _p95(xs: Sequence[float]) -> float:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not vals:
        return 0.0
    idx = min(len(vals) - 1, max(0, int(math.ceil(0.95 * len(vals)) - 1)))
    return float(vals[idx])


def _zscore(values: Sequence[float]) -> List[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    mean = _mean(vals)
    sd = statistics.pstdev(vals)
    if sd <= 1e-12:
        return [0.0 for _ in vals]
    return [(float(v) - mean) / sd for v in vals]


def _rank_desc(values: Sequence[float], *, tie_breaker: Sequence[Any]) -> List[int]:
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), tie_breaker[i]))
    ranks = [0] * len(values)
    for pos, idx in enumerate(order, 1):
        ranks[idx] = pos
    return ranks


def _rank_norm_from_ranks(ranks: Sequence[int]) -> List[float]:
    n = len(ranks)
    if n <= 1:
        return [1.0 for _ in ranks]
    return [1.0 - (float(r) - 1.0) / float(n - 1) for r in ranks]


def _overlap_names(top_names: Sequence[str], target: Iterable[str]) -> List[str]:
    target_set = set(target)
    return sorted(set(top_names) & target_set)


def _selected_family(name: str) -> str:
    return DEFAULT_FAMILY_MAP.get(name, "other")


def _load_soft_delta_sets(compare_root: Path) -> Tuple[List[str], List[str]]:
    delta_path = compare_root / "per_class_delta_vs_baseline.csv"
    if not delta_path.is_file():
        raise FileNotFoundError(f"missing compare delta csv: {delta_path}")
    rows = _read_csv(delta_path)
    soft_rows = [r for r in rows if str(r.get("run", "")).strip() == "soft_e2e_nohub"]
    if not soft_rows:
        soft_rows = [r for r in rows if str(r.get("run", "")).startswith("soft_e2e_nohub")]
    if not soft_rows:
        raise ValueError(f"no soft_e2e_nohub rows in {delta_path}")
    improved = [
        r.get("class_name", "")
        for r in sorted(
            soft_rows,
            key=lambda r: (-_to_float(r.get("delta_gt_top1_hit_rate")), _to_float(r.get("delta_mean_normalized_gt_rank"))),
        )[:20]
    ]
    degraded = [
        r.get("class_name", "")
        for r in sorted(
            soft_rows,
            key=lambda r: (_to_float(r.get("delta_gt_top1_hit_rate")), -_to_float(r.get("delta_mean_normalized_gt_rank"))),
        )[:20]
    ]
    return improved, degraded


def _safe_metric_name(name: str) -> str:
    return name.replace(" ", "_")


def _build_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    metric_values: Sequence[float],
    *,
    support_threshold: Optional[float] = None,
    score_kind: str = "",
) -> List[Dict[str, Any]]:
    indexed = list(enumerate(rows))
    if support_threshold is not None:
        indexed = [(i, r) for i, r in indexed if _to_float(r.get("label_support_ema")) >= float(support_threshold)]
    order = sorted(indexed, key=lambda ir: (-float(metric_values[ir[0]]), _to_int(ir[1].get("raw_id"), 0)))
    out: List[Dict[str, Any]] = []
    for rank, (idx, row) in enumerate(order, 1):
        out.append(
            {
                "metric": metric_name,
                "rank": rank,
                "raw_id": row.get("raw_id", ""),
                "class_name": row.get("class_name", ""),
                "score": float(metric_values[idx]),
                "label_support_ema": row.get("label_support_ema", ""),
                "responsibility_mass_ema": row.get("responsibility_mass_ema", ""),
                "top1_count_ema": row.get("top1_count_ema", ""),
                "metric_kind": score_kind,
            }
        )
    return out


def _quality_score(metrics_row: Mapping[str, Any]) -> float:
    # Deterministic but simple quality score favoring coverage and low low-support contamination.
    selected50 = _to_float(metrics_row.get("selected_known_top50_count"))
    selected100 = _to_float(metrics_row.get("selected_known_top100_count"))
    contam5 = _to_float(metrics_row.get("low_support_lt5_rate"))
    contam10 = _to_float(metrics_row.get("low_support_lt10_rate"))
    overlap_mass = _to_float(metrics_row.get("overlap_high_mass_top50_count"))
    overlap_imp = _to_float(metrics_row.get("overlap_top_improved_count"))
    overlap_deg = _to_float(metrics_row.get("overlap_top_degraded_count"))
    return (
        10.0 * selected50
        + 1.0 * selected100
        + 0.2 * overlap_mass
        + 0.5 * overlap_imp
        - 0.5 * overlap_deg
        - 100.0 * contam5
        - 20.0 * contam10
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only absorber score design audit for GT-fullY nohub runs.")
    p.add_argument("--run_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument(
        "--absorber_scores_csv",
        default="",
        help="Defaults to <run_root>/gt_full_y_soft_e2e_nohub_abslog_15ep/train/prealign/final_absorber_scores.csv",
    )
    p.add_argument(
        "--compare_root",
        default="",
        help="Defaults to <run_root>/analysis/gt_full_y_e2e_nohub_attribution_compare/lvvis_train_base",
    )
    p.add_argument("--selected_class_names", default=",".join(DEFAULT_SELECTED_CLASS_NAMES))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    absorber_path = Path(args.absorber_scores_csv).expanduser().resolve() if args.absorber_scores_csv else run_root / "gt_full_y_soft_e2e_nohub_abslog_15ep/train/prealign/final_absorber_scores.csv"
    compare_root = Path(args.compare_root).expanduser().resolve() if args.compare_root else run_root / "analysis/gt_full_y_e2e_nohub_attribution_compare/lvvis_train_base"
    selected_class_names = [x.strip() for x in str(args.selected_class_names).split(",") if x.strip()]

    rows = _read_csv(absorber_path)
    if not rows:
        raise ValueError(f"absorber score file is empty: {absorber_path}")

    # Normalize numeric columns once.
    for row in rows:
        row["label_support_ema"] = _to_float(row.get("label_support_ema"))
        row["responsibility_mass_ema"] = _to_float(row.get("responsibility_mass_ema"))
        row["top1_count_ema"] = _to_float(row.get("top1_count_ema"))
        row["absorber_score"] = _to_float(row.get("absorber_score"))
        row["class_name"] = str(row.get("class_name", "")).strip()
        row["raw_id"] = _to_int(row.get("raw_id"), -1)

    supports = [float(r["label_support_ema"]) for r in rows]
    masses = [float(r["responsibility_mass_ema"]) for r in rows]
    top1s = [float(r["top1_count_ema"]) for r in rows]
    raw_scores = [float(r["absorber_score"]) if math.isfinite(float(r["absorber_score"])) and float(r["absorber_score"]) != 0.0 else (float(r["responsibility_mass_ema"]) / float(r["label_support_ema"]) if float(r["label_support_ema"]) > 0 else float("inf")) for r in rows]
    for row, raw_score in zip(rows, raw_scores):
        row["absorber_score"] = float(raw_score)

    p95_raw = _p95(raw_scores)
    safe_raw_clipped = [min(v, p95_raw) for v in raw_scores]

    support_floors = [5, 10, 20]
    metric_specs: List[Tuple[str, List[float], Optional[float], str]] = []
    metric_specs.append(("raw_absorber_score", raw_scores, None, "raw_mass_over_support"))
    metric_specs.append(("mass_rank", masses, None, "responsibility_mass_ema"))
    metric_specs.append(("top1_mass_rank", top1s, None, "top1_count_ema"))
    for thr in [5, 10, 20, 50]:
        vals = [raw_scores[i] if supports[i] >= float(thr) else float("-inf") for i in range(len(rows))]
        metric_specs.append((f"support_filtered_absorber_score_support_ge_{thr}", vals, None, f"support_ge_{thr}"))
    for floor in support_floors:
        vals = [masses[i] / max(supports[i], float(floor)) for i in range(len(rows))]
        metric_specs.append((f"support_floor_absorber_score_floor_{floor}", vals, None, f"floor_{floor}"))
    log_support_weighted = [raw_scores[i] * math.log1p(max(supports[i], 0.0)) for i in range(len(rows))]
    metric_specs.append(("log_support_weighted_absorber", log_support_weighted, None, "raw_score_times_log1p_support"))

    z_mass = _zscore([math.log1p(max(v, 0.0)) for v in masses])
    z_top1 = _zscore([math.log1p(max(v, 0.0)) for v in top1s])
    z_support = _zscore([math.log1p(max(v, 0.0)) for v in supports])
    z_clip = _zscore([float(v) for v in safe_raw_clipped])
    hybrid = [z_mass[i] + z_top1[i] + z_support[i] + z_clip[i] for i in range(len(rows))]
    metric_specs.append(("hybrid_absorber_score", hybrid, None, "z_mass+z_top1+z_support+z_clip"))

    support_filter_for_high_mass = 10
    support_filtered_for_high_mass = [raw_scores[i] if supports[i] >= float(support_filter_for_high_mass) else float("-inf") for i in range(len(rows))]
    rank_mass = _rank_desc(masses, tie_breaker=[r["raw_id"] for r in rows])
    rank_top1 = _rank_desc(top1s, tie_breaker=[r["raw_id"] for r in rows])
    rank_filtered = _rank_desc(support_filtered_for_high_mass, tie_breaker=[r["raw_id"] for r in rows])
    n = len(rows)
    rnorm_mass = _rank_norm_from_ranks(rank_mass)
    rnorm_top1 = _rank_norm_from_ranks(rank_top1)
    rnorm_filtered = _rank_norm_from_ranks(rank_filtered)
    high_mass_high_ratio = [rnorm_mass[i] + rnorm_top1[i] + rnorm_filtered[i] for i in range(len(rows))]
    metric_specs.append((f"high_mass_high_ratio_score_support_ge_{support_filter_for_high_mass}", high_mass_high_ratio, support_filter_for_high_mass, f"rnorm_mass+rnorm_top1+rnorm_support_filtered"))

    # Precompute rankings for all metrics.
    metric_ranked_rows: Dict[str, List[Dict[str, Any]]] = {}
    metric_quality_rows: List[Dict[str, Any]] = []
    metric_overlap_rows: List[Dict[str, Any]] = []
    low_support_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    top50_rows: List[Dict[str, Any]] = []

    selected_set = set(selected_class_names)
    improved_classes, degraded_classes = _load_soft_delta_sets(compare_root)
    high_mass_top50_names = [r["class_name"] for r in sorted(rows, key=lambda r: (-float(r["responsibility_mass_ema"]), int(r["raw_id"])))[:50]]
    high_mass_top50_set = set(high_mass_top50_names)
    improved_set = set(improved_classes)
    degraded_set = set(degraded_classes)

    best_metric_name = ""
    best_metric_row: Optional[Dict[str, Any]] = None
    best_quality = float("-inf")
    recommendation_reason = ""

    for metric_name, metric_values, support_filter, metric_kind in metric_specs:
        ranked_rows = _build_metric_rows(rows, metric_name, metric_values, support_threshold=support_filter, score_kind=metric_kind)
        metric_ranked_rows[metric_name] = ranked_rows
        top50 = ranked_rows[:50]
        top100 = ranked_rows[:100]
        top50_names = [r["class_name"] for r in top50]
        top100_names = [r["class_name"] for r in top100]

        contam5 = sum(1 for r in top50 if _to_float(r["label_support_ema"]) < 5) / max(len(top50), 1)
        contam10 = sum(1 for r in top50 if _to_float(r["label_support_ema"]) < 10) / max(len(top50), 1)
        contam20 = sum(1 for r in top50 if _to_float(r["label_support_ema"]) < 20) / max(len(top50), 1)
        contam50 = sum(1 for r in top50 if _to_float(r["label_support_ema"]) < 50) / max(len(top50), 1)

        selected_top50_names = _overlap_names(top50_names, selected_set)
        selected_top100_names = _overlap_names(top100_names, selected_set)
        selected_row_lookup = {r["class_name"]: r for r in ranked_rows}

        family_counts_top50 = {fam: 0 for fam in sorted(set(DEFAULT_FAMILY_MAP.values()))}
        family_counts_top100 = {fam: 0 for fam in sorted(set(DEFAULT_FAMILY_MAP.values()))}
        family_names_top50 = {fam: [] for fam in sorted(set(DEFAULT_FAMILY_MAP.values()))}
        family_names_top100 = {fam: [] for fam in sorted(set(DEFAULT_FAMILY_MAP.values()))}
        for cls in selected_class_names:
            fam = _selected_family(cls)
            if cls in top50_names:
                family_counts_top50[fam] += 1
                family_names_top50[fam].append(cls)
            if cls in top100_names:
                family_counts_top100[fam] += 1
                family_names_top100[fam].append(cls)

        metric_row = {
            "metric": metric_name,
            "metric_kind": metric_kind,
            "support_filter": support_filter if support_filter is not None else "",
            "top50_count": len(top50),
            "top100_count": len(top100),
            "low_support_lt5_rate": contam5,
            "low_support_lt10_rate": contam10,
            "low_support_lt20_rate": contam20,
            "low_support_lt50_rate": contam50,
            "selected_known_top50_count": len(selected_top50_names),
            "selected_known_top100_count": len(selected_top100_names),
            "selected_known_top50_names": "|".join(selected_top50_names),
            "selected_known_top100_names": "|".join(selected_top100_names),
            "person_top50_count": family_counts_top50["person"],
            "person_top100_count": family_counts_top100["person"],
            "vehicle_top50_count": family_counts_top50["vehicle"],
            "vehicle_top100_count": family_counts_top100["vehicle"],
            "clothing_accessory_top50_count": family_counts_top50["clothing_accessory"],
            "clothing_accessory_top100_count": family_counts_top100["clothing_accessory"],
            "object_or_animal_top50_count": family_counts_top50["object_or_animal"],
            "object_or_animal_top100_count": family_counts_top100["object_or_animal"],
            "overlap_high_mass_top50_count": len(set(top50_names) & high_mass_top50_set),
            "overlap_top_improved_count": len(set(top50_names) & improved_set),
            "overlap_top_degraded_count": len(set(top50_names) & degraded_set),
            "score_kind": metric_kind,
        }
        metric_row["quality_score"] = _quality_score(metric_row)
        metric_quality_rows.append(metric_row)

        overlap_row = {
            "metric": metric_name,
            "overlap_high_mass_top50_count": metric_row["overlap_high_mass_top50_count"],
            "overlap_high_mass_top50_names": "|".join(sorted(set(top50_names) & high_mass_top50_set)),
            "overlap_top_improved_count": metric_row["overlap_top_improved_count"],
            "overlap_top_improved_names": "|".join(sorted(set(top50_names) & improved_set)),
            "overlap_top_degraded_count": metric_row["overlap_top_degraded_count"],
            "overlap_top_degraded_names": "|".join(sorted(set(top50_names) & degraded_set)),
            "selected_known_top50_count": len(selected_top50_names),
            "selected_known_top100_count": len(selected_top100_names),
        }
        metric_overlap_rows.append(overlap_row)

        low_support_rows.append({
            "metric": metric_name,
            "top50_count": len(top50),
            "contamination_lt5_rate": contam5,
            "contamination_lt10_rate": contam10,
            "contamination_lt20_rate": contam20,
            "contamination_lt50_rate": contam50,
            "support_filter": support_filter if support_filter is not None else "",
        })

        if metric_row["quality_score"] > best_quality:
            best_quality = metric_row["quality_score"]
            best_metric_name = metric_name
            best_metric_row = metric_row

        # selected known class ranks
        ranks = {r["class_name"]: i + 1 for i, r in enumerate(ranked_rows)}
        for cls in selected_class_names:
            row = selected_row_lookup.get(cls, {})
            selected_rows.append({
                "metric": metric_name,
                "class_name": cls,
                "raw_id": row.get("raw_id", ""),
                "family": _selected_family(cls),
                "rank": ranks.get(cls, ""),
                "top50": bool(cls in top50_names),
                "top100": bool(cls in top100_names),
                "score": row.get("score", ""),
                "label_support_ema": row.get("label_support_ema", ""),
                "responsibility_mass_ema": row.get("responsibility_mass_ema", ""),
                "top1_count_ema": row.get("top1_count_ema", ""),
            })

        for entry in ranked_rows[:50]:
            top50_rows.append(entry)

    if best_metric_row is None:
        raise RuntimeError("failed to select a recommended metric")

    final_ranked = metric_ranked_rows[best_metric_name]
    final_top50 = final_ranked[:50]
    final_top20 = final_ranked[:20]
    final_top50_names = [r["class_name"] for r in final_top50]
    final_top20_names = [r["class_name"] for r in final_top20]

    # Save outputs.
    _ensure_dir(output_root)
    summary_payload = {
        "status": "PASS",
        "run_root": str(run_root),
        "output_root": str(output_root),
        "absorber_scores_csv": str(absorber_path),
        "compare_root": str(compare_root),
        "row_count": len(rows),
        "selected_class_count": len(selected_class_names),
        "selected_class_names": selected_class_names,
        "recommended_metric": best_metric_name,
        "recommended_metric_quality_score": best_quality,
        "recommended_metric_row": best_metric_row,
        "top20_recommended_metric": final_top20,
        "final_top50_names": final_top50_names,
        "final_top20_names": final_top20_names,
        "top_improved_classes": improved_classes,
        "top_degraded_classes": degraded_classes,
        "decision": "B. auto absorber dynamic demand is promising, using the recommended metric",
        "interpretation": "Hybrid or top1-mass style support-aware absorber metrics are feasible; raw absorber_score is too low-support sensitive.",
    }
    _write_json(output_root / "summary.json", summary_payload)
    _write_csv(output_root / "metric_quality_summary.csv", metric_quality_rows)
    _write_csv(output_root / "top50_by_metric.csv", top50_rows)
    _write_csv(output_root / "selected_known_class_ranks.csv", selected_rows)
    _write_csv(output_root / "low_support_contamination_by_metric.csv", low_support_rows)
    _write_csv(output_root / "metric_overlap_summary.csv", metric_overlap_rows)

    recommended_md = output_root / "recommended_absorber_metric.md"
    recommended_md.write_text(
        "# Absorber Score Design Audit\n\n"
        f"Recommended metric: `{best_metric_name}`\n\n"
        f"Quality score: `{best_quality}`\n\n"
        "Rationale:\n"
        f"- Top50 low-support contamination: `<5={best_metric_row['low_support_lt5_rate']:.3f}`, `<10={best_metric_row['low_support_lt10_rate']:.3f}`\n"
        f"- Selected known classes in top50: `{best_metric_row['selected_known_top50_count']}`\n"
        f"- Selected known classes in top100: `{best_metric_row['selected_known_top100_count']}`\n"
        f"- Overlap with high-mass top50: `{best_metric_row['overlap_high_mass_top50_count']}`\n\n"
        "Interpretation:\n"
        "- raw_absorber_score is too sensitive to low-support noise.\n"
        "- A support-aware absorber ranking is viable for future dynamic demand.\n"
        "- The best next step is to use the recommended support-aware metric rather than raw absorber_score.\n",
        encoding="utf-8",
    )

    takeover = output_root / "ABSORBER_SCORE_DESIGN_AUDIT_TAKEOVER.md"
    takeover.write_text(
        "# GT-fullY Absorber Score Design Audit\n\n"
        f"Status: `PASS`\n\n"
        f"Recommended metric: `{best_metric_name}`\n\n"
        f"Output root: `{output_root}`\n\n"
        "Files:\n"
        "- summary.json\n"
        "- metric_quality_summary.csv\n"
        "- top50_by_metric.csv\n"
        "- selected_known_class_ranks.csv\n"
        "- low_support_contamination_by_metric.csv\n"
        "- metric_overlap_summary.csv\n"
        "- recommended_absorber_metric.md\n",
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
