#!/usr/bin/env python3
"""Read-only differential audit for GT-fullY clean nohub vs auto-absorber veryweak.

This script never trains, never touches checkpoints, and never changes predictions.
It joins existing per-class attribution outputs with the replay absorber-pressure table
and reports whether auto_absorb_veryweak actually changed classes relative to nohub.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
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


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return repr(v)
    return v


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


def _pick(row: Mapping[str, Any], names: Sequence[str], default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name])
    return default


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


def _id_candidates(row: Mapping[str, Any], names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for name in names:
        if name not in row:
            continue
        val = _safe_int_str(row.get(name))
        if val and val not in out:
            out.append(val)
    return out


def _choose_replay_metric_column(rows: Sequence[Mapping[str, str]]) -> str:
    if not rows:
        return ""
    cols = set(rows[0].keys())
    for name in [
        "metric_support_floor_absorber_score_floor_20",
        "metric_support_floor_absorber_score_floor_10",
        "metric_support_floor_absorber_score_floor_5",
        "metric_support_filtered_absorber_score_ge_20",
        "metric_support_filtered_absorber_score_ge_10",
        "metric_support_filtered_absorber_score_ge_5",
        "metric_raw_absorber_score",
    ]:
        if name in cols:
            return name
    return ""


def _normalize_run_name(name: str, aliases: Mapping[str, str]) -> str:
    return aliases.get(name, name)


def _resolve_run_aliases(raw_aliases: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in raw_aliases or []:
        if not item:
            continue
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


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


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _infer_fields_from_per_class(rows: Sequence[Mapping[str, str]]) -> Dict[str, str]:
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
        "raw_id": first(["raw_id", "gt_raw_id", "class_raw_id", "class_id"]),
        "class_name": first(["class_name", "gt_class_name", "name"]),
        "gt_count": first(["gt_count", "row_count", "count"]),
        "top1": first(["gt_top1_hit_rate", "top1_is_gt_mean", "top1_hit_rate"]),
        "norm_rank": first(["mean_normalized_gt_rank", "normalized_gt_rank_mean"]),
        "gt_rank": first(["gt_rank_mean", "mean_gt_rank"]),
        "candidate_size": first(["candidate_size_mean"]),
        "certificate_family": first(["certificate_family"]),
        "certificate_type": first(["certificate_type"]),
        "resolved_round": first(["resolved_round"]),
        "base_group": first(["base_group", "base_observed_unobserved"]),
    }


def _index_per_class(rows: Sequence[Mapping[str, str]], fields: Mapping[str, str], aliases: Mapping[str, str]) -> Dict[Tuple[str, str], Mapping[str, str]]:
    out: Dict[Tuple[str, str], Mapping[str, str]] = {}
    run_col = fields.get("run") or "checkpoint"
    raw_col = fields.get("raw_id") or "raw_id"
    for row in rows:
        raw_id = _safe_int_str(row.get(raw_col))
        run = _normalize_run_name(str(row.get(run_col, "")), aliases)
        if raw_id and run:
            out[(raw_id, run)] = row
    return out


def _read_replay_table(replay_dir: Path) -> Dict[str, Mapping[str, str]]:
    path = replay_dir / "class_absorber_replay_table.csv"
    if not path.exists():
        # fall back to top/risk/opportunity tables if a future replay only emits those.
        return {}
    rows = _read_csv(path)
    out: Dict[str, Mapping[str, str]] = {}
    for r in rows:
        for raw_id in _id_candidates(r, ["raw_id", "class_raw_id", "class_id", "category_id", "gt_raw_id"]):
            out[raw_id] = r
    return out


def _build_class_delta_rows(
    compare_dir: Path,
    replay_dir: Path,
    nohub_run: str,
    auto_run: str,
    aliases: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    attr_path = compare_dir / "per_class_attribution.csv"
    rows = _read_csv(attr_path)
    fields = _infer_fields_from_per_class(rows)
    missing = [k for k, v in fields.items() if k in ("run", "raw_id", "top1", "norm_rank") and not v]
    if missing:
        raise RuntimeError(f"Cannot infer required columns from {attr_path}: missing {missing}; columns={list(rows[0].keys()) if rows else []}")
    idx = _index_per_class(rows, fields, aliases)
    replay = _read_replay_table(replay_dir)
    replay_metric_col = _choose_replay_metric_column(list(replay.values()))
    replay_metric_vals = [_num(r.get(replay_metric_col)) for r in replay.values()] if replay_metric_col else []
    replay_metric_clip = statistics.quantiles(replay_metric_vals, n=100)[94] if len(replay_metric_vals) >= 100 else (max(replay_metric_vals) if replay_metric_vals else 0.0)
    if replay_metric_clip <= 1e-12:
        replay_metric_clip = 0.0

    raw_ids = sorted({raw for (raw, run) in idx.keys() if run in {nohub_run, auto_run}}, key=lambda x: int(x) if x.isdigit() else x)
    out: List[Dict[str, Any]] = []
    joined_rows = 0
    for raw_id in raw_ids:
        nr = idx.get((raw_id, nohub_run))
        ar = idx.get((raw_id, auto_run))
        if nr is None or ar is None:
            continue
        rr = replay.get(raw_id, {})
        if rr:
            joined_rows += 1
        n_top1 = _num(nr.get(fields["top1"]))
        a_top1 = _num(ar.get(fields["top1"]))
        n_rank = _num(nr.get(fields["norm_rank"]))
        a_rank = _num(ar.get(fields["norm_rank"]))
        n_gt_rank = _num(nr.get(fields.get("gt_rank", ""))) if fields.get("gt_rank") else 0.0
        a_gt_rank = _num(ar.get(fields.get("gt_rank", ""))) if fields.get("gt_rank") else 0.0
        gt_count = _num(ar.get(fields.get("gt_count", "")) or nr.get(fields.get("gt_count", ""))) if fields.get("gt_count") else 0.0
        class_name = _pick(ar, [fields.get("class_name", "")], "") or _pick(nr, [fields.get("class_name", "")], "") or _pick(rr, ["class_name", "name"], "")
        replay_metric_value = _num(_pick(rr, [replay_metric_col], "")) if replay_metric_col else 0.0
        normalized_pressure = (replay_metric_value / replay_metric_clip) if replay_metric_clip > 1e-12 else 0.0
        normalized_pressure = max(0.0, min(1.0, normalized_pressure))
        class_weight = max(0.90, 1.0 - 0.05 * normalized_pressure)
        downweight = 1.0 - class_weight
        delta_top1 = a_top1 - n_top1
        delta_rank = a_rank - n_rank
        replay_opportunity_score = max(0.0, -delta_top1) * normalized_pressure
        replay_risk_score = max(0.0, delta_top1) * normalized_pressure
        opportunity_score = replay_opportunity_score
        risk_score = replay_risk_score
        row: Dict[str, Any] = {
            "raw_id": raw_id,
            "class_name": class_name,
            "gt_count": gt_count,
            "nohub_gt_top1_hit_rate": n_top1,
            "auto_gt_top1_hit_rate": a_top1,
            "delta_top1_auto_vs_nohub": delta_top1,
            "nohub_mean_normalized_gt_rank": n_rank,
            "auto_mean_normalized_gt_rank": a_rank,
            "delta_rank_auto_vs_nohub": a_rank - n_rank,
            "nohub_gt_rank_mean": n_gt_rank,
            "auto_gt_rank_mean": a_gt_rank,
            "delta_gt_rank_auto_vs_nohub": a_gt_rank - n_gt_rank,
            "certificate_family": _pick(ar, [fields.get("certificate_family", "")], _pick(nr, [fields.get("certificate_family", "")], _pick(rr, ["certificate_family"], ""))),
            "certificate_type": _pick(ar, [fields.get("certificate_type", "")], _pick(nr, [fields.get("certificate_type", "")], _pick(rr, ["certificate_type"], ""))),
            "resolved_round": _pick(ar, [fields.get("resolved_round", "")], _pick(nr, [fields.get("resolved_round", "")], _pick(rr, ["resolved_round"], ""))),
            "base_group": _pick(ar, [fields.get("base_group", "")], _pick(nr, [fields.get("base_group", "")], _pick(rr, ["base_group"], ""))),
            "label_support_ema": _num(_pick(rr, ["label_support_ema", "support", "label_support"])),
            "responsibility_mass_ema": _num(_pick(rr, ["responsibility_mass_ema", "mass", "responsibility_mass"])),
            "top1_count_ema": _num(_pick(rr, ["top1_count_ema", "top1_mass", "top1"])),
            "recommended_metric_value": replay_metric_value,
            "normalized_pressure": normalized_pressure,
            "class_weight": class_weight,
            "downweight": downweight,
            "replay_opportunity_score": replay_opportunity_score,
            "replay_risk_score": replay_risk_score,
            "opportunity_score": opportunity_score,
            "risk_score": risk_score,
            "replay_metric_column": replay_metric_col,
        }
        row["replay_rows_joined"] = 1 if rr else 0
        out.append(row)
    meta = {
        "per_class_attribution": str(attr_path),
        "field_map": fields,
        "paired_class_count": len(out),
        "replay_metric_column": replay_metric_col,
        "replay_metric_p95": replay_metric_clip,
        "replay_rows_joined": joined_rows,
        "replay_row_count": len(replay),
        "replay_join_key_used": "raw_id",
        "replay_id_columns_accepted": ["raw_id", "class_raw_id", "class_id", "category_id", "gt_raw_id"],
    }
    return out, meta


def _weighted_mean(rows: Sequence[Mapping[str, Any]], key: str, weight_key: str = "gt_count") -> float:
    denom = sum(_num(r.get(weight_key)) for r in rows)
    if denom <= 0:
        return 0.0
    return sum(_num(r.get(key)) * _num(r.get(weight_key)) for r in rows) / denom


def _summarize_groups(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    group_specs = [
        ("overall", lambda r: "overall"),
        ("certificate_family", lambda r: str(r.get("certificate_family", ""))),
        ("certificate_type", lambda r: str(r.get("certificate_type", ""))),
        ("resolved_round", lambda r: str(r.get("resolved_round", ""))),
        ("base_group", lambda r: str(r.get("base_group", ""))),
        ("person_conditioned", lambda r: "true" if "person" in (str(r.get("certificate_family", "")) + " " + str(r.get("certificate_type", ""))).lower() else "false"),
    ]
    out: List[Dict[str, Any]] = []
    for gname, getter in group_specs:
        buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for r in rows:
            gv = getter(r) or "unknown"
            buckets[gv].append(r)
        for gv, rs in sorted(buckets.items()):
            gt_count = sum(_num(r.get("gt_count")) for r in rs)
            out.append({
                "group_name": gname,
                "group_value": gv,
                "class_count": len(rs),
                "gt_count": gt_count,
                "delta_top1_auto_vs_nohub_weighted": _weighted_mean(rs, "delta_top1_auto_vs_nohub"),
                "delta_rank_auto_vs_nohub_weighted": _weighted_mean(rs, "delta_rank_auto_vs_nohub"),
                "delta_gt_rank_auto_vs_nohub_weighted": _weighted_mean(rs, "delta_gt_rank_auto_vs_nohub"),
                "mean_downweight_weighted": _weighted_mean(rs, "downweight"),
                "mean_pressure_weighted": _weighted_mean(rs, "normalized_pressure"),
                "mean_class_weight_weighted": _weighted_mean(rs, "class_weight"),
                "changed_top1_class_count": sum(1 for r in rs if abs(_num(r.get("delta_top1_auto_vs_nohub"))) > 1e-12),
                "changed_rank_class_count": sum(1 for r in rs if abs(_num(r.get("delta_rank_auto_vs_nohub"))) > 1e-12),
            })
    return out


def _correlation_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    predictors = [
        "label_support_ema",
        "responsibility_mass_ema",
        "top1_count_ema",
        "recommended_metric_value",
        "normalized_pressure",
        "class_weight",
        "downweight",
        "replay_opportunity_score",
        "replay_risk_score",
        "opportunity_score",
        "risk_score",
    ]
    targets = [
        "delta_top1_auto_vs_nohub",
        "delta_rank_auto_vs_nohub",
        "delta_gt_rank_auto_vs_nohub",
    ]
    out: List[Dict[str, Any]] = []
    for p in predictors:
        for t in targets:
            pairs = [(float(_num(r.get(p))), float(_num(r.get(t)))) for r in rows if (p in r and t in r)]
            # Keep rows with at least one non-zero on either side to avoid empty signal; but do not filter true zeros away.
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
            out.append({
                "predictor": p,
                "target": t,
                "n": len(xs),
                "pearson": _pearson(xs, ys),
                "spearman": _spearman(xs, ys),
                "predictor_nonzero_count": sum(1 for x in xs if abs(x) > 1e-12),
                "target_nonzero_count": sum(1 for y in ys if abs(y) > 1e-12),
            })
    return out


def _sort_desc(rows: Sequence[Mapping[str, Any]], key: str) -> List[Mapping[str, Any]]:
    return sorted(rows, key=lambda r: (_num(r.get(key)), _num(r.get("gt_count"))), reverse=True)


def _sort_asc(rows: Sequence[Mapping[str, Any]], key: str) -> List[Mapping[str, Any]]:
    return sorted(rows, key=lambda r: (_num(r.get(key)), -_num(r.get("gt_count"))))


def _take_top(rows: Sequence[Mapping[str, Any]], key: str, n: int = 20, reverse: bool = True) -> List[Mapping[str, Any]]:
    return (_sort_desc(rows, key) if reverse else _sort_asc(rows, key))[:n]


def _validate_replay(rows: Sequence[Mapping[str, Any]], key: str, n: int = 30) -> List[Mapping[str, Any]]:
    ranked = _sort_desc(rows, key)[:n]
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(ranked, 1):
        out.append({
            **dict(r),
            "replay_rank": i,
            "replay_key": key,
            "actual_helped_top1": _num(r.get("delta_top1_auto_vs_nohub")) > 0,
            "actual_hurt_top1": _num(r.get("delta_top1_auto_vs_nohub")) < 0,
            "actual_improved_rank": _num(r.get("delta_rank_auto_vs_nohub")) < 0,
            "actual_worsened_rank": _num(r.get("delta_rank_auto_vs_nohub")) > 0,
        })
    return out


BASE_FIELDS = [
    "raw_id", "class_name", "gt_count",
    "certificate_family", "certificate_type", "resolved_round", "base_group",
    "nohub_gt_top1_hit_rate", "auto_gt_top1_hit_rate", "delta_top1_auto_vs_nohub",
    "nohub_mean_normalized_gt_rank", "auto_mean_normalized_gt_rank", "delta_rank_auto_vs_nohub",
    "nohub_gt_rank_mean", "auto_gt_rank_mean", "delta_gt_rank_auto_vs_nohub",
    "label_support_ema", "responsibility_mass_ema", "top1_count_ema",
    "recommended_metric_value", "normalized_pressure", "class_weight", "downweight",
    "replay_opportunity_score", "replay_risk_score", "opportunity_score", "risk_score",
    "replay_metric_column", "replay_rows_joined",
]


def _make_takeover(summary: Mapping[str, Any], out_dir: Path) -> str:
    return f"""# Auto Absorb Veryweak Differential Audit Takeover

Status: `{summary.get('status')}`
Output: `{out_dir}`

## Scope

Read-only GT-fullY clean differential audit. No training, no checkpoint modification, no VideoCutLER/Y′/extra/mAP.
GT attribution is used only for post-hoc diagnosis.

## Compared runs

- nohub: `{summary.get('nohub_run')}`
- auto: `{summary.get('auto_run')}`

## Key findings

- paired classes: `{summary.get('paired_class_count')}`
- top1 changed class count: `{summary.get('top1_changed_class_count')}`
- rank changed class count: `{summary.get('rank_changed_class_count')}`
- overall weighted delta top1 auto-vs-nohub: `{summary.get('overall_delta_top1_weighted')}`
- overall weighted delta normalized rank auto-vs-nohub: `{summary.get('overall_delta_rank_weighted')}`
- max abs top1 delta: `{summary.get('max_abs_delta_top1')}`
- max abs normalized-rank delta: `{summary.get('max_abs_delta_rank')}`

## Interpretation

{summary.get('interpretation')}

## Required follow-up

Use `summary_nohub_vs_auto_by_group.csv`, `pressure_effect_correlation.csv`, `replay_top_opportunity_actual_result.csv`, and `replay_top_risk_actual_result.csv` to decide whether auto absorber is too weak, mis-targeted, or only rank-shaping.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare_dir", required=True, help="Directory containing per_class_attribution.csv for baseline/nohub/auto compare.")
    ap.add_argument("--replay_dir", required=True, help="Directory containing auto absorber replay outputs, especially class_absorber_replay_table.csv.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--nohub_run", default="soft_e2e_nohub")
    ap.add_argument("--auto_run", default="soft_e2e_auto_absorb_veryweak")
    ap.add_argument("--run_alias", action="append", default=[], help="Optional raw=canonical run alias; may repeat.")
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()

    compare_dir = Path(args.compare_dir)
    replay_dir = Path(args.replay_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aliases = _resolve_run_aliases(args.run_alias)
    # Common aliases observed in previous runs.
    aliases.setdefault("auto_absorb_veryweak", args.auto_run)
    aliases.setdefault("soft_e2e_auto_absorb_veryweak_15ep", args.auto_run)
    aliases.setdefault("gt_full_y_soft_e2e_auto_absorb_veryweak_15ep", args.auto_run)
    aliases.setdefault("gt_full_y_soft_e2e_nohub_15ep", args.nohub_run)

    rows, meta = _build_class_delta_rows(compare_dir, replay_dir, args.nohub_run, args.auto_run, aliases)
    if not rows:
        raise SystemExit("No paired nohub/auto class rows found. Check --nohub_run/--auto_run and per_class_attribution.csv run names.")
    replay_joined = int(meta.get("replay_rows_joined", 0))
    replay_blocker = ""
    if replay_joined <= 0:
        replay_blocker = f"Replay directory provided but no replay rows joined from {replay_dir}; check raw_id normalization and replay schema."

    group_rows = _summarize_groups(rows)
    corr_rows = _correlation_rows(rows)
    top_improved = _take_top(rows, "delta_top1_auto_vs_nohub", args.top_k, True)
    top_degraded = _take_top(rows, "delta_top1_auto_vs_nohub", args.top_k, False)
    top_rank_improved = _take_top(rows, "delta_rank_auto_vs_nohub", args.top_k, False)  # lower rank is better
    top_rank_degraded = _take_top(rows, "delta_rank_auto_vs_nohub", args.top_k, True)
    opp_actual = _validate_replay(rows, "replay_opportunity_score", max(30, args.top_k))
    risk_actual = _validate_replay(rows, "replay_risk_score", max(30, args.top_k))

    overall = next((r for r in group_rows if r["group_name"] == "overall" and r["group_value"] == "overall"), {})
    max_abs_top1 = max(abs(_num(r.get("delta_top1_auto_vs_nohub"))) for r in rows)
    max_abs_rank = max(abs(_num(r.get("delta_rank_auto_vs_nohub"))) for r in rows)
    changed_top1 = sum(1 for r in rows if abs(_num(r.get("delta_top1_auto_vs_nohub"))) > 1e-12)
    changed_rank = sum(1 for r in rows if abs(_num(r.get("delta_rank_auto_vs_nohub"))) > 1e-12)
    pressure_corr_top1 = next((r for r in corr_rows if r["predictor"] == "downweight" and r["target"] == "delta_top1_auto_vs_nohub"), {})
    pressure_corr_rank = next((r for r in corr_rows if r["predictor"] == "downweight" and r["target"] == "delta_rank_auto_vs_nohub"), {})

    if changed_top1 <= max(5, len(rows) * 0.03) and changed_rank <= max(10, len(rows) * 0.05):
        interpretation = "Veryweak auto absorber appears safe but mostly inert: few class-level decisions changed. This supports keeping soft_e2e_nohub as the main mechanism."
    elif pressure_corr_top1.get("spearman") is not None and abs(float(pressure_corr_top1["spearman"])) < 0.05:
        interpretation = "Downweight pressure has little monotonic relation to top1 changes; current class-weight action may be too weak or too shallow."
    elif pressure_corr_top1.get("spearman") is not None and float(pressure_corr_top1["spearman"]) < -0.05:
        interpretation = "Higher downweight tends to reduce top1; auto absorber may be suppressing useful classes and should not be strengthened."
    else:
        interpretation = "Auto absorber changes are measurable; inspect risk/opportunity validation before increasing strength."

    summary: Dict[str, Any] = {
        "status": "PASS" if replay_joined > 0 else "FAIL",
        "compare_dir": str(compare_dir),
        "replay_dir": str(replay_dir),
        "output_dir": str(out_dir),
        "nohub_run": args.nohub_run,
        "auto_run": args.auto_run,
        **meta,
        "paired_class_count": len(rows),
        "top1_changed_class_count": changed_top1,
        "rank_changed_class_count": changed_rank,
        "overall_delta_top1_weighted": overall.get("delta_top1_auto_vs_nohub_weighted"),
        "overall_delta_rank_weighted": overall.get("delta_rank_auto_vs_nohub_weighted"),
        "overall_mean_downweight_weighted": overall.get("mean_downweight_weighted"),
        "max_abs_delta_top1": max_abs_top1,
        "max_abs_delta_rank": max_abs_rank,
        "replay_join_blocker": replay_blocker,
        "downweight_vs_delta_top1": pressure_corr_top1,
        "downweight_vs_delta_rank": pressure_corr_rank,
        "interpretation": interpretation,
        "outputs": {
            "per_class_delta_csv": str(out_dir / "per_class_nohub_vs_auto_delta.csv"),
            "summary_by_group_csv": str(out_dir / "summary_nohub_vs_auto_by_group.csv"),
            "correlation_csv": str(out_dir / "pressure_effect_correlation.csv"),
            "replay_opportunity_csv": str(out_dir / "replay_top_opportunity_actual_result.csv"),
            "replay_risk_csv": str(out_dir / "replay_top_risk_actual_result.csv"),
            "takeover_md": str(out_dir / "AUTO_ABSORB_VERYWEAK_DIFFERENTIAL_AUDIT_TAKEOVER.md"),
        },
    }

    _write_csv(out_dir / "per_class_nohub_vs_auto_delta.csv", rows, BASE_FIELDS)
    _write_csv(out_dir / "summary_nohub_vs_auto_by_group.csv", group_rows, [
        "group_name", "group_value", "class_count", "gt_count",
        "delta_top1_auto_vs_nohub_weighted", "delta_rank_auto_vs_nohub_weighted", "delta_gt_rank_auto_vs_nohub_weighted",
        "mean_downweight_weighted", "mean_pressure_weighted", "mean_class_weight_weighted",
        "changed_top1_class_count", "changed_rank_class_count",
    ])
    _write_csv(out_dir / "pressure_effect_correlation.csv", corr_rows, [
        "predictor", "target", "n", "pearson", "spearman", "predictor_nonzero_count", "target_nonzero_count"
    ])
    _write_csv(out_dir / "replay_top_opportunity_actual_result.csv", opp_actual, ["replay_rank", "replay_key"] + BASE_FIELDS + ["actual_helped_top1", "actual_hurt_top1", "actual_improved_rank", "actual_worsened_rank"])
    _write_csv(out_dir / "replay_top_risk_actual_result.csv", risk_actual, ["replay_rank", "replay_key"] + BASE_FIELDS + ["actual_helped_top1", "actual_hurt_top1", "actual_improved_rank", "actual_worsened_rank"])
    _write_csv(out_dir / "top20_auto_improved_classes.csv", top_improved, BASE_FIELDS)
    _write_csv(out_dir / "top20_auto_degraded_classes.csv", top_degraded, BASE_FIELDS)
    _write_csv(out_dir / "top20_rank_improved_classes.csv", top_rank_improved, BASE_FIELDS)
    _write_csv(out_dir / "top20_rank_degraded_classes.csv", top_rank_degraded, BASE_FIELDS)
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "AUTO_ABSORB_VERYWEAK_DIFFERENTIAL_AUDIT_TAKEOVER.md").write_text(_make_takeover(summary, out_dir), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if replay_joined <= 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
