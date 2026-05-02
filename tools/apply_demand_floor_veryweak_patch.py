#!/usr/bin/env python3
"""Apply GT-fullY clean nohub demand-floor veryweak training patch.

This patch is intentionally small and incremental. It adds an optional static
class-level additive score bonus to the existing GT-fullY clean soft-routing
trainer. The bonus is computed from the observable under_assigned_class_table.csv
produced by the read-only under-assigned class audit. GT correctness/count is not
used to construct the bonus.

Default behavior is unchanged unless --enable_demand_floor_bonus is passed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

PATCH_TOKEN = "DEMAND_FLOOR_VERYWEAK_PATCH_V1"

HELPER_BLOCK = r'''

# DEMAND_FLOOR_VERYWEAK_PATCH_V1_BEGIN
def _df_safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return v
    except Exception:
        try:
            v = float(str(x).strip())
            if not math.isfinite(v):
                return float(default)
            return v
        except Exception:
            return float(default)


def _df_load_under_assigned_table(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing under-assigned table: {path}")
    out: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as h:
        reader = csv.DictReader(h)
        for row in reader:
            rid = _as_int(row.get("raw_id"))
            if rid is None:
                continue
            out[int(rid)] = dict(row)
    if not out:
        raise ValueError(f"under-assigned table has no usable raw_id rows: {path}")
    return out


def _df_rank_norm(vals_by_id: Mapping[int, float]) -> Dict[int, float]:
    items = sorted(((int(k), float(v)) for k, v in vals_by_id.items()), key=lambda kv: kv[1], reverse=True)
    n = len(items)
    if n <= 0:
        return {}
    if n == 1:
        return {items[0][0]: 1.0}
    return {int(k): float(1.0 - idx / (n - 1)) for idx, (k, _v) in enumerate(items)}


def _df_minmax_norm(vals_by_id: Mapping[int, float]) -> Dict[int, float]:
    if not vals_by_id:
        return {}
    vals = [float(v) for v in vals_by_id.values() if math.isfinite(float(v))]
    if not vals:
        return {int(k): 0.0 for k in vals_by_id}
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1.0e-12:
        return {int(k): 0.0 for k in vals_by_id}
    return {int(k): max(0.0, min(1.0, (float(v) - lo) / (hi - lo))) for k, v in vals_by_id.items()}


def _df_zclip_norm(vals_by_id: Mapping[int, float]) -> Dict[int, float]:
    if not vals_by_id:
        return {}
    vals = [float(v) for v in vals_by_id.values() if math.isfinite(float(v))]
    if not vals:
        return {int(k): 0.0 for k in vals_by_id}
    mean = float(np.mean(np.asarray(vals, dtype=np.float64)))
    std = float(np.std(np.asarray(vals, dtype=np.float64)))
    if std <= 1.0e-12:
        return {int(k): 0.0 for k in vals_by_id}
    return {int(k): max(0.0, min(1.0, ((float(v) - mean) / std + 2.0) / 4.0)) for k, v in vals_by_id.items()}


def _df_excluded_high_support_ids(table: Mapping[int, Mapping[str, Any]], policy: str) -> Set[int]:
    if policy == "exclude_top_1pct_support":
        items = sorted(
            ((int(rid), _df_safe_float(row.get("candidate_support"))) for rid, row in table.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        k = max(1, int(math.ceil(0.01 * len(items))))
        return {int(rid) for rid, _support in items[:k]}
    if policy == "exclude_top_5_support":
        items = sorted(
            ((int(rid), _df_safe_float(row.get("candidate_support"))) for rid, row in table.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return {int(rid) for rid, _support in items[:5]}
    if policy in {"none", "log_squash"}:
        return set()
    raise ValueError(f"unsupported demand_floor_high_support_policy: {policy}")


def _df_metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == "negative_mean_responsibility_per_support":
        return -_df_safe_float(row.get("mean_responsibility_per_support"))
    if metric == "low_mass_per_support":
        if "low_mass_per_support" in row and str(row.get("low_mass_per_support", "")) != "":
            return _df_safe_float(row.get("low_mass_per_support"))
        return 1.0 - _df_safe_float(row.get("mean_responsibility_per_support"))
    return _df_safe_float(row.get(metric))


def _build_demand_floor_static_bonus_from_under_table(
    *,
    under_assigned_csv: Path,
    metric: str,
    alpha: float,
    max_bonus: float,
    support_threshold: float,
    high_support_policy: str,
    normalization: str,
    allowed_raw_ids: Optional[Set[int]] = None,
) -> Tuple[Dict[int, float], List[Dict[str, Any]], Dict[str, Any]]:
    table = _df_load_under_assigned_table(under_assigned_csv)
    excluded = _df_excluded_high_support_ids(table, high_support_policy)
    raw_vals: Dict[int, float] = {}
    for rid, row in table.items():
        rid_i = int(rid)
        if allowed_raw_ids is not None and rid_i not in allowed_raw_ids:
            continue
        if rid_i in excluded:
            continue
        support = _df_safe_float(row.get("candidate_support"))
        if support < float(support_threshold):
            continue
        v = _df_metric_value(row, metric)
        if high_support_policy == "log_squash":
            v = math.log1p(max(0.0, float(v)))
        if not math.isfinite(float(v)):
            continue
        raw_vals[rid_i] = float(v)
    if normalization == "rank":
        norm = _df_rank_norm(raw_vals)
    elif normalization == "minmax":
        norm = _df_minmax_norm(raw_vals)
    elif normalization == "zclip":
        norm = _df_zclip_norm(raw_vals)
    else:
        raise ValueError(f"unsupported demand_floor_normalization: {normalization}")
    bonus_map = {
        int(rid): min(float(max_bonus), max(0.0, float(alpha) * float(norm.get(int(rid), 0.0))))
        for rid in raw_vals
    }
    rank_items = sorted(raw_vals.items(), key=lambda kv: kv[1], reverse=True)
    rows: List[Dict[str, Any]] = []
    for rank, (rid, raw_v) in enumerate(rank_items, start=1):
        row = table.get(int(rid), {})
        b = float(bonus_map.get(int(rid), 0.0))
        rows.append({
            "raw_id": int(rid),
            "class_name": row.get("class_name", ""),
            "bonus_rank": int(rank),
            "metric": str(metric),
            "metric_raw_value": float(raw_v),
            "metric_norm_value": float(norm.get(int(rid), 0.0)),
            "bonus": b,
            "candidate_support": row.get("candidate_support", ""),
            "responsibility_mass": row.get("responsibility_mass", ""),
            "top1_count": row.get("top1_count", ""),
            "gt_count": row.get("gt_count", ""),
            "delta_gt_top1_hit_rate": row.get("delta_gt_top1_hit_rate", ""),
            "delta_mean_normalized_gt_rank": row.get("delta_mean_normalized_gt_rank", ""),
            "is_nohub_degraded_either": row.get("is_nohub_degraded_either", ""),
            "certificate_family": row.get("certificate_family", ""),
            "certificate_type": row.get("certificate_type", ""),
            "resolved_round": row.get("resolved_round", ""),
            "base_group": row.get("base_group", ""),
            "person_conditioned": row.get("person_conditioned", ""),
        })
    nonzero = [v for v in bonus_map.values() if float(v) > 0.0]
    meta = {
        "enabled": True,
        "under_assigned_csv": str(under_assigned_csv),
        "metric": str(metric),
        "alpha": float(alpha),
        "max_bonus": float(max_bonus),
        "support_threshold": float(support_threshold),
        "high_support_policy": str(high_support_policy),
        "normalization": str(normalization),
        "raw_candidate_count": int(len(raw_vals)),
        "bonus_class_count": int(len(nonzero)),
        "bonus_mean_nonzero": float(np.mean(np.asarray(nonzero, dtype=np.float64))) if nonzero else 0.0,
        "bonus_max": float(max(nonzero)) if nonzero else 0.0,
        "excluded_high_support_count": int(len(excluded)),
    }
    return bonus_map, rows, meta
# DEMAND_FLOOR_VERYWEAK_PATCH_V1_END
'''

ARG_BLOCK = r'''
    # DEMAND_FLOOR_VERYWEAK_PATCH_V1 CLI BEGIN
    p.add_argument("--enable_demand_floor_bonus", action="store_true", help="Enable static veryweak demand-floor class score bonus for soft_routing.")
    p.add_argument("--demand_floor_under_assigned_csv", default="", help="under_assigned_class_table.csv from the under-assigned class audit.")
    p.add_argument("--demand_floor_metric", default="support_mass_gap", choices=("support_mass_gap", "support_mass_ratio", "hybrid_under_assignment_score", "low_mass_per_support", "under_top1_ratio", "negative_mean_responsibility_per_support"))
    p.add_argument("--demand_floor_alpha", type=float, default=0.02)
    p.add_argument("--demand_floor_max_bonus", type=float, default=0.02)
    p.add_argument("--demand_floor_support_threshold", type=float, default=20.0)
    p.add_argument("--demand_floor_high_support_policy", default="exclude_top_1pct_support", choices=("none", "log_squash", "exclude_top_1pct_support", "exclude_top_5_support"))
    p.add_argument("--demand_floor_normalization", default="rank", choices=("rank", "minmax", "zclip"))
    # DEMAND_FLOOR_VERYWEAK_PATCH_V1 CLI END
'''

VALIDATION_BLOCK = r'''
    # DEMAND_FLOOR_VERYWEAK_PATCH_V1 validation BEGIN
    if bool(getattr(args, "enable_demand_floor_bonus", False)) and str(args.protocol) != "soft_routing":
        raise ValueError("--enable_demand_floor_bonus is only supported for --protocol soft_routing")
    if bool(getattr(args, "enable_demand_floor_bonus", False)):
        if not str(getattr(args, "demand_floor_under_assigned_csv", "")):
            raise ValueError("--demand_floor_under_assigned_csv is required when --enable_demand_floor_bonus is used")
        if not Path(str(args.demand_floor_under_assigned_csv)).is_file():
            raise FileNotFoundError(f"missing --demand_floor_under_assigned_csv: {args.demand_floor_under_assigned_csv}")
    # DEMAND_FLOOR_VERYWEAK_PATCH_V1 validation END
'''

LOAD_BLOCK = r'''
    # DEMAND_FLOOR_VERYWEAK_PATCH_V1 load BEGIN
    demand_floor_bonus_enabled = bool(getattr(args, "enable_demand_floor_bonus", False))
    demand_floor_bonus_map: Dict[int, float] = {}
    demand_floor_bonus_rows: List[Dict[str, Any]] = []
    demand_floor_bonus_meta: Dict[str, Any] = {"enabled": False}
    demand_floor_bonus_tensor: Optional[torch.Tensor] = None
    if demand_floor_bonus_enabled:
        demand_floor_bonus_map, demand_floor_bonus_rows, demand_floor_bonus_meta = _build_demand_floor_static_bonus_from_under_table(
            under_assigned_csv=Path(str(args.demand_floor_under_assigned_csv)),
            metric=str(args.demand_floor_metric),
            alpha=float(args.demand_floor_alpha),
            max_bonus=float(args.demand_floor_max_bonus),
            support_threshold=float(args.demand_floor_support_threshold),
            high_support_policy=str(args.demand_floor_high_support_policy),
            normalization=str(args.demand_floor_normalization),
            allowed_raw_ids={int(x) for x in raw_to_idx.keys()},
        )
        demand_floor_bonus_map = {int(k): float(v) for k, v in demand_floor_bonus_map.items() if int(k) in raw_to_idx and float(v) > 0.0}
        demand_floor_bonus_tensor = torch.zeros((len(text_vocab_ids),), device=device, dtype=torch.float32)
        for rid, bonus in demand_floor_bonus_map.items():
            demand_floor_bonus_tensor[int(raw_to_idx[int(rid)])] = float(bonus)
        demand_floor_bonus_meta.update({
            "after_vocab_filter_bonus_class_count": int(len(demand_floor_bonus_map)),
            "after_vocab_filter_bonus_max": float(max(demand_floor_bonus_map.values())) if demand_floor_bonus_map else 0.0,
        })
        print(json.dumps({"demand_floor_bonus_meta": demand_floor_bonus_meta}, ensure_ascii=False))
    # DEMAND_FLOOR_VERYWEAK_PATCH_V1 load END
'''

SCORE_BONUS_BLOCK = r'''
                # DEMAND_FLOOR_VERYWEAK_PATCH_V1 score bonus BEGIN
                if demand_floor_bonus_enabled and demand_floor_bonus_tensor is not None:
                    demand_floor_cand_bonus = demand_floor_bonus_tensor[cand_idx]
                    if int(demand_floor_cand_bonus.numel()) > 0:
                        scores = scores + demand_floor_cand_bonus.view(1, -1)
                        _dfb = demand_floor_cand_bonus.detach().cpu().numpy().astype(np.float64)
                        batch_float_stats["demand_floor_bonus_candidate_mean"].append(float(_dfb.mean()) if _dfb.size else 0.0)
                        batch_float_stats["demand_floor_bonus_candidate_max"].append(float(_dfb.max()) if _dfb.size else 0.0)
                        batch_float_stats["demand_floor_bonus_candidate_nonzero_count"].append(float((_dfb > 0.0).sum()) if _dfb.size else 0.0)
                # DEMAND_FLOOR_VERYWEAK_PATCH_V1 score bonus END
'''

OUTPUT_BLOCK = r'''
    # DEMAND_FLOOR_VERYWEAK_PATCH_V1 outputs BEGIN
    demand_floor_bonus_outputs: Dict[str, str] = {}
    if demand_floor_bonus_enabled:
        demand_floor_fields = [
            "raw_id", "class_name", "bonus_rank", "metric", "metric_raw_value", "metric_norm_value", "bonus",
            "candidate_support", "responsibility_mass", "top1_count", "gt_count",
            "delta_gt_top1_hit_rate", "delta_mean_normalized_gt_rank", "is_nohub_degraded_either",
            "certificate_family", "certificate_type", "resolved_round", "base_group", "person_conditioned",
        ]
        _write_csv_rows(train_dir / "demand_floor_bonus_static.csv", demand_floor_bonus_rows, demand_floor_fields)
        _write_json(train_dir / "demand_floor_bonus_config.json", demand_floor_bonus_meta)
        demand_floor_bonus_outputs = {
            "demand_floor_bonus_static": str((Path("train") / "prealign" / "demand_floor_bonus_static.csv").as_posix()),
            "demand_floor_bonus_config": str((Path("train") / "prealign" / "demand_floor_bonus_config.json").as_posix()),
        }
    # DEMAND_FLOOR_VERYWEAK_PATCH_V1 outputs END
'''

STAGE_CONFIG_BLOCK = r'''
            # DEMAND_FLOOR_VERYWEAK_PATCH_V1 config BEGIN
            "enable_demand_floor_bonus": bool(demand_floor_bonus_enabled),
            "demand_floor_bonus_meta": demand_floor_bonus_meta,
            "demand_floor_bonus_outputs": demand_floor_bonus_outputs,
            # DEMAND_FLOOR_VERYWEAK_PATCH_V1 config END
'''

CKPT_BLOCK = r'''
        # DEMAND_FLOOR_VERYWEAK_PATCH_V1 checkpoint BEGIN
        "demand_floor_bonus_enabled": bool(demand_floor_bonus_enabled),
        "demand_floor_bonus_state": {
            "meta": demand_floor_bonus_meta,
            "bonus_by_raw_id": {str(int(k)): float(v) for k, v in sorted(demand_floor_bonus_map.items())},
        } if demand_floor_bonus_enabled else None,
        # DEMAND_FLOOR_VERYWEAK_PATCH_V1 checkpoint END
'''

def insert_after(s: str, marker: str, insert: str, label: str) -> str:
    idx = s.find(marker)
    if idx < 0:
        raise RuntimeError(f"patch marker not found: {label}")
    return s[: idx + len(marker)] + insert + s[idx + len(marker):]


def insert_before(s: str, marker: str, insert: str, label: str) -> str:
    idx = s.find(marker)
    if idx < 0:
        raise RuntimeError(f"patch marker not found: {label}")
    return s[:idx] + insert + s[idx:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--target", default="videocutler/run_stageb_train_gt_full_y_clean.py")
    args = ap.parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    target = repo / args.target
    if not target.is_file():
        raise FileNotFoundError(target)
    s = target.read_text(encoding="utf-8")
    if PATCH_TOKEN in s:
        print(f"{PATCH_TOKEN} already present in {target}; no changes made")
        return

    s = insert_before(s, "\ndef train_clean(args: argparse.Namespace) -> Dict[str, Any]:\n", HELPER_BLOCK, "helper insertion")

    validation_anchor = '    if bool(args.enable_absorber_logging) and str(args.protocol) != "soft_routing":\n        raise ValueError("--enable_absorber_logging is only supported for --protocol soft_routing")\n'
    if validation_anchor in s:
        s = insert_after(s, validation_anchor, VALIDATION_BLOCK, "validation insertion")
    else:
        s = insert_before(s, '    if str(args.protocol) == "static_residual" and args.epochs is None:\n', VALIDATION_BLOCK, "validation fallback")

    load_anchor = '    text_vocab_tensor = torch.from_numpy(np.asarray(text_vocab_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)\n'
    s = insert_after(s, load_anchor, LOAD_BLOCK, "bonus loading")

    score_anchor = '                scores = torch.matmul(Z, cand_text.t()) / temperature\n'
    s = insert_after(s, score_anchor, SCORE_BONUS_BLOCK, "score bonus")

    ckpt_anchor = '        "global_step": int(global_step),\n'
    s = insert_after(s, ckpt_anchor, CKPT_BLOCK, "checkpoint state")

    output_anchor = '    absorber_outputs: Dict[str, str] = {}\n'
    s = insert_before(s, output_anchor, OUTPUT_BLOCK, "output files")

    stage_anchor = '            "absorber_outputs": absorber_outputs,\n'
    s = insert_after(s, stage_anchor, STAGE_CONFIG_BLOCK, "stage config")

    args_anchor = '    return p.parse_args()\n'
    s = insert_before(s, args_anchor, ARG_BLOCK, "argparse flags")

    target.write_text(s, encoding="utf-8")
    print(f"Applied {PATCH_TOKEN} to {target}")


if __name__ == "__main__":
    main()
