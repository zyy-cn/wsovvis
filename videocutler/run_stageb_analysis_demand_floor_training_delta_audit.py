#!/usr/bin/env python3
"""Demand-floor training delta audit.

Read-only diagnostic for GT-fullY clean NoHub demand-floor pilot.

It compares a trained demand-floor checkpoint against the original NoHub
checkpoint at the per-class attribution level, then joins the static bonus table
and the replay deltas to explain why replay-positive demand floor did not close
positively after training.

This script does not train, modify checkpoints, or touch evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TOP1_COL = "gt_top1_hit_rate"
RANK_COL = "mean_normalized_gt_rank"
GT_COUNT_COL = "gt_count"


RUN_COL_CANDIDATES = ("checkpoint", "run", "run_name", "name", "variant")
RAW_ID_CANDIDATES = ("raw_id", "class_raw_id", "category_id", "gt_raw_id")
CLASS_NAME_CANDIDATES = ("class_name", "name", "category_name")


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path, *, required: bool = True) -> Dict:
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_col(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True, label: str = "column") -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find {label}; candidates={list(candidates)} columns={list(df.columns)}")
    return None


def _ensure_raw_id(df: pd.DataFrame) -> pd.DataFrame:
    raw_col = _pick_col(df, RAW_ID_CANDIDATES, label="raw id column")
    if raw_col != "raw_id":
        df = df.rename(columns={raw_col: "raw_id"})
    df["raw_id"] = pd.to_numeric(df["raw_id"], errors="coerce").astype("Int64")
    return df


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pearson(x: pd.Series, y: pd.Series) -> float:
    v = pd.DataFrame({"x": _safe_num(x), "y": _safe_num(y)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) < 2 or v["x"].nunique() <= 1 or v["y"].nunique() <= 1:
        return float("nan")
    return float(v["x"].corr(v["y"], method="pearson"))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    v = pd.DataFrame({"x": _safe_num(x), "y": _safe_num(y)}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) < 2 or v["x"].nunique() <= 1 or v["y"].nunique() <= 1:
        return float("nan")
    return float(v["x"].corr(v["y"], method="spearman"))


def _bool_rate(s: pd.Series) -> float:
    if len(s) == 0:
        return float("nan")
    return float(s.astype(bool).mean())


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if not math.isfinite(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if pd.isna(obj):
        return None
    return obj


def _load_per_class_attribution(compare_dir: Path, nohub_run: str, demand_run: str) -> pd.DataFrame:
    df = _read_csv(compare_dir / "per_class_attribution.csv")
    df = _ensure_raw_id(df)
    run_col = _pick_col(df, RUN_COL_CANDIDATES, label="run/checkpoint column")
    if run_col != "checkpoint":
        df = df.rename(columns={run_col: "checkpoint"})
    required = {"raw_id", "checkpoint", TOP1_COL, RANK_COL}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"per_class_attribution.csv missing columns: {missing}")

    keep_meta = [
        c
        for c in [
            "raw_id",
            "class_name",
            "gt_count",
            "candidate_size_mean",
            "gt_rank_mean",
            "certificate_family",
            "certificate_type",
            "resolved_round",
            "base_group",
            "person_conditioned",
        ]
        if c in df.columns
    ]
    value_cols = [c for c in [TOP1_COL, RANK_COL, "gt_rank_mean", "candidate_size_mean"] if c in df.columns]

    sub = df[df["checkpoint"].isin([nohub_run, demand_run])].copy()
    if sub["checkpoint"].nunique() < 2:
        found = sorted(df["checkpoint"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"Need both runs in per_class_attribution.csv. nohub_run={nohub_run}, demand_run={demand_run}, found={found}"
        )

    wide = sub.pivot_table(index="raw_id", columns="checkpoint", values=value_cols, aggfunc="first")
    wide.columns = [f"{metric}__{run}" for metric, run in wide.columns]
    wide = wide.reset_index()

    meta_cols = [c for c in keep_meta if c != "raw_id"]
    meta = sub.sort_values("checkpoint").drop_duplicates("raw_id")[["raw_id"] + meta_cols]
    merged = meta.merge(wide, on="raw_id", how="left")

    # Standard names.
    rename = {
        f"{TOP1_COL}__{nohub_run}": "nohub_gt_top1_hit_rate",
        f"{TOP1_COL}__{demand_run}": "demand_gt_top1_hit_rate",
        f"{RANK_COL}__{nohub_run}": "nohub_mean_normalized_gt_rank",
        f"{RANK_COL}__{demand_run}": "demand_mean_normalized_gt_rank",
        f"gt_rank_mean__{nohub_run}": "nohub_gt_rank_mean",
        f"gt_rank_mean__{demand_run}": "demand_gt_rank_mean",
    }
    merged = merged.rename(columns={k: v for k, v in rename.items() if k in merged.columns})
    merged["training_delta_gt_top1_hit_rate"] = (
        _safe_num(merged["demand_gt_top1_hit_rate"]) - _safe_num(merged["nohub_gt_top1_hit_rate"])
    )
    merged["training_delta_mean_normalized_gt_rank"] = (
        _safe_num(merged["demand_mean_normalized_gt_rank"]) - _safe_num(merged["nohub_mean_normalized_gt_rank"])
    )
    if "demand_gt_rank_mean" in merged.columns and "nohub_gt_rank_mean" in merged.columns:
        merged["training_delta_gt_rank_mean"] = _safe_num(merged["demand_gt_rank_mean"]) - _safe_num(merged["nohub_gt_rank_mean"])
    merged["training_improved_top1"] = merged["training_delta_gt_top1_hit_rate"] > 0
    merged["training_improved_rank"] = merged["training_delta_mean_normalized_gt_rank"] < 0
    merged["training_improved_either"] = merged["training_improved_top1"] | merged["training_improved_rank"]
    merged["training_safe_improved"] = (merged["training_delta_gt_top1_hit_rate"] >= 0) & (
        merged["training_delta_mean_normalized_gt_rank"] <= 0
    ) & (
        (merged["training_delta_gt_top1_hit_rate"] > 0) | (merged["training_delta_mean_normalized_gt_rank"] < 0)
    )
    merged["training_degraded_top1"] = merged["training_delta_gt_top1_hit_rate"] < 0
    merged["training_degraded_rank"] = merged["training_delta_mean_normalized_gt_rank"] > 0
    merged["training_degraded_either"] = merged["training_degraded_top1"] | merged["training_degraded_rank"]
    merged["training_strictly_degraded"] = (merged["training_delta_gt_top1_hit_rate"] <= 0) & (
        merged["training_delta_mean_normalized_gt_rank"] >= 0
    ) & (
        (merged["training_delta_gt_top1_hit_rate"] < 0) | (merged["training_delta_mean_normalized_gt_rank"] > 0)
    )
    return merged


def _load_bonus_table(demand_train_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    cfg = _read_json(demand_train_dir / "demand_floor_bonus_config.json", required=False)
    bonus = _read_csv(demand_train_dir / "demand_floor_bonus_static.csv", required=False)
    if bonus.empty:
        return bonus, cfg
    bonus = _ensure_raw_id(bonus)
    for c in ["bonus", "bonus_rank", "candidate_support", "responsibility_mass", "top1_count", "gt_count"]:
        if c in bonus.columns:
            bonus[c] = _safe_num(bonus[c])
    keep = [
        c
        for c in [
            "raw_id",
            "class_name",
            "bonus",
            "bonus_rank",
            "metric",
            "metric_raw_value",
            "metric_norm_value",
            "candidate_support",
            "responsibility_mass",
            "top1_count",
            "gt_count",
            "is_nohub_degraded_either",
            "certificate_family",
            "certificate_type",
            "resolved_round",
            "base_group",
            "person_conditioned",
        ]
        if c in bonus.columns
    ]
    bonus = bonus[keep].drop_duplicates("raw_id")
    bonus = bonus.rename(columns={
        "class_name": "bonus_class_name",
        "gt_count": "bonus_gt_count",
        "certificate_family": "bonus_certificate_family",
        "certificate_type": "bonus_certificate_type",
        "resolved_round": "bonus_resolved_round",
        "base_group": "bonus_base_group",
        "person_conditioned": "bonus_person_conditioned",
    })
    return bonus, cfg


def _load_replay_table(replay_dir: Path, explicit_setting_id: Optional[str]) -> Tuple[pd.DataFrame, Dict, Optional[str]]:
    summary = _read_json(replay_dir / "summary.json", required=False)
    setting_id = explicit_setting_id
    if not setting_id:
        setting_id = (summary.get("recommended_setting") or {}).get("setting_id")
    replay = _read_csv(replay_dir / "per_class_replay_delta.csv", required=False)
    if replay.empty:
        return replay, summary, setting_id
    replay = _ensure_raw_id(replay)
    if setting_id and "setting_id" in replay.columns:
        r2 = replay[replay["setting_id"].astype(str) == str(setting_id)].copy()
        if not r2.empty:
            replay = r2
    # Find likely replay delta columns.
    rename = {}
    for cand in ["delta_gt_top1_hit_rate_vs_no_bonus", "replay_delta_gt_top1_hit_rate", "delta_gt_top1_hit_rate"]:
        if cand in replay.columns:
            rename[cand] = "replay_delta_gt_top1_hit_rate"
            break
    for cand in [
        "delta_mean_normalized_gt_rank_vs_no_bonus",
        "replay_delta_mean_normalized_gt_rank",
        "delta_mean_normalized_gt_rank",
    ]:
        if cand in replay.columns:
            rename[cand] = "replay_delta_mean_normalized_gt_rank"
            break
    replay = replay.rename(columns=rename)
    keep = [
        c
        for c in [
            "raw_id",
            "class_name",
            "setting_id",
            "replay_delta_gt_top1_hit_rate",
            "replay_delta_mean_normalized_gt_rank",
            "delta_gt_rank_mean_vs_no_bonus",
            "bonus",
            "bonus_rank",
            "gt_count",
            "certificate_family",
            "certificate_type",
            "resolved_round",
            "person_conditioned",
            "base_group",
        ]
        if c in replay.columns
    ]
    replay = replay[keep].drop_duplicates("raw_id")
    replay = replay.rename(columns={
        "class_name": "replay_class_name",
        "bonus": "replay_bonus",
        "bonus_rank": "replay_bonus_rank",
        "gt_count": "replay_gt_count",
        "certificate_family": "replay_certificate_family",
        "certificate_type": "replay_certificate_type",
        "resolved_round": "replay_resolved_round",
        "person_conditioned": "replay_person_conditioned",
        "base_group": "replay_base_group",
    })
    return replay, summary, setting_id


def _merge_tables(pc: pd.DataFrame, bonus: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    df = pc.copy()
    if not bonus.empty:
        df = df.merge(bonus, on="raw_id", how="left")
    else:
        df["bonus"] = np.nan
    if not replay.empty:
        df = df.merge(replay, on="raw_id", how="left", suffixes=("", "__replay"))
    df["bonus"] = _safe_num(df.get("bonus", pd.Series(index=df.index, dtype=float))).fillna(0.0)
    df["has_bonus"] = df["bonus"] > 0
    if "bonus_rank" in df.columns:
        df["bonus_rank"] = _safe_num(df["bonus_rank"])
    if "gt_count" not in df.columns and "bonus_gt_count" in df.columns:
        df["gt_count"] = df["bonus_gt_count"]
    if "class_name" not in df.columns:
        for c in ["bonus_class_name", "replay_class_name"]:
            if c in df.columns:
                df["class_name"] = df[c]
                break
    for c in ["replay_delta_gt_top1_hit_rate", "replay_delta_mean_normalized_gt_rank"]:
        if c not in df.columns:
            df[c] = np.nan
    df["replay_improved_top1"] = _safe_num(df["replay_delta_gt_top1_hit_rate"]) > 0
    df["replay_improved_rank"] = _safe_num(df["replay_delta_mean_normalized_gt_rank"]) < 0
    df["replay_improved_either"] = df["replay_improved_top1"] | df["replay_improved_rank"]
    df["replay_degraded_either"] = (_safe_num(df["replay_delta_gt_top1_hit_rate"]) < 0) | (
        _safe_num(df["replay_delta_mean_normalized_gt_rank"]) > 0
    )
    df["replay_training_top1_same_sign"] = np.sign(_safe_num(df["replay_delta_gt_top1_hit_rate"]).fillna(0)) == np.sign(
        _safe_num(df["training_delta_gt_top1_hit_rate"]).fillna(0)
    )
    df["replay_training_rank_same_sign"] = np.sign(_safe_num(df["replay_delta_mean_normalized_gt_rank"]).fillna(0)) == np.sign(
        _safe_num(df["training_delta_mean_normalized_gt_rank"]).fillna(0)
    )
    return df


def _group_delta_summary(group_df: pd.DataFrame, nohub_run: str, demand_run: str) -> pd.DataFrame:
    if group_df.empty:
        return pd.DataFrame()
    group_df = group_df.copy()
    run_col = _pick_col(group_df, RUN_COL_CANDIDATES, required=False, label="group run column")
    if run_col is None:
        return pd.DataFrame()
    if run_col != "checkpoint":
        group_df = group_df.rename(columns={run_col: "checkpoint"})
    for c in [TOP1_COL, RANK_COL, "gt_rank_mean"]:
        if c in group_df.columns:
            group_df[c] = _safe_num(group_df[c])
    idx_cols = [c for c in ["group_name", "group_value"] if c in group_df.columns]
    if not idx_cols:
        return pd.DataFrame()
    rows = []
    for keys, g in group_df[group_df["checkpoint"].isin([nohub_run, demand_run])].groupby(idx_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        gmap = {str(r["checkpoint"]): r for _, r in g.iterrows()}
        if nohub_run not in gmap or demand_run not in gmap:
            continue
        out = {c: v for c, v in zip(idx_cols, keys)}
        out["gt_count"] = int(gmap[demand_run].get("gt_count", gmap[nohub_run].get("gt_count", 0)))
        out["nohub_gt_top1_hit_rate"] = gmap[nohub_run].get(TOP1_COL)
        out["demand_gt_top1_hit_rate"] = gmap[demand_run].get(TOP1_COL)
        out["training_delta_gt_top1_hit_rate"] = out["demand_gt_top1_hit_rate"] - out["nohub_gt_top1_hit_rate"]
        out["nohub_mean_normalized_gt_rank"] = gmap[nohub_run].get(RANK_COL)
        out["demand_mean_normalized_gt_rank"] = gmap[demand_run].get(RANK_COL)
        out["training_delta_mean_normalized_gt_rank"] = out["demand_mean_normalized_gt_rank"] - out["nohub_mean_normalized_gt_rank"]
        if "gt_rank_mean" in group_df.columns:
            out["nohub_gt_rank_mean"] = gmap[nohub_run].get("gt_rank_mean")
            out["demand_gt_rank_mean"] = gmap[demand_run].get("gt_rank_mean")
            out["training_delta_gt_rank_mean"] = out["demand_gt_rank_mean"] - out["nohub_gt_rank_mean"]
        rows.append(out)
    return pd.DataFrame(rows)


def _correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    x_cols = [
        c
        for c in [
            "bonus",
            "bonus_rank",
            "metric_raw_value",
            "metric_norm_value",
            "candidate_support",
            "responsibility_mass",
            "top1_count",
            "replay_delta_gt_top1_hit_rate",
            "replay_delta_mean_normalized_gt_rank",
        ]
        if c in df.columns
    ]
    y_cols = ["training_delta_gt_top1_hit_rate", "training_delta_mean_normalized_gt_rank"]
    rows = []
    for x in x_cols:
        for y in y_cols:
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "n": int(pd.DataFrame({"x": _safe_num(df[x]), "y": _safe_num(df[y])}).dropna().shape[0]),
                    "pearson": _pearson(df[x], df[y]),
                    "spearman": _spearman(df[x], df[y]),
                }
            )
    # Bonus-only slice.
    bdf = df[df["has_bonus"]].copy()
    for x in x_cols:
        for y in y_cols:
            if x not in bdf.columns:
                continue
            rows.append(
                {
                    "x": f"bonus_only::{x}",
                    "y": y,
                    "n": int(pd.DataFrame({"x": _safe_num(bdf[x]), "y": _safe_num(bdf[y])}).dropna().shape[0]),
                    "pearson": _pearson(bdf[x], bdf[y]),
                    "spearman": _spearman(bdf[x], bdf[y]),
                }
            )
    return pd.DataFrame(rows)


def _replay_consistency(df: pd.DataFrame) -> pd.DataFrame:
    slices = {
        "all_classes": df,
        "bonus_classes": df[df["has_bonus"]],
        "non_bonus_classes": df[~df["has_bonus"]],
        "replay_improved_either": df[df["replay_improved_either"]],
        "replay_degraded_either": df[df["replay_degraded_either"]],
    }
    rows = []
    for name, g in slices.items():
        if len(g) == 0:
            rows.append({"slice": name, "class_count": 0})
            continue
        rows.append(
            {
                "slice": name,
                "class_count": int(len(g)),
                "gt_count_sum": int(_safe_num(g.get("gt_count", pd.Series(dtype=float))).fillna(0).sum()),
                "training_improved_either_rate": _bool_rate(g["training_improved_either"]),
                "training_safe_improved_rate": _bool_rate(g["training_safe_improved"]),
                "training_degraded_either_rate": _bool_rate(g["training_degraded_either"]),
                "training_strictly_degraded_rate": _bool_rate(g["training_strictly_degraded"]),
                "mean_training_delta_top1": float(_safe_num(g["training_delta_gt_top1_hit_rate"]).mean()),
                "mean_training_delta_rank": float(_safe_num(g["training_delta_mean_normalized_gt_rank"]).mean()),
                "mean_replay_delta_top1": float(_safe_num(g["replay_delta_gt_top1_hit_rate"]).mean()),
                "mean_replay_delta_rank": float(_safe_num(g["replay_delta_mean_normalized_gt_rank"]).mean()),
                "top1_sign_agreement_rate": _bool_rate(g["replay_training_top1_same_sign"]),
                "rank_sign_agreement_rate": _bool_rate(g["replay_training_rank_same_sign"]),
                "replay_top1_vs_training_top1_pearson": _pearson(g["replay_delta_gt_top1_hit_rate"], g["training_delta_gt_top1_hit_rate"]),
                "replay_rank_vs_training_rank_pearson": _pearson(
                    g["replay_delta_mean_normalized_gt_rank"], g["training_delta_mean_normalized_gt_rank"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _select_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df[[c for c in cols if c in df.columns]].copy()


def _write_takeover(
    path: Path,
    summary: Dict,
    group_summary: pd.DataFrame,
    corr: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    rec = summary.get("recommendation", {})
    key = summary.get("key_deltas", {})
    with path.open("w", encoding="utf-8") as f:
        f.write("# Demand-floor Training Delta Audit\n\n")
        f.write(f"Status: `{summary.get('status')}`\n\n")
        f.write(f"Interpretation: `{summary.get('interpretation')}`\n\n")
        f.write("## Key deltas: demand-floor training vs NoHub\n\n")
        for k, v in key.items():
            f.write(f"- {k}: `{v}`\n")
        f.write("\n## Recommendation\n\n")
        for k, v in rec.items():
            f.write(f"- {k}: `{v}`\n")
        f.write("\n## Notes\n\n")
        for note in summary.get("notes", []):
            f.write(f"- {note}\n")
        f.write("\n## Outputs\n\n")
        for name, p in summary.get("outputs", {}).items():
            f.write(f"- {name}: `{p}`\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only attribution audit for demand-floor training deltas.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--compare_dir", required=True, help="Directory containing nohub vs demand-floor attribution compare outputs.")
    ap.add_argument("--replay_dir", required=True, help="Directory containing demand_floor_replay outputs.")
    ap.add_argument("--demand_floor_train_dir", required=True, help="train/prealign dir with demand_floor_bonus_static.csv/config.json.")
    ap.add_argument("--nohub_run", default="soft_e2e_nohub")
    ap.add_argument("--demand_floor_run", default="soft_e2e_nohub_demand_floor_veryweak")
    ap.add_argument("--replay_setting_id", default=None)
    ap.add_argument("--top_k", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    compare_dir = Path(args.compare_dir)
    replay_dir = Path(args.replay_dir)
    train_dir = Path(args.demand_floor_train_dir)

    pc = _load_per_class_attribution(compare_dir, args.nohub_run, args.demand_floor_run)
    bonus, bonus_cfg = _load_bonus_table(train_dir)
    replay, replay_summary, replay_setting_id = _load_replay_table(replay_dir, args.replay_setting_id)
    df = _merge_tables(pc, bonus, replay)

    group_df = _read_csv(compare_dir / "summary_by_group.csv", required=False)
    group_delta = _group_delta_summary(group_df, args.nohub_run, args.demand_floor_run)

    corr = _correlation_table(df)
    consistency = _replay_consistency(df)

    # Sort outputs.
    base_cols = [
        "raw_id",
        "class_name",
        "gt_count",
        "bonus",
        "bonus_rank",
        "metric_raw_value",
        "metric_norm_value",
        "candidate_support",
        "responsibility_mass",
        "top1_count",
        "nohub_gt_top1_hit_rate",
        "demand_gt_top1_hit_rate",
        "training_delta_gt_top1_hit_rate",
        "nohub_mean_normalized_gt_rank",
        "demand_mean_normalized_gt_rank",
        "training_delta_mean_normalized_gt_rank",
        "replay_delta_gt_top1_hit_rate",
        "replay_delta_mean_normalized_gt_rank",
        "training_improved_either",
        "training_degraded_either",
        "training_safe_improved",
        "training_strictly_degraded",
        "certificate_family",
        "certificate_type",
        "resolved_round",
        "base_group",
        "person_conditioned",
        "is_nohub_degraded_either",
    ]
    delta_table = _select_columns(df, base_cols).sort_values(
        ["training_delta_gt_top1_hit_rate", "training_delta_mean_normalized_gt_rank"], ascending=[False, True]
    )

    top_improved = _select_columns(df[df["training_improved_either"]], base_cols).sort_values(
        ["training_delta_gt_top1_hit_rate", "training_delta_mean_normalized_gt_rank"], ascending=[False, True]
    ).head(args.top_k)
    top_degraded = _select_columns(df[df["training_degraded_either"]], base_cols).sort_values(
        ["training_delta_gt_top1_hit_rate", "training_delta_mean_normalized_gt_rank"], ascending=[True, False]
    ).head(args.top_k)

    fam = df.get("certificate_family", pd.Series(index=df.index, dtype=object)).astype(str)
    ctype = df.get("certificate_type", pd.Series(index=df.index, dtype=object)).astype(str)
    anchor_mask = fam.str.contains("anchor", case=False, na=False) | ctype.str.contains("anchor", case=False, na=False)
    initial_person_mask = fam.str.contains("initial|person", case=False, na=False) | ctype.str.contains(
        "initial|person", case=False, na=False
    )
    anchor_gain = _select_columns(df[anchor_mask], base_cols).sort_values(
        ["training_delta_gt_top1_hit_rate", "training_delta_mean_normalized_gt_rank"], ascending=[False, True]
    )
    initial_person_reg = _select_columns(df[initial_person_mask & df["training_degraded_either"]], base_cols).sort_values(
        ["training_delta_gt_top1_hit_rate", "training_delta_mean_normalized_gt_rank"], ascending=[True, False]
    )

    # Risk: high bonus but bad training delta.
    risk = df[df["has_bonus"]].copy()
    risk["risk_score"] = (
        (-_safe_num(risk["training_delta_gt_top1_hit_rate"]).fillna(0) * 10.0)
        + (_safe_num(risk["training_delta_mean_normalized_gt_rank"]).fillna(0) * 5.0)
        + (_safe_num(risk.get("gt_count", pd.Series(index=risk.index))).fillna(0) / max(float(_safe_num(risk.get("gt_count", pd.Series(index=risk.index))).fillna(0).max()), 1.0))
    )
    risk = _select_columns(risk.sort_values("risk_score", ascending=False), base_cols + ["risk_score"]).head(max(args.top_k, 50))

    # Sparse candidates: safe training improvement, preferably replay-aligned, and not obvious regressors.
    cand = df[df["has_bonus"]].copy()
    cand["sparse_candidate_score"] = (
        _safe_num(cand["training_delta_gt_top1_hit_rate"]).fillna(0) * 20.0
        - _safe_num(cand["training_delta_mean_normalized_gt_rank"]).fillna(0) * 10.0
        + _safe_num(cand["replay_delta_gt_top1_hit_rate"]).fillna(0) * 5.0
        - _safe_num(cand["replay_delta_mean_normalized_gt_rank"]).fillna(0) * 3.0
    )
    sparse = cand[cand["training_safe_improved"]].sort_values("sparse_candidate_score", ascending=False)
    sparse = _select_columns(sparse, base_cols + ["sparse_candidate_score"]).head(max(args.top_k, 80))

    # Write tables.
    paths = {
        "training_delta_by_class": out / "training_delta_by_class.csv",
        "bonus_effect_correlation": out / "bonus_effect_correlation.csv",
        "replay_vs_training_consistency": out / "replay_vs_training_consistency.csv",
        "summary_delta_vs_nohub_by_group": out / "summary_delta_vs_nohub_by_group.csv",
        "top20_training_improved_classes": out / "top20_training_improved_classes.csv",
        "top20_training_degraded_classes": out / "top20_training_degraded_classes.csv",
        "anchor_gain_breakdown": out / "anchor_gain_breakdown.csv",
        "initial_person_regression_breakdown": out / "initial_person_regression_breakdown.csv",
        "bonus_rank_risk_table": out / "bonus_rank_risk_table.csv",
        "sparse_bonus_candidate_table": out / "sparse_bonus_candidate_table.csv",
    }
    delta_table.to_csv(paths["training_delta_by_class"], index=False)
    corr.to_csv(paths["bonus_effect_correlation"], index=False)
    consistency.to_csv(paths["replay_vs_training_consistency"], index=False)
    group_delta.to_csv(paths["summary_delta_vs_nohub_by_group"], index=False)
    top_improved.to_csv(paths["top20_training_improved_classes"], index=False)
    top_degraded.to_csv(paths["top20_training_degraded_classes"], index=False)
    anchor_gain.to_csv(paths["anchor_gain_breakdown"], index=False)
    initial_person_reg.to_csv(paths["initial_person_regression_breakdown"], index=False)
    risk.to_csv(paths["bonus_rank_risk_table"], index=False)
    sparse.to_csv(paths["sparse_bonus_candidate_table"], index=False)

    # Key deltas from group table.
    key_deltas = {}
    if not group_delta.empty:
        def _g(group_name, group_value="overall"):
            g = group_delta[(group_delta.get("group_name") == group_name) & (group_delta.get("group_value").astype(str) == str(group_value))]
            if g.empty:
                return None
            return g.iloc[0]
        for label, gn, gv in [
            ("overall", "overall", "overall"),
            ("anchor_conditioned", "certificate_family", "anchor_conditioned"),
            ("initial_context_identifiable", "certificate_family", "initial_context_identifiable"),
            ("person_conditioned", "certificate_family", "person_conditioned"),
            ("base_unobserved", "base_observed_unobserved", "base_unobserved"),
        ]:
            r = _g(gn, gv)
            if r is not None:
                key_deltas[f"{label}_delta_gt_top1_hit_rate"] = float(r["training_delta_gt_top1_hit_rate"])
                key_deltas[f"{label}_delta_mean_normalized_gt_rank"] = float(r["training_delta_mean_normalized_gt_rank"])

    bonus_classes = df[df["has_bonus"]]
    non_bonus_classes = df[~df["has_bonus"]]
    notes = []
    interpretation = "DEMAND_FLOOR_TRAINING_DELTA_AUDIT_COMPLETE"
    if key_deltas.get("overall_delta_gt_top1_hit_rate", 0.0) < 0 or key_deltas.get("overall_delta_mean_normalized_gt_rank", 0.0) > 0:
        interpretation = "TRAINING_NEGATIVE_OVERALL__DO_NOT_ADOPT_CURRENT_DEMAND_FLOOR"
        notes.append("Demand-floor training is worse than NoHub overall; do not adopt the current veryweak training variant.")
    if key_deltas.get("anchor_conditioned_delta_gt_top1_hit_rate", 0.0) > 0:
        notes.append("Anchor-conditioned classes improved, so demand-side signal is not empty; the issue is global/static bonus breadth and optimization coupling.")
    if len(sparse) > 0:
        notes.append("Sparse candidate table contains classes with safe positive training deltas; use it only for follow-up replay, not as direct training proof.")

    summary = {
        "status": "PASS",
        "output_dir": str(out),
        "compare_dir": str(compare_dir),
        "replay_dir": str(replay_dir),
        "demand_floor_train_dir": str(train_dir),
        "nohub_run": args.nohub_run,
        "demand_floor_run": args.demand_floor_run,
        "replay_setting_id": replay_setting_id,
        "bonus_config": bonus_cfg,
        "class_count": int(len(df)),
        "bonus_class_count_joined": int(df["has_bonus"].sum()),
        "non_bonus_class_count_joined": int((~df["has_bonus"]).sum()),
        "training_improved_either_rate_all": _bool_rate(df["training_improved_either"]),
        "training_degraded_either_rate_all": _bool_rate(df["training_degraded_either"]),
        "training_improved_either_rate_bonus": _bool_rate(bonus_classes["training_improved_either"]) if len(bonus_classes) else None,
        "training_degraded_either_rate_bonus": _bool_rate(bonus_classes["training_degraded_either"]) if len(bonus_classes) else None,
        "replay_training_top1_pearson_all": _pearson(df["replay_delta_gt_top1_hit_rate"], df["training_delta_gt_top1_hit_rate"]),
        "replay_training_rank_pearson_all": _pearson(
            df["replay_delta_mean_normalized_gt_rank"], df["training_delta_mean_normalized_gt_rank"]
        ),
        "key_deltas": key_deltas,
        "interpretation": interpretation,
        "recommendation": {
            "adopt_current_demand_floor_training": False,
            "safe_next_step": "Run sparse demand-floor replay only if the sparse candidate table is compact and aligned with anchor/base_unobserved gains.",
            "do_not_do": "Do not run another broad 263-class static-bonus 15ep training before a sparse replay verifies reduced collateral damage.",
        },
        "notes": notes,
        "outputs": {k: str(v) for k, v in paths.items()},
    }

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, indent=2, ensure_ascii=False)
    _write_takeover(out / "DEMAND_FLOOR_TRAINING_DELTA_AUDIT_TAKEOVER.md", summary, group_delta, corr, consistency)

    print(json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
