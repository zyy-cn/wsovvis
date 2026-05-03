#!/usr/bin/env python3
"""Row-level rescue audit for residual-gated GT-clean pilot.

This is a read-only reducer. It joins:
  * old row-level oracle/weak_base/weak_nohub assignment gap CSV,
  * residual-gated row seed pool,
  * balanced training manifest,
  * pilot row-level after predictions for one or more modes.

It answers the missing causal accounting question:
  old weak/nohub wrong -> pilot after correct?

No training, no control-plane edits, no feature loading, no GPU required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (list(rows[0].keys()) if rows else ["empty"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(obj), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _truth(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(str(x)))
        except Exception:
            return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _mean(vals: Sequence[float]) -> float:
    return float(sum(vals) / max(len(vals), 1))


def _key(row: Mapping[str, Any]) -> Tuple[str, str]:
    tid = str(row.get("trajectory_id", "")).strip()
    raw = row.get("gt_raw_id", row.get("raw_id", ""))
    rid = _as_int(raw)
    return tid, str(rid if rid is not None else str(raw).strip())


def _default_manifest_path(run_root: Path, dataset: str) -> Path:
    return run_root / "analysis" / "residual_gated_training_manifest" / dataset / "balanced_training_manifest.csv"


def _default_seed_pool_path(run_root: Path, dataset: str) -> Path:
    return run_root / "analysis" / "residual_gated_row_seed_pool" / dataset / "residual_gated_row_seed_pool.csv"


def _default_row_gap_path(repo_root: Path, dataset: str) -> Path:
    return (
        repo_root / "codex" / "outputs" / "G8_inference_and_eval"
        / "gt_clean_weak_fully_overfit_capacity_20260502"
        / "analysis" / "assignment_oracle_gap_audit" / dataset / "base_vocab" / "row_level_assignment_gap.csv"
    )


def _default_pilot_root(run_root: Path, dataset: str, mode: str) -> Path:
    return run_root / "outputs" / "residual_gated_gtclean_pilot" / dataset / mode


def _default_out_dir(run_root: Path, dataset: str) -> Path:
    return run_root / "analysis" / "residual_gated_row_rescue" / dataset


def _load_mode_predictions(run_root: Path, dataset: str, mode: str, explicit_root: str = "") -> Tuple[Dict[Tuple[str, str], Dict[str, str]], Dict[str, Any]]:
    root = Path(explicit_root).expanduser().resolve() if explicit_root else _default_pilot_root(run_root, dataset, mode)
    pred_csv = root / "analysis" / "eval_after_row_predictions.csv"
    summary_json = root / "final_summary.json"
    if not pred_csv.is_file():
        raise FileNotFoundError(f"missing row-level after prediction CSV for mode={mode}: {pred_csv}")
    pred_rows = _read_csv(pred_csv)
    pred_by_key: Dict[Tuple[str, str], Dict[str, str]] = {}
    duplicate = 0
    for r in pred_rows:
        k = _key(r)
        if k in pred_by_key:
            duplicate += 1
        pred_by_key.setdefault(k, r)
    summary: Dict[str, Any] = {"mode": mode, "output_root": str(root), "prediction_csv": str(pred_csv), "prediction_rows": len(pred_rows), "prediction_unique_keys": len(pred_by_key), "duplicate_prediction_keys": duplicate}
    if summary_json.is_file():
        try:
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            summary["final_summary"] = payload
        except Exception as e:
            summary["final_summary_error"] = str(e)
    return pred_by_key, summary


def _summarize_bool(rows: Sequence[Mapping[str, Any]], field: str) -> Tuple[int, float]:
    n = len(rows)
    hit = sum(1 for r in rows if _truth(r.get(field)))
    return hit, float(hit / max(n, 1))


def _aggregate_group(rows: Sequence[Mapping[str, Any]], group_name: str) -> Dict[str, Any]:
    n = len(rows)
    if n <= 0:
        return {
            "group": group_name,
            "row_count": 0,
            "old_oracle_top1_rate": 0.0,
            "old_weak_base_top1_rate": 0.0,
            "old_weak_nohub_top1_rate": 0.0,
            "after_top1_rate": 0.0,
            "after_top5_rate": 0.0,
            "old_nohub_wrong_after_correct_count": 0,
            "old_nohub_wrong_after_correct_rate": 0.0,
            "old_base_wrong_after_correct_count": 0,
            "old_base_wrong_after_correct_rate": 0.0,
            "old_nohub_other_positive_after_correct_count": 0,
            "old_nohub_other_positive_after_correct_rate": 0.0,
            "old_base_other_positive_after_correct_count": 0,
            "old_base_other_positive_after_correct_rate": 0.0,
            "old_nohub_correct_after_wrong_count": 0,
            "old_nohub_correct_after_wrong_rate": 0.0,
        }
    oracle_hit, oracle_rate = _summarize_bool(rows, "old_oracle_top1_is_gt")
    base_hit, base_rate = _summarize_bool(rows, "old_weak_base_top1_is_gt")
    nohub_hit, nohub_rate = _summarize_bool(rows, "old_weak_nohub_top1_is_gt")
    after_hit, after_rate = _summarize_bool(rows, "after_gt_top1_hit")
    after5_hit, after5_rate = _summarize_bool(rows, "after_gt_top5_hit")

    old_nohub_wrong = [r for r in rows if not _truth(r.get("old_weak_nohub_top1_is_gt"))]
    old_base_wrong = [r for r in rows if not _truth(r.get("old_weak_base_top1_is_gt"))]
    old_nohub_op = [r for r in rows if str(r.get("old_weak_nohub_error_type", "")) == "other_positive_confusion"]
    old_base_op = [r for r in rows if str(r.get("old_weak_base_error_type", "")) == "other_positive_confusion"]
    old_nohub_correct = [r for r in rows if _truth(r.get("old_weak_nohub_top1_is_gt"))]

    def rescued(sub: Sequence[Mapping[str, Any]]) -> Tuple[int, float]:
        c = sum(1 for r in sub if _truth(r.get("after_gt_top1_hit")))
        return c, float(c / max(len(sub), 1))

    nh_rescue_c, nh_rescue_r = rescued(old_nohub_wrong)
    b_rescue_c, b_rescue_r = rescued(old_base_wrong)
    nh_op_c, nh_op_r = rescued(old_nohub_op)
    b_op_c, b_op_r = rescued(old_base_op)
    nh_regress_c = sum(1 for r in old_nohub_correct if not _truth(r.get("after_gt_top1_hit")))
    return {
        "group": group_name,
        "row_count": n,
        "old_oracle_top1_count": oracle_hit,
        "old_oracle_top1_rate": oracle_rate,
        "old_weak_base_top1_count": base_hit,
        "old_weak_base_top1_rate": base_rate,
        "old_weak_nohub_top1_count": nohub_hit,
        "old_weak_nohub_top1_rate": nohub_rate,
        "after_top1_count": after_hit,
        "after_top1_rate": after_rate,
        "after_top5_count": after5_hit,
        "after_top5_rate": after5_rate,
        "old_nohub_wrong_count": len(old_nohub_wrong),
        "old_nohub_wrong_after_correct_count": nh_rescue_c,
        "old_nohub_wrong_after_correct_rate": nh_rescue_r,
        "old_base_wrong_count": len(old_base_wrong),
        "old_base_wrong_after_correct_count": b_rescue_c,
        "old_base_wrong_after_correct_rate": b_rescue_r,
        "old_nohub_other_positive_count": len(old_nohub_op),
        "old_nohub_other_positive_after_correct_count": nh_op_c,
        "old_nohub_other_positive_after_correct_rate": nh_op_r,
        "old_base_other_positive_count": len(old_base_op),
        "old_base_other_positive_after_correct_count": b_op_c,
        "old_base_other_positive_after_correct_rate": b_op_r,
        "old_nohub_correct_count": len(old_nohub_correct),
        "old_nohub_correct_after_wrong_count": nh_regress_c,
        "old_nohub_correct_after_wrong_rate": float(nh_regress_c / max(len(old_nohub_correct), 1)),
        "mean_after_norm_rank": _mean([_as_float(r.get("after_gt_norm_rank"), 1.0) for r in rows]),
        "mean_after_rank": _mean([_as_float(r.get("after_gt_rank"), 9999.0) for r in rows]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Residual-gated row-level rescue audit")
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    ap.add_argument("--manifest_csv", default="")
    ap.add_argument("--seed_pool_csv", default="")
    ap.add_argument("--row_gap_csv", default="")
    ap.add_argument("--modes", default="hard_ce,hard_soft_proto")
    ap.add_argument("--primary_mode", default="hard_soft_proto")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--top_class_rows", type=int, default=200)
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    dataset = str(args.dataset_name)
    manifest_csv = Path(args.manifest_csv).expanduser().resolve() if args.manifest_csv else _default_manifest_path(run_root, dataset)
    seed_pool_csv = Path(args.seed_pool_csv).expanduser().resolve() if args.seed_pool_csv else _default_seed_pool_path(run_root, dataset)
    row_gap_csv = Path(args.row_gap_csv).expanduser().resolve() if args.row_gap_csv else _default_row_gap_path(repo_root, dataset)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else _default_out_dir(run_root, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p, name in [(manifest_csv, "manifest_csv"), (seed_pool_csv, "seed_pool_csv"), (row_gap_csv, "row_gap_csv")]:
        if not p.is_file():
            raise FileNotFoundError(f"{name} not found: {p}")

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    if str(args.primary_mode) not in modes:
        modes.append(str(args.primary_mode))

    old_rows = _read_csv(row_gap_csv)
    seed_rows = _read_csv(seed_pool_csv)
    manifest_rows = _read_csv(manifest_csv)

    row_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in old_rows:
        k = _key(r)
        row_meta.setdefault(k, {}).update({
            "trajectory_id": k[0],
            "gt_raw_id": k[1],
            "clip_id": r.get("clip_id", ""),
            "gt_class_name": r.get("gt_class_name", ""),
            "clip_y_size": r.get("clip_y_size", ""),
            "old_oracle_top1_is_gt": int(_truth(r.get("oracle_top1_is_gt"))),
            "old_oracle_gt_rank": r.get("oracle_gt_rank", ""),
            "old_weak_base_top1_is_gt": int(_truth(r.get("weak_base_top1_is_gt"))),
            "old_weak_base_gt_rank": r.get("weak_base_gt_rank", ""),
            "old_weak_base_top1_class_name": r.get("weak_base_top1_class_name", ""),
            "old_weak_base_error_type": r.get("weak_base_error_type", ""),
            "old_weak_nohub_top1_is_gt": int(_truth(r.get("weak_nohub_top1_is_gt"))),
            "old_weak_nohub_gt_rank": r.get("weak_nohub_gt_rank", ""),
            "old_weak_nohub_top1_class_name": r.get("weak_nohub_top1_class_name", ""),
            "old_weak_nohub_error_type": r.get("weak_nohub_error_type", ""),
            "old_nohub_rescued_baseline_wrong": int(_truth(r.get("nohub_rescued_baseline_wrong"))),
            "old_oracle_correct_weak_base_wrong": int(_truth(r.get("oracle_correct_weak_base_wrong"))),
        })
    for r in seed_rows:
        k = _key(r)
        row_meta.setdefault(k, {"trajectory_id": k[0], "gt_raw_id": k[1]}).update({
            "seed_type": r.get("seed_type", ""),
            "policy": r.get("policy", ""),
            "certificate_type": r.get("certificate_type", ""),
            "seed_clip_count": r.get("clip_count", ""),
            "seed_instance_count": r.get("instance_count", ""),
        })
    for r in manifest_rows:
        k = _key(r)
        lf = str(r.get("loss_family", ""))
        row_meta.setdefault(k, {"trajectory_id": k[0], "gt_raw_id": k[1]}).update({
            "manifest_selected": 1 if str(r.get("manifest_use", "train")) == "train" else 0,
            "manifest_loss_family": lf,
            "manifest_sample_weight": r.get("sample_weight", ""),
            "manifest_selection_reason": r.get("selection_reason", ""),
        })

    mode_predictions: Dict[str, Dict[Tuple[str, str], Dict[str, str]]] = {}
    mode_summaries: Dict[str, Any] = {}
    missing_modes: List[str] = []
    for mode in modes:
        try:
            pred, summary = _load_mode_predictions(run_root, dataset, mode)
            mode_predictions[mode] = pred
            mode_summaries[mode] = summary
        except FileNotFoundError as e:
            missing_modes.append(f"{mode}: {e}")

    if str(args.primary_mode) not in mode_predictions:
        raise FileNotFoundError(f"primary_mode={args.primary_mode} predictions missing. Missing modes: {missing_modes}")

    primary_pred = mode_predictions[str(args.primary_mode)]
    joined_rows: List[Dict[str, Any]] = []
    missing_after = 0
    for k, pred in sorted(primary_pred.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        meta = dict(row_meta.get(k, {"trajectory_id": k[0], "gt_raw_id": k[1]}))
        if k not in row_meta:
            meta["meta_missing_from_old_seed_manifest"] = 1
        meta.setdefault("seed_type", "absent_from_seed_pool")
        meta.setdefault("policy", "")
        meta.setdefault("certificate_type", "")
        meta.setdefault("manifest_selected", 0)
        meta.setdefault("manifest_loss_family", "unselected")
        meta["row_group"] = (
            f"selected_{meta.get('manifest_loss_family')}" if _as_int(meta.get("manifest_selected"), 0) == 1
            else f"seed_{meta.get('seed_type', 'absent_from_seed_pool')}_not_selected"
        )
        meta.update({
            "after_mode": str(args.primary_mode),
            "after_top1_raw_id": pred.get("top1_raw_id", ""),
            "after_gt_rank": pred.get("gt_rank", ""),
            "after_gt_norm_rank": pred.get("gt_norm_rank", ""),
            "after_gt_top1_hit": int(_truth(pred.get("gt_top1_hit"))),
            "after_gt_top5_hit": int(_truth(pred.get("gt_top5_hit"))),
        })
        for mode, pred_map in mode_predictions.items():
            p = pred_map.get(k)
            if p is None:
                meta[f"{mode}_after_top1"] = ""
                meta[f"{mode}_after_rank"] = ""
            else:
                meta[f"{mode}_after_top1"] = int(_truth(p.get("gt_top1_hit")))
                meta[f"{mode}_after_rank"] = p.get("gt_rank", "")
        joined_rows.append(meta)
    # Count old/seed/manifest keys missing from primary prediction.
    for k in row_meta.keys():
        if k not in primary_pred:
            missing_after += 1

    _write_csv(out_dir / "row_level_rescue_joined.csv", joined_rows)

    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    groups["all_primary_after_rows"] = joined_rows
    groups["manifest_selected_all"] = [r for r in joined_rows if _as_int(r.get("manifest_selected"), 0) == 1]
    groups["manifest_unselected_all"] = [r for r in joined_rows if _as_int(r.get("manifest_selected"), 0) != 1]
    for r in joined_rows:
        groups[str(r.get("row_group", "unknown"))].append(r)
        lf = str(r.get("manifest_loss_family", "unselected"))
        if _as_int(r.get("manifest_selected"), 0) == 1:
            groups[f"selected_loss_{lf}"].append(r)
        st = str(r.get("seed_type", "absent_from_seed_pool"))
        groups[f"seed_type_{st}"].append(r)
        old_err = str(r.get("old_weak_nohub_error_type", ""))
        if old_err:
            groups[f"old_nohub_error_{old_err}"].append(r)
        old_base_err = str(r.get("old_weak_base_error_type", ""))
        if old_base_err:
            groups[f"old_base_error_{old_base_err}"].append(r)

    group_rows = [_aggregate_group(rows, name) for name, rows in sorted(groups.items())]
    _write_csv(out_dir / "row_level_rescue_by_group.csv", group_rows)

    by_class: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in joined_rows:
        by_class[str(r.get("gt_raw_id", ""))].append(r)
    class_rows: List[Dict[str, Any]] = []
    for rid, rows in sorted(by_class.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999999):
        agg = _aggregate_group(rows, f"class_{rid}")
        selected_rows = [r for r in rows if _as_int(r.get("manifest_selected"), 0) == 1]
        agg.update({
            "raw_id": rid,
            "class_name": next((str(r.get("gt_class_name", "")) for r in rows if str(r.get("gt_class_name", ""))), ""),
            "selected_rows": len(selected_rows),
            "selected_loss_families": ";".join(sorted(set(str(r.get("manifest_loss_family", "")) for r in selected_rows if str(r.get("manifest_loss_family", ""))))),
            "seed_types": ";".join(sorted(set(str(r.get("seed_type", "")) for r in rows if str(r.get("seed_type", ""))))),
            "policies": ";".join(sorted(set(str(r.get("policy", "")) for r in rows if str(r.get("policy", ""))))),
        })
        class_rows.append(agg)
    class_rows_sorted = sorted(class_rows, key=lambda r: (_as_float(r.get("old_nohub_wrong_after_correct_count"), 0.0), _as_float(r.get("after_top1_count"), 0.0)), reverse=True)
    _write_csv(out_dir / "row_level_rescue_by_class.csv", class_rows_sorted)
    _write_csv(out_dir / "row_level_rescue_top_classes.csv", class_rows_sorted[: int(args.top_class_rows)])

    comparative_rows: List[Dict[str, Any]] = []
    for mode, pred_map in mode_predictions.items():
        mode_join: List[Dict[str, Any]] = []
        for k, pred in pred_map.items():
            meta = dict(row_meta.get(k, {"trajectory_id": k[0], "gt_raw_id": k[1]}))
            meta["after_gt_top1_hit"] = int(_truth(pred.get("gt_top1_hit")))
            meta["after_gt_top5_hit"] = int(_truth(pred.get("gt_top5_hit")))
            meta["after_gt_rank"] = pred.get("gt_rank", "")
            meta["after_gt_norm_rank"] = pred.get("gt_norm_rank", "")
            meta.setdefault("manifest_selected", 0)
            meta.setdefault("manifest_loss_family", "unselected")
            mode_join.append(meta)
        for name, rows in {
            "all": mode_join,
            "manifest_selected": [r for r in mode_join if _as_int(r.get("manifest_selected"), 0) == 1],
            "selected_hard_ce": [r for r in mode_join if str(r.get("manifest_loss_family", "")) == "hard_ce"],
            "selected_soft_ce": [r for r in mode_join if str(r.get("manifest_loss_family", "")) == "soft_ce"],
            "selected_prototype_calibration": [r for r in mode_join if str(r.get("manifest_loss_family", "")) == "prototype_calibration"],
        }.items():
            agg = _aggregate_group(rows, f"{mode}:{name}")
            agg["mode"] = mode
            agg["mode_group"] = name
            comparative_rows.append(agg)
    _write_csv(out_dir / "row_level_rescue_by_mode_group.csv", comparative_rows)

    summary = {
        "status": "PASS",
        "timestamp": _now(),
        "run_root": str(run_root),
        "dataset_name": dataset,
        "primary_mode": str(args.primary_mode),
        "modes_requested": modes,
        "missing_modes": missing_modes,
        "manifest_csv": str(manifest_csv),
        "seed_pool_csv": str(seed_pool_csv),
        "row_gap_csv": str(row_gap_csv),
        "out_dir": str(out_dir),
        "input_counts": {
            "old_gap_rows": len(old_rows),
            "seed_pool_rows": len(seed_rows),
            "manifest_rows": len(manifest_rows),
            "primary_prediction_rows": len(primary_pred),
            "joined_primary_rows": len(joined_rows),
            "row_meta_keys_missing_from_primary_prediction": missing_after,
        },
        "mode_prediction_summaries": mode_summaries,
        "primary_all_group": _aggregate_group(joined_rows, "all_primary_after_rows"),
        "primary_selected_group": _aggregate_group(groups["manifest_selected_all"], "manifest_selected_all"),
        "primary_unselected_group": _aggregate_group(groups["manifest_unselected_all"], "manifest_unselected_all"),
        "key_groups": {r["group"]: r for r in group_rows if r["group"] in {
            "selected_loss_hard_ce",
            "selected_loss_soft_ce",
            "selected_loss_prototype_calibration",
            "old_nohub_error_other_positive_confusion",
            "old_base_error_other_positive_confusion",
            "seed_type_deferred",
            "manifest_selected_all",
        }},
        "comparative_gates": [],
        "limitations": [
            "This is still a GT-clean pilot scorer, not the old weak_base/nohub scorer itself; old metrics are used for rescue-group stratification.",
            "Row-level after rescue is exact for rows exported by the pilot eval_after_row_predictions.csv.",
        ],
        "outputs": {
            "row_level_rescue_joined": str(out_dir / "row_level_rescue_joined.csv"),
            "row_level_rescue_by_group": str(out_dir / "row_level_rescue_by_group.csv"),
            "row_level_rescue_by_class": str(out_dir / "row_level_rescue_by_class.csv"),
            "row_level_rescue_top_classes": str(out_dir / "row_level_rescue_top_classes.csv"),
            "row_level_rescue_by_mode_group": str(out_dir / "row_level_rescue_by_mode_group.csv"),
            "summary": str(out_dir / "row_level_rescue_summary.json"),
            "takeover": str(out_dir / "RESIDUAL_GATED_ROW_RESCUE_TAKEOVER.md"),
        },
    }

    # Comparative gates if hard_ce and hard_soft_proto are both present.
    comp_by = {(r.get("mode"), r.get("mode_group")): r for r in comparative_rows}
    hs = comp_by.get(("hard_soft_proto", "manifest_selected"))
    hc = comp_by.get(("hard_ce", "manifest_selected"))
    if hs and hc:
        summary["comparative_gates"].append({
            "name": "hard_soft_proto_selected_old_nohub_wrong_rescue_ge_hard_ce",
            "status": "PASS" if _as_float(hs.get("old_nohub_wrong_after_correct_rate")) >= _as_float(hc.get("old_nohub_wrong_after_correct_rate")) else "FAIL",
            "hard_soft_proto_rate": hs.get("old_nohub_wrong_after_correct_rate"),
            "hard_ce_rate": hc.get("old_nohub_wrong_after_correct_rate"),
            "hard": False,
        })
        summary["comparative_gates"].append({
            "name": "hard_soft_proto_selected_after_top1_ge_hard_ce",
            "status": "PASS" if _as_float(hs.get("after_top1_rate")) >= _as_float(hc.get("after_top1_rate")) else "FAIL",
            "hard_soft_proto_rate": hs.get("after_top1_rate"),
            "hard_ce_rate": hc.get("after_top1_rate"),
            "hard": False,
        })

    _write_json(out_dir / "row_level_rescue_summary.json", summary)
    lines = [
        "# Residual-Gated Row-Level Rescue TAKEOVER",
        "",
        f"- status: {summary['status']}",
        f"- primary_mode: {args.primary_mode}",
        f"- joined_primary_rows: {len(joined_rows)}",
        f"- all after_top1_rate: {summary['primary_all_group'].get('after_top1_rate', 0.0):.6f}",
        f"- all old_nohub_wrong_after_correct: {summary['primary_all_group'].get('old_nohub_wrong_after_correct_count', 0)} / {summary['primary_all_group'].get('old_nohub_wrong_count', 0)} = {summary['primary_all_group'].get('old_nohub_wrong_after_correct_rate', 0.0):.6f}",
        f"- selected after_top1_rate: {summary['primary_selected_group'].get('after_top1_rate', 0.0):.6f}",
        f"- selected old_nohub_wrong_after_correct: {summary['primary_selected_group'].get('old_nohub_wrong_after_correct_count', 0)} / {summary['primary_selected_group'].get('old_nohub_wrong_count', 0)} = {summary['primary_selected_group'].get('old_nohub_wrong_after_correct_rate', 0.0):.6f}",
        f"- selected old_nohub_correct_after_wrong: {summary['primary_selected_group'].get('old_nohub_correct_after_wrong_count', 0)} / {summary['primary_selected_group'].get('old_nohub_correct_count', 0)} = {summary['primary_selected_group'].get('old_nohub_correct_after_wrong_rate', 0.0):.6f}",
        "",
        "## Outputs",
        "- row_level_rescue_summary.json",
        "- row_level_rescue_joined.csv",
        "- row_level_rescue_by_group.csv",
        "- row_level_rescue_by_class.csv",
        "- row_level_rescue_by_mode_group.csv",
    ]
    (out_dir / "RESIDUAL_GATED_ROW_RESCUE_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
