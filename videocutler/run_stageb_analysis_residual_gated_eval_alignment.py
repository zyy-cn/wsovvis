#!/usr/bin/env python3
"""Residual-gated GT-clean pilot eval-alignment audit.

This is a read-only reducer. It aligns:
  1) old row-level assignment gap metrics: oracle / weak_base / weak_nohub,
  2) residual-gated seed pool and balanced training manifest,
  3) pilot final summaries and by-class before/after deltas.

It does not train, does not touch control-plane files, and does not read large feature tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _norm_id(x: Any) -> str:
    s = "" if x is None else str(x).strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def _truth(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def _float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.mean(vals) if vals else 0.0


def _mode_output_root(run_root: Path, dataset_name: str, mode: str) -> Path:
    return run_root / "outputs" / "residual_gated_gtclean_pilot" / dataset_name / mode


def _default_manifest(run_root: Path, dataset_name: str) -> Path:
    return run_root / "analysis" / "residual_gated_training_manifest" / dataset_name / "balanced_training_manifest.csv"


def _default_seed_pool(run_root: Path, dataset_name: str) -> Path:
    return run_root / "analysis" / "residual_gated_row_seed_pool" / dataset_name / "residual_gated_row_seed_pool.csv"


def _default_row_gap() -> Path:
    return Path("/mnt/sda/zyy/code/wsovvis/codex/outputs/G8_inference_and_eval/gt_clean_weak_fully_overfit_capacity_20260502/analysis/assignment_oracle_gap_audit/lvvis_train_base/base_vocab/row_level_assignment_gap.csv")


def _row_key(row: Dict[str, str]) -> Tuple[str, str]:
    return (str(row.get("trajectory_id", "")).strip(), _norm_id(row.get("gt_raw_id") or row.get("raw_id")))


def _aggregate_old_gap(row_gap_rows: List[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    by_class: Dict[str, Dict[str, Any]] = {}
    total = Counter()
    error_base = Counter()
    error_nohub = Counter()

    for r in row_gap_rows:
        rid = _norm_id(r.get("gt_raw_id"))
        if not rid:
            continue
        c = by_class.setdefault(rid, {
            "raw_id": rid,
            "class_name": r.get("gt_class_name", ""),
            "old_row_count": 0,
            "old_oracle_top1_count": 0,
            "old_weak_base_top1_count": 0,
            "old_weak_nohub_top1_count": 0,
            "old_nohub_rescued_count": 0,
            "old_oracle_correct_weak_base_wrong_count": 0,
            "old_base_other_positive_count": 0,
            "old_nohub_other_positive_count": 0,
        })
        c["old_row_count"] += 1
        total["rows"] += 1

        oracle_ok = _truth(r.get("oracle_top1_is_gt"))
        base_ok = _truth(r.get("weak_base_top1_is_gt"))
        nohub_ok = _truth(r.get("weak_nohub_top1_is_gt"))
        rescued = _truth(r.get("nohub_rescued_baseline_wrong"))
        oracle_gap = _truth(r.get("oracle_correct_weak_base_wrong"))
        base_err = str(r.get("weak_base_error_type", ""))
        nohub_err = str(r.get("weak_nohub_error_type", ""))

        c["old_oracle_top1_count"] += int(oracle_ok)
        c["old_weak_base_top1_count"] += int(base_ok)
        c["old_weak_nohub_top1_count"] += int(nohub_ok)
        c["old_nohub_rescued_count"] += int(rescued)
        c["old_oracle_correct_weak_base_wrong_count"] += int(oracle_gap)
        c["old_base_other_positive_count"] += int(base_err == "other_positive_confusion")
        c["old_nohub_other_positive_count"] += int(nohub_err == "other_positive_confusion")

        total["oracle_ok"] += int(oracle_ok)
        total["base_ok"] += int(base_ok)
        total["nohub_ok"] += int(nohub_ok)
        total["rescued"] += int(rescued)
        total["oracle_gap"] += int(oracle_gap)
        error_base[base_err or ""] += 1
        error_nohub[nohub_err or ""] += 1

    for c in by_class.values():
        n = c["old_row_count"]
        c["old_oracle_top1_rate"] = _safe_div(c["old_oracle_top1_count"], n)
        c["old_weak_base_top1_rate"] = _safe_div(c["old_weak_base_top1_count"], n)
        c["old_weak_nohub_top1_rate"] = _safe_div(c["old_weak_nohub_top1_count"], n)
        c["old_nohub_rescued_rate"] = _safe_div(c["old_nohub_rescued_count"], n)
        c["old_oracle_gap_rate"] = _safe_div(c["old_oracle_correct_weak_base_wrong_count"], n)
        c["old_base_other_positive_rate"] = _safe_div(c["old_base_other_positive_count"], n)
        c["old_nohub_other_positive_rate"] = _safe_div(c["old_nohub_other_positive_count"], n)

    summary = {
        "old_row_count": total["rows"],
        "old_oracle_top1_rate": _safe_div(total["oracle_ok"], total["rows"]),
        "old_weak_base_top1_rate": _safe_div(total["base_ok"], total["rows"]),
        "old_weak_nohub_top1_rate": _safe_div(total["nohub_ok"], total["rows"]),
        "old_nohub_rescued_rate": _safe_div(total["rescued"], total["rows"]),
        "old_oracle_gap_rate": _safe_div(total["oracle_gap"], total["rows"]),
        "old_weak_base_error_type_counts": dict(error_base),
        "old_weak_nohub_error_type_counts": dict(error_nohub),
    }
    return by_class, summary


def _aggregate_seed_and_manifest(
    seed_rows: List[Dict[str, str]],
    manifest_rows: List[Dict[str, str]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]:
    by_class: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "seed_hard_ce_seed_rows": 0,
        "seed_soft_ce_seed_rows": 0,
        "seed_prototype_seed_rows": 0,
        "seed_deferred_rows": 0,
        "manifest_hard_ce_rows": 0,
        "manifest_soft_ce_rows": 0,
        "manifest_prototype_calibration_rows": 0,
        "manifest_total_selected_rows": 0,
        "policy": "",
        "certificate_type": "",
        "clip_count": "",
        "instance_count": "",
    })
    row_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in seed_rows:
        rid = _norm_id(r.get("gt_raw_id") or r.get("raw_id"))
        if not rid:
            continue
        d = by_class[rid]
        d["raw_id"] = rid
        d["class_name"] = r.get("gt_class_name") or r.get("class_name") or d.get("class_name", "")
        st = str(r.get("seed_type", ""))
        if st == "hard_ce_seed":
            d["seed_hard_ce_seed_rows"] += 1
        elif st == "soft_ce_seed":
            d["seed_soft_ce_seed_rows"] += 1
        elif st == "prototype_seed":
            d["seed_prototype_seed_rows"] += 1
        elif st == "deferred":
            d["seed_deferred_rows"] += 1
        d["policy"] = d["policy"] or r.get("policy", "")
        d["certificate_type"] = d["certificate_type"] or r.get("certificate_type", "")
        d["clip_count"] = d["clip_count"] or r.get("clip_count", "")
        d["instance_count"] = d["instance_count"] or r.get("instance_count", "")
        row_meta[_row_key(r)].update({
            "seed_type": st,
            "seed_policy": r.get("policy", ""),
            "seed_certificate_type": r.get("certificate_type", ""),
        })

    for r in manifest_rows:
        rid = _norm_id(r.get("gt_raw_id") or r.get("raw_id"))
        if not rid:
            continue
        d = by_class[rid]
        d["raw_id"] = rid
        d["class_name"] = r.get("gt_class_name") or r.get("class_name") or d.get("class_name", "")
        lf = str(r.get("loss_family", ""))
        if lf == "hard_ce":
            d["manifest_hard_ce_rows"] += 1
        elif lf == "soft_ce":
            d["manifest_soft_ce_rows"] += 1
        elif lf == "prototype_calibration":
            d["manifest_prototype_calibration_rows"] += 1
        d["manifest_total_selected_rows"] += 1
        d["policy"] = d["policy"] or r.get("policy", "")
        d["certificate_type"] = d["certificate_type"] or r.get("certificate_type", "")
        d["clip_count"] = d["clip_count"] or r.get("clip_count", "")
        d["instance_count"] = d["instance_count"] or r.get("instance_count", "")
        key = _row_key(r)
        row_meta[key].update({
            "manifest_selected": True,
            "manifest_loss_family": lf,
            "manifest_sample_weight": r.get("sample_weight", ""),
            "manifest_use": r.get("manifest_use", ""),
        })

    group_counts = Counter()
    for meta in row_meta.values():
        if meta.get("manifest_loss_family"):
            group_counts[f"selected_{meta['manifest_loss_family']}"] += 1
        else:
            st = meta.get("seed_type") or "no_seed"
            group_counts[f"seed_{st}_not_selected"] += 1

    return dict(by_class), row_meta, {"row_membership_counts": dict(group_counts)}


def _load_mode_class_delta(run_root: Path, dataset_name: str, mode: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    mode_root = _mode_output_root(run_root, dataset_name, mode)
    final_json = mode_root / "final_summary.json"
    delta_csv = mode_root / "analysis" / "eval_before_after_by_class_delta.csv"
    out: Dict[str, Dict[str, Any]] = {}

    final = _read_json(final_json)
    rows = _read_csv(delta_csv)

    for r in rows:
        rid = _norm_id(r.get("raw_id"))
        if not rid:
            continue
        out[rid] = {
            f"{mode}_before_top1": _float(r.get("before_top1")),
            f"{mode}_after_top1": _float(r.get("after_top1")),
            f"{mode}_delta_top1": _float(r.get("delta_top1")),
            f"{mode}_before_mean_norm_rank": _float(r.get("before_mean_norm_rank")),
            f"{mode}_after_mean_norm_rank": _float(r.get("after_mean_norm_rank")),
            f"{mode}_delta_mean_norm_rank": _float(r.get("delta_mean_norm_rank")),
            f"{mode}_eval_gt_count": _float(r.get("gt_count")),
        }
    return out, final


def _class_selection_group(row: Dict[str, Any]) -> str:
    if int(row.get("manifest_hard_ce_rows", 0) or 0) > 0:
        return "class_selected_hard_ce"
    if int(row.get("manifest_soft_ce_rows", 0) or 0) > 0:
        return "class_selected_soft_ce_only"
    if int(row.get("manifest_prototype_calibration_rows", 0) or 0) > 0:
        return "class_selected_prototype_only"
    if int(row.get("seed_deferred_rows", 0) or 0) > 0:
        return "class_deferred_seed_only_or_mixed_unselected"
    return "class_not_in_seed_pool"


def _aggregate_row_groups(row_gap_rows: List[Dict[str, str]], row_meta: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg: Dict[str, Counter] = defaultdict(Counter)
    for r in row_gap_rows:
        key = _row_key(r)
        meta = row_meta.get(key, {})
        if meta.get("manifest_loss_family"):
            group = f"selected_{meta.get('manifest_loss_family')}"
        elif meta.get("seed_type"):
            group = f"unselected_{meta.get('seed_type')}"
        else:
            group = "not_in_seed_pool"
        a = agg[group]
        a["rows"] += 1
        a["oracle_top1"] += int(_truth(r.get("oracle_top1_is_gt")))
        a["weak_base_top1"] += int(_truth(r.get("weak_base_top1_is_gt")))
        a["weak_nohub_top1"] += int(_truth(r.get("weak_nohub_top1_is_gt")))
        a["nohub_rescued"] += int(_truth(r.get("nohub_rescued_baseline_wrong")))
        a["oracle_gap"] += int(_truth(r.get("oracle_correct_weak_base_wrong")))
        a["base_other_positive"] += int(str(r.get("weak_base_error_type", "")) == "other_positive_confusion")
        a["nohub_other_positive"] += int(str(r.get("weak_nohub_error_type", "")) == "other_positive_confusion")

    rows = []
    for group, c in sorted(agg.items()):
        n = c["rows"]
        rows.append({
            "row_group": group,
            "rows": n,
            "old_oracle_top1_rate": _safe_div(c["oracle_top1"], n),
            "old_weak_base_top1_rate": _safe_div(c["weak_base_top1"], n),
            "old_weak_nohub_top1_rate": _safe_div(c["weak_nohub_top1"], n),
            "old_nohub_rescued_rate": _safe_div(c["nohub_rescued"], n),
            "old_oracle_gap_rate": _safe_div(c["oracle_gap"], n),
            "old_base_other_positive_rate": _safe_div(c["base_other_positive"], n),
            "old_nohub_other_positive_rate": _safe_div(c["nohub_other_positive"], n),
        })
    return rows


def _aggregate_class_groups(class_rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in class_rows:
        buckets[str(r.get("class_selection_group", ""))].append(r)

    out = []
    for g, rows in sorted(buckets.items()):
        out.append({
            "class_group": g,
            "class_count": len(rows),
            "old_row_count_sum": sum(int(_float(r.get("old_row_count"), 0)) for r in rows),
            "old_oracle_top1_rate_macro": _mean(_float(r.get("old_oracle_top1_rate"), 0) for r in rows),
            "old_weak_base_top1_rate_macro": _mean(_float(r.get("old_weak_base_top1_rate"), 0) for r in rows),
            "old_weak_nohub_top1_rate_macro": _mean(_float(r.get("old_weak_nohub_top1_rate"), 0) for r in rows),
            f"{mode}_after_top1_macro": _mean(_float(r.get(f"{mode}_after_top1"), 0) for r in rows),
            f"{mode}_delta_top1_macro": _mean(_float(r.get(f"{mode}_delta_top1"), 0) for r in rows),
            f"{mode}_after_mean_norm_rank_macro": _mean(_float(r.get(f"{mode}_after_mean_norm_rank"), 0) for r in rows),
            f"{mode}_delta_mean_norm_rank_macro": _mean(_float(r.get(f"{mode}_delta_mean_norm_rank"), 0) for r in rows),
            f"{mode}_improved_top1_classes": sum(1 for r in rows if _float(r.get(f"{mode}_delta_top1"), 0) > 0),
            f"{mode}_degraded_top1_classes": sum(1 for r in rows if _float(r.get(f"{mode}_delta_top1"), 0) < 0),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--manifest_csv", default="")
    ap.add_argument("--seed_pool_csv", default="")
    ap.add_argument("--row_gap_csv", default="")
    ap.add_argument("--modes", default="eval_only,hard_ce,hard_soft_proto")
    ap.add_argument("--primary_mode", default="hard_soft_proto")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    dataset_name = args.dataset_name
    manifest_csv = Path(args.manifest_csv) if args.manifest_csv else _default_manifest(run_root, dataset_name)
    seed_pool_csv = Path(args.seed_pool_csv) if args.seed_pool_csv else _default_seed_pool(run_root, dataset_name)
    row_gap_csv = Path(args.row_gap_csv) if args.row_gap_csv else _default_row_gap()
    out_dir = Path(args.out_dir) if args.out_dir else run_root / "analysis" / "residual_gated_eval_alignment" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    primary_mode = args.primary_mode

    row_gap_rows = _read_csv(row_gap_csv)
    manifest_rows = _read_csv(manifest_csv)
    seed_rows = _read_csv(seed_pool_csv) if seed_pool_csv.exists() else []

    old_by_class, old_summary = _aggregate_old_gap(row_gap_rows)
    seed_by_class, row_meta, seed_summary = _aggregate_seed_and_manifest(seed_rows, manifest_rows)

    mode_by_class: Dict[str, Dict[str, Dict[str, Any]]] = {}
    mode_summaries: Dict[str, Dict[str, Any]] = {}
    missing_modes = []
    for mode in modes:
        try:
            mode_class, final = _load_mode_class_delta(run_root, dataset_name, mode)
            mode_by_class[mode] = mode_class
            mode_summaries[mode] = {
                "status": final.get("status"),
                "mode": final.get("mode"),
                "train_summary": final.get("train_summary", {}),
                "eval_before": final.get("eval_before", {}),
                "eval_after": final.get("eval_after", {}),
                "eval_delta": final.get("eval_delta", {}),
                "output_root": final.get("output_root"),
            }
        except FileNotFoundError as e:
            missing_modes.append({"mode": mode, "missing": str(e)})

    all_raw_ids = set(old_by_class) | set(seed_by_class)
    for mode, d in mode_by_class.items():
        all_raw_ids |= set(d)

    class_rows: List[Dict[str, Any]] = []
    for rid in sorted(all_raw_ids, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        row: Dict[str, Any] = {"raw_id": rid}
        row.update(old_by_class.get(rid, {}))
        row.update(seed_by_class.get(rid, {}))
        if "class_name" not in row or not row.get("class_name"):
            for mode in modes:
                if rid in mode_by_class.get(mode, {}):
                    break
        for mode in modes:
            row.update(mode_by_class.get(mode, {}).get(rid, {}))
        row["class_selection_group"] = _class_selection_group(row)
        row["selected_any_rows"] = int(row.get("manifest_total_selected_rows", 0) or 0)
        row["selected_hard_or_soft_rows"] = int(row.get("manifest_hard_ce_rows", 0) or 0) + int(row.get("manifest_soft_ce_rows", 0) or 0)
        # Conservative interpretation labels.
        old_nohub = _float(row.get("old_weak_nohub_top1_rate"), 0)
        delta = _float(row.get(f"{primary_mode}_delta_top1"), 0)
        after = _float(row.get(f"{primary_mode}_after_top1"), 0)
        if row["selected_hard_or_soft_rows"] > 0 and delta > 0:
            row["alignment_label"] = "selected_class_improved"
        elif row["selected_hard_or_soft_rows"] == 0 and delta > 0:
            row["alignment_label"] = "unselected_class_improved_possible_spillover"
        elif old_nohub > 0 and after == 0:
            row["alignment_label"] = "old_nohub_capability_not_recovered_by_pilot"
        elif delta < 0:
            row["alignment_label"] = "degraded"
        else:
            row["alignment_label"] = "unchanged"
        class_rows.append(row)

    row_group_rows = _aggregate_row_groups(row_gap_rows, row_meta)
    class_group_rows = _aggregate_class_groups(class_rows, primary_mode) if primary_mode in mode_by_class else []

    _write_csv(out_dir / "eval_alignment_by_class.csv", class_rows)
    _write_csv(out_dir / "eval_alignment_row_group_old_metrics.csv", row_group_rows)
    _write_csv(out_dir / "eval_alignment_class_group_pilot_delta.csv", class_group_rows)

    primary_final = mode_summaries.get(primary_mode, {})
    hard_final = mode_summaries.get("hard_ce", {})
    hsp_final = mode_summaries.get("hard_soft_proto", {})

    def _delta_metric(final: Dict[str, Any], key: str) -> float:
        return _float(final.get("eval_delta", {}).get(key), 0.0)

    selected_improved = sum(1 for r in class_rows if r.get("alignment_label") == "selected_class_improved")
    unselected_improved = sum(1 for r in class_rows if r.get("alignment_label") == "unselected_class_improved_possible_spillover")
    degraded = sum(1 for r in class_rows if r.get("alignment_label") == "degraded")

    summary = {
        "status": "PASS" if not missing_modes else "PASS_WITH_MISSING_MODES",
        "run_root": str(run_root),
        "dataset_name": dataset_name,
        "manifest_csv": str(manifest_csv),
        "seed_pool_csv": str(seed_pool_csv),
        "row_gap_csv": str(row_gap_csv),
        "out_dir": str(out_dir),
        "modes_requested": modes,
        "primary_mode": primary_mode,
        "missing_modes": missing_modes,
        "old_row_gap_summary": old_summary,
        "seed_manifest_summary": seed_summary,
        "mode_summaries": mode_summaries,
        "primary_mode_alignment": {
            "selected_class_improved": selected_improved,
            "unselected_class_improved_possible_spillover": unselected_improved,
            "degraded_classes": degraded,
            "class_count": len(class_rows),
        },
        "comparative_gates": [
            {
                "name": "hard_soft_proto_beats_hard_ce_micro_top1_delta",
                "status": "PASS" if _delta_metric(hsp_final, "micro_top1_delta") > _delta_metric(hard_final, "micro_top1_delta") else "WARN",
                "hard_ce": _delta_metric(hard_final, "micro_top1_delta"),
                "hard_soft_proto": _delta_metric(hsp_final, "micro_top1_delta"),
                "hard": False,
            },
            {
                "name": "hard_soft_proto_beats_hard_ce_macro_rank1_delta",
                "status": "PASS" if _delta_metric(hsp_final, "macro_mean_rank1_delta") > _delta_metric(hard_final, "macro_mean_rank1_delta") else "WARN",
                "hard_ce": _delta_metric(hard_final, "macro_mean_rank1_delta"),
                "hard_soft_proto": _delta_metric(hsp_final, "macro_mean_rank1_delta"),
                "hard": False,
            },
            {
                "name": "primary_mean_norm_rank_improves",
                "status": "PASS" if _delta_metric(primary_final, "mean_normalized_gt_rank_delta") < 0 else "FAIL",
                "delta": _delta_metric(primary_final, "mean_normalized_gt_rank_delta"),
                "hard": True,
            },
        ],
        "limitations": [
            "This reducer aligns pilot after metrics at class level because the current pilot did not export row-level after predictions.",
            "It can state whether selected classes improved and whether old weak/nohub capability existed, but cannot exactly count old-nohub-wrong rows rescued after training until row-level pilot predictions are exported.",
            "Pilot eval_before is a fresh residual-adapter scorer state, not the old weak_base/nohub scorer; old scorer metrics are reported side-by-side rather than treated as the same metric."
        ],
        "outputs": {
            "eval_alignment_by_class": str(out_dir / "eval_alignment_by_class.csv"),
            "eval_alignment_row_group_old_metrics": str(out_dir / "eval_alignment_row_group_old_metrics.csv"),
            "eval_alignment_class_group_pilot_delta": str(out_dir / "eval_alignment_class_group_pilot_delta.csv"),
            "eval_alignment_summary": str(out_dir / "eval_alignment_summary.json"),
            "takeover": str(out_dir / "RESIDUAL_GATED_EVAL_ALIGNMENT_TAKEOVER.md"),
        }
    }

    (out_dir / "eval_alignment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Residual-Gated Eval Alignment TAKEOVER",
        "",
        f"- status: {summary['status']}",
        f"- dataset: {dataset_name}",
        f"- primary_mode: {primary_mode}",
        f"- old weak_base top1: {old_summary.get('old_weak_base_top1_rate', 0):.6f}",
        f"- old weak_nohub top1: {old_summary.get('old_weak_nohub_top1_rate', 0):.6f}",
        f"- old oracle top1: {old_summary.get('old_oracle_top1_rate', 0):.6f}",
    ]
    if primary_final:
        lines += [
            f"- primary after micro_top1: {_float(primary_final.get('eval_after', {}).get('micro_top1'), 0):.6f}",
            f"- primary delta micro_top1: {_delta_metric(primary_final, 'micro_top1_delta'):.6f}",
            f"- primary delta macro_rank1: {_delta_metric(primary_final, 'macro_mean_rank1_delta'):.6f}",
            f"- primary delta mean_norm_rank: {_delta_metric(primary_final, 'mean_normalized_gt_rank_delta'):.6f}",
        ]
    lines += [
        "",
        "## Alignment labels",
        f"- selected_class_improved: {selected_improved}",
        f"- unselected_class_improved_possible_spillover: {unselected_improved}",
        f"- degraded_classes: {degraded}",
        "",
        "## Comparative gates",
    ]
    for g in summary["comparative_gates"]:
        got = ", ".join(f"{k}={v}" for k, v in g.items() if k not in {"name", "status", "hard"})
        lines.append(f"- {g['name']}: {g['status']} {got}")
    lines += [
        "",
        "## Important limitation",
        "- Current pilot after predictions are only available in by-class delta form. Exact row-level rescue counts require a future row-level after-prediction export.",
        "",
        "## Outputs",
        "- eval_alignment_summary.json",
        "- eval_alignment_by_class.csv",
        "- eval_alignment_row_group_old_metrics.csv",
        "- eval_alignment_class_group_pilot_delta.csv",
    ]
    (out_dir / "RESIDUAL_GATED_EVAL_ALIGNMENT_TAKEOVER.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
