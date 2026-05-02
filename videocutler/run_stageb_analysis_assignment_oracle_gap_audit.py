#!/usr/bin/env python3
"""Assignment oracle gap audit for GT-clean weak full-Y overfit.

Purpose
-------
This is a read-only row-level diagnosis. It compares the assignments produced by
three already-trained GT-clean base checkpoints on the same GT-upper-bound rows:

  A. oracle supervised GT-class checkpoint
  B. weak full-Y baseline checkpoint
  C. weak full-Y NoHub checkpoint

The audit does not train. It uses instance GT labels only as clean denominator /
evaluation truth. Its goal is to explain *where* the weak full-Y objective loses
the oracle capacity that was proven by the GT-clean oracle overfit test.

Main questions
--------------
* When weak top1 is wrong, is it another positive in the same clip-Y set, or an
  outside-positive base absorber?
* Which GT->predicted-class confusions dominate the oracle gap?
* Which rows are rescued or broken by NoHub relative to weak baseline?
* Does weak failure concentrate in high co-occurrence / large clip-Y settings?

Boundary
--------
* Read-only analysis; no training, no checkpoint modification.
* GT labels are used only for row-level diagnosis/evaluation.
* No VideoCutLER/mainline trajectories, no mAP, no Y-prime/extra/unknown.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _bootstrap_repo_root_for_direct_cli() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


_BOOT_REPO_ROOT = _bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab  # noqa: E402
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig  # noqa: E402
import videocutler.run_stageb_train_gt_clean_base_oracle_overfit as oracle  # noqa: E402


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _mean(vals: Sequence[float]) -> float:
    arr = [float(v) for v in vals if not math.isnan(float(v))]
    return float(np.mean(np.asarray(arr, dtype=np.float64))) if arr else 0.0


def _pct(num: int, den: int) -> float:
    return float(num / max(int(den), 1))


def _row_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('clip_id')}|{row.get('trajectory_id')}|{row.get('target_raw_id')}"


def _checkpoint_path(root: Path) -> Path:
    p = root / "train" / "prealign" / "checkpoints" / "prealign_last.pth"
    if not p.is_file():
        raise FileNotFoundError(f"missing checkpoint: {p}")
    return p


def _load_projector_from_root(root: Path, device: torch.device) -> Tuple[Projector, torch.nn.Parameter, Dict[str, Any]]:
    ckpt_path = _checkpoint_path(root)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg_obj = ckpt.get("text_projector_config", {}) if isinstance(ckpt, Mapping) else {}
    try:
        cfg = ProjectorConfig(**dict(cfg_obj))
    except Exception:
        cfg = ProjectorConfig()
    projector = Projector(cfg).to(device)
    state = ckpt.get("text_projector_state_dict") if isinstance(ckpt, Mapping) else None
    if state is None:
        raise KeyError(f"checkpoint has no text_projector_state_dict: {ckpt_path}")
    projector.load_state_dict(state, strict=True)
    projector.eval()
    theta_raw = ckpt.get("theta_T", 0.0) if isinstance(ckpt, Mapping) else 0.0
    theta_t = torch.nn.Parameter(torch.tensor(float(theta_raw), device=device, dtype=torch.float32), requires_grad=False)
    meta = {
        "checkpoint_path": str(ckpt_path),
        "stage_id": ckpt.get("stage_id", "") if isinstance(ckpt, Mapping) else "",
        "protocol": ckpt.get("protocol", "") if isinstance(ckpt, Mapping) else "",
        "pipeline": ckpt.get("pipeline", "") if isinstance(ckpt, Mapping) else "",
        "epoch": ckpt.get("epoch", "") if isinstance(ckpt, Mapping) else "",
        "global_step": ckpt.get("global_step", "") if isinstance(ckpt, Mapping) else "",
        "temperature": float(oracle._compute_t_dis(theta_t).detach().cpu().item()),
    }
    return projector, theta_t, meta


def _score_model_rows(
    *,
    name: str,
    root: Path,
    examples: Sequence[Mapping[str, Any]],
    text_vocab_tensor: torch.Tensor,
    text_vocab_ids: Sequence[int],
    raw_to_idx: Mapping[int, int],
    clip_y_base: Mapping[int, Set[int]],
    base_ids: Set[int],
    candidate_scope: str,
    device: torch.device,
    class_name_map: Optional[Mapping[int, str]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    projector, theta_t, meta = _load_projector_from_root(root, device)
    summary, _per_class, rows = oracle._score_rows(
        examples=examples,
        projector=projector,
        theta_t=theta_t,
        text_vocab_tensor=text_vocab_tensor,
        text_vocab_ids=text_vocab_ids,
        raw_to_idx=raw_to_idx,
        clip_y_base=clip_y_base,
        base_ids=base_ids,
        candidate_scope=str(candidate_scope),
        device=device,
        class_name_map=class_name_map,
        max_rows_out=max(int(len(examples)) + 10, 1),
    )
    for r in rows:
        r["run"] = str(name)
        r["candidate_scope"] = str(candidate_scope)
    meta.update({"run": str(name), "summary": summary})
    return summary, rows, meta


def _error_type(row: Mapping[str, Any], clip_y: Set[int]) -> str:
    gt = int(row.get("target_raw_id"))
    top = int(row.get("top1_raw_id"))
    if top == gt:
        return "correct"
    if top in clip_y:
        return "other_positive_confusion"
    return "outside_positive_base_absorption"


def _clip_y_bin(size: int) -> str:
    s = int(size)
    if s <= 1:
        return "01"
    if s == 2:
        return "02"
    if s <= 5:
        return "03_05"
    if s <= 10:
        return "06_10"
    return "gt10"


def _empty_run_counts() -> Dict[str, int]:
    return {
        "count": 0,
        "top1_correct": 0,
        "top5_correct": 0,
        "top10_correct": 0,
        "other_positive_confusion": 0,
        "outside_positive_base_absorption": 0,
        "top1_in_clip_y": 0,
    }


def _summarize_run(rows: Sequence[Mapping[str, Any]], clip_y_base: Mapping[int, Set[int]]) -> Dict[str, Any]:
    c = _empty_run_counts()
    ranks: List[float] = []
    raw_ranks: List[float] = []
    top_hist = Counter()
    for r in rows:
        clip_id = int(r.get("clip_id"))
        clip_y = set(int(x) for x in clip_y_base.get(clip_id, set()))
        et = _error_type(r, clip_y)
        c["count"] += 1
        c["top1_correct"] += int(et == "correct")
        c["top5_correct"] += int(bool(r.get("gt_top5_hit")))
        c["top10_correct"] += int(bool(r.get("gt_top10_hit")))
        c["other_positive_confusion"] += int(et == "other_positive_confusion")
        c["outside_positive_base_absorption"] += int(et == "outside_positive_base_absorption")
        c["top1_in_clip_y"] += int(int(r.get("top1_raw_id")) in clip_y)
        ranks.append(_as_float(r.get("normalized_gt_rank")))
        raw_ranks.append(_as_float(r.get("gt_rank")))
        top_hist[int(r.get("top1_raw_id"))] += 1
    n = c["count"]
    return {
        "row_count": int(n),
        "top1_hit_rate": _pct(c["top1_correct"], n),
        "top5_hit_rate": _pct(c["top5_correct"], n),
        "top10_hit_rate": _pct(c["top10_correct"], n),
        "mean_normalized_gt_rank": _mean(ranks),
        "mean_gt_rank": _mean(raw_ranks),
        "top1_in_clip_y_rate": _pct(c["top1_in_clip_y"], n),
        "other_positive_confusion_rate_all": _pct(c["other_positive_confusion"], n),
        "outside_positive_absorption_rate_all": _pct(c["outside_positive_base_absorption"], n),
        "other_positive_confusion_rate_among_wrong": _pct(c["other_positive_confusion"], max(n - c["top1_correct"], 1)),
        "outside_positive_absorption_rate_among_wrong": _pct(c["outside_positive_base_absorption"], max(n - c["top1_correct"], 1)),
        "top_pred_unique_count": int(len(top_hist)),
        "top_pred_max_count": int(top_hist.most_common(1)[0][1]) if top_hist else 0,
        "top_pred_max_share": float(top_hist.most_common(1)[0][1] / max(n, 1)) if top_hist else 0.0,
    }


def _build_gap_rows(
    *,
    scope: str,
    oracle_rows: Sequence[Mapping[str, Any]],
    weak_base_rows: Sequence[Mapping[str, Any]],
    weak_nohub_rows: Sequence[Mapping[str, Any]],
    clip_y_base: Mapping[int, Set[int]],
    class_name_map: Mapping[int, str],
) -> List[Dict[str, Any]]:
    oidx = {_row_key(r): r for r in oracle_rows}
    bidx = {_row_key(r): r for r in weak_base_rows}
    nidx = {_row_key(r): r for r in weak_nohub_rows}
    keys = sorted(set(oidx) & set(bidx) & set(nidx))
    out: List[Dict[str, Any]] = []
    for k in keys:
        o = oidx[k]
        b = bidx[k]
        n = nidx[k]
        clip_id = int(o.get("clip_id"))
        clip_y = set(int(x) for x in clip_y_base.get(clip_id, set()))
        gt = int(o.get("target_raw_id"))
        btop = int(b.get("top1_raw_id"))
        ntop = int(n.get("top1_raw_id"))
        otop = int(o.get("top1_raw_id"))
        b_rank = int(b.get("gt_rank"))
        n_rank = int(n.get("gt_rank"))
        o_rank = int(o.get("gt_rank"))
        brow = {
            "candidate_scope": str(scope),
            "clip_id": int(clip_id),
            "trajectory_id": str(o.get("trajectory_id", "")),
            "gt_raw_id": int(gt),
            "gt_class_name": str(class_name_map.get(int(gt), "")),
            "clip_y_size": int(len(clip_y)),
            "oracle_top1_raw_id": int(otop),
            "oracle_top1_class_name": str(class_name_map.get(int(otop), "")),
            "oracle_top1_is_gt": bool(otop == gt),
            "oracle_gt_rank": int(o_rank),
            "oracle_norm_rank": float(o.get("normalized_gt_rank")),
            "weak_base_top1_raw_id": int(btop),
            "weak_base_top1_class_name": str(class_name_map.get(int(btop), "")),
            "weak_base_top1_is_gt": bool(btop == gt),
            "weak_base_gt_rank": int(b_rank),
            "weak_base_norm_rank": float(b.get("normalized_gt_rank")),
            "weak_base_error_type": _error_type(b, clip_y),
            "weak_nohub_top1_raw_id": int(ntop),
            "weak_nohub_top1_class_name": str(class_name_map.get(int(ntop), "")),
            "weak_nohub_top1_is_gt": bool(ntop == gt),
            "weak_nohub_gt_rank": int(n_rank),
            "weak_nohub_norm_rank": float(n.get("normalized_gt_rank")),
            "weak_nohub_error_type": _error_type(n, clip_y),
            "nohub_delta_gt_rank": int(n_rank - b_rank),
            "nohub_delta_norm_rank": float(n.get("normalized_gt_rank")) - float(b.get("normalized_gt_rank")),
            "nohub_rescued_baseline_wrong": bool((btop != gt) and (ntop == gt)),
            "nohub_broke_baseline_correct": bool((btop == gt) and (ntop != gt)),
            "oracle_correct_weak_base_wrong": bool((otop == gt) and (btop != gt)),
            "oracle_correct_weak_nohub_wrong": bool((otop == gt) and (ntop != gt)),
            "weak_base_top1_in_clip_y": bool(btop in clip_y),
            "weak_nohub_top1_in_clip_y": bool(ntop in clip_y),
        }
        out.append(brow)
    return out


def _summary_from_gap(gap_rows: Sequence[Mapping[str, Any]], run_prefix: str) -> Dict[str, Any]:
    n = len(gap_rows)
    top1 = sum(1 for r in gap_rows if bool(r.get(f"{run_prefix}_top1_is_gt")))
    oracle_top1 = sum(1 for r in gap_rows if bool(r.get("oracle_top1_is_gt")))
    oracle_correct_weak_wrong = sum(1 for r in gap_rows if bool(r.get(f"oracle_correct_{run_prefix}_wrong")))
    other_pos = sum(1 for r in gap_rows if str(r.get(f"{run_prefix}_error_type")) == "other_positive_confusion")
    outside = sum(1 for r in gap_rows if str(r.get(f"{run_prefix}_error_type")) == "outside_positive_base_absorption")
    top_in_clip = sum(1 for r in gap_rows if bool(r.get(f"{run_prefix}_top1_in_clip_y")))
    ranks = [_as_float(r.get(f"{run_prefix}_norm_rank")) for r in gap_rows]
    return {
        "run": str(run_prefix),
        "row_count": int(n),
        "top1_hit_rate": _pct(top1, n),
        "mean_normalized_gt_rank": _mean(ranks),
        "oracle_top1_hit_rate": _pct(oracle_top1, n),
        "oracle_correct_weak_wrong_rate": _pct(oracle_correct_weak_wrong, n),
        "oracle_correct_weak_wrong_count": int(oracle_correct_weak_wrong),
        "top1_in_clip_y_rate": _pct(top_in_clip, n),
        "other_positive_confusion_rate_all": _pct(other_pos, n),
        "outside_positive_absorption_rate_all": _pct(outside, n),
        "other_positive_confusion_rate_among_wrong": _pct(other_pos, max(n - top1, 1)),
        "outside_positive_absorption_rate_among_wrong": _pct(outside, max(n - top1, 1)),
    }


def _nohub_delta_summary(gap_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(gap_rows)
    rescued = sum(1 for r in gap_rows if bool(r.get("nohub_rescued_baseline_wrong")))
    broke = sum(1 for r in gap_rows if bool(r.get("nohub_broke_baseline_correct")))
    both_correct = sum(1 for r in gap_rows if bool(r.get("weak_base_top1_is_gt")) and bool(r.get("weak_nohub_top1_is_gt")))
    both_wrong = sum(1 for r in gap_rows if (not bool(r.get("weak_base_top1_is_gt"))) and (not bool(r.get("weak_nohub_top1_is_gt"))))
    improved_rank = sum(1 for r in gap_rows if _as_float(r.get("nohub_delta_norm_rank")) < 0.0)
    degraded_rank = sum(1 for r in gap_rows if _as_float(r.get("nohub_delta_norm_rank")) > 0.0)
    delta_rank = [_as_float(r.get("nohub_delta_norm_rank")) for r in gap_rows]
    return {
        "row_count": int(n),
        "nohub_rescued_count": int(rescued),
        "nohub_rescued_rate": _pct(rescued, n),
        "nohub_broke_count": int(broke),
        "nohub_broke_rate": _pct(broke, n),
        "both_correct_count": int(both_correct),
        "both_correct_rate": _pct(both_correct, n),
        "both_wrong_count": int(both_wrong),
        "both_wrong_rate": _pct(both_wrong, n),
        "rank_improved_count": int(improved_rank),
        "rank_improved_rate": _pct(improved_rank, n),
        "rank_degraded_count": int(degraded_rank),
        "rank_degraded_rate": _pct(degraded_rank, n),
        "mean_nohub_delta_norm_rank": _mean(delta_rank),
    }


def _per_class_gap(gap_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for r in gap_rows:
        by[int(r.get("gt_raw_id"))].append(r)
    rows: List[Dict[str, Any]] = []
    for rid, rs in sorted(by.items()):
        n = len(rs)
        rows.append({
            "raw_id": int(rid),
            "class_name": str(rs[0].get("gt_class_name", "")),
            "gt_count": int(n),
            "oracle_top1_rate": _pct(sum(bool(r.get("oracle_top1_is_gt")) for r in rs), n),
            "weak_base_top1_rate": _pct(sum(bool(r.get("weak_base_top1_is_gt")) for r in rs), n),
            "weak_nohub_top1_rate": _pct(sum(bool(r.get("weak_nohub_top1_is_gt")) for r in rs), n),
            "weak_base_mean_norm_rank": _mean([_as_float(r.get("weak_base_norm_rank")) for r in rs]),
            "weak_nohub_mean_norm_rank": _mean([_as_float(r.get("weak_nohub_norm_rank")) for r in rs]),
            "oracle_mean_norm_rank": _mean([_as_float(r.get("oracle_norm_rank")) for r in rs]),
            "weak_base_other_positive_confusion_rate": _pct(sum(str(r.get("weak_base_error_type")) == "other_positive_confusion" for r in rs), n),
            "weak_base_outside_absorption_rate": _pct(sum(str(r.get("weak_base_error_type")) == "outside_positive_base_absorption" for r in rs), n),
            "weak_nohub_other_positive_confusion_rate": _pct(sum(str(r.get("weak_nohub_error_type")) == "other_positive_confusion" for r in rs), n),
            "weak_nohub_outside_absorption_rate": _pct(sum(str(r.get("weak_nohub_error_type")) == "outside_positive_base_absorption" for r in rs), n),
            "nohub_rescued_rate": _pct(sum(bool(r.get("nohub_rescued_baseline_wrong")) for r in rs), n),
            "nohub_broke_rate": _pct(sum(bool(r.get("nohub_broke_baseline_correct")) for r in rs), n),
            "mean_clip_y_size": _mean([float(r.get("clip_y_size")) for r in rs]),
        })
    return rows


def _confusion_pairs(gap_rows: Sequence[Mapping[str, Any]], run_prefix: str, top_k: int) -> List[Dict[str, Any]]:
    pair_counter: Counter[Tuple[int, int, str]] = Counter()
    target_count: Counter[int] = Counter()
    pred_count: Counter[int] = Counter()
    name_by: Dict[int, str] = {}
    for r in gap_rows:
        gt = int(r.get("gt_raw_id"))
        pred = int(r.get(f"{run_prefix}_top1_raw_id"))
        target_count[gt] += 1
        pred_count[pred] += 1
        name_by[gt] = str(r.get("gt_class_name", ""))
        name_by[pred] = str(r.get(f"{run_prefix}_top1_class_name", ""))
        if pred == gt:
            continue
        et = str(r.get(f"{run_prefix}_error_type"))
        pair_counter[(gt, pred, et)] += 1
    rows: List[Dict[str, Any]] = []
    for (gt, pred, et), cnt in pair_counter.most_common(int(top_k)):
        rows.append({
            "run": str(run_prefix),
            "gt_raw_id": int(gt),
            "gt_class_name": str(name_by.get(gt, "")),
            "pred_raw_id": int(pred),
            "pred_class_name": str(name_by.get(pred, "")),
            "error_type": str(et),
            "count": int(cnt),
            "share_of_gt": _pct(cnt, target_count[gt]),
            "pred_total_top1_count": int(pred_count[pred]),
        })
    return rows


def _top_pred_frequency(gap_rows: Sequence[Mapping[str, Any]], run_prefix: str, top_k: int) -> List[Dict[str, Any]]:
    c = Counter()
    correct = Counter()
    name_by: Dict[int, str] = {}
    for r in gap_rows:
        pred = int(r.get(f"{run_prefix}_top1_raw_id"))
        gt = int(r.get("gt_raw_id"))
        c[pred] += 1
        correct[pred] += int(pred == gt)
        name_by[pred] = str(r.get(f"{run_prefix}_top1_class_name", ""))
    n = len(gap_rows)
    rows: List[Dict[str, Any]] = []
    for pred, cnt in c.most_common(int(top_k)):
        rows.append({
            "run": str(run_prefix),
            "pred_raw_id": int(pred),
            "pred_class_name": str(name_by.get(pred, "")),
            "top1_count": int(cnt),
            "top1_share": _pct(cnt, n),
            "correct_when_predicted_rate": _pct(correct[pred], cnt),
            "wrong_when_predicted_rate": _pct(cnt - correct[pred], cnt),
        })
    return rows


def _by_clip_y_size(gap_rows: Sequence[Mapping[str, Any]], run_prefix: str) -> List[Dict[str, Any]]:
    by: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in gap_rows:
        by[_clip_y_bin(int(r.get("clip_y_size", 0)))].append(r)
    rows: List[Dict[str, Any]] = []
    for b, rs in sorted(by.items()):
        n = len(rs)
        rows.append({
            "run": str(run_prefix),
            "clip_y_size_bin": b,
            "row_count": int(n),
            "top1_hit_rate": _pct(sum(bool(r.get(f"{run_prefix}_top1_is_gt")) for r in rs), n),
            "mean_normalized_gt_rank": _mean([_as_float(r.get(f"{run_prefix}_norm_rank")) for r in rs]),
            "other_positive_confusion_rate_all": _pct(sum(str(r.get(f"{run_prefix}_error_type")) == "other_positive_confusion" for r in rs), n),
            "outside_positive_absorption_rate_all": _pct(sum(str(r.get(f"{run_prefix}_error_type")) == "outside_positive_base_absorption" for r in rs), n),
            "top1_in_clip_y_rate": _pct(sum(bool(r.get(f"{run_prefix}_top1_in_clip_y")) for r in rs), n),
        })
    return rows


def _diagnose(summary_by_scope: Mapping[str, Any]) -> str:
    # Primary diagnosis from base-vocab scope, because it tests global base-class discrimination.
    base = summary_by_scope.get("base_vocab", {}) if isinstance(summary_by_scope.get("base_vocab"), Mapping) else {}
    weak = base.get("weak_nohub", {}) if isinstance(base.get("weak_nohub"), Mapping) else {}
    oracle_s = base.get("oracle", {}) if isinstance(base.get("oracle"), Mapping) else {}
    oracle_top1 = _as_float(oracle_s.get("top1_hit_rate"))
    weak_top1 = _as_float(weak.get("top1_hit_rate"))
    if oracle_top1 >= 0.70 and weak_top1 < 0.50:
        return "ORACLE_STRONG__WEAK_ASSIGNMENT_GAP_DOMINANT"
    if oracle_top1 >= 0.70 and weak_top1 >= 0.50:
        return "ORACLE_STRONG__WEAK_PARTIAL_OR_STRONG__INSPECT_GAP_MODES"
    if oracle_top1 < 0.70:
        return "ORACLE_WEAK__CAPACITY_BOTTLENECK"
    return "MIXED__INSPECT_ASSIGNMENT_GAP_TABLES"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only row-level assignment oracle gap audit.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--repo_root", default=str(_BOOT_REPO_ROOT))
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--gt_identity_binding_jsonl", required=True)
    p.add_argument("--oracle_root", required=True)
    p.add_argument("--weak_baseline_root", required=True)
    p.add_argument("--weak_nohub_root", required=True)
    p.add_argument("--candidate_scopes", default="base_vocab,clip_y_base")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--write_row_level", action="store_true", default=True)
    p.add_argument("--no_write_row_level", action="store_false", dest="write_row_level")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    # Materialize the exact clean GT upper-bound denominator used by the oracle/weak trainers.
    load_args = argparse.Namespace(
        output_root=str(out),
        repo_root=str(Path(args.repo_root).expanduser().resolve()),
        asset_root=str(Path(args.asset_root).expanduser().resolve()),
        dataset_name=str(args.dataset_name),
        annotation_json=str(args.annotation_json),
        split_json=str(args.split_json),
        gt_identity_binding_jsonl=str(args.gt_identity_binding_jsonl),
        smoke=False,
        smoke_max_trajectories=0,
        subset_fraction=None,
        seed=int(args.seed),
        require_target_in_clip_y_base=True,
    )
    pack = oracle._load_oracle_examples(load_args)
    examples = list(pack.examples)
    class_name_map = oracle._class_name_map_from_annotation(Path(args.annotation_json))
    text_vocab_ids, _text_records, text_vocab_matrix = load_text_vocab(out)
    raw_to_idx = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
    text_vocab_tensor = torch.from_numpy(np.asarray(text_vocab_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)

    scopes = [s.strip() for s in str(args.candidate_scopes).split(",") if s.strip()]
    if not scopes:
        scopes = ["base_vocab"]

    model_roots = {
        "oracle": Path(args.oracle_root).expanduser().resolve(),
        "weak_base": Path(args.weak_baseline_root).expanduser().resolve(),
        "weak_nohub": Path(args.weak_nohub_root).expanduser().resolve(),
    }
    checkpoint_meta: Dict[str, Any] = {}
    rows_by_scope: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    summaries_by_scope: Dict[str, Dict[str, Any]] = {}
    gap_rows_by_scope: Dict[str, List[Dict[str, Any]]] = {}

    for scope in scopes:
        rows_by_scope[scope] = {}
        summaries_by_scope[scope] = {}
        for model_name, root in model_roots.items():
            summary, rows, meta = _score_model_rows(
                name=model_name,
                root=root,
                examples=examples,
                text_vocab_tensor=text_vocab_tensor,
                text_vocab_ids=text_vocab_ids,
                raw_to_idx=raw_to_idx,
                clip_y_base=pack.clip_y_base,
                base_ids=pack.base_ids,
                candidate_scope=scope,
                device=device,
                class_name_map=class_name_map,
            )
            rows_by_scope[scope][model_name] = rows
            summaries_by_scope[scope][model_name] = _summarize_run(rows, pack.clip_y_base)
            summaries_by_scope[scope][model_name].update({
                "candidate_scope": scope,
                "source_root": str(root),
                "checkpoint_path": meta.get("checkpoint_path", ""),
                "checkpoint_protocol": meta.get("protocol", ""),
                "checkpoint_temperature": meta.get("temperature", ""),
            })
            checkpoint_meta[f"{scope}:{model_name}"] = meta
        gap_rows = _build_gap_rows(
            scope=scope,
            oracle_rows=rows_by_scope[scope]["oracle"],
            weak_base_rows=rows_by_scope[scope]["weak_base"],
            weak_nohub_rows=rows_by_scope[scope]["weak_nohub"],
            clip_y_base=pack.clip_y_base,
            class_name_map=class_name_map,
        )
        gap_rows_by_scope[scope] = gap_rows

        scope_dir = out / scope
        scope_dir.mkdir(parents=True, exist_ok=True)
        if bool(args.write_row_level):
            _write_csv(scope_dir / "row_level_assignment_gap.csv", gap_rows)
        _write_csv(scope_dir / "run_summary.csv", [
            {"run": k, **v} for k, v in summaries_by_scope[scope].items()
        ])
        _write_csv(scope_dir / "oracle_gap_error_type_summary.csv", [
            {"candidate_scope": scope, **_summary_from_gap(gap_rows, "weak_base")},
            {"candidate_scope": scope, **_summary_from_gap(gap_rows, "weak_nohub")},
        ])
        _write_json(scope_dir / "nohub_delta_summary.json", _nohub_delta_summary(gap_rows))
        _write_csv(scope_dir / "per_class_oracle_gap.csv", _per_class_gap(gap_rows))
        pairs = _confusion_pairs(gap_rows, "weak_base", int(args.top_k)) + _confusion_pairs(gap_rows, "weak_nohub", int(args.top_k))
        _write_csv(scope_dir / "top_confusion_pairs.csv", pairs)
        topfreq = _top_pred_frequency(gap_rows, "weak_base", int(args.top_k)) + _top_pred_frequency(gap_rows, "weak_nohub", int(args.top_k))
        _write_csv(scope_dir / "top_pred_frequency.csv", topfreq)
        bins = _by_clip_y_size(gap_rows, "weak_base") + _by_clip_y_size(gap_rows, "weak_nohub")
        _write_csv(scope_dir / "summary_by_clip_y_size_bin.csv", bins)
        # Compact rescued/broken samples for quick inspection.
        rescued = [r for r in gap_rows if bool(r.get("nohub_rescued_baseline_wrong"))]
        broken = [r for r in gap_rows if bool(r.get("nohub_broke_baseline_correct"))]
        rank_improved = sorted(gap_rows, key=lambda r: _as_float(r.get("nohub_delta_norm_rank")))[: int(args.top_k)]
        rank_degraded = sorted(gap_rows, key=lambda r: _as_float(r.get("nohub_delta_norm_rank")), reverse=True)[: int(args.top_k)]
        _write_csv(scope_dir / "nohub_rescued_rows_top.csv", rescued[: int(args.top_k)])
        _write_csv(scope_dir / "nohub_broken_rows_top.csv", broken[: int(args.top_k)])
        _write_csv(scope_dir / "nohub_rank_improved_rows_top.csv", rank_improved)
        _write_csv(scope_dir / "nohub_rank_degraded_rows_top.csv", rank_degraded)

    top_summary_rows: List[Dict[str, Any]] = []
    for scope in scopes:
        for model_name, s in summaries_by_scope[scope].items():
            top_summary_rows.append({"candidate_scope": scope, "run": model_name, **s})
        gap_rows = gap_rows_by_scope[scope]
        top_summary_rows.append({"candidate_scope": scope, "run": "gap_weak_base", **_summary_from_gap(gap_rows, "weak_base")})
        top_summary_rows.append({"candidate_scope": scope, "run": "gap_weak_nohub", **_summary_from_gap(gap_rows, "weak_nohub")})
        top_summary_rows.append({"candidate_scope": scope, "run": "nohub_delta", **_nohub_delta_summary(gap_rows)})
    _write_csv(out / "assignment_oracle_gap_summary.csv", top_summary_rows)

    summary_by_scope = {
        scope: {
            "run_summaries": summaries_by_scope[scope],
            "gap_weak_base": _summary_from_gap(gap_rows_by_scope[scope], "weak_base"),
            "gap_weak_nohub": _summary_from_gap(gap_rows_by_scope[scope], "weak_nohub"),
            "nohub_delta": _nohub_delta_summary(gap_rows_by_scope[scope]),
        }
        for scope in scopes
    }
    diagnosis = _diagnose(summary_by_scope)
    summary = {
        "status": "PASS",
        "diagnosis": diagnosis,
        "output_dir": str(out),
        "dataset_name": str(args.dataset_name),
        "candidate_scopes": scopes,
        "example_count": int(len(examples)),
        "boundary": "read-only; GT labels only for assignment diagnosis/evaluation; no training",
        "materialization": {
            "identity_binding_paths_used": list(pack.identity_binding_paths_used),
            "target_attach_counters": dict(pack.target_attach_counters),
        },
        "checkpoint_meta": checkpoint_meta,
        "summary_by_scope": summary_by_scope,
    }
    _write_json(out / "summary.json", summary)

    md = [
        "# Assignment Oracle Gap Audit",
        "",
        "Status: `PASS`",
        f"Diagnosis: `{diagnosis}`",
        "",
        "Purpose:",
        "- Compare oracle supervised, weak full-Y baseline, and weak full-Y NoHub on the same GT-clean rows.",
        "- Decompose the weak assignment gap into other-positive confusion, outside-positive base absorption, and NoHub rescue/break cases.",
        "",
        "Core outputs:",
        "- summary.json",
        "- assignment_oracle_gap_summary.csv",
        "- <scope>/row_level_assignment_gap.csv",
        "- <scope>/oracle_gap_error_type_summary.csv",
        "- <scope>/per_class_oracle_gap.csv",
        "- <scope>/top_confusion_pairs.csv",
        "- <scope>/top_pred_frequency.csv",
        "- <scope>/summary_by_clip_y_size_bin.csv",
        "- <scope>/nohub_delta_summary.json",
        "",
        "Boundary:",
        "- Read-only analysis.",
        "- No training, no checkpoint changes, no VC trajectory, no mAP.",
        "- GT target is used only for clean assignment diagnosis/evaluation.",
        "",
    ]
    (out / "ASSIGNMENT_ORACLE_GAP_AUDIT_TAKEOVER.md").write_text("\n".join(md), encoding="utf-8")
    print(str(out / "summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
