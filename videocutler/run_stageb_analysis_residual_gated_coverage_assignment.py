#!/usr/bin/env python3
"""A8.0 residual-gated coverage-constrained assignment audit.

Side-path, read-only analysis. It intentionally does not train, does not modify
WS-OVVIS control-plane files, and does not consume row-level GT information for
assignment generation.

Allowed assignment inputs:
  * GT carrier trajectory features z_i
  * clip-level full-Y base labels Y(v)
  * text prototype bank t_c
  * trajectory-text similarity matrix s(i,c)
  * optional text-side projector checkpoint as a score initializer

Forbidden for assignment generation:
  * gt_raw_id / gt_class_name as row labels
  * oracle_top1_is_gt / weak_nohub_top1_is_gt
  * weak_nohub_error_type
  * hand-written person/hub raw id prior
  * A4 GT-defined soft/preserve split

Row-level GT and NoHub correctness fields are only joined after assignments are
created, for audit-only coverage and rescue diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab  # noqa: E402
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig  # noqa: E402
from videocutler.run_stageb_train_gt_full_y_clean import (  # noqa: E402
    _bootstrap_asset_links,
    _class_name_map_from_annotation_json,
    _class_name_map_from_text_records,
    _compute_t_dis,
    _inverse_softplus,
    _load_materialized_gt_examples,
    _normalize_np,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (list(rows[0].keys()) if rows else ["empty"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _truth(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(str(x)))
        except Exception:
            return None


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(list(vals), dtype=np.float64)))


def _norm_id(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def _softmax_np(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64) / max(float(temperature), 1.0e-6)
    arr = arr - np.max(arr)
    e = np.exp(arr)
    denom = float(np.sum(e))
    if denom <= 0.0:
        return np.full_like(arr, fill_value=1.0 / max(arr.size, 1), dtype=np.float64)
    return e / denom


def _entropy_np(p: np.ndarray) -> float:
    arr = np.asarray(p, dtype=np.float64)
    arr = arr[arr > 0.0]
    if arr.size <= 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def _pick(row: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        if name in row and str(row.get(name, "")).strip():
            return str(row.get(name, "")).strip()
    return ""


NOHUB_TOP1_ID_FIELDS = [
    "weak_nohub_top1_raw_id",
    "weak_nohub_top1_raw_id_canonical",
    "weak_nohub_top1_class_raw_id",
    "weak_nohub_top1_category_id",
    "weak_nohub_top1_id",
]
NOHUB_TOP1_NAME_FIELDS = [
    "weak_nohub_top1_class_name",
    "weak_nohub_top1_name",
    "weak_nohub_top1_class",
]


def _default_row_gap_path(repo_root: Path, dataset_name: str) -> Path:
    return (
        repo_root
        / "codex" / "outputs" / "G8_inference_and_eval"
        / "gt_clean_weak_fully_overfit_capacity_20260502"
        / "analysis" / "assignment_oracle_gap_audit" / str(dataset_name) / "base_vocab" / "row_level_assignment_gap.csv"
    )


def _auto_find_checkpoint(repo_root: Path, dataset_name: str) -> str:
    """Best-effort score initializer discovery.

    This is optional and is recorded explicitly as an additional score prior.
    It must not be confused with using NoHub correctness as a training label.
    """
    candidates = [
        _default_row_gap_path(repo_root, dataset_name).parents[1] / "assignment_oracle_gap_summary.csv",
        repo_root
        / "codex" / "outputs" / "G8_inference_and_eval"
        / "gt_clean_weak_fully_overfit_capacity_20260502"
        / "analysis" / "assignment_oracle_gap_audit" / str(dataset_name) / "assignment_oracle_gap_summary.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            rows = _read_csv(path)
        except Exception:
            continue
        preferred = sorted(
            rows,
            key=lambda r: (
                0 if str(r.get("run", "")).lower() == "weak_nohub" else 1,
                0 if str(r.get("candidate_scope", "")).lower() == "base_vocab" else 1,
            ),
        )
        for row in preferred:
            ckpt = str(row.get("checkpoint_path", "")).strip()
            if ckpt and Path(ckpt).is_file():
                return ckpt
    return ""


def _load_checkpoint_if_requested(projector: Projector, theta_t: torch.nn.Parameter, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    if not checkpoint_path:
        return {"loaded": False, "reason": "no checkpoint requested"}
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        return {"loaded": False, "checkpoint_path": str(path), "reason": "checkpoint not found"}
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        return {"loaded": False, "checkpoint_path": str(path), "reason": "checkpoint payload is not a mapping"}
    loaded_projector = False
    for key in ("text_projector_state_dict", "projector_state_dict", "state_dict"):
        sd = payload.get(key)
        if isinstance(sd, Mapping):
            try:
                projector.load_state_dict(sd, strict=False)
                loaded_projector = True
                break
            except Exception:
                continue
    loaded_theta = False
    for key in ("theta_T", "theta_t", "theta"):
        if key in payload:
            try:
                theta_t.data = torch.tensor(float(payload[key]), device=device, dtype=torch.float32)
                loaded_theta = True
                break
            except Exception:
                pass
    return {
        "loaded": bool(loaded_projector or loaded_theta),
        "loaded_projector": bool(loaded_projector),
        "loaded_theta": bool(loaded_theta),
        "checkpoint_path": str(path),
        "keys": sorted(str(k) for k in payload.keys()),
    }


@dataclass
class PreparedData:
    examples: List[Dict[str, Any]]
    by_clip: Dict[int, List[Dict[str, Any]]]
    clip_y_base: Dict[int, Set[int]]
    base_ids: Set[int]
    raw_to_text_idx: Dict[int, int]
    text_ids: List[int]
    text_records: List[Mapping[str, Any]]
    text_matrix: np.ndarray
    class_names: Dict[int, str]
    materialization_summary: Dict[str, Any]


def _prepare_data(args: argparse.Namespace) -> PreparedData:
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    out_root_for_assets = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else Path(args.run_root).expanduser().resolve()
    _bootstrap_asset_links(repo_root, asset_root)
    _bootstrap_asset_links(out_root_for_assets, asset_root)
    examples, clip_y_base, base_ids, materialization_summary = _load_materialized_gt_examples(
        repo_root=repo_root,
        output_root=out_root_for_assets,
        asset_root=asset_root,
        dataset_name=str(args.dataset_name),
        annotation_json=Path(args.annotation_json).expanduser().resolve(),
        split_json=Path(args.split_json).expanduser().resolve(),
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
        subset_fraction=args.subset_fraction,
        seed=int(args.seed),
    )
    if not examples:
        raise RuntimeError("no materialized GT carrier examples were loaded")
    by_clip: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_clip[int(ex["clip_id"])].append(dict(ex))
    text_ids_raw, text_records, text_matrix = load_text_vocab(out_root_for_assets)
    text_ids = [int(x) for x in text_ids_raw]
    raw_to_text_idx = {int(raw_id): idx for idx, raw_id in enumerate(text_ids)}
    ann_names = _class_name_map_from_annotation_json(Path(args.annotation_json).expanduser().resolve())
    text_names = _class_name_map_from_text_records(text_records)
    class_names = dict(text_names)
    class_names.update({int(k): str(v) for k, v in ann_names.items()})
    materialization_summary = dict(materialization_summary)
    materialization_summary["clip_count_after_grouping"] = int(len(by_clip))
    materialization_summary["base_ids_count"] = int(len(base_ids))
    materialization_summary["text_bank_count"] = int(len(text_ids))
    return PreparedData(
        examples=list(examples),
        by_clip=dict(by_clip),
        clip_y_base={int(k): set(int(x) for x in v) for k, v in clip_y_base.items()},
        base_ids=set(int(x) for x in base_ids),
        raw_to_text_idx=raw_to_text_idx,
        text_ids=text_ids,
        text_records=list(text_records),
        text_matrix=np.asarray(text_matrix, dtype=np.float32),
        class_names=class_names,
        materialization_summary=materialization_summary,
    )


def _carrier_tensor(group: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    vecs = []
    for ex in group:
        vecs.append(torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))))
    z = torch.stack(vecs, dim=0).to(device=device, dtype=torch.float32)
    return F.normalize(z, p=2.0, dim=-1)


def _build_text_projection(
    *,
    text_matrix: np.ndarray,
    projector: Projector,
    device: torch.device,
) -> torch.Tensor:
    text_tensor = torch.tensor(np.asarray(text_matrix, dtype=np.float32), device=device, dtype=torch.float32)
    projector.eval()
    with torch.no_grad():
        return F.normalize(projector(text_tensor), p=2.0, dim=-1)


def _rank_info(scores_for_row: np.ndarray, assigned_col: int) -> Tuple[int, float, float, float, int, int, float]:
    scores = np.asarray(scores_for_row, dtype=np.float64)
    if scores.size <= 0:
        return 0, 0.0, 0.0, 0.0, -1, -1, 0.0
    order = np.argsort(-scores)
    top1 = int(order[0])
    top2 = int(order[1]) if scores.size > 1 else int(order[0])
    top1_score = float(scores[top1])
    top2_score = float(scores[top2]) if scores.size > 1 else float(scores[top1])
    assigned_score = float(scores[int(assigned_col)])
    rank = int(1 + np.sum(scores > assigned_score))
    # margin relative to the best competing class for this row.
    if scores.size > 1:
        mask = np.ones(scores.shape[0], dtype=bool)
        mask[int(assigned_col)] = False
        best_other = float(np.max(scores[mask]))
    else:
        best_other = assigned_score
    assigned_margin = float(assigned_score - best_other)
    probs = _softmax_np(scores, temperature=1.0)
    entropy = _entropy_np(probs)
    return rank, assigned_score, assigned_margin, entropy, top1, top2, float(top1_score - top2_score)


def _assign_clip(
    *,
    clip_id: int,
    group: Sequence[Mapping[str, Any]],
    y_ids: Sequence[int],
    score_matrix: np.ndarray,
    class_names: Mapping[int, str],
    min_primary_score: float,
    min_extra_score: float,
    min_extra_margin: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy coverage assignment for one clip.

    No row-level GT is used here. The only inputs are clip ID, trajectory rows,
    clip-level Y ids, and score matrix over Y.
    """
    n_traj = int(len(group))
    n_cls = int(len(y_ids))
    assignments: List[Dict[str, Any]] = []
    clip_class_rows: List[Dict[str, Any]] = []
    if n_traj <= 0 or n_cls <= 0:
        return assignments, clip_class_rows

    S = np.asarray(score_matrix, dtype=np.float64)
    assert S.shape == (n_traj, n_cls)

    # Scarcity summary per class: best possible score and margin over trajectories.
    class_stats: List[Tuple[int, float, float]] = []
    for ccol, _rid in enumerate(y_ids):
        col = S[:, ccol]
        best = float(np.max(col))
        if col.size > 1:
            order = np.argsort(-col)
            gap = float(col[order[0]] - col[order[1]])
        else:
            gap = float(best)
        class_stats.append((ccol, best, gap))

    if n_traj >= n_cls:
        # Harder classes first: if a class has only weak candidate support, reserve it early.
        ordered_classes = [x[0] for x in sorted(class_stats, key=lambda x: (x[1], x[2], x[0]))]
    else:
        # In infeasible clips, assign the classes with best available evidence and slack the rest.
        ordered_classes = [x[0] for x in sorted(class_stats, key=lambda x: (-x[1], -x[2], x[0]))]

    unused_traj: Set[int] = set(range(n_traj))
    assigned_by_class: Dict[int, Dict[str, Any]] = {}
    used_by_traj: Dict[int, int] = {}

    for ccol in ordered_classes:
        rid = int(y_ids[ccol])
        if not unused_traj:
            assigned_by_class[ccol] = {
                "is_slack": 1,
                "slack_reason": "trajectory_count_lt_class_count_or_all_trajectories_used",
            }
            continue
        avail = sorted(unused_traj, key=lambda i: float(S[i, ccol]), reverse=True)
        ti = int(avail[0])
        score = float(S[ti, ccol])
        if score < float(min_primary_score):
            assigned_by_class[ccol] = {
                "is_slack": 1,
                "slack_reason": "best_available_score_below_min_primary_score",
                "best_available_score": score,
                "best_available_trajectory_id": str(group[ti].get("trajectory_id", "")),
            }
            continue
        rank, assigned_score, assigned_margin, entropy, top1_col, top2_col, top1_top2_margin = _rank_info(S[ti, :], ccol)
        ex = group[ti]
        row = {
            "clip_id": str(clip_id),
            "trajectory_id": str(ex.get("trajectory_id", "")),
            "assigned_raw_id": str(rid),
            "assigned_class_name": str(class_names.get(rid, rid)),
            "assignment_role": "primary_coverage",
            "assignment_score": assigned_score,
            "assignment_margin_vs_best_other": assigned_margin,
            "assignment_rank_in_row": rank,
            "assignment_entropy": entropy,
            "row_top1_raw_id_in_full_y": str(int(y_ids[top1_col])) if top1_col >= 0 else "",
            "row_top1_class_name_in_full_y": str(class_names.get(int(y_ids[top1_col]), int(y_ids[top1_col]))) if top1_col >= 0 else "",
            "row_top2_raw_id_in_full_y": str(int(y_ids[top2_col])) if top2_col >= 0 else "",
            "row_top2_class_name_in_full_y": str(class_names.get(int(y_ids[top2_col]), int(y_ids[top2_col]))) if top2_col >= 0 else "",
            "row_top1_top2_margin": top1_top2_margin,
            "coverage_rank": 1,
            "is_primary_coverage": 1,
            "is_slack": 0,
            "slack_reason": "",
            "clip_y_size": n_cls,
            "trajectory_count": n_traj,
        }
        assignments.append(row)
        assigned_by_class[ccol] = row
        used_by_traj[ti] = ccol
        unused_traj.remove(ti)

    # Optional routing for leftover trajectories: weak-visible, no GT.
    for ti in sorted(unused_traj):
        row_scores = S[ti, :]
        top_col = int(np.argmax(row_scores))
        rank, assigned_score, assigned_margin, entropy, top1_col, top2_col, top1_top2_margin = _rank_info(row_scores, top_col)
        rid = int(y_ids[top_col])
        role = "extra_support" if assigned_score >= float(min_extra_score) and top1_top2_margin >= float(min_extra_margin) else "deferred_unused"
        ex = group[ti]
        assignments.append({
            "clip_id": str(clip_id),
            "trajectory_id": str(ex.get("trajectory_id", "")),
            "assigned_raw_id": str(rid),
            "assigned_class_name": str(class_names.get(rid, rid)),
            "assignment_role": role,
            "assignment_score": assigned_score,
            "assignment_margin_vs_best_other": assigned_margin,
            "assignment_rank_in_row": rank,
            "assignment_entropy": entropy,
            "row_top1_raw_id_in_full_y": str(int(y_ids[top1_col])) if top1_col >= 0 else "",
            "row_top1_class_name_in_full_y": str(class_names.get(int(y_ids[top1_col]), int(y_ids[top1_col]))) if top1_col >= 0 else "",
            "row_top2_raw_id_in_full_y": str(int(y_ids[top2_col])) if top2_col >= 0 else "",
            "row_top2_class_name_in_full_y": str(class_names.get(int(y_ids[top2_col]), int(y_ids[top2_col]))) if top2_col >= 0 else "",
            "row_top1_top2_margin": top1_top2_margin,
            "coverage_rank": 0,
            "is_primary_coverage": 0,
            "is_slack": 0,
            "slack_reason": "",
            "clip_y_size": n_cls,
            "trajectory_count": n_traj,
        })

    # One row per clip-class, including slack classes.
    for ccol, rid0 in enumerate(y_ids):
        rid = int(rid0)
        a = assigned_by_class.get(ccol, {})
        is_slack = int(a.get("is_slack", 0) or 0)
        clip_class_rows.append({
            "clip_id": str(clip_id),
            "raw_id": str(rid),
            "class_name": str(class_names.get(rid, rid)),
            "has_coverage_support": int(not is_slack and bool(a.get("trajectory_id", ""))),
            "support_trajectory_id": str(a.get("trajectory_id", "")),
            "support_score": a.get("assignment_score", a.get("best_available_score", "")),
            "support_margin_vs_best_other": a.get("assignment_margin_vs_best_other", ""),
            "support_entropy": a.get("assignment_entropy", ""),
            "support_rank_in_row": a.get("assignment_rank_in_row", ""),
            "support_row_top1_raw_id_in_full_y": a.get("row_top1_raw_id_in_full_y", ""),
            "support_row_top1_class_name_in_full_y": a.get("row_top1_class_name_in_full_y", ""),
            "is_slack": int(is_slack),
            "slack_reason": str(a.get("slack_reason", "")),
            "clip_y_size": n_cls,
            "trajectory_count": n_traj,
        })
    return assignments, clip_class_rows


def _load_row_gap(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[Dict[str, str]], Dict[str, Any]]:
    if not path.is_file():
        return {}, [], {"available": False, "row_gap_csv": str(path), "reason": "not found"}
    rows = _read_csv(path)
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dup = 0
    for r in rows:
        key = (str(r.get("clip_id", "")), str(r.get("trajectory_id", "")))
        if not key[0] or not key[1]:
            continue
        if key in by_key:
            dup += 1
        by_key.setdefault(key, dict(r))
    return by_key, rows, {"available": True, "row_gap_csv": str(path), "rows": len(rows), "unique_keys": len(by_key), "duplicate_keys": dup}


def _nohub_clip_class_coverage_from_row_gap(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not rows:
        return {}
    header_keys = set(rows[0].keys()) if rows else set()
    use_id_top1 = any(k in header_keys for k in NOHUB_TOP1_ID_FIELDS)
    clip_gt_classes: Dict[str, Dict[str, str]] = defaultdict(dict)
    clip_pred_support: Dict[str, Set[str]] = defaultdict(set)
    clip_correct_support: Dict[str, Set[str]] = defaultdict(set)
    clip_gt_row_count: Counter = Counter()
    for r in rows:
        clip = str(r.get("clip_id", ""))
        gt_id = _norm_id(r.get("gt_raw_id"))
        gt_name = str(r.get("gt_class_name", ""))
        if not clip or not gt_id:
            continue
        clip_gt_classes[clip][gt_id] = gt_name
        clip_gt_row_count[(clip, gt_id)] += 1
        if use_id_top1:
            pred = _norm_id(_pick(r, NOHUB_TOP1_ID_FIELDS))
            pred_key = pred
        else:
            pred_key = _pick(r, NOHUB_TOP1_NAME_FIELDS)
        if pred_key:
            clip_pred_support[clip].add(str(pred_key))
        if _truth(r.get("weak_nohub_top1_is_gt")):
            clip_correct_support[clip].add(gt_id)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for clip, cls_map in clip_gt_classes.items():
        for gt_id, gt_name in cls_map.items():
            out[(clip, gt_id)] = {
                "nohub_gt_row_count_in_clip": int(clip_gt_row_count[(clip, gt_id)]),
                "nohub_predicted_support": int(gt_id in clip_pred_support[clip] or gt_name in clip_pred_support[clip]),
                "nohub_gt_correct_support": int(gt_id in clip_correct_support[clip]),
            }
    return out


def _audit_assignments(
    *,
    assignment_rows: List[Dict[str, Any]],
    clip_class_rows: List[Dict[str, Any]],
    row_gap_by_key: Mapping[Tuple[str, str], Mapping[str, Any]],
    row_gap_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    nohub_cc = _nohub_clip_class_coverage_from_row_gap(row_gap_rows)
    role_counters: Dict[str, Counter] = defaultdict(Counter)
    role_scores: Dict[str, List[float]] = defaultdict(list)
    audited_assignment_rows: List[Dict[str, Any]] = []

    for row in assignment_rows:
        out = dict(row)
        key = (str(row.get("clip_id", "")), str(row.get("trajectory_id", "")))
        rg = row_gap_by_key.get(key)
        assigned = _norm_id(row.get("assigned_raw_id"))
        role = str(row.get("assignment_role", ""))
        role_counters[role]["rows"] += 1
        role_scores[role].append(_as_float(row.get("assignment_score"), 0.0))
        if rg:
            gt = _norm_id(rg.get("gt_raw_id"))
            gt_name = str(rg.get("gt_class_name", ""))
            old_nohub_ok = _truth(rg.get("weak_nohub_top1_is_gt"))
            old_base_ok = _truth(rg.get("weak_base_top1_is_gt"))
            err = str(rg.get("weak_nohub_error_type", ""))
            match = bool(assigned and gt and assigned == gt)
            out.update({
                "audit_gt_raw_id": gt,
                "audit_gt_class_name": gt_name,
                "audit_assignment_matches_gt": int(match),
                "audit_weak_nohub_top1_is_gt": int(old_nohub_ok),
                "audit_weak_base_top1_is_gt": int(old_base_ok),
                "audit_weak_nohub_error_type": err,
                "audit_old_nohub_wrong": int(not old_nohub_ok),
                "audit_old_nohub_other_positive_confusion": int(err == "other_positive_confusion"),
            })
            role_counters[role]["joined_gt_rows"] += 1
            role_counters[role]["assignment_matches_gt"] += int(match)
            role_counters[role]["old_nohub_wrong"] += int(not old_nohub_ok)
            role_counters[role]["old_nohub_other_positive_confusion"] += int(err == "other_positive_confusion")
            role_counters[role]["old_nohub_correct"] += int(old_nohub_ok)
            if str(row.get("is_primary_coverage", "0")) in {"1", "true", "True"}:
                role_counters[role]["primary_rows"] += 1
                role_counters[role]["primary_assignment_matches_gt"] += int(match)
        else:
            out.update({
                "audit_gt_raw_id": "",
                "audit_gt_class_name": "",
                "audit_assignment_matches_gt": "",
                "audit_weak_nohub_top1_is_gt": "",
                "audit_weak_base_top1_is_gt": "",
                "audit_weak_nohub_error_type": "",
                "audit_old_nohub_wrong": "",
                "audit_old_nohub_other_positive_confusion": "",
            })
            role_counters[role]["missing_row_gap"] += 1
        audited_assignment_rows.append(out)

    audited_cc_rows: List[Dict[str, Any]] = []
    per_class: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    per_class_scores: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in clip_class_rows:
        out = dict(row)
        clip = str(row.get("clip_id", ""))
        rid = _norm_id(row.get("raw_id"))
        cname = str(row.get("class_name", ""))
        cc_key = (clip, rid)
        nohub = nohub_cc.get(cc_key, {})
        out["nohub_predicted_support"] = int(nohub.get("nohub_predicted_support", 0))
        out["nohub_gt_correct_support"] = int(nohub.get("nohub_gt_correct_support", 0))
        out["nohub_gt_row_count_in_clip"] = int(nohub.get("nohub_gt_row_count_in_clip", 0))
        support_tid = str(row.get("support_trajectory_id", ""))
        gt_correct = 0
        if support_tid:
            rg = row_gap_by_key.get((clip, support_tid))
            if rg and rid and _norm_id(rg.get("gt_raw_id")) == rid:
                gt_correct = 1
        out["coverage_gt_correct_support"] = int(gt_correct)
        out["coverage_gt_correct_uncovered"] = int(not gt_correct)
        out["coverage_assigned_uncovered"] = int(not int(row.get("has_coverage_support", 0)))
        audited_cc_rows.append(out)
        key = (rid, cname)
        per_class[key]["clip_class_count"] += 1
        per_class[key]["coverage_supported_clips"] += int(row.get("has_coverage_support", 0))
        per_class[key]["coverage_gt_correct_supported_clips"] += int(gt_correct)
        per_class[key]["nohub_predicted_supported_clips"] += int(out["nohub_predicted_support"])
        per_class[key]["nohub_gt_correct_supported_clips"] += int(out["nohub_gt_correct_support"])
        per_class[key]["slack_clips"] += int(row.get("is_slack", 0))
        sc = _as_float(row.get("support_score"), float("nan"))
        if math.isfinite(sc):
            per_class_scores[key].append(sc)

    by_class_rows: List[Dict[str, Any]] = []
    for (rid, cname), ctr in sorted(per_class.items(), key=lambda kv: (-kv[1]["clip_class_count"], kv[0][1])):
        n = int(ctr["clip_class_count"])
        by_class_rows.append({
            "raw_id": rid,
            "class_name": cname,
            "clip_class_count": n,
            "coverage_supported_clips": int(ctr["coverage_supported_clips"]),
            "coverage_support_rate": int(ctr["coverage_supported_clips"]) / n if n else 0.0,
            "coverage_gt_correct_supported_clips": int(ctr["coverage_gt_correct_supported_clips"]),
            "coverage_gt_correct_uncovered_clips": n - int(ctr["coverage_gt_correct_supported_clips"]),
            "coverage_gt_correct_uncovered_rate": (n - int(ctr["coverage_gt_correct_supported_clips"])) / n if n else 0.0,
            "nohub_predicted_supported_clips": int(ctr["nohub_predicted_supported_clips"]),
            "nohub_predicted_uncovered_clips": n - int(ctr["nohub_predicted_supported_clips"]),
            "nohub_predicted_uncovered_rate": (n - int(ctr["nohub_predicted_supported_clips"])) / n if n else 0.0,
            "nohub_gt_correct_supported_clips": int(ctr["nohub_gt_correct_supported_clips"]),
            "nohub_gt_correct_uncovered_clips": n - int(ctr["nohub_gt_correct_supported_clips"]),
            "nohub_gt_correct_uncovered_rate": (n - int(ctr["nohub_gt_correct_supported_clips"])) / n if n else 0.0,
            "slack_clips": int(ctr["slack_clips"]),
            "mean_support_score": _mean(per_class_scores.get((rid, cname), [])),
        })

    total_cc = len(audited_cc_rows)
    cov_supported = sum(int(r.get("has_coverage_support", 0)) for r in audited_cc_rows)
    cov_gt = sum(int(r.get("coverage_gt_correct_support", 0)) for r in audited_cc_rows)
    nohub_pred = sum(int(r.get("nohub_predicted_support", 0)) for r in audited_cc_rows)
    nohub_gt = sum(int(r.get("nohub_gt_correct_support", 0)) for r in audited_cc_rows)
    slack = sum(int(r.get("is_slack", 0)) for r in audited_cc_rows)

    role_summary: Dict[str, Any] = {}
    for role, ctr in sorted(role_counters.items()):
        n = int(ctr.get("rows", 0))
        joined = int(ctr.get("joined_gt_rows", 0))
        role_summary[role] = {
            "rows": n,
            "joined_gt_rows": joined,
            "assignment_matches_gt": int(ctr.get("assignment_matches_gt", 0)),
            "assignment_matches_gt_rate": int(ctr.get("assignment_matches_gt", 0)) / joined if joined else 0.0,
            "old_nohub_wrong_count": int(ctr.get("old_nohub_wrong", 0)),
            "old_nohub_wrong_rate": int(ctr.get("old_nohub_wrong", 0)) / joined if joined else 0.0,
            "old_nohub_other_positive_confusion_count": int(ctr.get("old_nohub_other_positive_confusion", 0)),
            "old_nohub_other_positive_confusion_rate": int(ctr.get("old_nohub_other_positive_confusion", 0)) / joined if joined else 0.0,
            "old_nohub_correct_count": int(ctr.get("old_nohub_correct", 0)),
            "old_nohub_correct_rate": int(ctr.get("old_nohub_correct", 0)) / joined if joined else 0.0,
            "mean_assignment_score": _mean(role_scores.get(role, [])),
            "primary_rows": int(ctr.get("primary_rows", 0)),
            "primary_assignment_matches_gt": int(ctr.get("primary_assignment_matches_gt", 0)),
            "primary_assignment_matches_gt_rate": int(ctr.get("primary_assignment_matches_gt", 0)) / int(ctr.get("primary_rows", 0)) if int(ctr.get("primary_rows", 0)) else 0.0,
        }

    summary = {
        "clip_class_count": total_cc,
        "coverage_supported_clip_classes": cov_supported,
        "coverage_uncovered_clip_classes": total_cc - cov_supported,
        "coverage_uncovered_rate": (total_cc - cov_supported) / total_cc if total_cc else 0.0,
        "coverage_gt_correct_supported_clip_classes": cov_gt,
        "coverage_gt_correct_uncovered_clip_classes": total_cc - cov_gt,
        "coverage_gt_correct_uncovered_rate": (total_cc - cov_gt) / total_cc if total_cc else 0.0,
        "coverage_slack_clip_classes": slack,
        "coverage_slack_rate": slack / total_cc if total_cc else 0.0,
        "nohub_predicted_supported_clip_classes": nohub_pred,
        "nohub_predicted_uncovered_clip_classes": total_cc - nohub_pred,
        "nohub_predicted_uncovered_rate": (total_cc - nohub_pred) / total_cc if total_cc else 0.0,
        "nohub_gt_correct_supported_clip_classes": nohub_gt,
        "nohub_gt_correct_uncovered_clip_classes": total_cc - nohub_gt,
        "nohub_gt_correct_uncovered_rate": (total_cc - nohub_gt) / total_cc if total_cc else 0.0,
        "coverage_minus_nohub_gt_correct_supported_delta": cov_gt - nohub_gt,
        "coverage_gt_correct_uncovered_rate_delta_vs_nohub": ((total_cc - cov_gt) / total_cc if total_cc else 0.0) - ((total_cc - nohub_gt) / total_cc if total_cc else 0.0),
        "role_summary": role_summary,
        "top_remaining_uncovered_classes": sorted(
            by_class_rows,
            key=lambda r: (-int(r["coverage_gt_correct_uncovered_clips"]), -int(r["clip_class_count"])),
        )[:30],
    }
    return audited_assignment_rows, audited_cc_rows, by_class_rows, summary


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    repo_root = Path(args.repo_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else run_root / "analysis" / "residual_gated_coverage_assignment" / str(args.dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _prepare_data(args)
    device = torch.device(str(args.device) if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    carrier_dim = int(np.asarray(data.examples[0]["carrier_vec"], dtype=np.float32).reshape(-1).shape[0])
    text_dim = int(data.text_matrix.shape[1])
    projector = Projector(ProjectorConfig(input_dim=text_dim, output_dim=carrier_dim)).to(device)
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(float(args.t_dis_init)), device=device, dtype=torch.float32))

    init_checkpoint = str(args.init_checkpoint).strip()
    if init_checkpoint == "auto":
        init_checkpoint = _auto_find_checkpoint(repo_root, str(args.dataset_name))
    checkpoint_summary = _load_checkpoint_if_requested(projector, theta_t, init_checkpoint, device)
    text_proj_all = _build_text_projection(text_matrix=data.text_matrix, projector=projector, device=device)

    assignment_rows: List[Dict[str, Any]] = []
    clip_class_rows: List[Dict[str, Any]] = []
    counters = Counter()
    clip_ids = sorted(int(k) for k in data.by_clip.keys())
    if int(args.max_clips) > 0 and len(clip_ids) > int(args.max_clips):
        rng = random.Random(int(args.seed))
        clip_ids = sorted(rng.sample(clip_ids, k=int(args.max_clips)))
        counters["subsampled_clips"] += 1

    iterator: Iterable[int] = clip_ids
    if bool(args.show_progress) and tqdm is not None:
        iterator = tqdm(clip_ids, desc="A8 coverage assignment", dynamic_ncols=True)

    with torch.no_grad():
        for clip_id in iterator:
            group = data.by_clip.get(int(clip_id), [])
            y_ids = sorted(int(x) for x in data.clip_y_base.get(int(clip_id), set()) if int(x) in data.raw_to_text_idx)
            if not group:
                counters["skip_empty_group"] += 1
                continue
            if not y_ids:
                counters["skip_no_full_y_base"] += 1
                continue
            Z = _carrier_tensor(group, device)
            text_idx = torch.tensor([int(data.raw_to_text_idx[int(rid)]) for rid in y_ids], device=device, dtype=torch.long)
            T = text_proj_all[text_idx]
            scores = torch.matmul(Z, T.t()).detach().cpu().numpy().astype(np.float32)
            rows, cc_rows = _assign_clip(
                clip_id=int(clip_id),
                group=group,
                y_ids=y_ids,
                score_matrix=scores,
                class_names=data.class_names,
                min_primary_score=float(args.min_primary_score),
                min_extra_score=float(args.min_extra_score),
                min_extra_margin=float(args.min_extra_margin),
            )
            assignment_rows.extend(rows)
            clip_class_rows.extend(cc_rows)
            counters["clips_processed"] += 1
            counters["assignment_rows"] += len(rows)
            counters["clip_class_rows"] += len(cc_rows)

    row_gap_path = Path(args.row_gap_csv).expanduser().resolve() if str(args.row_gap_csv).strip() else _default_row_gap_path(repo_root, str(args.dataset_name))
    row_gap_by_key, row_gap_rows, row_gap_summary = _load_row_gap(row_gap_path)
    audited_rows, audited_cc_rows, by_class_rows, audit_summary = _audit_assignments(
        assignment_rows=assignment_rows,
        clip_class_rows=clip_class_rows,
        row_gap_by_key=row_gap_by_key,
        row_gap_rows=row_gap_rows,
    )

    assignment_fields = [
        "clip_id", "trajectory_id", "assigned_raw_id", "assigned_class_name", "assignment_role",
        "assignment_score", "assignment_margin_vs_best_other", "assignment_rank_in_row", "assignment_entropy",
        "row_top1_raw_id_in_full_y", "row_top1_class_name_in_full_y", "row_top2_raw_id_in_full_y", "row_top2_class_name_in_full_y",
        "row_top1_top2_margin", "coverage_rank", "is_primary_coverage", "is_slack", "slack_reason",
        "clip_y_size", "trajectory_count", "audit_gt_raw_id", "audit_gt_class_name", "audit_assignment_matches_gt",
        "audit_weak_nohub_top1_is_gt", "audit_weak_base_top1_is_gt", "audit_weak_nohub_error_type",
        "audit_old_nohub_wrong", "audit_old_nohub_other_positive_confusion",
    ]
    cc_fields = [
        "clip_id", "raw_id", "class_name", "has_coverage_support", "support_trajectory_id", "support_score",
        "support_margin_vs_best_other", "support_entropy", "support_rank_in_row", "support_row_top1_raw_id_in_full_y",
        "support_row_top1_class_name_in_full_y", "is_slack", "slack_reason", "clip_y_size", "trajectory_count",
        "nohub_predicted_support", "nohub_gt_correct_support", "nohub_gt_row_count_in_clip",
        "coverage_gt_correct_support", "coverage_gt_correct_uncovered", "coverage_assigned_uncovered",
    ]
    class_fields = [
        "raw_id", "class_name", "clip_class_count", "coverage_supported_clips", "coverage_support_rate",
        "coverage_gt_correct_supported_clips", "coverage_gt_correct_uncovered_clips", "coverage_gt_correct_uncovered_rate",
        "nohub_predicted_supported_clips", "nohub_predicted_uncovered_clips", "nohub_predicted_uncovered_rate",
        "nohub_gt_correct_supported_clips", "nohub_gt_correct_uncovered_clips", "nohub_gt_correct_uncovered_rate",
        "slack_clips", "mean_support_score",
    ]
    _write_csv(out_dir / "coverage_assignment_rows.csv", audited_rows, assignment_fields)
    _write_csv(out_dir / "coverage_assignment_by_clip_class.csv", audited_cc_rows, cc_fields)
    _write_csv(out_dir / "coverage_assignment_by_class.csv", by_class_rows, class_fields)

    final_summary: Dict[str, Any] = {
        "status": "PASS",
        "timestamp": _now(),
        "run_root": str(run_root),
        "dataset_name": str(args.dataset_name),
        "out_dir": str(out_dir),
        "policy": {
            "does_not_train": True,
            "does_not_modify_control_plane": True,
            "assignment_generation_uses_row_level_gt": False,
            "assignment_generation_uses_oracle_correctness": False,
            "assignment_generation_uses_nohub_correctness": False,
            "assignment_generation_uses_manual_person_or_hub_prior": False,
            "assignment_allowed_inputs": [
                "GT carrier feature z_i",
                "clip-level full-Y base labels Y(v)",
                "text prototype bank t_c",
                "trajectory-text score matrix s(i,c)",
                "optional checkpoint only as score initializer, recorded separately",
            ],
            "audit_uses_row_level_gt": bool(row_gap_summary.get("available")),
            "score_initializer_checkpoint": checkpoint_summary,
        },
        "setup": {
            "repo_root": str(repo_root),
            "asset_root": str(Path(args.asset_root).expanduser().resolve()),
            "annotation_json": str(Path(args.annotation_json).expanduser().resolve()),
            "split_json": str(Path(args.split_json).expanduser().resolve()),
            "device": str(device),
            "carrier_dim": carrier_dim,
            "text_dim": text_dim,
            "t_dis": float(_compute_t_dis(theta_t).detach().cpu().item()),
            "min_primary_score": float(args.min_primary_score),
            "min_extra_score": float(args.min_extra_score),
            "min_extra_margin": float(args.min_extra_margin),
        },
        "materialization_summary": data.materialization_summary,
        "row_gap_summary": row_gap_summary,
        "assignment_generation_summary": dict(counters),
        "coverage_audit_summary": audit_summary,
        "gates": {
            "nohub_baseline_available": bool(row_gap_summary.get("available")),
            "coverage_assignment_generated": bool(len(audited_rows) > 0 and len(audited_cc_rows) > 0),
            "forbidden_gt_not_used_for_assignment": True,
            "manual_hub_prior_not_used": True,
        },
        "outputs": {
            "coverage_assignment_rows": str(out_dir / "coverage_assignment_rows.csv"),
            "coverage_assignment_by_clip_class": str(out_dir / "coverage_assignment_by_clip_class.csv"),
            "coverage_assignment_by_class": str(out_dir / "coverage_assignment_by_class.csv"),
            "coverage_assignment_summary": str(out_dir / "coverage_assignment_summary.json"),
            "coverage_vs_nohub_and_gt_audit": str(out_dir / "coverage_vs_nohub_and_gt_audit.json"),
            "takeover": str(out_dir / "COVERAGE_ASSIGNMENT_TAKEOVER.md"),
        },
    }
    _write_json(out_dir / "coverage_assignment_summary.json", final_summary)
    _write_json(out_dir / "coverage_vs_nohub_and_gt_audit.json", audit_summary)

    a = audit_summary
    lines = [
        "# A8 Coverage-Constrained Assignment TAKEOVER",
        "",
        "- status: PASS",
        f"- dataset_name: {args.dataset_name}",
        f"- clips_processed: {counters.get('clips_processed', 0)}",
        f"- clip_class_count: {a.get('clip_class_count', 0)}",
        f"- NoHub GT-correct uncovered: {a.get('nohub_gt_correct_uncovered_clip_classes', 0)} / {a.get('clip_class_count', 0)} = {a.get('nohub_gt_correct_uncovered_rate', 0.0):.6f}",
        f"- A8 coverage GT-correct uncovered: {a.get('coverage_gt_correct_uncovered_clip_classes', 0)} / {a.get('clip_class_count', 0)} = {a.get('coverage_gt_correct_uncovered_rate', 0.0):.6f}",
        f"- A8 assigned support uncovered: {a.get('coverage_uncovered_clip_classes', 0)} / {a.get('clip_class_count', 0)} = {a.get('coverage_uncovered_rate', 0.0):.6f}",
        f"- slack clip-classes: {a.get('coverage_slack_clip_classes', 0)} / {a.get('clip_class_count', 0)} = {a.get('coverage_slack_rate', 0.0):.6f}",
        f"- score initializer loaded: {bool(checkpoint_summary.get('loaded'))}",
        "",
        "## Policy",
        "- Assignment generation used GT carrier + full-Y + text scores only.",
        "- Row-level GT / oracle correctness / NoHub correctness are audit-only.",
        "- No manual person/hub raw-id prior is used.",
        "",
        "## Outputs",
        "- coverage_assignment_rows.csv",
        "- coverage_assignment_by_clip_class.csv",
        "- coverage_assignment_by_class.csv",
        "- coverage_assignment_summary.json",
        "- coverage_vs_nohub_and_gt_audit.json",
    ]
    (out_dir / "COVERAGE_ASSIGNMENT_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2, default=str))
    return final_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A8.0 coverage-constrained assignment audit")
    p.add_argument("--run_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--output_dir", default="")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--row_gap_csv", default="")
    p.add_argument("--init_checkpoint", default="", help="Empty for random text projector; 'auto' for existing score initializer; or explicit checkpoint path. Recorded as a score prior if used.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--t_dis_init", type=float, default=0.07)
    p.add_argument("--min_primary_score", type=float, default=-1.0, help="Primary coverage slack threshold. -1.0 means cover whenever a trajectory is available.")
    p.add_argument("--min_extra_score", type=float, default=0.20)
    p.add_argument("--min_extra_margin", type=float, default=0.05)
    p.add_argument("--max_clips", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--show_progress", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    args = p.parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not str(args.annotation_json).strip():
        args.annotation_json = str(Path(args.asset_root).expanduser().resolve() / "dataset" / "LV-VIS" / "annotations" / "train_instances.json")
    if not str(args.split_json).strip():
        args.split_json = str(repo_root / "package" / "reference" / "lvvis_official_base_novel_split.json")
    return args


def main() -> int:
    _run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
