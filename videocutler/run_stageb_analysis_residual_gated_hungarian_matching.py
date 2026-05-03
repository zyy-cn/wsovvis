#!/usr/bin/env python3
"""A8.1 per-clip partial one-to-one Hungarian matching audit.

Read-only side-path. It generates hard partial one-to-one trajectory-class
matches from GT carrier features and clip-level full-Y labels only.

Policy:
  * no training
  * no control-plane mutation
  * no row-level GT / oracle correctness / NoHub correctness for matching
  * no manual person/hub prior
  * no dummy, no slack, no extra-support routing

GT / NoHub correctness is joined only after matching for audit statistics.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:  # pragma: no cover
    linear_sum_assignment = None  # type: ignore
    _SCIPY_IMPORT_ERROR = repr(exc)
else:
    _SCIPY_IMPORT_ERROR = ""

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

from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig  # noqa: E402
from videocutler.run_stageb_analysis_residual_gated_coverage_assignment import (  # noqa: E402
    _auto_find_checkpoint,
    _build_text_projection,
    _carrier_tensor,
    _compute_t_dis,
    _default_row_gap_path,
    _entropy_np,
    _inverse_softplus,
    _load_checkpoint_if_requested,
    _load_row_gap,
    _nohub_clip_class_coverage_from_row_gap,
    _prepare_data,
    _rank_info,
    _truth,
    _write_csv,
    _write_json,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x)))
    except Exception:
        return int(default)


def _mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(list(vals), dtype=np.float64)))


def _hungarian_clip(
    *,
    clip_id: int,
    group: Sequence[Mapping[str, Any]],
    y_ids: Sequence[int],
    score_matrix: np.ndarray,
    class_names: Mapping[int, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partial hard one-to-one matching for one clip.

    No dummy/slack is introduced. SciPy rectangular Hungarian returns exactly
    min(num_trajectories, num_classes) pairs.
    """
    if linear_sum_assignment is None:
        raise RuntimeError(f"scipy.optimize.linear_sum_assignment import failed: {_SCIPY_IMPORT_ERROR}")
    n = int(len(group))
    m = int(len(y_ids))
    if n <= 0 or m <= 0:
        return [], [], []
    S = np.asarray(score_matrix, dtype=np.float64)
    if S.shape != (n, m):
        raise RuntimeError(f"score matrix shape mismatch for clip {clip_id}: {S.shape} vs {(n, m)}")
    row_ind, col_ind = linear_sum_assignment(S, maximize=True)
    matched_rows: List[Dict[str, Any]] = []
    used_i: Set[int] = set()
    used_c: Set[int] = set()
    for i0, c0 in zip(row_ind.tolist(), col_ind.tolist()):
        i = int(i0)
        c = int(c0)
        used_i.add(i)
        used_c.add(c)
        rid = int(y_ids[c])
        rank, assigned_score, assigned_margin, entropy, top1_col, top2_col, top1_top2_margin = _rank_info(S[i, :], c)
        ex = group[i]
        matched_rows.append({
            "clip_id": str(clip_id),
            "trajectory_id": str(ex.get("trajectory_id", "")),
            "matched_raw_id": str(rid),
            "matched_class_name": str(class_names.get(rid, rid)),
            "match_score": assigned_score,
            "match_margin_vs_best_other": assigned_margin,
            "match_rank_in_row": rank,
            "match_entropy": entropy,
            "row_top1_raw_id_in_full_y": str(int(y_ids[top1_col])) if top1_col >= 0 else "",
            "row_top1_class_name_in_full_y": str(class_names.get(int(y_ids[top1_col]), int(y_ids[top1_col]))) if top1_col >= 0 else "",
            "row_top2_raw_id_in_full_y": str(int(y_ids[top2_col])) if top2_col >= 0 else "",
            "row_top2_class_name_in_full_y": str(class_names.get(int(y_ids[top2_col]), int(y_ids[top2_col]))) if top2_col >= 0 else "",
            "row_top1_top2_margin": top1_top2_margin,
            "clip_y_size": m,
            "trajectory_count": n,
            "matched_pair_count_in_clip": min(n, m),
        })
    unmatched_classes: List[Dict[str, Any]] = []
    for c, rid0 in enumerate(y_ids):
        if c in used_c:
            continue
        rid = int(rid0)
        col = S[:, c]
        best_i = int(np.argmax(col)) if col.size else -1
        unmatched_classes.append({
            "clip_id": str(clip_id),
            "raw_id": str(rid),
            "class_name": str(class_names.get(rid, rid)),
            "best_available_trajectory_id": str(group[best_i].get("trajectory_id", "")) if best_i >= 0 else "",
            "best_available_score": float(col[best_i]) if best_i >= 0 else "",
            "clip_y_size": m,
            "trajectory_count": n,
            "unmatched_reason": "class_count_gt_trajectory_count" if m > n else "not_selected_by_partial_matching",
        })
    unmatched_trajectories: List[Dict[str, Any]] = []
    for i, ex in enumerate(group):
        if i in used_i:
            continue
        row_scores = S[i, :]
        top_c = int(np.argmax(row_scores)) if row_scores.size else -1
        unmatched_trajectories.append({
            "clip_id": str(clip_id),
            "trajectory_id": str(ex.get("trajectory_id", "")),
            "best_class_raw_id": str(int(y_ids[top_c])) if top_c >= 0 else "",
            "best_class_name": str(class_names.get(int(y_ids[top_c]), int(y_ids[top_c]))) if top_c >= 0 else "",
            "best_score": float(row_scores[top_c]) if top_c >= 0 else "",
            "clip_y_size": m,
            "trajectory_count": n,
            "unmatched_reason": "trajectory_count_gt_class_count" if n > m else "not_selected_by_partial_matching",
        })
    return matched_rows, unmatched_classes, unmatched_trajectories


def _audit(
    *,
    matched_rows: List[Dict[str, Any]],
    unmatched_classes: List[Dict[str, Any]],
    unmatched_trajectories: List[Dict[str, Any]],
    row_gap_by_key: Mapping[Tuple[str, str], Mapping[str, Any]],
    row_gap_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    nohub_cc = _nohub_clip_class_coverage_from_row_gap(row_gap_rows)
    matched_audited: List[Dict[str, Any]] = []
    by_class_counter: Dict[str, Counter] = defaultdict(Counter)
    score_by_class: Dict[str, List[float]] = defaultdict(list)
    global_c = Counter()
    transition = Counter()

    for row in matched_rows:
        out = dict(row)
        clip = str(row.get("clip_id", ""))
        tid = str(row.get("trajectory_id", ""))
        rid = _norm_id(row.get("matched_raw_id"))
        rg = row_gap_by_key.get((clip, tid))
        cc = nohub_cc.get((clip, rid), {})
        nohub_correct_for_clip_class = bool(int(cc.get("nohub_gt_correct_support", 0) or 0))
        out["nohub_gt_correct_support_for_matched_class"] = int(nohub_correct_for_clip_class)
        out["nohub_predicted_support_for_matched_class"] = int(cc.get("nohub_predicted_support", 0) or 0)
        match = False
        if rg:
            gt = _norm_id(rg.get("gt_raw_id"))
            gt_name = str(rg.get("gt_class_name", ""))
            old_nohub_ok = _truth(rg.get("weak_nohub_top1_is_gt"))
            old_base_ok = _truth(rg.get("weak_base_top1_is_gt"))
            err = str(rg.get("weak_nohub_error_type", ""))
            match = bool(rid and gt and rid == gt)
            out.update({
                "audit_gt_raw_id": gt,
                "audit_gt_class_name": gt_name,
                "audit_assignment_matches_gt": int(match),
                "audit_mismatch": int(not match),
                "audit_weak_nohub_top1_is_gt": int(old_nohub_ok),
                "audit_weak_base_top1_is_gt": int(old_base_ok),
                "audit_weak_nohub_error_type": err,
                "audit_old_nohub_wrong": int(not old_nohub_ok),
                "audit_old_nohub_other_positive_confusion": int(err == "other_positive_confusion"),
            })
            global_c["joined_gt_rows"] += 1
            global_c["matches_gt"] += int(match)
            global_c["old_nohub_wrong"] += int(not old_nohub_ok)
            global_c["old_nohub_correct"] += int(old_nohub_ok)
            global_c["other_positive"] += int(err == "other_positive_confusion")
            by_class_counter[rid]["matched_pairs"] += 1
            by_class_counter[rid]["assignment_matches_gt"] += int(match)
            by_class_counter[rid]["old_nohub_wrong_rows"] += int(not old_nohub_ok)
            by_class_counter[rid]["old_nohub_correct_rows"] += int(old_nohub_ok)
            by_class_counter[rid]["other_positive_rows"] += int(err == "other_positive_confusion")
            if nohub_correct_for_clip_class and match:
                transition["nohub_correct->hungarian_correct"] += 1
            elif nohub_correct_for_clip_class and not match:
                transition["nohub_correct->hungarian_wrong"] += 1
            elif (not nohub_correct_for_clip_class) and match:
                transition["nohub_wrong_or_uncovered->hungarian_correct"] += 1
            else:
                transition["nohub_wrong_or_uncovered->hungarian_wrong"] += 1
        else:
            out.update({
                "audit_gt_raw_id": "",
                "audit_gt_class_name": "",
                "audit_assignment_matches_gt": "",
                "audit_mismatch": "",
                "audit_weak_nohub_top1_is_gt": "",
                "audit_weak_base_top1_is_gt": "",
                "audit_weak_nohub_error_type": "",
                "audit_old_nohub_wrong": "",
                "audit_old_nohub_other_positive_confusion": "",
            })
            by_class_counter[rid]["matched_pairs"] += 1
        score_by_class[rid].append(_as_float(row.get("match_score"), 0.0))
        matched_audited.append(out)

    unmatched_classes_audited: List[Dict[str, Any]] = []
    for row in unmatched_classes:
        out = dict(row)
        clip = str(row.get("clip_id", ""))
        rid = _norm_id(row.get("raw_id"))
        cc = nohub_cc.get((clip, rid), {})
        out["nohub_predicted_support"] = int(cc.get("nohub_predicted_support", 0) or 0)
        out["nohub_gt_correct_support"] = int(cc.get("nohub_gt_correct_support", 0) or 0)
        out["nohub_gt_row_count_in_clip"] = int(cc.get("nohub_gt_row_count_in_clip", 0) or 0)
        by_class_counter[rid]["unmatched_class_clips"] += 1
        unmatched_classes_audited.append(out)

    unmatched_traj_audited: List[Dict[str, Any]] = []
    for row in unmatched_trajectories:
        out = dict(row)
        rg = row_gap_by_key.get((str(row.get("clip_id", "")), str(row.get("trajectory_id", ""))))
        if rg:
            out.update({
                "audit_gt_raw_id": _norm_id(rg.get("gt_raw_id")),
                "audit_gt_class_name": str(rg.get("gt_class_name", "")),
                "audit_weak_nohub_top1_is_gt": int(_truth(rg.get("weak_nohub_top1_is_gt"))),
                "audit_weak_nohub_error_type": str(rg.get("weak_nohub_error_type", "")),
            })
        else:
            out.update({"audit_gt_raw_id": "", "audit_gt_class_name": "", "audit_weak_nohub_top1_is_gt": "", "audit_weak_nohub_error_type": ""})
        unmatched_traj_audited.append(out)

    by_class_rows: List[Dict[str, Any]] = []
    # add NoHub clip-class coverage universe for fair per-class comparison.
    nohub_class_c = defaultdict(Counter)
    for (_clip, rid), cc in nohub_cc.items():
        nohub_class_c[str(rid)]["clip_class_count"] += 1
        nohub_class_c[str(rid)]["nohub_predicted_supported_clips"] += int(cc.get("nohub_predicted_support", 0) or 0)
        nohub_class_c[str(rid)]["nohub_gt_correct_supported_clips"] += int(cc.get("nohub_gt_correct_support", 0) or 0)
    for rid in sorted(set(by_class_counter) | set(nohub_class_c), key=lambda x: int(float(x)) if str(x).strip() else -1):
        bc = by_class_counter[rid]
        nc = nohub_class_c[rid]
        clip_count = int(nc.get("clip_class_count", 0))
        nohub_correct = int(nc.get("nohub_gt_correct_supported_clips", 0))
        h_correct = int(bc.get("assignment_matches_gt", 0))
        by_class_rows.append({
            "raw_id": rid,
            "clip_class_count": clip_count,
            "matched_pairs": int(bc.get("matched_pairs", 0)),
            "unmatched_class_clips": int(bc.get("unmatched_class_clips", 0)),
            "hungarian_gt_correct_supported_clips_or_rows": h_correct,
            "nohub_gt_correct_supported_clips": nohub_correct,
            "support_delta_hungarian_minus_nohub": h_correct - nohub_correct,
            "hungarian_match_rate_on_pairs": h_correct / max(int(bc.get("matched_pairs", 0)), 1),
            "nohub_gt_correct_support_rate": nohub_correct / max(clip_count, 1),
            "mean_match_score": _mean(score_by_class.get(rid, [])),
        })

    total_pairs = len(matched_audited)
    total_classes = total_pairs + len(unmatched_classes_audited)
    total_traj = total_pairs + len(unmatched_traj_audited)
    joined = int(global_c.get("joined_gt_rows", 0))
    transition_rows = [{"transition": k, "count": int(v), "rate_on_joined_pairs": int(v) / max(joined, 1)} for k, v in sorted(transition.items())]
    summary = {
        "matched_pair_count": total_pairs,
        "unmatched_class_count": len(unmatched_classes_audited),
        "unmatched_trajectory_count": len(unmatched_traj_audited),
        "class_item_count": total_classes,
        "trajectory_item_count": total_traj,
        "unmatched_class_rate": len(unmatched_classes_audited) / max(total_classes, 1),
        "unmatched_trajectory_rate": len(unmatched_traj_audited) / max(total_traj, 1),
        "joined_gt_rows": joined,
        "assignment_gt_match_count": int(global_c.get("matches_gt", 0)),
        "assignment_gt_match_rate": int(global_c.get("matches_gt", 0)) / max(joined, 1),
        "mismatch_rate": 1.0 - (int(global_c.get("matches_gt", 0)) / max(joined, 1)),
        "old_nohub_wrong_on_matched_pairs": int(global_c.get("old_nohub_wrong", 0)),
        "old_nohub_correct_on_matched_pairs": int(global_c.get("old_nohub_correct", 0)),
        "old_nohub_other_positive_on_matched_pairs": int(global_c.get("other_positive", 0)),
        "transition_summary": dict(transition),
        "transition_rows": transition_rows,
        "net_clip_class_gain_proxy": int(transition.get("nohub_wrong_or_uncovered->hungarian_correct", 0)) - int(transition.get("nohub_correct->hungarian_wrong", 0)),
    }
    return matched_audited, unmatched_classes_audited, unmatched_traj_audited, by_class_rows, summary


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    if linear_sum_assignment is None:
        raise RuntimeError(f"SciPy is required for Hungarian matching: {_SCIPY_IMPORT_ERROR}")
    run_root = Path(args.run_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else run_root / "analysis" / "residual_gated_hungarian_matching" / str(args.dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed)); np.random.seed(int(args.seed)); torch.manual_seed(int(args.seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(str(args.device))

    if not str(args.annotation_json).strip():
        args.annotation_json = str(Path(args.repo_root) / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json")
    if not str(args.split_json).strip():
        args.split_json = str(Path(args.repo_root) / "package" / "reference" / "lvvis_official_base_novel_split.json")

    data = _prepare_data(args)
    text_tensor = torch.tensor(np.asarray(data.text_matrix, dtype=np.float32), device=device, dtype=torch.float32)
    projector = Projector(ProjectorConfig()).to(device)
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(args.t_dis_init) - 1.0e-4, 1.0e-6)), device=device, dtype=torch.float32))
    ckpt = _auto_find_checkpoint(repo_root, str(args.dataset_name)) if str(args.init_checkpoint).strip() == "auto" else str(args.init_checkpoint).strip()
    checkpoint_summary = _load_checkpoint_if_requested(projector, theta_t, ckpt, device)
    text_proj_all = _build_text_projection(text_matrix=data.text_matrix, projector=projector, device=device)

    matched_rows: List[Dict[str, Any]] = []
    unmatched_classes: List[Dict[str, Any]] = []
    unmatched_trajectories: List[Dict[str, Any]] = []
    counters = Counter()
    clip_ids = sorted(int(k) for k in data.by_clip.keys())
    if int(args.max_clips) > 0 and len(clip_ids) > int(args.max_clips):
        rng = random.Random(int(args.seed))
        clip_ids = sorted(rng.sample(clip_ids, int(args.max_clips)))
        counters["subsampled_clips"] += 1
    iterator: Iterable[int] = clip_ids
    if bool(args.show_progress) and tqdm is not None:
        iterator = tqdm(clip_ids, desc="A8.1 Hungarian matching", dynamic_ncols=True)

    with torch.no_grad():
        for clip_id in iterator:
            group = data.by_clip.get(int(clip_id), [])
            y_ids = sorted(int(x) for x in data.clip_y_base.get(int(clip_id), set()) if int(x) in data.raw_to_text_idx)
            if not group:
                counters["skip_empty_group"] += 1; continue
            if not y_ids:
                counters["skip_no_full_y_base"] += 1; continue
            Z = _carrier_tensor(group, device)
            text_idx = torch.tensor([int(data.raw_to_text_idx[int(rid)]) for rid in y_ids], device=device, dtype=torch.long)
            T = text_proj_all[text_idx]
            S = torch.matmul(Z, T.t()).detach().cpu().numpy().astype(np.float32)
            mr, uc, ut = _hungarian_clip(clip_id=int(clip_id), group=group, y_ids=y_ids, score_matrix=S, class_names=data.class_names)
            matched_rows.extend(mr); unmatched_classes.extend(uc); unmatched_trajectories.extend(ut)
            counters["clips_processed"] += 1
            counters["matched_pairs"] += len(mr)
            counters["unmatched_classes"] += len(uc)
            counters["unmatched_trajectories"] += len(ut)

    row_gap_path = Path(args.row_gap_csv).expanduser().resolve() if str(args.row_gap_csv).strip() else _default_row_gap_path(repo_root, str(args.dataset_name))
    row_gap_by_key, row_gap_rows, row_gap_summary = _load_row_gap(row_gap_path)
    matched_a, uc_a, ut_a, by_class_rows, audit_summary = _audit(
        matched_rows=matched_rows,
        unmatched_classes=unmatched_classes,
        unmatched_trajectories=unmatched_trajectories,
        row_gap_by_key=row_gap_by_key,
        row_gap_rows=row_gap_rows,
    )

    matched_fields = [
        "clip_id", "trajectory_id", "matched_raw_id", "matched_class_name", "match_score", "match_margin_vs_best_other", "match_rank_in_row", "match_entropy",
        "row_top1_raw_id_in_full_y", "row_top1_class_name_in_full_y", "row_top2_raw_id_in_full_y", "row_top2_class_name_in_full_y", "row_top1_top2_margin",
        "clip_y_size", "trajectory_count", "matched_pair_count_in_clip", "nohub_gt_correct_support_for_matched_class", "nohub_predicted_support_for_matched_class",
        "audit_gt_raw_id", "audit_gt_class_name", "audit_assignment_matches_gt", "audit_mismatch", "audit_weak_nohub_top1_is_gt", "audit_weak_base_top1_is_gt", "audit_weak_nohub_error_type", "audit_old_nohub_wrong", "audit_old_nohub_other_positive_confusion",
    ]
    _write_csv(out_dir / "hungarian_matched_pairs.csv", matched_a, matched_fields)
    _write_csv(out_dir / "hungarian_unmatched_classes.csv", uc_a)
    _write_csv(out_dir / "hungarian_unmatched_trajectories.csv", ut_a)
    _write_csv(out_dir / "hungarian_by_class_delta_vs_nohub.csv", by_class_rows)
    _write_csv(out_dir / "hungarian_transition_summary.csv", audit_summary.get("transition_rows", []))

    final = {
        "status": "PASS",
        "timestamp": _now(),
        "run_root": str(run_root),
        "dataset_name": str(args.dataset_name),
        "out_dir": str(out_dir),
        "policy": {
            "does_not_train": True,
            "does_not_modify_control_plane": True,
            "uses_dummy": False,
            "uses_slack": False,
            "uses_extra_support": False,
            "matching_generation_uses_row_level_gt": False,
            "matching_generation_uses_oracle_correctness": False,
            "matching_generation_uses_nohub_correctness": False,
            "matching_generation_uses_manual_person_or_hub_prior": False,
            "audit_uses_row_level_gt": bool(row_gap_summary.get("available")),
            "allowed_matching_inputs": ["GT carrier feature z_i", "clip-level full-Y base labels Y(v)", "text prototype bank t_c", "trajectory-text score matrix s(i,c)", "optional checkpoint as score initializer"],
            "score_initializer_checkpoint": checkpoint_summary,
        },
        "setup": {
            "repo_root": str(repo_root),
            "asset_root": str(Path(args.asset_root).expanduser().resolve()),
            "device": str(device),
            "t_dis": float(_compute_t_dis(theta_t).detach().cpu().item()),
            "algorithm": "scipy.optimize.linear_sum_assignment(maximize=True)",
        },
        "materialization_summary": data.materialization_summary,
        "row_gap_summary": row_gap_summary,
        "matching_generation_summary": dict(counters),
        "hungarian_audit_summary": audit_summary,
        "outputs": {
            "matched_pairs": str(out_dir / "hungarian_matched_pairs.csv"),
            "unmatched_classes": str(out_dir / "hungarian_unmatched_classes.csv"),
            "unmatched_trajectories": str(out_dir / "hungarian_unmatched_trajectories.csv"),
            "by_class_delta_vs_nohub": str(out_dir / "hungarian_by_class_delta_vs_nohub.csv"),
            "transition_summary": str(out_dir / "hungarian_transition_summary.csv"),
            "match_summary": str(out_dir / "hungarian_match_summary.json"),
            "audit": str(out_dir / "hungarian_vs_nohub_gt_audit.json"),
            "takeover": str(out_dir / "HUNGARIAN_MATCHING_TAKEOVER.md"),
        },
    }
    _write_json(out_dir / "hungarian_match_summary.json", final)
    _write_json(out_dir / "hungarian_vs_nohub_gt_audit.json", audit_summary)
    a = audit_summary
    lines = [
        "# A8.1 Hungarian Partial One-to-One Matching TAKEOVER",
        "",
        "- status: PASS",
        f"- dataset_name: {args.dataset_name}",
        f"- clips_processed: {counters.get('clips_processed', 0)}",
        f"- matched_pair_count: {a.get('matched_pair_count', 0)}",
        f"- unmatched_class_count/rate: {a.get('unmatched_class_count', 0)} / {a.get('class_item_count', 0)} = {a.get('unmatched_class_rate', 0.0):.6f}",
        f"- unmatched_trajectory_count/rate: {a.get('unmatched_trajectory_count', 0)} / {a.get('trajectory_item_count', 0)} = {a.get('unmatched_trajectory_rate', 0.0):.6f}",
        f"- assignment_gt_match_rate: {a.get('assignment_gt_match_rate', 0.0):.6f}",
        f"- mismatch_rate: {a.get('mismatch_rate', 0.0):.6f}",
        f"- net_clip_class_gain_proxy: {a.get('net_clip_class_gain_proxy', 0)}",
        f"- score initializer loaded: {bool(checkpoint_summary.get('loaded'))}",
        "",
        "## Policy",
        "- Matching used GT carrier + full-Y + text scores only.",
        "- No dummy, no slack, no extra_support.",
        "- Row-level GT / NoHub correctness are audit-only.",
        "- No manual person/hub prior is used.",
    ]
    (out_dir / "HUNGARIAN_MATCHING_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2, default=str))
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A8.1 partial one-to-one Hungarian matching audit")
    p.add_argument("--run_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--output_dir", default="")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--row_gap_csv", default="")
    p.add_argument("--init_checkpoint", default="auto")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--max_clips", type=int, default=0)
    p.add_argument("--show_progress", action="store_true", default=True)
    p.add_argument("--t_dis_init", type=float, default=0.07)
    return p.parse_args()


def main() -> int:
    _run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
