#!/usr/bin/env python3
"""A8 joint prealign + train-time dynamic Hungarian training.

This side-path tool implements the corrected A8 dynamic interface:

  L_total = lambda_prealign * L_prealign_current
          + lambda_dynamic   * L_dynamic_hungarian_current

where:
  * L_prealign_current is exactly the current full-Y/base-vocab prealign bag
    objective from videocutler.run_stageb_train_hungarian_prealign._prealign_loss_for_clip.
  * L_dynamic_hungarian_current recomputes the row x full-Y score matrix with
    current model parameters at every clip/iteration, runs Hungarian assignment
    on that current matrix, and trains CE/InfoNCE on the resulting dynamic pairs.

It intentionally does NOT use matched_pairs_csv.matched_raw_id as a training
label. Any clip-universe CSV is used only to select clips / optional row universe.
It also does NOT introduce GT-target CE, visible525 CE, rank-margin loss,
hard-negative loss, dummy/slack rows, extra support, or NoHub correctness labels.

Primary metrics are canonical visible-525 rank@K emitted by
  tools/a8_visible525_candidate_rankk_audit.py
Legacy row_gap / clip-local full-Y micro_top1 is not reported as a headline.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover
    _scipy_linear_sum_assignment = None

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
    _compute_t_dis,
    _inverse_softplus,
    _load_checkpoint_if_requested,
    _prepare_data as _prepare_residual_gt_data,
    _write_csv,
    _write_json,
)
from videocutler.run_stageb_train_hungarian_prealign import (  # noqa: E402
    _group_by_clip,
    _prealign_loss_for_clip,
    _project_text,
)
from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab  # noqa: E402
from videocutler.ext_stageb_ovvis.algorithms.prealign import _prepare_examples as _prepare_prealign_examples  # noqa: E402
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (  # noqa: E402
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)
from videocutler.run_stageb_train_gt_full_y_clean import (  # noqa: E402
    _bootstrap_asset_links,
    _class_name_map_from_annotation_json,
    _class_name_map_from_text_records,
    _load_base_ids,
    _load_clip_y_base,
)


SUPPORTED_TRAJECTORY_SOURCE_BRANCHES = {"gt_upper_bound", "mainline"}


def _prepare_data(args: argparse.Namespace) -> Any:
    """Load training examples for either GT or VideoCutLER trajectories.

    This local wrapper preserves the old default (gt_upper_bound), but exposes
    the materialization branch so A8 dynamic Hungarian can be run on the normal
    VideoCutLER trajectory/carrier assets:

      * gt_upper_bound -> exports_gt/ + carrier_bank_gt/
      * mainline       -> exports/    + carrier_bank/

    It does not change the weak/full-Y candidate semantics or the Hungarian
    assignment rule; only the trajectory/carrier source branch changes.
    """
    branch = str(getattr(args, "trajectory_source_branch", "gt_upper_bound")).strip() or "gt_upper_bound"
    if branch not in SUPPORTED_TRAJECTORY_SOURCE_BRANCHES:
        raise ValueError(f"unsupported trajectory_source_branch={branch!r}; expected one of {sorted(SUPPORTED_TRAJECTORY_SOURCE_BRANCHES)}")

    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    out_root_for_assets = Path(args.output_dir).expanduser().resolve() if str(getattr(args, "output_dir", "")).strip() else Path(args.run_root).expanduser().resolve()
    _bootstrap_asset_links(repo_root, asset_root)
    _bootstrap_asset_links(out_root_for_assets, asset_root)

    base_ids = _load_base_ids(Path(args.split_json).expanduser().resolve())
    clip_y_base = _load_clip_y_base(Path(args.annotation_json).expanduser().resolve(), base_ids)

    # Use the canonical G7 phase-1 materializer for both branches.  This is the
    # same branch abstraction used by the normal G7 training code and avoids
    # hand-assembling paths in this side-path A8 tool.
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(repo_root)
        materialized = materialize_phase1_training_samples(
            repo_root,
            Phase1MaterializationConfig(
                dataset_name=str(args.dataset_name),
                trajectory_source_branch=str(branch),
                smoke=bool(args.smoke),
                smoke_max_trajectories=int(args.smoke_max_trajectories),
                subset_fraction=args.subset_fraction,
                subset_seed=int(args.seed),
            ),
        )
    finally:
        import os
        os.chdir(old_cwd)

    samples_raw = materialized.get("valid_samples") or materialized.get("samples") or []
    samples: List[Dict[str, Any]] = []
    sample_counters = Counter()
    for sample in samples_raw:
        if not bool(sample.get("sample_valid", False)):
            sample_counters["skip_sample_not_valid"] += 1
            continue
        clip = _as_int(sample.get("clip_id"))
        if clip is None:
            sample_counters["skip_no_clip_id"] += 1
            continue
        y_base = sorted(int(x) for x in clip_y_base.get(int(clip), set()))
        if not y_base:
            sample_counters["skip_no_y_base"] += 1
            continue
        row = dict(sample)
        row["observed_raw_ids"] = [int(x) for x in y_base]
        row["clean_label_source"] = "full_Y_base_from_GT_annotations"
        row["trajectory_source_branch"] = str(branch)
        samples.append(row)

    prepared = _prepare_prealign_examples(
        samples,
        output_root=out_root_for_assets,
        dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(branch),
    )
    examples = list(prepared.get("examples", []))
    if not examples:
        raise RuntimeError(f"no materialized carrier examples were loaded for trajectory_source_branch={branch}")

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

    materialization_summary = {
        "trajectory_source_branch": str(branch),
        "uses_gt_upper_bound_trajectory": bool(branch == "gt_upper_bound"),
        "uses_videocutler_mainline_trajectory": bool(branch == "mainline"),
        "materialized_stats": materialized.get("stats", {}),
        "materialized_resolution": materialized.get("resolution", {}),
        "sample_counters": dict(sample_counters),
        "prepare_skipped_reason_histogram": dict(prepared.get("skipped_reason_histogram", {})),
        "sample_count_after_full_y_base_filter": int(len(samples)),
        "trainable_example_count": int(len(examples)),
        "clip_count_after_grouping": int(len(by_clip)),
        "base_ids_count": int(len(base_ids)),
        "text_bank_count": int(len(text_ids)),
    }

    return SimpleNamespace(
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(str(x)))
    except Exception:
        return default


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(np.mean(np.asarray(list(xs), dtype=np.float64)))


def _save_checkpoint(
    path: Path,
    *,
    projector: Projector,
    theta_t: torch.nn.Parameter,
    epoch: int,
    global_step: int,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text_projector_state_dict": projector.state_dict(),
            "text_projector_config": {
                "input_dim": 512,
                "hidden_dim": 1024,
                "output_dim": 768,
                "dropout": 0.0,
                "use_layernorm": True,
            },
            "theta_T": float(theta_t.detach().cpu().item()),
            "stage_id": "a8_joint_train_time_dynamic_hungarian",
            "loss": "joint_prealign_train_time_dynamic_hungarian",
            "epoch": int(epoch),
            "global_step": int(global_step),
            **dict(payload),
        },
        path,
    )


def _load_train_visible_ids_from_csv(path: Optional[Path]) -> set[int]:
    """Load the canonical train-visible-525 raw ids when the visibility CSV exists.

    This mirrors tools/a8_visible525_candidate_rankk_audit.py but is intentionally
    non-fatal for training setup: graph regularization can fall back to the base
    vocabulary when the visibility audit asset is absent in a new snapshot.
    """
    if path is None or not path.is_file():
        return set()
    ids: set[int] = set()
    try:
        rows = _read_csv(path)
    except Exception:
        return set()
    for r in rows:
        rid = _as_int(r.get("raw_id"))
        if rid is None:
            continue
        if str(r.get("in_row_gap", "0")).strip() == "1":
            ids.add(int(rid))
    return ids


def _build_graph_preserve_cache(
    *,
    data: Any,
    text_tensor: torch.Tensor,
    base_raw_ids: Sequence[int],
    device: torch.device,
    mode: str,
    scope: str,
    topk: int,
    tau: float,
    visible_csv: Optional[Path],
    seed: int,
) -> Dict[str, Any]:
    """Precompute local raw-text graph targets for structure-preserving loss.

    The loss is intentionally a text-prototype-only regularizer: it does not use
    row-level GT, dynamic Hungarian assignments, or visual prototype targets.  It
    preserves the local raw CLIP text neighborhood after projection:

      KL(softmax(S_raw[i, N_i]/tau) || softmax(S_proj[i, N_i]/tau)).

    `random_text_topk` keeps the same implementation surface but randomizes the
    neighborhood set as a control for "just adding a regularizer".
    """
    mode = str(mode).strip().lower()
    if mode in {"", "none", "off", "false", "0"}:
        return {"enabled": False, "mode": "none", "reason": "graph_preserve_mode_disabled"}

    if mode not in {"raw_text_topk", "random_text_topk"}:
        raise ValueError(f"unsupported graph_preserve_mode={mode!r}")
    scope = str(scope).strip().lower()
    if scope not in {"visible525", "base_vocab"}:
        raise ValueError(f"unsupported graph_preserve_scope={scope!r}")

    raw_ids: List[int]
    visible_ids: set[int] = set()
    visible_csv_status = "NOT_USED"
    if scope == "visible525":
        visible_ids = _load_train_visible_ids_from_csv(visible_csv)
        if len(visible_ids) == 525:
            raw_ids = [int(rid) for rid in sorted(visible_ids) if int(rid) in data.raw_to_text_idx]
            visible_csv_status = "PASS"
        else:
            # Robust fallback for clean snapshots without prior visibility audit.
            raw_ids = [int(rid) for rid in sorted(base_raw_ids) if int(rid) in data.raw_to_text_idx]
            visible_csv_status = f"FALLBACK_TO_BASE_VOCAB_visible_count={len(visible_ids)}"
    else:
        raw_ids = [int(rid) for rid in sorted(base_raw_ids) if int(rid) in data.raw_to_text_idx]
        visible_csv_status = "NOT_USED_BASE_VOCAB_SCOPE"

    raw_ids = [int(rid) for rid in raw_ids if int(rid) in data.raw_to_text_idx]
    if len(raw_ids) < 3:
        raise RuntimeError(f"graph preserve scope has too few classes: {len(raw_ids)}")
    k = min(int(topk), len(raw_ids) - 1)
    if k <= 0:
        raise RuntimeError(f"graph_preserve_topk must be positive after clipping, got {topk}")

    text_indices_np = np.asarray([int(data.raw_to_text_idx[int(rid)]) for rid in raw_ids], dtype=np.int64)
    with torch.no_grad():
        T = F.normalize(text_tensor.index_select(0, torch.tensor(text_indices_np, device=device, dtype=torch.long)), p=2.0, dim=-1)
        sim = torch.matmul(T, T.t())
        sim.fill_diagonal_(-float("inf"))
        if mode == "raw_text_topk":
            neighbor_idx = torch.topk(sim, k=int(k), dim=1, largest=True, sorted=True).indices
        else:
            rng = np.random.default_rng(int(seed))
            neigh_rows: List[np.ndarray] = []
            all_idx = np.arange(len(raw_ids), dtype=np.int64)
            for i in range(len(raw_ids)):
                choices = all_idx[all_idx != i]
                neigh_rows.append(rng.choice(choices, size=int(k), replace=False))
            neighbor_idx = torch.tensor(np.stack(neigh_rows, axis=0), device=device, dtype=torch.long)
        raw_neighbor_sim = torch.gather(sim, 1, neighbor_idx)
        # Random neighbors may include mostly low-similarity pairs; keep the real
        # raw-sim target distribution over the chosen neighbors for a fair control.
        target_prob = torch.softmax(raw_neighbor_sim / max(float(tau), 1.0e-6), dim=1).detach()

    return {
        "enabled": True,
        "mode": mode,
        "scope": scope,
        "resolved_scope": "visible525" if scope == "visible525" and visible_csv_status == "PASS" else "base_vocab",
        "visible_csv": str(visible_csv) if visible_csv is not None else "",
        "visible_csv_status": visible_csv_status,
        "class_count": int(len(raw_ids)),
        "topk": int(k),
        "tau": float(tau),
        "raw_ids_head": [int(x) for x in raw_ids[:20]],
        "scope_text_indices": torch.tensor(text_indices_np, device=device, dtype=torch.long),
        "neighbor_idx": neighbor_idx,
        "target_prob": target_prob,
    }


def _graph_preserve_loss(
    *,
    text_proj_all: torch.Tensor,
    cache: Mapping[str, Any],
    tau: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if not bool(cache.get("enabled", False)):
        zero = text_proj_all.sum() * 0.0
        return zero, {"graph_preserve_enabled": False, "graph_preserve_loss": 0.0}
    scope_text_indices = cache["scope_text_indices"]
    neighbor_idx = cache["neighbor_idx"]
    target_prob = cache["target_prob"]
    P = F.normalize(text_proj_all.index_select(0, scope_text_indices), p=2.0, dim=-1)
    sim_proj = torch.matmul(P, P.t())
    proj_neighbor_sim = torch.gather(sim_proj, 1, neighbor_idx)
    log_prob = torch.log_softmax(proj_neighbor_sim / max(float(tau), 1.0e-6), dim=1)
    tgt = target_prob.to(device=log_prob.device, dtype=log_prob.dtype)
    cross_entropy = -(tgt * log_prob).sum(dim=1).mean()
    target_entropy = -(tgt * torch.log(torch.clamp(tgt, min=1.0e-12))).sum(dim=1).mean().detach()
    loss = cross_entropy - target_entropy
    with torch.no_grad():
        stats = {
            "graph_preserve_enabled": True,
            "graph_preserve_mode": str(cache.get("mode", "")),
            "graph_preserve_scope": str(cache.get("resolved_scope", cache.get("scope", ""))),
            "graph_preserve_class_count": int(cache.get("class_count", 0)),
            "graph_preserve_topk": int(cache.get("topk", 0)),
            "graph_preserve_loss": float(loss.detach().cpu().item()),
            "graph_preserve_cross_entropy": float(cross_entropy.detach().cpu().item()),
            "graph_preserve_target_entropy": float(target_entropy.detach().cpu().item()),
            "graph_preserve_proj_neighbor_sim_mean": float(proj_neighbor_sim.detach().mean().cpu().item()),
        }
    return loss, stats


def _default_clip_universe_csv(run_root: Path, dataset_name: str) -> Path:
    candidates = [
        run_root / "analysis" / "residual_gated_hungarian_matching_baseline_full_y_5ep" / str(dataset_name) / "hungarian_matched_pairs.csv",
        run_root / "analysis" / "residual_gated_hungarian_matching" / str(dataset_name) / "hungarian_matched_pairs.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def _default_output_root(run_root: Path, dataset_name: str, name: str) -> Path:
    return run_root / "outputs" / "a8_joint_train_time_dynamic_hungarian" / str(dataset_name) / str(name)


def _load_clip_universe(path: Optional[Path]) -> Tuple[Dict[int, List[str]], Dict[str, Any]]:
    if path is None or not path.is_file():
        return {}, {"status": "NOT_USED", "reason": "clip_universe_csv_missing_or_empty", "path": str(path) if path else ""}
    rows = _read_csv(path)
    by_clip: Dict[int, List[str]] = defaultdict(list)
    counters = Counter()
    for r in rows:
        cid = _as_int(r.get("clip_id"))
        tid = str(r.get("trajectory_id", "")).strip()
        if cid is None:
            counters["skip_missing_clip_id"] += 1
            continue
        if tid:
            by_clip[int(cid)].append(tid)
        else:
            by_clip[int(cid)]
        counters["rows"] += 1
    # unique trajectory ids per clip, preserving order
    out: Dict[int, List[str]] = {}
    for cid, tids in by_clip.items():
        seen = set()
        cur: List[str] = []
        for tid in tids:
            if tid in seen:
                continue
            seen.add(tid)
            cur.append(tid)
        out[int(cid)] = cur
    return out, {"status": "PASS", "path": str(path), "input_rows": len(rows), "clip_count": len(out), "counters": dict(counters)}


def _example_by_tid(examples: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for ex in examples:
        tid = str(ex.get("trajectory_id", ""))
        if tid and tid not in out:
            out[tid] = ex
    return out


def _select_clip_rows(
    *,
    clip_id: int,
    all_by_clip: Mapping[int, Sequence[Mapping[str, Any]]],
    example_by_tid: Mapping[str, Mapping[str, Any]],
    clip_universe_by_clip: Mapping[int, Sequence[str]],
    row_source: str,
) -> List[Mapping[str, Any]]:
    row_source = str(row_source).strip().lower()
    if row_source == "all_clip_trajectories":
        return [dict(x) for x in all_by_clip.get(int(clip_id), [])]
    if row_source == "clip_universe_rows":
        tids = list(clip_universe_by_clip.get(int(clip_id), []))
        rows: List[Mapping[str, Any]] = []
        for tid in tids:
            ex = example_by_tid.get(str(tid))
            if ex is not None:
                rows.append(dict(ex))
        return rows
    raise ValueError(f"unsupported dynamic_row_source={row_source!r}")


def _carrier_tensor(rows: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    vecs: List[torch.Tensor] = []
    for ex in rows:
        if "carrier_vec" not in ex:
            continue
        arr = np.asarray(ex["carrier_vec"], dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        vecs.append(torch.from_numpy(arr.astype(np.float32)))
    if not vecs:
        raise RuntimeError("empty carrier rows for dynamic Hungarian")
    z = torch.stack(vecs, dim=0).to(device=device, dtype=torch.float32)
    return F.normalize(z, p=2.0, dim=-1)


def _hungarian_maximize(score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return row/col assignment for a score matrix.

    SciPy is used when available. The greedy fallback is deterministic and is
    logged as a fallback, but formal runs should normally use SciPy.
    """
    if score.ndim != 2:
        raise ValueError(f"score must be 2D, got shape={score.shape}")
    if score.shape[0] == 0 or score.shape[1] == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), "empty"
    if _scipy_linear_sum_assignment is not None:
        r, c = _scipy_linear_sum_assignment(score, maximize=True)
        return np.asarray(r, dtype=np.int64), np.asarray(c, dtype=np.int64), "scipy_linear_sum_assignment_maximize"
    # fallback: greedy maximum without replacement
    flat_order = np.argsort(-score.reshape(-1), kind="mergesort")
    used_r: set[int] = set()
    used_c: set[int] = set()
    out_r: List[int] = []
    out_c: List[int] = []
    n_cols = int(score.shape[1])
    for idx in flat_order:
        rr = int(idx // n_cols)
        cc = int(idx % n_cols)
        if rr in used_r or cc in used_c:
            continue
        used_r.add(rr); used_c.add(cc); out_r.append(rr); out_c.append(cc)
        if len(out_r) >= min(score.shape[0], score.shape[1]):
            break
    return np.asarray(out_r, dtype=np.int64), np.asarray(out_c, dtype=np.int64), "greedy_fallback_no_scipy"


def _dynamic_hungarian_loss_for_clip(
    *,
    clip_id: int,
    rows: Sequence[Mapping[str, Any]],
    data: Any,
    text_proj_all: torch.Tensor,
    theta_t: torch.nn.Parameter,
    device: torch.device,
    loss_name: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    y_ids = sorted(int(x) for x in data.clip_y_base.get(int(clip_id), set()) if int(x) in data.base_ids and int(x) in data.raw_to_text_idx)
    if not y_ids:
        raise RuntimeError(f"clip {clip_id} has empty full-Y base candidates after filter")
    valid_rows = [dict(r) for r in rows if "carrier_vec" in r]
    if not valid_rows:
        raise RuntimeError(f"clip {clip_id} has no dynamic rows with carrier_vec")
    Z = _carrier_tensor(valid_rows, device=device)
    text_idx = torch.tensor([int(data.raw_to_text_idx[int(rid)]) for rid in y_ids], device=device, dtype=torch.long)
    T = text_proj_all.index_select(0, text_idx)
    logits = torch.matmul(Z, T.t()) / _compute_t_dis(theta_t)

    # Critical point: assignment is recomputed from current logits every iteration.
    score_np = logits.detach().cpu().numpy().astype(np.float64)
    row_idx_np, col_idx_np, solver_name = _hungarian_maximize(score_np)
    if len(row_idx_np) == 0:
        raise RuntimeError(f"clip {clip_id} produced empty Hungarian assignment")
    row_idx = torch.tensor(row_idx_np, device=device, dtype=torch.long)
    col_idx = torch.tensor(col_idx_np, device=device, dtype=torch.long)
    assigned_logits = logits.index_select(0, row_idx)
    if str(loss_name) == "ce":
        loss = F.cross_entropy(assigned_logits, col_idx, reduction="mean")
    elif str(loss_name) == "infonce":
        pos = assigned_logits.gather(1, col_idx.view(-1, 1)).squeeze(1)
        loss = -(pos - torch.logsumexp(assigned_logits, dim=1)).mean()
    else:
        raise ValueError(f"unsupported dynamic loss: {loss_name}")

    with torch.no_grad():
        pred = torch.argmax(assigned_logits, dim=1)
        pseudo_top1 = float((pred == col_idx).float().mean().detach().cpu().item())
        # Margin between selected column and best non-selected column in that row.
        #
        # Important: clips with a single candidate column have no valid competitor.
        # Older diagnostics masked the selected column to -1e30 and then subtracted
        # it, producing sentinel-sized margins such as 1e29.  That value is not a
        # real scientific signal and must not enter epoch summaries.  We therefore
        # compute margin only when at least two columns exist; otherwise report a
        # finite neutral margin with explicit valid/skipped counts.
        selected = assigned_logits.gather(1, col_idx.view(-1, 1)).squeeze(1)
        if assigned_logits.shape[1] >= 2:
            masked = assigned_logits.clone()
            masked[torch.arange(masked.shape[0], device=device), col_idx] = -float("inf")
            best_other = torch.max(masked, dim=1).values
            margin = selected - best_other
            finite_margin = margin[torch.isfinite(margin)]
            if finite_margin.numel() > 0:
                margin_mean_value = float(finite_margin.mean().detach().cpu().item())
                margin_min_value = float(finite_margin.min().detach().cpu().item())
                margin_valid_pairs = int(finite_margin.numel())
            else:
                margin_mean_value = 0.0
                margin_min_value = 0.0
                margin_valid_pairs = 0
            margin_skipped_single_candidate_pairs = 0
        else:
            margin = torch.empty((0,), device=device, dtype=assigned_logits.dtype)
            margin_mean_value = 0.0
            margin_min_value = 0.0
            margin_valid_pairs = 0
            margin_skipped_single_candidate_pairs = int(len(row_idx_np))
        assigned_raw_ids = [int(y_ids[int(c)]) for c in col_idx.detach().cpu().tolist()]
        row_tids = [str(valid_rows[int(r)].get("trajectory_id", "")) for r in row_idx_np.tolist()]
        # raw_id 773/person is the known dominant hub in the current A8 audits; keep
        # this as a diagnostic only, not as a training rule.
        hub_count = sum(1 for rid in assigned_raw_ids if int(rid) == 773)
    stats = {
        "clip_id": int(clip_id),
        "dynamic_rows": int(len(valid_rows)),
        "dynamic_candidate_count": int(len(y_ids)),
        "dynamic_assigned_pairs": int(len(row_idx_np)),
        "dynamic_assignment_solver": solver_name,
        "dynamic_assignment_uses_current_logits": True,
        "dynamic_pseudo_top1_acc": pseudo_top1,
        "dynamic_selected_margin_mean": margin_mean_value,
        "dynamic_selected_margin_min": margin_min_value,
        "dynamic_selected_margin_valid_pairs": margin_valid_pairs,
        "dynamic_selected_margin_skipped_single_candidate_pairs": margin_skipped_single_candidate_pairs,
        "dynamic_person_raw773_assignment_rate": float(hub_count / max(len(assigned_raw_ids), 1)),
        "dynamic_loss": float(loss.detach().cpu().item()),
        "dynamic_assigned_raw_ids_head": assigned_raw_ids[:20],
        "dynamic_assigned_trajectory_ids_head": row_tids[:20],
    }
    return loss, stats


def _default_visible_csv(run_root: Path) -> Path:
    return run_root / "analysis" / "a8_base_116_visibility_audit" / "lvvis_train_base" / "base_641_visibility_by_class.csv"


def _annotation_default(repo_root: Path, dataset_name: str) -> Path:
    if "val" in str(dataset_name):
        return repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "val_instances.json"
    return repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json"


def _run_visible525_eval(
    *,
    repo_root: Path,
    asset_root: Path,
    run_root: Path,
    dataset_name: str,
    checkpoint_path: Path,
    output_root: Path,
    visible_csv: Path,
    annotation_json: Path,
    device: str,
    max_rows: int,
    show_progress: bool,
) -> Dict[str, Any]:
    tool = repo_root / "tools" / "a8_visible525_candidate_rankk_audit.py"
    out_dir = output_root / "analysis" / f"canonical_visible525_{dataset_name}"
    required = [
        tool,
        checkpoint_path,
        asset_root / "carrier_bank_gt" / str(dataset_name) / "carrier_records.jsonl",
        asset_root / "carrier_bank_gt" / str(dataset_name) / "gt_carrier_identity_binding.jsonl",
        asset_root / "exports_gt" / str(dataset_name) / "trajectory_records.jsonl",
        annotation_json,
        visible_csv,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("canonical visible525 eval missing required files: " + json.dumps(missing, ensure_ascii=False))
    cmd = [
        sys.executable,
        str(tool),
        "--dataset_name", str(dataset_name),
        "--output_root", str(out_dir),
        "--checkpoint_path", str(checkpoint_path),
        "--asset_root", str(asset_root),
        "--gt_carrier_path", str(asset_root / "carrier_bank_gt" / str(dataset_name) / "carrier_records.jsonl"),
        "--gt_identity_path", str(asset_root / "carrier_bank_gt" / str(dataset_name) / "gt_carrier_identity_binding.jsonl"),
        "--gt_trajectory_path", str(asset_root / "exports_gt" / str(dataset_name) / "trajectory_records.jsonl"),
        "--annotation_json", str(annotation_json),
        "--visible_csv", str(visible_csv),
        "--device", str(device),
        "--score_mode", "logit",
    ]
    if int(max_rows) > 0:
        cmd += ["--max_rows", str(int(max_rows))]
    if bool(show_progress):
        cmd.append("--show_progress")
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "visible525_eval.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"canonical visible525 eval failed with code {proc.returncode}; see {out_dir / 'visible525_eval.log'}")
    summary_path = out_dir / "visible525_candidate_rankk_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _write_takeover(out_root: Path, payload: Mapping[str, Any]) -> None:
    primary = payload.get("primary_metric", {}) if isinstance(payload.get("primary_metric", {}), Mapping) else {}
    setup = payload.get("setup", {}) if isinstance(payload.get("setup", {}), Mapping) else {}
    graph = setup.get("graph_preserve", {}) if isinstance(setup.get("graph_preserve", {}), Mapping) else {}
    lines = [
        "# A8 Joint Train-Time Dynamic Hungarian TAKEOVER",
        "",
        f"- status: {payload.get('status')}",
        f"- name: {payload.get('name')}",
        f"- checkpoint: {payload.get('checkpoint')}",
        "- training objective: lambda_prealign * current prealign bag loss + lambda_dynamic * train-time dynamic Hungarian loss",
        f"- graph preserve: mode={graph.get('mode', 'none')}, scope={graph.get('resolved_scope', graph.get('scope', ''))}, topK={graph.get('topk', '')}, lambda={setup.get('lambda_graph_preserve', 0.0)}",
        "- dynamic target source: current logits only; matched_raw_id is not used as target",
        "- primary metric: canonical visible525 rank@1",
        f"- primary value: {primary.get('value', '')}",
        "- legacy row_gap micro_top1: not emitted as headline",
        "",
        "## Non-goals enforced",
        "- no fixed matched-pair target CE",
        "- no GT-target CE",
        "- no visible525 CE",
        "- no rank-margin or hard-negative loss",
        "- no dummy/slack or extra support",
        "- no NoHub correctness labels",
        "- graph preserve does not use row-level GT or visual prototype targets and does not change Hungarian assignment",
    ]
    (out_root / "A8_JOINT_TRAIN_TIME_DYNAMIC_HUNGARIAN_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve() if str(args.output_root).strip() else _default_output_root(run_root, str(args.dataset_name), str(args.name))
    train_dir = out_root / "train" / "joint_train_time_dynamic_hungarian"
    train_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed)); np.random.seed(int(args.seed)); torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(str(args.device) if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    if not str(args.annotation_json).strip():
        args.annotation_json = str(_annotation_default(repo_root, str(args.dataset_name)))
    if not str(args.split_json).strip():
        args.split_json = str(repo_root / "package" / "reference" / "lvvis_official_base_novel_split.json")
    if not hasattr(args, "output_dir"):
        args.output_dir = ""

    data = _prepare_data(args)
    all_by_clip = _group_by_clip(data.examples)
    example_by_tid = _example_by_tid(data.examples)
    base_raw_ids = sorted(int(x) for x in data.base_ids if int(x) in data.raw_to_text_idx)
    if not base_raw_ids:
        raise RuntimeError("empty base vocabulary after text-bank filtering")
    base_col_by_raw_id = {int(rid): j for j, rid in enumerate(base_raw_ids)}
    base_text_indices = torch.tensor([int(data.raw_to_text_idx[int(rid)]) for rid in base_raw_ids], device=device, dtype=torch.long)
    text_tensor = torch.tensor(np.asarray(data.text_matrix, dtype=np.float32), device=device, dtype=torch.float32)

    clip_universe_csv = Path(args.clip_universe_csv).expanduser().resolve() if str(args.clip_universe_csv).strip() else _default_clip_universe_csv(run_root, str(args.dataset_name))
    clip_universe_by_clip, clip_universe_summary = _load_clip_universe(clip_universe_csv)
    if clip_universe_by_clip:
        clip_ids = sorted(int(cid) for cid in clip_universe_by_clip.keys() if int(cid) in all_by_clip)
        clip_universe_mode = "clip_universe_csv"
    else:
        clip_ids = sorted(int(cid) for cid in all_by_clip.keys())
        clip_universe_mode = "all_data_examples"
    if int(args.max_train_clips) > 0 and len(clip_ids) > int(args.max_train_clips):
        rng = random.Random(int(args.seed))
        clip_ids = sorted(rng.sample(clip_ids, int(args.max_train_clips)))
    if not clip_ids:
        raise RuntimeError("no train clips selected")

    projector = Projector(ProjectorConfig()).to(device)
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(args.t_dis_init) - 1.0e-4, 1.0e-6)), device=device, dtype=torch.float32))
    init = str(args.init_checkpoint).strip()
    ckpt = _auto_find_checkpoint(repo_root, str(args.dataset_name)) if init == "auto" else init
    checkpoint_summary = _load_checkpoint_if_requested(projector, theta_t, ckpt, device)

    graph_visible_csv = Path(args.graph_preserve_visible_csv).expanduser().resolve() if str(args.graph_preserve_visible_csv).strip() else _default_visible_csv(run_root)
    graph_cache = _build_graph_preserve_cache(
        data=data,
        text_tensor=text_tensor,
        base_raw_ids=base_raw_ids,
        device=device,
        mode=str(args.graph_preserve_mode),
        scope=str(args.graph_preserve_scope),
        topk=int(args.graph_preserve_topk),
        tau=float(args.graph_preserve_tau),
        visible_csv=graph_visible_csv,
        seed=int(args.seed),
    )

    setup = {
        "timestamp": _now(),
        "name": str(args.name),
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": str(args.trajectory_source_branch),
        "repo_root": str(repo_root),
        "asset_root": str(asset_root),
        "run_root": str(run_root),
        "output_root": str(out_root),
        "device": str(device),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "lambda_prealign": float(args.lambda_prealign),
        "lambda_dynamic": float(args.lambda_dynamic),
        "lambda_graph_preserve": float(args.lambda_graph_preserve),
        "graph_preserve": {
            k: v for k, v in graph_cache.items()
            if k not in {"scope_text_indices", "neighbor_idx", "target_prob"}
        },
        "dynamic_loss": str(args.dynamic_loss),
        "dynamic_row_source": str(args.dynamic_row_source),
        "dynamic_candidate_source": "full_y_base",
        "clip_universe_mode": clip_universe_mode,
        "clip_universe_summary": clip_universe_summary,
        "init_checkpoint": str(ckpt),
        "checkpoint_summary": checkpoint_summary,
        "materialization_summary": data.materialization_summary,
        "base_vocab_count": len(base_raw_ids),
        "policy": {
            "uses_current_prealign_loss": True,
            "prealign_loss_symbol": "run_stageb_train_hungarian_prealign._prealign_loss_for_clip",
            "uses_train_time_dynamic_hungarian": True,
            "uses_matched_raw_id_as_target": False,
            "uses_row_level_gt_for_training": False,
            "uses_visible525_ce_for_training": False,
            "uses_rank_margin_or_hard_negative": False,
            "uses_nohub_correctness_for_training": False,
            "uses_dummy_or_slack": False,
            "uses_extra_support": False,
            "uses_gt_upper_bound_trajectory": bool(str(args.trajectory_source_branch) == "gt_upper_bound"),
            "uses_videocutler_mainline_trajectory": bool(str(args.trajectory_source_branch) == "mainline"),
            "trajectory_source_branch_changes_only_carrier_source": True,
            "uses_local_text_graph_preserve_regularizer": bool(graph_cache.get("enabled", False)) and float(args.lambda_graph_preserve) > 0.0,
            "graph_preserve_uses_row_level_gt": False,
            "graph_preserve_uses_visual_prototype_target": False,
            "graph_preserve_changes_hungarian_assignment": False,
        },
    }
    _write_json(out_root / "setup.json", setup)

    optimizer = torch.optim.AdamW([*projector.parameters(), theta_t], lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    log_path = train_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    rng = random.Random(int(args.seed))
    global_step = 0
    all_total: List[float] = []
    all_pre: List[float] = []
    all_dyn: List[float] = []
    all_graph: List[float] = []
    epoch_rows: List[Dict[str, Any]] = []
    iterator = tqdm(range(int(args.epochs)), desc=f"a8_dyn_hun_{args.name}", dynamic_ncols=True) if bool(args.show_progress) and tqdm is not None else range(int(args.epochs))
    for epoch in iterator:
        cur_clip_ids = list(clip_ids)
        rng.shuffle(cur_clip_ids)
        ep_total: List[float] = []
        ep_pre: List[float] = []
        ep_dyn: List[float] = []
        ep_graph: List[float] = []
        ep_pre_rows = 0
        ep_dyn_rows = 0
        ep_pairs = 0
        ep_dyn_margin: List[float] = []
        ep_dyn_person: List[float] = []
        ep_pre_mass: List[float] = []
        ep_solver = Counter()
        for cid in cur_clip_ids:
            pre_group = [dict(x) for x in all_by_clip.get(int(cid), [])]
            dyn_rows = _select_clip_rows(
                clip_id=int(cid),
                all_by_clip=all_by_clip,
                example_by_tid=example_by_tid,
                clip_universe_by_clip=clip_universe_by_clip,
                row_source=str(args.dynamic_row_source),
            )
            y_ids = [int(x) for x in data.clip_y_base.get(int(cid), set()) if int(x) in base_col_by_raw_id]
            if not pre_group or not dyn_rows or not y_ids:
                continue
            optimizer.zero_grad(set_to_none=True)
            pre_loss, pre_stats = _prealign_loss_for_clip(
                clip_id=int(cid),
                group=pre_group,
                data=data,
                projector=projector,
                text_tensor=text_tensor,
                base_text_indices=base_text_indices,
                base_raw_ids=base_raw_ids,
                base_col_by_raw_id=base_col_by_raw_id,
                theta_t=theta_t,
                device=device,
                protocol="baseline_full_y",
                row_weight_gamma=float(args.row_weight_gamma),
                row_weight_conf_threshold=float(args.row_weight_conf_threshold),
                min_row_weight=float(args.min_row_weight),
            )
            text_proj_all = _project_text(projector, text_tensor)
            dyn_loss, dyn_stats = _dynamic_hungarian_loss_for_clip(
                clip_id=int(cid),
                rows=dyn_rows,
                data=data,
                text_proj_all=text_proj_all,
                theta_t=theta_t,
                device=device,
                loss_name=str(args.dynamic_loss),
            )
            if bool(graph_cache.get("enabled", False)) and float(args.lambda_graph_preserve) > 0.0 and int(args.graph_preserve_every_n_steps) > 0 and (global_step % int(args.graph_preserve_every_n_steps) == 0):
                graph_loss, graph_stats = _graph_preserve_loss(
                    text_proj_all=text_proj_all,
                    cache=graph_cache,
                    tau=float(args.graph_preserve_tau),
                )
            else:
                graph_loss, graph_stats = (text_proj_all.sum() * 0.0), {"graph_preserve_enabled": bool(graph_cache.get("enabled", False)), "graph_preserve_loss": 0.0, "graph_preserve_skipped_this_step": True}
            total_loss = float(args.lambda_prealign) * pre_loss + float(args.lambda_dynamic) * dyn_loss + float(args.lambda_graph_preserve) * graph_loss
            total_loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_([*projector.parameters(), theta_t], max_norm=float(args.grad_clip_norm))
            optimizer.step()
            global_step += 1
            tv = float(total_loss.detach().cpu().item())
            pv = float(pre_loss.detach().cpu().item())
            dv = float(dyn_loss.detach().cpu().item())
            gv = float(graph_loss.detach().cpu().item())
            all_total.append(tv); all_pre.append(pv); all_dyn.append(dv); all_graph.append(gv)
            ep_total.append(tv); ep_pre.append(pv); ep_dyn.append(dv); ep_graph.append(gv)
            ep_pre_rows += int(pre_stats.get("rows", 0)); ep_dyn_rows += int(dyn_stats.get("dynamic_rows", 0)); ep_pairs += int(dyn_stats.get("dynamic_assigned_pairs", 0))
            ep_dyn_margin.append(float(dyn_stats.get("dynamic_selected_margin_mean", 0.0)))
            ep_dyn_person.append(float(dyn_stats.get("dynamic_person_raw773_assignment_rate", 0.0)))
            ep_pre_mass.append(float(pre_stats.get("mass_in_y_mean", 0.0)))
            ep_solver[str(dyn_stats.get("dynamic_assignment_solver", ""))] += 1
            if int(args.log_every_steps) > 0 and global_step % int(args.log_every_steps) == 0:
                _append_jsonl(log_path, {"timestamp": _now(), "row_type": "step", "epoch": int(epoch) + 1, "global_step": global_step, "loss_total": tv, "loss_prealign": pv, "loss_dynamic": dv, "loss_graph_preserve": gv, **pre_stats, **dyn_stats, **graph_stats})
        epoch_row = {
            "timestamp": _now(),
            "row_type": "epoch_summary",
            "epoch": int(epoch) + 1,
            "global_step": int(global_step),
            "loss_total_mean": _mean(ep_total),
            "loss_prealign_mean": _mean(ep_pre),
            "loss_dynamic_mean": _mean(ep_dyn),
            "loss_graph_preserve_mean": _mean(ep_graph),
            "prealign_mass_in_y_mean": _mean(ep_pre_mass),
            "dynamic_selected_margin_mean": _mean(ep_dyn_margin),
            "dynamic_person_raw773_assignment_rate_mean": _mean(ep_dyn_person),
            "epoch_prealign_rows": int(ep_pre_rows),
            "epoch_dynamic_rows": int(ep_dyn_rows),
            "epoch_dynamic_assigned_pairs": int(ep_pairs),
            "epoch_clips": int(len(cur_clip_ids)),
            "dynamic_solver_counts": dict(ep_solver),
        }
        epoch_rows.append(epoch_row)
        _append_jsonl(log_path, epoch_row)
        if bool(args.print_epoch_summary):
            print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        if int(args.save_every_epochs) > 0 and ((int(epoch) + 1) % int(args.save_every_epochs) == 0):
            _save_checkpoint(train_dir / f"a8_joint_train_time_dynamic_epoch_{int(epoch)+1:03d}.pth", projector=projector, theta_t=theta_t, epoch=int(epoch)+1, global_step=global_step, payload=setup)

    ckpt_out = train_dir / "a8_joint_train_time_dynamic_last.pth"
    _save_checkpoint(ckpt_out, projector=projector, theta_t=theta_t, epoch=int(args.epochs), global_step=global_step, payload=setup)
    _write_csv(train_dir / "epoch_metrics.csv", epoch_rows)

    canonical_eval: Dict[str, Any] = {"status": "SKIPPED"}
    primary_metric: Dict[str, Any] = {"name": "canonical_visible525_rank@1", "value": None, "status": "SKIPPED"}
    if not bool(args.skip_canonical_visible525_eval):
        visible_csv = Path(args.canonical_visible525_csv).expanduser().resolve() if str(args.canonical_visible525_csv).strip() else _default_visible_csv(run_root)
        canonical_dataset = str(args.canonical_eval_dataset_name or args.dataset_name)
        annotation_json = Path(args.canonical_eval_annotation_json).expanduser().resolve() if str(args.canonical_eval_annotation_json).strip() else _annotation_default(repo_root, canonical_dataset)
        canonical_eval = _run_visible525_eval(
            repo_root=repo_root,
            asset_root=asset_root,
            run_root=run_root,
            dataset_name=canonical_dataset,
            checkpoint_path=ckpt_out,
            output_root=out_root,
            visible_csv=visible_csv,
            annotation_json=annotation_json,
            device=str(args.device),
            max_rows=int(args.canonical_eval_max_rows),
            show_progress=bool(args.show_progress),
        )
        primary_metric = dict(canonical_eval.get("primary_metric", {}))
        primary_metric.setdefault("status", canonical_eval.get("status"))

    final = {
        "status": "PASS",
        "timestamp": _now(),
        "name": str(args.name),
        "definition": "joint current prealign bag loss + train-time dynamic Hungarian over current logits",
        "output_root": str(out_root),
        "checkpoint": str(ckpt_out),
        "primary_metric": primary_metric,
        "canonical_visible525_eval": canonical_eval,
        "train_summary": {
            "epochs": int(args.epochs),
            "global_step": int(global_step),
            "clip_count": int(len(clip_ids)),
            "loss_total_mean": _mean(all_total),
            "loss_prealign_mean": _mean(all_pre),
            "loss_dynamic_mean": _mean(all_dyn),
            "loss_graph_preserve_mean": _mean(all_graph),
            "loss_total_last": all_total[-1] if all_total else 0.0,
        },
        "setup": setup,
        "headline_policy": "Use canonical visible525 rank@K only. Do not use retired row_gap micro_top1 as primary metric.",
    }
    _write_json(out_root / "final_summary.json", final)
    _write_takeover(out_root, final)
    print(json.dumps(final, ensure_ascii=False, indent=2, default=str), flush=True)
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A8 joint prealign + train-time dynamic Hungarian training")
    p.add_argument("--run_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--trajectory_source_branch", choices=["gt_upper_bound", "mainline"], default="gt_upper_bound", help="Trajectory/carrier source: gt_upper_bound uses exports_gt/carrier_bank_gt; mainline uses VideoCutLER exports/carrier_bank.")
    p.add_argument("--name", default="D-J1_pre1_dyn1_ep5")
    p.add_argument("--output_root", default="")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--output_dir", default="")
    p.add_argument("--clip_universe_csv", default="", help="Optional old matched-pairs CSV used only for clip/row universe; matched_raw_id is ignored.")
    p.add_argument("--dynamic_row_source", choices=["all_clip_trajectories", "clip_universe_rows"], default="all_clip_trajectories")
    p.add_argument("--dynamic_loss", choices=["ce", "infonce"], default="ce")
    p.add_argument("--lambda_prealign", type=float, default=1.0)
    p.add_argument("--lambda_dynamic", type=float, default=1.0)
    p.add_argument("--lambda_graph_preserve", type=float, default=0.0)
    p.add_argument("--graph_preserve_mode", choices=["none", "raw_text_topk", "random_text_topk"], default="none")
    p.add_argument("--graph_preserve_scope", choices=["visible525", "base_vocab"], default="visible525")
    p.add_argument("--graph_preserve_topk", type=int, default=20)
    p.add_argument("--graph_preserve_tau", type=float, default=0.1)
    p.add_argument("--graph_preserve_every_n_steps", type=int, default=1)
    p.add_argument("--graph_preserve_visible_csv", default="", help="Optional base_641_visibility_by_class.csv; used when graph_preserve_scope=visible525.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--max_train_clips", type=int, default=0)
    p.add_argument("--init_checkpoint", default="", help="empty: train from fresh projector; auto: load current auto checkpoint intentionally")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--t_dis_init", type=float, default=0.07)
    p.add_argument("--row_weight_gamma", type=float, default=8.0)
    p.add_argument("--row_weight_conf_threshold", type=float, default=0.5)
    p.add_argument("--min_row_weight", type=float, default=0.25)
    p.add_argument("--canonical_eval_dataset_name", default="")
    p.add_argument("--canonical_eval_annotation_json", default="")
    p.add_argument("--canonical_visible525_csv", default="")
    p.add_argument("--canonical_eval_max_rows", type=int, default=0)
    p.add_argument("--skip_canonical_visible525_eval", action="store_true")
    p.add_argument("--show_progress", action="store_true", default=True)
    p.add_argument("--print_epoch_summary", action="store_true", default=True)
    p.add_argument("--log_every_steps", type=int, default=200)
    p.add_argument("--save_every_epochs", type=int, default=0)
    return p.parse_args()


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
