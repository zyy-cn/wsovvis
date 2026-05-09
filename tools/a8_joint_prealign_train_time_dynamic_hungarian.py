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
import hashlib
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
from videocutler.ext_stageb_ovvis.banks.carrier_bank import (  # noqa: E402
    _coerce_token_feature_matrix,
    _decode_mask_rle,
    _mask_to_token_weights,
    _read_feature_vector_cached,
    _resize_pad_mask,
)
from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import (  # noqa: E402
    reconstruct_valid_token_mask_from_geometry,
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

    # Preserve the original trajectory record for optional train-side
    # patch-token carrier augmentation.  The canonical prealign preparation only
    # needs carrier_vec and therefore drops masks/frame indices; patch sampling
    # must recover them from the phase-1 materialized samples without changing
    # the weak-label / Hungarian protocol.
    sample_by_tid: Dict[str, Mapping[str, Any]] = {str(s.get("trajectory_id", "")): s for s in samples if str(s.get("trajectory_id", ""))}
    restored_trajectory_records = 0
    for ex in examples:
        src = sample_by_tid.get(str(ex.get("trajectory_id", "")))
        if src is None:
            continue
        if isinstance(src.get("trajectory_record"), Mapping):
            ex["trajectory_record"] = dict(src["trajectory_record"])
            restored_trajectory_records += 1
        ex["trajectory_source_branch"] = str(branch)

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
        "restored_trajectory_record_count_for_patchsample_aug": int(restored_trajectory_records),
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


def _sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_npz_first_array(path: Path) -> Tuple[np.ndarray, str]:
    obj = np.load(path)
    keys = list(obj.keys())
    if not keys:
        raise RuntimeError(f"empty npz payload: {path}")
    for key in ("protos", "features", "arr_0", "llama_hidden_mean", "clip_of_llm_mean", "llama_direct_concept_mean"):
        if key in obj:
            return np.asarray(obj[key]), str(key)
    key0 = str(keys[0])
    return np.asarray(obj[key0]), key0


def _load_lvvis_text_bank_classes(bank_root: Path) -> Tuple[List[int], Dict[int, str], Dict[str, Any]]:
    class_path = bank_root / "lvvis_class_names.json"
    if not class_path.is_file():
        raise FileNotFoundError(f"missing lvvis_class_names.json under text bank root: {class_path}")
    payload = json.loads(class_path.read_text(encoding="utf-8"))
    rows = payload.get("classes", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list):
        raise RuntimeError(f"invalid lvvis_class_names.json schema: {class_path}")
    ids: List[int] = []
    names: Dict[int, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        rid = _as_int(row.get("raw_id"))
        if rid is None:
            continue
        name = str(row.get("name", row.get("class_name", rid)))
        ids.append(int(rid))
        names[int(rid)] = name
    if not ids:
        raise RuntimeError(f"no class ids found in {class_path}")
    if ids != sorted(ids):
        raise RuntimeError(f"text bank raw ids are not ascending: {class_path}")
    return ids, names, payload if isinstance(payload, dict) else {"classes": rows}


def _resolve_text_bank_payload_path(bank_root: Path, variant: str) -> Path:
    payload = bank_root / "payload"
    candidates = {
        "clip_of_llm_mean": payload / "clip_of_llm_mean.fp16.npz",
        "llama_hidden_mean": payload / "llama_hidden_mean.fp16.npz",
        "llama_direct_concept_mean": payload / "llama_direct_concept_mean.fp16.npz",
    }
    if variant not in candidates:
        raise ValueError(f"unsupported external text_bank_variant={variant!r}; expected one of {sorted(candidates)}")
    path = candidates[variant]
    if not path.is_file():
        raise FileNotFoundError(f"missing text bank payload for {variant}: {path}")
    return path


def _resolve_training_text_bank(args: argparse.Namespace, data: Any) -> Dict[str, Any]:
    """Resolve the text feature matrix used as train-time class anchors.

    clip_current preserves the historical CLIP text bank from load_text_vocab().
    Other variants load canonical LV-VIS text-bank assets under wsovvis_asserts
    and replace only the text anchor matrix / raw-id mapping.  The clip-wise
    Hungarian protocol, candidate set, trajectory source, and losses are not
    changed by this function.
    """
    variant = str(getattr(args, "text_bank_variant", "clip_current")).strip() or "clip_current"
    if variant == "clip_current":
        mat = np.asarray(data.text_matrix, dtype=np.float32)
        inferred_dim = int(mat.shape[1]) if mat.ndim == 2 else 0
        requested_dim = int(getattr(args, "text_feature_dim", 0) or 0)
        if requested_dim and requested_dim != inferred_dim:
            raise RuntimeError(f"--text_feature_dim={requested_dim} does not match clip_current dim={inferred_dim}")
        return {
            "status": "PASS",
            "variant": "clip_current",
            "root": "canonical_asset_text_bank",
            "feature_dim": int(inferred_dim),
            "projector_input_dim": int(inferred_dim),
            "class_count": int(len(data.text_ids)),
            "raw_id_order": "from_canonical_text_bank",
            "payload_path": "",
            "payload_sha256": "",
            "manifest_path": "",
            "manifest_sha256": "",
            "records_path": "",
            "replaces_only_text_anchor_source": True,
            "same_dynamic_hungarian_protocol": True,
            "same_clip_wise_training_protocol": True,
            "same_candidate_set": True,
        }

    bank_root_arg = str(getattr(args, "text_bank_root", "")).strip()
    if not bank_root_arg:
        raise RuntimeError(f"--text_bank_root is required when --text_bank_variant={variant}")
    bank_root = Path(bank_root_arg).expanduser().resolve()
    manifest_path = bank_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing text bank manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("status", "")).upper() != "PASS":
        raise RuntimeError(f"text bank manifest status is not PASS: {manifest_path}")
    ids, names, classes_payload = _load_lvvis_text_bank_classes(bank_root)
    payload_path = _resolve_text_bank_payload_path(bank_root, variant)
    arr, arr_key = _load_npz_first_array(payload_path)
    if arr.ndim != 2:
        raise RuntimeError(f"text bank payload must be 2D [class_count, dim], got shape={arr.shape} from {payload_path}")
    if int(arr.shape[0]) != len(ids):
        raise RuntimeError(f"text bank class count mismatch: payload rows={arr.shape[0]} class ids={len(ids)}")
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"non-finite values in text bank payload: {payload_path}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-12)
    feature_dim = int(arr.shape[1])
    requested_dim = int(getattr(args, "text_feature_dim", 0) or 0)
    if requested_dim and requested_dim != feature_dim:
        raise RuntimeError(f"--text_feature_dim={requested_dim} does not match loaded feature_dim={feature_dim} from {payload_path}")

    data.text_ids = [int(x) for x in ids]
    data.raw_to_text_idx = {int(rid): idx for idx, rid in enumerate(ids)}
    data.text_matrix = arr
    merged_names = dict(getattr(data, "class_names", {}) or {})
    merged_names.update({int(k): str(v) for k, v in names.items()})
    data.class_names = merged_names

    return {
        "status": "PASS",
        "variant": variant,
        "root": str(bank_root),
        "feature_dim": int(feature_dim),
        "projector_input_dim": int(feature_dim),
        "class_count": int(len(ids)),
        "raw_id_order": str((classes_payload or {}).get("raw_id_order", "ascending")),
        "payload_path": str(payload_path),
        "payload_array_key": str(arr_key),
        "payload_sha256": _sha256_file(payload_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_profile_id": str(manifest.get("profile_id", "")),
        "manifest_profile_type": str(manifest.get("profile_type", "")),
        "uses_old_corr_feats": bool(manifest.get("uses_old_corr_feats", False)),
        "token_feature_alignment": str(manifest.get("token_feature_alignment", "")),
        "all_vectors_finite": bool(manifest.get("all_vectors_finite", True)),
        "all_mean_vectors_l2_normalized": bool(manifest.get("all_mean_vectors_l2_normalized", True)),
        "records_path": str(bank_root / "records" / f"{variant}_text_prototype_records.jsonl"),
        "replaces_only_text_anchor_source": True,
        "same_dynamic_hungarian_protocol": True,
        "same_clip_wise_training_protocol": True,
        "same_candidate_set": True,
    }


def _save_checkpoint(
    path: Path,
    *,
    projector: Projector,
    theta_t: torch.nn.Parameter,
    epoch: int,
    global_step: int,
    payload: Mapping[str, Any],
    text_projector_config: Optional[Mapping[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text_projector_state_dict": projector.state_dict(),
            "text_projector_config": dict(text_projector_config or {
                "input_dim": int(getattr(getattr(projector, "config", None), "input_dim", 512)),
                "hidden_dim": int(getattr(getattr(projector, "config", None), "hidden_dim", 1024)),
                "output_dim": int(getattr(getattr(projector, "config", None), "output_dim", 768)),
                "dropout": float(getattr(getattr(projector, "config", None), "dropout", 0.0)),
                "use_layernorm": bool(getattr(getattr(projector, "config", None), "use_layernorm", True)),
                "projector_type": str(getattr(getattr(projector, "config", None), "projector_type", "mlp") or "mlp"),
            }),
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
    row-level GT, dynamic Hungarian assignments, or visual prototype targets.

    Supported modes:
      * raw_text_topk: legacy KL local-neighborhood preservation
        KL(softmax(S_raw[i, N_i]/tau) || softmax(S_proj[i, N_i]/tau)).
      * topk_local / topk_local_mse / raw_text_topk_mse: local edge cosine MSE
        mean_j (S_proj[i, j] - S_raw[i, j])^2 over raw-text topK neighbors.
      * random_*: same targets on randomized neighborhoods, for a control.

    The MSE mode is the intended A8 Llama-hidden + linear projector ablation: it
    directly penalizes distortion of the useful raw-text local manifold without
    changing dynamic Hungarian, the candidate set, or trajectory source.
    """
    mode = str(mode).strip().lower()
    if mode in {"", "none", "off", "false", "0"}:
        return {"enabled": False, "mode": "none", "reason": "graph_preserve_mode_disabled"}

    aliases = {
        "topk_local": "topk_local_mse",
        "raw_text_topk_mse": "topk_local_mse",
        "random_topk_local": "random_topk_local_mse",
        "random_text_topk_mse": "random_topk_local_mse",
    }
    mode = aliases.get(mode, mode)
    supported_modes = {"raw_text_topk", "random_text_topk", "topk_local_mse", "random_topk_local_mse"}
    if mode not in supported_modes:
        raise ValueError(f"unsupported graph_preserve_mode={mode!r}; expected one of {sorted(supported_modes)}")
    loss_type = "edge_mse" if mode in {"topk_local_mse", "random_topk_local_mse"} else "kl"
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
        if mode in {"raw_text_topk", "topk_local_mse"}:
            neighbor_idx = torch.topk(sim, k=int(k), dim=1, largest=True, sorted=True).indices
        else:
            rng = np.random.default_rng(int(seed))
            neigh_rows: List[np.ndarray] = []
            all_idx = np.arange(len(raw_ids), dtype=np.int64)
            for i in range(len(raw_ids)):
                choices = all_idx[all_idx != i]
                neigh_rows.append(rng.choice(choices, size=int(k), replace=False))
            neighbor_idx = torch.tensor(np.stack(neigh_rows, axis=0), device=device, dtype=torch.long)
        raw_neighbor_sim = torch.gather(sim, 1, neighbor_idx).detach()
        # Random neighbors may include mostly low-similarity pairs; keep the real
        # raw-sim target distribution/edge values over the chosen neighbors for a
        # fair control.
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
        "loss_type": loss_type,
        "raw_ids_head": [int(x) for x in raw_ids[:20]],
        "scope_text_indices": torch.tensor(text_indices_np, device=device, dtype=torch.long),
        "neighbor_idx": neighbor_idx,
        "target_prob": target_prob,
        "raw_neighbor_sim": raw_neighbor_sim,
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
    raw_neighbor_sim = cache.get("raw_neighbor_sim")
    loss_type = str(cache.get("loss_type", "kl")).strip().lower()
    P = F.normalize(text_proj_all.index_select(0, scope_text_indices), p=2.0, dim=-1)
    sim_proj = torch.matmul(P, P.t())
    proj_neighbor_sim = torch.gather(sim_proj, 1, neighbor_idx)
    raw_sim_t = raw_neighbor_sim.to(device=proj_neighbor_sim.device, dtype=proj_neighbor_sim.dtype) if torch.is_tensor(raw_neighbor_sim) else None
    if loss_type == "edge_mse":
        if raw_sim_t is None:
            raise RuntimeError("graph preserve edge_mse requires raw_neighbor_sim in cache")
        edge_mse = F.mse_loss(proj_neighbor_sim, raw_sim_t)
        loss = edge_mse
        cross_entropy = proj_neighbor_sim.sum() * 0.0
        target_entropy = proj_neighbor_sim.sum() * 0.0
    else:
        log_prob = torch.log_softmax(proj_neighbor_sim / max(float(tau), 1.0e-6), dim=1)
        tgt = target_prob.to(device=log_prob.device, dtype=log_prob.dtype)
        cross_entropy = -(tgt * log_prob).sum(dim=1).mean()
        target_entropy = -(tgt * torch.log(torch.clamp(tgt, min=1.0e-12))).sum(dim=1).mean().detach()
        loss = cross_entropy - target_entropy
        edge_mse = F.mse_loss(proj_neighbor_sim, raw_sim_t) if raw_sim_t is not None else proj_neighbor_sim.sum() * 0.0
    with torch.no_grad():
        abs_err = torch.mean(torch.abs(proj_neighbor_sim - raw_sim_t)) if raw_sim_t is not None else proj_neighbor_sim.sum() * 0.0
        stats = {
            "graph_preserve_enabled": True,
            "graph_preserve_mode": str(cache.get("mode", "")),
            "graph_preserve_loss_type": str(loss_type),
            "graph_preserve_scope": str(cache.get("resolved_scope", cache.get("scope", ""))),
            "graph_preserve_class_count": int(cache.get("class_count", 0)),
            "graph_preserve_topk": int(cache.get("topk", 0)),
            "graph_preserve_loss": float(loss.detach().cpu().item()),
            "graph_preserve_edge_mse": float(edge_mse.detach().cpu().item()),
            "graph_preserve_abs_err_mean": float(abs_err.detach().cpu().item()),
            "graph_preserve_cross_entropy": float(cross_entropy.detach().cpu().item()),
            "graph_preserve_target_entropy": float(target_entropy.detach().cpu().item()),
            "graph_preserve_proj_neighbor_sim_mean": float(proj_neighbor_sim.detach().mean().cpu().item()),
            "graph_preserve_raw_neighbor_sim_mean": float(raw_sim_t.detach().mean().cpu().item()) if raw_sim_t is not None else 0.0,
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


def _load_frame_maps(frame_bank_dir: Path) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Any]]]:
    frame_records_path = frame_bank_dir / "frame_records.jsonl"
    geom_records_path = frame_bank_dir / "frame_geom_records.jsonl"
    if not frame_records_path.is_file():
        raise FileNotFoundError(frame_records_path)
    if not geom_records_path.is_file():
        raise FileNotFoundError(geom_records_path)
    frame_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    geom_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in _iter_jsonl(frame_records_path):
        frame_map[(str(row.get("clip_id")), int(row.get("frame_index")))] = dict(row)
    for row in _iter_jsonl(geom_records_path):
        geom_map[(str(row.get("clip_id")), int(row.get("frame_index")))] = dict(row)
    return frame_map, geom_map


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield dict(obj)


def _normalize_np_vec(vec: np.ndarray, eps: float = 1.0e-12) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= eps or not np.isfinite(norm):
        return None
    return (arr / norm).astype(np.float32)


def _frame_patch_candidates_for_aug(
    *,
    frame_bank_dir: Path,
    frame_record: Mapping[str, Any],
    geom_record: Mapping[str, Any],
    mask_item: Any,
    image_size: Sequence[int],
    payload_cache: Dict[Path, np.lib.npyio.NpzFile],
    min_token_weight: float,
    max_frame_candidate_tokens: int,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    feature = _read_feature_vector_cached(frame_bank_dir, str(frame_record["feat_path"]), payload_cache)
    grid_h = int(geom_record["grid_h"])
    grid_w = int(geom_record["grid_w"])
    patch_size = int(geom_record["patch_size"])
    token_matrix = _coerce_token_feature_matrix(feature, grid_h, grid_w)
    if token_matrix is None:
        return None
    valid_mask = reconstruct_valid_token_mask_from_geometry(geom_record).astype(np.float32)
    decoded_mask = _decode_mask_rle(mask_item, image_size)
    projected_mask = _resize_pad_mask(
        decoded_mask,
        resized_h=int(geom_record["resized_h"]),
        resized_w=int(geom_record["resized_w"]),
        padded_h=int(geom_record["padded_h"]),
        padded_w=int(geom_record["padded_w"]),
    )
    weights = _mask_to_token_weights(projected_mask, patch_size, grid_h, grid_w) * valid_mask
    flat = weights.reshape(-1).astype(np.float64)
    valid_idx = np.flatnonzero(flat > float(min_token_weight)) if float(min_token_weight) > 0 else np.flatnonzero(flat > 0)
    if int(valid_idx.size) <= 0:
        return None
    cand_weights = flat[valid_idx].astype(np.float64)
    denom = float(np.sum(cand_weights))
    if denom <= 1.0e-12 or not np.isfinite(denom):
        return None
    cand_weights = cand_weights / denom
    if int(max_frame_candidate_tokens or 0) > 0 and int(valid_idx.size) > int(max_frame_candidate_tokens):
        cap = int(max_frame_candidate_tokens)
        chosen_pos = rng.choice(int(valid_idx.size), size=cap, replace=False, p=cand_weights)
        valid_idx = valid_idx[chosen_pos]
        cand_weights = cand_weights[chosen_pos]
        cand_weights = cand_weights / max(1.0e-12, float(np.sum(cand_weights)))
    cand_tokens = np.asarray(token_matrix[valid_idx], dtype=np.float32)
    return cand_tokens, cand_weights.astype(np.float64), denom


def _sample_patch_carrier_for_row(
    *,
    row: Mapping[str, Any],
    frame_bank_dir: Path,
    frame_map: Mapping[Tuple[str, int], Mapping[str, Any]],
    geom_map: Mapping[Tuple[str, int], Mapping[str, Any]],
    payload_cache: Dict[Path, np.lib.npyio.NpzFile],
    tokens_per_view: int,
    min_token_weight: float,
    max_frame_candidate_tokens: int,
    seed: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    traj = row.get("trajectory_record") if isinstance(row.get("trajectory_record"), Mapping) else {}
    trajectory_id = str(row.get("trajectory_id", traj.get("trajectory_id", "")))
    clip_id = str(row.get("clip_id", traj.get("clip_id", row.get("video_id", traj.get("video_id", "")))))
    frame_indices = [int(x) for x in list(traj.get("frame_indices", []))]
    masks_rle = list(traj.get("masks_rle", []))
    image_size = list(traj.get("image_size", []))
    if len(image_size) != 2:
        for frame_index in frame_indices:
            geom = geom_map.get((clip_id, int(frame_index)))
            if geom is not None:
                image_size = [int(geom["orig_h"]), int(geom["orig_w"])]
                break
    if not trajectory_id or not frame_indices or len(frame_indices) != len(masks_rle) or len(image_size) != 2:
        return None, {"status": "SKIP", "reason": "malformed_trajectory_record"}

    rng = np.random.default_rng(int(seed))
    frame_candidates: List[Tuple[np.ndarray, np.ndarray, float, int]] = []
    counters: Counter = Counter()
    for frame_index, mask_item in zip(frame_indices, masks_rle):
        key = (clip_id, int(frame_index))
        frame_record = frame_map.get(key)
        geom_record = geom_map.get(key)
        if frame_record is None:
            counters["missing_frame_record"] += 1
            continue
        if geom_record is None:
            counters["missing_frame_geom_record"] += 1
            continue
        try:
            cand = _frame_patch_candidates_for_aug(
                frame_bank_dir=frame_bank_dir,
                frame_record=frame_record,
                geom_record=geom_record,
                mask_item=mask_item,
                image_size=image_size,
                payload_cache=payload_cache,
                min_token_weight=float(min_token_weight),
                max_frame_candidate_tokens=int(max_frame_candidate_tokens or 0),
                rng=rng,
            )
        except Exception:
            counters["frame_candidate_failed"] += 1
            continue
        if cand is None:
            counters["empty_token_occupancy"] += 1
            continue
        tokens, weights, denom = cand
        frame_candidates.append((tokens, weights, float(denom), int(frame_index)))
    if not frame_candidates:
        return None, {"status": "SKIP", "reason": "no_valid_frames", "counters": dict(counters)}

    frame_mass = np.asarray([x[2] for x in frame_candidates], dtype=np.float64)
    frame_prob = frame_mass / float(np.sum(frame_mass)) if float(np.sum(frame_mass)) > 1.0e-12 else np.full((len(frame_candidates),), 1.0 / float(len(frame_candidates)), dtype=np.float64)
    frame_draws = rng.choice(len(frame_candidates), size=int(tokens_per_view), replace=True, p=frame_prob)
    pieces: List[np.ndarray] = []
    for fidx in np.unique(frame_draws):
        count = int(np.sum(frame_draws == fidx))
        tokens, weights, _denom, _frame_index = frame_candidates[int(fidx)]
        if int(tokens.shape[0]) <= 0 or count <= 0:
            continue
        chosen = rng.choice(int(tokens.shape[0]), size=count, replace=int(tokens.shape[0]) < count, p=weights)
        pieces.append(np.asarray(tokens[chosen], dtype=np.float32))
    if not pieces:
        return None, {"status": "SKIP", "reason": "no_valid_sampled_tokens", "counters": dict(counters)}
    sampled = np.concatenate(pieces, axis=0).astype(np.float32)
    proto = _normalize_np_vec(np.mean(sampled, axis=0))
    if proto is None:
        return None, {"status": "SKIP", "reason": "zero_norm_sampled_carrier", "counters": dict(counters)}
    return proto.astype(np.float32), {
        "status": "PASS",
        "valid_frame_count": int(len(frame_candidates)),
        "tokens_per_view": int(sampled.shape[0]),
        "token_candidate_count_sum": int(sum(int(x[0].shape[0]) for x in frame_candidates)),
        "token_candidate_count_mean_per_frame": float(np.mean([int(x[0].shape[0]) for x in frame_candidates])),
        "counters": dict(counters),
    }


def _patchsample_aug_seed(global_seed: int, epoch: int, clip_id: int, trajectory_id: str) -> int:
    # Stable 32-bit seed without relying on Python's salted hash().
    import hashlib
    payload = f"{int(global_seed)}|{int(epoch)}|{int(clip_id)}|{trajectory_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)




def _load_patchsample_cache(cache_dir: Path) -> Dict[str, Any]:
    """Load an offline patchsample carrier cache without materializing vectors.

    Expected files:
      * patchsample_manifest.json
      * patchsample_index.json
      * patchsample_vectors.fp16.mmap  (shape from manifest)
      * patchsample_valid.npy          (bool, mmap readable)

    The cache contains per-epoch, per-trajectory sampled carriers.  Training
    still gates replacement by patchsample_prob and preserves the original
    clip-wise Hungarian protocol; this cache only replaces expensive online
    frame_bank sampling with a deterministic table lookup.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "patchsample_manifest.json"
    index_path = cache_dir / "patchsample_index.json"
    vectors_path = cache_dir / "patchsample_vectors.fp16.mmap"
    valid_path = cache_dir / "patchsample_valid.npy"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not vectors_path.is_file():
        raise FileNotFoundError(vectors_path)
    if not valid_path.is_file():
        raise FileNotFoundError(valid_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    trajectory_ids = [str(x) for x in index.get("trajectory_ids", [])]
    shape = tuple(int(x) for x in manifest.get("shape", []))
    if len(shape) != 3:
        raise RuntimeError(f"invalid patchsample cache shape in {manifest_path}: {shape!r}")
    if shape[1] != len(trajectory_ids):
        raise RuntimeError(f"patchsample cache index/shape mismatch: shape={shape}, trajectory_ids={len(trajectory_ids)}")
    vec = np.memmap(vectors_path, mode="r", dtype=np.float16, shape=shape)
    valid = np.load(valid_path, mmap_mode="r")
    if tuple(int(x) for x in valid.shape) != shape[:2]:
        raise RuntimeError(f"patchsample valid shape mismatch: valid={valid.shape}, expected={shape[:2]}")
    return {
        "status": "PASS",
        "cache_dir": str(cache_dir),
        "manifest": manifest,
        "trajectory_ids": trajectory_ids,
        "index_by_trajectory_id": {tid: i for i, tid in enumerate(trajectory_ids)},
        "vectors": vec,
        "valid": valid,
        "shape": shape,
        "epochs": int(shape[0]),
        "trajectory_count": int(shape[1]),
        "dim": int(shape[2]),
        "vectors_path": str(vectors_path),
        "valid_path": str(valid_path),
    }

def _maybe_patchsample_augment_clip_rows(
    *,
    clip_id: int,
    pre_group: Sequence[Mapping[str, Any]],
    dyn_rows: Sequence[Mapping[str, Any]],
    epoch: int,
    args: argparse.Namespace,
    frame_bank_dir: Optional[Path],
    frame_map: Optional[Mapping[Tuple[str, int], Mapping[str, Any]]],
    geom_map: Optional[Mapping[Tuple[str, int], Mapping[str, Any]]],
    patchsample_cache: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], Dict[str, Any]]:
    mode = str(getattr(args, "train_carrier_aug_mode", "none")).strip().lower()
    prob = float(getattr(args, "patchsample_prob", 0.0))
    if mode in {"", "none", "off", "false", "0"} or prob <= 0.0:
        return [dict(x) for x in pre_group], [dict(x) for x in dyn_rows], {"train_carrier_aug_enabled": False, "train_carrier_aug_mode": "none"}
    if mode not in {"patchsample_mixed", "patchsample_cached_mixed"}:
        raise ValueError(f"unsupported train_carrier_aug_mode={mode!r}")

    patch_prob = min(max(prob, 0.0), 1.0)
    token_k = int(getattr(args, "patchsample_tokens_per_view", 64))
    min_weight = float(getattr(args, "patchsample_min_token_weight", 0.0))
    max_cand = int(getattr(args, "patchsample_max_frame_candidate_tokens", 4096))
    base_seed = int(getattr(args, "patchsample_seed", getattr(args, "seed", 0)))

    replacements: Dict[str, np.ndarray] = {}
    counters: Counter = Counter()
    candidate_means: List[float] = []

    rows_by_tid: Dict[str, Mapping[str, Any]] = {}
    for r in list(pre_group) + list(dyn_rows):
        tid = str(r.get("trajectory_id", ""))
        if tid and tid not in rows_by_tid:
            rows_by_tid[tid] = r

    if mode == "patchsample_cached_mixed":
        if patchsample_cache is None or not bool(patchsample_cache.get("status") == "PASS"):
            raise RuntimeError("patchsample_cached_mixed requested but patchsample cache is not loaded")
        cache_epochs = int(patchsample_cache.get("epochs", 0))
        if int(epoch) < 1 or int(epoch) > cache_epochs:
            raise RuntimeError(f"patchsample cache has {cache_epochs} epochs but training requested epoch={epoch}; build a larger cache or reduce --epochs")
        idx_by_tid = patchsample_cache["index_by_trajectory_id"]
        vectors = patchsample_cache["vectors"]
        valid = patchsample_cache["valid"]
        eidx = int(epoch) - 1
        for tid, row in rows_by_tid.items():
            gate_seed = _patchsample_aug_seed(base_seed, int(epoch), int(clip_id), str(tid))
            gate_rng = np.random.default_rng(gate_seed)
            if float(gate_rng.random()) >= patch_prob:
                counters["kept_mean_by_probability"] += 1
                continue
            counters["patchsample_attempted"] += 1
            tidx = idx_by_tid.get(str(tid))
            if tidx is None:
                counters["patchsample_fallback_to_mean"] += 1
                counters["patchsample_skip_missing_cache_trajectory"] += 1
                continue
            if not bool(valid[eidx, int(tidx)]):
                counters["patchsample_fallback_to_mean"] += 1
                counters["patchsample_skip_invalid_cache_vector"] += 1
                continue
            vec = _normalize_np_vec(np.asarray(vectors[eidx, int(tidx)], dtype=np.float32))
            if vec is None:
                counters["patchsample_fallback_to_mean"] += 1
                counters["patchsample_skip_zero_norm_cache_vector"] += 1
                continue
            replacements[str(tid)] = np.asarray(vec, dtype=np.float32)
            counters["patchsample_used"] += 1
        payload_cache_scope = "offline_cache_lookup"
    else:
        if frame_bank_dir is None or frame_map is None or geom_map is None:
            raise RuntimeError("patchsample_mixed requested but frame_bank maps are not loaded")
        # One clip-local payload cache preserves the original clip-wise protocol
        # while avoiding repeated opens of the same frame_bank .npz payload for
        # trajectories within the clip. It is deliberately released before the
        # next clip. This online path is retained for smoke/debug only; full
        # runs should use patchsample_cached_mixed.
        payload_cache: Dict[Path, np.lib.npyio.NpzFile] = {}
        try:
            for tid, row in rows_by_tid.items():
                gate_seed = _patchsample_aug_seed(base_seed, int(epoch), int(clip_id), str(tid))
                gate_rng = np.random.default_rng(gate_seed)
                if float(gate_rng.random()) >= patch_prob:
                    counters["kept_mean_by_probability"] += 1
                    continue
                counters["patchsample_attempted"] += 1
                vec, st = _sample_patch_carrier_for_row(
                    row=row,
                    frame_bank_dir=frame_bank_dir,
                    frame_map=frame_map,
                    geom_map=geom_map,
                    payload_cache=payload_cache,
                    tokens_per_view=token_k,
                    min_token_weight=min_weight,
                    max_frame_candidate_tokens=max_cand,
                    seed=gate_seed,
                )
                if vec is None:
                    counters["patchsample_fallback_to_mean"] += 1
                    reason = str(st.get("reason", "unknown")) if isinstance(st, Mapping) else "unknown"
                    counters[f"patchsample_skip_{reason}"] += 1
                    continue
                replacements[str(tid)] = np.asarray(vec, dtype=np.float32)
                counters["patchsample_used"] += 1
                if isinstance(st, Mapping) and st.get("token_candidate_count_mean_per_frame") is not None:
                    candidate_means.append(float(st.get("token_candidate_count_mean_per_frame", 0.0)))
        finally:
            for payload in payload_cache.values():
                try:
                    payload.close()
                except Exception:
                    pass
        payload_cache_scope = "clip_local"

    def apply(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        out: List[Mapping[str, Any]] = []
        for r in rows:
            rr = dict(r)
            tid = str(rr.get("trajectory_id", ""))
            if tid in replacements:
                rr["carrier_vec"] = replacements[tid]
                rr["carrier_aug_source"] = str(mode)
            else:
                rr["carrier_aug_source"] = "mean_carrier"
            out.append(rr)
        return out

    stats = {
        "train_carrier_aug_enabled": True,
        "train_carrier_aug_mode": str(mode),
        "patchsample_prob": float(patch_prob),
        "patchsample_tokens_per_view": int(token_k),
        "patchsample_unique_trajectories": int(len(set([str(r.get("trajectory_id", "")) for r in list(pre_group) + list(dyn_rows) if str(r.get("trajectory_id", ""))]))),
        "patchsample_attempted": int(counters.get("patchsample_attempted", 0)),
        "patchsample_used": int(counters.get("patchsample_used", 0)),
        "patchsample_fallback_to_mean": int(counters.get("patchsample_fallback_to_mean", 0)),
        "patchsample_kept_mean_by_probability": int(counters.get("kept_mean_by_probability", 0)),
        "patchsample_token_candidate_count_mean_per_frame": float(np.mean(candidate_means)) if candidate_means else 0.0,
        "patchsample_counters": dict(counters),
        "payload_cache_scope": payload_cache_scope,
    }
    return apply(pre_group), apply(dyn_rows), stats

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


def _resolve_projector_constraint(args: argparse.Namespace) -> Dict[str, Any]:
    constraint = str(getattr(args, "projector_constraint", "none") or "none").strip().lower()
    if constraint not in {"none", "hard_semi_orth", "orth_penalty"}:
        raise RuntimeError(f"unsupported --projector_constraint={constraint!r}")
    requested_type = str(getattr(args, "text_projector_type", "mlp") or "mlp").strip().lower()
    if constraint == "hard_semi_orth":
        effective_type = "semi_orthogonal_linear"
    elif constraint == "orth_penalty":
        effective_type = "linear"
    else:
        effective_type = requested_type
    if effective_type not in {"mlp", "linear", "linear_ln", "semi_orthogonal_linear"}:
        raise RuntimeError(f"unsupported effective text projector type={effective_type!r}")
    weight = float(getattr(args, "orth_penalty_weight", 0.0) or 0.0)
    every = max(int(getattr(args, "orth_penalty_every_n_steps", 1) or 1), 1)
    return {
        "constraint": constraint,
        "requested_text_projector_type": requested_type,
        "effective_text_projector_type": effective_type,
        "orth_penalty_weight": float(weight),
        "orth_penalty_every_n_steps": int(every),
        "hard_orth_project_every_n_steps": max(int(getattr(args, "hard_orth_project_every_n_steps", 1) or 1), 1),
        "forces_text_projector_type": bool(constraint in {"hard_semi_orth", "orth_penalty"}),
        "hard_constraint": bool(constraint == "hard_semi_orth"),
        "soft_penalty": bool(constraint == "orth_penalty" and weight > 0.0),
    }


def _orthogonality_loss_for_step(
    *,
    projector: Projector,
    text_proj_all: torch.Tensor,
    projector_constraint: Mapping[str, Any],
    global_step: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    constraint = str(projector_constraint.get("constraint", "none"))
    weight = float(projector_constraint.get("orth_penalty_weight", 0.0) or 0.0)
    every = max(int(projector_constraint.get("orth_penalty_every_n_steps", 1) or 1), 1)
    report: Dict[str, Any]
    try:
        report = dict(projector.orthogonality_report())
    except Exception as exc:
        report = {"orthogonality_applicable": False, "orthogonality_error": repr(exc)}
    if constraint == "orth_penalty" and weight > 0.0 and (int(global_step) % every == 0):
        loss = projector.orthogonality_penalty()
        report.update({"orth_penalty_applied_this_step": True, "orth_penalty_loss": float(loss.detach().cpu().item())})
        return loss, report
    zero = text_proj_all.sum() * 0.0
    report.update({"orth_penalty_applied_this_step": False, "orth_penalty_loss": 0.0})
    return zero, report


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
    text_bank_summary = _resolve_training_text_bank(args, data)
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

    projector_constraint = _resolve_projector_constraint(args)
    text_projector_type = str(projector_constraint["effective_text_projector_type"])
    text_projector_config = {
        "input_dim": int(text_bank_summary["projector_input_dim"]),
        "hidden_dim": int(args.text_projector_hidden_dim) if text_projector_type == "mlp" else 0,
        "output_dim": int(args.text_projector_out_dim),
        "dropout": 0.0,
        "use_layernorm": bool(text_projector_type in {"mlp", "linear_ln"}),
        "projector_type": text_projector_type,
        "projector_constraint": str(projector_constraint["constraint"]),
        "requested_projector_type": str(projector_constraint["requested_text_projector_type"]),
        "orth_penalty_weight": float(projector_constraint["orth_penalty_weight"]),
        "orth_penalty_every_n_steps": int(projector_constraint["orth_penalty_every_n_steps"]),
        "hard_orth_project_every_n_steps": int(projector_constraint["hard_orth_project_every_n_steps"]),
    }
    projector = Projector(ProjectorConfig(**text_projector_config)).to(device)
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(args.t_dis_init) - 1.0e-4, 1.0e-6)), device=device, dtype=torch.float32))
    init = str(args.init_checkpoint).strip()
    ckpt = _auto_find_checkpoint(repo_root, str(args.dataset_name)) if init == "auto" else init
    checkpoint_summary = _load_checkpoint_if_requested(projector, theta_t, ckpt, device)
    if str(projector_constraint["constraint"]) == "hard_semi_orth" and hasattr(projector, "project_semi_orthogonal_"):
        projector.project_semi_orthogonal_()

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

    train_aug_mode = str(args.train_carrier_aug_mode).strip().lower()
    train_aug_enabled = bool(train_aug_mode in {"patchsample_mixed", "patchsample_cached_mixed"} and float(args.patchsample_prob) > 0.0)
    frame_bank_dir: Optional[Path] = None
    frame_map: Optional[Mapping[Tuple[str, int], Mapping[str, Any]]] = None
    geom_map: Optional[Mapping[Tuple[str, int], Mapping[str, Any]]] = None
    patchsample_cache: Optional[Mapping[str, Any]] = None
    frame_bank_summary: Dict[str, Any] = {"status": "NOT_USED", "reason": "train_carrier_aug_disabled"}
    cache_summary: Dict[str, Any] = {"status": "NOT_USED", "reason": "patchsample cache not requested"}
    if train_aug_enabled and train_aug_mode == "patchsample_cached_mixed":
        if not str(getattr(args, "patchsample_cache_dir", "")).strip():
            raise RuntimeError("--train_carrier_aug_mode patchsample_cached_mixed requires --patchsample_cache_dir")
        patchsample_cache = _load_patchsample_cache(Path(args.patchsample_cache_dir))
        if int(patchsample_cache["epochs"]) < int(args.epochs):
            raise RuntimeError(f"patchsample cache epochs={patchsample_cache['epochs']} < requested training epochs={args.epochs}")
        cache_summary = {
            "status": "PASS",
            "cache_dir": str(patchsample_cache["cache_dir"]),
            "epochs": int(patchsample_cache["epochs"]),
            "trajectory_count": int(patchsample_cache["trajectory_count"]),
            "dim": int(patchsample_cache["dim"]),
            "shape": list(patchsample_cache["shape"]),
            "vectors_path": str(patchsample_cache["vectors_path"]),
            "valid_path": str(patchsample_cache["valid_path"]),
            "runs_dinov2_encoder": False,
        }
        frame_bank_summary = {"status": "NOT_USED", "reason": "offline patchsample cache lookup"}
    elif train_aug_enabled:
        frame_bank_dir = asset_root / "frame_bank" / str(args.dataset_name)
        frame_map_loaded, geom_map_loaded = _load_frame_maps(frame_bank_dir)
        frame_map = frame_map_loaded
        geom_map = geom_map_loaded
        frame_bank_summary = {
            "status": "PASS",
            "frame_bank_dir": str(frame_bank_dir),
            "frame_record_count": int(len(frame_map_loaded)),
            "frame_geom_record_count": int(len(geom_map_loaded)),
            "payload_cache_scope": "clip_local",
            "runs_dinov2_encoder": False,
        }

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
        "projector_constraint": dict(projector_constraint),
        "text_bank": text_bank_summary,
        "text_projector_config": dict(text_projector_config),
        "train_carrier_aug": {
            "enabled": bool(train_aug_enabled),
            "mode": str(args.train_carrier_aug_mode),
            "patchsample_prob": float(args.patchsample_prob),
            "patchsample_tokens_per_view": int(args.patchsample_tokens_per_view),
            "patchsample_seed": int(args.patchsample_seed),
            "patchsample_min_token_weight": float(args.patchsample_min_token_weight),
            "patchsample_max_frame_candidate_tokens": int(args.patchsample_max_frame_candidate_tokens),
            "frame_bank": frame_bank_summary,
            "offline_cache": cache_summary,
            "patchsample_cache_dir": str(getattr(args, "patchsample_cache_dir", "")),
            "uses_frame_bank_patch_token_cache": bool(train_aug_enabled and train_aug_mode == "patchsample_mixed"),
            "uses_offline_patchsample_cache": bool(train_aug_enabled and train_aug_mode == "patchsample_cached_mixed"),
            "runs_dinov2_encoder": False,
            "changes_inference_protocol": False,
            "preserves_clip_wise_training_protocol": True,
        },
        "graph_preserve": {
            k: v for k, v in graph_cache.items()
            if k not in {"scope_text_indices", "neighbor_idx", "target_prob", "raw_neighbor_sim"}
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
            "uses_external_text_bank_variant": bool(str(args.text_bank_variant) != "clip_current"),
            "text_bank_replaces_only_text_anchor_source": True,
            "text_projector_type": str(text_projector_type),
            "requested_text_projector_type": str(projector_constraint["requested_text_projector_type"]),
            "projector_constraint": str(projector_constraint["constraint"]),
            "uses_hard_semi_orthogonal_projector": bool(projector_constraint["constraint"] == "hard_semi_orth"),
            "uses_soft_orthogonality_penalty": bool(projector_constraint["constraint"] == "orth_penalty" and float(projector_constraint["orth_penalty_weight"]) > 0.0),
            "uses_linear_text_projector": bool(text_projector_type in {"linear", "linear_ln", "semi_orthogonal_linear"}),
            "text_bank_changes_dynamic_hungarian_protocol": False,
            "text_bank_changes_clip_wise_training_protocol": False,
            "text_bank_changes_candidate_set": False,
            "text_bank_changes_trajectory_source": False,
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
            "uses_train_side_patchsample_carrier_augmentation": bool(train_aug_enabled),
            "patchsample_aug_changes_inference_protocol": False,
            "patchsample_aug_preserves_clip_wise_training_protocol": True,
            "patchsample_aug_does_not_random_batch_trajectories": True,
            "patchsample_aug_does_not_cross_clip_hungarian": True,
            "patchsample_aug_prefetch_unit": "offline_cache_lookup" if train_aug_mode == "patchsample_cached_mixed" else "clip_payload_only",
            "patchsample_aug_uses_offline_cache": bool(train_aug_enabled and train_aug_mode == "patchsample_cached_mixed"),
            "patchsample_aug_runs_dinov2_encoder": False,
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
    all_orth: List[float] = []
    all_patch_attempted = 0
    all_patch_used = 0
    all_patch_fallback = 0
    all_patch_kept_mean = 0
    epoch_rows: List[Dict[str, Any]] = []
    iterator = tqdm(range(int(args.epochs)), desc=f"a8_dyn_hun_{args.name}", dynamic_ncols=True) if bool(args.show_progress) and tqdm is not None else range(int(args.epochs))
    for epoch in iterator:
        cur_clip_ids = list(clip_ids)
        rng.shuffle(cur_clip_ids)
        ep_total: List[float] = []
        ep_pre: List[float] = []
        ep_dyn: List[float] = []
        ep_graph: List[float] = []
        ep_orth: List[float] = []
        ep_pre_rows = 0
        ep_dyn_rows = 0
        ep_pairs = 0
        ep_dyn_margin: List[float] = []
        ep_dyn_person: List[float] = []
        ep_pre_mass: List[float] = []
        ep_patch_attempted = 0
        ep_patch_used = 0
        ep_patch_fallback = 0
        ep_patch_kept_mean = 0
        ep_patch_candidate_mean: List[float] = []
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
            pre_group, dyn_rows, patch_stats = _maybe_patchsample_augment_clip_rows(
                clip_id=int(cid),
                pre_group=pre_group,
                dyn_rows=dyn_rows,
                epoch=int(epoch) + 1,
                args=args,
                frame_bank_dir=frame_bank_dir,
                frame_map=frame_map,
                geom_map=geom_map,
                patchsample_cache=patchsample_cache,
            )
            ep_patch_attempted += int(patch_stats.get("patchsample_attempted", 0))
            ep_patch_used += int(patch_stats.get("patchsample_used", 0))
            ep_patch_fallback += int(patch_stats.get("patchsample_fallback_to_mean", 0))
            ep_patch_kept_mean += int(patch_stats.get("patchsample_kept_mean_by_probability", 0))
            all_patch_attempted += int(patch_stats.get("patchsample_attempted", 0))
            all_patch_used += int(patch_stats.get("patchsample_used", 0))
            all_patch_fallback += int(patch_stats.get("patchsample_fallback_to_mean", 0))
            all_patch_kept_mean += int(patch_stats.get("patchsample_kept_mean_by_probability", 0))
            if float(patch_stats.get("patchsample_token_candidate_count_mean_per_frame", 0.0)) > 0:
                ep_patch_candidate_mean.append(float(patch_stats.get("patchsample_token_candidate_count_mean_per_frame", 0.0)))
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
            orth_loss, orth_stats = _orthogonality_loss_for_step(
                projector=projector,
                text_proj_all=text_proj_all,
                projector_constraint=projector_constraint,
                global_step=int(global_step),
            )
            total_loss = (
                float(args.lambda_prealign) * pre_loss
                + float(args.lambda_dynamic) * dyn_loss
                + float(args.lambda_graph_preserve) * graph_loss
                + float(projector_constraint["orth_penalty_weight"]) * orth_loss
            )
            total_loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_([*projector.parameters(), theta_t], max_norm=float(args.grad_clip_norm))
            optimizer.step()
            if (
                str(projector_constraint["constraint"]) == "hard_semi_orth"
                and hasattr(projector, "project_semi_orthogonal_")
                and ((int(global_step) + 1) % int(projector_constraint["hard_orth_project_every_n_steps"]) == 0)
            ):
                projector.project_semi_orthogonal_()
            global_step += 1
            tv = float(total_loss.detach().cpu().item())
            pv = float(pre_loss.detach().cpu().item())
            dv = float(dyn_loss.detach().cpu().item())
            gv = float(graph_loss.detach().cpu().item())
            ov = float(orth_loss.detach().cpu().item())
            all_total.append(tv); all_pre.append(pv); all_dyn.append(dv); all_graph.append(gv); all_orth.append(ov)
            ep_total.append(tv); ep_pre.append(pv); ep_dyn.append(dv); ep_graph.append(gv); ep_orth.append(ov)
            ep_pre_rows += int(pre_stats.get("rows", 0)); ep_dyn_rows += int(dyn_stats.get("dynamic_rows", 0)); ep_pairs += int(dyn_stats.get("dynamic_assigned_pairs", 0))
            ep_dyn_margin.append(float(dyn_stats.get("dynamic_selected_margin_mean", 0.0)))
            ep_dyn_person.append(float(dyn_stats.get("dynamic_person_raw773_assignment_rate", 0.0)))
            ep_pre_mass.append(float(pre_stats.get("mass_in_y_mean", 0.0)))
            ep_solver[str(dyn_stats.get("dynamic_assignment_solver", ""))] += 1
            if int(args.log_every_steps) > 0 and global_step % int(args.log_every_steps) == 0:
                _append_jsonl(log_path, {"timestamp": _now(), "row_type": "step", "epoch": int(epoch) + 1, "global_step": global_step, "loss_total": tv, "loss_prealign": pv, "loss_dynamic": dv, "loss_graph_preserve": gv, "loss_orthogonality": ov, **pre_stats, **dyn_stats, **graph_stats, **orth_stats, **patch_stats})
        epoch_row = {
            "timestamp": _now(),
            "row_type": "epoch_summary",
            "epoch": int(epoch) + 1,
            "global_step": int(global_step),
            "loss_total_mean": _mean(ep_total),
            "loss_prealign_mean": _mean(ep_pre),
            "loss_dynamic_mean": _mean(ep_dyn),
            "loss_graph_preserve_mean": _mean(ep_graph),
            "loss_orthogonality_mean": _mean(ep_orth),
            "orthogonality_report": projector.orthogonality_report(),
            "prealign_mass_in_y_mean": _mean(ep_pre_mass),
            "dynamic_selected_margin_mean": _mean(ep_dyn_margin),
            "dynamic_person_raw773_assignment_rate_mean": _mean(ep_dyn_person),
            "patchsample_attempted": int(ep_patch_attempted),
            "patchsample_used": int(ep_patch_used),
            "patchsample_fallback_to_mean": int(ep_patch_fallback),
            "patchsample_kept_mean_by_probability": int(ep_patch_kept_mean),
            "patchsample_used_rate_among_attempted": float(ep_patch_used / max(ep_patch_attempted, 1)),
            "patchsample_token_candidate_count_mean_per_frame": _mean(ep_patch_candidate_mean),
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
            _save_checkpoint(train_dir / f"a8_joint_train_time_dynamic_epoch_{int(epoch)+1:03d}.pth", projector=projector, theta_t=theta_t, epoch=int(epoch)+1, global_step=global_step, payload=setup, text_projector_config=text_projector_config)

    ckpt_out = train_dir / "a8_joint_train_time_dynamic_last.pth"
    _save_checkpoint(ckpt_out, projector=projector, theta_t=theta_t, epoch=int(args.epochs), global_step=global_step, payload=setup, text_projector_config=text_projector_config)
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
            "loss_orthogonality_mean": _mean(all_orth),
            "orthogonality_report": projector.orthogonality_report(),
            "loss_total_last": all_total[-1] if all_total else 0.0,
            "patchsample_attempted": int(all_patch_attempted),
            "patchsample_used": int(all_patch_used),
            "patchsample_fallback_to_mean": int(all_patch_fallback),
            "patchsample_kept_mean_by_probability": int(all_patch_kept_mean),
            "patchsample_used_rate_among_attempted": float(all_patch_used / max(all_patch_attempted, 1)),
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
    p.add_argument("--text_bank_variant", choices=["clip_current", "clip_of_llm_mean", "llama_hidden_mean", "llama_direct_concept_mean"], default="clip_current", help="Text anchor source. clip_current preserves the canonical CLIP text bank; other variants load LV-VIS Llama3/CLIP-of-LLM banks.")
    p.add_argument("--text_bank_root", default="", help="Root directory of an external text bank profile, e.g. $ASSERT_ROOT/text_bank_llama3/lvvis/lvvis_visual_only_v1.")
    p.add_argument("--text_feature_dim", type=int, default=0, help="Optional guard for loaded text feature dimension. 0 means infer.")
    p.add_argument("--text_projector_type", choices=["mlp", "linear", "linear_ln", "semi_orthogonal_linear"], default="mlp", help="Text projector family. mlp preserves the historical LayerNorm+Linear+GELU+Linear mapper; linear uses one Linear(D->out)+L2 normalize; linear_ln adds input LayerNorm before that single linear map. semi_orthogonal_linear is normally activated via --projector_constraint hard_semi_orth.")
    p.add_argument("--projector_constraint", choices=["none", "hard_semi_orth", "orth_penalty"], default="none", help="A8 text-bank projector constraint ablation. none preserves historical behavior; hard_semi_orth forces a QR semi-orthogonal linear projector; orth_penalty uses a linear projector plus an orthogonality penalty.")
    p.add_argument("--orth_penalty_weight", type=float, default=0.0, help="Weight for --projector_constraint orth_penalty. Ignored by hard_semi_orth and none.")
    p.add_argument("--orth_penalty_every_n_steps", type=int, default=1, help="Apply the soft orthogonality penalty every N optimizer steps.")
    p.add_argument("--hard_orth_project_every_n_steps", type=int, default=1, help="For --projector_constraint hard_semi_orth, project the semi-orthogonal weight back to the Stiefel set every N optimizer steps. Use 1 for exact step-wise retraction; larger values are a pragmatic Llama3-cost control.")
    p.add_argument("--text_projector_hidden_dim", type=int, default=1024)
    p.add_argument("--text_projector_out_dim", type=int, default=768)
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--output_dir", default="")
    p.add_argument("--clip_universe_csv", default="", help="Optional old matched-pairs CSV used only for clip/row universe; matched_raw_id is ignored.")
    p.add_argument("--dynamic_row_source", choices=["all_clip_trajectories", "clip_universe_rows"], default="all_clip_trajectories")
    p.add_argument("--dynamic_loss", choices=["ce", "infonce"], default="ce")
    p.add_argument("--lambda_prealign", type=float, default=1.0)
    p.add_argument("--lambda_dynamic", type=float, default=1.0)
    p.add_argument("--lambda_graph_preserve", type=float, default=0.0)
    p.add_argument("--graph_preserve_mode", choices=["none", "raw_text_topk", "random_text_topk", "topk_local", "topk_local_mse", "raw_text_topk_mse", "random_topk_local", "random_topk_local_mse", "random_text_topk_mse"], default="none", help="Text-only local graph regularizer. raw_text_topk is legacy KL; topk_local/topk_local_mse uses local edge cosine MSE; random_* are controls.")
    p.add_argument("--graph_preserve_scope", choices=["visible525", "base_vocab"], default="visible525")
    p.add_argument("--graph_preserve_topk", type=int, default=20)
    p.add_argument("--graph_preserve_tau", type=float, default=0.1)
    p.add_argument("--graph_preserve_every_n_steps", type=int, default=1)
    p.add_argument("--graph_preserve_visible_csv", default="", help="Optional base_641_visibility_by_class.csv; used when graph_preserve_scope=visible525.")
    p.add_argument("--train_carrier_aug_mode", choices=["none", "patchsample_mixed", "patchsample_cached_mixed"], default="none", help="Train-side carrier augmentation only; inference remains mean-carrier unless separately changed. patchsample_cached_mixed uses offline cached sampled carriers.")
    p.add_argument("--patchsample_cache_dir", default="", help="Offline patchsample carrier cache directory produced by tools/a8_build_patchsample_carrier_cache.py; required for patchsample_cached_mixed.")
    p.add_argument("--patchsample_prob", type=float, default=0.0, help="Probability of replacing a trajectory mean carrier with one patch-token sampled carrier during training.")
    p.add_argument("--patchsample_tokens_per_view", type=int, default=64, help="Number of mask-inside patch tokens sampled per augmented carrier.")
    p.add_argument("--patchsample_seed", type=int, default=3407, help="Stable seed used for epoch/clip/trajectory patch sampling.")
    p.add_argument("--patchsample_min_token_weight", type=float, default=0.0, help="Minimum projected mask token weight for sampled-token eligibility.")
    p.add_argument("--patchsample_max_frame_candidate_tokens", type=int, default=4096, help="Optional per-frame cap on eligible mask tokens before weighted sampling; 0 disables cap.")
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
