#!/usr/bin/env python3
"""Residual-gated GT-clean balanced training pilot.

This is a side-path experiment entry. It intentionally does not modify the
normal WS-OVVIS training path or any package/control-plane files.

Inputs:
  * residual_gated balanced training manifest CSV produced by
    run_stageb_analysis_residual_gated_training_manifest.py
  * GT-clean row-level assignment gap CSV for evaluation rows
  * existing GT-carrier/text-bank assets through the repository asset links

Modes:
  * eval_only: materialize/evaluate only; no optimizer step.
  * hard_ce: train only hard_ce manifest rows.
  * hard_soft_proto: train hard_ce + soft_ce + prototype_calibration rows.

The script is GPU-friendly: it projects the full text bank once per train/eval
batch and evaluates rows in vectorized chunks. It avoids row-wise full-vocab
loops and keeps all outputs lightweight CSV/JSON/MD.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    _compute_t_dis,
    _inverse_softplus,
    _load_base_ids,
    _load_materialized_gt_examples,
    _normalize_np,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


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


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(list(values), dtype=np.float64)))


def _default_manifest_path(run_root: Path, dataset_name: str) -> Path:
    return run_root / "analysis" / "residual_gated_training_manifest" / dataset_name / "balanced_training_manifest.csv"


def _default_row_gap_path(repo_root: Path, dataset_name: str) -> Path:
    return (
        repo_root
        / "codex" / "outputs" / "G8_inference_and_eval"
        / "gt_clean_weak_fully_overfit_capacity_20260502"
        / "analysis" / "assignment_oracle_gap_audit" / dataset_name / "base_vocab" / "row_level_assignment_gap.csv"
    )


def _default_output_root(run_root: Path, dataset_name: str, mode: str) -> Path:
    return run_root / "outputs" / "residual_gated_gtclean_pilot" / dataset_name / str(mode)


def _iter_progress(iterable: Iterable[Any], *, enabled: bool, desc: str) -> Iterable[Any]:
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, dynamic_ncols=True)
    return iterable


@dataclass
class PreparedData:
    examples: List[Dict[str, Any]]
    example_by_tid: Dict[str, Dict[str, Any]]
    base_ids: List[int]
    raw_to_text_idx: Dict[int, int]
    text_ids: List[int]
    text_matrix: np.ndarray
    materialization_summary: Dict[str, Any]


def _prepare_data(args: argparse.Namespace, output_root: Path) -> PreparedData:
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    _bootstrap_asset_links(repo_root, asset_root)
    _bootstrap_asset_links(output_root, asset_root)

    examples, _clip_y_base, base_ids_set, materialization_summary = _load_materialized_gt_examples(
        repo_root=repo_root,
        output_root=output_root,
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
        raise RuntimeError("no materialized GT-clean examples were loaded")
    example_by_tid: Dict[str, Dict[str, Any]] = {}
    duplicate_tid = 0
    for ex in examples:
        tid = str(ex.get("trajectory_id", ""))
        if not tid:
            continue
        if tid in example_by_tid:
            duplicate_tid += 1
        example_by_tid.setdefault(tid, ex)

    text_ids, _text_records, text_matrix = load_text_vocab(output_root)
    raw_to_text_idx = {int(raw_id): idx for idx, raw_id in enumerate(text_ids)}
    base_ids = sorted(int(x) for x in base_ids_set if int(x) in raw_to_text_idx)
    if not base_ids:
        raise RuntimeError("no official-base raw ids overlap with text bank")
    materialization_summary = dict(materialization_summary)
    materialization_summary["example_by_trajectory_id_count"] = int(len(example_by_tid))
    materialization_summary["duplicate_trajectory_id_count"] = int(duplicate_tid)
    materialization_summary["base_text_class_count"] = int(len(base_ids))
    return PreparedData(
        examples=examples,
        example_by_tid=example_by_tid,
        base_ids=base_ids,
        raw_to_text_idx=raw_to_text_idx,
        text_ids=[int(x) for x in text_ids],
        text_matrix=np.asarray(text_matrix, dtype=np.float32),
        materialization_summary=materialization_summary,
    )


def _load_manifest_rows(path: Path, *, mode: str, example_by_tid: Mapping[str, Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = _read_csv(path)
    selected: List[Dict[str, Any]] = []
    counters = Counter()
    for row in rows:
        manifest_use = str(row.get("manifest_use", "train"))
        if manifest_use != "train":
            counters["skip_not_train"] += 1
            continue
        loss_family = str(row.get("loss_family", ""))
        if mode == "eval_only":
            continue
        if mode == "hard_ce" and loss_family != "hard_ce":
            counters["skip_not_hard_mode"] += 1
            continue
        if mode == "hard_soft_proto" and loss_family not in {"hard_ce", "soft_ce", "prototype_calibration"}:
            counters["skip_unrecognized_loss_family"] += 1
            continue
        tid = str(row.get("trajectory_id", ""))
        rid = _as_int(row.get("gt_raw_id"))
        if not tid or rid is None:
            counters["skip_missing_tid_or_gt"] += 1
            continue
        if tid not in example_by_tid:
            counters["skip_missing_example_by_tid"] += 1
            continue
        out = dict(row)
        out["gt_raw_id_int"] = int(rid)
        out["sample_weight_float"] = _as_float(row.get("sample_weight"), 1.0)
        selected.append(out)
        counters[f"selected_{loss_family}"] += 1
    summary = {
        "manifest_csv": str(path),
        "mode": str(mode),
        "input_manifest_rows": int(len(rows)),
        "selected_train_rows": int(len(selected)),
        "counters": dict(counters),
    }
    return selected, summary


def _load_row_gap_eval_rows(
    path: Path,
    *,
    example_by_tid: Mapping[str, Mapping[str, Any]],
    raw_to_text_idx: Mapping[int, int],
    max_rows: int = 0,
    seed: int = 3407,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = _read_csv(path)
    valid: List[Dict[str, Any]] = []
    counters = Counter()
    for row in rows:
        tid = str(row.get("trajectory_id", ""))
        rid = _as_int(row.get("gt_raw_id"))
        if not tid or rid is None:
            counters["skip_missing_tid_or_gt"] += 1
            continue
        if int(rid) not in raw_to_text_idx:
            counters["skip_gt_not_in_text_bank"] += 1
            continue
        if tid not in example_by_tid:
            counters["skip_missing_example_by_tid"] += 1
            continue
        r = dict(row)
        r["gt_raw_id_int"] = int(rid)
        valid.append(r)
    if int(max_rows) > 0 and len(valid) > int(max_rows):
        rng = random.Random(int(seed))
        idxs = sorted(rng.sample(range(len(valid)), k=int(max_rows)))
        valid = [valid[i] for i in idxs]
        counters["subsampled_eval_rows"] += 1
    summary = {
        "row_gap_csv": str(path),
        "input_rows": int(len(rows)),
        "valid_eval_rows": int(len(valid)),
        "counters": dict(counters),
    }
    return valid, summary


def _auto_find_checkpoint(args: argparse.Namespace) -> str:
    if str(args.init_checkpoint).strip() != "auto":
        return str(args.init_checkpoint).strip()
    repo_root = Path(args.repo_root).expanduser().resolve()
    summary_candidates = [
        _default_row_gap_path(repo_root, str(args.dataset_name)).parents[1] / "assignment_oracle_gap_summary.csv",
        repo_root
        / "codex" / "outputs" / "G8_inference_and_eval"
        / "gt_clean_weak_fully_overfit_capacity_20260502"
        / "analysis" / "assignment_oracle_gap_audit" / str(args.dataset_name) / "assignment_oracle_gap_summary.csv",
    ]
    for summary_path in summary_candidates:
        if not summary_path.is_file():
            continue
        try:
            rows = _read_csv(summary_path)
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
        return {"loaded": False, "reason": "no checkpoint requested or found"}
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


def _base_text_projection(
    projector: Projector,
    text_tensor: torch.Tensor,
    base_text_indices: torch.Tensor,
) -> torch.Tensor:
    text_proj_all = F.normalize(projector(text_tensor), p=2.0, dim=-1)
    return text_proj_all[base_text_indices]


def _batch_carrier_tensor(batch_rows: Sequence[Mapping[str, Any]], example_by_tid: Mapping[str, Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    vecs = []
    for row in batch_rows:
        ex = example_by_tid[str(row["trajectory_id"])]
        vecs.append(torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))))
    Z = torch.stack(vecs, dim=0).to(device=device, dtype=torch.float32)
    return F.normalize(Z, p=2.0, dim=-1)


def _loss_on_batch(
    *,
    rows: Sequence[Mapping[str, Any]],
    example_by_tid: Mapping[str, Mapping[str, Any]],
    base_ids: Sequence[int],
    base_raw_to_col: Mapping[int, int],
    text_proj_base: torch.Tensor,
    theta_t: torch.nn.Parameter,
    device: torch.device,
    soft_label_smoothing: float,
    proto_loss_weight: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if not rows:
        raise ValueError("empty training batch")
    Z = _batch_carrier_tensor(rows, example_by_tid, device)
    temperature = _compute_t_dis(theta_t)
    logits = torch.matmul(Z, text_proj_base.t()) / temperature
    targets = torch.tensor([int(base_raw_to_col[int(row["gt_raw_id_int"])]) for row in rows], device=device, dtype=torch.long)
    weights = torch.tensor([float(row.get("sample_weight_float", 1.0)) for row in rows], device=device, dtype=torch.float32)
    loss_fams = [str(row.get("loss_family", "")) for row in rows]
    loss_terms: List[torch.Tensor] = []
    stats = Counter()
    if any(x == "hard_ce" for x in loss_fams):
        idx = torch.tensor([i for i, x in enumerate(loss_fams) if x == "hard_ce"], device=device, dtype=torch.long)
        ce = F.cross_entropy(logits[idx], targets[idx], reduction="none")
        w = weights[idx]
        loss_terms.append((ce * w).sum() / torch.clamp(w.sum(), min=1.0))
        stats["hard_ce_rows"] = int(idx.numel())
    if any(x == "soft_ce" for x in loss_fams):
        idx = torch.tensor([i for i, x in enumerate(loss_fams) if x == "soft_ce"], device=device, dtype=torch.long)
        log_probs = torch.log_softmax(logits[idx], dim=1)
        eps = float(max(0.0, min(float(soft_label_smoothing), 0.9)))
        n_cls = int(log_probs.shape[1])
        target_dist = torch.full_like(log_probs, fill_value=eps / max(n_cls - 1, 1))
        target_dist.scatter_(1, targets[idx].view(-1, 1), 1.0 - eps)
        soft_loss = -(target_dist * log_probs).sum(dim=1)
        w = weights[idx]
        loss_terms.append((soft_loss * w).sum() / torch.clamp(w.sum(), min=1.0))
        stats["soft_ce_rows"] = int(idx.numel())
    if any(x == "prototype_calibration" for x in loss_fams):
        idx = torch.tensor([i for i, x in enumerate(loss_fams) if x == "prototype_calibration"], device=device, dtype=torch.long)
        proto = text_proj_base[targets[idx]]
        z = F.normalize(Z[idx], p=2.0, dim=-1)
        proto_loss = 1.0 - torch.sum(z * proto, dim=1)
        w = weights[idx]
        loss_terms.append(float(proto_loss_weight) * (proto_loss * w).sum() / torch.clamp(w.sum(), min=1.0))
        stats["prototype_calibration_rows"] = int(idx.numel())
    if not loss_terms:
        raise RuntimeError("batch has no recognized trainable loss family")
    loss = torch.stack(loss_terms).sum()
    stats["batch_rows"] = int(len(rows))
    stats["temperature"] = float(temperature.detach().cpu().item())
    stats["loss"] = float(loss.detach().cpu().item())
    return loss, dict(stats)


def _evaluate_rows(
    *,
    stage: str,
    rows: Sequence[Mapping[str, Any]],
    example_by_tid: Mapping[str, Mapping[str, Any]],
    projector: Projector,
    text_tensor: torch.Tensor,
    base_ids: Sequence[int],
    raw_to_text_idx: Mapping[int, int],
    theta_t: torch.nn.Parameter,
    device: torch.device,
    batch_size: int,
    out_dir: Path,
) -> Dict[str, Any]:
    projector.eval()
    base_text_indices = torch.tensor([int(raw_to_text_idx[int(x)]) for x in base_ids], device=device, dtype=torch.long)
    base_ids_list = [int(x) for x in base_ids]
    base_raw_to_col = {int(rid): idx for idx, rid in enumerate(base_ids_list)}
    pred_rows: List[Dict[str, Any]] = []
    by_class: Dict[int, Counter] = defaultdict(Counter)
    norm_ranks: List[float] = []
    with torch.no_grad():
        text_proj_base = _base_text_projection(projector, text_tensor, base_text_indices)
        temperature = _compute_t_dis(theta_t)
        for start in _iter_progress(range(0, len(rows), int(batch_size)), enabled=False, desc=f"eval {stage}"):
            batch = rows[start:start + int(batch_size)]
            Z = _batch_carrier_tensor(batch, example_by_tid, device)
            logits = torch.matmul(Z, text_proj_base.t()) / temperature
            top_idx = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64).tolist()
            logits_cpu = logits.detach().cpu().numpy().astype(np.float32)
            for i, row in enumerate(batch):
                gt = int(row["gt_raw_id_int"])
                if gt not in base_raw_to_col:
                    continue
                gt_col = int(base_raw_to_col[gt])
                gt_logit = float(logits_cpu[i, gt_col])
                rank = int(1 + np.sum(logits_cpu[i, :] > gt_logit))
                top_raw = int(base_ids_list[int(top_idx[i])])
                top1 = bool(top_raw == gt)
                top5 = bool(rank <= 5)
                denom = max(len(base_ids_list) - 1, 1)
                norm_rank = float((rank - 1) / denom)
                norm_ranks.append(norm_rank)
                by_class[gt]["count"] += 1
                by_class[gt]["top1"] += int(top1)
                by_class[gt]["top5"] += int(top5)
                by_class[gt]["rank_sum"] += rank
                by_class[gt]["norm_rank_sum"] += norm_rank
                pred_rows.append({
                    "stage": stage,
                    "clip_id": row.get("clip_id", ""),
                    "trajectory_id": row.get("trajectory_id", ""),
                    "gt_raw_id": gt,
                    "gt_class_name": row.get("gt_class_name", ""),
                    "top1_raw_id": top_raw,
                    "gt_rank": rank,
                    "gt_norm_rank": norm_rank,
                    "gt_top1_hit": int(top1),
                    "gt_top5_hit": int(top5),
                    "source_seed_type": row.get("seed_type", ""),
                    "source_loss_family": row.get("loss_family", ""),
                    "source_policy": row.get("policy", ""),
                })
    by_class_rows: List[Dict[str, Any]] = []
    for rid, c in sorted(by_class.items()):
        n = int(c["count"])
        by_class_rows.append({
            "raw_id": int(rid),
            "class_name": "",
            "gt_count": n,
            "gt_top1_hit_rate": float(c["top1"] / max(n, 1)),
            "gt_top5_hit_rate": float(c["top5"] / max(n, 1)),
            "mean_gt_rank": float(c["rank_sum"] / max(n, 1)),
            "mean_normalized_gt_rank": float(c["norm_rank_sum"] / max(n, 1)),
        })
    total = len(pred_rows)
    summary = {
        "stage": stage,
        "eval_rows": int(total),
        "class_count": int(len(by_class_rows)),
        "micro_top1": float(sum(r["gt_top1_hit"] for r in pred_rows) / max(total, 1)),
        "micro_top5": float(sum(r["gt_top5_hit"] for r in pred_rows) / max(total, 1)),
        "mean_normalized_gt_rank": _mean(norm_ranks),
        "macro_mean_rank1": _mean([float(r["gt_top1_hit_rate"]) for r in by_class_rows]),
        "macro_mean_top5": _mean([float(r["gt_top5_hit_rate"]) for r in by_class_rows]),
    }
    _write_csv(out_dir / f"eval_{stage}_row_predictions.csv", pred_rows)
    _write_csv(out_dir / f"eval_{stage}_by_class.csv", by_class_rows)
    _write_json(out_dir / f"eval_{stage}_summary.json", summary)
    projector.train()
    return summary


def _compare_by_class(before_csv: Path, after_csv: Path, out_csv: Path) -> Dict[str, Any]:
    before = {str(row.get("raw_id")): row for row in _read_csv(before_csv)}
    after = {str(row.get("raw_id")): row for row in _read_csv(after_csv)}
    rows: List[Dict[str, Any]] = []
    improved = degraded = same = 0
    for rid in sorted(set(before) | set(after), key=lambda x: int(float(x)) if str(x).strip() else 0):
        b = before.get(rid, {})
        a = after.get(rid, {})
        b1 = _as_float(b.get("gt_top1_hit_rate"), 0.0)
        a1 = _as_float(a.get("gt_top1_hit_rate"), 0.0)
        delta = a1 - b1
        if delta > 1e-9:
            improved += 1
        elif delta < -1e-9:
            degraded += 1
        else:
            same += 1
        rows.append({
            "raw_id": rid,
            "before_top1": b1,
            "after_top1": a1,
            "delta_top1": delta,
            "before_mean_norm_rank": _as_float(b.get("mean_normalized_gt_rank"), 0.0),
            "after_mean_norm_rank": _as_float(a.get("mean_normalized_gt_rank"), 0.0),
            "delta_mean_norm_rank": _as_float(a.get("mean_normalized_gt_rank"), 0.0) - _as_float(b.get("mean_normalized_gt_rank"), 0.0),
            "gt_count": a.get("gt_count", b.get("gt_count", "")),
        })
    _write_csv(out_csv, rows)
    return {"class_count": len(rows), "improved_top1_classes": improved, "degraded_top1_classes": degraded, "same_top1_classes": same}


def _train(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else _default_output_root(run_root, str(args.dataset_name), str(args.mode))
    output_root.mkdir(parents=True, exist_ok=True)
    train_dir = output_root / "train" / "residual_gated_gtclean"
    train_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(str(args.device))

    manifest_csv = Path(args.manifest_csv).expanduser().resolve() if args.manifest_csv else _default_manifest_path(run_root, str(args.dataset_name))
    row_gap_csv = Path(args.row_gap_csv).expanduser().resolve() if args.row_gap_csv else _default_row_gap_path(repo_root, str(args.dataset_name))
    if not manifest_csv.is_file():
        raise FileNotFoundError(f"manifest_csv not found: {manifest_csv}")
    if not row_gap_csv.is_file():
        raise FileNotFoundError(f"row_gap_csv not found: {row_gap_csv}")

    data = _prepare_data(args, output_root)
    train_rows, manifest_summary = _load_manifest_rows(manifest_csv, mode=str(args.mode), example_by_tid=data.example_by_tid)
    eval_rows, eval_row_summary = _load_row_gap_eval_rows(
        row_gap_csv,
        example_by_tid=data.example_by_tid,
        raw_to_text_idx=data.raw_to_text_idx,
        max_rows=int(args.max_eval_rows),
        seed=int(args.seed),
    )
    if not eval_rows:
        raise RuntimeError("no eval rows available after joining row_gap_csv to materialized examples")

    text_tensor = torch.from_numpy(np.asarray(data.text_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
    projector_cfg = ProjectorConfig()
    projector = Projector(projector_cfg).to(device)
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(args.t_dis_init) - 1.0e-4, 1.0e-6)), device=device, dtype=torch.float32))

    resolved_ckpt = _auto_find_checkpoint(args)
    checkpoint_summary = _load_checkpoint_if_requested(projector, theta_t, resolved_ckpt, device)

    base_text_indices = torch.tensor([int(data.raw_to_text_idx[int(x)]) for x in data.base_ids], device=device, dtype=torch.long)
    base_raw_to_col = {int(rid): idx for idx, rid in enumerate(data.base_ids)}

    setup = {
        "timestamp": _now(),
        "mode": str(args.mode),
        "run_root": str(run_root),
        "output_root": str(output_root),
        "manifest_csv": str(manifest_csv),
        "row_gap_csv": str(row_gap_csv),
        "device": str(device),
        "checkpoint": checkpoint_summary,
        "materialization_summary": data.materialization_summary,
        "manifest_summary": manifest_summary,
        "eval_row_summary": eval_row_summary,
        "base_text_class_count": int(len(data.base_ids)),
    }
    _write_json(output_root / "pilot_setup.json", setup)

    before = _evaluate_rows(
        stage="before",
        rows=eval_rows,
        example_by_tid=data.example_by_tid,
        projector=projector,
        text_tensor=text_tensor,
        base_ids=data.base_ids,
        raw_to_text_idx=data.raw_to_text_idx,
        theta_t=theta_t,
        device=device,
        batch_size=int(args.eval_batch_size),
        out_dir=output_root / "analysis",
    )

    train_log_path = train_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()

    optimizer = torch.optim.AdamW([*projector.parameters(), theta_t], lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    train_summary: Dict[str, Any] = {"trained": False, "epochs": 0}
    if str(args.mode) != "eval_only":
        if not train_rows:
            raise RuntimeError(f"mode={args.mode} requested training but selected no manifest rows")
        rng = random.Random(int(args.seed))
        global_step = 0
        losses: List[float] = []
        for epoch in _iter_progress(range(int(args.epochs)), enabled=bool(args.show_progress), desc=f"{args.mode} epochs"):
            rows_epoch = list(train_rows)
            rng.shuffle(rows_epoch)
            epoch_losses: List[float] = []
            epoch_counts = Counter()
            for start in range(0, len(rows_epoch), int(args.batch_size_rows)):
                batch = rows_epoch[start:start + int(args.batch_size_rows)]
                optimizer.zero_grad(set_to_none=True)
                text_proj_base = _base_text_projection(projector, text_tensor, base_text_indices)
                loss, stats = _loss_on_batch(
                    rows=batch,
                    example_by_tid=data.example_by_tid,
                    base_ids=data.base_ids,
                    base_raw_to_col=base_raw_to_col,
                    text_proj_base=text_proj_base,
                    theta_t=theta_t,
                    device=device,
                    soft_label_smoothing=float(args.soft_label_smoothing),
                    proto_loss_weight=float(args.prototype_loss_weight),
                )
                loss.backward()
                if float(args.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_([*projector.parameters(), theta_t], max_norm=float(args.grad_clip_norm))
                optimizer.step()
                global_step += 1
                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)
                epoch_losses.append(loss_value)
                epoch_counts.update({k: int(v) for k, v in stats.items() if k.endswith("_rows") or k == "batch_rows"})
                _append_jsonl(train_log_path, {
                    "timestamp": _now(),
                    "epoch": int(epoch) + 1,
                    "global_step": int(global_step),
                    **stats,
                })
            epoch_row = {
                "timestamp": _now(),
                "row_type": "epoch_summary",
                "epoch": int(epoch) + 1,
                "loss_mean": _mean(epoch_losses),
                "loss_last": float(epoch_losses[-1]) if epoch_losses else 0.0,
                "epoch_batches": int(math.ceil(len(rows_epoch) / max(int(args.batch_size_rows), 1))),
                **{f"epoch_{k}": int(v) for k, v in epoch_counts.items()},
            }
            _append_jsonl(train_log_path, epoch_row)
            if bool(args.print_epoch_summary):
                print(json.dumps(epoch_row, ensure_ascii=False))
        train_summary = {
            "trained": True,
            "epochs": int(args.epochs),
            "global_step": int(global_step),
            "loss_mean": _mean(losses),
            "loss_last": float(losses[-1]) if losses else 0.0,
            "train_row_count": int(len(train_rows)),
        }

    after = _evaluate_rows(
        stage="after",
        rows=eval_rows,
        example_by_tid=data.example_by_tid,
        projector=projector,
        text_tensor=text_tensor,
        base_ids=data.base_ids,
        raw_to_text_idx=data.raw_to_text_idx,
        theta_t=theta_t,
        device=device,
        batch_size=int(args.eval_batch_size),
        out_dir=output_root / "analysis",
    )
    cmp_summary = _compare_by_class(
        output_root / "analysis" / "eval_before_by_class.csv",
        output_root / "analysis" / "eval_after_by_class.csv",
        output_root / "analysis" / "eval_before_after_by_class_delta.csv",
    )

    ckpt_path = train_dir / "residual_gated_gtclean_last.pth"
    torch.save({
        "pipeline": "residual_gated_gtclean_pilot",
        "mode": str(args.mode),
        "epoch": int(args.epochs) if str(args.mode) != "eval_only" else 0,
        "text_projector_state_dict": projector.state_dict(),
        "text_projector_config": {
            "input_dim": int(projector_cfg.input_dim),
            "hidden_dim": int(projector_cfg.hidden_dim),
            "output_dim": int(projector_cfg.output_dim),
            "dropout": float(projector_cfg.dropout),
            "use_layernorm": bool(projector_cfg.use_layernorm),
        },
        "theta_T": float(theta_t.detach().cpu().item()),
        "seed": int(args.seed),
        "manifest_csv": str(manifest_csv),
        "row_gap_csv": str(row_gap_csv),
        "checkpoint_init": checkpoint_summary,
    }, ckpt_path)

    final_summary = {
        "status": "PASS",
        "mode": str(args.mode),
        "output_root": str(output_root),
        "setup": setup,
        "train_summary": train_summary,
        "eval_before": before,
        "eval_after": after,
        "eval_delta": {
            "micro_top1_delta": float(after.get("micro_top1", 0.0) - before.get("micro_top1", 0.0)),
            "macro_mean_rank1_delta": float(after.get("macro_mean_rank1", 0.0) - before.get("macro_mean_rank1", 0.0)),
            "mean_normalized_gt_rank_delta": float(after.get("mean_normalized_gt_rank", 0.0) - before.get("mean_normalized_gt_rank", 0.0)),
            **cmp_summary,
        },
        "checkpoint_path": str(ckpt_path),
        "outputs": {
            "train_log": str(train_log_path),
            "eval_before_summary": str(output_root / "analysis" / "eval_before_summary.json"),
            "eval_after_summary": str(output_root / "analysis" / "eval_after_summary.json"),
            "eval_before_after_by_class_delta": str(output_root / "analysis" / "eval_before_after_by_class_delta.csv"),
        },
        "interpretation": {
            "primary_gate": "Check macro_mean_rank1_delta and mean_normalized_gt_rank_delta. Eval-only may be unchanged; train modes should improve if the manifest is useful.",
            "does_not_modify_control_plane": True,
            "gpu_friendly": True,
            "rowwise_large_matrix_compute": False,
        },
    }
    _write_json(output_root / "final_summary.json", final_summary)
    lines = [
        "# Residual-Gated GT-Clean Pilot TAKEOVER",
        "",
        f"- status: {final_summary['status']}",
        f"- mode: {args.mode}",
        f"- output_root: {output_root}",
        f"- trained: {train_summary.get('trained')}",
        f"- train_rows: {train_summary.get('train_row_count', 0)}",
        f"- before micro_top1: {before.get('micro_top1', 0.0):.6f}",
        f"- after micro_top1: {after.get('micro_top1', 0.0):.6f}",
        f"- delta micro_top1: {final_summary['eval_delta']['micro_top1_delta']:.6f}",
        f"- before macro_mean_rank1: {before.get('macro_mean_rank1', 0.0):.6f}",
        f"- after macro_mean_rank1: {after.get('macro_mean_rank1', 0.0):.6f}",
        f"- delta macro_mean_rank1: {final_summary['eval_delta']['macro_mean_rank1_delta']:.6f}",
        f"- before mean_norm_rank: {before.get('mean_normalized_gt_rank', 0.0):.6f}",
        f"- after mean_norm_rank: {after.get('mean_normalized_gt_rank', 0.0):.6f}",
        f"- delta mean_norm_rank: {final_summary['eval_delta']['mean_normalized_gt_rank_delta']:.6f}",
        "",
        "## Outputs",
        "- final_summary.json",
        "- train/residual_gated_gtclean/train_log.jsonl",
        "- analysis/eval_before_summary.json",
        "- analysis/eval_after_summary.json",
        "- analysis/eval_before_after_by_class_delta.csv",
        "- train/residual_gated_gtclean/residual_gated_gtclean_last.pth",
    ]
    (output_root / "RESIDUAL_GATED_GTCLEAN_PILOT_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2, default=str))
    return final_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Residual-gated GT-clean balanced training pilot")
    p.add_argument("--run_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--mode", required=True, choices=("eval_only", "hard_ce", "hard_soft_proto"))
    p.add_argument("--output_root", default="")
    p.add_argument("--manifest_csv", default="")
    p.add_argument("--row_gap_csv", default="")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--init_checkpoint", default="", help="Checkpoint path, empty for random init, or 'auto' to read the oracle-gap summary when available.")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size_rows", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--t_dis_init", type=float, default=0.07)
    p.add_argument("--soft_label_smoothing", type=float, default=0.2)
    p.add_argument("--prototype_loss_weight", type=float, default=0.25)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--max_eval_rows", type=int, default=0, help="0 means evaluate all joined row-gap rows.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--show_progress", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    p.add_argument("--print_epoch_summary", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    args = p.parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if not str(args.annotation_json).strip():
        args.annotation_json = str(Path(args.asset_root).expanduser().resolve() / "dataset" / "LV-VIS" / "annotations" / "train_instances.json")
    if not str(args.split_json).strip():
        args.split_json = str(repo_root / "package" / "reference" / "lvvis_official_base_novel_split.json")
    return args


def main() -> int:
    _train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
