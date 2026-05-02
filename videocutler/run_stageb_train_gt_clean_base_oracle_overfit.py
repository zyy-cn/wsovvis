#!/usr/bin/env python3
"""GT-clean base oracle overfit trainer.

Purpose
-------
This entry is a capacity sanity test, not a weakly-supervised method. It removes
proposal noise and weak assignment ambiguity by training on GT-carrier trajectories
with an instance-level GT class target. It answers one question:

    Can the current text projector + text prototypes + fixed GT trajectory features
    overfit base semantic labels in a clean setting?

Boundary
--------
* GT upper-bound trajectories only;
* official base raw ids only by default;
* no VideoCutLER/mainline trajectories;
* no Y-prime, extra mining, unknown, certificates, EMA, absorber, or demand floor;
* GT identity is used intentionally because this is an oracle capacity test.

The checkpoint is intentionally compatible with the existing G8 eval bridge:
text_projector_state_dict + text_projector_config + theta_T are written.
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
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _bootstrap_repo_root_for_direct_cli() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


_BOOT_REPO_ROOT = _bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab  # noqa: E402
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig  # noqa: E402

# Reuse the already-established GT-fullY clean materialization helpers. This keeps
# this oracle test aligned with the same GT carrier and full-Y assets used by the
# current NoHub experiments.
import videocutler.run_stageb_train_gt_full_y_clean as clean  # noqa: E402


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for row in rows:
            h.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


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


def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _mean(vals: Sequence[float]) -> float:
    return float(np.mean(np.asarray(list(vals), dtype=np.float64))) if vals else 0.0


def _percentile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(list(vals), dtype=np.float64), float(q)))


def _normalize_np(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(arr))
    if denom <= 1.0e-12:
        return arr.astype(np.float32, copy=False)
    return (arr / denom).astype(np.float32, copy=False)


def _inverse_softplus(value: float) -> float:
    target = max(float(value), 1.0e-6)
    return float(math.log(math.expm1(target)))


def _compute_t_dis(theta_t: torch.nn.Parameter) -> torch.Tensor:
    return F.softplus(theta_t) + 1.0e-4


def _iter_progress(iterable: Iterable[Any], *, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    return -(p * torch.log(torch.clamp(p, min=1.0e-12))).sum(dim=-1)


def _row_id_candidates(row: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("trajectory_id", "join_key", "carrier_id", "sample_id", "track_id", "id"):
        val = row.get(key)
        if val is not None:
            out.append(str(val))
    # Deduplicate while preserving order.
    seen: Set[str] = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _extract_raw_id_from_record(row: Mapping[str, Any]) -> Optional[int]:
    """Robustly extract a raw LV-VIS class id from known sidecar/binding schemas."""
    direct_keys = (
        "matched_gt_raw_id_canonical",
        "matched_gt_raw_id",
        "gt_raw_id",
        "gt_category_raw_id",
        "raw_category_id",
        "category_raw_id",
        "category_id",
        "raw_id",
        "label_raw_id",
        "gt_label_raw_id",
        "target_raw_id",
    )
    for key in direct_keys:
        val = _as_int(row.get(key))
        if val is not None:
            return int(val)

    # Some GT-carrier exports store zero-based pred_label_raw. Prior project notes
    # established raw LV-VIS category id = pred_label_raw + 1 for this binding.
    pred_label = _as_int(row.get("pred_label_raw"))
    if pred_label is not None:
        return int(pred_label) + 1

    nested_keys = ("gt", "match", "matched_gt", "category", "label")
    for nk in nested_keys:
        v = row.get(nk)
        if isinstance(v, Mapping):
            inner = _extract_raw_id_from_record(v)
            if inner is not None:
                return int(inner)
    return None


def _candidate_identity_binding_paths(repo_root: Path, asset_root: Path, dataset_name: str) -> List[Path]:
    rels = [
        Path("carrier_bank_gt") / dataset_name / "gt_carrier_identity_binding.jsonl",
        Path("carrier_bank_gt") / dataset_name / "identity_binding.jsonl",
        Path("gt_sidecar_bank") / dataset_name / "gt_upper_bound" / "trajectory_gt_match_train_gt_upper_bound.jsonl",
        Path("gt_sidecar_bank") / dataset_name / "gt_upper_bound" / "trajectory_gt_match_train_mainline.jsonl",
        Path("gt_sidecar_bank") / dataset_name / "mainline" / "trajectory_gt_match_train_mainline.jsonl",
    ]
    roots = [repo_root, asset_root]
    out: List[Path] = []
    seen: Set[str] = set()
    for root in roots:
        for rel in rels:
            p = (root / rel).expanduser()
            key = str(p)
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def _load_identity_binding(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, Mapping):
                continue
            raw_id = _extract_raw_id_from_record(row)
            if raw_id is None:
                continue
            for rid in _row_id_candidates(row):
                mapping[str(rid)] = int(raw_id)
    return mapping


def _load_identity_bindings(paths: Sequence[Path]) -> Tuple[Dict[str, int], List[str], Dict[str, Any]]:
    merged: Dict[str, int] = {}
    used: List[str] = []
    stats: Dict[str, Any] = {"path_stats": []}
    for p in paths:
        m = _load_identity_binding(p)
        if not m:
            continue
        before = len(merged)
        merged.update(m)
        used.append(str(p))
        stats["path_stats"].append({"path": str(p), "rows_loaded": len(m), "new_keys": len(merged) - before})
    stats["merged_key_count"] = len(merged)
    return merged, used, stats


def _extract_target_for_example(ex: Mapping[str, Any], binding: Mapping[str, int]) -> Optional[int]:
    raw = _extract_raw_id_from_record(ex)
    if raw is not None:
        return int(raw)
    for rid in _row_id_candidates(ex):
        if str(rid) in binding:
            return int(binding[str(rid)])
    return None


def _class_name_map_from_annotation(annotation_json: Path) -> Dict[int, str]:
    try:
        obj = json.loads(annotation_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cats = obj.get("categories", []) if isinstance(obj, Mapping) else []
    out: Dict[int, str] = {}
    for c in cats:
        if not isinstance(c, Mapping):
            continue
        rid = _as_int(c.get("id", c.get("raw_id", c.get("category_id"))))
        if rid is None:
            continue
        out[int(rid)] = str(c.get("name", c.get("class_name", rid)))
    return out


def _build_candidate_ids(
    *,
    scope: str,
    clip_id: int,
    target_raw_id: int,
    clip_y_base: Mapping[int, Set[int]],
    base_ids: Set[int],
    text_vocab_ids: Sequence[int],
    raw_to_idx: Mapping[int, int],
) -> List[int]:
    s = str(scope)
    if s == "clip_y_base":
        cand = set(clip_y_base.get(int(clip_id), set()))
    elif s == "base_vocab":
        cand = set(base_ids)
    elif s == "full_vocab":
        cand = set(int(x) for x in text_vocab_ids)
    elif s == "gt_only_smoke":
        cand = {int(target_raw_id)}
    else:
        raise ValueError(f"unsupported candidate scope: {scope}")
    cand = {int(x) for x in cand if int(x) in raw_to_idx}
    if int(target_raw_id) not in cand:
        return []
    return sorted(cand)


@dataclass
class ExamplePack:
    examples: List[Dict[str, Any]]
    clip_y_base: Dict[int, Set[int]]
    base_ids: Set[int]
    materialization_summary: Dict[str, Any]
    identity_binding_paths_used: List[str]
    identity_binding_stats: Dict[str, Any]
    target_attach_counters: Dict[str, int]


def _load_oracle_examples(args: argparse.Namespace) -> ExamplePack:
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    clean._bootstrap_asset_links(repo_root, asset_root)
    clean._bootstrap_asset_links(output_root, asset_root)

    examples, clip_y_base, base_ids, materialization_summary = clean._load_materialized_gt_examples(
        repo_root=repo_root,
        output_root=output_root,
        asset_root=asset_root,
        dataset_name=str(args.dataset_name),
        annotation_json=Path(args.annotation_json),
        split_json=Path(args.split_json),
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
        subset_fraction=args.subset_fraction,
        seed=int(args.seed),
    )

    binding_paths: List[Path]
    if str(args.gt_identity_binding_jsonl or "").strip():
        binding_paths = [Path(x).expanduser() for x in str(args.gt_identity_binding_jsonl).split(",") if str(x).strip()]
    else:
        binding_paths = _candidate_identity_binding_paths(repo_root, asset_root, str(args.dataset_name))
    binding, used, binding_stats = _load_identity_bindings(binding_paths)

    out: List[Dict[str, Any]] = []
    counters = Counter()
    for ex in examples:
        target = _extract_target_for_example(ex, binding)
        if target is None:
            counters["skip_no_target_raw_id"] += 1
            continue
        if int(target) not in base_ids:
            counters["skip_target_not_base"] += 1
            continue
        clip_id = _as_int(ex.get("clip_id"))
        if clip_id is None:
            counters["skip_no_clip_id"] += 1
            continue
        yb = set(clip_y_base.get(int(clip_id), set()))
        if bool(args.require_target_in_clip_y_base) and int(target) not in yb:
            counters["skip_target_not_in_clip_y_base"] += 1
            continue
        row = dict(ex)
        row["target_raw_id"] = int(target)
        row["target_source"] = "identity_binding_or_embedded_gt"
        out.append(row)
        counters["kept"] += 1

    if not out:
        raise RuntimeError(
            "no oracle examples with base target_raw_id. "
            "Pass --gt_identity_binding_jsonl or inspect GT-carrier schema."
        )
    materialization_summary = dict(materialization_summary)
    materialization_summary.update({
        "oracle_target_attach_counters": dict(counters),
        "oracle_example_count": int(len(out)),
        "identity_binding_paths_used": list(used),
        "identity_binding_stats": binding_stats,
    })
    return ExamplePack(out, dict(clip_y_base), set(base_ids), materialization_summary, list(used), dict(binding_stats), dict(counters))


def _iter_minibatches(n: int, batch_size: int, *, rng: random.Random) -> Iterator[List[int]]:
    idxs = list(range(int(n)))
    rng.shuffle(idxs)
    bs = max(1, int(batch_size))
    for i in range(0, len(idxs), bs):
        yield idxs[i:i + bs]


def _score_rows(
    *,
    examples: Sequence[Mapping[str, Any]],
    projector: Projector,
    theta_t: torch.nn.Parameter,
    text_vocab_tensor: torch.Tensor,
    text_vocab_ids: Sequence[int],
    raw_to_idx: Mapping[int, int],
    clip_y_base: Mapping[int, Set[int]],
    base_ids: Set[int],
    candidate_scope: str,
    device: torch.device,
    class_name_map: Optional[Mapping[int, str]] = None,
    max_rows_out: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    projector.eval()
    rows_out: List[Dict[str, Any]] = []
    per_class: Dict[int, Dict[str, Any]] = {}
    ranks: List[int] = []
    norm_ranks: List[float] = []
    losses: List[float] = []
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    mrr_vals: List[float] = []
    cand_sizes: List[int] = []
    skipped = Counter()
    with torch.no_grad():
        text_proj_all = F.normalize(projector(text_vocab_tensor), p=2.0, dim=-1)
        temperature = _compute_t_dis(theta_t)
        for ex in examples:
            clip_id = int(ex["clip_id"])
            target = int(ex["target_raw_id"])
            candidates = _build_candidate_ids(
                scope=candidate_scope,
                clip_id=clip_id,
                target_raw_id=target,
                clip_y_base=clip_y_base,
                base_ids=base_ids,
                text_vocab_ids=text_vocab_ids,
                raw_to_idx=raw_to_idx,
            )
            if not candidates:
                skipped["target_not_in_candidate_scope"] += 1
                continue
            cand_idx = torch.tensor([raw_to_idx[int(x)] for x in candidates], device=device, dtype=torch.long)
            cand_text = text_proj_all[cand_idx]
            z = torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))).to(device=device, dtype=torch.float32).view(1, -1)
            z = F.normalize(z, p=2.0, dim=-1)
            scores = (z @ cand_text.t()).squeeze(0) / temperature
            target_pos = candidates.index(target)
            target_tensor = torch.tensor([target_pos], device=device, dtype=torch.long)
            loss = F.cross_entropy(scores.unsqueeze(0), target_tensor)
            order = torch.argsort(scores, descending=True).detach().cpu().numpy().astype(np.int64).tolist()
            rank = int(order.index(int(target_pos)) + 1)
            M = int(len(candidates))
            norm_rank = float((rank - 1) / max(M - 1, 1))
            top1 = int(rank == 1)
            top5 = int(rank <= 5)
            top10 = int(rank <= 10)
            top_raw = int(candidates[int(order[0])]) if order else -1
            ranks.append(rank)
            norm_ranks.append(norm_rank)
            losses.append(float(loss.detach().cpu().item()))
            cand_sizes.append(M)
            top1_hits += top1
            top5_hits += top5
            top10_hits += top10
            mrr_vals.append(float(1.0 / rank))
            pc = per_class.setdefault(target, {
                "raw_id": int(target),
                "class_name": str(class_name_map.get(int(target), "")) if class_name_map else "",
                "gt_count": 0,
                "rank_sum": 0.0,
                "norm_rank_sum": 0.0,
                "top1": 0,
                "top5": 0,
                "top10": 0,
                "candidate_size_sum": 0.0,
                "loss_sum": 0.0,
            })
            pc["gt_count"] += 1
            pc["rank_sum"] += float(rank)
            pc["norm_rank_sum"] += float(norm_rank)
            pc["top1"] += int(top1)
            pc["top5"] += int(top5)
            pc["top10"] += int(top10)
            pc["candidate_size_sum"] += float(M)
            pc["loss_sum"] += float(loss.detach().cpu().item())
            if max_rows_out > 0 and len(rows_out) < int(max_rows_out):
                rows_out.append({
                    "clip_id": int(clip_id),
                    "trajectory_id": str(ex.get("trajectory_id", "")),
                    "target_raw_id": int(target),
                    "target_class_name": str(class_name_map.get(int(target), "")) if class_name_map else "",
                    "candidate_scope": str(candidate_scope),
                    "candidate_size": int(M),
                    "gt_rank": int(rank),
                    "normalized_gt_rank": float(norm_rank),
                    "gt_top1_hit": bool(top1),
                    "gt_top5_hit": bool(top5),
                    "gt_top10_hit": bool(top10),
                    "top1_raw_id": int(top_raw),
                    "top1_class_name": str(class_name_map.get(int(top_raw), "")) if class_name_map else "",
                    "loss": float(loss.detach().cpu().item()),
                })
    n = len(ranks)
    summary = {
        "evaluated_gt_count": int(n),
        "skipped": dict(skipped),
        "candidate_scope": str(candidate_scope),
        "loss_mean": _mean(losses),
        "gt_top1_hit_rate": float(top1_hits / max(n, 1)),
        "gt_top5_hit_rate": float(top5_hits / max(n, 1)),
        "gt_top10_hit_rate": float(top10_hits / max(n, 1)),
        "mean_normalized_gt_rank": _mean(norm_ranks),
        "median_normalized_gt_rank": _percentile(norm_ranks, 50),
        "gt_rank_mean": _mean([float(x) for x in ranks]),
        "mrr": _mean(mrr_vals),
        "candidate_size_mean": _mean([float(x) for x in cand_sizes]),
        "temperature": float(_compute_t_dis(theta_t).detach().cpu().item()),
    }
    per_class_rows: List[Dict[str, Any]] = []
    for rid, pc in sorted(per_class.items()):
        c = max(int(pc["gt_count"]), 1)
        per_class_rows.append({
            "raw_id": int(rid),
            "class_name": str(pc.get("class_name", "")),
            "gt_count": int(pc["gt_count"]),
            "gt_top1_hit_rate": float(pc["top1"] / c),
            "gt_top5_hit_rate": float(pc["top5"] / c),
            "gt_top10_hit_rate": float(pc["top10"] / c),
            "mean_gt_rank": float(pc["rank_sum"] / c),
            "mean_normalized_gt_rank": float(pc["norm_rank_sum"] / c),
            "candidate_size_mean": float(pc["candidate_size_sum"] / c),
            "loss_mean": float(pc["loss_sum"] / c),
        })
    projector.train()
    return summary, per_class_rows, rows_out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: List[str] = []
    seen: Set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def train_oracle(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(str(args.device))

    pack = _load_oracle_examples(args)
    examples = pack.examples
    class_name_map = _class_name_map_from_annotation(Path(args.annotation_json))

    text_vocab_ids, _text_records, text_vocab_matrix = load_text_vocab(output_root)
    raw_to_idx = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
    text_vocab_tensor = torch.from_numpy(np.asarray(text_vocab_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
    projector_cfg = ProjectorConfig()
    projector = Projector(projector_cfg).to(device)
    projector.train()
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(args.t_dis_init) - 1.0e-4, 1.0e-6)), device=device, dtype=torch.float32))
    optim_params: List[torch.nn.Parameter] = list(projector.parameters())
    if not bool(args.freeze_temperature):
        optim_params.append(theta_t)
    optimizer = torch.optim.AdamW(optim_params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    train_dir = output_root / "train" / "prealign"
    runtime_metrics_path = train_dir / "runtime_metrics.jsonl"
    if runtime_metrics_path.exists():
        runtime_metrics_path.unlink()
    oracle_metrics_path = train_dir / "gt_clean_base_oracle_overfit_metrics.jsonl"
    if oracle_metrics_path.exists():
        oracle_metrics_path.unlink()

    train_start = datetime.now(timezone.utc).isoformat()
    global_step = 0
    batch_losses: List[float] = []
    epoch_summaries: List[Dict[str, Any]] = []

    # Initial zero-shot/cold-start measurement before optimization.
    eval_summary, per_class_rows, example_rows = _score_rows(
        examples=examples,
        projector=projector,
        theta_t=theta_t,
        text_vocab_tensor=text_vocab_tensor,
        text_vocab_ids=text_vocab_ids,
        raw_to_idx=raw_to_idx,
        clip_y_base=pack.clip_y_base,
        base_ids=pack.base_ids,
        candidate_scope=str(args.eval_candidate_scope),
        device=device,
        class_name_map=class_name_map,
        max_rows_out=int(args.max_example_rows_out),
    )
    initial_summary = dict(eval_summary)
    initial_summary.update({"row_type": "eval_summary", "epoch": 0, "phase": "initial"})
    _append_jsonl(runtime_metrics_path, initial_summary)
    _append_jsonl(oracle_metrics_path, initial_summary)

    for epoch_idx in _iter_progress(range(int(args.epochs)), enabled=bool(args.show_progress), desc="gt-clean oracle epochs", leave=True):
        rng = random.Random(int(args.seed) + int(epoch_idx))
        epoch_losses: List[float] = []
        epoch_entropy: List[float] = []
        epoch_cand_sizes: List[float] = []
        epoch_skipped = Counter()
        for batch_ids in _iter_minibatches(len(examples), int(args.batch_size), rng=rng):
            optimizer.zero_grad(set_to_none=True)
            text_proj_all = F.normalize(projector(text_vocab_tensor), p=2.0, dim=-1)
            temperature = _compute_t_dis(theta_t)
            batch_loss_terms: List[torch.Tensor] = []
            for idx in batch_ids:
                ex = examples[int(idx)]
                clip_id = int(ex["clip_id"])
                target = int(ex["target_raw_id"])
                candidates = _build_candidate_ids(
                    scope=str(args.train_candidate_scope),
                    clip_id=clip_id,
                    target_raw_id=target,
                    clip_y_base=pack.clip_y_base,
                    base_ids=pack.base_ids,
                    text_vocab_ids=text_vocab_ids,
                    raw_to_idx=raw_to_idx,
                )
                if not candidates:
                    epoch_skipped["target_not_in_train_candidate_scope"] += 1
                    continue
                cand_idx = torch.tensor([raw_to_idx[int(x)] for x in candidates], device=device, dtype=torch.long)
                cand_text = text_proj_all[cand_idx]
                z = torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))).to(device=device, dtype=torch.float32).view(1, -1)
                z = F.normalize(z, p=2.0, dim=-1)
                scores = (z @ cand_text.t()) / temperature
                target_pos = int(candidates.index(target))
                target_tensor = torch.tensor([target_pos], device=device, dtype=torch.long)
                loss = F.cross_entropy(scores, target_tensor)
                if float(args.entropy_penalty) != 0.0:
                    loss = loss + float(args.entropy_penalty) * _softmax_entropy(scores).mean()
                batch_loss_terms.append(loss)
                epoch_entropy.append(float(_softmax_entropy(scores.detach()).mean().cpu().item()))
                epoch_cand_sizes.append(float(len(candidates)))
            if not batch_loss_terms:
                continue
            loss_batch = torch.stack(batch_loss_terms).mean()
            loss_batch.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(optim_params, max_norm=float(args.grad_clip_norm))
            optimizer.step()
            global_step += 1
            bval = float(loss_batch.detach().cpu().item())
            batch_losses.append(bval)
            epoch_losses.append(bval)
            _append_jsonl(runtime_metrics_path, {
                "row_type": "microbatch",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "epoch": int(epoch_idx) + 1,
                "global_step": int(global_step),
                "loss": bval,
                "batch_size_effective": int(len(batch_loss_terms)),
                "candidate_size_mean": _mean(epoch_cand_sizes[-len(batch_loss_terms):]),
                "temperature": float(_compute_t_dis(theta_t).detach().cpu().item()),
            })

        # Full train-set evaluation each epoch. This is intentionally done on the
        # same split because this audit is an overfit capacity test.
        eval_summary, per_class_rows, example_rows = _score_rows(
            examples=examples,
            projector=projector,
            theta_t=theta_t,
            text_vocab_tensor=text_vocab_tensor,
            text_vocab_ids=text_vocab_ids,
            raw_to_idx=raw_to_idx,
            clip_y_base=pack.clip_y_base,
            base_ids=pack.base_ids,
            candidate_scope=str(args.eval_candidate_scope),
            device=device,
            class_name_map=class_name_map,
            max_rows_out=int(args.max_example_rows_out),
        )
        epoch_summary = {
            "row_type": "epoch_summary",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": "prealign",
            "protocol": "oracle_supervised_gt_class",
            "epoch": int(epoch_idx) + 1,
            "global_step": int(global_step),
            "loss_mean_epoch_train_batches": _mean(epoch_losses),
            "loss_last_epoch_train_batch": float(epoch_losses[-1]) if epoch_losses else 0.0,
            "entropy_mean_epoch": _mean(epoch_entropy),
            "candidate_size_mean_epoch_train": _mean(epoch_cand_sizes),
            "skipped_epoch": dict(epoch_skipped),
            **eval_summary,
        }
        epoch_summaries.append(epoch_summary)
        _append_jsonl(runtime_metrics_path, epoch_summary)
        _append_jsonl(oracle_metrics_path, epoch_summary)
        if bool(args.print_epoch_summary):
            print(json.dumps({
                "epoch": epoch_summary["epoch"],
                "train_loss": epoch_summary["loss_mean_epoch_train_batches"],
                "eval_top1": epoch_summary["gt_top1_hit_rate"],
                "eval_rank": epoch_summary["mean_normalized_gt_rank"],
                "eval_loss": epoch_summary["loss_mean"],
                "temperature": epoch_summary["temperature"],
            }, ensure_ascii=False))

    ckpt_dir = train_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "prealign_last.pth"
    torch.save({
        "stage_id": "prealign",
        "epoch": int(args.epochs),
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
        "protocol": "oracle_supervised_gt_class",
        "pipeline": "gt_clean_base_overfit",
        "label_source": "instance_gt_raw_id_oracle",
        "trajectory_source_branch": "gt_upper_bound",
        "global_step": int(global_step),
        "train_candidate_scope": str(args.train_candidate_scope),
        "eval_candidate_scope": str(args.eval_candidate_scope),
    }, ckpt_path)

    final_eval, per_class_rows, example_rows = _score_rows(
        examples=examples,
        projector=projector,
        theta_t=theta_t,
        text_vocab_tensor=text_vocab_tensor,
        text_vocab_ids=text_vocab_ids,
        raw_to_idx=raw_to_idx,
        clip_y_base=pack.clip_y_base,
        base_ids=pack.base_ids,
        candidate_scope=str(args.eval_candidate_scope),
        device=device,
        class_name_map=class_name_map,
        max_rows_out=int(args.max_example_rows_out),
    )
    _write_csv(train_dir / "per_class_oracle_overfit.csv", per_class_rows)
    _write_jsonl(train_dir / "oracle_overfit_example_rows.jsonl", example_rows)

    # Emit responsibility records compatible with existing attribution compare.
    response_rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        projector.eval()
        text_proj_all = F.normalize(projector(text_vocab_tensor), p=2.0, dim=-1)
        temperature = _compute_t_dis(theta_t)
        for ex in examples:
            clip_id = int(ex["clip_id"])
            target = int(ex["target_raw_id"])
            candidates = _build_candidate_ids(
                scope=str(args.eval_candidate_scope),
                clip_id=clip_id,
                target_raw_id=target,
                clip_y_base=pack.clip_y_base,
                base_ids=pack.base_ids,
                text_vocab_ids=text_vocab_ids,
                raw_to_idx=raw_to_idx,
            )
            if not candidates:
                continue
            cand_idx = torch.tensor([raw_to_idx[int(x)] for x in candidates], device=device, dtype=torch.long)
            cand_text = text_proj_all[cand_idx]
            z = torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))).to(device=device, dtype=torch.float32).view(1, -1)
            z = F.normalize(z, p=2.0, dim=-1)
            scores = ((z @ cand_text.t()).squeeze(0) / temperature)
            probs = torch.softmax(scores, dim=0).detach().cpu().numpy().astype(np.float64).tolist()
            response_rows.append({
                "dataset_name": str(args.dataset_name),
                "clip_id": int(clip_id),
                "video_id": int(ex.get("video_id", clip_id)),
                "trajectory_id": str(ex.get("trajectory_id", "")),
                "target_raw_id": int(target),
                "candidate_ids_known": list(candidates),
                "candidate_ids_extra": [],
                "candidate_ids_null": [],
                "candidate_scope_policy": {"policy": "GT_CLEAN_BASE_ORACLE_OVERFIT", "label_source": "instance_gt_raw_id_oracle", "eval_candidate_scope": str(args.eval_candidate_scope)},
                "candidate_demand_by_raw_id": {str(int(x)): 1.0 for x in candidates},
                "candidate_kind_by_raw_id": {str(int(x)): 1 for x in candidates},
                "unknown_disabled": True,
                "training_semantics": "gt_clean_base_oracle_supervised_overfit",
                "stage_id": "prealign",
                "r_init": {},
                "r_final": {str(int(raw_id)): float(prob) for raw_id, prob in zip(candidates, probs)},
                "join_key": str(ex.get("trajectory_id", "")),
            })
    _write_jsonl(train_dir / "responsibility_records.jsonl", response_rows)

    stage_summary = {
        "stage_id": "prealign",
        "pipeline": "gt_clean_base_overfit",
        "protocol": "oracle_supervised_gt_class",
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": "gt_upper_bound",
        "label_source": "instance_gt_raw_id_oracle",
        "epochs": int(args.epochs),
        "train_candidate_scope": str(args.train_candidate_scope),
        "eval_candidate_scope": str(args.eval_candidate_scope),
        "initial_eval": initial_summary,
        "final_eval": final_eval,
        "loss_mean": _mean(batch_losses),
        "loss_last": float(batch_losses[-1]) if batch_losses else 0.0,
        "global_step": int(global_step),
        "trainable_example_count": int(len(examples)),
        "checkpoint_last": str((Path("train") / "prealign" / "checkpoints" / "prealign_last.pth").as_posix()),
        "runtime_metrics": str((Path("train") / "prealign" / "runtime_metrics.jsonl").as_posix()),
        "oracle_metrics": str((Path("train") / "prealign" / "gt_clean_base_oracle_overfit_metrics.jsonl").as_posix()),
        "per_class_oracle_overfit": str((Path("train") / "prealign" / "per_class_oracle_overfit.csv").as_posix()),
        "materialization_summary": pack.materialization_summary,
        "identity_binding_paths_used": pack.identity_binding_paths_used,
        "identity_binding_stats": pack.identity_binding_stats,
        "target_attach_counters": pack.target_attach_counters,
        "boundary": "capacity audit only; GT target is used intentionally; not a weak-supervised result",
    }
    _write_json(train_dir / "stage_summary.json", stage_summary)
    _write_json(train_dir / "train_state.json", {
        "stage_id": "prealign",
        "epoch": int(args.epochs),
        "checkpoint_last": str((Path("train") / "prealign" / "checkpoints" / "prealign_last.pth").as_posix()),
        "checkpoint_selected": str((Path("train") / "prealign" / "checkpoints" / "prealign_last.pth").as_posix()),
        "selected_for_infer": True,
        "selected_for_infer_authority": "gt_clean_base_oracle_overfit",
        "pipeline": "gt_clean_base_overfit",
        "protocol": "oracle_supervised_gt_class",
        "global_step": int(global_step),
    })
    pipeline_summary = {
        "status": "PASS",
        "exp_name": str(args.exp_name),
        "pipeline": "gt_clean_base_overfit",
        "protocol": "oracle_supervised_gt_class",
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": "gt_upper_bound",
        "label_source": "instance_gt_raw_id_oracle",
        "repo_root": str(repo_root),
        "asset_root": str(asset_root),
        "output_root": str(output_root),
        "train_started_at": train_start,
        "train_finished_at": datetime.now(timezone.utc).isoformat(),
        "selected_checkpoint_path": str(ckpt_path),
        "initial_eval": initial_summary,
        "final_eval": final_eval,
        "stages": {"prealign": stage_summary},
        "interpretation": "If this oracle arm cannot strongly overfit, the bottleneck is projector/text/feature capacity rather than weak assignment or novel transfer.",
    }
    _write_json(output_root / "train" / "pipeline_train_summary.json", pipeline_summary)
    _write_json(output_root / "GT_CLEAN_BASE_ORACLE_OVERFIT_TAKEOVER.json", pipeline_summary)
    print(output_root / "train" / "pipeline_train_summary.json")
    return pipeline_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GT-clean base oracle overfit capacity trainer.")
    p.add_argument("--exp_name", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--gt_identity_binding_jsonl", default="", help="Optional comma-separated GT-carrier identity binding jsonl paths. If omitted, common repo/asset locations are auto-detected.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--t_dis_init", type=float, default=0.07)
    p.add_argument("--freeze_temperature", action="store_true")
    p.add_argument("--grad_clip_norm", type=float, default=5.0)
    p.add_argument("--entropy_penalty", type=float, default=0.0)
    p.add_argument("--train_candidate_scope", choices=("clip_y_base", "base_vocab", "full_vocab", "gt_only_smoke"), default="base_vocab")
    p.add_argument("--eval_candidate_scope", choices=("clip_y_base", "base_vocab", "full_vocab", "gt_only_smoke"), default="base_vocab")
    p.add_argument("--require_target_in_clip_y_base", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    p.add_argument("--show_progress", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    p.add_argument("--print_epoch_summary", type=lambda x: str(x).lower() not in {"0", "false", "no"}, default=True)
    p.add_argument("--max_example_rows_out", type=int, default=2000)
    return p.parse_args()


def main() -> int:
    train_oracle(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
