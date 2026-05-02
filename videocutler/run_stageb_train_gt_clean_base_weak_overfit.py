#!/usr/bin/env python3
"""GT-clean weak full-Y base overfit trainer.

Purpose
-------
This entry is a weak-assignment capacity audit. It uses GT upper-bound
trajectories to remove proposal noise, but *does not* use instance-level GT
class targets for training. GT identity is attached only to define the clean
base evaluation denominator and to report attribution metrics.

It answers one question:

    Can a video/clip-level full-Y weak objective release the base-class
    fitting capacity that the oracle supervised GT-clean overfit already proved
    exists?

Boundary
--------
* GT upper-bound trajectories only;
* official base rows only for clean evaluation denominator;
* weak training target is the clip/video full-Y base candidate set, not the
  instance GT class;
* no VideoCutLER/mainline trajectories;
* no Y-prime, extra mining, unknown, certificate, EMA, absorber, or demand floor;
* optional nohub row weighting is row-level only and uses local positive-set
  confidence, not GT correctness.

The checkpoint is compatible with the existing G8 eval bridge:
text_projector_state_dict + text_projector_config + theta_T are written.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

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
import videocutler.run_stageb_train_gt_clean_base_oracle_overfit as oracle  # noqa: E402


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def _mean(vals: Sequence[float]) -> float:
    return float(np.mean(np.asarray(list(vals), dtype=np.float64))) if vals else 0.0


def _percentile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(list(vals), dtype=np.float64), float(q)))


def _iter_progress(iterable: Iterable[Any], *, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _iter_minibatches(n: int, batch_size: int, *, rng: random.Random) -> Iterator[List[int]]:
    idxs = list(range(int(n)))
    rng.shuffle(idxs)
    bs = max(1, int(batch_size))
    for i in range(0, len(idxs), bs):
        yield idxs[i:i + bs]


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


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    return -(p * torch.log(torch.clamp(p, min=1.0e-12))).sum(dim=-1)


def _candidate_set_for_scope(
    *,
    scope: str,
    clip_id: int,
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
        cand = {int(x) for x in text_vocab_ids}
    else:
        raise ValueError(f"unsupported weak candidate scope: {scope}")
    return sorted(int(x) for x in cand if int(x) in raw_to_idx)


def _weak_loss_for_row(
    *,
    ex: Mapping[str, Any],
    text_proj_all: torch.Tensor,
    raw_to_idx: Mapping[int, int],
    text_vocab_ids: Sequence[int],
    clip_y_base: Mapping[int, Set[int]],
    base_ids: Set[int],
    positive_scope: str,
    denominator_scope: str,
    theta_t: torch.nn.Parameter,
    device: torch.device,
    nohub: bool,
    soft_tau: float,
    soft_gamma: float,
    min_row_weight: float,
    entropy_penalty: float,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    clip_id = int(ex["clip_id"])
    pos = _candidate_set_for_scope(
        scope=positive_scope,
        clip_id=clip_id,
        clip_y_base=clip_y_base,
        base_ids=base_ids,
        text_vocab_ids=text_vocab_ids,
        raw_to_idx=raw_to_idx,
    )
    den = _candidate_set_for_scope(
        scope=denominator_scope,
        clip_id=clip_id,
        clip_y_base=clip_y_base,
        base_ids=base_ids,
        text_vocab_ids=text_vocab_ids,
        raw_to_idx=raw_to_idx,
    )
    if not pos:
        return None, {"skip": "empty_positive_scope"}
    if not den:
        return None, {"skip": "empty_denominator_scope"}
    den_set = set(int(x) for x in den)
    pos = [int(x) for x in pos if int(x) in den_set]
    if not pos:
        return None, {"skip": "positive_not_in_denominator"}

    den_idx = torch.tensor([raw_to_idx[int(x)] for x in den], device=device, dtype=torch.long)
    den_text = text_proj_all[den_idx]
    z = torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))).to(device=device, dtype=torch.float32).view(1, -1)
    z = F.normalize(z, p=2.0, dim=-1)
    scores_den = (z @ den_text.t()).squeeze(0) / _compute_t_dis(theta_t)

    den_pos = {int(raw): i for i, raw in enumerate(den)}
    pos_positions = torch.tensor([den_pos[int(x)] for x in pos], device=device, dtype=torch.long)
    scores_pos = scores_den[pos_positions]

    # Weak multi-label set objective: maximize probability mass assigned to the
    # observed full-Y positive set against the chosen denominator. This uses no
    # instance GT target.
    loss = torch.logsumexp(scores_den, dim=0) - torch.logsumexp(scores_pos, dim=0)

    pos_probs = torch.softmax(scores_pos.detach(), dim=-1)
    local_conf = float(pos_probs.max().cpu().item()) if pos_probs.numel() else 0.0
    local_explained = float(torch.sigmoid(torch.tensor(float(soft_gamma) * (local_conf - float(soft_tau)))).item())
    row_weight = 1.0
    if bool(nohub):
        row_weight = max(float(min_row_weight), 1.0 - local_explained)
        loss = loss * float(row_weight)
    if float(entropy_penalty) != 0.0:
        loss = loss + float(entropy_penalty) * _entropy_from_logits(scores_den).mean()

    top_den_idx = int(torch.argmax(scores_den.detach()).cpu().item())
    top_raw = int(den[top_den_idx])
    stats = {
        "skip": "",
        "positive_size": int(len(pos)),
        "denominator_size": int(len(den)),
        "local_conf": float(local_conf),
        "local_explained": float(local_explained),
        "row_weight": float(row_weight),
        "den_entropy": float(_entropy_from_logits(scores_den.detach()).mean().cpu().item()),
        "pos_entropy": float(_entropy_from_logits(scores_pos.detach()).mean().cpu().item()) if scores_pos.numel() > 1 else 0.0,
        "top_raw_id": int(top_raw),
    }
    return loss, stats


def _score_multi_scope(
    *,
    examples: Sequence[Mapping[str, Any]],
    projector: Projector,
    theta_t: torch.nn.Parameter,
    text_vocab_tensor: torch.Tensor,
    text_vocab_ids: Sequence[int],
    raw_to_idx: Mapping[int, int],
    clip_y_base: Mapping[int, Set[int]],
    base_ids: Set[int],
    eval_scopes: Sequence[str],
    device: torch.device,
    class_name_map: Optional[Mapping[int, str]],
    max_rows_out: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    per_class: Dict[str, List[Dict[str, Any]]] = {}
    rows: Dict[str, List[Dict[str, Any]]] = {}
    for scope in eval_scopes:
        s, pc, erows = oracle._score_rows(
            examples=examples,
            projector=projector,
            theta_t=theta_t,
            text_vocab_tensor=text_vocab_tensor,
            text_vocab_ids=text_vocab_ids,
            raw_to_idx=raw_to_idx,
            clip_y_base=clip_y_base,
            base_ids=base_ids,
            candidate_scope=str(scope),
            device=device,
            class_name_map=class_name_map,
            max_rows_out=int(max_rows_out),
        )
        summaries[str(scope)] = dict(s)
        per_class[str(scope)] = list(pc)
        rows[str(scope)] = list(erows)
    return summaries, per_class, rows


def _extract_primary_metric(eval_by_scope: Mapping[str, Mapping[str, Any]], preferred_scope: str) -> Dict[str, Any]:
    if preferred_scope in eval_by_scope:
        return dict(eval_by_scope[preferred_scope])
    if eval_by_scope:
        return dict(next(iter(eval_by_scope.values())))
    return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GT-clean weak full-Y base overfit trainer.")
    p.add_argument("--exp_name", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--repo_root", default=str(_BOOT_REPO_ROOT))
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--gt_identity_binding_jsonl", default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--protocol", choices=["weak_fullY_baseline", "weak_fullY_nohub"], default="weak_fullY_baseline")
    p.add_argument("--positive_scope", choices=["clip_y_base", "base_vocab", "full_vocab"], default="clip_y_base")
    p.add_argument("--denominator_scope", choices=["base_vocab", "full_vocab", "clip_y_base"], default="base_vocab")
    p.add_argument("--eval_candidate_scopes", default="clip_y_base,base_vocab")
    p.add_argument("--primary_eval_scope", choices=["clip_y_base", "base_vocab", "full_vocab"], default="base_vocab")
    p.add_argument("--learning_rate", type=float, default=3.0e-4)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--t_dis_init", type=float, default=0.07)
    p.add_argument("--freeze_temperature", action="store_true")
    p.add_argument("--entropy_penalty", type=float, default=0.0)
    p.add_argument("--soft_tau", type=float, default=0.7)
    p.add_argument("--soft_gamma", type=float, default=10.0)
    p.add_argument("--min_row_weight", type=float, default=0.25)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--require_target_in_clip_y_base", action="store_true", default=True)
    p.add_argument("--no_require_target_in_clip_y_base", action="store_false", dest="require_target_in_clip_y_base")
    p.add_argument("--max_example_rows_out", type=int, default=200)
    p.add_argument("--show_progress", action="store_true", default=True)
    p.add_argument("--no_progress", action="store_false", dest="show_progress")
    p.add_argument("--print_epoch_summary", action="store_true", default=True)
    p.add_argument("--no_print_epoch_summary", action="store_false", dest="print_epoch_summary")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(str(args.device))

    pack = oracle._load_oracle_examples(args)
    examples = pack.examples
    class_name_map = oracle._class_name_map_from_annotation(Path(args.annotation_json))
    eval_scopes = [x.strip() for x in str(args.eval_candidate_scopes).split(",") if x.strip()]
    if not eval_scopes:
        eval_scopes = [str(args.primary_eval_scope)]

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
    weak_metrics_path = train_dir / "gt_clean_base_weak_overfit_metrics.jsonl"
    for p in (runtime_metrics_path, weak_metrics_path):
        if p.exists():
            p.unlink()

    train_start = datetime.now(timezone.utc).isoformat()
    global_step = 0
    batch_losses: List[float] = []
    epoch_summaries: List[Dict[str, Any]] = []

    eval_by_scope, per_class_by_scope, rows_by_scope = _score_multi_scope(
        examples=examples,
        projector=projector,
        theta_t=theta_t,
        text_vocab_tensor=text_vocab_tensor,
        text_vocab_ids=text_vocab_ids,
        raw_to_idx=raw_to_idx,
        clip_y_base=pack.clip_y_base,
        base_ids=pack.base_ids,
        eval_scopes=eval_scopes,
        device=device,
        class_name_map=class_name_map,
        max_rows_out=int(args.max_example_rows_out),
    )
    initial_summary = {
        "row_type": "eval_summary",
        "epoch": 0,
        "phase": "initial",
        "eval_by_scope": eval_by_scope,
        "primary_eval_scope": str(args.primary_eval_scope),
        **{f"primary_{k}": v for k, v in _extract_primary_metric(eval_by_scope, str(args.primary_eval_scope)).items()},
    }
    _append_jsonl(runtime_metrics_path, initial_summary)
    _append_jsonl(weak_metrics_path, initial_summary)

    nohub_enabled = str(args.protocol) == "weak_fullY_nohub"
    for epoch_idx in _iter_progress(range(int(args.epochs)), enabled=bool(args.show_progress), desc="gt-clean weak full-Y epochs", leave=True):
        rng = random.Random(int(args.seed) + int(epoch_idx))
        epoch_losses: List[float] = []
        epoch_skipped = Counter()
        local_conf_vals: List[float] = []
        row_weight_vals: List[float] = []
        local_explained_vals: List[float] = []
        den_entropy_vals: List[float] = []
        pos_entropy_vals: List[float] = []
        positive_sizes: List[float] = []
        denominator_sizes: List[float] = []
        top_pred_hist = Counter()

        for batch_ids in _iter_minibatches(len(examples), int(args.batch_size), rng=rng):
            optimizer.zero_grad(set_to_none=True)
            text_proj_all = F.normalize(projector(text_vocab_tensor), p=2.0, dim=-1)
            batch_loss_terms: List[torch.Tensor] = []
            for idx in batch_ids:
                ex = examples[int(idx)]
                loss, stats = _weak_loss_for_row(
                    ex=ex,
                    text_proj_all=text_proj_all,
                    raw_to_idx=raw_to_idx,
                    text_vocab_ids=text_vocab_ids,
                    clip_y_base=pack.clip_y_base,
                    base_ids=pack.base_ids,
                    positive_scope=str(args.positive_scope),
                    denominator_scope=str(args.denominator_scope),
                    theta_t=theta_t,
                    device=device,
                    nohub=nohub_enabled,
                    soft_tau=float(args.soft_tau),
                    soft_gamma=float(args.soft_gamma),
                    min_row_weight=float(args.min_row_weight),
                    entropy_penalty=float(args.entropy_penalty),
                )
                if loss is None:
                    epoch_skipped[str(stats.get("skip", "unknown"))] += 1
                    continue
                batch_loss_terms.append(loss)
                local_conf_vals.append(float(stats["local_conf"]))
                local_explained_vals.append(float(stats["local_explained"]))
                row_weight_vals.append(float(stats["row_weight"]))
                den_entropy_vals.append(float(stats["den_entropy"]))
                pos_entropy_vals.append(float(stats["pos_entropy"]))
                positive_sizes.append(float(stats["positive_size"]))
                denominator_sizes.append(float(stats["denominator_size"]))
                top_pred_hist[int(stats["top_raw_id"])] += 1
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
            recent_n = min(len(batch_loss_terms), len(local_conf_vals))
            _append_jsonl(runtime_metrics_path, {
                "row_type": "microbatch",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "epoch": int(epoch_idx) + 1,
                "global_step": int(global_step),
                "loss": bval,
                "batch_size_effective": int(len(batch_loss_terms)),
                "local_conf_mean_recent": _mean(local_conf_vals[-recent_n:]) if recent_n else 0.0,
                "row_weight_mean_recent": _mean(row_weight_vals[-recent_n:]) if recent_n else 1.0,
                "temperature": float(_compute_t_dis(theta_t).detach().cpu().item()),
            })

        eval_by_scope, per_class_by_scope, rows_by_scope = _score_multi_scope(
            examples=examples,
            projector=projector,
            theta_t=theta_t,
            text_vocab_tensor=text_vocab_tensor,
            text_vocab_ids=text_vocab_ids,
            raw_to_idx=raw_to_idx,
            clip_y_base=pack.clip_y_base,
            base_ids=pack.base_ids,
            eval_scopes=eval_scopes,
            device=device,
            class_name_map=class_name_map,
            max_rows_out=int(args.max_example_rows_out),
        )
        primary = _extract_primary_metric(eval_by_scope, str(args.primary_eval_scope))
        total_top = sum(top_pred_hist.values())
        most_common = top_pred_hist.most_common(20)
        epoch_summary = {
            "row_type": "epoch_summary",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": "prealign",
            "protocol": str(args.protocol),
            "epoch": int(epoch_idx) + 1,
            "global_step": int(global_step),
            "loss_mean_epoch_train_batches": _mean(epoch_losses),
            "loss_last_epoch_train_batch": float(epoch_losses[-1]) if epoch_losses else 0.0,
            "positive_scope": str(args.positive_scope),
            "denominator_scope": str(args.denominator_scope),
            "positive_size_mean_epoch": _mean(positive_sizes),
            "denominator_size_mean_epoch": _mean(denominator_sizes),
            "local_conf_mean_epoch": _mean(local_conf_vals),
            "local_conf_p90_epoch": _percentile(local_conf_vals, 90),
            "local_explained_mean_epoch": _mean(local_explained_vals),
            "row_weight_mean_epoch": _mean(row_weight_vals),
            "row_weight_min_epoch": min(row_weight_vals) if row_weight_vals else 1.0,
            "den_entropy_mean_epoch": _mean(den_entropy_vals),
            "pos_entropy_mean_epoch": _mean(pos_entropy_vals),
            "top_pred_unique_count_epoch": int(len(top_pred_hist)),
            "top_pred_max_share_epoch": float(most_common[0][1] / total_top) if total_top and most_common else 0.0,
            "top_pred_top20_epoch": [{"raw_id": int(k), "count": int(v), "share": float(v / max(total_top, 1))} for k, v in most_common],
            "skipped_epoch": dict(epoch_skipped),
            "eval_by_scope": eval_by_scope,
            "primary_eval_scope": str(args.primary_eval_scope),
            **{f"primary_{k}": v for k, v in primary.items()},
        }
        epoch_summaries.append(epoch_summary)
        _append_jsonl(runtime_metrics_path, epoch_summary)
        _append_jsonl(weak_metrics_path, epoch_summary)
        if bool(args.print_epoch_summary):
            print(json.dumps({
                "epoch": epoch_summary["epoch"],
                "train_loss": epoch_summary["loss_mean_epoch_train_batches"],
                "primary_scope": str(args.primary_eval_scope),
                "primary_top1": primary.get("gt_top1_hit_rate"),
                "primary_rank": primary.get("mean_normalized_gt_rank"),
                "clip_y_top1": eval_by_scope.get("clip_y_base", {}).get("gt_top1_hit_rate"),
                "base_vocab_top1": eval_by_scope.get("base_vocab", {}).get("gt_top1_hit_rate"),
                "row_weight_mean": epoch_summary["row_weight_mean_epoch"],
                "top_pred_max_share": epoch_summary["top_pred_max_share_epoch"],
                "temperature": float(_compute_t_dis(theta_t).detach().cpu().item()),
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
        "protocol": str(args.protocol),
        "pipeline": "gt_clean_weak_fullY_overfit",
        "label_source": "weak_clip_full_y_base_only__gt_for_eval_only",
        "trajectory_source_branch": "gt_upper_bound",
        "global_step": int(global_step),
        "positive_scope": str(args.positive_scope),
        "denominator_scope": str(args.denominator_scope),
        "eval_candidate_scopes": eval_scopes,
        "primary_eval_scope": str(args.primary_eval_scope),
        "nohub_enabled": bool(nohub_enabled),
        "soft_tau": float(args.soft_tau),
        "soft_gamma": float(args.soft_gamma),
        "min_row_weight": float(args.min_row_weight),
    }, ckpt_path)

    final_eval_by_scope, per_class_by_scope, rows_by_scope = _score_multi_scope(
        examples=examples,
        projector=projector,
        theta_t=theta_t,
        text_vocab_tensor=text_vocab_tensor,
        text_vocab_ids=text_vocab_ids,
        raw_to_idx=raw_to_idx,
        clip_y_base=pack.clip_y_base,
        base_ids=pack.base_ids,
        eval_scopes=eval_scopes,
        device=device,
        class_name_map=class_name_map,
        max_rows_out=int(args.max_example_rows_out),
    )
    for scope, rows in per_class_by_scope.items():
        _write_csv(train_dir / f"per_class_weak_overfit__{scope}.csv", rows)
    for scope, rows in rows_by_scope.items():
        _write_jsonl(train_dir / f"weak_overfit_example_rows__{scope}.jsonl", rows)

    train_state = {
        "stage_id": "prealign",
        "epoch": int(args.epochs),
        "selected_for_infer": "prealign_only",
        "selected_for_infer_authority": "explicit_train_state_field",
        "checkpoint_last": "train/prealign/checkpoints/prealign_last.pth",
        "checkpoint_selected": "train/prealign/checkpoints/prealign_last.pth",
        "global_step": int(global_step),
        "runtime_asset_source": pack.materialization_summary.get("materialized_resolution", {}).get("runtime_asset_source", "local_canonical_assets"),
        "runtime_asset_output_root": str(repo_root),
        "pipeline": "gt_clean_weak_fullY_overfit",
        "protocol": str(args.protocol),
        "positive_scope": str(args.positive_scope),
        "denominator_scope": str(args.denominator_scope),
    }
    _write_json(train_dir / "train_state.json", train_state)

    final_primary = _extract_primary_metric(final_eval_by_scope, str(args.primary_eval_scope))
    stage_summary = {
        "stage_id": "prealign",
        "pipeline": "gt_clean_weak_fullY_overfit",
        "protocol": str(args.protocol),
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": "gt_upper_bound",
        "label_source": "weak_clip_full_y_base_only__gt_for_eval_only",
        "epochs": int(args.epochs),
        "positive_scope": str(args.positive_scope),
        "denominator_scope": str(args.denominator_scope),
        "eval_candidate_scopes": eval_scopes,
        "primary_eval_scope": str(args.primary_eval_scope),
        "initial_eval_by_scope": initial_summary["eval_by_scope"],
        "final_eval_by_scope": final_eval_by_scope,
        "final_primary_eval": final_primary,
        "loss_mean": _mean(batch_losses),
        "loss_last": float(batch_losses[-1]) if batch_losses else 0.0,
        "global_step": int(global_step),
        "trainable_example_count": int(len(examples)),
        "checkpoint_last": "train/prealign/checkpoints/prealign_last.pth",
        "runtime_metrics": "train/prealign/runtime_metrics.jsonl",
        "weak_metrics": "train/prealign/gt_clean_base_weak_overfit_metrics.jsonl",
        "per_class_weak_overfit": {scope: f"train/prealign/per_class_weak_overfit__{scope}.csv" for scope in per_class_by_scope},
        "materialization_summary": pack.materialization_summary,
        "identity_binding_paths_used": pack.identity_binding_paths_used,
        "identity_binding_stats": pack.identity_binding_stats,
        "target_attach_counters": pack.target_attach_counters,
        "boundary": "weak assignment audit only; GT target is used for clean denominator/eval, not for training loss",
    }
    _write_json(train_dir / "stage_summary.json", stage_summary)

    summary = {
        "status": "PASS",
        "exp_name": str(args.exp_name),
        "pipeline": "gt_clean_weak_fullY_overfit",
        "protocol": str(args.protocol),
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": "gt_upper_bound",
        "label_source": "weak_clip_full_y_base_only__gt_for_eval_only",
        "repo_root": str(repo_root),
        "asset_root": str(Path(args.asset_root).expanduser().resolve()),
        "output_root": str(output_root),
        "train_started_at": train_start,
        "train_finished_at": datetime.now(timezone.utc).isoformat(),
        "selected_checkpoint_path": str(ckpt_path),
        "initial_eval_by_scope": initial_summary["eval_by_scope"],
        "final_eval_by_scope": final_eval_by_scope,
        "final_primary_eval": final_primary,
        "stages": {"prealign": stage_summary},
        "interpretation": "If oracle supervised overfit is strong but this weak full-Y arm remains weak, the bottleneck is the weak assignment objective.",
    }
    _write_json(output_root / "train" / "pipeline_train_summary.json", summary)
    print(str(output_root / "train" / "pipeline_train_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
