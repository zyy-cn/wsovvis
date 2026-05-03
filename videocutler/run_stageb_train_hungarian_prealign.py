#!/usr/bin/env python3
"""A8 prealign ablation trainer for Hungarian pseudo-label pipeline.

This is a clean side-path prealign stage. It trains only a score initializer
from GT carrier features and clip-level full-Y labels. It does not generate or
consume Hungarian labels, row-level GT labels, NoHub correctness, oracle
correctness, manual person/hub priors, dummy/slack, or extra-support rows.

Protocols:
  * baseline_full_y: positive-set bag loss over base vocabulary
      L_i = logsumexp_{c in B} s(i,c) - logsumexp_{c in Y(v)} s(i,c)
  * nohub_style: same base loss multiplied by stop-gradient confidence weight

The output checkpoint is intended for:
  prealign checkpoint -> Hungarian matching -> clean base matched-pair CE
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

from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig  # noqa: E402
from videocutler.run_stageb_analysis_residual_gated_coverage_assignment import (  # noqa: E402
    _auto_find_checkpoint,
    _compute_t_dis,
    _default_row_gap_path,
    _inverse_softplus,
    _load_checkpoint_if_requested,
    _prepare_data,
    _truth,
    _write_csv,
    _write_json,
)
from videocutler.run_stageb_train_residual_gated_hungarian_matched import (  # noqa: E402
    _compare_eval,
    _evaluate_full_y,
    _example_by_tid,
    _load_eval_rows,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(list(vals), dtype=np.float64)))


def _group_by_clip(examples: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        out[int(ex["clip_id"])].append(dict(ex))
    return dict(out)


def _carrier_tensor(group: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    vecs = []
    for ex in group:
        arr = np.asarray(ex["carrier_vec"], dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        vecs.append(torch.from_numpy(arr.astype(np.float32)))
    z = torch.stack(vecs, dim=0).to(device=device, dtype=torch.float32)
    return F.normalize(z, p=2.0, dim=-1)


def _project_text(projector: Projector, text_tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(projector(text_tensor), p=2.0, dim=-1)


def _prealign_loss_for_clip(
    *,
    clip_id: int,
    group: Sequence[Mapping[str, Any]],
    data: Any,
    projector: Projector,
    text_tensor: torch.Tensor,
    base_text_indices: torch.Tensor,
    base_raw_ids: Sequence[int],
    base_col_by_raw_id: Mapping[int, int],
    theta_t: torch.nn.Parameter,
    device: torch.device,
    protocol: str,
    row_weight_gamma: float,
    row_weight_conf_threshold: float,
    min_row_weight: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    y_ids = sorted(int(x) for x in data.clip_y_base.get(int(clip_id), set()) if int(x) in base_col_by_raw_id)
    if not y_ids:
        raise RuntimeError(f"clip {clip_id} has empty full-Y positives after base/text filter")
    Z = _carrier_tensor(group, device=device)
    text_proj_all = _project_text(projector, text_tensor)
    T_base = text_proj_all[base_text_indices]
    logits_base = torch.matmul(Z, T_base.t()) / _compute_t_dis(theta_t)
    pos_cols = torch.tensor([int(base_col_by_raw_id[int(rid)]) for rid in y_ids], device=device, dtype=torch.long)
    logits_pos = logits_base.index_select(1, pos_cols)

    # Positive-set bag loss: probability mass should enter the clip-level full-Y set.
    loss_i = torch.logsumexp(logits_base, dim=1) - torch.logsumexp(logits_pos, dim=1)

    with torch.no_grad():
        local_prob = torch.softmax(logits_pos.detach(), dim=1)
        local_conf = torch.max(local_prob, dim=1).values
        explained = torch.sigmoid(float(row_weight_gamma) * (local_conf - float(row_weight_conf_threshold)))
        row_weight = torch.clamp(1.0 - explained, min=float(min_row_weight), max=1.0)

    if str(protocol) == "baseline_full_y":
        weighted = loss_i
        used_row_weight = torch.ones_like(loss_i)
    elif str(protocol) == "nohub_style":
        weighted = row_weight * loss_i
        used_row_weight = row_weight
    else:
        raise ValueError(f"unsupported protocol: {protocol}")

    with torch.no_grad():
        pred_col = torch.argmax(logits_base, dim=1)
        pos_set = set(int(x) for x in pos_cols.detach().cpu().tolist())
        mass_in_y = torch.exp(torch.logsumexp(logits_pos, dim=1) - torch.logsumexp(logits_base, dim=1))
        top1_in_y = float(torch.tensor([1.0 if int(x) in pos_set else 0.0 for x in pred_col.detach().cpu().tolist()]).mean().item())

    return weighted.mean(), {
        "clip_id": int(clip_id),
        "rows": int(len(group)),
        "clip_y_size": int(len(y_ids)),
        "loss_mean": float(loss_i.detach().mean().cpu().item()),
        "weighted_loss_mean": float(weighted.detach().mean().cpu().item()),
        "row_weight_mean": float(used_row_weight.detach().mean().cpu().item()),
        "local_conf_mean": float(local_conf.detach().mean().cpu().item()),
        "mass_in_y_mean": float(mass_in_y.detach().mean().cpu().item()),
        "top1_in_y_rate": float(top1_in_y),
    }


def _default_output_root(run_root: Path, dataset_name: str, protocol: str) -> Path:
    return run_root / "outputs" / "a8_hungarian_prealign_ablation" / str(dataset_name) / f"{protocol}_50ep"


def _train(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve() if str(args.output_root).strip() else _default_output_root(run_root, str(args.dataset_name), str(args.protocol))
    out_root.mkdir(parents=True, exist_ok=True)
    train_dir = out_root / "train" / f"prealign_{args.protocol}"
    train_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed)); np.random.seed(int(args.seed)); torch.manual_seed(int(args.seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(str(args.device))

    if not str(args.annotation_json).strip():
        args.annotation_json = str(Path(args.repo_root) / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json")
    if not str(args.split_json).strip():
        args.split_json = str(Path(args.repo_root) / "package" / "reference" / "lvvis_official_base_novel_split.json")
    # _prepare_data requires output_dir for optional asset-root resolution.
    if not hasattr(args, "output_dir"):
        args.output_dir = ""

    data = _prepare_data(args)
    by_clip = _group_by_clip(data.examples)
    clip_ids = sorted(int(k) for k in by_clip.keys())

    base_raw_ids = sorted(int(x) for x in data.base_ids if int(x) in data.raw_to_text_idx)
    if not base_raw_ids:
        raise RuntimeError("empty base vocabulary after text-bank filter")
    base_col_by_raw_id = {int(rid): j for j, rid in enumerate(base_raw_ids)}
    base_text_indices = torch.tensor([int(data.raw_to_text_idx[int(rid)]) for rid in base_raw_ids], device=device, dtype=torch.long)

    text_tensor = torch.tensor(np.asarray(data.text_matrix, dtype=np.float32), device=device, dtype=torch.float32)
    projector = Projector(ProjectorConfig()).to(device)
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(args.t_dis_init) - 1.0e-4, 1.0e-6)), device=device, dtype=torch.float32))
    ckpt = _auto_find_checkpoint(repo_root, str(args.dataset_name)) if str(args.init_checkpoint).strip() == "auto" else str(args.init_checkpoint).strip()
    checkpoint_summary = _load_checkpoint_if_requested(projector, theta_t, ckpt, device)

    row_gap_csv = Path(args.row_gap_csv).expanduser().resolve() if str(args.row_gap_csv).strip() else _default_row_gap_path(repo_root, str(args.dataset_name))
    eval_rows, eval_summary = _load_eval_rows(row_gap_csv, data, max_rows=int(args.max_eval_rows), seed=int(args.seed))
    example_by_tid = _example_by_tid(data)

    setup = {
        "timestamp": _now(),
        "protocol": str(args.protocol),
        "run_root": str(run_root),
        "output_root": str(out_root),
        "device": str(device),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "policy": {
            "uses_row_level_gt_for_training": False,
            "uses_hungarian_labels_for_prealign": False,
            "uses_nohub_correctness_for_training": False,
            "uses_manual_person_or_hub_prior": False,
            "uses_dummy_or_slack": False,
            "uses_extra_support": False,
            "candidate_set": "full-Y positives over base-vocabulary denominator",
            "loss_baseline_full_y": "logsumexp_B - logsumexp_Y",
            "loss_nohub_style": "stopgrad(row_weight) * (logsumexp_B - logsumexp_Y)",
        },
        "checkpoint": checkpoint_summary,
        "base_vocab_count": len(base_raw_ids),
        "materialization_summary": data.materialization_summary,
        "eval_row_summary": eval_summary,
    }
    _write_json(out_root / "prealign_setup.json", setup)

    before = _evaluate_full_y(stage="before", rows=eval_rows, data=data, example_by_tid=example_by_tid, projector=projector, text_tensor=text_tensor, theta_t=theta_t, device=device, out_dir=out_root / "analysis")

    optimizer = torch.optim.AdamW([*projector.parameters(), theta_t], lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    rng = random.Random(int(args.seed))
    log_path = train_dir / "train_log.jsonl"
    if log_path.exists(): log_path.unlink()
    global_step = 0
    all_losses: List[float] = []
    for epoch in (tqdm(range(int(args.epochs)), desc=f"prealign_{args.protocol}_epochs", dynamic_ncols=True) if bool(args.show_progress) and tqdm is not None else range(int(args.epochs))):
        rng.shuffle(clip_ids)
        epoch_losses: List[float] = []
        epoch_weighted: List[float] = []
        epoch_mass: List[float] = []
        epoch_top1_in_y: List[float] = []
        epoch_weights: List[float] = []
        epoch_rows = 0
        for clip_id in clip_ids:
            group = by_clip[int(clip_id)]
            y_ids = [int(x) for x in data.clip_y_base.get(int(clip_id), set()) if int(x) in base_col_by_raw_id]
            if not group or not y_ids:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, stats = _prealign_loss_for_clip(
                clip_id=int(clip_id), group=group, data=data, projector=projector, text_tensor=text_tensor,
                base_text_indices=base_text_indices, base_raw_ids=base_raw_ids, base_col_by_raw_id=base_col_by_raw_id,
                theta_t=theta_t, device=device, protocol=str(args.protocol),
                row_weight_gamma=float(args.row_weight_gamma), row_weight_conf_threshold=float(args.row_weight_conf_threshold), min_row_weight=float(args.min_row_weight),
            )
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_([*projector.parameters(), theta_t], max_norm=float(args.grad_clip_norm))
            optimizer.step()
            global_step += 1
            lv = float(loss.detach().cpu().item())
            all_losses.append(lv); epoch_losses.append(float(stats["loss_mean"])); epoch_weighted.append(float(stats["weighted_loss_mean"])); epoch_mass.append(float(stats["mass_in_y_mean"])); epoch_top1_in_y.append(float(stats["top1_in_y_rate"])); epoch_weights.append(float(stats["row_weight_mean"])); epoch_rows += int(stats["rows"])
            if int(args.log_every_steps) > 0 and global_step % int(args.log_every_steps) == 0:
                _append_jsonl(log_path, {"timestamp": _now(), "epoch": int(epoch)+1, "global_step": global_step, **stats})
        epoch_row = {
            "timestamp": _now(), "row_type": "epoch_summary", "epoch": int(epoch)+1, "global_step": global_step,
            "loss_mean": _mean(epoch_losses), "weighted_loss_mean": _mean(epoch_weighted), "mass_in_y_mean": _mean(epoch_mass), "top1_in_y_rate_mean": _mean(epoch_top1_in_y), "row_weight_mean": _mean(epoch_weights), "epoch_rows": epoch_rows, "epoch_clips": len(clip_ids),
        }
        _append_jsonl(log_path, epoch_row)
        if bool(args.print_epoch_summary): print(json.dumps(epoch_row, ensure_ascii=False))

    after = _evaluate_full_y(stage="after", rows=eval_rows, data=data, example_by_tid=example_by_tid, projector=projector, text_tensor=text_tensor, theta_t=theta_t, device=device, out_dir=out_root / "analysis")
    cmp = _compare_eval(before, after)
    ckpt_out = train_dir / "prealign_last.pth"
    torch.save({
        "text_projector_state_dict": projector.state_dict(),
        "theta_T": float(theta_t.detach().cpu().item()),
        "epoch": int(args.epochs),
        "global_step": int(global_step),
        "protocol": str(args.protocol),
        "positive_scope": "full_y_base",
        "denominator_scope": "base_vocab",
        "loss": "positive_set_bag",
        "row_weight_enabled": str(args.protocol) == "nohub_style",
        "min_row_weight": float(args.min_row_weight),
        "row_weight_gamma": float(args.row_weight_gamma),
        "row_weight_conf_threshold": float(args.row_weight_conf_threshold),
        "seed": int(args.seed),
    }, ckpt_out)

    final = {
        "status": "PASS", "timestamp": _now(), "protocol": str(args.protocol), "output_root": str(out_root), "checkpoint": str(ckpt_out),
        "train_summary": {"epochs": int(args.epochs), "global_step": int(global_step), "loss_mean": _mean(all_losses), "loss_last": all_losses[-1] if all_losses else 0.0, "train_clip_count": len(clip_ids), "train_example_count": len(data.examples)},
        "eval_before": before, "eval_after": after, "eval_delta": cmp, "setup": setup,
    }
    _write_json(out_root / "final_summary.json", final)
    lines = [
        f"# A8 Prealign Ablation TAKEOVER ({args.protocol})", "", "- status: PASS", f"- protocol: {args.protocol}", f"- epochs: {args.epochs}", f"- checkpoint: {ckpt_out}", f"- before micro_top1: {before.get('micro_top1', 0.0):.6f}", f"- after micro_top1: {after.get('micro_top1', 0.0):.6f}", f"- delta micro_top1: {cmp.get('delta_micro_top1', 0.0):.6f}", f"- before macro_rank1: {before.get('macro_rank1', 0.0):.6f}", f"- after macro_rank1: {after.get('macro_rank1', 0.0):.6f}", f"- delta macro_rank1: {cmp.get('delta_macro_rank1', 0.0):.6f}", "", "## Policy", "- Prealign uses GT carrier + full-Y clip labels + text prototypes only.", "- No row-level GT / Hungarian labels / NoHub correctness are used for prealign training.", "- baseline_full_y loss is logsumexp_B - logsumexp_Y.", "- nohub_style only adds stop-gradient row weighting.",
    ]
    (out_root / "A8_PREALIGN_ABLATION_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2, default=str))
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A8 prealign ablation: baseline_full_y/nohub_style score initializer")
    p.add_argument("--run_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--output_root", default="")
    p.add_argument("--output_dir", default="")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--row_gap_csv", default="")
    p.add_argument("--init_checkpoint", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--protocol", choices=["baseline_full_y", "nohub_style"], default="baseline_full_y")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--max_eval_rows", type=int, default=0)
    p.add_argument("--t_dis_init", type=float, default=0.07)
    p.add_argument("--row_weight_gamma", type=float, default=8.0)
    p.add_argument("--row_weight_conf_threshold", type=float, default=0.5)
    p.add_argument("--min_row_weight", type=float, default=0.25)
    p.add_argument("--show_progress", action="store_true", default=True)
    p.add_argument("--print_epoch_summary", action="store_true", default=True)
    p.add_argument("--log_every_steps", type=int, default=200)
    return p.parse_args()


def main() -> int:
    _train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
