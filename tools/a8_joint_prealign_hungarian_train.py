#!/usr/bin/env python3
"""A8 joint prealign + dynamic Hungarian diagnostic training.

This side-path tool intentionally combines ONLY two existing objectives:

  L_prealign = logsumexp over base-vocab B - logsumexp over clip full-Y base set
  L_hungarian = current clip-local matched-pair CE/InfoNCE, including the current
                dynamic Hungarian assignment implementation when requested.

It does NOT introduce GT-target CE, visible525 CE, rank-margin, hard-negative,
suppressor-aware loss, dummy/slack, extra support, or NoHub correctness labels.

Primary evaluation is intentionally left to the canonical visible525 audit tool:
  tools/a8_visible525_candidate_rankk_audit.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
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
    _write_csv,
    _write_json,
)
from videocutler.run_stageb_train_residual_gated_hungarian_matched import (  # noqa: E402
    _example_by_tid,
    _group_rows_by_clip,
    _load_train_pairs,
    _loss_for_clip_assignment_mode,
    _project_text,
    _save_training_checkpoint,
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
        return int(float(str(x)))
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _mean(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(list(vals), dtype=np.float64)))


def _default_matched_pairs_path(run_root: Path, dataset_name: str) -> Path:
    # Prefer the already-used fixed baseline path; fall back to older canonical path.
    candidates = [
        run_root / "analysis" / "residual_gated_hungarian_matching_baseline_full_y_5ep" / str(dataset_name) / "hungarian_matched_pairs.csv",
        run_root / "analysis" / "residual_gated_hungarian_matching" / str(dataset_name) / "hungarian_matched_pairs.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def _default_output_root(run_root: Path, dataset_name: str, name: str) -> Path:
    return run_root / "outputs" / "a8_joint_prealign_hungarian" / str(dataset_name) / str(name)


def _clip_source_rows(
    *,
    clip_id: int,
    matched_rows: Sequence[Mapping[str, Any]],
    data: Any,
    example_by_tid: Mapping[str, Mapping[str, Any]],
    row_source: str,
) -> List[Mapping[str, Any]]:
    row_source = str(row_source or "all_clip_trajectories").strip().lower()
    if row_source == "matched_rows":
        return list(matched_rows)
    if row_source == "all_clip_trajectories":
        return [dict(ex) for ex in data.by_clip.get(int(clip_id), [])]
    raise ValueError(f"unsupported prealign_row_source={row_source!r}; expected matched_rows|all_clip_trajectories")


def _current_prealign_bag_loss_for_clip(
    *,
    clip_id: int,
    matched_rows: Sequence[Mapping[str, Any]],
    data: Any,
    example_by_tid: Mapping[str, Mapping[str, Any]],
    text_proj_all: torch.Tensor,
    theta_t: torch.nn.Parameter,
    device: torch.device,
    row_source: str = "all_clip_trajectories",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Current baseline_full_y prealign bag loss for one clip.

    L_i = logsumexp_{c in B} s(i,c) - logsumexp_{c in Y_base(v)} s(i,c)

    B is the official base vocabulary filtered by available text-bank ids.
    Y_base(v) is the current clip full-Y base label set filtered by B/text ids.
    """
    base_ids = sorted(int(x) for x in data.base_ids if int(x) in data.raw_to_text_idx)
    if not base_ids:
        raise RuntimeError("base vocabulary B is empty after text-bank filtering")
    y_ids = sorted(int(x) for x in data.clip_y_base.get(int(clip_id), set()) if int(x) in data.raw_to_text_idx and int(x) in data.base_ids)
    if not y_ids:
        raise RuntimeError(f"clip {clip_id} has empty Y_base after text-bank/base filtering")

    b_col = {int(rid): j for j, rid in enumerate(base_ids)}
    pos_cols = [b_col[int(rid)] for rid in y_ids if int(rid) in b_col]
    if not pos_cols:
        raise RuntimeError(f"clip {clip_id} has no positive cols inside B")

    rows = _clip_source_rows(
        clip_id=int(clip_id),
        matched_rows=matched_rows,
        data=data,
        example_by_tid=example_by_tid,
        row_source=row_source,
    )
    z_vecs: List[torch.Tensor] = []
    kept = 0
    for r in rows:
        tid = str(r.get("trajectory_id", ""))
        if not tid:
            continue
        ex = r if "carrier_vec" in r else example_by_tid.get(tid)
        if ex is None or "carrier_vec" not in ex:
            continue
        z_vecs.append(torch.from_numpy(np.asarray(ex["carrier_vec"], dtype=np.float32)))
        kept += 1
    if not z_vecs:
        raise RuntimeError(f"no prealign rows remained in clip {clip_id} for row_source={row_source}")

    Z = torch.stack(z_vecs, dim=0).to(device=device, dtype=torch.float32)
    Z = F.normalize(Z, p=2.0, dim=-1)
    base_text_idx = torch.tensor([int(data.raw_to_text_idx[int(rid)]) for rid in base_ids], device=device, dtype=torch.long)
    T_base = text_proj_all.index_select(0, base_text_idx)
    logits = torch.matmul(Z, T_base.t()) / _compute_t_dis(theta_t)
    pos = torch.tensor(pos_cols, device=device, dtype=torch.long)
    denom = torch.logsumexp(logits, dim=1)
    numer = torch.logsumexp(logits.index_select(1, pos), dim=1)
    losses = denom - numer
    loss = losses.mean()
    with torch.no_grad():
        prob_mass = torch.exp(numer - denom)
        stats = {
            "prealign_rows": int(kept),
            "prealign_base_candidate_count": int(len(base_ids)),
            "prealign_positive_count": int(len(pos_cols)),
            "prealign_loss": float(loss.detach().cpu().item()),
            "prealign_positive_mass_mean": float(prob_mass.mean().detach().cpu().item()),
        }
    return loss, stats


def _write_readme(out_root: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# A8 Joint Prealign-Hungarian Diagnostic",
        "",
        "- status: PASS" if payload.get("status") == "PASS" else f"- status: {payload.get('status')}",
        "- objective: L_total = lambda_prealign * L_prealign_current + lambda_hungarian * L_hungarian_current",
        "- L_prealign_current: logsumexp_B - logsumexp_Y_base(v)",
        "- L_hungarian_current: current fixed/dynamic Hungarian clip-local CE/InfoNCE implementation",
        "- no GT-target CE, no visible525 CE, no rank-margin, no hard-negative, no suppressor-aware loss",
        f"- checkpoint: {payload.get('checkpoint')}",
        "",
        "Primary metric must be computed with tools/a8_visible525_candidate_rankk_audit.py.",
        "Do not use retired row_gap micro_top1 as the primary metric.",
    ]
    (out_root / "A8_JOINT_PREALIGN_HUNGARIAN_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _train(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve() if str(args.output_root).strip() else _default_output_root(run_root, str(args.dataset_name), str(args.name))
    train_dir = out_root / "train" / "joint_prealign_hungarian"
    train_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device) if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    data_args = argparse.Namespace(
        repo_root=str(repo_root),
        asset_root=str(asset_root),
        run_root=str(run_root),
        output_dir=str(out_root),
        dataset_name=str(args.dataset_name),
        annotation_json=(str(args.annotation_json).strip() or str(repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json")),
        split_json=(str(args.split_json).strip() or str(repo_root / "package" / "reference" / "lvvis_official_base_novel_split.json")),
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
        subset_fraction=args.subset_fraction,
        seed=int(args.seed),
    )
    data = _prepare_data(data_args)
    example_by_tid = _example_by_tid(data)

    matched_csv = Path(args.matched_pairs_csv).expanduser().resolve() if str(args.matched_pairs_csv).strip() else _default_matched_pairs_path(run_root, str(args.dataset_name))
    if not matched_csv.is_file():
        raise FileNotFoundError(f"matched_pairs_csv not found: {matched_csv}")
    train_rows, train_row_summary = _load_train_pairs(matched_csv, data, max_rows=int(args.max_train_rows), seed=int(args.seed))
    if not train_rows:
        raise RuntimeError("no train rows selected from matched_pairs_csv")

    text_tensor = torch.tensor(np.asarray(data.text_matrix, dtype=np.float32), device=device, dtype=torch.float32)
    projector = Projector(ProjectorConfig()).to(device)
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(args.t_dis_init) - 1.0e-4, 1.0e-6)), device=device, dtype=torch.float32))
    init = str(args.init_checkpoint).strip()
    ckpt = _auto_find_checkpoint(repo_root, str(args.dataset_name)) if init == "auto" else init
    checkpoint_summary = _load_checkpoint_if_requested(projector, theta_t, ckpt, device)

    optimizer = torch.optim.AdamW([*projector.parameters(), theta_t], lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    train_by_clip = _group_rows_by_clip(train_rows)
    clip_ids = list(train_by_clip.keys())
    rng = random.Random(int(args.seed))
    log_path = train_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    setup = {
        "timestamp": _now(),
        "name": str(args.name),
        "run_root": str(run_root),
        "output_root": str(out_root),
        "dataset_name": str(args.dataset_name),
        "matched_pairs_csv": str(matched_csv),
        "device": str(device),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "lambda_prealign": float(args.lambda_prealign),
        "lambda_hungarian": float(args.lambda_hungarian),
        "hungarian_loss": str(args.hungarian_loss),
        "assignment_mode": str(args.assignment_mode),
        "dynamic_row_source": str(args.dynamic_row_source),
        "dynamic_candidate_source": str(args.dynamic_candidate_source),
        "prealign_row_source": str(args.prealign_row_source),
        "policy": {
            "uses_existing_prealign_loss": True,
            "uses_existing_hungarian_loss": True,
            "uses_row_level_gt_for_training": False,
            "uses_visible525_ce_for_training": False,
            "uses_rank_margin_or_hard_negative": False,
            "uses_nohub_correctness_for_training": False,
            "uses_dummy_or_slack": False,
            "uses_extra_support": False,
        },
        "checkpoint": checkpoint_summary,
        "materialization_summary": data.materialization_summary,
        "train_row_summary": train_row_summary,
    }
    _write_json(out_root / "a8_joint_prealign_hungarian_setup.json", setup)

    global_step = 0
    all_total_losses: List[float] = []
    all_pre_losses: List[float] = []
    all_hun_losses: List[float] = []
    epoch_metric_rows: List[Dict[str, Any]] = []
    iterator = tqdm(range(int(args.epochs)), desc="a8_joint_pre_hun_epochs", dynamic_ncols=True) if bool(args.show_progress) and tqdm is not None else range(int(args.epochs))
    for epoch in iterator:
        rng.shuffle(clip_ids)
        epoch_total: List[float] = []
        epoch_pre: List[float] = []
        epoch_hun: List[float] = []
        epoch_pmass: List[float] = []
        epoch_accs: List[float] = []
        epoch_rows = 0
        epoch_pre_rows = 0
        epoch_dyn_hub: List[float] = []
        epoch_dyn_margin: List[float] = []
        epoch_dyn_universe: List[float] = []
        for clip_id in clip_ids:
            rows = train_by_clip[int(clip_id)]
            optimizer.zero_grad(set_to_none=True)
            text_proj_all = _project_text(projector, text_tensor)
            pre_loss, pre_stats = _current_prealign_bag_loss_for_clip(
                clip_id=int(clip_id),
                matched_rows=rows,
                data=data,
                example_by_tid=example_by_tid,
                text_proj_all=text_proj_all,
                theta_t=theta_t,
                device=device,
                row_source=str(args.prealign_row_source),
            )
            hun_loss, hun_stats = _loss_for_clip_assignment_mode(
                rows=rows,
                data=data,
                example_by_tid=example_by_tid,
                text_proj_all=text_proj_all,
                theta_t=theta_t,
                device=device,
                loss_name=str(args.hungarian_loss),
                assignment_mode=str(args.assignment_mode),
                dynamic_row_source=str(args.dynamic_row_source),
                dynamic_candidate_source=str(args.dynamic_candidate_source),
                dynamic_hub_raw_ids=str(args.dynamic_hub_raw_ids),
            )
            total_loss = float(args.lambda_prealign) * pre_loss + float(args.lambda_hungarian) * hun_loss
            total_loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_([*projector.parameters(), theta_t], max_norm=float(args.grad_clip_norm))
            optimizer.step()
            global_step += 1
            tv = float(total_loss.detach().cpu().item())
            pv = float(pre_loss.detach().cpu().item())
            hv = float(hun_loss.detach().cpu().item())
            all_total_losses.append(tv); all_pre_losses.append(pv); all_hun_losses.append(hv)
            epoch_total.append(tv); epoch_pre.append(pv); epoch_hun.append(hv)
            epoch_pmass.append(float(pre_stats.get("prealign_positive_mass_mean", 0.0)))
            epoch_accs.append(float(hun_stats.get("pseudo_top1_acc", 0.0)))
            epoch_rows += int(hun_stats.get("rows", 0))
            epoch_pre_rows += int(pre_stats.get("prealign_rows", 0))
            epoch_dyn_hub.append(float(hun_stats.get("dynamic_assignment_hub_rate", 0.0)))
            epoch_dyn_margin.append(float(hun_stats.get("dynamic_mean_assignment_margin", 0.0)))
            epoch_dyn_universe.append(float(hun_stats.get("dynamic_universe_rows", 0.0)))
            if int(args.log_every_steps) > 0 and global_step % int(args.log_every_steps) == 0:
                _append_jsonl(log_path, {"timestamp": _now(), "row_type": "step", "epoch": int(epoch)+1, "global_step": global_step, "loss_total": tv, "loss_prealign": pv, "loss_hungarian": hv, **pre_stats, **hun_stats})
        epoch_row = {
            "timestamp": _now(),
            "row_type": "epoch_summary",
            "epoch": int(epoch) + 1,
            "loss_total_mean": _mean(epoch_total),
            "loss_prealign_mean": _mean(epoch_pre),
            "loss_hungarian_mean": _mean(epoch_hun),
            "prealign_positive_mass_mean": _mean(epoch_pmass),
            "hungarian_pseudo_top1_acc_mean": _mean(epoch_accs),
            "lambda_prealign": float(args.lambda_prealign),
            "lambda_hungarian": float(args.lambda_hungarian),
            "assignment_mode": str(args.assignment_mode),
            "dynamic_row_source": str(args.dynamic_row_source),
            "prealign_row_source": str(args.prealign_row_source),
            "epoch_hungarian_rows": int(epoch_rows),
            "epoch_prealign_rows": int(epoch_pre_rows),
            "epoch_clips": int(len(clip_ids)),
            "dynamic_assignment_hub_rate_mean": _mean(epoch_dyn_hub),
            "dynamic_mean_assignment_margin": _mean(epoch_dyn_margin),
            "dynamic_universe_rows_mean": _mean(epoch_dyn_universe),
        }
        _append_jsonl(log_path, epoch_row)
        epoch_metric_rows.append(dict(epoch_row))
        if int(args.save_every_epochs) > 0 and ((int(epoch) + 1) % int(args.save_every_epochs) == 0):
            _save_training_checkpoint(
                train_dir / f"a8_joint_epoch_{int(epoch)+1:03d}.pth",
                projector=projector,
                theta_t=theta_t,
                loss_name="joint_prealign_hungarian",
                epoch=int(epoch) + 1,
                global_step=global_step,
                matched_pairs_csv=matched_csv,
                assignment_mode=str(args.assignment_mode),
                dynamic_row_source=str(args.dynamic_row_source),
                dynamic_candidate_source=str(args.dynamic_candidate_source),
            )
        if bool(args.print_epoch_summary):
            print(json.dumps(epoch_row, ensure_ascii=False), flush=True)

    ckpt_out = train_dir / "a8_joint_last.pth"
    _save_training_checkpoint(
        ckpt_out,
        projector=projector,
        theta_t=theta_t,
        loss_name="joint_prealign_hungarian",
        epoch=int(args.epochs),
        global_step=global_step,
        matched_pairs_csv=matched_csv,
        assignment_mode=str(args.assignment_mode),
        dynamic_row_source=str(args.dynamic_row_source),
        dynamic_candidate_source=str(args.dynamic_candidate_source),
    )
    if bool(args.write_epoch_metrics):
        _write_csv(train_dir / "epoch_metrics.csv", epoch_metric_rows)
    final = {
        "status": "PASS",
        "timestamp": _now(),
        "definition": "A8 joint training using only current prealign bag loss plus current Hungarian CE/InfoNCE loss",
        "output_root": str(out_root),
        "checkpoint": str(ckpt_out),
        "train_summary": {
            "epochs": int(args.epochs),
            "global_step": int(global_step),
            "loss_total_mean": _mean(all_total_losses),
            "loss_prealign_mean": _mean(all_pre_losses),
            "loss_hungarian_mean": _mean(all_hun_losses),
            "train_row_count_from_matched_csv": int(len(train_rows)),
            "train_clip_count": int(len(clip_ids)),
        },
        "setup": setup,
        "primary_metric_note": "Run tools/a8_visible525_candidate_rankk_audit.py on checkpoint for canonical visible525 rank@K. Retired row_gap micro_top1 is not reported here.",
    }
    _write_json(out_root / "final_summary.json", final)
    _write_readme(out_root, final)
    print(json.dumps(final, ensure_ascii=False, indent=2, default=str))
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A8 joint current prealign bag loss + current Hungarian loss diagnostic training")
    p.add_argument("--run_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--name", default="j1_pre1_hun1_ep5")
    p.add_argument("--matched_pairs_csv", default="")
    p.add_argument("--output_root", default="")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--init_checkpoint", default="", help="Use empty string for no staged prealign init; use auto only when intentionally loading legacy initializer.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hungarian_loss", choices=["ce", "infonce"], default="ce")
    p.add_argument("--assignment_mode", choices=["fixed", "dynamic"], default="dynamic")
    p.add_argument("--dynamic_row_source", choices=["matched_rows", "all_clip_trajectories"], default="all_clip_trajectories")
    p.add_argument("--dynamic_candidate_source", choices=["full_y"], default="full_y")
    p.add_argument("--dynamic_hub_raw_ids", default="63,135,173,527,577,580,773,868,931,936,970,1044,1112,1114")
    p.add_argument("--prealign_row_source", choices=["matched_rows", "all_clip_trajectories"], default="all_clip_trajectories")
    p.add_argument("--lambda_prealign", type=float, default=1.0)
    p.add_argument("--lambda_hungarian", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--max_train_rows", type=int, default=0)
    p.add_argument("--t_dis_init", type=float, default=0.07)
    p.add_argument("--show_progress", action="store_true", default=True)
    p.add_argument("--print_epoch_summary", action="store_true", default=True)
    p.add_argument("--log_every_steps", type=int, default=200)
    p.add_argument("--save_every_epochs", type=int, default=0)
    p.add_argument("--write_epoch_metrics", action="store_true")
    return p.parse_args()


def main() -> int:
    _train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
