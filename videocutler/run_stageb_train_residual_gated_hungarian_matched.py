#!/usr/bin/env python3
"""A8.2 minimal Hungarian matched-pair training.

This is a side-path experiment. It consumes A8.1 Hungarian matched pairs and
trains the existing text-side projector / temperature using only GT carrier
features, clip-level full-Y candidate sets, and the fixed matched pseudo labels.

Two arms are supported with identical data/settings:
  * --loss ce: matched-pair cross entropy over full-Y classes in the clip
  * --loss infonce: row-wise InfoNCE over the same full-Y class denominator

Under this strict same-denominator setting, InfoNCE and CE are mathematically
very close; the purpose is to keep the loss ablation locked while avoiding any
extra sample/negative-pool change.
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
    _build_text_projection,
    _carrier_tensor,
    _compute_t_dis,
    _default_row_gap_path,
    _inverse_softplus,
    _load_checkpoint_if_requested,
    _load_row_gap,
    _prepare_data,
    _truth,
    _write_csv,
    _write_json,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


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


def _save_training_checkpoint(
    checkpoint_path: Path,
    *,
    projector: Projector,
    theta_t: torch.nn.Parameter,
    loss_name: str,
    epoch: int,
    global_step: int,
    matched_pairs_csv: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text_projector_state_dict": projector.state_dict(),
            "theta_T": float(theta_t.detach().cpu().item()),
            "loss": str(loss_name),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "matched_pairs_csv": str(matched_pairs_csv),
        },
        checkpoint_path,
    )


def _default_matched_pairs_path(run_root: Path, dataset_name: str) -> Path:
    return run_root / "analysis" / "residual_gated_hungarian_matching" / str(dataset_name) / "hungarian_matched_pairs.csv"


def _default_output_root(run_root: Path, dataset_name: str, loss_name: str) -> Path:
    return run_root / "outputs" / "a8_hungarian_matched_training" / str(dataset_name) / str(loss_name)


def _example_by_tid(data: Any) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for ex in data.examples:
        tid = str(ex.get("trajectory_id", ""))
        if tid and tid not in out:
            out[tid] = ex
    return out


def _load_train_pairs(path: Path, data: Any, max_rows: int = 0, seed: int = 3407) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_rows = _read_csv(path)
    example_by_tid = _example_by_tid(data)
    rows: List[Dict[str, Any]] = []
    counters = Counter()
    for r in raw_rows:
        tid = str(r.get("trajectory_id", ""))
        rid = _as_int(r.get("matched_raw_id"))
        clip_id = _as_int(r.get("clip_id"))
        if not tid or rid is None or clip_id is None:
            counters["skip_missing_key"] += 1; continue
        if tid not in example_by_tid:
            counters["skip_missing_example"] += 1; continue
        y_ids = sorted(int(x) for x in data.clip_y_base.get(int(clip_id), set()) if int(x) in data.raw_to_text_idx)
        if int(rid) not in y_ids:
            counters["skip_matched_class_not_in_full_y"] += 1; continue
        rows.append({
            "clip_id": int(clip_id),
            "trajectory_id": tid,
            "matched_raw_id": int(rid),
            "matched_class_name": str(r.get("matched_class_name", "")),
            "match_score": _as_float(r.get("match_score"), 0.0),
            "match_rank_in_row": _as_int(r.get("match_rank_in_row"), 0) or 0,
            "match_margin_vs_best_other": _as_float(r.get("match_margin_vs_best_other"), 0.0),
            "audit_gt_raw_id": _norm_id(r.get("audit_gt_raw_id")),
            "audit_assignment_matches_gt": int(_truth(r.get("audit_assignment_matches_gt"))),
            "audit_old_nohub_wrong": int(_truth(r.get("audit_old_nohub_wrong"))),
        })
        counters["selected_rows"] += 1
    if int(max_rows) > 0 and len(rows) > int(max_rows):
        rng = random.Random(int(seed))
        rows = rng.sample(rows, int(max_rows))
        counters["subsampled_train_rows"] += 1
    return rows, {"matched_pairs_csv": str(path), "input_rows": len(raw_rows), "selected_rows": len(rows), "counters": dict(counters)}


def _group_rows_by_clip(rows: Sequence[Mapping[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[int(r["clip_id"])].append(dict(r))
    return dict(out)


def _project_text(projector: Projector, text_tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(projector(text_tensor), p=2.0, dim=-1)


def _loss_for_clip(
    *,
    rows: Sequence[Mapping[str, Any]],
    data: Any,
    example_by_tid: Mapping[str, Mapping[str, Any]],
    text_proj_all: torch.Tensor,
    theta_t: torch.nn.Parameter,
    device: torch.device,
    loss_name: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    clip_id = int(rows[0]["clip_id"])
    y_ids = sorted(int(x) for x in data.clip_y_base.get(int(clip_id), set()) if int(x) in data.raw_to_text_idx)
    if not y_ids:
        raise RuntimeError(f"clip {clip_id} has empty full-Y after text-bank filter")
    y_col = {int(rid): j for j, rid in enumerate(y_ids)}
    z_vecs = []
    targets = []
    kept_rows = []
    for r in rows:
        rid = int(r["matched_raw_id"])
        if rid not in y_col:
            continue
        ex = example_by_tid[str(r["trajectory_id"])]
        z_vecs.append(torch.from_numpy(np.asarray(ex["carrier_vec"], dtype=np.float32)))
        targets.append(y_col[rid])
        kept_rows.append(r)
    if not z_vecs:
        raise RuntimeError(f"no trainable rows remained in clip {clip_id}")
    Z = torch.stack(z_vecs, dim=0).to(device=device, dtype=torch.float32)
    Z = F.normalize(Z, p=2.0, dim=-1)
    text_idx = torch.tensor([int(data.raw_to_text_idx[int(rid)]) for rid in y_ids], device=device, dtype=torch.long)
    T = text_proj_all[text_idx]
    logits = torch.matmul(Z, T.t()) / _compute_t_dis(theta_t)
    target = torch.tensor(targets, device=device, dtype=torch.long)
    if str(loss_name) == "ce":
        loss = F.cross_entropy(logits, target, reduction="mean")
    elif str(loss_name) == "infonce":
        # Row-wise InfoNCE over the exact same full-Y denominator as CE.
        pos = logits.gather(1, target.view(-1, 1)).squeeze(1)
        loss = -(pos - torch.logsumexp(logits, dim=1)).mean()
    else:
        raise ValueError(f"unsupported loss: {loss_name}")
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        pseudo_acc = float((pred == target).float().mean().detach().cpu().item())
    return loss, {"rows": len(kept_rows), "clip_id": clip_id, "pseudo_top1_acc": pseudo_acc}


def _load_eval_rows(row_gap_csv: Path, data: Any, max_rows: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    row_gap_by_key, row_gap_rows, summary = _load_row_gap(row_gap_csv)
    example_by_tid = _example_by_tid(data)
    rows: List[Dict[str, Any]] = []
    counters = Counter()
    for r in row_gap_rows:
        tid = str(r.get("trajectory_id", ""))
        gt = _as_int(r.get("gt_raw_id"))
        clip_id = _as_int(r.get("clip_id"))
        if not tid or gt is None or clip_id is None:
            counters["skip_missing_key"] += 1; continue
        if tid not in example_by_tid:
            counters["skip_missing_example"] += 1; continue
        y_ids = sorted(int(x) for x in data.clip_y_base.get(int(clip_id), set()) if int(x) in data.raw_to_text_idx)
        if int(gt) not in y_ids:
            counters["skip_gt_not_in_full_y"] += 1; continue
        rows.append({
            "clip_id": int(clip_id),
            "trajectory_id": tid,
            "gt_raw_id": int(gt),
            "gt_class_name": str(r.get("gt_class_name", "")),
            "weak_nohub_top1_is_gt": int(_truth(r.get("weak_nohub_top1_is_gt"))),
            "weak_base_top1_is_gt": int(_truth(r.get("weak_base_top1_is_gt"))),
            "weak_nohub_error_type": str(r.get("weak_nohub_error_type", "")),
        })
    if int(max_rows) > 0 and len(rows) > int(max_rows):
        rng = random.Random(int(seed)); rows = rng.sample(rows, int(max_rows)); counters["subsampled_eval_rows"] += 1
    summary = dict(summary)
    summary.update({"valid_eval_rows": len(rows), "counters": dict(counters)})
    return rows, summary


def _evaluate_full_y(
    *,
    stage: str,
    rows: Sequence[Mapping[str, Any]],
    data: Any,
    example_by_tid: Mapping[str, Mapping[str, Any]],
    projector: Projector,
    text_tensor: torch.Tensor,
    theta_t: torch.nn.Parameter,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, Any]:
    projector.eval()
    pred_rows: List[Dict[str, Any]] = []
    by_class = defaultdict(lambda: Counter())
    ranks: List[int] = []
    norm_ranks: List[float] = []
    with torch.no_grad():
        text_proj_all = _project_text(projector, text_tensor)
        grouped = _group_rows_by_clip(rows)
        iterator = grouped.items()
        for clip_id, crs in iterator:
            y_ids = sorted(int(x) for x in data.clip_y_base.get(int(clip_id), set()) if int(x) in data.raw_to_text_idx)
            if not y_ids:
                continue
            y_col = {int(rid): j for j, rid in enumerate(y_ids)}
            z_vecs = []
            valid = []
            for r in crs:
                gt = int(r["gt_raw_id"])
                if gt not in y_col:
                    continue
                ex = example_by_tid[str(r["trajectory_id"])]
                z_vecs.append(torch.from_numpy(np.asarray(ex["carrier_vec"], dtype=np.float32)))
                valid.append(r)
            if not valid:
                continue
            Z = torch.stack(z_vecs, dim=0).to(device=device, dtype=torch.float32)
            Z = F.normalize(Z, p=2.0, dim=-1)
            text_idx = torch.tensor([int(data.raw_to_text_idx[int(rid)]) for rid in y_ids], device=device, dtype=torch.long)
            T = text_proj_all[text_idx]
            logits = (torch.matmul(Z, T.t()) / _compute_t_dis(theta_t)).detach().cpu().numpy().astype(np.float32)
            for i, r in enumerate(valid):
                gt = int(r["gt_raw_id"])
                gt_col = int(y_col[gt])
                vals = logits[i]
                order = np.argsort(-vals)
                top_col = int(order[0])
                top2_col = int(order[1]) if len(order) > 1 else top_col
                gt_score = float(vals[gt_col])
                top1_score = float(vals[top_col])
                top2_score = float(vals[top2_col]) if len(order) > 1 else None
                rank = int(1 + np.sum(vals > vals[gt_col]))
                top1 = int(y_ids[top_col])
                top5_cols = set(int(x) for x in order[: min(5, len(order))])
                top1_hit = int(top1 == gt)
                top5_hit = int(gt_col in top5_cols)
                norm_rank = (rank - 1) / max(len(y_ids) - 1, 1)
                score_margin = float(gt_score - top1_score)
                wrong_abs_gap = float(max(top1_score - gt_score, 0.0)) if top1_hit == 0 else 0.0
                ranks.append(rank); norm_ranks.append(float(norm_rank))
                by_class[str(gt)]["rows"] += 1
                by_class[str(gt)]["top1"] += top1_hit
                by_class[str(gt)]["top5"] += top5_hit
                by_class[str(gt)]["rank_sum"] += rank
                pred_rows.append({
                    "clip_id": r["clip_id"], "trajectory_id": r["trajectory_id"], "gt_raw_id": gt, "gt_class_name": r.get("gt_class_name", ""),
                    "top1_raw_id": top1, "top1_hit": top1_hit, "top5_hit": top5_hit, "gt_rank": rank, "normalized_gt_rank": norm_rank,
                    "clip_y_size": len(y_ids), "weak_nohub_top1_is_gt": r.get("weak_nohub_top1_is_gt", ""), "weak_nohub_error_type": r.get("weak_nohub_error_type", ""),
                    "score_domain": "full_y_clip_logits_div_t_dis",
                    "gt_score": gt_score,
                    "top1_score": top1_score,
                    "top2_score": top2_score,
                    "score_margin": score_margin,
                    "wrong_abs_gap": wrong_abs_gap,
                    "candidate_count": len(y_ids),
                    "candidate_raw_ids_json": json.dumps([int(x) for x in y_ids], ensure_ascii=False),
                    "candidate_scores_json": json.dumps([float(x) for x in vals.tolist()], ensure_ascii=False),
                })
    by_class_rows: List[Dict[str, Any]] = []
    for rid, c in sorted(by_class.items(), key=lambda x: int(float(x[0]))):
        n = int(c["rows"])
        by_class_rows.append({"raw_id": rid, "gt_count": n, "gt_top1_hit_rate": c["top1"] / max(n, 1), "gt_top5_hit_rate": c["top5"] / max(n, 1), "mean_rank": c["rank_sum"] / max(n, 1)})
    summary = {
        "stage": stage,
        "eval_rows": len(pred_rows),
        "class_count": len(by_class_rows),
        "micro_top1": sum(int(r["top1_hit"]) for r in pred_rows) / max(len(pred_rows), 1),
        "micro_top5": sum(int(r["top5_hit"]) for r in pred_rows) / max(len(pred_rows), 1),
        "mean_rank": _mean([float(x) for x in ranks]),
        "mean_normalized_gt_rank": _mean(norm_ranks),
        "macro_rank1": _mean([float(r["gt_top1_hit_rate"]) for r in by_class_rows]),
        "macro_top5": _mean([float(r["gt_top5_hit_rate"]) for r in by_class_rows]),
    }
    _write_csv(out_dir / f"eval_{stage}_row_predictions.csv", pred_rows)
    _write_csv(out_dir / f"eval_{stage}_by_class.csv", by_class_rows)
    _write_json(out_dir / f"eval_{stage}_summary.json", summary)
    projector.train()
    return summary


def _compare_eval(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ["micro_top1", "micro_top5", "mean_normalized_gt_rank", "macro_rank1", "macro_top5"]
    return {f"delta_{k}": float(after.get(k, 0.0)) - float(before.get(k, 0.0)) for k in keys}


def _train(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve() if str(args.output_root).strip() else _default_output_root(run_root, str(args.dataset_name), str(args.loss))
    out_root.mkdir(parents=True, exist_ok=True)
    train_dir = out_root / "train" / "a8_hungarian_matched"
    train_dir.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed)); np.random.seed(int(args.seed)); torch.manual_seed(int(args.seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device(str(args.device))

    if not str(args.annotation_json).strip():
        args.annotation_json = str(Path(args.repo_root) / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json")
    if not str(args.split_json).strip():
        args.split_json = str(Path(args.repo_root) / "package" / "reference" / "lvvis_official_base_novel_split.json")

    # _prepare_data is reused from the A8 coverage-assignment audit path.
    # That helper expects args.output_dir for optional asset-root resolution.
    # In this training entrypoint, assets resolve from run_root while training
    # artifacts are written to output_root. Keep output_dir empty unless a
    # future caller explicitly sets it.
    if not hasattr(args, "output_dir"):
        args.output_dir = ""
    data = _prepare_data(args)
    example_by_tid = _example_by_tid(data)
    matched_csv = Path(args.matched_pairs_csv).expanduser().resolve() if str(args.matched_pairs_csv).strip() else _default_matched_pairs_path(run_root, str(args.dataset_name))
    if not matched_csv.is_file():
        raise FileNotFoundError(f"matched_pairs_csv not found: {matched_csv}")
    train_rows, train_row_summary = _load_train_pairs(matched_csv, data, max_rows=int(args.max_train_rows), seed=int(args.seed))
    if not train_rows:
        raise RuntimeError("no train rows selected from matched_pairs_csv")
    row_gap_csv = Path(args.row_gap_csv).expanduser().resolve() if str(args.row_gap_csv).strip() else _default_row_gap_path(repo_root, str(args.dataset_name))
    eval_rows, eval_summary = _load_eval_rows(row_gap_csv, data, max_rows=int(args.max_eval_rows), seed=int(args.seed))
    if not eval_rows:
        raise RuntimeError("no eval rows available")

    text_tensor = torch.tensor(np.asarray(data.text_matrix, dtype=np.float32), device=device, dtype=torch.float32)
    projector = Projector(ProjectorConfig()).to(device)
    theta_t = torch.nn.Parameter(torch.tensor(_inverse_softplus(max(float(args.t_dis_init) - 1.0e-4, 1.0e-6)), device=device, dtype=torch.float32))
    ckpt = _auto_find_checkpoint(repo_root, str(args.dataset_name)) if str(args.init_checkpoint).strip() == "auto" else str(args.init_checkpoint).strip()
    checkpoint_summary = _load_checkpoint_if_requested(projector, theta_t, ckpt, device)

    setup = {
        "timestamp": _now(), "loss": str(args.loss), "run_root": str(run_root), "output_root": str(out_root), "matched_pairs_csv": str(matched_csv), "row_gap_csv": str(row_gap_csv),
        "device": str(device), "epochs": int(args.epochs), "learning_rate": float(args.learning_rate), "seed": int(args.seed),
        "policy": {"uses_row_level_gt_for_training": False, "uses_nohub_correctness_for_training": False, "uses_dummy_or_slack": False, "uses_extra_support": False, "loss_only_arm": str(args.loss)},
        "checkpoint": checkpoint_summary, "materialization_summary": data.materialization_summary, "train_row_summary": train_row_summary, "eval_row_summary": eval_summary,
    }
    _write_json(out_root / "a8_hungarian_train_setup.json", setup)

    before = _evaluate_full_y(stage="before", rows=eval_rows, data=data, example_by_tid=example_by_tid, projector=projector, text_tensor=text_tensor, theta_t=theta_t, device=device, out_dir=out_root / "analysis")

    optimizer = torch.optim.AdamW([*projector.parameters(), theta_t], lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    train_by_clip = _group_rows_by_clip(train_rows)
    clip_ids = list(train_by_clip.keys())
    rng = random.Random(int(args.seed))
    log_path = train_dir / "train_log.jsonl"
    if log_path.exists(): log_path.unlink()
    epoch_metric_rows: List[Dict[str, Any]] = []
    global_step = 0
    all_losses: List[float] = []
    for epoch in (tqdm(range(int(args.epochs)), desc=f"a8_hungarian_{args.loss}_epochs", dynamic_ncols=True) if bool(args.show_progress) and tqdm is not None else range(int(args.epochs))):
        rng.shuffle(clip_ids)
        epoch_losses: List[float] = []
        epoch_rows = 0
        epoch_accs: List[float] = []
        for clip_id in clip_ids:
            rows = train_by_clip[clip_id]
            optimizer.zero_grad(set_to_none=True)
            text_proj_all = _project_text(projector, text_tensor)
            loss, stats = _loss_for_clip(rows=rows, data=data, example_by_tid=example_by_tid, text_proj_all=text_proj_all, theta_t=theta_t, device=device, loss_name=str(args.loss))
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_([*projector.parameters(), theta_t], max_norm=float(args.grad_clip_norm))
            optimizer.step()
            global_step += 1
            lv = float(loss.detach().cpu().item())
            epoch_losses.append(lv); all_losses.append(lv); epoch_rows += int(stats["rows"]); epoch_accs.append(float(stats["pseudo_top1_acc"]))
            if int(args.log_every_steps) > 0 and global_step % int(args.log_every_steps) == 0:
                _append_jsonl(log_path, {"timestamp": _now(), "epoch": int(epoch)+1, "global_step": global_step, "loss": lv, **stats})
        epoch_row = {"timestamp": _now(), "row_type": "epoch_summary", "epoch": int(epoch)+1, "loss_mean": _mean(epoch_losses), "loss_last": epoch_losses[-1] if epoch_losses else 0.0, "pseudo_top1_acc_mean": _mean(epoch_accs), "epoch_rows": epoch_rows, "epoch_clips": len(clip_ids)}
        _append_jsonl(log_path, epoch_row)
        epoch_metric_row = dict(epoch_row)
        if int(args.eval_every_epochs) > 0 and ((int(epoch) + 1) % int(args.eval_every_epochs) == 0):
            epoch_eval = _evaluate_full_y(
                stage=f"epoch_{int(epoch)+1:03d}",
                rows=eval_rows,
                data=data,
                example_by_tid=example_by_tid,
                projector=projector,
                text_tensor=text_tensor,
                theta_t=theta_t,
                device=device,
                out_dir=out_root / "analysis",
            )
            epoch_metric_row.update({
                "eval_micro_top1": float(epoch_eval.get("micro_top1", 0.0)),
                "eval_micro_top5": float(epoch_eval.get("micro_top5", 0.0)),
                "eval_mean_rank": float(epoch_eval.get("mean_rank", 0.0)),
                "eval_mean_normalized_gt_rank": float(epoch_eval.get("mean_normalized_gt_rank", 0.0)),
                "eval_macro_rank1": float(epoch_eval.get("macro_rank1", 0.0)),
                "eval_macro_top5": float(epoch_eval.get("macro_top5", 0.0)),
            })
        if int(args.save_every_epochs) > 0 and ((int(epoch) + 1) % int(args.save_every_epochs) == 0):
            _save_training_checkpoint(
                train_dir / f"a8_hungarian_epoch_{int(epoch)+1:03d}.pth",
                projector=projector,
                theta_t=theta_t,
                loss_name=str(args.loss),
                epoch=int(epoch) + 1,
                global_step=global_step,
                matched_pairs_csv=matched_csv,
            )
        epoch_metric_rows.append(epoch_metric_row)
        if bool(args.print_epoch_summary): print(json.dumps(epoch_row, ensure_ascii=False))

    after = _evaluate_full_y(stage="after", rows=eval_rows, data=data, example_by_tid=example_by_tid, projector=projector, text_tensor=text_tensor, theta_t=theta_t, device=device, out_dir=out_root / "analysis")
    cmp = _compare_eval(before, after)
    ckpt_out = train_dir / "a8_hungarian_last.pth"
    _save_training_checkpoint(
        ckpt_out,
        projector=projector,
        theta_t=theta_t,
        loss_name=str(args.loss),
        epoch=int(args.epochs),
        global_step=global_step,
        matched_pairs_csv=matched_csv,
    )
    if bool(args.write_epoch_metrics):
        _write_csv(train_dir / "epoch_metrics.csv", epoch_metric_rows)

    final = {
        "status": "PASS", "timestamp": _now(), "loss": str(args.loss), "output_root": str(out_root), "train_summary": {"epochs": int(args.epochs), "global_step": global_step, "loss_mean": _mean(all_losses), "loss_last": all_losses[-1] if all_losses else 0.0, "train_row_count": len(train_rows), "train_clip_count": len(clip_ids)},
        "eval_before": before, "eval_after": after, "eval_delta": cmp, "checkpoint": str(ckpt_out), "setup": setup,
    }
    _write_json(out_root / "final_summary.json", final)
    lines = [
        f"# A8.2 Hungarian Matched Training TAKEOVER ({args.loss})", "", "- status: PASS", f"- loss: {args.loss}", f"- epochs: {args.epochs}", f"- train_rows: {len(train_rows)}", f"- train_clips: {len(clip_ids)}", f"- before micro_top1: {before.get('micro_top1', 0.0):.6f}", f"- after micro_top1: {after.get('micro_top1', 0.0):.6f}", f"- delta micro_top1: {cmp.get('delta_micro_top1', 0.0):.6f}", f"- before macro_rank1: {before.get('macro_rank1', 0.0):.6f}", f"- after macro_rank1: {after.get('macro_rank1', 0.0):.6f}", f"- delta macro_rank1: {cmp.get('delta_macro_rank1', 0.0):.6f}", "", "## Policy", "- Uses fixed Hungarian matched pairs only.", "- No dummy/slack/extra_support.", "- Row-level GT and NoHub correctness are not used for training.",]
    (out_root / "A8_HUNGARIAN_TRAINING_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2, default=str))
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A8.2 minimal Hungarian matched-pair training")
    p.add_argument("--run_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--matched_pairs_csv", default="")
    p.add_argument("--output_root", default="")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--row_gap_csv", default="")
    p.add_argument("--init_checkpoint", default="auto")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--loss", choices=["ce", "infonce"], default="ce")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--max_train_rows", type=int, default=0)
    p.add_argument("--max_eval_rows", type=int, default=0)
    p.add_argument("--t_dis_init", type=float, default=0.07)
    p.add_argument("--show_progress", action="store_true", default=True)
    p.add_argument("--print_epoch_summary", action="store_true", default=True)
    p.add_argument("--log_every_steps", type=int, default=200)
    p.add_argument("--eval_every_epochs", type=int, default=0)
    p.add_argument("--save_every_epochs", type=int, default=0)
    p.add_argument("--write_epoch_metrics", action="store_true")
    return p.parse_args()


def main() -> int:
    _train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
