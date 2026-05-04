#!/usr/bin/env python3
"""
A8 CE Drift-Onset Audit

Read-only diagnostic. It does NOT train or modify model/evaluator semantics.
For each checkpoint epoch, it calls tools/a8_true_margin_export_audit.py, then
computes row-level drift relative to a baseline epoch.

Primary questions:
- When does CE training stop improving GT alignment and start corrupting rows?
- Which hubs expand at/after the drift point?
- Is drift concentrated in rows whose Hungarian pseudo label differs from GT?

This script is intentionally robust to minor CSV schema differences. When a
metric cannot be computed from available columns, it reports null instead of
silently fabricating values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SMALL_TH = 1.0
LARGE_TH = 3.0


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def _first_nonempty(row: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v) != "":
            return str(v)
    return default


def _row_key(row: Dict[str, Any]) -> str:
    # Prefer a stable explicit key if exporter provides one.
    for k in ["row_key", "eval_row_key", "key", "instance_key"]:
        v = row.get(k)
        if v:
            return str(v)
    clip = _first_nonempty(row, ["clip_id", "video_id", "movie_id"])
    traj = _first_nonempty(row, ["trajectory_id", "traj_id", "trajectory_key", "proposal_id"])
    if clip or traj:
        return f"{clip}||{traj}"
    # Fallback: this should rarely happen; preserve determinism.
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def _is_gt_top1(row: Dict[str, Any]) -> bool:
    for k in ["top1_hit", "gt_top1_hit", "is_top1_gt", "top1_is_gt"]:
        if k in row and row.get(k) != "":
            return _truthy(row.get(k)) or _as_int(row.get(k), 0) == 1
    gt = _first_nonempty(row, ["gt_raw_id", "gt_category_id", "raw_gt_id"])
    top1 = _first_nonempty(row, ["top1_raw_id", "wrong_top1_raw_id", "pred_raw_id", "top1_category_id"])
    return bool(gt and top1 and gt == top1)


def _gt_raw_id(row: Dict[str, Any]) -> str:
    return _first_nonempty(row, ["gt_raw_id", "gt_category_id", "raw_gt_id", "category_id"])


def _gt_class_name(row: Dict[str, Any]) -> str:
    return _first_nonempty(row, ["gt_class_name", "gt_name", "category_name", "class_name"])


def _top1_raw_id(row: Dict[str, Any]) -> str:
    return _first_nonempty(row, ["top1_raw_id", "wrong_top1_raw_id", "pred_raw_id", "top1_category_id"])


def _top1_class_name(row: Dict[str, Any]) -> str:
    return _first_nonempty(row, ["top1_class_name", "wrong_top1_class_name", "pred_class_name", "top1_name"])


def _wrong_abs_gap(row: Dict[str, Any]) -> float:
    # Positive only for wrong rows in current exporter; robust to alternative names.
    return _as_float(_first_nonempty(row, ["wrong_abs_gap", "margin_abs_gap", "top1_minus_gt", "abs_gap"]), 0.0)


def _score_margin(row: Dict[str, Any]) -> float:
    if "score_margin" in row and row.get("score_margin") != "":
        return _as_float(row.get("score_margin"), 0.0)
    if _is_gt_top1(row):
        return 0.0
    return -_wrong_abs_gap(row)


def _margin_bucket(row: Dict[str, Any]) -> str:
    if _is_gt_top1(row):
        return "correct"
    gap = _wrong_abs_gap(row)
    if gap <= SMALL_TH:
        return "small_le_1"
    if gap >= LARGE_TH:
        return "large_ge_3"
    return "middle_1_to_3"


def _load_residual_buckets(path: Optional[Path]) -> Dict[str, str]:
    if not path or not path.exists():
        return {}
    by_id: Dict[str, Dict[str, str]] = {}
    for r in _read_csv(path):
        rid = _first_nonempty(r, ["raw_category_id", "raw_id", "category_id", "gt_raw_id"])
        if not rid:
            continue
        old = by_id.get(rid)
        # If multiple variants exist, prefer person-aware rows.
        if old is None or "person" in str(r.get("variant") or "").lower():
            by_id[rid] = r
    out = {}
    for rid, r in by_id.items():
        if not _truthy(r.get("resolved")):
            out[rid] = "unresolved"
            continue
        it = _as_int(r.get("resolved_at_iteration"), -1)
        if it == 0:
            out[rid] = "iter0_initial_anchor"
        elif it == 1:
            out[rid] = "iter1_first_peeling"
        elif it >= 2:
            out[rid] = "iter2plus_late_chain"
        else:
            out[rid] = "resolved_unknown_iteration"
    return out


def _load_matched_pairs(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not path or not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for r in _read_csv(path):
        k = _row_key(r)
        out[k] = r
    return out


def _pseudo_raw_id(matched_row: Optional[Dict[str, str]]) -> str:
    if not matched_row:
        return ""
    return _first_nonempty(
        matched_row,
        [
            "matched_raw_id",
            "target_raw_id",
            "pseudo_raw_id",
            "matched_category_id",
            "category_id",
            "raw_category_id",
        ],
    )


def _discover_checkpoints(checkpoint_dir: Path, epochs: Optional[List[int]]) -> Dict[int, Path]:
    ckpts: Dict[int, Path] = {}
    if epochs:
        for ep in epochs:
            candidates = [
                checkpoint_dir / f"a8_hungarian_epoch_{ep:03d}.pth",
                checkpoint_dir / f"a8_hungarian_epoch_{ep}.pth",
                checkpoint_dir / f"epoch_{ep:03d}.pth",
                checkpoint_dir / f"epoch_{ep}.pth",
            ]
            for c in candidates:
                if c.exists():
                    ckpts[ep] = c
                    break
        return ckpts

    for p in sorted(checkpoint_dir.glob("*.pth")):
        m = re.search(r"epoch[_-]?(\d+)", p.name)
        if m:
            ckpts[int(m.group(1))] = p
    return ckpts


def _run_true_margin_export(
    repo_root: Path,
    run_root: Path,
    dataset_name: str,
    checkpoint_path: Path,
    row_gap_csv: Path,
    output_dir: Path,
    asset_root: Optional[Path],
    device: str,
    force: bool,
) -> None:
    validation = output_dir / "validation_summary.json"
    row_csv = output_dir / "true_score_margin_row_audit.csv"
    if validation.exists() and row_csv.exists() and not force:
        return
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "a8_true_margin_export_audit.py"),
        "--run_root",
        str(run_root),
        "--dataset_name",
        dataset_name,
        "--checkpoint_path",
        str(checkpoint_path),
        "--repo_root",
        str(repo_root),
        "--row_gap_csv",
        str(row_gap_csv),
        "--device",
        device,
        "--output_dir",
        str(output_dir),
    ]
    if asset_root:
        cmd.extend(["--asset_root", str(asset_root)])
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def _epoch_summary(
    epoch: int,
    validation_json: Path,
    row_csv: Path,
    baseline_rows: Optional[Dict[str, Dict[str, str]]],
    residual_buckets: Dict[str, str],
    matched_pairs: Dict[str, Dict[str, str]],
    hub_ids: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    val = _load_json(validation_json)
    rows = _read_csv(row_csv)
    eval_summary = val.get("eval_summary", {}) if isinstance(val, dict) else {}

    total_wrong = 0
    small = middle = large = 0
    small_i01 = middle_i01 = large_i01 = 0
    iter2plus = unresolved = 0
    absorber = Counter()
    absorber_src: Dict[str, set] = defaultdict(set)
    hub_gap_sum = Counter()

    pseudo_mismatch_rows = 0
    pseudo_wrong_top1_rows = 0
    pseudo_correct_rows = 0
    pseudo_label_available_rows = 0

    row_transitions: List[Dict[str, Any]] = []
    correct_to_wrong = wrong_to_correct = stable_correct = stable_wrong = 0
    wrong_same_top1 = wrong_changed_top1 = 0
    corrupted_to_hub = 0
    fixed_from_hub = 0

    gt_score_sum = top1_score_sum = 0.0
    score_margin_sum = 0.0
    score_margin_wrong_sum = 0.0
    wrong_count_for_margin = 0

    for r in rows:
        k = _row_key(r)
        gt = _gt_raw_id(r)
        gt_name = _gt_class_name(r)
        top1 = _top1_raw_id(r)
        top1_name = _top1_class_name(r)
        correct = _is_gt_top1(r)
        gap = _wrong_abs_gap(r)
        bucket = _margin_bucket(r)
        residual_bucket = residual_buckets.get(gt, "unknown")
        pseudo = _pseudo_raw_id(matched_pairs.get(k))
        pseudo_available = bool(pseudo)

        if pseudo_available:
            pseudo_label_available_rows += 1
            if pseudo != gt:
                pseudo_mismatch_rows += 1
            else:
                pseudo_correct_rows += 1
            if top1 and top1 == pseudo and pseudo != gt:
                pseudo_wrong_top1_rows += 1

        gt_score_sum += _as_float(r.get("gt_score"), 0.0)
        top1_score_sum += _as_float(r.get("top1_score"), 0.0)
        score_margin_sum += _score_margin(r)

        if not correct:
            total_wrong += 1
            wrong_count_for_margin += 1
            score_margin_wrong_sum += _score_margin(r)
            if bucket == "small_le_1":
                small += 1
            elif bucket == "middle_1_to_3":
                middle += 1
            elif bucket == "large_ge_3":
                large += 1

            if residual_bucket in {"iter0_initial_anchor", "iter1_first_peeling"}:
                if bucket == "small_le_1":
                    small_i01 += 1
                elif bucket == "middle_1_to_3":
                    middle_i01 += 1
                elif bucket == "large_ge_3":
                    large_i01 += 1
            elif residual_bucket == "iter2plus_late_chain":
                iter2plus += 1
            elif residual_bucket == "unresolved":
                unresolved += 1

            if top1:
                absorber[top1] += 1
                absorber_src[top1].add(gt)
                hub_gap_sum[top1] += gap

        if baseline_rows is not None and k in baseline_rows:
            br = baseline_rows[k]
            b_correct = _is_gt_top1(br)
            b_top1 = _top1_raw_id(br)
            if b_correct and correct:
                stable_correct += 1
            elif (not b_correct) and (not correct):
                stable_wrong += 1
                if b_top1 == top1:
                    wrong_same_top1 += 1
                else:
                    wrong_changed_top1 += 1
            elif b_correct and not correct:
                correct_to_wrong += 1
                if top1 in hub_ids:
                    corrupted_to_hub += 1
                row_transitions.append(
                    {
                        "epoch": epoch,
                        "transition": "correct_to_wrong",
                        "row_key": k,
                        "gt_raw_id": gt,
                        "gt_class_name": gt_name,
                        "baseline_top1_raw_id": b_top1,
                        "epoch_top1_raw_id": top1,
                        "epoch_top1_class_name": top1_name,
                        "wrong_abs_gap": gap,
                        "margin_bucket": bucket,
                        "residual_bucket": residual_bucket,
                        "pseudo_raw_id": pseudo,
                        "pseudo_matches_gt": int(pseudo == gt) if pseudo_available else "",
                        "top1_matches_pseudo": int(top1 == pseudo) if pseudo_available else "",
                    }
                )
            elif (not b_correct) and correct:
                wrong_to_correct += 1
                if b_top1 in hub_ids:
                    fixed_from_hub += 1
                row_transitions.append(
                    {
                        "epoch": epoch,
                        "transition": "wrong_to_correct",
                        "row_key": k,
                        "gt_raw_id": gt,
                        "gt_class_name": gt_name,
                        "baseline_top1_raw_id": b_top1,
                        "epoch_top1_raw_id": top1,
                        "epoch_top1_class_name": top1_name,
                        "wrong_abs_gap": gap,
                        "margin_bucket": bucket,
                        "residual_bucket": residual_bucket,
                        "pseudo_raw_id": pseudo,
                        "pseudo_matches_gt": int(pseudo == gt) if pseudo_available else "",
                        "top1_matches_pseudo": int(top1 == pseudo) if pseudo_available else "",
                    }
                )

    n_rows = max(len(rows), 1)
    hub_rows = []
    for hid, n in absorber.most_common():
        hub_rows.append(
            {
                "epoch": epoch,
                "hub_raw_id": hid,
                "absorbed_wrong_rows": n,
                "source_class_count": len(absorber_src[hid]),
                "mean_wrong_abs_gap": hub_gap_sum[hid] / max(n, 1),
                "is_tracked_hub": int(hid in hub_ids),
            }
        )

    summary = {
        "epoch": epoch,
        "row_count": len(rows),
        "micro_top1": eval_summary.get("micro_top1"),
        "macro_rank1": eval_summary.get("macro_rank1"),
        "mean_normalized_gt_rank": eval_summary.get("mean_normalized_gt_rank"),
        "micro_top5": eval_summary.get("micro_top5"),
        "macro_top5": eval_summary.get("macro_top5"),
        "total_wrong_rows": total_wrong,
        "small_wrong_rows": small,
        "middle_wrong_rows": middle,
        "large_wrong_rows": large,
        "small_iter0_plus_iter1": small_i01,
        "middle_iter0_plus_iter1": middle_i01,
        "large_iter0_plus_iter1": large_i01,
        "iter2plus_wrong_rows": iter2plus,
        "unresolved_wrong_rows": unresolved,
        "mean_score_margin_all": score_margin_sum / n_rows,
        "mean_score_margin_wrong": score_margin_wrong_sum / max(wrong_count_for_margin, 1),
        "pseudo_label_available_rows": pseudo_label_available_rows,
        "pseudo_correct_rows": pseudo_correct_rows,
        "pseudo_mismatch_rows": pseudo_mismatch_rows,
        "pseudo_wrong_top1_rows": pseudo_wrong_top1_rows,
        "pseudo_wrong_top1_rate_among_available": pseudo_wrong_top1_rows / max(pseudo_label_available_rows, 1),
        "stable_correct_vs_baseline": stable_correct,
        "stable_wrong_vs_baseline": stable_wrong,
        "correct_to_wrong_vs_baseline": correct_to_wrong,
        "wrong_to_correct_vs_baseline": wrong_to_correct,
        "wrong_same_top1_vs_baseline": wrong_same_top1,
        "wrong_changed_top1_vs_baseline": wrong_changed_top1,
        "corrupted_to_tracked_hub": corrupted_to_hub,
        "fixed_from_tracked_hub": fixed_from_hub,
    }
    # Add tracked hub columns.
    for hid in hub_ids:
        summary[f"hub_{hid}_absorbed_wrong_rows"] = absorber.get(hid, 0)
        summary[f"hub_{hid}_source_class_count"] = len(absorber_src.get(hid, set()))
        summary[f"hub_{hid}_mean_wrong_abs_gap"] = hub_gap_sum.get(hid, 0.0) / max(absorber.get(hid, 0), 1)

    return summary, row_transitions, hub_rows


def _parse_epochs(s: str) -> Optional[List[int]]:
    if not s or s.strip().lower() in {"auto", "discover"}:
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _read_loss_by_epoch(checkpoint_dir: Path) -> Dict[int, Dict[str, Any]]:
    # epoch_metrics.csv is next to checkpoints in train/a8_hungarian_matched.
    p = checkpoint_dir / "epoch_metrics.csv"
    if not p.exists():
        return {}
    out = {}
    for r in _read_csv(p):
        if r.get("row_type") != "epoch_summary":
            continue
        ep = _as_int(r.get("epoch"), -1)
        if ep < 0:
            continue
        out[ep] = r
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="A8 CE drift-onset audit over checkpoint epochs.")
    ap.add_argument("--repo_root", type=Path, default=Path("/mnt/sda/zyy/code/wsovvis"))
    ap.add_argument("--run_root", type=Path, required=True)
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--checkpoint_dir", type=Path, required=True)
    ap.add_argument("--epochs", default="5,10,15,20,25,30,35,40,45,50,75,100,150,200")
    ap.add_argument("--baseline_epoch", type=int, default=5)
    ap.add_argument("--row_gap_csv", type=Path, required=True)
    ap.add_argument("--matched_pairs_csv", type=Path, default=None)
    ap.add_argument("--residual_csv", type=Path, default=None)
    ap.add_argument("--asset_root", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--hub_raw_ids", default="auto", help="Comma-separated hub raw ids, or 'auto' from baseline top absorbers.")
    ap.add_argument("--auto_hub_topk", type=int, default=12)
    ap.add_argument("--force_export", action="store_true")
    ap.add_argument("--skip_export", action="store_true", help="Assume true-margin outputs already exist under output_dir/epoch_XXX_true_margin.")
    args = ap.parse_args(argv)

    repo_root: Path = args.repo_root
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_epochs = _parse_epochs(args.epochs)
    ckpts = _discover_checkpoints(args.checkpoint_dir, requested_epochs)
    if not ckpts:
        raise SystemExit(f"No checkpoints found in {args.checkpoint_dir} for epochs={args.epochs}")
    if args.baseline_epoch not in ckpts:
        raise SystemExit(f"Baseline epoch {args.baseline_epoch} not found. Available: {sorted(ckpts)}")

    export_dirs: Dict[int, Path] = {}
    export_status = []
    for ep, ckpt in sorted(ckpts.items()):
        ep_out = out_dir / f"epoch_{ep:03d}_true_margin"
        export_dirs[ep] = ep_out
        if not args.skip_export:
            _run_true_margin_export(
                repo_root=repo_root,
                run_root=args.run_root,
                dataset_name=args.dataset_name,
                checkpoint_path=ckpt,
                row_gap_csv=args.row_gap_csv,
                output_dir=ep_out,
                asset_root=args.asset_root,
                device=args.device,
                force=args.force_export,
            )
        row_csv = ep_out / "true_score_margin_row_audit.csv"
        val_json = ep_out / "validation_summary.json"
        export_status.append(
            {
                "epoch": ep,
                "checkpoint": str(ckpt),
                "true_margin_dir": str(ep_out),
                "row_csv_exists": row_csv.exists(),
                "validation_json_exists": val_json.exists(),
            }
        )
        if not row_csv.exists() or not val_json.exists():
            raise SystemExit(f"Missing true-margin output for epoch {ep}: {ep_out}")

    baseline_rows_list = _read_csv(export_dirs[args.baseline_epoch] / "true_score_margin_row_audit.csv")
    baseline_rows = {_row_key(r): r for r in baseline_rows_list}

    # Hub ids: auto from baseline top wrong absorbers unless explicitly provided.
    if args.hub_raw_ids.strip().lower() == "auto":
        cnt = Counter()
        for r in baseline_rows_list:
            if not _is_gt_top1(r):
                top1 = _top1_raw_id(r)
                if top1:
                    cnt[top1] += 1
        hub_ids = [hid for hid, _ in cnt.most_common(args.auto_hub_topk)]
    else:
        hub_ids = [x.strip() for x in args.hub_raw_ids.split(",") if x.strip()]

    residual_buckets = _load_residual_buckets(args.residual_csv)
    matched_pairs = _load_matched_pairs(args.matched_pairs_csv)
    loss_rows = _read_loss_by_epoch(args.checkpoint_dir)

    epoch_rows: List[Dict[str, Any]] = []
    all_transitions: List[Dict[str, Any]] = []
    all_hubs: List[Dict[str, Any]] = []
    for ep in sorted(export_dirs):
        summary, transitions, hub_rows = _epoch_summary(
            ep,
            export_dirs[ep] / "validation_summary.json",
            export_dirs[ep] / "true_score_margin_row_audit.csv",
            baseline_rows=baseline_rows,
            residual_buckets=residual_buckets,
            matched_pairs=matched_pairs,
            hub_ids=hub_ids,
        )
        if ep in loss_rows:
            lr = loss_rows[ep]
            summary["loss_mean"] = _as_float(lr.get("loss_mean"), math.nan)
            summary["loss_last"] = _as_float(lr.get("loss_last"), math.nan)
            summary["pseudo_top1_acc_mean_train"] = _as_float(lr.get("pseudo_top1_acc_mean"), math.nan)
        else:
            summary["loss_mean"] = ""
            summary["loss_last"] = ""
            summary["pseudo_top1_acc_mean_train"] = ""
        epoch_rows.append(summary)
        all_transitions.extend(transitions)
        all_hubs.extend(hub_rows)

    # Determine drift onset. Conservative rule: first epoch after baseline where
    # micro_top1 decreases vs previous saved epoch OR correct_to_wrong increases,
    # while loss decreases if loss is available.
    drift_candidates = []
    prev = None
    for r in epoch_rows:
        if r["epoch"] <= args.baseline_epoch:
            prev = r
            continue
        if prev is None:
            prev = r
            continue
        micro = _as_float(r.get("micro_top1"), math.nan)
        prev_micro = _as_float(prev.get("micro_top1"), math.nan)
        c2w = _as_int(r.get("correct_to_wrong_vs_baseline"), 0)
        prev_c2w = _as_int(prev.get("correct_to_wrong_vs_baseline"), 0)
        loss = _as_float(r.get("loss_mean"), math.nan)
        prev_loss = _as_float(prev.get("loss_mean"), math.nan)
        loss_decreasing = (not math.isnan(loss) and not math.isnan(prev_loss) and loss <= prev_loss)
        metric_worse = (not math.isnan(micro) and not math.isnan(prev_micro) and micro < prev_micro)
        c2w_worse = c2w > prev_c2w
        if metric_worse or c2w_worse:
            drift_candidates.append(
                {
                    "epoch": r["epoch"],
                    "prev_epoch": prev["epoch"],
                    "metric_worse": metric_worse,
                    "correct_to_wrong_increased": c2w_worse,
                    "loss_decreasing": loss_decreasing,
                    "micro_top1": micro,
                    "prev_micro_top1": prev_micro,
                    "correct_to_wrong": c2w,
                    "prev_correct_to_wrong": prev_c2w,
                    "loss_mean": loss,
                    "prev_loss_mean": prev_loss,
                }
            )
        prev = r

    _write_csv(out_dir / "epoch_level_drift_table.csv", epoch_rows)
    _write_csv(out_dir / "row_transition_events.csv", all_transitions)
    _write_csv(out_dir / "hub_onset_table.csv", all_hubs)

    # Compact markdown.
    md = []
    md.append("# A8 CE Drift-Onset Audit")
    md.append("")
    md.append(f"- dataset: `{args.dataset_name}`")
    md.append(f"- checkpoint_dir: `{args.checkpoint_dir}`")
    md.append(f"- baseline_epoch: `{args.baseline_epoch}`")
    md.append(f"- tracked_hub_raw_ids: `{','.join(hub_ids)}`")
    md.append("")
    md.append("## Epoch-level table")
    md.append("")
    md.append("| epoch | loss | micro | macro | mean_norm_rank | total_wrong | large_i01 | middle_i01 | small_i01 | c2w | w2c | pseudo_wrong_top1 |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in epoch_rows:
        md.append(
            f"| {r['epoch']} | {r.get('loss_mean','')} | {r.get('micro_top1','')} | {r.get('macro_rank1','')} | {r.get('mean_normalized_gt_rank','')} | "
            f"{r.get('total_wrong_rows','')} | {r.get('large_iter0_plus_iter1','')} | {r.get('middle_iter0_plus_iter1','')} | {r.get('small_iter0_plus_iter1','')} | "
            f"{r.get('correct_to_wrong_vs_baseline','')} | {r.get('wrong_to_correct_vs_baseline','')} | {r.get('pseudo_wrong_top1_rows','')} |"
        )
    md.append("")
    md.append("## Drift candidates")
    md.append("")
    if drift_candidates:
        for c in drift_candidates[:10]:
            md.append(
                f"- epoch {c['epoch']} vs {c['prev_epoch']}: metric_worse={c['metric_worse']}, "
                f"correct_to_wrong_increased={c['correct_to_wrong_increased']}, loss_decreasing={c['loss_decreasing']}"
            )
    else:
        md.append("- No drift candidate detected by the conservative rule in the available saved epochs.")
    md.append("")
    md.append("## Notes")
    md.append("- This audit is read-only and uses true-margin exporter outputs for each checkpoint.")
    md.append("- `correct_to_wrong_vs_baseline` uses the baseline epoch as reference.")
    md.append("- `pseudo_wrong_top1_rows` is a proxy for pseudo-label overfit: top1 equals the Hungarian pseudo label while pseudo label differs from GT.")
    (out_dir / "A8_CE_DRIFT_ONSET_AUDIT.md").write_text("\n".join(md), encoding="utf-8")

    summary = {
        "status": "PASS",
        "run_root": str(args.run_root),
        "dataset_name": args.dataset_name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "epochs": sorted(ckpts),
        "baseline_epoch": args.baseline_epoch,
        "hub_raw_ids": hub_ids,
        "export_status": export_status,
        "drift_candidates": drift_candidates,
        "outputs": {
            "epoch_level_drift_table": str(out_dir / "epoch_level_drift_table.csv"),
            "row_transition_events": str(out_dir / "row_transition_events.csv"),
            "hub_onset_table": str(out_dir / "hub_onset_table.csv"),
            "markdown": str(out_dir / "A8_CE_DRIFT_ONSET_AUDIT.md"),
        },
    }
    _write_json(out_dir / "ce_drift_onset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("WROTE", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
