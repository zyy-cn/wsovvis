#!/usr/bin/env python3
"""
Read-only A8 score-margin + convergence audit.

This script is intentionally conservative:
- It does not modify training, matching, inference, checkpoints, or eval outputs.
- It only reads existing CSV / JSON / logs and writes new analysis files.
- True score-margin is computed only when row-level score columns are present.
  If those columns are absent, it writes rank/confusion/support aggregates and
  marks true score-margin as unavailable instead of inventing scores.

Typical usage:
python tools/a8_score_margin_convergence_audit.py \
  --run_root /mnt/sda/zyy/code/wsovvis/codex/outputs/G8_inference_and_eval/sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427 \
  --dataset_name lvvis_train_base \
  --base_out /mnt/sda/zyy/code/wsovvis/codex/outputs/G8_inference_and_eval/sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427/outputs/a8_hungarian_prealign_ablation/lvvis_train_base/baseline_full_y_5ep_base_ce_50ep
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        if not fieldnames:
            fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_value(r.get(k, "")) for k in fieldnames})


def _csv_value(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fnum(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        if x == "":
            return default
        return float(x)
    except Exception:
        return default


def fnum_or_none(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def inum(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    try:
        if x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def first_present(row: Dict[str, str], names: Iterable[str], default: str = "") -> str:
    for n in names:
        v = row.get(n)
        if v is not None and v != "":
            return v
    return default


def first_float_present(row: Dict[str, str], names: Iterable[str]) -> Optional[float]:
    for n in names:
        if n in row:
            v = fnum_or_none(row.get(n))
            if v is not None:
                return v
    return None


def support_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 2:
        return "2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    return ">20"


def quality_bucket(top1: float) -> str:
    if top1 >= 0.8:
        return "solved_high"
    if top1 >= 0.5:
        return "partly_solved"
    if top1 > 0:
        return "weakly_solved"
    return "zero_top1"


def pct(vals: List[float], threshold: float) -> Optional[float]:
    if not vals:
        return None
    return sum(1 for v in vals if v < threshold) / len(vals)


def safe_mean(vals: List[float]) -> Optional[float]:
    return mean(vals) if vals else None


def safe_median(vals: List[float]) -> Optional[float]:
    return median(vals) if vals else None


def detect_score_columns(rows: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
    if not rows:
        return {"gt_score": None, "top1_score": None, "top2_score": None}

    fields = set(rows[0].keys())

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in fields:
                # Require at least one non-empty numeric value.
                for r in rows[: min(len(rows), 200)]:
                    if fnum_or_none(r.get(c)) is not None:
                        return c
        return None

    return {
        "gt_score": pick([
            "gt_score", "score_gt", "after_gt_score", "gt_logit", "target_score",
            "score_for_gt", "gt_class_score"
        ]),
        "top1_score": pick([
            "top1_score", "score_top1", "after_top1_score", "top1_logit",
            "pred_score", "top1_class_score"
        ]),
        "top2_score": pick([
            "top2_score", "score_top2", "after_top2_score", "top2_logit",
            "second_score"
        ]),
    }


def infer_paths(args: argparse.Namespace) -> Dict[str, Path]:
    run_root = Path(args.run_root).resolve()
    dataset = args.dataset_name
    if args.base_out:
        base_out = Path(args.base_out).resolve()
    else:
        base_out = run_root / "outputs/a8_hungarian_prealign_ablation" / dataset / "baseline_full_y_5ep_base_ce_50ep"

    base_analysis = Path(args.base_analysis).resolve() if args.base_analysis else base_out / "analysis"

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = run_root / "analysis/a8_baseline_full_y_5ep_margin_convergence_audit" / dataset

    return {
        "run_root": run_root,
        "base_out": base_out,
        "base_analysis": base_analysis,
        "output_dir": out_dir,
        "row_csv": Path(args.row_csv).resolve() if args.row_csv else base_analysis / "eval_after_row_predictions.csv",
        "by_class_csv": Path(args.by_class_csv).resolve() if args.by_class_csv else base_analysis / "eval_after_by_class.csv",
        "final_summary": Path(args.final_summary).resolve() if args.final_summary else base_out / "final_summary.json",
    }


def build_class_info(class_rows: List[Dict[str, str]], row_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    names: Dict[str, str] = {}
    for r in row_rows:
        gt = str(first_present(r, ["gt_raw_id", "raw_id", "target_raw_id"])).strip()
        if gt and first_present(r, ["gt_class_name", "class_name", "target_class_name"]):
            names[gt] = first_present(r, ["gt_class_name", "class_name", "target_class_name"])

    info: Dict[str, Dict[str, Any]] = {}
    for c in class_rows:
        rid = str(first_present(c, ["raw_id", "gt_raw_id", "target_raw_id"])).strip()
        if not rid:
            continue
        gt_count = inum(first_present(c, ["gt_count", "count", "rows", "row_count"]))
        top1 = fnum(first_present(c, ["gt_top1_hit_rate", "top1_rate", "top1_hit_rate"]))
        top5 = fnum(first_present(c, ["gt_top5_hit_rate", "top5_rate", "top5_hit_rate"]))
        mean_rank = fnum(first_present(c, ["mean_rank", "gt_mean_rank"]))
        mean_norm = fnum(first_present(c, ["mean_normalized_gt_rank", "mean_normalized_rank"]))
        info[rid] = {
            "raw_id": rid,
            "class_name": first_present(c, ["class_name", "gt_class_name"], names.get(rid, "")),
            "gt_count": gt_count,
            "top1_rate": top1,
            "top5_rate": top5,
            "mean_rank": mean_rank,
            "mean_normalized_rank": mean_norm,
        }
    return info


def build_row_audit(
    row_rows: List[Dict[str, str]],
    class_info: Dict[str, Dict[str, Any]],
    score_cols: Dict[str, Optional[str]],
) -> Tuple[List[Dict[str, Any]], bool]:
    has_true_scores = score_cols.get("gt_score") is not None and score_cols.get("top1_score") is not None
    audited: List[Dict[str, Any]] = []

    for idx, r in enumerate(row_rows):
        gt = str(first_present(r, ["gt_raw_id", "target_raw_id", "raw_id"])).strip()
        top1 = str(first_present(r, ["top1_raw_id", "pred_raw_id", "top1_class_raw_id"])).strip()
        top1_hit = inum(first_present(r, ["top1_hit", "gt_top1_hit", "is_top1_gt"]))
        top5_hit = inum(first_present(r, ["top5_hit", "gt_top5_hit", "is_top5_gt"]))
        gt_rank = inum(first_present(r, ["gt_rank", "rank_of_gt", "rank"]), default=0)

        gt_score = first_float_present(r, [score_cols["gt_score"]]) if score_cols.get("gt_score") else None
        top1_score = first_float_present(r, [score_cols["top1_score"]]) if score_cols.get("top1_score") else None
        top2_score = first_float_present(r, [score_cols["top2_score"]]) if score_cols.get("top2_score") else None

        score_margin = None
        margin_abs_gap = None
        if gt_score is not None and top1_score is not None:
            score_margin = gt_score - top1_score
            margin_abs_gap = max(0.0, top1_score - gt_score) if top1_hit == 0 else 0.0

        info = class_info.get(gt, {})
        audited.append({
            "row_index": idx,
            "video_id": first_present(r, ["video_id", "video", "ytid"]),
            "clip_id": first_present(r, ["clip_id", "clip", "video_clip_key", "row_clip_key"]),
            "trajectory_id": first_present(r, ["trajectory_id", "traj_id", "track_id", "instance_id"]),
            "row_key": first_present(r, ["row_key", "key", "sample_key"]),
            "gt_raw_id": gt,
            "gt_class_name": first_present(r, ["gt_class_name", "class_name", "target_class_name"], info.get("class_name", "")),
            "top1_raw_id": top1,
            "top1_class_name": first_present(r, ["top1_class_name", "pred_class_name", "top1_name"]),
            "gt_rank": gt_rank,
            "top1_hit": top1_hit,
            "top5_hit": top5_hit,
            "gt_score": gt_score,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "score_margin": score_margin,
            "margin_abs_gap": margin_abs_gap,
            "rank_gap_proxy": max(gt_rank - 1, 0) if gt_rank else "",
            "candidate_count": first_present(r, ["candidate_count", "num_classes", "vocab_count"]),
            "support_bucket": support_bucket(int(info.get("gt_count", 0))),
            "gt_count": info.get("gt_count", ""),
            "class_top1_rate": info.get("top1_rate", ""),
            "class_top5_rate": info.get("top5_rate", ""),
        })

    return audited, has_true_scores


def aggregate_by_class(row_audit: List[Dict[str, Any]], class_info: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows_by_gt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in row_audit:
        if r["gt_raw_id"]:
            rows_by_gt[str(r["gt_raw_id"])].append(r)

    out: List[Dict[str, Any]] = []
    for gt, info in class_info.items():
        rows = rows_by_gt.get(gt, [])
        wrong = [r for r in rows if inum(r.get("top1_hit")) == 0]
        rank2_wrong = [r for r in wrong if inum(r.get("gt_rank")) == 2]

        gaps = [r["margin_abs_gap"] for r in wrong if isinstance(r.get("margin_abs_gap"), (int, float))]
        rank2_gaps = [r["margin_abs_gap"] for r in rank2_wrong if isinstance(r.get("margin_abs_gap"), (int, float))]
        margins = [r["score_margin"] for r in rows if isinstance(r.get("score_margin"), (int, float))]
        wrong_counter = Counter(str(r.get("top1_raw_id", "")) for r in wrong)

        dominant_id, dominant_n = ("", 0)
        if wrong_counter:
            dominant_id, dominant_n = wrong_counter.most_common(1)[0]

        out.append({
            "gt_raw_id": gt,
            "gt_class_name": info.get("class_name", ""),
            "gt_count": info.get("gt_count", len(rows)),
            "support_bucket": support_bucket(int(info.get("gt_count", len(rows)) or 0)),
            "quality_bucket": quality_bucket(float(info.get("top1_rate", 0.0) or 0.0)),
            "top1_rate": info.get("top1_rate", ""),
            "top5_rate": info.get("top5_rate", ""),
            "mean_rank": info.get("mean_rank", ""),
            "mean_normalized_rank": info.get("mean_normalized_rank", ""),
            "mean_score_margin": safe_mean(margins),
            "median_score_margin": safe_median(margins),
            "wrong_count": len(wrong),
            "mean_wrong_abs_gap": safe_mean(gaps),
            "median_wrong_abs_gap": safe_median(gaps),
            "rank2_wrong_count": len(rank2_wrong),
            "rank2_wrong_mean_abs_gap": safe_mean(rank2_gaps),
            "rank2_wrong_median_abs_gap": safe_median(rank2_gaps),
            "pct_wrong_gap_lt_0p01": pct(gaps, 0.01),
            "pct_wrong_gap_lt_0p03": pct(gaps, 0.03),
            "pct_wrong_gap_lt_0p05": pct(gaps, 0.05),
            "pct_wrong_gap_lt_0p10": pct(gaps, 0.10),
            "dominant_wrong_top1_raw_id": dominant_id,
            "dominant_wrong_top1_count": dominant_n,
        })

    return sorted(out, key=lambda r: (fnum(r.get("top1_rate")), -inum(r.get("gt_count"))))


def aggregate_by_edge(row_audit: List[Dict[str, Any]], class_info: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in row_audit:
        if inum(r.get("top1_hit")) == 0:
            gt = str(r.get("gt_raw_id", ""))
            pred = str(r.get("top1_raw_id", ""))
            groups[(gt, pred)].append(r)

    out: List[Dict[str, Any]] = []
    for (gt, pred), rows in groups.items():
        gaps = [r["margin_abs_gap"] for r in rows if isinstance(r.get("margin_abs_gap"), (int, float))]
        rank_hist = Counter(str(inum(r.get("gt_rank"), 0)) for r in rows)
        info = class_info.get(gt, {})
        out.append({
            "gt_raw_id": gt,
            "gt_class_name": info.get("class_name", ""),
            "wrong_top1_raw_id": pred,
            "wrong_count": len(rows),
            "mean_abs_gap": safe_mean(gaps),
            "median_abs_gap": safe_median(gaps),
            "pct_gap_lt_0p01": pct(gaps, 0.01),
            "pct_gap_lt_0p03": pct(gaps, 0.03),
            "pct_gap_lt_0p05": pct(gaps, 0.05),
            "pct_gap_lt_0p10": pct(gaps, 0.10),
            "gt_rank_hist": dict(rank_hist),
            "gt_top1_rate": info.get("top1_rate", ""),
            "gt_top5_rate": info.get("top5_rate", ""),
            "gt_count": info.get("gt_count", ""),
        })
    return sorted(out, key=lambda r: (-inum(r.get("wrong_count")), fnum(r.get("gt_top1_rate"))))


def aggregate_absorbers(edge_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_pred: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in edge_rows:
        by_pred[str(e["wrong_top1_raw_id"])].append(e)

    out: List[Dict[str, Any]] = []
    for pred, edges in by_pred.items():
        wrong_rows = sum(inum(e["wrong_count"]) for e in edges)
        gaps: List[float] = []
        for e in edges:
            if e.get("mean_abs_gap") is not None:
                # Weighted approximate expansion from edge mean.
                gaps.extend([float(e["mean_abs_gap"])] * inum(e["wrong_count"]))
        top_sources = sorted(edges, key=lambda e: -inum(e["wrong_count"]))[:20]
        top_sources_by_gap = sorted(
            [e for e in edges if e.get("median_abs_gap") is not None],
            key=lambda e: -fnum(e.get("median_abs_gap")),
        )[:20]

        out.append({
            "wrong_top1_raw_id": pred,
            "absorbed_wrong_rows": wrong_rows,
            "absorbed_source_class_count": len(edges),
            "mean_abs_gap": safe_mean(gaps),
            "median_abs_gap": safe_median(gaps),
            "pct_gap_lt_0p05": pct(gaps, 0.05),
            "top_source_classes_by_wrong_count": [
                {
                    "gt_raw_id": e["gt_raw_id"],
                    "gt_class_name": e.get("gt_class_name", ""),
                    "wrong_count": e["wrong_count"],
                    "median_abs_gap": e.get("median_abs_gap"),
                    "gt_top1_rate": e.get("gt_top1_rate"),
                    "gt_top5_rate": e.get("gt_top5_rate"),
                }
                for e in top_sources
            ],
            "top_source_classes_by_median_gap": [
                {
                    "gt_raw_id": e["gt_raw_id"],
                    "gt_class_name": e.get("gt_class_name", ""),
                    "wrong_count": e["wrong_count"],
                    "median_abs_gap": e.get("median_abs_gap"),
                    "gt_top1_rate": e.get("gt_top1_rate"),
                    "gt_top5_rate": e.get("gt_top5_rate"),
                }
                for e in top_sources_by_gap
            ],
        })
    return sorted(out, key=lambda r: (-inum(r.get("absorbed_wrong_rows")), -inum(r.get("absorbed_source_class_count"))))


LOSS_PATTERNS = [
    re.compile(r"(?:epoch|ep)[^\d]*(\d+).*?(?:loss|mean_loss|train_loss)[=: ]+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.I),
    re.compile(r"(?:loss|mean_loss|train_loss)[=: ]+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?).*?(?:epoch|ep)[^\d]*(\d+)", re.I),
]


def scan_training_logs(base_out: Path, run_root: Path, max_bytes_per_log: int = 4_000_000) -> Dict[str, Any]:
    candidates: List[Path] = []
    for root in [base_out, run_root / "logs"]:
        if root.exists():
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                name = p.name.lower()
                if name.endswith((".log", ".txt", ".out")) and any(s in str(p).lower() for s in [
                    "bfy", "baseline_full_y", "hungarian", "pre5", "5ep", "base50", "ce_50"
                ]):
                    candidates.append(p)

    # Deduplicate and keep small/medium logs.
    seen = set()
    logs = []
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            logs.append(p)

    epoch_losses: Dict[int, List[float]] = defaultdict(list)
    log_summaries = []
    for p in logs:
        size = p.stat().st_size
        parsed = 0
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                read_bytes = 0
                for line in f:
                    read_bytes += len(line.encode("utf-8", errors="ignore"))
                    if read_bytes > max_bytes_per_log:
                        break
                    for pat in LOSS_PATTERNS:
                        m = pat.search(line)
                        if not m:
                            continue
                        if "loss" in pat.pattern[:30].lower():
                            # pattern 2: loss then epoch
                            loss = fnum(m.group(1), None)
                            ep = inum(m.group(2), None)
                        else:
                            ep = inum(m.group(1), None)
                            loss = fnum(m.group(2), None)
                        if ep is not None and loss is not None:
                            epoch_losses[ep].append(loss)
                            parsed += 1
                        break
        except Exception as exc:
            log_summaries.append({"path": str(p), "size": size, "error": repr(exc), "parsed_loss_rows": parsed})
            continue
        log_summaries.append({"path": str(p), "size": size, "parsed_loss_rows": parsed})

    series = []
    for ep in sorted(epoch_losses):
        vals = epoch_losses[ep]
        series.append({"epoch": ep, "loss_mean": mean(vals), "loss_min": min(vals), "n": len(vals)})

    return {"available_training_logs": log_summaries, "epoch_loss_series": series}


def convergence_status(base_out: Path, run_root: Path, final_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    checkpoints = []
    if base_out.exists():
        for p in base_out.rglob("*.pth"):
            try:
                checkpoints.append({"path": str(p), "size": p.stat().st_size})
            except OSError:
                checkpoints.append({"path": str(p)})

    log_scan = scan_training_logs(base_out, run_root)
    series = log_scan["epoch_loss_series"]

    summary: Dict[str, Any] = {
        "available_training_logs": log_scan["available_training_logs"],
        "available_epoch_metric_files": [],
        "available_checkpoints": checkpoints,
        "checkpoint_epochs_found": [],
        "train_loss_first": None,
        "train_loss_last": None,
        "train_loss_min": None,
        "train_loss_trend_last_10_epochs": None,
        "eval_metric_by_epoch": None,
        "best_epoch_by_micro_top1": None,
        "best_epoch_by_macro_rank1": None,
        "best_epoch_by_mean_normalized_rank": None,
        "final_epoch": None,
        "final_vs_best_delta": None,
        "convergence_status": "insufficient_logs",
        "evidence": [],
    }

    # Collect epoch metric-like files without reading fully.
    if base_out.exists():
        for p in base_out.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".csv", ".json", ".jsonl"]:
                low = p.name.lower()
                if any(k in low for k in ["epoch", "loss", "metric", "history", "train_summary"]):
                    summary["available_epoch_metric_files"].append(str(p))

    # Checkpoint epoch numbers.
    for c in checkpoints:
        m = re.search(r"(?:epoch|ep)[_\-]?(\d+)", Path(c["path"]).name, re.I)
        if m:
            summary["checkpoint_epochs_found"].append(int(m.group(1)))

    if final_summary:
        train_summary = final_summary.get("train_summary", {}) if isinstance(final_summary, dict) else {}
        final_epoch = train_summary.get("epochs") or final_summary.get("epochs")
        if final_epoch is not None:
            summary["final_epoch"] = final_epoch
        if final_summary.get("loss") is not None:
            summary["final_summary_loss"] = final_summary.get("loss")

    if len(series) >= 2:
        losses = [float(x["loss_mean"]) for x in series]
        epochs = [int(x["epoch"]) for x in series]
        summary["train_loss_first"] = losses[0]
        summary["train_loss_last"] = losses[-1]
        summary["train_loss_min"] = min(losses)
        summary["final_epoch"] = summary.get("final_epoch") or epochs[-1]

        tail = losses[-10:]
        if len(tail) >= 2:
            delta = tail[-1] - tail[0]
            rel = delta / max(abs(tail[0]), 1e-12)
            summary["train_loss_trend_last_10_epochs"] = {"delta": delta, "relative_delta": rel, "n": len(tail)}

            min_loss = min(losses)
            last = losses[-1]
            if rel < -0.05:
                status = "likely_undertrained"
                evidence = "Training loss still decreases by more than 5% over the last observed window."
            elif last > min_loss * 1.05:
                status = "likely_overtrained"
                evidence = "Final observed loss is more than 5% worse than the minimum observed loss."
            else:
                status = "likely_plateaued"
                evidence = "Last-window loss change is small or final loss remains close to the observed minimum."
            summary["convergence_status"] = status
            summary["evidence"].append(evidence)
        else:
            summary["convergence_status"] = "insufficient_logs"
            summary["evidence"].append("Only limited epoch-loss points were parsed.")
    else:
        summary["evidence"].append(
            "No parseable epoch-loss series found. Existing artifacts may only contain final metrics."
        )

    if not summary["checkpoint_epochs_found"]:
        summary["evidence"].append("No epoch-numbered checkpoints found; final-vs-best epoch cannot be proven.")

    if not summary["available_epoch_metric_files"]:
        summary["evidence"].append("No explicit epoch metric/history files found by filename scan.")

    return summary


def make_takeover(
    out_dir: Path,
    row_count: int,
    class_count: int,
    has_true_scores: bool,
    score_cols: Dict[str, Optional[str]],
    by_class: List[Dict[str, Any]],
    by_edge: List[Dict[str, Any]],
    absorbers: List[Dict[str, Any]],
    conv: Dict[str, Any],
) -> str:
    rank2_edges = [e for e in by_edge if "2" in (e.get("gt_rank_hist") or {})]
    total_wrong = sum(inum(e.get("wrong_count")) for e in by_edge)
    rank2_wrong = 0
    for e in by_edge:
        hist = e.get("gt_rank_hist") or {}
        rank2_wrong += inum(hist.get("2", 0))

    lines = []
    lines.append("# A8 score-margin + convergence audit")
    lines.append("")
    lines.append(f"- row_count: {row_count}")
    lines.append(f"- class_count: {class_count}")
    lines.append(f"- true_score_margin_available: {str(has_true_scores)}")
    lines.append(f"- detected_score_columns: {json.dumps(score_cols, ensure_ascii=False)}")
    lines.append(f"- total_wrong_rows_from_edges: {total_wrong}")
    lines.append(f"- rank2_wrong_rows_from_edges: {rank2_wrong}")
    if total_wrong:
        lines.append(f"- rank2_wrong_rate_among_wrong: {rank2_wrong / total_wrong:.6f}")
    lines.append(f"- convergence_status: {conv.get('convergence_status')}")
    lines.append("")
    if not has_true_scores:
        lines.append("## True score-margin blocker")
        lines.append("")
        lines.append(
            "Current row prediction CSV does not contain `gt_score/top1_score`; this script therefore did not invent numeric margins. "
            "It exported rank/confusion/support aggregates and marked true score margin as unavailable."
        )
        lines.append("")
    lines.append("## Top absorbers")
    for r in absorbers[:10]:
        lines.append(
            f"- wrong_top1={r.get('wrong_top1_raw_id')}: "
            f"wrong_rows={r.get('absorbed_wrong_rows')}, "
            f"source_classes={r.get('absorbed_source_class_count')}, "
            f"median_abs_gap={r.get('median_abs_gap')}"
        )
    lines.append("")
    lines.append("## Top confusion edges")
    for e in by_edge[:10]:
        lines.append(
            f"- {e.get('gt_class_name')}({e.get('gt_raw_id')}) -> {e.get('wrong_top1_raw_id')}: "
            f"wrong_count={e.get('wrong_count')}, "
            f"gt_rank_hist={json.dumps(e.get('gt_rank_hist'), ensure_ascii=False)}, "
            f"median_abs_gap={e.get('median_abs_gap')}"
        )
    lines.append("")
    lines.append("## Top unresolved classes")
    for r in by_class[:10]:
        lines.append(
            f"- {r.get('gt_class_name')}({r.get('gt_raw_id')}): "
            f"gt_count={r.get('gt_count')}, top1={r.get('top1_rate')}, top5={r.get('top5_rate')}, "
            f"wrong_count={r.get('wrong_count')}, median_wrong_abs_gap={r.get('median_wrong_abs_gap')}, "
            f"dominant_wrong={r.get('dominant_wrong_top1_raw_id')}({r.get('dominant_wrong_top1_count')})"
        )
    lines.append("")
    lines.append("## Convergence evidence")
    for ev in conv.get("evidence", []):
        lines.append(f"- {ev}")
    lines.append("")
    lines.append("## Output files")
    for name in [
        "score_margin_row_audit.csv",
        "score_margin_by_class.csv",
        "score_margin_by_confusion_edge.csv",
        "absorber_margin_summary.csv",
        "convergence_audit_summary.json",
        "margin_convergence_audit_summary.json",
    ]:
        lines.append(f"- {out_dir / name}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--base_out", default="")
    ap.add_argument("--base_analysis", default="")
    ap.add_argument("--row_csv", default="")
    ap.add_argument("--by_class_csv", default="")
    ap.add_argument("--final_summary", default="")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--strict_true_margin", action="store_true", help="Exit non-zero if row CSV lacks score columns.")
    args = ap.parse_args()

    paths = infer_paths(args)
    out_dir = paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    row_csv = paths["row_csv"]
    by_class_csv = paths["by_class_csv"]
    if not row_csv.exists():
        raise FileNotFoundError(f"Missing row predictions CSV: {row_csv}")
    if not by_class_csv.exists():
        raise FileNotFoundError(f"Missing by-class CSV: {by_class_csv}")

    row_rows = read_csv_rows(row_csv)
    class_rows = read_csv_rows(by_class_csv)

    score_cols = detect_score_columns(row_rows)
    class_info = build_class_info(class_rows, row_rows)
    row_audit, has_true_scores = build_row_audit(row_rows, class_info, score_cols)

    if args.strict_true_margin and not has_true_scores:
        # Still write a minimal blocker summary for usability.
        blocker = {
            "status": "BLOCKED_TRUE_MARGIN_COLUMNS_MISSING",
            "row_csv": str(row_csv),
            "detected_score_columns": score_cols,
            "required": ["gt_score", "top1_score"],
            "message": "Row CSV lacks true score columns. Re-run/evaluate with score export enabled or patch scorer to export scores.",
        }
        (out_dir / "margin_convergence_audit_summary.json").write_text(
            json.dumps(blocker, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SystemExit(2)

    by_class = aggregate_by_class(row_audit, class_info)
    by_edge = aggregate_by_edge(row_audit, class_info)
    absorbers = aggregate_absorbers(by_edge)

    write_csv(out_dir / "score_margin_row_audit.csv", row_audit)
    write_csv(out_dir / "score_margin_by_class.csv", by_class)
    write_csv(out_dir / "score_margin_by_confusion_edge.csv", by_edge)
    write_csv(out_dir / "absorber_margin_summary.csv", absorbers)

    final_summary = load_json(paths["final_summary"])
    conv = convergence_status(paths["base_out"], paths["run_root"], final_summary)
    (out_dir / "convergence_audit_summary.json").write_text(
        json.dumps(conv, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_wrong = sum(inum(e.get("wrong_count")) for e in by_edge)
    rank2_wrong = sum(inum((e.get("gt_rank_hist") or {}).get("2", 0)) for e in by_edge)
    summary = {
        "status": "PASS_WITH_TRUE_MARGIN" if has_true_scores else "PASS_RANK_ONLY_TRUE_MARGIN_UNAVAILABLE",
        "inputs": {
            "row_csv": str(row_csv),
            "by_class_csv": str(by_class_csv),
            "base_out": str(paths["base_out"]),
            "final_summary": str(paths["final_summary"]),
        },
        "outputs": {
            "output_dir": str(out_dir),
            "row_audit": str(out_dir / "score_margin_row_audit.csv"),
            "by_class": str(out_dir / "score_margin_by_class.csv"),
            "by_confusion_edge": str(out_dir / "score_margin_by_confusion_edge.csv"),
            "absorber_summary": str(out_dir / "absorber_margin_summary.csv"),
            "convergence": str(out_dir / "convergence_audit_summary.json"),
        },
        "score_margin": {
            "true_score_margin_available": has_true_scores,
            "detected_score_columns": score_cols,
            "note": None if has_true_scores else (
                "Existing row predictions do not contain gt_score/top1_score. "
                "Numeric score margins require a score-export patch or a re-eval that writes row-level scores."
            ),
        },
        "rank_confusion": {
            "row_count": len(row_rows),
            "class_count": len(class_info),
            "confusion_edge_count": len(by_edge),
            "absorber_count": len(absorbers),
            "total_wrong_rows_from_edges": total_wrong,
            "rank2_wrong_rows_from_edges": rank2_wrong,
            "rank2_wrong_rate_among_wrong": rank2_wrong / total_wrong if total_wrong else None,
            "top_absorbers": absorbers[:20],
            "top_confusion_edges": by_edge[:20],
            "top_unresolved_classes": by_class[:20],
        },
        "convergence": conv,
        "next_command_if_true_margin_needed": (
            "Patch the scorer/eval row exporter to include gt_score/top1_score/top2_score in "
            "eval_after_row_predictions.csv, then rerun this audit with --strict_true_margin."
        ) if not has_true_scores else None,
    }
    (out_dir / "margin_convergence_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    takeover = make_takeover(
        out_dir=out_dir,
        row_count=len(row_rows),
        class_count=len(class_info),
        has_true_scores=has_true_scores,
        score_cols=score_cols,
        by_class=by_class,
        by_edge=by_edge,
        absorbers=absorbers,
        conv=conv,
    )
    (out_dir / "A8_MARGIN_CONVERGENCE_TAKEOVER.md").write_text(takeover, encoding="utf-8")

    print(takeover)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
