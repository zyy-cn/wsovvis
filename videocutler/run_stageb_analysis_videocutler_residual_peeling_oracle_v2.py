#!/usr/bin/env python3
"""Read-only VideoCutLER residual-peeling oracle v2.

This audit reuses the exact prealign projector/checkpoint text-scoring backend
that produced the D_full row scores, but applies it to VideoCutLER carrier rows
grouped by clip and evaluated under iterative-residual candidate policies.

It is read-only: no training, inference, checkpoint writes, or label edits.
Large carrier/trajectory inputs are streamed and the outputs are compact.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab
from videocutler.ext_stageb_ovvis.audit._matrix_vocab_scoring import build_carrier_matrix_pack
from videocutler.ext_stageb_ovvis.eval.g8_bridge import load_projector_bundle
from videocutler.run_stageb_analysis_gtcarrier_latent_rowdump import _load_projector, _project_text_matrix
from videocutler.run_stageb_analysis_videocutler_residual_peeling_oracle import (
    _as_float,
    _as_int,
    _as_str_id,
    _extract_clip_key,
    _extract_gt_id,
    _json_loads_maybe,
    _rate,
    _truth,
    known_before_iteration,
    load_annotation_contexts,
    load_iterative_labels,
    load_split,
)

Record = Dict[str, Any]

DEFAULT_BACKEND_RUN_ROOT = Path(
    "/mnt/sda/zyy/code/wsovvis/codex/outputs/G8_inference_and_eval/"
    "support_null_v2b_relmargin_cap035_k32_b010_resp_rerun_20260428"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/sda/zyy/code/wsovvis/codex/outputs/G8_inference_and_eval/"
    "oracle_clean_data_ablation_20260429"
)
DEFAULT_DATASET = "lvvis_train_base"
DEFAULT_ANN_JSON = Path("/home/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations/train_instances.json")
DEFAULT_SPLIT_JSON = Path("package/reference/lvvis_official_base_novel_split.json")
DEFAULT_CARRIER_ROOT = Path("/home/zyy/code/wsovvis_asserts")
DEFAULT_CARRIER_JSONL = DEFAULT_CARRIER_ROOT / "carrier_bank" / DEFAULT_DATASET / "carrier_records.jsonl"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _iter_jsonl(path: Path) -> Iterable[Record]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _load_csv_rows(path: Path) -> List[Record]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _median(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(statistics.median(clean)) if clean else None


def _mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else None


def _normalize_np(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= eps or not math.isfinite(norm):
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32, copy=False)


def _safe_rank(scores: np.ndarray, gt_index: int) -> Tuple[int, int, float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    gt_score = float(scores[int(gt_index)])
    rank = int(np.sum(scores > gt_score)) + 1
    top_idx = int(np.argmax(scores))
    if len(scores) <= 1:
        margin = float("inf")
    else:
        if top_idx != gt_index:
            best_non = float(scores[top_idx])
        else:
            tmp = scores.copy()
            tmp[int(gt_index)] = -np.inf
            best_non = float(np.max(tmp))
        margin = float(gt_score - best_non)
    return rank, top_idx, gt_score, margin


def _topk_counts(counter: Counter[int], *, topn: int = 20) -> List[Dict[str, Any]]:
    return [{"raw_category_id": int(raw_id), "count": int(count)} for raw_id, count in counter.most_common(topn)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only VideoCutLER residual-peeling oracle v2")
    p.add_argument("--run_root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--run_root_v2b", default=str(DEFAULT_BACKEND_RUN_ROOT))
    p.add_argument("--dataset_name", default=DEFAULT_DATASET)
    p.add_argument("--variant", default="person_aware")
    p.add_argument("--annotation_json", default=str(DEFAULT_ANN_JSON))
    p.add_argument("--split_json", default=str(DEFAULT_SPLIT_JSON))
    p.add_argument("--per_class_csv", default="")
    p.add_argument("--trajectory_precision_rows_csv", default="")
    p.add_argument("--carrier_records_jsonl", default=str(DEFAULT_CARRIER_JSONL))
    p.add_argument("--carrier_asset_root", default=str(DEFAULT_CARRIER_ROOT))
    p.add_argument("--candidate_policies", default="fullY,base_residual,all_visible_residual,fullY_minus_known")
    p.add_argument("--person_raw_id", default="773")
    p.add_argument("--min_iou", type=float, default=0.5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score_chunk_size", type=int, default=4096)
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--top_examples", type=int, default=100)
    p.add_argument("--max_rows", type=int, default=0)
    return p.parse_args()


def _load_trajectory_precision_rows(path: Path, *, min_iou: float) -> Tuple[Dict[str, List[Record]], Dict[str, int], Dict[str, Any]]:
    rows_by_clip: Dict[str, List[Record]] = defaultdict(list)
    stats = {
        "rows_seen": 0,
        "rows_kept": 0,
        "skipped_no_tid": 0,
        "skipped_no_gt": 0,
        "skipped_no_clip": 0,
        "skipped_low_iou": 0,
    }
    tids: Dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stats["rows_seen"] += 1
            tid = _as_str_id(row.get("trajectory_id", row.get("traj_id", row.get("carrier_id"))))
            if tid is None:
                stats["skipped_no_tid"] += 1
                continue
            gt_raw = _as_int(row.get("best_gt_raw_id", row.get("matched_gt_raw_id", row.get("gt_raw_id", row.get("raw_category_id")))), default=-1)
            if gt_raw < 0:
                stats["skipped_no_gt"] += 1
                continue
            clip_key = _as_str_id(row.get("video_id", row.get("clip_id", row.get("clip_key"))))
            if clip_key is None:
                stats["skipped_no_clip"] += 1
                continue
            best_iou = _as_float(row.get("best_gt_iou", row.get("matched_gt_iou", row.get("gt_iou", 0.0))), 0.0)
            if best_iou < float(min_iou):
                stats["skipped_low_iou"] += 1
                continue
            row_dict: Record = dict(row)
            row_dict["trajectory_id"] = tid
            row_dict["clip_id"] = clip_key
            row_dict["best_gt_raw_id"] = int(gt_raw)
            row_dict["best_gt_iou"] = float(best_iou)
            rows_by_clip[str(clip_key)].append(row_dict)
            tids[tid] = 1
            stats["rows_kept"] += 1
    return rows_by_clip, tids, stats


def _load_relevant_carriers(carrier_jsonl: Path, relevant_tids: Set[str]) -> Tuple[Dict[str, Record], Dict[str, Any]]:
    out: Dict[str, Record] = {}
    stats = {"rows_seen": 0, "rows_kept": 0, "rows_missing_tid": 0}
    with carrier_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats["rows_seen"] += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            tid = _as_str_id(row.get("trajectory_id", row.get("carrier_id", row.get("track_id"))))
            if tid is None:
                stats["rows_missing_tid"] += 1
                continue
            if tid not in relevant_tids:
                continue
            if tid not in out:
                out[tid] = dict(row)
                stats["rows_kept"] += 1
    return out, stats


def _policy_source_set(
    policy: str,
    *,
    gt_raw_id: str,
    clip_key: str,
    known: Set[str],
    base_ctx: Mapping[str, Set[str]],
    all_ctx: Mapping[str, Set[str]],
    full_vocab: Set[str],
) -> Set[str]:
    if policy == "fullY":
        cand = set(full_vocab)
    elif policy == "base_residual":
        cand = set(base_ctx.get(clip_key, set())) - set(known)
    elif policy == "all_visible_residual":
        cand = set(all_ctx.get(clip_key, set())) - set(known)
    elif policy == "fullY_minus_known":
        cand = set(full_vocab) - set(known)
    else:
        raise ValueError(f"unsupported candidate policy: {policy}")
    return cand


def _score_policy_row(
    *,
    full_logits: np.ndarray,
    raw_to_idx: Mapping[str, int],
    policy_source: Set[str],
    gt_raw_id: str,
) -> Optional[Record]:
    if gt_raw_id not in raw_to_idx:
        return None
    candidate_source = set(policy_source)
    contains_gt = gt_raw_id in candidate_source
    candidate_eval = set(candidate_source)
    candidate_eval.add(gt_raw_id)
    candidate_indices = [int(raw_to_idx[x]) for x in candidate_eval if x in raw_to_idx]
    if not candidate_indices:
        return None
    gt_index = int(raw_to_idx[gt_raw_id])
    if gt_index not in candidate_indices:
        candidate_indices.append(gt_index)
    candidate_indices = list(dict.fromkeys(candidate_indices))
    local_gt_index = candidate_indices.index(gt_index)
    row_scores = np.asarray(full_logits[np.asarray(candidate_indices, dtype=np.int64)], dtype=np.float64)
    rank, top_local_idx, gt_score, margin = _safe_rank(row_scores, local_gt_index)
    top1_idx = int(candidate_indices[top_local_idx])
    top1_raw = next((rid for rid, idx in raw_to_idx.items() if int(idx) == top1_idx), str(top1_idx))
    top5 = rank <= 5
    top20 = rank <= 20
    norm_rank = float((rank - 1) / max(len(candidate_indices) - 1, 1))
    probs = np.exp(row_scores - float(np.max(row_scores)))
    probs = probs / max(float(np.sum(probs)), 1e-12)
    entropy = float(-np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))))
    masked = np.asarray(row_scores, dtype=np.float64).copy()
    masked[local_gt_index] = -np.inf
    best_non = float(np.max(masked)) if masked.size else float("-inf")
    return {
        "candidate_contains_gt": bool(contains_gt),
        "candidate_source_size": int(len(candidate_source)),
        "candidate_eval_size": int(len(candidate_indices)),
        "gt_rank": int(rank),
        "gt_top1": bool(rank == 1),
        "gt_top5": bool(top5),
        "gt_top20": bool(top20),
        "gt_normalized_gt_rank": float(norm_rank),
        "gt_margin_vs_best_non_gt": float(margin),
        "assignment_entropy": entropy,
        "top1_raw_id": str(top1_raw),
        "score_gt": float(gt_score),
        "score_best_non_gt": float(best_non) if math.isfinite(best_non) else None,
        "score_top1": float(row_scores[top_local_idx]),
    }


def _summarize_policy_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    evaluable = [r for r in rows if bool(r.get("candidate_contains_gt"))]
    wrong = [r for r in evaluable if not bool(r.get("gt_top1"))]
    top1_counter: Counter[int] = Counter()
    wrong_top1_counter: Counter[int] = Counter()
    confusion_pairs: Counter[Tuple[int, int]] = Counter()
    for row in rows:
        top1_raw = _as_int(row.get("top1_raw_id"), default=-1)
        if top1_raw >= 0:
            top1_counter[top1_raw] += 1
    for row in wrong:
        gt = _as_int(row.get("gt_raw_id"), default=-1)
        top1 = _as_int(row.get("top1_raw_id"), default=-1)
        if top1 >= 0:
            wrong_top1_counter[top1] += 1
        if gt >= 0 and top1 >= 0:
            confusion_pairs[(gt, top1)] += 1
    wrong_count = len(wrong)
    total_top1 = sum(top1_counter.values())
    return {
        "row_count_total": int(len(rows)),
        "row_count_with_label_record": int(len(rows)),
        "row_count_in_scope": int(len(evaluable)),
        "candidate_contains_gt_count": int(len(evaluable)),
        "candidate_contains_gt_rate": _rate(len(evaluable), len(rows)),
        "latent_evaluable_row_count": int(len(evaluable)),
        "fullY_rank1_rate": _rate(sum(1 for r in evaluable if bool(r.get("fullY_top1"))), len(evaluable)),
        "residual_rank1_rate": _rate(sum(1 for r in evaluable if bool(r.get("gt_top1"))), len(evaluable)),
        "residual_top5_rate": _rate(sum(1 for r in evaluable if bool(r.get("gt_top5"))), len(evaluable)),
        "residual_top20_rate": _rate(sum(1 for r in evaluable if bool(r.get("gt_top20"))), len(evaluable)),
        "mean_gt_rank": _mean([float(r.get("gt_rank")) for r in evaluable if r.get("gt_rank") is not None]),
        "median_gt_rank": _median([float(r.get("gt_rank")) for r in evaluable if r.get("gt_rank") is not None]),
        "mean_normalized_gt_rank": _mean([float(r.get("gt_normalized_gt_rank")) for r in evaluable if r.get("gt_normalized_gt_rank") is not None]),
        "mean_gt_margin_vs_best_non_gt": _mean([float(r.get("gt_margin_vs_best_non_gt")) for r in evaluable if r.get("gt_margin_vs_best_non_gt") is not None]),
        "positive_gt_margin_rate": _rate(sum(1 for r in evaluable if r.get("gt_margin_vs_best_non_gt") is not None and float(r.get("gt_margin_vs_best_non_gt")) > 0), len(evaluable)),
        "top1_wrong_rate": _rate(wrong_count, len(evaluable)),
        "wrong_large_negative_margin_rate": _rate(sum(1 for r in wrong if r.get("gt_margin_vs_best_non_gt") is not None and float(r.get("gt_margin_vs_best_non_gt")) < -0.20), wrong_count),
        "wrong_near_tie_margin_rate": _rate(sum(1 for r in wrong if r.get("gt_margin_vs_best_non_gt") is not None and -0.05 <= float(r.get("gt_margin_vs_best_non_gt")) < 0.0), wrong_count),
        "assignment_entropy_mean": _mean([float(r.get("assignment_entropy")) for r in rows if r.get("assignment_entropy") is not None]),
        "hub_like_top1_concentration": (float(max(top1_counter.values()) / total_top1) if total_top1 else None),
        "top1_label_top_counts": _topk_counts(top1_counter, topn=20),
        "wrong_top1_label_top_counts": _topk_counts(wrong_top1_counter, topn=20),
        "confusion_pairs_top": [
            {"gt_raw_id": int(gt), "wrong_top1_raw_id": int(top1), "count": int(count)}
            for (gt, top1), count in confusion_pairs.most_common(20)
        ],
    }


def _policy_summary_row(policy: str, summary: Mapping[str, Any]) -> Dict[str, Any]:
    row = {"policy": policy}
    row.update({k: summary.get(k) for k in [
        "row_count_total",
        "row_count_with_label_record",
        "row_count_in_scope",
        "candidate_contains_gt_count",
        "candidate_contains_gt_rate",
        "latent_evaluable_row_count",
        "fullY_rank1_rate",
        "residual_rank1_rate",
        "residual_top5_rate",
        "residual_top20_rate",
        "mean_gt_rank",
        "median_gt_rank",
        "mean_normalized_gt_rank",
        "mean_gt_margin_vs_best_non_gt",
        "positive_gt_margin_rate",
        "top1_wrong_rate",
        "wrong_large_negative_margin_rate",
        "wrong_near_tie_margin_rate",
        "assignment_entropy_mean",
        "hub_like_top1_concentration",
    ]})
    row["top1_label_top_counts"] = json.dumps(summary.get("top1_label_top_counts", []), ensure_ascii=False)
    row["wrong_top1_label_top_counts"] = json.dumps(summary.get("wrong_top1_label_top_counts", []), ensure_ascii=False)
    row["confusion_pairs_top"] = json.dumps(summary.get("confusion_pairs_top", []), ensure_ascii=False)
    row["comparison_to_E_in_scope"] = json.dumps({"E_in_scope_oracle_valid_rate": 1.0, "latent_gap_to_oracle": float(1.0 - float(summary.get("residual_rank1_rate") or 0.0))}, ensure_ascii=False)
    return row


def _write_takeover(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# VideoCutLER Residual Peeling Oracle v2",
        "",
        f"- status: `{summary.get('status')}`",
        f"- scorer_backend: `{summary.get('scorer_backend')}`",
        f"- run_root_v2b: `{summary.get('run_root_v2b')}`",
        f"- output_dir: `{summary.get('output_dir')}`",
        f"- filtered_rows: `{summary.get('row_count_total')}`",
        f"- scored_rows: `{summary.get('scored_rows_total')}`",
        f"- carrier_rows_loaded: `{summary.get('carrier_rows_loaded')}`",
        f"- text_vocab_rows: `{summary.get('text_vocab_rows')}`",
        f"- text_dim: `{summary.get('text_dim')}`",
        f"- carrier_dim: `{summary.get('carrier_dim')}`",
        "",
        "## Policy summary",
    ]
    for policy, row in summary.get("policy_summaries", {}).items():
        lines.append(
            f"- {policy}: fullY_rank1={row.get('fullY_rank1_rate')}, "
            f"residual_rank1={row.get('residual_rank1_rate')}, "
            f"candidate_contains_gt={row.get('candidate_contains_gt_rate')}, "
            f"gap_to_oracle={row.get('comparison_to_E_in_scope', {}).get('latent_gap_to_oracle') if isinstance(row.get('comparison_to_E_in_scope'), dict) else None}"
        )
    lines.append("")
    lines.append("## Backend trace")
    lines.append(f"- trace file: `{summary.get('backend_trace_path')}`")
    lines.append(f"- checkpoint: `{summary.get('checkpoint_path')}`")
    lines.append(f"- projected text dim: `{summary.get('projected_text_dim')}`")
    lines.append("")
    lines.append("This audit uses the same prealign text projector/checkpoint backend as `D_full_row_scores.jsonl`.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    output_root = Path(args.run_root).expanduser().resolve()
    backend_run_root = Path(args.run_root_v2b).expanduser().resolve()
    out_dir = output_root / "analysis" / "videocutler_residual_peeling_oracle_v2" / args.dataset_name / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    annotation_json = Path(args.annotation_json).expanduser().resolve()
    split_json = Path(args.split_json).expanduser().resolve()
    per_class_csv = Path(args.per_class_csv).expanduser().resolve() if args.per_class_csv else output_root / "analysis" / "iterative_residual_label_identifiability" / args.dataset_name / "per_class_iterative_residual_identifiability.csv"
    precision_csv = Path(args.trajectory_precision_rows_csv).expanduser().resolve() if args.trajectory_precision_rows_csv else output_root / "analysis" / "videocutler_multiplicity_precision" / args.dataset_name / "trajectory_precision_rows.csv"
    carrier_jsonl = Path(args.carrier_records_jsonl).expanduser().resolve()
    carrier_asset_root = Path(args.carrier_asset_root).expanduser().resolve()

    required = [annotation_json, split_json, per_class_csv, precision_csv, carrier_jsonl]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing required input(s): " + "; ".join(missing))

    base_ids, novel_ids, split_names = load_split(split_json)
    base_ctx, all_ctx, ann_names = load_annotation_contexts(annotation_json, base_ids, novel_ids)
    label_by_id, resolved_by_iter, initial_known, label_meta = load_iterative_labels(per_class_csv, args.variant, str(args.person_raw_id))
    text_vocab_ids, _text_records, text_vocab_matrix = load_text_vocab(backend_run_root)
    if not text_vocab_ids or text_vocab_matrix.ndim != 2:
        raise RuntimeError(f"no text vocab loaded from {backend_run_root}")

    checkpoint_path = backend_run_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing prealign checkpoint for scorer backend: {checkpoint_path}")
    device_str = str(args.device)
    device = torch.device(device_str if str(device_str).startswith("cuda") and torch.cuda.is_available() else "cpu")
    projector, temperature, ckpt = _load_projector(checkpoint_path, device=device)
    projected_text_matrix = _project_text_matrix(projector, np.asarray(text_vocab_matrix, dtype=np.float32), device=device, batch_size=2048)
    text_vocab_ids = [int(x) for x in text_vocab_ids]
    raw_to_idx = {str(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
    full_vocab: Set[str] = set(raw_to_idx.keys())

    row_groups, relevant_tids, precision_stats = _load_trajectory_precision_rows(precision_csv, min_iou=float(args.min_iou))
    carrier_by_tid, carrier_stats = _load_relevant_carriers(carrier_jsonl, set(relevant_tids.keys()))

    policies = [p.strip() for p in str(args.candidate_policies).split(",") if p.strip()]
    wanted_policies = [p for p in ["fullY", "base_residual", "all_visible_residual", "fullY_minus_known"] if p in policies or p == "fullY"]
    if not wanted_policies:
        raise ValueError("no candidate policies requested")

    all_row_records: List[Record] = []
    policy_rows: Dict[str, List[Record]] = {p: [] for p in wanted_policies}
    by_cert_rows: Dict[Tuple[str, str], List[Record]] = defaultdict(list)
    by_class_rows: Dict[Tuple[str, str], List[Record]] = defaultdict(list)
    failure_examples: List[Record] = []

    row_count_total = int(precision_stats["rows_kept"])
    carrier_join_miss = 0
    carrier_clip_mismatch = 0
    label_missing_count = 0
    gt_missing_vocab_count = 0
    scored_rows_total = 0
    clip_counter = 0
    t0 = time.perf_counter()

    for clip_key in sorted(row_groups.keys(), key=lambda x: (_as_str_id(x) or x)):
        clip_rows = row_groups[clip_key]
        clip_samples: List[Record] = []
        clip_row_meta: List[Record] = []
        for row in clip_rows:
            tid = _as_str_id(row.get("trajectory_id"))
            if tid is None:
                continue
            carrier = carrier_by_tid.get(tid)
            if carrier is None:
                carrier_join_miss += 1
                continue
            c_clip = _as_str_id(carrier.get("clip_id", carrier.get("video_id")))
            if c_clip is not None and c_clip != str(clip_key):
                carrier_clip_mismatch += 1
            clip_samples.append({"carrier_record": dict(carrier)})
            clip_row_meta.append(dict(row))
        if not clip_samples:
            continue
        carrier_pack = build_carrier_matrix_pack(
            clip_samples,
            output_root=carrier_asset_root,
            dataset_name=args.dataset_name,
            trajectory_source_branch="mainline",
        )
        carrier_matrix = np.asarray(carrier_pack["carrier_matrix"], dtype=np.float32)
        if carrier_matrix.ndim != 2 or int(carrier_matrix.shape[0]) != len(clip_samples):
            continue
        carrier_matrix = np.asarray([_normalize_np(row) for row in carrier_matrix], dtype=np.float32)
        carrier_dim = int(carrier_matrix.shape[1]) if carrier_matrix.size else 0
        text_dim = int(projected_text_matrix.shape[1]) if projected_text_matrix.size else 0
        if carrier_dim != text_dim:
            raise RuntimeError(f"DIM_MISMATCH: carrier_dim={carrier_dim} text_dim={text_dim}")
        full_logits = np.matmul(carrier_matrix, projected_text_matrix.T) / float(temperature)
        full_logits = np.asarray(full_logits, dtype=np.float32)
        clip_counter += 1
        if int(args.progress_every) > 0 and clip_counter % int(args.progress_every) == 0:
            elapsed = max(1e-9, time.perf_counter() - t0)
            print(
                f"[videocutler-residual-peeling-v2] clips={clip_counter}/{len(row_groups)} "
                f"rows={scored_rows_total} rate={scored_rows_total/elapsed:.1f}/s elapsed={elapsed:.1f}s",
                flush=True,
            )

        for row_index, row in enumerate(clip_row_meta):
            gt_raw_id = _as_str_id(row.get("best_gt_raw_id", row.get("matched_gt_raw_id", row.get("gt_raw_id"))))
            if gt_raw_id is None:
                continue
            gt_raw_id = str(gt_raw_id)
            label_row = label_by_id.get(gt_raw_id)
            if not label_row:
                label_missing_count += 1
                continue
            if gt_raw_id not in raw_to_idx:
                gt_missing_vocab_count += 1
                continue
            resolved_at = _as_int(label_row.get("resolved_at_iteration", label_row.get("iteration", 0)), default=0)
            known = known_before_iteration(resolved_by_iter, initial_known, resolved_at)
            if gt_raw_id in known:
                known = set(known)
                known.discard(gt_raw_id)
            cert = str(label_row.get("certificate_type", label_row.get("certificate", "unknown")))
            clip_id = str(row.get("clip_id", row.get("video_id", clip_key)))
            row_record: Record = {
                "clip_id": clip_id,
                "video_id": _as_int(row.get("video_id", clip_key), default=_as_int(clip_key, 0)),
                "trajectory_id": row.get("trajectory_id"),
                "raw_category_id": int(gt_raw_id),
                "best_gt_iou": float(_as_float(row.get("best_gt_iou", 0.0), 0.0)),
                "certificate_type": cert,
                "resolved_at_iteration": int(resolved_at),
                "known_size_before": int(len(known)),
            }

            policy_source_rows: Dict[str, Set[str]] = {
                "fullY": set(full_vocab),
                "base_residual": _policy_source_set(
                    "base_residual",
                    gt_raw_id=gt_raw_id,
                    clip_key=str(clip_key),
                    known=known,
                    base_ctx=base_ctx,
                    all_ctx=all_ctx,
                    full_vocab=full_vocab,
                ),
                "all_visible_residual": _policy_source_set(
                    "all_visible_residual",
                    gt_raw_id=gt_raw_id,
                    clip_key=str(clip_key),
                    known=known,
                    base_ctx=base_ctx,
                    all_ctx=all_ctx,
                    full_vocab=full_vocab,
                ),
                "fullY_minus_known": _policy_source_set(
                    "fullY_minus_known",
                    gt_raw_id=gt_raw_id,
                    clip_key=str(clip_key),
                    known=known,
                    base_ctx=base_ctx,
                    all_ctx=all_ctx,
                    full_vocab=full_vocab,
                ),
            }

            full_logits_row = np.asarray(full_logits[row_index], dtype=np.float32)
            row_record["candidate_fullY_contains_gt"] = bool(gt_raw_id in policy_source_rows["fullY"])
            row_record["fullY_contains_gt"] = bool(gt_raw_id in policy_source_rows["fullY"])
            row_record["fullY_score_rank"] = None
            row_record["fullY_top1"] = None
            row_record["fullY_top5"] = None
            row_record["fullY_top20"] = None
            row_record["fullY_margin"] = None
            row_record["fullY_candidate_size"] = None

            for policy in wanted_policies:
                source = set(policy_source_rows[policy])
                contains_gt = gt_raw_id in source
                source.add(gt_raw_id)
                if gt_raw_id not in raw_to_idx:
                    continue
                candidate_indices = [int(raw_to_idx[x]) for x in source if x in raw_to_idx]
                if not candidate_indices:
                    continue
                candidate_indices = list(dict.fromkeys(candidate_indices))
                gt_index = int(raw_to_idx[gt_raw_id])
                if gt_index not in candidate_indices:
                    candidate_indices.append(gt_index)
                local_gt_idx = candidate_indices.index(gt_index)
                row_scores = np.asarray(full_logits_row[np.asarray(candidate_indices, dtype=np.int64)], dtype=np.float64)
                rank, top_local, gt_score, margin = _safe_rank(row_scores, local_gt_idx)
                top1_idx = int(candidate_indices[top_local])
                top1_raw = next((rid for rid, idx in raw_to_idx.items() if int(idx) == top1_idx), str(top1_idx))
                probs = np.exp(row_scores - float(np.max(row_scores)))
                probs = probs / max(float(np.sum(probs)), 1e-12)
                entropy = float(-np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))))
                masked = np.asarray(row_scores, dtype=np.float64).copy()
                masked[local_gt_idx] = -np.inf
                best_non = float(np.max(masked)) if masked.size else float("-inf")
                row_out = {
                    **row_record,
                    "policy": policy,
                    "candidate_contains_gt": bool(contains_gt),
                    "candidate_source_size": int(len(policy_source_rows[policy])),
                    "candidate_eval_size": int(len(candidate_indices)),
                    "gt_rank": int(rank),
                    "gt_top1": bool(rank == 1),
                    "gt_top5": bool(rank <= 5),
                    "gt_top20": bool(rank <= 20),
                    "gt_normalized_gt_rank": float((rank - 1) / max(len(candidate_indices) - 1, 1)),
                    "gt_margin_vs_best_non_gt": float(margin),
                    "assignment_entropy": entropy,
                    "top1_raw_id": str(top1_raw),
                    "score_gt": float(gt_score),
                    "score_best_non_gt": float(best_non) if math.isfinite(best_non) else None,
                    "score_top1": float(row_scores[top_local]),
                }
                policy_rows[policy].append(row_out)
                scored_rows_total += 1
                all_row_records.append(row_out)
                by_cert_rows[(cert, policy)].append(row_out)
                by_class_rows[(str(gt_raw_id), policy)].append(row_out)
                if len(failure_examples) < int(args.top_examples) and (not contains_gt or not bool(rank == 1)):
                    failure_examples.append(
                        {
                            "policy": policy,
                            "clip_id": clip_id,
                            "trajectory_id": row.get("trajectory_id"),
                            "raw_category_id": int(gt_raw_id),
                            "best_gt_iou": float(_as_float(row.get("best_gt_iou", 0.0), 0.0)),
                            "candidate_contains_gt": bool(contains_gt),
                            "candidate_source_size": int(len(policy_source_rows[policy])),
                            "candidate_eval_size": int(len(candidate_indices)),
                            "gt_rank": int(rank),
                            "gt_margin_vs_best_non_gt": float(margin),
                            "top1_raw_id": str(top1_raw),
                            "certificate_type": cert,
                        }
                    )

    policy_summaries: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    for policy in wanted_policies:
        summary = _summarize_policy_rows(policy_rows[policy])
        summary["status"] = "PASS_SCORER" if summary["row_count_total"] > 0 else "FAIL_SCHEMA"
        summary["comparison_to_E_in_scope"] = {
            "E_in_scope_oracle_valid_rate": 1.0,
            "latent_gap_to_oracle": float(1.0 - float(summary["residual_rank1_rate"] or 0.0)),
        }
        policy_summaries[policy] = summary
        summary_rows.append(_policy_summary_row(policy, summary))

    cert_summary_rows: List[Dict[str, Any]] = []
    for (cert, policy), rows in sorted(by_cert_rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        s = _summarize_policy_rows(rows)
        cert_summary_rows.append({
            "certificate_type": cert,
            "policy": policy,
            "row_count": int(len(rows)),
            "fullY_rank1_rate": s.get("fullY_rank1_rate"),
            "residual_rank1_rate": s.get("residual_rank1_rate"),
            "residual_top5_rate": s.get("residual_top5_rate"),
            "residual_top20_rate": s.get("residual_top20_rate"),
            "candidate_contains_gt_rate": s.get("candidate_contains_gt_rate"),
            "mean_gt_rank": s.get("mean_gt_rank"),
            "median_gt_rank": s.get("median_gt_rank"),
            "mean_normalized_gt_rank": s.get("mean_normalized_gt_rank"),
            "mean_gt_margin_vs_best_non_gt": s.get("mean_gt_margin_vs_best_non_gt"),
            "positive_gt_margin_rate": s.get("positive_gt_margin_rate"),
            "top1_wrong_rate": s.get("top1_wrong_rate"),
        })

    class_summary_rows: List[Dict[str, Any]] = []
    for (raw_id, policy), rows in sorted(by_class_rows.items(), key=lambda kv: (kv[0][1], int(kv[0][0]) if str(kv[0][0]).isdigit() else str(kv[0][0]))):
        s = _summarize_policy_rows(rows)
        class_summary_rows.append({
            "raw_category_id": raw_id,
            "class_name": ann_names.get(raw_id, split_names.get(raw_id, raw_id)),
            "policy": policy,
            "row_count": int(len(rows)),
            "certificate_type": rows[0].get("certificate_type", "") if rows else "",
            "fullY_rank1_rate": s.get("fullY_rank1_rate"),
            "residual_rank1_rate": s.get("residual_rank1_rate"),
            "residual_top5_rate": s.get("residual_top5_rate"),
            "residual_top20_rate": s.get("residual_top20_rate"),
            "candidate_contains_gt_count": s.get("candidate_contains_gt_count"),
            "candidate_contains_gt_rate": s.get("candidate_contains_gt_rate"),
            "mean_gt_rank": s.get("mean_gt_rank"),
            "median_gt_rank": s.get("median_gt_rank"),
            "mean_normalized_gt_rank": s.get("mean_normalized_gt_rank"),
            "mean_gt_margin_vs_best_non_gt": s.get("mean_gt_margin_vs_best_non_gt"),
            "positive_gt_margin_rate": s.get("positive_gt_margin_rate"),
        })

    backend_trace = {
        "generator_script": "videocutler/run_stageb_analysis_gtcarrier_latent_rowdump.py",
        "generator_function_chain": [
            "main",
            "_load_projector",
            "_project_text_matrix",
        ],
        "scoring_backend": "current_v2b_prealign_projector_text_cosine",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_stage_id": str(getattr(projector, "stage_id", "prealign")),
        "projector_input_dim": int(getattr(getattr(projector, "config", None), "input_dim", 0)),
        "projector_output_dim": int(getattr(getattr(projector, "config", None), "output_dim", 0)),
        "projector_temperature": float(temperature),
        "text_vocab_records_path": str(backend_run_root / "text_bank" / "text_prototype_records.jsonl"),
        "text_vocab_rows": int(len(text_vocab_ids)),
        "text_dim": int(np.asarray(text_vocab_matrix).shape[1]) if np.asarray(text_vocab_matrix).ndim == 2 else 0,
        "projected_text_dim": int(np.asarray(projected_text_matrix).shape[1]) if np.asarray(projected_text_matrix).ndim == 2 else 0,
        "carrier_asset_root": str(carrier_asset_root),
        "carrier_records_jsonl": str(carrier_jsonl),
        "carrier_dim": int(np.asarray(next(iter(policy_rows["fullY"]), {}).get("candidate_eval_size", 0)) if False else int(np.asarray(projected_text_matrix).shape[1])),
        "why_naive_overlay_fails": (
            "text prototypes are 512-d before projection; the scorer projects them into the "
            "projector output space (768-d) before dotting against carrier vectors. A direct "
            "512-d overlay skips that projector and produces a dimension mismatch."
        ),
    }
    # real carrier dimension from the first loaded carrier vector, if any
    if carrier_by_tid:
        first_vec_samples = [{"carrier_record": dict(next(iter(carrier_by_tid.values())))}]
        try:
            carrier_pack = build_carrier_matrix_pack(
                first_vec_samples,
                output_root=carrier_asset_root,
                dataset_name=args.dataset_name,
                trajectory_source_branch="mainline",
            )
            carrier_dim = int(np.asarray(carrier_pack["carrier_matrix"]).shape[1])
        except Exception:
            carrier_dim = 0
        backend_trace["carrier_dim"] = carrier_dim

    summary = {
        "status": "PASS" if row_count_total > 0 else "FAIL_SCHEMA",
        "dataset_name": args.dataset_name,
        "variant": args.variant,
        "run_root": str(output_root),
        "run_root_v2b": str(backend_run_root),
        "output_dir": str(out_dir),
        "annotation_json": str(annotation_json),
        "split_json": str(split_json),
        "per_class_csv": str(per_class_csv),
        "trajectory_precision_rows_csv": str(precision_csv),
        "carrier_records_jsonl": str(carrier_jsonl),
        "carrier_asset_root": str(carrier_asset_root),
        "min_iou": float(args.min_iou),
        "person_raw_id": _as_int(args.person_raw_id, default=773),
        "label_meta": label_meta,
        "split_counts": {"base": len(base_ids), "novel": len(novel_ids)},
        "precision_rows": precision_stats,
        "carrier_rows_loaded": int(carrier_stats["rows_kept"]),
        "carrier_rows_seen": int(carrier_stats["rows_seen"]),
        "carrier_join_miss": int(carrier_join_miss),
        "carrier_clip_mismatch": int(carrier_clip_mismatch),
        "label_missing_count": int(label_missing_count),
        "gt_missing_vocab_count": int(gt_missing_vocab_count),
        "score_chunk_size": int(args.score_chunk_size),
        "row_count_total": int(row_count_total),
        "row_count_with_label_record": int(row_count_total - precision_stats["skipped_no_tid"] - precision_stats["skipped_no_gt"] - precision_stats["skipped_no_clip"] - precision_stats["skipped_low_iou"]),
        "row_count_in_scope": int(len(all_row_records)),
        "scored_rows_total": int(scored_rows_total),
        "backend_trace": backend_trace,
        "policy_summaries": policy_summaries,
        "outputs": {
            "summary_json": str(out_dir / "summary.json"),
            "summary_by_policy_csv": str(out_dir / "summary_by_policy.csv"),
            "summary_by_certificate_csv": str(out_dir / "summary_by_certificate.csv"),
            "per_class_csv": str(out_dir / "per_class_videocutler_residual_oracle.csv"),
            "failure_examples_jsonl": str(out_dir / "failure_examples.jsonl"),
            "scorer_backend_trace_md": str(out_dir / "SCORER_BACKEND_TRACE.md"),
            "takeover_md": str(out_dir / "VIDEOCUTLER_RESIDUAL_PEELING_ORACLE_V2_TAKEOVER.md"),
        },
        "warnings": [],
    }
    if backend_trace["projector_output_dim"] != backend_trace["carrier_dim"]:
        summary["warnings"].append("projector output dim does not match carrier dim")
    if len(all_row_records) <= 0:
        summary["warnings"].append("no evaluable rows after filtering and carrier join")

    _write_json(out_dir / "summary.json", summary)
    _write_csv(
        out_dir / "summary_by_policy.csv",
        summary_rows,
        [
            "policy",
            "row_count_total",
            "row_count_with_label_record",
            "row_count_in_scope",
            "candidate_contains_gt_count",
            "candidate_contains_gt_rate",
            "latent_evaluable_row_count",
            "fullY_rank1_rate",
            "residual_rank1_rate",
            "residual_top5_rate",
            "residual_top20_rate",
            "mean_gt_rank",
            "median_gt_rank",
            "mean_normalized_gt_rank",
            "mean_gt_margin_vs_best_non_gt",
            "positive_gt_margin_rate",
            "top1_wrong_rate",
            "wrong_large_negative_margin_rate",
            "wrong_near_tie_margin_rate",
            "assignment_entropy_mean",
            "hub_like_top1_concentration",
            "top1_label_top_counts",
            "wrong_top1_label_top_counts",
            "confusion_pairs_top",
            "comparison_to_E_in_scope",
        ],
    )
    _write_csv(
        out_dir / "summary_by_certificate.csv",
        cert_summary_rows,
        [
            "certificate_type",
            "policy",
            "row_count",
            "fullY_rank1_rate",
            "residual_rank1_rate",
            "residual_top5_rate",
            "residual_top20_rate",
            "candidate_contains_gt_rate",
            "mean_gt_rank",
            "median_gt_rank",
            "mean_normalized_gt_rank",
            "mean_gt_margin_vs_best_non_gt",
            "positive_gt_margin_rate",
            "top1_wrong_rate",
        ],
    )
    _write_csv(
        out_dir / "per_class_videocutler_residual_oracle.csv",
        class_summary_rows,
        [
            "raw_category_id",
            "class_name",
            "policy",
            "row_count",
            "certificate_type",
            "fullY_rank1_rate",
            "residual_rank1_rate",
            "residual_top5_rate",
            "residual_top20_rate",
            "candidate_contains_gt_count",
            "candidate_contains_gt_rate",
            "mean_gt_rank",
            "median_gt_rank",
            "mean_normalized_gt_rank",
            "mean_gt_margin_vs_best_non_gt",
            "positive_gt_margin_rate",
        ],
    )
    with (out_dir / "failure_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in failure_examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    trace_md = out_dir / "SCORER_BACKEND_TRACE.md"
    trace_lines = [
        "# Scorer Backend Trace",
        "",
        "Exact D-full row score generator:",
        "- `videocutler/run_stageb_analysis_gtcarrier_latent_rowdump.py`",
        "- function chain: `main -> _load_projector -> _project_text_matrix -> row-wise scoring`",
        "",
        "Backend components reused here:",
        f"- checkpoint: `{checkpoint_path}`",
        f"- projector input dim: `{backend_trace['projector_input_dim']}`",
        f"- projector output dim: `{backend_trace['projector_output_dim']}`",
        f"- text vocab records: `{backend_trace['text_vocab_records_path']}`",
        f"- text vocab rows: `{backend_trace['text_vocab_rows']}`",
        f"- text dim before projection: `{backend_trace['text_dim']}`",
        f"- projected text dim: `{backend_trace['projected_text_dim']}`",
        f"- carrier dim: `{backend_trace['carrier_dim']}`",
        f"- temperature: `{backend_trace['projector_temperature']}`",
        "",
        "Why the naive overlay fails:",
        "- The text bank stores 512-d prototype vectors.",
        "- The scorer projects them through the prealign text projector into the 768-d carrier space before scoring.",
        "- A direct 512-d dot-product against 768-d carrier vectors skips that projector and produces a dimension mismatch.",
        "",
        "The v2 audit therefore scores each clip with the projected 768-d text matrix and the 768-d carrier matrix, then applies the residual candidate-policy subsets on top of that exact backend.",
    ]
    trace_md.write_text("\n".join(trace_lines).rstrip() + "\n", encoding="utf-8")

    takeover_path = out_dir / "VIDEOCUTLER_RESIDUAL_PEELING_ORACLE_V2_TAKEOVER.md"
    _write_takeover(takeover_path, summary)

    print(json.dumps({
        "status": summary["status"],
        "output_dir": str(out_dir),
        "row_count_total": summary["row_count_total"],
        "scored_rows_total": summary["scored_rows_total"],
        "warnings": summary["warnings"],
        "backend_trace": backend_trace,
        "policy_summaries": policy_summaries,
    }, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
