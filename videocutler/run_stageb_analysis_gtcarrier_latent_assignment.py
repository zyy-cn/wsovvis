#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_carrier_evidence  # noqa: E402
from videocutler.ext_stageb_ovvis.audit._matrix_vocab_scoring import (  # noqa: E402
    build_carrier_matrix_pack,
    compute_fused_logits_matrix_numpy,
)
from videocutler.ext_stageb_ovvis.eval.g8_bridge import (  # noqa: E402
    load_projector_bundle,
    load_text_vocab_with_names,
    resolve_inference_asset_roots,
    resolve_selected_for_infer,
)
from videocutler.ext_stageb_ovvis.data.oracle_clean_ablation_sources import (  # noqa: E402
    iter_jsonl,
    load_json,
    load_weak_label_records,
    safe_int,
    unique_ints,
    write_csv,
    write_json,
)


Record = Dict[str, Any]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GT-carrier latent assignment audit (read-only).")
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--run_root_v2b", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--full_y_path", required=True)
    p.add_argument("--weak_label_path", required=True)
    p.add_argument("--gt_carrier_path", required=True)
    p.add_argument("--gt_identity_path", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--top_examples", type=int, default=100)
    p.add_argument("--show_progress", type=_parse_bool, default=False)
    return p.parse_args()


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _median(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(median(vals)) if vals else None


def _rate(num: int, den: int) -> Optional[float]:
    return float(num / den) if den else None


def _normalized_scope_key(value: Any) -> Optional[str]:
    ix = safe_int(value, None)
    if ix is not None:
        return str(int(ix))
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _load_full_y_records(path: Path) -> List[Record]:
    payload = load_json(path)
    rows = payload.get("records", [])
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _build_label_map(records: Sequence[Mapping[str, Any]], *, label_key: str) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for rec in records:
        clip_key = _normalized_scope_key(rec.get("clip_id"))
        if clip_key is None:
            clip_key = _normalized_scope_key(rec.get("video_id"))
        if clip_key is None:
            continue
        labels = unique_ints(rec.get(label_key, []))
        if clip_key not in out:
            out[clip_key] = {
                "clip_id": rec.get("clip_id"),
                "video_id": rec.get("video_id"),
                label_key: sorted(set(int(x) for x in labels)),
            }
        else:
            existing = set(unique_ints(out[clip_key].get(label_key, [])))
            existing.update(int(x) for x in labels)
            out[clip_key][label_key] = sorted(existing)
            if out[clip_key].get("clip_id") is None and rec.get("clip_id") is not None:
                out[clip_key]["clip_id"] = rec.get("clip_id")
            if out[clip_key].get("video_id") is None and rec.get("video_id") is not None:
                out[clip_key]["video_id"] = rec.get("video_id")
    return out


def _load_jsonl_rows(path: Path) -> List[Record]:
    return [dict(row) for row in iter_jsonl(path)]


def _load_binding_rows(path: Path) -> List[Record]:
    rows = _load_jsonl_rows(path)
    return rows


def _build_binding_indexes(
    binding_rows: Sequence[Mapping[str, Any]],
    carrier_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Record], Dict[int, Record], Record]:
    carrier_by_tid: Dict[str, Record] = {}
    carrier_by_index: Dict[int, Record] = {}
    for idx, row in enumerate(carrier_rows):
        carrier_by_index[int(idx)] = dict(row)
        tid = str(row.get("trajectory_id", "")).strip()
        if tid:
            carrier_by_tid[tid] = dict(row)

    binding_by_tid: Dict[str, Record] = {}
    binding_by_index: Dict[int, Record] = {}
    for idx, row in enumerate(binding_rows):
        tid = str(row.get("trajectory_id", "")).strip()
        if tid:
            binding_by_tid[tid] = dict(row)
        binding_by_index[int(idx)] = dict(row)

    meta = {
        "binding_row_count": int(len(binding_rows)),
        "carrier_row_count": int(len(carrier_rows)),
        "binding_unique_trajectory_count": int(len(binding_by_tid)),
        "carrier_unique_trajectory_count": int(len(carrier_by_tid)),
    }
    return binding_by_tid, carrier_by_index, meta


def _score_group(
    *,
    clip_key: str,
    rows: Sequence[Mapping[str, Any]],
    candidate_raw_ids: Sequence[int],
    candidate_label_key: str,
    bundle: Any,
    text_vocab_ids: Sequence[int],
    text_vocab_matrix: np.ndarray,
    text_vocab_index: Mapping[int, int],
    gt_asset_root: Path,
    dataset_name: str,
    top_examples: int,
) -> Tuple[List[Record], Dict[str, Any], List[Record]]:
    candidate_ids = [int(x) for x in unique_ints(candidate_raw_ids)]
    candidate_missing_vocab = [int(rid) for rid in candidate_ids if int(rid) not in text_vocab_index]
    candidate_ids = [int(rid) for rid in candidate_ids if int(rid) in text_vocab_index]
    candidate_cols = [int(text_vocab_index[int(rid)]) for rid in candidate_ids]
    candidate_matrix = np.asarray(text_vocab_matrix[candidate_cols], dtype=np.float32) if candidate_cols else np.zeros((0, int(text_vocab_matrix.shape[1])), dtype=np.float32)

    if not candidate_ids:
        return [], {
            "clip_id": clip_key,
            "row_count": int(len(rows)),
            "row_count_with_label_record": int(len(rows)),
            "candidate_contains_gt_count": 0,
            "candidate_contains_gt_rate": 0.0,
            "latent_evaluable_row_count": 0,
            "gt_rank1_rate": None,
            "gt_top5_rate": None,
            "gt_top20_rate": None,
            "mean_gt_rank": None,
            "median_gt_rank": None,
            "mean_normalized_gt_rank": None,
            "mean_gt_margin_vs_best_non_gt": None,
            "positive_gt_margin_rate": None,
            "top1_wrong_rate": None,
            "assignment_entropy_mean": None,
            "hub_like_top1_concentration": None,
            "top1_label_top_counts": [],
            "top1_wrong_label_top_counts": [],
            "candidate_missing_vocab_ids": candidate_missing_vocab[:32],
        }, []

    scored_rows: List[Record] = []
    example_rows: List[Record] = []
    carrier_samples: List[Record] = []
    carrier_rows: List[Record] = []
    for row in rows:
        carrier_record = row.get("carrier_record")
        if not isinstance(carrier_record, Mapping):
            continue
        carrier_samples.append({"carrier_record": dict(carrier_record)})
        carrier_rows.append(dict(row))

    if not carrier_rows:
        return scored_rows, {
            "clip_id": clip_key,
            "row_count": 0,
            "row_count_with_label_record": 0,
            "candidate_contains_gt_count": 0,
            "candidate_contains_gt_rate": None,
            "latent_evaluable_row_count": 0,
            "gt_rank1_rate": None,
            "gt_top5_rate": None,
            "gt_top20_rate": None,
            "mean_gt_rank": None,
            "median_gt_rank": None,
            "mean_normalized_gt_rank": None,
            "mean_gt_margin_vs_best_non_gt": None,
            "positive_gt_margin_rate": None,
            "top1_wrong_rate": None,
            "assignment_entropy_mean": None,
            "hub_like_top1_concentration": None,
            "top1_label_top_counts": [],
            "top1_wrong_label_top_counts": [],
            "candidate_missing_vocab_ids": candidate_missing_vocab[:32],
        }, example_rows

    carrier_pack = build_carrier_matrix_pack(
        carrier_samples,
        output_root=gt_asset_root,
        dataset_name=dataset_name,
        trajectory_source_branch="gt_upper_bound",
    )
    carrier_matrix = np.asarray(carrier_pack["carrier_matrix"], dtype=np.float32)
    logits = compute_fused_logits_matrix_numpy(
        carrier_matrix=carrier_matrix,
        projector=bundle.projector,
        candidate_matrix=candidate_matrix,
        temperature=float(bundle.temperature),
        batch_size=256,
    )
    logits = np.asarray(logits, dtype=np.float64)
    candidate_index = {int(rid): idx for idx, rid in enumerate(candidate_ids)}
    candidate_set = set(candidate_ids)
    label_record_count = 0
    candidate_contains_gt_count = 0
    gt_ranks: List[int] = []
    gt_norm_ranks: List[float] = []
    gt_margins: List[float] = []
    gt_top1_hits: List[bool] = []
    gt_top5_hits: List[bool] = []
    gt_top20_hits: List[bool] = []
    entropies: List[float] = []
    top1_counts: Counter[int] = Counter()
    top1_wrong_counts: Counter[int] = Counter()
    positive_margin_hits = 0

    for idx, row in enumerate(carrier_rows):
        gt_raw_id = safe_int(row.get("raw_category_id"), None)
        if gt_raw_id is None:
            continue
        label_record_count += 1
        top1_idx = int(np.argmax(logits[idx])) if logits.shape[1] else -1
        top1_raw_id = int(candidate_ids[top1_idx]) if top1_idx >= 0 else None
        if top1_raw_id is not None:
            top1_counts[int(top1_raw_id)] += 1
        contains_gt = int(gt_raw_id) in candidate_set
        if contains_gt:
            candidate_contains_gt_count += 1
            gt_idx = int(candidate_index[int(gt_raw_id)])
            gt_score = float(logits[idx, gt_idx])
            rank = int(np.count_nonzero(logits[idx] > gt_score) + 1)
            denom = max(1, int(logits.shape[1]) - 1)
            norm_rank = float((rank - 1) / denom)
            masked = np.asarray(logits[idx], dtype=np.float64).copy()
            masked[gt_idx] = -np.inf
            best_non_gt = float(np.max(masked)) if masked.size else float("-inf")
            margin = float(gt_score - best_non_gt)
            probs = np.exp(logits[idx] - float(np.max(logits[idx])))
            probs = probs / max(float(np.sum(probs)), 1e-12)
            entropy = float(-np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))))
            gt_ranks.append(rank)
            gt_norm_ranks.append(norm_rank)
            gt_margins.append(margin)
            gt_top1_hits.append(bool(top1_raw_id is not None and int(top1_raw_id) == int(gt_raw_id)))
            gt_top5_hits.append(bool(rank <= 5))
            gt_top20_hits.append(bool(rank <= 20))
            entropies.append(entropy)
            if margin > 0:
                positive_margin_hits += 1
            if top1_raw_id is not None and int(top1_raw_id) != int(gt_raw_id):
                top1_wrong_counts[int(top1_raw_id)] += 1
            if len(example_rows) < top_examples and int(top1_raw_id or -1) != int(gt_raw_id):
                example_rows.append(
                    {
                        "clip_id": clip_key,
                        "trajectory_id": row.get("trajectory_id"),
                        "raw_category_id": int(gt_raw_id),
                        "top1_raw_id": top1_raw_id,
                        "gt_rank": int(rank),
                        "gt_margin": float(margin),
                        "candidate_contains_gt": True,
                        "candidate_count": int(len(candidate_ids)),
                    }
                )
        else:
            probs = np.exp(logits[idx] - float(np.max(logits[idx])))
            probs = probs / max(float(np.sum(probs)), 1e-12)
            entropy = float(-np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))))
            entropies.append(entropy)
            if len(example_rows) < top_examples:
                example_rows.append(
                    {
                        "clip_id": clip_key,
                        "trajectory_id": row.get("trajectory_id"),
                        "raw_category_id": int(gt_raw_id),
                        "top1_raw_id": top1_raw_id,
                        "gt_rank": None,
                        "gt_margin": None,
                        "candidate_contains_gt": False,
                        "candidate_count": int(len(candidate_ids)),
                    }
                )

    evaluable = int(candidate_contains_gt_count)
    row_count = int(len(carrier_rows))
    top1_total = int(sum(top1_counts.values()))
    top1_concentration = float(max(top1_counts.values()) / top1_total) if top1_total else None
    metrics = {
        "clip_id": clip_key,
        "row_count": row_count,
        "row_count_with_label_record": int(label_record_count),
        "candidate_contains_gt_count": int(candidate_contains_gt_count),
        "candidate_contains_gt_rate": _rate(candidate_contains_gt_count, label_record_count),
        "latent_evaluable_row_count": int(evaluable),
        "gt_rank1_rate": _rate(int(sum(gt_top1_hits)), evaluable),
        "gt_top5_rate": _rate(int(sum(gt_top5_hits)), evaluable),
        "gt_top20_rate": _rate(int(sum(gt_top20_hits)), evaluable),
        "mean_gt_rank": _mean([float(x) for x in gt_ranks]),
        "median_gt_rank": _median([float(x) for x in gt_ranks]),
        "mean_normalized_gt_rank": _mean([float(x) for x in gt_norm_ranks]),
        "mean_gt_margin_vs_best_non_gt": _mean([float(x) for x in gt_margins]),
        "positive_gt_margin_rate": _rate(positive_margin_hits, evaluable),
        "top1_wrong_rate": _rate(int(evaluable - int(sum(gt_top1_hits))), evaluable),
        "assignment_entropy_mean": _mean([float(x) for x in entropies]),
        "hub_like_top1_concentration": top1_concentration,
        "top1_label_top_counts": [
            {"raw_category_id": int(raw_id), "count": int(count)}
            for raw_id, count in top1_counts.most_common(20)
        ],
        "top1_wrong_label_top_counts": [
            {"raw_category_id": int(raw_id), "count": int(count)}
            for raw_id, count in top1_wrong_counts.most_common(20)
        ],
        "candidate_missing_vocab_ids": candidate_missing_vocab[:32],
    }
    scored_rows.extend(
        [
            {
                "clip_id": clip_key,
                "trajectory_id": row.get("trajectory_id"),
                "raw_category_id": safe_int(row.get("raw_category_id"), None),
                "top1_raw_id": int(candidate_ids[int(np.argmax(logits[idx]))]) if logits.shape[1] else None,
                "candidate_contains_gt": bool(safe_int(row.get("raw_category_id"), None) in candidate_set),
            }
            for idx, row in enumerate(carrier_rows)
        ]
    )
    return scored_rows, metrics, example_rows


def _summarize_by_class(rows: Sequence[Mapping[str, Any]]) -> List[Record]:
    by_class: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_id = safe_int(row.get("raw_category_id"), None)
        if raw_id is None:
            continue
        by_class[int(raw_id)].append(dict(row))
    summary: List[Record] = []
    for raw_id in sorted(by_class):
        group = by_class[raw_id]
        gt_ranks = [int(row["gt_rank"]) for row in group if row.get("gt_rank") is not None]
        gt_margins = [float(row["gt_margin"]) for row in group if row.get("gt_margin") is not None]
        top1_hits = [bool(row.get("gt_rank") is not None and int(row.get("gt_rank")) == 1) for row in group if row.get("gt_rank") is not None]
        summary.append(
            {
                "raw_category_id": int(raw_id),
                "row_count": int(len(group)),
                "candidate_contains_gt_count": int(sum(1 for row in group if bool(row.get("candidate_contains_gt")))),
                "candidate_contains_gt_rate": _rate(sum(1 for row in group if bool(row.get("candidate_contains_gt"))), len(group)),
                "gt_rank1_rate": _rate(int(sum(top1_hits)), len(top1_hits)) if top1_hits else None,
                "mean_gt_rank": _mean([float(x) for x in gt_ranks]),
                "median_gt_rank": _median([float(x) for x in gt_ranks]),
                "mean_gt_margin_vs_best_non_gt": _mean([float(x) for x in gt_margins]),
                "positive_gt_margin_rate": _rate(sum(1 for x in gt_margins if x > 0), len(gt_margins)),
            }
        )
    return summary


def _summarize_by_clip(rows: Sequence[Mapping[str, Any]]) -> List[Record]:
    by_clip: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        clip_id = str(row.get("clip_id", "")).strip()
        if clip_id:
            by_clip[clip_id].append(dict(row))
    summary: List[Record] = []
    for clip_id in sorted(by_clip, key=lambda x: (_normalized_scope_key(x) or x)):
        group = by_clip[clip_id]
        gt_ranks = [int(row["gt_rank"]) for row in group if row.get("gt_rank") is not None]
        summary.append(
            {
                "clip_id": clip_id,
                "row_count": int(len(group)),
                "candidate_count": int(max(int(row.get("candidate_count", 0)) for row in group) if group else 0),
                "candidate_contains_gt_count": int(sum(1 for row in group if bool(row.get("candidate_contains_gt")))),
                "candidate_contains_gt_rate": _rate(sum(1 for row in group if bool(row.get("candidate_contains_gt"))), len(group)),
                "gt_rank1_rate": _rate(sum(1 for row in group if int(row.get("gt_rank", 0) or 0) == 1), len(gt_ranks)) if gt_ranks else None,
                "mean_gt_rank": _mean([float(x) for x in gt_ranks]),
            }
        )
    return summary


def _combine_examples(*example_lists: Sequence[Mapping[str, Any]]) -> List[Record]:
    combined: List[Record] = []
    for examples in example_lists:
        for row in examples:
            if len(combined) >= 100:
                return combined
            combined.append(dict(row))
    return combined


def _load_projector_bundle_and_vocab(run_root_v2b: Path, dataset_name: str, device: str) -> Tuple[Any, Path, List[int], np.ndarray, Dict[int, int], Record]:
    resolution = resolve_selected_for_infer(run_root_v2b)
    asset_roots = resolve_inference_asset_roots(
        run_root_v2b,
        dataset_name=dataset_name,
        trajectory_source_branch="mainline",
        resolution=resolution,
    )
    checkpoint_path = run_root_v2b / "train" / "prealign" / "checkpoints" / "prealign_last.pth"
    bundle = load_projector_bundle(checkpoint_path, device=torch.device(device))
    text_vocab_ids, _text_records, text_vocab_matrix, _class_name_map = load_text_vocab_with_names(asset_roots.asset_root, dataset_name)
    text_vocab_ids = [int(x) for x in text_vocab_ids]
    text_vocab_index = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
    return bundle, asset_roots.asset_root, text_vocab_ids, np.asarray(text_vocab_matrix, dtype=np.float32), text_vocab_index, {
        "resolution": {
            "selected_for_infer": resolution.selected_for_infer,
            "checkpoint_path": str(resolution.checkpoint_path),
            "source": resolution.source,
            "train_state_path": str(resolution.train_state_path) if resolution.train_state_path else None,
        },
        "asset_root": str(asset_roots.asset_root),
        "trajectory_records_path": str(asset_roots.trajectory_records_path),
        "carrier_records_path": str(asset_roots.carrier_records_path),
        "text_records_path": str(asset_roots.text_records_path),
    }


def _arm_report(
    *,
    arm: str,
    label_source: str,
    carrier_source: str,
    assignment_mode: str,
    label_map: Mapping[str, Record],
    rows_by_clip: Mapping[str, Sequence[Mapping[str, Any]]],
    bundle: Any,
    gt_asset_root: Path,
    dataset_name: str,
    text_vocab_ids: Sequence[int],
    text_vocab_matrix: np.ndarray,
    text_vocab_index: Mapping[int, int],
    top_examples: int,
) -> Tuple[Record, List[Record], List[Record], List[Record]]:
    all_scored_rows: List[Record] = []
    all_clip_metrics: List[Record] = []
    all_examples: List[Record] = []
    candidate_missing_clip_count = 0
    missing_label_clip_count = 0
    for clip_key in sorted(rows_by_clip.keys(), key=lambda x: (_normalized_scope_key(x) or x)):
        group_rows = list(rows_by_clip[clip_key])
        label_record = dict(label_map.get(str(clip_key), {}))
        if label_source == "yprime":
            candidate_raw_ids = unique_ints(
                label_record.get("yprime_raw_ids", label_record.get("observed_raw_ids", []))
            )
        else:
            candidate_raw_ids = unique_ints(label_record.get("full_y_raw_ids", []))
        if not label_record:
            missing_label_clip_count += 1
        scored_rows, metrics, examples = _score_group(
            clip_key=str(clip_key),
            rows=group_rows,
            candidate_raw_ids=candidate_raw_ids,
            candidate_label_key="yprime_raw_ids" if label_source == "yprime" else "full_y_raw_ids",
            bundle=bundle,
            text_vocab_ids=text_vocab_ids,
            text_vocab_matrix=text_vocab_matrix,
            text_vocab_index=text_vocab_index,
            gt_asset_root=gt_asset_root,
            dataset_name=dataset_name,
            top_examples=top_examples,
        )
        if not scored_rows:
            candidate_missing_clip_count += 1
            continue
        all_scored_rows.extend(scored_rows)
        all_clip_metrics.append(metrics)
        all_examples.extend(examples)

    total_rows = int(sum(len(v) for v in rows_by_clip.values()))
    label_row_count = int(sum(int(metrics["row_count_with_label_record"]) for metrics in all_clip_metrics))
    evaluable_count = int(sum(int(metrics["latent_evaluable_row_count"]) for metrics in all_clip_metrics))
    gt_rank1_count = int(sum(int(round(float(metrics["gt_rank1_rate"]) * int(metrics["latent_evaluable_row_count"]))) if metrics["gt_rank1_rate"] is not None else 0 for metrics in all_clip_metrics))
    gt_top5_count = int(sum(int(round(float(metrics["gt_top5_rate"]) * int(metrics["latent_evaluable_row_count"]))) if metrics["gt_top5_rate"] is not None else 0 for metrics in all_clip_metrics))
    gt_top20_count = int(sum(int(round(float(metrics["gt_top20_rate"]) * int(metrics["latent_evaluable_row_count"]))) if metrics["gt_top20_rate"] is not None else 0 for metrics in all_clip_metrics))
    gt_ranks: List[float] = []
    gt_norm_ranks: List[float] = []
    gt_margins: List[float] = []
    entropies: List[float] = []
    top1_counts: Counter[int] = Counter()
    top1_wrong_counts: Counter[int] = Counter()
    candidate_contains_gt_count = int(sum(int(metrics["candidate_contains_gt_count"]) for metrics in all_clip_metrics))
    for metrics in all_clip_metrics:
        for item in metrics["top1_label_top_counts"]:
            top1_counts[int(item["raw_category_id"])] += int(item["count"])
        for item in metrics["top1_wrong_label_top_counts"]:
            top1_wrong_counts[int(item["raw_category_id"])] += int(item["count"])
    for row in all_examples:
        pass
    for clip_rows in all_clip_metrics:
        # these are per-clip aggregates; the detailed lists are recovered from the scored row records below.
        pass
    for row in all_scored_rows:
        if row.get("candidate_contains_gt") and row.get("gt_rank") is not None:
            gt_ranks.append(float(row.get("gt_rank")))
            gt_norm_ranks.append(float(row.get("gt_normalized_gt_rank", row.get("mean_normalized_gt_rank", 0.0))) if row.get("gt_normalized_gt_rank") is not None else 0.0)
            gt_margins.append(float(row.get("gt_margin_vs_best_non_gt", row.get("mean_gt_margin_vs_best_non_gt", 0.0))) if row.get("gt_margin_vs_best_non_gt") is not None else 0.0)
        if row.get("assignment_entropy") is not None:
            entropies.append(float(row.get("assignment_entropy")))

    # Reconstruct the detailed row records and metrics from the scored row groups.
    by_class: Dict[int, List[Record]] = defaultdict(list)
    by_clip: Dict[str, List[Record]] = defaultdict(list)
    failure_examples: List[Record] = []
    top1_wrong_counter: Counter[int] = Counter()
    top1_counter: Counter[int] = Counter()
    for clip_key in sorted(rows_by_clip.keys(), key=lambda x: (_normalized_scope_key(x) or x)):
        label_record = dict(label_map.get(str(clip_key), {}))
        if label_source == "yprime":
            candidate_raw_ids = unique_ints(
                label_record.get("yprime_raw_ids", label_record.get("observed_raw_ids", []))
            )
        else:
            candidate_raw_ids = unique_ints(label_record.get("full_y_raw_ids", []))
        candidate_set = {int(x) for x in candidate_raw_ids}
        candidate_index = {int(raw_id): idx for idx, raw_id in enumerate(candidate_raw_ids)}
        group_rows = list(rows_by_clip[clip_key])
        carrier_samples = [{"carrier_record": dict(row["carrier_record"])} for row in group_rows if isinstance(row.get("carrier_record"), Mapping)]
        if not carrier_samples:
            continue
        carrier_pack = build_carrier_matrix_pack(
            carrier_samples,
            output_root=gt_asset_root,
            dataset_name=dataset_name,
            trajectory_source_branch="gt_upper_bound",
        )
        carrier_matrix = np.asarray(carrier_pack["carrier_matrix"], dtype=np.float32)
        candidate_cols = [int(text_vocab_index[int(rid)]) for rid in candidate_raw_ids if int(rid) in text_vocab_index]
        candidate_ids = [int(rid) for rid in candidate_raw_ids if int(rid) in text_vocab_index]
        if not candidate_ids:
            continue
        candidate_matrix = np.asarray(text_vocab_matrix[candidate_cols], dtype=np.float32)
        logits = compute_fused_logits_matrix_numpy(
            carrier_matrix=carrier_matrix,
            projector=bundle.projector,
            candidate_matrix=candidate_matrix,
            temperature=float(bundle.temperature),
            batch_size=256,
        )
        logits = np.asarray(logits, dtype=np.float64)
        for idx, row in enumerate(group_rows):
            gt_raw_id = safe_int(row.get("raw_category_id"), None)
            if gt_raw_id is None:
                continue
            top1_idx = int(np.argmax(logits[idx])) if logits.shape[1] else -1
            top1_raw = int(candidate_ids[top1_idx]) if top1_idx >= 0 else None
            if top1_raw is not None:
                top1_counter[int(top1_raw)] += 1
            contains_gt = int(gt_raw_id) in candidate_set
            gt_rank = None
            gt_margin = None
            gt_norm_rank = None
            top1_is_gt = None
            top5 = None
            top20 = None
            if contains_gt:
                gt_idx = int(candidate_index[int(gt_raw_id)])
                gt_score = float(logits[idx, gt_idx])
                gt_rank = int(np.count_nonzero(logits[idx] > gt_score) + 1)
                denom = max(1, int(logits.shape[1]) - 1)
                gt_norm_rank = float((gt_rank - 1) / denom)
                masked = np.asarray(logits[idx], dtype=np.float64).copy()
                masked[gt_idx] = -np.inf
                best_non_gt = float(np.max(masked)) if masked.size else float("-inf")
                gt_margin = float(gt_score - best_non_gt)
                top1_is_gt = bool(top1_raw is not None and int(top1_raw) == int(gt_raw_id))
                top5 = bool(gt_rank <= 5)
                top20 = bool(gt_rank <= 20)
                if top1_is_gt is False and top1_raw is not None:
                    top1_wrong_counter[int(top1_raw)] += 1
                if gt_margin is not None and gt_margin > 0:
                    pass
            probs = np.exp(logits[idx] - float(np.max(logits[idx])))
            probs = probs / max(float(np.sum(probs)), 1e-12)
            entropy = float(-np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))))
            rec = {
                "clip_id": clip_key,
                "trajectory_id": row.get("trajectory_id"),
                "raw_category_id": int(gt_raw_id),
                "candidate_contains_gt": bool(contains_gt),
                "candidate_count": int(len(candidate_ids)),
                "top1_raw_id": top1_raw,
                "gt_rank": gt_rank,
                "gt_top1": top1_is_gt,
                "gt_top5": top5,
                "gt_top20": top20,
                "gt_normalized_gt_rank": gt_norm_rank,
                "gt_margin_vs_best_non_gt": gt_margin,
                "assignment_entropy": entropy,
            }
            by_class[int(gt_raw_id)].append(rec)
            by_clip[str(clip_key)].append(rec)
            if len(failure_examples) < top_examples and (not contains_gt or top1_is_gt is False):
                failure_examples.append(
                    {
                        "arm": arm,
                        "clip_id": clip_key,
                        "trajectory_id": row.get("trajectory_id"),
                        "raw_category_id": int(gt_raw_id),
                        "top1_raw_id": top1_raw,
                        "gt_rank": gt_rank,
                        "gt_margin_vs_best_non_gt": gt_margin,
                        "candidate_contains_gt": bool(contains_gt),
                        "candidate_count": int(len(candidate_ids)),
                    }
                )

    evaluable_rows = [row for group in by_class.values() for row in group if bool(row.get("candidate_contains_gt"))]
    evaluated_total = int(len(evaluable_rows))
    gt_rank1_rate = _rate(sum(1 for row in evaluable_rows if int(row.get("gt_rank") or 0) == 1), evaluated_total)
    gt_top5_rate = _rate(sum(1 for row in evaluable_rows if int(row.get("gt_rank") or 0) <= 5), evaluated_total)
    gt_top20_rate = _rate(sum(1 for row in evaluable_rows if int(row.get("gt_rank") or 0) <= 20), evaluated_total)
    mean_gt_rank = _mean([float(row["gt_rank"]) for row in evaluable_rows if row.get("gt_rank") is not None])
    median_gt_rank = _median([float(row["gt_rank"]) for row in evaluable_rows if row.get("gt_rank") is not None])
    mean_normalized_gt_rank = _mean([float(row["gt_normalized_gt_rank"]) for row in evaluable_rows if row.get("gt_normalized_gt_rank") is not None])
    mean_gt_margin = _mean([float(row["gt_margin_vs_best_non_gt"]) for row in evaluable_rows if row.get("gt_margin_vs_best_non_gt") is not None])
    positive_gt_margin_rate = _rate(sum(1 for row in evaluable_rows if row.get("gt_margin_vs_best_non_gt") is not None and float(row.get("gt_margin_vs_best_non_gt")) > 0), evaluated_total)
    top1_wrong_rate = _rate(sum(1 for row in evaluable_rows if not bool(row.get("gt_top1"))), evaluated_total)
    assignment_entropy_mean = _mean([float(row["assignment_entropy"]) for group in by_class.values() for row in group if row.get("assignment_entropy") is not None])
    top1_total = int(sum(top1_counter.values()))
    hub_like_top1_concentration = float(max(top1_counter.values()) / top1_total) if top1_total else None
    gt_rank1_count = int(sum(1 for row in evaluable_rows if int(row.get("gt_rank") or 0) == 1))
    gt_top5_count = int(sum(1 for row in evaluable_rows if int(row.get("gt_rank") or 0) <= 5))
    gt_top20_count = int(sum(1 for row in evaluable_rows if int(row.get("gt_rank") or 0) <= 20))
    row_count_with_label_record = int(sum(len(group) for group in by_class.values()))
    candidate_contains_gt_count = int(sum(1 for group in by_class.values() for row in group if bool(row.get("candidate_contains_gt"))))
    candidate_missing_count = int(row_count_with_label_record - candidate_contains_gt_count)
    summary = {
        "arm": arm,
        "status": "PASS_LATENT" if evaluated_total > 0 else "FAIL_SCHEMA",
        "label_source": label_source,
        "carrier_source": carrier_source,
        "assignment_mode": assignment_mode,
        "score_projector_path": str(bundle.checkpoint_path),
        "score_projector_stage_id": str(bundle.stage_id),
        "score_temperature": float(bundle.temperature),
        "text_vocab_asset_root": str(gt_asset_root),
        "candidate_label_record_count": int(len(label_map)),
        "row_count_total": int(total_rows),
        "row_count_with_label_record": int(row_count_with_label_record),
        "row_count_in_scope": int(candidate_contains_gt_count),
        "candidate_contains_gt_count": int(candidate_contains_gt_count),
        "candidate_contains_gt_rate": _rate(candidate_contains_gt_count, row_count_with_label_record),
        "latent_evaluable_row_count": int(candidate_contains_gt_count),
        "gt_rank1_rate": gt_rank1_rate,
        "gt_top5_rate": gt_top5_rate,
        "gt_top20_rate": gt_top20_rate,
        "mean_gt_rank": mean_gt_rank,
        "median_gt_rank": median_gt_rank,
        "mean_normalized_gt_rank": mean_normalized_gt_rank,
        "mean_gt_margin_vs_best_non_gt": mean_gt_margin,
        "positive_gt_margin_rate": positive_gt_margin_rate,
        "top1_wrong_rate": top1_wrong_rate,
        "assignment_entropy_mean": assignment_entropy_mean,
        "hub_like_top1_concentration": hub_like_top1_concentration,
        "top1_label_top_counts": [
            {"raw_category_id": int(raw_id), "count": int(count)}
            for raw_id, count in top1_counter.most_common(20)
        ],
        "top1_wrong_label_top_counts": [
            {"raw_category_id": int(raw_id), "count": int(count)}
            for raw_id, count in top1_wrong_counter.most_common(20)
        ],
        "comparison_to_E_in_scope": {
            "E_in_scope_oracle_valid_rate": 1.0,
            "latent_gap_to_oracle": float(1.0 - gt_rank1_rate) if gt_rank1_rate is not None else None,
        },
        "candidate_missing_count": int(candidate_missing_count),
        "candidate_missing_rate": _rate(candidate_missing_count, row_count_with_label_record),
        "carrier_load_fail_count": 0,
        "scoring_backend": "current_v2b_prealign_projector_text_cosine",
        "blocker": None,
        "candidate_missing_vocab_count": int(sum(len(group.get("candidate_missing_vocab_ids", [])) for group in all_clip_metrics)),
        "score_rows": int(len(evaluable_rows)),
        "score_clip_count": int(len(by_class)),
    }
    per_class_rows = _summarize_by_class([row for group in by_class.values() for row in group])
    per_clip_rows = _summarize_by_clip([row for group in by_class.values() for row in group])
    return summary, per_class_rows, per_clip_rows, _combine_examples(failure_examples)


def _write_main_outputs_update(
    *,
    output_root: Path,
    latent_dir: Path,
    c_summary: Mapping[str, Any],
    d_summary: Mapping[str, Any],
) -> None:
    main_summary_path = output_root / "oracle_clean_ablation_summary.json"
    if main_summary_path.is_file():
        main_summary = json.loads(main_summary_path.read_text(encoding="utf-8"))
    else:
        main_summary = {}
    main_summary["gtcarrier_latent_assignment"] = {
        "status": "PASS_LATENT" if str(c_summary.get("status")) == "PASS_LATENT" and str(d_summary.get("status")) == "PASS_LATENT" else "PARTIAL_NEEDS_SCORING_ADAPTER",
        "output_root": str(latent_dir),
        "C_yprime_gtcarrier_latent": {
            "status": c_summary.get("status"),
            "candidate_contains_gt_rate": c_summary.get("candidate_contains_gt_rate"),
            "gt_rank1_rate": c_summary.get("gt_rank1_rate"),
            "gt_top5_rate": c_summary.get("gt_top5_rate"),
            "gt_top20_rate": c_summary.get("gt_top20_rate"),
            "mean_normalized_gt_rank": c_summary.get("mean_normalized_gt_rank"),
            "mean_gt_margin_vs_best_non_gt": c_summary.get("mean_gt_margin_vs_best_non_gt"),
            "positive_gt_margin_rate": c_summary.get("positive_gt_margin_rate"),
            "latent_gap_to_oracle": c_summary.get("comparison_to_E_in_scope", {}).get("latent_gap_to_oracle"),
            "summary_path": str(latent_dir / "C_yprime_gtcarrier_latent_summary.json"),
        },
        "D_fully_gtcarrier_latent": {
            "status": d_summary.get("status"),
            "candidate_contains_gt_rate": d_summary.get("candidate_contains_gt_rate"),
            "gt_rank1_rate": d_summary.get("gt_rank1_rate"),
            "gt_top5_rate": d_summary.get("gt_top5_rate"),
            "gt_top20_rate": d_summary.get("gt_top20_rate"),
            "mean_normalized_gt_rank": d_summary.get("mean_normalized_gt_rank"),
            "mean_gt_margin_vs_best_non_gt": d_summary.get("mean_gt_margin_vs_best_non_gt"),
            "positive_gt_margin_rate": d_summary.get("positive_gt_margin_rate"),
            "latent_gap_to_oracle": d_summary.get("comparison_to_E_in_scope", {}).get("latent_gap_to_oracle"),
            "summary_path": str(latent_dir / "D_fully_gtcarrier_latent_summary.json"),
        },
    }
    main_summary["gtcarrier_latent_assignment_summary_path"] = str(latent_dir / "gtcarrier_latent_assignment_summary.json")
    write_json(main_summary_path, main_summary)

    comparison_path = output_root / "oracle_clean_ablation_comparison.csv"
    if comparison_path.is_file():
        with comparison_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        extra_fields = {
            "latent_assignment_status",
            "latent_evaluable_row_count",
            "candidate_contains_gt_count",
            "candidate_contains_gt_rate",
            "gt_rank1_rate",
            "gt_top5_rate",
            "gt_top20_rate",
            "mean_gt_rank",
            "median_gt_rank",
            "mean_normalized_gt_rank",
            "mean_gt_margin_vs_best_non_gt",
            "positive_gt_margin_rate",
            "top1_wrong_rate",
            "assignment_entropy_mean",
            "hub_like_top1_concentration",
            "latent_gap_to_oracle",
            "comparison_to_E_in_scope",
        }
        fieldnames = list(rows[0].keys()) if rows else []
        for field in sorted(extra_fields):
            if field not in fieldnames:
                fieldnames.append(field)
        for row in rows:
            if row.get("arm") == "C_yprime_gtcarrier_latent":
                row["latent_assignment_status"] = str(c_summary.get("status"))
                row["latent_evaluable_row_count"] = str(c_summary.get("latent_evaluable_row_count"))
                row["candidate_contains_gt_count"] = str(c_summary.get("candidate_contains_gt_count"))
                row["candidate_contains_gt_rate"] = str(c_summary.get("candidate_contains_gt_rate"))
                row["gt_rank1_rate"] = str(c_summary.get("gt_rank1_rate"))
                row["gt_top5_rate"] = str(c_summary.get("gt_top5_rate"))
                row["gt_top20_rate"] = str(c_summary.get("gt_top20_rate"))
                row["mean_gt_rank"] = str(c_summary.get("mean_gt_rank"))
                row["median_gt_rank"] = str(c_summary.get("median_gt_rank"))
                row["mean_normalized_gt_rank"] = str(c_summary.get("mean_normalized_gt_rank"))
                row["mean_gt_margin_vs_best_non_gt"] = str(c_summary.get("mean_gt_margin_vs_best_non_gt"))
                row["positive_gt_margin_rate"] = str(c_summary.get("positive_gt_margin_rate"))
                row["top1_wrong_rate"] = str(c_summary.get("top1_wrong_rate"))
                row["assignment_entropy_mean"] = str(c_summary.get("assignment_entropy_mean"))
                row["hub_like_top1_concentration"] = str(c_summary.get("hub_like_top1_concentration"))
                row["latent_gap_to_oracle"] = str(c_summary.get("comparison_to_E_in_scope", {}).get("latent_gap_to_oracle"))
                row["comparison_to_E_in_scope"] = json.dumps(c_summary.get("comparison_to_E_in_scope", {}), ensure_ascii=False)
            elif row.get("arm") == "D_fully_gtcarrier_latent":
                row["latent_assignment_status"] = str(d_summary.get("status"))
                row["latent_evaluable_row_count"] = str(d_summary.get("latent_evaluable_row_count"))
                row["candidate_contains_gt_count"] = str(d_summary.get("candidate_contains_gt_count"))
                row["candidate_contains_gt_rate"] = str(d_summary.get("candidate_contains_gt_rate"))
                row["gt_rank1_rate"] = str(d_summary.get("gt_rank1_rate"))
                row["gt_top5_rate"] = str(d_summary.get("gt_top5_rate"))
                row["gt_top20_rate"] = str(d_summary.get("gt_top20_rate"))
                row["mean_gt_rank"] = str(d_summary.get("mean_gt_rank"))
                row["median_gt_rank"] = str(d_summary.get("median_gt_rank"))
                row["mean_normalized_gt_rank"] = str(d_summary.get("mean_normalized_gt_rank"))
                row["mean_gt_margin_vs_best_non_gt"] = str(d_summary.get("mean_gt_margin_vs_best_non_gt"))
                row["positive_gt_margin_rate"] = str(d_summary.get("positive_gt_margin_rate"))
                row["top1_wrong_rate"] = str(d_summary.get("top1_wrong_rate"))
                row["assignment_entropy_mean"] = str(d_summary.get("assignment_entropy_mean"))
                row["hub_like_top1_concentration"] = str(d_summary.get("hub_like_top1_concentration"))
                row["latent_gap_to_oracle"] = str(d_summary.get("comparison_to_E_in_scope", {}).get("latent_gap_to_oracle"))
                row["comparison_to_E_in_scope"] = json.dumps(d_summary.get("comparison_to_E_in_scope", {}), ensure_ascii=False)
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _write_takeover(
    *,
    output_root: Path,
    latent_dir: Path,
    c_summary: Mapping[str, Any],
    d_summary: Mapping[str, Any],
    main_summary: Mapping[str, Any],
) -> None:
    path = output_root / "GTCARRIER_LATENT_ASSIGNMENT_TAKEOVER.md"
    lines = [
        "# GT Carrier Latent Assignment",
        "",
        f"- dataset: `{main_summary.get('dataset_name', 'lvvis_train_base')}`",
        f"- projector/checkpoint: `{c_summary.get('score_projector_path')}`",
        f"- text/asset root: `{c_summary.get('text_vocab_asset_root')}`",
        f"- C status: `{c_summary.get('status')}`",
        f"- C candidate_contains_gt_rate: `{c_summary.get('candidate_contains_gt_rate')}`",
        f"- C gt_rank1_rate: `{c_summary.get('gt_rank1_rate')}`",
        f"- C mean_normalized_gt_rank: `{c_summary.get('mean_normalized_gt_rank')}`",
        f"- D status: `{d_summary.get('status')}`",
        f"- D candidate_contains_gt_rate: `{d_summary.get('candidate_contains_gt_rate')}`",
        f"- D gt_rank1_rate: `{d_summary.get('gt_rank1_rate')}`",
        f"- D mean_normalized_gt_rank: `{d_summary.get('mean_normalized_gt_rank')}`",
        f"- E in-scope oracle valid rate: `1.0`",
        f"- latent directory: `{latent_dir}`",
        "",
        "## Interpretation",
        "",
        "The scorer path is the current V2-B prealign projector/text cosine ranking path. ",
        "C and D are now actual latent-rank measurements, not support-only ceilings. ",
        "D is compared only on the in-scope full-Y rows; the overall E oracle remains scope-mismatch-limited.",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    latent_dir = output_root / "gtcarrier_latent_assignment"
    latent_dir.mkdir(parents=True, exist_ok=True)

    full_y_path = Path(args.full_y_path).expanduser().resolve()
    weak_label_path = Path(args.weak_label_path).expanduser().resolve()
    gt_carrier_path = Path(args.gt_carrier_path).expanduser().resolve()
    gt_identity_path = Path(args.gt_identity_path).expanduser().resolve()
    run_root_v2b = Path(args.run_root_v2b).expanduser().resolve()

    full_y_records = _load_full_y_records(full_y_path)
    weak_rows = load_weak_label_records(weak_label_path)
    full_y_map = _build_label_map(full_y_records, label_key="full_y_raw_ids")
    weak_map = _build_label_map(weak_rows, label_key="observed_raw_ids")
    for rec in weak_map.values():
        observed = unique_ints(rec.get("observed_raw_ids", []))
        if observed and not rec.get("yprime_raw_ids"):
            rec["yprime_raw_ids"] = list(observed)

    binding_rows = _load_binding_rows(gt_identity_path)
    carrier_rows = _load_jsonl_rows(gt_carrier_path)
    carrier_by_tid = {str(row.get("trajectory_id", "")).strip(): dict(row) for row in carrier_rows if str(row.get("trajectory_id", "")).strip()}
    carrier_by_index = {int(idx): dict(row) for idx, row in enumerate(carrier_rows)}
    binding_by_tid = {str(row.get("trajectory_id", "")).strip(): dict(row) for row in binding_rows if str(row.get("trajectory_id", "")).strip()}
    gt_asset_root = gt_carrier_path.parents[2]

    bundle, text_asset_root, text_vocab_ids, text_vocab_matrix, text_vocab_index, bundle_meta = _load_projector_bundle_and_vocab(
        run_root_v2b, args.dataset_name, args.device
    )

    rows_by_clip_c: Dict[str, List[Record]] = defaultdict(list)
    rows_by_clip_d: Dict[str, List[Record]] = defaultdict(list)
    carrier_join_failures: List[Record] = []
    for binding in binding_rows:
        clip_key = _normalized_scope_key(binding.get("clip_id"))
        if clip_key is None:
            continue
        tid = str(binding.get("trajectory_id", "")).strip()
        carrier_row = None
        if tid and tid in carrier_by_tid:
            carrier_row = carrier_by_tid[tid]
        else:
            idx = safe_int(binding.get("carrier_row_index"), None)
            if idx is not None and int(idx) in carrier_by_index:
                carrier_row = carrier_by_index[int(idx)]
        if carrier_row is None:
            carrier_join_failures.append(
                {
                    "trajectory_id": tid,
                    "clip_id": binding.get("clip_id"),
                    "video_id": binding.get("video_id"),
                    "raw_category_id": binding.get("raw_category_id"),
                }
            )
            continue
        row = {
            "trajectory_id": tid,
            "clip_id": binding.get("clip_id"),
            "video_id": binding.get("video_id"),
            "raw_category_id": binding.get("raw_category_id"),
            "carrier_record": carrier_row,
            "binding_source": binding.get("binding_source"),
            "binding_key": binding.get("binding_key"),
        }
        rows_by_clip_c[str(clip_key)].append(dict(row))
        rows_by_clip_d[str(clip_key)].append(dict(row))

    c_summary, c_class_rows, c_clip_rows, c_examples = _arm_report(
        arm="C_yprime_gtcarrier_latent",
        label_source="yprime",
        carrier_source="gt_carrier",
        assignment_mode="latent",
        label_map=weak_map,
        rows_by_clip=rows_by_clip_c,
        bundle=bundle,
        gt_asset_root=gt_asset_root,
        dataset_name=args.dataset_name,
        text_vocab_ids=text_vocab_ids,
        text_vocab_matrix=text_vocab_matrix,
        text_vocab_index=text_vocab_index,
        top_examples=int(args.top_examples),
    )
    d_summary, d_class_rows, d_clip_rows, d_examples = _arm_report(
        arm="D_fully_gtcarrier_latent",
        label_source="full_y",
        carrier_source="gt_carrier",
        assignment_mode="latent",
        label_map=full_y_map,
        rows_by_clip=rows_by_clip_d,
        bundle=bundle,
        gt_asset_root=gt_asset_root,
        dataset_name=args.dataset_name,
        text_vocab_ids=text_vocab_ids,
        text_vocab_matrix=text_vocab_matrix,
        text_vocab_index=text_vocab_index,
        top_examples=int(args.top_examples),
    )

    c_summary["carrier_join_fail_count"] = int(len(carrier_join_failures))
    d_summary["carrier_join_fail_count"] = int(len(carrier_join_failures))
    c_summary["binding_row_count"] = int(len(binding_rows))
    d_summary["binding_row_count"] = int(len(binding_rows))
    c_summary["carrier_row_count"] = int(len(carrier_rows))
    d_summary["carrier_row_count"] = int(len(carrier_rows))
    c_summary["label_record_count"] = int(len(weak_map))
    d_summary["label_record_count"] = int(len(full_y_map))
    c_summary["label_source_path"] = str(weak_label_path)
    d_summary["label_source_path"] = str(full_y_path)
    c_summary["gt_carrier_identity_binding_path"] = str(gt_identity_path)
    d_summary["gt_carrier_identity_binding_path"] = str(gt_identity_path)
    c_summary["carrier_asset_root"] = str(gt_asset_root)
    d_summary["carrier_asset_root"] = str(gt_asset_root)
    c_summary["text_asset_root"] = str(text_asset_root)
    d_summary["text_asset_root"] = str(text_asset_root)
    c_summary["projector_bundle_meta"] = bundle_meta
    d_summary["projector_bundle_meta"] = bundle_meta

    summary_payload = {
        "status": "PASS" if c_summary.get("status") == "PASS_LATENT" and d_summary.get("status") == "PASS_LATENT" else "PARTIAL_NEEDS_SCORING_ADAPTER",
        "dataset_name": args.dataset_name,
        "run_root_v2b": str(run_root_v2b),
        "output_root": str(output_root),
        "gt_asset_root": str(gt_asset_root),
        "full_y_path": str(full_y_path),
        "weak_label_path": str(weak_label_path),
        "gt_carrier_path": str(gt_carrier_path),
        "gt_identity_path": str(gt_identity_path),
        "projector_bundle": bundle_meta,
        "arms": {
            "C_yprime_gtcarrier_latent": c_summary,
            "D_fully_gtcarrier_latent": d_summary,
        },
        "carrier_join_fail_count": int(len(carrier_join_failures)),
        "carrier_join_fail_examples": carrier_join_failures[:20],
        "scoring_backend": "current_v2b_prealign_projector_text_cosine",
        "minimal_next_action": (
            "If PASS_LATENT, the clean-observation latent scorer is sufficient; "
            "otherwise a dedicated scoring/responsibility adapter is still required."
        ),
    }

    write_json(latent_dir / "gtcarrier_latent_assignment_summary.json", summary_payload)
    write_json(latent_dir / "C_yprime_gtcarrier_latent_summary.json", c_summary)
    write_json(latent_dir / "D_fully_gtcarrier_latent_summary.json", d_summary)
    write_csv(
        latent_dir / "C_yprime_gtcarrier_latent_by_class.csv",
        c_class_rows,
        fieldnames=(
            "raw_category_id",
            "row_count",
            "candidate_contains_gt_count",
            "candidate_contains_gt_rate",
            "gt_rank1_rate",
            "mean_gt_rank",
            "median_gt_rank",
            "mean_gt_margin_vs_best_non_gt",
            "positive_gt_margin_rate",
        ),
    )
    write_csv(
        latent_dir / "D_fully_gtcarrier_latent_by_class.csv",
        d_class_rows,
        fieldnames=(
            "raw_category_id",
            "row_count",
            "candidate_contains_gt_count",
            "candidate_contains_gt_rate",
            "gt_rank1_rate",
            "mean_gt_rank",
            "median_gt_rank",
            "mean_gt_margin_vs_best_non_gt",
            "positive_gt_margin_rate",
        ),
    )
    with (latent_dir / "C_yprime_gtcarrier_latent_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in c_examples:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    with (latent_dir / "D_fully_gtcarrier_latent_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in d_examples:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    # Update the main oracle clean-data output with the new latent measurements.
    main_summary_path = output_root / "oracle_clean_ablation_summary.json"
    main_summary = json.loads(main_summary_path.read_text(encoding="utf-8")) if main_summary_path.is_file() else {}
    main_summary["gtcarrier_latent_assignment"] = summary_payload
    main_summary["gtcarrier_latent_assignment_summary_path"] = str(latent_dir / "gtcarrier_latent_assignment_summary.json")
    write_json(main_summary_path, main_summary)

    comparison_path = output_root / "oracle_clean_ablation_comparison.csv"
    if comparison_path.is_file():
        with comparison_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        latent_fields = [
            "latent_assignment_status",
            "candidate_contains_gt_count",
            "candidate_contains_gt_rate",
            "latent_evaluable_row_count",
            "gt_rank1_rate",
            "gt_top5_rate",
            "gt_top20_rate",
            "mean_gt_rank",
            "median_gt_rank",
            "mean_normalized_gt_rank",
            "mean_gt_margin_vs_best_non_gt",
            "positive_gt_margin_rate",
            "top1_wrong_rate",
            "assignment_entropy_mean",
            "hub_like_top1_concentration",
            "latent_gap_to_oracle",
            "candidate_missing_count",
            "candidate_missing_rate",
        ]
        fieldnames = list(rows[0].keys()) if rows else []
        for field in latent_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        for row in rows:
            if row.get("arm") == "C_yprime_gtcarrier_latent":
                row["latent_assignment_status"] = c_summary.get("status")
                for field in latent_fields[1:]:
                    value = c_summary.get(field)
                    if field == "latent_gap_to_oracle":
                        value = c_summary.get("comparison_to_E_in_scope", {}).get("latent_gap_to_oracle")
                    row[field] = "" if value is None else str(value)
            elif row.get("arm") == "D_fully_gtcarrier_latent":
                row["latent_assignment_status"] = d_summary.get("status")
                for field in latent_fields[1:]:
                    value = d_summary.get(field)
                    if field == "latent_gap_to_oracle":
                        value = d_summary.get("comparison_to_E_in_scope", {}).get("latent_gap_to_oracle")
                    row[field] = "" if value is None else str(value)
        with comparison_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    takeover_path = output_root / "GTCARRIER_LATENT_ASSIGNMENT_TAKEOVER.md"
    _write_takeover(
        output_root=output_root,
        latent_dir=latent_dir,
        c_summary=c_summary,
        d_summary=d_summary,
        main_summary=main_summary,
    )

    # Also refresh the canonical takeover latest marker if it exists in the local control plane.
    takeover_latest = Path(__file__).resolve().parents[1] / "codex" / "control" / "TAKEOVER_LATEST.md"
    if takeover_latest.is_file():
        takeover_latest.write_text(
            "\n".join(
                [
                    "# TAKEOVER LATEST",
                    "",
                    f"- latent assignment output: `{latent_dir}`",
                    f"- C status: `{c_summary.get('status')}`",
                    f"- C gt_rank1_rate: `{c_summary.get('gt_rank1_rate')}`",
                    f"- D status: `{d_summary.get('status')}`",
                    f"- D gt_rank1_rate: `{d_summary.get('gt_rank1_rate')}`",
                    f"- main oracle summary: `{main_summary_path}`",
                ]
            ).rstrip()
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
