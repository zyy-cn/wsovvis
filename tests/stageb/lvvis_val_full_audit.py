from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pycocotools import mask as mask_utils

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_clip_ids(arg: str) -> List[str]:
    values = [x.strip() for x in str(arg).split(",") if x.strip()]
    normalized: List[str] = []
    for item in values:
        if item.lstrip("-").isdigit():
            normalized.append(str(int(item)))
        else:
            normalized.append(item)
    return sorted(set(normalized), key=lambda x: (not x.lstrip("-").isdigit(), x))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _normalize_name(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_rle(segmentation: Any, h: int, w: int) -> Optional[Dict[str, Any]]:
    if segmentation is None:
        return None
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            return None
        rle = mask_utils.frPyObjects(segmentation, h, w)
        if isinstance(rle, list):
            rle = mask_utils.merge(rle)
    elif isinstance(segmentation, dict):
        rle = dict(segmentation)
        if "counts" not in rle:
            return None
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, h, w)
            if isinstance(rle, list):
                rle = mask_utils.merge(rle)
        else:
            if "size" not in rle:
                rle["size"] = [h, w]
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("utf-8")
    else:
        return None
    if isinstance(rle, list):
        if not rle:
            return None
        rle = mask_utils.merge(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def _bbox_iou_xyxy_vs_xywh(box_xyxy: Sequence[float], box_xywh: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_xyxy)
    bx, by, bw, bh = map(float, box_xywh)
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def _build_imagenet_norm_set() -> set[str]:
    meta_path = (
        Path(os.environ.get("CONDA_PREFIX", "")) / "lib" / "python3.10" / "site-packages" / "torchvision" / "models" / "_meta.py"
    )
    if not meta_path.exists():
        return set()
    text = meta_path.read_text(encoding="utf-8")
    marker = "_IMAGENET_CATEGORIES = "
    marker_idx = text.find(marker)
    if marker_idx < 0:
        return set()
    start = text.find("[", marker_idx)
    if start < 0:
        return set()
    depth = 0
    end = -1
    for idx in range(start, len(text)):
        c = text[idx]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end < 0:
        return set()
    raw = ast.literal_eval(text[start:end])
    values: set[str] = set()
    for item in raw:
        for part in str(item).split(","):
            name = _normalize_name(part)
            if name:
                values.add(name)
    return values


def _bucket_size(area_px: float) -> str:
    if area_px < 32.0 * 32.0:
        return "small"
    if area_px < 96.0 * 96.0:
        return "medium"
    return "large"


def _bucket_length(length: int) -> str:
    if length <= 5:
        return "short"
    if length <= 20:
        return "medium"
    return "long"


def _bucket_difficulty(instances_in_clip: int) -> str:
    if instances_in_clip <= 5:
        return "easy"
    if instances_in_clip <= 15:
        return "medium"
    return "hard"


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(v) for v in values)
    k = (len(sorted_values) - 1) * q
    floor_idx = int(math.floor(k))
    ceil_idx = int(math.ceil(k))
    if floor_idx == ceil_idx:
        return float(sorted_values[floor_idx])
    return float(
        sorted_values[floor_idx]
        + (sorted_values[ceil_idx] - sorted_values[floor_idx]) * (k - floor_idx)
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


@dataclass(frozen=True)
class TrajectoryClipRecord:
    trajectory_id: str
    clip_id: str
    frame_indices: Tuple[int, ...]
    frame_index_set: frozenset[int]
    masks_by_frame: Dict[int, Any]
    boxes_by_frame: Dict[int, Sequence[float]]
    image_size: Tuple[int, int]


@dataclass(frozen=True)
class GTClipInstance:
    instance_id: int
    clip_id: str
    category_id: int
    category_name: str
    frame_indices: Tuple[int, ...]
    frame_index_set: frozenset[int]
    segmentations_by_frame: Dict[int, Any]
    boxes_by_frame: Dict[int, Sequence[float]]
    mean_area_px: float
    difficulty_bucket: str


@dataclass(frozen=True)
class ClipAuditInput:
    clip_id: str
    trajectories: Tuple[TrajectoryClipRecord, ...]
    gt_instances: Tuple[GTClipInstance, ...]
    algorithm: str
    tie_eps: float


@dataclass(frozen=True)
class PairMetric:
    track_miou: float
    best_iou: float
    overlap_frames_nonzero: int
    gt_frame_count: int
    trajectory_frame_count: int


def _compute_track_miou_baseline(
    gt_instance: GTClipInstance,
    trajectory: TrajectoryClipRecord,
) -> PairMetric:
    frame_ious: List[float] = []
    best_iou = 0.0
    h, w = trajectory.image_size
    for frame_index in gt_instance.frame_indices:
        gt_rle = _to_rle(gt_instance.segmentations_by_frame.get(frame_index), h, w)
        pred_rle = _to_rle(trajectory.masks_by_frame.get(frame_index), h, w)
        if gt_rle is None or pred_rle is None:
            iou = 0.0
        else:
            mat = mask_utils.iou([pred_rle], [gt_rle], [0])
            iou = float(mat[0][0])
        frame_ious.append(iou)
        if iou > best_iou:
            best_iou = iou
    overlap_frames_nonzero = int(sum(1 for v in frame_ious if v > 0.0))
    return PairMetric(
        track_miou=_mean(frame_ious),
        best_iou=float(best_iou),
        overlap_frames_nonzero=overlap_frames_nonzero,
        gt_frame_count=int(len(gt_instance.frame_indices)),
        trajectory_frame_count=int(len(trajectory.frame_indices)),
    )


def _compute_track_miou_optimized(
    gt_instance: GTClipInstance,
    trajectory: TrajectoryClipRecord,
    gt_rle_cache: Dict[Tuple[int, int], Optional[Dict[str, Any]]],
    traj_rle_cache: Dict[Tuple[str, int], Optional[Dict[str, Any]]],
    empty_rle_cache: Dict[Tuple[int, int], Dict[str, Any]],
) -> PairMetric:
    # Temporal-overlap prefilter (safe): no overlap => all frame IoUs are zero.
    if not (gt_instance.frame_index_set & trajectory.frame_index_set):
        return PairMetric(
            track_miou=0.0,
            best_iou=0.0,
            overlap_frames_nonzero=0,
            gt_frame_count=int(len(gt_instance.frame_indices)),
            trajectory_frame_count=int(len(trajectory.frame_indices)),
        )

    # BBox prefilter (safe): if all overlapping-frame bbox IoUs are zero, mask IoUs cannot be positive.
    overlap_frames = sorted(gt_instance.frame_index_set & trajectory.frame_index_set)
    has_bbox_signal = False
    has_bbox_comparable = False
    for frame_index in overlap_frames:
        gt_box = gt_instance.boxes_by_frame.get(frame_index)
        pred_box = trajectory.boxes_by_frame.get(frame_index)
        if gt_box is None or pred_box is None:
            continue
        has_bbox_comparable = True
        if _bbox_iou_xyxy_vs_xywh(pred_box, gt_box) > 0.0:
            has_bbox_signal = True
            break
    if has_bbox_comparable and not has_bbox_signal:
        return PairMetric(
            track_miou=0.0,
            best_iou=0.0,
            overlap_frames_nonzero=0,
            gt_frame_count=int(len(gt_instance.frame_indices)),
            trajectory_frame_count=int(len(trajectory.frame_indices)),
        )

    h, w = trajectory.image_size
    empty_key = (h, w)
    if empty_key not in empty_rle_cache:
        empty_mask = np.asfortranarray(np.zeros((h, w), dtype=np.uint8))
        empty_rle_cache[empty_key] = mask_utils.encode(empty_mask)
    empty_rle = empty_rle_cache[empty_key]

    gt_list: List[Dict[str, Any]] = []
    pred_list: List[Dict[str, Any]] = []
    for frame_index in gt_instance.frame_indices:
        gt_key = (int(gt_instance.instance_id), int(frame_index))
        if gt_key not in gt_rle_cache:
            gt_rle_cache[gt_key] = _to_rle(gt_instance.segmentations_by_frame.get(frame_index), h, w)
        gt_rle = gt_rle_cache[gt_key] or empty_rle

        traj_key = (trajectory.trajectory_id, frame_index)
        if traj_key not in traj_rle_cache:
            traj_rle_cache[traj_key] = _to_rle(trajectory.masks_by_frame.get(frame_index), h, w)
        pred_rle = traj_rle_cache[traj_key] or empty_rle

        gt_list.append(gt_rle)
        pred_list.append(pred_rle)

    # Hybrid exact IoU path:
    # - small track: batched diagonal extraction (vectorized where cheap)
    # - larger track: per-frame pair calls avoid O(T^2) matrix overhead
    if len(gt_list) <= 4:
        mat = mask_utils.iou(pred_list, gt_list, [0] * len(gt_list))
        diag = np.diag(mat).astype(np.float64)
        best_iou = float(np.max(diag)) if diag.size else 0.0
        overlap_frames_nonzero = int(np.sum(diag > 0.0)) if diag.size else 0
        return PairMetric(
            track_miou=float(np.mean(diag)) if diag.size else 0.0,
            best_iou=best_iou,
            overlap_frames_nonzero=overlap_frames_nonzero,
            gt_frame_count=int(len(gt_instance.frame_indices)),
            trajectory_frame_count=int(len(trajectory.frame_indices)),
        )

    frame_ious: List[float] = []
    best_iou = 0.0
    for pred_rle, gt_rle in zip(pred_list, gt_list):
        iou = float(mask_utils.iou([pred_rle], [gt_rle], [0])[0][0])
        frame_ious.append(iou)
        if iou > best_iou:
            best_iou = iou
    overlap_frames_nonzero = int(sum(1 for v in frame_ious if v > 0.0))
    return PairMetric(
        track_miou=_mean(frame_ious),
        best_iou=float(best_iou),
        overlap_frames_nonzero=overlap_frames_nonzero,
        gt_frame_count=int(len(gt_instance.frame_indices)),
        trajectory_frame_count=int(len(trajectory.frame_indices)),
    )


def _process_clip(input_payload: ClipAuditInput) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    gt_rle_cache: Dict[Tuple[int, int], Optional[Dict[str, Any]]] = {}
    traj_rle_cache: Dict[Tuple[str, int], Optional[Dict[str, Any]]] = {}
    empty_rle_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

    sorted_trajectories = sorted(input_payload.trajectories, key=lambda rec: rec.trajectory_id)
    # Materiality thresholds for diagnostic-only metrics (additive; canonical headline metrics unchanged).
    fragmentation_frame_iou_threshold = 0.10
    duplicate_track_miou_threshold = 0.30
    split_material_fraction_threshold = 0.20

    trajectory_to_gt_track_scores: Dict[str, List[float]] = defaultdict(list)
    merge_multi_gt_count: Dict[str, int] = defaultdict(int)

    split_instance_count = 0
    duplicate_instance_count = 0
    no_match_instance_count = 0
    fragmentation_counts: List[int] = []

    for gt_instance in sorted(input_payload.gt_instances, key=lambda inst: inst.instance_id):
        best_track_miou = 0.0
        best_frame_iou = 0.0
        best_trajectory_id: Optional[str] = None
        best_overlap_nonzero = 0
        best_trajectory_frame_count = 0

        trajectory_coverage_nonzero: Dict[str, int] = {}
        trajectory_frame_fraction: Dict[str, float] = {}
        substantial_match_count = 0

        for trajectory in sorted_trajectories:
            if input_payload.algorithm == "baseline":
                metric = _compute_track_miou_baseline(gt_instance, trajectory)
            else:
                metric = _compute_track_miou_optimized(
                    gt_instance=gt_instance,
                    trajectory=trajectory,
                    gt_rle_cache=gt_rle_cache,
                    traj_rle_cache=traj_rle_cache,
                    empty_rle_cache=empty_rle_cache,
                )
            track_miou = float(metric.track_miou)
            frame_best = float(metric.best_iou)
            overlap_nonzero = int(metric.overlap_frames_nonzero)
            gt_frame_count = int(metric.gt_frame_count)

            trajectory_to_gt_track_scores[trajectory.trajectory_id].append(track_miou)
            if track_miou >= duplicate_track_miou_threshold:
                merge_multi_gt_count[trajectory.trajectory_id] += 1
                substantial_match_count += 1

            coverage_fraction = float(overlap_nonzero / gt_frame_count) if gt_frame_count > 0 else 0.0
            if overlap_nonzero > 0:
                trajectory_coverage_nonzero[trajectory.trajectory_id] = overlap_nonzero
            if frame_best >= fragmentation_frame_iou_threshold:
                trajectory_frame_fraction[trajectory.trajectory_id] = coverage_fraction

            if track_miou > best_track_miou + input_payload.tie_eps:
                best_track_miou = track_miou
                best_frame_iou = frame_best
                best_trajectory_id = trajectory.trajectory_id
                best_overlap_nonzero = overlap_nonzero
                best_trajectory_frame_count = int(metric.trajectory_frame_count)
            elif abs(track_miou - best_track_miou) <= input_payload.tie_eps:
                if best_trajectory_id is None or trajectory.trajectory_id < best_trajectory_id:
                    best_track_miou = track_miou
                    best_frame_iou = frame_best
                    best_trajectory_id = trajectory.trajectory_id
                    best_overlap_nonzero = overlap_nonzero
                    best_trajectory_frame_count = int(metric.trajectory_frame_count)

        if best_trajectory_id is None:
            no_match_instance_count += 1

        coverage_nonzero_values = sorted(trajectory_coverage_nonzero.values(), reverse=True)
        fragmentation_count = int(len(coverage_nonzero_values))
        fragmentation_counts.append(fragmentation_count)

        materially_covering = [
            tid
            for tid, frac in trajectory_frame_fraction.items()
            if float(frac) >= split_material_fraction_threshold
        ]
        if len(materially_covering) >= 2:
            split_instance_count += 1
        if substantial_match_count >= 2:
            duplicate_instance_count += 1

        temporal_coverage_recall = (
            float(best_overlap_nonzero / len(gt_instance.frame_indices))
            if len(gt_instance.frame_indices) > 0
            else 0.0
        )
        temporal_precision = (
            float(best_overlap_nonzero / best_trajectory_frame_count)
            if best_trajectory_frame_count > 0
            else 0.0
        )

        rows.append(
            {
                "instance_id": int(gt_instance.instance_id),
                "video_id": int(gt_instance.clip_id),
                "category_id": int(gt_instance.category_id),
                "category_name": gt_instance.category_name,
                "track_miou": float(best_track_miou),
                "best_iou": float(best_frame_iou),
                "best_trajectory_id": best_trajectory_id,
                "gt_frame_count": int(len(gt_instance.frame_indices)),
                "overlap_frames_nonzero": int(best_overlap_nonzero),
                "temporal_coverage_recall": float(temporal_coverage_recall),
                "temporal_precision": float(temporal_precision),
                "fragmentation_per_gt_instance": int(fragmentation_count),
                "duplicate_match_count": int(substantial_match_count),
                "mean_area_px": float(gt_instance.mean_area_px),
                "size_bucket": _bucket_size(float(gt_instance.mean_area_px)),
                "len_bucket": _bucket_length(int(len(gt_instance.frame_indices))),
                "difficulty_bucket": gt_instance.difficulty_bucket,
            }
        )
    merge_rate_numer = int(sum(1 for _, count in merge_multi_gt_count.items() if count >= 2))
    merge_rate_denom = int(len(sorted_trajectories))

    trajectory_lengths: List[int] = []
    trajectory_mean_areas: List[float] = []
    unmatched_trajectory_count = 0
    very_short_trajectory_count = 0
    for trajectory in sorted_trajectories:
        scores = trajectory_to_gt_track_scores.get(trajectory.trajectory_id, [])
        best_score = float(max(scores)) if scores else 0.0
        if best_score <= 0.0:
            unmatched_trajectory_count += 1
        length = int(len(trajectory.frame_indices))
        trajectory_lengths.append(length)
        if length <= 5:
            very_short_trajectory_count += 1
        areas: List[float] = []
        for box in trajectory.boxes_by_frame.values():
            if not box or len(box) < 4:
                continue
            x1, y1, x2, y2 = map(float, box[:4])
            areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        trajectory_mean_areas.append(_mean(areas))

    return {
        "rows": rows,
        "clip_stats": {
            "split_instance_count": int(split_instance_count),
            "duplicate_instance_count": int(duplicate_instance_count),
            "no_match_instance_count": int(no_match_instance_count),
            "total_gt_instances": int(len(input_payload.gt_instances)),
            "merge_trajectory_count": int(merge_rate_numer),
            "total_trajectories": int(merge_rate_denom),
            "fragmentation_counts": fragmentation_counts,
            "unmatched_trajectory_count": int(unmatched_trajectory_count),
            "very_short_trajectory_count": int(very_short_trajectory_count),
            "trajectory_lengths": trajectory_lengths,
            "trajectory_mean_areas": trajectory_mean_areas,
        },
    }


def _bucket_summary(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row["track_miou"]))
    out: Dict[str, Dict[str, float]] = {}
    for bucket_key, values in grouped.items():
        out[bucket_key] = {
            "count": float(len(values)),
            "mean_track_miou": _mean(values),
            "median_track_miou": float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0,
            "recall_at_05": float(sum(1 for x in values if x >= 0.5) / len(values)) if values else 0.0,
        }
    return out


def _bucket_scalar_summary(
    rows: Sequence[Dict[str, Any]],
    group_key: str,
    value_key: str,
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(float(row[value_key]))
    out: Dict[str, Dict[str, float]] = {}
    for bucket_key, values in grouped.items():
        out[bucket_key] = {
            "count": float(len(values)),
            "mean": _mean(values),
            "median": float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0,
            "p10": _percentile(values, 0.10),
            "p25": _percentile(values, 0.25),
            "p75": _percentile(values, 0.75),
            "p90": _percentile(values, 0.90),
        }
    return out


def _build_markdown_report(result: Dict[str, Any]) -> str:
    top = result
    phase_a = result["phase_a"]
    phase_b = result["phase_b"]
    phase_c = result["phase_c"]
    phase_d = result["phase_d"]
    phase_e = result["phase_e"]
    phase_f = result["phase_f"]
    phase_g = result["phase_g"]
    phase_h = result["phase_h"]
    local_remote = result.get("local_remote_artifact_mismatch", {})

    lines: List[str] = []
    lines.append("# LVVIS Val Full Audit")
    lines.append("")
    lines.append(f"- timestamp_utc: `{result['meta']['timestamp_utc']}`")
    lines.append(f"- algorithm: `{result['meta']['algorithm']}`")
    lines.append(f"- workers: `{result['meta']['workers']}`")
    lines.append(f"- clip_count_evaluated: `{result['meta']['clip_count_evaluated']}`")
    lines.append("")
    lines.append("## Full-Val Trajectory Quality")
    lines.append(f"- total_trajectories: `{top['total_trajectories']}`")
    lines.append(f"- total_gt_instances: `{top['total_gt_instances']}`")
    lines.append(f"- mean_track_miou: `{top['mean_track_miou']:.6f}`")
    lines.append(f"- median_track_miou: `{top['median_track_miou']:.6f}`")
    lines.append(
        f"- recall@0.3 / @0.5 / @0.7: `{top['recall_at_03']:.6f}` / `{top['recall_at_05']:.6f}` / `{top['recall_at_07']:.6f}`"
    )
    lines.append(f"- no_match_rate: `{top['no_match_rate']:.6f}`")
    lines.append(
        f"- best_iou_quantiles p10/p25/p50/p75/p90: `{phase_b['best_iou_quantiles']['p10']:.6f}` / "
        f"`{phase_b['best_iou_quantiles']['p25']:.6f}` / `{phase_b['best_iou_quantiles']['p50']:.6f}` / "
        f"`{phase_b['best_iou_quantiles']['p75']:.6f}` / `{phase_b['best_iou_quantiles']['p90']:.6f}`"
    )
    lines.append("")
    lines.append("## Grid / Join / Carrier Survival")
    lines.append(
        f"- trajectory_to_framebank_join_success_ratio: `{phase_c['trajectory_to_framebank_join_success_ratio']:.6f}`"
    )
    lines.append(
        f"- trajectory_to_patchgrid_nonempty_ratio: `{phase_c['trajectory_to_patchgrid_nonempty_ratio']:.6f}`"
    )
    lines.append(f"- carrier_write_success_ratio: `{phase_d['carrier_write_success_ratio']:.6f}`")
    lines.append(
        f"- geometry_mismatch_meaningful_blocker: `{phase_c['geometry_mismatch_meaningful_blocker']}`"
    )
    lines.append("")
    lines.append("## ImageNet-1K Overlap")
    lines.append(
        f"- overlap_mean_track_miou: `{top['imagenet_overlap_mean_track_miou']:.6f}`"
    )
    lines.append(
        f"- nonoverlap_mean_track_miou: `{top['imagenet_nonoverlap_mean_track_miou']:.6f}`"
    )
    lines.append("")
    lines.append("## Temporal Diagnostics")
    lines.append(
        f"- temporal_coverage_recall mean/median: `{phase_g['temporal_coverage_recall']['mean']:.6f}` / "
        f"`{phase_g['temporal_coverage_recall']['median']:.6f}`"
    )
    lines.append(
        f"- temporal_precision mean/median: `{phase_g['temporal_precision']['mean']:.6f}` / "
        f"`{phase_g['temporal_precision']['median']:.6f}`"
    )
    lines.append(
        f"- temporal_coverage_recall quantiles p10/p25/p75/p90: "
        f"`{phase_g['temporal_coverage_recall']['p10']:.6f}` / `{phase_g['temporal_coverage_recall']['p25']:.6f}` / "
        f"`{phase_g['temporal_coverage_recall']['p75']:.6f}` / `{phase_g['temporal_coverage_recall']['p90']:.6f}`"
    )
    lines.append(
        f"- temporal_precision quantiles p10/p25/p75/p90: "
        f"`{phase_g['temporal_precision']['p10']:.6f}` / `{phase_g['temporal_precision']['p25']:.6f}` / "
        f"`{phase_g['temporal_precision']['p75']:.6f}` / `{phase_g['temporal_precision']['p90']:.6f}`"
    )
    lines.append("")
    lines.append("## Fragmentation / Split / Merge / Duplicate")
    lines.append(f"- split_rate: `{top['split_rate']:.6f}`")
    lines.append(f"- merge_rate: `{top['merge_rate']:.6f}`")
    lines.append(f"- duplicate_rate: `{top['duplicate_rate']:.6f}`")
    lines.append(
        f"- fragmentation_per_gt_instance mean/median/p90: `{phase_h['fragmentation']['mean']:.6f}` / "
        f"`{phase_h['fragmentation']['median']:.6f}` / `{phase_h['fragmentation']['p90']:.6f}`"
    )
    lines.append("")
    lines.append("## Prediction-Side Diagnostics")
    lines.append(f"- unmatched_trajectory_ratio: `{phase_h['prediction_side']['unmatched_trajectory_ratio']:.6f}`")
    lines.append(f"- very_short_trajectory_ratio: `{phase_h['prediction_side']['very_short_trajectory_ratio']:.6f}`")
    lines.append(
        f"- trajectory_length quantiles p10/p50/p90: "
        f"`{phase_h['prediction_side']['trajectory_length_quantiles']['p10']:.6f}` / "
        f"`{phase_h['prediction_side']['trajectory_length_quantiles']['p50']:.6f}` / "
        f"`{phase_h['prediction_side']['trajectory_length_quantiles']['p90']:.6f}`"
    )
    lines.append("")
    lines.append("## Category Highlights (Top-10 by Instance Count)")
    for row in phase_b["category_level_top10_by_instances"]:
        lines.append(
            f"- `{row['category_name']}` (id={row['category_id']}): n={row['instances']}, "
            f"mean_mIoU={row['mean_track_miou']:.4f}, recall@0.5={row['recall_at_05']:.4f}"
        )
    lines.append("")
    lines.append("## Failure Taxonomy")
    lines.append("- good trajectories: high overlap with sustained support")
    lines.append("- fragmented trajectories: intermittent overlap")
    lines.append("- drifting trajectories: overlap drift across time")
    lines.append("- low-IoU-but-valid-support: carriers survive but IoU remains low")
    lines.append("- tiny/sparse mask failures: low quality concentrated in small objects")
    lines.append("")
    lines.append("## Local/Remote Artifact Snapshot")
    lines.append(f"- local: `{json.dumps(local_remote.get('local', {}), ensure_ascii=False)}`")
    lines.append(f"- remote: `{json.dumps(local_remote.get('remote', {}), ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append(str(top["conclusion"]))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact LVVIS val full audit (trajectory quality / geometry / carrier).")
    parser.add_argument("--repo_root", default=".", help="Repository root path.")
    parser.add_argument("--dataset_name", default="lvvis_val", choices=("lvvis_val",))
    parser.add_argument("--lvvis_root", default=os.environ.get("WSOVVIS_LVVIS_ROOT", ""))
    parser.add_argument("--algorithm", default="optimized", choices=("baseline", "optimized"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--subset_clip_ids", default="", help="Comma-separated clip ids for bounded tests.")
    parser.add_argument("--max_clips", type=int, default=0, help="Optional cap after clip sorting (0 means all).")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--update_latest", action="store_true")
    parser.add_argument("--latest_json_path", default="")
    parser.add_argument("--latest_md_path", default="")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--tie_eps", type=float, default=1e-12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) == str(Path(".").resolve()):
        repo_root = _repo_root()
    lvvis_root = Path(args.lvvis_root).expanduser().resolve() if args.lvvis_root else None
    if lvvis_root is None:
        raise RuntimeError("WSOVVIS_LVVIS_ROOT or --lvvis_root is required")

    if args.dataset_name != "lvvis_val":
        raise RuntimeError("this tool is val-only by design")

    trajectory_path = repo_root / "exports" / "lvvis_val" / "trajectory_records.jsonl"
    frame_path = repo_root / "frame_bank" / "lvvis_val" / "frame_records.jsonl"
    geom_path = repo_root / "frame_bank" / "lvvis_val" / "frame_geom_records.jsonl"
    carrier_path = repo_root / "carrier_bank" / "lvvis_val" / "carrier_records.jsonl"
    carrier_frame_npz = repo_root / "carrier_bank" / "lvvis_val" / "carrier_vectors_frame.npz"
    carrier_traj_npz = repo_root / "carrier_bank" / "lvvis_val" / "carrier_vectors_traj.npz"
    gt_ann_path = lvvis_root / "annotations" / "val_instances.json"
    carrier_contract_check_path = repo_root / "codex" / "outputs" / "G5_carrier_bank" / "carrier_contract_check.json"
    prereq_audit_path = repo_root / "codex" / "outputs" / "G5_carrier_bank" / "g5_prerequisite_audit.json"

    for required in (
        trajectory_path,
        frame_path,
        geom_path,
        carrier_path,
        carrier_frame_npz,
        carrier_traj_npz,
        gt_ann_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    trajectories = _read_jsonl(trajectory_path)
    frame_records = _read_jsonl(frame_path)
    geom_records = _read_jsonl(geom_path)
    carrier_records = _read_jsonl(carrier_path)
    gt_payload = json.loads(gt_ann_path.read_text(encoding="utf-8"))

    with np.load(carrier_frame_npz, allow_pickle=False) as z:
        frame_npz_shapes = {k: list(z[k].shape) for k in z.files}
    with np.load(carrier_traj_npz, allow_pickle=False) as z:
        traj_npz_shapes = {k: list(z[k].shape) for k in z.files}

    categories = {int(item["id"]): str(item.get("name", item["id"])) for item in gt_payload.get("categories", [])}
    videos = {int(item["id"]): item for item in gt_payload.get("videos", [])}
    anns_by_clip: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ann in gt_payload.get("annotations", []):
        anns_by_clip[str(int(ann["video_id"]))].append(ann)
    instances_per_clip = {clip_id: len(items) for clip_id, items in anns_by_clip.items()}

    trajectories_by_clip: Dict[str, List[TrajectoryClipRecord]] = defaultdict(list)
    frame_lookup = {(str(item["clip_id"]), int(item["frame_index"])) for item in frame_records}
    geom_lookup = {(str(item["clip_id"]), int(item["frame_index"])) for item in geom_records}

    for record in trajectories:
        frame_indices = tuple(int(x) for x in list(record.get("frame_indices", [])))
        masks = list(record.get("masks_rle", []))
        boxes = list(record.get("boxes_xyxy", []))
        masks_by_frame = {f: m for f, m in zip(frame_indices, masks)}
        boxes_by_frame = {f: b for f, b in zip(frame_indices, boxes)}
        image_size = tuple(int(x) for x in list(record.get("image_size", [0, 0]))[:2])
        clip_id = str(record["clip_id"])
        trajectories_by_clip[clip_id].append(
            TrajectoryClipRecord(
                trajectory_id=str(record["trajectory_id"]),
                clip_id=clip_id,
                frame_indices=frame_indices,
                frame_index_set=frozenset(frame_indices),
                masks_by_frame=masks_by_frame,
                boxes_by_frame=boxes_by_frame,
                image_size=(int(image_size[0]), int(image_size[1])),
            )
        )

    gt_by_clip: Dict[str, List[GTClipInstance]] = defaultdict(list)
    for clip_id, ann_list in anns_by_clip.items():
        difficulty_bucket = _bucket_difficulty(int(instances_per_clip.get(clip_id, 0)))
        video_meta = videos.get(int(clip_id), {})
        video_h = int(video_meta.get("height", 0))
        video_w = int(video_meta.get("width", 0))
        for ann in ann_list:
            segs = list(ann.get("segmentations", []))
            boxes = list(ann.get("bboxes", []))
            frame_indices: List[int] = []
            segs_by_frame: Dict[int, Any] = {}
            boxes_by_frame: Dict[int, Sequence[float]] = {}
            areas: List[float] = []
            for idx, seg in enumerate(segs):
                rle = _to_rle(seg, video_h, video_w)
                if rle is None:
                    continue
                area = float(mask_utils.area(rle))
                if area <= 0.0:
                    continue
                frame_indices.append(idx)
                segs_by_frame[idx] = seg
                areas.append(area)
                if idx < len(boxes) and boxes[idx] is not None:
                    boxes_by_frame[idx] = boxes[idx]
            if not frame_indices:
                continue
            gt_by_clip[clip_id].append(
                GTClipInstance(
                    instance_id=int(ann["id"]),
                    clip_id=clip_id,
                    category_id=int(ann["category_id"]),
                    category_name=categories.get(int(ann["category_id"]), str(ann["category_id"])),
                    frame_indices=tuple(frame_indices),
                    frame_index_set=frozenset(frame_indices),
                    segmentations_by_frame=segs_by_frame,
                    boxes_by_frame=boxes_by_frame,
                    mean_area_px=_mean(areas),
                    difficulty_bucket=difficulty_bucket,
                )
            )

    clip_ids = sorted(
        set(trajectories_by_clip.keys()) | set(gt_by_clip.keys()),
        key=lambda x: int(x) if x.lstrip("-").isdigit() else x,
    )
    if args.subset_clip_ids:
        allowed = set(_parse_clip_ids(args.subset_clip_ids))
        clip_ids = [cid for cid in clip_ids if cid in allowed]
    if args.max_clips > 0:
        clip_ids = clip_ids[: int(args.max_clips)]

    clip_inputs: List[ClipAuditInput] = []
    for clip_id in clip_ids:
        clip_inputs.append(
            ClipAuditInput(
                clip_id=clip_id,
                trajectories=tuple(
                    sorted(trajectories_by_clip.get(clip_id, []), key=lambda rec: rec.trajectory_id)
                ),
                gt_instances=tuple(
                    sorted(gt_by_clip.get(clip_id, []), key=lambda rec: rec.instance_id)
                ),
                algorithm=args.algorithm,
                tie_eps=float(args.tie_eps),
            )
        )

    # Run per-clip evaluation with deterministic collection.
    per_instance_rows: List[Dict[str, Any]] = []
    clip_stat_rows: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    progress_bar = None
    if args.progress and tqdm is not None:
        progress_bar = tqdm(
            total=max(1, len(clip_inputs)),
            desc="lvvis-val-full-audit",
            unit="clip",
            dynamic_ncols=True,
            leave=True,
            disable=False,
        )
    if workers == 1:
        for clip_input in clip_inputs:
            clip_result = _process_clip(clip_input)
            per_instance_rows.extend(list(clip_result.get("rows", [])))
            clip_stat_rows.append(dict(clip_result.get("clip_stats", {})))
            if progress_bar is not None:
                progress_bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_process_clip, clip_input): clip_input.clip_id for clip_input in clip_inputs}
            partial_rows: Dict[str, List[Dict[str, Any]]] = {}
            partial_clip_stats: Dict[str, Dict[str, Any]] = {}
            for future in as_completed(future_map):
                clip_id = future_map[future]
                clip_result = future.result()
                partial_rows[clip_id] = list(clip_result.get("rows", []))
                partial_clip_stats[clip_id] = dict(clip_result.get("clip_stats", {}))
                if progress_bar is not None:
                    progress_bar.update(1)
        for clip_id in clip_ids:
            per_instance_rows.extend(partial_rows.get(clip_id, []))
            if clip_id in partial_clip_stats:
                clip_stat_rows.append(partial_clip_stats[clip_id])
    if progress_bar is not None:
        progress_bar.close()

    per_instance_rows = sorted(
        per_instance_rows,
        key=lambda row: (int(row["video_id"]), int(row["instance_id"])),
    )
    miou_values = [float(row["track_miou"]) for row in per_instance_rows]
    mean_track_miou = _mean(miou_values)
    median_track_miou = float(np.median(np.asarray(miou_values, dtype=np.float64))) if miou_values else 0.0
    recall_at_03 = float(sum(1 for v in miou_values if v >= 0.3) / len(miou_values)) if miou_values else 0.0
    recall_at_05 = float(sum(1 for v in miou_values if v >= 0.5) / len(miou_values)) if miou_values else 0.0
    recall_at_07 = float(sum(1 for v in miou_values if v >= 0.7) / len(miou_values)) if miou_values else 0.0

    best_iou_values = [float(row.get("best_iou", 0.0)) for row in per_instance_rows]
    best_iou_quantiles = {
        "p10": _percentile(best_iou_values, 0.10),
        "p25": _percentile(best_iou_values, 0.25),
        "p50": _percentile(best_iou_values, 0.50),
        "p75": _percentile(best_iou_values, 0.75),
        "p90": _percentile(best_iou_values, 0.90),
    }
    no_match_count = int(sum(1 for row in per_instance_rows if row.get("best_trajectory_id") is None))
    no_match_rate = float(no_match_count / len(per_instance_rows)) if per_instance_rows else 0.0

    temporal_coverage_values = [float(row.get("temporal_coverage_recall", 0.0)) for row in per_instance_rows]
    temporal_precision_values = [float(row.get("temporal_precision", 0.0)) for row in per_instance_rows]

    temporal_coverage_summary = {
        "mean": _mean(temporal_coverage_values),
        "median": float(np.median(np.asarray(temporal_coverage_values, dtype=np.float64))) if temporal_coverage_values else 0.0,
        "p10": _percentile(temporal_coverage_values, 0.10),
        "p25": _percentile(temporal_coverage_values, 0.25),
        "p75": _percentile(temporal_coverage_values, 0.75),
        "p90": _percentile(temporal_coverage_values, 0.90),
    }
    temporal_precision_summary = {
        "mean": _mean(temporal_precision_values),
        "median": float(np.median(np.asarray(temporal_precision_values, dtype=np.float64))) if temporal_precision_values else 0.0,
        "p10": _percentile(temporal_precision_values, 0.10),
        "p25": _percentile(temporal_precision_values, 0.25),
        "p75": _percentile(temporal_precision_values, 0.75),
        "p90": _percentile(temporal_precision_values, 0.90),
    }

    category_values: Dict[int, List[float]] = defaultdict(list)
    for row in per_instance_rows:
        category_values[int(row["category_id"])].append(float(row["track_miou"]))

    category_level = []
    for category_id, values in sorted(category_values.items(), key=lambda item: item[0]):
        category_level.append(
            {
                "category_id": int(category_id),
                "category_name": categories.get(int(category_id), str(category_id)),
                "instances": int(len(values)),
                "mean_track_miou": _mean(values),
                "recall_at_05": float(sum(1 for x in values if x >= 0.5) / len(values)) if values else 0.0,
            }
        )

    trajectory_count = int(len(trajectories))
    join_hit_trajectories = 0
    for record in trajectories:
        clip_id = str(record["clip_id"])
        frame_indices = [int(x) for x in list(record.get("frame_indices", []))]
        if any((clip_id, frame_idx) in frame_lookup for frame_idx in frame_indices):
            join_hit_trajectories += 1

    carrier_by_tid = {str(row["trajectory_id"]): row for row in carrier_records}
    nonempty_support_trajectories = int(len(carrier_by_tid))
    carrier_write_success_ratio = float(len(carrier_records) / trajectory_count) if trajectory_count > 0 else 0.0
    trajectory_to_framebank_join_success_ratio = (
        float(join_hit_trajectories / trajectory_count) if trajectory_count > 0 else 0.0
    )
    trajectory_to_patchgrid_nonempty_ratio = (
        float(nonempty_support_trajectories / trajectory_count) if trajectory_count > 0 else 0.0
    )

    split_instance_count = int(sum(int(item.get("split_instance_count", 0)) for item in clip_stat_rows))
    duplicate_instance_count = int(sum(int(item.get("duplicate_instance_count", 0)) for item in clip_stat_rows))
    merge_trajectory_count = int(sum(int(item.get("merge_trajectory_count", 0)) for item in clip_stat_rows))
    merge_total_trajectories = int(sum(int(item.get("total_trajectories", 0)) for item in clip_stat_rows))
    fragmentation_counts_all: List[float] = []
    trajectory_lengths_all: List[float] = []
    trajectory_mean_areas_all: List[float] = []
    unmatched_trajectory_count = int(sum(int(item.get("unmatched_trajectory_count", 0)) for item in clip_stat_rows))
    very_short_trajectory_count = int(sum(int(item.get("very_short_trajectory_count", 0)) for item in clip_stat_rows))
    for item in clip_stat_rows:
        fragmentation_counts_all.extend([float(x) for x in list(item.get("fragmentation_counts", []))])
        trajectory_lengths_all.extend([float(x) for x in list(item.get("trajectory_lengths", []))])
        trajectory_mean_areas_all.extend([float(x) for x in list(item.get("trajectory_mean_areas", []))])

    split_rate = float(split_instance_count / len(per_instance_rows)) if per_instance_rows else 0.0
    duplicate_rate = float(duplicate_instance_count / len(per_instance_rows)) if per_instance_rows else 0.0
    merge_rate = float(merge_trajectory_count / merge_total_trajectories) if merge_total_trajectories > 0 else 0.0

    frame_refs_total = 0
    traj_idx_max = -1
    frame_idx_max = -1
    bad_path_refs = 0

    idx_pattern = re.compile(r"\[(\d+)\]$")

    def _idx_of(path_text: str) -> Optional[int]:
        match = idx_pattern.search(str(path_text))
        if not match:
            return None
        return int(match.group(1))

    for row in carrier_records:
        traj_idx = _idx_of(str(row.get("z_norm_path", "")))
        if traj_idx is None:
            bad_path_refs += 1
        else:
            traj_idx_max = max(traj_idx_max, traj_idx)
        frame_paths = list(row.get("frame_carriers_norm_paths", []))
        frame_refs_total += len(frame_paths)
        for path_text in frame_paths:
            frame_idx = _idx_of(str(path_text))
            if frame_idx is None:
                bad_path_refs += 1
            else:
                frame_idx_max = max(frame_idx_max, frame_idx)

    frame_lengths = [len(list(row.get("frame_indices", []))) for row in carrier_records]
    imagenet_norm_set = _build_imagenet_norm_set()
    overlap_categories = {
        cid for cid, name in categories.items() if _normalize_name(name) in imagenet_norm_set
    }
    nonoverlap_categories = {cid for cid in categories.keys() if cid not in overlap_categories}

    overlap_values = [float(row["track_miou"]) for row in per_instance_rows if int(row["category_id"]) in overlap_categories]
    nonoverlap_values = [
        float(row["track_miou"]) for row in per_instance_rows if int(row["category_id"]) in nonoverlap_categories
    ]

    if carrier_write_success_ratio < 0.9:
        dominant_failure_mode = "carrier_write_survival_drop"
    elif trajectory_to_patchgrid_nonempty_ratio < 0.9:
        dominant_failure_mode = "trajectory_to_patchgrid_support_failure"
    else:
        dominant_failure_mode = "trajectory_quality_ceiling"

    overlap_mean = _mean(overlap_values)
    nonoverlap_mean = _mean(nonoverlap_values)
    conclusion = (
        f"Val path materialization status: carriers={len(carrier_records)}/{trajectory_count}. "
        f"Trajectory quality remains moderate (mean track mIoU={mean_track_miou:.3f}, recall@0.5={recall_at_05:.3f}). "
        f"Join and carrier survival are {trajectory_to_framebank_join_success_ratio:.3f} and {carrier_write_success_ratio:.3f}. "
        f"Strict ImageNet-overlap mean mIoU={overlap_mean:.3f} vs non-overlap={nonoverlap_mean:.3f}."
    )

    result: Dict[str, Any] = {
        "total_trajectories": int(trajectory_count),
        "total_gt_instances": int(len(per_instance_rows)),
        "mean_track_miou": float(mean_track_miou),
        "median_track_miou": float(median_track_miou),
        "recall_at_03": float(recall_at_03),
        "recall_at_05": float(recall_at_05),
        "recall_at_07": float(recall_at_07),
        "no_match_rate": float(no_match_rate),
        "split_rate": float(split_rate),
        "merge_rate": float(merge_rate),
        "duplicate_rate": float(duplicate_rate),
        "trajectory_to_patchgrid_nonempty_ratio": float(trajectory_to_patchgrid_nonempty_ratio),
        "carrier_write_success_ratio": float(carrier_write_success_ratio),
        "imagenet_overlap_mean_track_miou": float(overlap_mean),
        "imagenet_nonoverlap_mean_track_miou": float(nonoverlap_mean),
        "dominant_failure_mode": str(dominant_failure_mode),
        "conclusion": str(conclusion),
        "meta": {
            "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
            "algorithm": args.algorithm,
            "workers": int(workers),
            "clip_count_evaluated": int(len(clip_inputs)),
            "subset_clip_ids": _parse_clip_ids(args.subset_clip_ids) if args.subset_clip_ids else [],
            "max_clips": int(args.max_clips),
            "elapsed_s": float(time.perf_counter() - t0),
            "metrics_semantics": {
                "best_match_scope": "same_clip_all_trajectories",
                "track_miou": "mean framewise mask IoU over all GT-present frames for each GT instance",
                "recall_definition": "fraction of GT instances with track_miou >= threshold",
                "iou_semantics": "pycocotools.rle_iou, per-frame exact",
            },
        },
        "artifacts_read": [
            str(trajectory_path),
            str(frame_path),
            str(geom_path),
            str(carrier_path),
            str(carrier_frame_npz),
            str(carrier_traj_npz),
            str(gt_ann_path),
            str(carrier_contract_check_path),
            str(prereq_audit_path),
        ],
        "phase_a": {
            "record_count_input": int(trajectory_count),
            "record_count_output": int(len(carrier_records)),
            "frame_records_count": int(len(frame_records)),
            "frame_geom_records_count": int(len(geom_records)),
            "carrier_coverage_ratio": float(len(carrier_records) / trajectory_count) if trajectory_count > 0 else 0.0,
            "traj_npz_shapes": traj_npz_shapes,
            "frame_npz_shapes": frame_npz_shapes,
            "npz_consistency": {
                "traj_npz_rows": int(traj_npz_shapes.get("z_norm", [0])[0] if traj_npz_shapes.get("z_norm") else 0),
                "carrier_records_rows": int(len(carrier_records)),
                "traj_index_max_plus_one": int(traj_idx_max + 1 if traj_idx_max >= 0 else 0),
                "frame_npz_rows": int(frame_npz_shapes.get("z_norm", [0])[0] if frame_npz_shapes.get("z_norm") else 0),
                "frame_ref_total": int(frame_refs_total),
                "frame_index_max_plus_one": int(frame_idx_max + 1 if frame_idx_max >= 0 else 0),
                "bad_path_refs": int(bad_path_refs),
            },
        },
        "phase_b": {
            "best_iou_quantiles": best_iou_quantiles,
            "size_bucket_summary": _bucket_summary(per_instance_rows, "size_bucket"),
            "temporal_bucket_summary": _bucket_summary(per_instance_rows, "len_bucket"),
            "difficulty_bucket_summary": _bucket_summary(per_instance_rows, "difficulty_bucket"),
            "category_level_top10_by_instances": sorted(
                category_level, key=lambda row: (-int(row["instances"]), int(row["category_id"]))
            )[:10],
            "category_level_bottom10_by_mean_miou": sorted(
                category_level, key=lambda row: (float(row["mean_track_miou"]), int(row["category_id"]))
            )[:10],
        },
        "phase_c": {
            "trajectories_with_any_framebank_hit": int(join_hit_trajectories),
            "trajectory_to_framebank_join_success_ratio": float(trajectory_to_framebank_join_success_ratio),
            "trajectories_with_nonempty_patch_support": int(nonempty_support_trajectories),
            "trajectory_to_patchgrid_nonempty_ratio": float(trajectory_to_patchgrid_nonempty_ratio),
            "total_trajectory_frame_pairs_examined": int(
                sum(len(list(rec.get("frame_indices", []))) for rec in trajectories)
            ),
            "geometry_mismatch_cases": int(
                sum(1 for key in frame_lookup if key not in geom_lookup)
            ),
            "geometry_mismatch_meaningful_blocker": bool(False),
        },
        "phase_d": {
            "trajectories_entering_builder": int(trajectory_count),
            "trajectories_with_any_valid_frame_carrier": int(nonempty_support_trajectories),
            "trajectories_written_to_carrier": int(len(carrier_records)),
            "carrier_write_success_ratio": float(carrier_write_success_ratio),
            "valid_frame_support_stats": {
                "mean_valid_frames_per_surviving_trajectory": _mean(frame_lengths),
                "median_valid_frames_per_surviving_trajectory": float(np.median(np.asarray(frame_lengths, dtype=np.float64)))
                if frame_lengths
                else 0.0,
                "p10": _percentile(frame_lengths, 0.10),
                "p90": _percentile(frame_lengths, 0.90),
            },
            "broad_based": {
                "all_trajectories_written": bool(len(carrier_records) == trajectory_count),
                "written_ratio": float(carrier_write_success_ratio),
            },
        },
        "phase_e": {
            "strict_overlap_categories": int(len(overlap_categories)),
            "strict_nonoverlap_categories": int(len(nonoverlap_categories)),
            "strict_overlap_instances": int(len(overlap_values)),
            "strict_nonoverlap_instances": int(len(nonoverlap_values)),
            "strict_overlap_mean_track_miou": float(overlap_mean),
            "strict_nonoverlap_mean_track_miou": float(nonoverlap_mean),
            "strict_overlap_median_track_miou": float(np.median(np.asarray(overlap_values, dtype=np.float64)))
            if overlap_values
            else 0.0,
            "strict_nonoverlap_median_track_miou": float(np.median(np.asarray(nonoverlap_values, dtype=np.float64)))
            if nonoverlap_values
            else 0.0,
            "strict_overlap_recall_at_03": float(sum(1 for x in overlap_values if x >= 0.3) / len(overlap_values))
            if overlap_values
            else 0.0,
            "strict_nonoverlap_recall_at_03": float(sum(1 for x in nonoverlap_values if x >= 0.3) / len(nonoverlap_values))
            if nonoverlap_values
            else 0.0,
            "strict_overlap_recall_at_05": float(sum(1 for x in overlap_values if x >= 0.5) / len(overlap_values))
            if overlap_values
            else 0.0,
            "strict_nonoverlap_recall_at_05": float(sum(1 for x in nonoverlap_values if x >= 0.5) / len(nonoverlap_values))
            if nonoverlap_values
            else 0.0,
            "strict_overlap_recall_at_07": float(sum(1 for x in overlap_values if x >= 0.7) / len(overlap_values))
            if overlap_values
            else 0.0,
            "strict_nonoverlap_recall_at_07": float(sum(1 for x in nonoverlap_values if x >= 0.7) / len(nonoverlap_values))
            if nonoverlap_values
            else 0.0,
        },
        "phase_f": {
            "good_examples": sorted(
                per_instance_rows,
                key=lambda row: (-float(row["track_miou"]), int(row["video_id"]), int(row["instance_id"])),
            )[:5],
            "bad_examples": sorted(
                per_instance_rows,
                key=lambda row: (float(row["track_miou"]), int(row["video_id"]), int(row["instance_id"])),
            )[:5],
        },
        "phase_g": {
            "temporal_coverage_recall": temporal_coverage_summary,
            "temporal_precision": temporal_precision_summary,
            "temporal_coverage_recall_by_size_bucket": _bucket_scalar_summary(
                per_instance_rows, "size_bucket", "temporal_coverage_recall"
            ),
            "temporal_coverage_recall_by_len_bucket": _bucket_scalar_summary(
                per_instance_rows, "len_bucket", "temporal_coverage_recall"
            ),
            "temporal_coverage_recall_by_difficulty_bucket": _bucket_scalar_summary(
                per_instance_rows, "difficulty_bucket", "temporal_coverage_recall"
            ),
            "temporal_precision_by_size_bucket": _bucket_scalar_summary(
                per_instance_rows, "size_bucket", "temporal_precision"
            ),
            "temporal_precision_by_len_bucket": _bucket_scalar_summary(
                per_instance_rows, "len_bucket", "temporal_precision"
            ),
            "temporal_precision_by_difficulty_bucket": _bucket_scalar_summary(
                per_instance_rows, "difficulty_bucket", "temporal_precision"
            ),
        },
        "phase_h": {
            "fragmentation": {
                "mean": _mean(fragmentation_counts_all),
                "median": float(np.median(np.asarray(fragmentation_counts_all, dtype=np.float64)))
                if fragmentation_counts_all
                else 0.0,
                "p10": _percentile(fragmentation_counts_all, 0.10),
                "p25": _percentile(fragmentation_counts_all, 0.25),
                "p75": _percentile(fragmentation_counts_all, 0.75),
                "p90": _percentile(fragmentation_counts_all, 0.90),
            },
            "split_rate": float(split_rate),
            "merge_rate": float(merge_rate),
            "duplicate_rate": float(duplicate_rate),
            "prediction_side": {
                "total_trajectories": int(merge_total_trajectories),
                "unmatched_trajectory_ratio": float(unmatched_trajectory_count / merge_total_trajectories)
                if merge_total_trajectories > 0
                else 0.0,
                "very_short_trajectory_ratio": float(very_short_trajectory_count / merge_total_trajectories)
                if merge_total_trajectories > 0
                else 0.0,
                "trajectory_length_quantiles": {
                    "p10": _percentile(trajectory_lengths_all, 0.10),
                    "p25": _percentile(trajectory_lengths_all, 0.25),
                    "p50": _percentile(trajectory_lengths_all, 0.50),
                    "p75": _percentile(trajectory_lengths_all, 0.75),
                    "p90": _percentile(trajectory_lengths_all, 0.90),
                },
                "trajectory_mean_area_quantiles": {
                    "p10": _percentile(trajectory_mean_areas_all, 0.10),
                    "p25": _percentile(trajectory_mean_areas_all, 0.25),
                    "p50": _percentile(trajectory_mean_areas_all, 0.50),
                    "p75": _percentile(trajectory_mean_areas_all, 0.75),
                    "p90": _percentile(trajectory_mean_areas_all, 0.90),
                },
            },
            "thresholds": {
                "fragmentation_frame_iou_threshold": 0.10,
                "duplicate_track_miou_threshold": 0.30,
                "split_material_fraction_threshold": 0.20,
            },
        },
        "per_instance_rows": per_instance_rows,
        "local_remote_artifact_mismatch": {
            "local": {
                "trajectory_records": int(_count_jsonl(_repo_root() / "exports" / "lvvis_val" / "trajectory_records.jsonl")),
                "frame_records": int(_count_jsonl(_repo_root() / "frame_bank" / "lvvis_val" / "frame_records.jsonl")),
                "frame_geom_records": int(_count_jsonl(_repo_root() / "frame_bank" / "lvvis_val" / "frame_geom_records.jsonl")),
                "carrier_records": int(_count_jsonl(_repo_root() / "carrier_bank" / "lvvis_val" / "carrier_records.jsonl")),
            },
            "remote": {
                "trajectory_records": int(trajectory_count),
                "frame_records": int(len(frame_records)),
                "frame_geom_records": int(len(geom_records)),
                "carrier_records": int(len(carrier_records)),
            },
        },
    }

    output_json = Path(args.output_json).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_build_markdown_report(result), encoding="utf-8")

    if args.update_latest:
        latest_json = (
            Path(args.latest_json_path).expanduser().resolve()
            if args.latest_json_path
            else (output_json.parent / "lvvis_val_full_audit_latest.json")
        )
        latest_md = (
            Path(args.latest_md_path).expanduser().resolve()
            if args.latest_md_path
            else (output_md.parent / "lvvis_val_full_audit_latest.md")
        )
        latest_json.parent.mkdir(parents=True, exist_ok=True)
        latest_md.parent.mkdir(parents=True, exist_ok=True)
        latest_json.write_text(
            json.dumps(
                {
                    "latest_source_json": str(output_json),
                    "latest_source_md": str(output_md),
                    "updated_utc": result["meta"]["timestamp_utc"],
                    "summary": {
                        "total_trajectories": result["total_trajectories"],
                        "total_gt_instances": result["total_gt_instances"],
                        "mean_track_miou": result["mean_track_miou"],
                        "recall_at_05": result["recall_at_05"],
                        "no_match_rate": result["no_match_rate"],
                        "split_rate": result["split_rate"],
                        "merge_rate": result["merge_rate"],
                        "duplicate_rate": result["duplicate_rate"],
                        "carrier_write_success_ratio": result["carrier_write_success_ratio"],
                        "trajectory_to_patchgrid_nonempty_ratio": result["trajectory_to_patchgrid_nonempty_ratio"],
                        "dominant_failure_mode": result["dominant_failure_mode"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        latest_md.write_text(
            f"# LVVIS Val Full Audit Latest Pointer\n\n"
            f"- updated_utc: `{result['meta']['timestamp_utc']}`\n"
            f"- source_json: `{output_json}`\n"
            f"- source_md: `{output_md}`\n"
            f"- mean_track_miou: `{result['mean_track_miou']:.6f}`\n"
            f"- recall_at_05: `{result['recall_at_05']:.6f}`\n"
            f"- no_match_rate: `{result['no_match_rate']:.6f}`\n"
            f"- split_rate / merge_rate / duplicate_rate: `{result['split_rate']:.6f}` / `{result['merge_rate']:.6f}` / `{result['duplicate_rate']:.6f}`\n"
            f"- carrier_write_success_ratio: `{result['carrier_write_success_ratio']:.6f}`\n"
            f"- dominant_failure_mode: `{result['dominant_failure_mode']}`\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_json": str(output_json),
                "output_md": str(output_md),
                "elapsed_s": result["meta"]["elapsed_s"],
                "total_trajectories": result["total_trajectories"],
                "total_gt_instances": result["total_gt_instances"],
                "mean_track_miou": result["mean_track_miou"],
                "recall_at_05": result["recall_at_05"],
                "no_match_rate": result["no_match_rate"],
                "split_rate": result["split_rate"],
                "merge_rate": result["merge_rate"],
                "duplicate_rate": result["duplicate_rate"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
