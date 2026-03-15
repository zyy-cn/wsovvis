#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mask_utils():
    try:
        import pycocotools.mask as mask_utils
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pycocotools.mask is required for duplicate suppression control") from exc
    return mask_utils


def _to_float(value: Any) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list):
        if not value:
            return 0.0
        value = value[0]
    return float(value)


def _seg_area(segmentation: Any) -> float:
    if not segmentation:
        return 0.0
    return _to_float(_mask_utils().area(segmentation))


def _seg_iou_and_union(left_seg: Any, right_seg: Any) -> Tuple[float, float, float]:
    left_area = _seg_area(left_seg)
    right_area = _seg_area(right_seg)
    if not left_seg and not right_seg:
        return 0.0, 0.0, 0.0
    if left_seg and right_seg:
        iou = _to_float(_mask_utils().iou([left_seg], [right_seg], [0]))
        if iou <= 0.0:
            inter = 0.0
        else:
            inter = iou * (left_area + right_area) / (1.0 + iou)
        union = left_area + right_area - inter
        return float(iou), float(inter), float(union)
    return 0.0, 0.0, float(left_area + right_area)


def _tube_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_segments = left.get("segmentations") or []
    right_segments = right.get("segmentations") or []
    max_len = max(len(left_segments), len(right_segments))
    total_inter = 0.0
    total_union = 0.0
    for index in range(max_len):
        left_seg = left_segments[index] if index < len(left_segments) else None
        right_seg = right_segments[index] if index < len(right_segments) else None
        _, inter, union = _seg_iou_and_union(left_seg, right_seg)
        total_inter += inter
        total_union += union
    if total_union <= 0.0:
        return 0.0
    return float(total_inter / total_union)


def _score(row: Mapping[str, Any]) -> float:
    return float(row.get("score", 0.0))


def _row_sort_key(index_row: Tuple[int, Mapping[str, Any]]) -> Tuple[float, int]:
    index, row = index_row
    return (-_score(row), int(row.get("track_id", row.get("id", index))))


def _stats(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Route A duplicate-suppression control to refined results.json.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dup-tube-iou-thresh", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = _load_json(args.input_json)
    if isinstance(payload, dict):
        rows_key = "predictions" if isinstance(payload.get("predictions"), list) else "annotations"
        rows = payload.get(rows_key)
        if not isinstance(rows, list):
            raise ValueError("input-json object must contain predictions or annotations list")
    elif isinstance(payload, list):
        rows_key = None
        rows = payload
    else:
        raise ValueError("input-json must be a list or object")

    by_video: Dict[str, List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        by_video[str(row.get("video_id"))].append((index, row))

    selected_rows: List[Mapping[str, Any]] = []
    per_video_rows: List[Dict[str, Any]] = []
    duplicate_pair_count_total = 0
    dropped_duplicate_total = 0
    dropped_topk_total = 0
    before_counts: List[int] = []
    after_counts: List[int] = []

    for video_id in sorted(by_video.keys(), key=lambda value: int(value) if value.isdigit() else value):
        ordered = sorted(by_video[video_id], key=_row_sort_key)
        before_count = len(ordered)
        before_counts.append(before_count)
        dynamic_rank = min(int(args.top_k), before_count)
        dynamic_threshold = _score(ordered[dynamic_rank - 1][1]) if dynamic_rank > 0 else 0.0
        candidate_pool = [row for _, row in ordered if _score(row) >= dynamic_threshold]

        duplicate_pair_count = 0
        for left_index in range(len(candidate_pool)):
            for right_index in range(left_index + 1, len(candidate_pool)):
                if _tube_iou(candidate_pool[left_index], candidate_pool[right_index]) >= float(args.dup_tube_iou_thresh):
                    duplicate_pair_count += 1
        duplicate_pair_count_total += duplicate_pair_count

        kept: List[Mapping[str, Any]] = []
        dropped_duplicate = 0
        dropped_topk = 0
        for row in candidate_pool:
            if len(kept) >= int(args.top_k):
                dropped_topk += 1
                continue
            if any(_tube_iou(row, selected) >= float(args.dup_tube_iou_thresh) for selected in kept):
                dropped_duplicate += 1
                continue
            kept.append(row)

        dropped_duplicate_total += dropped_duplicate
        dropped_topk_total += dropped_topk
        after_counts.append(len(kept))
        selected_rows.extend(kept)
        per_video_rows.append(
            {
                "video_id": video_id,
                "count_before": before_count,
                "dynamic_threshold": float(dynamic_threshold),
                "candidate_pool_count": len(candidate_pool),
                "count_after": len(kept),
                "dropped_duplicate": dropped_duplicate,
                "dropped_topk": dropped_topk,
                "duplicate_pair_count_at_or_above_thresh": duplicate_pair_count,
            }
        )

    output_payload: Any
    if rows_key is None:
        output_payload = selected_rows
    else:
        output_payload = dict(payload)
        output_payload[rows_key] = selected_rows
    _dump_json(args.output_json, output_payload)
    _dump_json(
        args.summary_json,
        {
            "schema_name": "wsovvis.s1_routea_dupselect_control",
            "schema_version": "1.0.0",
            "input_json": str(args.input_json),
            "output_json": str(args.output_json),
            "selection_policy": {
                "dynamic_threshold_rule": "score_of_kth_ranked_prediction_per_video",
                "top_k": int(args.top_k),
                "dup_tube_iou_thresh": float(args.dup_tube_iou_thresh),
            },
            "counts": {
                "predictions_before": int(sum(before_counts)),
                "predictions_after": int(sum(after_counts)),
                "dropped_duplicate": int(dropped_duplicate_total),
                "dropped_topk": int(dropped_topk_total),
                "duplicate_pair_count_at_or_above_thresh": int(duplicate_pair_count_total),
            },
            "per_video_count_before_stats": _stats(before_counts),
            "per_video_count_after_stats": _stats(after_counts),
            "per_video_rows": per_video_rows,
        },
    )
    print(
        json.dumps(
            {
                "predictions_before": int(sum(before_counts)),
                "predictions_after": int(sum(after_counts)),
                "dropped_duplicate": int(dropped_duplicate_total),
                "dropped_topk": int(dropped_topk_total),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
