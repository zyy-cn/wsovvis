from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .external_lvvis import resolve_lvvis_annotation_paths
from .g8_bridge import G8Paths, validate_json_artifact, write_json


@dataclass(frozen=True)
class InternalEvalConfig:
    exp_name: str
    dataset_name: str
    output_root: Path
    seed: int
    smoke: bool = False


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_pred_main(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    validate_json_artifact(payload, "pred_main.schema.json")
    return [dict(row) for row in payload]


def _read_pred_diag(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    validate_json_artifact(payload, "pred_diag.schema.json")
    return [dict(row) for row in payload]


def _subset_annotation_payload(payload: Mapping[str, Any], video_ids: Iterable[int]) -> Dict[str, Any]:
    keep = {int(v) for v in video_ids}
    return {
        **dict(payload),
        "videos": [dict(video) for video in payload.get("videos", []) if int(video.get("id", -1)) in keep],
        "annotations": [dict(ann) for ann in payload.get("annotations", []) if int(ann.get("video_id", -1)) in keep],
    }


def _normalize_counts_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_uncompressed_rle(counts: Sequence[int], size: Sequence[int]) -> List[List[int]]:
    h, w = int(size[0]), int(size[1])
    total = h * w
    flat = [0] * total
    cursor = 0
    fill = 0
    for raw_run in counts:
        run = int(raw_run)
        if run < 0:
            raise ValueError("RLE counts must be non-negative")
        for index in range(cursor, min(cursor + run, total)):
            flat[index] = fill
        cursor += run
        fill = 1 - fill
    if cursor < total:
        for index in range(cursor, total):
            flat[index] = fill
    mask = [[0 for _ in range(w)] for _ in range(h)]
    for x in range(w):
        for y in range(h):
            mask[y][x] = int(flat[x * h + y])
    return mask


def _mask_from_rle_fallback(rle: Mapping[str, Any]) -> Optional[List[List[int]]]:
    counts = rle.get("counts")
    size = rle.get("size")
    if not isinstance(size, Sequence) or len(size) != 2:
        return None
    if isinstance(counts, list):
        return _decode_uncompressed_rle(counts, size)
    return None


def _mask_iou_from_dense(pred_mask: Sequence[Sequence[int]], gt_mask: Sequence[Sequence[int]]) -> float:
    if not pred_mask or not gt_mask:
        return 0.0
    h = min(len(pred_mask), len(gt_mask))
    w = min(len(pred_mask[0]), len(gt_mask[0]))
    inter = 0
    union = 0
    for y in range(h):
        for x in range(w):
            p = int(pred_mask[y][x]) != 0
            g = int(gt_mask[y][x]) != 0
            if p and g:
                inter += 1
            if p or g:
                union += 1
    if union == 0:
        return 0.0
    return float(inter / union)


def _try_import_mask_utils():
    try:  # pragma: no cover - environment dependent
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        return None
    return mask_utils


def _normalize_rle_for_backend(rle: Mapping[str, Any], *, h: int, w: int, mask_utils: Any) -> Any:
    payload = dict(rle)
    payload.setdefault("size", [h, w])
    if isinstance(payload.get("counts"), list):
        normalized = mask_utils.frPyObjects(payload, h, w)
        if isinstance(normalized, list):
            normalized = mask_utils.merge(normalized)
        payload = dict(normalized)
    if isinstance(payload.get("counts"), str):
        payload["counts"] = payload["counts"].encode("utf-8")
    return payload


def _frame_iou(pred_seg: Any, gt_seg: Any, *, h: int, w: int) -> Optional[float]:
    if pred_seg is None and gt_seg is None:
        return None
    if pred_seg is None or gt_seg is None:
        return 0.0
    mask_utils = _try_import_mask_utils()
    if mask_utils is not None:
        pred_rle = _normalize_rle_for_backend(pred_seg, h=h, w=w, mask_utils=mask_utils)
        gt_rle = _normalize_rle_for_backend(gt_seg, h=h, w=w, mask_utils=mask_utils)
        mat = mask_utils.iou([pred_rle], [gt_rle], [0])
        return float(mat[0][0])
    pred_mask = _mask_from_rle_fallback(pred_seg)
    gt_mask = _mask_from_rle_fallback(gt_seg)
    if pred_mask is not None and gt_mask is not None:
        return _mask_iou_from_dense(pred_mask, gt_mask)
    if (
        isinstance(pred_seg, Mapping)
        and isinstance(gt_seg, Mapping)
        and list(pred_seg.get("size", [h, w])) == list(gt_seg.get("size", [h, w]))
        and _normalize_counts_text(pred_seg.get("counts", "")) == _normalize_counts_text(gt_seg.get("counts", ""))
    ):
        return 1.0
    return 0.0


def _video_iou(pred_segmentations: Sequence[Any], gt_segmentations: Sequence[Any], *, h: int, w: int) -> float:
    frame_count = max(len(pred_segmentations), len(gt_segmentations))
    scores: List[float] = []
    for frame_index in range(frame_count):
        pred_seg = pred_segmentations[frame_index] if frame_index < len(pred_segmentations) else None
        gt_seg = gt_segmentations[frame_index] if frame_index < len(gt_segmentations) else None
        frame_iou = _frame_iou(pred_seg, gt_seg, h=h, w=w)
        if frame_iou is None:
            continue
        scores.append(float(frame_iou))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _assert_pred_alignment(pred_main: Sequence[Mapping[str, Any]], pred_diag: Sequence[Mapping[str, Any]]) -> None:
    if len(pred_main) != len(pred_diag):
        raise ValueError("pred_main/pred_diag length mismatch")
    for index, (main_row, diag_row) in enumerate(zip(pred_main, pred_diag)):
        if int(diag_row.get("pred_main_index", -1)) != index:
            raise ValueError(f"pred_diag index mismatch at row {index}")
        if str(main_row.get("trajectory_id")) != str(diag_row.get("trajectory_id")):
            raise ValueError(f"trajectory_id alignment mismatch at row {index}")
        if int(main_row.get("video_id", -1)) != int(diag_row.get("video_id", -1)):
            raise ValueError(f"video_id alignment mismatch at row {index}")


def _video_map(videos: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(video["id"]): dict(video) for video in videos}


def _prediction_rows(pred_main: Sequence[Mapping[str, Any]], pred_diag: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for main_row, diag_row in zip(pred_main, pred_diag):
        rows.append(
            {
                "trajectory_id": str(main_row["trajectory_id"]),
                "video_id": int(main_row["video_id"]),
                "segmentations": list(main_row.get("segmentations", [])),
                "pred_category_id": int(diag_row["top1_known_raw_id"]),
                "pred_category_name": str(diag_row.get("top1_known_name", "")),
                "generator_score": float(diag_row.get("generator_score", 0.0)),
                "main_score": float(main_row.get("score", 0.0)),
            }
        )
    return rows


def _ground_truth_rows(gt_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ann in gt_payload.get("annotations", []):
        rows.append(
            {
                "gt_id": int(ann["id"]),
                "video_id": int(ann["video_id"]),
                "category_id": int(ann["category_id"]),
                "segmentations": list(ann.get("segmentations", [])),
            }
        )
    return rows


def _percent(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer / denom)


def _match_predictions(
    prediction_rows: Sequence[Mapping[str, Any]],
    gt_rows: Sequence[Mapping[str, Any]],
    videos_by_id: Mapping[int, Mapping[str, Any]],
    *,
    match_iou_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[int]], Dict[str, List[int]]]:
    match_rows: List[Dict[str, Any]] = []
    gt_to_pred: Dict[int, List[int]] = {int(gt["gt_id"]): [] for gt in gt_rows}
    pred_to_gt_hits: Dict[str, List[int]] = {str(pred["trajectory_id"]): [] for pred in prediction_rows}

    gt_by_video: Dict[int, List[Mapping[str, Any]]] = {}
    for gt_row in gt_rows:
        gt_by_video.setdefault(int(gt_row["video_id"]), []).append(gt_row)

    for pred_row in prediction_rows:
        video_id = int(pred_row["video_id"])
        video = dict(videos_by_id.get(video_id, {}))
        h = int(video.get("height", 0) or 0)
        w = int(video.get("width", 0) or 0)
        candidates = gt_by_video.get(video_id, [])
        best_gt_id = None
        best_gt_category = None
        best_iou = 0.0
        overlap_gt_ids: List[int] = []
        for gt_row in candidates:
            iou = _video_iou(pred_row.get("segmentations", []), gt_row.get("segmentations", []), h=h, w=w)
            if iou >= match_iou_threshold:
                overlap_gt_ids.append(int(gt_row["gt_id"]))
            if iou > best_iou:
                best_iou = float(iou)
                best_gt_id = int(gt_row["gt_id"])
                best_gt_category = int(gt_row["category_id"])
        is_matched = bool(best_gt_id is not None and best_iou >= match_iou_threshold)
        if is_matched:
            gt_to_pred[int(best_gt_id)].append(len(match_rows))
        pred_to_gt_hits[str(pred_row["trajectory_id"])] = list(overlap_gt_ids)
        match_rows.append(
            {
                "trajectory_id": str(pred_row["trajectory_id"]),
                "video_id": int(video_id),
                "pred_category_id": int(pred_row["pred_category_id"]),
                "best_gt_id": int(best_gt_id) if best_gt_id is not None else None,
                "best_gt_category_id": int(best_gt_category) if best_gt_category is not None else None,
                "best_iou": float(best_iou),
                "is_matched": bool(is_matched),
                "is_correct": bool(is_matched and int(pred_row["pred_category_id"]) == int(best_gt_category)),
            }
        )
    return match_rows, gt_to_pred, pred_to_gt_hits


def _build_internal_metrics_payload(
    *,
    matched_predictions: int,
    correct_predictions: int,
    mean_best_iou: float,
    smoke: bool,
) -> Dict[str, Any]:
    payload = {
        "benchmark_name": "lvvis",
        "dataset_name": "lvvis_val",
        "split_tag": "val_smoke" if smoke else "val",
        "matched_predictions": int(matched_predictions),
        "correct_predictions": int(correct_predictions),
        "top1_known_acc": float(_percent(correct_predictions, matched_predictions)),
        "mean_best_iou": float(mean_best_iou),
        "match_iou_threshold": 0.5,
        "metric_valid": bool(matched_predictions > 0),
        "metric_status": "ok" if matched_predictions > 0 else "no_matched_predictions",
    }
    validate_json_artifact(payload, "internal_metrics.schema.json")
    return payload


def _build_companion_payload(
    *,
    smoke: bool,
    match_rows: Sequence[Mapping[str, Any]],
    gt_rows: Sequence[Mapping[str, Any]],
    gt_to_pred: Mapping[int, Sequence[int]],
    pred_to_gt_hits: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    total_gt = int(len(gt_rows))
    total_predictions = int(len(match_rows))
    split_instances = sum(1 for gt_id in gt_to_pred if len(list(gt_to_pred[gt_id])) > 1)
    duplicate_excess = sum(max(0, len(list(gt_to_pred[gt_id])) - 1) for gt_id in gt_to_pred)
    merge_predictions = sum(1 for hits in pred_to_gt_hits.values() if len(list(hits)) > 1)
    unmatched_predictions = sum(1 for row in match_rows if not bool(row["is_matched"]))
    companion = {
        "benchmark_name": "lvvis",
        "dataset_name": "lvvis_val",
        "split_tag": "val_smoke" if smoke else "val",
        "fragmentation_per_gt_instance": float(duplicate_excess / total_gt) if total_gt > 0 else 0.0,
        "split_rate": float(split_instances / total_gt) if total_gt > 0 else 0.0,
        "merge_rate": float(merge_predictions / total_predictions) if total_predictions > 0 else 0.0,
        "duplicate_rate": float(duplicate_excess / total_gt) if total_gt > 0 else 0.0,
        "unmatched_trajectory_ratio": float(unmatched_predictions / total_predictions) if total_predictions > 0 else 0.0,
        "counts": {
            "gt_instances": total_gt,
            "predictions": total_predictions,
            "matched_predictions": int(sum(1 for row in match_rows if bool(row["is_matched"]))),
            "correct_predictions": int(sum(1 for row in match_rows if bool(row["is_correct"]))),
            "split_instances": int(split_instances),
            "merge_predictions": int(merge_predictions),
            "duplicate_excess": int(duplicate_excess),
            "unmatched_predictions": int(unmatched_predictions),
        },
    }
    return companion


def run_internal_eval(config: InternalEvalConfig) -> Dict[str, Any]:
    if config.dataset_name != "lvvis_val":
        raise ValueError("run_stageb_eval_internal only supports lvvis_val")

    paths = G8Paths(config.output_root, config.dataset_name)
    if not paths.pred_main_path.is_file():
        raise FileNotFoundError(f"missing canonical pred_main artifact: {paths.pred_main_path}")
    if not paths.pred_diag_path.is_file():
        raise FileNotFoundError(f"missing canonical pred_diag artifact: {paths.pred_diag_path}")

    pred_main = _read_pred_main(paths.pred_main_path)
    pred_diag = _read_pred_diag(paths.pred_diag_path)
    _assert_pred_alignment(pred_main, pred_diag)

    ann_paths = resolve_lvvis_annotation_paths()
    gt_payload = _read_json(ann_paths.val_json)
    if config.smoke:
        gt_payload = _subset_annotation_payload(gt_payload, [int(row["video_id"]) for row in pred_main])

    videos_by_id = _video_map(gt_payload.get("videos", []))
    prediction_rows = _prediction_rows(pred_main, pred_diag)
    gt_rows = _ground_truth_rows(gt_payload)
    match_rows, gt_to_pred, pred_to_gt_hits = _match_predictions(
        prediction_rows,
        gt_rows,
        videos_by_id,
        match_iou_threshold=0.5,
    )

    matched_rows = [row for row in match_rows if bool(row["is_matched"])]
    correct_rows = [row for row in matched_rows if bool(row["is_correct"])]
    mean_best_iou = float(sum(float(row["best_iou"]) for row in matched_rows) / len(matched_rows)) if matched_rows else 0.0

    payload = _build_internal_metrics_payload(
        matched_predictions=len(matched_rows),
        correct_predictions=len(correct_rows),
        mean_best_iou=mean_best_iou,
        smoke=config.smoke,
    )
    companion = _build_companion_payload(
        smoke=config.smoke,
        match_rows=match_rows,
        gt_rows=gt_rows,
        gt_to_pred=gt_to_pred,
        pred_to_gt_hits=pred_to_gt_hits,
    )
    write_json(paths.internal_metrics_path, payload)
    write_json(paths.internal_companion_metrics_path, companion)

    return {
        "status": "OK",
        "artifact": str(paths.internal_metrics_path),
        "companion_artifact": str(paths.internal_companion_metrics_path),
        "pred_main_path": str(paths.pred_main_path),
        "pred_diag_path": str(paths.pred_diag_path),
        "video_count_evaluated": int(len(videos_by_id)),
        "metrics": {
            "matched_predictions": int(payload["matched_predictions"]),
            "correct_predictions": int(payload["correct_predictions"]),
            "top1_known_acc": float(payload["top1_known_acc"]),
            "mean_best_iou": float(payload["mean_best_iou"]),
            "fragmentation_per_gt_instance": float(companion["fragmentation_per_gt_instance"]),
            "split_rate": float(companion["split_rate"]),
            "merge_rate": float(companion["merge_rate"]),
            "duplicate_rate": float(companion["duplicate_rate"]),
            "unmatched_trajectory_ratio": float(companion["unmatched_trajectory_ratio"]),
        },
    }
