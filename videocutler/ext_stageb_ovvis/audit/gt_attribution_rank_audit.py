from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_combined_evidence
from videocutler.ext_stageb_ovvis.eval.external_lvvis import resolve_lvvis_annotation_paths
from videocutler.ext_stageb_ovvis.eval.external_ytvis2019 import resolve_ytvis2019_annotation_paths
from videocutler.ext_stageb_ovvis.eval.g8_bridge import (
    G8Paths,
    InferenceAssetRoots,
    ProjectorBundle,
    build_infer_rows,
    canonical_checkpoint_relpath,
    compute_fused_logits_chunked,
    load_json,
    load_projector_bundle,
    load_text_vocab_with_names,
    load_video_meta,
    resolve_inference_asset_roots,
    resolve_selected_for_infer,
    write_json,
)

ALLOWED_DATASETS = ("lvvis_val", "ytvis_2019_val")
STAGE_TO_SELECTED = {
    "prealign": "prealign_only",
    "softem_base": "base_only",
    "softem_aug": "augmented",
}
ALL_STAGES = ("prealign", "softem_base", "softem_aug")


@dataclass(frozen=True)
class StageAuditResult:
    dataset_name: str
    stage: str
    stage_status: str
    class_space_size: int
    total_prediction_count: int
    matched_prediction_count: int
    match_rate: float
    mean_normalized_gt_rank: Optional[float]
    gt_top1_hit_rate: Optional[float]
    checkpoint_path: Optional[str]
    ledger_path: Optional[str]
    note: Optional[str] = None


@dataclass(frozen=True)
class GTAttributionRankAuditConfig:
    dataset_name: str
    output_root: Path
    stage: str
    device: torch.device
    logit_chunk_size: int = 256
    trajectory_source_branch: str = "mainline"


def _require_dataset_name(dataset_name: str) -> str:
    if dataset_name not in ALLOWED_DATASETS:
        raise ValueError(f"dataset_name must be one of {ALLOWED_DATASETS}, got {dataset_name!r}")
    return dataset_name


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_pred_main(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    return [dict(row) for row in payload]


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
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception:
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


def _load_gt_payload(dataset_name: str) -> Dict[str, Any]:
    if dataset_name == "lvvis_val":
        return load_json(resolve_lvvis_annotation_paths().val_json)
    if dataset_name == "ytvis_2019_val":
        return load_json(resolve_ytvis2019_annotation_paths().val_json)
    raise ValueError(f"unsupported dataset_name: {dataset_name}")


def _video_map(videos: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(video["id"]): dict(video) for video in videos}


def _prediction_rows(pred_main: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for main_row in pred_main:
        rows.append(
            {
                "trajectory_id": str(main_row["trajectory_id"]),
                "video_id": int(main_row["video_id"]),
                "segmentations": list(main_row.get("segmentations", [])),
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


def _match_predictions(
    prediction_rows: Sequence[Mapping[str, Any]],
    gt_rows: Sequence[Mapping[str, Any]],
    videos_by_id: Mapping[int, Mapping[str, Any]],
    *,
    match_iou_threshold: float,
) -> List[Dict[str, Any]]:
    match_rows: List[Dict[str, Any]] = []
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
        for gt_row in candidates:
            iou = _video_iou(pred_row.get("segmentations", []), gt_row.get("segmentations", []), h=h, w=w)
            if iou > best_iou:
                best_iou = float(iou)
                best_gt_id = int(gt_row["gt_id"])
                best_gt_category = int(gt_row["category_id"])
        is_matched = bool(best_gt_id is not None and best_iou >= match_iou_threshold)
        match_rows.append(
            {
                "trajectory_id": str(pred_row["trajectory_id"]),
                "video_id": int(video_id),
                "best_gt_id": int(best_gt_id) if best_gt_id is not None else None,
                "best_gt_category_id": int(best_gt_category) if best_gt_category is not None else None,
                "best_iou": float(best_iou),
                "is_matched": bool(is_matched),
            }
        )
    return match_rows


def _dataset_vocab(asset_root: Path, dataset_name: str, gt_payload: Mapping[str, Any]) -> Tuple[List[int], np.ndarray]:
    raw_ids, _records, matrix, _class_name_map = load_text_vocab_with_names(asset_root, dataset_name)
    raw_ids = [int(x) for x in raw_ids]
    index_by_id = {raw_id: idx for idx, raw_id in enumerate(raw_ids)}
    dataset_ids = [int(cat["id"]) for cat in gt_payload.get("categories", [])]
    missing = [raw_id for raw_id in dataset_ids if raw_id not in index_by_id]
    if missing:
        raise ValueError(f"dataset categories missing from text vocab: {missing[:16]}")
    dataset_ids_sorted = sorted(dataset_ids)
    indices = np.asarray([index_by_id[raw_id] for raw_id in dataset_ids_sorted], dtype=np.int64)
    return dataset_ids_sorted, np.asarray(matrix[indices], dtype=np.float32)


def _score_row_full_vocab(
    row: Mapping[str, Any],
    *,
    bundle: ProjectorBundle,
    asset_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
    vocab_matrix: np.ndarray,
    logit_chunk_size: int,
) -> np.ndarray:
    carrier_vec, frame_vectors, frame_vec, _combined = load_combined_evidence(
        row,
        output_root=asset_root,
        dataset_name=dataset_name,
        trajectory_source_branch=trajectory_source_branch,
    )
    _carrier_logits, _frame_logits, fused_logits = compute_fused_logits_chunked(
        projector=bundle.projector,
        carrier_vec=carrier_vec,
        frame_vec=frame_vec,
        candidate_matrix=vocab_matrix,
        temperature=float(bundle.temperature),
        frame_vectors=frame_vectors,
        logit_chunk_size=logit_chunk_size,
    )
    return np.asarray(fused_logits, dtype=np.float32)


def _rank_and_top1(logits: np.ndarray, gt_index: int) -> Tuple[int, float, int]:
    gt_score = float(logits[gt_index])
    rank = int(np.count_nonzero(logits > gt_score) + 1)
    denom = max(1, int(logits.shape[0]) - 1)
    normalized_rank = float((rank - 1) / denom)
    top1 = int(int(np.argmax(logits)) == int(gt_index))
    return rank, normalized_rank, top1


def _match_rate(matched: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(matched / total)


def _resolve_asset_roots(output_root: Path) -> InferenceAssetRoots:
    resolution = resolve_selected_for_infer(output_root)
    return resolve_inference_asset_roots(
        output_root,
        dataset_name=resolution.train_state_payload.get("dataset_name", "lvvis_val") if resolution.train_state_payload else "lvvis_val",
        trajectory_source_branch="mainline",
        resolution=resolution,
    )


def _resolve_asset_roots_for_dataset(output_root: Path, dataset_name: str) -> InferenceAssetRoots:
    resolution = resolve_selected_for_infer(output_root)
    return resolve_inference_asset_roots(
        output_root,
        dataset_name=dataset_name,
        trajectory_source_branch="mainline",
        resolution=resolution,
    )


def _stage_checkpoint_path(output_root: Path, stage: str) -> Path:
    selected = STAGE_TO_SELECTED[stage]
    return (output_root / canonical_checkpoint_relpath(selected)).resolve()


def _stage_ledger_path(output_root: Path, dataset_name: str, stage: str) -> Path:
    return output_root / "audit" / "gt_attribution_rank" / dataset_name / f"{stage}_ledger.jsonl"


def _stage_summary_path(output_root: Path, dataset_name: str, stage: str) -> Path:
    return output_root / "audit" / "gt_attribution_rank" / dataset_name / f"{stage}_summary.json"


def _package_summary_path(output_root: Path, dataset_name: str) -> Path:
    return output_root / "audit" / "gt_attribution_rank" / dataset_name / "summary.json"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def run_stage_gt_attribution_rank_audit(config: GTAttributionRankAuditConfig, stage: str) -> StageAuditResult:
    _require_dataset_name(config.dataset_name)
    if stage not in ALL_STAGES:
        raise ValueError(f"stage must be one of {ALL_STAGES}, got {stage!r}")

    checkpoint_path = _stage_checkpoint_path(config.output_root, stage)
    if not checkpoint_path.is_file():
        result = StageAuditResult(
            dataset_name=config.dataset_name,
            stage=stage,
            stage_status="STAGE_NOT_PRESENT",
            class_space_size=0,
            total_prediction_count=0,
            matched_prediction_count=0,
            match_rate=0.0,
            mean_normalized_gt_rank=None,
            gt_top1_hit_rate=None,
            checkpoint_path=str(checkpoint_path),
            ledger_path=None,
            note="checkpoint missing for requested stage",
        )
        write_json(_stage_summary_path(config.output_root, config.dataset_name, stage), result.__dict__)
        return result

    paths = G8Paths(config.output_root, config.dataset_name)
    pred_main = _read_pred_main(paths.pred_main_path)
    gt_payload = _load_gt_payload(config.dataset_name)
    videos_by_id = _video_map(gt_payload.get("videos", []))
    prediction_rows = _prediction_rows(pred_main)
    gt_rows = _ground_truth_rows(gt_payload)
    match_rows = _match_predictions(prediction_rows, gt_rows, videos_by_id, match_iou_threshold=0.5)
    matched_rows = [row for row in match_rows if bool(row.get("is_matched"))]

    asset_roots = _resolve_asset_roots_for_dataset(config.output_root, config.dataset_name)
    infer_rows, _skipped, _asset_counts = build_infer_rows(asset_roots, dataset_name=config.dataset_name)
    infer_row_by_tid = {str(row["trajectory_id"]): row for row in infer_rows}

    vocab_ids, vocab_matrix = _dataset_vocab(asset_roots.asset_root, config.dataset_name, gt_payload)
    vocab_index = {raw_id: idx for idx, raw_id in enumerate(vocab_ids)}
    bundle = load_projector_bundle(checkpoint_path, device=config.device)

    ledger_rows: List[Dict[str, Any]] = []
    normalized_ranks: List[float] = []
    top1_values: List[int] = []
    for row in matched_rows:
        trajectory_id = str(row["trajectory_id"])
        infer_row = infer_row_by_tid.get(trajectory_id)
        if infer_row is None:
            raise KeyError(f"matched trajectory_id missing from infer rows: {trajectory_id}")
        gt_raw_id = int(row["best_gt_category_id"])
        if gt_raw_id not in vocab_index:
            raise KeyError(f"matched GT raw id {gt_raw_id} missing from dataset vocab")
        logits = _score_row_full_vocab(
            infer_row,
            bundle=bundle,
            asset_root=asset_roots.asset_root,
            dataset_name=config.dataset_name,
            trajectory_source_branch=config.trajectory_source_branch,
            vocab_matrix=vocab_matrix,
            logit_chunk_size=config.logit_chunk_size,
        )
        rank, normalized_rank, top1 = _rank_and_top1(logits, vocab_index[gt_raw_id])
        normalized_ranks.append(float(normalized_rank))
        top1_values.append(int(top1))
        ledger_rows.append(
            {
                "trajectory_id": trajectory_id,
                "video_id": int(row["video_id"]),
                "gt_id": int(row["best_gt_id"]),
                "gt_class_id": int(gt_raw_id),
                "stage": stage,
                "rank": int(rank),
                "normalized_rank": float(normalized_rank),
                "top1": int(top1),
            }
        )

    ledger_path = _stage_ledger_path(config.output_root, config.dataset_name, stage)
    _write_jsonl(ledger_path, ledger_rows)
    mean_rank = float(sum(normalized_ranks) / len(normalized_ranks)) if normalized_ranks else None
    top1_rate = float(sum(top1_values) / len(top1_values)) if top1_values else None
    result = StageAuditResult(
        dataset_name=config.dataset_name,
        stage=stage,
        stage_status="STAGE_PRESENT",
        class_space_size=len(vocab_ids),
        total_prediction_count=len(match_rows),
        matched_prediction_count=len(matched_rows),
        match_rate=_match_rate(len(matched_rows), len(match_rows)),
        mean_normalized_gt_rank=mean_rank,
        gt_top1_hit_rate=top1_rate,
        checkpoint_path=str(checkpoint_path),
        ledger_path=str(ledger_path),
        note=None,
    )
    write_json(_stage_summary_path(config.output_root, config.dataset_name, stage), result.__dict__)
    return result


def run_gt_attribution_rank_audit(config: GTAttributionRankAuditConfig) -> Dict[str, Any]:
    stage_names = ALL_STAGES if config.stage == "all" else (config.stage,)
    results: Dict[str, Any] = {}
    for stage in stage_names:
        results[stage] = run_stage_gt_attribution_rank_audit(config, stage).__dict__
    summary = {
        "dataset_name": config.dataset_name,
        "output_root": str(config.output_root),
        "stages": results,
    }
    write_json(_package_summary_path(config.output_root, config.dataset_name), summary)
    return summary
