from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_combined_evidence
from videocutler.ext_stageb_ovvis.audit.dropped_gt_attribution_audit import _as_int, _load_lvvis_split_reference
from videocutler.ext_stageb_ovvis.audit.extra_recovery_audit import _load_or_generate_gt_sidecar_lookup
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
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
    resolve_runtime_assets,
)

TRAIN_DATASETS = ("lvvis_train_base",)
VAL_BASE_NOVEL_DATASETS = ("lvvis_val", "ytvis_2019_val")
ALLOWED_DATASETS = TRAIN_DATASETS + VAL_BASE_NOVEL_DATASETS
STAGE_TO_SELECTED = {
    "prealign": "prealign_only",
    "softem_base": "base_only",
    "softem_aug": "augmented",
}
ALL_STAGES = ("prealign", "softem_base", "softem_aug")
ALL_GT_TRAIN_SPLIT_ORDER = ("base_observed", "base_unobserved")
ALL_GT_VAL_SPLIT_ORDER = ("base", "novel")


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
    all_gt_only: bool = False
    all_gt_generate_sidecars_if_missing: bool = False
    all_gt_heartbeat_every_rows: int = 256


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
    if dataset_name == "lvvis_train_base":
        return load_json(resolve_lvvis_annotation_paths().train_json)
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


def _score_all_gt_rows(
    *,
    stage: str,
    materialized_samples: Sequence[Mapping[str, Any]],
    gt_sidecar_lookup: Mapping[str, Mapping[str, Any]],
    full_vocab_ids: Sequence[int],
    vocab_matrix: np.ndarray,
    bundle: ProjectorBundle,
    asset_roots: InferenceAssetRoots,
    dataset_name: str,
    trajectory_source_branch: str,
    logit_chunk_size: int,
    base_vocab_ids: Sequence[int],
    progress_callback: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_vocab = {int(x) for x in base_vocab_ids}
    vocab_index = {int(raw_id): idx for idx, raw_id in enumerate(full_vocab_ids)}
    rows: List[Dict[str, Any]] = []
    unsupported_hist: Dict[str, int] = {}
    total_rows = int(len(materialized_samples))
    for row_index, sample in enumerate(materialized_samples, start=1):
        trajectory_id = str(sample.get("trajectory_id", "")).strip()
        sidecar = gt_sidecar_lookup.get(trajectory_id, {})
        gt_raw_id = _as_int(sidecar.get("matched_gt_class_id"))
        gt_available = bool(sidecar.get("audit_usable", False)) and gt_raw_id is not None
        observed_raw_ids = [int(x) for x in list(sample.get("observed_raw_ids", []))]
        split_label = None
        if gt_available and gt_raw_id is not None:
            split_label = _all_gt_split_label(
                dataset_name=dataset_name,
                gt_raw_id=int(gt_raw_id),
                observed_raw_ids=observed_raw_ids,
                base_vocab_ids=base_vocab,
            )
            if split_label is None:
                unsupported_hist["unsupported_split"] = unsupported_hist.get("unsupported_split", 0) + 1

        row: Dict[str, Any] = {
            "stage_id": str(stage),
            "trajectory_id": trajectory_id,
            "clip_id": _as_int(sample.get("clip_id"))
            if _as_int(sample.get("clip_id")) is not None
            else (
                int(sample.get("trajectory_record", {}).get("clip_id"))
                if isinstance(sample.get("trajectory_record"), Mapping) and sample.get("trajectory_record", {}).get("clip_id") is not None
                else None
            ),
            "video_id": _as_int(sample.get("trajectory_record", {}).get("video_id")) if isinstance(sample.get("trajectory_record"), Mapping) else None,
            "observed_raw_ids": observed_raw_ids,
            "gt_available_for_audit": gt_available,
            "gt_class_id": gt_raw_id,
            "all_gt_split": split_label,
            "gt_rank": None,
            "normalized_gt_rank": None,
            "gt_is_top1": False,
            "gt_top5": False,
            "gt_top10": False,
            "mrr": None,
            "margin_to_best_wrong": None,
            "stage_top1_id": None,
            "wrong_top1_is_base": False,
        }

        if not gt_available or gt_raw_id is None or split_label is None:
            rows.append(row)
            continue

        gt_index = vocab_index.get(int(gt_raw_id))
        if gt_index is None:
            rows.append(row)
            continue

        logits = _score_row_full_vocab(
            sample,
            bundle=bundle,
            asset_root=asset_roots.asset_root,
            dataset_name=dataset_name,
            trajectory_source_branch=trajectory_source_branch,
            vocab_matrix=vocab_matrix,
            logit_chunk_size=logit_chunk_size,
        )
        rank_payload = _rank_metrics_from_logits(logits, int(gt_index))
        top1_id = rank_payload["top1_id"]
        row.update(
            {
                "gt_rank": int(rank_payload["rank"]),
                "normalized_gt_rank": float(rank_payload["normalized_rank"]),
                "gt_is_top1": bool(rank_payload["top1"]),
                "gt_top5": bool(rank_payload["top5"]),
                "gt_top10": bool(rank_payload["top10"]),
                "mrr": float(rank_payload["mrr"]),
                "margin_to_best_wrong": rank_payload["margin_to_best_wrong"],
                "stage_top1_id": top1_id,
                "wrong_top1_is_base": bool(top1_id is not None and int(top1_id) != int(gt_raw_id) and int(top1_id) in base_vocab),
            }
        )
        rows.append(row)
        if progress_callback is not None and (row_index == total_rows or row_index % max(1, int(progress_callback.__dict__.get("heartbeat_every", 256))) == 0):
            progress_callback(row_index, total_rows)

    summary = _summarize_all_gt_subset(rows, stage_id=stage, split_order=split_order)
    summary["unsupported_gt_histogram"] = dict(sorted(unsupported_hist.items()))
    return rows, summary


def _rank_and_top1(logits: np.ndarray, gt_index: int) -> Tuple[int, float, int]:
    gt_score = float(logits[gt_index])
    rank = int(np.count_nonzero(logits > gt_score) + 1)
    denom = max(1, int(logits.shape[0]) - 1)
    normalized_rank = float((rank - 1) / denom)
    top1 = int(int(np.argmax(logits)) == int(gt_index))
    return rank, normalized_rank, top1


def _rank_metrics_from_logits(logits: np.ndarray, gt_index: int) -> Dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float32)
    gt_score = float(logits[int(gt_index)])
    ordered = sorted(((int(idx), float(score)) for idx, score in enumerate(logits)), key=lambda item: (-item[1], item[0]))
    rank = 1 + sum(1 for _idx, score in ordered if float(score) > gt_score)
    denom = max(1, int(logits.shape[0]) - 1)
    normalized_rank = float((rank - 1) / denom)
    top1_id = int(ordered[0][0]) if ordered else None
    best_wrong = next((float(score) for idx, score in ordered if int(idx) != int(gt_index)), None)
    return {
        "rank": int(rank),
        "normalized_rank": float(normalized_rank),
        "top1": bool(top1_id is not None and int(top1_id) == int(gt_index)),
        "top5": bool(rank <= 5),
        "top10": bool(rank <= 10),
        "mrr": float(1.0 / rank),
        "margin_to_best_wrong": float(gt_score - best_wrong) if best_wrong is not None else None,
        "top1_id": top1_id,
    }


def _match_rate(matched: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(matched / total)


def _all_gt_split_order_for_dataset(dataset_name: str) -> Tuple[str, ...]:
    if dataset_name in TRAIN_DATASETS:
        return ALL_GT_TRAIN_SPLIT_ORDER
    if dataset_name in VAL_BASE_NOVEL_DATASETS:
        return ALL_GT_VAL_SPLIT_ORDER
    raise ValueError(f"unsupported dataset_name for all-GT split order: {dataset_name}")


def _all_gt_split_label(*, dataset_name: str, gt_raw_id: int, observed_raw_ids: Sequence[int], base_vocab_ids: Sequence[int]) -> Optional[str]:
    base_vocab = {int(x) for x in base_vocab_ids}
    observed_vocab = {int(x) for x in observed_raw_ids}
    gt_raw_id = int(gt_raw_id)
    if dataset_name in TRAIN_DATASETS:
        if gt_raw_id in base_vocab and gt_raw_id in observed_vocab:
            return "base_observed"
        if gt_raw_id in base_vocab and gt_raw_id not in observed_vocab:
            return "base_unobserved"
        return None
    if dataset_name in VAL_BASE_NOVEL_DATASETS:
        if gt_raw_id in base_vocab:
            return "base"
        return "novel"
    raise ValueError(f"unsupported dataset_name for all-GT split label: {dataset_name}")


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


def _stage_all_gt_summary_by_split_path(output_root: Path, dataset_name: str, stage: str) -> Path:
    return output_root / "audit" / "gt_attribution_rank" / dataset_name / f"{stage}_all_gt_summary_by_split.json"


def _package_summary_path(output_root: Path, dataset_name: str) -> Path:
    return output_root / "audit" / "gt_attribution_rank" / dataset_name / "summary.json"


def _package_all_gt_summary_by_split_path(output_root: Path, dataset_name: str) -> Path:
    return output_root / "audit" / "gt_attribution_rank" / dataset_name / "all_gt_summary_by_split.json"


def _package_all_gt_comparison_by_split_path(output_root: Path, dataset_name: str) -> Path:
    return output_root / "audit" / "gt_attribution_rank" / dataset_name / "gt_attribution_rank_all_gt_comparison_by_split.json"


def _package_all_gt_summary_export_path(output_root: Path, dataset_name: str) -> Path:
    return output_root / "audit" / "gt_attribution_rank" / dataset_name / "gt_attribution_rank_all_gt_summary_by_split.json"


def _stage_all_gt_progress_path(output_root: Path, dataset_name: str, stage: str) -> Path:
    return output_root / "audit" / "gt_attribution_rank" / dataset_name / f"{stage}_all_gt_progress.json"


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _load_base_vocab_ids_for_dataset(dataset_name: str, gt_payload: Mapping[str, Any]) -> List[int]:
    if dataset_name in {"lvvis_train_base", "lvvis_val"}:
        base_vocab_ids, _novel_vocab_reference = _load_lvvis_split_reference()
        return [int(x) for x in base_vocab_ids]
    return sorted({int(cat["id"]) for cat in gt_payload.get("categories", [])})


def _load_or_generate_gt_sidecar_lookup_cached(
    *,
    output_root: Path,
    dataset_name: str,
    clip_ids: Sequence[int],
    generate_if_missing: bool,
) -> Dict[str, Mapping[str, Any]]:
    lookup = _load_or_generate_gt_sidecar_lookup(
        output_root=output_root,
        dataset_name=dataset_name,
        clip_ids=clip_ids,
        generate_sidecars=False,
    )
    if lookup or not generate_if_missing:
        return lookup
    return _load_or_generate_gt_sidecar_lookup(
        output_root=output_root,
        dataset_name=dataset_name,
        clip_ids=clip_ids,
        generate_sidecars=True,
    )


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _prepare_all_gt_val_samples(config: GTAttributionRankAuditConfig) -> List[Dict[str, Any]]:
    resolution = resolve_runtime_assets(
        config.output_root,
        dataset_name=config.dataset_name,
        trajectory_source_branch=config.trajectory_source_branch,
    )
    runtime_output_root = Path(str(resolution["runtime_output_root"]))
    assets = resolution["assets"]
    trajectory_records = sorted(
        _load_jsonl_records(runtime_output_root / assets["trajectory_records"]["path"]),
        key=lambda rec: str(rec.get("trajectory_id", "")),
    )
    carrier_records = _load_jsonl_records(runtime_output_root / assets["carrier_records"]["path"])
    carrier_by_tid = {str(rec.get("trajectory_id", "")): rec for rec in carrier_records}
    samples: List[Dict[str, Any]] = []
    for traj in trajectory_records:
        trajectory_id = str(traj.get("trajectory_id", "")).strip()
        if not trajectory_id:
            continue
        carrier_rec = carrier_by_tid.get(trajectory_id)
        if carrier_rec is None:
            continue
        samples.append(
            {
                "trajectory_id": trajectory_id,
                "clip_id": str(traj.get("clip_id", "")),
                "trajectory_record": dict(traj),
                "carrier_record": dict(carrier_rec),
                "weak_label_record": None,
                "candidate_text_prototypes": [],
                "observed_raw_ids": [],
                "observed_set_semantics": "not_applicable_val_base_novel_only",
                "observed_set_source": "not_applicable_val_base_novel_only",
                "candidate_ids_known": [],
                "candidate_ids_extra": [],
                "sample_valid": True,
            }
        )
    return samples


def _prepare_all_gt_shared_inputs(config: GTAttributionRankAuditConfig) -> Dict[str, Any]:
    if config.dataset_name in TRAIN_DATASETS:
        materialized = materialize_phase1_training_samples(
            config.output_root,
            Phase1MaterializationConfig(
                dataset_name=config.dataset_name,
                trajectory_source_branch=config.trajectory_source_branch,
                smoke=False,
            ),
        )
        samples = [dict(x) for x in materialized["samples"] if bool(x.get("sample_valid", False))]
    elif config.dataset_name in VAL_BASE_NOVEL_DATASETS:
        samples = _prepare_all_gt_val_samples(config)
    else:
        raise ValueError(f"unsupported dataset_name for all-GT preparation: {config.dataset_name}")
    clip_ids = sorted(
        {
            int(sample.get("trajectory_record", {}).get("clip_id"))
            for sample in samples
            if isinstance(sample.get("trajectory_record"), Mapping) and sample.get("trajectory_record", {}).get("clip_id") is not None
        }
    )
    gt_sidecar_lookup = _load_or_generate_gt_sidecar_lookup_cached(
        output_root=config.output_root,
        dataset_name=config.dataset_name,
        clip_ids=[int(x) for x in clip_ids],
        generate_if_missing=bool(config.all_gt_generate_sidecars_if_missing),
    )
    asset_roots = _resolve_asset_roots_for_dataset(config.output_root, config.dataset_name)
    gt_payload = _load_gt_payload(config.dataset_name)
    full_vocab_ids, vocab_matrix = _dataset_vocab(asset_roots.asset_root, config.dataset_name, gt_payload)
    base_vocab_ids = _load_base_vocab_ids_for_dataset(config.dataset_name, gt_payload)
    observed_sources = sorted({str(sample.get("observed_set_source", "unknown")) for sample in samples})
    observed_semantics = sorted({str(sample.get("observed_set_semantics", "unknown")) for sample in samples})
    return {
        "samples": samples,
        "clip_ids": clip_ids,
        "gt_sidecar_lookup": gt_sidecar_lookup,
        "asset_roots": asset_roots,
        "gt_payload": gt_payload,
        "full_vocab_ids": full_vocab_ids,
        "vocab_matrix": vocab_matrix,
        "base_vocab_ids": base_vocab_ids,
        "observed_set_sources": observed_sources,
        "observed_set_semantics": observed_semantics,
    }


def _write_all_gt_progress(
    output_root: Path,
    dataset_name: str,
    stage: str,
    *,
    processed_rows: int,
    total_rows: int,
    checkpoint_path: Path,
    status: str,
) -> None:
    write_json(
        _stage_all_gt_progress_path(output_root, dataset_name, stage),
        {
            "dataset_name": dataset_name,
            "stage": stage,
            "status": status,
            "processed_rows": int(processed_rows),
            "total_rows": int(total_rows),
            "progress": float(processed_rows / total_rows) if total_rows > 0 else 1.0,
            "checkpoint_path": str(checkpoint_path),
        },
    )


def _build_all_gt_comparison_by_split(results: Mapping[str, Mapping[str, Any]], *, split_order: Sequence[str]) -> Dict[str, Any]:
    by_split: Dict[str, Dict[str, Any]] = {}
    for split in split_order:
        by_split[split] = {}
        for stage, stage_summary in results.items():
            split_summary = dict(stage_summary.get("split_summaries", {}).get(split, {}))
            by_split[split][stage] = {
                "gt_count": split_summary.get("gt_count"),
                "mean_normalized_gt_rank": split_summary.get("mean_normalized_gt_rank"),
                "gt_top1_hit_rate": split_summary.get("gt_top1_hit_rate"),
                "status": split_summary.get("status"),
            }
    return {
        "split_order": [str(x) for x in split_order],
        "stage_order": list(results.keys()),
        "by_split": by_split,
    }


def _summarize_all_gt_subset(rows: Sequence[Mapping[str, Any]], *, stage_id: str, split_order: Sequence[str]) -> Dict[str, Any]:
    gt_rows = [row for row in rows if bool(row.get("gt_available_for_audit")) and row.get("gt_rank") is not None and row.get("normalized_gt_rank") is not None]
    unsupported_rows = [row for row in rows if not str(row.get("all_gt_split", "")).strip()]
    split_summaries: Dict[str, Any] = {}
    split_counts: Dict[str, int] = {}
    for split in split_order:
        split_rows = [row for row in gt_rows if str(row.get("all_gt_split", "")).strip() == split]
        split_counts[split] = int(len(split_rows))
        if not split_rows:
            split_summaries[split] = {
                "gt_count": 0,
                "match_rate": None,
                "mean_normalized_gt_rank": None,
                "gt_top1_hit_rate": None,
                "gt_top5_hit_rate": None,
                "gt_top10_hit_rate": None,
                "mrr": None,
                "margin_to_best_wrong_mean": None,
                "status": "EMPTY",
            }
            continue
        normalized_ranks = [float(row["normalized_gt_rank"]) for row in split_rows]
        top1s = [1.0 if bool(row.get("gt_is_top1")) else 0.0 for row in split_rows]
        top5s = [1.0 if bool(row.get("gt_rank") is not None and int(row["gt_rank"]) <= 5) else 0.0 for row in split_rows]
        top10s = [1.0 if bool(row.get("gt_rank") is not None and int(row["gt_rank"]) <= 10) else 0.0 for row in split_rows]
        mrrs = [float(row["mrr"]) for row in split_rows if row.get("mrr") is not None]
        margins = [float(row["margin_to_best_wrong"]) for row in split_rows if row.get("margin_to_best_wrong") is not None]
        split_summaries[split] = {
            "gt_count": int(len(split_rows)),
            "match_rate": float(len(split_rows) / len(gt_rows)) if gt_rows else None,
            "mean_normalized_gt_rank": float(np.mean(np.asarray(normalized_ranks, dtype=np.float64))),
            "gt_top1_hit_rate": float(np.mean(np.asarray(top1s, dtype=np.float64))),
            "gt_top5_hit_rate": float(np.mean(np.asarray(top5s, dtype=np.float64))),
            "gt_top10_hit_rate": float(np.mean(np.asarray(top10s, dtype=np.float64))),
            "mrr": float(np.mean(np.asarray(mrrs, dtype=np.float64))) if mrrs else None,
            "margin_to_best_wrong_mean": float(np.mean(np.asarray(margins, dtype=np.float64))) if margins else None,
            "status": "PASS",
        }
    return {
        "stage_id": str(stage_id),
        "status": "PASS" if rows else "EMPTY",
        "row_count": int(len(rows)),
        "gt_available_row_count": int(len(gt_rows)),
        "gt_count": int(len(gt_rows)),
        "match_rate": float(len(gt_rows) / len(rows)) if rows else 0.0,
        "mean_normalized_gt_rank": float(np.mean(np.asarray([float(row["normalized_gt_rank"]) for row in gt_rows], dtype=np.float64))) if gt_rows else None,
        "gt_top1_hit_rate": float(np.mean(np.asarray([1.0 if bool(row.get("gt_is_top1")) else 0.0 for row in gt_rows], dtype=np.float64))) if gt_rows else None,
        "gt_top5_hit_rate": float(np.mean(np.asarray([1.0 if bool(row.get("gt_rank") is not None and int(row["gt_rank"]) <= 5) else 0.0 for row in gt_rows], dtype=np.float64))) if gt_rows else None,
        "gt_top10_hit_rate": float(np.mean(np.asarray([1.0 if bool(row.get("gt_rank") is not None and int(row["gt_rank"]) <= 10) else 0.0 for row in gt_rows], dtype=np.float64))) if gt_rows else None,
        "mrr": float(np.mean(np.asarray([float(row["mrr"]) for row in gt_rows if row.get("mrr") is not None], dtype=np.float64))) if gt_rows else None,
        "wrong_top1_is_base_rate": float(sum(1 for row in gt_rows if bool(row.get("wrong_top1_is_base"))) / len(gt_rows)) if gt_rows else None,
        "margin_to_best_wrong_mean": float(np.mean(np.asarray([float(row["margin_to_best_wrong"]) for row in gt_rows if row.get("margin_to_best_wrong") is not None], dtype=np.float64))) if gt_rows else None,
        "split_counts": split_counts,
        "split_summaries": split_summaries,
        "unsupported_gt_row_count": int(len(unsupported_rows)),
    }


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


def run_stage_all_gt_attribution_rank_audit(
    config: GTAttributionRankAuditConfig,
    stage: str,
    *,
    prepared_inputs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    _require_dataset_name(config.dataset_name)
    if stage not in ALL_STAGES:
        raise ValueError(f"stage must be one of {ALL_STAGES}, got {stage!r}")

    checkpoint_path = _stage_checkpoint_path(config.output_root, stage)
    summary_path = _stage_all_gt_summary_by_split_path(config.output_root, config.dataset_name, stage)
    split_order = _all_gt_split_order_for_dataset(config.dataset_name)
    observed_set_sources = list((prepared_inputs or {}).get("observed_set_sources", []))
    observed_set_semantics = list((prepared_inputs or {}).get("observed_set_semantics", []))
    if not checkpoint_path.is_file():
        result = {
            "dataset_name": config.dataset_name,
            "stage": stage,
            "stage_status": "STAGE_NOT_PRESENT",
            "class_space_size": 0,
            "row_count": 0,
            "gt_available_row_count": 0,
            "gt_count": 0,
            "match_rate": 0.0,
            "mean_normalized_gt_rank": None,
            "gt_top1_hit_rate": None,
            "gt_top5_hit_rate": None,
            "gt_top10_hit_rate": None,
            "mrr": None,
            "wrong_top1_is_base_rate": None,
            "margin_to_best_wrong_mean": None,
            "checkpoint_path": str(checkpoint_path),
            "summary_by_split_path": str(summary_path),
            "note": "checkpoint missing for requested stage",
            "split_counts": {split: 0 for split in split_order},
            "split_summaries": {split: {"gt_count": 0, "status": "STAGE_NOT_PRESENT"} for split in split_order},
            "unsupported_gt_histogram": {},
            "observed_set_sources": observed_set_sources,
            "observed_set_semantics": observed_set_semantics,
        }
        write_json(summary_path, result)
        return result

    prepared = dict(prepared_inputs or _prepare_all_gt_shared_inputs(config))
    samples = [dict(x) for x in prepared["samples"]]
    observed_set_sources = list(prepared.get("observed_set_sources", []))
    observed_set_semantics = list(prepared.get("observed_set_semantics", []))
    gt_sidecar_lookup = dict(prepared["gt_sidecar_lookup"])
    asset_roots = prepared["asset_roots"]
    full_vocab_ids = list(prepared["full_vocab_ids"])
    vocab_matrix = np.asarray(prepared["vocab_matrix"], dtype=np.float32)
    base_vocab_ids = list(prepared["base_vocab_ids"])

    bundle = load_projector_bundle(checkpoint_path, device=config.device)

    def _progress_callback(processed_rows: int, total_rows: int) -> None:
        _write_all_gt_progress(
            config.output_root,
            config.dataset_name,
            stage,
            processed_rows=processed_rows,
            total_rows=total_rows,
            checkpoint_path=checkpoint_path,
            status="RUNNING",
        )

    _progress_callback.__dict__["heartbeat_every"] = int(config.all_gt_heartbeat_every_rows)

    _write_all_gt_progress(
        config.output_root,
        config.dataset_name,
        stage,
        processed_rows=0,
        total_rows=len(samples),
        checkpoint_path=checkpoint_path,
        status="RUNNING",
    )

    rows, summary = _score_all_gt_rows(
        stage=stage,
        materialized_samples=samples,
        gt_sidecar_lookup=gt_sidecar_lookup,
        full_vocab_ids=full_vocab_ids,
        vocab_matrix=vocab_matrix,
        bundle=bundle,
        asset_roots=asset_roots,
        dataset_name=config.dataset_name,
        trajectory_source_branch=config.trajectory_source_branch,
        logit_chunk_size=config.logit_chunk_size,
        base_vocab_ids=base_vocab_ids,
        progress_callback=_progress_callback,
    )
    summary.update(
        {
            "dataset_name": config.dataset_name,
            "stage": stage,
            "stage_status": "STAGE_PRESENT",
            "class_space_size": len(full_vocab_ids),
            "checkpoint_path": str(checkpoint_path),
            "summary_by_split_path": str(summary_path),
            "legacy_summary_path": str(_stage_summary_path(config.output_root, config.dataset_name, stage)),
            "ledger_path": None,
            "observed_set_sources": observed_set_sources,
            "observed_set_semantics": observed_set_semantics,
        }
    )
    write_json(summary_path, summary)
    _write_all_gt_progress(
        config.output_root,
        config.dataset_name,
        stage,
        processed_rows=len(samples),
        total_rows=len(samples),
        checkpoint_path=checkpoint_path,
        status="COMPLETE",
    )
    return summary


def run_gt_attribution_rank_all_gt_audit(config: GTAttributionRankAuditConfig) -> Dict[str, Any]:
    stage_names = ALL_STAGES if config.stage == "all" else (config.stage,)
    prepared = _prepare_all_gt_shared_inputs(config)
    results: Dict[str, Any] = {}
    for stage in stage_names:
        results[stage] = run_stage_all_gt_attribution_rank_audit(config, stage, prepared_inputs=prepared)
    comparison = _build_all_gt_comparison_by_split(results, split_order=_all_gt_split_order_for_dataset(config.dataset_name))
    summary = {
        "dataset_name": config.dataset_name,
        "output_root": str(config.output_root),
        "split_order": list(_all_gt_split_order_for_dataset(config.dataset_name)),
        "stages": results,
        "comparison_by_split": comparison,
        "observed_set_sources": list(prepared.get("observed_set_sources", [])),
        "observed_set_semantics": list(prepared.get("observed_set_semantics", [])),
    }
    write_json(_package_all_gt_summary_by_split_path(config.output_root, config.dataset_name), summary)
    write_json(_package_all_gt_summary_export_path(config.output_root, config.dataset_name), summary)
    write_json(_package_all_gt_comparison_by_split_path(config.output_root, config.dataset_name), comparison)
    return summary


def run_gt_attribution_rank_audit(config: GTAttributionRankAuditConfig) -> Dict[str, Any]:
    if bool(config.all_gt_only):
        return run_gt_attribution_rank_all_gt_audit(config)
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
    run_gt_attribution_rank_all_gt_audit(config)
    return summary
