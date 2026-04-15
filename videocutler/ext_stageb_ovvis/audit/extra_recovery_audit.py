from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_carrier_records
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)


Record = Dict[str, Any]


def _load_jsonl(path: Path) -> List[Record]:
    if not path.is_file():
        return []
    rows: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _unique_ints(values: Sequence[Any]) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    for value in values:
        try:
            item = int(value)
        except Exception:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "codex" / "control" / "CURRENT_TASK.json").exists():
            return parent
    return current.parents[3]


def _dataset_split(dataset_name: str) -> str:
    if "train" in str(dataset_name):
        return "train"
    if "val" in str(dataset_name):
        return "val"
    raise ValueError(f"unsupported dataset_name: {dataset_name}")


def _valid_carrier_records(records: Sequence[Mapping[str, Any]]) -> List[Record]:
    out: List[Record] = []
    for record in records:
        if not bool(record.get("valid_carrier", False)):
            continue
        trajectory_id = str(record.get("trajectory_id", "")).strip()
        if not trajectory_id:
            continue
        out.append(dict(record))
    return out


def _decode_mask_rle(mask_item: Any, image_size: Sequence[int]) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pycocotools is required for extra recovery GT matching") from exc

    if len(image_size) != 2:
        raise ValueError("image_size must contain [H, W]")
    h, w = int(image_size[0]), int(image_size[1])
    if h <= 0 or w <= 0:
        raise ValueError("image_size values must be positive")

    if mask_item is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(mask_item, dict):
        rle = dict(mask_item)
        if "size" not in rle:
            rle["size"] = [h, w]
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, h, w)
        return np.asarray(mask_utils.decode(rle), dtype=np.uint8)
    if isinstance(mask_item, str):
        return np.asarray(mask_utils.decode({"size": [h, w], "counts": mask_item.encode("utf-8")}), dtype=np.uint8)
    raise ValueError("unsupported mask rle format")


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    if mask_a.shape != mask_b.shape:
        raise ValueError("mask shape mismatch")
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 1e-12:
        return 0.0
    return inter / union


def _mean_iou_over_overlapping_frames(main_record: Mapping[str, Any], gt_record: Mapping[str, Any]) -> Tuple[float, float, int]:
    main_frames = [int(x) for x in list(main_record.get("frame_indices", []))]
    gt_frames = [int(x) for x in list(gt_record.get("frame_indices", []))]
    if not main_frames or not gt_frames:
        return 0.0, 0.0, 0
    main_masks = list(main_record.get("masks_rle", []))
    gt_masks = list(gt_record.get("masks_rle", []))
    if len(main_frames) != len(main_masks) or len(gt_frames) != len(gt_masks):
        return 0.0, 0.0, 0
    main_map = {frame: mask for frame, mask in zip(main_frames, main_masks)}
    gt_map = {frame: mask for frame, mask in zip(gt_frames, gt_masks)}
    overlap = sorted(set(main_map) & set(gt_map))
    if not overlap:
        return 0.0, 0.0, 0
    image_size = list(main_record.get("image_size") or gt_record.get("image_size") or [0, 0])
    ious: List[float] = []
    for frame_index in overlap:
        main_mask = _decode_mask_rle(main_map[frame_index], image_size)
        gt_mask = _decode_mask_rle(gt_map[frame_index], image_size)
        ious.append(_mask_iou(main_mask, gt_mask))
    arr = np.asarray(ious, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), int(len(overlap))


def _rank_from_scores(scores: Sequence[Tuple[int, float]], target_id: int) -> int:
    ordered = sorted(scores, key=lambda item: (-float(item[1]), int(item[0])))
    for idx, (candidate_id, _score) in enumerate(ordered, start=1):
        if int(candidate_id) == int(target_id):
            return idx
    return -1


def _entropy_from_scores(scores: Sequence[float]) -> float:
    arr = np.asarray(list(scores), dtype=np.float64)
    total = float(arr.sum())
    if total <= 1e-12:
        return 0.0
    probs = np.clip(arr / total, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _resolve_responsibility_records(output_root: Path) -> Dict[str, Record]:
    path = output_root / "train" / "softem_aug" / "responsibility_records.jsonl"
    records = _load_jsonl(path)
    by_tid: Dict[str, Record] = {}
    for row in records:
        trajectory_id = str(row.get("trajectory_id", "")).strip()
        if trajectory_id:
            by_tid[trajectory_id] = dict(row)
    return by_tid


def _candidate_domain_from_sample(sample: Mapping[str, Any]) -> Tuple[List[int], List[int], List[int]]:
    known = _unique_ints(list(sample.get("candidate_ids_known", [])))
    extra = _unique_ints(list(sample.get("candidate_ids_extra", [])))
    union: List[int] = []
    seen: set[int] = set()
    for raw_id in [*known, *extra]:
        if raw_id in seen:
            continue
        seen.add(raw_id)
        union.append(raw_id)
    return known, extra, union


def _sidecar_match_rows(
    *,
    output_root: Path,
    dataset_name: str,
    clip_ids: Sequence[int],
    trajectory_source_branch: str,
) -> Dict[str, Path]:
    split = _dataset_split(dataset_name)
    output_dir = output_root / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_id_set = {int(x) for x in clip_ids}
    if trajectory_source_branch == "mainline":
        main_records = _valid_carrier_records(
            [
                rec
                for rec in read_carrier_records(output_root / "exports" / dataset_name / "trajectory_records.jsonl")
                if int(rec.get("clip_id", -1)) in clip_id_set
            ]
        )
        gt_records = _valid_carrier_records(
            [
                rec
                for rec in read_carrier_records(output_root / "exports_gt" / dataset_name / "trajectory_records.jsonl")
                if int(rec.get("clip_id", -1)) in clip_id_set
            ]
        )
        main_by_clip: Dict[int, List[Record]] = defaultdict(list)
        gt_by_clip: Dict[int, List[Record]] = defaultdict(list)
        for rec in main_records:
            main_by_clip[int(rec["clip_id"])].append(dict(rec))
        for rec in gt_records:
            gt_by_clip[int(rec["clip_id"])].append(dict(rec))
        match_rows: List[Record] = []
        for clip_id in sorted(clip_id_set):
            for main_rec in main_by_clip.get(clip_id, []):
                best_row: Optional[Record] = None
                best_key: Tuple[float, int, str] = (-1.0, -1, "")
                for gt_rec in gt_by_clip.get(clip_id, []):
                    mean_iou, median_iou, support_frame_count = _mean_iou_over_overlapping_frames(main_rec, gt_rec)
                    key = (float(mean_iou), int(support_frame_count), str(gt_rec.get("trajectory_id", "")))
                    if key > best_key:
                        best_key = key
                        gt_class_id = _as_int(gt_rec.get("pred_label_raw"))
                        best_row = {
                            "dataset_name": dataset_name,
                            "trajectory_source_branch": "mainline",
                            "split_tag": main_rec.get("split_tag"),
                            "trajectory_id": str(main_rec.get("trajectory_id", "")),
                            "clip_id": int(main_rec.get("clip_id", -1)),
                            "video_id": _as_int(main_rec.get("video_id")),
                            "matched_gt_track_id": str(gt_rec.get("trajectory_id", "")),
                            "matched_gt_raw_id": gt_class_id,
                            "matched_gt_class_id": gt_class_id,
                            "match_iou_mean": float(mean_iou),
                            "match_iou_p50": float(median_iou),
                            "match_support_frame_count": int(support_frame_count),
                            "match_quality": "high" if mean_iou >= 0.5 else "medium" if mean_iou >= 0.25 else "low",
                            "semantic_purity_flag": bool(mean_iou >= 0.25),
                            "audit_usable": bool(mean_iou >= 0.25 and support_frame_count > 0 and gt_class_id is not None),
                        }
                if best_row is None:
                    best_row = {
                        "dataset_name": dataset_name,
                        "trajectory_source_branch": "mainline",
                        "split_tag": main_rec.get("split_tag"),
                        "trajectory_id": str(main_rec.get("trajectory_id", "")),
                        "clip_id": int(main_rec.get("clip_id", -1)),
                        "video_id": _as_int(main_rec.get("video_id")),
                        "matched_gt_track_id": None,
                        "matched_gt_raw_id": None,
                        "matched_gt_class_id": None,
                        "match_iou_mean": 0.0,
                        "match_iou_p50": 0.0,
                        "match_support_frame_count": 0,
                        "match_quality": "no_match",
                        "semantic_purity_flag": False,
                        "audit_usable": False,
                    }
                match_rows.append(best_row)
        match_path = output_dir / f"trajectory_gt_match_{split}_mainline.jsonl"
        _write_jsonl(match_path, sorted(match_rows, key=lambda row: str(row.get("trajectory_id", ""))))
        return {"match_path": match_path}

    if trajectory_source_branch == "gt_upper_bound":
        gt_records = _valid_carrier_records(
            [
                rec
                for rec in read_carrier_records(output_root / "exports_gt" / dataset_name / "trajectory_records.jsonl")
                if int(rec.get("clip_id", -1)) in clip_id_set
            ]
        )
        identity_rows: List[Record] = []
        for gt_rec in gt_records:
            gt_class_id = _as_int(gt_rec.get("pred_label_raw"))
            identity_rows.append(
                {
                    "dataset_name": dataset_name,
                    "trajectory_source_branch": "gt_upper_bound",
                    "split_tag": gt_rec.get("split_tag"),
                    "trajectory_id": str(gt_rec.get("trajectory_id", "")),
                    "clip_id": int(gt_rec.get("clip_id", -1)),
                    "video_id": _as_int(gt_rec.get("video_id")),
                    "matched_gt_track_id": str(gt_rec.get("trajectory_id", "")),
                    "matched_gt_raw_id": gt_class_id,
                    "matched_gt_class_id": gt_class_id,
                    "match_iou_mean": 1.0,
                    "match_iou_p50": 1.0,
                    "match_support_frame_count": int(len(gt_rec.get("frame_indices", []))),
                    "match_quality": "identity",
                    "semantic_purity_flag": True,
                    "audit_usable": bool(gt_class_id is not None),
                }
            )
        identity_path = output_dir / f"trajectory_gt_identity_{split}_gt.jsonl"
        _write_jsonl(identity_path, sorted(identity_rows, key=lambda row: str(row.get("trajectory_id", ""))))
        return {"identity_path": identity_path}

    raise ValueError(f"unsupported trajectory_source_branch: {trajectory_source_branch}")


def _load_or_generate_gt_sidecar_lookup(
    *,
    output_root: Path,
    dataset_name: str,
    clip_ids: Sequence[int],
    generate_sidecars: bool = True,
) -> Dict[str, Record]:
    if generate_sidecars:
        _sidecar_match_rows(
            output_root=output_root,
            dataset_name=dataset_name,
            clip_ids=clip_ids,
            trajectory_source_branch="mainline",
        )
        _sidecar_match_rows(
            output_root=output_root,
            dataset_name=dataset_name,
            clip_ids=clip_ids,
            trajectory_source_branch="gt_upper_bound",
        )
    sidecar_dir = output_root / "audit"
    lookup: Dict[str, Record] = {}
    for name in (
        f"trajectory_gt_match_{_dataset_split(dataset_name)}_mainline.jsonl",
        f"trajectory_gt_identity_{_dataset_split(dataset_name)}_gt.jsonl",
    ):
        for row in _load_jsonl(sidecar_dir / name):
            trajectory_id = str(row.get("trajectory_id", "")).strip()
            if trajectory_id:
                lookup[trajectory_id] = dict(row)
    return lookup


def _clip_summary_from_rows(rows: Sequence[Record]) -> List[Record]:
    by_clip: Dict[int, List[Record]] = defaultdict(list)
    for row in rows:
        by_clip[int(row.get("clip_id", -1))].append(dict(row))
    clip_rows: List[Record] = []
    for clip_id in sorted(by_clip.keys()):
        clip_group = by_clip[clip_id]
        observed_gt: set[int] = set()
        recovered_gt: set[int] = set()
        extra_admitted: set[int] = set()
        for row in clip_group:
            gt_class_id = _as_int(row.get("gt_class_id"))
            if gt_class_id is None or not bool(row.get("gt_available_for_audit")):
                continue
            if bool(row.get("gt_missing_from_observed")):
                observed_gt.add(gt_class_id)
            if bool(row.get("gt_recovered_via_extra")):
                recovered_gt.add(gt_class_id)
            for extra_id in list(row.get("candidate_ids_extra", [])):
                extra_admitted.add(int(extra_id))
        missing_count = int(len(observed_gt))
        recovered_count = int(len(recovered_gt))
        extra_count = int(len(extra_admitted))
        clip_rows.append(
            {
                "clip_id": int(clip_id),
                "clip_missing_class_count": missing_count,
                "clip_recovered_missing_class_count": recovered_count,
                "clip_missing_recall": float(recovered_count / missing_count) if missing_count else None,
                "clip_extra_class_count": extra_count,
                "clip_extra_precision": float(recovered_count / extra_count) if extra_count else None,
                "clip_extra_false_discovery_rate": float(1.0 - (recovered_count / extra_count)) if extra_count else None,
            }
        )
    return clip_rows


def _select_topk_ids_scores(candidate_scores: Sequence[Tuple[int, float]], topk: int) -> Tuple[List[int], List[float]]:
    ordered = sorted(candidate_scores, key=lambda item: (-float(item[1]), int(item[0])))
    top = ordered[: max(1, int(topk))]
    return [int(cid) for cid, _ in top], [float(score) for _, score in top]


def build_extra_recovery_rows(
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
    stage_id: str,
    snapshot_id: str,
    materialized_samples: Sequence[Mapping[str, Any]],
    responsibility_records: Sequence[Mapping[str, Any]],
    gt_sidecar_lookup: Mapping[str, Mapping[str, Any]],
    topk: int = 5,
) -> Tuple[List[Record], Dict[str, Any]]:
    resp_by_tid = {str(row.get("trajectory_id", "")).strip(): dict(row) for row in responsibility_records if str(row.get("trajectory_id", "")).strip()}
    rows: List[Record] = []
    skipped_reason_histogram: Counter[str] = Counter()
    for sample in sorted(materialized_samples, key=lambda row: str(row.get("trajectory_id", ""))):
        trajectory_id = str(sample.get("trajectory_id", "")).strip()
        if not trajectory_id:
            skipped_reason_histogram["missing_trajectory_id"] += 1
            continue
        resp = resp_by_tid.get(trajectory_id)
        if resp is None:
            skipped_reason_histogram["missing_responsibility_record"] += 1
            continue
        if not bool(sample.get("sample_valid", False)):
            skipped_reason_histogram["sample_not_valid_from_phase1"] += 1
            continue
        gt_record = dict(gt_sidecar_lookup.get(trajectory_id, {}))
        gt_class_id = _as_int(gt_record.get("matched_gt_class_id"))
        gt_available_for_audit = bool(gt_record) and bool(gt_record.get("audit_usable", False)) and gt_class_id is not None
        if not gt_available_for_audit:
            skipped_reason_histogram["gt_sidecar_unusable"] += 1
            continue

        observed_raw_ids = _unique_ints(list(sample.get("observed_raw_ids", [])))
        candidate_ids_known = _unique_ints(list(resp.get("candidate_ids_known", sample.get("candidate_ids_known", []))))
        candidate_ids_extra = _unique_ints(list(resp.get("candidate_ids_extra", sample.get("candidate_ids_extra", []))))
        candidate_ids_union = _unique_ints([*candidate_ids_known, *candidate_ids_extra])
        r_final = dict(resp.get("r_final", {}))
        r_known_extra = []
        invalid_reasons: List[str] = []
        for raw_id in candidate_ids_union:
            score = _as_float(r_final.get(str(raw_id), r_final.get(raw_id)))
            if score is None:
                invalid_reasons.append(f"missing_resp_mass:{raw_id}")
                score = 0.0
            r_known_extra.append((int(raw_id), float(score)))
        if not r_known_extra:
            skipped_reason_histogram["empty_candidate_domain"] += 1
            continue
        unknown_score = _as_float(r_final.get("unknown", 0.0)) or 0.0
        topk_ids, topk_scores = _select_topk_ids_scores(r_known_extra, topk=topk)
        top1_id = int(topk_ids[0]) if topk_ids else None
        top1_score = float(topk_scores[0]) if topk_scores else None
        margin_top1_top2 = float(topk_scores[0] - topk_scores[1]) if len(topk_scores) >= 2 else (float(topk_scores[0]) if topk_scores else None)
        entropy_final = _entropy_from_scores([score for _raw_id, score in r_known_extra])
        extra_scores = [(raw_id, score) for raw_id, score in r_known_extra if raw_id in set(candidate_ids_extra)]
        extra_top1_ids, extra_top1_scores = _select_topk_ids_scores(extra_scores, topk=1) if extra_scores else ([], [])
        extra_top1_id = int(extra_top1_ids[0]) if extra_top1_ids else None
        extra_top1_is_gt = bool(extra_top1_id is not None and gt_class_id is not None and int(extra_top1_id) == int(gt_class_id))
        gt_in_known_domain = bool(gt_class_id in candidate_ids_known)
        gt_in_extra_domain = bool(gt_class_id in candidate_ids_extra)
        gt_in_union_domain = bool(gt_class_id in candidate_ids_union)
        gt_rank_within_extra_union = _rank_from_scores(r_known_extra, gt_class_id) if gt_in_union_domain else None
        gt_score = float(dict(r_known_extra).get(int(gt_class_id), 0.0)) if gt_in_union_domain else None
        gt_missing_from_observed = bool(gt_class_id not in observed_raw_ids)
        gt_recovered_via_extra = bool(gt_missing_from_observed and gt_in_extra_domain and extra_top1_is_gt)
        row: Record = {
            "dataset_name": str(dataset_name),
            "trajectory_source_branch": str(trajectory_source_branch),
            "stage_id": str(stage_id),
            "snapshot_id": str(snapshot_id),
            "trajectory_id": trajectory_id,
            "clip_id": int(sample.get("clip_id", -1)),
            "video_id": _as_int(sample.get("video_id")),
            "observed_raw_ids": observed_raw_ids,
            "candidate_ids_known": candidate_ids_known,
            "candidate_ids_extra": candidate_ids_extra,
            "candidate_ids_union": candidate_ids_union,
            "top1_id": top1_id,
            "top1_score": top1_score,
            "topk_ids": topk_ids,
            "topk_scores": topk_scores,
            "entropy_final": entropy_final,
            "margin_top1_top2": margin_top1_top2,
            "gt_available_for_audit": gt_available_for_audit,
            "gt_class_id": gt_class_id,
            "gt_in_known_domain": gt_in_known_domain,
            "gt_in_extra_domain": gt_in_extra_domain,
            "gt_in_union_domain": gt_in_union_domain,
            "gt_missing_from_observed": gt_missing_from_observed,
            "gt_rank_within_extra_union": gt_rank_within_extra_union,
            "gt_score": gt_score,
            "extra_top1_id": extra_top1_id,
            "extra_top1_is_gt": extra_top1_is_gt,
            "gt_recovered_via_extra": gt_recovered_via_extra,
            "unknown_score": float(unknown_score),
            "invalid_reasons": sorted(set(invalid_reasons + [str(x) for x in list(sample.get("invalid_reasons", []))])),
            "missing_views": sorted(set(str(x) for x in list(sample.get("missing_views", [])))),
        }
        rows.append(row)

    clip_rows = _clip_summary_from_rows(rows)
    total_gt_missing = sum(int(row.get("clip_missing_class_count", 0)) for row in clip_rows)
    total_gt_recovered = sum(int(row.get("clip_recovered_missing_class_count", 0)) for row in clip_rows)
    total_extra_topk_pred = sum(int(row.get("clip_extra_class_count", 0)) for row in clip_rows)
    summary = {
        "status": "PASS" if rows else "EMPTY",
        "row_count": int(len(rows)),
        "gt_available_row_count": int(sum(1 for row in rows if bool(row.get("gt_available_for_audit")))),
        "gt_missing_row_count": int(total_gt_missing),
        "gt_recovered_row_count": int(total_gt_recovered),
        "clip_count": int(len(clip_rows)),
        "skipped_sample_count": int(sum(skipped_reason_histogram.values())),
        "skip_reason_histogram": dict(sorted(skipped_reason_histogram.items())),
        "extra_gt_recall@K": float(total_gt_recovered / total_gt_missing) if total_gt_missing else None,
        "extra_precision@K": float(total_gt_recovered / total_extra_topk_pred) if total_extra_topk_pred else None,
        "missing_class_recovery_rate": float(total_gt_recovered / total_gt_missing) if total_gt_missing else None,
        "spurious_extra_rate": float((total_extra_topk_pred - total_gt_recovered) / total_extra_topk_pred) if total_extra_topk_pred else None,
        "topk": int(topk),
        "stage_id": str(stage_id),
        "snapshot_id": str(snapshot_id),
        "dataset_name": str(dataset_name),
        "trajectory_source_branch": str(trajectory_source_branch),
        "clip_summaries": clip_rows,
    }
    return rows, summary


@dataclass
class ExtraRecoveryAuditConfig:
    dataset_name: str = "lvvis_train_base"
    trajectory_source_branch: str = "mainline"
    smoke: bool = False
    smoke_max_trajectories: int = 128
    topk: int = 5
    generate_val_sidecars: bool = True
    gt_sidecar_dir: str = "audit"


class ExtraRecoveryAuditBuffer:
    def __init__(
        self,
        *,
        output_root: Path,
        dataset_name: str,
        trajectory_source_branch: str,
        topk: int = 5,
        gt_sidecar_dir: str = "audit",
    ) -> None:
        self.output_root = Path(output_root)
        self.dataset_name = str(dataset_name)
        self.trajectory_source_branch = str(trajectory_source_branch)
        self.topk = int(topk) if int(topk) > 0 else 5
        self.gt_sidecar_dir = str(gt_sidecar_dir)
        self._rows: List[Record] = []
        self._summary: Optional[Dict[str, Any]] = None
        self._stage_written: set[str] = set()

    def _materialize_context(self, config: ExtraRecoveryAuditConfig) -> Tuple[List[Record], Dict[str, Any], Dict[str, Record]]:
        materialized = materialize_phase1_training_samples(
            self.output_root,
            Phase1MaterializationConfig(
                dataset_name=config.dataset_name,
                trajectory_source_branch=config.trajectory_source_branch,
                smoke=config.smoke,
                smoke_max_trajectories=config.smoke_max_trajectories,
            ),
        )
        samples = list(materialized["samples"])
        clip_ids = sorted({int(sample.get("clip_id", -1)) for sample in samples if sample.get("clip_id") is not None})
        responsibility_records = _load_jsonl(self.output_root / "train" / "softem_aug" / "responsibility_records.jsonl")
        sidecar_lookup = _load_or_generate_gt_sidecar_lookup(
            output_root=self.output_root,
            dataset_name=config.dataset_name,
            clip_ids=clip_ids,
            generate_sidecars=True,
        )
        if config.generate_val_sidecars and config.dataset_name == "lvvis_train_base":
            val_materialized = materialize_phase1_training_samples(
                self.output_root,
                Phase1MaterializationConfig(
                    dataset_name="lvvis_val",
                    trajectory_source_branch=config.trajectory_source_branch,
                    smoke=config.smoke,
                    smoke_max_trajectories=config.smoke_max_trajectories,
                ),
            )
            val_clip_ids = sorted({int(sample.get("clip_id", -1)) for sample in list(val_materialized["samples"]) if sample.get("clip_id") is not None})
            _load_or_generate_gt_sidecar_lookup(
                output_root=self.output_root,
                dataset_name="lvvis_val",
                clip_ids=val_clip_ids,
                generate_sidecars=True,
            )
        return samples, {"responsibility_records": responsibility_records}, sidecar_lookup

    def record_snapshot(self, context: Mapping[str, Any]) -> List[Record]:
        stage_id = str(context.get("stage_id", "")).strip()
        snapshot_id = str(context.get("snapshot_id", "")).strip() or "stage_end"
        phase = str(context.get("phase", "")).strip() or "stage_end"
        if not stage_id:
            raise ValueError("extra recovery snapshot missing stage_id")
        if "materialized_samples" not in context:
            raise ValueError("extra recovery snapshot missing materialized_samples")
        if "responsibility_records" not in context:
            raise ValueError("extra recovery snapshot missing responsibility_records")
        if "gt_sidecar_lookup" not in context:
            raise ValueError("extra recovery snapshot missing gt_sidecar_lookup")
        rows, _summary = build_extra_recovery_rows(
            output_root=self.output_root,
            dataset_name=self.dataset_name,
            trajectory_source_branch=self.trajectory_source_branch,
            stage_id=stage_id,
            snapshot_id=snapshot_id,
            materialized_samples=list(context.get("materialized_samples", [])),
            responsibility_records=list(context.get("responsibility_records", [])),
            gt_sidecar_lookup=dict(context.get("gt_sidecar_lookup", {})),
            topk=self.topk,
        )
        self._rows = rows
        if phase == "stage_end":
            self.flush_stage(stage_id)
        return rows

    def flush_stage(self, stage_id: str) -> Path:
        stage_dir = self.output_root / "train" / str(stage_id)
        stage_path = stage_dir / "extra_recovery_ledger.jsonl"
        _write_jsonl(stage_path, sorted(self._rows, key=lambda row: str(row.get("trajectory_id", ""))))
        self._stage_written.add(str(stage_id))
        return stage_path

    def finalize(self) -> Dict[str, Any]:
        if "softem_aug" not in self._stage_written and self._rows:
            self.flush_stage("softem_aug")
        # Recompute summary from the cached rows produced in the last snapshot.
        clip_rows = _clip_summary_from_rows(self._rows)
        gt_available_rows = [row for row in self._rows if bool(row.get("gt_available_for_audit"))]
        gt_missing_rows = [row for row in gt_available_rows if bool(row.get("gt_missing_from_observed"))]
        gt_recovered_rows = [row for row in gt_available_rows if bool(row.get("gt_recovered_via_extra"))]
        total_gt_missing = sum(int(row.get("clip_missing_class_count", 0)) for row in clip_rows)
        total_gt_recovered = sum(int(row.get("clip_recovered_missing_class_count", 0)) for row in clip_rows)
        total_extra_topk_pred = sum(int(row.get("clip_extra_class_count", 0)) for row in clip_rows)
        skipped_hist = Counter()
        for row in self._rows:
            for reason in row.get("invalid_reasons", []):
                if reason:
                    skipped_hist[str(reason)] += 1
        summary = {
            "status": "PASS" if self._rows else "EMPTY",
            "row_count": int(len(self._rows)),
            "gt_available_row_count": int(len(gt_available_rows)),
            "gt_missing_row_count": int(total_gt_missing),
            "gt_recovered_row_count": int(total_gt_recovered),
            "clip_count": int(len(clip_rows)),
            "skipped_sample_count": int(sum(skipped_hist.values())),
            "skip_reason_histogram": dict(sorted(skipped_hist.items())),
            "extra_gt_recall@K": float(total_gt_recovered / total_gt_missing) if total_gt_missing else None,
            "extra_precision@K": float(total_gt_recovered / total_extra_topk_pred) if total_extra_topk_pred else None,
            "missing_class_recovery_rate": float(total_gt_recovered / total_gt_missing) if total_gt_missing else None,
            "spurious_extra_rate": float((total_extra_topk_pred - total_gt_recovered) / total_extra_topk_pred) if total_extra_topk_pred else None,
            "topk": int(self.topk),
            "stage_id": "softem_aug",
            "snapshot_id": "stage_end",
            "dataset_name": self.dataset_name,
            "trajectory_source_branch": self.trajectory_source_branch,
            "clip_summaries": clip_rows,
        }
        self._summary = summary
        summary_path = self.output_root / "train" / "audit" / "extra_recovery_summary.json"
        _write_json(summary_path, summary)
        return summary


def run_extra_recovery_audit(
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
    smoke: bool,
    smoke_max_trajectories: int,
    topk: int = 5,
    generate_val_sidecars: bool = True,
    gt_sidecar_dir: str = "audit",
) -> Dict[str, Any]:
    config = ExtraRecoveryAuditConfig(
        dataset_name=dataset_name,
        trajectory_source_branch=trajectory_source_branch,
        smoke=smoke,
        smoke_max_trajectories=smoke_max_trajectories,
        topk=topk,
        generate_val_sidecars=generate_val_sidecars,
        gt_sidecar_dir=gt_sidecar_dir,
    )
    buffer = ExtraRecoveryAuditBuffer(
        output_root=output_root,
        dataset_name=dataset_name,
        trajectory_source_branch=trajectory_source_branch,
        topk=topk,
        gt_sidecar_dir=gt_sidecar_dir,
    )
    materialized, extra_context, sidecar_lookup = buffer._materialize_context(config)
    responsibility_records = list(extra_context["responsibility_records"])
    rows = buffer.record_snapshot(
        {
            "stage_id": "softem_aug",
            "snapshot_id": "stage_end",
            "phase": "stage_end",
            "materialized_samples": materialized,
            "responsibility_records": responsibility_records,
            "gt_sidecar_lookup": sidecar_lookup,
        }
    )
    summary = buffer.finalize()
    return {
        "status": "PASS" if rows else "EMPTY",
        "dataset_name": dataset_name,
        "trajectory_source_branch": trajectory_source_branch,
        "smoke": bool(smoke),
        "smoke_max_trajectories": int(smoke_max_trajectories),
        "topk": int(topk),
        "generate_val_sidecars": bool(generate_val_sidecars),
        "sidecar_dir": gt_sidecar_dir,
        "sidecar_paths": {
            "train_mainline": str(output_root / gt_sidecar_dir / "trajectory_gt_match_train_mainline.jsonl"),
            "train_gt": str(output_root / gt_sidecar_dir / "trajectory_gt_identity_train_gt.jsonl"),
            "val_mainline": str(output_root / gt_sidecar_dir / "trajectory_gt_match_val_mainline.jsonl"),
            "val_gt": str(output_root / gt_sidecar_dir / "trajectory_gt_identity_val_gt.jsonl"),
        },
        "ledger_path": str(output_root / "train" / "softem_aug" / "extra_recovery_ledger.jsonl"),
        "summary_path": str(output_root / "train" / "audit" / "extra_recovery_summary.json"),
        "row_count": len(rows),
        "summary": summary,
        "current_asset_mode_behavior": "bounded_provisional_extra_recovery_audit_with_offline_gt_sidecar",
        "audit_only": True,
        "training_semantics_changed": False,
        "formal_training_ready": False,
    }
