from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_vector_from_locator
from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import fuse_carrier_frame_logits
from videocutler.ext_stageb_ovvis.banks.text_bank import resolve_text_prototype


Record = Dict[str, Any]

_STAGE_ORDER = {"prealign": 0, "softem_base": 1, "softem_aug": 2}


def _load_jsonl(path: Path) -> List[Record]:
    rows: List[Record] = []
    if not path.is_file():
        return rows
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


def _unique_ints(values: Sequence[Any]) -> List[int]:
    seen = set()
    out: List[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _extract_gt_class_id(record: Mapping[str, Any]) -> Optional[int]:
    for key in (
        "gt_class_id",
        "matched_gt_raw_id",
        "matched_gt_class_id",
        "gt_raw_id",
        "raw_id",
        "class_id",
    ):
        if key in record:
            out = _as_int(record.get(key))
            if out is not None:
                return out
    return None


def _extract_gt_available(record: Mapping[str, Any]) -> bool:
    if "audit_usable" in record:
        return bool(record.get("audit_usable"))
    if "match_quality" in record:
        quality = str(record.get("match_quality", "")).strip().lower()
        if quality:
            return quality not in {"bad", "failed", "reject", "rejected"}
    return True


def _load_gt_sidecar_records(
    output_root: Path,
    *,
    dataset_name: str,
    trajectory_source_branch: str,
    gt_sidecar_dir: str = "audit",
) -> List[Record]:
    split = "train" if "train" in str(dataset_name) else "val"
    branch_suffix = "mainline" if trajectory_source_branch == "mainline" else "gt"
    candidates = [
        output_root / gt_sidecar_dir / f"trajectory_gt_match_{split}_{branch_suffix}.jsonl",
        output_root / gt_sidecar_dir / f"trajectory_gt_identity_{split}_{branch_suffix}.jsonl",
        output_root / "audit" / f"trajectory_gt_match_{split}_{branch_suffix}.jsonl",
        output_root / "audit" / f"trajectory_gt_identity_{split}_{branch_suffix}.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return _load_jsonl(candidate)
    return []


def load_gt_sidecar_lookup(
    output_root: Path,
    *,
    dataset_name: str,
    trajectory_source_branch: str,
    gt_sidecar_dir: str = "audit",
) -> Dict[str, Record]:
    records = _load_gt_sidecar_records(
        output_root,
        dataset_name=dataset_name,
        trajectory_source_branch=trajectory_source_branch,
        gt_sidecar_dir=gt_sidecar_dir,
    )
    by_tid: Dict[str, Record] = {}
    for row in records:
        trajectory_id = str(row.get("trajectory_id", "")).strip()
        if not trajectory_id:
            continue
        by_tid[trajectory_id] = dict(row)
    return by_tid


def _rank_from_scores(scores: np.ndarray, index: int) -> int:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    matches = np.where(order == int(index))[0]
    if matches.size == 0:
        return -1
    return int(matches[0]) + 1


def _safe_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _snapshot_order_key(snapshot_id: Any) -> tuple[int, int, str]:
    text = str(snapshot_id).strip()
    if text == "stage_start":
        return (0, 0, text)
    if text.startswith("epoch_"):
        try:
            return (1, int(text.split("_", 1)[1]), text)
        except Exception:
            return (1, 10**9, text)
    if text == "stage_end":
        return (2, 0, text)
    return (1, 10**9, text)


@lru_cache(maxsize=65536)
def _cached_text_vector(records_path_str: str, raw_id: int, proto_path: str) -> np.ndarray:
    records_path = Path(records_path_str)
    return np.asarray(
        resolve_text_prototype(
            records_path,
            {
                "raw_id": int(raw_id),
                "proto_path": str(proto_path),
                "path_base_mode": "artifact_parent_dir",
            },
        ),
        dtype=np.float32,
    )


def _compute_candidate_vectors(
    output_root: Path,
    candidate_records: Sequence[Mapping[str, Any]],
) -> Tuple[List[int], List[np.ndarray], List[str]]:
    text_records_path = output_root / "text_bank" / "text_prototype_records.jsonl"
    candidate_ids: List[int] = []
    candidate_vectors: List[np.ndarray] = []
    invalid_reasons: List[str] = []
    for record in candidate_records:
        raw_id = _as_int(record.get("raw_id"))
        proto_path = str(record.get("proto_path", "")).strip()
        if raw_id is None or not proto_path:
            invalid_reasons.append("candidate_raw_id_missing")
            continue
        try:
            vector = _cached_text_vector(str(text_records_path), int(raw_id), proto_path)
        except Exception:  # pragma: no cover - defensive path
            invalid_reasons.append(f"candidate_vector_load_failed:{raw_id}")
            continue
        candidate_ids.append(int(raw_id))
        candidate_vectors.append(np.asarray(vector, dtype=np.float32))
    if candidate_records and len(candidate_ids) != len(candidate_records):
        invalid_reasons.append("candidate_id_text_record_length_mismatch")
    return candidate_ids, candidate_vectors, invalid_reasons


@lru_cache(maxsize=32768)
def _cached_carrier_vector(locator_key: str, artifact_parent_dir: str) -> np.ndarray:
    return np.asarray(read_vector_from_locator(Path(artifact_parent_dir), locator_key), dtype=np.float32)


def _trajectory_vector(output_root: Path, carrier_locator: str, trajectory_source_branch: str, dataset_name: str) -> np.ndarray:
    carrier_base = "carrier_bank" if trajectory_source_branch == "mainline" else "carrier_bank_gt"
    artifact_parent_dir = output_root / carrier_base / dataset_name
    return _cached_carrier_vector(str(carrier_locator), str(artifact_parent_dir))


def build_attribution_rows(
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
    stage_id: str,
    snapshot_id: str,
    materialized_samples: Sequence[Mapping[str, Any]],
    projector: Any,
    topk: int,
    gt_sidecar_lookup: Mapping[str, Mapping[str, Any]],
    temperature: float,
    previous_by_trajectory: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Record]:
    previous_by_trajectory = dict(previous_by_trajectory or {})
    rows: List[Record] = []
    device = next(projector.parameters()).device if hasattr(projector, "parameters") else torch.device("cpu")
    projector_was_training = bool(getattr(projector, "training", False))
    projector.eval()
    with torch.no_grad():
        for sample in materialized_samples:
            trajectory_id = str(sample.get("trajectory_id", "")).strip()
            clip_id = str(sample.get("clip_id", "")).strip()
            trajectory_record = sample.get("trajectory_record") if isinstance(sample.get("trajectory_record"), Mapping) else {}
            video_id = _as_int(trajectory_record.get("video_id")) if isinstance(trajectory_record, Mapping) else None
            carrier_record = sample.get("carrier_record") if isinstance(sample.get("carrier_record"), Mapping) else {}
            missing_views = [str(x) for x in list(sample.get("missing_views", []))]
            invalid_reasons = [str(x) for x in list(sample.get("invalid_reasons", []))]
            observed_raw_ids = _unique_ints(sample.get("observed_raw_ids", []))
            candidate_ids_known = _unique_ints(sample.get("candidate_ids_known", []))
            candidate_ids_extra = _unique_ints(sample.get("candidate_ids_extra", []))
            candidate_ids_union = _unique_ints([*candidate_ids_known, *candidate_ids_extra])

            gt_record = gt_sidecar_lookup.get(trajectory_id, {})
            gt_class_id = _extract_gt_class_id(gt_record) if gt_record else None
            gt_available_for_audit = bool(gt_record) and _extract_gt_available(gt_record)

            candidate_text_records = list(sample.get("candidate_text_prototypes", []))
            candidate_known_ids, candidate_vectors, candidate_invalids = _compute_candidate_vectors(output_root, candidate_text_records)
            if candidate_invalids:
                invalid_reasons.extend(candidate_invalids)

            row: Record = {
                "dataset_name": str(dataset_name),
                "trajectory_source_branch": str(trajectory_source_branch),
                "stage_id": str(stage_id),
                "snapshot_id": str(snapshot_id),
                "trajectory_id": trajectory_id,
                "clip_id": clip_id,
                "video_id": int(video_id) if video_id is not None else None,
                "observed_raw_ids": observed_raw_ids,
                "candidate_ids_known": candidate_ids_known,
                "candidate_ids_extra": candidate_ids_extra,
                "candidate_ids_union": candidate_ids_union,
                "mass_final_topk": [],
                "top1_id": None,
                "top1_score": None,
                "topk_ids": [],
                "topk_scores": [],
                "entropy_final": None,
                "margin_top1_top2": None,
                "gt_available_for_audit": gt_available_for_audit,
                "gt_class_id": gt_class_id,
                "gt_in_known_domain": bool(gt_class_id is not None and gt_class_id in candidate_ids_known),
                "gt_in_extra_domain": bool(gt_class_id is not None and gt_class_id in candidate_ids_extra),
                "gt_in_union_domain": bool(gt_class_id is not None and gt_class_id in candidate_ids_union),
                "gt_rank": None,
                "gt_score": None,
                "is_gt_top1": False,
                "delta_gt_score_vs_prev": None,
                "delta_gt_rank_vs_prev": None,
                "missing_views": sorted(set(missing_views)),
                "invalid_reasons": sorted(set(invalid_reasons)),
            }

            sample_valid = bool(sample.get("sample_valid", False))
            traj_locator = ""
            if isinstance(carrier_record, Mapping):
                traj_locator = str(carrier_record.get("z_norm_path", "")).strip()
            if sample_valid and traj_locator and candidate_vectors and candidate_known_ids and len(candidate_known_ids) == len(candidate_vectors):
                try:
                    traj_vec = _trajectory_vector(output_root, traj_locator, trajectory_source_branch, dataset_name)
                    frame_vectors: List[np.ndarray] = []
                    frame_rows = list(sample.get("frame_feature_rows", []))
                    geom_rows = list(sample.get("frame_geometry_rows", []))
                    if len(frame_rows) == len(geom_rows) and frame_rows:
                        from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import (
                            read_feature_vector,
                            reconstruct_valid_token_mask_from_geometry,
                        )

                        def _coerce_token_feature_matrix(feature: np.ndarray, grid_h: int, grid_w: int) -> Optional[np.ndarray]:
                            feature = np.asarray(feature, dtype=np.float32)
                            if feature.ndim != 2:
                                return None
                            grid_tokens = int(grid_h) * int(grid_w)
                            if int(feature.shape[0]) == grid_tokens:
                                return feature
                            if int(feature.shape[0]) == grid_tokens + 1:
                                return feature[1:]
                            return None

                        frame_parent = output_root / "frame_bank" / dataset_name
                        for frame_row, geom_row in zip(frame_rows, geom_rows):
                            feat_path = str(frame_row.get("feat_path", ""))
                            if not feat_path:
                                continue
                            feature = read_feature_vector(frame_parent, feat_path)
                            token_matrix = _coerce_token_feature_matrix(feature, int(geom_row["grid_h"]), int(geom_row["grid_w"]))
                            if token_matrix is None:
                                continue
                            valid_mask = reconstruct_valid_token_mask_from_geometry(geom_row).astype(np.float32).reshape(-1)
                            denom = float(np.sum(valid_mask))
                            if denom <= 1e-12:
                                continue
                            frame_vec = np.sum(token_matrix * valid_mask[:, None], axis=0).astype(np.float32) / denom
                            frame_vectors.append(frame_vec)
                    frame_vec = np.mean(np.stack(frame_vectors, axis=0), axis=0).astype(np.float32) if frame_vectors else np.zeros_like(traj_vec)
                    _, _, fused_logits_np = fuse_carrier_frame_logits(
                        projector=projector,
                        carrier_vec=traj_vec,
                        frame_vec=frame_vec,
                        candidate_matrix=np.asarray(candidate_vectors, dtype=np.float32),
                        temperature=float(temperature),
                    )
                    logits = torch.from_numpy(np.asarray(fused_logits_np, dtype=np.float32)).to(device=device, dtype=torch.float32)
                    scores = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float64)
                    order = np.argsort(-scores, kind="mergesort")
                    k = int(topk) if int(topk) > 0 else 5
                    k = min(k, int(scores.shape[0]))
                    top_indices = order[:k]
                    topk_ids = [int(candidate_known_ids[idx]) for idx in top_indices]
                    topk_scores = [float(scores[idx]) for idx in top_indices]
                    row["top1_id"] = int(topk_ids[0]) if topk_ids else None
                    row["top1_score"] = float(topk_scores[0]) if topk_scores else None
                    row["topk_ids"] = topk_ids
                    row["topk_scores"] = topk_scores
                    row["mass_final_topk"] = [
                        {"candidate_id": int(candidate_known_ids[idx]), "score": float(scores[idx])}
                        for idx in top_indices
                    ]
                    row["entropy_final"] = _safe_entropy(scores)
                    if len(top_indices) >= 2:
                        row["margin_top1_top2"] = float(topk_scores[0] - topk_scores[1])
                    elif topk_scores:
                        row["margin_top1_top2"] = float(topk_scores[0])
                    if gt_class_id is not None and gt_class_id in candidate_known_ids:
                        gt_index = candidate_known_ids.index(gt_class_id)
                        row["gt_rank"] = _rank_from_scores(scores, gt_index)
                        row["gt_score"] = float(scores[gt_index])
                        row["is_gt_top1"] = bool(row["gt_rank"] == 1)
                except Exception as exc:  # pragma: no cover - defensive path
                    row["invalid_reasons"] = sorted(set(row["invalid_reasons"] + [f"audit_projection_failed:{type(exc).__name__}"]))

            prev = previous_by_trajectory.get(trajectory_id)
            if prev is not None and row["gt_available_for_audit"] and row["gt_rank"] is not None and row["gt_score"] is not None:
                prev_rank = prev.get("gt_rank")
                prev_score = prev.get("gt_score")
                if prev_rank is not None:
                    try:
                        row["delta_gt_rank_vs_prev"] = float(prev_rank) - float(row["gt_rank"])
                    except Exception:
                        row["delta_gt_rank_vs_prev"] = None
                if prev_score is not None:
                    try:
                        row["delta_gt_score_vs_prev"] = float(row["gt_score"]) - float(prev_score)
                    except Exception:
                        row["delta_gt_score_vs_prev"] = None

            rows.append(row)

    if projector_was_training:
        projector.train()
    return rows


def _flatten_rows(stage_rows: Mapping[str, Sequence[Record]]) -> List[Record]:
    ordered_stage_ids = sorted(stage_rows.keys(), key=lambda item: (_STAGE_ORDER.get(str(item), 99), str(item)))
    flattened: List[Record] = []
    for stage_id in ordered_stage_ids:
        stage_rows_list = list(stage_rows[stage_id])
        flattened.extend(stage_rows_list)
    return flattened


def summarize_attribution_rows(
    stage_rows: Mapping[str, Sequence[Record]],
) -> Dict[str, Any]:
    rows = _flatten_rows(stage_rows)
    gt_rows = [row for row in rows if bool(row.get("gt_available_for_audit")) and row.get("gt_rank") is not None and row.get("gt_score") is not None]
    gt_top1_hits = sum(1 for row in gt_rows if bool(row.get("is_gt_top1")))
    gt_in_domain_hits = sum(1 for row in gt_rows if bool(row.get("gt_in_union_domain")))
    gt_ranks = [float(row["gt_rank"]) for row in gt_rows if row.get("gt_rank") is not None]
    gt_scores = [float(row["gt_score"]) for row in gt_rows if row.get("gt_score") is not None]

    transition_counts = {
        "wrong_to_wrong": 0,
        "wrong_to_right": 0,
        "right_to_right": 0,
        "right_to_wrong": 0,
    }
    per_trajectory: Dict[str, List[Record]] = {}
    for row in rows:
        per_trajectory.setdefault(str(row.get("trajectory_id", "")), []).append(row)
    for traj_rows in per_trajectory.values():
        ordered = list(traj_rows)
        ordered.sort(key=lambda row: (_STAGE_ORDER.get(str(row.get("stage_id", "")), 99), _snapshot_order_key(row.get("snapshot_id", ""))))
        prev_right: Optional[bool] = None
        for row in ordered:
            if not (row.get("gt_available_for_audit") and row.get("gt_rank") is not None):
                continue
            current_right = bool(row.get("is_gt_top1"))
            if prev_right is not None:
                if not prev_right and not current_right:
                    transition_counts["wrong_to_wrong"] += 1
                elif not prev_right and current_right:
                    transition_counts["wrong_to_right"] += 1
                elif prev_right and current_right:
                    transition_counts["right_to_right"] += 1
                elif prev_right and not current_right:
                    transition_counts["right_to_wrong"] += 1
            prev_right = current_right

    wrong_prev = transition_counts["wrong_to_wrong"] + transition_counts["wrong_to_right"]
    right_prev = transition_counts["right_to_right"] + transition_counts["right_to_wrong"]
    wrong_to_right_correction_rate = float(transition_counts["wrong_to_right"] / wrong_prev) if wrong_prev else None
    right_to_wrong_regression_rate = float(transition_counts["right_to_wrong"] / right_prev) if right_prev else None

    monotonic_rank_total = 0
    monotonic_rank_hits = 0
    monotonic_score_total = 0
    monotonic_score_hits = 0
    for traj_rows in per_trajectory.values():
        ordered = list(traj_rows)
        ordered.sort(key=lambda row: (_STAGE_ORDER.get(str(row.get("stage_id", "")), 99), _snapshot_order_key(row.get("snapshot_id", ""))))
        gt_sequence = [row for row in ordered if row.get("gt_available_for_audit") and row.get("gt_rank") is not None and row.get("gt_score") is not None]
        if len(gt_sequence) < 2:
            continue
        monotonic_rank_total += 1
        monotonic_score_total += 1
        ranks = [float(row["gt_rank"]) for row in gt_sequence]
        scores = [float(row["gt_score"]) for row in gt_sequence]
        rank_deltas = [ranks[idx - 1] - ranks[idx] for idx in range(1, len(ranks))]
        score_deltas = [scores[idx] - scores[idx - 1] for idx in range(1, len(scores))]
        if all(delta >= 0.0 for delta in rank_deltas) and any(delta > 0.0 for delta in rank_deltas):
            monotonic_rank_hits += 1
        if all(delta >= 0.0 for delta in score_deltas) and any(delta > 0.0 for delta in score_deltas):
            monotonic_score_hits += 1

    monotonic_gt_rank_improve_rate = float(monotonic_rank_hits / monotonic_rank_total) if monotonic_rank_total else None
    monotonic_gt_score_improve_rate = float(monotonic_score_hits / monotonic_score_total) if monotonic_score_total else None

    return {
        "status": "PASS" if rows else "EMPTY",
        "row_count": int(len(rows)),
        "gt_row_count": int(len(gt_rows)),
        "gt_top1_rate": float(gt_top1_hits / len(gt_rows)) if gt_rows else None,
        "gt_in_domain_rate": float(gt_in_domain_hits / len(gt_rows)) if gt_rows else None,
        "mean_gt_rank": float(np.mean(gt_ranks)) if gt_ranks else None,
        "median_gt_rank": float(np.median(gt_ranks)) if gt_ranks else None,
        "mean_gt_score": float(np.mean(gt_scores)) if gt_scores else None,
        "wrong_to_right_correction_rate": wrong_to_right_correction_rate,
        "right_to_wrong_regression_rate": right_to_wrong_regression_rate,
        "monotonic_gt_rank_improve_rate": monotonic_gt_rank_improve_rate,
        "monotonic_gt_score_improve_rate": monotonic_gt_score_improve_rate,
        "transition_matrix": transition_counts,
    }
