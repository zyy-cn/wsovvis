from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    build_stage_domain_indices,
    fuse_carrier_frame_logits,
    load_combined_evidence,
    load_text_vocab,
)
from videocutler.ext_stageb_ovvis.audit.trajectory_gt_audit import (
    _STAGE_ORDER,  # reuse stable ordering for summaries
    _extract_gt_available,
    _extract_gt_class_id,
    _load_jsonl,
    _snapshot_order_key,
    _trajectory_vector,
    load_gt_sidecar_lookup,
)
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig


Record = Dict[str, Any]


@dataclass(frozen=True)
class StageCheckpointSpec:
    stage_id: str
    train_state_path: str
    checkpoint_path: str
    selected_for_infer: str


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_projector_from_checkpoint(checkpoint_path: Path, *, device: torch.device) -> Projector:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_payload = dict(checkpoint.get("projector_config", {}))
    projector = Projector(
        ProjectorConfig(
            input_dim=int(config_payload.get("input_dim", 768)),
            hidden_dim=int(config_payload.get("hidden_dim", 512)),
            output_dim=int(config_payload.get("output_dim", 512)),
            dropout=float(config_payload.get("dropout", 0.0)),
            use_layernorm=bool(config_payload.get("use_layernorm", True)),
        )
    ).to(device)
    projector.load_state_dict(checkpoint["projector_state_dict"])
    projector.eval()
    return projector


def _stage_checkpoint_specs(output_root: Path) -> List[StageCheckpointSpec]:
    return [
        StageCheckpointSpec(
            stage_id="prealign",
            train_state_path="train/prealign/train_state.json",
            checkpoint_path="train/prealign/checkpoints/prealign_last.pth",
            selected_for_infer="prealign_only",
        ),
        StageCheckpointSpec(
            stage_id="softem_base",
            train_state_path="train/softem_base/train_state.json",
            checkpoint_path="train/softem_base/checkpoints/softem_base_last.pth",
            selected_for_infer="base_only",
        ),
        StageCheckpointSpec(
            stage_id="softem_aug",
            train_state_path="train/softem_aug/train_state.json",
            checkpoint_path="train/softem_aug/checkpoints/softem_aug_last.pth",
            selected_for_infer="augmented",
        ),
    ]


def _normalize_rows(rows: Sequence[Record]) -> List[Record]:
    return sorted(list(rows), key=lambda row: str(row.get("trajectory_id", "")))


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _safe_median(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def _rank_from_scores(scores: np.ndarray, index: int) -> int:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    matches = np.where(order == int(index))[0]
    if matches.size == 0:
        return -1
    return int(matches[0]) + 1


def _compute_query_and_scores(
    *,
    projector: Projector,
    carrier_vec: np.ndarray,
    frame_vectors: Sequence[np.ndarray],
    frame_vec: np.ndarray,
    text_matrix: np.ndarray,
    temperature: float,
    lambda_frame: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = next(projector.parameters()).device if hasattr(projector, "parameters") else torch.device("cpu")
    carrier_tensor = torch.from_numpy(np.asarray(carrier_vec, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    if frame_vectors:
        frame_tensor = torch.from_numpy(np.stack([np.asarray(vec, dtype=np.float32) for vec in frame_vectors], axis=0)).to(device=device, dtype=torch.float32)
        frame_q = projector(frame_tensor).mean(dim=0, keepdim=True)
    else:
        frame_tensor = torch.from_numpy(np.asarray(frame_vec, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
        frame_q = projector(frame_tensor)
    carrier_q = projector(carrier_tensor)
    fused_q = (1.0 - float(lambda_frame)) * carrier_q + float(lambda_frame) * frame_q
    candidate_tensor = torch.from_numpy(np.asarray(text_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
    candidate_tensor = F.normalize(candidate_tensor, p=2.0, dim=-1)
    carrier_logits = torch.matmul(carrier_q, candidate_tensor.t()).squeeze(0) / float(temperature)
    frame_logits = torch.matmul(frame_q, candidate_tensor.t()).squeeze(0) / float(temperature)
    fused_logits = torch.matmul(fused_q, candidate_tensor.t()).squeeze(0) / float(temperature)
    cosine_scores = torch.matmul(F.normalize(fused_q, p=2.0, dim=-1), candidate_tensor.t()).squeeze(0)
    return (
        carrier_logits.detach().cpu().numpy().astype(np.float32),
        frame_logits.detach().cpu().numpy().astype(np.float32),
        fused_logits.detach().cpu().numpy().astype(np.float32),
        cosine_scores.detach().cpu().numpy().astype(np.float32),
        fused_q.detach().cpu().numpy().astype(np.float32),
    )


def build_projector_quality_rows(
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
    stage_id: str,
    snapshot_id: str,
    materialized_samples: Sequence[Mapping[str, Any]],
    projector: Projector,
    topk: int,
    gt_sidecar_lookup: Mapping[str, Mapping[str, Any]],
    temperature: float,
    previous_by_trajectory: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Record]:
    previous_by_trajectory = dict(previous_by_trajectory or {})
    text_vocab_ids, _text_records, text_matrix = load_text_vocab(output_root)
    text_id_to_index = {int(raw_id): idx for idx, raw_id in enumerate(text_vocab_ids)}
    rows: List[Record] = []

    for sample in materialized_samples:
        trajectory_id = str(sample.get("trajectory_id", "")).strip()
        clip_id = str(sample.get("clip_id", "")).strip()
        trajectory_record = sample.get("trajectory_record") if isinstance(sample.get("trajectory_record"), Mapping) else {}
        video_id = int(trajectory_record.get("video_id")) if isinstance(trajectory_record, Mapping) and trajectory_record.get("video_id") is not None else None
        carrier_record = sample.get("carrier_record") if isinstance(sample.get("carrier_record"), Mapping) else {}
        missing_views = [str(x) for x in list(sample.get("missing_views", []))]
        invalid_reasons = [str(x) for x in list(sample.get("invalid_reasons", []))]
        observed_raw_ids = [int(x) for x in list(sample.get("observed_raw_ids", []))]
        candidate_ids_known = [int(x) for x in list(sample.get("candidate_ids_known", []))]
        candidate_ids_extra = [int(x) for x in list(sample.get("candidate_ids_extra", []))]
        candidate_ids_union = sorted(dict.fromkeys([*candidate_ids_known, *candidate_ids_extra]))

        gt_record = gt_sidecar_lookup.get(trajectory_id, {})
        gt_class_id = _extract_gt_class_id(gt_record) if gt_record else None
        gt_available_for_audit = bool(gt_record) and _extract_gt_available(gt_record)

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
            "gt_available_for_audit": gt_available_for_audit,
            "gt_class_id": gt_class_id,
            "gt_in_known_domain": bool(gt_class_id is not None and gt_class_id in candidate_ids_known),
            "gt_in_extra_domain": bool(gt_class_id is not None and gt_class_id in candidate_ids_extra),
            "gt_in_union_domain": bool(gt_class_id is not None and gt_class_id in candidate_ids_union),
            "gt_rank_full_vocab": None,
            "gt_rank_stage_domain": None,
            "gt_score": None,
            "best_wrong_id": None,
            "best_wrong_score": None,
            "margin_gt_minus_best_wrong": None,
            "is_gt_top1": False,
            "is_gt_in_topk": False,
            "cosine_to_gt_text": None,
            "cosine_to_best_wrong_text": None,
            "top1_id": None,
            "top1_score": None,
            "topk_ids": [],
            "topk_scores": [],
            "observed_top1_rate": None,
            "observed_mass_quality": None,
            "closed_set_top1_id": None,
            "closed_set_top1_score": None,
            "closed_set_gt_rank": None,
            "closed_set_gt_score": None,
            "closed_set_is_gt_top1": False,
            "closed_set_candidate_ids": [],
            "invalid_reasons": sorted(set(invalid_reasons)),
            "missing_views": sorted(set(missing_views)),
            "delta_gt_score_vs_prev": None,
            "delta_gt_rank_vs_prev": None,
        }

        sample_valid = bool(sample.get("sample_valid", False))
        traj_locator = str(carrier_record.get("z_norm_path", "")).strip() if isinstance(carrier_record, Mapping) else ""
        if sample_valid and traj_locator and gt_available_for_audit:
            try:
                carrier_vec, frame_vectors, frame_vec, _combined = load_combined_evidence(
                    sample,
                    output_root=output_root,
                    dataset_name=dataset_name,
                    trajectory_source_branch=trajectory_source_branch,
                )
                _, _, fused_logits, cosine_scores, fused_query = _compute_query_and_scores(
                    projector=projector,
                    carrier_vec=carrier_vec,
                    frame_vectors=frame_vectors,
                    frame_vec=frame_vec,
                    text_matrix=text_matrix,
                    temperature=temperature,
                )
                probs = torch.softmax(torch.from_numpy(fused_logits), dim=0).detach().cpu().numpy().astype(np.float64)
                order = np.argsort(-np.asarray(fused_logits, dtype=np.float64), kind="mergesort")
                k = min(max(int(topk), 1), int(len(order)))
                top_indices = order[:k]
                topk_ids = [int(text_vocab_ids[idx]) for idx in top_indices]
                topk_scores = [float(probs[idx]) for idx in top_indices]
                row["top1_id"] = int(topk_ids[0]) if topk_ids else None
                row["top1_score"] = float(topk_scores[0]) if topk_scores else None
                row["topk_ids"] = topk_ids
                row["topk_scores"] = topk_scores
                row["observed_top1_rate"] = bool(row["top1_id"] in observed_raw_ids) if topk_ids else False
                row["observed_mass_quality"] = float(np.sum([probs[text_id_to_index[x]] for x in observed_raw_ids if x in text_id_to_index])) if observed_raw_ids else 0.0
                row["cosine_to_gt_text"] = float(cosine_scores[text_id_to_index[gt_class_id]]) if gt_class_id in text_id_to_index else None
                if topk_ids:
                    best_wrong_idx = None
                    for idx in order:
                        raw_id = int(text_vocab_ids[idx])
                        if gt_class_id is None or raw_id != gt_class_id:
                            best_wrong_idx = int(idx)
                            break
                    if best_wrong_idx is not None:
                        row["best_wrong_id"] = int(text_vocab_ids[best_wrong_idx])
                        row["best_wrong_score"] = float(probs[best_wrong_idx])
                        row["cosine_to_best_wrong_text"] = float(cosine_scores[best_wrong_idx])
                        row["margin_gt_minus_best_wrong"] = (
                            float(probs[text_id_to_index[gt_class_id]]) - float(probs[best_wrong_idx])
                            if gt_class_id in text_id_to_index
                            else None
                        )
                if gt_class_id in text_id_to_index:
                    gt_idx = text_id_to_index[gt_class_id]
                    row["gt_rank_full_vocab"] = _rank_from_scores(fused_logits, gt_idx)
                    row["gt_score"] = float(probs[gt_idx])
                    row["is_gt_top1"] = bool(row["gt_rank_full_vocab"] == 1)
                    row["is_gt_in_topk"] = bool(row["gt_rank_full_vocab"] is not None and row["gt_rank_full_vocab"] <= k)

                stage_domain_ids, _, extra_domain_ids = build_stage_domain_indices(
                    candidate_ids_known,
                    candidate_ids_extra,
                    stage_id=stage_id,
                )
                closed_candidate_ids = sorted(dict.fromkeys([*candidate_ids_known, *candidate_ids_extra, *( [gt_class_id] if gt_class_id is not None else [] )]))
                row["closed_set_candidate_ids"] = closed_candidate_ids
                stage_domain_logits = [0.0]
                stage_domain_ids_for_scores: List[int] = []
                for raw_id in stage_domain_ids:
                    idx = text_id_to_index.get(int(raw_id))
                    if idx is None:
                        continue
                    stage_domain_logits.append(float(fused_logits[idx]))
                    stage_domain_ids_for_scores.append(int(raw_id))
                if gt_class_id is not None and gt_class_id in text_id_to_index and gt_class_id not in stage_domain_ids_for_scores:
                    gt_idx = text_id_to_index[gt_class_id]
                    stage_domain_logits.append(float(fused_logits[gt_idx]))
                    stage_domain_ids_for_scores.append(int(gt_class_id))
                stage_domain_logits_arr = np.asarray(stage_domain_logits, dtype=np.float64)
                if gt_class_id is not None and gt_class_id in stage_domain_ids_for_scores:
                    gt_stage_idx = stage_domain_ids_for_scores.index(int(gt_class_id)) + 1  # +1 for unknown slot
                    row["gt_rank_stage_domain"] = _rank_from_scores(stage_domain_logits_arr, gt_stage_idx)
                    row["closed_set_gt_rank"] = row["gt_rank_stage_domain"]
                    row["closed_set_gt_score"] = float(torch.softmax(torch.from_numpy(stage_domain_logits_arr), dim=0).detach().cpu().numpy()[gt_stage_idx])
                    row["closed_set_top1_id"] = "unknown"
                    if len(stage_domain_logits_arr) > 1:
                        best_known_idx = int(np.argmax(stage_domain_logits_arr[1:])) + 1
                        if best_known_idx >= 1:
                            row["closed_set_top1_id"] = int(stage_domain_ids_for_scores[best_known_idx - 1])
                            row["closed_set_top1_score"] = float(torch.softmax(torch.from_numpy(stage_domain_logits_arr), dim=0).detach().cpu().numpy()[best_known_idx])
                    row["closed_set_is_gt_top1"] = bool(row["gt_rank_stage_domain"] == 1)
                elif gt_class_id is not None and gt_class_id in text_id_to_index:
                    row["gt_rank_stage_domain"] = None
                    row["closed_set_gt_rank"] = None

            except Exception as exc:  # pragma: no cover - defensive path
                row["invalid_reasons"] = sorted(set(row["invalid_reasons"] + [f"audit_projection_failed:{type(exc).__name__}"]))

        prev = previous_by_trajectory.get(trajectory_id)
        if prev is not None and row["gt_available_for_audit"] and row["gt_rank_full_vocab"] is not None and row["gt_score"] is not None:
            prev_rank = prev.get("gt_rank_full_vocab")
            prev_score = prev.get("gt_score")
            if prev_rank is not None:
                try:
                    row["delta_gt_rank_vs_prev"] = float(prev_rank) - float(row["gt_rank_full_vocab"])
                except Exception:
                    row["delta_gt_rank_vs_prev"] = None
            if prev_score is not None:
                try:
                    row["delta_gt_score_vs_prev"] = float(row["gt_score"]) - float(prev_score)
                except Exception:
                    row["delta_gt_score_vs_prev"] = None

        rows.append(row)

    return rows


def summarize_projector_quality_rows(stage_rows: Mapping[str, Sequence[Record]]) -> Dict[str, Any]:
    ordered_stage_ids = sorted(stage_rows.keys(), key=lambda item: (_STAGE_ORDER.get(str(item), 99), str(item)))
    summary_by_stage: Dict[str, Any] = {}
    filtered_stage_rows: Dict[str, List[Record]] = {}
    for stage_id in ordered_stage_ids:
        filtered = [row for row in stage_rows[stage_id] if row.get("gt_available_for_audit") and row.get("gt_rank_full_vocab") is not None and row.get("gt_score") is not None]
        filtered_stage_rows[stage_id] = filtered
        gt_rows = filtered
        gt_top1_hits = sum(1 for row in gt_rows if bool(row.get("is_gt_top1")))
        gt_topk_hits = sum(1 for row in gt_rows if bool(row.get("is_gt_in_topk")))
        observed_top1_hits = sum(1 for row in gt_rows if bool(row.get("observed_top1_rate")))
        gt_ranks = [float(row["gt_rank_full_vocab"]) for row in gt_rows]
        gt_scores = [float(row["gt_score"]) for row in gt_rows]
        margins = [float(row["margin_gt_minus_best_wrong"]) for row in gt_rows if row.get("margin_gt_minus_best_wrong") is not None]
        cosine_gt = [float(row["cosine_to_gt_text"]) for row in gt_rows if row.get("cosine_to_gt_text") is not None]
        cosine_wrong = [float(row["cosine_to_best_wrong_text"]) for row in gt_rows if row.get("cosine_to_best_wrong_text") is not None]
        observed_mass = [float(row["observed_mass_quality"]) for row in gt_rows if row.get("observed_mass_quality") is not None]
        summary_by_stage[stage_id] = {
            "status": "PASS" if gt_rows else "EMPTY",
            "row_count": int(len(stage_rows[stage_id])),
            "gt_row_count": int(len(gt_rows)),
            "gt_top1_rate": float(gt_top1_hits / len(gt_rows)) if gt_rows else None,
            "gt_topk_rate": float(gt_topk_hits / len(gt_rows)) if gt_rows else None,
            "mean_gt_rank": _safe_mean(gt_ranks),
            "median_gt_rank": _safe_median(gt_ranks),
            "mean_gt_score": _safe_mean(gt_scores),
            "mean_margin_gt_minus_best_wrong": _safe_mean(margins),
            "mean_cosine_to_gt_text": _safe_mean(cosine_gt),
            "mean_cosine_to_best_wrong_text": _safe_mean(cosine_wrong),
            "observed_top1_rate": float(observed_top1_hits / len(gt_rows)) if gt_rows else None,
            "observed_mass_quality": _safe_mean(observed_mass),
        }

    transition_summary: Dict[str, Any] = {
        "prealign_to_softem_base": _transition_metrics(filtered_stage_rows.get("prealign", []), filtered_stage_rows.get("softem_base", [])),
        "softem_base_to_softem_aug": _transition_metrics(filtered_stage_rows.get("softem_base", []), filtered_stage_rows.get("softem_aug", [])),
        "prealign_to_softem_aug": _transition_metrics(filtered_stage_rows.get("prealign", []), filtered_stage_rows.get("softem_aug", [])),
    }
    return {
        "status": "PASS" if summary_by_stage else "EMPTY",
        "stage_ids": ordered_stage_ids,
        "stage_summaries": summary_by_stage,
        "transition_summary": transition_summary,
    }


def _transition_metrics(prev_rows: Sequence[Record], next_rows: Sequence[Record]) -> Dict[str, Any]:
    prev_by_tid = {str(row.get("trajectory_id", "")): row for row in prev_rows if row.get("trajectory_id")}
    next_by_tid = {str(row.get("trajectory_id", "")): row for row in next_rows if row.get("trajectory_id")}
    gt_rank_improved_count = 0
    gt_rank_worsened_count = 0
    gt_score_improved_count = 0
    margin_improved_count = 0
    compared = 0
    for trajectory_id, prev in prev_by_tid.items():
        nxt = next_by_tid.get(trajectory_id)
        if not nxt:
            continue
        if prev.get("gt_rank_full_vocab") is None or nxt.get("gt_rank_full_vocab") is None:
            continue
        compared += 1
        if float(nxt["gt_rank_full_vocab"]) < float(prev["gt_rank_full_vocab"]):
            gt_rank_improved_count += 1
        elif float(nxt["gt_rank_full_vocab"]) > float(prev["gt_rank_full_vocab"]):
            gt_rank_worsened_count += 1
        if float(nxt.get("gt_score", 0.0)) > float(prev.get("gt_score", 0.0)):
            gt_score_improved_count += 1
        prev_margin = float(prev.get("margin_gt_minus_best_wrong") or 0.0)
        next_margin = float(nxt.get("margin_gt_minus_best_wrong") or 0.0)
        if next_margin > prev_margin:
            margin_improved_count += 1
    return {
        "compared_trajectory_count": int(compared),
        "gt_rank_improved_count": int(gt_rank_improved_count),
        "gt_rank_worsened_count": int(gt_rank_worsened_count),
        "gt_score_improved_count": int(gt_score_improved_count),
        "margin_improved_count": int(margin_improved_count),
    }


def _stage_checkpoint_path(output_root: Path, stage_id: str) -> Path:
    if stage_id == "prealign":
        return output_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth"
    if stage_id == "softem_base":
        return output_root / "train" / "softem_base" / "checkpoints" / "softem_base_last.pth"
    if stage_id == "softem_aug":
        return output_root / "train" / "softem_aug" / "checkpoints" / "softem_aug_last.pth"
    raise ValueError(f"unsupported stage_id: {stage_id}")


def run_projector_quality_audit(
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
    smoke: bool,
    smoke_max_trajectories: int,
    topk: int = 5,
    gt_sidecar_dir: str = "audit",
    temperature: float = 0.07,
) -> Dict[str, Any]:
    if dataset_name != "lvvis_train_base":
        raise ValueError("projector quality audit currently supports dataset_name=lvvis_train_base only")
    if trajectory_source_branch != "mainline":
        raise ValueError("projector quality audit currently supports trajectory_source_branch=mainline only")

    from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
        Phase1MaterializationConfig,
        materialize_phase1_training_samples,
    )

    materialized = materialize_phase1_training_samples(
        output_root,
        Phase1MaterializationConfig(
            dataset_name=dataset_name,
            trajectory_source_branch=trajectory_source_branch,
            smoke=smoke,
            smoke_max_trajectories=smoke_max_trajectories,
        ),
    )
    gt_lookup = load_gt_sidecar_lookup(
        output_root,
        dataset_name=dataset_name,
        trajectory_source_branch=trajectory_source_branch,
        gt_sidecar_dir=gt_sidecar_dir,
    )
    stage_rows: Dict[str, List[Record]] = {}
    previous_by_trajectory: Dict[str, Record] = {}
    stage_details: Dict[str, Any] = {}
    for spec in _stage_checkpoint_specs(output_root):
        ckpt_path = output_root / spec.checkpoint_path
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"missing projector checkpoint for stage {spec.stage_id}: {ckpt_path}")
        projector = _load_projector_from_checkpoint(ckpt_path, device=torch.device("cpu"))
        rows = build_projector_quality_rows(
            output_root=output_root,
            dataset_name=dataset_name,
            trajectory_source_branch=trajectory_source_branch,
            stage_id=spec.stage_id,
            snapshot_id="stage_end",
            materialized_samples=materialized["samples"],
            projector=projector,
            topk=topk,
            gt_sidecar_lookup=gt_lookup,
            temperature=temperature,
            previous_by_trajectory=previous_by_trajectory,
        )
        previous_by_trajectory = {str(row.get("trajectory_id", "")): dict(row) for row in rows if row.get("trajectory_id")}
        stage_rows[spec.stage_id] = rows
        stage_path = output_root / "train" / "audit" / f"{spec.stage_id}_projector_quality.json"
        stage_summary = summarize_projector_quality_rows({spec.stage_id: rows})
        stage_details[spec.stage_id] = {
            "stage_id": spec.stage_id,
            "train_state_path": spec.train_state_path,
            "checkpoint_path": spec.checkpoint_path,
            "selected_for_infer": spec.selected_for_infer,
            "materialization_stats": materialized["stats"],
            "summary": stage_summary["stage_summaries"][spec.stage_id],
            "rows": rows,
        }
        _write_json(stage_path, stage_details[spec.stage_id])

    summary = summarize_projector_quality_rows(stage_rows)
    transition_summary_path = output_root / "train" / "audit" / "stagewise_projector_quality_transition_summary.json"
    _write_json(transition_summary_path, summary["transition_summary"])
    casebook_path = output_root / "train" / "audit" / "projector_quality_casebook.jsonl"
    casebook_rows: List[Record] = []
    for stage_id in ("prealign", "softem_base", "softem_aug"):
        casebook_rows.extend(_normalize_rows(stage_rows.get(stage_id, [])))
    _write_jsonl(casebook_path, casebook_rows)
    payload = {
        "status": summary["status"],
        "dataset_name": dataset_name,
        "trajectory_source_branch": trajectory_source_branch,
        "smoke": bool(smoke),
        "smoke_max_trajectories": int(smoke_max_trajectories),
        "topk": int(topk),
        "gt_sidecar_dir": gt_sidecar_dir,
        "materialization_stats": materialized["stats"],
        "stage_summaries": summary["stage_summaries"],
        "transition_summary": summary["transition_summary"],
        "artifacts": {
            "prealign": "train/audit/prealign_projector_quality.json",
            "softem_base": "train/audit/softem_base_projector_quality.json",
            "softem_aug": "train/audit/softem_aug_projector_quality.json",
            "transition_summary": "train/audit/stagewise_projector_quality_transition_summary.json",
            "casebook": "train/audit/projector_quality_casebook.jsonl",
        },
        "training_semantics_changed": False,
        "formal_training_ready": False,
    }
    _write_json(output_root / "codex" / "outputs" / "G7_training" / "g7_stagewise_projector_quality_latest.json", payload)
    _write_md(
        output_root / "codex" / "outputs" / "G7_training" / "g7_stagewise_projector_quality_latest.md",
        [
            "# G7 Stagewise Projector Quality Audit",
            "",
            f"- status: {payload['status']}",
            f"- dataset_name: {dataset_name}",
            f"- trajectory_source_branch: {trajectory_source_branch}",
            f"- smoke: {bool(smoke)}",
            f"- topk: {int(topk)}",
            "- audit_only: true",
            "- training_semantics_changed: false",
            "- formal_training_ready: false",
        ],
    )
    return payload
