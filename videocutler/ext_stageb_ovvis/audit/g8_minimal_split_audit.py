from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    _coerce_temperature_tensor,
    _project_candidate_matrix,
    load_combined_evidence,
)
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import (
    GTAttributionRankAuditConfig,
    TRAIN_DATASETS,
    VAL_BASE_NOVEL_DATASETS,
    _all_gt_split_label,
    _all_gt_split_order_for_dataset,
    _prepare_all_gt_shared_inputs,
    _require_dataset_name,
    _stage_checkpoint_path,
)
from videocutler.ext_stageb_ovvis.eval.g8_bridge import load_projector_bundle, write_json

MINIMAL_STAGES: Tuple[str, ...] = ("prealign", "softem_base")
TRAIN_SPLIT_ORDER: Tuple[str, ...] = ("base_observed", "base_unobserved")
VAL_SPLIT_ORDER: Tuple[str, ...] = ("base", "novel")
DEFAULT_LAMBDA_FRAME = 0.25
NEW_CHAIN_AUDIT_PIPELINE_ID = "g8_minimal_split_v1"
NEW_CHAIN_AUDIT_ENTRYPOINT = "videocutler/run_stageb_audit_g8_minimal_split.py"
NEW_CHAIN_AUDIT_SCOPE = "g8_default_minimal_split"
NEW_CHAIN_METRIC_SCOPE: Tuple[str, ...] = ("gt_count", "mean_normalized_gt_rank", "gt_top1_hit_rate")


@dataclass(frozen=True)
class MinimalSplitAuditConfig:
    dataset_name: str
    output_root: Path
    stage: str
    device: torch.device
    trajectory_source_branch: str = "mainline"
    all_gt_generate_sidecars_if_missing: bool = False
    heartbeat_every_rows: int = 256
    batch_size_rows: int = 128
    candidate_chunk_size: int = 0


@dataclass(frozen=True)
class CachedEvidence:
    trajectory_id: str
    carrier_vec: np.ndarray
    frame_vectors: Tuple[np.ndarray, ...]
    frame_vec: np.ndarray


def _stage_summary_path(output_root: Path, dataset_name: str, stage: str) -> Path:
    return output_root / "audit" / "minimal_split" / dataset_name / f"{stage}_minimal_split_summary.json"


def _dataset_summary_path(output_root: Path, dataset_name: str) -> Path:
    suffix = "train_minimal_split_summary.json" if dataset_name in TRAIN_DATASETS else "val_minimal_split_summary.json"
    return output_root / "audit" / "minimal_split" / dataset_name / suffix


def _package_comparison_path(output_root: Path, dataset_name: str) -> Path:
    return output_root / "audit" / "minimal_split" / dataset_name / "minimal_split_comparison.json"


def _stage_progress_path(output_root: Path, dataset_name: str, stage: str) -> Path:
    return output_root / "audit" / "minimal_split" / dataset_name / f"{stage}_minimal_split_progress.json"


def _split_order_for_dataset(dataset_name: str) -> Tuple[str, ...]:
    if dataset_name in TRAIN_DATASETS:
        return TRAIN_SPLIT_ORDER
    if dataset_name in VAL_BASE_NOVEL_DATASETS:
        return VAL_SPLIT_ORDER
    raise ValueError(f"unsupported dataset_name: {dataset_name!r}")


def _validate_stage(stage: str) -> str:
    if stage not in (*MINIMAL_STAGES, "all"):
        raise ValueError(f"stage must be one of {MINIMAL_STAGES + ('all',)}, got {stage!r}")
    return stage


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _write_progress(path: Path, *, status: str, processed_rows: int, total_rows: int, checkpoint_path: Path) -> None:
    payload = {
        "status": str(status),
        "processed_rows": int(processed_rows),
        "total_rows": int(total_rows),
        "checkpoint_path": str(checkpoint_path),
    }
    write_json(path, payload)


def _iter_stage_names(stage: str) -> Tuple[str, ...]:
    return MINIMAL_STAGES if stage == "all" else (stage,)


def _new_chain_provenance(*, dataset_name: str, split_order: Sequence[str], stage_scope: Sequence[str]) -> Dict[str, Any]:
    return {
        "audit_pipeline_id": NEW_CHAIN_AUDIT_PIPELINE_ID,
        "audit_entrypoint": NEW_CHAIN_AUDIT_ENTRYPOINT,
        "audit_scope": NEW_CHAIN_AUDIT_SCOPE,
        "dataset_name": str(dataset_name),
        "stage_scope": list(stage_scope),
        "metric_scope": list(NEW_CHAIN_METRIC_SCOPE),
        "split_scope": [str(x) for x in split_order],
        "generated_by_new_chain": True,
    }


def _materialize_shared_inputs(config: MinimalSplitAuditConfig) -> Dict[str, Any]:
    legacy_config = GTAttributionRankAuditConfig(
        dataset_name=config.dataset_name,
        output_root=config.output_root,
        stage="prealign",
        device=config.device,
        trajectory_source_branch=config.trajectory_source_branch,
        all_gt_generate_sidecars_if_missing=config.all_gt_generate_sidecars_if_missing,
        all_gt_heartbeat_every_rows=config.heartbeat_every_rows,
    )
    prepared = dict(_prepare_all_gt_shared_inputs(legacy_config))
    samples = [dict(x) for x in prepared.get("samples", [])]
    gt_sidecar_lookup = dict(prepared.get("gt_sidecar_lookup", {}))
    if samples:
        gt_available = 0
        for sample in samples:
            trajectory_id = str(sample.get("trajectory_id", sample.get("trajectory_record", {}).get("trajectory_id", ""))).strip()
            sidecar = dict(gt_sidecar_lookup.get(trajectory_id, {}))
            gt_raw_id = _as_int(sidecar.get("matched_gt_raw_id", sidecar.get("matched_gt_class_id")))
            if bool(sidecar.get("audit_usable", False)) and gt_raw_id is not None:
                gt_available += 1
        if gt_available == 0:
            sidecar_name = "train" if config.dataset_name in TRAIN_DATASETS else "val"
            raise RuntimeError(
                f"{sidecar_name.upper()}_GT_SIDECAR_MISSING_OR_EMPTY: row_count={len(samples)} gt_available_row_count=0 "
                f"for dataset={config.dataset_name} under output_root={config.output_root}"
            )
    return prepared


def _build_rows_and_cache(
    *,
    config: MinimalSplitAuditConfig,
    prepared: Mapping[str, Any],
    stage: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, CachedEvidence], Dict[int, int], torch.Tensor, torch.Tensor]:
    samples = [dict(x) for x in prepared["samples"]]
    gt_sidecar_lookup = dict(prepared["gt_sidecar_lookup"])
    asset_roots = prepared["asset_roots"]
    full_vocab_ids = [int(x) for x in prepared["full_vocab_ids"]]
    vocab_matrix = np.asarray(prepared["vocab_matrix"], dtype=np.float32)
    base_vocab_ids = [int(x) for x in prepared["base_vocab_ids"]]
    vocab_index = {int(raw_id): idx for idx, raw_id in enumerate(full_vocab_ids)}
    bundle = load_projector_bundle(_stage_checkpoint_path(config.output_root, stage), device=config.device)
    candidate_tensor = _project_candidate_matrix(
        projector=bundle.projector,
        candidate_matrix=vocab_matrix,
        device=config.device,
    )
    temperature_tensor = _coerce_temperature_tensor(bundle.temperature, device=config.device)

    rows: List[Dict[str, Any]] = []
    cache: Dict[str, CachedEvidence] = {}
    split_order = _split_order_for_dataset(config.dataset_name)
    split_order_set = set(split_order)
    for sample in samples:
        trajectory_id = str(sample.get("trajectory_id", sample.get("trajectory_record", {}).get("trajectory_id", ""))).strip()
        if not trajectory_id:
            continue
        sidecar = dict(gt_sidecar_lookup.get(trajectory_id, {}))
        gt_raw_id = _as_int(sidecar.get("matched_gt_raw_id", sidecar.get("matched_gt_class_id")))
        gt_available = bool(sidecar.get("audit_usable", False)) and gt_raw_id is not None
        observed_raw_ids = [int(x) for x in list(sample.get("observed_raw_ids", []))]
        split_label = None
        if gt_available and gt_raw_id is not None:
            split_label = _all_gt_split_label(
                dataset_name=config.dataset_name,
                gt_raw_id=int(gt_raw_id),
                observed_raw_ids=observed_raw_ids,
                base_vocab_ids=base_vocab_ids,
            )
        row = {
            "trajectory_id": trajectory_id,
            "video_id": _as_int(sample.get("video_id", sample.get("trajectory_record", {}).get("video_id"))),
            "clip_id": _as_int(sample.get("clip_id", sample.get("trajectory_record", {}).get("clip_id"))),
            "gt_class_id": gt_raw_id,
            "gt_available_for_audit": bool(gt_available),
            "split": split_label,
            "observed_raw_ids": observed_raw_ids,
            "stage": stage,
        }
        rows.append(row)
        if not gt_available or split_label not in split_order_set or gt_raw_id not in vocab_index:
            continue
        if trajectory_id in cache:
            continue
        carrier_vec, frame_vectors, frame_vec, _combined = load_combined_evidence(
            sample,
            output_root=asset_roots.asset_root,
            dataset_name=config.dataset_name,
            trajectory_source_branch=config.trajectory_source_branch,
        )
        cache[trajectory_id] = CachedEvidence(
            trajectory_id=trajectory_id,
            carrier_vec=np.asarray(carrier_vec, dtype=np.float32),
            frame_vectors=tuple(np.asarray(vec, dtype=np.float32) for vec in frame_vectors),
            frame_vec=np.asarray(frame_vec, dtype=np.float32),
        )
    return rows, cache, vocab_index, candidate_tensor, temperature_tensor


def _normalize_batch(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor, p=2.0, dim=-1)


def _score_batch(
    *,
    batch_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, CachedEvidence],
    candidate_tensor: torch.Tensor,
    temperature_tensor: torch.Tensor,
    vocab_index: Mapping[int, int],
    device: torch.device,
    candidate_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not batch_rows:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    carrier_np = np.stack([cache[str(row["trajectory_id"])].carrier_vec for row in batch_rows], axis=0).astype(np.float32)
    carrier_tensor = _normalize_batch(torch.from_numpy(carrier_np).to(device=device, dtype=torch.float32))

    frame_chunks: List[np.ndarray] = []
    frame_counts: List[int] = []
    for row in batch_rows:
        evidence = cache[str(row["trajectory_id"])]
        if evidence.frame_vectors:
            vecs = [np.asarray(x, dtype=np.float32) for x in evidence.frame_vectors]
        else:
            vecs = [np.asarray(evidence.frame_vec, dtype=np.float32)]
        frame_counts.append(len(vecs))
        frame_chunks.extend(vecs)
    flat_frames = np.stack(frame_chunks, axis=0).astype(np.float32)
    frame_tensor = _normalize_batch(torch.from_numpy(flat_frames).to(device=device, dtype=torch.float32))

    def _batched_logits(candidates: torch.Tensor) -> torch.Tensor:
        carrier_logits = torch.matmul(carrier_tensor, candidates.t()) / temperature_tensor
        frame_logits_flat = torch.matmul(frame_tensor, candidates.t()) / temperature_tensor
        frame_logits_parts = torch.split(frame_logits_flat, frame_counts, dim=0)
        frame_logits = torch.stack([part.mean(dim=0) for part in frame_logits_parts], dim=0)
        return (1.0 - float(DEFAULT_LAMBDA_FRAME)) * carrier_logits + float(DEFAULT_LAMBDA_FRAME) * frame_logits

    with torch.no_grad():
        if int(candidate_chunk_size) > 0 and int(candidate_chunk_size) < int(candidate_tensor.shape[0]):
            parts: List[torch.Tensor] = []
            chunk = int(candidate_chunk_size)
            for start in range(0, int(candidate_tensor.shape[0]), chunk):
                parts.append(_batched_logits(candidate_tensor[start : start + chunk]))
            fused_logits = torch.cat(parts, dim=1)
        else:
            fused_logits = _batched_logits(candidate_tensor)

        gt_indices = torch.tensor([int(vocab_index[int(row["gt_class_id"])]) for row in batch_rows], device=device, dtype=torch.long)
        gt_scores = fused_logits.gather(1, gt_indices.unsqueeze(1)).squeeze(1)
        ranks = (fused_logits > gt_scores.unsqueeze(1)).sum(dim=1).to(dtype=torch.float32) + 1.0
        denom = max(1, int(fused_logits.shape[1]) - 1)
        normalized = (ranks - 1.0) / float(denom)
        top1 = (torch.argmax(fused_logits, dim=1) == gt_indices).to(dtype=torch.float32)
    return normalized.detach().cpu().numpy().astype(np.float32), top1.detach().cpu().numpy().astype(np.float32)


def _summarize_minimal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage_id: str,
    split_order: Sequence[str],
) -> Dict[str, Any]:
    gt_rows = [row for row in rows if bool(row.get("gt_available_for_audit")) and row.get("normalized_gt_rank") is not None]
    split_summaries: Dict[str, Any] = {}
    split_counts: Dict[str, int] = {}
    for split in split_order:
        split_rows = [row for row in gt_rows if str(row.get("split")) == str(split)]
        split_counts[split] = int(len(split_rows))
        if not split_rows:
            split_summaries[split] = {
                "gt_count": 0,
                "mean_normalized_gt_rank": None,
                "gt_top1_hit_rate": None,
                "status": "EMPTY",
            }
            continue
        normalized = np.asarray([float(row["normalized_gt_rank"]) for row in split_rows], dtype=np.float64)
        top1 = np.asarray([float(row["gt_top1_hit_rate"]) for row in split_rows], dtype=np.float64)
        split_summaries[split] = {
            "gt_count": int(len(split_rows)),
            "mean_normalized_gt_rank": float(normalized.mean()),
            "gt_top1_hit_rate": float(top1.mean()),
            "status": "PASS",
        }
    summary = {
        "stage_id": str(stage_id),
        "status": "PASS" if rows else "EMPTY",
        "row_count": int(len(rows)),
        "gt_available_row_count": int(sum(1 for row in gt_rows)),
        "gt_count": int(len(gt_rows)),
        "mean_normalized_gt_rank": float(np.asarray([float(row["normalized_gt_rank"]) for row in gt_rows], dtype=np.float64).mean()) if gt_rows else None,
        "gt_top1_hit_rate": float(np.asarray([float(row["gt_top1_hit_rate"]) for row in gt_rows], dtype=np.float64).mean()) if gt_rows else None,
        "split_counts": split_counts,
        "split_summaries": split_summaries,
    }
    return summary


def _build_dataset_comparison(results: Mapping[str, Mapping[str, Any]], *, split_order: Sequence[str]) -> Dict[str, Any]:
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


def run_stage_minimal_split_audit(
    config: MinimalSplitAuditConfig,
    stage: str,
    *,
    prepared_inputs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    _require_dataset_name(config.dataset_name)
    if stage not in MINIMAL_STAGES:
        raise ValueError(f"stage must be one of {MINIMAL_STAGES}, got {stage!r}")
    checkpoint_path = _stage_checkpoint_path(config.output_root, stage)
    summary_path = _stage_summary_path(config.output_root, config.dataset_name, stage)
    split_order = _split_order_for_dataset(config.dataset_name)
    if not checkpoint_path.is_file():
        result = {
            "dataset_name": config.dataset_name,
            "stage": stage,
            "stage_status": "STAGE_NOT_PRESENT",
            "class_space_size": 0,
            "row_count": 0,
            "gt_available_row_count": 0,
            "gt_count": 0,
            "mean_normalized_gt_rank": None,
            "gt_top1_hit_rate": None,
            "checkpoint_path": str(checkpoint_path),
            "summary_path": str(summary_path),
            "split_counts": {split: 0 for split in split_order},
            "split_summaries": {split: {"gt_count": 0, "mean_normalized_gt_rank": None, "gt_top1_hit_rate": None, "status": "STAGE_NOT_PRESENT"} for split in split_order},
            "note": "checkpoint missing for requested stage",
            "observed_set_sources": list((prepared_inputs or {}).get("observed_set_sources", [])),
            "observed_set_semantics": list((prepared_inputs or {}).get("observed_set_semantics", [])),
        }
        result.update(_new_chain_provenance(dataset_name=config.dataset_name, split_order=split_order, stage_scope=MINIMAL_STAGES))
        write_json(summary_path, result)
        return result

    prepared = dict(prepared_inputs or _materialize_shared_inputs(config))
    observed_set_sources = list(prepared.get("observed_set_sources", []))
    observed_set_semantics = list(prepared.get("observed_set_semantics", []))
    rows, cache, vocab_index, candidate_tensor, temperature_tensor = _build_rows_and_cache(config=config, prepared=prepared, stage=stage)

    scored_rows = [row for row in rows if bool(row.get("gt_available_for_audit")) and row.get("split") in set(split_order) and int(row.get("gt_class_id")) in vocab_index]
    if rows and config.dataset_name in TRAIN_DATASETS and not scored_rows:
        raise RuntimeError(
            f"TRAIN_GT_AVAILABLE_FILTERED_TO_ZERO: row_count={len(rows)} gt_available_row_count=0_or_filtered for "
            f"dataset={config.dataset_name} stage={stage} output_root={config.output_root}"
        )
    progress_path = _stage_progress_path(config.output_root, config.dataset_name, stage)
    _write_progress(progress_path, status="RUNNING", processed_rows=0, total_rows=len(scored_rows), checkpoint_path=checkpoint_path)

    batch_size = max(1, int(config.batch_size_rows))
    for start in range(0, len(scored_rows), batch_size):
        batch_rows = scored_rows[start : start + batch_size]
        normalized, top1 = _score_batch(
            batch_rows=batch_rows,
            cache=cache,
            candidate_tensor=candidate_tensor,
            temperature_tensor=temperature_tensor,
            vocab_index=vocab_index,
            device=config.device,
            candidate_chunk_size=int(config.candidate_chunk_size),
        )
        for row, norm_rank, top1_hit in zip(batch_rows, normalized.tolist(), top1.tolist()):
            row["normalized_gt_rank"] = float(norm_rank)
            row["gt_top1_hit_rate"] = float(top1_hit)
        if ((start + len(batch_rows)) % max(1, int(config.heartbeat_every_rows)) == 0) or (start + len(batch_rows) >= len(scored_rows)):
            _write_progress(progress_path, status="RUNNING", processed_rows=start + len(batch_rows), total_rows=len(scored_rows), checkpoint_path=checkpoint_path)

    summary = _summarize_minimal_rows(rows, stage_id=stage, split_order=split_order)
    summary.update(
        {
            "dataset_name": config.dataset_name,
            "stage": stage,
            "stage_status": "STAGE_PRESENT",
            "class_space_size": int(candidate_tensor.shape[0]),
            "checkpoint_path": str(checkpoint_path),
            "summary_path": str(summary_path),
            "observed_set_sources": observed_set_sources,
            "observed_set_semantics": observed_set_semantics,
        }
    )
    summary.update(_new_chain_provenance(dataset_name=config.dataset_name, split_order=split_order, stage_scope=MINIMAL_STAGES))
    write_json(summary_path, summary)
    _write_progress(progress_path, status="COMPLETE", processed_rows=len(scored_rows), total_rows=len(scored_rows), checkpoint_path=checkpoint_path)
    return summary


def run_minimal_split_audit(config: MinimalSplitAuditConfig) -> Dict[str, Any]:
    _require_dataset_name(config.dataset_name)
    _validate_stage(config.stage)
    prepared = _materialize_shared_inputs(config)
    stage_names = _iter_stage_names(config.stage)
    results: Dict[str, Any] = {}
    for stage in stage_names:
        results[stage] = run_stage_minimal_split_audit(config, stage, prepared_inputs=prepared)
    split_order = _split_order_for_dataset(config.dataset_name)
    comparison = _build_dataset_comparison(results, split_order=split_order)
    summary = {
        "dataset_name": config.dataset_name,
        "output_root": str(config.output_root),
        "split_order": list(split_order),
        "stages": results,
        "comparison_by_split": comparison,
        "metric_scope": ["gt_count", "mean_normalized_gt_rank", "gt_top1_hit_rate"],
        "stage_scope": list(stage_names),
        "observed_set_sources": list(prepared.get("observed_set_sources", [])),
        "observed_set_semantics": list(prepared.get("observed_set_semantics", [])),
    }
    summary.update(_new_chain_provenance(dataset_name=config.dataset_name, split_order=split_order, stage_scope=stage_names))
    write_json(_dataset_summary_path(config.output_root, config.dataset_name), summary)
    comparison.update(_new_chain_provenance(dataset_name=config.dataset_name, split_order=split_order, stage_scope=stage_names))
    write_json(_package_comparison_path(config.output_root, config.dataset_name), comparison)
    return summary
