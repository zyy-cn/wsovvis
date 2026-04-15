from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_vector_from_locator
from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import (
    read_feature_vector,
    reconstruct_valid_token_mask_from_geometry,
)
from videocutler.ext_stageb_ovvis.banks.text_bank import read_text_prototype_records, resolve_text_prototype


Record = Dict[str, Any]


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None
    return (vec / norm).astype(np.float32)


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


def load_text_vocab(output_root: Path) -> Tuple[List[int], List[Record], np.ndarray]:
    text_records_path = output_root / "text_bank" / "text_prototype_records.jsonl"
    records = read_text_prototype_records(text_records_path)
    raw_ids: List[int] = []
    vectors: List[np.ndarray] = []
    for record in records:
        raw_ids.append(int(record["raw_id"]))
        vectors.append(np.asarray(resolve_text_prototype(text_records_path, record), dtype=np.float32))
    if not vectors:
        raise RuntimeError("text bank is empty")
    matrix = np.stack(vectors, axis=0).astype(np.float32)
    return raw_ids, records, matrix


def load_combined_evidence(
    sample: Mapping[str, Any],
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if trajectory_source_branch == "mainline":
        carrier_parent = output_root / "carrier_bank" / dataset_name
    elif trajectory_source_branch == "gt_upper_bound":
        carrier_parent = output_root / "carrier_bank_gt" / dataset_name
    else:
        raise ValueError(f"unsupported trajectory_source_branch: {trajectory_source_branch}")
    frame_parent = output_root / "frame_bank" / dataset_name

    carrier_record = sample.get("carrier_record")
    if not isinstance(carrier_record, Mapping):
        raise ValueError("missing carrier_record")
    z_norm_path = str(carrier_record.get("z_norm_path", ""))
    if not z_norm_path:
        raise ValueError("missing carrier z_norm_path")
    carrier_vec = np.asarray(read_vector_from_locator(carrier_parent, z_norm_path), dtype=np.float32)

    frame_rows = list(sample.get("frame_feature_rows", []))
    geom_rows = list(sample.get("frame_geometry_rows", []))
    if not frame_rows or len(frame_rows) != len(geom_rows):
        raise ValueError("frame evidence rows missing or mismatched")

    frame_vectors: List[np.ndarray] = []
    for frame_row, geom_row in zip(frame_rows, geom_rows):
        feat_path = str(frame_row.get("feat_path", ""))
        if not feat_path:
            raise ValueError("missing frame feat_path")
        feature = read_feature_vector(frame_parent, feat_path)
        token_matrix = _coerce_token_feature_matrix(
            feature,
            int(geom_row["grid_h"]),
            int(geom_row["grid_w"]),
        )
        if token_matrix is None:
            raise ValueError("frame token matrix shape mismatch")
        valid_mask = reconstruct_valid_token_mask_from_geometry(geom_row).astype(np.float32).reshape(-1)
        denom = float(np.sum(valid_mask))
        if denom <= 1e-12:
            raise ValueError("empty frame valid token mask")
        frame_vec = np.sum(token_matrix * valid_mask[:, None], axis=0).astype(np.float32) / denom
        frame_vectors.append(frame_vec)

    frame_stack = np.stack(frame_vectors, axis=0).astype(np.float32)
    frame_vec = np.mean(frame_stack, axis=0).astype(np.float32)
    combined = np.mean(np.stack([_normalize(carrier_vec), _normalize(frame_vec)], axis=0), axis=0)
    if combined is None:
        raise ValueError("combined evidence is zero norm")
    return carrier_vec.astype(np.float32), frame_vec.astype(np.float32), combined.astype(np.float32)


def fuse_carrier_frame_logits_torch(
    *,
    projector: Any,
    carrier_vec: np.ndarray,
    frame_vec: np.ndarray,
    candidate_matrix: np.ndarray,
    temperature: float,
    lambda_frame: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_matrix = np.asarray(candidate_matrix, dtype=np.float32)
    if candidate_matrix.ndim != 2:
        raise ValueError("candidate_matrix must be rank-2")
    device = next(projector.parameters()).device if hasattr(projector, "parameters") else torch.device("cpu")
    carrier_tensor = torch.from_numpy(np.asarray(carrier_vec, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    frame_tensor = torch.from_numpy(np.asarray(frame_vec, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    candidate_tensor = torch.from_numpy(candidate_matrix.astype(np.float32)).to(device=device, dtype=torch.float32)
    candidate_tensor = F.normalize(candidate_tensor, p=2.0, dim=-1)
    carrier_q = projector(carrier_tensor)
    frame_q = projector(frame_tensor)
    carrier_logits = torch.matmul(carrier_q, candidate_tensor.t()).squeeze(0) / float(temperature)
    frame_logits = torch.matmul(frame_q, candidate_tensor.t()).squeeze(0) / float(temperature)
    fused_logits = (1.0 - float(lambda_frame)) * carrier_logits + float(lambda_frame) * frame_logits
    return carrier_logits, frame_logits, fused_logits


def fuse_carrier_frame_logits(
    *,
    projector: Any,
    carrier_vec: np.ndarray,
    frame_vec: np.ndarray,
    candidate_matrix: np.ndarray,
    temperature: float,
    lambda_frame: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        carrier_logits, frame_logits, fused_logits = fuse_carrier_frame_logits_torch(
            projector=projector,
            carrier_vec=carrier_vec,
            frame_vec=frame_vec,
            candidate_matrix=candidate_matrix,
            temperature=temperature,
            lambda_frame=lambda_frame,
        )
    return (
        np.asarray(carrier_logits.detach().cpu().numpy(), dtype=np.float32),
        np.asarray(frame_logits.detach().cpu().numpy(), dtype=np.float32),
        np.asarray(fused_logits.detach().cpu().numpy(), dtype=np.float32),
    )


def observed_mass_loss(
    logits: torch.Tensor,
    observed_indices: Sequence[int],
    *,
    unknown_logit: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError("logits must be rank-1")
    observed = list(dict.fromkeys(int(i) for i in observed_indices))
    if not observed:
        raise ValueError("observed_indices cannot be empty")
    observed_logits = logits[torch.tensor(observed, device=logits.device, dtype=torch.long)]
    all_logits = torch.cat([unknown_logit.reshape(1), logits], dim=0)
    return torch.logsumexp(all_logits, dim=0) - torch.logsumexp(observed_logits, dim=0)


def _stage_allows_extra(stage_id: str) -> bool:
    return str(stage_id) == "softem_aug"


def build_stage_domain_indices(
    candidate_ids_known: Sequence[int],
    candidate_ids_extra: Sequence[int],
    *,
    stage_id: str,
) -> Tuple[List[int], List[int], List[int]]:
    known = [int(x) for x in candidate_ids_known]
    extra = [int(x) for x in candidate_ids_extra]
    if _stage_allows_extra(stage_id):
        domain = [*known, *extra]
        extra_domain = list(extra)
    else:
        domain = list(known)
        extra_domain = []
    return domain, known, extra_domain


def refine_responsibilities(
    *,
    initial_mass: Mapping[str, float],
    model_probs: Sequence[float],
    candidate_ids_known: Sequence[int],
    candidate_ids_extra: Sequence[int],
    stage_id: str,
    coverage_bonus: float = 0.15,
    extra_penalty: float = 0.05,
) -> Tuple[Dict[str, float], Dict[str, float], List[int]]:
    domain_ids, known_ids, extra_ids = build_stage_domain_indices(
        candidate_ids_known,
        candidate_ids_extra,
        stage_id=stage_id,
    )
    model_probs_arr = np.asarray(model_probs, dtype=np.float64)
    if model_probs_arr.ndim != 1 or int(model_probs_arr.shape[0]) != len(domain_ids):
        raise ValueError("model probability shape mismatch")
    init = {str(key): max(0.0, float(value)) for key, value in dict(initial_mass).items()}
    unknown_init = max(0.0, float(init.get("unknown", 0.0)))
    base_unknown = math.log(max(unknown_init, 1e-12))
    scores: Dict[str, float] = {"unknown": base_unknown + 0.0}
    weight_by_id: Dict[str, float] = {"unknown": 1.0}
    coverage_bonus_applied_to: List[int] = []

    for raw_id, model_prob in zip(domain_ids, model_probs_arr.tolist()):
        init_mass = max(0.0, float(init.get(str(int(raw_id)), 0.0)))
        score = math.log(max(model_prob, 1e-12)) + math.log(max(init_mass, 1e-12))
        weight = 1.0
        if int(raw_id) in known_ids:
            weight = 1.0 + float(coverage_bonus)
            coverage_bonus_applied_to.append(int(raw_id))
        elif int(raw_id) in extra_ids:
            weight = max(1e-6, 1.0 - float(extra_penalty))
        scores[str(int(raw_id))] = score + math.log(max(weight, 1e-12))
        weight_by_id[str(int(raw_id))] = float(weight)

    ordered_keys = ["unknown", *[str(int(raw_id)) for raw_id in domain_ids]]
    score_tensor = torch.tensor([scores[key] for key in ordered_keys], dtype=torch.float64)
    probs = torch.softmax(score_tensor, dim=0).cpu().numpy().astype(np.float64)
    final_mass = {key: float(prob) for key, prob in zip(ordered_keys, probs.tolist())}
    init_mass = {key: float(max(0.0, float(init.get(key, 0.0)))) for key in ordered_keys}
    init_mass = _normalize(init_mass) if init_mass else {"unknown": 1.0}
    final_mass = _normalize(final_mass)
    return final_mass, init_mass, sorted(set(coverage_bonus_applied_to))


def _normalize(mass: Mapping[str, float]) -> Dict[str, float]:
    total = 0.0
    normalized: Dict[str, float] = {}
    for key, value in mass.items():
        v = max(0.0, float(value))
        normalized[str(key)] = v
        total += v
    if total <= 0.0:
        return {"unknown": 1.0}
    return {key: float(value / total) for key, value in normalized.items()}
