from __future__ import annotations

import json
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_vector_from_locator
from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import read_feature_vector, reconstruct_valid_token_mask_from_geometry
from videocutler.ext_stageb_ovvis.banks.text_bank import read_text_prototype_records, resolve_text_prototype
from videocutler.ext_stageb_ovvis.algorithms._memory_audit import timing_checkpoint


Record = Dict[str, Any]
_LOAD_EVIDENCE_AUDIT_COUNT = 0


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None
    return (vec / norm).astype(np.float32)

    grid_tokens = int(grid_h) * int(grid_w)
    if int(feature.shape[0]) == grid_tokens:
        return feature
    if int(feature.shape[0]) == grid_tokens + 1:
        return feature[1:]
    return None


def _resolve_module_device(module: Any) -> torch.device:
    if hasattr(module, 'parameters'):
        try:
            return next(module.parameters()).device
        except StopIteration:
            pass
    return torch.device('cpu')


def _coerce_temperature_tensor(temperature: float | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        temp = temperature.to(device=device, dtype=torch.float32)
    else:
        temp = torch.tensor(float(temperature), device=device, dtype=torch.float32)
    return torch.clamp(temp, min=1e-6)


def _project_candidate_matrix(*, projector: Any, candidate_matrix: np.ndarray, device: torch.device) -> torch.Tensor:
    candidate_np = np.asarray(candidate_matrix, dtype=np.float32)
    if candidate_np.ndim != 2:
        raise ValueError('candidate_matrix must be rank-2')
    candidate_tensor = torch.from_numpy(candidate_np).to(device=device, dtype=torch.float32)
    input_dim = int(getattr(getattr(projector, 'config', None), 'input_dim', candidate_tensor.shape[-1]))
    output_dim = int(getattr(getattr(projector, 'config', None), 'output_dim', candidate_tensor.shape[-1]))
    if int(candidate_tensor.shape[-1]) == input_dim:
        candidate_tensor = projector(candidate_tensor)
    elif int(candidate_tensor.shape[-1]) == output_dim:
        candidate_tensor = F.normalize(candidate_tensor, p=2.0, dim=-1)
    else:
        raise ValueError(
            f'candidate_matrix width {int(candidate_tensor.shape[-1])} does not match projector input/output dims '
            f'({input_dim}, {output_dim})'
        )
    return F.normalize(candidate_tensor, p=2.0, dim=-1)


def load_text_vocab(output_root: Path) -> Tuple[List[int], List[Record], np.ndarray]:
    text_records_path = output_root / 'text_bank' / 'text_prototype_records.jsonl'
    records = read_text_prototype_records(text_records_path)
    raw_ids: List[int] = []
    vectors: List[np.ndarray] = []
    for record in records:
        raw_ids.append(int(record['raw_id']))
        vectors.append(np.asarray(resolve_text_prototype(text_records_path, record), dtype=np.float32))
    if not vectors:
        raise RuntimeError('text bank is empty')
    matrix = np.stack(vectors, axis=0).astype(np.float32)
    return raw_ids, records, matrix


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


def _read_jsonl(path: Path) -> List[Record]:
    rows: List[Record] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=16)
def _frame_bank_lookup(output_root_text: str, dataset_name: str) -> Tuple[Dict[Tuple[str, int], Record], Dict[Tuple[str, int], Record]]:
    output_root = Path(output_root_text)
    frame_root = output_root / 'frame_bank' / dataset_name
    frame_rows = _read_jsonl(frame_root / 'frame_records.jsonl')
    geom_rows = _read_jsonl(frame_root / 'frame_geom_records.jsonl')
    frame_lookup = {(str(row['clip_id']), int(row['frame_index'])): dict(row) for row in frame_rows}
    geom_lookup = {(str(row['clip_id']), int(row['frame_index'])): dict(row) for row in geom_rows}
    return frame_lookup, geom_lookup


def _carrier_parent_dir(output_root: Path, dataset_name: str, trajectory_source_branch: str) -> Path:
    if trajectory_source_branch == 'mainline':
        return output_root / 'carrier_bank' / dataset_name
    if trajectory_source_branch == 'gt_upper_bound':
        return output_root / 'carrier_bank_gt' / dataset_name
    raise ValueError(f'unsupported trajectory_source_branch: {trajectory_source_branch}')


def _collect_runtime_frame_vectors(
    sample: Mapping[str, Any],
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
) -> List[np.ndarray]:
    carrier_record = sample.get('carrier_record')
    if not isinstance(carrier_record, Mapping):
        raise ValueError('missing carrier_record')
    carrier_frame_paths = carrier_record.get('frame_carriers_norm_paths', None)
    if carrier_frame_paths is not None:
        frame_locators = [str(locator) for locator in list(carrier_frame_paths)]
        if not frame_locators:
            raise ValueError('empty carrier_record.frame_carriers_norm_paths')
        carrier_parent = _carrier_parent_dir(output_root, dataset_name, trajectory_source_branch)
        return [
            np.asarray(read_vector_from_locator(carrier_parent, locator), dtype=np.float32)
            for locator in frame_locators
        ]
    return []


def load_combined_evidence(
    sample: Mapping[str, Any],
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    global _LOAD_EVIDENCE_AUDIT_COUNT
    carrier_parent = _carrier_parent_dir(output_root, dataset_name, trajectory_source_branch)
    audit_index = int(_LOAD_EVIDENCE_AUDIT_COUNT)
    _LOAD_EVIDENCE_AUDIT_COUNT += 1
    audit_enabled = audit_index < 3
    audit_t0 = time.perf_counter()

    carrier_record = sample.get('carrier_record')
    if not isinstance(carrier_record, Mapping):
        raise ValueError('missing carrier_record')
    z_norm_path = str(carrier_record.get('z_norm_path', ''))
    if not z_norm_path:
        raise ValueError('missing carrier z_norm_path')
    if audit_enabled:
        timing_checkpoint(
            'load_combined_evidence_before_carrier_vec',
            started_at=audit_t0,
            trajectory_id=str(sample.get('trajectory_id', '')),
            z_norm_path=z_norm_path,
        )
    carrier_vec = np.asarray(read_vector_from_locator(carrier_parent, z_norm_path), dtype=np.float32)
    if audit_enabled:
        timing_checkpoint(
            'load_combined_evidence_after_carrier_vec',
            started_at=audit_t0,
            trajectory_id=str(sample.get('trajectory_id', '')),
            carrier_vec_shape=getattr(carrier_vec, 'shape', None),
        )

    frame_vectors = _collect_runtime_frame_vectors(
        sample,
        output_root=output_root,
        dataset_name=dataset_name,
        trajectory_source_branch=trajectory_source_branch,
    )
    if audit_enabled:
        timing_checkpoint(
            'load_combined_evidence_after_frame_vectors',
            started_at=audit_t0,
            trajectory_id=str(sample.get('trajectory_id', '')),
            frame_vectors_count=len(frame_vectors),
        )
    if frame_vectors:
        frame_stack = np.stack([np.asarray(vec, dtype=np.float32) for vec in frame_vectors], axis=0).astype(np.float32)
        frame_vec = np.mean(frame_stack, axis=0).astype(np.float32)
    else:
        raise ValueError('missing runtime frame evidence: carrier_record.frame_carriers_norm_paths')
    if audit_enabled:
        timing_checkpoint(
            'load_combined_evidence_after_frame_vec',
            started_at=audit_t0,
            trajectory_id=str(sample.get('trajectory_id', '')),
            frame_vec_shape=getattr(frame_vec, 'shape', None),
            frame_vectors_count=len(frame_vectors),
        )

    carrier_norm = _normalize(carrier_vec)
    frame_norm = _normalize(frame_vec)
    if carrier_norm is None or frame_norm is None:
        raise ValueError('combined evidence is zero norm')
    combined = np.mean(np.stack([carrier_norm, frame_norm], axis=0), axis=0).astype(np.float32)
    if audit_enabled:
        timing_checkpoint(
            'load_combined_evidence_after_combined_vec',
            started_at=audit_t0,
            trajectory_id=str(sample.get('trajectory_id', '')),
            combined_shape=getattr(combined, 'shape', None),
        )
    return carrier_vec.astype(np.float32), [np.asarray(vec, dtype=np.float32) for vec in frame_vectors], frame_vec.astype(np.float32), combined.astype(np.float32)


def fuse_carrier_frame_logits_torch(
    *,
    projector: Any,
    carrier_vec: np.ndarray,
    frame_vec: np.ndarray,
    candidate_matrix: np.ndarray,
    temperature: float | torch.Tensor,
    lambda_frame: float = 0.25,
    frame_vectors: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = _resolve_module_device(projector)
    temperature_tensor = _coerce_temperature_tensor(temperature, device=device)
    candidate_tensor = _project_candidate_matrix(projector=projector, candidate_matrix=candidate_matrix, device=device)
    carrier_tensor = torch.from_numpy(np.asarray(carrier_vec, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
    carrier_tensor = F.normalize(carrier_tensor, p=2.0, dim=-1)
    carrier_logits = torch.matmul(carrier_tensor, candidate_tensor.t()).squeeze(0) / temperature_tensor
    if frame_vectors is not None:
        frame_list = [np.asarray(vec, dtype=np.float32) for vec in frame_vectors]
        if frame_list:
            frame_tensor = torch.from_numpy(np.stack(frame_list, axis=0).astype(np.float32)).to(device=device, dtype=torch.float32)
            frame_tensor = F.normalize(frame_tensor, p=2.0, dim=-1)
            frame_logits = torch.matmul(frame_tensor, candidate_tensor.t()) / temperature_tensor
            frame_logits = frame_logits.mean(dim=0)
        else:
            frame_vectors = None
    if frame_vectors is None:
        frame_tensor = torch.from_numpy(np.asarray(frame_vec, dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0)
        frame_tensor = F.normalize(frame_tensor, p=2.0, dim=-1)
        frame_logits = torch.matmul(frame_tensor, candidate_tensor.t()).squeeze(0) / temperature_tensor
    fused_logits = (1.0 - float(lambda_frame)) * carrier_logits + float(lambda_frame) * frame_logits
    return carrier_logits, frame_logits, fused_logits


def fuse_carrier_frame_logits(
    *,
    projector: Any,
    carrier_vec: np.ndarray,
    frame_vec: np.ndarray,
    candidate_matrix: np.ndarray,
    temperature: float | torch.Tensor,
    lambda_frame: float = 0.25,
    frame_vectors: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        carrier_logits, frame_logits, fused_logits = fuse_carrier_frame_logits_torch(
            projector=projector,
            carrier_vec=carrier_vec,
            frame_vec=frame_vec,
            candidate_matrix=candidate_matrix,
            temperature=temperature,
            lambda_frame=lambda_frame,
            frame_vectors=frame_vectors,
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
        raise ValueError('logits must be rank-1')
    observed = list(dict.fromkeys(int(i) for i in observed_indices))
    if not observed:
        raise ValueError('observed_indices cannot be empty')
    observed_logits = logits[torch.tensor(observed, device=logits.device, dtype=torch.long)]
    all_logits = torch.cat([unknown_logit.reshape(1), logits], dim=0)
    return torch.logsumexp(all_logits, dim=0) - torch.logsumexp(observed_logits, dim=0)


def _stage_allows_extra(stage_id: str) -> bool:
    return str(stage_id) == 'softem_aug'


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
    model_logits: Sequence[float],
    candidate_ids_known: Sequence[int],
    candidate_ids_extra: Sequence[int],
    stage_id: str,
    coverage_bonus: float = 0.1,
    coverage_epsilon: float = 1.0,
    extra_penalty: float = 0.1,
    b_u_value: float = 0.0,
    coverage_context: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    domain_ids, known_ids, extra_ids = build_stage_domain_indices(
        candidate_ids_known,
        candidate_ids_extra,
        stage_id=stage_id,
    )
    model_logits_arr = np.asarray(model_logits, dtype=np.float64)
    if model_logits_arr.ndim != 1 or int(model_logits_arr.shape[0]) != len(domain_ids):
        raise ValueError('model logit shape mismatch')
    init = {str(key): max(0.0, float(value)) for key, value in dict(initial_mass).items()}
    scores: Dict[str, float] = {'unknown': float(b_u_value)}
    coverage_bonus_applied_to: List[int] = []
    extra_penalty_applied_to: List[int] = []
    coverage_map = {str(key): max(0.0, float(value)) for key, value in dict(coverage_context or {}).items()}
    for raw_id, model_logit in zip(domain_ids, model_logits_arr.tolist()):
        score = float(model_logit)
        if int(raw_id) in known_ids:
            coverage_mass = max(0.0, float(coverage_map.get(str(int(raw_id)), 0.0)))
            coverage_term = float(coverage_bonus) * math.log(float(coverage_epsilon) + max(coverage_mass, 0.0))
            score = score + coverage_term
            coverage_bonus_applied_to.append(int(raw_id))
        elif int(raw_id) in extra_ids:
            score = score - float(extra_penalty)
            extra_penalty_applied_to.append(int(raw_id))
        scores[str(int(raw_id))] = score

    ordered_keys = ['unknown', *[str(int(raw_id)) for raw_id in domain_ids]]
    score_tensor = torch.tensor([scores[key] for key in ordered_keys], dtype=torch.float64)
    probs = torch.softmax(score_tensor, dim=0).cpu().numpy().astype(np.float64)
    refined_final_mass = {key: float(prob) for key, prob in zip(ordered_keys, probs.tolist())}
    refined_init_mass = {key: float(max(0.0, float(init.get(key, 0.0)))) for key in ordered_keys}
    refined_init_mass = _normalize_mass_dict(refined_init_mass) if refined_init_mass else {'unknown': 1.0}
    refined_final_mass = _normalize_mass_dict(refined_final_mass)
    refine_trace: Dict[str, Any] = {
        'domain_ids': [int(x) for x in domain_ids],
        'known_ids': [int(x) for x in known_ids],
        'extra_ids': [int(x) for x in extra_ids],
        'coverage_bonus_applied_to': sorted(set(int(x) for x in coverage_bonus_applied_to)),
        'extra_penalty_applied_to': sorted(set(int(x) for x in extra_penalty_applied_to)),
        'b_u': float(b_u_value),
        'init_mass': dict(refined_init_mass),
        'final_mass': dict(refined_final_mass),
    }
    return refined_init_mass, refined_final_mass, refine_trace


def _normalize_mass_dict(mass: Mapping[str, float]) -> Dict[str, float]:
    total = 0.0
    normalized: Dict[str, float] = {}
    for key, value in mass.items():
        v = max(0.0, float(value))
        normalized[str(key)] = v
        total += v
    if total <= 0.0:
        return {'unknown': 1.0}
    if abs(total - 1.0) <= 1e-12:
        return {key: float(value) for key, value in normalized.items()}
    return {key: float(value / total) for key, value in normalized.items()}
