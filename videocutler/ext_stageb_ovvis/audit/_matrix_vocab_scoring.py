from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    _coerce_temperature_tensor,
    _project_candidate_matrix,
    _resolve_module_device,
    load_carrier_evidence,
)

Record = Dict[str, Any]


def _iter_batch_slices(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    if batch_size <= 0:
        batch_size = total
    start = 0
    while start < total:
        end = min(total, start + batch_size)
        yield start, end
        start = end


def build_carrier_matrix_pack(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: Any,
    dataset_name: str,
    trajectory_source_branch: str,
) -> Dict[str, Any]:
    carrier_vectors: List[np.ndarray] = []
    kept_rows: List[Dict[str, Any]] = []
    for row in rows:
        carrier_vec = load_carrier_evidence(
            row,
            output_root=output_root,
            dataset_name=dataset_name,
            trajectory_source_branch=trajectory_source_branch,
        )
        carrier_vectors.append(np.asarray(carrier_vec, dtype=np.float32))
        kept_rows.append(dict(row))
    if not carrier_vectors:
        return {
            "carrier_matrix": np.zeros((0, 0), dtype=np.float32),
            "rows": [],
        }
    return {
        "carrier_matrix": np.stack(carrier_vectors, axis=0).astype(np.float32),
        "rows": kept_rows,
    }


def _prepare_projected_candidate_tensor(
    *,
    projector: Any,
    candidate_matrix: np.ndarray,
) -> Tuple[torch.device, torch.Tensor]:
    device = _resolve_module_device(projector)
    candidate_tensor = _project_candidate_matrix(
        projector=projector,
        candidate_matrix=np.asarray(candidate_matrix, dtype=np.float32),
        device=device,
    )
    return device, candidate_tensor


def compute_fused_logits_matrix_numpy(
    *,
    carrier_matrix: np.ndarray,
    projector: Any,
    candidate_matrix: np.ndarray,
    temperature: float | torch.Tensor,
    batch_size: int = 512,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    carrier_np = np.asarray(carrier_matrix, dtype=np.float32)
    if carrier_np.ndim != 2:
        raise ValueError("carrier_matrix must be rank-2")
    if int(carrier_np.shape[0]) == 0:
        candidate_np = np.asarray(candidate_matrix, dtype=np.float32)
        width = int(candidate_np.shape[0]) if candidate_np.ndim == 2 else 0
        return np.zeros((0, width), dtype=np.float32)
    device, candidate_tensor = _prepare_projected_candidate_tensor(
        projector=projector,
        candidate_matrix=candidate_matrix,
    )
    temperature_tensor = _coerce_temperature_tensor(temperature, device=device)
    outputs: List[np.ndarray] = []
    total = int(carrier_np.shape[0])
    with torch.no_grad():
        for start, end in _iter_batch_slices(total, int(batch_size)):
            carrier_tensor = torch.from_numpy(carrier_np[start:end]).to(device=device, dtype=torch.float32)
            carrier_tensor = F.normalize(carrier_tensor, p=2.0, dim=-1)
            logits_t = torch.matmul(carrier_tensor, candidate_tensor.t()) / temperature_tensor
            outputs.append(np.asarray(logits_t.detach().cpu().numpy(), dtype=np.float32))
            if progress_callback is not None:
                progress_callback(int(end), int(total))
    return np.concatenate(outputs, axis=0).astype(np.float32) if outputs else np.zeros((0, int(candidate_tensor.shape[0])), dtype=np.float32)


def compute_fused_logits_and_cosine_matrix_numpy(
    *,
    carrier_matrix: np.ndarray,
    projector: Any,
    candidate_matrix: np.ndarray,
    temperature: float | torch.Tensor,
    batch_size: int = 512,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    carrier_np = np.asarray(carrier_matrix, dtype=np.float32)
    if carrier_np.ndim != 2:
        raise ValueError("carrier_matrix must be rank-2")
    if int(carrier_np.shape[0]) == 0:
        candidate_np = np.asarray(candidate_matrix, dtype=np.float32)
        width = int(candidate_np.shape[0]) if candidate_np.ndim == 2 else 0
        empty = np.zeros((0, width), dtype=np.float32)
        return empty, empty
    device, candidate_tensor = _prepare_projected_candidate_tensor(
        projector=projector,
        candidate_matrix=candidate_matrix,
    )
    temperature_tensor = _coerce_temperature_tensor(temperature, device=device)
    logits_parts: List[np.ndarray] = []
    cosine_parts: List[np.ndarray] = []
    total = int(carrier_np.shape[0])
    with torch.no_grad():
        for start, end in _iter_batch_slices(total, int(batch_size)):
            carrier_tensor = torch.from_numpy(carrier_np[start:end]).to(device=device, dtype=torch.float32)
            carrier_tensor = F.normalize(carrier_tensor, p=2.0, dim=-1)
            cosine_t = torch.matmul(carrier_tensor, candidate_tensor.t())
            logits_t = cosine_t / temperature_tensor
            logits_parts.append(np.asarray(logits_t.detach().cpu().numpy(), dtype=np.float32))
            cosine_parts.append(np.asarray(cosine_t.detach().cpu().numpy(), dtype=np.float32))
            if progress_callback is not None:
                progress_callback(int(end), int(total))
    logits = np.concatenate(logits_parts, axis=0).astype(np.float32) if logits_parts else np.zeros((0, int(candidate_tensor.shape[0])), dtype=np.float32)
    cosine = np.concatenate(cosine_parts, axis=0).astype(np.float32) if cosine_parts else np.zeros_like(logits)
    return logits, cosine


def compute_rank_metrics_batched(
    *,
    carrier_matrix: np.ndarray,
    projector: Any,
    candidate_matrix: np.ndarray,
    temperature: float | torch.Tensor,
    gt_indices: Sequence[int],
    batch_size: int = 512,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, np.ndarray]:
    carrier_np = np.asarray(carrier_matrix, dtype=np.float32)
    if carrier_np.ndim != 2:
        raise ValueError("carrier_matrix must be rank-2")
    gt_idx_np = np.asarray(list(gt_indices), dtype=np.int64)
    if int(carrier_np.shape[0]) != int(gt_idx_np.shape[0]):
        raise ValueError("gt_indices length must match carrier_matrix rows")
    if int(carrier_np.shape[0]) == 0:
        empty_i64 = np.zeros((0,), dtype=np.int64)
        empty_f32 = np.zeros((0,), dtype=np.float32)
        empty_bool = np.zeros((0,), dtype=np.bool_)
        return {
            "rank": empty_i64,
            "normalized_rank": empty_f32,
            "top1": empty_bool,
            "top5": empty_bool,
            "top10": empty_bool,
            "mrr": empty_f32,
            "margin_to_best_wrong": empty_f32,
            "top1_index": empty_i64,
        }
    device, candidate_tensor = _prepare_projected_candidate_tensor(
        projector=projector,
        candidate_matrix=candidate_matrix,
    )
    temperature_tensor = _coerce_temperature_tensor(temperature, device=device)
    total = int(carrier_np.shape[0])
    class_count = int(candidate_tensor.shape[0])
    denom = max(1, class_count - 1)
    rank_parts: List[np.ndarray] = []
    normalized_parts: List[np.ndarray] = []
    top1_parts: List[np.ndarray] = []
    top5_parts: List[np.ndarray] = []
    top10_parts: List[np.ndarray] = []
    mrr_parts: List[np.ndarray] = []
    margin_parts: List[np.ndarray] = []
    top1_idx_parts: List[np.ndarray] = []
    with torch.no_grad():
        for start, end in _iter_batch_slices(total, int(batch_size)):
            carrier_tensor = torch.from_numpy(carrier_np[start:end]).to(device=device, dtype=torch.float32)
            carrier_tensor = F.normalize(carrier_tensor, p=2.0, dim=-1)
            logits_t = torch.matmul(carrier_tensor, candidate_tensor.t()) / temperature_tensor
            gt_idx_t = torch.from_numpy(gt_idx_np[start:end]).to(device=device, dtype=torch.long)
            row_ids = torch.arange(int(end - start), device=device, dtype=torch.long)
            gt_scores_t = logits_t[row_ids, gt_idx_t]
            rank_t = torch.sum(logits_t > gt_scores_t.unsqueeze(1), dim=1, dtype=torch.long) + 1
            top1_idx_t = torch.argmax(logits_t, dim=1)
            top1_t = top1_idx_t.eq(gt_idx_t)
            top5_t = rank_t.le(5)
            top10_t = rank_t.le(10)
            normalized_t = (rank_t.to(dtype=torch.float32) - 1.0) / float(denom)
            mrr_t = 1.0 / rank_t.to(dtype=torch.float32)
            masked_logits_t = logits_t.clone()
            masked_logits_t[row_ids, gt_idx_t] = -torch.inf
            best_wrong_t = torch.max(masked_logits_t, dim=1).values
            margin_t = gt_scores_t - best_wrong_t
            rank_parts.append(np.asarray(rank_t.detach().cpu().numpy(), dtype=np.int64))
            normalized_parts.append(np.asarray(normalized_t.detach().cpu().numpy(), dtype=np.float32))
            top1_parts.append(np.asarray(top1_t.detach().cpu().numpy(), dtype=np.bool_))
            top5_parts.append(np.asarray(top5_t.detach().cpu().numpy(), dtype=np.bool_))
            top10_parts.append(np.asarray(top10_t.detach().cpu().numpy(), dtype=np.bool_))
            mrr_parts.append(np.asarray(mrr_t.detach().cpu().numpy(), dtype=np.float32))
            margin_parts.append(np.asarray(margin_t.detach().cpu().numpy(), dtype=np.float32))
            top1_idx_parts.append(np.asarray(top1_idx_t.detach().cpu().numpy(), dtype=np.int64))
            if progress_callback is not None:
                progress_callback(int(end), int(total))
    return {
        "rank": np.concatenate(rank_parts, axis=0).astype(np.int64),
        "normalized_rank": np.concatenate(normalized_parts, axis=0).astype(np.float32),
        "top1": np.concatenate(top1_parts, axis=0).astype(np.bool_),
        "top5": np.concatenate(top5_parts, axis=0).astype(np.bool_),
        "top10": np.concatenate(top10_parts, axis=0).astype(np.bool_),
        "mrr": np.concatenate(mrr_parts, axis=0).astype(np.float32),
        "margin_to_best_wrong": np.concatenate(margin_parts, axis=0).astype(np.float32),
        "top1_index": np.concatenate(top1_idx_parts, axis=0).astype(np.int64),
    }
