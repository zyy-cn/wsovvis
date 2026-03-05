from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class StageCSemanticSliceError(ValueError):
    """Raised when Stage C semantic C0 interface inputs are invalid."""


def _err(field_path: str, rule_summary: str) -> StageCSemanticSliceError:
    return StageCSemanticSliceError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path, rule_summary)


def _validate_mask(mask: np.ndarray | None, *, name: str, expected_length: int) -> np.ndarray:
    if mask is None:
        return np.ones((expected_length,), dtype=np.bool_)
    _require(isinstance(mask, np.ndarray), name, "must be numpy ndarray when provided")
    _require(mask.ndim == 1, name, "must be rank-1")
    _require(mask.shape[0] == expected_length, name, f"length mismatch, expected {expected_length}")
    _require(mask.dtype == np.bool_, name, "must have bool dtype")
    return mask


@dataclass(frozen=True)
class StageCSemanticBatchV1:
    """C0 semantic interface contract container.

    Fields:
      - track_features (F): [N_track, D]
      - prototype_features (G): [N_cand, D]
      - candidate_matrix (Y_hat): [N_track, N_cand]
      - valid_track_mask: [N_track] bool
      - valid_column_mask: [N_cand] bool
    """

    track_features: np.ndarray
    prototype_features: np.ndarray
    candidate_matrix: np.ndarray
    candidate_label_ids: tuple[int | str, ...]
    valid_track_mask: np.ndarray | None = None
    valid_column_mask: np.ndarray | None = None


@dataclass(frozen=True)
class StageCSemanticAssignmentOutputV1:
    soft_assignment: np.ndarray
    valid_track_mask: np.ndarray
    valid_column_mask: np.ndarray
    backend: str


@dataclass(frozen=True)
class StageCSemanticLossHookInputV1:
    batch: StageCSemanticBatchV1
    assignment: StageCSemanticAssignmentOutputV1
    loss_key: str = "loss_stage_c_semantic"
    loss_weight: float = 0.0


@dataclass(frozen=True)
class StageCSemanticLossHookOutputV1:
    enabled: bool
    applied: bool
    loss_key: str
    loss_value: float
    diagnostics: Mapping[str, Any]


def normalize_stagec_semantic_batch_v1(batch: StageCSemanticBatchV1) -> StageCSemanticBatchV1:
    _require(isinstance(batch.track_features, np.ndarray), "batch.track_features", "must be numpy ndarray")
    _require(batch.track_features.ndim == 2, "batch.track_features", "must be rank-2 [N_track, D]")
    _require(batch.track_features.dtype == np.float32, "batch.track_features", "must be float32")
    _require(np.isfinite(batch.track_features).all(), "batch.track_features", "must be finite")
    n_track, embedding_dim = batch.track_features.shape
    _require(n_track >= 0, "batch.track_features.shape[0]", "must be >= 0")
    _require(embedding_dim > 0, "batch.track_features.shape[1]", "must be > 0")

    _require(isinstance(batch.prototype_features, np.ndarray), "batch.prototype_features", "must be numpy ndarray")
    _require(batch.prototype_features.ndim == 2, "batch.prototype_features", "must be rank-2 [N_cand, D]")
    _require(batch.prototype_features.dtype == np.float32, "batch.prototype_features", "must be float32")
    _require(np.isfinite(batch.prototype_features).all(), "batch.prototype_features", "must be finite")
    n_cand, proto_dim = batch.prototype_features.shape
    _require(proto_dim == embedding_dim, "batch.prototype_features.shape[1]", "must match F embedding dim")

    _require(isinstance(batch.candidate_matrix, np.ndarray), "batch.candidate_matrix", "must be numpy ndarray")
    _require(batch.candidate_matrix.ndim == 2, "batch.candidate_matrix", "must be rank-2 [N_track, N_cand]")
    _require(batch.candidate_matrix.dtype in (np.float32, np.bool_), "batch.candidate_matrix", "must be float32 or bool")
    _require(np.isfinite(batch.candidate_matrix.astype(np.float32)).all(), "batch.candidate_matrix", "must be finite")
    _require(
        batch.candidate_matrix.shape == (n_track, n_cand),
        "batch.candidate_matrix.shape",
        "must match [N_track, N_cand] from F/G",
    )

    _require(isinstance(batch.candidate_label_ids, tuple), "batch.candidate_label_ids", "must be tuple")
    _require(len(batch.candidate_label_ids) == n_cand, "batch.candidate_label_ids", "length must match N_cand")
    for idx, label_id in enumerate(batch.candidate_label_ids):
        ok = (isinstance(label_id, int) and not isinstance(label_id, bool)) or (
            isinstance(label_id, str) and bool(label_id)
        )
        _require(ok, f"batch.candidate_label_ids[{idx}]", "must be non-empty string or integer")

    valid_track_mask = _validate_mask(batch.valid_track_mask, name="batch.valid_track_mask", expected_length=n_track)
    valid_column_mask = _validate_mask(batch.valid_column_mask, name="batch.valid_column_mask", expected_length=n_cand)

    _require(
        not (n_track > 0 and n_cand > 0 and valid_column_mask.sum() == 0 and valid_track_mask.any()),
        "batch.valid_column_mask",
        "must have at least one valid column when there are valid tracks",
    )
    return StageCSemanticBatchV1(
        track_features=batch.track_features,
        prototype_features=batch.prototype_features,
        candidate_matrix=batch.candidate_matrix.astype(np.float32),
        candidate_label_ids=batch.candidate_label_ids,
        valid_track_mask=valid_track_mask,
        valid_column_mask=valid_column_mask,
    )


def build_stagec_prototype_candidate_stub_v1(
    *,
    track_features: np.ndarray,
    prototype_features: np.ndarray,
    candidate_label_ids: tuple[int | str, ...],
    candidate_matrix: np.ndarray | None = None,
    valid_track_mask: np.ndarray | None = None,
    valid_column_mask: np.ndarray | None = None,
) -> StageCSemanticBatchV1:
    n_track = int(track_features.shape[0]) if isinstance(track_features, np.ndarray) and track_features.ndim == 2 else -1
    n_cand = (
        int(prototype_features.shape[0])
        if isinstance(prototype_features, np.ndarray) and prototype_features.ndim == 2
        else len(candidate_label_ids)
    )
    if candidate_matrix is None:
        _require(n_track >= 0 and n_cand >= 0, "candidate_matrix", "cannot infer shape from invalid F/G")
        candidate_matrix = np.zeros((n_track, n_cand), dtype=np.float32)
    raw = StageCSemanticBatchV1(
        track_features=track_features,
        prototype_features=prototype_features,
        candidate_matrix=candidate_matrix,
        candidate_label_ids=candidate_label_ids,
        valid_track_mask=valid_track_mask,
        valid_column_mask=valid_column_mask,
    )
    return normalize_stagec_semantic_batch_v1(raw)


def run_stagec_assignment_stub_v1(batch: StageCSemanticBatchV1) -> StageCSemanticAssignmentOutputV1:
    normalized = normalize_stagec_semantic_batch_v1(batch)
    n_track, _ = normalized.track_features.shape
    n_cand = normalized.prototype_features.shape[0]
    valid_track_mask = normalized.valid_track_mask if normalized.valid_track_mask is not None else np.ones((n_track,), dtype=np.bool_)
    valid_column_mask = (
        normalized.valid_column_mask if normalized.valid_column_mask is not None else np.ones((n_cand,), dtype=np.bool_)
    )

    soft = np.zeros((n_track, n_cand), dtype=np.float32)
    num_valid_cols = int(valid_column_mask.sum())
    if num_valid_cols > 0:
        soft[np.ix_(valid_track_mask, valid_column_mask)] = 1.0 / float(num_valid_cols)
    return StageCSemanticAssignmentOutputV1(
        soft_assignment=soft,
        valid_track_mask=valid_track_mask,
        valid_column_mask=valid_column_mask,
        backend="c0_uniform_stub_v1",
    )


def compute_stagec_semantic_loss_hook_stub_v1(
    hook_input: StageCSemanticLossHookInputV1,
) -> StageCSemanticLossHookOutputV1:
    batch = normalize_stagec_semantic_batch_v1(hook_input.batch)
    assignment = hook_input.assignment
    _require(isinstance(assignment.soft_assignment, np.ndarray), "assignment.soft_assignment", "must be numpy ndarray")
    _require(
        assignment.soft_assignment.shape == batch.candidate_matrix.shape,
        "assignment.soft_assignment.shape",
        "must match [N_track, N_cand]",
    )
    _require(np.isfinite(assignment.soft_assignment).all(), "assignment.soft_assignment", "must be finite")
    _require(isinstance(hook_input.loss_key, str) and hook_input.loss_key, "hook_input.loss_key", "must be non-empty")
    _require(np.isfinite(float(hook_input.loss_weight)), "hook_input.loss_weight", "must be finite")
    diagnostics = {
        "interface_version": "stagec_semantic_c0_v1",
        "loss_hook_version": "stagec_semantic_loss_hook_c0_v1",
        "n_track": int(batch.track_features.shape[0]),
        "n_cand": int(batch.prototype_features.shape[0]),
        "embedding_dim": int(batch.track_features.shape[1]),
        "assignment_backend": assignment.backend,
    }
    return StageCSemanticLossHookOutputV1(
        enabled=True,
        applied=False,
        loss_key=hook_input.loss_key,
        loss_value=0.0,
        diagnostics=diagnostics,
    )
