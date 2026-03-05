from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

STAGEC_BG_LABEL_ID = "__bg__"
STAGEC_UNK_FG_LABEL_ID = "__unk_fg__"


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


@dataclass(frozen=True)
class StageCCandidateSetV1:
    candidate_label_ids: tuple[int | str, ...]
    candidate_matrix: np.ndarray


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _is_valid_label_id(label_id: object) -> bool:
    return (isinstance(label_id, int) and not isinstance(label_id, bool)) or (
        isinstance(label_id, str) and bool(label_id)
    )


def _normalize_label_id(label_id: object, *, field_path: str) -> int | str:
    _require(_is_valid_label_id(label_id), field_path, "must be non-empty string or integer")
    return label_id  # type: ignore[return-value]


def _dedupe_stable_label_ids(label_ids: Iterable[int | str]) -> list[int | str]:
    seen: set[int | str] = set()
    ordered: list[int | str] = []
    for label_id in label_ids:
        if label_id in seen:
            continue
        seen.add(label_id)
        ordered.append(label_id)
    return ordered


def merge_stagec_candidate_label_ids_v1(
    *,
    positive_label_ids: Sequence[int | str] | None = None,
    topk_label_ids: Sequence[int | str] | None = None,
    merge_mode: str = "yp_plus_topk",
    include_bg: bool = False,
    include_unk_fg: bool = False,
    bg_label_id: int | str = STAGEC_BG_LABEL_ID,
    unk_fg_label_id: int | str = STAGEC_UNK_FG_LABEL_ID,
) -> tuple[int | str, ...]:
    """Build deterministic Stage C C1 candidate labels from Y'/TopK sources."""

    _require(
        merge_mode in ("yp_only", "topk_only", "yp_plus_topk"),
        "merge_mode",
        "must be one of {'yp_only','topk_only','yp_plus_topk'}",
    )
    positives = [
        _normalize_label_id(v, field_path=f"positive_label_ids[{i}]")
        for i, v in enumerate(() if positive_label_ids is None else positive_label_ids)
    ]
    topk = [
        _normalize_label_id(v, field_path=f"topk_label_ids[{i}]")
        for i, v in enumerate(() if topk_label_ids is None else topk_label_ids)
    ]

    merged: list[int | str] = []
    if merge_mode in ("yp_only", "yp_plus_topk"):
        merged.extend(positives)
    if merge_mode in ("topk_only", "yp_plus_topk"):
        merged.extend(topk)
    merged = _dedupe_stable_label_ids(merged)

    if include_bg:
        bg = _normalize_label_id(bg_label_id, field_path="bg_label_id")
        if bg not in merged:
            merged.append(bg)
    if include_unk_fg:
        unk = _normalize_label_id(unk_fg_label_id, field_path="unk_fg_label_id")
        if unk not in merged:
            merged.append(unk)
    return tuple(merged)


def select_stagec_topk_label_ids_from_scores_v1(
    *,
    label_score_items: Sequence[tuple[int | str, float]],
    topk_k: int,
) -> tuple[int | str, ...]:
    """Select TopK labels with deterministic tie-breaking.

    Scores are ranked descending. Ties are broken by first appearance index.
    Duplicated labels keep the maximum score seen.
    """

    _require(isinstance(topk_k, int), "topk_k", "must be integer")
    _require(topk_k >= 0, "topk_k", "must be >= 0")
    best_score_by_label: dict[int | str, float] = {}
    first_idx_by_label: dict[int | str, int] = {}
    for idx, (raw_label, raw_score) in enumerate(label_score_items):
        label = _normalize_label_id(raw_label, field_path=f"label_score_items[{idx}][0]")
        score = float(raw_score)
        _require(np.isfinite(score), f"label_score_items[{idx}][1]", "must be finite")
        if label not in first_idx_by_label:
            first_idx_by_label[label] = idx
        prev = best_score_by_label.get(label)
        if prev is None or score > prev:
            best_score_by_label[label] = score
    ranked = sorted(best_score_by_label.items(), key=lambda kv: (-kv[1], first_idx_by_label[kv[0]]))
    return tuple(label for label, _ in ranked[:topk_k])


def build_stagec_candidate_matrix_from_track_label_sets_v1(
    *,
    track_label_ids: Sequence[Sequence[int | str]],
    candidate_label_ids: Sequence[int | str],
) -> np.ndarray:
    """Build [N_track, N_cand] candidate membership matrix from per-track labels."""

    normalized_candidates = [
        _normalize_label_id(v, field_path=f"candidate_label_ids[{i}]") for i, v in enumerate(candidate_label_ids)
    ]
    n_track = len(track_label_ids)
    n_cand = len(normalized_candidates)
    matrix = np.zeros((n_track, n_cand), dtype=np.float32)
    col_by_label = {label_id: col for col, label_id in enumerate(normalized_candidates)}
    for row, labels in enumerate(track_label_ids):
        normalized_labels = {
            _normalize_label_id(v, field_path=f"track_label_ids[{row}]") for v in labels
        }
        for label_id in normalized_labels:
            col = col_by_label.get(label_id)
            if col is not None:
                matrix[row, col] = 1.0
    return matrix


def build_stagec_candidate_set_v1(
    *,
    n_track: int,
    positive_label_ids: Sequence[int | str] | None = None,
    topk_label_ids: Sequence[int | str] | None = None,
    topk_score_items: Sequence[tuple[int | str, float]] | None = None,
    topk_k: int = 0,
    merge_mode: str = "yp_plus_topk",
    track_label_ids: Sequence[Sequence[int | str]] | None = None,
    include_bg: bool = False,
    include_unk_fg: bool = False,
) -> StageCCandidateSetV1:
    """C1 candidate-set assembly with deterministic order and dedup."""

    _require(isinstance(n_track, int), "n_track", "must be integer")
    _require(n_track >= 0, "n_track", "must be >= 0")
    _require(isinstance(topk_k, int), "topk_k", "must be integer")
    _require(topk_k >= 0, "topk_k", "must be >= 0")
    if topk_label_ids is None and topk_score_items is not None:
        topk_label_ids = select_stagec_topk_label_ids_from_scores_v1(label_score_items=topk_score_items, topk_k=topk_k)
    elif topk_label_ids is not None and topk_k > 0:
        topk_label_ids = tuple(topk_label_ids[:topk_k])

    candidate_label_ids = merge_stagec_candidate_label_ids_v1(
        positive_label_ids=positive_label_ids,
        topk_label_ids=topk_label_ids,
        merge_mode=merge_mode,
        include_bg=include_bg,
        include_unk_fg=include_unk_fg,
    )
    n_cand = len(candidate_label_ids)
    if track_label_ids is not None:
        _require(len(track_label_ids) == n_track, "track_label_ids", "length must match n_track")
        matrix = build_stagec_candidate_matrix_from_track_label_sets_v1(
            track_label_ids=track_label_ids,
            candidate_label_ids=candidate_label_ids,
        )
    else:
        matrix = np.ones((n_track, n_cand), dtype=np.float32) if n_cand > 0 else np.zeros((n_track, 0), dtype=np.float32)
    return StageCCandidateSetV1(candidate_label_ids=candidate_label_ids, candidate_matrix=matrix)


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
        _require(_is_valid_label_id(label_id), f"batch.candidate_label_ids[{idx}]", "must be non-empty string or integer")

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


def run_stagec_assignment_sinkhorn_minimal_v1(
    batch: StageCSemanticBatchV1,
    *,
    temperature: float = 0.10,
    iterations: int = 20,
    tolerance: float = 1e-6,
    eps: float = 1e-12,
    bg_capacity_weight: float = 1.5,
    unk_fg_capacity_weight: float = 1.5,
) -> StageCSemanticAssignmentOutputV1:
    """C2 minimal masked Sinkhorn assignment.

    Semantics:
      - valid rows are normalized to row-mass ~= 1;
      - column masses follow a soft target capacity vector;
      - `candidate_matrix <= 0` entries are treated as forbidden transport edges.
      - richer diagnostics and semantic-loss coupling are intentionally deferred to C3/C4.
    """

    normalized = normalize_stagec_semantic_batch_v1(batch)
    _require(np.isfinite(float(temperature)) and float(temperature) > 0.0, "temperature", "must be finite and > 0")
    _require(isinstance(iterations, int) and iterations >= 1, "iterations", "must be integer >= 1")
    _require(np.isfinite(float(tolerance)) and float(tolerance) >= 0.0, "tolerance", "must be finite and >= 0")
    _require(np.isfinite(float(eps)) and float(eps) > 0.0, "eps", "must be finite and > 0")
    _require(
        np.isfinite(float(bg_capacity_weight)) and float(bg_capacity_weight) >= 0.0,
        "bg_capacity_weight",
        "must be finite and >= 0",
    )
    _require(
        np.isfinite(float(unk_fg_capacity_weight)) and float(unk_fg_capacity_weight) >= 0.0,
        "unk_fg_capacity_weight",
        "must be finite and >= 0",
    )

    n_track, _ = normalized.track_features.shape
    n_cand = normalized.prototype_features.shape[0]
    valid_track_mask = normalized.valid_track_mask if normalized.valid_track_mask is not None else np.ones((n_track,), dtype=np.bool_)
    valid_column_mask = (
        normalized.valid_column_mask if normalized.valid_column_mask is not None else np.ones((n_cand,), dtype=np.bool_)
    )

    soft = np.zeros((n_track, n_cand), dtype=np.float32)
    valid_rows = np.flatnonzero(valid_track_mask)
    valid_cols = np.flatnonzero(valid_column_mask)
    if valid_rows.size == 0 or valid_cols.size == 0:
        return StageCSemanticAssignmentOutputV1(
            soft_assignment=soft,
            valid_track_mask=valid_track_mask,
            valid_column_mask=valid_column_mask,
            backend="c2_sinkhorn_minimal_v1",
        )

    row_sub = normalized.track_features[valid_rows].astype(np.float64, copy=False)
    col_sub = normalized.prototype_features[valid_cols].astype(np.float64, copy=False)
    score = row_sub @ col_sub.T
    _require(np.isfinite(score).all(), "score_matrix", "must be finite")

    cand = normalized.candidate_matrix[np.ix_(valid_rows, valid_cols)].astype(np.float64, copy=False)
    allowed = cand > 0.0
    valid_label_ids = tuple(normalized.candidate_label_ids[col] for col in valid_cols)
    bg_pos = valid_label_ids.index(STAGEC_BG_LABEL_ID) if STAGEC_BG_LABEL_ID in valid_label_ids else None
    unk_pos = valid_label_ids.index(STAGEC_UNK_FG_LABEL_ID) if STAGEC_UNK_FG_LABEL_ID in valid_label_ids else None
    for i in range(allowed.shape[0]):
        if allowed[i].any():
            continue
        if bg_pos is not None:
            allowed[i, int(bg_pos)] = True
        elif unk_pos is not None:
            allowed[i, int(unk_pos)] = True
        else:
            allowed[i, :] = True

    score_masked = np.where(allowed, score, -1e12)
    row_max = np.max(score_masked, axis=1, keepdims=True)
    stable = (score_masked - row_max) / float(temperature)
    kernel = np.exp(stable)
    kernel = np.where(allowed, np.maximum(kernel, float(eps)), 0.0)
    _require(np.isfinite(kernel).all(), "kernel", "must be finite")

    target_col_mass = np.full((valid_cols.size,), 1.0, dtype=np.float64)
    if bg_pos is not None:
        target_col_mass[int(bg_pos)] *= float(bg_capacity_weight)
    if unk_pos is not None:
        target_col_mass[int(unk_pos)] *= float(unk_fg_capacity_weight)
    target_col_mass_sum = float(np.sum(target_col_mass))
    _require(target_col_mass_sum > 0.0, "target_col_mass", "sum must be > 0")
    target_col_mass = target_col_mass / target_col_mass_sum * float(valid_rows.size)

    plan = kernel.copy()
    for _ in range(int(iterations)):
        row_sums = np.maximum(np.sum(plan, axis=1, keepdims=True), float(eps))
        plan = plan / row_sums
        col_sums = np.maximum(np.sum(plan, axis=0, keepdims=True), float(eps))
        plan = plan * (target_col_mass.reshape(1, -1) / col_sums)
        row_err = float(np.sum(np.abs(np.sum(plan, axis=1) - 1.0)))
        col_err = float(np.sum(np.abs(np.sum(plan, axis=0) - target_col_mass)))
        if max(row_err, col_err) <= float(tolerance):
            break

    row_sums = np.maximum(np.sum(plan, axis=1, keepdims=True), float(eps))
    posterior = plan / row_sums
    posterior = np.where(allowed, posterior, 0.0)
    posterior = posterior / np.maximum(np.sum(posterior, axis=1, keepdims=True), float(eps))
    soft[np.ix_(valid_rows, valid_cols)] = posterior.astype(np.float32)
    return StageCSemanticAssignmentOutputV1(
        soft_assignment=soft,
        valid_track_mask=valid_track_mask,
        valid_column_mask=valid_column_mask,
        backend="c2_sinkhorn_minimal_v1",
    )


def run_stagec_assignment_mil_minimal_v1(
    batch: StageCSemanticBatchV1,
    *,
    temperature: float = 0.10,
    eps: float = 1e-12,
) -> StageCSemanticAssignmentOutputV1:
    """C9 minimal MIL-style row-wise softmax assignment.

    Semantics:
      - valid rows are normalized to row-mass ~= 1 via masked softmax;
      - `candidate_matrix <= 0` entries are forbidden edges;
      - if a row has no allowed columns, fallback prefers bg, then unk_fg, else all cols.
    """

    normalized = normalize_stagec_semantic_batch_v1(batch)
    _require(np.isfinite(float(temperature)) and float(temperature) > 0.0, "temperature", "must be finite and > 0")
    _require(np.isfinite(float(eps)) and float(eps) > 0.0, "eps", "must be finite and > 0")

    n_track, _ = normalized.track_features.shape
    n_cand = normalized.prototype_features.shape[0]
    valid_track_mask = normalized.valid_track_mask if normalized.valid_track_mask is not None else np.ones((n_track,), dtype=np.bool_)
    valid_column_mask = (
        normalized.valid_column_mask if normalized.valid_column_mask is not None else np.ones((n_cand,), dtype=np.bool_)
    )

    soft = np.zeros((n_track, n_cand), dtype=np.float32)
    valid_rows = np.flatnonzero(valid_track_mask)
    valid_cols = np.flatnonzero(valid_column_mask)
    if valid_rows.size == 0 or valid_cols.size == 0:
        return StageCSemanticAssignmentOutputV1(
            soft_assignment=soft,
            valid_track_mask=valid_track_mask,
            valid_column_mask=valid_column_mask,
            backend="c9_mil_minimal_v1",
        )

    row_sub = normalized.track_features[valid_rows].astype(np.float64, copy=False)
    col_sub = normalized.prototype_features[valid_cols].astype(np.float64, copy=False)
    score = row_sub @ col_sub.T
    _require(np.isfinite(score).all(), "score_matrix", "must be finite")

    cand = normalized.candidate_matrix[np.ix_(valid_rows, valid_cols)].astype(np.float64, copy=False)
    allowed = cand > 0.0
    valid_label_ids = tuple(normalized.candidate_label_ids[col] for col in valid_cols)
    bg_pos = valid_label_ids.index(STAGEC_BG_LABEL_ID) if STAGEC_BG_LABEL_ID in valid_label_ids else None
    unk_pos = valid_label_ids.index(STAGEC_UNK_FG_LABEL_ID) if STAGEC_UNK_FG_LABEL_ID in valid_label_ids else None
    for i in range(allowed.shape[0]):
        if allowed[i].any():
            continue
        if bg_pos is not None:
            allowed[i, int(bg_pos)] = True
        elif unk_pos is not None:
            allowed[i, int(unk_pos)] = True
        else:
            allowed[i, :] = True

    score_masked = np.where(allowed, score, -1e12)
    row_max = np.max(score_masked, axis=1, keepdims=True)
    stable = (score_masked - row_max) / float(temperature)
    exp_scores = np.exp(stable)
    exp_scores = np.where(allowed, np.maximum(exp_scores, float(eps)), 0.0)
    row_sums = np.maximum(np.sum(exp_scores, axis=1, keepdims=True), float(eps))
    posterior = exp_scores / row_sums
    posterior = np.where(allowed, posterior, 0.0)
    posterior = posterior / np.maximum(np.sum(posterior, axis=1, keepdims=True), float(eps))
    soft[np.ix_(valid_rows, valid_cols)] = posterior.astype(np.float32)
    return StageCSemanticAssignmentOutputV1(
        soft_assignment=soft,
        valid_track_mask=valid_track_mask,
        valid_column_mask=valid_column_mask,
        backend="c9_mil_minimal_v1",
    )


def summarize_stagec_assignment_observability_c4_minimal_v1(
    *,
    batch: StageCSemanticBatchV1,
    assignment: StageCSemanticAssignmentOutputV1,
    positive_label_ids: Sequence[int | str] | None = None,
    track_objectness: np.ndarray | None = None,
    high_objectness_threshold: float = 0.5,
    bg_dominance_threshold: float = 0.5,
    coverage_presence_eps: float = 1e-6,
    config_echo: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """C4 minimal observability summary for Stage C assignment outputs."""

    normalized = normalize_stagec_semantic_batch_v1(batch)
    _require(isinstance(assignment.soft_assignment, np.ndarray), "assignment.soft_assignment", "must be numpy ndarray")
    _require(
        assignment.soft_assignment.shape == normalized.candidate_matrix.shape,
        "assignment.soft_assignment.shape",
        "must match [N_track, N_cand]",
    )
    _require(np.isfinite(assignment.soft_assignment).all(), "assignment.soft_assignment", "must be finite")
    _require(
        np.isfinite(float(high_objectness_threshold)),
        "high_objectness_threshold",
        "must be finite",
    )
    _require(np.isfinite(float(bg_dominance_threshold)), "bg_dominance_threshold", "must be finite")
    _require(
        np.isfinite(float(coverage_presence_eps)) and float(coverage_presence_eps) >= 0.0,
        "coverage_presence_eps",
        "must be finite and >= 0",
    )

    n_track, n_cand = normalized.candidate_matrix.shape
    valid_track_mask = assignment.valid_track_mask if assignment.valid_track_mask is not None else np.ones((n_track,), dtype=np.bool_)
    valid_column_mask = (
        assignment.valid_column_mask if assignment.valid_column_mask is not None else np.ones((n_cand,), dtype=np.bool_)
    )
    valid_rows = np.flatnonzero(valid_track_mask)
    valid_cols = np.flatnonzero(valid_column_mask)

    valid_p = assignment.soft_assignment[np.ix_(valid_rows, valid_cols)].astype(np.float64, copy=False)
    valid_label_ids = tuple(normalized.candidate_label_ids[col] for col in valid_cols)

    bg_pos = valid_label_ids.index(STAGEC_BG_LABEL_ID) if STAGEC_BG_LABEL_ID in valid_label_ids else None
    unk_fg_pos = valid_label_ids.index(STAGEC_UNK_FG_LABEL_ID) if STAGEC_UNK_FG_LABEL_ID in valid_label_ids else None

    total_mass = float(valid_p.sum())
    bg_mass = float(valid_p[:, int(bg_pos)].sum()) if bg_pos is not None else 0.0
    unk_fg_mass = float(valid_p[:, int(unk_fg_pos)].sum()) if unk_fg_pos is not None else 0.0
    non_special_mass = max(0.0, total_mass - bg_mass - unk_fg_mass)

    assignment_mass = {
        "total_mass": total_mass,
        "bg_mass_fraction": _safe_fraction(bg_mass, total_mass),
        "unk_fg_mass_fraction": _safe_fraction(unk_fg_mass, total_mass),
        "non_special_mass_fraction": _safe_fraction(non_special_mass, total_mass),
    }

    positive_ids = tuple(
        _normalize_label_id(label_id, field_path=f"positive_label_ids[{idx}]")
        for idx, label_id in enumerate(positive_label_ids or ())
    )
    positive_set = set(positive_ids)
    positive_cols = [idx for idx, label_id in enumerate(valid_label_ids) if label_id in positive_set]
    if positive_cols and valid_rows.size > 0:
        pos_p = valid_p[:, positive_cols]
        positive_presence = 1.0 - np.prod(1.0 - np.clip(pos_p, 0.0, 1.0), axis=0)
        positives_covered = int(np.sum(positive_presence > float(coverage_presence_eps)))
    else:
        positives_covered = 0
    positives_present = int(len(positive_cols))
    positives_uncovered = max(0, positives_present - positives_covered)
    coverage = {
        "positives_total": int(len(positive_ids)),
        "positives_present_in_candidates": positives_present,
        "positives_covered": positives_covered,
        "positives_uncovered": positives_uncovered,
        "coverage_ratio_present": _safe_fraction(float(positives_covered), float(positives_present)),
    }

    if valid_rows.size > 0 and valid_cols.size > 0:
        row_sums = np.maximum(valid_p.sum(axis=1, keepdims=True), 1e-12)
        row_p = valid_p / row_sums
        row_entropy = -np.sum(row_p * np.log(np.maximum(row_p, 1e-12)), axis=1)
        top1_mass = np.max(row_p, axis=1)
        distribution = {
            "mean_row_entropy": float(np.mean(row_entropy)),
            "mean_top1_mass": float(np.mean(top1_mass)),
            "valid_row_count_for_entropy": int(row_entropy.shape[0]),
        }
    else:
        distribution = {
            "mean_row_entropy": 0.0,
            "mean_top1_mass": 0.0,
            "valid_row_count_for_entropy": 0,
        }

    if track_objectness is None:
        objectness = np.ones((n_track,), dtype=np.float64)
    else:
        _require(isinstance(track_objectness, np.ndarray), "track_objectness", "must be numpy ndarray when provided")
        _require(track_objectness.shape == (n_track,), "track_objectness.shape", "must match [N_track]")
        _require(np.isfinite(track_objectness).all(), "track_objectness", "must be finite")
        objectness = track_objectness.astype(np.float64, copy=False)
    high_objectness_mask = np.logical_and(valid_track_mask, objectness >= float(high_objectness_threshold))

    bg_prob_by_track = np.zeros((n_track,), dtype=np.float64)
    if bg_pos is not None and valid_rows.size > 0:
        bg_prob_by_track[valid_rows] = valid_p[:, int(bg_pos)]
    bg_dominant_mask = np.logical_and(high_objectness_mask, bg_prob_by_track >= float(bg_dominance_threshold))
    high_objectness_count = int(np.sum(high_objectness_mask))
    bg_dominant_count = int(np.sum(bg_dominant_mask))
    fg_not_bg_monitor = {
        "high_objectness_track_count": high_objectness_count,
        "high_objectness_bg_dominant_count": bg_dominant_count,
        "high_objectness_bg_dominant_fraction": _safe_fraction(float(bg_dominant_count), float(high_objectness_count)),
        "high_objectness_threshold": float(high_objectness_threshold),
        "bg_dominance_threshold": float(bg_dominance_threshold),
    }

    backend_echo = {
        "assignment_backend": assignment.backend,
        "special_columns_present": {
            "bg": bool(bg_pos is not None),
            "unk_fg": bool(unk_fg_pos is not None),
        },
        "config": dict(config_echo) if isinstance(config_echo, Mapping) else {},
    }
    return {
        "assignment_mass": assignment_mass,
        "coverage": coverage,
        "distribution": distribution,
        "fg_not_bg_monitor": fg_not_bg_monitor,
        "backend_echo": backend_echo,
    }


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


def compute_stagec_semantic_loss_hook_c3_minimal_v1(
    hook_input: StageCSemanticLossHookInputV1,
    *,
    track_features_tensor: Any,
    prototype_features_tensor: Any,
    positive_label_ids: Sequence[int | str] | None = None,
    track_objectness: Any | None = None,
    score_temperature: float = 0.10,
    coverage_target: float = 0.50,
    alignment_weight: float = 1.0,
    coverage_weight: float = 0.25,
    fg_not_bg_weight: float = 0.10,
    temporal_consistency_enabled: bool = False,
    temporal_consistency_weight: float = 0.0,
    temporal_consistency_mode: str = "sym_kl",
    temporal_pair_indices: Sequence[tuple[int, int]] | np.ndarray | None = None,
    eps: float = 1e-8,
) -> tuple[Any, StageCSemanticLossHookOutputV1]:
    """C3 minimal semantic loss coupling with torch autograd support."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - validated in remote canonical env
        raise StageCSemanticSliceError("torch is required for C3 minimal semantic loss path") from exc

    batch = normalize_stagec_semantic_batch_v1(hook_input.batch)
    assignment = hook_input.assignment
    _require(
        assignment.backend in {"c2_sinkhorn_minimal_v1", "c9_mil_minimal_v1"},
        "assignment.backend",
        "must be one of {'c2_sinkhorn_minimal_v1','c9_mil_minimal_v1'} for C3",
    )
    _require(isinstance(hook_input.loss_key, str) and hook_input.loss_key, "hook_input.loss_key", "must be non-empty")
    _require(np.isfinite(float(hook_input.loss_weight)), "hook_input.loss_weight", "must be finite")
    _require(np.isfinite(float(score_temperature)) and float(score_temperature) > 0.0, "score_temperature", "must be finite and > 0")
    _require(np.isfinite(float(coverage_target)) and float(coverage_target) >= 0.0, "coverage_target", "must be finite and >= 0")
    _require(np.isfinite(float(alignment_weight)) and float(alignment_weight) >= 0.0, "alignment_weight", "must be finite and >= 0")
    _require(np.isfinite(float(coverage_weight)) and float(coverage_weight) >= 0.0, "coverage_weight", "must be finite and >= 0")
    _require(np.isfinite(float(fg_not_bg_weight)) and float(fg_not_bg_weight) >= 0.0, "fg_not_bg_weight", "must be finite and >= 0")
    _require(isinstance(temporal_consistency_enabled, bool), "temporal_consistency_enabled", "must be boolean")
    _require(
        np.isfinite(float(temporal_consistency_weight)) and float(temporal_consistency_weight) >= 0.0,
        "temporal_consistency_weight",
        "must be finite and >= 0",
    )
    _require(
        temporal_consistency_mode in {"kl", "sym_kl"},
        "temporal_consistency_mode",
        "must be one of {'kl','sym_kl'}",
    )
    _require(np.isfinite(float(eps)) and float(eps) > 0.0, "eps", "must be finite and > 0")

    _require(
        tuple(track_features_tensor.shape) == tuple(batch.track_features.shape),
        "track_features_tensor.shape",
        "must match batch.track_features shape",
    )
    _require(
        tuple(prototype_features_tensor.shape) == tuple(batch.prototype_features.shape),
        "prototype_features_tensor.shape",
        "must match batch.prototype_features shape",
    )

    if track_objectness is None:
        objectness = torch.ones((batch.track_features.shape[0],), dtype=track_features_tensor.dtype, device=track_features_tensor.device)
    else:
        objectness = track_objectness.to(dtype=track_features_tensor.dtype, device=track_features_tensor.device)
        _require(tuple(objectness.shape) == (batch.track_features.shape[0],), "track_objectness.shape", "must be [N_track]")

    p = torch.as_tensor(assignment.soft_assignment, dtype=track_features_tensor.dtype, device=track_features_tensor.device)
    valid_track = torch.as_tensor(
        assignment.valid_track_mask.astype(np.float32),
        dtype=track_features_tensor.dtype,
        device=track_features_tensor.device,
    )
    valid_col = torch.as_tensor(
        assignment.valid_column_mask.astype(np.float32),
        dtype=track_features_tensor.dtype,
        device=track_features_tensor.device,
    )
    p = p * valid_track.unsqueeze(1) * valid_col.unsqueeze(0)
    p = p / torch.clamp(p.sum(dim=1, keepdim=True), min=float(eps))

    logits = (track_features_tensor @ prototype_features_tensor.transpose(0, 1)) / float(score_temperature)
    log_probs = torch.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    alignment_per_track = -(p * log_probs).sum(dim=1)
    alignment_loss = (alignment_per_track * valid_track).sum() / torch.clamp(valid_track.sum(), min=1.0)

    positives = set(positive_label_ids or ())
    pos_cols = [idx for idx, label_id in enumerate(batch.candidate_label_ids) if label_id in positives]
    if pos_cols:
        pos_p = p[:, pos_cols]
        positive_presence = 1.0 - torch.prod(1.0 - pos_p, dim=0)
        coverage_loss = torch.relu(float(coverage_target) - positive_presence).mean()
    else:
        coverage_loss = torch.zeros((), dtype=track_features_tensor.dtype, device=track_features_tensor.device)

    if STAGEC_BG_LABEL_ID in batch.candidate_label_ids:
        bg_idx = int(batch.candidate_label_ids.index(STAGEC_BG_LABEL_ID))
        fg_not_bg_loss = ((objectness * p[:, bg_idx]) * valid_track).sum() / torch.clamp(valid_track.sum(), min=1.0)
    else:
        fg_not_bg_loss = torch.zeros((), dtype=track_features_tensor.dtype, device=track_features_tensor.device)

    if temporal_pair_indices is None:
        pair_idx_array = np.zeros((0, 2), dtype=np.int64)
    else:
        pair_idx_array = np.asarray(temporal_pair_indices, dtype=np.int64)
    _require(pair_idx_array.ndim == 2 and pair_idx_array.shape[1] == 2, "temporal_pair_indices", "must be [N_pair, 2] when provided")
    if pair_idx_array.size > 0:
        _require(
            int(pair_idx_array.min()) >= 0 and int(pair_idx_array.max()) < batch.track_features.shape[0],
            "temporal_pair_indices",
            "indices must be in [0, N_track)",
        )

    temporal_pair_count = int(pair_idx_array.shape[0])
    temporal_consistency_loss = torch.zeros((), dtype=track_features_tensor.dtype, device=track_features_tensor.device)
    if temporal_consistency_enabled and float(temporal_consistency_weight) > 0.0 and temporal_pair_count > 0:
        pair_losses = []
        for src_idx, dst_idx in pair_idx_array.tolist():
            if not bool(assignment.valid_track_mask[int(src_idx)]) or not bool(assignment.valid_track_mask[int(dst_idx)]):
                continue
            src_p = torch.clamp(probs[int(src_idx)], min=float(eps))
            dst_p = torch.clamp(probs[int(dst_idx)], min=float(eps))
            kl_src_dst = torch.sum(src_p * (torch.log(src_p) - torch.log(dst_p)))
            if temporal_consistency_mode == "sym_kl":
                kl_dst_src = torch.sum(dst_p * (torch.log(dst_p) - torch.log(src_p)))
                pair_losses.append(0.5 * (kl_src_dst + kl_dst_src))
            else:
                pair_losses.append(kl_src_dst)
        if pair_losses:
            temporal_consistency_loss = torch.stack(pair_losses).mean()

    total = (
        float(alignment_weight) * alignment_loss
        + float(coverage_weight) * coverage_loss
        + float(fg_not_bg_weight) * fg_not_bg_loss
        + float(temporal_consistency_weight) * temporal_consistency_loss
    )
    total = float(hook_input.loss_weight) * total

    diagnostics = {
        "interface_version": "stagec_semantic_c3_v1",
        "loss_hook_version": "stagec_semantic_loss_hook_c3_minimal_v1",
        "n_track": int(batch.track_features.shape[0]),
        "n_cand": int(batch.prototype_features.shape[0]),
        "embedding_dim": int(batch.track_features.shape[1]),
        "assignment_backend": assignment.backend,
        "component_alignment": float(alignment_loss.detach().cpu().item()),
        "component_coverage": float(coverage_loss.detach().cpu().item()),
        "component_fg_not_bg": float(fg_not_bg_loss.detach().cpu().item()),
        "component_temporal_consistency": float(temporal_consistency_loss.detach().cpu().item()),
        "temporal_consistency_enabled": bool(temporal_consistency_enabled),
        "temporal_consistency_mode": temporal_consistency_mode,
        "temporal_pair_count": temporal_pair_count,
        "num_positive_columns": int(len(pos_cols)),
    }
    hook_output = StageCSemanticLossHookOutputV1(
        enabled=True,
        applied=True,
        loss_key=hook_input.loss_key,
        loss_value=float(total.detach().cpu().item()),
        diagnostics=diagnostics,
    )
    return total, hook_output
