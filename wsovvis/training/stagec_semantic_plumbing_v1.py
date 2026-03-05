from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wsovvis.track_feature_export.stagec_clip_text_prototype_cache_v1 import (
    get_or_build_stagec_clip_text_prototype_cache_v1,
)
from wsovvis.track_feature_export.stagec_semantic_slice_v1 import (
    StageCSemanticLossHookInputV1,
    build_stagec_candidate_set_v1,
    build_stagec_prototype_candidate_stub_v1,
    compute_stagec_semantic_loss_hook_stub_v1,
    run_stagec_assignment_stub_v1,
)


class StageCSemanticPlumbingError(ValueError):
    """Raised when Stage C semantic plumbing configuration is invalid."""


def _err(field_path: str, rule_summary: str) -> StageCSemanticPlumbingError:
    return StageCSemanticPlumbingError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path, rule_summary)


def build_stagec_semantic_plumbing_c0(
    raw_config: Mapping[str, Any] | None,
    *,
    track_features: np.ndarray,
    prototype_features: np.ndarray,
    candidate_label_ids: tuple[int | str, ...],
    candidate_matrix: np.ndarray | None = None,
    valid_track_mask: np.ndarray | None = None,
    valid_column_mask: np.ndarray | None = None,
    loss_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """C0 training-path hook for Stage C semantic slice.

    This function intentionally produces only a placeholder zero loss while freezing
    the interface from semantic batch -> assignment output -> loss hook output.
    """

    cfg = dict(raw_config) if isinstance(raw_config, Mapping) else {}
    enabled = cfg.get("enabled", False)
    _require(isinstance(enabled, bool), "stage_c_semantic.enabled", "must be boolean")
    if not enabled:
        return {
            "enabled": False,
            "interface_version": "stagec_semantic_c0_v1",
            "hook_status": "disabled_noop",
            "loss_applied": False,
            "loss_key": None,
            "loss_value": 0.0,
        }

    loss_key = cfg.get("loss_key", "loss_stage_c_semantic")
    _require(isinstance(loss_key, str) and loss_key, "stage_c_semantic.loss_key", "must be non-empty string")

    loss_weight = cfg.get("loss_weight", 0.0)
    _require(isinstance(loss_weight, (int, float)) and not isinstance(loss_weight, bool), "stage_c_semantic.loss_weight", "must be numeric")

    batch = build_stagec_prototype_candidate_stub_v1(
        track_features=track_features,
        prototype_features=prototype_features,
        candidate_label_ids=candidate_label_ids,
        candidate_matrix=candidate_matrix,
        valid_track_mask=valid_track_mask,
        valid_column_mask=valid_column_mask,
    )
    assignment = run_stagec_assignment_stub_v1(batch)
    loss_output = compute_stagec_semantic_loss_hook_stub_v1(
        StageCSemanticLossHookInputV1(
            batch=batch,
            assignment=assignment,
            loss_key=loss_key,
            loss_weight=float(loss_weight),
        )
    )

    inserted_into_loss_dict = False
    if isinstance(loss_dict, dict):
        if loss_output.loss_key not in loss_dict:
            loss_dict[loss_output.loss_key] = float(loss_output.loss_value)
        inserted_into_loss_dict = True

    return {
        "enabled": True,
        "interface_version": "stagec_semantic_c0_v1",
        "hook_status": "active_noop",
        "loss_applied": False,
        "loss_key": loss_output.loss_key,
        "loss_value": float(loss_output.loss_value),
        "inserted_into_loss_dict": inserted_into_loss_dict,
        "shape_summary": {
            "F": tuple(batch.track_features.shape),
            "G": tuple(batch.prototype_features.shape),
            "Y_hat": tuple(batch.candidate_matrix.shape),
            "P": tuple(assignment.soft_assignment.shape),
        },
        "mask_summary": {
            "valid_tracks": int(assignment.valid_track_mask.sum()),
            "valid_columns": int(assignment.valid_column_mask.sum()),
        },
        "diagnostics": dict(loss_output.diagnostics),
    }


def build_stagec_semantic_plumbing_c1_clip_text_default(
    raw_config: Mapping[str, Any] | None,
    *,
    track_features: np.ndarray,
    positive_label_ids: tuple[int | str, ...] | None = None,
    topk_label_ids: tuple[int | str, ...] | None = None,
    topk_score_items: tuple[tuple[int | str, float], ...] | None = None,
    topk_k: int = 0,
    merge_mode: str = "yp_plus_topk",
    track_label_ids: tuple[tuple[int | str, ...], ...] | None = None,
    include_bg: bool = False,
    include_unk_fg: bool = False,
    cache_root: str | Path = ".cache/stagec_clip_text_proto_v1",
    label_text_by_id: Mapping[int | str, str] | None = None,
    model_name: str = "clip-vit-b32",
    prompt_variant: str = "default",
    valid_track_mask: np.ndarray | None = None,
    valid_column_mask: np.ndarray | None = None,
    loss_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """C1 additive plumbing: candidate assembly + CLIP text prototype cache path."""

    _require(isinstance(track_features, np.ndarray), "track_features", "must be numpy ndarray")
    _require(track_features.ndim == 2, "track_features", "must be rank-2 [N_track, D]")
    _require(track_features.shape[1] > 0, "track_features.shape[1]", "must be > 0")
    n_track, embedding_dim = int(track_features.shape[0]), int(track_features.shape[1])

    candidate_set = build_stagec_candidate_set_v1(
        n_track=n_track,
        positive_label_ids=positive_label_ids,
        topk_label_ids=topk_label_ids,
        topk_score_items=topk_score_items,
        topk_k=topk_k,
        merge_mode=merge_mode,
        track_label_ids=track_label_ids,
        include_bg=include_bg,
        include_unk_fg=include_unk_fg,
    )
    _require(len(candidate_set.candidate_label_ids) > 0, "candidate_label_ids", "must be non-empty for enabled C1 path")

    proto_cache = get_or_build_stagec_clip_text_prototype_cache_v1(
        cache_root=cache_root,
        candidate_label_ids=candidate_set.candidate_label_ids,
        label_text_by_id=label_text_by_id,
        model_name=model_name,
        prompt_variant=prompt_variant,
        embedding_dim=embedding_dim,
    )
    result = build_stagec_semantic_plumbing_c0(
        raw_config=raw_config,
        track_features=track_features,
        prototype_features=proto_cache.prototype_features,
        candidate_label_ids=candidate_set.candidate_label_ids,
        candidate_matrix=candidate_set.candidate_matrix,
        valid_track_mask=valid_track_mask,
        valid_column_mask=valid_column_mask,
        loss_dict=loss_dict,
    )
    result["interface_version"] = "stagec_semantic_c1_v1"
    result["candidate_summary"] = {
        "merge_mode": merge_mode,
        "candidate_label_ids": tuple(candidate_set.candidate_label_ids),
        "n_cand": int(len(candidate_set.candidate_label_ids)),
    }
    result["prototype_cache"] = {
        "backend": "clip_text_default_v1",
        "cache_key": proto_cache.cache_key,
        "cache_hit": bool(proto_cache.cache_hit),
        "model_name": proto_cache.model_name,
        "prompt_variant": proto_cache.prompt_variant,
        "metadata_path": str(proto_cache.metadata_path),
        "tensor_path": str(proto_cache.tensor_path),
    }
    return result
