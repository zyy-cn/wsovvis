from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wsovvis.track_feature_export.stagec_semantic_slice_v1 import (
    StageCSemanticLossHookInputV1,
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
