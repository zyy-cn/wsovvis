from __future__ import annotations

import numpy as np
import pytest

from wsovvis.track_feature_export import (
    StageCSemanticBatchV1,
    StageCSemanticSliceError,
    normalize_stagec_semantic_batch_v1,
    run_stagec_assignment_stub_v1,
)
from wsovvis.training import build_stagec_semantic_plumbing_c0


def _nominal_batch() -> StageCSemanticBatchV1:
    return StageCSemanticBatchV1(
        track_features=np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        prototype_features=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        candidate_matrix=np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        candidate_label_ids=(10, 11, "__bg__", "__unk_fg__"),
        valid_track_mask=np.array([True, True, False], dtype=np.bool_),
        valid_column_mask=np.array([True, True, False, True], dtype=np.bool_),
    )


def test_stagec_semantic_batch_contract_nominal_shapes_and_masks() -> None:
    batch = normalize_stagec_semantic_batch_v1(_nominal_batch())
    assert batch.track_features.shape == (3, 4)
    assert batch.prototype_features.shape == (4, 4)
    assert batch.candidate_matrix.shape == (3, 4)
    assert tuple(batch.candidate_label_ids) == (10, 11, "__bg__", "__unk_fg__")
    assert batch.valid_track_mask is not None and batch.valid_track_mask.dtype == np.bool_
    assert batch.valid_column_mask is not None and batch.valid_column_mask.dtype == np.bool_


def test_stagec_semantic_batch_contract_rejects_dimension_mismatch() -> None:
    bad = _nominal_batch()
    bad = StageCSemanticBatchV1(
        track_features=bad.track_features,
        prototype_features=bad.prototype_features[:, :3],
        candidate_matrix=bad.candidate_matrix,
        candidate_label_ids=bad.candidate_label_ids,
        valid_track_mask=bad.valid_track_mask,
        valid_column_mask=bad.valid_column_mask,
    )
    with pytest.raises(StageCSemanticSliceError, match="must match F embedding dim"):
        normalize_stagec_semantic_batch_v1(bad)


def test_stagec_semantic_assignment_stub_respects_masks() -> None:
    batch = normalize_stagec_semantic_batch_v1(_nominal_batch())
    out = run_stagec_assignment_stub_v1(batch)
    assert out.soft_assignment.shape == (3, 4)
    assert np.allclose(out.soft_assignment[2], 0.0)
    assert np.allclose(out.soft_assignment[:, 2], 0.0)
    assert np.allclose(out.soft_assignment[0].sum(), 1.0)
    assert np.allclose(out.soft_assignment[1].sum(), 1.0)


def test_stagec_semantic_plumbing_c0_hook_callable_with_loss_dict() -> None:
    batch = _nominal_batch()
    loss_dict: dict[str, float] = {}
    result = build_stagec_semantic_plumbing_c0(
        {"enabled": True, "loss_key": "loss_stage_c_semantic"},
        track_features=batch.track_features,
        prototype_features=batch.prototype_features,
        candidate_label_ids=batch.candidate_label_ids,
        candidate_matrix=batch.candidate_matrix,
        valid_track_mask=batch.valid_track_mask,
        valid_column_mask=batch.valid_column_mask,
        loss_dict=loss_dict,
    )
    assert result["enabled"] is True
    assert result["hook_status"] == "active_noop"
    assert result["shape_summary"] == {"F": (3, 4), "G": (4, 4), "Y_hat": (3, 4), "P": (3, 4)}
    assert "loss_stage_c_semantic" in loss_dict
    assert float(loss_dict["loss_stage_c_semantic"]) == 0.0
