from __future__ import annotations

import numpy as np

from wsovvis.track_feature_export import (
    STAGEC_BG_LABEL_ID,
    STAGEC_UNK_FG_LABEL_ID,
    StageCSemanticBatchV1,
    run_stagec_assignment_em_minimal_v1,
)


def _base_batch() -> StageCSemanticBatchV1:
    return StageCSemanticBatchV1(
        track_features=np.array(
            [
                [1.3, 0.0, 0.1, 0.0],
                [0.1, 1.2, 0.0, 0.0],
                [0.0, 0.0, 1.1, 0.0],
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
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        candidate_label_ids=(1, 2, STAGEC_BG_LABEL_ID, STAGEC_UNK_FG_LABEL_ID),
        valid_track_mask=np.array([True, True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True, True, True], dtype=np.bool_),
    )


def test_em_c9_shape_nonnegative_finite_and_contract_compatible() -> None:
    out = run_stagec_assignment_em_minimal_v1(_base_batch(), temperature=0.10, em_iterations=4)
    assert out.backend == "c9_em_minimal_v1"
    assert out.soft_assignment.shape == (3, 4)
    assert np.isfinite(out.soft_assignment).all()
    assert np.min(out.soft_assignment) >= 0.0
    assert np.allclose(np.sum(out.soft_assignment[0]), 1.0, atol=1e-5)
    assert np.allclose(np.sum(out.soft_assignment[1]), 1.0, atol=1e-5)
    assert np.allclose(np.sum(out.soft_assignment[2]), 1.0, atol=1e-5)
    assert out.soft_assignment[0, 2] == 0.0


def test_em_c9_no_nan_or_inf_extreme_scores() -> None:
    batch = StageCSemanticBatchV1(
        track_features=np.array([[200.0, 0.0], [0.0, 200.0]], dtype=np.float32),
        prototype_features=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
        candidate_matrix=np.ones((2, 3), dtype=np.float32),
        candidate_label_ids=(10, 11, STAGEC_BG_LABEL_ID),
        valid_track_mask=np.array([True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True, True], dtype=np.bool_),
    )
    out = run_stagec_assignment_em_minimal_v1(batch, temperature=0.05, em_iterations=6)
    assert np.isfinite(out.soft_assignment).all()
    assert not np.isnan(out.soft_assignment).any()
    assert np.allclose(np.sum(out.soft_assignment, axis=1), np.array([1.0, 1.0]), atol=1e-5)

