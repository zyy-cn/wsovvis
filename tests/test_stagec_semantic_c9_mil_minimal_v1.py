from __future__ import annotations

import numpy as np

from wsovvis.track_feature_export import (
    STAGEC_BG_LABEL_ID,
    STAGEC_UNK_FG_LABEL_ID,
    StageCSemanticBatchV1,
    run_stagec_assignment_mil_minimal_v1,
)


def _base_batch() -> StageCSemanticBatchV1:
    return StageCSemanticBatchV1(
        track_features=np.array(
            [
                [1.2, 0.0, 0.0, 0.0],
                [0.0, 1.2, 0.0, 0.0],
                [0.0, 0.0, 1.2, 0.0],
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
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        candidate_label_ids=(1, 2, STAGEC_BG_LABEL_ID, STAGEC_UNK_FG_LABEL_ID),
        valid_track_mask=np.array([True, True, False], dtype=np.bool_),
        valid_column_mask=np.array([True, True, True, False], dtype=np.bool_),
    )


def test_mil_c9_shape_nonnegative_finite_and_mask_respected() -> None:
    out = run_stagec_assignment_mil_minimal_v1(_base_batch(), temperature=0.10)
    assert out.backend == "c9_mil_minimal_v1"
    assert out.soft_assignment.shape == (3, 4)
    assert np.isfinite(out.soft_assignment).all()
    assert np.min(out.soft_assignment) >= 0.0
    assert np.allclose(out.soft_assignment[2], 0.0)
    assert np.allclose(out.soft_assignment[:, 3], 0.0)
    assert np.allclose(np.sum(out.soft_assignment[0]), 1.0, atol=1e-5)
    assert np.allclose(np.sum(out.soft_assignment[1]), 1.0, atol=1e-5)


def test_mil_c9_temperature_controls_sharpness() -> None:
    batch = StageCSemanticBatchV1(
        track_features=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        prototype_features=np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4]], dtype=np.float32),
        candidate_matrix=np.ones((2, 3), dtype=np.float32),
        candidate_label_ids=(10, 11, STAGEC_BG_LABEL_ID),
    )
    sharp = run_stagec_assignment_mil_minimal_v1(batch, temperature=0.05)
    diffuse = run_stagec_assignment_mil_minimal_v1(batch, temperature=0.40)

    sharp_top1 = float(np.mean(np.max(sharp.soft_assignment, axis=1)))
    diffuse_top1 = float(np.mean(np.max(diffuse.soft_assignment, axis=1)))
    assert sharp_top1 > diffuse_top1

