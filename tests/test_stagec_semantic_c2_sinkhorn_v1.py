from __future__ import annotations

import numpy as np

from wsovvis.track_feature_export import (
    STAGEC_BG_LABEL_ID,
    STAGEC_UNK_FG_LABEL_ID,
    StageCSemanticBatchV1,
    normalize_stagec_semantic_batch_v1,
    run_stagec_assignment_sinkhorn_minimal_v1,
)
from wsovvis.training import build_stagec_semantic_plumbing_c1_clip_text_default


def _base_batch() -> StageCSemanticBatchV1:
    return StageCSemanticBatchV1(
        track_features=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
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


def test_sinkhorn_c2_shape_nonnegative_finite_and_mask_respected() -> None:
    batch = normalize_stagec_semantic_batch_v1(_base_batch())
    out = run_stagec_assignment_sinkhorn_minimal_v1(batch)

    assert out.backend == "c2_sinkhorn_minimal_v1"
    assert out.soft_assignment.shape == (3, 4)
    assert np.isfinite(out.soft_assignment).all()
    assert np.min(out.soft_assignment) >= 0.0
    assert np.allclose(out.soft_assignment[2], 0.0)
    assert np.allclose(out.soft_assignment[:, 3], 0.0)
    assert np.allclose(np.sum(out.soft_assignment[0]), 1.0, atol=1e-5)
    assert np.allclose(np.sum(out.soft_assignment[1]), 1.0, atol=1e-5)


def test_sinkhorn_c2_capacity_weights_raise_bg_and_unk_mass() -> None:
    n_track = 6
    batch = StageCSemanticBatchV1(
        track_features=np.zeros((n_track, 4), dtype=np.float32),
        prototype_features=np.zeros((4, 4), dtype=np.float32),
        candidate_matrix=np.ones((n_track, 4), dtype=np.float32),
        candidate_label_ids=(10, 11, STAGEC_BG_LABEL_ID, STAGEC_UNK_FG_LABEL_ID),
    )
    out = run_stagec_assignment_sinkhorn_minimal_v1(
        batch,
        bg_capacity_weight=3.0,
        unk_fg_capacity_weight=2.0,
    )
    col_sums = np.sum(out.soft_assignment, axis=0)

    assert col_sums[2] > col_sums[0]
    assert col_sums[2] > col_sums[1]
    assert col_sums[3] > col_sums[0]
    assert np.allclose(np.sum(out.soft_assignment, axis=1), 1.0, atol=1e-5)


def test_sinkhorn_c2_numeric_stability_extreme_logits() -> None:
    batch = StageCSemanticBatchV1(
        track_features=np.array([[1e6, -1e6], [-1e6, 1e6]], dtype=np.float32),
        prototype_features=np.array([[1e6, -1e6], [-1e6, 1e6], [0.0, 0.0]], dtype=np.float32),
        candidate_matrix=np.ones((2, 3), dtype=np.float32),
        candidate_label_ids=(1, 2, STAGEC_BG_LABEL_ID),
    )
    out = run_stagec_assignment_sinkhorn_minimal_v1(
        batch,
        temperature=0.05,
        iterations=30,
        eps=1e-12,
    )

    assert np.isfinite(out.soft_assignment).all()
    assert np.min(out.soft_assignment) >= 0.0
    assert np.allclose(np.sum(out.soft_assignment, axis=1), 1.0, atol=1e-5)


def test_stagec_c2_plumbing_via_c1_candidate_and_clip_path(tmp_path) -> None:
    track_features = np.array([[0.2, 0.1, 0.0, 0.3], [0.3, 0.0, 0.1, 0.2]], dtype=np.float32)
    out = build_stagec_semantic_plumbing_c1_clip_text_default(
        {
            "enabled": True,
            "loss_key": "loss_stage_c_semantic",
            "assignment_backend": "c2_sinkhorn_minimal_v1",
            "sinkhorn_iterations": 16,
            "sinkhorn_bg_capacity_weight": 2.0,
            "sinkhorn_unk_fg_capacity_weight": 2.0,
        },
        track_features=track_features,
        positive_label_ids=(3, 7),
        topk_label_ids=(7, 9),
        merge_mode="yp_plus_topk",
        include_bg=True,
        include_unk_fg=True,
        cache_root=tmp_path / "clip_cache_c2_smoke",
    )

    assert out["enabled"] is True
    assert out["assignment_backend"] == "c2_sinkhorn_minimal_v1"
    assert out["shape_summary"] == {"F": (2, 4), "G": (5, 4), "Y_hat": (2, 5), "P": (2, 5)}
    assert STAGEC_BG_LABEL_ID in out["candidate_summary"]["candidate_label_ids"]
    assert STAGEC_UNK_FG_LABEL_ID in out["candidate_summary"]["candidate_label_ids"]
