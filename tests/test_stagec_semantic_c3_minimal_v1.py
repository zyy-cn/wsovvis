from __future__ import annotations

import numpy as np
import pytest
import torch

from wsovvis.track_feature_export import (
    STAGEC_BG_LABEL_ID,
    StageCSemanticAssignmentOutputV1,
    StageCSemanticBatchV1,
    StageCSemanticLossHookInputV1,
    compute_stagec_semantic_loss_hook_c3_minimal_v1,
)
from wsovvis.training import (
    StageCSemanticPlumbingError,
    build_stagec_semantic_plumbing_c3_minimal_coupled,
)


def test_stagec_c3_minimal_plumbing_nonzero_finite_and_backward(tmp_path) -> None:
    track = torch.nn.Parameter(
        torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.2, 0.2, 0.2, 0.2],
            ],
            dtype=torch.float32,
        )
    )
    losses: dict[str, torch.Tensor] = {}
    out = build_stagec_semantic_plumbing_c3_minimal_coupled(
        {
            "enabled": True,
            "loss_key": "loss_stage_c_semantic",
            "loss_weight": 1.0,
            "assignment_backend": "c2_sinkhorn_minimal_v1",
            "sinkhorn_iterations": 24,
            "c3_alignment_weight": 1.0,
            "c3_coverage_weight": 0.3,
            "c3_fg_not_bg_weight": 0.1,
            "c3_score_temperature": 0.2,
            "c8_temporal_consistency_enabled": True,
            "c8_temporal_consistency_weight": 0.2,
            "c8_temporal_consistency_mode": "sym_kl",
        },
        track_features_tensor=track,
        positive_label_ids=(101, 202),
        topk_label_ids=(101, 202, 303),
        merge_mode="yp_plus_topk",
        include_bg=True,
        include_unk_fg=True,
        cache_root=tmp_path / "clip_cache_c3_smoke",
        label_text_by_id={
            101: "person",
            202: "dog",
            303: "car",
            STAGEC_BG_LABEL_ID: "background",
            "__unk_fg__": "unknown foreground",
        },
        temporal_pair_indices=((0, 1), (1, 2)),
        loss_dict=losses,
    )

    assert out["assignment_backend"] == "c2_sinkhorn_minimal_v1"
    assert out["loss_applied"] is True
    loss = out["semantic_loss_tensor"]
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
    assert float(loss.detach().item()) > 0.0
    assert float(out["diagnostics"]["component_temporal_consistency"]) >= 0.0
    assert int(out["diagnostics"]["temporal_pair_count"]) == 2
    assert "loss_stage_c_semantic" in losses
    assert losses["loss_stage_c_semantic"] is loss

    loss.backward()
    assert track.grad is not None
    assert torch.isfinite(track.grad).all()
    assert float(track.grad.abs().sum().item()) > 0.0


def test_stagec_c3_minimal_plumbing_rejects_unknown_backend() -> None:
    track = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True)
    with pytest.raises(StageCSemanticPlumbingError, match="requires assignment_backend in"):
        build_stagec_semantic_plumbing_c3_minimal_coupled(
            {
                "enabled": True,
                "assignment_backend": "c0_uniform_stub_v1",
            },
            track_features_tensor=track,
            positive_label_ids=(1,),
            topk_label_ids=(1,),
            include_bg=True,
            include_unk_fg=True,
        )


def test_stagec_c3_minimal_plumbing_supports_c9_mil_backend(tmp_path) -> None:
    track = torch.nn.Parameter(
        torch.tensor(
            [
                [1.2, 0.0, 0.1, 0.0],
                [0.1, 1.1, 0.0, 0.0],
                [0.2, 0.1, 0.9, 0.0],
            ],
            dtype=torch.float32,
        )
    )
    out = build_stagec_semantic_plumbing_c3_minimal_coupled(
        {
            "enabled": True,
            "loss_key": "loss_stage_c_semantic",
            "assignment_backend": "c9_mil_minimal_v1",
            "mil_temperature": 0.12,
            "c3_alignment_weight": 1.0,
            "c3_coverage_weight": 0.3,
            "c3_fg_not_bg_weight": 0.1,
            "c3_score_temperature": 0.2,
        },
        track_features_tensor=track,
        positive_label_ids=(101, 202),
        topk_label_ids=(101, 202, 303),
        merge_mode="yp_plus_topk",
        include_bg=True,
        include_unk_fg=True,
        cache_root=tmp_path / "clip_cache_c3_mil_smoke",
        label_text_by_id={
            101: "person",
            202: "dog",
            303: "car",
            STAGEC_BG_LABEL_ID: "background",
            "__unk_fg__": "unknown foreground",
        },
    )

    assert out["assignment_backend"] == "c9_mil_minimal_v1"
    assert out["loss_applied"] is True
    loss = out["semantic_loss_tensor"]
    assert torch.isfinite(loss)
    loss.backward()
    assert track.grad is not None
    assert torch.isfinite(track.grad).all()
    assert float(track.grad.abs().sum().item()) > 0.0


def test_stagec_c3_minimal_loss_changes_when_transport_p_changes() -> None:
    batch = StageCSemanticBatchV1(
        track_features=np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        prototype_features=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        candidate_matrix=np.ones((2, 2), dtype=np.float32),
        candidate_label_ids=(1, STAGEC_BG_LABEL_ID),
        valid_track_mask=np.array([True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True], dtype=np.bool_),
    )
    track = torch.tensor(batch.track_features, dtype=torch.float32, requires_grad=True)
    proto = torch.tensor(batch.prototype_features, dtype=torch.float32)

    aligned = StageCSemanticAssignmentOutputV1(
        soft_assignment=np.array([[0.95, 0.05], [0.95, 0.05]], dtype=np.float32),
        valid_track_mask=np.array([True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True], dtype=np.bool_),
        backend="c2_sinkhorn_minimal_v1",
    )
    bg_shifted = StageCSemanticAssignmentOutputV1(
        soft_assignment=np.array([[0.05, 0.95], [0.05, 0.95]], dtype=np.float32),
        valid_track_mask=np.array([True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True], dtype=np.bool_),
        backend="c2_sinkhorn_minimal_v1",
    )

    loss_aligned, _ = compute_stagec_semantic_loss_hook_c3_minimal_v1(
        StageCSemanticLossHookInputV1(batch=batch, assignment=aligned, loss_weight=1.0),
        track_features_tensor=track,
        prototype_features_tensor=proto,
        positive_label_ids=(1,),
        alignment_weight=1.0,
        coverage_weight=0.0,
        fg_not_bg_weight=0.0,
    )
    loss_bg_shifted, _ = compute_stagec_semantic_loss_hook_c3_minimal_v1(
        StageCSemanticLossHookInputV1(batch=batch, assignment=bg_shifted, loss_weight=1.0),
        track_features_tensor=track,
        prototype_features_tensor=proto,
        positive_label_ids=(1,),
        alignment_weight=1.0,
        coverage_weight=0.0,
        fg_not_bg_weight=0.0,
    )

    assert torch.isfinite(loss_aligned)
    assert torch.isfinite(loss_bg_shifted)
    assert float(loss_aligned.detach().item()) != float(loss_bg_shifted.detach().item())


def test_stagec_c3_minimal_temporal_consistency_toggle_changes_loss() -> None:
    batch = StageCSemanticBatchV1(
        track_features=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.2, 0.8, 0.0],
                [0.0, 0.2, 0.8],
            ],
            dtype=np.float32,
        ),
        prototype_features=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        candidate_matrix=np.ones((3, 3), dtype=np.float32),
        candidate_label_ids=(1, 2, STAGEC_BG_LABEL_ID),
        valid_track_mask=np.array([True, True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True, True], dtype=np.bool_),
    )
    assignment = StageCSemanticAssignmentOutputV1(
        soft_assignment=np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
            ],
            dtype=np.float32,
        ),
        valid_track_mask=np.array([True, True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True, True], dtype=np.bool_),
        backend="c2_sinkhorn_minimal_v1",
    )
    hook_input = StageCSemanticLossHookInputV1(batch=batch, assignment=assignment, loss_weight=1.0)

    track_no_temporal = torch.tensor(batch.track_features, dtype=torch.float32, requires_grad=True)
    proto_no_temporal = torch.tensor(batch.prototype_features, dtype=torch.float32)
    loss_no_temporal, out_no_temporal = compute_stagec_semantic_loss_hook_c3_minimal_v1(
        hook_input,
        track_features_tensor=track_no_temporal,
        prototype_features_tensor=proto_no_temporal,
        positive_label_ids=(1, 2),
        coverage_weight=0.0,
        fg_not_bg_weight=0.0,
        temporal_consistency_enabled=False,
        temporal_consistency_weight=0.2,
        temporal_pair_indices=((0, 1), (1, 2)),
    )
    loss_no_temporal.backward()
    assert track_no_temporal.grad is not None
    assert torch.isfinite(track_no_temporal.grad).all()

    track_temporal = torch.tensor(batch.track_features, dtype=torch.float32, requires_grad=True)
    proto_temporal = torch.tensor(batch.prototype_features, dtype=torch.float32)
    loss_temporal, out_temporal = compute_stagec_semantic_loss_hook_c3_minimal_v1(
        hook_input,
        track_features_tensor=track_temporal,
        prototype_features_tensor=proto_temporal,
        positive_label_ids=(1, 2),
        coverage_weight=0.0,
        fg_not_bg_weight=0.0,
        temporal_consistency_enabled=True,
        temporal_consistency_weight=0.2,
        temporal_pair_indices=((0, 1), (1, 2)),
    )
    loss_temporal.backward()
    assert track_temporal.grad is not None
    assert torch.isfinite(track_temporal.grad).all()
    assert float(track_temporal.grad.abs().sum().item()) > 0.0
    assert float(out_no_temporal.diagnostics["component_temporal_consistency"]) == 0.0
    assert float(out_temporal.diagnostics["component_temporal_consistency"]) > 0.0
    assert float(loss_no_temporal.detach().item()) != float(loss_temporal.detach().item())
