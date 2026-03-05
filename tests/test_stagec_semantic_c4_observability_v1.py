from __future__ import annotations

import math

import numpy as np

from wsovvis.track_feature_export import (
    STAGEC_BG_LABEL_ID,
    STAGEC_UNK_FG_LABEL_ID,
    StageCSemanticAssignmentOutputV1,
    StageCSemanticBatchV1,
    summarize_stagec_assignment_observability_c4_minimal_v1,
)
from wsovvis.training import build_stagec_semantic_plumbing_c1_clip_text_default


def _batch_for_observability() -> StageCSemanticBatchV1:
    return StageCSemanticBatchV1(
        track_features=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.3, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        prototype_features=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.6, 0.6, 0.6],
            ],
            dtype=np.float32,
        ),
        candidate_matrix=np.ones((3, 4), dtype=np.float32),
        candidate_label_ids=(101, 202, STAGEC_BG_LABEL_ID, STAGEC_UNK_FG_LABEL_ID),
        valid_track_mask=np.array([True, True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True, True, True], dtype=np.bool_),
    )


def test_stagec_c4_observability_summary_keys_ranges_and_fg_not_bg_monitor() -> None:
    batch = _batch_for_observability()
    assignment = StageCSemanticAssignmentOutputV1(
        soft_assignment=np.array(
            [
                [0.80, 0.10, 0.10, 0.00],
                [0.05, 0.05, 0.70, 0.20],
                [0.40, 0.40, 0.10, 0.10],
            ],
            dtype=np.float32,
        ),
        valid_track_mask=np.array([True, True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True, True, True], dtype=np.bool_),
        backend="c2_sinkhorn_minimal_v1",
    )

    summary = summarize_stagec_assignment_observability_c4_minimal_v1(
        batch=batch,
        assignment=assignment,
        positive_label_ids=(101, 202, 999),
        track_objectness=np.array([0.90, 0.95, 0.20], dtype=np.float32),
        config_echo={"sinkhorn_iterations": 20},
    )

    for key in ("assignment_mass", "coverage", "distribution", "fg_not_bg_monitor", "backend_echo"):
        assert key in summary

    mass = summary["assignment_mass"]
    assert 0.0 <= mass["bg_mass_fraction"] <= 1.0
    assert 0.0 <= mass["unk_fg_mass_fraction"] <= 1.0
    assert 0.0 <= mass["non_special_mass_fraction"] <= 1.0
    assert math.isfinite(mass["bg_mass_fraction"])
    assert math.isfinite(mass["unk_fg_mass_fraction"])
    assert math.isfinite(mass["non_special_mass_fraction"])
    assert math.isclose(
        mass["bg_mass_fraction"] + mass["unk_fg_mass_fraction"] + mass["non_special_mass_fraction"],
        1.0,
        rel_tol=1e-6,
    )

    coverage = summary["coverage"]
    assert coverage["positives_total"] == 3
    assert coverage["positives_present_in_candidates"] == 2
    assert coverage["positives_covered"] == 2
    assert coverage["positives_uncovered"] == 0
    assert math.isclose(coverage["coverage_ratio_present"], 1.0, rel_tol=1e-6)

    dist = summary["distribution"]
    assert math.isfinite(dist["mean_row_entropy"])
    assert math.isfinite(dist["mean_top1_mass"])
    assert dist["valid_row_count_for_entropy"] == 3

    monitor = summary["fg_not_bg_monitor"]
    assert monitor["high_objectness_track_count"] == 2
    assert monitor["high_objectness_bg_dominant_count"] == 1
    assert math.isclose(monitor["high_objectness_bg_dominant_fraction"], 0.5, rel_tol=1e-6)

    backend_echo = summary["backend_echo"]
    assert backend_echo["assignment_backend"] == "c2_sinkhorn_minimal_v1"
    assert backend_echo["config"]["sinkhorn_iterations"] == 20


def test_stagec_c4_observability_entropy_responds_to_diffuse_vs_sharp() -> None:
    batch = _batch_for_observability()
    sharp = StageCSemanticAssignmentOutputV1(
        soft_assignment=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        valid_track_mask=np.array([True, True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True, True, True], dtype=np.bool_),
        backend="c2_sinkhorn_minimal_v1",
    )
    diffuse = StageCSemanticAssignmentOutputV1(
        soft_assignment=np.full((3, 4), 0.25, dtype=np.float32),
        valid_track_mask=np.array([True, True, True], dtype=np.bool_),
        valid_column_mask=np.array([True, True, True, True], dtype=np.bool_),
        backend="c2_sinkhorn_minimal_v1",
    )

    sharp_summary = summarize_stagec_assignment_observability_c4_minimal_v1(batch=batch, assignment=sharp)
    diffuse_summary = summarize_stagec_assignment_observability_c4_minimal_v1(batch=batch, assignment=diffuse)

    assert math.isfinite(sharp_summary["distribution"]["mean_row_entropy"])
    assert math.isfinite(diffuse_summary["distribution"]["mean_row_entropy"])
    assert diffuse_summary["distribution"]["mean_row_entropy"] > sharp_summary["distribution"]["mean_row_entropy"]
    assert diffuse_summary["distribution"]["mean_top1_mass"] < sharp_summary["distribution"]["mean_top1_mass"]


def test_stagec_c4_backend_config_echo_present_in_c1_plumbing(tmp_path) -> None:
    out = build_stagec_semantic_plumbing_c1_clip_text_default(
        {
            "enabled": True,
            "assignment_backend": "c2_sinkhorn_minimal_v1",
            "sinkhorn_temperature": 0.09,
            "sinkhorn_iterations": 17,
            "sinkhorn_tolerance": 1e-7,
            "sinkhorn_bg_capacity_weight": 2.5,
            "sinkhorn_unk_fg_capacity_weight": 1.7,
        },
        track_features=np.array(
            [
                [0.2, 0.1, 0.0, 0.3],
                [0.3, 0.0, 0.1, 0.2],
            ],
            dtype=np.float32,
        ),
        positive_label_ids=(3, 7),
        topk_label_ids=(7, 9),
        merge_mode="yp_plus_topk",
        include_bg=True,
        include_unk_fg=True,
        cache_root=tmp_path / "clip_cache_c4_echo",
    )

    summary = out["diagnostics"]["c4_observability"]
    assert summary["backend_echo"]["assignment_backend"] == "c2_sinkhorn_minimal_v1"
    cfg = summary["backend_echo"]["config"]
    assert cfg["assignment_backend"] == "c2_sinkhorn_minimal_v1"
    assert cfg["sinkhorn_iterations"] == 17
    assert math.isclose(cfg["sinkhorn_temperature"], 0.09, rel_tol=1e-8)
    assert math.isclose(cfg["sinkhorn_tolerance"], 1e-7, rel_tol=1e-8)
    assert math.isclose(cfg["sinkhorn_bg_capacity_weight"], 2.5, rel_tol=1e-8)
    assert math.isclose(cfg["sinkhorn_unk_fg_capacity_weight"], 1.7, rel_tol=1e-8)
