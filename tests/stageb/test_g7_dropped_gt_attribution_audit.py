from __future__ import annotations

from videocutler.ext_stageb_ovvis.audit.dropped_gt_attribution_audit import (
    build_dropped_gt_attribution_rows,
    summarize_dropped_gt_attribution_rows,
)


def test_prealign_stage_supports_full_vocab_rank_for_dropped_gt() -> None:
    rows, summary = build_dropped_gt_attribution_rows(
        stage_id="prealign",
        materialized_samples=[
            {
                "trajectory_id": "traj-1",
                "clip_id": 3,
                "video_id": 9,
                "observed_raw_ids": [1],
            }
        ],
        stage_records=[
            {
                "trajectory_id": "traj-1",
                "proxy_mass": {"unknown": 0.1, "1": 0.6, "7": 0.2, "9": 0.1},
            }
        ],
        gt_sidecar_lookup={"traj-1": {"audit_usable": True, "matched_gt_class_id": 7}},
        full_vocab_ids=[1, 7, 9],
        base_vocab_ids=[1, 7, 9],
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["gt_missing_from_observed"] is True
    assert row["dropped_gt_rank"] == 2
    assert row["dropped_gt_top1"] is False
    assert row["dropped_gt_top5"] is True
    assert row["stage_top1_id"] == 1
    assert row["wrong_top1_is_base"] is True
    assert summary["dropped_gt_mean_rank"] == 2.0
    assert summary["dropped_gt_top5_hit_rate"] == 1.0


def test_softem_stage_marks_out_of_domain_gt_with_sentinel_rank() -> None:
    rows, summary = build_dropped_gt_attribution_rows(
        stage_id="softem_base",
        materialized_samples=[
            {
                "trajectory_id": "traj-2",
                "clip_id": 4,
                "video_id": 12,
                "observed_raw_ids": [1],
            }
        ],
        stage_records=[
            {
                "trajectory_id": "traj-2",
                "candidate_ids_known": [1],
                "candidate_ids_extra": [5],
                "r_final": {"unknown": 0.1, "1": 0.8, "5": 0.1},
            }
        ],
        gt_sidecar_lookup={"traj-2": {"audit_usable": True, "matched_gt_class_id": 7}},
        full_vocab_ids=[1, 5, 7, 9],
        base_vocab_ids=[1, 5, 7, 9],
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["gt_missing_from_observed"] is True
    assert row["dropped_gt_in_stage_domain"] is False
    assert row["dropped_gt_rank"] == 5
    assert row["dropped_gt_mrr"] == 0.0
    assert summary["dropped_gt_in_stage_domain_rate"] == 0.0


def test_summary_aggregates_multiple_rows() -> None:
    summary = summarize_dropped_gt_attribution_rows(
        [
            {
                "gt_available_for_audit": True,
                "gt_missing_from_observed": True,
                "dropped_gt_in_stage_domain": True,
                "dropped_gt_rank": 2,
                "dropped_gt_mrr": 0.5,
                "dropped_gt_top1": False,
                "dropped_gt_top5": True,
                "dropped_gt_top10": True,
                "dropped_gt_score": 0.2,
                "dropped_gt_margin_to_best_wrong": -0.1,
                "wrong_top1_is_base": True,
                "stage_top1_id": 1,
                "gt_class_id": 7,
            },
            {
                "gt_available_for_audit": True,
                "gt_missing_from_observed": True,
                "dropped_gt_in_stage_domain": True,
                "dropped_gt_rank": 1,
                "dropped_gt_mrr": 1.0,
                "dropped_gt_top1": True,
                "dropped_gt_top5": True,
                "dropped_gt_top10": True,
                "dropped_gt_score": 0.8,
                "dropped_gt_margin_to_best_wrong": 0.3,
                "wrong_top1_is_base": False,
                "stage_top1_id": 7,
                "gt_class_id": 7,
            },
        ],
        stage_id="softem_aug",
    )
    assert summary["dropped_gt_count"] == 2
    assert summary["dropped_gt_mean_rank"] == 1.5
    assert summary["dropped_gt_top1_hit_rate"] == 0.5
    assert summary["wrong_top1_is_base_rate"] == 0.5
