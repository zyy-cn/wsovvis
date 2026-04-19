from __future__ import annotations

import math

from videocutler.ext_stageb_ovvis.audit.train_gt_rank_audit import _aggregate_rows, _rank_and_top1_from_logits, _split_label


def test_split_label_observed_vs_dropped() -> None:
    assert _split_label(gt_class_id=3, observed_raw_ids=[1, 3, 5]) == "observed"
    assert _split_label(gt_class_id=4, observed_raw_ids=[1, 3, 5]) == "dropped"


def test_rank_and_top1_from_logits() -> None:
    rank, normalized_rank, is_top1 = _rank_and_top1_from_logits(
        logits=[0.1, 0.6, 0.2, 0.3],
        gt_index=3,
    )
    assert rank == 2
    assert math.isclose(normalized_rank, 1.0 / 3.0)
    assert is_top1 is False


def test_aggregate_rows_observed_and_dropped() -> None:
    rows = [
        {"supervision_split": "observed", "normalized_gt_rank": 0.0, "gt_is_top1": True},
        {"supervision_split": "observed", "normalized_gt_rank": 0.5, "gt_is_top1": False},
        {"supervision_split": "dropped", "normalized_gt_rank": 1.0, "gt_is_top1": False},
    ]
    summary = _aggregate_rows(rows, total_prediction_count=5)
    assert math.isclose(summary["match_rate"], 3.0 / 5.0)
    assert math.isclose(summary["observed"]["mean_normalized_gt_rank"], 0.25)
    assert math.isclose(summary["observed"]["gt_top1_hit_rate"], 0.5)
    assert math.isclose(summary["dropped"]["mean_normalized_gt_rank"], 1.0)
    assert math.isclose(summary["dropped"]["gt_top1_hit_rate"], 0.0)
