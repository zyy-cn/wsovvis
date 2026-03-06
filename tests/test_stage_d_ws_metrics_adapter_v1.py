from __future__ import annotations

import pytest

from wsovvis.metrics import build_ws_metrics_summary_v1_from_stage_d_round_summary
from wsovvis.metrics.ws_metrics_stage_d_adapter_v1 import _build_predictions_by_missing_rate


def _round_summary_fixture() -> dict:
    return {
        "schema_name": "wsovvis.stage_d_round_summary_v1",
        "round_index": 1,
        "round_output_summary": {
            "selected_video_id": "9",
            "assignment_backend": "c9_em_minimal_v1",
            "positive_label_ids": [48, 145, 314],
            "candidate_label_ids": [48, 145, 314, 909001],
        },
    }


def test_stage_d_ws_metrics_adapter_returns_expected_schema_and_types() -> None:
    out = build_ws_metrics_summary_v1_from_stage_d_round_summary(_round_summary_fixture())
    assert out["schema_name"] == "wsovvis.ws_metrics_summary_v1"
    assert out["schema_version"] == "1.0"
    assert set(out["metrics"].keys()) == {"scr", "missing_rate_curve", "aurc"}
    assert isinstance(out["metrics"]["scr"], float)
    assert isinstance(out["metrics"]["aurc"], float)
    assert isinstance(out["metrics"]["missing_rate_curve"], list)
    assert out["metrics"]["scr"] == pytest.approx(1.0)
    assert out["metrics"]["aurc"] == pytest.approx(7.0 / 12.0)
    assert out["metrics"]["missing_rate_curve"] == [
        {"missing_rate": 0.0, "scr": 1.0},
        {"missing_rate": 0.5, "scr": pytest.approx(2.0 / 3.0)},
        {"missing_rate": 1.0, "scr": 0.0},
    ]
    extras = out["stage_d_extras"]
    assert extras["candidate_size"] == 4
    assert extras["positive_size"] == 3
    assert extras["jaccard"] == pytest.approx(0.75)
    assert extras["overreach"] == 1


def test_stage_d_ws_metrics_adapter_subset_curve_varies_and_is_non_increasing() -> None:
    preds = _build_predictions_by_missing_rate([10, 20, 30], (0.0, 0.5, 1.0))
    assert preds["0.0"] == [10, 20, 30]
    assert preds["0.5"] == [10, 20]
    assert preds["1.0"] == []

    out = build_ws_metrics_summary_v1_from_stage_d_round_summary(
        {
            "schema_name": "wsovvis.stage_d_round_summary_v1",
            "round_index": 0,
            "round_output_summary": {
                "selected_video_id": "v0",
                "assignment_backend": "c9_em_minimal_v1",
                "positive_label_ids": [20, 30],
                "candidate_label_ids": [10, 20, 30],
            },
        }
    )
    curve = out["metrics"]["missing_rate_curve"]
    scr_values = [float(row["scr"]) for row in curve]
    assert len(set(scr_values)) > 1
    assert scr_values[0] >= scr_values[1] >= scr_values[2]


def test_stage_d_ws_metrics_adapter_fail_fast_missing_fields() -> None:
    with pytest.raises(ValueError, match="stage_d_round_summary missing required fields"):
        build_ws_metrics_summary_v1_from_stage_d_round_summary(
            {
                "round_output_summary": {
                    "selected_video_id": "9",
                    "candidate_label_ids": [1, 2],
                }
            }
        )
