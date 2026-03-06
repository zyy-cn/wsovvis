from __future__ import annotations

import pytest

from wsovvis.metrics import build_ws_metrics_summary_v1_from_stage_d_round_summary


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
    assert out["metrics"]["aurc"] == pytest.approx(1.0)


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
