from __future__ import annotations

import pytest

from wsovvis.metrics import build_ws_metrics_summary_v1


def _tiny_stage_summary_fixture() -> dict:
    return {
        "video_id": "9",
        "assignment_backend": "c9_em_minimal_v1",
        "steps": 6,
        "seed": 20260305,
        "ws_eval_bundle": {
            "gt_entities": [48, 145, 314, 465, 475, 704],
            "predicted_entities": [48],
            "predictions_by_missing_rate": {
                "0.0": [48],
                "0.5": [48],
                "1.0": [48],
            },
        },
    }


def test_ws_metrics_summary_v1_schema_and_fields_complete() -> None:
    out = build_ws_metrics_summary_v1(_tiny_stage_summary_fixture())

    assert out["schema_name"] == "wsovvis.ws_metrics_summary_v1"
    assert out["schema_version"] == "1.0"
    assert set(out.keys()) == {"schema_name", "schema_version", "metrics", "source_metadata"}

    metrics = out["metrics"]
    assert set(metrics.keys()) == {"scr", "missing_rate_curve", "aurc"}
    assert isinstance(metrics["missing_rate_curve"], list)
    assert metrics["missing_rate_curve"] == [
        {"missing_rate": 0.0, "scr": pytest.approx(1.0 / 6.0)},
        {"missing_rate": 0.5, "scr": pytest.approx(1.0 / 6.0)},
        {"missing_rate": 1.0, "scr": pytest.approx(1.0 / 6.0)},
    ]


def test_ws_metrics_summary_v1_fail_fast_on_missing_required_field() -> None:
    with pytest.raises(ValueError, match="gt_entities: required"):
        build_ws_metrics_summary_v1(
            {
                "predicted_entities": [1, 2],
                "predictions_by_missing_rate": {"0.0": [1, 2]},
            }
        )


def test_ws_metrics_summary_v1_c10a_fixture_consistency() -> None:
    out = build_ws_metrics_summary_v1(_tiny_stage_summary_fixture())
    metrics = out["metrics"]

    assert metrics["scr"] == pytest.approx(1.0 / 6.0)
    assert metrics["aurc"] == pytest.approx(1.0 / 6.0)
    assert out["source_metadata"] == {
        "video_id": "9",
        "assignment_backend": "c9_em_minimal_v1",
        "steps": 6,
        "seed": 20260305,
    }
