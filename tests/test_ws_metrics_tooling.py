from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from wsovvis.metrics import aurc_from_curve, missing_rate_curve_from_predictions, set_coverage_recall


def test_case_a_perfect_prediction() -> None:
    gt = {1, 2, 3}
    preds = {1, 2, 3}
    curve = missing_rate_curve_from_predictions(
        gt,
        {
            0.0: {1, 2, 3},
            0.5: {1, 2, 3},
            1.0: {1, 2, 3},
        },
    )
    assert set_coverage_recall(gt, preds) == 1.0
    assert curve == [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]
    assert aurc_from_curve(curve) == 1.0


def test_case_b_half_coverage() -> None:
    gt = {1, 2, 3, 4}
    preds = {1, 2}
    curve = missing_rate_curve_from_predictions(
        gt,
        {
            0.0: {1, 2},
            0.5: {1, 2},
            1.0: {1, 2},
        },
    )
    assert set_coverage_recall(gt, preds) == 0.5
    assert curve == [(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)]
    assert aurc_from_curve(curve) == 0.5


def test_case_c_empty_prediction() -> None:
    gt = {5, 6}
    preds = set()
    curve = missing_rate_curve_from_predictions(
        gt,
        {
            0.0: set(),
            0.5: set(),
            1.0: set(),
        },
    )
    assert set_coverage_recall(gt, preds) == 0.0
    assert curve == [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
    assert aurc_from_curve(curve) == 0.0


def test_aurc_trapezoid_nontrivial_curve() -> None:
    curve = [
        (0.0, 1.0),
        (0.25, 0.75),
        (0.75, 0.25),
        (1.0, 0.0),
    ]
    # Hand-computed area:
    # [0,0.25]: (1.0+0.75)/2 * 0.25 = 0.21875
    # [0.25,0.75]: (0.75+0.25)/2 * 0.50 = 0.25
    # [0.75,1.0]: (0.25+0.0)/2 * 0.25 = 0.03125
    # total = 0.5, normalized by (1.0 - 0.0) => 0.5
    assert aurc_from_curve(curve) == 0.5


def test_optional_cli_demo(tmp_path: Path) -> None:
    payload = {
        "gt_entities": [1, 2, 3, 4],
        "predicted_entities": [1, 2],
        "predictions_by_missing_rate": {
            "0.0": [1, 2, 3, 4],
            "0.5": [1, 2],
            "1.0": [1],
        },
    }
    input_json = tmp_path / "demo.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "tools/ws_metrics_demo.py", "--input-json", str(input_json)],
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(proc.stdout)
    assert output["scr"] == 0.5
    assert output["curve"] == [[0.0, 1.0], [0.5, 0.5], [1.0, 0.25]]
    assert output["aurc"] == 0.5625
