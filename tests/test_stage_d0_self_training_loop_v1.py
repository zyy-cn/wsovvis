from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "tools/run_stage_d0_self_training_loop.py", *args],
        cwd=str(_repo_root()),
        text=True,
        capture_output=True,
        check=False,
    )


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_d0_tiny_round0_pass_through_no_refine(tmp_path: Path) -> None:
    out_json = tmp_path / "d0_summary_round0.json"
    round_root = tmp_path / "rounds_round0"
    proc = _run(
        [
            "--tiny-pinned",
            "--round-index",
            "0",
            "--max-rounds",
            "1",
            "--refine-mode",
            "none",
            "--round-summary-root",
            str(round_root),
            "--out-json",
            str(out_json),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    run_summary = _load_json(out_json)
    assert run_summary["status"] == "PASS"
    assert run_summary["rounds_executed"] == 1
    round0 = run_summary["round_summaries"][0]
    assert round0["round_index"] == 0
    assert round0["refine_mode_requested"] == "none"
    assert round0["refine_applied"] is False
    assert round0["round_input_summary"]["source_kind"] == "tiny_pinned_synthetic_seed"
    assert round0["round_output_summary"]["orchestration_status"] == "pass_through_baseline"


def test_d0_round0_round1_with_minimal_refine_and_stagec_seed(tmp_path: Path) -> None:
    stagec_summary = {
        "selected_video_id": "video_9",
        "selected_positive_label_ids": [5, 7],
        "assignment_backend_requested": "c9_em_minimal_v1",
        "final": {"candidate_label_ids": [5, 7, 11]},
        "ws_metrics_summary_v1": {
            "schema_name": "wsovvis.ws_metrics_summary_v1",
            "schema_version": "1.0",
            "scr_macro": 1.0,
        },
    }
    stagec_path = tmp_path / "stagec_summary.json"
    stagec_path.write_text(json.dumps(stagec_summary), encoding="utf-8")
    out_json = tmp_path / "d0_summary_round01.json"
    round_root = tmp_path / "rounds_round01"
    proc = _run(
        [
            "--stagec-summary-in-json",
            str(stagec_path),
            "--tiny-pinned",
            "--round-index",
            "0",
            "--max-rounds",
            "2",
            "--refine-mode",
            "minimal",
            "--round-summary-root",
            str(round_root),
            "--out-json",
            str(out_json),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    run_summary = _load_json(out_json)
    assert run_summary["rounds_executed"] == 2
    round0 = run_summary["round_summaries"][0]
    round1 = run_summary["round_summaries"][1]
    assert round0["round_index"] == 0
    assert round0["refine_applied"] is False
    assert round0["round_input_summary"]["source_kind"] == "stagec_summary_json"
    assert round0["ws_metrics_summary_v1"]["schema_name"] == "wsovvis.ws_metrics_summary_v1"
    assert round1["round_index"] == 1
    assert round1["refine_mode_requested"] == "minimal"
    assert round1["refine_applied"] is True
    refine_summary = round1["round_input_summary"]["refine_summary"]
    assert refine_summary["applied"] is True
    assert refine_summary["refine_mode"] == "minimal"
    assert 909001 in round1["round_output_summary"]["candidate_label_ids"]
