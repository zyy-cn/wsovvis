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
            "--round-policy",
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
    assert round0["round_policy_name"] == "none"
    assert round0["round_policy_applied"] is False
    assert round0["round_input_summary"]["source_kind"] == "tiny_pinned_synthetic_seed"
    assert round0["round_output_summary"]["orchestration_status"] == "pass_through_baseline"


def test_d0_round0_round1_with_minimal_refine_and_stagec_seed(tmp_path: Path) -> None:
    fixture_path = _repo_root() / "tests/fixtures/stagec_summary_d1_tiny.json"
    out_json = tmp_path / "d0_summary_round01.json"
    round_root = tmp_path / "rounds_round01"
    proc = _run(
        [
            "--stagec-summary-json",
            str(fixture_path),
            "--tiny-pinned",
            "--round-index",
            "0",
            "--max-rounds",
            "2",
            "--refine-mode",
            "minimal",
            "--round-policy",
            "minimal_curriculum_v1",
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
    assert round0["round_input_summary"]["selected_video_id"] == "video_9"
    assert round0["round_output_summary"]["candidate_label_ids"] == [5, 7, 11]
    assert round0["ws_metrics_summary_v1"]["schema_name"] == "wsovvis.ws_metrics_summary_v1"
    assert round1["round_index"] == 1
    assert round1["refine_mode_requested"] == "minimal"
    assert round1["refine_applied"] is True
    assert round1["round_policy_name"] == "minimal_curriculum_v1"
    assert round1["round_policy_applied"] is True
    assert "cap_additions_k=1" in round1["round_policy_notes"]
    assert round1["round_policy_stats"]["k"] == 1
    assert round1["round_input_summary"]["source_kind"] == "previous_round_output"
    refine_summary = round1["round_input_summary"]["refine_summary"]
    assert refine_summary["applied"] is True
    assert refine_summary["refine_mode"] == "minimal"
    assert 909001 in round1["round_output_summary"]["candidate_label_ids"]
    assert run_summary["schema_name"] == "wsovvis.stage_d_loop_summary_v1"
    assert run_summary["schema_version"] == "1.0"
    assert run_summary["round_policy_name"] == "minimal_curriculum_v1"
    assert run_summary["round_policy_applied"] is True
    assert run_summary["round_policy_stats"]["policy_applied_round_count"] == 1
    assert len(run_summary["round_paths"]) == 2


def test_d0_stagec_contract_validation_rejects_missing_video_id(tmp_path: Path) -> None:
    stagec_path = tmp_path / "bad_stagec_summary.json"
    stagec_path.write_text(
        json.dumps(
            {
                "selected_positive_label_ids": [10],
                "final": {"candidate_label_ids": [10]},
            }
        ),
        encoding="utf-8",
    )
    proc = _run(
        [
            "--stagec-summary-json",
            str(stagec_path),
            "--round-index",
            "0",
            "--max-rounds",
            "1",
            "--refine-mode",
            "none",
            "--round-policy",
            "none",
            "--round-summary-root",
            str(tmp_path / "rounds_bad"),
            "--out-json",
            str(tmp_path / "bad_out.json"),
        ]
    )
    assert proc.returncode != 0
    assert "selected_video_id" in proc.stderr


def test_d0_round_policy_cli_rejects_invalid_choice() -> None:
    proc = _run(
        [
            "--tiny-pinned",
            "--round-index",
            "0",
            "--max-rounds",
            "1",
            "--refine-mode",
            "none",
            "--round-policy",
            "invalid_policy",
            "--round-summary-root",
            "outputs/stage_d0_round_summaries_invalid",
        ]
    )
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr
