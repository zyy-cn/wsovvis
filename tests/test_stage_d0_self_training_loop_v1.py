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
    assert round0["round_refine_additions_count"] == 0
    assert round0["round_refine_added_label_ids"] == []
    assert round0["round_policy_kept_count"] == 0
    assert round0["round_policy_dropped_count"] == 0
    assert round0["candidate_label_ids_count_before"] == 3
    assert round0["candidate_label_ids_count_after"] == 3
    assert round0["candidate_label_ids_count_delta"] == 0
    assert round0["round_input_summary"]["source_kind"] == "tiny_pinned_synthetic_seed"
    assert round0["round_output_summary"]["orchestration_status"] == "pass_through_baseline"
    assert round0["upstream_risk_guardrail_v1"] is None
    assert round0["train_summary"] is None
    assert run_summary["train_hook"] == "none"
    assert run_summary["train_steps"] == 2
    assert run_summary["train_seed"] == 20260305
    assert run_summary["train_hook_applied_round_count"] == 0


def test_d0_round0_round1_with_minimal_refine_and_stagec_seed(tmp_path: Path) -> None:
    fixture_path = _repo_root() / "tests/fixtures/stagec_summary_d1_tiny.json"
    out_json = tmp_path / "d0_summary_round01.json"
    round_root = tmp_path / "rounds_round01"
    proc = _run(
        [
            "--stagec-summary-json",
            str(fixture_path),
            "--seed",
            "20260305",
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
    assert round1["round_refine_additions_count"] == 1
    assert round1["round_refine_added_label_ids"] == [909001]
    assert round1["round_policy_kept_count"] == 1
    assert round1["round_policy_dropped_count"] == 0
    assert round1["candidate_label_ids_count_before"] == 3
    assert round1["candidate_label_ids_count_after"] == 4
    assert round1["candidate_label_ids_count_delta"] == 1
    assert round1["round_input_summary"]["source_kind"] == "previous_round_output"
    refine_summary = round1["round_input_summary"]["refine_summary"]
    assert refine_summary["applied"] is True
    assert refine_summary["refine_mode"] == "minimal"
    assert 909001 in round1["round_output_summary"]["candidate_label_ids"]
    assert run_summary["schema_name"] == "wsovvis.stage_d_loop_summary_v1"
    assert run_summary["schema_version"] == "1.0"
    assert run_summary["seed"] == 20260305
    assert run_summary["round_policy_name"] == "minimal_curriculum_v1"
    assert run_summary["round_policy_applied"] is True
    assert run_summary["round_policy_stats"]["policy_applied_round_count"] == 1
    assert run_summary["round_refine_additions_count_total"] == 1
    assert run_summary["round_policy_kept_count_total"] == 1
    assert run_summary["round_policy_dropped_count_total"] == 0
    assert run_summary["candidate_label_ids_count_before_start"] == 3
    assert run_summary["candidate_label_ids_count_after_end"] == 4
    assert run_summary["candidate_label_ids_count_delta_total"] == 1
    assert run_summary["upstream_risk_guardrail_v1_present"] is False
    assert run_summary["upstream_risk_guardrail_v1"] is None
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


def test_d0_stagec_contract_validation_rejects_missing_candidate_labels(tmp_path: Path) -> None:
    stagec_path = tmp_path / "bad_stagec_summary_missing_candidates.json"
    stagec_path.write_text(
        json.dumps(
            {
                "selected_video_id": "video_1",
                "selected_positive_label_ids": [],
                "final": {"candidate_label_ids": []},
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
            str(tmp_path / "rounds_bad_candidates"),
            "--out-json",
            str(tmp_path / "bad_candidates_out.json"),
        ]
    )
    assert proc.returncode != 0
    assert "candidate labels" in proc.stderr


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


def test_d0_minimal_multiadd_round_policy_none_keeps_all_added_ids(tmp_path: Path) -> None:
    fixture_path = _repo_root() / "tests/fixtures/stagec_summary_d1_tiny.json"
    out_json = tmp_path / "d0_multiadd_none_summary.json"
    round_root = tmp_path / "rounds_multiadd_none"
    proc = _run(
        [
            "--stagec-summary-json",
            str(fixture_path),
            "--seed",
            "20260305",
            "--round-index",
            "0",
            "--max-rounds",
            "2",
            "--refine-mode",
            "minimal_multiadd_v1",
            "--refine-multiadd-count",
            "3",
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
    round1 = run_summary["round_summaries"][1]
    assert round1["refine_mode_requested"] == "minimal_multiadd_v1"
    assert round1["round_refine_mode"] == "minimal_multiadd_v1"
    assert round1["round_refine_multiadd_count"] == 3
    assert round1["round_refine_additions_count"] == 3
    assert round1["round_refine_added_label_ids"] == [909001, 909002, 909003]
    assert round1["round_policy_name"] == "none"
    assert round1["round_policy_applied"] is False
    assert round1["round_policy_kept_count"] == 0
    assert round1["round_policy_dropped_count"] == 0
    assert round1["candidate_label_ids_count_before"] == 3
    assert round1["candidate_label_ids_count_after"] == 6
    assert round1["candidate_label_ids_count_delta"] == 3
    assert round1["round_output_summary"]["candidate_label_ids"] == [5, 7, 11, 909001, 909002, 909003]


def test_d0_minimal_multiadd_policy_cap1_drops_n_minus_1(tmp_path: Path) -> None:
    fixture_path = _repo_root() / "tests/fixtures/stagec_summary_d1_tiny.json"
    out_json = tmp_path / "d0_multiadd_cap1_summary.json"
    round_root = tmp_path / "rounds_multiadd_cap1"
    proc = _run(
        [
            "--stagec-summary-json",
            str(fixture_path),
            "--seed",
            "20260305",
            "--round-index",
            "0",
            "--max-rounds",
            "2",
            "--refine-mode",
            "minimal_multiadd_v1",
            "--refine-multiadd-count",
            "3",
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
    round1 = run_summary["round_summaries"][1]
    assert round1["round_refine_additions_count"] == 3
    assert round1["round_refine_added_label_ids"] == [909001, 909002, 909003]
    assert round1["round_policy_applied"] is True
    assert round1["round_policy_stats"]["k"] == 1
    assert round1["round_policy_stats"]["input_additions_count"] == 3
    assert round1["round_policy_stats"]["kept_addition_ids"] == [909001]
    assert round1["round_policy_stats"]["dropped_addition_ids"] == [909002, 909003]
    assert round1["round_policy_kept_count"] == 1
    assert round1["round_policy_dropped_count"] == 2
    assert round1["candidate_label_ids_count_before"] == 3
    assert round1["candidate_label_ids_count_after"] == 4
    assert round1["candidate_label_ids_count_delta"] == 1
    assert round1["round_output_summary"]["candidate_label_ids"] == [5, 7, 11, 909001]
    assert run_summary["round_refine_additions_count_total"] == 3
    assert run_summary["round_policy_kept_count_total"] == 1
    assert run_summary["round_policy_dropped_count_total"] == 2
    assert run_summary["candidate_label_ids_count_delta_total"] == 1


def test_d0_minimal_multiadd_iterative_adds_new_ids_in_round1_and_round2(tmp_path: Path) -> None:
    fixture_path = _repo_root() / "tests/fixtures/stagec_summary_d1_tiny.json"
    out_json = tmp_path / "d0_multiadd_iter_summary.json"
    round_root = tmp_path / "rounds_multiadd_iter"
    proc = _run(
        [
            "--stagec-summary-json",
            str(fixture_path),
            "--seed",
            "20260305",
            "--round-index",
            "0",
            "--max-rounds",
            "3",
            "--refine-mode",
            "minimal_multiadd_iter_v1",
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
    round1 = run_summary["round_summaries"][1]
    round2 = run_summary["round_summaries"][2]
    assert round1["round_refine_mode"] == "minimal_multiadd_iter_v1"
    assert round2["round_refine_mode"] == "minimal_multiadd_iter_v1"
    assert round1["round_refine_additions_count"] == 1
    assert round2["round_refine_additions_count"] == 1
    assert round1["round_refine_added_label_ids"] == [909001]
    assert round2["round_refine_added_label_ids"] == [909002]
    assert round1["round_policy_kept_count"] == 1
    assert round2["round_policy_kept_count"] == 1
    assert round1["round_policy_dropped_count"] == 0
    assert round2["round_policy_dropped_count"] == 0
    assert round1["candidate_label_ids_count_after"] == 4
    assert round2["candidate_label_ids_count_after"] == 5
    assert run_summary["round_refine_additions_count_total"] == 2
    assert run_summary["round_policy_kept_count_total"] == 2
    assert run_summary["round_policy_dropped_count_total"] == 0
    assert run_summary["candidate_label_ids_count_delta_total"] == 2


def test_d0_stagec_micro_train_hook_runs_per_round(tmp_path: Path) -> None:
    fixture_path = _repo_root() / "tests/fixtures/stagec_summary_d1_tiny.json"
    out_json = tmp_path / "d0_train_hook_summary.json"
    round_root = tmp_path / "rounds_train_hook"
    proc = _run(
        [
            "--stagec-summary-json",
            str(fixture_path),
            "--round-index",
            "0",
            "--max-rounds",
            "2",
            "--refine-mode",
            "minimal",
            "--round-policy",
            "none",
            "--train-hook",
            "stagec_micro_train_v1",
            "--train-steps",
            "2",
            "--train-seed",
            "20260305",
            "--round-summary-root",
            str(round_root),
            "--out-json",
            str(out_json),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    run_summary = _load_json(out_json)
    assert run_summary["train_hook"] == "stagec_micro_train_v1"
    assert run_summary["train_steps"] == 2
    assert run_summary["train_seed"] == 20260305
    assert run_summary["train_data_mode"] == "synthetic_v1"
    assert run_summary["train_real_run_root"] is None
    assert run_summary["train_hook_applied_round_count"] == 2
    round0 = run_summary["round_summaries"][0]
    round1 = run_summary["round_summaries"][1]
    train0 = round0["train_summary"]
    train1 = round1["train_summary"]
    assert train0["status"] == "PASS"
    assert train1["status"] == "PASS"
    assert train0["hook_name"] == "stagec_micro_train_v1"
    assert train1["hook_name"] == "stagec_micro_train_v1"
    assert train0["train_steps"] == 2
    assert train1["train_steps"] == 2
    assert train0["train_seed_effective"] == 20260305
    assert train1["train_seed_effective"] == 20260306
    assert train0["data_mode"] == "synthetic_v1"
    assert train1["data_mode"] == "synthetic_v1"
    assert train0["train_data_mode_requested"] == "synthetic_v1"
    assert train1["train_data_mode_requested"] == "synthetic_v1"
    assert train0["train_real_run_root_requested"] is None
    assert train1["train_real_run_root_requested"] is None
    assert train0["train_candidate_label_ids_count"] == 3
    assert train1["train_candidate_label_ids_count"] == 4
    assert train0["train_candidate_label_ids_effective_count"] >= train0["train_candidate_label_ids_count"]
    assert train1["train_candidate_label_ids_effective_count"] >= train1["train_candidate_label_ids_count"]
    assert Path(train0["candidate_label_ids_json_path"]).exists()
    assert Path(train1["candidate_label_ids_json_path"]).exists()


def test_d0_emit_ws_metrics_writes_round_sidecars(tmp_path: Path) -> None:
    fixture_path = _repo_root() / "tests/fixtures/stagec_summary_d1_tiny.json"
    out_json = tmp_path / "d0_emit_ws_metrics_summary.json"
    round_root = tmp_path / "rounds_emit_ws_metrics"
    proc = _run(
        [
            "--stagec-summary-json",
            str(fixture_path),
            "--round-index",
            "0",
            "--max-rounds",
            "2",
            "--refine-mode",
            "minimal",
            "--round-policy",
            "minimal_curriculum_v1",
            "--emit-ws-metrics",
            "--round-summary-root",
            str(round_root),
            "--out-json",
            str(out_json),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    run_summary = _load_json(out_json)
    assert run_summary["emit_ws_metrics"] is True
    assert len(run_summary["round_summaries"]) == 2

    for idx, summary in enumerate(run_summary["round_summaries"]):
        ws = summary["ws_metrics_summary_v1"]
        assert ws["schema_name"] == "wsovvis.ws_metrics_summary_v1"
        assert ws["schema_version"] == "1.0"
        assert isinstance(ws["metrics"]["scr"], float)
        assert isinstance(ws["metrics"]["aurc"], float)
        ws_path = round_root / f"round{idx}_ws_metrics_summary.json"
        assert ws_path.exists()
        sidecar = _load_json(ws_path)
        assert sidecar["schema_name"] == "wsovvis.ws_metrics_summary_v1"


def test_d0_minimal_regression_guard_blocks_round2_high_risk_additions(tmp_path: Path) -> None:
    stagec_path = tmp_path / "stagec_high_risk_summary.json"
    stagec_path.write_text(
        json.dumps(
            {
                "assignment_backend_requested": "c9_em_minimal_v1",
                "selected_video_id": "video_9",
                "selected_positive_label_ids": [5, 7],
                "final": {"candidate_label_ids": [5, 7, 11]},
                "unknown_handling_diagnostics_v1": {
                    "risk_guardrail_v1": {
                        "score": 3,
                        "level": "high",
                        "triggered": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "d0_guarded_summary.json"
    round_root = tmp_path / "rounds_guarded"
    proc = _run(
        [
            "--stagec-summary-json",
            str(stagec_path),
            "--seed",
            "20260305",
            "--round-index",
            "0",
            "--max-rounds",
            "3",
            "--refine-mode",
            "minimal_multiadd_iter_v1",
            "--round-policy",
            "minimal_curriculum_v1",
            "--round-guard",
            "minimal_regression_guard_v1",
            "--round-summary-root",
            str(round_root),
            "--out-json",
            str(out_json),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    run_summary = _load_json(out_json)
    round1 = run_summary["round_summaries"][1]
    round2 = run_summary["round_summaries"][2]

    assert round1["proposed_addition_ids"] == [909001]
    assert round1["accepted_addition_ids"] == [909001]
    assert round1["rejected_addition_ids"] == []
    assert round1["guard_decision"] == "allow_additions_under_guard_v1"
    assert round1["candidate_label_ids_count_after"] == 4

    assert round2["proposed_addition_ids"] == [909002]
    assert round2["accepted_addition_ids"] == []
    assert round2["rejected_addition_ids"] == [909002]
    assert round2["guard_decision"] == "reject_all_additions_due_to_high_risk_v1"
    assert "upstream_risk_guardrail_v1 indicates high risk" in round2["guard_decision_reason"]
    assert round2["candidate_label_ids_count_after"] == 4
    assert round2["candidate_label_ids_count_delta"] == 0
    assert round2["round_guard_stats"]["upstream_risk_guardrail_v1_present"] is True
    assert round2["round_guard_stats"]["fallback_high_risk_used"] is False

    assert run_summary["round_guard_name"] == "minimal_regression_guard_v1"
    assert run_summary["guard_variant"] == "minimal_regression_guard_v1"
    assert run_summary["round_guard_stats"]["rounds_with_rejected_additions"] == [2]
    assert run_summary["round_guard_stats"]["rejected_additions_total"] == 1
    assert run_summary["round_guard_stats"]["accepted_additions_total"] == 1
    assert run_summary["candidate_label_ids_count_delta_total"] == 1


def test_d0_minimal_regression_guard_uses_fallback_when_upstream_risk_missing(tmp_path: Path) -> None:
    stagec_path = tmp_path / "stagec_no_risk_summary.json"
    stagec_path.write_text(
        json.dumps(
            {
                "assignment_backend_requested": "c9_em_minimal_v1",
                "selected_video_id": "video_9",
                "selected_positive_label_ids": [5, 7],
                "final": {"candidate_label_ids": [5, 7, 11]},
                "unknown_handling_diagnostics_v1": {},
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "d0_guarded_fallback_summary.json"
    round_root = tmp_path / "rounds_guarded_fallback"
    proc = _run(
        [
            "--stagec-summary-json",
            str(stagec_path),
            "--seed",
            "20260306",
            "--round-index",
            "0",
            "--max-rounds",
            "3",
            "--refine-mode",
            "minimal_multiadd_iter_v1",
            "--round-policy",
            "minimal_curriculum_v1",
            "--round-guard",
            "minimal_regression_guard_v1",
            "--round-summary-root",
            str(round_root),
            "--out-json",
            str(out_json),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    run_summary = _load_json(out_json)
    round2 = run_summary["round_summaries"][2]

    assert round2["proposed_addition_ids"] == [909002]
    assert round2["accepted_addition_ids"] == []
    assert round2["rejected_addition_ids"] == [909002]
    assert round2["guard_decision"] == "reject_all_additions_due_to_high_risk_v1"
    assert "fallback_guard_v1 triggered" in round2["guard_decision_reason"]
    assert round2["round_guard_stats"]["upstream_risk_guardrail_v1_present"] is False
    assert round2["round_guard_stats"]["fallback_high_risk_used"] is True
    assert round2["candidate_label_ids_count_after"] == 4


def test_d0_accept_top1_guard_keeps_one_under_high_risk(tmp_path: Path) -> None:
    stagec_path = tmp_path / "stagec_high_risk_summary_keep_top1.json"
    stagec_path.write_text(
        json.dumps(
            {
                "assignment_backend_requested": "c9_em_minimal_v1",
                "selected_video_id": "video_9",
                "selected_positive_label_ids": [5, 7],
                "final": {"candidate_label_ids": [5, 7, 11]},
                "unknown_handling_diagnostics_v1": {
                    "risk_guardrail_v1": {
                        "risk_score": 3,
                        "risk_level": "high",
                        "triggered": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "d0_guarded_keep_top1_summary.json"
    round_root = tmp_path / "rounds_guarded_keep_top1"
    proc = _run(
        [
            "--stagec-summary-json",
            str(stagec_path),
            "--seed",
            "20260306",
            "--round-index",
            "0",
            "--max-rounds",
            "3",
            "--refine-mode",
            "minimal_multiadd_v1",
            "--refine-multiadd-count",
            "3",
            "--round-policy",
            "minimal_curriculum_v1",
            "--round-guard",
            "accept_top1_only_under_guard_v1",
            "--round-summary-root",
            str(round_root),
            "--out-json",
            str(out_json),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    run_summary = _load_json(out_json)
    round2 = run_summary["round_summaries"][2]

    assert round2["guard_variant"] == "accept_top1_only_under_guard_v1"
    assert round2["proposed_addition_ids"] == [909002]
    assert round2["accepted_addition_ids"] == [909002]
    assert round2["rejected_addition_ids"] == []
    assert round2["guard_decision"] == "accept_top1_only_due_to_high_risk_v1"
    assert round2["candidate_label_ids_count_after"] == 5
    assert round2["round_output_summary"]["candidate_label_ids"] == [5, 7, 11, 909001, 909002]
    assert run_summary["guard_variant"] == "accept_top1_only_under_guard_v1"
    assert run_summary["round_guard_stats"]["accepted_additions_total"] == 2
    assert run_summary["round_guard_stats"]["rejected_additions_total"] == 0


def test_d0_accept_top1_guard_uses_fallback_trigger(tmp_path: Path) -> None:
    stagec_path = tmp_path / "stagec_no_risk_summary_keep_top1.json"
    stagec_path.write_text(
        json.dumps(
            {
                "assignment_backend_requested": "c9_em_minimal_v1",
                "selected_video_id": "video_9",
                "selected_positive_label_ids": [5, 7],
                "final": {"candidate_label_ids": [5, 7, 11]},
                "unknown_handling_diagnostics_v1": {},
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "d0_guarded_keep_top1_fallback_summary.json"
    round_root = tmp_path / "rounds_guarded_keep_top1_fallback"
    proc = _run(
        [
            "--stagec-summary-json",
            str(stagec_path),
            "--seed",
            "20260307",
            "--round-index",
            "0",
            "--max-rounds",
            "3",
            "--refine-mode",
            "minimal_multiadd_v1",
            "--refine-multiadd-count",
            "3",
            "--round-policy",
            "minimal_curriculum_v1",
            "--round-guard",
            "accept_top1_only_under_guard_v1",
            "--round-summary-root",
            str(round_root),
            "--out-json",
            str(out_json),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    run_summary = _load_json(out_json)
    round2 = run_summary["round_summaries"][2]

    assert round2["proposed_addition_ids"] == [909002]
    assert round2["accepted_addition_ids"] == [909002]
    assert round2["rejected_addition_ids"] == []
    assert round2["guard_decision"] == "accept_top1_only_due_to_high_risk_v1"
    assert "fallback_guard_v1 triggered" in round2["guard_decision_reason"]
    assert round2["round_guard_stats"]["fallback_high_risk_used"] is True
