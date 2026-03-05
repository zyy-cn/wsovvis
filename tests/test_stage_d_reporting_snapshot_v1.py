from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "tools/stage_d_reporting_snapshot.py", *args],
        cwd=str(_repo_root()),
        text=True,
        capture_output=True,
        check=False,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_stage_d_snapshot_happy_path_detects_round_paths(tmp_path: Path) -> None:
    loop_path = tmp_path / "d4_tiny_loop_summary.json"
    rounds_root = tmp_path / "rounds"
    round0_path = rounds_root / "round0_summary.json"
    round1_path = rounds_root / "round1_summary.json"
    out_json = tmp_path / "snapshot.json"

    _write_json(
        loop_path,
        {
            "schema_name": "wsovvis.stage_d_loop_summary_v1",
            "rounds_executed": 2,
            "round_refine_additions_count_total": 1,
            "round_policy_kept_count_total": 1,
            "round_policy_dropped_count_total": 0,
            "candidate_label_ids_count_before_start": 3,
            "candidate_label_ids_count_after_end": 4,
            "candidate_label_ids_count_delta_total": 1,
            "upstream_risk_guardrail_v1": {"guardrail_mode": "strict"},
            "round_paths": [
                {"round_summary_path": "rounds/round0_summary.json"},
                {"round_summary_path": "rounds/round1_summary.json"},
            ],
        },
    )
    _write_json(
        round0_path,
        {
            "candidate_label_ids_count_before": 3,
            "candidate_label_ids_count_after": 3,
            "candidate_label_ids_count_delta": 0,
            "round_refine_additions_count": 0,
            "round_refine_added_label_ids": [],
            "round_policy_kept_count": 0,
            "round_policy_dropped_count": 0,
            "upstream_risk_guardrail_v1": {"guardrail_mode": "strict"},
        },
    )
    _write_json(
        round1_path,
        {
            "candidate_label_ids_count_before": 3,
            "candidate_label_ids_count_after": 4,
            "candidate_label_ids_count_delta": 1,
            "round_refine_additions_count": 1,
            "round_refine_added_label_ids": [909001],
            "round_policy_kept_count": 1,
            "round_policy_dropped_count": 0,
            "upstream_risk_guardrail_v1": {"guardrail_mode": "strict"},
        },
    )

    proc = _run(["--loop", str(loop_path), "--out-json", str(out_json)])
    assert proc.returncode == 0, proc.stderr
    assert "STAGE_D_SNAPSHOT" in proc.stdout
    assert "round1" in proc.stdout
    assert "additions_ids=[909001]" in proc.stdout

    snapshot = _load_json(out_json)
    assert snapshot["schema_name"] == "wsovvis.stage_d_reporting_snapshot_v1"
    assert snapshot["round0"]["candidate_label_ids_count_before"] == 3
    assert snapshot["round1"]["candidate_label_ids_count_after"] == 4
    assert snapshot["round1"]["round_refine_added_label_ids"] == [909001]
    assert snapshot["loop"]["rounds_executed"] == 2
    assert snapshot["loop"]["candidate_label_ids_count_delta_total"] == 1
    assert snapshot["loop"]["upstream_risk_guardrail_v1_present"] is True
    assert snapshot["artifact_paths"]["round0_summary_json"] == str(round0_path.resolve())
    assert snapshot["artifact_paths"]["round1_summary_json"] == str(round1_path.resolve())


def test_stage_d_snapshot_missing_required_field_fails_fast(tmp_path: Path) -> None:
    loop_path = tmp_path / "loop_summary.json"
    round0_path = tmp_path / "round0_summary.json"
    _write_json(loop_path, {"rounds_executed": 1})
    _write_json(
        round0_path,
        {
            "candidate_label_ids_count_before": 2,
            "candidate_label_ids_count_after": 2,
        },
    )
    proc = _run(
        [
            "--loop-summary-json",
            str(loop_path),
            "--round0",
            str(round0_path),
        ]
    )
    assert proc.returncode != 0
    assert "candidate_label_ids_count_delta" in proc.stderr
