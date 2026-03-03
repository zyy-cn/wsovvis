from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _write_stub_scripts(repo_root: Path) -> Path:
    tools_dir = repo_root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    log_path = repo_root / "order.log"

    helper_stub = """#!/usr/bin/env bash
set -euo pipefail
echo "helper:$*" >> "${PWD}/order.log"
"""
    replay_stub = """#!/usr/bin/env bash
set -euo pipefail
echo "replay:$*" >> "${PWD}/order.log"
"""
    (tools_dir / "run_stage_d9_helper_tests_quick.sh").write_text(helper_stub, encoding="utf-8")
    (tools_dir / "run_stage_d11_canonical_replay.sh").write_text(replay_stub, encoding="utf-8")
    return log_path


def test_d13_ci_quick_pipeline_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d13_ci_quick_pipeline.sh"
    proc = subprocess.run(
        ["bash", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Runs a branch-local CI mirror pipeline in order" in proc.stdout


def test_d13_ci_quick_pipeline_runs_helper_then_replay(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d13_ci_quick_pipeline.sh"
    stub_repo = tmp_path / "stub_repo"
    log_path = _write_stub_scripts(stub_repo)

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--repo-root",
            str(stub_repo),
            "--python-bin",
            sys.executable,
            "--on-weight",
            "0.25",
            "--pilot-scale",
            "2e-6",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "D13_CI_QUICK_PIPELINE_STAGE=helper_fast_gate PASS" in proc.stdout
    assert "D13_CI_QUICK_PIPELINE_STAGE=canonical_replay PASS" in proc.stdout
    assert "D13_CI_QUICK_PIPELINE=PASS" in proc.stdout

    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("helper:")
    assert lines[1].startswith("replay:")
    assert "--on-weight 0.25" in lines[1]
    assert "--pilot-scale 2e-6" in lines[1]
