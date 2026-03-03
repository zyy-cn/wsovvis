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
    quick_stub = """#!/usr/bin/env bash
set -euo pipefail
echo "quick:$*" >> "${PWD}/order.log"
"""
    (tools_dir / "run_stage_d9_helper_tests_quick.sh").write_text(helper_stub, encoding="utf-8")
    (tools_dir / "run_stage_d10_quick_checks.sh").write_text(quick_stub, encoding="utf-8")
    return log_path


def test_layered_fast_gate_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d10_layered_fast_gate.sh"
    proc = subprocess.run(
        ["bash", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Runs layered fast-gate validation in order" in proc.stdout


def test_layered_fast_gate_default_runs_helper_only(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d10_layered_fast_gate.sh"
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
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "D10_LAYERED_FAST_GATE_STAGE=helper_coverage PASS" in proc.stdout
    assert "D10_LAYERED_FAST_GATE_STAGE=pilot_quick_check SKIP" in proc.stdout

    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("helper:")


def test_layered_fast_gate_with_pilot_smoke_runs_both_and_forwards_args(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d10_layered_fast_gate.sh"
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
            "--with-pilot-smoke",
            "--pilot-on-mode",
            "pilot",
            "--pilot-on-weight",
            "0.25",
            "--pilot-scale",
            "1e-6",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "D10_LAYERED_FAST_GATE_STAGE=pilot_quick_check PASS" in proc.stdout

    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("helper:")
    assert lines[1].startswith("quick:")
    assert "--on-mode pilot" in lines[1]
    assert "--on-weight 0.25" in lines[1]
    assert "--pilot-scale 1e-6" in lines[1]


def test_layered_fast_gate_rejects_pilot_args_without_flag(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d10_layered_fast_gate.sh"
    stub_repo = tmp_path / "stub_repo"
    _write_stub_scripts(stub_repo)

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--repo-root",
            str(stub_repo),
            "--pilot-on-weight",
            "0.25",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "Pilot quick-check args require --with-pilot-smoke" in proc.stderr
