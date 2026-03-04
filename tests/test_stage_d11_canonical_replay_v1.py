from __future__ import annotations

import subprocess
from pathlib import Path


def _write_stub_scripts(
    repo_root: Path,
    *,
    helper_lines: list[str],
    helper_exit_code: int = 0,
) -> Path:
    tools_dir = repo_root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    log_path = repo_root / "order.log"

    layered_stub = """#!/usr/bin/env bash
set -euo pipefail
echo "layered:$*" >> "${PWD}/order.log"
"""
    helper_output = "\n".join(f'echo "{line}"' for line in helper_lines)
    helper_stub = f"""#!/usr/bin/env bash
set -euo pipefail
{helper_output}
echo "helper:$*" >> "${PWD}/order.log"
exit {helper_exit_code}
"""
    (tools_dir / "run_stage_d10_layered_fast_gate.sh").write_text(layered_stub, encoding="utf-8")
    (tools_dir / "run_stage_d9_smoke_helper.py").write_text(helper_stub, encoding="utf-8")
    return log_path


def test_n11_replay_help() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d11_canonical_replay.sh"
    proc = subprocess.run(
        ["bash", str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Replay the N11 canonical validation sequence in order" in proc.stdout


def test_n11_replay_runs_layered_then_pilot_with_defaults(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d11_canonical_replay.sh"
    stub_repo = tmp_path / "stub_repo"
    log_path = _write_stub_scripts(
        stub_repo,
        helper_lines=[
            "D10_PILOT_DIAGNOSTICS_CHECK_ENABLED=True",
            "D10_PILOT_DIAGNOSTICS_CHECKS_PASS=True",
            "D10_PILOT_NONZERO_MODE=gradient_coupled_pilot_v1",
            "D10_PILOT_NONZERO_ENABLED=True",
            "D10_PILOT_LOSS_WEIGHT=0.25",
            "D10_PILOT_SCALE=1e-06",
            "D10_PILOT_APPLIED=True",
            "D10_PILOT_STATE=applied",
            "D10_PILOT_SKIP_REASON=none",
            "D10_PILOT_NONZERO_STATE=nonzero_applied",
            "D10_PILOT_NONZERO_SKIP_REASON=none",
        ],
    )

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--repo-root",
            str(stub_repo),
            "--python-bin",
            "bash",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "D11_CANONICAL_REPLAY_STAGE=n10_layered_fast_gate PASS" in proc.stdout
    assert "D11_CANONICAL_REPLAY_STAGE=pilot_helper_smoke PASS" in proc.stdout
    assert "D11_CANONICAL_REPLAY=PASS" in proc.stdout
    assert "D10_PILOT_DIAGNOSTICS_CHECK_ENABLED=True" in proc.stdout
    assert "D10_PILOT_DIAGNOSTICS_CHECKS_PASS=True" in proc.stdout
    assert "D10_PILOT_NONZERO_MODE=gradient_coupled_pilot_v1" in proc.stdout
    assert "D10_PILOT_NONZERO_ENABLED=True" in proc.stdout
    assert "D10_PILOT_LOSS_WEIGHT=0.25" in proc.stdout
    assert "D10_PILOT_SCALE=1e-06" in proc.stdout
    assert "D10_PILOT_APPLIED=True" in proc.stdout
    assert "D10_PILOT_STATE=applied" in proc.stdout
    assert "D10_PILOT_SKIP_REASON=none" in proc.stdout
    assert "D10_PILOT_NONZERO_STATE=nonzero_applied" in proc.stdout
    assert "D10_PILOT_NONZERO_SKIP_REASON=none" in proc.stdout

    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("layered:")
    assert lines[1].startswith("helper:")
    assert "--on-mode pilot" in lines[1]
    assert "--pilot-scale 1e-6" in lines[1]


def test_n11_replay_forwards_optional_helper_args(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d11_canonical_replay.sh"
    stub_repo = tmp_path / "stub_repo"
    log_path = _write_stub_scripts(
        stub_repo,
        helper_lines=[
            "D10_PILOT_DIAGNOSTICS_CHECK_ENABLED=True",
            "D10_PILOT_DIAGNOSTICS_CHECKS_PASS=True",
            "D10_PILOT_APPLIED=True",
            "D10_PILOT_NONZERO_STATE=nonzero_applied",
        ],
    )

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--repo-root",
            str(stub_repo),
            "--python-bin",
            "bash",
            "--output-root",
            "/tmp/n11-smoke",
            "--config-path",
            "/tmp/n11-smoke.yaml",
            "--stagec-artifact-root",
            "/tmp/stagec",
            "--on-weight",
            "0.25",
            "--pilot-scale",
            "2e-6",
            "--keep-output",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert "--output-root /tmp/n11-smoke" in lines[1]
    assert "--config-path /tmp/n11-smoke.yaml" in lines[1]
    assert "--stagec-artifact-root /tmp/stagec" in lines[1]
    assert "--on-weight 0.25" in lines[1]
    assert "--pilot-scale 2e-6" in lines[1]
    assert "--keep-output" in lines[1]


def test_n11_replay_surfaces_skip_path_pilot_diagnostics_lines(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d11_canonical_replay.sh"
    stub_repo = tmp_path / "stub_repo"
    _write_stub_scripts(
        stub_repo,
        helper_lines=[
            "D10_PILOT_DIAGNOSTICS_CHECK_ENABLED=True",
            "D10_PILOT_DIAGNOSTICS_CHECKS_PASS=True",
            "D10_PILOT_NONZERO_MODE=gradient_coupled_pilot_v1",
            "D10_PILOT_APPLIED=False",
            "D10_PILOT_STATE=skipped",
            "D10_PILOT_SKIP_REASON=gradient_coupled_reference_unavailable",
            "D10_PILOT_NONZERO_STATE=skipped",
            "D10_PILOT_NONZERO_SKIP_REASON=gradient_coupled_reference_unavailable",
        ],
    )

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--repo-root",
            str(stub_repo),
            "--python-bin",
            "bash",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "D11_CANONICAL_REPLAY=PASS" in proc.stdout
    assert "D10_PILOT_DIAGNOSTICS_CHECK_ENABLED=True" in proc.stdout
    assert "D10_PILOT_DIAGNOSTICS_CHECKS_PASS=True" in proc.stdout
    assert "D10_PILOT_NONZERO_MODE=gradient_coupled_pilot_v1" in proc.stdout
    assert "D10_PILOT_APPLIED=False" in proc.stdout
    assert "D10_PILOT_STATE=skipped" in proc.stdout
    assert "D10_PILOT_SKIP_REASON=gradient_coupled_reference_unavailable" in proc.stdout
    assert "D10_PILOT_NONZERO_STATE=skipped" in proc.stdout
    assert "D10_PILOT_NONZERO_SKIP_REASON=gradient_coupled_reference_unavailable" in proc.stdout


def test_n11_replay_fails_fast_when_helper_reports_contradictory_pilot_lines(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d11_canonical_replay.sh"
    stub_repo = tmp_path / "stub_repo"
    _write_stub_scripts(
        stub_repo,
        helper_lines=[
            "D10_PILOT_DIAGNOSTICS_CHECK_ENABLED=True",
            "D10_PILOT_DIAGNOSTICS_CHECKS_PASS=False",
            "D10_PILOT_APPLIED=True",
            "D10_PILOT_STATE=skipped",
            "D10_PILOT_SKIP_REASON=gradient_coupled_reference_unavailable",
            "D10_PILOT_NONZERO_STATE=skipped",
        ],
        helper_exit_code=1,
    )

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--repo-root",
            str(stub_repo),
            "--python-bin",
            "bash",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "D11_CANONICAL_REPLAY=PASS" not in proc.stdout
    assert "D11_CANONICAL_REPLAY_STAGE=pilot_helper_smoke PASS" not in proc.stdout
