from __future__ import annotations

import subprocess
from pathlib import Path


def _pilot_fields(stdout: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        if not line.startswith("D10_PILOT_") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _assert_pilot_fields(stdout: str, expected: dict[str, str]) -> None:
    fields = _pilot_fields(stdout)
    for key, value in expected.items():
        assert key in fields, f"missing pilot field: {key}"
        assert fields[key] == value


def _write_stub_scripts(
    repo_root: Path,
    *,
    helper_lines: list[str],
    helper_exit_code: int = 0,
    bootstrap_lines: list[str] | None = None,
    bootstrap_exit_code: int = 0,
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
echo "helper:$*" >> "${{PWD}}/order.log"
exit {helper_exit_code}
"""
    (tools_dir / "run_stage_d10_layered_fast_gate.sh").write_text(layered_stub, encoding="utf-8")
    (tools_dir / "run_stage_d9_smoke_helper.py").write_text(helper_stub, encoding="utf-8")
    if bootstrap_lines is not None:
        bootstrap_output = "\n".join(f'echo "{line}"' for line in bootstrap_lines)
        bootstrap_stub = f"""#!/usr/bin/env bash
set -euo pipefail
{bootstrap_output}
echo "bootstrap:$*" >> "${{PWD}}/order.log"
exit {bootstrap_exit_code}
"""
        (tools_dir / "check_canonical_runner_bootstrap_links.py").write_text(
            bootstrap_stub,
            encoding="utf-8",
        )
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
            "D10_PILOT_APPLY_MODE=loss_dict_insert_zero",
            "D10_PILOT_INSERTED_INTO_LOSS_DICT=True",
            "D10_PILOT_USED_PLACEHOLDER_PATH=False",
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
    _assert_pilot_fields(
        proc.stdout,
        {
            "D10_PILOT_DIAGNOSTICS_CHECK_ENABLED": "True",
            "D10_PILOT_DIAGNOSTICS_CHECKS_PASS": "True",
            "D10_PILOT_APPLIED": "False",
            "D10_PILOT_STATE": "skipped",
            "D10_PILOT_SKIP_REASON": "gradient_coupled_reference_unavailable",
            "D10_PILOT_NONZERO_STATE": "skipped",
            "D10_PILOT_NONZERO_SKIP_REASON": "gradient_coupled_reference_unavailable",
            "D10_PILOT_APPLY_MODE": "loss_dict_insert_zero",
            "D10_PILOT_INSERTED_INTO_LOSS_DICT": "True",
            "D10_PILOT_USED_PLACEHOLDER_PATH": "False",
        },
    )


def test_n11_replay_surfaces_placeholder_skip_path_pilot_diagnostics_lines(tmp_path: Path) -> None:
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
            "D10_PILOT_SKIP_REASON=gradient_coupled_requires_loss_dict",
            "D10_PILOT_NONZERO_STATE=skipped",
            "D10_PILOT_NONZERO_SKIP_REASON=gradient_coupled_requires_loss_dict",
            "D10_PILOT_APPLY_MODE=placeholder_zero",
            "D10_PILOT_INSERTED_INTO_LOSS_DICT=False",
            "D10_PILOT_USED_PLACEHOLDER_PATH=True",
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
    _assert_pilot_fields(
        proc.stdout,
        {
            "D10_PILOT_DIAGNOSTICS_CHECK_ENABLED": "True",
            "D10_PILOT_DIAGNOSTICS_CHECKS_PASS": "True",
            "D10_PILOT_APPLIED": "False",
            "D10_PILOT_STATE": "skipped",
            "D10_PILOT_SKIP_REASON": "gradient_coupled_requires_loss_dict",
            "D10_PILOT_NONZERO_STATE": "skipped",
            "D10_PILOT_NONZERO_SKIP_REASON": "gradient_coupled_requires_loss_dict",
            "D10_PILOT_APPLY_MODE": "placeholder_zero",
            "D10_PILOT_INSERTED_INTO_LOSS_DICT": "False",
            "D10_PILOT_USED_PLACEHOLDER_PATH": "True",
        },
    )


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
    assert "D10_PILOT_DIAGNOSTICS_CHECKS_PASS=False" in proc.stdout
    assert "D10_PILOT_DIAGNOSTICS_CHECKS_PASS=True" not in proc.stdout
    assert "D11_CANONICAL_REPLAY=PASS" not in proc.stdout
    assert "D11_CANONICAL_REPLAY_STAGE=pilot_helper_smoke PASS" not in proc.stdout
    assert "D11_CANONICAL_REPLAY_STAGE=n10_layered_fast_gate PASS" in proc.stdout


def test_n11_replay_runs_bootstrap_check_before_replay_when_opted_in(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d11_canonical_replay.sh"
    stub_repo = tmp_path / "stub_repo"
    log_path = _write_stub_scripts(
        stub_repo,
        helper_lines=[
            "D10_PILOT_DIAGNOSTICS_CHECK_ENABLED=True",
            "D10_PILOT_DIAGNOSTICS_CHECKS_PASS=True",
        ],
        bootstrap_lines=[
            "BOOTSTRAP_LINK_CHECK_BEGIN",
            "path\tstatus\texpected\tactual\tnote",
            "runs\tOK\t../wsovvis_live/runs\t../wsovvis_live/runs\t-",
            "BOOTSTRAP_LINK_CHECK_END",
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
            "--bootstrap-link-check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "D11_CANONICAL_REPLAY_STAGE=bootstrap_link_preflight_check PASS" in proc.stdout
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3
    assert lines[0].startswith("bootstrap:")
    assert lines[0].endswith("--check")
    assert lines[1].startswith("layered:")
    assert lines[2].startswith("helper:")


def test_n11_replay_runs_bootstrap_fix_then_recheck_before_replay_when_opted_in(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d11_canonical_replay.sh"
    stub_repo = tmp_path / "stub_repo"
    log_path = _write_stub_scripts(
        stub_repo,
        helper_lines=[
            "D10_PILOT_DIAGNOSTICS_CHECK_ENABLED=True",
            "D10_PILOT_DIAGNOSTICS_CHECKS_PASS=True",
        ],
        bootstrap_lines=[
            "BOOTSTRAP_LINK_CHECK_BEGIN",
            "path\tstatus\texpected\tactual\tnote",
            "runs\tFIXED\t../wsovvis_live/runs\t../wsovvis_live/runs\t-",
            "BOOTSTRAP_LINK_CHECK_END",
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
            "--bootstrap-link-fix",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "D11_CANONICAL_REPLAY_STAGE=bootstrap_link_preflight_fix PASS" in proc.stdout
    assert "D11_CANONICAL_REPLAY_STAGE=bootstrap_link_preflight_recheck PASS" in proc.stdout
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 4
    assert lines[0].startswith("bootstrap:")
    assert lines[0].endswith("--fix")
    assert lines[1].startswith("bootstrap:")
    assert lines[1].endswith("--check")
    assert lines[2].startswith("layered:")
    assert lines[3].startswith("helper:")


def test_n11_replay_hard_stops_before_replay_on_bootstrap_failure(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d11_canonical_replay.sh"
    stub_repo = tmp_path / "stub_repo"
    log_path = _write_stub_scripts(
        stub_repo,
        helper_lines=["D10_PILOT_DIAGNOSTICS_CHECK_ENABLED=True"],
        bootstrap_lines=[
            "BOOTSTRAP_LINK_CHECK_BEGIN",
            "path\tstatus\texpected\tactual\tnote",
            "runs\tWRONG_TARGET\t../wsovvis_live/runs\t/tmp/other\t-",
            "BOOTSTRAP_LINK_CHECK_END",
        ],
        bootstrap_exit_code=1,
    )

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--repo-root",
            str(stub_repo),
            "--python-bin",
            "bash",
            "--bootstrap-link-check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "ERROR: bootstrap link preflight stage 'bootstrap_link_preflight_check' failed." in proc.stderr
    assert "D11_CANONICAL_REPLAY_STAGE=n10_layered_fast_gate START" not in proc.stdout
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("bootstrap:")


def test_n11_replay_hard_stops_on_bootstrap_skipped_status(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "run_stage_d11_canonical_replay.sh"
    stub_repo = tmp_path / "stub_repo"
    log_path = _write_stub_scripts(
        stub_repo,
        helper_lines=["D10_PILOT_DIAGNOSTICS_CHECK_ENABLED=True"],
        bootstrap_lines=[
            "BOOTSTRAP_LINK_CHECK_BEGIN",
            "path\tstatus\texpected\tactual\tnote",
            "runs\tSKIPPED\t../wsovvis_live/runs\t<not-a-symlink>\trefusing to replace non-symlink path",
            "BOOTSTRAP_LINK_CHECK_END",
        ],
        bootstrap_exit_code=0,
    )

    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--repo-root",
            str(stub_repo),
            "--python-bin",
            "bash",
            "--bootstrap-link-fix",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "ERROR: bootstrap link preflight reported SKIPPED status" in proc.stderr
    assert "D11_CANONICAL_REPLAY_STAGE=n10_layered_fast_gate START" not in proc.stdout
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("bootstrap:")
    assert lines[0].endswith("--fix")
