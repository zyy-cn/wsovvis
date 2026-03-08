from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "tools" / "run_mainline_loop.py"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _base_status(*, active_gate: str, gate_judgment: str, terminal_mode: str | None = None) -> str:
    lines = [
        "# WSOVVIS Mainline Status",
        "",
        "## Current state",
        f"- Active gate: `{active_gate}`",
        f"- Current {active_gate.split()[0]} judgment: `{gate_judgment}`",
    ]
    if terminal_mode is not None:
        lines.append(f"- Terminal mainline mode: `{terminal_mode}`")
    lines.append("")
    return "\n".join(lines)


def _run_script(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_script_path()), "--repo-root", str(repo_root), *args],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        check=False,
    )


def test_run_mainline_loop_non_terminal_writes_progress_prompt(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_text(
        repo_root / "docs/mainline/STATUS.md",
        _base_status(active_gate="G5 - Full-video linking and inference closure", gate_judgment="PASS"),
    )

    proc = _run_script(repo_root, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    prompt = (repo_root / "docs/mainline/reports/supervisor_prompt_latest.txt").read_text(encoding="utf-8")
    assert "Run exactly one bounded supervisor iteration." in prompt
    assert "If a scoped implementation step is required, perform only that step." in prompt
    assert not (repo_root / "docs/mainline/reports/mainline_terminal_summary.txt").exists()


def test_run_mainline_loop_terminal_mode_stops_and_writes_summary(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_text(
        repo_root / "docs/mainline/STATUS.md",
        _base_status(
            active_gate="G6 - Single-round bounded refinement",
            gate_judgment="PASS",
            terminal_mode="active",
        ),
    )
    _write_text(
        repo_root / "docs/mainline/reports/acceptance_latest.txt",
        "7. May the next gate activate?\n- `no further gate`\n",
    )

    proc = _run_script(repo_root, "--dry-run")
    assert proc.returncode == 0, proc.stderr

    prompt = (repo_root / "docs/mainline/reports/supervisor_prompt_latest.txt").read_text(encoding="utf-8")
    summary = (repo_root / "docs/mainline/reports/mainline_terminal_summary.txt").read_text(encoding="utf-8")

    assert "No new coding step was generated." in prompt
    assert "terminal mainline summary" in summary.lower()
    assert "tests/test_g5_bounded_policy_v1.py" in summary
    assert "tests/test_stage_d0_self_training_loop_v1.py::test_d0_emit_ws_metrics_preserves_hidden_positive_fields_for_hpr_uar" in summary


def test_run_mainline_loop_terminal_revalidate_writes_bounded_prompt(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write_text(
        repo_root / "docs/mainline/STATUS.md",
        _base_status(
            active_gate="G6 - Single-round bounded refinement",
            gate_judgment="PASS",
            terminal_mode="active",
        ),
    )
    _write_text(
        repo_root / "docs/mainline/reports/acceptance_latest.txt",
        "7. May the next gate activate?\n- `no further gate`\n",
    )

    proc = _run_script(repo_root, "--dry-run", "--terminal-revalidate")
    assert proc.returncode == 0, proc.stderr

    prompt = (repo_root / "docs/mainline/reports/supervisor_prompt_latest.txt").read_text(encoding="utf-8")
    assert "Operate strictly in bounded terminal revalidation mode." in prompt
    assert "gpu4090d" in prompt
    assert "tests/test_stagec4_sinkhorn_scorer_v1.py::test_sinkhorn_c43_unk_fg_gating_schema_and_behavior" in prompt
    assert "tests/test_g5_bounded_policy_v1.py" in prompt
    assert "tests/test_stage_d_reporting_snapshot_v1.py::test_stage_d_snapshot_happy_path_detects_round_paths" in prompt
    assert "Do not continue algorithm development." in prompt
