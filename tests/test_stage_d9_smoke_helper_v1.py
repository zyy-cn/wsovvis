from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_helper_module():
    repo_root = Path(__file__).resolve().parents[1]
    helper_path = repo_root / "tools" / "run_stage_d9_smoke_helper.py"
    spec = importlib.util.spec_from_file_location("run_stage_d9_smoke_helper", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_metrics_checks_off_on_parity_and_zero_attr() -> None:
    helper = _load_helper_module()
    off_rows = [{"iteration": 0, "total_loss": 1.25}]
    on_rows = [
        {"iteration": 0, "loss_stage_d_attr": 0.0, "total_loss": 1.25},
        {"iteration": 1, "loss_stage_d_attr": 0, "total_loss": 1.25},
    ]

    out = helper._evaluate_metrics_checks(
        off_rows=off_rows,
        on_rows=on_rows,
        total_loss_key="total_loss",
        parity_tol=1e-9,
    )

    assert out["off_has_attr"] is False
    assert out["on_has_attr"] is True
    assert out["on_last_attr"] == pytest.approx(0.0)
    assert out["parity_ok"] is True
    assert out["on_zero_ok"] is True
    assert out["checks_ok"] is True


def test_evaluate_metrics_checks_total_loss_parity_within_tolerance() -> None:
    helper = _load_helper_module()
    off_rows = [{"total_loss": 3.0}]
    on_rows = [{"loss_stage_d_attr": 0.0, "total_loss": 3.0 + 5e-10}]

    out = helper._evaluate_metrics_checks(
        off_rows=off_rows,
        on_rows=on_rows,
        total_loss_key="total_loss",
        parity_tol=1e-9,
    )

    assert out["parity_ok"] is True
    assert out["checks_ok"] is True


def test_evaluate_metrics_checks_nonzero_mode_requires_nonzero_attr_and_total_increase() -> None:
    helper = _load_helper_module()
    off_rows = [{"total_loss": 1.5}]
    on_rows = [{"loss_stage_d_attr": 0.25, "total_loss": 1.75}]

    out = helper._evaluate_metrics_checks(
        off_rows=off_rows,
        on_rows=on_rows,
        total_loss_key="total_loss",
        parity_tol=1e-9,
        expect_nonzero_on=True,
        nonzero_eps=1e-12,
    )

    assert out["expect_nonzero_on"] is True
    assert out["on_nonzero_ok"] is True
    assert out["total_increase_ok"] is True
    assert out["checks_ok"] is True


def test_evaluate_metrics_checks_nonzero_mode_fails_when_attr_is_zero() -> None:
    helper = _load_helper_module()
    off_rows = [{"total_loss": 1.5}]
    on_rows = [{"loss_stage_d_attr": 0.0, "total_loss": 1.5}]

    out = helper._evaluate_metrics_checks(
        off_rows=off_rows,
        on_rows=on_rows,
        total_loss_key="total_loss",
        parity_tol=1e-9,
        expect_nonzero_on=True,
        nonzero_eps=1e-12,
    )

    assert out["on_nonzero_ok"] is False
    assert out["total_increase_ok"] is False
    assert out["checks_ok"] is False


def test_load_metrics_rows_fail_fast_on_missing_or_bad_json(tmp_path: Path) -> None:
    helper = _load_helper_module()
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        helper._load_metrics_rows(missing)

    bad = tmp_path / "bad.json"
    bad.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        helper._load_metrics_rows(bad)


def test_evaluate_metrics_checks_fail_fast_on_missing_total_loss_key() -> None:
    helper = _load_helper_module()
    off_rows = [{"loss": 1.0}]
    on_rows = [{"loss_stage_d_attr": 0.0, "loss": 1.0}]

    with pytest.raises(RuntimeError, match="no numeric 'total_loss' found"):
        helper._evaluate_metrics_checks(
            off_rows=off_rows,
            on_rows=on_rows,
            total_loss_key="total_loss",
            parity_tol=1e-9,
        )


def test_cli_dry_run_prints_commands_without_training(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    helper_path = repo_root / "tools" / "run_stage_d9_smoke_helper.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(helper_path),
            "--repo-root",
            str(repo_root),
            "--output-root",
            str(tmp_path / "smoke"),
            "--config-path",
            str(tmp_path / "cfg.yaml"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "D10_DRY_RUN=1" in proc.stdout
    assert "D10_OFF_CMD=" in proc.stdout
    assert "D10_ON_CMD=" in proc.stdout
