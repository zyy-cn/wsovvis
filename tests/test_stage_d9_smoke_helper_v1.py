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


def test_evaluate_pilot_diagnostics_checks_passes_with_applied_payload() -> None:
    helper = _load_helper_module()
    on_runtime_cfg = {
        "stage_d_attribution_d6_loss_key": {
            "applied": True,
            "skip_reason": "none",
            "nonzero_semantics_mode": "gradient_coupled_pilot_v1",
            "nonzero_semantics_enabled_by_config": True,
            "gate_status": {
                "gradient_coupled_mode_requested": True,
                "gradient_coupled_tensor_ready": True,
            },
            "planned_loss": {
                "apply_mode": "loss_dict_insert_nonzero_gradient_coupled_pilot",
                "loss_weight": 0.25,
                "gradient_coupled_scale": 1e-6,
            },
            "diagnostics": {
                "gradient_coupled_pilot_applied": True,
                "gradient_coupled_pilot_state": "applied",
                "gradient_coupled_pilot_skip_reason": "none",
                "nonzero_semantics_state": "nonzero_applied",
                "nonzero_skip_reason": "none",
            },
        }
    }

    out = helper._evaluate_pilot_diagnostics_checks(
        on_mode="pilot",
        on_runtime_cfg=on_runtime_cfg,
        expected_weight=0.25,
        pilot_scale=1e-6,
    )

    assert out["enabled"] is True
    assert out["checks_ok"] is True
    assert out["pilot_applied"] is True
    assert out["pilot_state"] == "applied"
    assert out["gate_tensor_ready"] is True
    assert out["apply_mode"] == "loss_dict_insert_nonzero_gradient_coupled_pilot"


def test_evaluate_pilot_diagnostics_checks_passes_with_small_weight_scale_boundary_values() -> None:
    helper = _load_helper_module()
    on_runtime_cfg = {
        "stage_d_attribution_d6_loss_key": {
            "applied": True,
            "skip_reason": "none",
            "nonzero_semantics_mode": "gradient_coupled_pilot_v1",
            "nonzero_semantics_enabled_by_config": True,
            "gate_status": {
                "gradient_coupled_mode_requested": True,
                "gradient_coupled_tensor_ready": True,
            },
            "planned_loss": {
                "apply_mode": "loss_dict_insert_nonzero_gradient_coupled_pilot",
                "loss_weight": 1e-9,
                "gradient_coupled_scale": 1e-6,
            },
            "diagnostics": {
                "gradient_coupled_pilot_applied": True,
                "gradient_coupled_pilot_state": "applied",
                "gradient_coupled_pilot_skip_reason": "none",
                "nonzero_semantics_state": "nonzero_applied",
                "nonzero_skip_reason": "none",
            },
        }
    }

    out = helper._evaluate_pilot_diagnostics_checks(
        on_mode="pilot",
        on_runtime_cfg=on_runtime_cfg,
        expected_weight=1e-9,
        pilot_scale=1e-6,
    )

    assert out["enabled"] is True
    assert out["checks_ok"] is True
    assert out["loss_weight"] == pytest.approx(1e-9, abs=1e-18)
    assert out["gradient_coupled_scale"] == pytest.approx(1e-6, abs=1e-18)
    assert out["pilot_applied"] is True
    assert out["pilot_state"] == "applied"


def test_evaluate_pilot_diagnostics_checks_fail_fast_on_zero_scale_invalid_payload() -> None:
    helper = _load_helper_module()
    on_runtime_cfg = {
        "stage_d_attribution_d6_loss_key": {
            "applied": False,
            "skip_reason": "invalid_gradient_coupled_scale",
            "nonzero_semantics_mode": "gradient_coupled_pilot_v1",
            "nonzero_semantics_enabled_by_config": True,
            "gate_status": {
                "gradient_coupled_mode_requested": True,
                "gradient_coupled_tensor_ready": False,
            },
            "planned_loss": {
                "apply_mode": "loss_dict_insert_zero",
                "loss_weight": 0.25,
                "gradient_coupled_scale": 0.0,
            },
            "diagnostics": {
                "gradient_coupled_pilot_applied": False,
                "gradient_coupled_pilot_state": "skipped",
                "gradient_coupled_pilot_skip_reason": "invalid_gradient_coupled_scale",
                "nonzero_semantics_state": "skipped",
                "nonzero_skip_reason": "invalid_gradient_coupled_scale",
            },
        }
    }

    with pytest.raises(RuntimeError, match="applied=True and skip_reason=none"):
        helper._evaluate_pilot_diagnostics_checks(
            on_mode="pilot",
            on_runtime_cfg=on_runtime_cfg,
            expected_weight=0.25,
            pilot_scale=0.0,
        )


def test_evaluate_pilot_diagnostics_checks_passes_with_skipped_payload() -> None:
    helper = _load_helper_module()
    on_runtime_cfg = {
        "stage_d_attribution_d6_loss_key": {
            "applied": True,
            "skip_reason": "none",
            "nonzero_semantics_mode": "gradient_coupled_pilot_v1",
            "nonzero_semantics_enabled_by_config": True,
            "gate_status": {
                "gradient_coupled_mode_requested": True,
                "gradient_coupled_tensor_ready": False,
            },
            "planned_loss": {
                "apply_mode": "loss_dict_insert_zero",
                "loss_weight": 0.25,
                "gradient_coupled_scale": 1e-6,
            },
            "diagnostics": {
                "gradient_coupled_pilot_applied": False,
                "gradient_coupled_pilot_state": "skipped",
                "gradient_coupled_pilot_skip_reason": "gradient_coupled_reference_unavailable",
                "nonzero_semantics_state": "skipped",
                "nonzero_skip_reason": "gradient_coupled_reference_unavailable",
            },
        }
    }

    out = helper._evaluate_pilot_diagnostics_checks(
        on_mode="pilot",
        on_runtime_cfg=on_runtime_cfg,
        expected_weight=0.25,
        pilot_scale=1e-6,
    )

    assert out["enabled"] is True
    assert out["checks_ok"] is True
    assert out["pilot_applied"] is False
    assert out["pilot_state"] == "skipped"
    assert out["gate_tensor_ready"] is False
    assert out["apply_mode"] == "loss_dict_insert_zero"


def test_evaluate_pilot_diagnostics_checks_fail_fast_on_missing_fields() -> None:
    helper = _load_helper_module()
    on_runtime_cfg = {
        "stage_d_attribution_d6_loss_key": {
            "applied": True,
            "skip_reason": "none",
            "nonzero_semantics_mode": "gradient_coupled_pilot_v1",
            "nonzero_semantics_enabled_by_config": True,
            "gate_status": {
                "gradient_coupled_mode_requested": True,
                "gradient_coupled_tensor_ready": False,
            },
            "planned_loss": {
                "apply_mode": "loss_dict_insert_zero",
                "loss_weight": 0.25,
                "gradient_coupled_scale": 1e-6,
            },
            "diagnostics": {
                "gradient_coupled_pilot_applied": False,
                # missing gradient_coupled_pilot_state
                "gradient_coupled_pilot_skip_reason": "gradient_coupled_reference_unavailable",
                "nonzero_semantics_state": "skipped",
                "nonzero_skip_reason": "gradient_coupled_reference_unavailable",
            },
        }
    }

    with pytest.raises(RuntimeError, match="gradient_coupled_pilot_state"):
        helper._evaluate_pilot_diagnostics_checks(
            on_mode="pilot",
            on_runtime_cfg=on_runtime_cfg,
            expected_weight=0.25,
            pilot_scale=1e-6,
        )


def test_evaluate_pilot_diagnostics_checks_fail_fast_on_inconsistent_applied_contract() -> None:
    helper = _load_helper_module()
    on_runtime_cfg = {
        "stage_d_attribution_d6_loss_key": {
            "applied": True,
            "skip_reason": "none",
            "nonzero_semantics_mode": "gradient_coupled_pilot_v1",
            "nonzero_semantics_enabled_by_config": True,
            "gate_status": {
                "gradient_coupled_mode_requested": True,
                "gradient_coupled_tensor_ready": True,
            },
            "planned_loss": {
                "apply_mode": "loss_dict_insert_nonzero_gradient_coupled_pilot",
                "loss_weight": 0.25,
                "gradient_coupled_scale": 1e-6,
            },
            "diagnostics": {
                "gradient_coupled_pilot_applied": True,
                "gradient_coupled_pilot_state": "applied",
                "gradient_coupled_pilot_skip_reason": "none",
                "nonzero_semantics_state": "skipped",
                "nonzero_skip_reason": "gradient_coupled_reference_unavailable",
            },
        }
    }

    with pytest.raises(RuntimeError, match="applied=True requires nonzero_semantics_state=nonzero_applied"):
        helper._evaluate_pilot_diagnostics_checks(
            on_mode="pilot",
            on_runtime_cfg=on_runtime_cfg,
            expected_weight=0.25,
            pilot_scale=1e-6,
        )


def test_evaluate_pilot_diagnostics_checks_fail_fast_on_gate_and_apply_mode_mismatch() -> None:
    helper = _load_helper_module()
    on_runtime_cfg = {
        "stage_d_attribution_d6_loss_key": {
            "applied": True,
            "skip_reason": "none",
            "nonzero_semantics_mode": "gradient_coupled_pilot_v1",
            "nonzero_semantics_enabled_by_config": True,
            "gate_status": {
                "gradient_coupled_mode_requested": True,
                "gradient_coupled_tensor_ready": False,
            },
            "planned_loss": {
                "apply_mode": "loss_dict_insert_nonzero_gradient_coupled_pilot",
                "loss_weight": 0.25,
                "gradient_coupled_scale": 1e-6,
            },
            "diagnostics": {
                "gradient_coupled_pilot_applied": True,
                "gradient_coupled_pilot_state": "applied",
                "gradient_coupled_pilot_skip_reason": "none",
                "nonzero_semantics_state": "nonzero_applied",
                "nonzero_skip_reason": "none",
            },
        }
    }

    with pytest.raises(
        RuntimeError,
        match="applied=True requires gate_status.gradient_coupled_tensor_ready=True",
    ):
        helper._evaluate_pilot_diagnostics_checks(
            on_mode="pilot",
            on_runtime_cfg=on_runtime_cfg,
            expected_weight=0.25,
            pilot_scale=1e-6,
        )


def test_evaluate_pilot_diagnostics_checks_noop_for_non_pilot_mode() -> None:
    helper = _load_helper_module()
    out = helper._evaluate_pilot_diagnostics_checks(
        on_mode="zero",
        on_runtime_cfg={},
        expected_weight=0.0,
        pilot_scale=None,
    )
    assert out["enabled"] is False
    assert out["checks_ok"] is True


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


def test_cli_dry_run_nonzero_mode_prints_nonzero_overrides(tmp_path: Path) -> None:
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
            "--on-mode",
            "nonzero",
            "--on-weight",
            "0.25",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "D10_DRY_RUN=1" in proc.stdout
    assert "D10_ON_MODE=nonzero" in proc.stdout
    assert "D10_ON_NONZERO_MODE=constant" in proc.stdout
    assert "stage_d_attribution.additive_loss_key.weight=0.25" in proc.stdout
    assert "stage_d_attribution.additive_loss_key.nonzero_semantics.enabled=True" in proc.stdout
    assert "gradient_coupled_pilot_v1" not in proc.stdout


def test_cli_dry_run_pilot_mode_prints_gradient_coupled_overrides(tmp_path: Path) -> None:
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
            "--on-mode",
            "pilot",
            "--on-weight",
            "0.25",
            "--pilot-scale",
            "1e-6",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "D10_DRY_RUN=1" in proc.stdout
    assert "D10_ON_MODE=pilot" in proc.stdout
    assert "D10_ON_NONZERO_MODE=gradient_coupled_pilot_v1" in proc.stdout
    assert "D10_ON_PILOT_SCALE=1e-06" in proc.stdout
    assert "stage_d_attribution.additive_loss_key.weight=0.25" in proc.stdout
    assert "stage_d_attribution.additive_loss_key.nonzero_semantics.enabled=True" in proc.stdout
    assert "stage_d_attribution.additive_loss_key.nonzero_semantics.mode=gradient_coupled_pilot_v1" in proc.stdout
    assert (
        "stage_d_attribution.additive_loss_key.nonzero_semantics.gradient_coupled_scale=1e-06"
        in proc.stdout
    )
