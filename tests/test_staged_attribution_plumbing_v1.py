from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wsovvis.training import (
    StageDAttributionPlumbingError,
    apply_stage_d_additive_loss_key,
    build_stage_d_attribution_consumption_boundary,
    build_stage_d_objective_coupling_decision,
    consume_stage_d_attribution_config,
    resolve_stage_d_attribution_plumbing,
)


def _write_stagec_artifacts(root: Path, *, num_tracks_scored: int = 2, embedding_dim: int = 256) -> None:
    root.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "split": "val",
        "embedding_dim": embedding_dim,
        "num_tracks_scored": num_tracks_scored,
        "scorer_backend": "mil_v1",
    }
    (root / "run_summary.json").write_text(json.dumps(run_summary), encoding="utf-8")
    rows = []
    for i in range(num_tracks_scored):
        rows.append(
            {
                "video_id": "v1",
                "track_id": i,
                "row_index": i,
                "status": "processed_with_tracks",
                "score": float(i + 1),
            }
        )
    (root / "track_scores.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def test_stage_d_plumbing_default_off_is_compatible(tmp_path: Path) -> None:
    resolved = resolve_stage_d_attribution_plumbing(None, repo_root=tmp_path)
    assert resolved == {"enabled": False}
    consumed = consume_stage_d_attribution_config(resolved)
    assert consumed["enabled"] is False
    assert consumed["mode"] == "disabled_noop"
    assert consumed["consumed"] is True
    assert consumed["runtime_diag_version"] == "d3_runtime_v1"
    assert consumed["consumer_status"] == "skipped"
    assert consumed["skip_reason"] == "disabled_by_config"
    assert consumed["compatibility"]["default_off_compatible"] is True
    assert consumed["compatibility"]["training_objective_affected"] is False
    assert consumed["summary_counters"] == {
        "rows_validated": 0,
        "rows_expected": 0,
        "videos_count": 0,
        "tracks_count": 0,
    }
    assert consumed["counters"]["enabled_config_consumed"] == 0
    assert consumed["counters"]["objective_changes"] == 0
    assert consumed["counters"]["loss_changes"] == 0


def test_stage_d_plumbing_enabled_requires_artifact_reference(tmp_path: Path) -> None:
    with pytest.raises(StageDAttributionPlumbingError, match="stage_d_attribution.stagec_artifact_root"):
        resolve_stage_d_attribution_plumbing({"enabled": True}, repo_root=tmp_path)


def test_stage_d_plumbing_nominal_from_artifact_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stagec_out"
    _write_stagec_artifacts(artifact_root, num_tracks_scored=2, embedding_dim=256)

    resolved = resolve_stage_d_attribution_plumbing(
        {
            "enabled": True,
            "stagec_artifact_root": str(artifact_root),
            "required_scorer_backend": "mil_v1",
            "expected_embedding_dim": 256,
        },
        repo_root=tmp_path,
    )

    assert resolved["enabled"] is True
    assert Path(resolved["stagec_run_summary_path"]) == artifact_root / "run_summary.json"
    assert Path(resolved["stagec_track_scores_path"]) == artifact_root / "track_scores.jsonl"
    assert resolved["summary"]["num_tracks_scored"] == 2
    assert resolved["track_score_rows_validated"] == 2


def test_stage_d_consumer_enabled_nominal_is_noop_with_diagnostics(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stagec_out"
    _write_stagec_artifacts(artifact_root, num_tracks_scored=2, embedding_dim=256)
    resolved = resolve_stage_d_attribution_plumbing(
        {"enabled": True, "stagec_artifact_root": str(artifact_root)},
        repo_root=tmp_path,
    )

    consumed = consume_stage_d_attribution_config(resolved)
    assert consumed["enabled"] is True
    assert consumed["mode"] == "enabled_noop"
    assert consumed["consumed"] is True
    assert consumed["runtime_diag_version"] == "d3_runtime_v1"
    assert consumed["consumer_status"] == "loaded"
    assert consumed["skip_reason"] == "none"
    assert consumed["compatibility"]["default_off_compatible"] is True
    assert consumed["compatibility"]["training_objective_affected"] is False
    assert consumed["summary"]["embedding_dim"] == 256
    assert consumed["track_score_rows_validated"] == 2
    assert consumed["summary_counters"] == {
        "rows_validated": 2,
        "rows_expected": 2,
        "videos_count": 2,
        "tracks_count": 2,
    }
    assert consumed["provenance"]["source_kind"] == "stagec_artifact"
    assert consumed["provenance"]["scorer_backend"] == "mil_v1"
    assert consumed["provenance"]["embedding_dim"] == 256
    assert consumed["provenance"]["run_summary_path"] == consumed["stagec_run_summary_path"]
    assert consumed["provenance"]["track_scores_path"] == consumed["stagec_track_scores_path"]
    assert consumed["counters"]["enabled_config_consumed"] == 1
    assert consumed["counters"]["objective_changes"] == 0
    assert consumed["counters"]["loss_changes"] == 0


def test_stage_d_plumbing_fail_fast_on_run_summary_schema(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stagec_out"
    _write_stagec_artifacts(artifact_root, num_tracks_scored=1, embedding_dim=256)
    payload = json.loads((artifact_root / "run_summary.json").read_text(encoding="utf-8"))
    payload.pop("scorer_backend")
    (artifact_root / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StageDAttributionPlumbingError, match="stagec_run_summary.scorer_backend"):
        resolve_stage_d_attribution_plumbing(
            {"enabled": True, "stagec_artifact_root": str(artifact_root)},
            repo_root=tmp_path,
        )


def test_stage_d_plumbing_fail_fast_on_track_scores_schema(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stagec_out"
    _write_stagec_artifacts(artifact_root, num_tracks_scored=1, embedding_dim=256)
    (artifact_root / "track_scores.jsonl").write_text(
        json.dumps({"video_id": "v1", "track_id": "t1", "row_index": 0, "status": "processed_with_tracks"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StageDAttributionPlumbingError, match="stagec_track_scores\\[1\\]\\.score"):
        resolve_stage_d_attribution_plumbing(
            {"enabled": True, "stagec_artifact_root": str(artifact_root)},
            repo_root=tmp_path,
        )


def test_stage_d_plumbing_fail_fast_on_embedding_dim_mismatch(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stagec_out"
    _write_stagec_artifacts(artifact_root, num_tracks_scored=1, embedding_dim=256)

    with pytest.raises(StageDAttributionPlumbingError, match="dimension mismatch"):
        resolve_stage_d_attribution_plumbing(
            {"enabled": True, "stagec_artifact_root": str(artifact_root), "expected_embedding_dim": 128},
            repo_root=tmp_path,
        )


def test_stage_d4_consume_boundary_default_off_is_explicit_skip() -> None:
    boundary = build_stage_d_attribution_consumption_boundary(
        {
            "stage_d_attribution": {"enabled": False},
            "stage_d_attribution_runtime": {"enabled": False, "runtime_diag_version": "d3_runtime_v1"},
        }
    )
    assert boundary["consume_boundary_version"] == "d4_consume_boundary_v1"
    assert boundary["enabled"] is False
    assert boundary["consume_status"] == "skipped"
    assert boundary["skip_reason"] == "disabled_by_config"
    assert boundary["compatibility"]["default_off_compatible"] is True
    assert boundary["compatibility"]["training_objective_affected"] is False
    assert boundary["objective_placeholder"]["coupling_status"] == "noop"
    assert boundary["objective_placeholder"]["ready_for_objective_coupling"] is False
    assert boundary["counters"]["boundary_active"] == 0
    assert boundary["counters"]["objective_changes"] == 0
    assert boundary["counters"]["loss_changes"] == 0


def test_stage_d4_consume_boundary_enabled_valid_runtime_active_noop(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stagec_out"
    _write_stagec_artifacts(artifact_root, num_tracks_scored=2, embedding_dim=256)
    resolved = resolve_stage_d_attribution_plumbing(
        {"enabled": True, "stagec_artifact_root": str(artifact_root)},
        repo_root=tmp_path,
    )
    runtime = consume_stage_d_attribution_config(resolved)

    boundary = build_stage_d_attribution_consumption_boundary(
        {
            "stage_d_attribution": resolved,
            "stage_d_attribution_runtime": runtime,
        }
    )
    assert boundary["enabled"] is True
    assert boundary["consume_status"] == "active_noop"
    assert boundary["skip_reason"] == "none"
    assert boundary["guard_flags"]["runtime_shape_valid_for_enabled"] is True
    assert boundary["summary"]["scorer_backend"] == "mil_v1"
    assert boundary["track_score_rows_validated"] == 2
    assert boundary["objective_placeholder"]["interface_version"] == "d4_objective_placeholder_v1"
    assert boundary["objective_placeholder"]["coupling_status"] == "noop"
    assert boundary["objective_placeholder"]["ready_for_objective_coupling"] is True
    assert boundary["counters"]["boundary_active"] == 1
    assert boundary["counters"]["objective_changes"] == 0
    assert boundary["counters"]["loss_changes"] == 0


def test_stage_d4_consume_boundary_enabled_invalid_runtime_skip_closed() -> None:
    boundary = build_stage_d_attribution_consumption_boundary(
        {
            "stage_d_attribution": {"enabled": True},
            "stage_d_attribution_runtime": {
                "enabled": True,
                "runtime_diag_version": "d3_runtime_v1",
                "summary": {"scorer_backend": "mil_v1", "embedding_dim": 256, "num_tracks_scored": 2},
            },
        }
    )
    assert boundary["enabled"] is True
    assert boundary["consume_status"] == "skipped"
    assert boundary["skip_reason"] == "runtime_missing_or_invalid_for_enabled_config"
    assert boundary["policy"]["enabled_runtime_invalid_action"] == "skip_closed"
    assert boundary["policy"]["hard_fail_on_runtime_invalid"] is False
    assert boundary["guard_flags"]["runtime_mapping_present"] is True
    assert boundary["guard_flags"]["runtime_shape_valid_for_enabled"] is False
    assert boundary["objective_placeholder"]["coupling_status"] == "noop"
    assert boundary["objective_placeholder"]["ready_for_objective_coupling"] is False
    assert boundary["counters"]["boundary_active"] == 0
    assert boundary["counters"]["boundary_skipped"] == 1


def test_stage_d5_objective_coupling_default_off_is_skip_closed() -> None:
    coupling = build_stage_d_objective_coupling_decision(
        {
            "stage_d_attribution": {"enabled": False},
            "stage_d_attribution_runtime": {"enabled": False, "runtime_diag_version": "d3_runtime_v1"},
            "stage_d_attribution_consumption": {"enabled": False, "consume_status": "skipped"},
        }
    )
    assert coupling["coupling_version"] == "d5_objective_coupling_v1"
    assert coupling["enabled_by_config"] is False
    assert coupling["eligible"] is False
    assert coupling["applied"] is False
    assert coupling["skip_reason"] == "stage_d_attribution_disabled"
    assert coupling["gate_status"]["stage_d_enabled"] is False
    assert coupling["gate_status"]["d4_boundary_ready"] is False
    assert coupling["diagnostics"]["considered"] is False
    assert coupling["diagnostics"]["applied"] is False
    assert coupling["planned_loss"]["loss_value"] == 0.0


def test_stage_d5_objective_coupling_enabled_valid_runtime_reaches_apply_path(tmp_path: Path) -> None:
    artifact_root = tmp_path / "stagec_out"
    _write_stagec_artifacts(artifact_root, num_tracks_scored=2, embedding_dim=256)
    resolved = resolve_stage_d_attribution_plumbing(
        {
            "enabled": True,
            "stagec_artifact_root": str(artifact_root),
            "objective_coupling": {"enabled": True},
        },
        repo_root=tmp_path,
    )
    resolved = {**resolved, "objective_coupling": {"enabled": True}}
    runtime = consume_stage_d_attribution_config(resolved)
    boundary = build_stage_d_attribution_consumption_boundary(
        {"stage_d_attribution": resolved, "stage_d_attribution_runtime": runtime}
    )

    coupling = build_stage_d_objective_coupling_decision(
        {
            "stage_d_attribution": resolved,
            "stage_d_attribution_runtime": runtime,
            "stage_d_attribution_consumption": boundary,
        }
    )
    assert coupling["enabled_by_config"] is True
    assert coupling["eligible"] is True
    assert coupling["applied"] is True
    assert coupling["skip_reason"] == "none"
    assert coupling["gate_status"]["stage_d_enabled"] is True
    assert coupling["gate_status"]["d4_boundary_ready"] is True
    assert coupling["gate_status"]["runtime_fields_valid"] is True
    assert coupling["gate_status"]["runtime_boundary_consistent"] is True
    assert coupling["diagnostics"]["considered"] is True
    assert coupling["diagnostics"]["applied"] is True
    assert coupling["planned_loss"]["loss_key"] == "loss_stage_d_attribution_coupling"
    assert coupling["planned_loss"]["apply_mode"] == "no_op_placeholder"
    assert coupling["planned_loss"]["loss_value"] == 0.0


def test_stage_d5_objective_coupling_enabled_invalid_runtime_boundary_skip_closed() -> None:
    boundary = build_stage_d_attribution_consumption_boundary(
        {
            "stage_d_attribution": {"enabled": True, "objective_coupling": {"enabled": True}},
            "stage_d_attribution_runtime": {
                "enabled": True,
                "runtime_diag_version": "d3_runtime_v1",
                "summary": {"scorer_backend": "mil_v1", "embedding_dim": 256, "num_tracks_scored": 2},
            },
        }
    )
    coupling = build_stage_d_objective_coupling_decision(
        {
            "stage_d_attribution": {"enabled": True, "objective_coupling": {"enabled": True}},
            "stage_d_attribution_runtime": {
                "enabled": True,
                "runtime_diag_version": "d3_runtime_v1",
                "summary": {"scorer_backend": "mil_v1", "embedding_dim": 256, "num_tracks_scored": 2},
            },
            "stage_d_attribution_consumption": boundary,
        }
    )
    assert coupling["enabled_by_config"] is True
    assert coupling["eligible"] is False
    assert coupling["applied"] is False
    assert coupling["skip_reason"] == "d4_boundary_not_ready"
    assert coupling["gate_status"]["stage_d_enabled"] is True
    assert coupling["gate_status"]["d4_boundary_ready"] is False
    assert coupling["diagnostics"]["considered"] is False
    assert coupling["planned_loss"]["loss_value"] == 0.0


def test_stage_d6_additive_loss_key_default_off_is_skip_closed() -> None:
    losses: dict[str, float] = {"loss_mask": 1.0}
    d6 = apply_stage_d_additive_loss_key(
        {
            "stage_d_attribution": {"enabled": False},
            "stage_d_attribution_coupling": {"applied": False},
        },
        losses,
    )
    assert d6["application_version"] == "d6_additive_loss_key_v1"
    assert d6["enabled_by_config"] is False
    assert d6["eligible"] is False
    assert d6["applied"] is False
    assert d6["skip_reason"] == "stage_d_attribution_disabled"
    assert d6["diagnostics"]["inserted_into_loss_dict"] is False
    assert "loss_stage_d_attr" not in losses


def test_stage_d6_additive_loss_key_weight_zero_inserts_observable_noop() -> None:
    losses: dict[str, float] = {"loss_mask": 1.0}
    d6 = apply_stage_d_additive_loss_key(
        {
            "stage_d_attribution": {
                "enabled": True,
                "additive_loss_key": {"enabled": True, "weight": 0.0},
            },
            "stage_d_attribution_coupling": {"applied": True},
        },
        losses,
    )
    assert d6["enabled_by_config"] is True
    assert d6["eligible"] is True
    assert d6["applied"] is True
    assert d6["skip_reason"] == "none"
    assert d6["planned_loss"]["loss_key"] == "loss_stage_d_attr"
    assert d6["planned_loss"]["loss_weight"] == 0.0
    assert d6["planned_loss"]["apply_mode"] == "loss_dict_insert_zero"
    assert d6["diagnostics"]["applied"] is True
    assert d6["diagnostics"]["inserted_into_loss_dict"] is True
    assert d6["diagnostics"]["weight_zero_noop_observed"] is True
    assert d6["diagnostics"]["nonzero_semantics_state"] == "zero_weight_noop"
    assert losses["loss_stage_d_attr"] == 0.0


def test_stage_d6_additive_loss_key_requires_d5_applied_and_skip_closed_when_missing() -> None:
    losses: dict[str, float] = {}
    d6 = apply_stage_d_additive_loss_key(
        {
            "stage_d_attribution": {
                "enabled": True,
                "additive_loss_key": {"enabled": True, "weight": 1.0},
            },
            "stage_d_attribution_coupling": {"applied": False},
        },
        losses,
    )
    assert d6["enabled_by_config"] is True
    assert d6["eligible"] is False
    assert d6["applied"] is False
    assert d6["skip_reason"] == "d5_coupling_not_applied"
    assert d6["gate_status"]["d5_coupling_applied"] is False
    assert "loss_stage_d_attr" not in losses


def test_stage_d6_additive_loss_key_nonzero_weight_path_requires_d4_and_applies_nonzero(tmp_path: Path) -> None:
    cfg_dict = _build_stage_d8_enabled_cfg(tmp_path, weight=0.25, nonzero_semantics_enabled=True)
    losses: dict[str, torch.Tensor] = {"loss_mask": torch.tensor(1.0)}
    d6 = apply_stage_d_additive_loss_key(cfg_dict, losses)

    assert d6["applied"] is True
    assert d6["skip_reason"] == "none"
    assert d6["gate_status"]["stage_d_enabled"] is True
    assert d6["gate_status"]["d4_boundary_ready"] is True
    assert d6["gate_status"]["d5_coupling_applied"] is True
    assert d6["gate_status"]["nonzero_prereqs_satisfied"] is True
    assert d6["planned_loss"]["apply_mode"] == "loss_dict_insert_nonzero"
    assert d6["planned_loss"]["loss_weight"] == pytest.approx(0.25)
    assert d6["planned_loss"]["loss_value"] == pytest.approx(0.25)
    assert d6["diagnostics"]["nonzero_semantics_state"] == "nonzero_applied"
    assert d6["diagnostics"]["nonzero_skip_reason"] == "none"
    assert isinstance(losses["loss_stage_d_attr"], torch.Tensor)
    assert float(losses["loss_stage_d_attr"].item()) == pytest.approx(0.25)


def test_stage_d6_additive_loss_key_nonzero_enabled_without_d4_boundary_is_observable_skip_closed() -> None:
    losses: dict[str, torch.Tensor] = {"loss_mask": torch.tensor(1.0)}
    d6 = apply_stage_d_additive_loss_key(
        {
            "stage_d_attribution": {
                "enabled": True,
                "additive_loss_key": {
                    "enabled": True,
                    "weight": 0.25,
                    "nonzero_semantics": {"enabled": True},
                },
            },
            "stage_d_attribution_coupling": {"applied": True},
        },
        losses,
    )
    assert d6["applied"] is True
    assert d6["skip_reason"] == "none"
    assert d6["gate_status"]["d4_boundary_ready"] is False
    assert d6["gate_status"]["nonzero_prereqs_satisfied"] is False
    assert d6["planned_loss"]["apply_mode"] == "loss_dict_insert_zero"
    assert d6["diagnostics"]["nonzero_semantics_state"] == "skipped"
    assert d6["diagnostics"]["nonzero_skip_reason"] == "d4_boundary_not_ready"
    assert isinstance(losses["loss_stage_d_attr"], torch.Tensor)
    assert float(losses["loss_stage_d_attr"].item()) == pytest.approx(0.0)


def test_stage_d6_additive_loss_key_uses_placeholder_when_loss_dict_missing() -> None:
    d6 = apply_stage_d_additive_loss_key(
        {
            "stage_d_attribution": {
                "enabled": True,
                "additive_loss_key": {"enabled": True, "weight": 1.0},
            },
            "stage_d_attribution_coupling": {"applied": True},
        },
    )
    assert d6["enabled_by_config"] is True
    assert d6["eligible"] is True
    assert d6["applied"] is True
    assert d6["skip_reason"] == "none"
    assert d6["diagnostics"]["inserted_into_loss_dict"] is False
    assert d6["diagnostics"]["used_placeholder_path"] is True
    assert d6["planned_loss"]["apply_mode"] == "placeholder_zero"


def test_stage_d7_real_hook_default_off_parity_keeps_loss_dict_unchanged() -> None:
    losses: dict[str, torch.Tensor] = {"loss_mask": torch.tensor(1.0)}
    d6 = apply_stage_d_additive_loss_key(
        {
            "stage_d_attribution": {"enabled": False},
            "stage_d_attribution_coupling": {"applied": False},
        },
        losses,
    )
    assert d6["applied"] is False
    assert d6["skip_reason"] == "stage_d_attribution_disabled"
    assert "loss_stage_d_attr" not in losses
    assert float(losses["loss_mask"].item()) == pytest.approx(1.0)


def test_stage_d7_real_hook_weight_zero_inserts_tensor_zero_observable_noop() -> None:
    losses: dict[str, torch.Tensor] = {"loss_mask": torch.tensor(1.0)}
    d6 = apply_stage_d_additive_loss_key(
        {
            "stage_d_attribution": {
                "enabled": True,
                "additive_loss_key": {"enabled": True, "weight": 0.0},
            },
            "stage_d_attribution_coupling": {"applied": True},
        },
        losses,
    )
    assert d6["applied"] is True
    assert d6["skip_reason"] == "none"
    assert d6["diagnostics"]["inserted_into_loss_dict"] is True
    assert d6["diagnostics"]["weight_zero_noop_observed"] is True
    assert isinstance(losses["loss_stage_d_attr"], torch.Tensor)
    assert float(losses["loss_stage_d_attr"].item()) == pytest.approx(0.0)


def test_stage_d7_real_hook_skip_closed_when_prereq_missing() -> None:
    losses: dict[str, torch.Tensor] = {"loss_ce": torch.tensor(2.0)}
    d6 = apply_stage_d_additive_loss_key(
        {
            "stage_d_attribution": {
                "enabled": True,
                "additive_loss_key": {"enabled": True, "weight": 1.0},
            },
            "stage_d_attribution_coupling": {"applied": False},
        },
        losses,
    )
    assert d6["applied"] is False
    assert d6["skip_reason"] == "d5_coupling_not_applied"
    assert "loss_stage_d_attr" not in losses


def test_stage_d7_real_hook_duplicate_call_is_single_insertion_with_conflict_skip() -> None:
    losses: dict[str, torch.Tensor] = {"loss_ce": torch.tensor(2.0)}
    first = apply_stage_d_additive_loss_key(
        {
            "stage_d_attribution": {
                "enabled": True,
                "additive_loss_key": {"enabled": True, "weight": 1.0},
            },
            "stage_d_attribution_coupling": {"applied": True},
        },
        losses,
    )
    second = apply_stage_d_additive_loss_key(
        {
            "stage_d_attribution": {
                "enabled": True,
                "additive_loss_key": {"enabled": True, "weight": 1.0},
            },
            "stage_d_attribution_coupling": {"applied": True},
        },
        losses,
    )
    assert first["applied"] is True
    assert second["applied"] is False
    assert second["skip_reason"] == "loss_key_conflict"
    assert second["gate_status"]["loss_key_conflict"] is True
    assert list(losses.keys()).count("loss_stage_d_attr") == 1


def _build_stage_d8_enabled_cfg(
    tmp_path: Path,
    *,
    weight: float = 0.0,
    nonzero_semantics_enabled: bool = False,
    nonzero_semantics_mode: str = "constant",
    gradient_coupled_scale: float | None = None,
) -> dict[str, object]:
    artifact_root = tmp_path / "stagec_out_d8"
    _write_stagec_artifacts(artifact_root, num_tracks_scored=2, embedding_dim=256)
    nonzero_semantics_cfg: dict[str, object] = {
        "enabled": nonzero_semantics_enabled,
        "mode": nonzero_semantics_mode,
    }
    if gradient_coupled_scale is not None:
        nonzero_semantics_cfg["gradient_coupled_scale"] = float(gradient_coupled_scale)
    resolved = resolve_stage_d_attribution_plumbing(
        {
            "enabled": True,
            "stagec_artifact_root": str(artifact_root),
            "objective_coupling": {"enabled": True, "weight": 0.0},
            "additive_loss_key": {
                "enabled": True,
                "weight": weight,
                "nonzero_semantics": nonzero_semantics_cfg,
            },
        },
        repo_root=tmp_path,
    )
    resolved = {
        **resolved,
        "objective_coupling": {"enabled": True, "weight": 0.0},
        "additive_loss_key": {
            "enabled": True,
            "weight": weight,
            "nonzero_semantics": nonzero_semantics_cfg,
        },
    }
    runtime = consume_stage_d_attribution_config(resolved)
    boundary = build_stage_d_attribution_consumption_boundary(
        {"stage_d_attribution": resolved, "stage_d_attribution_runtime": runtime}
    )
    coupling = build_stage_d_objective_coupling_decision(
        {
            "stage_d_attribution": resolved,
            "stage_d_attribution_runtime": runtime,
            "stage_d_attribution_consumption": boundary,
        }
    )
    return {
        "stage_d_attribution": resolved,
        "stage_d_attribution_runtime": runtime,
        "stage_d_attribution_consumption": boundary,
        "stage_d_attribution_coupling": coupling,
    }


def test_stage_d8_short_training_smoke_default_off_two_steps_stable() -> None:
    for _ in range(2):
        losses: dict[str, torch.Tensor] = {
            "loss_mask": torch.tensor(1.0),
            "loss_dice": torch.tensor(0.5),
        }
        d6 = apply_stage_d_additive_loss_key(
            {
                "stage_d_attribution": {"enabled": False},
                "stage_d_attribution_coupling": {"applied": False},
            },
            losses,
        )
        # Trainer-style scalar loss reduction and logging conversion.
        reduced = sum(losses.values())
        logged = {name: float(value.item()) for name, value in losses.items()}

        assert d6["applied"] is False
        assert d6["skip_reason"] == "stage_d_attribution_disabled"
        assert "loss_stage_d_attr" not in losses
        assert float(reduced.item()) == pytest.approx(1.5)
        assert logged == {"loss_mask": pytest.approx(1.0), "loss_dice": pytest.approx(0.5)}


def test_stage_d8_short_training_smoke_enabled_weight_zero_two_steps_stable(tmp_path: Path) -> None:
    cfg_dict = _build_stage_d8_enabled_cfg(tmp_path, weight=0.0)
    coupling = cfg_dict["stage_d_attribution_coupling"]
    assert isinstance(coupling, dict)
    assert coupling["applied"] is True

    for _ in range(2):
        losses: dict[str, torch.Tensor] = {
            "loss_mask": torch.tensor(1.0),
            "loss_dice": torch.tensor(0.5),
        }
        d6 = apply_stage_d_additive_loss_key(cfg_dict, losses)
        reduced = sum(losses.values())
        logged = {name: float(value.item()) for name, value in losses.items()}

        assert d6["applied"] is True
        assert d6["skip_reason"] == "none"
        assert d6["gate_status"]["d5_coupling_applied"] is True
        assert d6["diagnostics"]["weight_zero_noop_observed"] is True
        assert list(losses.keys()).count("loss_stage_d_attr") == 1
        assert isinstance(losses["loss_stage_d_attr"], torch.Tensor)
        assert float(losses["loss_stage_d_attr"].item()) == pytest.approx(0.0)
        assert float(reduced.item()) == pytest.approx(1.5)
        assert logged["loss_stage_d_attr"] == pytest.approx(0.0)


def test_stage_d8_nonzero_semantics_integration_lock_reduced_loss_delta_equals_weight(tmp_path: Path) -> None:
    weight = 0.25
    tolerance = 1e-6
    baseline_cfg = _build_stage_d8_enabled_cfg(tmp_path, weight=weight, nonzero_semantics_enabled=False)
    nonzero_cfg = _build_stage_d8_enabled_cfg(tmp_path, weight=weight, nonzero_semantics_enabled=True)

    baseline_losses: dict[str, torch.Tensor] = {
        "loss_mask": torch.tensor(1.0),
        "loss_dice": torch.tensor(0.5),
    }
    nonzero_losses: dict[str, torch.Tensor] = {
        "loss_mask": torch.tensor(1.0),
        "loss_dice": torch.tensor(0.5),
    }
    baseline_d6 = apply_stage_d_additive_loss_key(baseline_cfg, baseline_losses)
    nonzero_d6 = apply_stage_d_additive_loss_key(nonzero_cfg, nonzero_losses)
    baseline_reduced = sum(baseline_losses.values())
    nonzero_reduced = sum(nonzero_losses.values())
    actual_delta = float((nonzero_reduced - baseline_reduced).item())

    assert baseline_d6["applied"] is True
    assert baseline_d6["skip_reason"] == "none"
    assert baseline_d6["gate_status"]["d4_boundary_ready"] is True
    assert baseline_d6["gate_status"]["d5_coupling_applied"] is True
    assert baseline_d6["gate_status"]["nonzero_prereqs_satisfied"] is False
    assert baseline_d6["diagnostics"]["nonzero_semantics_state"] == "disabled"
    assert float(baseline_reduced.item()) == pytest.approx(1.5, abs=tolerance)

    assert nonzero_d6["applied"] is True
    assert nonzero_d6["skip_reason"] == "none"
    assert nonzero_d6["gate_status"]["d4_boundary_ready"] is True
    assert nonzero_d6["gate_status"]["d5_coupling_applied"] is True
    assert nonzero_d6["gate_status"]["nonzero_prereqs_satisfied"] is True
    assert nonzero_d6["diagnostics"]["nonzero_semantics_state"] == "nonzero_applied"
    assert float(nonzero_losses["loss_stage_d_attr"].item()) == pytest.approx(weight, abs=tolerance)

    assert actual_delta == pytest.approx(weight, abs=tolerance)


def test_stage_d8_n6_gradient_coupled_nonzero_pilot_applies_and_backward_step_smoke(tmp_path: Path) -> None:
    cfg_dict = _build_stage_d8_enabled_cfg(
        tmp_path,
        weight=0.25,
        nonzero_semantics_enabled=True,
        nonzero_semantics_mode="gradient_coupled_pilot_v1",
    )
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.1)
    losses: dict[str, torch.Tensor] = {
        "loss_mask": (param - 2.0).pow(2),
        "loss_dice": 0.5 * (param + 1.0).pow(2),
    }

    d6 = apply_stage_d_additive_loss_key(cfg_dict, losses)
    attr_loss = losses["loss_stage_d_attr"]
    expected_attr = 0.25 + 1e-6 * float(losses["loss_mask"].detach().item())
    reduced = sum(losses.values())
    param_before = float(param.detach().item())
    optimizer.zero_grad()
    reduced.backward()
    optimizer.step()
    param_after = float(param.detach().item())

    assert d6["applied"] is True
    assert d6["skip_reason"] == "none"
    assert d6["nonzero_semantics_mode"] == "gradient_coupled_pilot_v1"
    assert d6["gate_status"]["gradient_coupled_mode_requested"] is True
    assert d6["gate_status"]["gradient_coupled_tensor_ready"] is True
    assert d6["planned_loss"]["apply_mode"] == "loss_dict_insert_nonzero_gradient_coupled_pilot"
    assert d6["planned_loss"]["loss_weight"] == pytest.approx(0.25)
    assert d6["planned_loss"]["gradient_coupled_scale"] == pytest.approx(1e-6)
    assert d6["diagnostics"]["nonzero_semantics_state"] == "nonzero_applied"
    assert d6["diagnostics"]["nonzero_skip_reason"] == "none"
    assert d6["diagnostics"]["gradient_coupled_pilot_applied"] is True
    assert d6["diagnostics"]["gradient_coupled_pilot_state"] == "applied"
    assert d6["diagnostics"]["gradient_coupled_pilot_skip_reason"] == "none"
    assert d6["diagnostics"]["gradient_coupled_reference_loss_key"] == "loss_mask"
    assert isinstance(attr_loss, torch.Tensor)
    assert attr_loss.dtype == losses["loss_mask"].dtype
    assert attr_loss.device == losses["loss_mask"].device
    assert attr_loss.requires_grad is True
    assert float(attr_loss.item()) == pytest.approx(expected_attr)
    assert param.grad is not None
    assert abs(param_after - param_before) > 0.0


def test_stage_d8_n6_gradient_coupled_nonzero_pilot_applies_with_small_weight_scale_and_near_zero_reference(
    tmp_path: Path,
) -> None:
    weight = 1e-9
    pilot_scale = 1e-6
    cfg_dict = _build_stage_d8_enabled_cfg(
        tmp_path,
        weight=weight,
        nonzero_semantics_enabled=True,
        nonzero_semantics_mode="gradient_coupled_pilot_v1",
        gradient_coupled_scale=pilot_scale,
    )
    near_zero_reference = 2e-12
    offset = float(torch.tensor(near_zero_reference, dtype=torch.float64).sqrt().item())
    param = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    losses: dict[str, torch.Tensor] = {
        "loss_mask": (param - (1.0 + offset)).pow(2),
        "loss_dice": 0.5 * (param + 1.0).pow(2),
    }

    d6 = apply_stage_d_additive_loss_key(cfg_dict, losses)
    attr_loss = losses["loss_stage_d_attr"]
    reference_value = float(losses["loss_mask"].detach().item())
    expected_attr = weight + pilot_scale * reference_value

    assert d6["applied"] is True
    assert d6["skip_reason"] == "none"
    assert d6["nonzero_semantics_mode"] == "gradient_coupled_pilot_v1"
    assert d6["planned_loss"]["apply_mode"] == "loss_dict_insert_nonzero_gradient_coupled_pilot"
    assert d6["planned_loss"]["loss_weight"] == pytest.approx(weight, abs=1e-18)
    assert d6["planned_loss"]["gradient_coupled_scale"] == pytest.approx(pilot_scale, abs=1e-18)
    assert d6["diagnostics"]["gradient_coupled_pilot_applied"] is True
    assert d6["diagnostics"]["gradient_coupled_pilot_state"] == "applied"
    assert d6["diagnostics"]["gradient_coupled_pilot_skip_reason"] == "none"
    assert d6["diagnostics"]["nonzero_semantics_state"] == "nonzero_applied"
    assert d6["diagnostics"]["nonzero_skip_reason"] == "none"
    assert isinstance(attr_loss, torch.Tensor)
    assert attr_loss.requires_grad is True
    assert float(attr_loss.item()) == pytest.approx(expected_attr, abs=1e-20)
    assert float(attr_loss.item()) > weight


@pytest.mark.parametrize(
    ("reference_value", "expected_direction"),
    [
        pytest.param(2.0, "above_weight", id="positive_scalar_reference"),
        pytest.param(-2.0, "below_weight", id="negative_scalar_reference"),
        pytest.param(1e-13, "near_weight", id="near_zero_positive_scalar_reference"),
        pytest.param(-1e-13, "near_weight", id="near_zero_negative_scalar_reference"),
    ],
)
def test_stage_d8_n6_gradient_coupled_nonzero_pilot_applied_scalar_reference_sign_direction_boundary(
    tmp_path: Path,
    reference_value: float,
    expected_direction: str,
) -> None:
    weight = 0.25
    pilot_scale = 1e-6
    cfg_dict = _build_stage_d8_enabled_cfg(
        tmp_path,
        weight=weight,
        nonzero_semantics_enabled=True,
        nonzero_semantics_mode="gradient_coupled_pilot_v1",
        gradient_coupled_scale=pilot_scale,
    )
    losses: dict[str, torch.Tensor] = {
        "loss_mask": torch.tensor(reference_value, dtype=torch.float64, requires_grad=True),
        "loss_dice": torch.tensor(0.5, dtype=torch.float64, requires_grad=True),
    }

    d6 = apply_stage_d_additive_loss_key(cfg_dict, losses)
    attr_loss = losses["loss_stage_d_attr"]
    expected_attr = weight + pilot_scale * reference_value
    tolerance = 1e-12

    assert d6["applied"] is True
    assert d6["skip_reason"] == "none"
    assert d6["planned_loss"]["apply_mode"] == "loss_dict_insert_nonzero_gradient_coupled_pilot"
    assert d6["planned_loss"]["loss_weight"] == pytest.approx(weight, abs=tolerance)
    assert d6["planned_loss"]["gradient_coupled_scale"] == pytest.approx(pilot_scale, abs=tolerance)
    assert d6["diagnostics"]["gradient_coupled_pilot_applied"] is True
    assert d6["diagnostics"]["gradient_coupled_pilot_state"] == "applied"
    assert d6["diagnostics"]["gradient_coupled_pilot_skip_reason"] == "none"
    assert d6["diagnostics"]["gradient_coupled_reference_loss_key"] == "loss_mask"
    assert d6["diagnostics"]["nonzero_semantics_state"] == "nonzero_applied"
    assert d6["diagnostics"]["nonzero_skip_reason"] == "none"
    assert isinstance(attr_loss, torch.Tensor)
    assert attr_loss.requires_grad is True
    assert float(attr_loss.item()) == pytest.approx(expected_attr, abs=tolerance)
    if expected_direction == "above_weight":
        assert float(attr_loss.item()) > weight
    elif expected_direction == "below_weight":
        assert float(attr_loss.item()) < weight
    else:
        assert float(attr_loss.item()) == pytest.approx(weight, abs=tolerance)


@pytest.mark.parametrize(
    ("reference_tensor_values", "expected_direction"),
    [
        pytest.param([1.5, 0.5], "above_weight", id="positive_tensor_sum_reference"),
        pytest.param([0.2, -1.2], "below_weight", id="negative_tensor_sum_reference"),
        pytest.param([1e-12, -5e-13], "near_weight", id="near_zero_tensor_sum_reference"),
    ],
)
def test_stage_d8_n6_gradient_coupled_nonzero_pilot_applied_tensor_reference_sign_direction_boundary(
    tmp_path: Path,
    reference_tensor_values: list[float],
    expected_direction: str,
) -> None:
    weight = 0.1
    pilot_scale = 1e-3
    cfg_dict = _build_stage_d8_enabled_cfg(
        tmp_path,
        weight=weight,
        nonzero_semantics_enabled=True,
        nonzero_semantics_mode="gradient_coupled_pilot_v1",
        gradient_coupled_scale=pilot_scale,
    )
    reference_tensor = torch.tensor(reference_tensor_values, dtype=torch.float64, requires_grad=True)
    losses: dict[str, torch.Tensor] = {
        "loss_mask": reference_tensor,
        "loss_dice": torch.tensor(0.5, dtype=torch.float64, requires_grad=True),
    }

    d6 = apply_stage_d_additive_loss_key(cfg_dict, losses)
    attr_loss = losses["loss_stage_d_attr"]
    reference_sum = float(reference_tensor.detach().sum().item())
    expected_attr = weight + pilot_scale * reference_sum
    tolerance = 1e-12

    assert d6["applied"] is True
    assert d6["skip_reason"] == "none"
    assert d6["planned_loss"]["apply_mode"] == "loss_dict_insert_nonzero_gradient_coupled_pilot"
    assert d6["diagnostics"]["gradient_coupled_pilot_applied"] is True
    assert d6["diagnostics"]["gradient_coupled_pilot_state"] == "applied"
    assert d6["diagnostics"]["gradient_coupled_reference_loss_key"] == "loss_mask"
    assert d6["diagnostics"]["nonzero_semantics_state"] == "nonzero_applied"
    assert isinstance(attr_loss, torch.Tensor)
    assert attr_loss.requires_grad is True
    assert float(attr_loss.item()) == pytest.approx(expected_attr, abs=tolerance)
    if expected_direction == "above_weight":
        assert float(attr_loss.item()) > weight
    elif expected_direction == "below_weight":
        assert float(attr_loss.item()) < weight
    else:
        assert float(attr_loss.item()) == pytest.approx(weight, abs=tolerance)


def test_stage_d6_n6_gradient_coupled_mode_skip_closed_when_scale_zero_with_weight_positive(tmp_path: Path) -> None:
    cfg_dict = _build_stage_d8_enabled_cfg(
        tmp_path,
        weight=0.25,
        nonzero_semantics_enabled=True,
        nonzero_semantics_mode="gradient_coupled_pilot_v1",
        gradient_coupled_scale=0.0,
    )
    losses: dict[str, torch.Tensor] = {
        "loss_mask": torch.tensor(1.0, requires_grad=True),
        "loss_dice": torch.tensor(0.5, requires_grad=True),
    }
    d6 = apply_stage_d_additive_loss_key(cfg_dict, losses)

    assert d6["applied"] is False
    assert d6["skip_reason"] == "invalid_gradient_coupled_scale"
    assert d6["planned_loss"]["gradient_coupled_scale"] == pytest.approx(0.0)
    assert d6["diagnostics"]["nonzero_semantics_state"] == "skipped"
    assert d6["diagnostics"]["nonzero_skip_reason"] == "invalid_gradient_coupled_scale"
    assert d6["diagnostics"]["gradient_coupled_pilot_state"] == "skipped"
    assert d6["diagnostics"]["gradient_coupled_pilot_skip_reason"] == "invalid_gradient_coupled_scale"
    assert "loss_stage_d_attr" not in losses


def test_stage_d6_n6_gradient_coupled_mode_skip_closed_when_reference_tensor_not_grad_ready(tmp_path: Path) -> None:
    cfg_dict = _build_stage_d8_enabled_cfg(
        tmp_path,
        weight=0.25,
        nonzero_semantics_enabled=True,
        nonzero_semantics_mode="gradient_coupled_pilot_v1",
    )
    losses: dict[str, torch.Tensor] = {"loss_mask": torch.tensor(1.0)}
    d6 = apply_stage_d_additive_loss_key(cfg_dict, losses)

    assert d6["applied"] is True
    assert d6["skip_reason"] == "none"
    assert d6["gate_status"]["nonzero_prereqs_satisfied"] is True
    assert d6["gate_status"]["gradient_coupled_tensor_ready"] is False
    assert d6["planned_loss"]["apply_mode"] == "loss_dict_insert_zero"
    assert d6["diagnostics"]["nonzero_semantics_state"] == "skipped"
    assert d6["diagnostics"]["nonzero_skip_reason"] == "gradient_coupled_reference_unavailable"
    assert d6["diagnostics"]["gradient_coupled_pilot_applied"] is False
    assert d6["diagnostics"]["gradient_coupled_pilot_state"] == "skipped"
    assert d6["diagnostics"]["gradient_coupled_pilot_skip_reason"] == "gradient_coupled_reference_unavailable"
    assert float(losses["loss_stage_d_attr"].item()) == pytest.approx(0.0)
