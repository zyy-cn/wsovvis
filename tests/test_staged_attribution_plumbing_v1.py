from __future__ import annotations

import json
from pathlib import Path

import pytest

from wsovvis.training import (
    StageDAttributionPlumbingError,
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
