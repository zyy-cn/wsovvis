from __future__ import annotations

import json
from pathlib import Path

import pytest

from wsovvis.training import (
    StageDAttributionPlumbingError,
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
    assert consumed["summary"]["embedding_dim"] == 256
    assert consumed["track_score_rows_validated"] == 2
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
