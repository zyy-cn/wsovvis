from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class StageDAttributionPlumbingError(ValueError):
    """Raised when Stage D attribution plumbing config/artifacts are invalid."""


@dataclass(frozen=True)
class StageDAttributionPlumbingConfig:
    enabled: bool = False
    stagec_artifact_root: str | None = None
    stagec_run_summary_path: str | None = None
    stagec_track_scores_path: str | None = None
    required_scorer_backend: str | None = None
    expected_embedding_dim: int | None = None


def _err(field_path: str, rule_summary: str) -> StageDAttributionPlumbingError:
    return StageDAttributionPlumbingError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path, rule_summary)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_config(raw: Mapping[str, Any] | None) -> StageDAttributionPlumbingConfig:
    if raw is None:
        return StageDAttributionPlumbingConfig()
    _require(isinstance(raw, Mapping), "stage_d_attribution", "must be an object")
    enabled = raw.get("enabled", False)
    _require(isinstance(enabled, bool), "stage_d_attribution.enabled", "must be boolean")
    expected_embedding_dim = raw.get("expected_embedding_dim", None)
    if expected_embedding_dim is not None:
        _require(
            isinstance(expected_embedding_dim, int) and expected_embedding_dim > 0,
            "stage_d_attribution.expected_embedding_dim",
            "must be integer > 0 when provided",
        )
    required_scorer_backend = raw.get("required_scorer_backend", None)
    if required_scorer_backend is not None:
        _require(
            isinstance(required_scorer_backend, str) and required_scorer_backend,
            "stage_d_attribution.required_scorer_backend",
            "must be non-empty string when provided",
        )
    return StageDAttributionPlumbingConfig(
        enabled=enabled,
        stagec_artifact_root=raw.get("stagec_artifact_root"),
        stagec_run_summary_path=raw.get("stagec_run_summary_path"),
        stagec_track_scores_path=raw.get("stagec_track_scores_path"),
        required_scorer_backend=required_scorer_backend,
        expected_embedding_dim=expected_embedding_dim,
    )


def _resolve_path(path_value: str, field_path: str, *, repo_root: Path) -> Path:
    _require(isinstance(path_value, str) and path_value, field_path, "must be non-empty string path")
    path = Path(path_value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _load_json(path: Path, *, field_path: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _err(field_path, f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise _err(field_path, f"invalid JSON: {exc}") from exc
    _require(isinstance(payload, dict), field_path, "top-level value must be object")
    return payload


def _validate_run_summary(payload: Mapping[str, Any], *, config: StageDAttributionPlumbingConfig) -> dict[str, Any]:
    scorer_backend = payload.get("scorer_backend")
    _require(isinstance(scorer_backend, str) and scorer_backend, "stagec_run_summary.scorer_backend", "required non-empty string")

    split = payload.get("split")
    _require(isinstance(split, str) and split, "stagec_run_summary.split", "required non-empty string")

    embedding_dim = payload.get("embedding_dim")
    _require(
        isinstance(embedding_dim, int) and embedding_dim > 0,
        "stagec_run_summary.embedding_dim",
        "required integer > 0",
    )

    num_tracks_scored = payload.get("num_tracks_scored")
    _require(
        isinstance(num_tracks_scored, int) and num_tracks_scored >= 0,
        "stagec_run_summary.num_tracks_scored",
        "required integer >= 0",
    )

    if config.required_scorer_backend is not None:
        _require(
            scorer_backend == config.required_scorer_backend,
            "stagec_run_summary.scorer_backend",
            (
                f"expected '{config.required_scorer_backend}' from "
                "stage_d_attribution.required_scorer_backend"
            ),
        )
    if config.expected_embedding_dim is not None:
        _require(
            int(embedding_dim) == int(config.expected_embedding_dim),
            "stagec_run_summary.embedding_dim",
            (
                f"dimension mismatch: artifact={int(embedding_dim)} "
                f"expected={int(config.expected_embedding_dim)}"
            ),
        )
    return {
        "scorer_backend": scorer_backend,
        "split": split,
        "embedding_dim": int(embedding_dim),
        "num_tracks_scored": int(num_tracks_scored),
    }


def _validate_track_scores(path: Path, *, expected_count: int) -> int:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise _err("stage_d_attribution.stagec_track_scores_path", f"missing file: {path}") from exc

    count = 0
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _err("stagec_track_scores", f"invalid JSONL at line {idx + 1}: {exc}") from exc
        _require(isinstance(row, dict), f"stagec_track_scores[{idx + 1}]", "must be object")
        _require(
            isinstance(row.get("video_id"), str) and row["video_id"],
            f"stagec_track_scores[{idx + 1}].video_id",
            "required non-empty string",
        )
        track_id = row.get("track_id")
        _require(
            (isinstance(track_id, str) and track_id) or isinstance(track_id, int),
            f"stagec_track_scores[{idx + 1}].track_id",
            "required non-empty string or integer",
        )
        _require(
            isinstance(row.get("row_index"), int) and row["row_index"] >= 0,
            f"stagec_track_scores[{idx + 1}].row_index",
            "required integer >= 0",
        )
        _require(
            isinstance(row.get("status"), str) and row["status"],
            f"stagec_track_scores[{idx + 1}].status",
            "required non-empty string",
        )
        _require(
            _is_number(row.get("score")) and math.isfinite(float(row["score"])),
            f"stagec_track_scores[{idx + 1}].score",
            "required finite numeric value",
        )
        count += 1
    _require(
        count == expected_count,
        "stagec_track_scores",
        f"row-count mismatch: {count} != stagec_run_summary.num_tracks_scored ({expected_count})",
    )
    return count


def resolve_stage_d_attribution_plumbing(raw: Mapping[str, Any] | None, *, repo_root: Path) -> dict[str, Any]:
    """Resolve and validate Stage C attribution artifact references for Stage D plumbing."""

    config = _parse_config(raw)
    if not config.enabled:
        return {"enabled": False}

    if config.stagec_run_summary_path is not None:
        run_summary_path = _resolve_path(
            config.stagec_run_summary_path,
            "stage_d_attribution.stagec_run_summary_path",
            repo_root=repo_root,
        )
    else:
        _require(
            isinstance(config.stagec_artifact_root, str) and config.stagec_artifact_root,
            "stage_d_attribution.stagec_artifact_root",
            "required when enabled and explicit stagec_run_summary_path is not provided",
        )
        artifact_root = _resolve_path(
            config.stagec_artifact_root,
            "stage_d_attribution.stagec_artifact_root",
            repo_root=repo_root,
        )
        run_summary_path = artifact_root / "run_summary.json"

    if config.stagec_track_scores_path is not None:
        track_scores_path = _resolve_path(
            config.stagec_track_scores_path,
            "stage_d_attribution.stagec_track_scores_path",
            repo_root=repo_root,
        )
    else:
        _require(
            isinstance(config.stagec_artifact_root, str) and config.stagec_artifact_root,
            "stage_d_attribution.stagec_artifact_root",
            "required when enabled and explicit stagec_track_scores_path is not provided",
        )
        artifact_root = _resolve_path(
            config.stagec_artifact_root,
            "stage_d_attribution.stagec_artifact_root",
            repo_root=repo_root,
        )
        track_scores_path = artifact_root / "track_scores.jsonl"

    run_summary = _load_json(run_summary_path, field_path="stage_d_attribution.stagec_run_summary_path")
    summary_meta = _validate_run_summary(run_summary, config=config)
    track_score_rows = _validate_track_scores(track_scores_path, expected_count=summary_meta["num_tracks_scored"])

    return {
        "enabled": True,
        "stagec_run_summary_path": str(run_summary_path),
        "stagec_track_scores_path": str(track_scores_path),
        "summary": summary_meta,
        "track_score_rows_validated": int(track_score_rows),
    }
