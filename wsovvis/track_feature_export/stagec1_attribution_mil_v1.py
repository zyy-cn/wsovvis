from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .stagec_loader_v1 import StageCTrackRecord, load_stageb_export_split_v1
from .v1_core import ALL_STATUSES, PROCESSED_STATUSES, ExportContractError


class StageC1AttributionError(ExportContractError):
    """Raised when Stage C1 offline MIL baseline scoring fails."""


@dataclass(frozen=True)
class StageC1MilConfig:
    embedding_abs_mean_weight: float = 1.0
    objectness_weight: float = 1.0
    length_log_weight: float = 0.25
    top_k_per_video: int = 3
    supported_video_statuses: Tuple[str, ...] = tuple(sorted(PROCESSED_STATUSES))


@dataclass(frozen=True)
class StageC1TrackScoreRecord:
    video_id: str
    track_id: str | int
    row_index: int
    status: str
    score: float


@dataclass(frozen=True)
class StageC1MilResult:
    track_scores: Tuple[StageC1TrackScoreRecord, ...]
    per_video_summary: Tuple[Dict[str, Any], ...]
    run_summary: Dict[str, Any]


def _err(field_path: str, rule_summary: str) -> StageC1AttributionError:
    return StageC1AttributionError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path=field_path, rule_summary=rule_summary)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_config(config: StageC1MilConfig) -> None:
    _require(_is_number(config.embedding_abs_mean_weight), "config.embedding_abs_mean_weight", "must be numeric")
    _require(_is_number(config.objectness_weight), "config.objectness_weight", "must be numeric")
    _require(_is_number(config.length_log_weight), "config.length_log_weight", "must be numeric")
    _require(
        isinstance(config.top_k_per_video, int) and config.top_k_per_video > 0,
        "config.top_k_per_video",
        "must be integer > 0",
    )
    _require(
        isinstance(config.supported_video_statuses, tuple) and len(config.supported_video_statuses) > 0,
        "config.supported_video_statuses",
        "must be a non-empty tuple",
    )
    unknown = set(config.supported_video_statuses) - ALL_STATUSES
    _require(
        not unknown,
        "config.supported_video_statuses",
        f"contains unknown statuses: {sorted(unknown)}",
    )
    unsupported_for_scoring = set(config.supported_video_statuses) - PROCESSED_STATUSES
    _require(
        not unsupported_for_scoring,
        "config.supported_video_statuses",
        f"contains non-processed statuses unsupported by MIL scoring: {sorted(unsupported_for_scoring)}",
    )


def _compute_track_score(track: StageCTrackRecord, config: StageC1MilConfig) -> float:
    embedding = track.embedding
    _require(isinstance(embedding, np.ndarray), "track.embedding", "must be numpy ndarray")
    _require(embedding.ndim == 1, "track.embedding", "must be rank-1")
    _require(np.isfinite(embedding).all(), "track.embedding", "must contain only finite values")

    metadata = track.metadata
    _require(hasattr(metadata, "objectness_score"), "track.metadata.objectness_score", "required field missing")
    _require(hasattr(metadata, "num_active_frames"), "track.metadata.num_active_frames", "required field missing")
    _require(_is_number(metadata.objectness_score), "track.metadata.objectness_score", "must be numeric")
    _require(
        isinstance(metadata.num_active_frames, int) and metadata.num_active_frames > 0,
        "track.metadata.num_active_frames",
        "must be integer > 0",
    )

    embedding_abs_mean = float(np.mean(np.abs(embedding), dtype=np.float64))
    length_log_term = math.log1p(metadata.num_active_frames)
    score = (
        float(config.embedding_abs_mean_weight) * embedding_abs_mean
        + float(config.objectness_weight) * float(metadata.objectness_score)
        + float(config.length_log_weight) * length_log_term
    )
    _require(math.isfinite(score), "track.score", "must be finite")
    return float(score)


def compute_stagec1_mil_baseline_scores(
    split_view: Any,
    *,
    config: StageC1MilConfig | None = None,
) -> StageC1MilResult:
    config = config or StageC1MilConfig()
    _validate_config(config)

    track_scores: list[StageC1TrackScoreRecord] = []
    seen_keys: set[tuple[str, str | int]] = set()
    expected_track_keys: list[tuple[str, str | int]] = []

    for video in split_view.iter_videos():
        _require(hasattr(video, "video_id") and isinstance(video.video_id, str) and video.video_id, "video.video_id", "must be non-empty string")
        _require(hasattr(video, "status"), "video.status", "required field missing")
        status = video.status
        _require(status in ALL_STATUSES, "video.status", f"must be one of {sorted(ALL_STATUSES)}")

        if status in PROCESSED_STATUSES and status not in config.supported_video_statuses:
            raise _err(
                "video.status",
                (
                    f"unsupported processed status '{status}' for video '{video.video_id}' under "
                    f"config.supported_video_statuses={list(config.supported_video_statuses)}"
                ),
            )

        if status != "processed_with_tracks":
            continue

        for track in split_view.iter_tracks(video.video_id):
            metadata = track.metadata
            _require(hasattr(metadata, "track_id"), "track.metadata.track_id", "required field missing")
            _require(hasattr(metadata, "row_index"), "track.metadata.row_index", "required field missing")
            _require(hasattr(metadata, "video_id"), "track.metadata.video_id", "required field missing")
            _require(
                (isinstance(metadata.track_id, str) and metadata.track_id) or isinstance(metadata.track_id, int),
                "track.metadata.track_id",
                "must be non-empty string or integer",
            )
            _require(
                isinstance(metadata.row_index, int) and metadata.row_index >= 0,
                "track.metadata.row_index",
                "must be integer >= 0",
            )
            _require(
                metadata.video_id == video.video_id,
                "track.metadata.video_id",
                f"must match parent video_id '{video.video_id}'",
            )

            key = (video.video_id, metadata.track_id)
            expected_track_keys.append(key)
            _require(key not in seen_keys, "track.identity", f"duplicate (video_id, track_id) key {key}")
            seen_keys.add(key)

            score = _compute_track_score(track, config)
            track_scores.append(
                StageC1TrackScoreRecord(
                    video_id=video.video_id,
                    track_id=metadata.track_id,
                    row_index=metadata.row_index,
                    status=status,
                    score=score,
                )
            )

    emitted_keys = [(record.video_id, record.track_id) for record in track_scores]
    _require(
        emitted_keys == expected_track_keys,
        "result.track_scores",
        "emitted key order mismatch against Stage C0 traversal order",
    )

    per_video_summary = _build_per_video_summary(track_scores, top_k_per_video=config.top_k_per_video)
    run_summary = _build_run_summary(split_view=split_view, track_scores=track_scores)

    return StageC1MilResult(
        track_scores=tuple(track_scores),
        per_video_summary=tuple(per_video_summary),
        run_summary=run_summary,
    )


def _build_per_video_summary(
    track_scores: Sequence[StageC1TrackScoreRecord],
    *,
    top_k_per_video: int,
) -> list[Dict[str, Any]]:
    by_video: Dict[str, list[StageC1TrackScoreRecord]] = {}
    for record in track_scores:
        by_video.setdefault(record.video_id, []).append(record)

    summaries: list[Dict[str, Any]] = []
    for video_id in sorted(by_video.keys()):
        records = by_video[video_id]
        values = [r.score for r in records]
        top_k = sorted(records, key=lambda r: r.score, reverse=True)[:top_k_per_video]
        summaries.append(
            {
                "video_id": video_id,
                "num_tracks": len(records),
                "score_min": float(min(values)),
                "score_max": float(max(values)),
                "score_mean": float(sum(values) / len(values)),
                "top_k": [
                    {
                        "track_id": rec.track_id,
                        "row_index": rec.row_index,
                        "score": rec.score,
                    }
                    for rec in top_k
                ],
            }
        )
    return summaries


def _build_run_summary(
    *,
    split_view: Any,
    track_scores: Sequence[StageC1TrackScoreRecord],
) -> Dict[str, Any]:
    scores = [rec.score for rec in track_scores]
    all_videos = list(split_view.iter_videos())
    return {
        "split": getattr(split_view, "split", "unknown"),
        "embedding_dim": int(getattr(split_view, "embedding_dim", -1)),
        "num_videos_total": len(all_videos),
        "num_tracks_scored": len(track_scores),
        "score_min": float(min(scores)) if scores else None,
        "score_max": float(max(scores)) if scores else None,
        "score_mean": float(sum(scores) / len(scores)) if scores else None,
    }


def write_stagec1_mil_artifacts(result: StageC1MilResult, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    track_scores_path = output_dir / "track_scores.jsonl"
    with track_scores_path.open("w", encoding="utf-8") as f:
        for record in result.track_scores:
            f.write(
                json.dumps(
                    {
                        "video_id": record.video_id,
                        "track_id": record.track_id,
                        "row_index": record.row_index,
                        "status": record.status,
                        "score": record.score,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    per_video_summary_path = output_dir / "per_video_summary.json"
    per_video_summary_path.write_text(json.dumps(list(result.per_video_summary), indent=2, sort_keys=True), encoding="utf-8")

    run_summary_path = output_dir / "run_summary.json"
    run_summary_path.write_text(json.dumps(result.run_summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "track_scores": track_scores_path,
        "per_video_summary": per_video_summary_path,
        "run_summary": run_summary_path,
    }


def run_stagec1_mil_baseline_offline(
    *,
    split_root: Path,
    output_dir: Path,
    config: StageC1MilConfig | None = None,
    eager_validate: bool = True,
) -> Dict[str, Any]:
    split_view = load_stageb_export_split_v1(split_root=split_root, eager_validate=eager_validate)
    result = compute_stagec1_mil_baseline_scores(split_view, config=config)
    artifact_paths = write_stagec1_mil_artifacts(result=result, output_dir=output_dir)
    return {
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "run_summary": result.run_summary,
    }
