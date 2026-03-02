from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from .stagec_loader_v1 import StageCTrackRecord, load_stageb_export_split_v1
from .v1_core import ALL_STATUSES, PROCESSED_STATUSES, ExportContractError

SCORER_BACKENDS = {"mil_v1", "labelset_proto_v1", "em_v1", "sinkhorn_v1"}
DECODER_BACKENDS = {"independent", "coverage_greedy_v1", "otlite_v1"}
EMPTY_LABELSET_POLICIES = {"use_all_prototypes", "error"}
PROTOTYPE_SCHEMA_NAME_V1 = "wsovvis.stagec.label_prototypes.v1"
SINKHORN_SPECIAL_BG_LABEL_ID = "__bg__"
SINKHORN_SPECIAL_UNK_FG_LABEL_ID = "__unk_fg__"
SINKHORN_C43_PREDICTED_LABEL_SOURCES = {"observed", "bg", "unk_fg"}


class StageC1AttributionError(ExportContractError):
    """Raised when Stage C offline baseline scoring fails."""


@dataclass(frozen=True)
class StageC1MilConfig:
    embedding_abs_mean_weight: float = 1.0
    objectness_weight: float = 1.0
    length_log_weight: float = 0.25
    top_k_per_video: int = 3
    supported_video_statuses: Tuple[str, ...] = tuple(sorted(PROCESSED_STATUSES))


@dataclass(frozen=True)
class StageC1DecoderConfig:
    backend: str = "independent"
    fg_score_min: float = -1.0
    bg_score_threshold: float | None = None
    bg_min_margin: float | None = None
    otlite_temperature: float = 0.10
    otlite_iters: int = 8
    otlite_eps: float = 1e-12
    otlite_ot_prob_min: float | None = None


@dataclass(frozen=True)
class StageC1EmConfig:
    temperature: float = 0.10
    iterations: int = 5
    prior_alpha: float = 1.0
    eps: float = 1e-12


@dataclass(frozen=True)
class StageC1SinkhornConfig:
    temperature: float = 0.10
    iterations: int = 12
    tolerance: float = 1e-6
    eps: float = 1e-12


@dataclass(frozen=True)
class StageC1SinkhornC43Config:
    enable: bool = False
    enable_bg: bool = False
    enable_unk_fg: bool = False
    bg_prior_weight: float = 0.0
    unk_fg_prior_weight: float = 0.0
    unk_fg_min_top_obs_score: float | None = None
    unk_fg_max_top_obs_score: float | None = None
    bg_score: float = 0.0


@dataclass(frozen=True)
class StageC1TrackScoreRecord:
    video_id: str
    track_id: str | int
    row_index: int
    status: str
    score: float
    predicted_label_id: str | int | None = None
    used_fallback_label_pool: bool = False
    decoder_predicted_label_id: str | int | None = None
    decoder_assigned_bg: bool | None = None
    decoder_assignment_source: str | None = None
    decoder_score: float | None = None
    decoder_margin: float | None = None
    decoder_bg_reason: str | None = None
    decoder_ot_prob: float | None = None
    predicted_label_source: str | None = None
    sinkhorn_active_special_columns: Tuple[str, ...] | None = None
    sinkhorn_bg_posterior: float | None = None
    sinkhorn_top_observed_score: float | None = None


@dataclass(frozen=True)
class StageC1MilResult:
    track_scores: Tuple[StageC1TrackScoreRecord, ...]
    per_video_summary: Tuple[Dict[str, Any], ...]
    run_summary: Dict[str, Any]


@dataclass(frozen=True)
class StageCLabelPrototypeInventory:
    schema_name: str
    schema_version: str
    embedding_dim: int
    dtype: str
    array_key: str
    labels_by_key: Mapping[tuple[int, int | str], int | str]
    row_index_by_key: Mapping[tuple[int, int | str], int]
    prototypes: np.ndarray


def _err(field_path: str, rule_summary: str) -> StageC1AttributionError:
    return StageC1AttributionError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path=field_path, rule_summary=rule_summary)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_label_id(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and value != "")


def _canonical_label_key(label_id: int | str) -> tuple[int, int | str]:
    if isinstance(label_id, int) and not isinstance(label_id, bool):
        return (0, int(label_id))
    if isinstance(label_id, str) and label_id:
        return (1, label_id)
    raise _err("label_id", "must be non-empty string or integer")


def _canonical_label_key_text(label_id: int | str) -> str:
    key = _canonical_label_key(label_id)
    return f"{key[0]}:{key[1]}"


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


def _validate_decoder_config(config: StageC1DecoderConfig) -> None:
    _require(config.backend in DECODER_BACKENDS, "decoder.backend", f"must be one of {sorted(DECODER_BACKENDS)}")
    _require(_is_number(config.fg_score_min), "decoder.fg_score_min", "must be numeric")
    _require(
        config.bg_score_threshold is None or _is_number(config.bg_score_threshold),
        "decoder.bg_score_threshold",
        "must be numeric or null",
    )
    _require(
        config.bg_min_margin is None or _is_number(config.bg_min_margin),
        "decoder.bg_min_margin",
        "must be numeric or null",
    )
    _require(_is_number(config.otlite_temperature), "decoder.otlite_temperature", "must be numeric")
    _require(float(config.otlite_temperature) > 0.0, "decoder.otlite_temperature", "must be > 0")
    _require(isinstance(config.otlite_iters, int), "decoder.otlite_iters", "must be integer >= 1")
    _require(int(config.otlite_iters) >= 1, "decoder.otlite_iters", "must be integer >= 1")
    _require(_is_number(config.otlite_eps), "decoder.otlite_eps", "must be numeric")
    _require(float(config.otlite_eps) > 0.0, "decoder.otlite_eps", "must be > 0")
    _require(
        config.otlite_ot_prob_min is None or _is_number(config.otlite_ot_prob_min),
        "decoder.otlite_ot_prob_min",
        "must be numeric or null",
    )
    if config.otlite_ot_prob_min is not None:
        _require(
            0.0 <= float(config.otlite_ot_prob_min) <= 1.0,
            "decoder.otlite_ot_prob_min",
            "must be in [0,1] when provided",
        )


def _validate_em_config(config: StageC1EmConfig) -> None:
    _require(_is_number(config.temperature), "em.temperature", "must be numeric")
    _require(float(config.temperature) > 0.0, "em.temperature", "must be > 0")
    _require(isinstance(config.iterations, int), "em.iterations", "must be integer >= 1")
    _require(int(config.iterations) >= 1, "em.iterations", "must be integer >= 1")
    _require(_is_number(config.prior_alpha), "em.prior_alpha", "must be numeric")
    _require(float(config.prior_alpha) >= 0.0, "em.prior_alpha", "must be >= 0")
    _require(_is_number(config.eps), "em.eps", "must be numeric")
    _require(float(config.eps) > 0.0, "em.eps", "must be > 0")


def _validate_sinkhorn_config(config: StageC1SinkhornConfig) -> None:
    _require(_is_number(config.temperature), "sinkhorn.temperature", "must be numeric")
    _require(float(config.temperature) > 0.0, "sinkhorn.temperature", "must be > 0")
    _require(isinstance(config.iterations, int), "sinkhorn.iterations", "must be integer >= 1")
    _require(int(config.iterations) >= 1, "sinkhorn.iterations", "must be integer >= 1")
    _require(_is_number(config.tolerance), "sinkhorn.tolerance", "must be numeric")
    _require(float(config.tolerance) >= 0.0, "sinkhorn.tolerance", "must be >= 0")
    _require(_is_number(config.eps), "sinkhorn.eps", "must be numeric")
    _require(float(config.eps) > 0.0, "sinkhorn.eps", "must be > 0")


def _validate_sinkhorn_c43_config(config: StageC1SinkhornC43Config) -> None:
    _require(isinstance(config.enable, bool), "sinkhorn.c43.enable", "must be boolean")
    _require(isinstance(config.enable_bg, bool), "sinkhorn.c43.enable_bg", "must be boolean")
    _require(isinstance(config.enable_unk_fg, bool), "sinkhorn.c43.enable_unk_fg", "must be boolean")
    _require(_is_number(config.bg_prior_weight), "sinkhorn.c43.bg_prior_weight", "must be numeric")
    _require(float(config.bg_prior_weight) >= 0.0, "sinkhorn.c43.bg_prior_weight", "must be >= 0")
    _require(_is_number(config.unk_fg_prior_weight), "sinkhorn.c43.unk_fg_prior_weight", "must be numeric")
    _require(float(config.unk_fg_prior_weight) >= 0.0, "sinkhorn.c43.unk_fg_prior_weight", "must be >= 0")
    _require(
        config.unk_fg_min_top_obs_score is None or _is_number(config.unk_fg_min_top_obs_score),
        "sinkhorn.c43.unk_fg_min_top_obs_score",
        "must be numeric or null",
    )
    _require(
        config.unk_fg_max_top_obs_score is None or _is_number(config.unk_fg_max_top_obs_score),
        "sinkhorn.c43.unk_fg_max_top_obs_score",
        "must be numeric or null",
    )
    _require(_is_number(config.bg_score), "sinkhorn.c43.bg_score", "must be numeric")
    _require(math.isfinite(float(config.bg_score)), "sinkhorn.c43.bg_score", "must be finite")
    if config.unk_fg_min_top_obs_score is not None and config.unk_fg_max_top_obs_score is not None:
        _require(
            float(config.unk_fg_min_top_obs_score) <= float(config.unk_fg_max_top_obs_score),
            "sinkhorn.c43.unk_fg_min_top_obs_score",
            "must be <= sinkhorn.c43.unk_fg_max_top_obs_score",
        )

    if not config.enable:
        _require(not config.enable_bg, "sinkhorn.c43.enable_bg", "requires sinkhorn.c43.enable=true")
        _require(not config.enable_unk_fg, "sinkhorn.c43.enable_unk_fg", "requires sinkhorn.c43.enable=true")
        _require(float(config.bg_prior_weight) == 0.0, "sinkhorn.c43.bg_prior_weight", "must be 0 when disabled")
        _require(float(config.unk_fg_prior_weight) == 0.0, "sinkhorn.c43.unk_fg_prior_weight", "must be 0 when disabled")
        _require(
            config.unk_fg_min_top_obs_score is None,
            "sinkhorn.c43.unk_fg_min_top_obs_score",
            "must be null when disabled",
        )
        _require(
            config.unk_fg_max_top_obs_score is None,
            "sinkhorn.c43.unk_fg_max_top_obs_score",
            "must be null when disabled",
        )
    else:
        _require(config.enable_bg, "sinkhorn.c43.enable_bg", "C4.3-A requires enable_bg=true when sinkhorn.c43.enable=true")
        _require(
            float(config.bg_prior_weight) > 0.0,
            "sinkhorn.c43.bg_prior_weight",
            "must be > 0 when sinkhorn.c43.enable_bg=true",
        )
        _require(not config.enable_unk_fg, "sinkhorn.c43.enable_unk_fg", "C4.3-A defers unk-fg to C4.3-B")
        _require(
            float(config.unk_fg_prior_weight) == 0.0,
            "sinkhorn.c43.unk_fg_prior_weight",
            "C4.3-A defers unk-fg to C4.3-B; expected 0",
        )
        _require(
            config.unk_fg_min_top_obs_score is None,
            "sinkhorn.c43.unk_fg_min_top_obs_score",
            "C4.3-A defers unk-fg gating to C4.3-B; expected null",
        )
        _require(
            config.unk_fg_max_top_obs_score is None,
            "sinkhorn.c43.unk_fg_max_top_obs_score",
            "C4.3-A defers unk-fg gating to C4.3-B; expected null",
        )


def _validate_no_sinkhorn_special_label_collisions(inventory: StageCLabelPrototypeInventory) -> None:
    reserved_keys = {
        _canonical_label_key(SINKHORN_SPECIAL_BG_LABEL_ID),
        _canonical_label_key(SINKHORN_SPECIAL_UNK_FG_LABEL_ID),
    }
    collisions = sorted(key for key in inventory.labels_by_key.keys() if key in reserved_keys)
    _require(
        not collisions,
        "prototype_manifest.labels",
        f"reserved sinkhorn special label IDs are forbidden: {sorted([SINKHORN_SPECIAL_BG_LABEL_ID, SINKHORN_SPECIAL_UNK_FG_LABEL_ID])}",
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


def _as_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "all_finite": True,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "all_finite": bool(np.isfinite(arr).all()),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
    }


def _validate_track_identity(track: StageCTrackRecord, video_id: str) -> tuple[str | int, int]:
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
        metadata.video_id == video_id,
        "track.metadata.video_id",
        f"must match parent video_id '{video_id}'",
    )
    return metadata.track_id, metadata.row_index


def _iter_processed_with_tracks(split_view: Any, config: StageC1MilConfig) -> tuple[list[Any], list[str], list[tuple[str, str | int]]]:
    all_videos = list(split_view.iter_videos())
    processed_with_tracks_video_ids: list[str] = []
    expected_track_keys: list[tuple[str, str | int]] = []

    for video in all_videos:
        _require(
            hasattr(video, "video_id") and isinstance(video.video_id, str) and video.video_id,
            "video.video_id",
            "must be non-empty string",
        )
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

        processed_with_tracks_video_ids.append(video.video_id)
        for track in split_view.iter_tracks(video.video_id):
            track_id, _ = _validate_track_identity(track, video.video_id)
            expected_track_keys.append((video.video_id, track_id))

    return all_videos, processed_with_tracks_video_ids, expected_track_keys


def _validate_emitted_order_and_non_empty(
    *,
    track_scores: Sequence[StageC1TrackScoreRecord],
    expected_track_keys: Sequence[tuple[str, str | int]],
    processed_with_tracks_video_ids: Sequence[str],
) -> list[tuple[str, str | int]]:
    emitted_keys = [(record.video_id, record.track_id) for record in track_scores]
    _require(
        emitted_keys == list(expected_track_keys),
        "result.track_scores",
        "emitted key order mismatch against Stage C0 traversal order",
    )
    scored_video_ids = {record.video_id for record in track_scores}
    _require(
        not processed_with_tracks_video_ids or bool(track_scores),
        "result.track_scores",
        "must be non-empty when split contains processed_with_tracks videos",
    )
    missing_scored_video_ids = sorted(set(processed_with_tracks_video_ids) - scored_video_ids)
    _require(
        not missing_scored_video_ids,
        "result.track_scores",
        f"missing scored tracks for processed_with_tracks videos: {missing_scored_video_ids}",
    )
    return emitted_keys


def compute_stagec1_mil_baseline_scores(
    split_view: Any,
    *,
    config: StageC1MilConfig | None = None,
) -> StageC1MilResult:
    config = config or StageC1MilConfig()
    _validate_config(config)

    track_scores: list[StageC1TrackScoreRecord] = []
    seen_keys: set[tuple[str, str | int]] = set()

    all_videos, processed_with_tracks_video_ids, expected_track_keys = _iter_processed_with_tracks(split_view, config)

    for video_id in processed_with_tracks_video_ids:
        for track in split_view.iter_tracks(video_id):
            track_id, row_index = _validate_track_identity(track, video_id)
            key = (video_id, track_id)
            _require(key not in seen_keys, "track.identity", f"duplicate (video_id, track_id) key {key}")
            seen_keys.add(key)

            score = _compute_track_score(track, config)
            track_scores.append(
                StageC1TrackScoreRecord(
                    video_id=video_id,
                    track_id=track_id,
                    row_index=row_index,
                    status="processed_with_tracks",
                    score=score,
                )
            )

    emitted_keys = _validate_emitted_order_and_non_empty(
        track_scores=track_scores,
        expected_track_keys=expected_track_keys,
        processed_with_tracks_video_ids=processed_with_tracks_video_ids,
    )

    per_video_summary = _build_per_video_summary(track_scores, top_k_per_video=config.top_k_per_video)
    run_summary = _build_run_summary(
        split_view=split_view,
        track_scores=track_scores,
        all_videos=all_videos,
        expected_track_keys=expected_track_keys,
        emitted_keys=emitted_keys,
        scorer_backend="mil_v1",
    )

    return StageC1MilResult(
        track_scores=tuple(track_scores),
        per_video_summary=tuple(per_video_summary),
        run_summary=run_summary,
    )


def _parse_semver_major(version: Any, field_path: str) -> int:
    _require(isinstance(version, str), field_path, "must be string")
    parts = version.split(".")
    _require(len(parts) == 3 and all(p.isdigit() for p in parts), field_path, "must follow MAJOR.MINOR.PATCH")
    return int(parts[0])


def _require_rel_path(path_value: Any, field_path: str) -> str:
    _require(isinstance(path_value, str) and path_value, field_path, "must be non-empty relative path")
    rel = PurePosixPath(path_value)
    _require(not rel.is_absolute(), field_path, "absolute path is forbidden")
    _require(".." not in rel.parts, field_path, "must not contain '..'")
    return str(rel)


def load_stagec_label_prototype_inventory_v1(manifest_json: Path) -> StageCLabelPrototypeInventory:
    try:
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _err("prototype_manifest_json", f"missing file: {manifest_json}") from exc
    except json.JSONDecodeError as exc:
        raise _err("prototype_manifest_json", f"invalid JSON: {exc}") from exc

    _require(isinstance(manifest, dict), "prototype_manifest.$", "top-level value must be an object")
    _require(
        manifest.get("schema_name") == PROTOTYPE_SCHEMA_NAME_V1,
        "prototype_manifest.schema_name",
        f"must equal '{PROTOTYPE_SCHEMA_NAME_V1}'",
    )
    _require(
        _parse_semver_major(manifest.get("schema_version"), "prototype_manifest.schema_version") == 1,
        "prototype_manifest.schema_version",
        "unsupported major version",
    )
    _require(
        isinstance(manifest.get("embedding_dim"), int) and manifest["embedding_dim"] > 0,
        "prototype_manifest.embedding_dim",
        "must be integer > 0",
    )
    _require(manifest.get("dtype") == "float32", "prototype_manifest.dtype", "must equal 'float32'")

    labels_raw = manifest.get("labels")
    _require(isinstance(labels_raw, list) and labels_raw, "prototype_manifest.labels", "must be non-empty list")

    array_key = manifest.get("array_key", "prototypes")
    _require(isinstance(array_key, str) and array_key, "prototype_manifest.array_key", "must be non-empty string")
    arrays_path_rel = _require_rel_path(manifest.get("arrays_path"), "prototype_manifest.arrays_path")
    arrays_path = manifest_json.parent / arrays_path_rel
    _require(arrays_path.exists(), "prototype_manifest.arrays_path", f"target file missing: {arrays_path}")

    try:
        with np.load(arrays_path, allow_pickle=False) as npz:
            _require(array_key in npz, "prototype_manifest.array_key", f"missing key '{array_key}' in {arrays_path}")
            prototypes = np.asarray(npz[array_key])
    except Exception as exc:  # noqa: BLE001
        raise _err("prototype_manifest.arrays_path", f"failed to load NPZ: {exc}") from exc

    _require(prototypes.ndim == 2, "prototype_arrays", "must be rank-2 [num_labels, embedding_dim]")
    _require(prototypes.shape[1] == manifest["embedding_dim"], "prototype_arrays", "embedding dim mismatch")
    _require(np.dtype(prototypes.dtype) == np.float32, "prototype_arrays.dtype", "must be float32")
    _require(np.isfinite(prototypes).all(), "prototype_arrays", "must contain only finite values")

    num_labels = prototypes.shape[0]
    _require(len(labels_raw) == num_labels, "prototype_manifest.labels", "length must match prototype row count")

    row_index_by_key: dict[tuple[int, int | str], int] = {}
    labels_by_key: dict[tuple[int, int | str], int | str] = {}
    seen_rows: set[int] = set()
    for i, record in enumerate(labels_raw):
        rpath = f"prototype_manifest.labels[{i}]"
        _require(isinstance(record, dict), rpath, "must be object")
        _require("row_index" in record, f"{rpath}.row_index", "required field missing")
        _require("label_id" in record, f"{rpath}.label_id", "required field missing")
        row_index = record["row_index"]
        label_id = record["label_id"]
        _require(isinstance(row_index, int) and row_index >= 0, f"{rpath}.row_index", "must be integer >= 0")
        _require(row_index < num_labels, f"{rpath}.row_index", f"must be < num_labels ({num_labels})")
        _require(row_index not in seen_rows, f"{rpath}.row_index", "duplicate row_index")
        _require(_is_label_id(label_id), f"{rpath}.label_id", "must be non-empty string or integer")
        key = _canonical_label_key(label_id)
        _require(key not in labels_by_key, f"{rpath}.label_id", "duplicate canonical label id")
        seen_rows.add(row_index)
        row_index_by_key[key] = int(row_index)
        labels_by_key[key] = label_id

    _require(len(seen_rows) == num_labels, "prototype_manifest.labels", "must cover every prototype row exactly once")

    row_norms = np.linalg.norm(prototypes.astype(np.float64), axis=1)
    _require(np.all(row_norms > 0.0), "prototype_arrays", "all prototype rows must have non-zero norm")

    return StageCLabelPrototypeInventory(
        schema_name=manifest["schema_name"],
        schema_version=manifest["schema_version"],
        embedding_dim=int(manifest["embedding_dim"]),
        dtype=str(manifest["dtype"]),
        array_key=array_key,
        labels_by_key=labels_by_key,
        row_index_by_key=row_index_by_key,
        prototypes=prototypes,
    )


def _extract_labelset_records(payload: Any) -> Sequence[tuple[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("videos"), list):
        out: list[tuple[str, Any]] = []
        for index, row in enumerate(payload["videos"]):
            _require(isinstance(row, dict), f"labelset.videos[{index}]", "must be object")
            video_id = row.get("video_id")
            _require(isinstance(video_id, str) and video_id, f"labelset.videos[{index}].video_id", "must be non-empty string")
            out.append((video_id, row))
        return out

    if isinstance(payload, dict) and isinstance(payload.get("per_video"), dict):
        return [(str(k), v) for k, v in payload["per_video"].items()]

    if isinstance(payload, dict):
        return [(str(k), v) for k, v in payload.items()]

    _require(False, "labelset.$", "must be object with either {videos:[...]} or mapping by video_id")
    return ()


def load_stagec_labelset_lookup(labelset_json: Path, *, labelset_key: str) -> Dict[str, Tuple[tuple[int, int | str], ...]]:
    try:
        payload = json.loads(labelset_json.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _err("labelset_json", f"missing file: {labelset_json}") from exc
    except json.JSONDecodeError as exc:
        raise _err("labelset_json", f"invalid JSON: {exc}") from exc

    _require(isinstance(labelset_key, str) and labelset_key, "labelset_key", "must be non-empty string")

    out: Dict[str, Tuple[tuple[int, int | str], ...]] = {}
    for raw_video_id, raw_value in _extract_labelset_records(payload):
        _require(isinstance(raw_video_id, str) and raw_video_id, "labelset.video_id", "must be non-empty string")
        value = raw_value
        if isinstance(raw_value, dict):
            _require(
                labelset_key in raw_value,
                f"labelset[{raw_video_id}].{labelset_key}",
                "required key missing for this video",
            )
            value = raw_value[labelset_key]

        _require(isinstance(value, list), f"labelset[{raw_video_id}]", "label list must be an array")
        keys: set[tuple[int, int | str]] = set()
        for idx, label_id in enumerate(value):
            _require(_is_label_id(label_id), f"labelset[{raw_video_id}][{idx}]", "must be non-empty string or integer")
            keys.add(_canonical_label_key(label_id))
        out[raw_video_id] = tuple(sorted(keys))

    return out


def _compute_proto_track_score_vector(
    *,
    embedding: np.ndarray,
    candidate_keys: Sequence[tuple[int, int | str]],
    inventory: StageCLabelPrototypeInventory,
    normalized_prototypes: np.ndarray,
) -> np.ndarray:
    _require(embedding.ndim == 1, "track.embedding", "must be rank-1")
    _require(np.isfinite(embedding).all(), "track.embedding", "must contain only finite values")
    _require(embedding.shape[0] == inventory.embedding_dim, "track.embedding", "embedding dim mismatch with prototype inventory")

    emb64 = embedding.astype(np.float64, copy=False)
    emb_norm = float(np.linalg.norm(emb64))
    if emb_norm <= 0.0:
        scores = np.zeros((len(candidate_keys),), dtype=np.float64)
    else:
        emb_unit = emb64 / emb_norm
        rows = [inventory.row_index_by_key[key] for key in candidate_keys]
        scores = normalized_prototypes[rows, :].dot(emb_unit)

    _require(len(candidate_keys) > 0, "candidate_labels", "must be non-empty")
    _require(np.isfinite(scores).all(), "track.score_vector", "must contain only finite values")
    return np.asarray(scores, dtype=np.float64)


def _best_label_with_margin(
    *,
    score_vector: np.ndarray,
    candidate_keys: Sequence[tuple[int, int | str]],
) -> tuple[tuple[int, int | str], float, float, float]:
    _require(score_vector.ndim == 1, "track.score_vector", "must be rank-1")
    _require(len(candidate_keys) == int(score_vector.shape[0]), "candidate_labels", "must align with score_vector length")
    _require(len(candidate_keys) > 0, "candidate_labels", "must be non-empty")

    best_idx = 0
    best_score = float(score_vector[0])
    for idx in range(1, score_vector.shape[0]):
        score = float(score_vector[idx])
        if score > best_score or (score == best_score and candidate_keys[idx] < candidate_keys[best_idx]):
            best_idx = idx
            best_score = score

    second_best = -float("inf")
    for idx in range(score_vector.shape[0]):
        if idx == best_idx:
            continue
        score = float(score_vector[idx])
        if score > second_best:
            second_best = score

    if math.isfinite(second_best):
        margin = float(best_score - second_best)
    else:
        margin = float("inf")
    return candidate_keys[best_idx], float(best_score), float(second_best), margin


def _best_label_from_weights(
    *,
    weight_vector: np.ndarray,
    candidate_keys: Sequence[tuple[int, int | str]],
) -> tuple[tuple[int, int | str], float, float, float]:
    _require(weight_vector.ndim == 1, "track.posterior_vector", "must be rank-1")
    _require(
        len(candidate_keys) == int(weight_vector.shape[0]),
        "candidate_labels",
        "must align with track.posterior_vector length",
    )
    _require(len(candidate_keys) > 0, "candidate_labels", "must be non-empty")
    _require(np.isfinite(weight_vector).all(), "track.posterior_vector", "must contain only finite values")
    return _best_label_with_margin(score_vector=weight_vector, candidate_keys=candidate_keys)


def _compute_em_posteriors(
    *,
    score_matrix: np.ndarray,
    temperature: float,
    iterations: int,
    prior_alpha: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    _require(score_matrix.ndim == 2, "em.score_matrix", "must be rank-2")
    _require(np.isfinite(score_matrix).all(), "em.score_matrix", "must contain only finite values")
    num_tracks, num_labels = score_matrix.shape
    _require(num_labels > 0, "candidate_labels", "must be non-empty")
    if num_tracks == 0:
        return np.zeros((0, num_labels), dtype=np.float64), np.full((num_labels,), 1.0 / float(num_labels), dtype=np.float64)

    score64 = score_matrix.astype(np.float64, copy=False)
    row_max = np.max(score64, axis=1, keepdims=True)
    stable = score64 - row_max
    logits = stable / float(temperature)
    emission = np.exp(logits)
    emission = np.maximum(emission, float(eps))
    _require(np.isfinite(emission).all(), "em.emission", "must contain only finite values")

    pi = np.full((num_labels,), 1.0 / float(num_labels), dtype=np.float64)
    gamma = np.full((num_tracks, num_labels), 1.0 / float(num_labels), dtype=np.float64)
    denom_prior = float(num_tracks) + float(num_labels) * float(prior_alpha)

    for _ in range(int(iterations)):
        weighted = emission * pi.reshape(1, -1)
        row_sums = np.maximum(np.sum(weighted, axis=1, keepdims=True), float(eps))
        gamma = weighted / row_sums
        _require(np.isfinite(gamma).all(), "em.posterior", "must contain only finite values")

        counts = np.sum(gamma, axis=0)
        if denom_prior > 0.0:
            pi = (counts + float(prior_alpha)) / denom_prior
        else:
            pi = counts / float(num_tracks)
        pi_sum = np.maximum(np.sum(pi), float(eps))
        pi = pi / pi_sum

    return gamma, pi


def _compute_sinkhorn_posteriors(
    *,
    score_matrix: np.ndarray,
    temperature: float,
    iterations: int,
    tolerance: float,
    eps: float,
    target_col_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    _require(score_matrix.ndim == 2, "sinkhorn.score_matrix", "must be rank-2")
    _require(np.isfinite(score_matrix).all(), "sinkhorn.score_matrix", "must contain only finite values")
    num_tracks, num_labels = score_matrix.shape
    _require(num_labels > 0, "candidate_labels", "must be non-empty")
    if target_col_weights is None:
        normalized_col_weights = None
    else:
        weights = np.asarray(target_col_weights, dtype=np.float64)
        _require(weights.ndim == 1, "sinkhorn.target_col_weights", "must be rank-1")
        _require(
            int(weights.shape[0]) == int(num_labels),
            "sinkhorn.target_col_weights",
            "must align with score_matrix column count",
        )
        _require(np.isfinite(weights).all(), "sinkhorn.target_col_weights", "must contain only finite values")
        _require(np.all(weights >= 0.0), "sinkhorn.target_col_weights", "must be >= 0")
        weight_sum = float(np.sum(weights))
        _require(weight_sum > 0.0, "sinkhorn.target_col_weights", "sum must be > 0")
        normalized_col_weights = weights / weight_sum
    if num_tracks == 0:
        return np.zeros((0, num_labels), dtype=np.float64), {
            "converged": True,
            "iterations_used": 0,
            "row_mass_l1_error": 0.0,
            "col_mass_l1_error": 0.0,
            "target_col_mass": 0.0,
            "target_col_mass_vector": [0.0] * int(num_labels),
            "posterior_col_mass": [0.0] * int(num_labels),
            "posterior_entropy_mean": None,
        }

    score64 = score_matrix.astype(np.float64, copy=False)
    row_max = np.max(score64, axis=1, keepdims=True)
    stable = score64 - row_max
    kernel = np.exp(stable / float(temperature))
    kernel = np.maximum(kernel, float(eps))
    _require(np.isfinite(kernel).all(), "sinkhorn.kernel", "must contain only finite values")

    if normalized_col_weights is None:
        target_col_mass_vector = np.full((num_labels,), float(num_tracks) / float(num_labels), dtype=np.float64)
    else:
        target_col_mass_vector = normalized_col_weights * float(num_tracks)
    target_col_mass = float(np.mean(target_col_mass_vector)) if num_labels > 0 else 0.0
    plan = kernel.astype(np.float64, copy=True)
    converged = False
    iterations_used = int(iterations)
    row_mass_l1_error = float("inf")
    col_mass_l1_error = float("inf")

    for idx in range(int(iterations)):
        row_sums = np.maximum(np.sum(plan, axis=1, keepdims=True), float(eps))
        plan = plan / row_sums
        col_sums = np.maximum(np.sum(plan, axis=0, keepdims=True), float(eps))
        plan = plan * (target_col_mass_vector.reshape(1, -1) / col_sums)

        row_mass_l1_error = float(np.sum(np.abs(np.sum(plan, axis=1) - 1.0)))
        col_mass_l1_error = float(np.sum(np.abs(np.sum(plan, axis=0) - target_col_mass_vector)))
        if max(row_mass_l1_error, col_mass_l1_error) <= float(tolerance):
            converged = True
            iterations_used = idx + 1
            break

    row_sums = np.maximum(np.sum(plan, axis=1, keepdims=True), float(eps))
    posterior = plan / row_sums
    row_mass_l1_error = float(np.sum(np.abs(np.sum(posterior, axis=1) - 1.0)))
    posterior_col_mass = np.sum(posterior, axis=0)
    col_mass_l1_error = float(np.sum(np.abs(posterior_col_mass - target_col_mass_vector)))
    clamped = np.maximum(posterior, float(eps))
    row_entropy = -np.sum(clamped * np.log(clamped), axis=1)

    return posterior, {
        "converged": bool(converged),
        "iterations_used": int(iterations_used),
        "row_mass_l1_error": float(row_mass_l1_error),
        "col_mass_l1_error": float(col_mass_l1_error),
        "target_col_mass": float(target_col_mass),
        "target_col_mass_vector": [float(x) for x in target_col_mass_vector.tolist()],
        "posterior_col_mass": [float(x) for x in posterior_col_mass.tolist()],
        "posterior_entropy_mean": float(np.mean(row_entropy)),
    }


def _track_tiebreak_key(track_id: str | int, row_index: int) -> tuple[int, str]:
    return (int(row_index), _canonical_label_key_text(track_id))


def _should_assign_bg(*, top1_score: float, margin: float, decoder_config: StageC1DecoderConfig) -> bool:
    if decoder_config.bg_score_threshold is not None and top1_score < float(decoder_config.bg_score_threshold):
        return True
    if decoder_config.bg_min_margin is not None and math.isfinite(margin) and margin < float(decoder_config.bg_min_margin):
        return True
    return False


def _bg_reason(*, score_for_gate: float, margin: float, decoder_config: StageC1DecoderConfig) -> str | None:
    if decoder_config.bg_score_threshold is not None and score_for_gate < float(decoder_config.bg_score_threshold):
        return "score_threshold"
    if decoder_config.bg_min_margin is not None and math.isfinite(margin) and margin < float(decoder_config.bg_min_margin):
        return "min_margin"
    return None


def _decode_video_assignments(
    *,
    track_ids: Sequence[str | int],
    row_indices: Sequence[int],
    candidate_keys: Sequence[tuple[int, int | str]],
    score_matrix: np.ndarray,
    scorer_top1_keys: Sequence[tuple[int, int | str]],
    scorer_top1_scores: Sequence[float],
    scorer_margins: Sequence[float],
    decoder_config: StageC1DecoderConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    num_tracks = len(track_ids)
    _require(score_matrix.shape == (num_tracks, len(candidate_keys)), "decoder.score_matrix", "shape mismatch")

    assignments: list[dict[str, Any]] = []
    for idx in range(num_tracks):
        assignments.append(
            {
                "label_key": scorer_top1_keys[idx],
                "assigned_bg": False,
                "source": "independent",
                "score": float(scorer_top1_scores[idx]),
                "margin": float(scorer_margins[idx]),
                "ot_prob": None,
            }
        )

    coverage_hit_count = 0
    tie_break_count = 0
    coverage_skip_fg_min_count = 0
    coverage_skip_bg_gate_count = 0
    fill_fg_count = 0
    fill_bg_count = 0
    bg_reason_counts: dict[str, int] = {"score_threshold": 0, "min_margin": 0, "fg_score_min": 0, "ot_prob_min": 0}
    otlite_col_mass_l1_error: float | None = None
    otlite_row_mass_l1_error: float | None = None
    otlite_bg_ot_prob_count = 0
    if decoder_config.backend == "coverage_greedy_v1":
        assigned = [False] * num_tracks
        assignments = [{} for _ in range(num_tracks)]

        for label_index, label_key in enumerate(candidate_keys):
            best_track_idx: int | None = None
            best_score = -float("inf")
            best_tie_key = (10**18, "~")
            for track_idx in range(num_tracks):
                if assigned[track_idx]:
                    continue
                score = float(score_matrix[track_idx, label_index])
                tie_key = _track_tiebreak_key(track_ids[track_idx], row_indices[track_idx])
                if score > best_score:
                    best_track_idx = track_idx
                    best_score = score
                    best_tie_key = tie_key
                elif score == best_score:
                    tie_break_count += 1
                    if tie_key < best_tie_key:
                        best_track_idx = track_idx
                        best_tie_key = tie_key

            if best_track_idx is None:
                continue

            assignment_score = float(score_matrix[best_track_idx, label_index])
            margin = float(scorer_margins[best_track_idx])
            if assignment_score < float(decoder_config.fg_score_min):
                coverage_skip_fg_min_count += 1
                continue
            if _should_assign_bg(top1_score=assignment_score, margin=margin, decoder_config=decoder_config):
                coverage_skip_bg_gate_count += 1
                continue

            assignments[best_track_idx] = {
                "label_key": label_key,
                "assigned_bg": False,
                "source": "coverage",
                "score": assignment_score,
                "margin": float(margin),
                "bg_reason": None,
                "ot_prob": None,
            }
            assigned[best_track_idx] = True
            coverage_hit_count += 1

        for track_idx in range(num_tracks):
            if assigned[track_idx]:
                continue
            top1_key = scorer_top1_keys[track_idx]
            top1_score = float(scorer_top1_scores[track_idx])
            margin = float(scorer_margins[track_idx])
            bg_reason = None
            if top1_score < float(decoder_config.fg_score_min):
                bg_reason = "fg_score_min"
            else:
                bg_reason = _bg_reason(score_for_gate=top1_score, margin=margin, decoder_config=decoder_config)
            assign_bg = bg_reason is not None
            if assign_bg:
                fill_bg_count += 1
                bg_reason_counts[str(bg_reason)] += 1
                assignments[track_idx] = {
                    "label_key": None,
                    "assigned_bg": True,
                    "source": "fill_bg",
                    "score": top1_score,
                    "margin": margin,
                    "bg_reason": bg_reason,
                    "ot_prob": None,
                }
            else:
                fill_fg_count += 1
                assignments[track_idx] = {
                    "label_key": top1_key,
                    "assigned_bg": False,
                    "source": "fill",
                    "score": top1_score,
                    "margin": margin,
                    "bg_reason": None,
                    "ot_prob": None,
                }
    elif decoder_config.backend == "otlite_v1":
        num_labels = len(candidate_keys)
        assignments = [{} for _ in range(num_tracks)]
        if num_tracks > 0 and num_labels > 0:
            score64 = score_matrix.astype(np.float64, copy=False)
            row_max = np.max(score64, axis=1, keepdims=True)
            stable_score = score64 - row_max
            kernel = np.exp(stable_score / float(decoder_config.otlite_temperature))
            kernel = np.maximum(kernel, float(decoder_config.otlite_eps))
            plan = kernel.astype(np.float64, copy=True)
            target_col_mass = float(num_tracks) / float(num_labels)

            for _ in range(int(decoder_config.otlite_iters)):
                row_sums = np.maximum(np.sum(plan, axis=1, keepdims=True), float(decoder_config.otlite_eps))
                plan = plan / row_sums
                col_sums = np.maximum(np.sum(plan, axis=0, keepdims=True), float(decoder_config.otlite_eps))
                plan = plan * (target_col_mass / col_sums)

            row_sums = np.maximum(np.sum(plan, axis=1, keepdims=True), float(decoder_config.otlite_eps))
            plan = plan / row_sums
            otlite_row_mass_l1_error = float(np.sum(np.abs(np.sum(plan, axis=1) - 1.0)))
            otlite_col_mass_l1_error = float(np.sum(np.abs(np.sum(plan, axis=0) - target_col_mass)))

            for track_idx in range(num_tracks):
                best_label_idx = 0
                best_prob = float(plan[track_idx, 0])
                for label_idx in range(1, num_labels):
                    candidate_prob = float(plan[track_idx, label_idx])
                    if candidate_prob > best_prob:
                        best_label_idx = label_idx
                        best_prob = candidate_prob
                    elif candidate_prob == best_prob:
                        tie_break_count += 1
                        if candidate_keys[label_idx] < candidate_keys[best_label_idx]:
                            best_label_idx = label_idx
                            best_prob = candidate_prob

                chosen_key = candidate_keys[best_label_idx]
                assignment_score = float(score_matrix[track_idx, best_label_idx])
                margin = float(scorer_margins[track_idx])
                bg_reason: str | None = None
                if assignment_score < float(decoder_config.fg_score_min):
                    bg_reason = "fg_score_min"
                else:
                    bg_reason = _bg_reason(
                        score_for_gate=assignment_score,
                        margin=margin,
                        decoder_config=decoder_config,
                    )
                    if (
                        bg_reason is None
                        and decoder_config.otlite_ot_prob_min is not None
                        and best_prob < float(decoder_config.otlite_ot_prob_min)
                    ):
                        bg_reason = "ot_prob_min"

                if bg_reason is None:
                    fill_fg_count += 1
                    assignments[track_idx] = {
                        "label_key": chosen_key,
                        "assigned_bg": False,
                        "source": "otlite",
                        "score": assignment_score,
                        "margin": margin,
                        "bg_reason": None,
                        "ot_prob": best_prob,
                    }
                else:
                    fill_bg_count += 1
                    bg_reason_counts[str(bg_reason)] += 1
                    if bg_reason == "ot_prob_min":
                        otlite_bg_ot_prob_count += 1
                    assignments[track_idx] = {
                        "label_key": None,
                        "assigned_bg": True,
                        "source": "otlite_bg",
                        "score": assignment_score,
                        "margin": margin,
                        "bg_reason": bg_reason,
                        "ot_prob": best_prob,
                    }

            coverage_hit_count = len(
                {
                    row["label_key"]
                    for row in assignments
                    if not bool(row.get("assigned_bg", False)) and row.get("label_key") is not None
                }
            )

    for row in assignments:
        row.setdefault("bg_reason", None)
        row.setdefault("ot_prob", None)
    num_bg = sum(1 for row in assignments if bool(row.get("assigned_bg", False)))
    num_fg = num_tracks - num_bg
    coverage_target_count = len(candidate_keys)
    coverage_ratio = float(coverage_hit_count / coverage_target_count) if coverage_target_count > 0 else None
    policy_version = "coverage_greedy_v1.r1b"
    if decoder_config.backend == "otlite_v1":
        policy_version = "otlite_v1.r2a"
    video_diag = {
        "decoder_backend": decoder_config.backend,
        "decoder_num_tracks_fg": num_fg,
        "decoder_num_tracks_bg": num_bg,
        "decoder_coverage_target_count": coverage_target_count,
        "decoder_coverage_hit_count": coverage_hit_count,
        "decoder_coverage_ratio": coverage_ratio,
        "decoder_tie_break_count": tie_break_count,
        "decoder_coverage_skip_fg_min_count": coverage_skip_fg_min_count,
        "decoder_coverage_skip_bg_gate_count": coverage_skip_bg_gate_count,
        "decoder_fill_fg_count": fill_fg_count,
        "decoder_fill_bg_count": fill_bg_count,
        "decoder_bg_reason_counts": bg_reason_counts,
        "decoder_policy_version": policy_version,
    }
    if decoder_config.backend == "otlite_v1":
        video_diag["decoder_otlite_col_mass_l1_error"] = otlite_col_mass_l1_error
        video_diag["decoder_otlite_row_mass_l1_error"] = otlite_row_mass_l1_error
        video_diag["decoder_otlite_bg_ot_prob_count"] = int(otlite_bg_ot_prob_count)
    return assignments, video_diag


def compute_stagec1_labelset_proto_baseline_scores(
    split_view: Any,
    *,
    prototype_inventory: StageCLabelPrototypeInventory,
    labelset_lookup: Mapping[str, Sequence[tuple[int, int | str]]],
    config: StageC1MilConfig | None = None,
    empty_labelset_policy: str = "use_all_prototypes",
    decoder_config: StageC1DecoderConfig | None = None,
) -> StageC1MilResult:
    config = config or StageC1MilConfig()
    decoder_config = decoder_config or StageC1DecoderConfig()
    _validate_config(config)
    _validate_decoder_config(decoder_config)
    _require(empty_labelset_policy in EMPTY_LABELSET_POLICIES, "empty_labelset_policy", f"must be one of {sorted(EMPTY_LABELSET_POLICIES)}")
    _require(
        int(getattr(split_view, "embedding_dim", -1)) == prototype_inventory.embedding_dim,
        "prototype_manifest.embedding_dim",
        "must match split embedding_dim",
    )

    all_videos, processed_with_tracks_video_ids, expected_track_keys = _iter_processed_with_tracks(split_view, config)
    prototype_keys_all = tuple(sorted(prototype_inventory.labels_by_key.keys()))

    proto = prototype_inventory.prototypes.astype(np.float64, copy=False)
    proto_norms = np.linalg.norm(proto, axis=1, keepdims=True)
    normalized_proto = proto / proto_norms

    track_scores: list[StageC1TrackScoreRecord] = []
    seen_keys: set[tuple[str, str | int]] = set()

    videos_missing_labelset: list[str] = []
    videos_with_nonempty_labelset = 0
    videos_using_fallback = 0
    fallback_tracks_total = 0
    labelset_size_by_video: Dict[str, int] = {}
    fallback_track_count_by_video: Dict[str, int] = {}
    decoder_diag_by_video: Dict[str, Dict[str, Any]] = {}

    for video_id in processed_with_tracks_video_ids:
        raw_keys = tuple(labelset_lookup.get(video_id, ()))
        labelset_size_by_video[video_id] = len(raw_keys)
        if video_id not in labelset_lookup:
            videos_missing_labelset.append(video_id)
        if raw_keys:
            videos_with_nonempty_labelset += 1

        candidate_keys = tuple(sorted(key for key in raw_keys if key in prototype_inventory.row_index_by_key))
        used_fallback = False
        if not candidate_keys:
            if empty_labelset_policy == "error":
                raise _err(
                    "labelset_lookup",
                    (
                        f"video '{video_id}' has empty candidate label intersection against prototype inventory "
                        "under empty_labelset_policy='error'"
                    ),
                )
            candidate_keys = prototype_keys_all
            used_fallback = True
            videos_using_fallback += 1

        local_fallback_tracks = 0
        local_track_ids: list[str | int] = []
        local_row_indices: list[int] = []
        local_score_vectors: list[np.ndarray] = []
        local_scorer_top1_keys: list[tuple[int, int | str]] = []
        local_scorer_top1_scores: list[float] = []
        local_scorer_margins: list[float] = []
        for track in split_view.iter_tracks(video_id):
            track_id, row_index = _validate_track_identity(track, video_id)
            key = (video_id, track_id)
            _require(key not in seen_keys, "track.identity", f"duplicate (video_id, track_id) key {key}")
            seen_keys.add(key)

            score_vector = _compute_proto_track_score_vector(
                embedding=track.embedding,
                candidate_keys=candidate_keys,
                inventory=prototype_inventory,
                normalized_prototypes=normalized_proto,
            )
            best_label_key, best_score, _, best_margin = _best_label_with_margin(
                score_vector=score_vector,
                candidate_keys=candidate_keys,
            )

            if used_fallback:
                local_fallback_tracks += 1
                fallback_tracks_total += 1

            local_track_ids.append(track_id)
            local_row_indices.append(row_index)
            local_score_vectors.append(score_vector)
            local_scorer_top1_keys.append(best_label_key)
            local_scorer_top1_scores.append(float(best_score))
            local_scorer_margins.append(float(best_margin))

        score_matrix = np.vstack(local_score_vectors) if local_score_vectors else np.zeros((0, len(candidate_keys)), dtype=np.float64)
        decoded_assignments, video_decoder_diag = _decode_video_assignments(
            track_ids=local_track_ids,
            row_indices=local_row_indices,
            candidate_keys=candidate_keys,
            score_matrix=score_matrix,
            scorer_top1_keys=local_scorer_top1_keys,
            scorer_top1_scores=local_scorer_top1_scores,
            scorer_margins=local_scorer_margins,
            decoder_config=decoder_config,
        )
        decoder_diag_by_video[video_id] = video_decoder_diag

        for idx, track_id in enumerate(local_track_ids):
            decoded = decoded_assignments[idx]
            decoded_key = decoded["label_key"]
            decoded_label_id = prototype_inventory.labels_by_key[decoded_key] if decoded_key is not None else None
            track_scores.append(
                StageC1TrackScoreRecord(
                    video_id=video_id,
                    track_id=track_id,
                    row_index=local_row_indices[idx],
                    status="processed_with_tracks",
                    score=local_scorer_top1_scores[idx],
                    predicted_label_id=prototype_inventory.labels_by_key[local_scorer_top1_keys[idx]],
                    used_fallback_label_pool=used_fallback,
                    decoder_predicted_label_id=decoded_label_id,
                    decoder_assigned_bg=bool(decoded["assigned_bg"]),
                    decoder_assignment_source=str(decoded["source"]),
                    decoder_score=float(decoded["score"]),
                    decoder_margin=float(decoded["margin"]),
                    decoder_bg_reason=str(decoded["bg_reason"]) if decoded["bg_reason"] is not None else None,
                    decoder_ot_prob=float(decoded["ot_prob"]) if decoded["ot_prob"] is not None else None,
                )
            )

        fallback_track_count_by_video[video_id] = local_fallback_tracks

    emitted_keys = _validate_emitted_order_and_non_empty(
        track_scores=track_scores,
        expected_track_keys=expected_track_keys,
        processed_with_tracks_video_ids=processed_with_tracks_video_ids,
    )

    top_pred_labels_by_video: Dict[str, list[dict[str, Any]]] = {}
    by_video: Dict[str, Dict[str, int]] = {}
    for rec in track_scores:
        if rec.predicted_label_id is None:
            continue
        video_map = by_video.setdefault(rec.video_id, {})
        ckey = _canonical_label_key_text(rec.predicted_label_id)
        video_map[ckey] = video_map.get(ckey, 0) + 1
    for video_id, counts in by_video.items():
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_pred_labels_by_video[video_id] = [
            {
                "canonical_label_key": label_key,
                "count": int(count),
            }
            for label_key, count in ranked[:3]
        ]

    per_video_summary = _build_per_video_summary(
        track_scores,
        top_k_per_video=config.top_k_per_video,
        labelset_size_by_video=labelset_size_by_video,
        fallback_track_count_by_video=fallback_track_count_by_video,
        top_predicted_labels_by_video=top_pred_labels_by_video,
        decoder_diag_by_video=decoder_diag_by_video,
    )

    decoder_total_bg = int(sum(int(diag["decoder_num_tracks_bg"]) for diag in decoder_diag_by_video.values()))
    decoder_total_fg = int(sum(int(diag["decoder_num_tracks_fg"]) for diag in decoder_diag_by_video.values()))
    decoder_coverage_target = int(sum(int(diag["decoder_coverage_target_count"]) for diag in decoder_diag_by_video.values()))
    decoder_coverage_hit = int(sum(int(diag["decoder_coverage_hit_count"]) for diag in decoder_diag_by_video.values()))
    decoder_tie_break_total = int(sum(int(diag["decoder_tie_break_count"]) for diag in decoder_diag_by_video.values()))
    decoder_coverage_skip_fg_min_total = int(
        sum(int(diag["decoder_coverage_skip_fg_min_count"]) for diag in decoder_diag_by_video.values())
    )
    decoder_coverage_skip_bg_gate_total = int(
        sum(int(diag["decoder_coverage_skip_bg_gate_count"]) for diag in decoder_diag_by_video.values())
    )
    decoder_fill_fg_total = int(sum(int(diag["decoder_fill_fg_count"]) for diag in decoder_diag_by_video.values()))
    decoder_fill_bg_total = int(sum(int(diag["decoder_fill_bg_count"]) for diag in decoder_diag_by_video.values()))
    decoder_bg_reason_counts: dict[str, int] = {"score_threshold": 0, "min_margin": 0, "fg_score_min": 0, "ot_prob_min": 0}
    for diag in decoder_diag_by_video.values():
        reason_counts = diag.get("decoder_bg_reason_counts")
        if isinstance(reason_counts, dict):
            for key in ("score_threshold", "min_margin", "fg_score_min", "ot_prob_min"):
                decoder_bg_reason_counts[key] += int(reason_counts.get(key, 0))
    decoder_coverage_ratio = float(decoder_coverage_hit / decoder_coverage_target) if decoder_coverage_target > 0 else None
    otlite_col_l1_values: list[float] = []
    otlite_row_l1_values: list[float] = []
    otlite_bg_ot_prob_count = 0
    if decoder_config.backend == "otlite_v1":
        for diag in decoder_diag_by_video.values():
            col_err = diag.get("decoder_otlite_col_mass_l1_error")
            row_err = diag.get("decoder_otlite_row_mass_l1_error")
            if isinstance(col_err, (int, float)):
                otlite_col_l1_values.append(float(col_err))
            if isinstance(row_err, (int, float)):
                otlite_row_l1_values.append(float(row_err))
            otlite_bg_ot_prob_count += int(diag.get("decoder_otlite_bg_ot_prob_count", 0))

    policy_version = "coverage_greedy_v1.r1b"
    if decoder_config.backend == "otlite_v1":
        policy_version = "otlite_v1.r2a"

    run_summary = _build_run_summary(
        split_view=split_view,
        track_scores=track_scores,
        all_videos=all_videos,
        expected_track_keys=expected_track_keys,
        emitted_keys=emitted_keys,
        scorer_backend="labelset_proto_v1",
        extra={
            "prototype_inventory": {
                "schema_name": prototype_inventory.schema_name,
                "schema_version": prototype_inventory.schema_version,
                "num_labels": len(prototype_inventory.labels_by_key),
                "embedding_dim": prototype_inventory.embedding_dim,
                "dtype": prototype_inventory.dtype,
                "array_key": prototype_inventory.array_key,
            },
            "labelset_coverage": {
                "videos_with_nonempty_labelset": videos_with_nonempty_labelset,
                "videos_missing_labelset": sorted(videos_missing_labelset),
                "num_videos_missing_labelset": len(videos_missing_labelset),
                "num_videos_using_fallback_label_pool": videos_using_fallback,
                "num_tracks_scored_with_fallback_label_pool": fallback_tracks_total,
                "empty_labelset_policy": empty_labelset_policy,
            },
            "label_conditioned_score_distribution": _as_stats([r.score for r in track_scores]),
            "decoder_backend": decoder_config.backend,
            "decoder_summary": {
                "num_tracks_fg": decoder_total_fg,
                "num_tracks_bg": decoder_total_bg,
                "coverage_target_count": decoder_coverage_target,
                "coverage_hit_count": decoder_coverage_hit,
                "coverage_ratio": decoder_coverage_ratio,
                "tie_break_count": decoder_tie_break_total,
                "coverage_skip_fg_min_count": decoder_coverage_skip_fg_min_total,
                "coverage_skip_bg_gate_count": decoder_coverage_skip_bg_gate_total,
                "fill_fg_count": decoder_fill_fg_total,
                "fill_bg_count": decoder_fill_bg_total,
                "bg_reason_counts": decoder_bg_reason_counts,
                "fg_score_min": float(decoder_config.fg_score_min),
                "bg_score_threshold": decoder_config.bg_score_threshold,
                "bg_min_margin": decoder_config.bg_min_margin,
                "policy_version": policy_version,
            },
        },
    )
    if decoder_config.backend == "otlite_v1":
        run_summary["decoder_summary"]["otlite_temperature"] = float(decoder_config.otlite_temperature)
        run_summary["decoder_summary"]["otlite_iters"] = int(decoder_config.otlite_iters)
        run_summary["decoder_summary"]["otlite_eps"] = float(decoder_config.otlite_eps)
        run_summary["decoder_summary"]["otlite_ot_prob_min"] = decoder_config.otlite_ot_prob_min
        run_summary["decoder_summary"]["otlite_col_mass_l1_error_mean"] = (
            float(np.mean(np.asarray(otlite_col_l1_values, dtype=np.float64))) if otlite_col_l1_values else None
        )
        run_summary["decoder_summary"]["otlite_col_mass_l1_error_max"] = max(otlite_col_l1_values) if otlite_col_l1_values else None
        run_summary["decoder_summary"]["otlite_row_mass_l1_error_mean"] = (
            float(np.mean(np.asarray(otlite_row_l1_values, dtype=np.float64))) if otlite_row_l1_values else None
        )
        run_summary["decoder_summary"]["otlite_row_mass_l1_error_max"] = max(otlite_row_l1_values) if otlite_row_l1_values else None
        run_summary["decoder_summary"]["otlite_bg_ot_prob_count"] = int(otlite_bg_ot_prob_count)

    return StageC1MilResult(
        track_scores=tuple(track_scores),
        per_video_summary=tuple(per_video_summary),
        run_summary=run_summary,
    )


def compute_stagec1_em_v1_scores(
    split_view: Any,
    *,
    prototype_inventory: StageCLabelPrototypeInventory,
    labelset_lookup: Mapping[str, Sequence[tuple[int, int | str]]],
    config: StageC1MilConfig | None = None,
    empty_labelset_policy: str = "use_all_prototypes",
    decoder_config: StageC1DecoderConfig | None = None,
    em_config: StageC1EmConfig | None = None,
) -> StageC1MilResult:
    config = config or StageC1MilConfig()
    decoder_config = decoder_config or StageC1DecoderConfig()
    em_config = em_config or StageC1EmConfig()
    _validate_config(config)
    _validate_decoder_config(decoder_config)
    _validate_em_config(em_config)
    _require(empty_labelset_policy in EMPTY_LABELSET_POLICIES, "empty_labelset_policy", f"must be one of {sorted(EMPTY_LABELSET_POLICIES)}")
    _require(
        int(getattr(split_view, "embedding_dim", -1)) == prototype_inventory.embedding_dim,
        "prototype_manifest.embedding_dim",
        "must match split embedding_dim",
    )

    all_videos, processed_with_tracks_video_ids, expected_track_keys = _iter_processed_with_tracks(split_view, config)
    prototype_keys_all = tuple(sorted(prototype_inventory.labels_by_key.keys()))
    proto = prototype_inventory.prototypes.astype(np.float64, copy=False)
    proto_norms = np.linalg.norm(proto, axis=1, keepdims=True)
    normalized_proto = proto / proto_norms

    track_scores: list[StageC1TrackScoreRecord] = []
    seen_keys: set[tuple[str, str | int]] = set()

    videos_missing_labelset: list[str] = []
    videos_with_nonempty_labelset = 0
    videos_using_fallback = 0
    fallback_tracks_total = 0
    labelset_size_by_video: Dict[str, int] = {}
    fallback_track_count_by_video: Dict[str, int] = {}
    decoder_diag_by_video: Dict[str, Dict[str, Any]] = {}
    em_posterior_entropy_by_video: Dict[str, float | None] = {}
    em_mixture_l1_uniform_by_video: Dict[str, float | None] = {}

    for video_id in processed_with_tracks_video_ids:
        raw_keys = tuple(labelset_lookup.get(video_id, ()))
        labelset_size_by_video[video_id] = len(raw_keys)
        if video_id not in labelset_lookup:
            videos_missing_labelset.append(video_id)
        if raw_keys:
            videos_with_nonempty_labelset += 1

        candidate_keys = tuple(sorted(key for key in raw_keys if key in prototype_inventory.row_index_by_key))
        used_fallback = False
        if not candidate_keys:
            if empty_labelset_policy == "error":
                raise _err(
                    "labelset_lookup",
                    (
                        f"video '{video_id}' has empty candidate label intersection against prototype inventory "
                        "under empty_labelset_policy='error'"
                    ),
                )
            candidate_keys = prototype_keys_all
            used_fallback = True
            videos_using_fallback += 1

        local_fallback_tracks = 0
        local_track_ids: list[str | int] = []
        local_row_indices: list[int] = []
        local_cos_score_vectors: list[np.ndarray] = []
        for track in split_view.iter_tracks(video_id):
            track_id, row_index = _validate_track_identity(track, video_id)
            key = (video_id, track_id)
            _require(key not in seen_keys, "track.identity", f"duplicate (video_id, track_id) key {key}")
            seen_keys.add(key)

            score_vector = _compute_proto_track_score_vector(
                embedding=track.embedding,
                candidate_keys=candidate_keys,
                inventory=prototype_inventory,
                normalized_prototypes=normalized_proto,
            )
            if used_fallback:
                local_fallback_tracks += 1
                fallback_tracks_total += 1
            local_track_ids.append(track_id)
            local_row_indices.append(row_index)
            local_cos_score_vectors.append(score_vector)

        local_cos_score_matrix = (
            np.vstack(local_cos_score_vectors) if local_cos_score_vectors else np.zeros((0, len(candidate_keys)), dtype=np.float64)
        )
        posterior_matrix, mixture_weights = _compute_em_posteriors(
            score_matrix=local_cos_score_matrix,
            temperature=float(em_config.temperature),
            iterations=int(em_config.iterations),
            prior_alpha=float(em_config.prior_alpha),
            eps=float(em_config.eps),
        )

        local_scorer_top1_keys: list[tuple[int, int | str]] = []
        local_scorer_top1_scores: list[float] = []
        local_scorer_margins: list[float] = []
        for idx in range(len(local_track_ids)):
            best_key, best_prob, _, best_margin = _best_label_from_weights(
                weight_vector=posterior_matrix[idx, :],
                candidate_keys=candidate_keys,
            )
            local_scorer_top1_keys.append(best_key)
            local_scorer_top1_scores.append(float(best_prob))
            local_scorer_margins.append(float(best_margin))

        decoded_assignments, video_decoder_diag = _decode_video_assignments(
            track_ids=local_track_ids,
            row_indices=local_row_indices,
            candidate_keys=candidate_keys,
            score_matrix=posterior_matrix,
            scorer_top1_keys=local_scorer_top1_keys,
            scorer_top1_scores=local_scorer_top1_scores,
            scorer_margins=local_scorer_margins,
            decoder_config=decoder_config,
        )
        decoder_diag_by_video[video_id] = video_decoder_diag

        if posterior_matrix.shape[0] > 0:
            clamped = np.maximum(posterior_matrix, float(em_config.eps))
            row_entropy = -np.sum(clamped * np.log(clamped), axis=1)
            em_posterior_entropy_by_video[video_id] = float(np.mean(row_entropy))
            uniform = np.full((posterior_matrix.shape[1],), 1.0 / float(posterior_matrix.shape[1]), dtype=np.float64)
            em_mixture_l1_uniform_by_video[video_id] = float(np.sum(np.abs(mixture_weights - uniform)))
        else:
            em_posterior_entropy_by_video[video_id] = None
            em_mixture_l1_uniform_by_video[video_id] = None

        for idx, track_id in enumerate(local_track_ids):
            decoded = decoded_assignments[idx]
            decoded_key = decoded["label_key"]
            decoded_label_id = prototype_inventory.labels_by_key[decoded_key] if decoded_key is not None else None
            track_scores.append(
                StageC1TrackScoreRecord(
                    video_id=video_id,
                    track_id=track_id,
                    row_index=local_row_indices[idx],
                    status="processed_with_tracks",
                    score=local_scorer_top1_scores[idx],
                    predicted_label_id=prototype_inventory.labels_by_key[local_scorer_top1_keys[idx]],
                    used_fallback_label_pool=used_fallback,
                    decoder_predicted_label_id=decoded_label_id,
                    decoder_assigned_bg=bool(decoded["assigned_bg"]),
                    decoder_assignment_source=str(decoded["source"]),
                    decoder_score=float(decoded["score"]),
                    decoder_margin=float(decoded["margin"]),
                    decoder_bg_reason=str(decoded["bg_reason"]) if decoded["bg_reason"] is not None else None,
                    decoder_ot_prob=float(decoded["ot_prob"]) if decoded["ot_prob"] is not None else None,
                )
            )

        fallback_track_count_by_video[video_id] = local_fallback_tracks

    emitted_keys = _validate_emitted_order_and_non_empty(
        track_scores=track_scores,
        expected_track_keys=expected_track_keys,
        processed_with_tracks_video_ids=processed_with_tracks_video_ids,
    )

    top_pred_labels_by_video: Dict[str, list[dict[str, Any]]] = {}
    by_video: Dict[str, Dict[str, int]] = {}
    for rec in track_scores:
        if rec.predicted_label_id is None:
            continue
        video_map = by_video.setdefault(rec.video_id, {})
        ckey = _canonical_label_key_text(rec.predicted_label_id)
        video_map[ckey] = video_map.get(ckey, 0) + 1
    for video_id, counts in by_video.items():
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_pred_labels_by_video[video_id] = [
            {
                "canonical_label_key": label_key,
                "count": int(count),
            }
            for label_key, count in ranked[:3]
        ]

    per_video_summary = _build_per_video_summary(
        track_scores,
        top_k_per_video=config.top_k_per_video,
        labelset_size_by_video=labelset_size_by_video,
        fallback_track_count_by_video=fallback_track_count_by_video,
        top_predicted_labels_by_video=top_pred_labels_by_video,
        decoder_diag_by_video=decoder_diag_by_video,
    )

    for row in per_video_summary:
        video_id = str(row["video_id"])
        row["em_posterior_entropy_mean"] = em_posterior_entropy_by_video.get(video_id)
        row["em_mixture_l1_uniform"] = em_mixture_l1_uniform_by_video.get(video_id)

    decoder_total_bg = int(sum(int(diag["decoder_num_tracks_bg"]) for diag in decoder_diag_by_video.values()))
    decoder_total_fg = int(sum(int(diag["decoder_num_tracks_fg"]) for diag in decoder_diag_by_video.values()))
    decoder_coverage_target = int(sum(int(diag["decoder_coverage_target_count"]) for diag in decoder_diag_by_video.values()))
    decoder_coverage_hit = int(sum(int(diag["decoder_coverage_hit_count"]) for diag in decoder_diag_by_video.values()))
    decoder_tie_break_total = int(sum(int(diag["decoder_tie_break_count"]) for diag in decoder_diag_by_video.values()))
    decoder_coverage_skip_fg_min_total = int(
        sum(int(diag["decoder_coverage_skip_fg_min_count"]) for diag in decoder_diag_by_video.values())
    )
    decoder_coverage_skip_bg_gate_total = int(
        sum(int(diag["decoder_coverage_skip_bg_gate_count"]) for diag in decoder_diag_by_video.values())
    )
    decoder_fill_fg_total = int(sum(int(diag["decoder_fill_fg_count"]) for diag in decoder_diag_by_video.values()))
    decoder_fill_bg_total = int(sum(int(diag["decoder_fill_bg_count"]) for diag in decoder_diag_by_video.values()))
    decoder_bg_reason_counts: dict[str, int] = {"score_threshold": 0, "min_margin": 0, "fg_score_min": 0, "ot_prob_min": 0}
    for diag in decoder_diag_by_video.values():
        reason_counts = diag.get("decoder_bg_reason_counts")
        if isinstance(reason_counts, dict):
            for key in ("score_threshold", "min_margin", "fg_score_min", "ot_prob_min"):
                decoder_bg_reason_counts[key] += int(reason_counts.get(key, 0))
    decoder_coverage_ratio = float(decoder_coverage_hit / decoder_coverage_target) if decoder_coverage_target > 0 else None
    otlite_col_l1_values: list[float] = []
    otlite_row_l1_values: list[float] = []
    otlite_bg_ot_prob_count = 0
    if decoder_config.backend == "otlite_v1":
        for diag in decoder_diag_by_video.values():
            col_err = diag.get("decoder_otlite_col_mass_l1_error")
            row_err = diag.get("decoder_otlite_row_mass_l1_error")
            if isinstance(col_err, (int, float)):
                otlite_col_l1_values.append(float(col_err))
            if isinstance(row_err, (int, float)):
                otlite_row_l1_values.append(float(row_err))
            otlite_bg_ot_prob_count += int(diag.get("decoder_otlite_bg_ot_prob_count", 0))

    policy_version = "coverage_greedy_v1.r1b"
    if decoder_config.backend == "otlite_v1":
        policy_version = "otlite_v1.r2a"

    em_posterior_entropy_values = [v for v in em_posterior_entropy_by_video.values() if isinstance(v, (int, float))]
    em_mixture_l1_uniform_values = [v for v in em_mixture_l1_uniform_by_video.values() if isinstance(v, (int, float))]
    run_summary = _build_run_summary(
        split_view=split_view,
        track_scores=track_scores,
        all_videos=all_videos,
        expected_track_keys=expected_track_keys,
        emitted_keys=emitted_keys,
        scorer_backend="em_v1",
        extra={
            "prototype_inventory": {
                "schema_name": prototype_inventory.schema_name,
                "schema_version": prototype_inventory.schema_version,
                "num_labels": len(prototype_inventory.labels_by_key),
                "embedding_dim": prototype_inventory.embedding_dim,
                "dtype": prototype_inventory.dtype,
                "array_key": prototype_inventory.array_key,
            },
            "labelset_coverage": {
                "videos_with_nonempty_labelset": videos_with_nonempty_labelset,
                "videos_missing_labelset": sorted(videos_missing_labelset),
                "num_videos_missing_labelset": len(videos_missing_labelset),
                "num_videos_using_fallback_label_pool": videos_using_fallback,
                "num_tracks_scored_with_fallback_label_pool": fallback_tracks_total,
                "empty_labelset_policy": empty_labelset_policy,
            },
            "label_conditioned_score_distribution": _as_stats([r.score for r in track_scores]),
            "decoder_backend": decoder_config.backend,
            "decoder_summary": {
                "num_tracks_fg": decoder_total_fg,
                "num_tracks_bg": decoder_total_bg,
                "coverage_target_count": decoder_coverage_target,
                "coverage_hit_count": decoder_coverage_hit,
                "coverage_ratio": decoder_coverage_ratio,
                "tie_break_count": decoder_tie_break_total,
                "coverage_skip_fg_min_count": decoder_coverage_skip_fg_min_total,
                "coverage_skip_bg_gate_count": decoder_coverage_skip_bg_gate_total,
                "fill_fg_count": decoder_fill_fg_total,
                "fill_bg_count": decoder_fill_bg_total,
                "bg_reason_counts": decoder_bg_reason_counts,
                "fg_score_min": float(decoder_config.fg_score_min),
                "bg_score_threshold": decoder_config.bg_score_threshold,
                "bg_min_margin": decoder_config.bg_min_margin,
                "policy_version": policy_version,
            },
            "em_summary": {
                "temperature": float(em_config.temperature),
                "iterations": int(em_config.iterations),
                "prior_alpha": float(em_config.prior_alpha),
                "eps": float(em_config.eps),
                "posterior_entropy_mean": (
                    float(np.mean(np.asarray(em_posterior_entropy_values, dtype=np.float64)))
                    if em_posterior_entropy_values
                    else None
                ),
                "posterior_entropy_max": max(em_posterior_entropy_values) if em_posterior_entropy_values else None,
                "mixture_l1_uniform_mean": (
                    float(np.mean(np.asarray(em_mixture_l1_uniform_values, dtype=np.float64)))
                    if em_mixture_l1_uniform_values
                    else None
                ),
                "mixture_l1_uniform_max": max(em_mixture_l1_uniform_values) if em_mixture_l1_uniform_values else None,
                "policy_version": "em_v1.r1",
            },
        },
    )
    if decoder_config.backend == "otlite_v1":
        run_summary["decoder_summary"]["otlite_temperature"] = float(decoder_config.otlite_temperature)
        run_summary["decoder_summary"]["otlite_iters"] = int(decoder_config.otlite_iters)
        run_summary["decoder_summary"]["otlite_eps"] = float(decoder_config.otlite_eps)
        run_summary["decoder_summary"]["otlite_ot_prob_min"] = decoder_config.otlite_ot_prob_min
        run_summary["decoder_summary"]["otlite_col_mass_l1_error_mean"] = (
            float(np.mean(np.asarray(otlite_col_l1_values, dtype=np.float64))) if otlite_col_l1_values else None
        )
        run_summary["decoder_summary"]["otlite_col_mass_l1_error_max"] = max(otlite_col_l1_values) if otlite_col_l1_values else None
        run_summary["decoder_summary"]["otlite_row_mass_l1_error_mean"] = (
            float(np.mean(np.asarray(otlite_row_l1_values, dtype=np.float64))) if otlite_row_l1_values else None
        )
        run_summary["decoder_summary"]["otlite_row_mass_l1_error_max"] = max(otlite_row_l1_values) if otlite_row_l1_values else None
        run_summary["decoder_summary"]["otlite_bg_ot_prob_count"] = int(otlite_bg_ot_prob_count)

    return StageC1MilResult(
        track_scores=tuple(track_scores),
        per_video_summary=tuple(per_video_summary),
        run_summary=run_summary,
    )


def compute_stagec1_sinkhorn_v1_scores(
    split_view: Any,
    *,
    prototype_inventory: StageCLabelPrototypeInventory,
    labelset_lookup: Mapping[str, Sequence[tuple[int, int | str]]],
    config: StageC1MilConfig | None = None,
    empty_labelset_policy: str = "use_all_prototypes",
    decoder_config: StageC1DecoderConfig | None = None,
    sinkhorn_config: StageC1SinkhornConfig | None = None,
    sinkhorn_c43_config: StageC1SinkhornC43Config | None = None,
) -> StageC1MilResult:
    config = config or StageC1MilConfig()
    decoder_config = decoder_config or StageC1DecoderConfig()
    sinkhorn_config = sinkhorn_config or StageC1SinkhornConfig()
    sinkhorn_c43_config = sinkhorn_c43_config or StageC1SinkhornC43Config()
    _validate_config(config)
    _validate_decoder_config(decoder_config)
    _validate_sinkhorn_config(sinkhorn_config)
    _validate_sinkhorn_c43_config(sinkhorn_c43_config)
    _require(empty_labelset_policy in EMPTY_LABELSET_POLICIES, "empty_labelset_policy", f"must be one of {sorted(EMPTY_LABELSET_POLICIES)}")
    _require(
        int(getattr(split_view, "embedding_dim", -1)) == prototype_inventory.embedding_dim,
        "prototype_manifest.embedding_dim",
        "must match split embedding_dim",
    )
    _validate_no_sinkhorn_special_label_collisions(prototype_inventory)

    all_videos, processed_with_tracks_video_ids, expected_track_keys = _iter_processed_with_tracks(split_view, config)
    prototype_keys_all = tuple(sorted(prototype_inventory.labels_by_key.keys()))
    proto = prototype_inventory.prototypes.astype(np.float64, copy=False)
    proto_norms = np.linalg.norm(proto, axis=1, keepdims=True)
    normalized_proto = proto / proto_norms

    track_scores: list[StageC1TrackScoreRecord] = []
    seen_keys: set[tuple[str, str | int]] = set()

    videos_missing_labelset: list[str] = []
    videos_with_nonempty_labelset = 0
    videos_using_fallback = 0
    fallback_tracks_total = 0
    labelset_size_by_video: Dict[str, int] = {}
    fallback_track_count_by_video: Dict[str, int] = {}
    decoder_diag_by_video: Dict[str, Dict[str, Any]] = {}
    sinkhorn_diag_by_video: Dict[str, Dict[str, Any]] = {}
    sinkhorn_bg_key = _canonical_label_key(SINKHORN_SPECIAL_BG_LABEL_ID)

    for video_id in processed_with_tracks_video_ids:
        raw_keys = tuple(labelset_lookup.get(video_id, ()))
        labelset_size_by_video[video_id] = len(raw_keys)
        if video_id not in labelset_lookup:
            videos_missing_labelset.append(video_id)
        if raw_keys:
            videos_with_nonempty_labelset += 1

        candidate_keys = tuple(sorted(key for key in raw_keys if key in prototype_inventory.row_index_by_key))
        used_fallback = False
        if not candidate_keys:
            if empty_labelset_policy == "error":
                raise _err(
                    "labelset_lookup",
                    (
                        f"video '{video_id}' has empty candidate label intersection against prototype inventory "
                        "under empty_labelset_policy='error'"
                    ),
                )
            candidate_keys = prototype_keys_all
            used_fallback = True
            videos_using_fallback += 1

        local_fallback_tracks = 0
        local_track_ids: list[str | int] = []
        local_row_indices: list[int] = []
        local_cos_score_vectors: list[np.ndarray] = []
        for track in split_view.iter_tracks(video_id):
            track_id, row_index = _validate_track_identity(track, video_id)
            key = (video_id, track_id)
            _require(key not in seen_keys, "track.identity", f"duplicate (video_id, track_id) key {key}")
            seen_keys.add(key)

            score_vector = _compute_proto_track_score_vector(
                embedding=track.embedding,
                candidate_keys=candidate_keys,
                inventory=prototype_inventory,
                normalized_prototypes=normalized_proto,
            )
            if used_fallback:
                local_fallback_tracks += 1
                fallback_tracks_total += 1
            local_track_ids.append(track_id)
            local_row_indices.append(row_index)
            local_cos_score_vectors.append(score_vector)

        local_observed_score_matrix = (
            np.vstack(local_cos_score_vectors) if local_cos_score_vectors else np.zeros((0, len(candidate_keys)), dtype=np.float64)
        )

        sinkhorn_candidate_keys = candidate_keys
        sinkhorn_score_matrix = local_observed_score_matrix
        sinkhorn_target_col_weights: np.ndarray | None = None
        active_special_columns: tuple[str, ...] = ()
        if sinkhorn_c43_config.enable:
            active_special_columns = (SINKHORN_SPECIAL_BG_LABEL_ID,)
            bg_column = np.full((local_observed_score_matrix.shape[0], 1), float(sinkhorn_c43_config.bg_score), dtype=np.float64)
            sinkhorn_score_matrix = np.concatenate([local_observed_score_matrix, bg_column], axis=1)
            sinkhorn_candidate_keys = tuple(list(candidate_keys) + [sinkhorn_bg_key])
            sinkhorn_target_col_weights = np.asarray(
                [1.0] * len(candidate_keys) + [float(sinkhorn_c43_config.bg_prior_weight)],
                dtype=np.float64,
            )

        posterior_matrix, sinkhorn_diag = _compute_sinkhorn_posteriors(
            score_matrix=sinkhorn_score_matrix,
            temperature=float(sinkhorn_config.temperature),
            iterations=int(sinkhorn_config.iterations),
            tolerance=float(sinkhorn_config.tolerance),
            eps=float(sinkhorn_config.eps),
            target_col_weights=sinkhorn_target_col_weights,
        )
        sinkhorn_diag["c43_enabled"] = bool(sinkhorn_c43_config.enable)
        sinkhorn_diag["active_special_columns"] = list(active_special_columns)
        if sinkhorn_c43_config.enable:
            observed_col_count = len(candidate_keys)
            bg_idx = observed_col_count
            sinkhorn_diag["c43_mass_observed_total"] = float(np.sum(posterior_matrix[:, :observed_col_count]))
            sinkhorn_diag["c43_mass_bg_total"] = float(np.sum(posterior_matrix[:, bg_idx]))
            sinkhorn_diag["c43_mass_unk_fg_total"] = 0.0
        sinkhorn_diag_by_video[video_id] = sinkhorn_diag

        local_scorer_top1_keys: list[tuple[int, int | str]] = []
        local_scorer_top1_scores: list[float] = []
        local_scorer_margins: list[float] = []
        local_predicted_label_sources: list[str | None] = []
        local_sinkhorn_bg_posteriors: list[float | None] = []
        local_sinkhorn_top_observed_scores: list[float | None] = []
        for idx in range(len(local_track_ids)):
            best_key, best_prob, _, best_margin = _best_label_from_weights(
                weight_vector=posterior_matrix[idx, :],
                candidate_keys=sinkhorn_candidate_keys,
            )
            local_scorer_top1_keys.append(best_key)
            local_scorer_top1_scores.append(float(best_prob))
            local_scorer_margins.append(float(best_margin))
            if sinkhorn_c43_config.enable:
                if best_key == sinkhorn_bg_key:
                    local_predicted_label_sources.append("bg")
                else:
                    local_predicted_label_sources.append("observed")
            else:
                local_predicted_label_sources.append(None)
            if sinkhorn_c43_config.enable:
                observed_probs = posterior_matrix[idx, : len(candidate_keys)]
                local_sinkhorn_top_observed_scores.append(float(np.max(observed_probs)))
                local_sinkhorn_bg_posteriors.append(float(posterior_matrix[idx, len(candidate_keys)]))
            else:
                local_sinkhorn_top_observed_scores.append(None)
                local_sinkhorn_bg_posteriors.append(None)

        decoder_score_matrix = posterior_matrix[:, : len(candidate_keys)] if sinkhorn_c43_config.enable else posterior_matrix
        decoder_scorer_top1_keys: list[tuple[int, int | str]] = []
        decoder_scorer_top1_scores: list[float] = []
        decoder_scorer_margins: list[float] = []
        for idx in range(len(local_track_ids)):
            best_key, best_prob, _, best_margin = _best_label_from_weights(
                weight_vector=decoder_score_matrix[idx, :],
                candidate_keys=candidate_keys,
            )
            decoder_scorer_top1_keys.append(best_key)
            decoder_scorer_top1_scores.append(float(best_prob))
            decoder_scorer_margins.append(float(best_margin))

        decoded_assignments, video_decoder_diag = _decode_video_assignments(
            track_ids=local_track_ids,
            row_indices=local_row_indices,
            candidate_keys=candidate_keys,
            score_matrix=decoder_score_matrix,
            scorer_top1_keys=decoder_scorer_top1_keys,
            scorer_top1_scores=decoder_scorer_top1_scores,
            scorer_margins=decoder_scorer_margins,
            decoder_config=decoder_config,
        )
        decoder_diag_by_video[video_id] = video_decoder_diag

        for idx, track_id in enumerate(local_track_ids):
            decoded = decoded_assignments[idx]
            decoded_key = decoded["label_key"]
            decoded_label_id = prototype_inventory.labels_by_key[decoded_key] if decoded_key is not None else None
            scorer_key = local_scorer_top1_keys[idx]
            if scorer_key == sinkhorn_bg_key:
                scorer_label_id: int | str = SINKHORN_SPECIAL_BG_LABEL_ID
            else:
                scorer_label_id = prototype_inventory.labels_by_key[scorer_key]
            track_scores.append(
                StageC1TrackScoreRecord(
                    video_id=video_id,
                    track_id=track_id,
                    row_index=local_row_indices[idx],
                    status="processed_with_tracks",
                    score=local_scorer_top1_scores[idx],
                    predicted_label_id=scorer_label_id,
                    used_fallback_label_pool=used_fallback,
                    decoder_predicted_label_id=decoded_label_id,
                    decoder_assigned_bg=bool(decoded["assigned_bg"]),
                    decoder_assignment_source=str(decoded["source"]),
                    decoder_score=float(decoded["score"]),
                    decoder_margin=float(decoded["margin"]),
                    decoder_bg_reason=str(decoded["bg_reason"]) if decoded["bg_reason"] is not None else None,
                    decoder_ot_prob=float(decoded["ot_prob"]) if decoded["ot_prob"] is not None else None,
                    predicted_label_source=local_predicted_label_sources[idx],
                    sinkhorn_active_special_columns=active_special_columns if sinkhorn_c43_config.enable else None,
                    sinkhorn_bg_posterior=local_sinkhorn_bg_posteriors[idx],
                    sinkhorn_top_observed_score=local_sinkhorn_top_observed_scores[idx],
                )
            )

        fallback_track_count_by_video[video_id] = local_fallback_tracks

    emitted_keys = _validate_emitted_order_and_non_empty(
        track_scores=track_scores,
        expected_track_keys=expected_track_keys,
        processed_with_tracks_video_ids=processed_with_tracks_video_ids,
    )

    top_pred_labels_by_video: Dict[str, list[dict[str, Any]]] = {}
    by_video: Dict[str, Dict[str, int]] = {}
    for rec in track_scores:
        if rec.predicted_label_id is None:
            continue
        video_map = by_video.setdefault(rec.video_id, {})
        ckey = _canonical_label_key_text(rec.predicted_label_id)
        video_map[ckey] = video_map.get(ckey, 0) + 1
    for video_id, counts in by_video.items():
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_pred_labels_by_video[video_id] = [
            {
                "canonical_label_key": label_key,
                "count": int(count),
            }
            for label_key, count in ranked[:3]
        ]

    per_video_summary = _build_per_video_summary(
        track_scores,
        top_k_per_video=config.top_k_per_video,
        labelset_size_by_video=labelset_size_by_video,
        fallback_track_count_by_video=fallback_track_count_by_video,
        top_predicted_labels_by_video=top_pred_labels_by_video,
        decoder_diag_by_video=decoder_diag_by_video,
    )

    for row in per_video_summary:
        video_id = str(row["video_id"])
        sinkhorn_diag = sinkhorn_diag_by_video.get(video_id, {})
        row["sinkhorn_converged"] = bool(sinkhorn_diag.get("converged", False))
        row["sinkhorn_iterations_used"] = int(sinkhorn_diag.get("iterations_used", 0))
        row["sinkhorn_row_mass_l1_error"] = sinkhorn_diag.get("row_mass_l1_error")
        row["sinkhorn_col_mass_l1_error"] = sinkhorn_diag.get("col_mass_l1_error")
        row["sinkhorn_target_col_mass"] = sinkhorn_diag.get("target_col_mass")
        row["sinkhorn_posterior_entropy_mean"] = sinkhorn_diag.get("posterior_entropy_mean")
        if sinkhorn_c43_config.enable:
            counts = {"observed": 0, "bg": 0, "unk_fg": 0}
            for rec in track_scores:
                if rec.video_id != video_id or rec.predicted_label_source is None:
                    continue
                counts[rec.predicted_label_source] = counts.get(rec.predicted_label_source, 0) + 1
            row["sinkhorn_c43_enabled"] = True
            row["sinkhorn_c43_active_special_columns"] = list(sinkhorn_diag.get("active_special_columns", []))
            row["sinkhorn_c43_num_tracks_observed"] = int(counts.get("observed", 0))
            row["sinkhorn_c43_num_tracks_bg"] = int(counts.get("bg", 0))
            row["sinkhorn_c43_num_tracks_unk_fg"] = int(counts.get("unk_fg", 0))
            row["sinkhorn_c43_mass_observed_total"] = sinkhorn_diag.get("c43_mass_observed_total")
            row["sinkhorn_c43_mass_bg_total"] = sinkhorn_diag.get("c43_mass_bg_total")
            row["sinkhorn_c43_mass_unk_fg_total"] = sinkhorn_diag.get("c43_mass_unk_fg_total")

    decoder_total_bg = int(sum(int(diag["decoder_num_tracks_bg"]) for diag in decoder_diag_by_video.values()))
    decoder_total_fg = int(sum(int(diag["decoder_num_tracks_fg"]) for diag in decoder_diag_by_video.values()))
    decoder_coverage_target = int(sum(int(diag["decoder_coverage_target_count"]) for diag in decoder_diag_by_video.values()))
    decoder_coverage_hit = int(sum(int(diag["decoder_coverage_hit_count"]) for diag in decoder_diag_by_video.values()))
    decoder_tie_break_total = int(sum(int(diag["decoder_tie_break_count"]) for diag in decoder_diag_by_video.values()))
    decoder_coverage_skip_fg_min_total = int(
        sum(int(diag["decoder_coverage_skip_fg_min_count"]) for diag in decoder_diag_by_video.values())
    )
    decoder_coverage_skip_bg_gate_total = int(
        sum(int(diag["decoder_coverage_skip_bg_gate_count"]) for diag in decoder_diag_by_video.values())
    )
    decoder_fill_fg_total = int(sum(int(diag["decoder_fill_fg_count"]) for diag in decoder_diag_by_video.values()))
    decoder_fill_bg_total = int(sum(int(diag["decoder_fill_bg_count"]) for diag in decoder_diag_by_video.values()))
    decoder_bg_reason_counts: dict[str, int] = {"score_threshold": 0, "min_margin": 0, "fg_score_min": 0, "ot_prob_min": 0}
    for diag in decoder_diag_by_video.values():
        reason_counts = diag.get("decoder_bg_reason_counts")
        if isinstance(reason_counts, dict):
            for key in ("score_threshold", "min_margin", "fg_score_min", "ot_prob_min"):
                decoder_bg_reason_counts[key] += int(reason_counts.get(key, 0))
    decoder_coverage_ratio = float(decoder_coverage_hit / decoder_coverage_target) if decoder_coverage_target > 0 else None
    otlite_col_l1_values: list[float] = []
    otlite_row_l1_values: list[float] = []
    otlite_bg_ot_prob_count = 0
    if decoder_config.backend == "otlite_v1":
        for diag in decoder_diag_by_video.values():
            col_err = diag.get("decoder_otlite_col_mass_l1_error")
            row_err = diag.get("decoder_otlite_row_mass_l1_error")
            if isinstance(col_err, (int, float)):
                otlite_col_l1_values.append(float(col_err))
            if isinstance(row_err, (int, float)):
                otlite_row_l1_values.append(float(row_err))
            otlite_bg_ot_prob_count += int(diag.get("decoder_otlite_bg_ot_prob_count", 0))

    policy_version = "coverage_greedy_v1.r1b"
    if decoder_config.backend == "otlite_v1":
        policy_version = "otlite_v1.r2a"

    sinkhorn_entropy_values: list[float] = []
    sinkhorn_row_l1_values: list[float] = []
    sinkhorn_col_l1_values: list[float] = []
    sinkhorn_converged_videos = 0
    sinkhorn_c43_mass_observed_values: list[float] = []
    sinkhorn_c43_mass_bg_values: list[float] = []
    sinkhorn_c43_mass_unk_fg_values: list[float] = []
    for diag in sinkhorn_diag_by_video.values():
        if bool(diag.get("converged", False)):
            sinkhorn_converged_videos += 1
        entropy = diag.get("posterior_entropy_mean")
        row_l1 = diag.get("row_mass_l1_error")
        col_l1 = diag.get("col_mass_l1_error")
        if isinstance(entropy, (int, float)):
            sinkhorn_entropy_values.append(float(entropy))
        if isinstance(row_l1, (int, float)):
            sinkhorn_row_l1_values.append(float(row_l1))
        if isinstance(col_l1, (int, float)):
            sinkhorn_col_l1_values.append(float(col_l1))
        mass_observed = diag.get("c43_mass_observed_total")
        mass_bg = diag.get("c43_mass_bg_total")
        mass_unk_fg = diag.get("c43_mass_unk_fg_total")
        if isinstance(mass_observed, (int, float)):
            sinkhorn_c43_mass_observed_values.append(float(mass_observed))
        if isinstance(mass_bg, (int, float)):
            sinkhorn_c43_mass_bg_values.append(float(mass_bg))
        if isinstance(mass_unk_fg, (int, float)):
            sinkhorn_c43_mass_unk_fg_values.append(float(mass_unk_fg))

    sinkhorn_summary_policy_version = "sinkhorn_v1.r2" if sinkhorn_c43_config.enable else "sinkhorn_v1.r1"
    c43_source_counts = {"observed": 0, "bg": 0, "unk_fg": 0}
    if sinkhorn_c43_config.enable:
        for rec in track_scores:
            source = rec.predicted_label_source
            if source is None:
                continue
            c43_source_counts[source] = c43_source_counts.get(source, 0) + 1

    run_summary = _build_run_summary(
        split_view=split_view,
        track_scores=track_scores,
        all_videos=all_videos,
        expected_track_keys=expected_track_keys,
        emitted_keys=emitted_keys,
        scorer_backend="sinkhorn_v1",
        extra={
            "prototype_inventory": {
                "schema_name": prototype_inventory.schema_name,
                "schema_version": prototype_inventory.schema_version,
                "num_labels": len(prototype_inventory.labels_by_key),
                "embedding_dim": prototype_inventory.embedding_dim,
                "dtype": prototype_inventory.dtype,
                "array_key": prototype_inventory.array_key,
            },
            "labelset_coverage": {
                "videos_with_nonempty_labelset": videos_with_nonempty_labelset,
                "videos_missing_labelset": sorted(videos_missing_labelset),
                "num_videos_missing_labelset": len(videos_missing_labelset),
                "num_videos_using_fallback_label_pool": videos_using_fallback,
                "num_tracks_scored_with_fallback_label_pool": fallback_tracks_total,
                "empty_labelset_policy": empty_labelset_policy,
            },
            "label_conditioned_score_distribution": _as_stats([r.score for r in track_scores]),
            "decoder_backend": decoder_config.backend,
            "decoder_summary": {
                "num_tracks_fg": decoder_total_fg,
                "num_tracks_bg": decoder_total_bg,
                "coverage_target_count": decoder_coverage_target,
                "coverage_hit_count": decoder_coverage_hit,
                "coverage_ratio": decoder_coverage_ratio,
                "tie_break_count": decoder_tie_break_total,
                "coverage_skip_fg_min_count": decoder_coverage_skip_fg_min_total,
                "coverage_skip_bg_gate_count": decoder_coverage_skip_bg_gate_total,
                "fill_fg_count": decoder_fill_fg_total,
                "fill_bg_count": decoder_fill_bg_total,
                "bg_reason_counts": decoder_bg_reason_counts,
                "fg_score_min": float(decoder_config.fg_score_min),
                "bg_score_threshold": decoder_config.bg_score_threshold,
                "bg_min_margin": decoder_config.bg_min_margin,
                "policy_version": policy_version,
            },
            "sinkhorn_summary": {
                "temperature": float(sinkhorn_config.temperature),
                "iterations": int(sinkhorn_config.iterations),
                "tolerance": float(sinkhorn_config.tolerance),
                "eps": float(sinkhorn_config.eps),
                "videos_converged": int(sinkhorn_converged_videos),
                "videos_total": int(len(sinkhorn_diag_by_video)),
                "posterior_entropy_mean": (
                    float(np.mean(np.asarray(sinkhorn_entropy_values, dtype=np.float64))) if sinkhorn_entropy_values else None
                ),
                "posterior_entropy_max": max(sinkhorn_entropy_values) if sinkhorn_entropy_values else None,
                "row_mass_l1_error_mean": (
                    float(np.mean(np.asarray(sinkhorn_row_l1_values, dtype=np.float64))) if sinkhorn_row_l1_values else None
                ),
                "row_mass_l1_error_max": max(sinkhorn_row_l1_values) if sinkhorn_row_l1_values else None,
                "col_mass_l1_error_mean": (
                    float(np.mean(np.asarray(sinkhorn_col_l1_values, dtype=np.float64))) if sinkhorn_col_l1_values else None
                ),
                "col_mass_l1_error_max": max(sinkhorn_col_l1_values) if sinkhorn_col_l1_values else None,
                "policy_version": sinkhorn_summary_policy_version,
            },
        },
    )
    if sinkhorn_c43_config.enable:
        run_summary["sinkhorn_summary"]["c43_enabled"] = True
        run_summary["sinkhorn_summary"]["c43_enable_bg"] = bool(sinkhorn_c43_config.enable_bg)
        run_summary["sinkhorn_summary"]["c43_enable_unk_fg"] = bool(sinkhorn_c43_config.enable_unk_fg)
        run_summary["sinkhorn_summary"]["c43_bg_prior_weight"] = float(sinkhorn_c43_config.bg_prior_weight)
        run_summary["sinkhorn_summary"]["c43_unk_fg_prior_weight"] = float(sinkhorn_c43_config.unk_fg_prior_weight)
        run_summary["sinkhorn_summary"]["c43_active_special_columns"] = [SINKHORN_SPECIAL_BG_LABEL_ID]
        run_summary["sinkhorn_summary"]["c43_num_tracks_observed"] = int(c43_source_counts["observed"])
        run_summary["sinkhorn_summary"]["c43_num_tracks_bg"] = int(c43_source_counts["bg"])
        run_summary["sinkhorn_summary"]["c43_num_tracks_unk_fg"] = int(c43_source_counts["unk_fg"])
        run_summary["sinkhorn_summary"]["c43_mass_observed_total"] = float(sum(sinkhorn_c43_mass_observed_values))
        run_summary["sinkhorn_summary"]["c43_mass_bg_total"] = float(sum(sinkhorn_c43_mass_bg_values))
        run_summary["sinkhorn_summary"]["c43_mass_unk_fg_total"] = float(sum(sinkhorn_c43_mass_unk_fg_values))
    if decoder_config.backend == "otlite_v1":
        run_summary["decoder_summary"]["otlite_temperature"] = float(decoder_config.otlite_temperature)
        run_summary["decoder_summary"]["otlite_iters"] = int(decoder_config.otlite_iters)
        run_summary["decoder_summary"]["otlite_eps"] = float(decoder_config.otlite_eps)
        run_summary["decoder_summary"]["otlite_ot_prob_min"] = decoder_config.otlite_ot_prob_min
        run_summary["decoder_summary"]["otlite_col_mass_l1_error_mean"] = (
            float(np.mean(np.asarray(otlite_col_l1_values, dtype=np.float64))) if otlite_col_l1_values else None
        )
        run_summary["decoder_summary"]["otlite_col_mass_l1_error_max"] = max(otlite_col_l1_values) if otlite_col_l1_values else None
        run_summary["decoder_summary"]["otlite_row_mass_l1_error_mean"] = (
            float(np.mean(np.asarray(otlite_row_l1_values, dtype=np.float64))) if otlite_row_l1_values else None
        )
        run_summary["decoder_summary"]["otlite_row_mass_l1_error_max"] = max(otlite_row_l1_values) if otlite_row_l1_values else None
        run_summary["decoder_summary"]["otlite_bg_ot_prob_count"] = int(otlite_bg_ot_prob_count)

    return StageC1MilResult(
        track_scores=tuple(track_scores),
        per_video_summary=tuple(per_video_summary),
        run_summary=run_summary,
    )


def _build_per_video_summary(
    track_scores: Sequence[StageC1TrackScoreRecord],
    *,
    top_k_per_video: int,
    labelset_size_by_video: Mapping[str, int] | None = None,
    fallback_track_count_by_video: Mapping[str, int] | None = None,
    top_predicted_labels_by_video: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    decoder_diag_by_video: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Dict[str, Any]]:
    by_video: Dict[str, list[StageC1TrackScoreRecord]] = {}
    for record in track_scores:
        by_video.setdefault(record.video_id, []).append(record)

    summaries: list[Dict[str, Any]] = []
    for video_id in sorted(by_video.keys()):
        records = by_video[video_id]
        values = [r.score for r in records]
        top_k = sorted(records, key=lambda r: r.score, reverse=True)[:top_k_per_video]
        row: Dict[str, Any] = {
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
        if labelset_size_by_video is not None:
            row["labelset_size"] = int(labelset_size_by_video.get(video_id, 0))
        if fallback_track_count_by_video is not None:
            row["num_tracks_scored_with_fallback_label_pool"] = int(fallback_track_count_by_video.get(video_id, 0))
        if top_predicted_labels_by_video is not None:
            row["top_predicted_labels"] = list(top_predicted_labels_by_video.get(video_id, ()))
        if decoder_diag_by_video is not None and video_id in decoder_diag_by_video:
            row.update(dict(decoder_diag_by_video[video_id]))
        summaries.append(row)
    return summaries


def _build_run_summary(
    *,
    split_view: Any,
    track_scores: Sequence[StageC1TrackScoreRecord],
    all_videos: Sequence[Any],
    expected_track_keys: Sequence[tuple[str, str | int]],
    emitted_keys: Sequence[tuple[str, str | int]],
    scorer_backend: str,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    scores = [rec.score for rec in track_scores]
    per_video_track_counts: Dict[str, int] = {}
    for rec in track_scores:
        per_video_track_counts[rec.video_id] = per_video_track_counts.get(rec.video_id, 0) + 1

    num_videos_processed_with_tracks = sum(1 for video in all_videos if getattr(video, "status", None) == "processed_with_tracks")
    num_videos_scored_non_empty = len(per_video_track_counts)
    track_count_stats = _as_stats(list(per_video_track_counts.values()))
    score_stats = _as_stats(scores)
    all_finite_scores = bool(score_stats["all_finite"])
    _require(all_finite_scores, "result.track_scores", "must contain only finite score values")

    run_summary: Dict[str, Any] = {
        "split": getattr(split_view, "split", "unknown"),
        "embedding_dim": int(getattr(split_view, "embedding_dim", -1)),
        "num_videos_total": len(all_videos),
        "num_videos_processed_with_tracks": num_videos_processed_with_tracks,
        "num_videos_scored_non_empty": num_videos_scored_non_empty,
        "num_tracks_scored": len(track_scores),
        "score_min": score_stats["min"],
        "score_max": score_stats["max"],
        "score_mean": score_stats["mean"],
        "score_distribution": score_stats,
        "per_video_track_count_distribution": track_count_stats,
        "ordering_identity_checks": {
            "expected_key_count": len(expected_track_keys),
            "emitted_key_count": len(emitted_keys),
            "unique_emitted_key_count": len(set(emitted_keys)),
            "key_order_matches_stagec0": list(emitted_keys) == list(expected_track_keys),
        },
        "scorer_backend": scorer_backend,
    }

    if extra:
        run_summary.update(dict(extra))
    return run_summary


def write_stagec1_mil_artifacts(result: StageC1MilResult, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    track_scores_path = output_dir / "track_scores.jsonl"
    with track_scores_path.open("w", encoding="utf-8") as f:
        for record in result.track_scores:
            row: Dict[str, Any] = {
                "video_id": record.video_id,
                "track_id": record.track_id,
                "row_index": record.row_index,
                "status": record.status,
                "score": record.score,
            }
            if record.predicted_label_id is not None:
                row["predicted_label_id"] = record.predicted_label_id
            if record.used_fallback_label_pool:
                row["used_fallback_label_pool"] = True
            if record.decoder_predicted_label_id is not None:
                row["decoder_predicted_label_id"] = record.decoder_predicted_label_id
            if record.decoder_assigned_bg is not None:
                row["decoder_assigned_bg"] = bool(record.decoder_assigned_bg)
            if record.decoder_assignment_source is not None:
                row["decoder_assignment_source"] = record.decoder_assignment_source
            if record.decoder_score is not None:
                row["decoder_score"] = float(record.decoder_score)
            if record.decoder_margin is not None:
                row["decoder_margin"] = float(record.decoder_margin)
            if record.decoder_bg_reason is not None:
                row["decoder_bg_reason"] = record.decoder_bg_reason
            if record.decoder_ot_prob is not None:
                row["decoder_ot_prob"] = float(record.decoder_ot_prob)
            if record.predicted_label_source is not None:
                row["predicted_label_source"] = record.predicted_label_source
            if record.sinkhorn_active_special_columns is not None:
                row["sinkhorn_active_special_columns"] = list(record.sinkhorn_active_special_columns)
            if record.sinkhorn_bg_posterior is not None:
                row["sinkhorn_bg_posterior"] = float(record.sinkhorn_bg_posterior)
            if record.sinkhorn_top_observed_score is not None:
                row["sinkhorn_top_observed_score"] = float(record.sinkhorn_top_observed_score)
            f.write(json.dumps(row, sort_keys=True) + "\n")

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
    scorer_backend: str = "mil_v1",
    decoder_backend: str = "independent",
    decoder_fg_score_min: float = -1.0,
    decoder_bg_score_threshold: float | None = None,
    decoder_bg_min_margin: float | None = None,
    decoder_otlite_temperature: float = 0.10,
    decoder_otlite_iters: int = 8,
    decoder_otlite_eps: float = 1e-12,
    decoder_otlite_ot_prob_min: float | None = None,
    labelset_json: Path | None = None,
    prototype_manifest_json: Path | None = None,
    labelset_key: str = "label_set_observed_ids",
    empty_labelset_policy: str = "use_all_prototypes",
    em_temperature: float = 0.10,
    em_iterations: int = 5,
    em_prior_alpha: float = 1.0,
    em_eps: float = 1e-12,
    sinkhorn_temperature: float = 0.10,
    sinkhorn_iterations: int = 12,
    sinkhorn_tolerance: float = 1e-6,
    sinkhorn_eps: float = 1e-12,
    sinkhorn_c43_enable: bool = False,
    sinkhorn_c43_enable_bg: bool = False,
    sinkhorn_c43_enable_unk_fg: bool = False,
    sinkhorn_c43_bg_prior_weight: float = 0.0,
    sinkhorn_c43_unk_fg_prior_weight: float = 0.0,
    sinkhorn_c43_unk_fg_min_top_obs_score: float | None = None,
    sinkhorn_c43_unk_fg_max_top_obs_score: float | None = None,
    sinkhorn_c43_bg_score: float = 0.0,
) -> Dict[str, Any]:
    _require(scorer_backend in SCORER_BACKENDS, "scorer_backend", f"must be one of {sorted(SCORER_BACKENDS)}")
    decoder_config = StageC1DecoderConfig(
        backend=decoder_backend,
        fg_score_min=decoder_fg_score_min,
        bg_score_threshold=decoder_bg_score_threshold,
        bg_min_margin=decoder_bg_min_margin,
        otlite_temperature=decoder_otlite_temperature,
        otlite_iters=decoder_otlite_iters,
        otlite_eps=decoder_otlite_eps,
        otlite_ot_prob_min=decoder_otlite_ot_prob_min,
    )
    _validate_decoder_config(decoder_config)
    em_config = StageC1EmConfig(
        temperature=em_temperature,
        iterations=em_iterations,
        prior_alpha=em_prior_alpha,
        eps=em_eps,
    )
    _validate_em_config(em_config)
    sinkhorn_config = StageC1SinkhornConfig(
        temperature=sinkhorn_temperature,
        iterations=sinkhorn_iterations,
        tolerance=sinkhorn_tolerance,
        eps=sinkhorn_eps,
    )
    _validate_sinkhorn_config(sinkhorn_config)
    sinkhorn_c43_config = StageC1SinkhornC43Config(
        enable=sinkhorn_c43_enable,
        enable_bg=sinkhorn_c43_enable_bg,
        enable_unk_fg=sinkhorn_c43_enable_unk_fg,
        bg_prior_weight=sinkhorn_c43_bg_prior_weight,
        unk_fg_prior_weight=sinkhorn_c43_unk_fg_prior_weight,
        unk_fg_min_top_obs_score=sinkhorn_c43_unk_fg_min_top_obs_score,
        unk_fg_max_top_obs_score=sinkhorn_c43_unk_fg_max_top_obs_score,
        bg_score=sinkhorn_c43_bg_score,
    )
    _validate_sinkhorn_c43_config(sinkhorn_c43_config)
    if scorer_backend == "mil_v1":
        _require(
            decoder_config.backend == "independent",
            "decoder_backend",
            "must be 'independent' when scorer_backend='mil_v1'",
        )

    split_view = load_stageb_export_split_v1(split_root=split_root, eager_validate=eager_validate)

    if scorer_backend == "mil_v1":
        result = compute_stagec1_mil_baseline_scores(split_view, config=config)
        run_summary = dict(result.run_summary)
        run_summary["decoder_backend"] = decoder_config.backend
        run_summary["decoder_summary"] = {
            "num_tracks_fg": 0,
            "num_tracks_bg": 0,
            "coverage_target_count": 0,
            "coverage_hit_count": 0,
            "coverage_ratio": None,
            "tie_break_count": 0,
            "coverage_skip_fg_min_count": 0,
            "coverage_skip_bg_gate_count": 0,
            "fill_fg_count": 0,
            "fill_bg_count": 0,
            "bg_reason_counts": {"score_threshold": 0, "min_margin": 0, "fg_score_min": 0, "ot_prob_min": 0},
            "fg_score_min": float(decoder_config.fg_score_min),
            "bg_score_threshold": decoder_config.bg_score_threshold,
            "bg_min_margin": decoder_config.bg_min_margin,
            "policy_version": "coverage_greedy_v1.r1b",
        }
        per_video_summary = []
        for row in result.per_video_summary:
            row_copy = dict(row)
            row_copy["decoder_backend"] = decoder_config.backend
            row_copy["decoder_num_tracks_fg"] = 0
            row_copy["decoder_num_tracks_bg"] = 0
            row_copy["decoder_coverage_target_count"] = 0
            row_copy["decoder_coverage_hit_count"] = 0
            row_copy["decoder_coverage_ratio"] = None
            row_copy["decoder_tie_break_count"] = 0
            row_copy["decoder_coverage_skip_fg_min_count"] = 0
            row_copy["decoder_coverage_skip_bg_gate_count"] = 0
            row_copy["decoder_fill_fg_count"] = 0
            row_copy["decoder_fill_bg_count"] = 0
            row_copy["decoder_bg_reason_counts"] = {"score_threshold": 0, "min_margin": 0, "fg_score_min": 0, "ot_prob_min": 0}
            row_copy["decoder_policy_version"] = "coverage_greedy_v1.r1b"
            per_video_summary.append(row_copy)
        result = StageC1MilResult(
            track_scores=result.track_scores,
            per_video_summary=tuple(per_video_summary),
            run_summary=run_summary,
        )
    elif scorer_backend == "labelset_proto_v1":
        _require(labelset_json is not None, "labelset_json", "required when scorer_backend='labelset_proto_v1'")
        _require(
            prototype_manifest_json is not None,
            "prototype_manifest_json",
            "required when scorer_backend='labelset_proto_v1'",
        )
        inventory = load_stagec_label_prototype_inventory_v1(prototype_manifest_json)
        labelset_lookup = load_stagec_labelset_lookup(labelset_json, labelset_key=labelset_key)
        result = compute_stagec1_labelset_proto_baseline_scores(
            split_view,
            prototype_inventory=inventory,
            labelset_lookup=labelset_lookup,
            config=config,
            empty_labelset_policy=empty_labelset_policy,
            decoder_config=decoder_config,
        )
    elif scorer_backend == "em_v1":
        _require(labelset_json is not None, "labelset_json", "required when scorer_backend='em_v1'")
        _require(
            prototype_manifest_json is not None,
            "prototype_manifest_json",
            "required when scorer_backend='em_v1'",
        )
        inventory = load_stagec_label_prototype_inventory_v1(prototype_manifest_json)
        labelset_lookup = load_stagec_labelset_lookup(labelset_json, labelset_key=labelset_key)
        result = compute_stagec1_em_v1_scores(
            split_view,
            prototype_inventory=inventory,
            labelset_lookup=labelset_lookup,
            config=config,
            empty_labelset_policy=empty_labelset_policy,
            decoder_config=decoder_config,
            em_config=em_config,
        )
    else:
        _require(labelset_json is not None, "labelset_json", "required when scorer_backend='sinkhorn_v1'")
        _require(
            prototype_manifest_json is not None,
            "prototype_manifest_json",
            "required when scorer_backend='sinkhorn_v1'",
        )
        inventory = load_stagec_label_prototype_inventory_v1(prototype_manifest_json)
        labelset_lookup = load_stagec_labelset_lookup(labelset_json, labelset_key=labelset_key)
        result = compute_stagec1_sinkhorn_v1_scores(
            split_view,
            prototype_inventory=inventory,
            labelset_lookup=labelset_lookup,
            config=config,
            empty_labelset_policy=empty_labelset_policy,
            decoder_config=decoder_config,
            sinkhorn_config=sinkhorn_config,
            sinkhorn_c43_config=sinkhorn_c43_config,
        )

    artifact_paths = write_stagec1_mil_artifacts(result=result, output_dir=output_dir)
    return {
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "run_summary": result.run_summary,
    }
