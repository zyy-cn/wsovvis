from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np

from .v1_core import (
    ALL_STATUSES,
    MANIFEST_SCHEMA_NAME,
    PROCESSED_STATUSES,
    VIDEO_SCHEMA_NAME,
    ExportContractError,
)


class StageCExportLoadError(ExportContractError):
    """Raised when loading Stage B export artifact v1 for Stage C fails."""


@dataclass(frozen=True)
class StageCVideoRecord:
    video_id: str
    status: str
    num_tracks: int
    track_metadata_path: Optional[str]
    track_arrays_path: Optional[str]


@dataclass(frozen=True)
class StageCTrackMetadata:
    video_id: str
    row_index: int
    track_id: str | int
    start_frame_idx: int
    end_frame_idx: int
    num_active_frames: int
    objectness_score: float


@dataclass(frozen=True)
class StageCTrackRecord:
    metadata: StageCTrackMetadata
    embedding: np.ndarray


@dataclass
class _LoadedVideoPayload:
    track_rows: Tuple[StageCTrackMetadata, ...]
    track_index_by_id: Mapping[str | int, int]
    embeddings: np.ndarray


def _err(field_path: str, rule_summary: str) -> StageCExportLoadError:
    return StageCExportLoadError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path=field_path, rule_summary=rule_summary)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_json(path: Path, file_label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StageCExportLoadError(f"{file_label} missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StageCExportLoadError(f"{file_label} invalid JSON at {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{file_label}.$", "top-level value must be an object")
    return payload


def _parse_major_version(version: Any, field_path: str) -> int:
    _require(isinstance(version, str), field_path, "must be a string")
    parts = version.split(".")
    _require(len(parts) == 3 and all(part.isdigit() for part in parts), field_path, "must follow MAJOR.MINOR.PATCH")
    return int(parts[0])


def _require_relative_path(path_value: Any, field_path: str) -> str:
    _require(isinstance(path_value, str) and path_value, field_path, "must be a non-empty relative path")
    rel = PurePosixPath(path_value)
    _require(not rel.is_absolute(), field_path, "absolute path is forbidden")
    _require(".." not in rel.parts, field_path, "must not contain '..'")
    return str(rel)


def _discover_manifest_path(split_root: Path) -> Path:
    manifest_v1 = split_root / "manifest.v1.json"
    manifest_compat = split_root / "manifest.json"
    if manifest_v1.exists():
        return manifest_v1
    if manifest_compat.exists():
        return manifest_compat
    raise StageCExportLoadError(
        f"manifest missing under split root {split_root}; expected manifest.v1.json or manifest.json"
    )


class StageCExportSplitView:
    """Read-only offline view for Stage B export artifact v1 split data."""

    def __init__(
        self,
        *,
        split_root: Path,
        manifest_path: Path,
        manifest: Mapping[str, Any],
        eager_validate: bool,
    ) -> None:
        self.split_root = split_root
        self.manifest_path = manifest_path
        self.manifest: Dict[str, Any] = dict(manifest)

        self.split = str(self.manifest["split"])
        self.embedding_dim = int(self.manifest["embedding_dim"])
        self.embedding_dtype = str(self.manifest["embedding_dtype"])
        self.embedding_pooling = str(self.manifest["embedding_pooling"])
        self.embedding_normalization = str(self.manifest["embedding_normalization"])
        self.producer = dict(self.manifest["producer"])

        self._video_order: Tuple[str, ...] = tuple()
        self._videos_by_id: Dict[str, StageCVideoRecord] = {}
        self._loaded_by_video_id: Dict[str, _LoadedVideoPayload] = {}
        self._split_track_id_type: Optional[type] = None

        self._validate_manifest_and_build_indexes()
        if eager_validate:
            self._eager_validate_processed_videos()

    def _validate_manifest_and_build_indexes(self) -> None:
        _require(
            self.manifest.get("schema_name") == MANIFEST_SCHEMA_NAME,
            "manifest.schema_name",
            f"must equal '{MANIFEST_SCHEMA_NAME}'",
        )
        major = _parse_major_version(self.manifest.get("schema_version"), "manifest.schema_version")
        _require(major == 1, "manifest.schema_version", "unsupported major version")

        _require(isinstance(self.manifest.get("split"), str) and self.manifest["split"], "manifest.split", "must be non-empty")
        _require(
            _is_int(self.manifest.get("embedding_dim")) and self.manifest["embedding_dim"] > 0,
            "manifest.embedding_dim",
            "must be integer > 0",
        )
        _require(
            self.manifest.get("embedding_dtype") == "float32",
            "manifest.embedding_dtype",
            "must equal 'float32'",
        )
        _require(
            self.manifest.get("embedding_pooling") == "track_pooled",
            "manifest.embedding_pooling",
            "must equal 'track_pooled'",
        )
        _require(
            self.manifest.get("embedding_normalization") in {"none", "l2"},
            "manifest.embedding_normalization",
            "must be one of ['none', 'l2']",
        )
        _require(isinstance(self.manifest.get("producer"), dict), "manifest.producer", "must be an object")

        videos = self.manifest.get("videos")
        _require(isinstance(videos, list), "manifest.videos", "must be a list")

        video_order: list[str] = []
        seen_video_ids = set()
        for index, video in enumerate(videos):
            vpath = f"manifest.videos[{index}]"
            _require(isinstance(video, dict), vpath, "must be an object")

            for key in ("video_id", "status", "num_tracks", "track_metadata_path", "track_arrays_path"):
                _require(key in video, f"{vpath}.{key}", "required field missing")

            video_id = video["video_id"]
            _require(isinstance(video_id, str) and video_id, f"{vpath}.video_id", "must be a non-empty string")
            _require(video_id not in seen_video_ids, f"{vpath}.video_id", f"duplicate video_id '{video_id}'")
            seen_video_ids.add(video_id)
            video_order.append(video_id)

            status = video["status"]
            _require(status in ALL_STATUSES, f"{vpath}.status", f"must be one of {sorted(ALL_STATUSES)}")

            num_tracks = video["num_tracks"]
            _require(_is_int(num_tracks) and num_tracks >= 0, f"{vpath}.num_tracks", "must be integer >= 0")

            metadata_path = video["track_metadata_path"]
            arrays_path = video["track_arrays_path"]
            if status == "processed_with_tracks":
                _require(num_tracks > 0, f"{vpath}.num_tracks", "must be > 0 for processed_with_tracks")
                metadata_path = _require_relative_path(metadata_path, f"{vpath}.track_metadata_path")
                arrays_path = _require_relative_path(arrays_path, f"{vpath}.track_arrays_path")
            elif status == "processed_zero_tracks":
                _require(num_tracks == 0, f"{vpath}.num_tracks", "must equal 0 for processed_zero_tracks")
                metadata_path = _require_relative_path(metadata_path, f"{vpath}.track_metadata_path")
                arrays_path = _require_relative_path(arrays_path, f"{vpath}.track_arrays_path")
            else:
                _require(num_tracks == 0, f"{vpath}.num_tracks", "must equal 0 for failed/unprocessed")
                _require(metadata_path is None, f"{vpath}.track_metadata_path", "must be null for failed/unprocessed")
                _require(arrays_path is None, f"{vpath}.track_arrays_path", "must be null for failed/unprocessed")

            self._videos_by_id[video_id] = StageCVideoRecord(
                video_id=video_id,
                status=status,
                num_tracks=num_tracks,
                track_metadata_path=metadata_path,
                track_arrays_path=arrays_path,
            )

        _require(
            video_order == sorted(video_order),
            "manifest.videos",
            "must be lexicographically sorted by video_id for deterministic iteration",
        )
        self._video_order = tuple(video_order)

    def _eager_validate_processed_videos(self) -> None:
        manifest_total_tracks = 0
        loaded_total_tracks = 0
        for video_id in self._video_order:
            video = self._videos_by_id[video_id]
            if video.status in PROCESSED_STATUSES:
                manifest_total_tracks += video.num_tracks
                payload = self._load_processed_video(video)
                loaded_total_tracks += len(payload.track_rows)
        _require(
            loaded_total_tracks == manifest_total_tracks,
            "manifest.videos",
            "processed track counts mismatch between manifest and loaded shards",
        )

    def iter_videos(self, include_statuses: Tuple[str, ...] | None = None) -> Iterator[StageCVideoRecord]:
        if include_statuses is None:
            accepted_statuses = None
        else:
            accepted_statuses = set(include_statuses)
            unknown = accepted_statuses - ALL_STATUSES
            _require(not unknown, "iter_videos.include_statuses", f"contains unknown statuses: {sorted(unknown)}")
        for video_id in self._video_order:
            video = self._videos_by_id[video_id]
            if accepted_statuses is not None and video.status not in accepted_statuses:
                continue
            yield video

    def iter_tracks(self, video_id: str) -> Iterator[StageCTrackRecord]:
        video = self._get_video(video_id)
        if video.status not in PROCESSED_STATUSES:
            return iter(())
        payload = self._load_processed_video(video)
        return (StageCTrackRecord(metadata=meta, embedding=payload.embeddings[meta.row_index]) for meta in payload.track_rows)

    def get_track_metadata(self, video_id: str, track_id: str | int) -> StageCTrackMetadata:
        video = self._get_video(video_id)
        payload = self._load_processed_video(video)
        _require(track_id in payload.track_index_by_id, "track_id", f"unknown track_id '{track_id}' in video '{video_id}'")
        return payload.track_rows[payload.track_index_by_id[track_id]]

    def get_track_embedding(self, video_id: str, track_id: str | int) -> np.ndarray:
        meta = self.get_track_metadata(video_id, track_id)
        return self._load_processed_video(self._videos_by_id[video_id]).embeddings[meta.row_index]

    def get_track_by_index(self, video_id: str, row_index: int) -> StageCTrackRecord:
        _require(_is_int(row_index) and row_index >= 0, "row_index", "must be integer >= 0")
        video = self._get_video(video_id)
        payload = self._load_processed_video(video)
        _require(row_index < len(payload.track_rows), "row_index", f"out of range for video '{video_id}'")
        meta = payload.track_rows[row_index]
        return StageCTrackRecord(metadata=meta, embedding=payload.embeddings[row_index])

    def _get_video(self, video_id: str) -> StageCVideoRecord:
        _require(isinstance(video_id, str) and video_id, "video_id", "must be a non-empty string")
        _require(video_id in self._videos_by_id, "video_id", f"unknown video_id '{video_id}'")
        return self._videos_by_id[video_id]

    def _load_processed_video(self, video: StageCVideoRecord) -> _LoadedVideoPayload:
        _require(video.status in PROCESSED_STATUSES, "video.status", "video is not processed and has no track payload")
        cached = self._loaded_by_video_id.get(video.video_id)
        if cached is not None:
            return cached

        _require(video.track_metadata_path is not None, "track_metadata_path", "required for processed video")
        _require(video.track_arrays_path is not None, "track_arrays_path", "required for processed video")

        metadata_path = self.split_root / Path(video.track_metadata_path)
        arrays_path = self.split_root / Path(video.track_arrays_path)
        _require(metadata_path.exists(), "track_metadata_path", f"target file missing on disk: {metadata_path}")
        _require(arrays_path.exists(), "track_arrays_path", f"target file missing on disk: {arrays_path}")

        metadata_payload = _load_json(metadata_path, "track_metadata.v1.json")
        _require(
            metadata_payload.get("schema_name") == VIDEO_SCHEMA_NAME,
            "track_metadata.v1.json.schema_name",
            f"must equal '{VIDEO_SCHEMA_NAME}'",
        )
        major = _parse_major_version(metadata_payload.get("schema_version"), "track_metadata.v1.json.schema_version")
        _require(major == 1, "track_metadata.v1.json.schema_version", "unsupported major version")

        _require(metadata_payload.get("split") == self.split, "track_metadata.v1.json.split", "must match manifest split")
        _require(
            metadata_payload.get("video_id") == video.video_id,
            "track_metadata.v1.json.video_id",
            "must match manifest video_id",
        )
        _require(
            _is_int(metadata_payload.get("num_tracks")) and metadata_payload["num_tracks"] >= 0,
            "track_metadata.v1.json.num_tracks",
            "must be integer >= 0",
        )
        _require(
            metadata_payload["num_tracks"] == video.num_tracks,
            "track_metadata.v1.json.num_tracks",
            "must match manifest num_tracks",
        )

        tracks = metadata_payload.get("tracks")
        _require(isinstance(tracks, list), "track_metadata.v1.json.tracks", "must be a list")
        _require(
            len(tracks) == video.num_tracks,
            "track_metadata.v1.json.tracks",
            "len(tracks) must match manifest num_tracks",
        )

        track_rows: list[StageCTrackMetadata] = []
        track_index_by_id: Dict[str | int, int] = {}
        for row_index, track in enumerate(tracks):
            tpath = f"track_metadata.v1.json.tracks[{row_index}]"
            _require(isinstance(track, dict), tpath, "must be an object")
            for key in (
                "row_index",
                "track_id",
                "start_frame_idx",
                "end_frame_idx",
                "num_active_frames",
                "objectness_score",
            ):
                _require(key in track, f"{tpath}.{key}", "required field missing")

            _require(
                _is_int(track["row_index"]) and track["row_index"] == row_index,
                f"{tpath}.row_index",
                "must equal row position",
            )
            track_id = track["track_id"]
            _require(
                (isinstance(track_id, str) and track_id) or _is_int(track_id),
                f"{tpath}.track_id",
                "must be non-empty string or integer",
            )
            if self._split_track_id_type is None:
                self._split_track_id_type = type(track_id)
            _require(
                type(track_id) is self._split_track_id_type,
                f"{tpath}.track_id",
                "serialized type must be consistent within split",
            )
            _require(track_id not in track_index_by_id, f"{tpath}.track_id", f"duplicate track_id '{track_id}' within video")

            _require(
                _is_int(track["start_frame_idx"]) and track["start_frame_idx"] >= 0,
                f"{tpath}.start_frame_idx",
                "must be integer >= 0",
            )
            _require(
                _is_int(track["end_frame_idx"]) and track["end_frame_idx"] >= track["start_frame_idx"],
                f"{tpath}.end_frame_idx",
                "must be integer >= start_frame_idx",
            )
            _require(
                _is_int(track["num_active_frames"]) and track["num_active_frames"] > 0,
                f"{tpath}.num_active_frames",
                "must be integer > 0",
            )
            _require(_is_number(track["objectness_score"]), f"{tpath}.objectness_score", "must be a number")

            track_index_by_id[track_id] = row_index
            track_rows.append(
                StageCTrackMetadata(
                    video_id=video.video_id,
                    row_index=row_index,
                    track_id=track_id,
                    start_frame_idx=int(track["start_frame_idx"]),
                    end_frame_idx=int(track["end_frame_idx"]),
                    num_active_frames=int(track["num_active_frames"]),
                    objectness_score=float(track["objectness_score"]),
                )
            )

        try:
            with np.load(arrays_path, allow_pickle=False) as npz_data:
                npz_keys = set(npz_data.files)
                _require(
                    {"embeddings", "track_row_index"}.issubset(npz_keys),
                    "track_arrays.v1.npz.keys",
                    "missing required keys ['embeddings', 'track_row_index']",
                )
                embeddings = np.asarray(npz_data["embeddings"])
                track_row_index = np.asarray(npz_data["track_row_index"])
        except Exception as exc:
            raise StageCExportLoadError(f"track_arrays.v1.npz failed to load at {arrays_path}: {exc}") from exc

        _require(embeddings.dtype == np.float32, "track_arrays.v1.npz.embeddings.dtype", "must be float32")
        _require(embeddings.ndim == 2, "track_arrays.v1.npz.embeddings.shape", "must be rank-2 [N, D]")
        _require(
            embeddings.shape[1] == self.embedding_dim,
            "track_arrays.v1.npz.embeddings.shape[1]",
            "must equal manifest embedding_dim",
        )
        _require(track_row_index.dtype == np.int64, "track_arrays.v1.npz.track_row_index.dtype", "must be int64")
        _require(track_row_index.ndim == 1, "track_arrays.v1.npz.track_row_index.shape", "must be rank-1 [N]")
        _require(
            embeddings.shape[0] == video.num_tracks,
            "track_arrays.v1.npz.embeddings.shape[0]",
            "must equal manifest num_tracks",
        )
        _require(
            track_row_index.shape[0] == video.num_tracks,
            "track_arrays.v1.npz.track_row_index.shape[0]",
            "must equal manifest num_tracks",
        )
        _require(
            np.array_equal(track_row_index, np.arange(video.num_tracks, dtype=np.int64)),
            "track_arrays.v1.npz.track_row_index",
            "must equal [0..N-1]",
        )
        _require(
            len(track_rows) == embeddings.shape[0],
            "track_arrays.v1.npz.embeddings.shape[0]",
            "must equal track_metadata row count",
        )
        _require(np.isfinite(embeddings).all(), "track_arrays.v1.npz.embeddings", "must contain only finite values")

        embeddings.setflags(write=False)
        payload = _LoadedVideoPayload(
            track_rows=tuple(track_rows),
            track_index_by_id=track_index_by_id,
            embeddings=embeddings,
        )
        self._loaded_by_video_id[video.video_id] = payload
        return payload


def load_stageb_export_split_v1(split_root: Path, *, eager_validate: bool = True) -> StageCExportSplitView:
    split_root = split_root.resolve()
    _require(split_root.exists() and split_root.is_dir(), "split_root", f"must be an existing directory: {split_root}")
    manifest_path = _discover_manifest_path(split_root)
    manifest = _load_json(manifest_path, manifest_path.name)
    return StageCExportSplitView(
        split_root=split_root,
        manifest_path=manifest_path,
        manifest=manifest,
        eager_validate=eager_validate,
    )
