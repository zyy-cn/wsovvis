from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wsovvis.track_feature_export import load_stageb_export_split_v1

SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_NAME = "wsovvis.global_track_bank"
VIDEO_SCHEMA_NAME = "wsovvis.global_track_bank_video"
TRACE_SCHEMA_NAME = "wsovvis.global_track_bank_trace"
PROCESSED_STATUSES = {"processed_with_tracks", "processed_zero_tracks"}
ALL_STATUSES = {"processed_with_tracks", "processed_zero_tracks", "failed", "unprocessed"}


class GlobalTrackBankError(RuntimeError):
    """Raised when building or loading the global-track-bank artifact fails."""


@dataclass(frozen=True)
class StitchingConfig:
    weight_temporal_iou: float = 0.5
    weight_query_cosine: float = 0.5
    min_temporal_iou: float = 1.0
    min_query_cosine: float = 0.995
    min_match_score: Optional[float] = None

    def canonical_dict(self) -> Dict[str, Any]:
        _require(self.weight_temporal_iou >= 0.0, "stitching_policy.weight_temporal_iou", "must be >= 0")
        _require(self.weight_query_cosine >= 0.0, "stitching_policy.weight_query_cosine", "must be >= 0")
        total_weight = self.weight_temporal_iou + self.weight_query_cosine
        _require(total_weight > 0.0, "stitching_policy.weights", "sum of weights must be > 0")
        _require(0.0 <= self.min_temporal_iou <= 1.0, "stitching_policy.min_temporal_iou", "must be in [0, 1]")
        _require(-1.0 <= self.min_query_cosine <= 1.0, "stitching_policy.min_query_cosine", "must be in [-1, 1]")
        min_match_score = self.min_match_score
        if min_match_score is None:
            min_match_score = (self.weight_temporal_iou * self.min_temporal_iou) + (
                self.weight_query_cosine * self.min_query_cosine
            )
        _require(
            -1.0 <= float(min_match_score) <= 1.0,
            "stitching_policy.min_match_score",
            "must be in [-1, 1]",
        )
        return {
            "policy_name": "temporal_interval_iou_plus_local_query_cosine",
            "policy_version": "v9.g3.r1",
            "geometry_signal": "temporal_interval_iou",
            "query_signal": "track_embedding_cosine",
            "weight_temporal_iou": float(self.weight_temporal_iou),
            "weight_query_cosine": float(self.weight_query_cosine),
            "min_temporal_iou": float(self.min_temporal_iou),
            "min_query_cosine": float(self.min_query_cosine),
            "min_match_score": float(min_match_score),
        }


@dataclass(frozen=True)
class GlobalTrackVideoRecord:
    video_id: str
    status: str
    num_local_tracklets: int
    num_global_tracks: int
    global_track_metadata_path: Optional[str]
    global_track_arrays_path: Optional[str]
    stitching_trace_path: Optional[str]


@dataclass(frozen=True)
class GlobalTrackMetadata:
    video_id: str
    row_index: int
    global_track_id: int
    start_frame_idx: int
    end_frame_idx: int
    num_active_frames: int
    objectness_score_mean: float
    objectness_score_max: float
    member_count: int
    member_local_row_indices: Tuple[int, ...]
    member_track_ids: Tuple[str | int, ...]


@dataclass(frozen=True)
class GlobalTrackRecord:
    metadata: GlobalTrackMetadata
    embedding: np.ndarray


@dataclass(frozen=True)
class _CandidateEdge:
    left_row_index: int
    right_row_index: int
    temporal_iou: float
    query_cosine: float
    match_score: float


@dataclass
class _LoadedVideoPayload:
    track_rows: Tuple[GlobalTrackMetadata, ...]
    track_index_by_id: Mapping[int, int]
    embeddings: np.ndarray


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, value: int) -> int:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        left_rank = self._rank[left_root]
        right_rank = self._rank[right_root]
        if left_rank < right_rank:
            self._parent[left_root] = right_root
            return
        if left_rank > right_rank:
            self._parent[right_root] = left_root
            return
        self._parent[right_root] = left_root
        self._rank[left_root] += 1


def _err(field_path: str, rule_summary: str) -> GlobalTrackBankError:
    return GlobalTrackBankError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path=field_path, rule_summary=rule_summary)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path, file_label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GlobalTrackBankError(f"{file_label} missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GlobalTrackBankError(f"{file_label} invalid JSON at {path}: {exc}") from exc
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
    raise GlobalTrackBankError(
        f"manifest missing under split root {split_root}; expected manifest.v1.json or manifest.json"
    )


def _temporal_interval_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b) + 1)
    if overlap <= 0:
        return 0.0
    len_a = end_a - start_a + 1
    len_b = end_b - start_b + 1
    union = len_a + len_b - overlap
    if union <= 0:
        return 0.0
    return float(overlap / union)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denom == 0.0:
        return 0.0
    score = float(np.dot(left64, right64) / denom)
    return max(-1.0, min(1.0, score))


def _component_sort_key(members: Sequence[Any]) -> Tuple[int, int, int]:
    min_start = min(int(track.metadata.start_frame_idx) for track in members)
    max_end = max(int(track.metadata.end_frame_idx) for track in members)
    min_row = min(int(track.metadata.row_index) for track in members)
    return (min_start, max_end, min_row)


def _format_track_id_list(values: Sequence[str | int]) -> List[str | int]:
    return [value for value in values]


def _compute_pairwise_structure(
    local_tracks: Sequence[Any],
    policy: Mapping[str, Any],
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[_CandidateEdge]]:
    count = len(local_tracks)
    temporal_iou_matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    query_cosine_matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    match_score_matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    candidate_edges: List[_CandidateEdge] = []
    for left_idx in range(count):
        left = local_tracks[left_idx]
        temporal_iou_matrix[left_idx][left_idx] = 1.0
        query_cosine_matrix[left_idx][left_idx] = 1.0
        match_score_matrix[left_idx][left_idx] = 1.0
        for right_idx in range(left_idx + 1, count):
            right = local_tracks[right_idx]
            temporal_iou = _temporal_interval_iou(
                left.metadata.start_frame_idx,
                left.metadata.end_frame_idx,
                right.metadata.start_frame_idx,
                right.metadata.end_frame_idx,
            )
            query_cosine = _cosine_similarity(left.embedding, right.embedding)
            match_score = (
                (float(policy["weight_temporal_iou"]) * temporal_iou)
                + (float(policy["weight_query_cosine"]) * query_cosine)
            )
            temporal_iou_matrix[left_idx][right_idx] = temporal_iou_matrix[right_idx][left_idx] = temporal_iou
            query_cosine_matrix[left_idx][right_idx] = query_cosine_matrix[right_idx][left_idx] = query_cosine
            match_score_matrix[left_idx][right_idx] = match_score_matrix[right_idx][left_idx] = match_score
            if (
                temporal_iou >= float(policy["min_temporal_iou"])
                and query_cosine >= float(policy["min_query_cosine"])
                and match_score >= float(policy["min_match_score"])
            ):
                candidate_edges.append(
                    _CandidateEdge(
                        left_row_index=left_idx,
                        right_row_index=right_idx,
                        temporal_iou=temporal_iou,
                        query_cosine=query_cosine,
                        match_score=match_score,
                    )
                )
    candidate_edges.sort(
        key=lambda edge: (
            -edge.match_score,
            -edge.temporal_iou,
            -edge.query_cosine,
            edge.left_row_index,
            edge.right_row_index,
        )
    )
    return temporal_iou_matrix, query_cosine_matrix, match_score_matrix, candidate_edges


def _build_components(
    local_tracks: Sequence[Any],
    candidate_edges: Sequence[_CandidateEdge],
) -> List[List[Any]]:
    uf = _UnionFind(len(local_tracks))
    for edge in candidate_edges:
        uf.union(edge.left_row_index, edge.right_row_index)
    buckets: Dict[int, List[Any]] = {}
    for row_index, track in enumerate(local_tracks):
        buckets.setdefault(uf.find(row_index), []).append(track)
    components = [sorted(members, key=lambda tr: int(tr.metadata.row_index)) for members in buckets.values()]
    components.sort(key=_component_sort_key)
    return components


def _build_video_artifacts(
    *,
    split_root: Path,
    split: str,
    video: Any,
    local_tracks: Sequence[Any],
    embedding_dim: int,
    embedding_dtype: str,
    policy: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    temporal_iou_matrix, query_cosine_matrix, match_score_matrix, candidate_edges = _compute_pairwise_structure(
        local_tracks=local_tracks,
        policy=policy,
    )
    components = _build_components(local_tracks=local_tracks, candidate_edges=candidate_edges)

    global_tracks_payload: List[Dict[str, Any]] = []
    global_embeddings: List[np.ndarray] = []
    merge_components: List[Dict[str, Any]] = []
    merged_local_tracklets = 0
    for row_index, members in enumerate(components):
        member_rows = tuple(int(track.metadata.row_index) for track in members)
        member_track_ids = tuple(track.metadata.track_id for track in members)
        start_frame_idx = min(int(track.metadata.start_frame_idx) for track in members)
        end_frame_idx = max(int(track.metadata.end_frame_idx) for track in members)
        objectness_values = [float(track.metadata.objectness_score) for track in members]
        num_active_frames = end_frame_idx - start_frame_idx + 1
        if len(member_rows) > 1:
            merged_local_tracklets += len(member_rows)
        global_embeddings.append(np.mean(np.stack([track.embedding for track in members], axis=0), axis=0))
        global_tracks_payload.append(
            {
                "row_index": row_index,
                "global_track_id": row_index,
                "start_frame_idx": start_frame_idx,
                "end_frame_idx": end_frame_idx,
                "num_active_frames": num_active_frames,
                "objectness_score_mean": float(sum(objectness_values) / len(objectness_values)),
                "objectness_score_max": float(max(objectness_values)),
                "member_count": len(member_rows),
                "member_local_row_indices": list(member_rows),
                "member_track_ids": _format_track_id_list(member_track_ids),
            }
        )
        merge_components.append(
            {
                "row_index": row_index,
                "global_track_id": row_index,
                "member_local_row_indices": list(member_rows),
                "member_track_ids": _format_track_id_list(member_track_ids),
            }
        )

    video_root = split_root / "videos" / video.video_id
    metadata_rel = PurePosixPath("videos") / video.video_id / "global_track_metadata.v1.json"
    arrays_rel = PurePosixPath("videos") / video.video_id / "global_track_arrays.v1.npz"
    trace_rel = PurePosixPath("videos") / video.video_id / "stitching_trace.v1.json"

    metadata_payload = {
        "schema_name": VIDEO_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "video_id": video.video_id,
        "source_status": video.status,
        "num_local_tracklets": len(local_tracks),
        "num_global_tracks": len(global_tracks_payload),
        "global_tracks": global_tracks_payload,
    }
    _dump_json(video_root / "global_track_metadata.v1.json", metadata_payload)

    if global_embeddings:
        embeddings = np.stack(global_embeddings, axis=0).astype(np.float32, copy=False)
    else:
        embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
    global_track_row_index = np.arange(len(global_tracks_payload), dtype=np.int64)
    np.savez(
        video_root / "global_track_arrays.v1.npz",
        global_embeddings=embeddings,
        global_track_row_index=global_track_row_index,
    )

    trace_payload = {
        "schema_name": TRACE_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "video_id": video.video_id,
        "source_status": video.status,
        "stitching_policy": dict(policy),
        "num_local_tracklets": len(local_tracks),
        "num_global_tracks": len(global_tracks_payload),
        "local_tracks": [
            {
                "row_index": int(track.metadata.row_index),
                "track_id": track.metadata.track_id,
                "start_frame_idx": int(track.metadata.start_frame_idx),
                "end_frame_idx": int(track.metadata.end_frame_idx),
                "num_active_frames": int(track.metadata.num_active_frames),
                "objectness_score": float(track.metadata.objectness_score),
            }
            for track in local_tracks
        ],
        "temporal_iou_matrix": temporal_iou_matrix,
        "query_cosine_matrix": query_cosine_matrix,
        "match_score_matrix": match_score_matrix,
        "candidate_edges": [
            {
                "left_row_index": edge.left_row_index,
                "right_row_index": edge.right_row_index,
                "temporal_iou": edge.temporal_iou,
                "query_cosine": edge.query_cosine,
                "match_score": edge.match_score,
            }
            for edge in candidate_edges
        ],
        "merge_components": merge_components,
        "merge_diagnostics": {
            "candidate_pair_count": len(candidate_edges),
            "merge_component_count": sum(1 for members in components if len(members) > 1),
            "merged_local_tracklets": merged_local_tracklets,
            "fragmentation_reduction": len(local_tracks) - len(global_tracks_payload),
        },
    }
    _dump_json(video_root / "stitching_trace.v1.json", trace_payload)

    manifest_row = {
        "video_id": video.video_id,
        "status": video.status,
        "num_local_tracklets": len(local_tracks),
        "num_global_tracks": len(global_tracks_payload),
        "global_track_metadata_path": str(metadata_rel),
        "global_track_arrays_path": str(arrays_rel),
        "stitching_trace_path": str(trace_rel),
    }
    return manifest_row, metadata_payload, embeddings, global_track_row_index


def build_global_track_bank_v9(
    source_split_root: Path,
    output_split_root: Path,
    *,
    overwrite: bool = False,
    config: StitchingConfig | None = None,
) -> Path:
    split_view = load_stageb_export_split_v1(Path(source_split_root), eager_validate=True)
    output_split_root = Path(output_split_root).resolve()
    if output_split_root.exists() and not overwrite:
        raise GlobalTrackBankError(f"producer output path already exists: {output_split_root}. Use overwrite=True to replace it.")

    canonical_config = (config or StitchingConfig()).canonical_dict()
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"global_track_bank_v9_{split_view.split}_", dir=str(output_split_root.parent))
    )
    try:
        manifest_videos: List[Dict[str, Any]] = []
        for video in split_view.iter_videos():
            if video.status in PROCESSED_STATUSES:
                local_tracks = list(split_view.iter_tracks(video.video_id))
                manifest_row, _, _, _ = _build_video_artifacts(
                    split_root=temp_dir,
                    split=split_view.split,
                    video=video,
                    local_tracks=local_tracks,
                    embedding_dim=split_view.embedding_dim,
                    embedding_dtype=split_view.embedding_dtype,
                    policy=canonical_config,
                )
                manifest_videos.append(manifest_row)
            else:
                manifest_videos.append(
                    {
                        "video_id": video.video_id,
                        "status": video.status,
                        "num_local_tracklets": 0,
                        "num_global_tracks": 0,
                        "global_track_metadata_path": None,
                        "global_track_arrays_path": None,
                        "stitching_trace_path": None,
                    }
                )

        manifest = {
            "schema_name": MANIFEST_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "split": split_view.split,
            "embedding_dim": int(split_view.embedding_dim),
            "embedding_dtype": str(split_view.embedding_dtype),
            "embedding_pooling": "global_track_member_mean",
            "embedding_normalization": str(split_view.embedding_normalization),
            "source_manifest_schema_name": str(split_view.manifest["schema_name"]),
            "source_manifest_schema_version": str(split_view.manifest["schema_version"]),
            "producer": dict(split_view.producer),
            "stitching_policy": canonical_config,
            "videos": manifest_videos,
        }
        _dump_json(temp_dir / "manifest.v1.json", manifest)
        load_global_track_bank_v9(temp_dir, eager_validate=True)
        if output_split_root.exists():
            shutil.rmtree(output_split_root)
        shutil.move(str(temp_dir), str(output_split_root))
        return output_split_root
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


class GlobalTrackBankSplitView:
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
        self.manifest = dict(manifest)

        self.split = str(self.manifest["split"])
        self.embedding_dim = int(self.manifest["embedding_dim"])
        self.embedding_dtype = str(self.manifest["embedding_dtype"])
        self.embedding_pooling = str(self.manifest["embedding_pooling"])
        self.embedding_normalization = str(self.manifest["embedding_normalization"])
        self.producer = dict(self.manifest["producer"])
        self.stitching_policy = dict(self.manifest["stitching_policy"])

        self._video_order: Tuple[str, ...] = tuple()
        self._videos_by_id: Dict[str, GlobalTrackVideoRecord] = {}
        self._loaded_by_video_id: Dict[str, _LoadedVideoPayload] = {}

        self._validate_manifest()
        if eager_validate:
            self._eager_validate_processed_videos()

    def _validate_manifest(self) -> None:
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
        _require(self.manifest.get("embedding_dtype") == "float32", "manifest.embedding_dtype", "must equal 'float32'")
        _require(
            self.manifest.get("embedding_pooling") == "global_track_member_mean",
            "manifest.embedding_pooling",
            "must equal 'global_track_member_mean'",
        )
        _require(
            self.manifest.get("embedding_normalization") in {"none", "l2"},
            "manifest.embedding_normalization",
            "must be one of ['none', 'l2']",
        )
        _require(isinstance(self.manifest.get("producer"), dict), "manifest.producer", "must be an object")
        _require(isinstance(self.manifest.get("stitching_policy"), dict), "manifest.stitching_policy", "must be an object")

        videos = self.manifest.get("videos")
        _require(isinstance(videos, list), "manifest.videos", "must be a list")
        seen = set()
        order: List[str] = []
        for index, video in enumerate(videos):
            vpath = f"manifest.videos[{index}]"
            _require(isinstance(video, dict), vpath, "must be an object")
            for key in (
                "video_id",
                "status",
                "num_local_tracklets",
                "num_global_tracks",
                "global_track_metadata_path",
                "global_track_arrays_path",
                "stitching_trace_path",
            ):
                _require(key in video, f"{vpath}.{key}", "required field missing")
            video_id = video["video_id"]
            _require(isinstance(video_id, str) and video_id, f"{vpath}.video_id", "must be a non-empty string")
            _require(video_id not in seen, f"{vpath}.video_id", f"duplicate video_id '{video_id}'")
            seen.add(video_id)
            order.append(video_id)

            status = video["status"]
            _require(status in ALL_STATUSES, f"{vpath}.status", f"must be one of {sorted(ALL_STATUSES)}")
            num_local_tracklets = video["num_local_tracklets"]
            num_global_tracks = video["num_global_tracks"]
            _require(_is_int(num_local_tracklets) and num_local_tracklets >= 0, f"{vpath}.num_local_tracklets", "must be integer >= 0")
            _require(_is_int(num_global_tracks) and num_global_tracks >= 0, f"{vpath}.num_global_tracks", "must be integer >= 0")

            metadata_path = video["global_track_metadata_path"]
            arrays_path = video["global_track_arrays_path"]
            trace_path = video["stitching_trace_path"]
            if status in PROCESSED_STATUSES:
                metadata_path = _require_relative_path(metadata_path, f"{vpath}.global_track_metadata_path")
                arrays_path = _require_relative_path(arrays_path, f"{vpath}.global_track_arrays_path")
                trace_path = _require_relative_path(trace_path, f"{vpath}.stitching_trace_path")
            else:
                _require(num_local_tracklets == 0, f"{vpath}.num_local_tracklets", "must equal 0 for failed/unprocessed")
                _require(num_global_tracks == 0, f"{vpath}.num_global_tracks", "must equal 0 for failed/unprocessed")
                _require(metadata_path is None, f"{vpath}.global_track_metadata_path", "must be null for failed/unprocessed")
                _require(arrays_path is None, f"{vpath}.global_track_arrays_path", "must be null for failed/unprocessed")
                _require(trace_path is None, f"{vpath}.stitching_trace_path", "must be null for failed/unprocessed")

            self._videos_by_id[video_id] = GlobalTrackVideoRecord(
                video_id=video_id,
                status=status,
                num_local_tracklets=num_local_tracklets,
                num_global_tracks=num_global_tracks,
                global_track_metadata_path=metadata_path,
                global_track_arrays_path=arrays_path,
                stitching_trace_path=trace_path,
            )
        _require(order == sorted(order), "manifest.videos", "must be lexicographically sorted by video_id")
        self._video_order = tuple(order)

    def _eager_validate_processed_videos(self) -> None:
        for video_id in self._video_order:
            video = self._videos_by_id[video_id]
            if video.status in PROCESSED_STATUSES:
                self._load_processed_video(video)

    def iter_videos(self, include_statuses: Tuple[str, ...] | None = None) -> Iterator[GlobalTrackVideoRecord]:
        accepted = None if include_statuses is None else set(include_statuses)
        if accepted is not None:
            unknown = accepted - ALL_STATUSES
            _require(not unknown, "iter_videos.include_statuses", f"contains unknown statuses: {sorted(unknown)}")
        for video_id in self._video_order:
            video = self._videos_by_id[video_id]
            if accepted is not None and video.status not in accepted:
                continue
            yield video

    def iter_global_tracks(self, video_id: str) -> Iterator[GlobalTrackRecord]:
        video = self._get_video(video_id)
        if video.status not in PROCESSED_STATUSES:
            return iter(())
        payload = self._load_processed_video(video)
        return (
            GlobalTrackRecord(metadata=meta, embedding=payload.embeddings[meta.row_index])
            for meta in payload.track_rows
        )

    def get_global_track_metadata(self, video_id: str, global_track_id: int) -> GlobalTrackMetadata:
        _require(_is_int(global_track_id) and global_track_id >= 0, "global_track_id", "must be integer >= 0")
        video = self._get_video(video_id)
        payload = self._load_processed_video(video)
        _require(global_track_id in payload.track_index_by_id, "global_track_id", f"unknown global_track_id '{global_track_id}'")
        return payload.track_rows[payload.track_index_by_id[global_track_id]]

    def get_global_track_embedding(self, video_id: str, global_track_id: int) -> np.ndarray:
        metadata = self.get_global_track_metadata(video_id, global_track_id)
        payload = self._load_processed_video(self._videos_by_id[video_id])
        return payload.embeddings[metadata.row_index]

    def _get_video(self, video_id: str) -> GlobalTrackVideoRecord:
        _require(isinstance(video_id, str) and video_id, "video_id", "must be a non-empty string")
        _require(video_id in self._videos_by_id, "video_id", f"unknown video_id '{video_id}'")
        return self._videos_by_id[video_id]

    def _load_processed_video(self, video: GlobalTrackVideoRecord) -> _LoadedVideoPayload:
        cached = self._loaded_by_video_id.get(video.video_id)
        if cached is not None:
            return cached
        _require(video.global_track_metadata_path is not None, "global_track_metadata_path", "required for processed video")
        _require(video.global_track_arrays_path is not None, "global_track_arrays_path", "required for processed video")

        metadata_path = self.split_root / Path(video.global_track_metadata_path)
        arrays_path = self.split_root / Path(video.global_track_arrays_path)
        trace_path = self.split_root / Path(video.stitching_trace_path) if video.stitching_trace_path is not None else None

        metadata = _load_json(metadata_path, "global_track_metadata.v1.json")
        _require(metadata.get("schema_name") == VIDEO_SCHEMA_NAME, "global_track_metadata.v1.json.schema_name", f"must equal '{VIDEO_SCHEMA_NAME}'")
        major = _parse_major_version(metadata.get("schema_version"), "global_track_metadata.v1.json.schema_version")
        _require(major == 1, "global_track_metadata.v1.json.schema_version", "unsupported major version")
        _require(metadata.get("split") == self.split, "global_track_metadata.v1.json.split", "must match manifest split")
        _require(metadata.get("video_id") == video.video_id, "global_track_metadata.v1.json.video_id", "must match manifest video_id")
        _require(
            _is_int(metadata.get("num_local_tracklets")) and metadata["num_local_tracklets"] == video.num_local_tracklets,
            "global_track_metadata.v1.json.num_local_tracklets",
            "must match manifest num_local_tracklets",
        )
        _require(
            _is_int(metadata.get("num_global_tracks")) and metadata["num_global_tracks"] == video.num_global_tracks,
            "global_track_metadata.v1.json.num_global_tracks",
            "must match manifest num_global_tracks",
        )
        tracks = metadata.get("global_tracks")
        _require(isinstance(tracks, list), "global_track_metadata.v1.json.global_tracks", "must be a list")
        _require(
            len(tracks) == video.num_global_tracks,
            "global_track_metadata.v1.json.global_tracks",
            "len(global_tracks) must match manifest num_global_tracks",
        )

        track_rows: List[GlobalTrackMetadata] = []
        track_index_by_id: Dict[int, int] = {}
        for row_index, track in enumerate(tracks):
            tpath = f"global_track_metadata.v1.json.global_tracks[{row_index}]"
            _require(isinstance(track, dict), tpath, "must be an object")
            for key in (
                "row_index",
                "global_track_id",
                "start_frame_idx",
                "end_frame_idx",
                "num_active_frames",
                "objectness_score_mean",
                "objectness_score_max",
                "member_count",
                "member_local_row_indices",
                "member_track_ids",
            ):
                _require(key in track, f"{tpath}.{key}", "required field missing")
            _require(_is_int(track["row_index"]) and track["row_index"] == row_index, f"{tpath}.row_index", "must equal row position")
            _require(_is_int(track["global_track_id"]) and track["global_track_id"] == row_index, f"{tpath}.global_track_id", "must equal row position")
            _require(track["global_track_id"] not in track_index_by_id, f"{tpath}.global_track_id", "duplicate global_track_id")
            _require(_is_int(track["start_frame_idx"]) and track["start_frame_idx"] >= 0, f"{tpath}.start_frame_idx", "must be integer >= 0")
            _require(
                _is_int(track["end_frame_idx"]) and track["end_frame_idx"] >= track["start_frame_idx"],
                f"{tpath}.end_frame_idx",
                "must be integer >= start_frame_idx",
            )
            _require(_is_int(track["num_active_frames"]) and track["num_active_frames"] > 0, f"{tpath}.num_active_frames", "must be integer > 0")
            _require(_is_number(track["objectness_score_mean"]), f"{tpath}.objectness_score_mean", "must be a number")
            _require(_is_number(track["objectness_score_max"]), f"{tpath}.objectness_score_max", "must be a number")
            _require(_is_int(track["member_count"]) and track["member_count"] > 0, f"{tpath}.member_count", "must be integer > 0")
            members = track["member_local_row_indices"]
            member_track_ids = track["member_track_ids"]
            _require(isinstance(members, list) and len(members) == track["member_count"], f"{tpath}.member_local_row_indices", "must be a list with member_count entries")
            _require(isinstance(member_track_ids, list) and len(member_track_ids) == track["member_count"], f"{tpath}.member_track_ids", "must be a list with member_count entries")
            for member_index, member_value in enumerate(members):
                _require(
                    _is_int(member_value) and member_value >= 0,
                    f"{tpath}.member_local_row_indices[{member_index}]",
                    "must be integer >= 0",
                )
            track_index_by_id[int(track["global_track_id"])] = row_index
            track_rows.append(
                GlobalTrackMetadata(
                    video_id=video.video_id,
                    row_index=row_index,
                    global_track_id=int(track["global_track_id"]),
                    start_frame_idx=int(track["start_frame_idx"]),
                    end_frame_idx=int(track["end_frame_idx"]),
                    num_active_frames=int(track["num_active_frames"]),
                    objectness_score_mean=float(track["objectness_score_mean"]),
                    objectness_score_max=float(track["objectness_score_max"]),
                    member_count=int(track["member_count"]),
                    member_local_row_indices=tuple(int(value) for value in members),
                    member_track_ids=tuple(member_track_ids),
                )
            )

        try:
            with np.load(arrays_path, allow_pickle=False) as npz_data:
                keys = set(npz_data.files)
                _require(
                    {"global_embeddings", "global_track_row_index"}.issubset(keys),
                    "global_track_arrays.v1.npz.keys",
                    "missing required keys ['global_embeddings', 'global_track_row_index']",
                )
                embeddings = np.asarray(npz_data["global_embeddings"])
                row_index = np.asarray(npz_data["global_track_row_index"])
        except Exception as exc:
            raise GlobalTrackBankError(f"global_track_arrays.v1.npz failed to load at {arrays_path}: {exc}") from exc

        _require(embeddings.dtype == np.float32, "global_track_arrays.v1.npz.global_embeddings.dtype", "must be float32")
        _require(embeddings.ndim == 2, "global_track_arrays.v1.npz.global_embeddings.shape", "must be rank-2 [N, D]")
        _require(
            embeddings.shape[1] == self.embedding_dim,
            "global_track_arrays.v1.npz.global_embeddings.shape[1]",
            "must equal manifest embedding_dim",
        )
        _require(row_index.dtype == np.int64, "global_track_arrays.v1.npz.global_track_row_index.dtype", "must be int64")
        _require(row_index.ndim == 1, "global_track_arrays.v1.npz.global_track_row_index.shape", "must be rank-1 [N]")
        _require(
            embeddings.shape[0] == video.num_global_tracks,
            "global_track_arrays.v1.npz.global_embeddings.shape[0]",
            "must equal manifest num_global_tracks",
        )
        _require(
            row_index.shape[0] == video.num_global_tracks,
            "global_track_arrays.v1.npz.global_track_row_index.shape[0]",
            "must equal manifest num_global_tracks",
        )
        _require(
            np.array_equal(row_index, np.arange(video.num_global_tracks, dtype=np.int64)),
            "global_track_arrays.v1.npz.global_track_row_index",
            "must equal [0..N-1]",
        )
        _require(np.isfinite(embeddings).all(), "global_track_arrays.v1.npz.global_embeddings", "must contain only finite values")
        if trace_path is not None:
            trace = _load_json(trace_path, "stitching_trace.v1.json")
            _require(trace.get("schema_name") == TRACE_SCHEMA_NAME, "stitching_trace.v1.json.schema_name", f"must equal '{TRACE_SCHEMA_NAME}'")
            trace_major = _parse_major_version(trace.get("schema_version"), "stitching_trace.v1.json.schema_version")
            _require(trace_major == 1, "stitching_trace.v1.json.schema_version", "unsupported major version")
            _require(trace.get("video_id") == video.video_id, "stitching_trace.v1.json.video_id", "must match manifest video_id")
            _require(trace.get("split") == self.split, "stitching_trace.v1.json.split", "must match manifest split")

        embeddings.setflags(write=False)
        payload = _LoadedVideoPayload(
            track_rows=tuple(track_rows),
            track_index_by_id=track_index_by_id,
            embeddings=embeddings,
        )
        self._loaded_by_video_id[video.video_id] = payload
        return payload


def load_global_track_bank_v9(split_root: Path, *, eager_validate: bool = True) -> GlobalTrackBankSplitView:
    split_root = Path(split_root).resolve()
    _require(split_root.exists() and split_root.is_dir(), "split_root", f"must be an existing directory: {split_root}")
    manifest_path = _discover_manifest_path(split_root)
    manifest = _load_json(manifest_path, manifest_path.name)
    return GlobalTrackBankSplitView(
        split_root=split_root,
        manifest_path=manifest_path,
        manifest=manifest,
        eager_validate=eager_validate,
    )


def _load_video_trace(split_root: Path, video: GlobalTrackVideoRecord) -> Dict[str, Any]:
    _require(video.stitching_trace_path is not None, "video.stitching_trace_path", "required for processed video")
    return _load_json(split_root / Path(video.stitching_trace_path), "stitching_trace.v1.json")


def summarize_global_track_bank_v9(split_root: Path) -> Dict[str, Any]:
    view = load_global_track_bank_v9(split_root, eager_validate=True)
    per_video_rows: List[Dict[str, Any]] = []
    total_local = 0
    total_global = 0
    total_candidate_pairs = 0
    total_merge_components = 0
    videos_with_merges = 0
    merged_local_tracklets = 0
    selected_video_id: Optional[str] = None
    selected_video_score = -math.inf
    selected_video_fragmentation = -1

    for video in view.iter_videos(include_statuses=("processed_with_tracks", "processed_zero_tracks")):
        trace = _load_video_trace(view.split_root, video)
        diagnostics = dict(trace["merge_diagnostics"])
        local_count = int(video.num_local_tracklets)
        global_count = int(video.num_global_tracks)
        total_local += local_count
        total_global += global_count
        total_candidate_pairs += int(diagnostics["candidate_pair_count"])
        total_merge_components += int(diagnostics["merge_component_count"])
        merged_local_tracklets += int(diagnostics["merged_local_tracklets"])
        fragmentation_reduction = int(diagnostics["fragmentation_reduction"])
        if fragmentation_reduction > 0:
            videos_with_merges += 1

        best_edge_score = -math.inf
        candidate_edges = trace["candidate_edges"]
        if candidate_edges:
            best_edge_score = max(float(edge["match_score"]) for edge in candidate_edges)
        candidate_video = False
        if fragmentation_reduction > selected_video_fragmentation:
            candidate_video = True
        elif fragmentation_reduction == selected_video_fragmentation and best_edge_score > selected_video_score:
            candidate_video = True
        elif selected_video_id is None and fragmentation_reduction == selected_video_fragmentation and best_edge_score == selected_video_score:
            candidate_video = True
        if candidate_video:
            selected_video_id = video.video_id
            selected_video_fragmentation = fragmentation_reduction
            selected_video_score = best_edge_score

        per_video_rows.append(
            {
                "video_id": video.video_id,
                "num_local_tracklets": local_count,
                "num_global_tracks": global_count,
                "candidate_pair_count": int(diagnostics["candidate_pair_count"]),
                "merge_component_count": int(diagnostics["merge_component_count"]),
                "fragmentation_reduction": fragmentation_reduction,
            }
        )

    fragmentation_ratio = 1.0 if total_local == 0 else float(total_global / total_local)
    merge_reduction_ratio = 0.0 if total_local == 0 else float((total_local - total_global) / total_local)
    summary = {
        "schema_name": "wsovvis.global_track_bank_summary",
        "schema_version": SCHEMA_VERSION,
        "split": view.split,
        "source_manifest_schema_name": view.manifest["source_manifest_schema_name"],
        "source_manifest_schema_version": view.manifest["source_manifest_schema_version"],
        "stitching_policy": dict(view.stitching_policy),
        "artifact_identity": {
            "split_root": str(view.split_root),
            "manifest_path": str(view.manifest_path),
        },
        "stitching_stats": {
            "total_videos": len(tuple(view.iter_videos())),
            "processed_videos": len(per_video_rows),
            "videos_with_merges": videos_with_merges,
            "total_local_tracklets": total_local,
            "total_global_tracks": total_global,
            "total_candidate_pairs": total_candidate_pairs,
            "total_merge_components": total_merge_components,
            "merged_local_tracklets": merged_local_tracklets,
            "fragmentation_reduction_total": total_local - total_global,
            "fragmentation_ratio": fragmentation_ratio,
            "merge_reduction_ratio": merge_reduction_ratio,
        },
        "selected_video_id": selected_video_id,
        "per_video_rows_sample": per_video_rows[:10],
    }
    return summary


def build_global_track_bank_v9_worked_example(split_root: Path, *, selected_video_id: Optional[str] = None) -> Dict[str, Any]:
    view = load_global_track_bank_v9(split_root, eager_validate=True)
    summary = summarize_global_track_bank_v9(split_root)
    video_id = selected_video_id or summary["selected_video_id"]
    _require(isinstance(video_id, str) and video_id, "selected_video_id", "must resolve to a processed video")
    video = next(iter([video for video in view.iter_videos() if video.video_id == video_id]), None)
    _require(video is not None, "selected_video_id", f"unknown video_id '{video_id}'")
    trace = _load_video_trace(view.split_root, video)
    metadata_rows = list(view.iter_global_tracks(video_id))
    best_edge = None
    if trace["candidate_edges"]:
        best_edge = max(
            trace["candidate_edges"],
            key=lambda edge: (
                float(edge["match_score"]),
                float(edge["temporal_iou"]),
                float(edge["query_cosine"]),
                -int(edge["left_row_index"]),
                -int(edge["right_row_index"]),
            ),
        )
    worked_example = {
        "schema_name": "wsovvis.global_track_bank_worked_example",
        "schema_version": SCHEMA_VERSION,
        "split": view.split,
        "selected_video_id": video_id,
        "stitching_policy": dict(view.stitching_policy),
        "local_tracks": trace["local_tracks"],
        "temporal_iou_matrix": trace["temporal_iou_matrix"],
        "query_cosine_matrix": trace["query_cosine_matrix"],
        "match_score_matrix": trace["match_score_matrix"],
        "candidate_edges": trace["candidate_edges"],
        "best_edge": best_edge,
        "merge_result": [
            {
                "global_track_id": record.metadata.global_track_id,
                "row_index": record.metadata.row_index,
                "start_frame_idx": record.metadata.start_frame_idx,
                "end_frame_idx": record.metadata.end_frame_idx,
                "num_active_frames": record.metadata.num_active_frames,
                "member_count": record.metadata.member_count,
                "member_local_row_indices": list(record.metadata.member_local_row_indices),
                "member_track_ids": _format_track_id_list(record.metadata.member_track_ids),
            }
            for record in metadata_rows
        ],
        "merge_diagnostics": trace["merge_diagnostics"],
    }
    return worked_example


def render_global_track_bank_coverage_svg(
    split_root: Path,
    output_path: Path,
    *,
    selected_video_id: Optional[str] = None,
) -> Path:
    worked_example = build_global_track_bank_v9_worked_example(split_root, selected_video_id=selected_video_id)
    local_tracks = list(worked_example["local_tracks"])
    merge_result = list(worked_example["merge_result"])
    max_frame = 0
    for row in local_tracks:
        max_frame = max(max_frame, int(row["end_frame_idx"]))
    for row in merge_result:
        max_frame = max(max_frame, int(row["end_frame_idx"]))
    frame_span = max(1, max_frame + 1)
    left_margin = 150
    row_height = 24
    chart_width = 780
    local_offset = 60
    global_offset = local_offset + (len(local_tracks) * row_height) + 70
    chart_height = global_offset + (len(merge_result) * row_height) + 80
    scale = chart_width / frame_span
    palette = ["#0B6E4F", "#C84C09", "#1D3557", "#B5179E", "#6A994E", "#BC4749", "#8E7DBE", "#3A86FF"]

    def x_pos(frame_idx: int) -> float:
        return left_margin + (float(frame_idx) * scale)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left_margin + chart_width + 40}" height="{chart_height}" viewBox="0 0 {left_margin + chart_width + 40} {chart_height}">',
        '<rect width="100%" height="100%" fill="#F8F7F2"/>',
        f'<text x="{left_margin}" y="28" font-family="monospace" font-size="18" fill="#1F2933">G3 coverage for video {worked_example["selected_video_id"]}</text>',
        f'<text x="{left_margin}" y="48" font-family="monospace" font-size="12" fill="#52606D">temporal interval IoU + local-query cosine, score-matrix-driven merge trace</text>',
    ]
    for frame_idx in range(frame_span + 1):
        xpos = x_pos(frame_idx)
        lines.append(f'<line x1="{xpos:.2f}" y1="54" x2="{xpos:.2f}" y2="{chart_height - 30}" stroke="#D9E2EC" stroke-width="1"/>')
        if frame_idx < frame_span:
            lines.append(
                f'<text x="{xpos + 2:.2f}" y="{chart_height - 10}" font-family="monospace" font-size="10" fill="#52606D">{frame_idx}</text>'
            )

    lines.append(f'<text x="20" y="{local_offset - 10}" font-family="monospace" font-size="14" fill="#102A43">local tracklets</text>')
    for idx, row in enumerate(local_tracks):
        y = local_offset + (idx * row_height)
        color = palette[idx % len(palette)]
        x = x_pos(int(row["start_frame_idx"]))
        width = max(6.0, (int(row["end_frame_idx"]) - int(row["start_frame_idx"]) + 1) * scale)
        lines.append(
            f'<text x="16" y="{y + 16}" font-family="monospace" font-size="11" fill="#102A43">L{idx} t{row["track_id"]}</text>'
        )
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="14" rx="3" fill="{color}" opacity="0.72"/>'
        )
    lines.append(
        f'<text x="20" y="{global_offset - 10}" font-family="monospace" font-size="14" fill="#102A43">global tracks</text>'
    )
    for idx, row in enumerate(merge_result):
        y = global_offset + (idx * row_height)
        color = palette[idx % len(palette)]
        x = x_pos(int(row["start_frame_idx"]))
        width = max(6.0, (int(row["end_frame_idx"]) - int(row["start_frame_idx"]) + 1) * scale)
        member_label = ",".join(str(value) for value in row["member_track_ids"])
        lines.append(
            f'<text x="16" y="{y + 16}" font-family="monospace" font-size="11" fill="#102A43">G{row["global_track_id"]}</text>'
        )
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="14" rx="3" fill="{color}" opacity="0.88"/>'
        )
        lines.append(
            f'<text x="{x + 6:.2f}" y="{y + 12}" font-family="monospace" font-size="10" fill="#F8F7F2">{member_label}</text>'
        )
    lines.append("</svg>")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
