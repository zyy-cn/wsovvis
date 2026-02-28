from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .v1_core import ExportContractError, build_track_feature_export_v1

_RUNTIME_STATUSES = {"success", "failed"}
_NORMALIZATION_MODES = {"none", "l2"}


def _err(field_path: str, rule_summary: str) -> ExportContractError:
    return ExportContractError(f"bridge_input field '{field_path}': {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path=field_path, rule_summary=rule_summary)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def load_stageb_bridge_input(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportContractError(f"bridge_input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExportContractError(f"bridge_input invalid JSON at {path}: {exc}") from exc
    _require(isinstance(payload, dict), "$", "top-level value must be an object")
    return payload


def _validate_top_level(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[str], List[Dict[str, Any]], str]:
    required = [
        "split",
        "producer",
        "split_domain_video_ids",
        "stageb_video_results",
        "embedding_pooling",
    ]
    for key in required:
        _require(key in payload, key, "required field missing")

    split = payload["split"]
    _require(isinstance(split, str) and split, "split", "must be a non-empty string")

    producer = payload["producer"]
    _require(isinstance(producer, dict), "producer", "must be an object")

    split_domain_video_ids = payload["split_domain_video_ids"]
    _require(
        isinstance(split_domain_video_ids, list),
        "split_domain_video_ids",
        "must be a list of non-empty strings",
    )

    stageb_video_results = payload["stageb_video_results"]
    _require(isinstance(stageb_video_results, list), "stageb_video_results", "must be a list")

    embedding_pooling = payload["embedding_pooling"]
    _require(
        embedding_pooling == "track_pooled",
        "embedding_pooling",
        "must equal 'track_pooled'",
    )

    embedding_normalization = payload.get("embedding_normalization", "none")
    _require(
        embedding_normalization in _NORMALIZATION_MODES,
        "embedding_normalization",
        "must be one of ['none', 'l2'] when provided",
    )

    seen_video_ids = set()
    canonical_split_ids: List[str] = []
    for idx, video_id in enumerate(split_domain_video_ids):
        field = f"split_domain_video_ids[{idx}]"
        _require(isinstance(video_id, str) and video_id, field, "must be a non-empty string")
        _require(video_id not in seen_video_ids, field, f"duplicate video_id '{video_id}'")
        seen_video_ids.add(video_id)
        canonical_split_ids.append(video_id)

    return split, producer, canonical_split_ids, stageb_video_results, embedding_normalization


def _validate_track(
    track: Any,
    field_path: str,
    seen_track_ids: set,
    embedding_dim: Optional[int],
    split_track_id_type: Optional[type],
) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[type], bool]:
    if not isinstance(track, dict):
        return None, embedding_dim, split_track_id_type, False

    required = [
        "track_id",
        "start_frame_idx",
        "end_frame_idx",
        "num_active_frames",
        "objectness_score",
        "embedding",
    ]
    for key in required:
        if key not in track:
            return None, embedding_dim, split_track_id_type, False

    track_id = track["track_id"]
    if not ((isinstance(track_id, str) and track_id) or _is_int(track_id)):
        return None, embedding_dim, split_track_id_type, False

    if track_id in seen_track_ids:
        return None, embedding_dim, split_track_id_type, False

    track_id_type = type(track_id)
    if split_track_id_type is None:
        split_track_id_type = track_id_type
    elif track_id_type is not split_track_id_type:
        raise _err(
            f"{field_path}.track_id",
            "track_id serialized type must be consistent within split",
        )

    start_frame_idx = track["start_frame_idx"]
    end_frame_idx = track["end_frame_idx"]
    num_active_frames = track["num_active_frames"]
    objectness_score = track["objectness_score"]
    embedding = track["embedding"]

    if not (_is_int(start_frame_idx) and start_frame_idx >= 0):
        return None, embedding_dim, split_track_id_type, False
    if not (_is_int(end_frame_idx) and end_frame_idx >= start_frame_idx):
        return None, embedding_dim, split_track_id_type, False
    if not (_is_int(num_active_frames) and num_active_frames > 0):
        return None, embedding_dim, split_track_id_type, False
    if not _is_finite_number(objectness_score):
        return None, embedding_dim, split_track_id_type, False

    if not (isinstance(embedding, list) and embedding):
        return None, embedding_dim, split_track_id_type, False

    numeric_embedding: List[float] = []
    for e_idx, value in enumerate(embedding):
        if not _is_finite_number(value):
            # D3 reject-only: non-finite embedding values are dropped, never repaired.
            return None, embedding_dim, split_track_id_type, False
        numeric_embedding.append(float(value))

    if embedding_dim is None:
        embedding_dim = len(numeric_embedding)
    if len(numeric_embedding) != embedding_dim:
        return None, embedding_dim, split_track_id_type, False

    seen_track_ids.add(track_id)
    return (
        {
            "track_id": track_id,
            "start_frame_idx": start_frame_idx,
            "end_frame_idx": end_frame_idx,
            "num_active_frames": num_active_frames,
            "objectness_score": float(objectness_score),
            "embedding": numeric_embedding,
        },
        embedding_dim,
        split_track_id_type,
        True,
    )


def _normalize_embedding(embedding: List[float], normalization: str) -> List[float]:
    if normalization != "l2":
        return embedding
    arr = np.asarray(embedding, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return embedding
    return [float(v) for v in (arr / norm)]


def convert_stageb_bridge_input_to_task_input_v1(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    split, producer, split_ids, stageb_results, embedding_normalization = _validate_top_level(payload)

    summary = {
        "total_split_domain_videos": len(split_ids),
        "stageb_result_records_seen": len(stageb_results),
        "processed_with_tracks_videos": 0,
        "processed_zero_tracks_videos": 0,
        "failed_videos": 0,
        "unprocessed_videos": 0,
        "invalid_tracks_dropped": 0,
        "duplicate_stageb_result_records": 0,
        "videos_with_runtime_success": 0,
        "videos_with_runtime_failed": 0,
    }

    split_set = set(split_ids)
    indexed_results: Dict[str, Dict[str, Any]] = {}
    for idx, result in enumerate(stageb_results):
        rpath = f"stageb_video_results[{idx}]"
        _require(isinstance(result, dict), rpath, "must be an object")

        _require("video_id" in result, f"{rpath}.video_id", "required field missing")
        video_id = result["video_id"]
        _require(
            isinstance(video_id, str) and video_id,
            f"{rpath}.video_id",
            "must be a non-empty string",
        )
        _require(
            video_id in split_set,
            f"{rpath}.video_id",
            f"video_id '{video_id}' is not present in split_domain_video_ids",
        )
        if video_id in indexed_results:
            summary["duplicate_stageb_result_records"] += 1
            raise _err(f"{rpath}.video_id", f"duplicate Stage-B result for video_id '{video_id}'")

        _require("runtime_status" in result, f"{rpath}.runtime_status", "required field missing")
        runtime_status = result["runtime_status"]
        _require(
            runtime_status in _RUNTIME_STATUSES,
            f"{rpath}.runtime_status",
            "must be one of ['failed', 'success']",
        )

        tracks = result.get("tracks", [])
        _require(isinstance(tracks, list), f"{rpath}.tracks", "must be a list when provided")

        if runtime_status == "failed":
            summary["videos_with_runtime_failed"] += 1
            _require(
                len(tracks) == 0,
                f"{rpath}.tracks",
                "must be empty when runtime_status is 'failed'",
            )
        else:
            summary["videos_with_runtime_success"] += 1

        indexed_results[video_id] = result

    embedding_dim: Optional[int] = None
    split_track_id_type: Optional[type] = None
    canonical_videos: List[Dict[str, Any]] = []

    for video_id in sorted(split_ids):
        if video_id not in indexed_results:
            summary["unprocessed_videos"] += 1
            canonical_videos.append({"video_id": video_id, "status": "unprocessed", "tracks": []})
            continue

        result = indexed_results[video_id]
        runtime_status = result["runtime_status"]

        if runtime_status == "failed":
            summary["failed_videos"] += 1
            canonical_videos.append({"video_id": video_id, "status": "failed", "tracks": []})
            continue

        tracks = result.get("tracks", [])
        seen_track_ids = set()
        valid_tracks: List[Dict[str, Any]] = []

        for t_idx, track in enumerate(tracks):
            tpath = f"stageb_video_results[{video_id!r}].tracks[{t_idx}]"
            canonical, embedding_dim, split_track_id_type, accepted = _validate_track(
                track=track,
                field_path=tpath,
                seen_track_ids=seen_track_ids,
                embedding_dim=embedding_dim,
                split_track_id_type=split_track_id_type,
            )
            if not accepted:
                summary["invalid_tracks_dropped"] += 1
                continue
            assert canonical is not None
            canonical["embedding"] = _normalize_embedding(canonical["embedding"], embedding_normalization)
            valid_tracks.append(canonical)

        if split_track_id_type is int:
            valid_tracks.sort(key=lambda tr: (tr["start_frame_idx"], tr["end_frame_idx"], int(tr["track_id"])))
        else:
            valid_tracks.sort(key=lambda tr: (tr["start_frame_idx"], tr["end_frame_idx"], str(tr["track_id"])))

        if valid_tracks:
            summary["processed_with_tracks_videos"] += 1
            canonical_videos.append(
                {
                    "video_id": video_id,
                    "status": "processed_with_tracks",
                    "tracks": valid_tracks,
                }
            )
        else:
            summary["processed_zero_tracks_videos"] += 1
            canonical_videos.append({"video_id": video_id, "status": "processed_zero_tracks", "tracks": []})

    _require(
        embedding_dim is not None and embedding_dim > 0,
        "embedding_dim",
        "cannot be inferred from surviving valid tracks",
    )

    return (
        {
            "split": split,
            "embedding_dim": embedding_dim,
            "embedding_dtype": "float32",
            "embedding_pooling": "track_pooled",
            "embedding_normalization": embedding_normalization,
            "producer": producer,
            "videos": canonical_videos,
        },
        summary,
    )


def build_track_feature_export_v1_from_stageb_bridge_input(
    payload: Dict[str, Any],
    output_split_root: Path,
    overwrite: bool = False,
) -> Tuple[Path, Dict[str, int]]:
    task_input, summary = convert_stageb_bridge_input_to_task_input_v1(payload)
    output_path = build_track_feature_export_v1(
        input_payload=task_input,
        output_split_root=output_split_root,
        overwrite=overwrite,
    )
    return output_path, summary


def build_track_feature_export_v1_from_stageb_bridge_input_json(
    input_json_path: Path,
    output_split_root: Path,
    overwrite: bool = False,
) -> Tuple[Path, Dict[str, int]]:
    payload = load_stageb_bridge_input(input_json_path)
    return build_track_feature_export_v1_from_stageb_bridge_input(
        payload=payload,
        output_split_root=output_split_root,
        overwrite=overwrite,
    )
