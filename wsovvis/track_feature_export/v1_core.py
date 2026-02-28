from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

MANIFEST_SCHEMA_NAME = "stage_b_track_feature_export"
VIDEO_SCHEMA_NAME = "stage_b_track_feature_export_video"
SCHEMA_VERSION = "1.0.0"
PROCESSED_STATUSES = {"processed_with_tracks", "processed_zero_tracks"}
ALL_STATUSES = PROCESSED_STATUSES | {"failed", "unprocessed"}


class ExportContractError(ValueError):
    """Raised when producer input or exported artifacts violate contract rules."""


def _err(file_type: str, field_path: str, rule_summary: str) -> ExportContractError:
    return ExportContractError(f"{file_type} field '{field_path}': {rule_summary}")


def _require(condition: bool, file_type: str, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(file_type=file_type, field_path=field_path, rule_summary=rule_summary)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def load_task_local_input(input_json_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportContractError(f"task_input file not found: {input_json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ExportContractError(f"task_input invalid JSON at {input_json_path}: {exc}") from exc
    _require(isinstance(payload, dict), "task_input", "$", "top-level value must be an object")
    return payload


def _validate_semver_exact(version: Any, file_type: str, field_path: str) -> None:
    _require(isinstance(version, str), file_type, field_path, "must be a string")
    _require(version == SCHEMA_VERSION, file_type, field_path, f"must equal '{SCHEMA_VERSION}'")


def _validate_producer_block(producer: Any, split: str) -> Dict[str, Any]:
    _require(isinstance(producer, dict), "task_input", "producer", "must be an object")
    required_fields = [
        "stage_b_checkpoint_id",
        "stage_b_checkpoint_hash",
        "stage_b_config_ref",
        "stage_b_config_hash",
        "pseudo_tube_manifest_id",
        "pseudo_tube_manifest_hash",
        "split",
        "extraction_settings",
    ]
    for key in required_fields:
        _require(key in producer, "task_input", f"producer.{key}", "required field missing")

    for key in required_fields[:-2]:
        _require(
            isinstance(producer[key], str) and producer[key],
            "task_input",
            f"producer.{key}",
            "must be a non-empty string",
        )

    _require(
        isinstance(producer["split"], str) and producer["split"],
        "task_input",
        "producer.split",
        "must be a non-empty string",
    )
    _require(
        producer["split"] == split,
        "task_input",
        "producer.split",
        "must match top-level split",
    )

    extraction = producer["extraction_settings"]
    _require(
        isinstance(extraction, dict),
        "task_input",
        "producer.extraction_settings",
        "must be an object",
    )
    for key in ("frame_sampling_rule", "pooling_rule", "min_track_length"):
        _require(
            key in extraction,
            "task_input",
            f"producer.extraction_settings.{key}",
            "required field missing",
        )
    _require(
        isinstance(extraction["frame_sampling_rule"], str) and extraction["frame_sampling_rule"],
        "task_input",
        "producer.extraction_settings.frame_sampling_rule",
        "must be a non-empty string",
    )
    _require(
        isinstance(extraction["pooling_rule"], str) and extraction["pooling_rule"],
        "task_input",
        "producer.extraction_settings.pooling_rule",
        "must be a non-empty string",
    )
    _require(
        _is_int(extraction["min_track_length"]) and extraction["min_track_length"] >= 1,
        "task_input",
        "producer.extraction_settings.min_track_length",
        "must be integer >= 1",
    )
    return producer


def _relative_video_paths(video_id: str) -> Tuple[str, str]:
    rel_dir = PurePosixPath("videos") / video_id
    return str(rel_dir / "track_metadata.v1.json"), str(rel_dir / "track_arrays.v1.npz")


def _validate_and_canonicalize_task_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "split",
        "embedding_dim",
        "embedding_dtype",
        "embedding_pooling",
        "embedding_normalization",
        "producer",
        "videos",
    ]
    for key in required:
        _require(key in payload, "task_input", key, "required field missing")

    split = payload["split"]
    _require(isinstance(split, str) and split, "task_input", "split", "must be a non-empty string")

    embedding_dim = payload["embedding_dim"]
    _require(_is_int(embedding_dim) and embedding_dim > 0, "task_input", "embedding_dim", "must be integer > 0")

    _require(
        payload["embedding_dtype"] == "float32",
        "task_input",
        "embedding_dtype",
        "must equal 'float32'",
    )
    _require(
        payload["embedding_pooling"] == "track_pooled",
        "task_input",
        "embedding_pooling",
        "must equal 'track_pooled'",
    )
    _require(
        payload["embedding_normalization"] in {"none", "l2"},
        "task_input",
        "embedding_normalization",
        "must be one of ['none', 'l2']",
    )

    producer = _validate_producer_block(payload["producer"], split=split)

    videos = payload["videos"]
    _require(isinstance(videos, list), "task_input", "videos", "must be a list")

    canonical_videos: List[Dict[str, Any]] = []
    seen_video_ids = set()
    split_track_id_type: Optional[type] = None

    for idx, video in enumerate(videos):
        vpath = f"videos[{idx}]"
        _require(isinstance(video, dict), "task_input", vpath, "must be an object")
        for key in ("video_id", "status"):
            _require(key in video, "task_input", f"{vpath}.{key}", "required field missing")

        video_id = video["video_id"]
        _require(
            isinstance(video_id, str) and video_id,
            "task_input",
            f"{vpath}.video_id",
            "must be a non-empty string",
        )
        _require(
            video_id not in seen_video_ids,
            "task_input",
            f"{vpath}.video_id",
            f"duplicate video_id '{video_id}'",
        )
        seen_video_ids.add(video_id)

        status = video["status"]
        _require(
            status in ALL_STATUSES,
            "task_input",
            f"{vpath}.status",
            f"must be one of {sorted(ALL_STATUSES)}",
        )

        tracks = video.get("tracks", [])
        _require(isinstance(tracks, list), "task_input", f"{vpath}.tracks", "must be a list when provided")

        if status == "processed_with_tracks":
            _require(tracks, "task_input", f"{vpath}.tracks", "must be non-empty for processed_with_tracks")
        elif status == "processed_zero_tracks":
            _require(not tracks, "task_input", f"{vpath}.tracks", "must be empty for processed_zero_tracks")
        else:
            _require(not tracks, "task_input", f"{vpath}.tracks", "must be empty for failed/unprocessed")

        canonical_tracks: List[Dict[str, Any]] = []
        seen_track_ids = set()

        for t_idx, track in enumerate(tracks):
            tpath = f"{vpath}.tracks[{t_idx}]"
            _require(isinstance(track, dict), "task_input", tpath, "must be an object")
            required_track = [
                "track_id",
                "start_frame_idx",
                "end_frame_idx",
                "num_active_frames",
                "objectness_score",
                "embedding",
            ]
            for key in required_track:
                _require(key in track, "task_input", f"{tpath}.{key}", "required field missing")

            track_id = track["track_id"]
            _require(
                (isinstance(track_id, str) and track_id) or _is_int(track_id),
                "task_input",
                f"{tpath}.track_id",
                "must be non-empty string or integer",
            )
            track_id_type = type(track_id)
            if split_track_id_type is None:
                split_track_id_type = track_id_type
            _require(
                track_id_type is split_track_id_type,
                "task_input",
                f"{tpath}.track_id",
                "track_id serialized type must be consistent within split",
            )
            _require(
                track_id not in seen_track_ids,
                "task_input",
                f"{tpath}.track_id",
                f"duplicate track_id '{track_id}' within video",
            )
            seen_track_ids.add(track_id)

            start_frame_idx = track["start_frame_idx"]
            end_frame_idx = track["end_frame_idx"]
            num_active_frames = track["num_active_frames"]
            objectness_score = track["objectness_score"]
            embedding = track["embedding"]

            _require(
                _is_int(start_frame_idx) and start_frame_idx >= 0,
                "task_input",
                f"{tpath}.start_frame_idx",
                "must be integer >= 0",
            )
            _require(
                _is_int(end_frame_idx) and end_frame_idx >= start_frame_idx,
                "task_input",
                f"{tpath}.end_frame_idx",
                "must be integer >= start_frame_idx",
            )
            _require(
                _is_int(num_active_frames) and num_active_frames > 0,
                "task_input",
                f"{tpath}.num_active_frames",
                "must be integer > 0",
            )
            _require(
                _is_number(objectness_score),
                "task_input",
                f"{tpath}.objectness_score",
                "must be a number",
            )
            _require(
                isinstance(embedding, list) and len(embedding) == embedding_dim,
                "task_input",
                f"{tpath}.embedding",
                f"must be a list of length embedding_dim ({embedding_dim})",
            )
            for e_idx, value in enumerate(embedding):
                _require(
                    _is_number(value),
                    "task_input",
                    f"{tpath}.embedding[{e_idx}]",
                    "must be a number",
                )

            canonical_tracks.append(
                {
                    "track_id": track_id,
                    "start_frame_idx": start_frame_idx,
                    "end_frame_idx": end_frame_idx,
                    "num_active_frames": num_active_frames,
                    "objectness_score": float(objectness_score),
                    "embedding": [float(v) for v in embedding],
                }
            )

        if split_track_id_type is int:
            sort_key = lambda tr: (tr["start_frame_idx"], tr["end_frame_idx"], int(tr["track_id"]))
        else:
            sort_key = lambda tr: (tr["start_frame_idx"], tr["end_frame_idx"], str(tr["track_id"]))

        canonical_tracks.sort(key=sort_key)
        for row_index, track in enumerate(canonical_tracks):
            track["row_index"] = row_index

        canonical_videos.append(
            {
                "video_id": video_id,
                "status": status,
                "num_tracks": len(canonical_tracks),
                "tracks": canonical_tracks,
            }
        )

    canonical_videos.sort(key=lambda x: x["video_id"])

    return {
        "split": split,
        "embedding_dim": embedding_dim,
        "embedding_dtype": "float32",
        "embedding_pooling": "track_pooled",
        "embedding_normalization": payload["embedding_normalization"],
        "producer": producer,
        "videos": canonical_videos,
    }


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_split_export(canonical: Dict[str, Any], split_root: Path) -> None:
    manifest_videos: List[Dict[str, Any]] = []

    for video in canonical["videos"]:
        video_id = video["video_id"]
        status = video["status"]
        num_tracks = video["num_tracks"]
        tracks = video["tracks"]

        if status in PROCESSED_STATUSES:
            meta_rel, npz_rel = _relative_video_paths(video_id)
            video_root = split_root / "videos" / video_id
            video_root.mkdir(parents=True, exist_ok=True)

            track_metadata = {
                "schema_name": VIDEO_SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "split": canonical["split"],
                "video_id": video_id,
                "num_tracks": num_tracks,
                "tracks": [
                    {
                        "row_index": tr["row_index"],
                        "track_id": tr["track_id"],
                        "start_frame_idx": tr["start_frame_idx"],
                        "end_frame_idx": tr["end_frame_idx"],
                        "num_active_frames": tr["num_active_frames"],
                        "objectness_score": tr["objectness_score"],
                    }
                    for tr in tracks
                ],
            }
            _dump_json(video_root / "track_metadata.v1.json", track_metadata)

            embeddings = np.asarray([tr["embedding"] for tr in tracks], dtype=np.float32)
            if num_tracks == 0:
                embeddings = np.zeros((0, canonical["embedding_dim"]), dtype=np.float32)
            track_row_index = np.arange(num_tracks, dtype=np.int64)
            np.savez(
                video_root / "track_arrays.v1.npz",
                embeddings=embeddings,
                track_row_index=track_row_index,
            )

            manifest_videos.append(
                {
                    "video_id": video_id,
                    "status": status,
                    "num_tracks": num_tracks,
                    "track_metadata_path": meta_rel,
                    "track_arrays_path": npz_rel,
                }
            )
        else:
            manifest_videos.append(
                {
                    "video_id": video_id,
                    "status": status,
                    "num_tracks": 0,
                    "track_metadata_path": None,
                    "track_arrays_path": None,
                }
            )

    manifest = {
        "schema_name": MANIFEST_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "split": canonical["split"],
        "embedding_dim": canonical["embedding_dim"],
        "embedding_dtype": canonical["embedding_dtype"],
        "embedding_pooling": canonical["embedding_pooling"],
        "embedding_normalization": canonical["embedding_normalization"],
        "producer": canonical["producer"],
        "videos": manifest_videos,
    }
    _dump_json(split_root / "manifest.v1.json", manifest)


def build_track_feature_export_v1(
    input_payload: Dict[str, Any],
    output_split_root: Path,
    overwrite: bool = False,
) -> Path:
    canonical = _validate_and_canonicalize_task_input(input_payload)

    output_split_root = output_split_root.resolve()
    if output_split_root.exists() and not overwrite:
        raise ExportContractError(
            f"producer output path already exists: {output_split_root}. Use --overwrite to replace it."
        )

    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"stageb_track_export_v1_{canonical['split']}_", dir=str(output_split_root.parent))
    )
    try:
        _write_split_export(canonical=canonical, split_root=temp_dir)
        validate_track_feature_export_v1(split_root=temp_dir)

        if output_split_root.exists():
            shutil.rmtree(output_split_root)
        shutil.move(str(temp_dir), str(output_split_root))
        return output_split_root
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _load_json(path: Path, file_type: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportContractError(f"{file_type} file missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExportContractError(f"{file_type} invalid JSON at {path}: {exc}") from exc
    _require(isinstance(payload, dict), file_type, "$", "top-level value must be an object")
    return payload


def _require_relative_path(path_value: Any, file_type: str, field_path: str) -> PurePosixPath:
    _require(isinstance(path_value, str) and path_value, file_type, field_path, "must be a non-empty relative path")
    rel = PurePosixPath(path_value)
    _require(not rel.is_absolute(), file_type, field_path, "must be relative (absolute path is forbidden)")
    _require(".." not in rel.parts, file_type, field_path, "must not contain '..'")
    return rel


def _sort_tracks_expected(tracks: Iterable[Dict[str, Any]], track_id_type: Optional[type]) -> List[Dict[str, Any]]:
    items = list(tracks)
    if track_id_type is int:
        items.sort(key=lambda tr: (tr["start_frame_idx"], tr["end_frame_idx"], int(tr["track_id"])))
    else:
        items.sort(key=lambda tr: (tr["start_frame_idx"], tr["end_frame_idx"], str(tr["track_id"])))
    return items


def validate_track_feature_export_v1(
    split_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> Path:
    _require(
        bool(split_root) ^ bool(manifest_path),
        "validator",
        "args",
        "provide exactly one of split_root or manifest_path",
    )

    if manifest_path is not None:
        manifest_path = manifest_path.resolve()
        split_root = manifest_path.parent
    else:
        split_root = split_root.resolve()  # type: ignore[union-attr]
        manifest_path = split_root / "manifest.v1.json"

    manifest = _load_json(manifest_path, "manifest.v1.json")

    _require(
        manifest.get("schema_name") == MANIFEST_SCHEMA_NAME,
        "manifest.v1.json",
        "schema_name",
        f"must equal '{MANIFEST_SCHEMA_NAME}'",
    )
    _validate_semver_exact(manifest.get("schema_version"), "manifest.v1.json", "schema_version")

    _require(
        isinstance(manifest.get("split"), str) and manifest["split"],
        "manifest.v1.json",
        "split",
        "must be a non-empty string",
    )
    split = manifest["split"]

    _require(
        _is_int(manifest.get("embedding_dim")) and manifest["embedding_dim"] > 0,
        "manifest.v1.json",
        "embedding_dim",
        "must be integer > 0",
    )
    _require(
        manifest.get("embedding_dtype") == "float32",
        "manifest.v1.json",
        "embedding_dtype",
        "must equal 'float32'",
    )
    _require(
        manifest.get("embedding_pooling") == "track_pooled",
        "manifest.v1.json",
        "embedding_pooling",
        "must equal 'track_pooled'",
    )
    _require(
        manifest.get("embedding_normalization") in {"none", "l2"},
        "manifest.v1.json",
        "embedding_normalization",
        "must be one of ['none', 'l2']",
    )
    _validate_producer_block(manifest.get("producer"), split=split)

    videos = manifest.get("videos")
    _require(isinstance(videos, list), "manifest.v1.json", "videos", "must be a list")

    seen_video_ids = set()
    expected_video_order = sorted([v.get("video_id") for v in videos if isinstance(v, dict)])
    actual_video_order = [v.get("video_id") for v in videos if isinstance(v, dict)]
    _require(
        expected_video_order == actual_video_order,
        "manifest.v1.json",
        "videos",
        "must be sorted lexicographically by video_id",
    )

    split_track_id_type: Optional[type] = None

    for idx, video in enumerate(videos):
        vpath = f"videos[{idx}]"
        _require(isinstance(video, dict), "manifest.v1.json", vpath, "must be an object")
        for key in ("video_id", "status", "num_tracks", "track_metadata_path", "track_arrays_path"):
            _require(key in video, "manifest.v1.json", f"{vpath}.{key}", "required field missing")

        video_id = video["video_id"]
        _require(
            isinstance(video_id, str) and video_id,
            "manifest.v1.json",
            f"{vpath}.video_id",
            "must be a non-empty string",
        )
        _require(
            video_id not in seen_video_ids,
            "manifest.v1.json",
            f"{vpath}.video_id",
            f"duplicate video_id '{video_id}'",
        )
        seen_video_ids.add(video_id)

        status = video["status"]
        _require(
            status in ALL_STATUSES,
            "manifest.v1.json",
            f"{vpath}.status",
            f"must be one of {sorted(ALL_STATUSES)}",
        )
        num_tracks = video["num_tracks"]
        _require(
            _is_int(num_tracks) and num_tracks >= 0,
            "manifest.v1.json",
            f"{vpath}.num_tracks",
            "must be integer >= 0",
        )

        if status in PROCESSED_STATUSES:
            if status == "processed_with_tracks":
                _require(
                    num_tracks > 0,
                    "manifest.v1.json",
                    f"{vpath}.num_tracks",
                    "must be > 0 for processed_with_tracks",
                )
            else:
                _require(
                    num_tracks == 0,
                    "manifest.v1.json",
                    f"{vpath}.num_tracks",
                    "must equal 0 for processed_zero_tracks",
                )

            meta_rel = _require_relative_path(
                video["track_metadata_path"], "manifest.v1.json", f"{vpath}.track_metadata_path"
            )
            npz_rel = _require_relative_path(
                video["track_arrays_path"], "manifest.v1.json", f"{vpath}.track_arrays_path"
            )
            meta_path = split_root / Path(str(meta_rel))
            npz_path = split_root / Path(str(npz_rel))
            _require(meta_path.exists(), "manifest.v1.json", f"{vpath}.track_metadata_path", "target file missing on disk")
            _require(npz_path.exists(), "manifest.v1.json", f"{vpath}.track_arrays_path", "target file missing on disk")

            video_meta = _load_json(meta_path, "track_metadata.v1.json")
            _require(
                video_meta.get("schema_name") == VIDEO_SCHEMA_NAME,
                "track_metadata.v1.json",
                "schema_name",
                f"must equal '{VIDEO_SCHEMA_NAME}'",
            )
            _validate_semver_exact(video_meta.get("schema_version"), "track_metadata.v1.json", "schema_version")
            _require(
                video_meta.get("split") == split,
                "track_metadata.v1.json",
                "split",
                "must match manifest split",
            )
            _require(
                video_meta.get("video_id") == video_id,
                "track_metadata.v1.json",
                "video_id",
                "must match manifest video_id",
            )
            _require(
                _is_int(video_meta.get("num_tracks")) and video_meta["num_tracks"] >= 0,
                "track_metadata.v1.json",
                "num_tracks",
                "must be integer >= 0",
            )

            tracks = video_meta.get("tracks")
            _require(isinstance(tracks, list), "track_metadata.v1.json", "tracks", "must be a list")
            _require(
                video_meta["num_tracks"] == len(tracks),
                "track_metadata.v1.json",
                "num_tracks",
                "must equal len(tracks)",
            )

            seen_track_ids = set()
            for t_idx, track in enumerate(tracks):
                tpath = f"tracks[{t_idx}]"
                _require(isinstance(track, dict), "track_metadata.v1.json", tpath, "must be an object")
                for key in (
                    "row_index",
                    "track_id",
                    "start_frame_idx",
                    "end_frame_idx",
                    "num_active_frames",
                    "objectness_score",
                ):
                    _require(key in track, "track_metadata.v1.json", f"{tpath}.{key}", "required field missing")

                _require(
                    _is_int(track["row_index"]) and track["row_index"] == t_idx,
                    "track_metadata.v1.json",
                    f"{tpath}.row_index",
                    "must equal its row position",
                )
                track_id = track["track_id"]
                _require(
                    (isinstance(track_id, str) and track_id) or _is_int(track_id),
                    "track_metadata.v1.json",
                    f"{tpath}.track_id",
                    "must be non-empty string or integer",
                )
                track_id_type = type(track_id)
                if split_track_id_type is None:
                    split_track_id_type = track_id_type
                _require(
                    track_id_type is split_track_id_type,
                    "track_metadata.v1.json",
                    f"{tpath}.track_id",
                    "track_id serialized type must be consistent within split",
                )
                _require(
                    track_id not in seen_track_ids,
                    "track_metadata.v1.json",
                    f"{tpath}.track_id",
                    f"duplicate track_id '{track_id}' within video",
                )
                seen_track_ids.add(track_id)

                _require(
                    _is_int(track["start_frame_idx"]) and track["start_frame_idx"] >= 0,
                    "track_metadata.v1.json",
                    f"{tpath}.start_frame_idx",
                    "must be integer >= 0",
                )
                _require(
                    _is_int(track["end_frame_idx"]) and track["end_frame_idx"] >= track["start_frame_idx"],
                    "track_metadata.v1.json",
                    f"{tpath}.end_frame_idx",
                    "must be integer >= start_frame_idx",
                )
                _require(
                    _is_int(track["num_active_frames"]) and track["num_active_frames"] > 0,
                    "track_metadata.v1.json",
                    f"{tpath}.num_active_frames",
                    "must be integer > 0",
                )
                _require(
                    _is_number(track["objectness_score"]),
                    "track_metadata.v1.json",
                    f"{tpath}.objectness_score",
                    "must be a number",
                )

            expected_sorted = _sort_tracks_expected(tracks, split_track_id_type)
            _require(
                expected_sorted == tracks,
                "track_metadata.v1.json",
                "tracks",
                "must be sorted by (start_frame_idx, end_frame_idx, track_id)",
            )

            try:
                npz_data = np.load(npz_path)
            except Exception as exc:
                raise ExportContractError(f"track_arrays.v1.npz could not be loaded at {npz_path}: {exc}") from exc

            required_npz_keys = {"embeddings", "track_row_index"}
            missing_npz = required_npz_keys - set(npz_data.files)
            _require(
                not missing_npz,
                "track_arrays.v1.npz",
                "keys",
                f"missing required keys: {sorted(missing_npz)}",
            )
            embeddings = npz_data["embeddings"]
            track_row_index = npz_data["track_row_index"]

            _require(
                embeddings.dtype == np.float32,
                "track_arrays.v1.npz",
                "embeddings.dtype",
                "must be float32",
            )
            _require(
                embeddings.ndim == 2,
                "track_arrays.v1.npz",
                "embeddings.shape",
                "must be rank-2 [N, D]",
            )
            _require(
                embeddings.shape[1] == manifest["embedding_dim"],
                "track_arrays.v1.npz",
                "embeddings.shape[1]",
                "must equal manifest embedding_dim",
            )
            _require(
                track_row_index.dtype == np.int64,
                "track_arrays.v1.npz",
                "track_row_index.dtype",
                "must be int64",
            )
            _require(
                track_row_index.ndim == 1,
                "track_arrays.v1.npz",
                "track_row_index.shape",
                "must be rank-1 [N]",
            )
            _require(
                track_row_index.shape[0] == num_tracks,
                "track_arrays.v1.npz",
                "track_row_index.shape[0]",
                "must equal manifest video num_tracks",
            )
            _require(
                embeddings.shape[0] == num_tracks,
                "track_arrays.v1.npz",
                "embeddings.shape[0]",
                "must equal manifest video num_tracks",
            )
            _require(
                video_meta["num_tracks"] == num_tracks,
                "manifest.v1.json",
                f"{vpath}.num_tracks",
                "must equal track_metadata.num_tracks",
            )

            expected_rows = np.arange(num_tracks, dtype=np.int64)
            _require(
                np.array_equal(track_row_index, expected_rows),
                "track_arrays.v1.npz",
                "track_row_index",
                "must equal [0..N-1]",
            )

            if "objectness_score" in npz_data.files:
                obj = npz_data["objectness_score"]
                _require(
                    obj.dtype == np.float32 and obj.ndim == 1 and obj.shape[0] == num_tracks,
                    "track_arrays.v1.npz",
                    "objectness_score",
                    "if present, must be float32 vector with length N",
                )
        else:
            _require(
                num_tracks == 0,
                "manifest.v1.json",
                f"{vpath}.num_tracks",
                "must equal 0 for failed/unprocessed",
            )
            _require(
                video["track_metadata_path"] is None,
                "manifest.v1.json",
                f"{vpath}.track_metadata_path",
                "must be null for failed/unprocessed",
            )
            _require(
                video["track_arrays_path"] is None,
                "manifest.v1.json",
                f"{vpath}.track_arrays_path",
                "must be null for failed/unprocessed",
            )

    return split_root
