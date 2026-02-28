from __future__ import annotations

import json
import math
import shutil
import struct
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Tuple

try:
    from .v1_core import ExportContractError
except Exception:  # pragma: no cover - fallback for standalone loading without package deps.
    class ExportContractError(ValueError):
        """Raised when feature-export enablement input violates the contract."""

CONTRACT_NAME = "stageb_feature_export_enablement_contract_v1"
CONTRACT_VERSION = "v1"
_ALLOWED_NORMALIZATION = {"none", "l2"}
_ALLOWED_COMPLETION_MARKERS = {"completed", "failed", "unknown"}
_ALLOWED_EVIDENCE_CONFIDENCE = {"explicit", "inferred"}
_EMBEDDING_DTYPE = "float32"


def _err(field_path: str, rule_summary: str) -> ExportContractError:
    return ExportContractError(f"feature_export_input field '{field_path}': {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path, rule_summary)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_feature_export_enablement_input(input_json_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExportContractError(f"feature_export_input file not found: {input_json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ExportContractError(f"feature_export_input invalid JSON at {input_json_path}: {exc}") from exc
    _require(isinstance(payload, dict), "$", "top-level value must be an object")
    return payload


def _validate_string(field_path: str, value: Any) -> str:
    _require(isinstance(value, str) and value, field_path, "must be a non-empty string")
    return value


def _validate_id(field_path: str, value: Any) -> Any:
    _require(
        (isinstance(value, str) and value) or _is_int(value),
        field_path,
        "must be a non-empty string or integer",
    )
    return value


def _sort_id_key(value: Any) -> Tuple[int, Any]:
    if _is_int(value):
        return (0, int(value))
    return (1, str(value))


def _validate_runtime_evidence(runtime_evidence: Any, field_path: str) -> Dict[str, Any]:
    _require(isinstance(runtime_evidence, dict), field_path, "must be an object")

    required = ["stageb_completion_marker", "evidence_source"]
    for key in required:
        _require(key in runtime_evidence, f"{field_path}.{key}", "required field missing")

    marker = runtime_evidence["stageb_completion_marker"]
    if marker == "unprocessed":
        raise _err(
            f"{field_path}.stageb_completion_marker",
            "must not use downstream token 'unprocessed'",
        )
    _require(
        isinstance(marker, str) and marker in _ALLOWED_COMPLETION_MARKERS,
        f"{field_path}.stageb_completion_marker",
        f"must be one of {sorted(_ALLOWED_COMPLETION_MARKERS)}",
    )

    evidence_source = runtime_evidence["evidence_source"]
    _require(
        isinstance(evidence_source, str) and evidence_source,
        f"{field_path}.evidence_source",
        "must be a non-empty string",
    )

    canonical: Dict[str, Any] = {
        "stageb_completion_marker": marker,
        "evidence_source": evidence_source,
    }

    if "evidence_confidence" in runtime_evidence:
        confidence = runtime_evidence["evidence_confidence"]
        _require(
            isinstance(confidence, str) and confidence in _ALLOWED_EVIDENCE_CONFIDENCE,
            f"{field_path}.evidence_confidence",
            f"must be one of {sorted(_ALLOWED_EVIDENCE_CONFIDENCE)} when provided",
        )
        canonical["evidence_confidence"] = confidence

    if "failure_reason" in runtime_evidence:
        failure_reason = runtime_evidence["failure_reason"]
        _require(
            isinstance(failure_reason, str) and failure_reason,
            f"{field_path}.failure_reason",
            "must be a non-empty string when provided",
        )
        canonical["failure_reason"] = failure_reason

    return canonical


def _validate_embedding(
    embedding: Any,
    embedding_dim: int,
    field_path: str,
) -> List[float]:
    _require(isinstance(embedding, list) and embedding, field_path, "must be a non-empty array")
    _require(len(embedding) == embedding_dim, field_path, f"length must equal embedding_dim={embedding_dim}")

    cast_values: List[float] = []
    for idx, value in enumerate(embedding):
        _require(_is_number(value), f"{field_path}[{idx}]", "must be numeric")
        _require(math.isfinite(float(value)), f"{field_path}[{idx}]", "must be finite (no NaN/Inf)")
        cast_value = struct.unpack("!f", struct.pack("!f", float(value)))[0]
        _require(
            math.isfinite(float(cast_value)),
            f"{field_path}[{idx}]",
            "must remain finite after float32 cast",
        )
        cast_values.append(cast_value)
    return cast_values


def _validate_tracks(
    tracks: Any,
    embedding_dim: int,
    run_normalization: str,
    field_path: str,
) -> List[Dict[str, Any]]:
    _require(isinstance(tracks, list), field_path, "must be an array")

    canonical_tracks: List[Dict[str, Any]] = []
    seen_track_ids = set()
    for idx, track in enumerate(tracks):
        tpath = f"{field_path}[{idx}]"
        _require(isinstance(track, dict), tpath, "must be an object")
        for key in ("track_id", "embedding", "embedding_normalization"):
            _require(key in track, f"{tpath}.{key}", "required field missing")

        track_id = _validate_id(f"{tpath}.track_id", track["track_id"])
        _require(track_id not in seen_track_ids, f"{tpath}.track_id", f"duplicate track_id '{track_id}'")
        seen_track_ids.add(track_id)

        track_normalization = track["embedding_normalization"]
        _require(
            isinstance(track_normalization, str) and track_normalization in _ALLOWED_NORMALIZATION,
            f"{tpath}.embedding_normalization",
            f"must be one of {sorted(_ALLOWED_NORMALIZATION)}",
        )
        _require(
            track_normalization == run_normalization,
            f"{tpath}.embedding_normalization",
            "must match run-level embedding_normalization for v1",
        )

        canonical_track: Dict[str, Any] = {
            "track_id": track_id,
            "embedding": _validate_embedding(track["embedding"], embedding_dim=embedding_dim, field_path=f"{tpath}.embedding"),
            "embedding_normalization": track_normalization,
        }

        if "start_frame_idx" in track:
            _require(_is_int(track["start_frame_idx"]) and track["start_frame_idx"] >= 0, f"{tpath}.start_frame_idx", "must be integer >= 0")
            canonical_track["start_frame_idx"] = track["start_frame_idx"]
        if "end_frame_idx" in track:
            _require(_is_int(track["end_frame_idx"]), f"{tpath}.end_frame_idx", "must be integer when provided")
            start = canonical_track.get("start_frame_idx", 0)
            _require(track["end_frame_idx"] >= start, f"{tpath}.end_frame_idx", "must be >= start_frame_idx when provided")
            canonical_track["end_frame_idx"] = track["end_frame_idx"]
        if "num_active_frames" in track:
            _require(_is_int(track["num_active_frames"]) and track["num_active_frames"] > 0, f"{tpath}.num_active_frames", "must be integer > 0 when provided")
            canonical_track["num_active_frames"] = track["num_active_frames"]
        if "objectness_score" in track:
            _require(_is_number(track["objectness_score"]), f"{tpath}.objectness_score", "must be numeric when provided")
            _require(math.isfinite(float(track["objectness_score"])), f"{tpath}.objectness_score", "must be finite when provided")
            canonical_track["objectness_score"] = float(track["objectness_score"])

        canonical_tracks.append(canonical_track)

    canonical_tracks.sort(key=lambda tr: _sort_id_key(tr["track_id"]))
    return canonical_tracks


def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    required_manifest_fields = [
        "run_id",
        "split",
        "embedding_dim",
        "embedding_normalization",
        "stageb_checkpoint_ref",
        "stageb_checkpoint_hash",
        "stageb_config_ref",
        "stageb_config_hash",
        "pseudo_tube_manifest_ref",
        "pseudo_tube_manifest_hash",
        "extraction_settings",
        "videos",
    ]
    for key in required_manifest_fields:
        _require(key in payload, key, "required field missing")

    run_id = _validate_string("run_id", payload["run_id"])
    split = _validate_string("split", payload["split"])

    embedding_dim = payload["embedding_dim"]
    _require(_is_int(embedding_dim) and embedding_dim > 0, "embedding_dim", "must be integer > 0")

    embedding_normalization = payload["embedding_normalization"]
    _require(
        isinstance(embedding_normalization, str) and embedding_normalization in _ALLOWED_NORMALIZATION,
        "embedding_normalization",
        f"must be one of {sorted(_ALLOWED_NORMALIZATION)}",
    )

    extraction_settings = payload["extraction_settings"]
    _require(isinstance(extraction_settings, dict), "extraction_settings", "must be an object")

    canonical: Dict[str, Any] = {
        "run_id": run_id,
        "split": split,
        "embedding_dim": embedding_dim,
        "embedding_dtype": _EMBEDDING_DTYPE,
        "embedding_normalization": embedding_normalization,
        "stageb_checkpoint_ref": _validate_string("stageb_checkpoint_ref", payload["stageb_checkpoint_ref"]),
        "stageb_checkpoint_hash": _validate_string("stageb_checkpoint_hash", payload["stageb_checkpoint_hash"]),
        "stageb_config_ref": _validate_string("stageb_config_ref", payload["stageb_config_ref"]),
        "stageb_config_hash": _validate_string("stageb_config_hash", payload["stageb_config_hash"]),
        "pseudo_tube_manifest_ref": _validate_string("pseudo_tube_manifest_ref", payload["pseudo_tube_manifest_ref"]),
        "pseudo_tube_manifest_hash": _validate_string("pseudo_tube_manifest_hash", payload["pseudo_tube_manifest_hash"]),
        "extraction_settings": extraction_settings,
    }

    if "created_at_utc" in payload:
        canonical["created_at_utc"] = _validate_string("created_at_utc", payload["created_at_utc"])
    if "notes" in payload:
        canonical["notes"] = _validate_string("notes", payload["notes"])

    videos = payload["videos"]
    _require(isinstance(videos, list), "videos", "must be an array")

    canonical_videos: List[Dict[str, Any]] = []
    seen_video_ids = set()
    for idx, video in enumerate(videos):
        vpath = f"videos[{idx}]"
        _require(isinstance(video, dict), vpath, "must be an object")
        for key in ("video_id", "runtime_evidence", "tracks"):
            _require(key in video, f"{vpath}.{key}", "required field missing")

        video_id = _validate_id(f"{vpath}.video_id", video["video_id"])
        _require(video_id not in seen_video_ids, f"{vpath}.video_id", f"duplicate video_id '{video_id}'")
        seen_video_ids.add(video_id)

        canonical_video: Dict[str, Any] = {
            "video_id": video_id,
            "runtime_evidence": _validate_runtime_evidence(video["runtime_evidence"], f"{vpath}.runtime_evidence"),
            "tracks": _validate_tracks(
                video["tracks"],
                embedding_dim=embedding_dim,
                run_normalization=embedding_normalization,
                field_path=f"{vpath}.tracks",
            ),
        }

        if "export_warnings" in video:
            warnings = video["export_warnings"]
            _require(isinstance(warnings, list), f"{vpath}.export_warnings", "must be an array when provided")
            for w_idx, warning in enumerate(warnings):
                _require(
                    isinstance(warning, str) and warning,
                    f"{vpath}.export_warnings[{w_idx}]",
                    "must be a non-empty string",
                )
            canonical_video["export_warnings"] = warnings

        if "source_artifacts" in video:
            _require(
                isinstance(video["source_artifacts"], dict),
                f"{vpath}.source_artifacts",
                "must be an object when provided",
            )
            canonical_video["source_artifacts"] = video["source_artifacts"]

        canonical_videos.append(canonical_video)

    canonical_videos.sort(key=lambda item: _sort_id_key(item["video_id"]))
    canonical["videos"] = canonical_videos
    return canonical


def build_feature_export_enablement_v1(
    input_payload: Dict[str, Any],
    run_root: Path,
    *,
    overwrite: bool = False,
    emit_video_index: bool = False,
) -> Path:
    canonical = _validate_payload(input_payload)

    output_root = run_root / "d2" / "inference" / "feature_export_v1"
    if output_root.exists():
        if not overwrite:
            raise ExportContractError(
                f"output path exists: {output_root} (use overwrite=True to replace existing directory)"
            )
        shutil.rmtree(output_root)

    videos_dir = output_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    video_shards: List[str] = []
    video_index: List[Dict[str, Any]] = []
    for video in canonical["videos"]:
        video_id = video["video_id"]
        video_id_str = str(video_id)
        shard_rel = PurePosixPath("videos") / f"{video_id_str}.json"
        shard_abs = output_root / shard_rel
        _json_write(shard_abs, video)
        video_shards.append(str(shard_rel))
        video_index.append({"video_id": video_id, "shard": str(shard_rel)})

    manifest: Dict[str, Any] = {
        "contract_name": CONTRACT_NAME,
        "contract_version": CONTRACT_VERSION,
        "run_id": canonical["run_id"],
        "split": canonical["split"],
        "embedding_dim": canonical["embedding_dim"],
        "embedding_dtype": canonical["embedding_dtype"],
        "embedding_normalization": canonical["embedding_normalization"],
        "video_shards": video_shards,
        "stageb_checkpoint_ref": canonical["stageb_checkpoint_ref"],
        "stageb_checkpoint_hash": canonical["stageb_checkpoint_hash"],
        "stageb_config_ref": canonical["stageb_config_ref"],
        "stageb_config_hash": canonical["stageb_config_hash"],
        "pseudo_tube_manifest_ref": canonical["pseudo_tube_manifest_ref"],
        "pseudo_tube_manifest_hash": canonical["pseudo_tube_manifest_hash"],
        "extraction_settings": canonical["extraction_settings"],
    }
    if "created_at_utc" in canonical:
        manifest["created_at_utc"] = canonical["created_at_utc"]
    if "notes" in canonical:
        manifest["notes"] = canonical["notes"]

    _json_write(output_root / "manifest.json", manifest)
    if emit_video_index:
        _json_write(output_root / "video_index.json", {"videos": video_index})

    return output_root
