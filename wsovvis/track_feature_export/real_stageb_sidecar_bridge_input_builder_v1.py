from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .v1_core import ExportContractError

_ALLOWED_RUNTIME = {"success", "failed"}
_QA_SUMMARY_SCHEMA_VERSION = "p3_1c2_real_stageb_bridge_qa_v1"
_MARKER_TO_RUNTIME = {
    "completed": "success",
    "failed": "failed",
    "unknown": "failed",
}
_DROP_REASONS = (
    "non_finite_embedding",
    "missing_required_field",
    "invalid_track_id",
    "invalid_temporal_range",
    "invalid_objectness",
    "embedding_dim_mismatch",
    "other",
)


def _err(field_path: str, rule_summary: str) -> ExportContractError:
    return ExportContractError(f"real_stageb_bridge_builder field '{field_path}': {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path, rule_summary)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _json_load(path: Path, *, field_path: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _err(field_path, f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise _err(field_path, f"invalid JSON at {path}: {exc}") from exc


def _resolve_split_domain_json_path(
    repo_root: Path,
    run_root: Optional[Path],
    config_json: Dict[str, Any],
) -> Path:
    data = config_json.get("data")
    _require(isinstance(data, dict), "config.json.data", "must be an object")
    val_json = data.get("val_json")
    _require(isinstance(val_json, str) and val_json, "config.json.data.val_json", "must be a non-empty string")
    path = Path(val_json)
    if not path.is_absolute():
        candidates: List[Path] = [repo_root / path]
        if run_root is not None:
            probe = run_root
            while probe != probe.parent:
                candidates.append(probe / path)
                probe = probe.parent
        candidates.append(Path.cwd() / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        path = candidates[0]
    return path


def _load_split_domain_video_ids(
    repo_root: Path,
    run_root: Optional[Path],
    config_json_path: Path,
) -> Tuple[str, List[str]]:
    config_json = _json_load(config_json_path, field_path="config.json")
    _require(isinstance(config_json, dict), "config.json", "top-level value must be an object")

    split_domain_path = _resolve_split_domain_json_path(repo_root, run_root, config_json)
    split_domain_payload = _json_load(split_domain_path, field_path="config.json.data.val_json")
    _require(isinstance(split_domain_payload, dict), "split_domain_json", "top-level value must be an object")

    videos = split_domain_payload.get("videos")
    _require(isinstance(videos, list), "split_domain_json.videos", "must be a list")

    canonical_ids: List[str] = []
    seen = set()
    for idx, video in enumerate(videos):
        field = f"split_domain_json.videos[{idx}].id"
        _require(isinstance(video, dict), f"split_domain_json.videos[{idx}]", "must be an object")
        _require("id" in video, field, "required field missing")
        raw_id = video["id"]
        _require((_is_int(raw_id) or (isinstance(raw_id, str) and raw_id)), field, "must be int or non-empty string")
        canonical = str(raw_id)
        _require(canonical not in seen, field, f"duplicate canonical video_id '{canonical}'")
        seen.add(canonical)
        canonical_ids.append(canonical)

    split = config_json.get("split")
    if not (isinstance(split, str) and split):
        split = "val"

    return split, sorted(canonical_ids)


def _infer_repo_root(run_root: Optional[Path], sidecar_root: Optional[Path]) -> Path:
    if run_root is not None:
        candidate = run_root
        while candidate != candidate.parent:
            if (candidate / "tools").exists() and (candidate / "wsovvis").exists():
                return candidate
            candidate = candidate.parent
    if sidecar_root is not None:
        candidate = sidecar_root
        while candidate != candidate.parent:
            if (candidate / "tools").exists() and (candidate / "wsovvis").exists():
                return candidate
            candidate = candidate.parent
    return Path.cwd()


def _load_manifest(sidecar_root: Path) -> Dict[str, Any]:
    manifest = _json_load(sidecar_root / "manifest.json", field_path="sidecar.manifest")
    _require(isinstance(manifest, dict), "sidecar.manifest", "top-level value must be an object")
    shards = manifest.get("video_shards")
    _require(isinstance(shards, list) and shards, "sidecar.manifest.video_shards", "must be a non-empty list")
    for idx, rel in enumerate(shards):
        _require(isinstance(rel, str) and rel, f"sidecar.manifest.video_shards[{idx}]", "must be a non-empty string")
    return manifest


def _canonical_video_id(video_id: Any) -> str:
    _require((_is_int(video_id) or (isinstance(video_id, str) and video_id)), "video_id", "must be int or non-empty string")
    return str(video_id)


def _canonical_track_id(track_id: Any, field_path: str) -> Any:
    _require((_is_int(track_id) or (isinstance(track_id, str) and track_id)), field_path, "must be int or non-empty string")
    return track_id


def _load_sidecar_tracks(sidecar_root: Path, manifest: Dict[str, Any], selected_videos: set[str]) -> Tuple[Dict[str, str], Dict[Tuple[str, Any], Dict[str, Any]], int]:
    video_runtime_status: Dict[str, str] = {}
    tracks_by_key: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    duplicate_keys = 0

    for idx, shard_rel in enumerate(manifest["video_shards"]):
        shard_path = sidecar_root / shard_rel
        payload = _json_load(shard_path, field_path=f"sidecar.videos[{idx}]")
        _require(isinstance(payload, dict), f"sidecar.videos[{idx}]", "top-level value must be an object")
        _require("video_id" in payload, f"sidecar.videos[{idx}].video_id", "required field missing")
        video_id = _canonical_video_id(payload["video_id"])
        if video_id not in selected_videos:
            continue

        runtime_evidence = payload.get("runtime_evidence")
        _require(isinstance(runtime_evidence, dict), f"sidecar.videos[{idx}].runtime_evidence", "must be an object")
        marker = runtime_evidence.get("stageb_completion_marker")
        _require(isinstance(marker, str) and marker in _MARKER_TO_RUNTIME, f"sidecar.videos[{idx}].runtime_evidence.stageb_completion_marker", "must be one of ['completed', 'failed', 'unknown']")
        runtime_status = _MARKER_TO_RUNTIME[marker]
        video_runtime_status[video_id] = runtime_status

        tracks = payload.get("tracks")
        _require(isinstance(tracks, list), f"sidecar.videos[{idx}].tracks", "must be a list")
        for t_idx, track in enumerate(tracks):
            tpath = f"sidecar.videos[{idx}].tracks[{t_idx}]"
            _require(isinstance(track, dict), tpath, "must be an object")
            _require("track_id" in track, f"{tpath}.track_id", "required field missing")
            canonical_track = _canonical_track_id(track["track_id"], f"{tpath}.track_id")
            key = (video_id, canonical_track)
            if key in tracks_by_key:
                duplicate_keys += 1
                raise _err(tpath, f"duplicate join key in sidecar: {key}")
            tracks_by_key[key] = dict(track)

    return video_runtime_status, tracks_by_key, duplicate_keys


def _coerce_predictions_json(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        # Common shapes seen in legacy dump paths.
        if isinstance(payload.get("predictions"), list):
            return [p for p in payload["predictions"] if isinstance(p, dict)]
        if isinstance(payload.get("annotations"), list):
            return [p for p in payload["annotations"] if isinstance(p, dict)]
    raise _err("results.json", "unsupported JSON prediction shape")


def _load_predictions(inference_root: Path) -> Tuple[List[Dict[str, Any]], str]:
    pth_path = inference_root / "instances_predictions.pth"
    if pth_path.exists():
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise _err("instances_predictions.pth", "torch is required to load .pth predictions") from exc

        try:
            payload = torch.load(pth_path, map_location="cpu")
        except Exception as exc:
            raise _err("instances_predictions.pth", f"failed to read .pth: {exc}") from exc
        _require(isinstance(payload, list), "instances_predictions.pth", "top-level value must be a list")
        return [p for p in payload if isinstance(p, dict)], str(pth_path)

    results_json = inference_root / "results.json"
    payload = _json_load(results_json, field_path="results.json")
    return _coerce_predictions_json(payload), str(results_json)


def _load_prediction_tracks(predictions: Sequence[Dict[str, Any]], selected_videos: set[str]) -> Tuple[Dict[Tuple[str, Any], Dict[str, Any]], int]:
    tracks_by_key: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    duplicate_keys = 0
    for idx, pred in enumerate(predictions):
        if "video_id" not in pred:
            continue
        video_id = _canonical_video_id(pred["video_id"])
        if video_id not in selected_videos:
            continue
        if "track_id" not in pred:
            continue
        track_id = _canonical_track_id(pred["track_id"], f"predictions[{idx}].track_id")
        key = (video_id, track_id)
        if key in tracks_by_key:
            duplicate_keys += 1
            raise _err(f"predictions[{idx}]", f"duplicate join key in predictions: {key}")
        tracks_by_key[key] = dict(pred)
    return tracks_by_key, duplicate_keys


def _merge_track(
    *,
    sidecar_track: Dict[str, Any],
    prediction_track: Dict[str, Any],
    embedding_dim: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    track_id = sidecar_track.get("track_id")
    if not (_is_int(track_id) or (isinstance(track_id, str) and track_id)):
        return None, "invalid_track_id"

    start_frame_idx = sidecar_track.get("start_frame_idx", prediction_track.get("start_frame_idx"))
    end_frame_idx = sidecar_track.get("end_frame_idx", prediction_track.get("end_frame_idx"))
    num_active_frames = sidecar_track.get("num_active_frames", prediction_track.get("num_active_frames"))
    objectness_score = sidecar_track.get("objectness_score", prediction_track.get("score"))
    embedding = sidecar_track.get("embedding")

    if start_frame_idx is None or end_frame_idx is None or num_active_frames is None or objectness_score is None or embedding is None:
        return None, "missing_required_field"
    if not (_is_int(start_frame_idx) and start_frame_idx >= 0):
        return None, "invalid_temporal_range"
    if not (_is_int(end_frame_idx) and end_frame_idx >= start_frame_idx):
        return None, "invalid_temporal_range"
    if not (_is_int(num_active_frames) and num_active_frames > 0):
        return None, "invalid_temporal_range"
    if not _is_finite_number(objectness_score):
        return None, "invalid_objectness"
    if not (isinstance(embedding, list) and len(embedding) == embedding_dim):
        if isinstance(embedding, list):
            return None, "embedding_dim_mismatch"
        return None, "missing_required_field"

    cast_embedding: List[float] = []
    for value in embedding:
        if not _is_finite_number(value):
            return None, "non_finite_embedding"
        cast_embedding.append(float(value))

    return {
        "track_id": track_id,
        "start_frame_idx": int(start_frame_idx),
        "end_frame_idx": int(end_frame_idx),
        "num_active_frames": int(num_active_frames),
        "objectness_score": float(objectness_score),
        "embedding": cast_embedding,
    }, ""


def _sort_track_key(track: Dict[str, Any]) -> Tuple[int, int, int, str]:
    track_id = track["track_id"]
    if _is_int(track_id):
        return (0, int(track["start_frame_idx"]), int(track["end_frame_idx"]), f"{track_id}")
    return (1, int(track["start_frame_idx"]), int(track["end_frame_idx"]), str(track_id))


def build_normalized_bridge_input_from_real_stageb_sidecar(
    *,
    run_root: Optional[Path],
    sidecar_root: Optional[Path] = None,
    inference_root: Optional[Path] = None,
    config_json_path: Optional[Path] = None,
    sample_video_limit: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _require(run_root is not None or sidecar_root is not None, "run_root", "provide run_root and/or sidecar_root")

    if run_root is not None:
        run_root = run_root.resolve()
        _require(run_root.exists(), "run_root", f"path not found: {run_root}")

    if sidecar_root is None:
        _require(run_root is not None, "sidecar_root", "required when run_root is absent")
        sidecar_root = run_root / "d2" / "inference" / "feature_export_v1"
    sidecar_root = sidecar_root.resolve()

    if inference_root is None:
        if run_root is not None:
            inference_root = run_root / "d2" / "inference"
        else:
            inference_root = sidecar_root.parent
    inference_root = inference_root.resolve()

    if config_json_path is None:
        if run_root is not None:
            config_json_path = run_root / "config.json"
        else:
            inferred_run_root = sidecar_root.parent.parent.parent
            config_json_path = inferred_run_root / "config.json"
    config_json_path = config_json_path.resolve()

    repo_root = _infer_repo_root(run_root=run_root, sidecar_root=sidecar_root)

    split_from_config, split_domain_ids = _load_split_domain_video_ids(
        repo_root=repo_root,
        run_root=run_root,
        config_json_path=config_json_path,
    )

    if sample_video_limit is not None:
        _require(_is_int(sample_video_limit) and sample_video_limit > 0, "sample_video_limit", "must be integer > 0 when provided")
        split_domain_ids = split_domain_ids[:sample_video_limit]

    selected_videos = set(split_domain_ids)

    manifest = _load_manifest(sidecar_root)
    manifest_split = manifest.get("split")
    split = str(manifest_split) if isinstance(manifest_split, str) and manifest_split else split_from_config

    sidecar_runtime_status, sidecar_tracks_by_key, sidecar_duplicates = _load_sidecar_tracks(
        sidecar_root=sidecar_root,
        manifest=manifest,
        selected_videos=selected_videos,
    )

    predictions, prediction_source_path = _load_predictions(inference_root)
    prediction_tracks_by_key, prediction_duplicates = _load_prediction_tracks(predictions, selected_videos=selected_videos)

    sidecar_keys = set(sidecar_tracks_by_key.keys())
    prediction_keys = set(prediction_tracks_by_key.keys())

    missing_sidecar = sorted(prediction_keys - sidecar_keys, key=lambda k: (k[0], str(k[1])))
    missing_prediction = sorted(sidecar_keys - prediction_keys, key=lambda k: (k[0], str(k[1])))
    if missing_sidecar:
        raise _err("join", f"prediction key(s) missing sidecar counterpart: {missing_sidecar[:5]} (count={len(missing_sidecar)})")
    if missing_prediction:
        raise _err("join", f"sidecar key(s) missing prediction counterpart: {missing_prediction[:5]} (count={len(missing_prediction)})")

    embedding_dim = manifest.get("embedding_dim")
    _require(_is_int(embedding_dim) and embedding_dim > 0, "sidecar.manifest.embedding_dim", "must be integer > 0")

    embedding_normalization = manifest.get("embedding_normalization")
    _require(
        isinstance(embedding_normalization, str) and embedding_normalization in {"none", "l2"},
        "sidecar.manifest.embedding_normalization",
        "must be one of ['none', 'l2']",
    )

    track_id_type: Optional[type] = None
    stageb_video_results: List[Dict[str, Any]] = []

    dropped_tracks = 0
    non_finite_rejects = 0
    drop_reason_counts: Dict[str, int] = {reason: 0 for reason in _DROP_REASONS}
    dropped_by_video: Dict[str, int] = {video_id: 0 for video_id in split_domain_ids}
    drop_reason_counts_by_video: Dict[str, Dict[str, int]] = {
        video_id: {reason: 0 for reason in _DROP_REASONS} for video_id in split_domain_ids
    }

    prediction_track_count_by_video: Dict[str, int] = {video_id: 0 for video_id in split_domain_ids}
    sidecar_track_count_by_video: Dict[str, int] = {video_id: 0 for video_id in split_domain_ids}
    matched_track_count_by_video: Dict[str, int] = {video_id: 0 for video_id in split_domain_ids}
    for video_id, _ in prediction_keys:
        prediction_track_count_by_video[video_id] += 1
    for video_id, _ in sidecar_keys:
        sidecar_track_count_by_video[video_id] += 1
    matched_keys = sidecar_keys & prediction_keys
    for video_id, _ in matched_keys:
        matched_track_count_by_video[video_id] += 1
    total_join_pairs = len(matched_keys)

    by_video_tracks: Dict[str, List[Dict[str, Any]]] = {video_id: [] for video_id in split_domain_ids}
    for video_id, track_id in sorted(sidecar_keys, key=lambda k: (k[0], str(k[1]))):
        sidecar_track = sidecar_tracks_by_key[(video_id, track_id)]
        prediction_track = prediction_tracks_by_key[(video_id, track_id)]
        merged, drop_reason = _merge_track(
            sidecar_track=sidecar_track,
            prediction_track=prediction_track,
            embedding_dim=int(embedding_dim),
        )
        if merged is None:
            dropped_tracks += 1
            reason_key = drop_reason if drop_reason in drop_reason_counts else "other"
            drop_reason_counts[reason_key] += 1
            drop_reason_counts_by_video[video_id][reason_key] += 1
            dropped_by_video[video_id] += 1
            if reason_key == "non_finite_embedding":
                non_finite_rejects += 1
            continue

        current_type = type(merged["track_id"])
        if track_id_type is None:
            track_id_type = current_type
        elif current_type is not track_id_type:
            raise _err(
                "join.track_id",
                "track_id serialized type must be consistent across split",
            )

        by_video_tracks[video_id].append(merged)

    for video_id in split_domain_ids:
        runtime_status = sidecar_runtime_status.get(video_id, "failed")
        _require(runtime_status in _ALLOWED_RUNTIME, f"stageb_video_results[{video_id}].runtime_status", "must be 'success' or 'failed'")

        tracks = by_video_tracks[video_id]
        tracks.sort(key=_sort_track_key)
        if runtime_status == "failed":
            tracks = []

        stageb_video_results.append(
            {
                "video_id": video_id,
                "runtime_status": runtime_status,
                "tracks": tracks,
            }
        )

    runtime_success_count = sum(1 for r in stageb_video_results if r["runtime_status"] == "success")
    runtime_failed_count = sum(1 for r in stageb_video_results if r["runtime_status"] == "failed")
    processed_zero_tracks_count = sum(
        1 for r in stageb_video_results if r["runtime_status"] == "success" and len(r["tracks"]) == 0
    )
    stageb_video_results_emitted = len(sidecar_runtime_status)
    unprocessed_estimate_count = len(split_domain_ids) - stageb_video_results_emitted

    producer = {
        "stage_b_checkpoint_id": manifest.get("stageb_checkpoint_ref"),
        "stage_b_checkpoint_hash": manifest.get("stageb_checkpoint_hash"),
        "stage_b_config_ref": manifest.get("stageb_config_ref"),
        "stage_b_config_hash": manifest.get("stageb_config_hash"),
        "pseudo_tube_manifest_id": manifest.get("pseudo_tube_manifest_ref"),
        "pseudo_tube_manifest_hash": manifest.get("pseudo_tube_manifest_hash"),
        "split": split,
        "extraction_settings": manifest.get("extraction_settings"),
    }

    for key, value in producer.items():
        if key == "extraction_settings":
            _require(isinstance(value, dict), f"producer.{key}", "must be an object")
            continue
        _require(isinstance(value, str) and value, f"producer.{key}", "must be a non-empty string")

    payload = {
        "split": split,
        "producer": producer,
        "split_domain_video_ids": split_domain_ids,
        "embedding_pooling": "track_pooled",
        "embedding_normalization": embedding_normalization,
        "stageb_video_results": stageb_video_results,
    }

    per_video_rows: List[Dict[str, Any]] = []
    for result in stageb_video_results:
        video_id = result["video_id"]
        row = {
            "video_id": video_id,
            "runtime_status": result["runtime_status"],
            "input_track_count_prediction": prediction_track_count_by_video[video_id],
            "input_track_count_sidecar": sidecar_track_count_by_video[video_id],
            "matched_track_count": matched_track_count_by_video[video_id],
            "dropped_track_count": dropped_by_video[video_id],
            "drop_reason_counts": drop_reason_counts_by_video[video_id],
            "final_track_count": len(result["tracks"]),
            "is_zero_tracks": bool(result["runtime_status"] == "success" and len(result["tracks"]) == 0),
        }
        per_video_rows.append(row)

    run_root_str = str(run_root) if run_root is not None else None
    summary: Dict[str, Any] = {
        "qa_summary_schema_version": _QA_SUMMARY_SCHEMA_VERSION,
        "split": split,
        "run_identity": {
            "run_root": run_root_str,
            "inference_root": str(inference_root),
            "sidecar_root": str(sidecar_root),
            "config_json_path": str(config_json_path),
            "sample_video_limit": sample_video_limit,
        },
        "prediction_source_path": prediction_source_path,
        "sidecar_root": str(sidecar_root),
        "sample_video_limit": sample_video_limit,
        "total_split_domain_videos": len(split_domain_ids),
        "total_join_pairs": total_join_pairs,
        "duplicates": {
            "sidecar": sidecar_duplicates,
            "prediction": prediction_duplicates,
        },
        "missing_counterparts": {
            "prediction_missing_sidecar": len(missing_sidecar),
            "sidecar_missing_prediction": len(missing_prediction),
        },
        "join": {
            "prediction_tracks_total": len(prediction_keys),
            "sidecar_tracks_total": len(sidecar_keys),
            "matched_tracks_total": total_join_pairs,
            "missing_sidecar_counterparts": len(missing_sidecar),
            "missing_prediction_counterparts": len(missing_prediction),
            "extra_prediction_tracks": len(missing_sidecar),
            "extra_sidecar_tracks": len(missing_prediction),
            "duplicate_prediction_keys": prediction_duplicates,
            "duplicate_sidecar_keys": sidecar_duplicates,
        },
        "dropped_tracks": dropped_tracks,
        "non_finite_rejects": non_finite_rejects,
        "drop": {
            "total": dropped_tracks,
            "by_reason": drop_reason_counts,
        },
        "runtime_status_counts": {
            "success": runtime_success_count,
            "failed": runtime_failed_count,
        },
        "runtime_status": {
            "success_videos": runtime_success_count,
            "failed_videos": runtime_failed_count,
        },
        "video_results": {
            "total_split_domain_videos": len(split_domain_ids),
            "with_stageb_result": stageb_video_results_emitted,
            "without_stageb_result": unprocessed_estimate_count,
            "processed_zero_tracks_count": processed_zero_tracks_count,
            "unprocessed_estimate_count": unprocessed_estimate_count,
        },
        "per_video_rows": per_video_rows,
    }

    return payload, summary


def build_normalized_bridge_input_from_real_stageb_sidecar_to_json(
    *,
    output_json_path: Path,
    run_root: Optional[Path],
    sidecar_root: Optional[Path] = None,
    inference_root: Optional[Path] = None,
    config_json_path: Optional[Path] = None,
    sample_video_limit: Optional[int] = None,
    summary_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    payload, summary = build_normalized_bridge_input_from_real_stageb_sidecar(
        run_root=run_root,
        sidecar_root=sidecar_root,
        inference_root=inference_root,
        config_json_path=config_json_path,
        sample_video_limit=sample_video_limit,
    )
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if summary_json_path is not None:
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return summary
