from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .feature_export_enablement_v1 import (
    ExportContractError,
    build_feature_export_enablement_v1,
)


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise ExportContractError(f"feature_export_input field '{path}': file not found for sha256")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return repo_root / path


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise ExportContractError("feature_export_input field 'embedding': value is not list-like")


def _optional_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExportContractError(f"feature_export_input field '{name}': must be integer when provided")
    return int(value)


def _build_tracks_from_predictions(
    predictions: Iterable[Dict[str, Any]],
    *,
    embedding_normalization: str,
) -> Dict[Any, List[Dict[str, Any]]]:
    videos: Dict[Any, List[Dict[str, Any]]] = {}
    for idx, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            raise ExportContractError(f"feature_export_input field 'predictions[{idx}]': must be an object")
        if "video_id" not in pred:
            raise ExportContractError(f"feature_export_input field 'predictions[{idx}].video_id': required field missing")
        if "track_id" not in pred:
            raise ExportContractError(
                f"feature_export_input field 'predictions[{idx}].track_id': missing explicit track_id from runner output"
            )
        if "embedding" not in pred:
            raise ExportContractError(
                f"feature_export_input field 'predictions[{idx}].embedding': missing per-track embedding from runner output"
            )

        video_id = pred["video_id"]
        track: Dict[str, Any] = {
            "track_id": pred["track_id"],
            "embedding": _as_list(pred["embedding"]),
            "embedding_normalization": str(pred.get("embedding_normalization", embedding_normalization)),
        }
        if "start_frame_idx" in pred:
            track["start_frame_idx"] = _optional_int(pred["start_frame_idx"], f"predictions[{idx}].start_frame_idx")
        if "end_frame_idx" in pred:
            track["end_frame_idx"] = _optional_int(pred["end_frame_idx"], f"predictions[{idx}].end_frame_idx")
        if "num_active_frames" in pred:
            track["num_active_frames"] = _optional_int(pred["num_active_frames"], f"predictions[{idx}].num_active_frames")
        if "score" in pred:
            track["objectness_score"] = float(pred["score"])

        videos.setdefault(video_id, []).append(track)
    return videos


def export_feature_enablement_from_real_run(
    *,
    run_root: Path,
    repo_root: Path,
    split: str,
    pseudo_tube_manifest_path: str,
    d2_cfg_ref: str,
    d2_opts: Sequence[str],
    embedding_normalization: str = "none",
    overwrite: bool = False,
    emit_video_index: bool = True,
) -> Path:
    inference_dir = run_root / "d2" / "inference"
    pred_path = inference_dir / "instances_predictions.pth"
    if not pred_path.exists():
        raise ExportContractError(f"feature_export_input field '{pred_path}': file not found")

    try:
        import torch  # local import so non-runner environments can still import module
    except Exception as exc:  # pragma: no cover
        raise ExportContractError("feature_export_input field 'torch': required to load instances_predictions.pth") from exc

    predictions = torch.load(pred_path, map_location="cpu")
    if not isinstance(predictions, list):
        raise ExportContractError("feature_export_input field 'instances_predictions.pth': top-level value must be a list")
    if not predictions:
        raise ExportContractError("feature_export_input field 'instances_predictions.pth': predictions list is empty")

    grouped_tracks = _build_tracks_from_predictions(
        predictions,
        embedding_normalization=embedding_normalization,
    )

    checkpoint_ref = (run_root / "d2" / "last_checkpoint").read_text(encoding="utf-8").strip()
    if not checkpoint_ref:
        raise ExportContractError("feature_export_input field 'stageb_checkpoint_ref': empty last_checkpoint")
    checkpoint_path = run_root / "d2" / checkpoint_ref
    if not checkpoint_path.exists():
        raise ExportContractError(f"feature_export_input field 'stageb_checkpoint_ref': checkpoint not found at {checkpoint_path}")

    config_path = run_root / "config.json"
    pseudo_path = _resolve_path(repo_root, pseudo_tube_manifest_path)

    first_video_id = next(iter(grouped_tracks))
    first_embedding = grouped_tracks[first_video_id][0]["embedding"]
    if not isinstance(first_embedding, list) or not first_embedding:
        raise ExportContractError("feature_export_input field 'embedding_dim': first embedding must be a non-empty list")

    payload: Dict[str, Any] = {
        "run_id": run_root.name,
        "split": split,
        "embedding_dim": len(first_embedding),
        "embedding_normalization": embedding_normalization,
        "stageb_checkpoint_ref": f"d2/{checkpoint_ref}",
        "stageb_checkpoint_hash": _sha256_file(checkpoint_path),
        "stageb_config_ref": "config.json",
        "stageb_config_hash": _sha256_file(config_path),
        "pseudo_tube_manifest_ref": str(pseudo_tube_manifest_path),
        "pseudo_tube_manifest_hash": _sha256_file(pseudo_path),
        "extraction_settings": {
            "frame_sampling_rule": "seqformer_runtime_default",
            "pooling_rule": "track_feature_vector_direct",
            "min_track_length": 1,
            "d2_cfg_ref": d2_cfg_ref,
            "d2_opts": list(d2_opts),
        },
        "videos": [],
    }

    for video_id, tracks in grouped_tracks.items():
        payload["videos"].append(
            {
                "video_id": video_id,
                "runtime_evidence": {
                    "stageb_completion_marker": "completed",
                    "evidence_source": "d2/inference/instances_predictions.pth",
                    "evidence_confidence": "explicit",
                },
                "tracks": tracks,
                "source_artifacts": {
                    "instances_predictions_path": "d2/inference/instances_predictions.pth",
                    "results_json_path": "d2/inference/results.json",
                },
            }
        )

    return build_feature_export_enablement_v1(
        input_payload=payload,
        run_root=run_root,
        overwrite=overwrite,
        emit_video_index=emit_video_index,
    )
