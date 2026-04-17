from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_vector_from_locator
from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import read_feature_vector, reconstruct_valid_token_mask_from_geometry

Record = Dict[str, Any]


def _safe_clip_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(text))


def _load_jsonl(path: Path) -> List[Record]:
    rows: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _coerce_token_feature_matrix(feature: np.ndarray, grid_h: int, grid_w: int) -> Optional[np.ndarray]:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 2:
        return None
    grid_tokens = int(grid_h) * int(grid_w)
    if int(feature.shape[0]) == grid_tokens:
        return feature
    if int(feature.shape[0]) == grid_tokens + 1:
        return feature[1:]
    return None


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        raise ValueError("zero-norm pooled frame vector")
    return (arr / norm).astype(np.float32)


def _trajectory_records_path(output_root: Path, dataset_name: str, trajectory_source_branch: str) -> Path:
    if trajectory_source_branch == "mainline":
        return output_root / "exports" / dataset_name / "trajectory_records.jsonl"
    if trajectory_source_branch == "gt_upper_bound":
        return output_root / "exports_gt" / dataset_name / "trajectory_records.jsonl"
    raise ValueError(f"unsupported trajectory_source_branch: {trajectory_source_branch}")


def _frame_records_path(output_root: Path, dataset_name: str) -> Path:
    return output_root / "frame_bank" / dataset_name / "frame_records.jsonl"


def _frame_geom_records_path(output_root: Path, dataset_name: str) -> Path:
    return output_root / "frame_bank" / dataset_name / "frame_geom_records.jsonl"


def _pooled_dir(output_root: Path, dataset_name: str) -> Path:
    return output_root / "frame_bank" / dataset_name


def pooled_frame_records_path(output_root: Path, dataset_name: str) -> Path:
    return _pooled_dir(output_root, dataset_name) / "pooled_frame_records.jsonl"


def pooled_frame_payload_rel(clip_id: str) -> str:
    return f"payload/clip_{_safe_clip_id(clip_id)}_pooled.npz"


def read_pooled_frame_vector(artifact_parent_dir: Path, locator: str) -> np.ndarray:
    return np.asarray(read_vector_from_locator(artifact_parent_dir, locator), dtype=np.float32)


def build_pooled_frame_bank(
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str = "mainline",
    smoke: bool = False,
    smoke_max_trajectories: int = 128,
) -> Dict[str, Any]:
    traj_path = _trajectory_records_path(output_root, dataset_name, trajectory_source_branch)
    frame_path = _frame_records_path(output_root, dataset_name)
    geom_path = _frame_geom_records_path(output_root, dataset_name)
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)
    if not frame_path.is_file():
        raise FileNotFoundError(frame_path)
    if not geom_path.is_file():
        raise FileNotFoundError(geom_path)

    traj_rows = _load_jsonl(traj_path)
    if smoke:
        traj_rows = sorted(traj_rows, key=lambda r: str(r.get("trajectory_id", "")))[: int(smoke_max_trajectories)]

    frame_rows = _load_jsonl(frame_path)
    geom_rows = _load_jsonl(geom_path)
    frame_by_key = {(str(r["clip_id"]), int(r["frame_index"])): r for r in frame_rows}
    geom_by_key = {(str(r["clip_id"]), int(r["frame_index"])): r for r in geom_rows}

    by_clip_vectors: Dict[str, List[np.ndarray]] = {}
    output_rows: List[Record] = []
    missing_hist: Dict[str, int] = {}

    def bump(reason: str) -> None:
        missing_hist[reason] = int(missing_hist.get(reason, 0)) + 1

    frame_parent = _pooled_dir(output_root, dataset_name)
    for traj in sorted(traj_rows, key=lambda r: str(r.get("trajectory_id", ""))):
        clip_id = str(traj.get("clip_id", ""))
        traj_id = str(traj.get("trajectory_id", ""))
        frame_indices = [int(x) for x in list(traj.get("frame_indices", []))]
        pooled_frames: List[np.ndarray] = []
        for frame_index in frame_indices:
            key = (clip_id, int(frame_index))
            fr = frame_by_key.get(key)
            gm = geom_by_key.get(key)
            if fr is None or gm is None:
                bump("missing_frame_row_for_pooled_asset")
                continue
            feat_path = str(fr.get("feat_path", ""))
            if not feat_path:
                bump("missing_feat_path_for_pooled_asset")
                continue
            feature = read_feature_vector(frame_parent, feat_path)
            token_matrix = _coerce_token_feature_matrix(feature, int(gm["grid_h"]), int(gm["grid_w"]))
            if token_matrix is None:
                bump("frame_token_matrix_shape_mismatch")
                continue
            valid_mask = reconstruct_valid_token_mask_from_geometry(gm).astype(np.float32).reshape(-1)
            denom = float(np.sum(valid_mask))
            if denom <= 1e-12:
                bump("empty_frame_valid_token_mask")
                continue
            frame_vec = np.sum(token_matrix * valid_mask[:, None], axis=0).astype(np.float32) / denom
            pooled_frames.append(frame_vec)
        if not pooled_frames:
            bump("trajectory_has_no_valid_pooled_frames")
            continue
        traj_pooled = _normalize(np.mean(np.stack(pooled_frames, axis=0).astype(np.float32), axis=0))
        clip_vectors = by_clip_vectors.setdefault(clip_id, [])
        slot = len(clip_vectors)
        clip_vectors.append(np.asarray(traj_pooled, dtype=np.float32))
        output_rows.append(
            {
                "trajectory_id": traj_id,
                "clip_id": clip_id,
                "trajectory_source_branch": str(trajectory_source_branch),
                "frame_count": len(pooled_frames),
                "frame_pooled_path": f"{pooled_frame_payload_rel(clip_id)}#frame_pooled[{slot}]",
                "path_base_mode": "artifact_parent_dir",
            }
        )

    for clip_id, vectors in by_clip_vectors.items():
        payload_path = _pooled_dir(output_root, dataset_name) / pooled_frame_payload_rel(clip_id)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack(vectors, axis=0).astype(np.float16) if vectors else np.zeros((0, 0), dtype=np.float16)
        np.savez_compressed(payload_path, frame_pooled=arr)

    records_path = pooled_frame_records_path(output_root, dataset_name)
    _write_jsonl(records_path, output_rows)
    return {
        "dataset_name": dataset_name,
        "trajectory_source_branch": trajectory_source_branch,
        "record_count_input": len(traj_rows),
        "record_count_output": len(output_rows),
        "coverage_ratio": float(len(output_rows)) / float(len(traj_rows)) if traj_rows else 0.0,
        "pooled_frame_records_path": str(records_path),
        "missing_reason_histogram": missing_hist,
    }
