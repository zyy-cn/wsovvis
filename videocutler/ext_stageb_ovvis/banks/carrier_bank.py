from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import (
    read_feature_vector,
    reconstruct_valid_token_mask_from_geometry,
)


Record = Dict[str, Any]


@dataclass(frozen=True)
class CarrierBuildConfig:
    dataset_name: str
    output_root: Path
    trajectory_source_branch: str = "mainline"
    smoke: bool = False
    smoke_max_trajectories: int = 64


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "codex" / "control" / "CURRENT_TASK.json").exists():
            return parent
    # Fallback keeps backward compatibility with the known repo layout.
    return current.parents[3]


def _safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text))


def _load_jsonl(path: Path) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _trajectory_records_path(repo_root: Path, dataset_name: str, trajectory_source_branch: str) -> Path:
    if trajectory_source_branch == "mainline":
        return repo_root / "exports" / dataset_name / "trajectory_records.jsonl"
    if trajectory_source_branch == "gt_upper_bound":
        return repo_root / "exports_gt" / dataset_name / "trajectory_records.jsonl"
    raise ValueError(f"unsupported trajectory_source_branch: {trajectory_source_branch}")


def _frame_records_path(repo_root: Path, dataset_name: str) -> Path:
    return repo_root / "frame_bank" / dataset_name / "frame_records.jsonl"


def _frame_geom_records_path(repo_root: Path, dataset_name: str) -> Path:
    return repo_root / "frame_bank" / dataset_name / "frame_geom_records.jsonl"


def _geometry_applicability_report_path(repo_root: Path) -> Path:
    return repo_root / "frame_bank" / "geometry" / "frame_geometry_applicability_report.json"


def _branch_output_dir(output_root: Path, dataset_name: str, trajectory_source_branch: str) -> Path:
    base = "carrier_bank" if trajectory_source_branch == "mainline" else "carrier_bank_gt"
    return output_root / base / dataset_name


def _decode_mask_rle(mask_item: Any, image_size: Sequence[int]) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pycocotools is required for G5 mask-to-token projection") from exc

    if len(image_size) != 2:
        raise ValueError("image_size must contain [H, W]")
    h, w = int(image_size[0]), int(image_size[1])
    if h <= 0 or w <= 0:
        raise ValueError("image_size values must be positive")

    if mask_item is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(mask_item, dict):
        rle = dict(mask_item)
        if "size" not in rle:
            rle["size"] = [h, w]
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, h, w)
        decoded = mask_utils.decode(rle)
        return np.asarray(decoded, dtype=np.uint8)
    if isinstance(mask_item, str):
        decoded = mask_utils.decode({"size": [h, w], "counts": mask_item.encode("utf-8")})
        return np.asarray(decoded, dtype=np.uint8)
    raise ValueError("unsupported mask rle format")


def _resize_pad_mask(mask: np.ndarray, resized_h: int, resized_w: int, padded_h: int, padded_w: int) -> np.ndarray:
    src_h, src_w = int(mask.shape[0]), int(mask.shape[1])
    if src_h <= 0 or src_w <= 0:
        return np.zeros((padded_h, padded_w), dtype=np.float32)
    src_y = (np.arange(resized_h) * float(src_h) / float(resized_h)).astype(np.int64)
    src_x = (np.arange(resized_w) * float(src_w) / float(resized_w)).astype(np.int64)
    src_y = np.clip(src_y, 0, src_h - 1)
    src_x = np.clip(src_x, 0, src_w - 1)
    resized = mask[src_y[:, None], src_x[None, :]].astype(np.float32)
    padded = np.zeros((padded_h, padded_w), dtype=np.float32)
    padded[:resized_h, :resized_w] = resized
    return padded


def _mask_to_token_weights(mask: np.ndarray, patch_size: int, grid_h: int, grid_w: int) -> np.ndarray:
    weights = np.zeros((grid_h, grid_w), dtype=np.float32)
    for row in range(grid_h):
        y0 = row * patch_size
        y1 = min((row + 1) * patch_size, mask.shape[0])
        if y1 <= y0:
            continue
        for col in range(grid_w):
            x0 = col * patch_size
            x1 = min((col + 1) * patch_size, mask.shape[1])
            if x1 <= x0:
                continue
            patch = mask[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            weights[row, col] = float(np.mean(patch))
    return weights


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return None
    return (vec / norm).astype(np.float32)


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


_VECTOR_LOCATOR_RE = re.compile(r"^(?P<path>[A-Za-z0-9_./-]+)#(?P<key>[A-Za-z0-9_]+)\[(?P<idx>[0-9]+)\]$")


def parse_vector_locator(locator: str) -> Tuple[Path, str, int]:
    match = _VECTOR_LOCATOR_RE.match(locator)
    if not match:
        raise ValueError(f"invalid vector locator: {locator}")
    rel_path = Path(match.group("path"))
    key = str(match.group("key"))
    idx = int(match.group("idx"))
    return rel_path, key, idx


def read_vector_from_locator(artifact_parent_dir: Path, locator: str) -> np.ndarray:
    rel_path, key, idx = parse_vector_locator(locator)
    payload_path = artifact_parent_dir / rel_path
    with np.load(payload_path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(f"missing key {key} in {payload_path}")
        arr = np.asarray(payload[key], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"payload key {key} is not 2D in {payload_path}")
        if idx < 0 or idx >= int(arr.shape[0]):
            raise IndexError(f"index out of range for {locator}")
        return np.asarray(arr[idx], dtype=np.float32)


def read_carrier_records(path: Path) -> List[Record]:
    records = _load_jsonl(path)
    return sorted(records, key=lambda rec: str(rec.get("trajectory_id", "")))


def _build_frame_lookup(records: Iterable[Record]) -> Dict[Tuple[str, int], Record]:
    lookup: Dict[Tuple[str, int], Record] = {}
    for record in records:
        key = (str(record["clip_id"]), int(record["frame_index"]))
        lookup[key] = record
    return lookup


def build_carrier_bank(config: CarrierBuildConfig) -> Dict[str, Any]:
    repo_root = _repo_root()
    trajectory_path = _trajectory_records_path(repo_root, config.dataset_name, config.trajectory_source_branch)
    frame_records_path = _frame_records_path(repo_root, config.dataset_name)
    frame_geom_records_path = _frame_geom_records_path(repo_root, config.dataset_name)
    geometry_report_path = _geometry_applicability_report_path(repo_root)

    for required in (trajectory_path, frame_records_path, frame_geom_records_path, geometry_report_path):
        if not required.exists():
            raise FileNotFoundError(required)

    trajectory_records = _load_jsonl(trajectory_path)
    if config.smoke:
        trajectory_records = trajectory_records[: config.smoke_max_trajectories]
    trajectory_records = sorted(trajectory_records, key=lambda rec: str(rec.get("trajectory_id", "")))

    frame_lookup = _build_frame_lookup(_load_jsonl(frame_records_path))
    geom_lookup = _build_frame_lookup(_load_jsonl(frame_geom_records_path))

    artifact_dir = _branch_output_dir(config.output_root, config.dataset_name, config.trajectory_source_branch)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    carrier_records_path = artifact_dir / "carrier_records.jsonl"
    traj_payload_rel = "carrier_vectors_traj.npz"
    frame_payload_rel = "carrier_vectors_frame.npz"
    traj_payload_path = artifact_dir / traj_payload_rel
    frame_payload_path = artifact_dir / frame_payload_rel

    written_records: List[Record] = []
    invalid_reason_stats: Dict[str, int] = {}
    traj_z_raw_rows: List[np.ndarray] = []
    traj_z_norm_rows: List[np.ndarray] = []
    frame_z_norm_rows: List[np.ndarray] = []

    def bump(reason: str) -> None:
        invalid_reason_stats[reason] = int(invalid_reason_stats.get(reason, 0)) + 1

    for record in trajectory_records:
        trajectory_id = str(record.get("trajectory_id", "")).strip()
        clip_id = str(record.get("clip_id"))
        frame_indices = [int(x) for x in list(record.get("frame_indices", []))]
        masks_rle = list(record.get("masks_rle", []))
        image_size = list(record.get("image_size", []))
        if not trajectory_id or len(frame_indices) == 0 or len(frame_indices) != len(masks_rle):
            bump("malformed_trajectory_record")
            continue

        frame_norm_vectors: List[np.ndarray] = []
        valid_frame_indices: List[int] = []
        for frame_index, mask_item in zip(frame_indices, masks_rle):
            key = (clip_id, int(frame_index))
            frame_record = frame_lookup.get(key)
            geom_record = geom_lookup.get(key)
            if frame_record is None:
                bump("missing_frame_record")
                continue
            if geom_record is None:
                bump("missing_frame_geom_record")
                continue

            feature = read_feature_vector(frame_records_path.parent, str(frame_record["feat_path"]))
            grid_h = int(geom_record["grid_h"])
            grid_w = int(geom_record["grid_w"])
            patch_size = int(geom_record["patch_size"])
            token_matrix = _coerce_token_feature_matrix(feature, grid_h, grid_w)
            if token_matrix is None:
                bump("token_shape_mismatch")
                continue

            valid_mask = reconstruct_valid_token_mask_from_geometry(geom_record).astype(np.float32)
            decoded_mask = _decode_mask_rle(mask_item, image_size)
            projected_mask = _resize_pad_mask(
                decoded_mask,
                resized_h=int(geom_record["resized_h"]),
                resized_w=int(geom_record["resized_w"]),
                padded_h=int(geom_record["padded_h"]),
                padded_w=int(geom_record["padded_w"]),
            )
            weights = _mask_to_token_weights(projected_mask, patch_size, grid_h, grid_w)
            weights = weights * valid_mask
            flat_weights = weights.reshape(-1)
            denom = float(np.sum(flat_weights))
            if denom <= 1e-12:
                bump("empty_token_occupancy")
                continue
            flat_weights = flat_weights / denom
            frame_raw = np.sum(token_matrix * flat_weights[:, None], axis=0).astype(np.float32)
            frame_norm = _normalize(frame_raw)
            if frame_norm is None:
                bump("zero_norm_frame_carrier")
                continue
            frame_norm_vectors.append(frame_norm)
            valid_frame_indices.append(int(frame_index))

        if not frame_norm_vectors:
            bump("no_valid_frames")
            continue

        frame_stack = np.stack(frame_norm_vectors, axis=0).astype(np.float32)
        z_raw = np.mean(frame_stack, axis=0).astype(np.float32)
        z_norm = _normalize(z_raw)
        if z_norm is None:
            bump("zero_norm_trajectory_carrier")
            continue

        traj_idx = len(traj_z_raw_rows)
        frame_start = len(frame_z_norm_rows)
        traj_z_raw_rows.append(z_raw.astype(np.float32))
        traj_z_norm_rows.append(z_norm.astype(np.float32))
        frame_z_norm_rows.extend([vec.astype(np.float32) for vec in frame_norm_vectors])

        written_records.append(
            {
                "trajectory_id": trajectory_id,
                "clip_id": clip_id,
                "frame_indices": valid_frame_indices,
                "z_raw_path": f"{traj_payload_rel}#z_raw[{traj_idx}]",
                "z_norm_path": f"{traj_payload_rel}#z_norm[{traj_idx}]",
                "frame_carriers_norm_paths": [
                    f"{frame_payload_rel}#z_norm[{frame_start + idx}]"
                    for idx in range(len(frame_norm_vectors))
                ],
                "path_base_mode": "artifact_parent_dir",
            }
        )

    written_records = sorted(written_records, key=lambda rec: str(rec.get("trajectory_id", "")))
    _write_jsonl(carrier_records_path, written_records)
    if traj_z_raw_rows:
        np.savez_compressed(
            traj_payload_path,
            z_raw=np.stack(traj_z_raw_rows, axis=0).astype(np.float32),
            z_norm=np.stack(traj_z_norm_rows, axis=0).astype(np.float32),
        )
    else:
        np.savez_compressed(
            traj_payload_path,
            z_raw=np.zeros((0, 0), dtype=np.float32),
            z_norm=np.zeros((0, 0), dtype=np.float32),
        )
    if frame_z_norm_rows:
        np.savez_compressed(
            frame_payload_path,
            z_norm=np.stack(frame_z_norm_rows, axis=0).astype(np.float32),
        )
    else:
        np.savez_compressed(
            frame_payload_path,
            z_norm=np.zeros((0, 0), dtype=np.float32),
        )

    total_records = len(trajectory_records)
    output_records = len(written_records)
    coverage_ratio = float(output_records) / float(total_records) if total_records > 0 else 0.0
    return {
        "dataset_name": config.dataset_name,
        "trajectory_source_branch": config.trajectory_source_branch,
        "run_scope": "smoke" if config.smoke else "full",
        "record_count_input": total_records,
        "record_count_output": output_records,
        "coverage_ratio": coverage_ratio,
        "invalid_reason_stats": invalid_reason_stats,
        "carrier_records_path": carrier_records_path,
        "artifact_parent_dir": artifact_dir,
        "traj_payload_path": traj_payload_path,
        "frame_payload_path": frame_payload_path,
    }
