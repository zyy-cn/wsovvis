from __future__ import annotations

import json
import multiprocessing as mp
import re
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import (
    parse_feat_path,
    reconstruct_valid_token_mask_from_geometry,
)

try:
    from tqdm.auto import tqdm as _tqdm_cls
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    _tqdm_cls = None


class _LineProgress:
    def __init__(
        self,
        iterable: Optional[Iterable[Any]] = None,
        total: Optional[int] = None,
        desc: str = "",
        unit: str = "it",
        dynamic_ncols: bool = True,
        leave: bool = True,
        file: Any = None,
        mininterval: float = 0.2,
    ) -> None:
        self._iterable = iterable
        self.total = int(total) if total is not None else (len(iterable) if iterable is not None else 0)
        self.desc = desc
        self.unit = unit
        self.leave = leave
        self.file = file or sys.stderr
        self.mininterval = float(mininterval)
        self.start = time.perf_counter()
        self.count = 0
        self._postfix: Dict[str, Any] = {}

    def __enter__(self) -> "_LineProgress":
        self._emit(force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self):
        if self._iterable is None:
            return iter(())
        for item in self._iterable:
            yield item
            self.update(1)

    def update(self, n: int = 1) -> None:
        self.count += int(n)
        self._emit()

    def set_postfix(self, values: Optional[Dict[str, Any]] = None, refresh: bool = True) -> None:
        if values is not None:
            self._postfix = dict(values)
        if refresh:
            self._emit()

    def close(self) -> None:
        self._emit(force=True)

    def _emit(self, force: bool = False) -> None:
        elapsed = max(1e-9, time.perf_counter() - self.start)
        rate = float(self.count) / elapsed
        remaining = max(0, int(self.total) - int(self.count)) if self.total else 0
        eta = (remaining / rate) if rate > 1e-9 and self.total else 0.0
        width = 24
        ratio = float(self.count) / float(self.total) if self.total else 0.0
        filled = max(0, min(width, int(round(width * ratio))))
        bar = "[" + ("=" * filled) + (">" if filled < width else "=") + ("." * max(0, width - filled - 1)) + "]"
        postfix = ""
        if self._postfix:
            postfix = " " + " ".join(f"{k}={v}" for k, v in self._postfix.items())
        msg = (
            f"{self.desc} {bar} {self.count}/{self.total} "
            f"[{rate:.2f} {self.unit}/s, eta {eta:.1f}s]{postfix}"
        )
        print(msg, file=self.file, flush=True)


def _make_progress_bar(**kwargs: Any):
    if _tqdm_cls is not None and sys.stderr.isatty():
        return _tqdm_cls(**kwargs)
    return _LineProgress(**kwargs)


Record = Dict[str, Any]


@dataclass(frozen=True)
class CarrierBuildConfig:
    dataset_name: str
    output_root: Path
    trajectory_source_branch: str = "mainline"
    smoke: bool = False
    smoke_max_trajectories: int = 64
    num_workers: int = 1
    device: str = "cpu"


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


_WORKER_FRAME_BY_CLIP: Dict[str, Dict[int, Tuple[Record, Optional[Record]]]] = {}


def _worker_init(frame_by_clip: Dict[str, Dict[int, Tuple[Record, Optional[Record]]]]) -> None:
    global _WORKER_FRAME_BY_CLIP
    _WORKER_FRAME_BY_CLIP = frame_by_clip


def _merge_reason_stats(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = int(dst.get(key, 0)) + int(value)


def _read_feature_vector_cached(
    artifact_parent_dir: Path,
    feat_path: str,
    payload_cache: Dict[Path, np.lib.npyio.NpzFile],
) -> np.ndarray:
    rel_path, slot = parse_feat_path(feat_path)
    payload_path = artifact_parent_dir / rel_path
    payload = payload_cache.get(payload_path)
    if payload is None:
        payload = np.load(payload_path, allow_pickle=True)
        payload_cache[payload_path] = payload
    slot_key = f"slot_{slot}"
    if slot_key in payload.files:
        return np.asarray(payload[slot_key], dtype=np.float32)
    if "feats" in payload.files:
        feats = np.asarray(payload["feats"])
        if feats.ndim == 0:
            raise ValueError("invalid feature payload")
        if feats.dtype == object:
            return np.asarray(feats[slot], dtype=np.float32)
        if slot < 0 or slot >= int(feats.shape[0]):
            raise IndexError("feat_path slot out of range")
        return np.asarray(feats[slot], dtype=np.float32)
    raise KeyError(slot_key)


def _process_clip_shard(
    task: Tuple[int, str, List[Tuple[int, Record]], str, str],
) -> Dict[str, Any]:
    task_index, clip_id, trajectory_items, frame_artifact_parent_dir_str, shard_root_str = task
    frame_artifact_parent_dir = Path(frame_artifact_parent_dir_str)
    shard_root = Path(shard_root_str)
    clip_lookup = _WORKER_FRAME_BY_CLIP.get(str(clip_id), {})
    invalid_reason_stats: Dict[str, int] = {}
    traj_rows_raw: List[np.ndarray] = []
    traj_rows_norm: List[np.ndarray] = []
    frame_rows_norm: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []
    payload_cache: Dict[Path, np.lib.npyio.NpzFile] = {}
    frame_static_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int, int, int, int, int]] = {}
    t0 = time.perf_counter()

    def bump(reason: str) -> None:
        invalid_reason_stats[reason] = int(invalid_reason_stats.get(reason, 0)) + 1

    try:
        for traj_pos, record in trajectory_items:
            trajectory_id = str(record.get("trajectory_id", "")).strip()
            frame_indices = [int(x) for x in list(record.get("frame_indices", []))]
            masks_rle = list(record.get("masks_rle", []))
            image_size = list(record.get("image_size", []))
            if not trajectory_id or len(frame_indices) == 0 or len(frame_indices) != len(masks_rle):
                bump("malformed_trajectory_record")
                continue

            frame_norm_vectors: List[np.ndarray] = []
            valid_frame_indices: List[int] = []
            for frame_index, mask_item in zip(frame_indices, masks_rle):
                frame_pair = clip_lookup.get(int(frame_index))
                if frame_pair is None:
                    bump("missing_frame_record")
                    continue
                frame_record, geom_record = frame_pair
                if geom_record is None:
                    bump("missing_frame_geom_record")
                    continue
                static = frame_static_cache.get(int(frame_index))
                if static is None:
                    feature = _read_feature_vector_cached(
                        frame_artifact_parent_dir, str(frame_record["feat_path"]), payload_cache
                    )
                    grid_h = int(geom_record["grid_h"])
                    grid_w = int(geom_record["grid_w"])
                    patch_size = int(geom_record["patch_size"])
                    token_matrix = _coerce_token_feature_matrix(feature, grid_h, grid_w)
                    if token_matrix is None:
                        bump("token_shape_mismatch")
                        continue
                    valid_mask = reconstruct_valid_token_mask_from_geometry(geom_record).astype(np.float32)
                    static = (
                        token_matrix,
                        valid_mask,
                        patch_size,
                        grid_h,
                        grid_w,
                        int(geom_record["resized_h"]),
                        int(geom_record["resized_w"]),
                    )
                    frame_static_cache[int(frame_index)] = static

                token_matrix, valid_mask, patch_size, grid_h, grid_w, resized_h, resized_w = static
                decoded_mask = _decode_mask_rle(mask_item, image_size)
                projected_mask = _resize_pad_mask(
                    decoded_mask,
                    resized_h=resized_h,
                    resized_w=resized_w,
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

            traj_local_idx = len(traj_rows_raw)
            frame_local_start = len(frame_rows_norm)
            frame_local_len = len(frame_norm_vectors)
            traj_rows_raw.append(z_raw)
            traj_rows_norm.append(z_norm.astype(np.float32))
            frame_rows_norm.extend([vec.astype(np.float32) for vec in frame_norm_vectors])
            meta_rows.append(
                {
                    "traj_pos": int(traj_pos),
                    "trajectory_id": trajectory_id,
                    "clip_id": str(record.get("clip_id")),
                    "valid_frame_indices": valid_frame_indices,
                    "traj_local_idx": int(traj_local_idx),
                    "frame_local_start": int(frame_local_start),
                    "frame_local_len": int(frame_local_len),
                }
            )
    finally:
        for payload in payload_cache.values():
            try:
                payload.close()
            except Exception:
                pass

    shard_prefix = f"shard_{task_index:06d}_{clip_id}"
    traj_shard = shard_root / f"{shard_prefix}_traj.npz"
    frame_shard = shard_root / f"{shard_prefix}_frame.npz"
    meta_shard = shard_root / f"{shard_prefix}_meta.jsonl"
    stat_shard = shard_root / f"{shard_prefix}_stats.json"

    if traj_rows_raw:
        np.savez_compressed(
            traj_shard,
            z_raw=np.stack(traj_rows_raw, axis=0).astype(np.float32),
            z_norm=np.stack(traj_rows_norm, axis=0).astype(np.float32),
        )
    else:
        np.savez_compressed(
            traj_shard,
            z_raw=np.zeros((0, 0), dtype=np.float32),
            z_norm=np.zeros((0, 0), dtype=np.float32),
        )
    if frame_rows_norm:
        np.savez_compressed(
            frame_shard,
            z_norm=np.stack(frame_rows_norm, axis=0).astype(np.float32),
        )
    else:
        np.savez_compressed(
            frame_shard,
            z_norm=np.zeros((0, 0), dtype=np.float32),
        )

    with meta_shard.open("w", encoding="utf-8") as handle:
        for row in meta_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stat_shard.write_text(
        json.dumps(
            {
                "invalid_reason_stats": invalid_reason_stats,
                "valid_trajectory_count": len(meta_rows),
                "valid_frame_vector_count": len(frame_rows_norm),
                "elapsed_sec": float(time.perf_counter() - t0),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "task_index": int(task_index),
        "clip_id": str(clip_id),
        "traj_shard": str(traj_shard),
        "frame_shard": str(frame_shard),
        "meta_shard": str(meta_shard),
        "stat_shard": str(stat_shard),
        "elapsed_sec": float(time.perf_counter() - t0),
    }


def build_carrier_bank(config: CarrierBuildConfig) -> Dict[str, Any]:
    repo_root = _repo_root()
    trajectory_path = _trajectory_records_path(repo_root, config.dataset_name, config.trajectory_source_branch)
    frame_records_path = _frame_records_path(repo_root, config.dataset_name)
    frame_geom_records_path = _frame_geom_records_path(repo_root, config.dataset_name)
    geometry_report_path = _geometry_applicability_report_path(repo_root)

    for required in (trajectory_path, frame_records_path, frame_geom_records_path, geometry_report_path):
        if not required.exists():
            raise FileNotFoundError(required)

    total_start = time.perf_counter()
    trajectory_records = _load_jsonl(trajectory_path)
    if config.smoke:
        trajectory_records = trajectory_records[: config.smoke_max_trajectories]
    trajectory_records = sorted(trajectory_records, key=lambda rec: str(rec.get("trajectory_id", "")))

    frame_lookup = _build_frame_lookup(_load_jsonl(frame_records_path))
    geom_lookup = _build_frame_lookup(_load_jsonl(frame_geom_records_path))
    frame_by_clip: Dict[str, Dict[int, Tuple[Record, Optional[Record]]]] = defaultdict(dict)
    for key, frame_record in frame_lookup.items():
        geom_record = geom_lookup.get(key)
        clip_id, frame_index = key
        frame_by_clip[str(clip_id)][int(frame_index)] = (frame_record, geom_record)

    clip_to_trajectories: Dict[str, List[Tuple[int, Record]]] = defaultdict(list)
    for traj_pos, record in enumerate(trajectory_records):
        clip_to_trajectories[str(record.get("clip_id"))].append((int(traj_pos), record))

    artifact_dir = _branch_output_dir(config.output_root, config.dataset_name, config.trajectory_source_branch)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    carrier_records_path = artifact_dir / "carrier_records.jsonl"
    traj_payload_rel = "carrier_vectors_traj.npz"
    frame_payload_rel = "carrier_vectors_frame.npz"
    traj_payload_path = artifact_dir / traj_payload_rel
    frame_payload_path = artifact_dir / frame_payload_rel

    invalid_reason_stats: Dict[str, int] = {}
    worker_count = max(1, int(config.num_workers))
    clip_items = sorted(clip_to_trajectories.items(), key=lambda kv: kv[0])
    tasks: List[Tuple[int, str, List[Tuple[int, Record]], str, str]] = []

    merge_t0 = 0.0
    with tempfile.TemporaryDirectory(prefix="g5_carrier_shards_", dir=str(artifact_dir)) as shard_dir_str:
        for task_index, (clip_id, items) in enumerate(clip_items):
            tasks.append((task_index, clip_id, items, str(frame_records_path.parent), shard_dir_str))
        compute_t0 = time.perf_counter()
        if worker_count <= 1 or len(tasks) <= 1:
            _worker_init(frame_by_clip)
            shard_results = []
            with _make_progress_bar(
                total=len(tasks),
                desc=f"g5-{config.dataset_name}-{config.trajectory_source_branch}",
                unit="clip",
                dynamic_ncols=True,
                leave=True,
                file=sys.stderr,
            ) as progress:
                for task in tasks:
                    shard_results.append(_process_clip_shard(task))
                    progress.update(1)
                    progress.set_postfix({"last_clip": task[1], "workers": 1}, refresh=False)
        else:
            start_methods = set(mp.get_all_start_methods())
            if "fork" in start_methods:
                ctx = mp.get_context("fork")
                with ctx.Pool(
                    processes=min(worker_count, len(tasks)),
                    initializer=_worker_init,
                    initargs=(frame_by_clip,),
                ) as pool:
                    shard_results = []
                    with _make_progress_bar(
                        total=len(tasks),
                        desc=f"g5-{config.dataset_name}-{config.trajectory_source_branch}",
                        unit="clip",
                        dynamic_ncols=True,
                        leave=True,
                        file=sys.stderr,
                    ) as progress:
                        for shard_result in pool.imap(_process_clip_shard, tasks, chunksize=1):
                            shard_results.append(shard_result)
                            progress.update(1)
                            progress.set_postfix(
                                {"last_clip": shard_result["clip_id"], "workers": min(worker_count, len(tasks))},
                                refresh=False,
                            )
            else:
                # Spawn-based fallback keeps behavior correct, without duplicate large state shipping.
                _worker_init(frame_by_clip)
                shard_results = []
                with _make_progress_bar(
                    total=len(tasks),
                    desc=f"g5-{config.dataset_name}-{config.trajectory_source_branch}",
                    unit="clip",
                    dynamic_ncols=True,
                    leave=True,
                    file=sys.stderr,
                ) as progress:
                    for task in tasks:
                        shard_results.append(_process_clip_shard(task))
                        progress.update(1)
                        progress.set_postfix({"last_clip": task[1], "workers": 1}, refresh=False)
                worker_count = 1
        compute_elapsed = float(time.perf_counter() - compute_t0)

        traj_meta_by_pos: Dict[int, Dict[str, Any]] = {}
        shard_cache: Dict[str, np.lib.npyio.NpzFile] = {}
        traj_dim = 0
        frame_dim = 0
        total_frame_rows = 0
        for shard_result in shard_results:
            stat_payload = json.loads(Path(shard_result["stat_shard"]).read_text(encoding="utf-8"))
            _merge_reason_stats(invalid_reason_stats, stat_payload.get("invalid_reason_stats", {}))
            with Path(shard_result["meta_shard"]).open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    traj_pos = int(row["traj_pos"])
                    if traj_pos in traj_meta_by_pos:
                        raise ValueError(f"duplicate trajectory position in shards: {traj_pos}")
                    row["traj_shard"] = str(shard_result["traj_shard"])
                    row["frame_shard"] = str(shard_result["frame_shard"])
                    traj_meta_by_pos[traj_pos] = row
                    total_frame_rows += int(row["frame_local_len"])
                    traj_npz = shard_cache.get(str(shard_result["traj_shard"]))
                    if traj_npz is None:
                        traj_npz = np.load(str(shard_result["traj_shard"]), allow_pickle=False)
                        shard_cache[str(shard_result["traj_shard"])] = traj_npz
                    if int(traj_npz["z_raw"].shape[0]) > 0:
                        traj_dim = int(traj_npz["z_raw"].shape[1])
                    frame_npz = shard_cache.get(str(shard_result["frame_shard"]))
                    if frame_npz is None:
                        frame_npz = np.load(str(shard_result["frame_shard"]), allow_pickle=False)
                        shard_cache[str(shard_result["frame_shard"])] = frame_npz
                    if int(frame_npz["z_norm"].shape[0]) > 0:
                        frame_dim = int(frame_npz["z_norm"].shape[1])

        ordered_positions = sorted(traj_meta_by_pos.keys())
        merge_t0 = time.perf_counter()
        output_records = len(ordered_positions)
        if output_records > 0 and traj_dim <= 0:
            raise ValueError("invalid traj vector dimension during merge")
        if total_frame_rows > 0 and frame_dim <= 0:
            raise ValueError("invalid frame vector dimension during merge")

        with tempfile.TemporaryDirectory(prefix="g5_carrier_merge_", dir=str(artifact_dir)) as merge_tmp:
            merge_dir = Path(merge_tmp)
            if output_records > 0:
                traj_z_raw_mm = np.memmap(
                    merge_dir / "traj_z_raw.fp16.mmap", dtype=np.float16, mode="w+", shape=(output_records, traj_dim)
                )
                traj_z_norm_mm = np.memmap(
                    merge_dir / "traj_z_norm.fp16.mmap", dtype=np.float16, mode="w+", shape=(output_records, traj_dim)
                )
            else:
                traj_z_raw_mm = None
                traj_z_norm_mm = None
            if total_frame_rows > 0:
                frame_z_norm_mm = np.memmap(
                    merge_dir / "frame_z_norm.fp16.mmap",
                    dtype=np.float16,
                    mode="w+",
                    shape=(total_frame_rows, frame_dim),
                )
            else:
                frame_z_norm_mm = None

            frame_global_cursor = 0
            with carrier_records_path.open("w", encoding="utf-8") as record_fh:
                for traj_global_idx, traj_pos in enumerate(ordered_positions):
                    row = traj_meta_by_pos[traj_pos]
                    traj_npz = shard_cache[str(row["traj_shard"])]
                    frame_npz = shard_cache[str(row["frame_shard"])]
                    traj_local_idx = int(row["traj_local_idx"])
                    frame_local_start = int(row["frame_local_start"])
                    frame_local_len = int(row["frame_local_len"])
                    if traj_z_raw_mm is not None:
                        traj_z_raw_mm[traj_global_idx] = np.asarray(
                            traj_npz["z_raw"][traj_local_idx], dtype=np.float16
                        )
                        traj_z_norm_mm[traj_global_idx] = np.asarray(
                            traj_npz["z_norm"][traj_local_idx], dtype=np.float16
                        )
                    if frame_local_len > 0 and frame_z_norm_mm is not None:
                        frame_z_norm_mm[frame_global_cursor : frame_global_cursor + frame_local_len] = np.asarray(
                            frame_npz["z_norm"][frame_local_start : frame_local_start + frame_local_len], dtype=np.float16
                        )
                    record = {
                        "trajectory_id": str(row["trajectory_id"]),
                        "clip_id": str(row["clip_id"]),
                        "frame_indices": [int(x) for x in list(row["valid_frame_indices"])],
                        "z_raw_path": f"{traj_payload_rel}#z_raw[{traj_global_idx}]",
                        "z_norm_path": f"{traj_payload_rel}#z_norm[{traj_global_idx}]",
                        "frame_carriers_norm_paths": [
                            f"{frame_payload_rel}#z_norm[{frame_global_cursor + idx}]"
                            for idx in range(frame_local_len)
                        ],
                        "path_base_mode": "artifact_parent_dir",
                    }
                    record_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    frame_global_cursor += frame_local_len

            if traj_z_raw_mm is not None and traj_z_norm_mm is not None:
                traj_z_raw_mm.flush()
                traj_z_norm_mm.flush()
                np.savez_compressed(traj_payload_path, z_raw=traj_z_raw_mm, z_norm=traj_z_norm_mm)
                del traj_z_raw_mm
                del traj_z_norm_mm
            else:
                np.savez_compressed(
                    traj_payload_path,
                    z_raw=np.zeros((0, 0), dtype=np.float16),
                    z_norm=np.zeros((0, 0), dtype=np.float16),
                )
            if frame_z_norm_mm is not None:
                frame_z_norm_mm.flush()
                np.savez_compressed(frame_payload_path, z_norm=frame_z_norm_mm)
                del frame_z_norm_mm
            else:
                np.savez_compressed(
                    frame_payload_path,
                    z_norm=np.zeros((0, 0), dtype=np.float16),
                )

        for payload in shard_cache.values():
            try:
                payload.close()
            except Exception:
                pass
        merge_elapsed = float(time.perf_counter() - merge_t0)

    total_records = len(trajectory_records)
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
        "num_workers_requested": int(config.num_workers),
        "num_workers_used": int(min(max(1, int(config.num_workers)), max(1, len(tasks)) if tasks else 1)),
        "device_requested": str(config.device),
        "device_effective": "cpu",
        "timing_sec": {
            "total": float(time.perf_counter() - total_start),
            "compute_shards": float(compute_elapsed if "compute_elapsed" in locals() else 0.0),
            "merge_write": float(merge_elapsed if merge_t0 > 0 else 0.0),
        },
    }
