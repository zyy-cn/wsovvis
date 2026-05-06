#!/usr/bin/env python3
"""Build an offline patch-token sampled carrier cache for A8 train-side augmentation.

This tool performs the expensive frame_bank patch-token sampling once, outside
of the training inner loop.  The training script can then use
--train_carrier_aug_mode patchsample_cached_mixed and read one sampled carrier
by epoch/trajectory id from a memory-mapped cache.

It preserves the A8 protocol boundary:
  * no DINOv2 encoder execution;
  * no row-level GT target labels for training;
  * no inference protocol change;
  * no trajectory random batching or cross-clip Hungarian;
  * cache generation is an offline feature-materialization step only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _bootstrap_repo_root_for_direct_cli() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_BOOTSTRAP_REPO_ROOT = _bootstrap_repo_root_for_direct_cli()


def _load_train_module(repo_root: Path):
    train_path = repo_root / "tools" / "a8_joint_prealign_train_time_dynamic_hungarian.py"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    spec = importlib.util.spec_from_file_location("a8_joint_prealign_train_time_dynamic_hungarian_for_cache", str(train_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {train_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _normalize_np_vec(vec: np.ndarray, eps: float = 1.0e-12) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= eps or not np.isfinite(norm):
        return None
    return (arr / norm).astype(np.float32)


def _sample_proto_from_frame_candidates(
    frame_candidates: Sequence[Tuple[np.ndarray, np.ndarray, float, int]],
    *,
    tokens_per_view: int,
    seed: int,
) -> Optional[np.ndarray]:
    if not frame_candidates:
        return None
    rng = np.random.default_rng(int(seed))
    frame_mass = np.asarray([x[2] for x in frame_candidates], dtype=np.float64)
    total = float(np.sum(frame_mass))
    if total > 1.0e-12 and np.isfinite(total):
        frame_prob = frame_mass / total
    else:
        frame_prob = np.full((len(frame_candidates),), 1.0 / float(len(frame_candidates)), dtype=np.float64)
    frame_draws = rng.choice(len(frame_candidates), size=int(tokens_per_view), replace=True, p=frame_prob)
    pieces: List[np.ndarray] = []
    for fidx in np.unique(frame_draws):
        count = int(np.sum(frame_draws == fidx))
        tokens, weights, _denom, _frame_index = frame_candidates[int(fidx)]
        if int(tokens.shape[0]) <= 0 or count <= 0:
            continue
        chosen = rng.choice(int(tokens.shape[0]), size=count, replace=int(tokens.shape[0]) < count, p=weights)
        pieces.append(np.asarray(tokens[chosen], dtype=np.float32))
    if not pieces:
        return None
    sampled = np.concatenate(pieces, axis=0).astype(np.float32)
    return _normalize_np_vec(np.mean(sampled, axis=0))


def _read_token_matrix_cached(
    train_mod: Any,
    *,
    frame_bank_dir: Path,
    feat_path: str,
    grid_h: int,
    grid_w: int,
    payload_cache: Dict[Path, np.lib.npyio.NpzFile],
    token_matrix_cache: Dict[Tuple[str, int, int], Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    key = (str(feat_path), int(grid_h), int(grid_w))
    if key in token_matrix_cache:
        return token_matrix_cache[key]
    feature = train_mod._read_feature_vector_cached(frame_bank_dir, str(feat_path), payload_cache)
    token_matrix = train_mod._coerce_token_feature_matrix(feature, int(grid_h), int(grid_w))
    if token_matrix is None:
        token_matrix_cache[key] = None
        return None
    token_matrix_cache[key] = np.asarray(token_matrix, dtype=np.float32)
    return token_matrix_cache[key]


def _build_frame_candidates_for_row(
    train_mod: Any,
    *,
    row: Mapping[str, Any],
    frame_bank_dir: Path,
    frame_map: Mapping[Tuple[str, int], Mapping[str, Any]],
    geom_map: Mapping[Tuple[str, int], Mapping[str, Any]],
    payload_cache: Dict[Path, np.lib.npyio.NpzFile],
    token_matrix_cache: Dict[Tuple[str, int, int], Optional[np.ndarray]],
    min_token_weight: float,
    max_frame_candidate_tokens: int,
    seed: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, int]], Dict[str, Any]]:
    traj = row.get("trajectory_record") if isinstance(row.get("trajectory_record"), Mapping) else {}
    trajectory_id = str(row.get("trajectory_id", traj.get("trajectory_id", "")))
    clip_id = str(row.get("clip_id", traj.get("clip_id", row.get("video_id", traj.get("video_id", "")))))
    frame_indices = [int(x) for x in list(traj.get("frame_indices", []))]
    masks_rle = list(traj.get("masks_rle", []))
    image_size = list(traj.get("image_size", []))
    if len(image_size) != 2:
        for frame_index in frame_indices:
            geom = geom_map.get((clip_id, int(frame_index)))
            if geom is not None:
                image_size = [int(geom["orig_h"]), int(geom["orig_w"])]
                break
    if not trajectory_id or not frame_indices or len(frame_indices) != len(masks_rle) or len(image_size) != 2:
        return [], {"status": "SKIP", "reason": "malformed_trajectory_record"}

    rng = np.random.default_rng(int(seed))
    frame_candidates: List[Tuple[np.ndarray, np.ndarray, float, int]] = []
    counters: Counter = Counter()
    candidate_counts: List[int] = []
    for frame_index, mask_item in zip(frame_indices, masks_rle):
        key = (clip_id, int(frame_index))
        frame_record = frame_map.get(key)
        geom_record = geom_map.get(key)
        if frame_record is None:
            counters["missing_frame_record"] += 1
            continue
        if geom_record is None:
            counters["missing_frame_geom_record"] += 1
            continue
        try:
            grid_h = int(geom_record["grid_h"])
            grid_w = int(geom_record["grid_w"])
            patch_size = int(geom_record["patch_size"])
            token_matrix = _read_token_matrix_cached(
                train_mod,
                frame_bank_dir=frame_bank_dir,
                feat_path=str(frame_record["feat_path"]),
                grid_h=grid_h,
                grid_w=grid_w,
                payload_cache=payload_cache,
                token_matrix_cache=token_matrix_cache,
            )
            if token_matrix is None:
                counters["coerce_token_matrix_failed"] += 1
                continue
            valid_mask = train_mod.reconstruct_valid_token_mask_from_geometry(geom_record).astype(np.float32)
            decoded_mask = train_mod._decode_mask_rle(mask_item, image_size)
            projected_mask = train_mod._resize_pad_mask(
                decoded_mask,
                resized_h=int(geom_record["resized_h"]),
                resized_w=int(geom_record["resized_w"]),
                padded_h=int(geom_record["padded_h"]),
                padded_w=int(geom_record["padded_w"]),
            )
            weights = train_mod._mask_to_token_weights(projected_mask, patch_size, grid_h, grid_w) * valid_mask
            flat = weights.reshape(-1).astype(np.float64)
            valid_idx = np.flatnonzero(flat > float(min_token_weight)) if float(min_token_weight) > 0 else np.flatnonzero(flat > 0)
            if int(valid_idx.size) <= 0:
                counters["empty_token_occupancy"] += 1
                continue
            cand_weights = flat[valid_idx].astype(np.float64)
            denom = float(np.sum(cand_weights))
            if denom <= 1.0e-12 or not np.isfinite(denom):
                counters["bad_token_weight_denom"] += 1
                continue
            cand_weights = cand_weights / denom
            if int(max_frame_candidate_tokens or 0) > 0 and int(valid_idx.size) > int(max_frame_candidate_tokens):
                cap = int(max_frame_candidate_tokens)
                chosen_pos = rng.choice(int(valid_idx.size), size=cap, replace=False, p=cand_weights)
                valid_idx = valid_idx[chosen_pos]
                cand_weights = cand_weights[chosen_pos]
                cand_weights = cand_weights / max(1.0e-12, float(np.sum(cand_weights)))
            cand_tokens = np.asarray(token_matrix[valid_idx], dtype=np.float32)
            candidate_counts.append(int(cand_tokens.shape[0]))
            frame_candidates.append((cand_tokens, cand_weights.astype(np.float64), float(denom), int(frame_index)))
        except Exception:
            counters["frame_candidate_failed"] += 1
            continue
    if not frame_candidates:
        return [], {"status": "SKIP", "reason": "no_valid_frames", "counters": dict(counters)}
    return frame_candidates, {
        "status": "PASS",
        "valid_frame_count": int(len(frame_candidates)),
        "token_candidate_count_mean_per_frame": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "token_candidate_count_sum": int(np.sum(candidate_counts)) if candidate_counts else 0,
        "counters": dict(counters),
    }


def _cache_seed(global_seed: int, epoch: int, clip_id: int, trajectory_id: str) -> int:
    payload = f"{int(global_seed)}|{int(epoch)}|{int(clip_id)}|{trajectory_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def build_cache(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "patchsample_cache_build_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    random.seed(int(args.seed)); np.random.seed(int(args.seed))
    train_mod = _load_train_module(repo_root)

    prep_args = argparse.Namespace(
        repo_root=str(repo_root),
        asset_root=str(asset_root),
        run_root=str(run_root),
        output_dir=str(repo_root),
        dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(args.trajectory_source_branch),
        annotation_json=str(args.annotation_json) if str(args.annotation_json).strip() else str(train_mod._annotation_default(repo_root, str(args.dataset_name))),
        split_json=str(args.split_json) if str(args.split_json).strip() else str(repo_root / "package" / "reference" / "lvvis_official_base_novel_split.json"),
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
        subset_fraction=args.subset_fraction,
        seed=int(args.seed),
    )
    data = train_mod._prepare_data(prep_args)
    examples = list(data.examples)
    if int(args.max_rows) > 0:
        examples = examples[: int(args.max_rows)]
    if int(args.max_train_clips) > 0:
        selected = set(sorted({int(x["clip_id"]) for x in examples})[: int(args.max_train_clips)])
        examples = [x for x in examples if int(x["clip_id"]) in selected]
    if not examples:
        raise RuntimeError("no examples selected for patchsample cache")

    frame_bank_dir = asset_root / "frame_bank" / str(args.dataset_name)
    frame_map, geom_map = train_mod._load_frame_maps(frame_bank_dir)
    examples_by_clip: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for ex in examples:
        examples_by_clip[int(ex["clip_id"])].append(ex)
    clip_ids = sorted(examples_by_clip.keys())
    trajectory_ids = [str(ex.get("trajectory_id", "")) for cid in clip_ids for ex in examples_by_clip[cid]]
    dim = int(args.dim)
    epochs = int(args.epochs)
    n = int(len(trajectory_ids))

    vectors_path = output_root / "patchsample_vectors.fp16.mmap"
    valid_path = output_root / "patchsample_valid.npy"
    if vectors_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"{vectors_path} exists; pass --overwrite")
    if valid_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"{valid_path} exists; pass --overwrite")
    vectors = np.memmap(vectors_path, mode="w+", dtype=np.float16, shape=(epochs, n, dim))
    vectors[:] = np.float16(0.0)
    valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.bool_, shape=(epochs, n))
    valid[:] = False

    manifest = {
        "status": "RUNNING",
        "timestamp": _now(),
        "definition": "offline patch-token sampled carrier cache for A8 train-side augmentation",
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": str(args.trajectory_source_branch),
        "output_root": str(output_root),
        "shape": [int(epochs), int(n), int(dim)],
        "epochs": int(epochs),
        "trajectory_count": int(n),
        "dim": int(dim),
        "tokens_per_view": int(args.tokens_per_view),
        "seed": int(args.seed),
        "min_token_weight": float(args.min_token_weight),
        "max_frame_candidate_tokens": int(args.max_frame_candidate_tokens),
        "frame_bank_dir": str(frame_bank_dir),
        "runs_dinov2_encoder": False,
        "changes_inference_protocol": False,
        "preserves_clip_wise_training_protocol": True,
        "cache_dtype": "float16",
        "valid_dtype": "bool",
    }
    _write_json(output_root / "patchsample_manifest.json", manifest)
    _write_json(output_root / "patchsample_index.json", {"trajectory_ids": trajectory_ids, "trajectory_count": int(n)})

    counters: Counter = Counter()
    token_candidate_means: List[float] = []
    tid_offset = 0
    clip_iter: Iterable[int] = clip_ids
    if bool(args.show_progress) and tqdm is not None:
        clip_iter = tqdm(clip_ids, desc=f"patchsample-cache {args.dataset_name} {args.trajectory_source_branch}", dynamic_ncols=True)
    for clip_i, cid in enumerate(clip_iter):
        payload_cache: Dict[Path, np.lib.npyio.NpzFile] = {}
        token_matrix_cache: Dict[Tuple[str, int, int], Optional[np.ndarray]] = {}
        clip_examples = list(examples_by_clip[int(cid)])
        try:
            for local_j, row in enumerate(clip_examples):
                tidx = tid_offset + local_j
                tid = str(row.get("trajectory_id", ""))
                seed0 = _cache_seed(int(args.seed), 0, int(cid), tid)
                frame_candidates, st = _build_frame_candidates_for_row(
                    train_mod,
                    row=row,
                    frame_bank_dir=frame_bank_dir,
                    frame_map=frame_map,
                    geom_map=geom_map,
                    payload_cache=payload_cache,
                    token_matrix_cache=token_matrix_cache,
                    min_token_weight=float(args.min_token_weight),
                    max_frame_candidate_tokens=int(args.max_frame_candidate_tokens),
                    seed=seed0,
                )
                if not frame_candidates:
                    counters["rows_no_valid_frame_candidates"] += 1
                    reason = str(st.get("reason", "unknown")) if isinstance(st, Mapping) else "unknown"
                    counters[f"skip_{reason}"] += 1
                    continue
                counters["rows_with_valid_frame_candidates"] += 1
                if isinstance(st, Mapping) and st.get("token_candidate_count_mean_per_frame") is not None:
                    token_candidate_means.append(float(st.get("token_candidate_count_mean_per_frame", 0.0)))
                for epoch in range(1, epochs + 1):
                    s = _cache_seed(int(args.seed), int(epoch), int(cid), tid)
                    proto = _sample_proto_from_frame_candidates(frame_candidates, tokens_per_view=int(args.tokens_per_view), seed=s)
                    if proto is None:
                        counters["sample_failed"] += 1
                        continue
                    vectors[int(epoch) - 1, tidx, :] = np.asarray(proto, dtype=np.float16)
                    valid[int(epoch) - 1, tidx] = True
        finally:
            for payload in payload_cache.values():
                try:
                    payload.close()
                except Exception:
                    pass
        tid_offset += len(clip_examples)
        if int(args.log_every_clips) > 0 and ((clip_i + 1) % int(args.log_every_clips) == 0 or (clip_i + 1) == len(clip_ids)):
            vectors.flush(); valid.flush()
            row = {
                "timestamp": _now(),
                "row_type": "progress",
                "clip_index": int(clip_i + 1),
                "clip_count": int(len(clip_ids)),
                "trajectory_offset": int(tid_offset),
                "trajectory_count": int(n),
                "valid_vectors_so_far": int(np.sum(valid[:, :tid_offset])),
                "counters": dict(counters),
            }
            _append_jsonl(log_path, row)
            if bool(args.print_progress):
                print(json.dumps(row, ensure_ascii=False), flush=True)

    vectors.flush(); valid.flush()
    valid_count = int(np.sum(valid))
    final = dict(manifest)
    final.update({
        "status": "PASS",
        "timestamp": _now(),
        "vectors_path": str(vectors_path),
        "valid_path": str(valid_path),
        "index_path": str(output_root / "patchsample_index.json"),
        "valid_vector_count": int(valid_count),
        "valid_ratio": float(valid_count / max(1, epochs * n)),
        "row_valid_candidate_count": int(counters.get("rows_with_valid_frame_candidates", 0)),
        "row_invalid_candidate_count": int(counters.get("rows_no_valid_frame_candidates", 0)),
        "token_candidate_count_mean_per_frame": float(np.mean(token_candidate_means)) if token_candidate_means else 0.0,
        "counters": dict(counters),
        "policy": {
            "runs_dinov2_encoder": False,
            "uses_frame_bank_patch_token_cache": True,
            "changes_inference_protocol": False,
            "preserves_clip_wise_training_protocol": True,
            "cache_generation_changes_training_labels": False,
        },
    })
    _write_json(output_root / "patchsample_manifest.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline patchsample carrier cache for A8 train-side augmentation")
    p.add_argument("--run_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--trajectory_source_branch", choices=["gt_upper_bound", "mainline"], default="gt_upper_bound")
    p.add_argument("--output_root", required=True)
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", default="")
    p.add_argument("--split_json", default="")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--tokens_per_view", type=int, default=64)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--min_token_weight", type=float, default=0.0)
    p.add_argument("--max_frame_candidate_tokens", type=int, default=4096)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--max_train_clips", type=int, default=0)
    p.add_argument("--log_every_clips", type=int, default=25)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--show_progress", action="store_true", default=True)
    p.add_argument("--print_progress", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    build_cache(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
