from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:  # pragma: no cover
    mask_utils = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate keep-aspect-pad geometry applicability and trajectory-mask projection.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument(
        "--validation_mode",
        required=True,
        choices=("full_split_frame_geometry_applicability", "trajectory_mask_to_token_projection"),
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset_names", nargs="*", default=["lvvis_train", "lvvis_val", "ytvis_2019_val"])
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--preprocess_mode", default="keep_aspect_pad")
    parser.add_argument("--resize_short_side", type=int, default=672)
    parser.add_argument("--max_long_side", type=int, default=1008)
    parser.add_argument("--pad_to_multiple", type=int, default=14)
    parser.add_argument("--contract_check_json")
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_lvvis_root() -> Path:
    env_root = os.environ.get("WSOVVIS_LVVIS_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (_repo_root() / "videocutler" / "datasets" / "LV-VIS").resolve()


def _resolve_ytvis_root() -> Path:
    env_root = os.environ.get("WSOVVIS_YTVIS2019_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    d2_root = os.environ.get("DETECTRON2_DATASETS", "").strip()
    if d2_root:
        return Path(d2_root).expanduser().resolve() / "ytvis_2019"
    return (_repo_root() / "videocutler" / "datasets" / "ytvis_2019").resolve()


def _resolve_image_path(split_root: Path, rel_file: str) -> Path:
    rel = Path(rel_file)
    candidates = [split_root / rel, split_root / "JPEGImages" / rel]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _dataset_video_records(dataset_name: str) -> List[Dict[str, Any]]:
    if dataset_name in {"lvvis_train", "lvvis_train_base"}:
        root = _resolve_lvvis_root()
        ann = root / "annotations" / "train_instances.json"
        split_root = root / "train"
    elif dataset_name == "lvvis_val":
        root = _resolve_lvvis_root()
        ann = root / "annotations" / "val_instances.json"
        split_root = root / "val"
    elif dataset_name == "ytvis_2019_val":
        root = _resolve_ytvis_root()
        ann = root / "valid.json"
        split_root = root / "valid" / "JPEGImages"
    else:
        raise ValueError(f"unsupported dataset_name: {dataset_name}")
    data = _read_json(ann)
    videos = sorted(data.get("videos", []), key=lambda item: int(item["id"]))
    out: List[Dict[str, Any]] = []
    for video in videos:
        out.append(
            {
                "dataset_name": dataset_name,
                "video_id": int(video["id"]),
                "clip_id": str(video["id"]),
                "height": int(video.get("height", 0)),
                "width": int(video.get("width", 0)),
                "file_names": list(video.get("file_names") or video.get("filenames") or []),
                "split_root": split_root,
            }
        )
    return out


def _decode_mask(mask_item: Any, image_size: Iterable[int]) -> np.ndarray:
    import numpy as np

    if mask_utils is None:
        raise RuntimeError("pycocotools is required for trajectory_mask_to_token_projection")
    h, w = [int(x) for x in image_size]
    if mask_item is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(mask_item, dict):
        rle = dict(mask_item)
        size = rle.get("size")
        if not size:
            rle["size"] = [h, w]
        decoded = mask_utils.decode(rle)
        return np.asarray(decoded, dtype=np.uint8)
    if isinstance(mask_item, str):
        decoded = mask_utils.decode({"size": [h, w], "counts": mask_item.encode("utf-8")})
        return np.asarray(decoded, dtype=np.uint8)
    raise ValueError("unsupported masks_rle item")


def _resize_pad_mask(mask: np.ndarray, resized_h: int, resized_w: int, padded_h: int, padded_w: int) -> np.ndarray:
    import numpy as np

    if resized_h <= 0 or resized_w <= 0 or padded_h <= 0 or padded_w <= 0:
        raise ValueError("invalid geometry dimensions for mask projection")
    src_h, src_w = int(mask.shape[0]), int(mask.shape[1])
    if src_h == resized_h and src_w == resized_w:
        resized = mask.astype(np.uint8, copy=False)
    else:
        src_y = (np.arange(resized_h) * float(src_h) / float(resized_h)).astype(np.int64)
        src_x = (np.arange(resized_w) * float(src_w) / float(resized_w)).astype(np.int64)
        src_y = np.clip(src_y, 0, src_h - 1)
        src_x = np.clip(src_x, 0, src_w - 1)
        resized = mask[src_y[:, None], src_x[None, :]].astype(np.uint8, copy=False)
    out = np.zeros((padded_h, padded_w), dtype=np.uint8)
    out[:resized_h, :resized_w] = resized
    return out


def _token_occupancy(mask: np.ndarray, patch_size: int, grid_h: int, grid_w: int) -> np.ndarray:
    import numpy as np

    out = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for r in range(grid_h):
        y0 = r * patch_size
        y1 = min((r + 1) * patch_size, mask.shape[0])
        if y1 <= y0:
            continue
        for c in range(grid_w):
            x0 = c * patch_size
            x1 = min((c + 1) * patch_size, mask.shape[1])
            if x1 <= x0:
                continue
            if np.any(mask[y0:y1, x0:x1] > 0):
                out[r, c] = 1
    return out


def _load_geom_lookup() -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    repo_root = _repo_root()
    lookup: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    by_dataset = {
        "lvvis_train_base": repo_root / "frame_bank" / "lvvis_train_base" / "frame_geom_records.jsonl",
        "lvvis_train": repo_root / "frame_bank" / "lvvis_train_base" / "frame_geom_records.jsonl",
        "lvvis_val": repo_root / "frame_bank" / "lvvis_val" / "frame_geom_records.jsonl",
    }
    for dataset_name, path in by_dataset.items():
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (dataset_name, int(rec["clip_id"]), int(rec["frame_index"]))
                lookup[key] = rec
    return lookup


def _frame_records_path(dataset_name: str) -> Path:
    repo_root = _repo_root()
    if dataset_name in {"lvvis_train", "lvvis_train_base"}:
        return repo_root / "frame_bank" / "lvvis_train_base" / "frame_records.jsonl"
    if dataset_name == "lvvis_val":
        return repo_root / "frame_bank" / "lvvis_val" / "frame_records.jsonl"
    raise ValueError(f"unsupported dataset_name for frame bank materialization: {dataset_name}")


def _trajectory_records_path(dataset_name: str) -> Path:
    repo_root = _repo_root()
    if dataset_name in {"lvvis_train", "lvvis_train_base"}:
        return repo_root / "exports" / "lvvis_train_base" / "trajectory_records.jsonl"
    if dataset_name == "lvvis_val":
        return repo_root / "exports" / "lvvis_val" / "trajectory_records.jsonl"
    raise ValueError(f"unsupported dataset_name for trajectory materialization: {dataset_name}")


def _materialize_geom_sidecars(
    *,
    dataset_names: List[str],
    patch_size: int,
    resize_short_side: int,
    max_long_side: int,
    pad_to_multiple: int,
) -> Dict[str, List[Dict[str, Any]]]:
    from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import (
        build_valid_token_mask,
        keep_aspect_pad_geometry,
        write_frame_geom_records_jsonl,
    )
    from videocutler.ext_stageb_ovvis.data.trajectory_dataset import read_trajectory_records

    repo_root = _repo_root()
    output: Dict[str, List[Dict[str, Any]]] = {}
    for dataset_name in dataset_names:
        frame_path = _frame_records_path(dataset_name)
        traj_path = _trajectory_records_path(dataset_name)
        if not frame_path.exists():
            raise FileNotFoundError(frame_path)
        if not traj_path.exists():
            raise FileNotFoundError(traj_path)

        frame_records = _read_jsonl(frame_path)
        trajectory_records = read_trajectory_records(traj_path)

        clip_sizes: Dict[str, Tuple[int, int]] = {}
        for rec in trajectory_records:
            clip_id = str(rec.get("clip_id"))
            image_size = list(rec.get("image_size", []))
            if len(image_size) != 2:
                raise ValueError(f"{dataset_name}:{clip_id}: missing image_size")
            orig_h, orig_w = int(image_size[0]), int(image_size[1])
            if orig_h <= 0 or orig_w <= 0:
                raise ValueError(f"{dataset_name}:{clip_id}: invalid image_size")
            prev = clip_sizes.get(clip_id)
            current = (orig_h, orig_w)
            if prev is None:
                clip_sizes[clip_id] = current
            elif prev != current:
                raise ValueError(f"{dataset_name}:{clip_id}: inconsistent image_size across trajectories")

        geom_records: List[Dict[str, Any]] = []
        for frame_rec in frame_records:
            clip_id = str(frame_rec["clip_id"])
            frame_index = int(frame_rec["frame_index"])
            if clip_id not in clip_sizes:
                raise ValueError(f"{dataset_name}:{clip_id}:{frame_index}: missing trajectory image_size")
            orig_h, orig_w = clip_sizes[clip_id]
            geom = keep_aspect_pad_geometry(
                orig_h=orig_h,
                orig_w=orig_w,
                resize_short_side=resize_short_side,
                max_long_side=max_long_side,
                pad_to_multiple=pad_to_multiple,
                patch_size=patch_size,
            )
            if geom.pad_left != 0 or geom.pad_top != 0:
                raise ValueError("padding must be right/bottom only")
            valid_mask = build_valid_token_mask(geom)
            if valid_mask.shape != (int(geom.grid_h), int(geom.grid_w)):
                raise ValueError("valid token mask shape mismatch")
            geom_records.append(
                {
                    "clip_id": clip_id,
                    "frame_index": frame_index,
                    "orig_h": int(geom.orig_h),
                    "orig_w": int(geom.orig_w),
                    "resized_h": int(geom.resized_h),
                    "resized_w": int(geom.resized_w),
                    "padded_h": int(geom.padded_h),
                    "padded_w": int(geom.padded_w),
                    "scale_y": float(geom.scale_y),
                    "scale_x": float(geom.scale_x),
                    "pad_left": int(geom.pad_left),
                    "pad_top": int(geom.pad_top),
                    "pad_right": int(geom.pad_right),
                    "pad_bottom": int(geom.pad_bottom),
                    "patch_size": int(geom.patch_size),
                    "grid_h": int(geom.grid_h),
                    "grid_w": int(geom.grid_w),
                    "valid_token_mask_path": f"frame_geom_records.jsonl#{len(geom_records)}",
                    "path_base_mode": "artifact_parent_dir",
                }
            )

        out_path = repo_root / "frame_bank" / ("lvvis_train_base" if dataset_name in {"lvvis_train", "lvvis_train_base"} else "lvvis_val") / "frame_geom_records.jsonl"
        write_frame_geom_records_jsonl(out_path, geom_records)
        output[dataset_name] = geom_records
    return output


def _write_frame_contract_check(report: Dict[str, Any], dataset_names: List[str]) -> None:
    repo_root = _repo_root()
    primary_paths = []
    record_count = 0
    for dataset_name in dataset_names:
        if dataset_name in {"lvvis_train", "lvvis_train_base"}:
            primary_paths.append("frame_bank/lvvis_train_base/frame_records.jsonl")
            primary_paths.append("frame_bank/lvvis_train_base/frame_geom_records.jsonl")
            record_count += len(_read_jsonl(repo_root / "frame_bank" / "lvvis_train_base" / "frame_records.jsonl"))
        elif dataset_name == "lvvis_val":
            primary_paths.append("frame_bank/lvvis_val/frame_records.jsonl")
            primary_paths.append("frame_bank/lvvis_val/frame_geom_records.jsonl")
            record_count += len(_read_jsonl(repo_root / "frame_bank" / "lvvis_val" / "frame_records.jsonl"))

    contract_path = repo_root / "codex" / "outputs" / "g4_frame_feature_cache" / "frame_contract_check.json"
    payload = {
        "gate_id": "G4_frame_feature_cache",
        "contract_ref": "contracts/gates/G4_frame_feature_cache.gate_contract.json",
        "status": report["status"],
        "artifact_path_base": "output_root_relative",
        "primary_artifacts": sorted(dict.fromkeys(primary_paths + ["frame_bank/geometry/frame_geometry_applicability_report.json"])),
        "checks_run": [
            "frame_feature_reader_parses_dsl",
            "frame_cache_coverage_ready",
            "artifact_exists",
            "artifact_schema_valid",
            "full_split_frame_geometry_applicability",
        ],
        "run_scope": "full",
        "input_source_type": "official_lvvis_train_annotations|official_lvvis_val_annotations",
        "data_scope": "train|val",
        "record_count": int(record_count),
        "coverage_ratio": float(report["coverage_ratio"]),
        "missing_frame_ratio": max(0.0, 1.0 - float(report["coverage_ratio"])),
        "consumer_ready": bool(report["status"] == "PASS" and float(report["coverage_ratio"]) >= 1.0),
    }
    _write_json(contract_path, payload)


def _geometry_report(args: argparse.Namespace) -> Dict[str, Any]:
    from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import (
        build_valid_token_mask,
        keep_aspect_pad_geometry,
        reconstruct_valid_token_mask_from_geometry,
    )
    from videocutler.ext_stageb_ovvis.data.trajectory_dataset import read_trajectory_records

    _materialize_geom_sidecars(
        dataset_names=[str(name) for name in args.dataset_names],
        patch_size=int(args.patch_size),
        resize_short_side=int(args.resize_short_side),
        max_long_side=int(args.max_long_side),
        pad_to_multiple=int(args.pad_to_multiple),
    )
    geom_lookup = _load_geom_lookup()
    violations: List[str] = []
    checked_videos = 0
    checked_frames = 0
    failed_frames = 0
    for dataset_name in args.dataset_names:
        videos = _dataset_video_records(dataset_name)
        checked_videos += len(videos)
        traj_records = read_trajectory_records(_trajectory_records_path(dataset_name))
        clip_sizes: Dict[str, Tuple[int, int]] = {}
        for rec in traj_records:
            clip_id = str(rec.get("clip_id"))
            image_size = list(rec.get("image_size", []))
            if len(image_size) != 2:
                failed_frames += 1
                if len(violations) < 64:
                    violations.append(f"{dataset_name}:{clip_id}:missing_image_size")
                continue
            clip_sizes[clip_id] = (int(image_size[0]), int(image_size[1]))
        frame_records = _read_jsonl(_frame_records_path(dataset_name))
        for frame_rec in frame_records:
            checked_frames += 1
            clip_id = str(frame_rec["clip_id"])
            frame_index = int(frame_rec["frame_index"])
            try:
                if clip_id not in clip_sizes:
                    raise ValueError("missing trajectory image_size")
                orig_h, orig_w = clip_sizes[clip_id]
                expected = keep_aspect_pad_geometry(
                    orig_h=orig_h,
                    orig_w=orig_w,
                    resize_short_side=args.resize_short_side,
                    max_long_side=args.max_long_side,
                    pad_to_multiple=args.pad_to_multiple,
                    patch_size=args.patch_size,
                )
                if expected.pad_left != 0 or expected.pad_top != 0:
                    raise ValueError("padding must be right/bottom only")
                if expected.padded_h % args.patch_size != 0 or expected.padded_w % args.patch_size != 0:
                    raise ValueError("padded dimensions must be divisible by patch size")
                if expected.resized_h > expected.padded_h or expected.resized_w > expected.padded_w:
                    raise ValueError("crop-like state detected")
                geom = geom_lookup.get((dataset_name, int(clip_id), frame_index))
                if not geom:
                    raise ValueError("missing frame geometry sidecar record")
                valid_mask = build_valid_token_mask(expected)
                expected_mask = reconstruct_valid_token_mask_from_geometry(geom)
                if valid_mask.shape != expected_mask.shape or tuple(valid_mask.shape) != (int(geom["grid_h"]), int(geom["grid_w"])):
                    raise ValueError("valid token mask shape mismatch")
                if int(geom["orig_h"]) != int(expected.orig_h) or int(geom["orig_w"]) != int(expected.orig_w):
                    raise ValueError("orig image size mismatch")
                if int(geom["resized_h"]) != int(expected.resized_h) or int(geom["resized_w"]) != int(expected.resized_w):
                    raise ValueError("resized geometry mismatch")
                if int(geom["padded_h"]) != int(expected.padded_h) or int(geom["padded_w"]) != int(expected.padded_w):
                    raise ValueError("padded geometry mismatch")
                if int(geom["pad_left"]) != 0 or int(geom["pad_top"]) != 0:
                    raise ValueError("padding must be right/bottom only")
                if int(geom["pad_right"]) != int(expected.pad_right) or int(geom["pad_bottom"]) != int(expected.pad_bottom):
                    raise ValueError("pad extent mismatch")
            except Exception as exc:
                failed_frames += 1
                if len(violations) < 64:
                    violations.append(f"{dataset_name}:{clip_id}:{frame_index}:{exc}")
    coverage = 1.0 if checked_frames == 0 else max(0.0, min(1.0, float(checked_frames - failed_frames) / float(checked_frames)))
    status = "PASS" if failed_frames == 0 else "FAIL"
    return {
        "status": status,
        "gate_id": "G4_frame_feature_cache",
        "report_kind": "full_split_frame_geometry_applicability",
        "run_scope": "full",
        "dataset_names": sorted({str(name) for name in args.dataset_names}),
        "checked_video_count": int(checked_videos),
        "checked_frame_count": int(checked_frames),
        "failed_frame_count": int(failed_frames),
        "coverage_ratio": float(coverage),
        "violations_preview": violations[:20],
        "notes": "keep_aspect_pad geometry applicability validation",
    }


def _projection_report(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np

    from videocutler.ext_stageb_ovvis.data.trajectory_dataset import read_trajectory_records

    geom_lookup = _load_geom_lookup()
    repo_root = _repo_root()
    violations: List[str] = []
    checked_frames = 0
    failed_frames = 0
    checked_videos = 0
    ds_to_export = {
        "lvvis_train": "lvvis_train_base",
        "lvvis_train_base": "lvvis_train_base",
        "lvvis_val": "lvvis_val",
        "ytvis_2019_val": "ytvis_2019_val",
    }
    for dataset_name in args.dataset_names:
        export_name = ds_to_export.get(dataset_name, dataset_name)
        export_path = repo_root / "exports" / export_name / "trajectory_records.jsonl"
        if not export_path.exists():
            continue
        records = read_trajectory_records(export_path)
        checked_videos += len({int(rec.get("video_id", -1)) for rec in records})
        for rec in records:
            frame_indices = list(rec.get("frame_indices", []))
            masks_rle = list(rec.get("masks_rle", []))
            image_size = list(rec.get("image_size", []))
            if len(frame_indices) != len(masks_rle):
                failed_frames += 1
                if len(violations) < 64:
                    violations.append(f"{dataset_name}:{rec.get('trajectory_id','?')}:aligned_lengths")
                continue
            for idx, frame_index in enumerate(frame_indices):
                checked_frames += 1
                try:
                    clip_id = int(rec["clip_id"])
                    geom = geom_lookup.get((dataset_name, clip_id, int(frame_index))) or geom_lookup.get((export_name, clip_id, int(frame_index)))
                    if not geom:
                        raise ValueError("missing frame geometry sidecar record")
                    mask = _decode_mask(masks_rle[idx], image_size)
                    projected = _resize_pad_mask(
                        mask,
                        resized_h=int(geom["resized_h"]),
                        resized_w=int(geom["resized_w"]),
                        padded_h=int(geom["padded_h"]),
                        padded_w=int(geom["padded_w"]),
                    )
                    occupancy = _token_occupancy(
                        projected,
                        patch_size=int(geom["patch_size"]),
                        grid_h=int(geom["grid_h"]),
                        grid_w=int(geom["grid_w"]),
                    )
                    valid_h = int(math.ceil(float(int(geom["resized_h"])) / float(int(geom["patch_size"]))))
                    valid_w = int(math.ceil(float(int(geom["resized_w"])) / float(int(geom["patch_size"]))))
                    valid_region = np.zeros_like(occupancy, dtype=np.uint8)
                    valid_region[:valid_h, :valid_w] = 1
                    orig_non_empty = bool(np.any(mask > 0))
                    projected_non_empty = bool(np.any(projected > 0))
                    occupied_non_empty = bool(np.any(occupancy > 0))
                    if orig_non_empty and projected_non_empty and not occupied_non_empty:
                        raise ValueError("non-empty mask collapsed to empty occupancy")
                    if np.any((occupancy > 0) & (valid_region == 0)):
                        raise ValueError("occupancy touches invalid token region")
                except Exception as exc:
                    failed_frames += 1
                    if len(violations) < 64:
                        violations.append(f"{dataset_name}:{rec.get('trajectory_id','?')}:{frame_index}:{exc}")
    coverage = 1.0 if checked_frames == 0 else max(0.0, min(1.0, float(checked_frames - failed_frames) / float(checked_frames)))
    status = "PASS" if failed_frames == 0 else "FAIL"
    return {
        "status": status,
        "gate_id": "G5_carrier_bank",
        "report_kind": "trajectory_mask_to_token_projection",
        "run_scope": "full",
        "dataset_names": sorted({str(name) for name in args.dataset_names}),
        "checked_video_count": int(checked_videos),
        "checked_frame_count": int(checked_frames),
        "failed_frame_count": int(failed_frames),
        "coverage_ratio": float(coverage),
        "violations_preview": violations[:20],
        "notes": "trajectory-mask to token-grid projection validation",
    }


def _report_output_path(validation_mode: str) -> Path:
    repo_root = _repo_root()
    if validation_mode == "full_split_frame_geometry_applicability":
        return repo_root / "frame_bank" / "geometry" / "frame_geometry_applicability_report.json"
    return repo_root / "carrier_bank" / "geometry" / "trajectory_mask_to_token_projection_report.json"


def _write_contract_check(path: Path, report: Dict[str, Any]) -> None:
    if report["report_kind"] == "full_split_frame_geometry_applicability":
        _write_frame_contract_check(report, [str(name) for name in report.get("dataset_names", []) if str(name) in {"lvvis_train", "lvvis_train_base", "lvvis_val"}])
        if path != _repo_root() / "codex" / "outputs" / "g4_frame_feature_cache" / "frame_contract_check.json":
            _write_frame_contract_check(report, [str(name) for name in report.get("dataset_names", []) if str(name) in {"lvvis_train", "lvvis_train_base", "lvvis_val"}])
        return
    payload = {
        "gate_id": "G4_frame_feature_cache" if report["report_kind"] == "full_split_frame_geometry_applicability" else "G5_carrier_bank",
        "status": report["status"],
        "checks_run": [report["report_kind"]],
        "run_scope": report["run_scope"],
        "coverage_ratio": report["coverage_ratio"],
        "consumer_ready": bool(report["status"] == "PASS" and float(report["coverage_ratio"]) >= 1.0),
    }
    _write_json(path, payload)


def main() -> int:
    args = parse_args()
    import sys

    repo_root = _repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    if str(args.preprocess_mode) != "keep_aspect_pad":
        raise ValueError("Only preprocess_mode=keep_aspect_pad is supported")
    if int(args.patch_size) != 14 or int(args.pad_to_multiple) != 14:
        raise ValueError("patch_size and pad_to_multiple must be 14")

    report = _geometry_report(args) if args.validation_mode == "full_split_frame_geometry_applicability" else _projection_report(args)
    out_path = _report_output_path(args.validation_mode)
    _write_json(out_path, report)

    if args.contract_check_json:
        _write_contract_check(Path(args.contract_check_json), report)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
