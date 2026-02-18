#!/usr/bin/env python3
"""Convert VideoCutler pseudo mask tubes to a (mostly) standard YouTube-VIS style JSON.

Scheme B (default):
  - videos: {id, file_names, height, width, length, name}
  - annotations: {id, video_id, category_id, iscrowd, segmentations, bboxes, areas}
  - categories: class-agnostic single category by default

Expected inputs (following README.md in this repo):
  mask_root/<video_id>/mask_00000.png, mask_00001.png, ...
  img_root/<video_id>/*.jpg (or .png)

Notes
-----
1) We align masks to frames by numeric index when possible (recommended), otherwise by sorted order.
2) Each mask PNG is an *instance id map* (0=background). For RGB PNGs, we decode id = R<<16 + G<<8 + B.
3) We emit COCO RLE for each instance in each frame.

This JSON is compatible with common YouTube-VIS loaders (including those used by SeqFormer in VNext).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


_RE_INT = re.compile(r"(\d+)")


def _stem_int(p: Path) -> Optional[int]:
    """Extract last integer in filename stem. Return None if not found."""
    m = _RE_INT.findall(p.stem)
    if not m:
        return None
    return int(m[-1])


def load_id_map(png_path: Path) -> np.ndarray:
    arr = np.array(Image.open(png_path))
    if arr.ndim == 2:
        return arr.astype(np.int32)
    if arr.ndim == 3 and arr.shape[2] == 3:
        r = arr[..., 0].astype(np.int32)
        g = arr[..., 1].astype(np.int32)
        b = arr[..., 2].astype(np.int32)
        return (r << 16) + (g << 8) + b
    raise ValueError(f"Unsupported mask shape {arr.shape}: {png_path}")


def binary_to_rle(mask: np.ndarray) -> dict:
    m = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(m)
    # pycocotools returns bytes for counts
    if isinstance(rle.get("counts"), (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def read_image_hw(img_path: Path) -> Tuple[int, int]:
    with Image.open(img_path) as im:
        w, h = im.size
    return int(h), int(w)


def list_frames(img_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    frames: List[Path] = []
    for ext in exts:
        frames.extend(sorted(img_dir.glob(f"*.{ext}")))
    # ensure stable sorting
    frames = sorted(frames)
    return frames


def build_frame_index(frames: List[Path]) -> Dict[int, Path]:
    """Map frame index -> path. If stem has integer, use it; otherwise use enumerate index."""
    idx_map: Dict[int, Path] = {}
    all_have_int = all((_stem_int(p) is not None) for p in frames)
    if all_have_int:
        for p in frames:
            idx_map[_stem_int(p)] = p  # type: ignore[arg-type]
    else:
        for i, p in enumerate(frames):
            idx_map[i] = p
    return idx_map


def build_mask_index(masks: List[Path]) -> Dict[int, Path]:
    """Map mask index -> path. If stem has integer, use it; otherwise use enumerate index."""
    idx_map: Dict[int, Path] = {}
    all_have_int = all((_stem_int(p) is not None) for p in masks)
    if all_have_int:
        for p in masks:
            idx_map[_stem_int(p)] = p  # type: ignore[arg-type]
    else:
        for i, p in enumerate(masks):
            idx_map[i] = p
    return idx_map


def convert_video(
    vid_id: int,
    video_name: str,
    img_root: Optional[Path],
    mask_dir: Path,
    img_exts: Tuple[str, ...],
    category_id: int,
    strict: bool,
    mask_glob: str,
    mask_glob_fallback: str,
    error_on_empty: bool,
    min_masks: int,
    min_coverage: float,
    skip_report: Optional[List[dict]] = None,
) -> Tuple[Optional[dict], List[dict]]:
    """Return (video_dict, annotations_for_video)."""

    mask_files = sorted(mask_dir.glob(mask_glob))
    if (not mask_files) and mask_glob_fallback:
        mask_files = sorted(mask_dir.glob(mask_glob_fallback))
    if not mask_files:
        msg = f"No mask PNGs found (glob={mask_glob}, fallback={mask_glob_fallback}): {mask_dir}"
        if error_on_empty:
            raise RuntimeError(msg)
        print(f"[WARN] skip {video_name}: {msg}")
        if skip_report is not None:
            skip_report.append(
                {
                    "video": video_name,
                    "reason": "no_masks",
                    "mask_dir": str(mask_dir),
                }
            )
        return None, []

    if img_root is None:
        # legacy: no image paths/size
        T = len(mask_files)
        if T < max(1, int(min_masks)):
            print(
                f"[WARN] skip {video_name}: too_few_masks masks={T} < min_masks={min_masks}"
            )
            if skip_report is not None:
                skip_report.append(
                    {
                        "video": video_name,
                        "reason": "too_few_masks",
                        "masks": T,
                        "min_masks": int(min_masks),
                        "mask_dir": str(mask_dir),
                    }
                )
            return None, []

        video = {"id": vid_id, "name": video_name, "length": T}
        frame_indices = list(range(T))
        # align by sorted order
        frame_files = {i: mask_files[i] for i in frame_indices}
    else:
        img_dir = img_root / video_name
        frames = list_frames(img_dir, img_exts)
        if not frames:
            msg = f"No frames found in: {img_dir}"
            if error_on_empty:
                raise RuntimeError(msg)
            print(f"[WARN] skip {video_name}: {msg}")
            if skip_report is not None:
                skip_report.append(
                    {
                        "video": video_name,
                        "reason": "no_frames",
                        "frame_dir": str(img_dir),
                        "mask_dir": str(mask_dir),
                    }
                )
            return None, []

        # Skip rules (before heavy decoding)
        T_frames = len(frames)
        M_masks = len(mask_files)
        cov = float(M_masks) / float(T_frames) if T_frames > 0 else 0.0
        if M_masks < max(1, int(min_masks)):
            print(
                f"[WARN] skip {video_name}: too_few_masks masks={M_masks} < min_masks={min_masks}"
            )
            if skip_report is not None:
                skip_report.append(
                    {
                        "video": video_name,
                        "reason": "too_few_masks",
                        "frames": T_frames,
                        "masks": M_masks,
                        "coverage": cov,
                        "min_masks": int(min_masks),
                        "min_coverage": float(min_coverage),
                        "frame_dir": str(img_dir),
                        "mask_dir": str(mask_dir),
                    }
                )
            return None, []
        if cov < float(min_coverage):
            print(
                f"[WARN] skip {video_name}: low_coverage masks={M_masks} frames={T_frames} coverage={cov:.3f} < min_coverage={min_coverage}"
            )
            if skip_report is not None:
                skip_report.append(
                    {
                        "video": video_name,
                        "reason": "low_coverage",
                        "frames": T_frames,
                        "masks": M_masks,
                        "coverage": cov,
                        "min_masks": int(min_masks),
                        "min_coverage": float(min_coverage),
                        "frame_dir": str(img_dir),
                        "mask_dir": str(mask_dir),
                    }
                )
            return None, []

        frame_map = build_frame_index(frames)
        mask_map = build_mask_index(mask_files)

        frame_indices = sorted(frame_map.keys())
        # Determine T by frames, not masks.
        T = len(frame_indices)
        if strict and (len(mask_map) != len(frame_map)):
            raise RuntimeError(
                f"Length mismatch for {video_name}: frames={len(frame_map)} masks={len(mask_map)}"
            )

        first_frame = frame_map[frame_indices[0]]
        h, w = read_image_hw(first_frame)

        # file_names should be relative to img_root
        file_names = [str((Path(video_name) / frame_map[i].name).as_posix()) for i in frame_indices]

        video = {
            "id": vid_id,
            "name": video_name,
            "file_names": file_names,
            "height": h,
            "width": w,
            "length": T,
        }
        frame_files = {i: mask_map.get(i) for i in frame_indices}

    # One pass: stream frames and fill instance tracks
    inst_tracks: Dict[int, dict] = {}

    def _ensure_track(inst_id: int):
        if inst_id not in inst_tracks:
            inst_tracks[inst_id] = {
                "segmentations": [None] * T,
                "bboxes": [None] * T,
                "areas": [0] * T,
            }

    for t, frame_i in enumerate(frame_indices):
        mp = frame_files[frame_i]
        if mp is None:
            # no mask for this frame
            continue
        id_map = load_id_map(mp)
        ids = np.unique(id_map)
        ids = ids[ids != 0]
        for inst_id in ids.tolist():
            inst_id = int(inst_id)
            _ensure_track(inst_id)
            binary = (id_map == inst_id)
            if not binary.any():
                continue
            rle = binary_to_rle(binary)
            bbox = mask_utils.toBbox(rle).tolist()  # [x,y,w,h]
            area = int(mask_utils.area(rle))
            inst_tracks[inst_id]["segmentations"][t] = rle
            inst_tracks[inst_id]["bboxes"][t] = bbox
            inst_tracks[inst_id]["areas"][t] = area

    anns: List[dict] = []
    return video, [
        {
            "video_id": vid_id,
            "category_id": category_id,
            "iscrowd": 0,
            "segmentations": v["segmentations"],
            "bboxes": v["bboxes"],
            "areas": v["areas"],
            "_inst_id": inst_id,  # for deterministic sorting (removed later)
        }
        for inst_id, v in inst_tracks.items()
        if any(s is not None for s in v["segmentations"])
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_root", required=True, help="Root dir: mask_root/<video_id>/mask_*.png")
    ap.add_argument("--img_root", default=None, help="Root dir: img_root/<video_id>/*.jpg")
    ap.add_argument("--img_exts", default="jpg,jpeg,png", help="Comma-separated image extensions")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--split_name", default="train")
    ap.add_argument("--category_id", type=int, default=1)
    ap.add_argument("--strict", action="store_true", help="Fail if masks/frames length mismatch")
    ap.add_argument("--mask_glob", default="mask_*.png", help="Glob for mask PNGs inside each video dir")
    ap.add_argument("--mask_glob_fallback", default="*.png", help="Fallback glob if --mask_glob matches nothing")
    ap.add_argument("--error_on_empty", action="store_true", help="Error if a video dir has no masks (default: skip with warning)")
    ap.add_argument(
        "--skip_list",
        default=None,
        help="Optional text file (one video_id per line) to skip when building JSON.",
    )
    ap.add_argument(
        "--min_masks",
        type=int,
        default=3,
        help="Skip a video if the number of mask PNGs is < min_masks (default: 3).",
    )
    ap.add_argument(
        "--min_coverage",
        type=float,
        default=0.2,
        help="Skip a video if (#mask_png / #frames) < min_coverage (default: 0.2).",
    )
    ap.add_argument(
        "--skip_report",
        default=None,
        help="Optional output path for a JSON skip report. Default: <out_json>.skipped.json",
    )
    args = ap.parse_args()

    mask_root = Path(args.mask_root)
    img_root = Path(args.img_root) if args.img_root else None
    img_exts = tuple([e.strip().lstrip(".") for e in args.img_exts.split(",") if e.strip()])

    video_dirs = sorted([p for p in mask_root.iterdir() if p.is_dir()])

    # optional skip list (video dir names)
    skip_set = set()
    if args.skip_list:
        with open(args.skip_list, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                skip_set.add(s)

    skip_report: List[dict] = []
    out = {
        "split": args.split_name,
        "videos": [],
        "annotations": [],
        "categories": [{"id": args.category_id, "name": "object"}],
    }

    ann_id = 1
    vid_id = 1
    for vdir in video_dirs:
        video_name = vdir.name
        if video_name in skip_set:
            print(f"[WARN] skip {video_name}: in_skip_list")
            skip_report.append(
                {
                    "video": video_name,
                    "reason": "skip_list",
                    "mask_dir": str(vdir),
                }
            )
            continue
        video, video_anns = convert_video(
            vid_id=vid_id,
            video_name=video_name,
            img_root=img_root,
            mask_dir=vdir,
            img_exts=img_exts,
            category_id=args.category_id,
            strict=args.strict,
            mask_glob=args.mask_glob,
            mask_glob_fallback=args.mask_glob_fallback,
            error_on_empty=args.error_on_empty,
            min_masks=args.min_masks,
            min_coverage=args.min_coverage,
            skip_report=skip_report,
        )
        if video is None:
            continue
        out["videos"].append(video)

        # deterministic: sort by original instance id
        video_anns = sorted(video_anns, key=lambda d: d.get("_inst_id", 0))
        for a in video_anns:
            a["id"] = ann_id
            a.pop("_inst_id", None)
            out["annotations"].append(a)
            ann_id += 1

        vid_id += 1

    os.makedirs(Path(args.out_json).parent, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    # write skip report (if any)
    skip_report_path = args.skip_report or (args.out_json + ".skipped.json")
    os.makedirs(Path(skip_report_path).parent, exist_ok=True)
    with open(skip_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "out_json": args.out_json,
                "min_masks": int(args.min_masks),
                "min_coverage": float(args.min_coverage),
                "skip_list": args.skip_list,
                "skipped": skip_report,
                "num_skipped": len(skip_report),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[OK] wrote {args.out_json} | videos={len(out['videos'])} anns={len(out['annotations'])} | skipped={len(skip_report)}"
    )
    print(f"[OK] skip report: {skip_report_path}")


if __name__ == "__main__":
    main()
