#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

def load_mask_ids(png_path: Path) -> np.ndarray:
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
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def mask_to_bbox_area(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    area = int(mask.sum())
    return bbox, area

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_root", required=True, help="Root dir containing per-video folders with mask_*.png")
    ap.add_argument("--out_json", required=True, help="Output JSON path")
    ap.add_argument("--split_name", default="train")
    args = ap.parse_args()

    mask_root = Path(args.mask_root)
    videos = sorted([p for p in mask_root.iterdir() if p.is_dir()])

    out = {
        "split": args.split_name,
        "videos": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],  # class-agnostic
    }

    ann_id = 1
    for vid_id, vdir in enumerate(videos, start=1):
        pngs = sorted(vdir.glob("mask_*.png"))
        if not pngs:
            continue

        frames = [load_mask_ids(p) for p in pngs]
        T = len(frames)

        all_ids = np.unique(np.concatenate([np.unique(f).reshape(-1) for f in frames]))
        inst_ids = [int(x) for x in all_ids.tolist() if int(x) != 0]

        out["videos"].append({"id": vid_id, "name": vdir.name, "length": T})

        for inst in inst_ids:
            segs, bboxes, areas = [], [], []
            any_valid = False
            for t in range(T):
                m = (frames[t] == inst)
                if m.any():
                    any_valid = True
                    segs.append(binary_to_rle(m))
                    bbox, area = mask_to_bbox_area(m)
                    bboxes.append(bbox)
                    areas.append(area)
                else:
                    segs.append(None)
                    bboxes.append(None)
                    areas.append(0)

            if not any_valid:
                continue

            out["annotations"].append({
                "id": ann_id,
                "video_id": vid_id,
                "category_id": 1,
                "segmentations": segs,
                "bboxes": bboxes,
                "areas": areas,
            })
            ann_id += 1

    os.makedirs(Path(args.out_json).parent, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[OK] wrote {args.out_json} | videos={len(out['videos'])} anns={len(out['annotations'])}")

if __name__ == "__main__":
    main()
