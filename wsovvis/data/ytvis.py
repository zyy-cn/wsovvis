from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


def _build_category_mapping(categories: List[Dict[str, Any]]) -> Dict[int, int]:
    """Map dataset category id -> contiguous id starting at 0."""
    ids = sorted({int(c["id"]) for c in categories})
    return {i: idx for idx, i in enumerate(ids)}



def _instances_to_per_frame_anns(inst_anns: List[Dict[str, Any]], num_frames: int) -> List[List[Dict[str, Any]]]:
    """Convert instance-level YTVIS annotations into per-frame annotations expected by SeqFormer.

    SeqFormer/VNext's YTVISDatasetMapper expects dataset_dict["annotations"] to be a list of length T,
    where each element is a list of per-instance dicts for that frame.
    """
    frame_anns: List[List[Dict[str, Any]]] = [[] for _ in range(num_frames)]

    for inst in inst_anns:
        segs = inst.get("segmentations") or []
        boxes = inst.get("bboxes") or []
        areas = inst.get("areas") or []

        # Pad/truncate to num_frames defensively
        if len(segs) < num_frames:
            segs = list(segs) + [None] * (num_frames - len(segs))
        else:
            segs = list(segs)[:num_frames]

        if len(boxes) < num_frames:
            boxes = list(boxes) + [None] * (num_frames - len(boxes))
        else:
            boxes = list(boxes)[:num_frames]

        if len(areas) < num_frames:
            areas = list(areas) + [None] * (num_frames - len(areas))
        else:
            areas = list(areas)[:num_frames]

        # Lazy import to avoid hard dependency issues in environments without pycocotools
        try:
            from pycocotools import mask as mask_utils  # type: ignore
        except Exception:
            mask_utils = None

        for t in range(num_frames):
            seg_t = segs[t]
            if seg_t is None:
                continue

            bbox_t = boxes[t] if t < len(boxes) else None
            area_t = areas[t] if t < len(areas) else None

            # If bbox/area is missing, try to compute from RLE segmentation
            if (bbox_t is None or area_t is None) and mask_utils is not None and isinstance(seg_t, dict) and "counts" in seg_t:
                try:
                    if bbox_t is None:
                        bb = mask_utils.toBbox(seg_t)  # [x,y,w,h]
                        bbox_t = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
                    if area_t is None:
                        area_t = float(mask_utils.area(seg_t))
                except Exception:
                    pass

            # Final fallbacks to satisfy detectron2 annotation schema
            if bbox_t is None:
                bbox_t = [0.0, 0.0, 0.0, 0.0]
            if area_t is None:
                area_t = 0.0

            per_frame = {
                "id": int(inst.get("id", 0)),
                "category_id": int(inst.get("category_id", 0)),
                "iscrowd": int(inst.get("iscrowd", 0)),
                "segmentation": seg_t,
                "bbox": bbox_t,
                "bbox_mode": BoxMode.XYWH_ABS,
                "area": area_t,
            }
            frame_anns[t].append(per_frame)

    return frame_anns

def load_ytvis_json(json_file: str, image_root: str, dataset_name: Optional[str] = None):
    """Load a YouTube-VIS style JSON.

    The resulting dicts follow the conventions used by SeqFormer/VNext:
      - each dataset dict corresponds to one video
      - fields: file_names (list of absolute paths), height, width, video_id, annotations
      - annotations is a per-frame list (length T); each element is a list of per-instance dicts for that frame
      - instance-level lists (segmentations/bboxes/areas with length T) are stored under `instances`
    """

    with PathManager.open(json_file, "r") as f:
        data = json.load(f)

    videos = {int(v["id"]): v for v in data.get("videos", [])}
    anns_by_video: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_video[int(ann["video_id"])].append(ann)

    # categories / metadata
    categories = data.get("categories", [])
    id_map = _build_category_mapping(categories) if categories else {}
    thing_classes = [c.get("name", str(c.get("id"))) for c in sorted(categories, key=lambda x: int(x["id"]))]

    dataset_dicts = []
    for vid, v in videos.items():
        file_names = v.get("file_names")
        if file_names is None:
            # fallback (non-standard json): allow a caller-provided mapping
            raise KeyError(
                f"JSON videos[{vid}] missing 'file_names'. Please run tools/convert_videocutler_png_to_json.py with --img_root."
            )

        abs_files = [
            os.path.normpath(fn) if os.path.isabs(fn) else os.path.normpath(os.path.join(image_root, fn))
            for fn in file_names
        ]
        record = {
            "file_names": abs_files,
            "height": int(v.get("height")) if v.get("height") is not None else None,
            "width": int(v.get("width")) if v.get("width") is not None else None,
            "length": int(v.get("length")) if v.get("length") is not None else len(abs_files),
            "video_id": vid,
            "dataset_name": dataset_name,
        }

        anns = []
        for ann in anns_by_video.get(vid, []):
            cat_id = int(ann.get("category_id", 1))
            anns.append(
                {
                    "id": int(ann.get("id", 0)),
                    "category_id": id_map.get(cat_id, cat_id),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                    "segmentations": ann.get("segmentations"),
                    "bboxes": ann.get("bboxes"),
                    "areas": ann.get("areas"),
                }
            )
        record["instances"] = anns
        record["annotations"] = _instances_to_per_frame_anns(anns, len(abs_files))
        dataset_dicts.append(record)

    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        if categories:
            meta.thing_dataset_id_to_contiguous_id = id_map
            meta.thing_classes = thing_classes

    return dataset_dicts


def register_ytvis_like_dataset(name, json_file, image_root, evaluator_type="ytvis"):
    from detectron2.data import DatasetCatalog, MetadataCatalog

    if name not in DatasetCatalog.list():
        DatasetCatalog.register(
            name, lambda jf=json_file, ir=image_root, dn=name: load_ytvis_json(jf, ir, dn)
        )
    else:
        print(f"[wsovvis] WARNING: dataset '{name}' already registered; overwrite metadata.")

    # ✅ 永远写 metadata（关键）
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type=evaluator_type,
    )
