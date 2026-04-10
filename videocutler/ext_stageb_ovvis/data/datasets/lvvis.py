from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


DATASET_NAMES = ("lvvis_train_base", "lvvis_val")
ROOT_ENV_VAR = "WSOVVIS_LVVIS_ROOT"
ROOT_FALLBACK = "videocutler/datasets/LV-VIS"
REQUIRED_CHILDREN = ("annotations", "train", "val")

_SPLITS: Dict[str, Tuple[str, str, str]] = {
    "lvvis_train_base": ("train", "annotations/train_instances.json", "train"),
    "lvvis_val": ("val", "annotations/val_instances.json", "val"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_lvvis_root() -> Path:
    env_value = os.environ.get(ROOT_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (_repo_root() / ROOT_FALLBACK).resolve()


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _category_metadata(json_file: Path) -> Dict[str, Any]:
    if not json_file.exists():
        return {
            "thing_classes": [],
            "thing_dataset_id_to_contiguous_id": {},
        }
    data = _load_json(json_file)
    categories = sorted(data.get("categories", []), key=lambda item: item["id"])
    thing_classes = [_normalize_name(str(item.get("name", item["id"]))) for item in categories]
    id_map = {int(item["id"]): index for index, item in enumerate(categories)}
    return {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": id_map,
    }


def _annotations_by_video(annotations: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for annotation in annotations:
        video_id = int(annotation["video_id"])
        grouped.setdefault(video_id, []).append(annotation)
    return grouped


def load_lvvis_json(json_file: str, image_root: str, dataset_name: str) -> List[Dict[str, Any]]:
    data = _load_json(Path(json_file))
    videos = sorted(data.get("videos", []), key=lambda item: item["id"])
    annotations_by_video = _annotations_by_video(data.get("annotations", []))
    id_map = MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id

    records: List[Dict[str, Any]] = []
    for video in videos:
        video_id = int(video["id"])
        file_names = video.get("file_names") or video.get("filenames") or []
        length = int(video.get("length", len(file_names)))
        record: Dict[str, Any] = {
            "file_names": [str(Path(image_root) / name) for name in file_names],
            "height": int(video["height"]),
            "width": int(video["width"]),
            "length": length,
            "video_id": video_id,
        }
        video_objects: List[List[Dict[str, Any]]] = [[] for _ in range(length)]
        for annotation in annotations_by_video.get(video_id, []):
            bboxes = annotation.get("bboxes") or []
            segmentations = annotation.get("segmentations") or []
            for frame_index in range(length):
                bbox = bboxes[frame_index] if frame_index < len(bboxes) else None
                segmentation = segmentations[frame_index] if frame_index < len(segmentations) else None
                if not bbox or not segmentation:
                    continue
                category_id = int(annotation["category_id"])
                video_objects[frame_index].append(
                    {
                        "id": annotation.get("id"),
                        "iscrowd": int(annotation.get("iscrowd", 0)),
                        "category_id": id_map.get(category_id, category_id),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": segmentation,
                        "raw_category_id": category_id,
                    }
                )
        record["annotations"] = video_objects
        records.append(record)
    return records


def register_lvvis_instances(name: str, json_file: Path, image_root: Path) -> None:
    if name in DatasetCatalog.list():
        return
    metadata = _category_metadata(json_file)
    DatasetCatalog.register(
        name,
        lambda json_file=str(json_file), image_root=str(image_root), name=name: load_lvvis_json(
            json_file, image_root, name
        ),
    )
    MetadataCatalog.get(name).set(
        json_file=str(json_file),
        image_root=str(image_root),
        evaluator_type="lvvis",
        root_binding_mode="env_first_repo_fallback",
        root_env_var=ROOT_ENV_VAR,
        root_required_children=list(REQUIRED_CHILDREN),
        **metadata,
    )


def register_all_lvvis() -> None:
    root = resolve_lvvis_root()
    for name, (_, annotation_rel, image_rel) in _SPLITS.items():
        register_lvvis_instances(name, root / annotation_rel, root / image_rel)


register_all_lvvis()

