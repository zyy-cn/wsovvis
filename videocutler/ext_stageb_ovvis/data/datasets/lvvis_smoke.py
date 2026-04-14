from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT_ENV_VAR = "WSOVVIS_LVVIS_ROOT"
ROOT_FALLBACK = "videocutler/datasets/LV-VIS"
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


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _subset_split_name(dataset_name: str, smoke_num_videos: int) -> str:
    return f"{dataset_name}_smoke_first{int(smoke_num_videos)}"


def load_lvvis_smoke_subset_data(
    dataset_name: str,
    smoke_num_videos: int,
) -> Tuple[str, Dict[str, Any], Path]:
    if dataset_name not in _SPLITS:
        raise ValueError(f"unsupported LV-VIS dataset: {dataset_name}")
    if smoke_num_videos <= 0:
        raise ValueError("smoke_num_videos must be positive")

    split_tag, annotation_rel, image_rel = _SPLITS[dataset_name]
    lvvis_root = resolve_lvvis_root()
    source_json = lvvis_root / annotation_rel
    source_data = _load_json(source_json)
    videos: List[Dict[str, Any]] = sorted(source_data.get("videos", []), key=lambda item: item["id"])[: smoke_num_videos]
    video_ids = {int(video["id"]) for video in videos}
    annotations = [
        annotation
        for annotation in source_data.get("annotations", [])
        if int(annotation.get("video_id", -1)) in video_ids
    ]
    subset_data = dict(source_data)
    subset_data["videos"] = videos
    subset_data["annotations"] = annotations
    smoke_name = _subset_split_name(dataset_name, smoke_num_videos)
    return smoke_name, subset_data, lvvis_root / image_rel


def register_lvvis_smoke_subset(
    dataset_name: str,
    smoke_num_videos: int,
    smoke_root: Path,
) -> Tuple[str, Path, Path]:
    from .lvvis import register_lvvis_instances

    split_tag, _, _ = _SPLITS[dataset_name]
    smoke_name, subset_data, image_root = load_lvvis_smoke_subset_data(dataset_name, smoke_num_videos)
    smoke_root = smoke_root.expanduser().resolve()
    smoke_annotation_path = smoke_root / "annotations" / f"{split_tag}_instances_smoke_first{smoke_num_videos}.json"
    smoke_annotation_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_annotation_path.write_text(
        json.dumps(subset_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    register_lvvis_instances(smoke_name, smoke_annotation_path, image_root)
    return smoke_name, smoke_annotation_path, image_root
