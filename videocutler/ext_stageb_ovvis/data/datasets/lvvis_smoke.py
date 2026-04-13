from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .lvvis import _SPLITS, register_lvvis_instances, resolve_lvvis_root


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _subset_split_name(dataset_name: str, smoke_num_videos: int) -> str:
    return f"{dataset_name}_smoke_first{int(smoke_num_videos)}"


def register_lvvis_smoke_subset(
    dataset_name: str,
    smoke_num_videos: int,
    smoke_root: Path,
) -> Tuple[str, Path, Path]:
    if dataset_name not in _SPLITS:
        raise ValueError(f"unsupported LV-VIS dataset: {dataset_name}")
    if smoke_num_videos <= 0:
        raise ValueError("smoke_num_videos must be positive")

    split_tag, annotation_rel, image_rel = _SPLITS[dataset_name]
    lvvis_root = resolve_lvvis_root()
    source_json = lvvis_root / annotation_rel
    source_data = _load_json(source_json)
    videos = sorted(source_data.get("videos", []), key=lambda item: item["id"])[: smoke_num_videos]
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
    smoke_root = smoke_root.expanduser().resolve()
    smoke_annotation_path = smoke_root / "annotations" / f"{split_tag}_instances_smoke_first{smoke_num_videos}.json"
    smoke_annotation_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_annotation_path.write_text(
        json.dumps(subset_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    register_lvvis_instances(smoke_name, smoke_annotation_path, lvvis_root / image_rel)
    return smoke_name, smoke_annotation_path, lvvis_root / image_rel
