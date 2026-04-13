from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple


Record = Dict[str, Any]
GENERATOR_TAG = "videocutler_r50_native"
GENERATOR_CFG_PATH = "videocutler/configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml"


def trajectory_sort_key(record: Record) -> Tuple[float, List[int], List[Any], int]:
    masks = record.get("masks_rle", [])
    masks_key = [
        json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for item in list(masks)
    ]
    return (
        -float(record["pred_score"]),
        list(record.get("frame_indices", [])),
        masks_key,
        int(record.get("rank_in_clip", 0)),
    )


def sort_clip_trajectories(records: Iterable[Record]) -> List[Record]:
    return sorted(records, key=trajectory_sort_key)


def build_trajectory_id(dataset_name: str, video_id: int, rank_in_clip: int) -> str:
    return f"{GENERATOR_TAG}:{dataset_name}:{int(video_id)}:{int(rank_in_clip):06d}"


def materialize_trajectory_bank(records: Iterable[Record]) -> List[Record]:
    materialized: List[Record] = []
    for record in sort_clip_trajectories(records):
        updated = dict(record)
        updated["generator_tag"] = GENERATOR_TAG
        updated["generator_cfg_path"] = GENERATOR_CFG_PATH
        updated["trajectory_id"] = build_trajectory_id(
            str(updated["dataset_name"]),
            int(updated["video_id"]),
            int(updated["rank_in_clip"]),
        )
        materialized.append(updated)
    return materialized


def validate_trajectory_record(record: Record) -> List[str]:
    errors: List[str] = []
    if record.get("generator_tag") != GENERATOR_TAG:
        errors.append("generator_tag")
    if record.get("generator_cfg_path") != GENERATOR_CFG_PATH:
        errors.append("generator_cfg_path")
    frame_indices = list(record.get("frame_indices", []))
    masks_rle = list(record.get("masks_rle", []))
    boxes_xyxy = list(record.get("boxes_xyxy", []))
    if not (len(frame_indices) == len(masks_rle) == len(boxes_xyxy)):
        errors.append("aligned_lengths")
    image_size = record.get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        errors.append("image_size")
    else:
        try:
            if int(image_size[0]) <= 0 or int(image_size[1]) <= 0:
                errors.append("image_size")
        except Exception:
            errors.append("image_size")
    expected_id = build_trajectory_id(
        str(record["dataset_name"]),
        int(record["video_id"]),
        int(record["rank_in_clip"]),
    )
    if record.get("trajectory_id") != expected_id:
        errors.append("trajectory_id")
    return errors
