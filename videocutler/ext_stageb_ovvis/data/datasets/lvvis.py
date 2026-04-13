from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


DATASET_NAMES = ("lvvis_train_base", "lvvis_val")
ROOT_ENV_VAR = "WSOVVIS_LVVIS_ROOT"
SANITIZED_ROOT_ENV_VAR = "WSOVVIS_LVVIS_SANITIZED_ROOT"
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


def _resolve_image_root(image_root: Path, json_data: Dict[str, Any]) -> Path:
    if not json_data.get("videos"):
        return image_root
    file_names = (
        json_data["videos"][0].get("file_names")
        or json_data["videos"][0].get("filenames")
        or []
    )
    if not file_names:
        return image_root
    first_rel = Path(str(file_names[0]))
    direct_candidate = image_root / first_rel
    if direct_candidate.exists():
        return image_root
    jpeg_candidate = image_root / "JPEGImages" / first_rel
    if jpeg_candidate.exists():
        return image_root / "JPEGImages"
    return image_root


def _resolve_frame_path(image_root: Path, file_name: str) -> Path:
    relative = Path(str(file_name))
    candidates = [
        image_root / relative,
        image_root / "JPEGImages" / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _probe_image_size(image_path: Path) -> Tuple[int, int]:
    try:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            height, width = image.shape[:2]
            return int(width), int(height)
    except Exception:
        pass

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
            return int(width), int(height)
    except Exception as exc:
        raise FileNotFoundError(f"unable to probe image size: {image_path}") from exc


def _collect_frame_sizes(frame_paths: List[Path]) -> Tuple[Tuple[int, int], List[Tuple[int, int]], bool]:
    canonical_size: Tuple[int, int] = (0, 0)
    variants: List[Tuple[int, int]] = []
    seen = set()
    inconsistent = False

    sample_indices = [0]
    if len(frame_paths) > 2:
        sample_indices.append(len(frame_paths) // 2)
    if len(frame_paths) > 1:
        sample_indices.append(len(frame_paths) - 1)

    for index in sorted(set(sample_indices)):
        frame_path = frame_paths[index]
        width, height = _probe_image_size(frame_path)
        size = (int(width), int(height))
        if canonical_size == (0, 0):
            canonical_size = size
        if size not in seen and len(variants) < 5:
            variants.append(size)
        if size not in seen and seen:
            inconsistent = True
        seen.add(size)

    return canonical_size, variants, inconsistent


def _sanitize_frame_sequence(
    frame_paths: List[Path],
    *,
    dataset_name: str,
    video_id: int,
    cache_root: Path,
    target_size: Tuple[int, int],
) -> List[str]:
    try:
        from PIL import Image, ImageOps
    except Exception as exc:  # pragma: no cover - remote runtime has Pillow
        raise RuntimeError("Pillow is required for LV-VIS frame normalization") from exc

    target_width, target_height = target_size
    video_cache_root = cache_root / dataset_name / f"video_{video_id:06d}" / f"{target_width}x{target_height}"
    video_cache_root.mkdir(parents=True, exist_ok=True)

    sanitized_paths: List[str] = []
    for frame_path in frame_paths:
        relative_frame = Path(frame_path.name)
        cached_path = video_cache_root / relative_frame
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        if not cached_path.exists():
            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                if image.size != (target_width, target_height):
                    image = ImageOps.pad(
                        image,
                        (target_width, target_height),
                        method=Image.Resampling.BILINEAR,
                        color=(0, 0, 0),
                        centering=(0.5, 0.5),
                    )
                image.save(cached_path, format="JPEG", quality=95, optimize=True)
        sanitized_paths.append(str(cached_path))
    return sanitized_paths


def _sanitize_video_record(
    video: Dict[str, Any],
    *,
    image_root: Path,
    dataset_name: str,
) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    file_names = video.get("file_names") or video.get("filenames") or []
    length = int(video.get("length", len(file_names)))
    if not file_names:
        raise ValueError(f"video {video.get('id')} in {dataset_name} has no frame file names")

    frame_paths = [_resolve_frame_path(image_root, str(name)) for name in file_names]
    first_frame = frame_paths[0]
    actual_width, actual_height = _probe_image_size(first_frame)
    canonical_size, size_variants, inconsistent = _collect_frame_sizes(frame_paths)
    if inconsistent:
        actual_width, actual_height = canonical_size
    annot_width = int(video.get("width", actual_width))
    annot_height = int(video.get("height", actual_height))
    mismatch = annot_width != actual_width or annot_height != actual_height or inconsistent

    sanitized_paths = [str(frame_path) for frame_path in frame_paths]
    if inconsistent:
        sanitized_root = Path(
            os.environ.get(
                SANITIZED_ROOT_ENV_VAR,
                str(image_root.parent / "_lvvis_sanitized_frames"),
            )
        )
        sanitized_paths = _sanitize_frame_sequence(
            frame_paths,
            dataset_name=dataset_name,
            video_id=int(video["id"]),
            cache_root=sanitized_root,
            target_size=(actual_width, actual_height),
        )

    record: Dict[str, Any] = {
        "file_names": sanitized_paths,
        "height": actual_height,
        "width": actual_width,
        "length": length,
        "video_id": int(video["id"]),
        "orig_annot_height": annot_height,
        "orig_annot_width": annot_width,
        "size_mismatch_fixed": bool(mismatch),
        "frame_size_inconsistent": bool(inconsistent),
        "frame_size_variants": [list(size) for size in size_variants],
    }
    mismatch_summary = {
        "video_id": int(video["id"]),
        "first_frame": str(first_frame),
        "orig_annot_height": annot_height,
        "orig_annot_width": annot_width,
        "actual_height": actual_height,
        "actual_width": actual_width,
        "frame_size_inconsistent": bool(inconsistent),
        "frame_size_variants": [list(size) for size in size_variants],
    }
    return record, mismatch, mismatch_summary


def load_lvvis_json_with_stats(
    json_file: str,
    image_root: str,
    dataset_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = _load_json(Path(json_file))
    videos = sorted(data.get("videos", []), key=lambda item: item["id"])
    annotations_by_video = _annotations_by_video(data.get("annotations", []))
    id_map = MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id
    resolved_image_root = _resolve_image_root(Path(image_root), data)

    records: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    sanitized_videos = 0
    for video in videos:
        record, mismatch, mismatch_summary = _sanitize_video_record(
            video,
            image_root=resolved_image_root,
            dataset_name=dataset_name,
        )
        if mismatch:
            sanitized_videos += 1
            if len(mismatches) < 3:
                mismatches.append(mismatch_summary)
        video_id = record["video_id"]
        length = record["length"]
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

    stats = {
        "dataset_name": dataset_name,
        "videos_checked": len(videos),
        "videos_sanitized": sanitized_videos,
        "mismatch_samples": mismatches,
        "image_root": str(resolved_image_root),
        "json_file": str(json_file),
    }
    return records, stats


def summarize_lvvis_sanitization(json_file: str, image_root: str, dataset_name: str) -> Dict[str, Any]:
    _, stats = load_lvvis_json_with_stats(json_file, image_root, dataset_name)
    return stats


def load_lvvis_json(json_file: str, image_root: str, dataset_name: str) -> List[Dict[str, Any]]:
    records, _ = load_lvvis_json_with_stats(json_file, image_root, dataset_name)
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
