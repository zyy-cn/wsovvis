from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, TextIO, Tuple

import numpy as np


@dataclass(frozen=True)
class FrameSample:
    clip_id: str
    frame_index: int
    image_path: Path


@dataclass(frozen=True)
class FrameGeometry:
    orig_h: int
    orig_w: int
    resized_h: int
    resized_w: int
    padded_h: int
    padded_w: int
    scale_y: float
    scale_x: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    patch_size: int
    grid_h: int
    grid_w: int


def frame_sort_key(sample: FrameSample) -> Tuple[str, int]:
    return (str(sample.clip_id), int(sample.frame_index))


def sorted_samples(samples: Iterable[FrameSample]) -> List[FrameSample]:
    return sorted(list(samples), key=frame_sort_key)


def _safe_clip_id(clip_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(clip_id))


def keep_aspect_pad_geometry(
    orig_h: int,
    orig_w: int,
    *,
    resize_short_side: int,
    max_long_side: int,
    pad_to_multiple: int,
    patch_size: int = 14,
) -> FrameGeometry:
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError("original image shape must be positive")
    if resize_short_side <= 0 or max_long_side <= 0 or pad_to_multiple <= 0:
        raise ValueError("resize and padding parameters must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    short = float(min(orig_h, orig_w))
    long = float(max(orig_h, orig_w))
    scale = float(resize_short_side) / short
    if long * scale > float(max_long_side):
        scale = float(max_long_side) / long

    resized_h = max(1, int(round(float(orig_h) * scale)))
    resized_w = max(1, int(round(float(orig_w) * scale)))
    if max(resized_h, resized_w) > int(max_long_side):
        shrink = float(max_long_side) / float(max(resized_h, resized_w))
        resized_h = max(1, int(math.floor(float(resized_h) * shrink)))
        resized_w = max(1, int(math.floor(float(resized_w) * shrink)))

    padded_h = int(math.ceil(float(resized_h) / float(pad_to_multiple)) * pad_to_multiple)
    padded_w = int(math.ceil(float(resized_w) / float(pad_to_multiple)) * pad_to_multiple)
    pad_right = int(padded_w - resized_w)
    pad_bottom = int(padded_h - resized_h)
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError("negative padding is not allowed")

    if padded_h % patch_size != 0 or padded_w % patch_size != 0:
        raise ValueError("padded dimensions must be divisible by patch_size")

    return FrameGeometry(
        orig_h=int(orig_h),
        orig_w=int(orig_w),
        resized_h=int(resized_h),
        resized_w=int(resized_w),
        padded_h=int(padded_h),
        padded_w=int(padded_w),
        scale_y=float(resized_h) / float(orig_h),
        scale_x=float(resized_w) / float(orig_w),
        pad_left=0,
        pad_top=0,
        pad_right=int(pad_right),
        pad_bottom=int(pad_bottom),
        patch_size=int(patch_size),
        grid_h=int(padded_h // patch_size),
        grid_w=int(padded_w // patch_size),
    )


def build_valid_token_mask(geometry: FrameGeometry) -> np.ndarray:
    valid_h = int(math.ceil(float(geometry.resized_h) / float(geometry.patch_size)))
    valid_w = int(math.ceil(float(geometry.resized_w) / float(geometry.patch_size)))
    valid_h = max(0, min(valid_h, int(geometry.grid_h)))
    valid_w = max(0, min(valid_w, int(geometry.grid_w)))
    mask = np.zeros((int(geometry.grid_h), int(geometry.grid_w)), dtype=np.uint8)
    if valid_h > 0 and valid_w > 0:
        mask[:valid_h, :valid_w] = 1
    return mask


def reconstruct_valid_token_mask_from_geometry(geometry: FrameGeometry | Dict[str, Any]) -> np.ndarray:
    if isinstance(geometry, FrameGeometry):
        geom = geometry
    else:
        geom = FrameGeometry(
            orig_h=int(geometry["orig_h"]),
            orig_w=int(geometry["orig_w"]),
            resized_h=int(geometry["resized_h"]),
            resized_w=int(geometry["resized_w"]),
            padded_h=int(geometry["padded_h"]),
            padded_w=int(geometry["padded_w"]),
            scale_y=float(geometry["scale_y"]),
            scale_x=float(geometry["scale_x"]),
            pad_left=int(geometry["pad_left"]),
            pad_top=int(geometry["pad_top"]),
            pad_right=int(geometry["pad_right"]),
            pad_bottom=int(geometry["pad_bottom"]),
            patch_size=int(geometry["patch_size"]),
            grid_h=int(geometry["grid_h"]),
            grid_w=int(geometry["grid_w"]),
        )
    return build_valid_token_mask(geom)


def _clip_payload_rel(clip_id: str) -> str:
    return f"payload/clip_{_safe_clip_id(clip_id)}_feats.npz"


def _load_feature_slot(payload: np.lib.npyio.NpzFile, slot: int) -> np.ndarray:
    slot_key = f"slot_{slot}"
    if slot_key in payload.files:
        return np.asarray(payload[slot_key], dtype=np.float32)
    if "feats" in payload.files:
        feats = np.asarray(payload["feats"])
        if feats.ndim == 0:
            raise ValueError("invalid feature payload")
        if feats.dtype == object:
            return np.asarray(feats[slot], dtype=np.float32)
        if slot < 0 or slot >= int(feats.shape[0]):
            raise IndexError("feat_path slot out of range")
        return np.asarray(feats[slot], dtype=np.float32)
    raise KeyError(slot_key)


def write_frame_records_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(list(records), key=lambda item: (str(item["clip_id"]), int(item["frame_index"])))
    with path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_frame_geom_records_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(list(records), key=lambda item: (str(item["clip_id"]), int(item["frame_index"])))
    with path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class FrameBankStreamWriter:
    def __init__(self, artifact_parent_dir: Path, frame_records_path: Path, frame_geom_records_path: Path) -> None:
        self.artifact_parent_dir = artifact_parent_dir
        self.payload_dir = artifact_parent_dir / "payload"
        self.frame_records_path = frame_records_path
        self.frame_geom_records_path = frame_geom_records_path
        self._frame_records_fh: TextIO | None = None
        self._frame_geom_records_fh: TextIO | None = None
        self._clip_id: str | None = None
        self._clip_features: List[np.ndarray] = []
        self._geom_slot = 0

    def __enter__(self) -> "FrameBankStreamWriter":
        self.artifact_parent_dir.mkdir(parents=True, exist_ok=True)
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self.frame_records_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_geom_records_path.parent.mkdir(parents=True, exist_ok=True)
        self._frame_records_fh = self.frame_records_path.open("w", encoding="utf-8")
        self._frame_geom_records_fh = self.frame_geom_records_path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._flush_clip_payload()
        if self._frame_records_fh is not None:
            self._frame_records_fh.close()
        if self._frame_geom_records_fh is not None:
            self._frame_geom_records_fh.close()
        self._frame_records_fh = None
        self._frame_geom_records_fh = None
        self._clip_id = None
        self._clip_features = []
        self._geom_slot = 0

    def _flush_clip_payload(self) -> None:
        if self._clip_id is None or not self._clip_features:
            return
        payload_rel = _clip_payload_rel(self._clip_id)
        payload_path = self.artifact_parent_dir / payload_rel
        payload = {f"slot_{idx}": np.asarray(feature, dtype=np.float16) for idx, feature in enumerate(self._clip_features)}
        np.savez_compressed(payload_path, **payload)
        self._clip_features = []

    def write_sample(
        self,
        sample: FrameSample,
        feature: np.ndarray,
        geometry: FrameGeometry,
        valid_mask: np.ndarray,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        if self._frame_records_fh is None or self._frame_geom_records_fh is None:
            raise RuntimeError("FrameBankStreamWriter must be used as a context manager")

        clip_id = str(sample.clip_id)
        frame_index = int(sample.frame_index)
        clip_safe = _safe_clip_id(clip_id)

        if self._clip_id is None:
            self._clip_id = clip_id
        elif self._clip_id != clip_id:
            self._flush_clip_payload()
            self._clip_id = clip_id

        feature = np.asarray(feature, dtype=np.float16)
        if feature.ndim != 2:
            raise ValueError("feature tensor must be [num_tokens, dim]")
        slot = len(self._clip_features)
        self._clip_features.append(feature)

        valid_mask = np.asarray(valid_mask, dtype=np.uint8)
        expected_mask = reconstruct_valid_token_mask_from_geometry(geometry)
        if valid_mask.shape != expected_mask.shape or not np.array_equal(valid_mask, expected_mask):
            raise ValueError("valid token mask does not match geometry")
        feat_payload_rel = _clip_payload_rel(clip_id)

        frame_record: Dict[str, object] = {
            "clip_id": clip_id,
            "frame_index": frame_index,
            "feat_path": f"{feat_payload_rel}#{slot}",
            "path_base_mode": "artifact_parent_dir",
        }
        geom_record: Dict[str, object] = {
            "clip_id": clip_id,
            "frame_index": frame_index,
            "orig_h": int(geometry.orig_h),
            "orig_w": int(geometry.orig_w),
            "resized_h": int(geometry.resized_h),
            "resized_w": int(geometry.resized_w),
            "padded_h": int(geometry.padded_h),
            "padded_w": int(geometry.padded_w),
            "scale_y": float(geometry.scale_y),
            "scale_x": float(geometry.scale_x),
            "pad_left": int(geometry.pad_left),
            "pad_top": int(geometry.pad_top),
            "pad_right": int(geometry.pad_right),
            "pad_bottom": int(geometry.pad_bottom),
            "patch_size": int(geometry.patch_size),
            "grid_h": int(geometry.grid_h),
            "grid_w": int(geometry.grid_w),
            "valid_token_mask_path": f"frame_geom_records.jsonl#{self._geom_slot}",
            "path_base_mode": "artifact_parent_dir",
        }
        self._frame_records_fh.write(json.dumps(frame_record, ensure_ascii=False) + "\n")
        self._frame_geom_records_fh.write(json.dumps(geom_record, ensure_ascii=False) + "\n")
        self._geom_slot += 1
        return frame_record, geom_record

    def close(self) -> None:
        self._flush_clip_payload()


def write_frame_bank(
    artifact_parent_dir: Path,
    samples: Iterable[FrameSample],
    feature_vectors: Sequence[np.ndarray] | np.ndarray,
    frame_geometries: Sequence[FrameGeometry],
    valid_token_masks: Sequence[np.ndarray],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    ordered_samples = sorted_samples(samples)
    if len(ordered_samples) != len(frame_geometries) or len(ordered_samples) != len(valid_token_masks):
        raise ValueError("sample/geometry/mask length mismatch")

    if isinstance(feature_vectors, np.ndarray):
        vector_list = [np.asarray(feature_vectors[idx]) for idx in range(int(feature_vectors.shape[0]))]
    else:
        vector_list = [np.asarray(item) for item in feature_vectors]
    if len(ordered_samples) != len(vector_list):
        raise ValueError("sample/feature length mismatch")

    artifact_parent_dir.mkdir(parents=True, exist_ok=True)

    frame_records: List[Dict[str, object]] = []
    geom_records: List[Dict[str, object]] = []
    current_clip_id: str | None = None
    current_features: List[np.ndarray] = []
    geom_slot = 0

    def flush_clip_payload(clip_id: str | None) -> None:
        nonlocal current_clip_id, current_features
        if clip_id is None or not current_features:
            return
        payload_rel = _clip_payload_rel(clip_id)
        payload_path = artifact_parent_dir / payload_rel
        payload = {f"slot_{idx}": np.asarray(feature, dtype=np.float16) for idx, feature in enumerate(current_features)}
        np.savez_compressed(payload_path, **payload)
        current_features = []

    for sample, feature, geom, valid_mask in zip(ordered_samples, vector_list, frame_geometries, valid_token_masks):
        clip_id = str(sample.clip_id)
        frame_index = int(sample.frame_index)
        if current_clip_id is None:
            current_clip_id = clip_id
        elif current_clip_id != clip_id:
            flush_clip_payload(current_clip_id)
            current_clip_id = clip_id

        feature = np.asarray(feature, dtype=np.float16)
        if feature.ndim != 2:
            raise ValueError("feature tensor must be [num_tokens, dim]")
        slot = len(current_features)
        current_features.append(feature)

        valid_mask = np.asarray(valid_mask, dtype=np.uint8)
        expected_mask = reconstruct_valid_token_mask_from_geometry(geom)
        if valid_mask.shape != expected_mask.shape or not np.array_equal(valid_mask, expected_mask):
            raise ValueError("valid token mask does not match geometry")
        feat_payload_rel = _clip_payload_rel(clip_id)

        frame_records.append(
            {
                "clip_id": clip_id,
                "frame_index": frame_index,
                "feat_path": f"{feat_payload_rel}#{slot}",
                "path_base_mode": "artifact_parent_dir",
            }
        )
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
                "valid_token_mask_path": f"frame_geom_records.jsonl#{geom_slot}",
                "path_base_mode": "artifact_parent_dir",
            }
        )
        geom_slot += 1
    flush_clip_payload(current_clip_id)
    return frame_records, geom_records


def parse_feat_path(feat_path: str) -> Tuple[Path, int]:
    if "#" not in feat_path:
        raise ValueError("feat_path missing #slot suffix")
    rel, slot = feat_path.rsplit("#", 1)
    if not rel or not slot.isdigit():
        raise ValueError("feat_path must follow <relative_file_path>#<slot>")
    return Path(rel), int(slot)


def read_feature_vector(artifact_parent_dir: Path, feat_path: str) -> np.ndarray:
    rel_path, slot = parse_feat_path(feat_path)
    with np.load(artifact_parent_dir / rel_path, allow_pickle=True) as payload:
        return _load_feature_slot(payload, slot)


def _read_jsonl_record_slot(path: Path, slot: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == slot:
                return json.loads(line)
    raise IndexError("jsonl slot out of range")


def read_valid_token_mask(artifact_parent_dir: Path, mask_path: str) -> np.ndarray:
    rel_path, slot = parse_feat_path(mask_path)
    resolved = artifact_parent_dir / rel_path
    if resolved.suffix == ".jsonl":
        geom_record = _read_jsonl_record_slot(resolved, slot)
        return reconstruct_valid_token_mask_from_geometry(geom_record)
    with np.load(resolved, allow_pickle=True) as payload:
        if "masks" not in payload.files:
            raise KeyError("masks")
        masks = np.asarray(payload["masks"])
        if slot < 0 or slot >= int(masks.shape[0]):
            raise IndexError("valid_token_mask_path slot out of range")
        return np.asarray(masks[slot], dtype=np.uint8)
