from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class FrameSample:
    clip_id: str
    frame_index: int
    image_path: Path


def frame_sort_key(sample: FrameSample) -> Tuple[str, int]:
    return (str(sample.clip_id), int(sample.frame_index))


def sorted_samples(samples: Iterable[FrameSample]) -> List[FrameSample]:
    return sorted(list(samples), key=frame_sort_key)


def _safe_clip_id(clip_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(clip_id))


def write_frame_bank(
    artifact_parent_dir: Path,
    samples: Iterable[FrameSample],
    feature_vectors: np.ndarray,
) -> List[Dict[str, object]]:
    ordered_samples = sorted_samples(samples)
    if len(ordered_samples) != int(feature_vectors.shape[0]):
        raise ValueError("sample/feature length mismatch")

    artifact_parent_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = artifact_parent_dir / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[np.ndarray]] = {}
    records: List[Dict[str, object]] = []
    for sample, feature in zip(ordered_samples, feature_vectors):
        clip_id = str(sample.clip_id)
        clip_vectors = grouped.setdefault(clip_id, [])
        slot = len(clip_vectors)
        clip_vectors.append(np.asarray(feature, dtype=np.float32))
        rel_payload = f"payload/frame_feats_{_safe_clip_id(clip_id)}.npz"
        records.append(
            {
                "clip_id": clip_id,
                "frame_index": int(sample.frame_index),
                "feat_path": f"{rel_payload}#{slot}",
                "path_base_mode": "artifact_parent_dir",
            }
        )

    for clip_id, vectors in grouped.items():
        payload_path = payload_dir / f"frame_feats_{_safe_clip_id(clip_id)}.npz"
        matrix = np.stack(vectors, axis=0).astype(np.float32, copy=False)
        np.savez_compressed(payload_path, feats=matrix)

    return records


def write_frame_records_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(list(records), key=lambda item: (str(item["clip_id"]), int(item["frame_index"])))
    with path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def parse_feat_path(feat_path: str) -> Tuple[Path, int]:
    if "#" not in feat_path:
        raise ValueError("feat_path missing #slot suffix")
    rel, slot = feat_path.rsplit("#", 1)
    if not rel or not slot.isdigit():
        raise ValueError("feat_path must follow <relative_file_path>#<slot>")
    return Path(rel), int(slot)


def read_feature_vector(artifact_parent_dir: Path, feat_path: str) -> np.ndarray:
    rel_path, slot = parse_feat_path(feat_path)
    payload = np.load(artifact_parent_dir / rel_path)
    feats = np.asarray(payload["feats"])
    if slot < 0 or slot >= int(feats.shape[0]):
        raise IndexError("feat_path slot out of range")
    return np.asarray(feats[slot], dtype=np.float32)
