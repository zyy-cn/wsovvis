from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

from videocutler.ext_stageb_ovvis.algorithms.prealign import PrealignConfig, train_prealign
from videocutler.ext_stageb_ovvis.algorithms.soft_em import SoftEMConfig, run_soft_em
from videocutler.ext_stageb_ovvis.audit.projector_quality_audit import build_projector_quality_rows
from videocutler.ext_stageb_ovvis.audit.trajectory_gt_audit import _extract_gt_available, _extract_gt_class_id, load_gt_sidecar_lookup
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)
from videocutler.run_stageb_train_softem import resolve_em_subiterations


Record = Dict[str, Any]

TIER2_CLASS_COUNT = 10
TIER2_CLIP_COUNT = 32
TIER2_DROP_RATIOS = (0.25, 0.50)
TIER2_TARGET_CLASS_IDS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
TIER2_DISTRACTOR_CLASS_IDS = [21, 23, 25, 27]
SEMIOPEN_DISTRACTOR_COUNT = 4
STRESSA_SPLIT_IDS = ("split_0", "split_1", "split_2")
STRESSA_TARGET_CLASS_IDS = list(TIER2_TARGET_CLASS_IDS)
STRESSA_DISTRACTOR_CLASS_IDS = [21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51]
STRESSA_CLASS_COUNT = 10
STRESSA_CLIP_COUNT = 32
STRESSA_DROP_RATIOS = (0.25, 0.50)
STRESSA_DISTRACTOR_COUNTS = (8, 12)
STRESSA_PROTOCOL_TYPES = ("ratio_drop", "keep_exactly_one_observed_per_clip")
MAINLINEB_SPLIT_IDS = ("split_0", "split_1", "split_2")
MAINLINEB_DISTRACTOR_COUNTS = (8, 16)
MAINLINEB_PROTOCOL_TYPES = ("ratio_drop",)
MAINLINEB_B1_CLASS_COUNT = 14
MAINLINEB_B1_CLIP_COUNT = 48
MAINLINEB_B2_CLASS_COUNT = 18
MAINLINEB_B2_CLIP_COUNT = 64
MAINLINEB_B1_TARGET_CLASS_IDS = list(range(1, 15))
MAINLINEB_B2_TARGET_CLASS_IDS = list(range(1, 19))
MAINLINEB_DISTRACTOR_CLASS_IDS = [21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51]
MAINLINEB_B1_START_CLIP_ID = 300
MAINLINEB_B2_START_CLIP_ID = 600
SMOKE_MAX_TRAJECTORIES = 32
SMOKE_PREALIGN_EPOCHS = 1
SMOKE_BASE_EPOCHS = 1
SMOKE_AUG_EPOCHS = 1
SMOKE_PREALIGN_LR = 1e-4
SMOKE_BASE_LR = 5e-5
SMOKE_AUG_LR = 5e-5
SMOKE_TEMPERATURE = 0.07
SMOKE_TOPK = 5
SMOKE_SEED = 0


@dataclass(frozen=True)
class SelectedClip:
    clip_id: str
    trajectory_id: str
    gt_class_id: int
    gt_class_name: str
    observed_raw_ids: List[int]
    complexity_score: int
    selection_rank: int
    sample: Mapping[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _copytree(src_root: Path, dst_root: Path, relpaths: Sequence[str]) -> None:
    for relpath in relpaths:
        src = src_root / relpath
        dst = dst_root / relpath
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _normalize(vec: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        return None
    return (arr / norm).astype(np.float32)


def _write_tier2_runtime_fixture(root: Path) -> Dict[str, Any]:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    text_dir = root / "text_bank"
    audit_dir = root / "audit"
    exports_dir = root / "exports" / "lvvis_train_base"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload", audit_dir, exports_dir):
        path.mkdir(parents=True, exist_ok=True)

    class_ids = [*TIER2_TARGET_CLASS_IDS, *TIER2_DISTRACTOR_CLASS_IDS]
    class_names = {int(cid): f"class_{int(cid)}" for cid in class_ids}

    protos = np.zeros((len(class_ids), 512), dtype=np.float32)
    for idx, _class_id in enumerate(class_ids):
        protos[idx, idx] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [
            {
                "raw_id": int(class_id),
                "proto_path": f"payload/text_prototypes.npz#protos[{idx}]",
                "path_base_mode": "artifact_parent_dir",
            }
            for idx, class_id in enumerate(class_ids)
        ],
    )

    trajectory_rows: List[Record] = []
    carrier_rows: List[Record] = []
    frame_rows: List[Record] = []
    geom_rows: List[Record] = []
    weak_rows: List[Record] = []
    gt_rows: List[Record] = []

    clip_ids = list(range(100, 132))
    class_distribution = [4, 4, 3, 3, 3, 3, 3, 3, 3, 3]
    clip_index = 0
    clip_counter = 0
    for class_idx, gt_class_id in enumerate(TIER2_TARGET_CLASS_IDS):
        support_1 = TIER2_TARGET_CLASS_IDS[(class_idx + 1) % len(TIER2_TARGET_CLASS_IDS)]
        support_2 = TIER2_TARGET_CLASS_IDS[(class_idx + 2) % len(TIER2_TARGET_CLASS_IDS)]
        support_3 = TIER2_TARGET_CLASS_IDS[(class_idx + 3) % len(TIER2_TARGET_CLASS_IDS)]
        for _ in range(class_distribution[class_idx]):
            clip_id = clip_ids[clip_counter]
            clip_counter += 1
            trajectory_id = f"tier2_traj_{clip_id:03d}"
            carrier_payload = np.zeros((1, 768), dtype=np.float16)
            carrier_payload[0, clip_index % 10] = 1.0
            carrier_path = carrier_dir / f"carrier_{clip_id:03d}.npz"
            np.savez(carrier_path, z_norm=carrier_payload)

            frame_payload = np.zeros((1, 4, 768), dtype=np.float16)
            frame_payload[0, 0, (clip_index + 1) % 10] = 1.0
            frame_path = frame_dir / "payload" / f"clip_{clip_id:03d}.npz"
            np.savez(frame_path, slot_0=frame_payload[0])

            observed_raw_ids = [int(gt_class_id), int(support_1), int(support_2), int(support_3)]
            carrier_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "clip_id": str(clip_id),
                    "z_norm_path": f"{carrier_path.name}#z_norm[0]",
                    "frame_indices": [0],
                    "frame_carriers_norm_paths": [],
                    "path_base_mode": "artifact_parent_dir",
                }
            )
            frame_rows.append(
                {
                    "clip_id": str(clip_id),
                    "frame_index": 0,
                    "feat_path": f"payload/{frame_path.name}#0",
                    "path_base_mode": "artifact_parent_dir",
                }
            )
            geom_rows.append(
                {
                    "clip_id": str(clip_id),
                    "frame_index": 0,
                    "orig_h": 28,
                    "orig_w": 28,
                    "resized_h": 28,
                    "resized_w": 28,
                    "padded_h": 28,
                    "padded_w": 28,
                    "scale_y": 1.0,
                    "scale_x": 1.0,
                    "pad_left": 0,
                    "pad_top": 0,
                    "pad_right": 0,
                    "pad_bottom": 0,
                    "patch_size": 14,
                    "grid_h": 2,
                    "grid_w": 2,
                    "valid_token_mask_path": f"frame_geom_records.jsonl#{clip_index}",
                    "path_base_mode": "artifact_parent_dir",
                }
            )
            trajectory_rows.append(
                {
                    "dataset_name": "lvvis_train_base",
                    "split_tag": "train",
                    "clip_id": clip_id,
                    "video_id": clip_id,
                    "rank_in_clip": 0,
                    "trajectory_id": trajectory_id,
                    "generator_tag": "synthetic_tier2",
                    "pred_score": 0.9,
                    "frame_indices": [0],
                    "masks_rle": [{}],
                    "boxes_xyxy": [[0, 0, 10, 10]],
                    "valid_carrier": True,
                    "invalid_reason": None,
                    "image_size": [28, 28],
                }
            )
            weak_rows.append(
                {
                    "clip_id": str(clip_id),
                    "video_id": clip_id,
                    "observed_raw_ids": list(observed_raw_ids),
                    "observation_protocol_id": "synthetic_tier2_v1",
                    "completeness_status": "unknown",
                }
            )
            gt_rows.append(
                {
                    "dataset_name": "lvvis_train_base",
                    "trajectory_source_branch": "mainline",
                    "split_tag": "train",
                    "trajectory_id": trajectory_id,
                    "clip_id": clip_id,
                    "video_id": clip_id,
                    "matched_gt_track_id": trajectory_id,
                    "matched_gt_raw_id": int(gt_class_id),
                    "matched_gt_class_id": int(gt_class_id),
                    "match_iou_mean": 1.0,
                    "match_iou_p50": 1.0,
                    "match_support_frame_count": 1,
                    "match_quality": "identity",
                    "semantic_purity_flag": True,
                    "audit_usable": True,
                }
            )
            clip_index += 1

    _write_jsonl(carrier_dir / "carrier_records.jsonl", carrier_rows)
    _write_jsonl(frame_dir / "frame_records.jsonl", frame_rows)
    _write_jsonl(frame_dir / "frame_geom_records.jsonl", geom_rows)
    _write_json(root / "weak_labels" / "weak_labels_train.json", weak_rows)
    _write_jsonl(exports_dir / "trajectory_records.jsonl", trajectory_rows)
    _write_jsonl(audit_dir / "trajectory_gt_match_train_mainline.jsonl", gt_rows)

    return {
        "class_names": class_names,
        "class_ids": class_ids,
        "clip_ids": clip_ids,
        "trajectory_ids": [row["trajectory_id"] for row in trajectory_rows],
    }


def _write_stressA_runtime_fixture(root: Path) -> Dict[str, Any]:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    text_dir = root / "text_bank"
    audit_dir = root / "audit"
    exports_dir = root / "exports" / "lvvis_train_base"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload", audit_dir, exports_dir):
        path.mkdir(parents=True, exist_ok=True)

    class_ids = [*STRESSA_TARGET_CLASS_IDS, *STRESSA_DISTRACTOR_CLASS_IDS]
    class_names = {int(cid): f"class_{int(cid)}" for cid in class_ids}

    protos = np.zeros((len(class_ids), 512), dtype=np.float32)
    for idx, _class_id in enumerate(class_ids):
        protos[idx, idx] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [
            {
                "raw_id": int(class_id),
                "proto_path": f"payload/text_prototypes.npz#protos[{idx}]",
                "path_base_mode": "artifact_parent_dir",
            }
            for idx, class_id in enumerate(class_ids)
        ],
    )

    trajectory_rows: List[Record] = []
    carrier_rows: List[Record] = []
    frame_rows: List[Record] = []
    geom_rows: List[Record] = []
    weak_rows: List[Record] = []
    gt_rows: List[Record] = []
    split_to_clip_ids: Dict[str, List[str]] = {str(split_id): [] for split_id in STRESSA_SPLIT_IDS}

    clip_id = 200
    for split_index, split_id in enumerate(STRESSA_SPLIT_IDS):
        split_start = clip_id
        split_clip_ids = list(range(split_start, split_start + STRESSA_CLIP_COUNT))
        split_to_clip_ids[str(split_id)] = [str(cid) for cid in split_clip_ids]
        class_distribution = [4, 4, 3, 3, 3, 3, 3, 3, 3, 3]
        split_clip_counter = 0
        for class_idx, gt_class_id in enumerate(STRESSA_TARGET_CLASS_IDS):
            support_1 = STRESSA_TARGET_CLASS_IDS[(class_idx + 1) % len(STRESSA_TARGET_CLASS_IDS)]
            support_2 = STRESSA_TARGET_CLASS_IDS[(class_idx + 2) % len(STRESSA_TARGET_CLASS_IDS)]
            support_3 = STRESSA_TARGET_CLASS_IDS[(class_idx + 3) % len(STRESSA_TARGET_CLASS_IDS)]
            for _ in range(class_distribution[class_idx]):
                current_clip_id = split_clip_ids[split_clip_counter]
                split_clip_counter += 1
                trajectory_id = f"stressa_{split_id}_traj_{current_clip_id:03d}"
                carrier_payload = np.zeros((1, 768), dtype=np.float16)
                carrier_payload[0, (current_clip_id - split_start) % 10] = 1.0
                carrier_path = carrier_dir / f"carrier_{current_clip_id:03d}.npz"
                np.savez(carrier_path, z_norm=carrier_payload)

                frame_payload = np.zeros((1, 4, 768), dtype=np.float16)
                frame_payload[0, 0, (current_clip_id - split_start + 1) % 10] = 1.0
                frame_path = frame_dir / "payload" / f"clip_{current_clip_id:03d}.npz"
                np.savez(frame_path, slot_0=frame_payload[0])

                observed_raw_ids = [int(gt_class_id), int(support_1), int(support_2), int(support_3)]
                carrier_rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "clip_id": str(current_clip_id),
                        "z_norm_path": f"{carrier_path.name}#z_norm[0]",
                        "frame_indices": [0],
                        "frame_carriers_norm_paths": [],
                        "path_base_mode": "artifact_parent_dir",
                    }
                )
                frame_rows.append(
                    {
                        "clip_id": str(current_clip_id),
                        "frame_index": 0,
                        "feat_path": f"payload/{frame_path.name}#0",
                        "path_base_mode": "artifact_parent_dir",
                    }
                )
                geom_rows.append(
                    {
                        "clip_id": str(current_clip_id),
                        "frame_index": 0,
                        "orig_h": 28,
                        "orig_w": 28,
                        "resized_h": 28,
                        "resized_w": 28,
                        "padded_h": 28,
                        "padded_w": 28,
                        "scale_y": 1.0,
                        "scale_x": 1.0,
                        "pad_left": 0,
                        "pad_top": 0,
                        "pad_right": 0,
                        "pad_bottom": 0,
                        "patch_size": 14,
                        "grid_h": 2,
                        "grid_w": 2,
                        "valid_token_mask_path": f"frame_geom_records.jsonl#{current_clip_id}",
                        "path_base_mode": "artifact_parent_dir",
                    }
                )
                trajectory_rows.append(
                    {
                        "dataset_name": "lvvis_train_base",
                        "split_tag": str(split_id),
                        "clip_id": current_clip_id,
                        "video_id": current_clip_id,
                        "rank_in_clip": 0,
                        "trajectory_id": trajectory_id,
                        "generator_tag": "synthetic_stressA",
                        "pred_score": 0.9,
                        "frame_indices": [0],
                        "masks_rle": [{}],
                        "boxes_xyxy": [[0, 0, 10, 10]],
                        "valid_carrier": True,
                        "invalid_reason": None,
                        "image_size": [28, 28],
                    }
                )
                weak_rows.append(
                    {
                        "clip_id": str(current_clip_id),
                        "video_id": current_clip_id,
                        "observed_raw_ids": list(observed_raw_ids),
                        "observation_protocol_id": "synthetic_stressa_v1",
                        "completeness_status": "unknown",
                    }
                )
                gt_rows.append(
                    {
                        "dataset_name": "lvvis_train_base",
                        "trajectory_source_branch": "mainline",
                        "split_tag": str(split_id),
                        "trajectory_id": trajectory_id,
                        "clip_id": current_clip_id,
                        "video_id": current_clip_id,
                        "matched_gt_track_id": trajectory_id,
                        "matched_gt_raw_id": int(gt_class_id),
                        "matched_gt_class_id": int(gt_class_id),
                        "match_iou_mean": 1.0,
                        "match_iou_p50": 1.0,
                        "match_support_frame_count": 1,
                        "match_quality": "identity",
                        "semantic_purity_flag": True,
                        "audit_usable": True,
                    }
                )
        clip_id += STRESSA_CLIP_COUNT

    _write_jsonl(carrier_dir / "carrier_records.jsonl", carrier_rows)
    _write_jsonl(frame_dir / "frame_records.jsonl", frame_rows)
    _write_jsonl(frame_dir / "frame_geom_records.jsonl", geom_rows)
    _write_json(root / "weak_labels" / "weak_labels_train.json", weak_rows)
    _write_jsonl(exports_dir / "trajectory_records.jsonl", trajectory_rows)
    _write_jsonl(audit_dir / "trajectory_gt_match_train_mainline.jsonl", gt_rows)

    return {
        "class_names": class_names,
        "class_ids": class_ids,
        "clip_ids": [str(row["clip_id"]) for row in trajectory_rows],
        "trajectory_ids": [row["trajectory_id"] for row in trajectory_rows],
        "split_to_clip_ids": split_to_clip_ids,
    }


def _write_mainline_ratio_drop_fixture(
    root: Path,
    *,
    split_ids: Sequence[str],
    target_class_ids: Sequence[int],
    distractor_class_ids: Sequence[int],
    clip_count_target: int,
    start_clip_id: int,
    generator_tag: str,
) -> Dict[str, Any]:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    text_dir = root / "text_bank"
    audit_dir = root / "audit"
    exports_dir = root / "exports" / "lvvis_train_base"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload", audit_dir, exports_dir):
        path.mkdir(parents=True, exist_ok=True)

    class_ids = [*target_class_ids, *distractor_class_ids]
    class_names = {int(cid): f"class_{int(cid)}" for cid in class_ids}

    protos = np.zeros((len(class_ids), 512), dtype=np.float32)
    for idx, _class_id in enumerate(class_ids):
        protos[idx, idx] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [
            {
                "raw_id": int(class_id),
                "proto_path": f"payload/text_prototypes.npz#protos[{idx}]",
                "path_base_mode": "artifact_parent_dir",
            }
            for idx, class_id in enumerate(class_ids)
        ],
    )

    def _class_distribution(class_count: int, total_clips: int) -> List[int]:
        base = int(total_clips) // int(class_count)
        remainder = int(total_clips) % int(class_count)
        return [base + 1 if idx < remainder else base for idx in range(int(class_count))]

    trajectory_rows: List[Record] = []
    carrier_rows: List[Record] = []
    frame_rows: List[Record] = []
    geom_rows: List[Record] = []
    weak_rows: List[Record] = []
    gt_rows: List[Record] = []
    split_to_clip_ids: Dict[str, List[str]] = {str(split_id): [] for split_id in split_ids}

    clip_id = int(start_clip_id)
    class_distribution = _class_distribution(len(target_class_ids), clip_count_target)
    for split_index, split_id in enumerate(split_ids):
        split_start = clip_id
        split_clip_ids = list(range(split_start, split_start + int(clip_count_target)))
        split_to_clip_ids[str(split_id)] = [str(cid) for cid in split_clip_ids]
        split_clip_counter = 0
        for class_idx, gt_class_id in enumerate(target_class_ids):
            support_1 = target_class_ids[(class_idx + 1) % len(target_class_ids)]
            support_2 = target_class_ids[(class_idx + 2) % len(target_class_ids)]
            support_3 = target_class_ids[(class_idx + 3) % len(target_class_ids)]
            for _ in range(class_distribution[class_idx]):
                current_clip_id = split_clip_ids[split_clip_counter]
                split_clip_counter += 1
                trajectory_id = f"{generator_tag}_{split_id}_traj_{current_clip_id:03d}"
                carrier_payload = np.zeros((1, 768), dtype=np.float16)
                carrier_payload[0, (current_clip_id - split_start) % 10] = 1.0
                carrier_path = carrier_dir / f"carrier_{current_clip_id:03d}.npz"
                np.savez(carrier_path, z_norm=carrier_payload)

                frame_payload = np.zeros((1, 4, 768), dtype=np.float16)
                frame_payload[0, 0, (current_clip_id - split_start + 1) % 10] = 1.0
                frame_path = frame_dir / "payload" / f"clip_{current_clip_id:03d}.npz"
                np.savez(frame_path, slot_0=frame_payload[0])

                observed_raw_ids = [int(gt_class_id), int(support_1), int(support_2), int(support_3)]
                carrier_rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "clip_id": str(current_clip_id),
                        "z_norm_path": f"{carrier_path.name}#z_norm[0]",
                        "frame_indices": [0],
                        "frame_carriers_norm_paths": [],
                        "path_base_mode": "artifact_parent_dir",
                    }
                )
                frame_rows.append(
                    {
                        "clip_id": str(current_clip_id),
                        "frame_index": 0,
                        "feat_path": f"payload/{frame_path.name}#0",
                        "path_base_mode": "artifact_parent_dir",
                    }
                )
                geom_rows.append(
                    {
                        "clip_id": str(current_clip_id),
                        "frame_index": 0,
                        "orig_h": 28,
                        "orig_w": 28,
                        "resized_h": 28,
                        "resized_w": 28,
                        "padded_h": 28,
                        "padded_w": 28,
                        "scale_y": 1.0,
                        "scale_x": 1.0,
                        "pad_left": 0,
                        "pad_top": 0,
                        "pad_right": 0,
                        "pad_bottom": 0,
                        "patch_size": 14,
                        "grid_h": 2,
                        "grid_w": 2,
                        "valid_token_mask_path": f"frame_geom_records.jsonl#{current_clip_id}",
                        "path_base_mode": "artifact_parent_dir",
                    }
                )
                trajectory_rows.append(
                    {
                        "dataset_name": "lvvis_train_base",
                        "split_tag": str(split_id),
                        "clip_id": current_clip_id,
                        "video_id": current_clip_id,
                        "rank_in_clip": 0,
                        "trajectory_id": trajectory_id,
                        "generator_tag": str(generator_tag),
                        "pred_score": 0.9,
                        "frame_indices": [0],
                        "masks_rle": [{}],
                        "boxes_xyxy": [[0, 0, 10, 10]],
                        "valid_carrier": True,
                        "invalid_reason": None,
                        "image_size": [28, 28],
                    }
                )
                weak_rows.append(
                    {
                        "clip_id": str(current_clip_id),
                        "video_id": current_clip_id,
                        "observed_raw_ids": list(observed_raw_ids),
                        "observation_protocol_id": f"synthetic_{generator_tag}_v1",
                        "completeness_status": "unknown",
                    }
                )
                gt_rows.append(
                    {
                        "dataset_name": "lvvis_train_base",
                        "trajectory_source_branch": "mainline",
                        "split_tag": str(split_id),
                        "trajectory_id": trajectory_id,
                        "clip_id": current_clip_id,
                        "video_id": current_clip_id,
                        "matched_gt_track_id": trajectory_id,
                        "matched_gt_raw_id": int(gt_class_id),
                        "matched_gt_class_id": int(gt_class_id),
                        "match_iou_mean": 1.0,
                        "match_iou_p50": 1.0,
                        "match_support_frame_count": 1,
                        "match_quality": "identity",
                        "semantic_purity_flag": True,
                        "audit_usable": True,
                    }
                )
        clip_id += int(clip_count_target)

    _write_jsonl(carrier_dir / "carrier_records.jsonl", carrier_rows)
    _write_jsonl(frame_dir / "frame_records.jsonl", frame_rows)
    _write_jsonl(frame_dir / "frame_geom_records.jsonl", geom_rows)
    _write_json(root / "weak_labels" / "weak_labels_train.json", weak_rows)
    _write_jsonl(exports_dir / "trajectory_records.jsonl", trajectory_rows)
    _write_jsonl(audit_dir / "trajectory_gt_match_train_mainline.jsonl", gt_rows)

    return {
        "class_names": class_names,
        "class_ids": class_ids,
        "clip_ids": [str(row["clip_id"]) for row in trajectory_rows],
        "trajectory_ids": [row["trajectory_id"] for row in trajectory_rows],
        "split_to_clip_ids": split_to_clip_ids,
    }


def _build_candidate_map(text_root: Path) -> Dict[int, Dict[str, Any]]:
    from videocutler.ext_stageb_ovvis.banks.text_bank import read_text_prototype_records

    records = read_text_prototype_records(text_root / "text_bank" / "text_prototype_records.jsonl")
    return {int(record["raw_id"]): dict(record) for record in records}


def _select_semiopen_distractors(
    class_name_lookup: Mapping[int, str],
    *,
    count: int = SEMIOPEN_DISTRACTOR_COUNT,
    source_class_ids: Sequence[int] = TIER2_DISTRACTOR_CLASS_IDS,
) -> List[Dict[str, Any]]:
    selected_ids = [int(cid) for cid in list(source_class_ids)[: int(count)]]
    return [
        {
            "class_id": int(class_id),
            "class_name": str(class_name_lookup.get(int(class_id), f"class_{int(class_id)}")),
            "selection_rank": int(index) + 1,
            "selection_rule": "deterministic_package_authorized_distractor_order",
        }
        for index, class_id in enumerate(selected_ids)
    ]


def _select_tier2_clips(
    materialized_samples: Sequence[Mapping[str, Any]],
    gt_lookup: Mapping[str, Mapping[str, Any]],
    *,
    class_count_target: int = TIER2_CLASS_COUNT,
    clip_count_target: int = TIER2_CLIP_COUNT,
    class_name_lookup: Optional[Mapping[int, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    class_name_lookup = dict(class_name_lookup or {})
    clip_records: List[Dict[str, Any]] = []
    by_clip: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for sample in materialized_samples:
        by_clip[str(sample.get("clip_id", ""))].append(sample)

    for clip_id, clip_samples in by_clip.items():
        clip_samples = sorted(list(clip_samples), key=lambda row: str(row.get("trajectory_id", "")))
        chosen_sample = None
        chosen_gt_class_id = None
        chosen_gt_record: Mapping[str, Any] = {}
        for sample in clip_samples:
            gt_record = gt_lookup.get(str(sample.get("trajectory_id", "")), {})
            gt_class_id = _extract_gt_class_id(gt_record) if gt_record else None
            if gt_class_id is None or not _extract_gt_available(gt_record):
                continue
            observed_raw_ids = sorted({int(x) for x in list(sample.get("observed_raw_ids", []))})
            if len(observed_raw_ids) < 4:
                continue
            chosen_sample = sample
            chosen_gt_class_id = int(gt_class_id)
            chosen_gt_record = gt_record
            break
        if chosen_sample is None or chosen_gt_class_id is None:
            continue
        observed_raw_ids = sorted({int(x) for x in list(chosen_sample.get("observed_raw_ids", []))})
        if int(chosen_gt_class_id) not in observed_raw_ids:
            continue
        clip_records.append(
            {
                "clip_id": str(clip_id),
                "trajectory_id": str(chosen_sample.get("trajectory_id", "")),
                "gt_class_id": int(chosen_gt_class_id),
                "gt_class_name": str(class_name_lookup.get(int(chosen_gt_class_id), f"class_{int(chosen_gt_class_id)}")),
                "observed_raw_ids": observed_raw_ids,
                "observed_count": int(len(observed_raw_ids)),
                "complexity_score": int(len(observed_raw_ids)),
                "selection_rank": 0,
                "gt_record": dict(chosen_gt_record),
                "sample": dict(chosen_sample),
            }
        )

    class_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for record in clip_records:
        class_groups[int(record["gt_class_id"])].append(record)

    class_rows: List[Dict[str, Any]] = []
    for class_id, group in class_groups.items():
        sorted_group = sorted(group, key=lambda row: (int(row["complexity_score"]), str(row["clip_id"]), str(row["trajectory_id"])))
        class_rows.append(
            {
                "class_id": int(class_id),
                "class_name": str(class_name_lookup.get(int(class_id), f"class_{int(class_id)}")),
                "clip_support_count": int(len(sorted_group)),
                "mean_clip_complexity": float(np.mean([int(row["complexity_score"]) for row in sorted_group])) if sorted_group else 0.0,
                "clip_ids": [str(row["clip_id"]) for row in sorted_group],
            }
        )
    class_rows.sort(key=lambda row: (-int(row["clip_support_count"]), float(row["mean_clip_complexity"]), int(row["class_id"])))
    selected_class_ids = [int(row["class_id"]) for row in class_rows[: int(class_count_target)]]
    selected_class_set = set(selected_class_ids)

    eligible_groups: Dict[int, List[Dict[str, Any]]] = {}
    for class_id in selected_class_ids:
        eligible = [
            dict(row)
            for row in class_groups.get(class_id, [])
            if any((rid in selected_class_set and rid != int(class_id)) for rid in [int(x) for x in row["observed_raw_ids"]])
        ]
        eligible.sort(key=lambda row: (int(row["complexity_score"]), str(row["clip_id"]), str(row["trajectory_id"])))
        eligible_groups[int(class_id)] = eligible

    selected_clip_rows: List[Dict[str, Any]] = []
    used_clip_ids: set[str] = set()
    round_index = 0
    while len(selected_clip_rows) < int(clip_count_target):
        progressed = False
        for class_id in selected_class_ids:
            group = eligible_groups.get(int(class_id), [])
            if round_index >= len(group):
                continue
            candidate = dict(group[round_index])
            if str(candidate["clip_id"]) in used_clip_ids:
                continue
            candidate["selected_class_rank"] = int(selected_class_ids.index(int(class_id))) + 1
            selected_clip_rows.append(candidate)
            used_clip_ids.add(str(candidate["clip_id"]))
            progressed = True
            if len(selected_clip_rows) >= int(clip_count_target):
                break
        if not progressed:
            break
        round_index += 1

    if len(selected_clip_rows) < int(clip_count_target):
        raise RuntimeError(
            f"could not build Tier-2 bounded set: selected={len(selected_clip_rows)} clips, target={clip_count_target}"
        )

    selected_clip_rows.sort(key=lambda row: (int(row["selected_class_rank"]), int(row["complexity_score"]), str(row["clip_id"]), str(row["trajectory_id"])))
    for index, row in enumerate(selected_clip_rows, start=1):
        row["selection_rank"] = int(index)

    return class_rows[: int(class_count_target)], selected_clip_rows, eligible_groups


def _select_split_tier2_clips(
    materialized_samples: Sequence[Mapping[str, Any]],
    gt_lookup: Mapping[str, Mapping[str, Any]],
    *,
    clip_ids: Sequence[str],
    class_count_target: int,
    clip_count_target: int,
    class_name_lookup: Optional[Mapping[int, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    clip_id_set = {str(clip_id) for clip_id in clip_ids}
    split_samples = [dict(sample) for sample in materialized_samples if str(sample.get("clip_id", "")) in clip_id_set]
    return _select_tier2_clips(
        split_samples,
        gt_lookup,
        class_count_target=class_count_target,
        clip_count_target=clip_count_target,
        class_name_lookup=class_name_lookup,
    )


def _derive_protocol_samples(
    selected_clip_rows: Sequence[Mapping[str, Any]],
    *,
    class_name_lookup: Mapping[int, str],
    candidate_map: Mapping[int, Mapping[str, Any]],
    drop_ratio: Optional[float],
    protocol_name: str,
    protocol_type: str = "ratio_drop",
    semiopen_distractor_ids: Sequence[int] = (),
) -> Tuple[List[Record], Dict[str, int], Dict[str, List[int]], List[Record]]:
    selected_class_ids = sorted({int(row["gt_class_id"]) for row in selected_clip_rows})
    selected_class_set = set(selected_class_ids)
    distractor_ids = [int(x) for x in semiopen_distractor_ids if int(x) not in selected_class_set]
    selected_samples: List[Record] = []
    selected_clip_ids: Dict[str, int] = {}
    selected_observed: Dict[str, List[int]] = {}
    selected_protocol_rows: List[Record] = []

    for row in selected_clip_rows:
        sample = dict(row["sample"])
        clip_id = str(sample["clip_id"])
        drop_class_id = int(row["gt_class_id"])
        observed_raw_ids = sorted({int(x) for x in list(sample.get("observed_raw_ids", []))})
        protocol_type = str(protocol_type)
        if protocol_type == "keep_exactly_one_observed_per_clip":
            keep_id = int(sorted(observed_raw_ids)[0])
            drop_ids = [int(rid) for rid in observed_raw_ids if int(rid) != int(keep_id)]
            if not drop_ids:
                raise RuntimeError(f"Tier-2 sample for clip {clip_id} would not drop any labels in keep-one mode")
            observed_after_drop = [int(keep_id)]
            primary_drop_class_id = int(drop_ids[0])
        else:
            if drop_ratio is None:
                raise RuntimeError(f"drop_ratio is required for protocol_type={protocol_type}")
            drop_count = max(1, int(round(len(observed_raw_ids) * float(drop_ratio))))
            drop_ids = [int(drop_class_id)]
            for rid in observed_raw_ids:
                if rid == int(drop_class_id):
                    continue
                if len(drop_ids) >= drop_count:
                    break
                drop_ids.append(int(rid))
            observed_after_drop = [rid for rid in observed_raw_ids if rid not in drop_ids]
            primary_drop_class_id = int(drop_class_id)
        if not observed_after_drop:
            raise RuntimeError(f"Tier-2 sample for clip {clip_id} would become empty after drop")
        candidate_ids_known = list(observed_after_drop)
        candidate_ids_extra = [*drop_ids, *distractor_ids]
        candidate_text_records = [dict(candidate_map[int(raw_id)]) for raw_id in [*candidate_ids_known, *candidate_ids_extra]]
        class_name = str(class_name_lookup.get(int(primary_drop_class_id), f"class_{int(primary_drop_class_id)}"))
        admission_reason = (
            f"tier2_drop_{int(float(drop_ratio) * 100):02d}_controlled_recovery_extra"
            if drop_ratio is not None
            else f"{protocol_type}_controlled_recovery_extra"
        )
        sample["observed_raw_ids"] = list(observed_after_drop)
        sample["weak_label_record"] = dict(sample.get("weak_label_record", {}))
        sample["weak_label_record"]["observed_raw_ids"] = list(observed_after_drop)
        sample["candidate_ids_known"] = candidate_ids_known
        sample["candidate_ids_extra"] = candidate_ids_extra
        sample["target_candidate_ids_extra"] = list(drop_ids)
        sample["semiopen_distractor_ids"] = list(distractor_ids)
        sample["candidate_text_prototypes"] = candidate_text_records
        sample["candidate_ids_extra_provenance"] = [
            {
                "raw_id": int(raw_id),
                "score": None,
                "rank": int(rank) + 1,
                "admission_reason": admission_reason,
                "proposal_source": protocol_name,
            }
            for rank, raw_id in enumerate(candidate_ids_extra)
        ]
        sample["candidate_proposal_source"] = protocol_name
        sample["missing_views"] = []
        sample["invalid_reasons"] = []
        sample["sample_valid"] = True
        sample["controlled_drop_protocol"] = {
            "protocol_name": protocol_name,
            "protocol_type": protocol_type,
            "drop_ratio": float(drop_ratio) if drop_ratio is not None else None,
            "drop_class_ids": list(candidate_ids_extra),
            "drop_class_names": [str(class_name_lookup.get(cid, f"class_{cid}")) for cid in candidate_ids_extra],
            "target_drop_class_ids": list(drop_ids),
            "target_drop_class_names": [str(class_name_lookup.get(cid, f"class_{cid}")) for cid in drop_ids],
            "distractor_class_ids": list(distractor_ids),
            "distractor_class_names": [str(class_name_lookup.get(cid, f"class_{cid}")) for cid in distractor_ids],
            "selected_class_ids": selected_class_ids,
            "selected_class_names": [str(class_name_lookup.get(cid, f"class_{cid}")) for cid in selected_class_ids],
            "observed_raw_ids_after_drop": list(observed_after_drop),
        }
        selected_samples.append(sample)
        selected_clip_ids[clip_id] = int(primary_drop_class_id)
        selected_observed[clip_id] = list(observed_after_drop)
        selected_protocol_rows.append(
            {
                "clip_id": clip_id,
                "trajectory_id": str(sample["trajectory_id"]),
                "drop_class_id": int(primary_drop_class_id),
                "drop_class_name": class_name,
                "dropped_class_ids": list(candidate_ids_extra),
                "dropped_class_names": [str(class_name_lookup.get(cid, f"class_{cid}")) for cid in candidate_ids_extra],
                "target_drop_class_ids": list(drop_ids),
                "target_drop_class_names": [str(class_name_lookup.get(cid, f"class_{cid}")) for cid in drop_ids],
                "distractor_class_ids": list(distractor_ids),
                "distractor_class_names": [str(class_name_lookup.get(cid, f"class_{cid}")) for cid in distractor_ids],
                "observed_raw_ids_before_drop": observed_raw_ids,
                "observed_raw_ids_after_drop": list(observed_after_drop),
                "selected_class_ids": selected_class_ids,
                "selected_class_names": [str(class_name_lookup.get(cid, f"class_{cid}")) for cid in selected_class_ids],
                "protocol_name": protocol_name,
                "protocol_type": protocol_type,
                "drop_ratio": float(drop_ratio) if drop_ratio is not None else None,
                "observed_keep_class_id": int(observed_after_drop[0]) if observed_after_drop else None,
            }
        )
    return selected_samples, selected_clip_ids, selected_observed, selected_protocol_rows


def _clone_target_only_eval_samples(selected_samples: Sequence[Mapping[str, Any]]) -> List[Record]:
    cloned_samples: List[Record] = []
    for sample in selected_samples:
        clone = dict(sample)
        target_extra_ids = [int(x) for x in list(sample.get("target_candidate_ids_extra", sample.get("candidate_ids_extra", [])))]
        known_ids = [int(x) for x in list(sample.get("candidate_ids_known", []))]
        allowed_ids = {*(known_ids), *target_extra_ids}
        clone["candidate_ids_extra"] = list(target_extra_ids)
        clone["candidate_text_prototypes"] = [
            dict(proto) for proto in list(sample.get("candidate_text_prototypes", [])) if int(proto.get("raw_id", -1)) in allowed_ids
        ]
        clone["candidate_ids_extra_provenance"] = [
            dict(proto)
            for proto in list(sample.get("candidate_ids_extra_provenance", []))
            if int(proto.get("raw_id", -1)) in set(target_extra_ids)
        ]
        clone["candidate_proposal_source"] = "target_only_eval"
        cloned_samples.append(clone)
    return cloned_samples


def _evaluate_rows(
    *,
    output_root: Path,
    selected_samples: Sequence[Mapping[str, Any]],
    gt_lookup: Mapping[str, Mapping[str, Any]],
    stage_id: str,
    checkpoint_path: Path,
    protocol_name: str,
    previous_by_trajectory: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Record]:
    from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    config_payload = dict(checkpoint.get("projector_config", {}))
    projector = Projector(
        ProjectorConfig(
            input_dim=int(config_payload.get("input_dim", 768)),
            hidden_dim=int(config_payload.get("hidden_dim", 512)),
            output_dim=int(config_payload.get("output_dim", 512)),
            dropout=float(config_payload.get("dropout", 0.0)),
            use_layernorm=bool(config_payload.get("use_layernorm", True)),
        )
    ).to("cpu")
    projector.load_state_dict(checkpoint["projector_state_dict"])
    projector.eval()

    rows = build_projector_quality_rows(
        output_root=output_root,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
        stage_id=stage_id,
        snapshot_id="stage_end",
        materialized_samples=selected_samples,
        projector=projector,
        topk=SMOKE_TOPK,
        gt_sidecar_lookup=gt_lookup,
        temperature=SMOKE_TEMPERATURE,
        previous_by_trajectory=previous_by_trajectory,
    )
    for row in rows:
        drop_class_id = int(row.get("gt_class_id")) if row.get("gt_class_id") is not None else None
        row["protocol_name"] = protocol_name
        row["gt_dropped_from_observed"] = bool(drop_class_id is not None and drop_class_id not in row.get("observed_raw_ids", []))
        row["gt_recovered_via_extra"] = bool(
            row["gt_dropped_from_observed"]
            and row.get("closed_set_is_gt_top1")
            and row.get("gt_class_id") is not None
            and row.get("gt_in_extra_domain")
        )
        row["gt_rank_closed_set"] = row.get("closed_set_gt_rank")
        row["gt_rank_broader_bounded"] = row.get("gt_rank_stage_domain")
        row["gt_rank_target_set"] = row.get("closed_set_gt_rank")
        row["gt_rank_eval_set"] = row.get("gt_rank_stage_domain")
        row["gt_rank_full_vocab"] = row.get("gt_rank_full_vocab")
        row["gt_score"] = row.get("gt_score")
    return rows


def _aggregate_protocol_summary(
    *,
    protocol_name: str,
    selected_class_rows: Sequence[Mapping[str, Any]],
    selected_clip_rows: Sequence[Mapping[str, Any]],
    rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]],
    prior_tier1_summary: Mapping[str, Any],
    prior_full_vocab_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    final_rows = list(rows_by_stage.get("softem_aug", []))
    dropped_rows = [row for row in final_rows if bool(row.get("gt_dropped_from_observed"))]
    recovered_rows = [row for row in dropped_rows if bool(row.get("gt_recovered_via_extra"))]
    topk_hits = [row for row in dropped_rows if bool(row.get("is_gt_in_topk"))]
    extra_top1_hits = [row for row in dropped_rows if bool(row.get("closed_set_is_gt_top1"))]
    recall = float(len(recovered_rows) / len(dropped_rows)) if dropped_rows else 0.0
    extra_recall = float(len(topk_hits) / len(dropped_rows)) if dropped_rows else 0.0
    extra_precision = float(len(recovered_rows) / len(extra_top1_hits)) if extra_top1_hits else 0.0
    stage_summaries: Dict[str, Any] = {}
    for stage_id in ("prealign", "softem_base", "softem_aug"):
        stage_rows = [row for row in rows_by_stage.get(stage_id, []) if row.get("gt_available_for_audit")]
        gt_ranks = [float(row["gt_rank_full_vocab"]) for row in stage_rows if row.get("gt_rank_full_vocab") is not None]
        gt_scores = [float(row["gt_score"]) for row in stage_rows if row.get("gt_score") is not None]
        margins = [float(row["margin_gt_minus_best_wrong"]) for row in stage_rows if row.get("margin_gt_minus_best_wrong") is not None]
        cosine_gt = [float(row["cosine_to_gt_text"]) for row in stage_rows if row.get("cosine_to_gt_text") is not None]
        cosine_wrong = [float(row["cosine_to_best_wrong_text"]) for row in stage_rows if row.get("cosine_to_best_wrong_text") is not None]
        observed_mass = [float(row["observed_mass_quality"]) for row in stage_rows if row.get("observed_mass_quality") is not None]
        stage_summaries[stage_id] = {
            "status": "PASS" if stage_rows else "EMPTY",
            "row_count": int(len(rows_by_stage.get(stage_id, []))),
            "gt_row_count": int(len(stage_rows)),
            "gt_top1_rate": float(sum(1 for row in stage_rows if bool(row.get("is_gt_top1"))) / len(stage_rows)) if stage_rows else None,
            "gt_topk_rate": float(sum(1 for row in stage_rows if bool(row.get("is_gt_in_topk"))) / len(stage_rows)) if stage_rows else None,
            "mean_gt_rank": float(np.mean(gt_ranks)) if gt_ranks else None,
            "median_gt_rank": float(np.median(gt_ranks)) if gt_ranks else None,
            "mean_gt_score": float(np.mean(gt_scores)) if gt_scores else None,
            "mean_margin_gt_minus_best_wrong": float(np.mean(margins)) if margins else None,
            "mean_cosine_to_gt_text": float(np.mean(cosine_gt)) if cosine_gt else None,
            "mean_cosine_to_best_wrong_text": float(np.mean(cosine_wrong)) if cosine_wrong else None,
            "observed_top1_rate": float(sum(1 for row in stage_rows if bool(row.get("observed_top1_rate"))) / len(stage_rows)) if stage_rows else None,
            "observed_mass_quality": float(np.mean(observed_mass)) if observed_mass else None,
        }

    prior_tier1_aug = prior_tier1_summary.get("stage_summaries", {}).get("softem_aug", {})
    prior_full_vocab_aug = prior_full_vocab_summary.get("stage_summaries", {}).get("softem_aug", {})
    smoothness = "smooth" if (prior_tier1_aug.get("gt_top1_rate") is not None and stage_summaries["softem_aug"]["gt_top1_rate"] is not None and stage_summaries["softem_aug"]["gt_top1_rate"] <= prior_tier1_aug.get("gt_top1_rate", 0.0)) else "abrupt"

    return {
        "protocol_name": protocol_name,
        "status": "PASS" if dropped_rows else "EMPTY",
        "selected_class_count": int(len(selected_class_rows)),
        "selected_clip_count": int(len(selected_clip_rows)),
        "selected_class_ids": [int(row["class_id"]) for row in selected_class_rows],
        "selected_clip_ids": [str(row["clip_id"]) for row in selected_clip_rows],
        "dropped_class_recovery_rate": recall,
        "extra_gt_recall@K": extra_recall,
        "extra_precision@K": extra_precision,
        "missing_class_recovery_rate": recall,
        "spurious_extra_rate": float(1.0 - extra_precision),
        "trajectory_level_rows": int(len(final_rows)),
        "dropped_trajectory_rows": int(len(dropped_rows)),
        "recovered_trajectory_rows": int(len(recovered_rows)),
        "stage_summaries": stage_summaries,
        "comparison_against_prior": {
            "tier1_closed_set": {
                "gt_top1_rate": prior_tier1_aug.get("gt_top1_rate"),
                "mean_gt_rank": prior_tier1_aug.get("mean_gt_rank"),
                "mean_gt_score": prior_tier1_aug.get("mean_gt_score"),
            },
            "weak_full_vocab": {
                "gt_top1_rate": prior_full_vocab_aug.get("gt_top1_rate"),
                "mean_gt_rank": prior_full_vocab_aug.get("mean_gt_rank"),
                "mean_gt_score": prior_full_vocab_aug.get("mean_gt_score"),
            },
        },
        "trend_from_tier1": smoothness,
    }


def _build_casebook(
    rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]],
    protocol_name: str,
    *,
    distractor_ids: Sequence[int] = (),
) -> List[Record]:
    final_rows = list(rows_by_stage.get("softem_aug", []))
    distractor_set = {int(x) for x in distractor_ids}
    casebook: List[Record] = []
    for row in final_rows:
        if not bool(row.get("gt_dropped_from_observed")):
            continue
        if bool(row.get("gt_recovered_via_extra")):
            failure_type = "recovered"
        elif bool(row.get("distractor_domination")) or int(row.get("closed_set_top1_id") or -1) in distractor_set:
            failure_type = "distractor_domination"
        elif not bool(row.get("gt_in_extra_domain")):
            failure_type = "proposal_failure"
        elif row.get("closed_set_top1_id") == "unknown":
            failure_type = "unknown_domination"
        elif row.get("closed_set_gt_rank") not in (None, 1):
            failure_type = "ranking_failure"
        else:
            failure_type = "competition_failure"
        casebook.append(
            {
                "protocol_name": protocol_name,
                "trajectory_id": str(row.get("trajectory_id")),
                "clip_id": str(row.get("clip_id")),
                "gt_class_id": row.get("gt_class_id"),
                "gt_dropped_from_observed": bool(row.get("gt_dropped_from_observed")),
                "recovered": bool(row.get("gt_recovered_via_extra")),
                "failure_type": failure_type,
                "topk_ids": list(row.get("topk_ids", [])),
                "topk_scores": list(row.get("topk_scores", [])),
                "closed_set_top1_id": row.get("closed_set_top1_id"),
                "closed_set_gt_rank": row.get("closed_set_gt_rank"),
                "closed_set_gt_score": row.get("closed_set_gt_score"),
                "observed_raw_ids": list(row.get("observed_raw_ids", [])),
                "candidate_ids_known": list(row.get("candidate_ids_known", [])),
                "candidate_ids_extra": list(row.get("candidate_ids_extra", [])),
                "candidate_proposal_source": row.get("candidate_proposal_source"),
                "invalid_reasons": list(row.get("invalid_reasons", [])),
            }
        )
    return casebook


def _emit_broader_validation_artifacts(
    *,
    repo_root: Path,
    summary: Mapping[str, Any],
    selected_class_rows: Sequence[Mapping[str, Any]],
    selected_clip_rows: Sequence[Mapping[str, Any]],
    protocol_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    ledger_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    casebook_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    audit_root = repo_root / "train" / "audit"
    artifact_root = repo_root / "codex" / "outputs" / "G7_training"
    audit_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_json(audit_root / "tier2_selected_classes.json", {"status": "PASS", "selected_classes": list(selected_class_rows)})
    _write_json(audit_root / "tier2_selected_clips.json", {"status": "PASS", "selected_clips": list(selected_clip_rows)})
    _write_json(
        audit_root / "tier2_drop_protocol_25.json",
        {
            "status": "PASS",
            "protocol_name": "tier2_25",
            "drop_ratio": 0.25,
            "drop_protocol_rows": list(protocol_rows["tier2_25"]),
            "selected_class_ids": summary["protocol_summaries"]["tier2_25"]["selected_class_ids"],
            "selected_clip_ids": summary["protocol_summaries"]["tier2_25"]["selected_clip_ids"],
        },
    )
    _write_json(
        audit_root / "tier2_drop_protocol_50.json",
        {
            "status": "PASS",
            "protocol_name": "tier2_50",
            "drop_ratio": 0.50,
            "drop_protocol_rows": list(protocol_rows["tier2_50"]),
            "selected_class_ids": summary["protocol_summaries"]["tier2_50"]["selected_class_ids"],
            "selected_clip_ids": summary["protocol_summaries"]["tier2_50"]["selected_clip_ids"],
        },
    )
    _write_jsonl(audit_root / "tier2_drop_recovery_ledger_25.jsonl", ledger_rows["tier2_25"])
    _write_jsonl(audit_root / "tier2_drop_recovery_ledger_50.jsonl", ledger_rows["tier2_50"])
    _write_json(audit_root / "tier2_drop_recovery_summary.json", summary)
    _write_jsonl(audit_root / "tier2_drop_recovery_casebook.jsonl", [*casebook_rows["tier2_25"], *casebook_rows["tier2_50"]])
    _write_json(artifact_root / "g7_broader_bounded_validation_latest.json", summary)
    (artifact_root / "g7_broader_bounded_validation_latest.md").write_text(
        "\n".join(
            [
                "# G7 Broader Bounded Validation",
                "",
                f"- status: {summary['status']}",
                f"- selected_class_count: {summary['selected_class_count']}",
                f"- selected_clip_count: {summary['selected_clip_count']}",
                f"- tier2_25_recovery_rate: {summary['protocol_summaries']['tier2_25']['dropped_class_recovery_rate']:.3f}",
                f"- tier2_50_recovery_rate: {summary['protocol_summaries']['tier2_50']['dropped_class_recovery_rate']:.3f}",
                f"- trend_from_tier1: {summary['overall_trend_from_tier1']}",
                f"- semiopen_run: {summary['semiopen_run']}",
                "",
                "## External Validity",
                summary["external_validity_statement"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_protocol(
    *,
    repo_root: Path,
    fixture_root: Path,
    protocol_name: str,
    drop_ratio: Optional[float],
    selected_clip_rows: Sequence[Mapping[str, Any]],
    class_name_lookup: Mapping[int, str],
    candidate_map: Mapping[int, Mapping[str, Any]],
    gt_lookup: Mapping[str, Mapping[str, Any]],
    protocol_type: str = "ratio_drop",
) -> Tuple[Dict[str, Any], List[Record], List[Record], Dict[str, Any], List[Record], List[Record], List[Record]]:
    protocol_root = fixture_root / protocol_name
    protocol_root.mkdir(parents=True, exist_ok=True)
    # The materialized samples store artifact-relative locators, so each protocol root
    # needs the same runtime fixture payloads as the source fixture root before training.
    _copytree(
        fixture_root,
        protocol_root,
        ["carrier_bank", "frame_bank", "text_bank", "weak_labels", "exports", "audit"],
    )
    selected_samples, selected_clip_ids, selected_observed, protocol_rows = _derive_protocol_samples(
        selected_clip_rows,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
        drop_ratio=drop_ratio,
        protocol_name=protocol_name,
        protocol_type=protocol_type,
    )
    prealign_result = train_prealign(
        output_root=protocol_root,
        materialized_samples=selected_samples,
        config=PrealignConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            device="cpu",
            seed=SMOKE_SEED,
            smoke=True,
            epochs=SMOKE_PREALIGN_EPOCHS,
            learning_rate=SMOKE_PREALIGN_LR,
            temperature=SMOKE_TEMPERATURE,
        ),
    )
    softem_result = run_soft_em(
        output_root=protocol_root,
        materialized_samples=selected_samples,
        config=SoftEMConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            mode="base_then_aug",
            device="cpu",
            seed=SMOKE_SEED,
            smoke=True,
            temperature=SMOKE_TEMPERATURE,
            em_subiterations=resolve_em_subiterations(smoke=True, explicit=None),
            base_epochs=SMOKE_BASE_EPOCHS,
            aug_epochs=SMOKE_AUG_EPOCHS,
            base_learning_rate=SMOKE_BASE_LR,
            aug_learning_rate=SMOKE_AUG_LR,
        ),
    )
    stage_rows = {
        "prealign": _evaluate_rows(
            output_root=protocol_root,
            selected_samples=selected_samples,
            gt_lookup=gt_lookup,
            stage_id="prealign",
            checkpoint_path=protocol_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth",
            protocol_name=protocol_name,
        ),
        "softem_base": [],
        "softem_aug": [],
    }
    stage_rows["softem_base"] = _evaluate_rows(
        output_root=protocol_root,
        selected_samples=selected_samples,
        gt_lookup=gt_lookup,
        stage_id="softem_base",
        checkpoint_path=protocol_root / "train" / "softem_base" / "checkpoints" / "softem_base_last.pth",
        protocol_name=protocol_name,
        previous_by_trajectory={row["trajectory_id"]: row for row in stage_rows["prealign"]},
    )
    stage_rows["softem_aug"] = _evaluate_rows(
        output_root=protocol_root,
        selected_samples=selected_samples,
        gt_lookup=gt_lookup,
        stage_id="softem_aug",
        checkpoint_path=protocol_root / "train" / "softem_aug" / "checkpoints" / "softem_aug_last.pth",
        protocol_name=protocol_name,
        previous_by_trajectory={row["trajectory_id"]: row for row in stage_rows["softem_base"]},
    )
    recovery_ledger_rows = [*stage_rows["prealign"], *stage_rows["softem_base"], *stage_rows["softem_aug"]]
    return stage_rows, recovery_ledger_rows, list(selected_samples), protocol_rows, list(selected_clip_ids.keys()), list(selected_observed.items()), [
        prealign_result,
        softem_result,
    ]


def _run_semiopen_protocol(
    *,
    repo_root: Path,
    fixture_root: Path,
    protocol_name: str,
    drop_ratio: Optional[float],
    selected_clip_rows: Sequence[Mapping[str, Any]],
    class_name_lookup: Mapping[int, str],
    candidate_map: Mapping[int, Mapping[str, Any]],
    gt_lookup: Mapping[str, Mapping[str, Any]],
    semiopen_distractor_ids: Sequence[int],
    protocol_type: str = "ratio_drop",
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Record], List[Record], List[Record], List[Record], List[Record]]:
    protocol_root = fixture_root / protocol_name
    protocol_root.mkdir(parents=True, exist_ok=True)
    _copytree(
        fixture_root,
        protocol_root,
        ["carrier_bank", "frame_bank", "text_bank", "weak_labels", "exports", "audit"],
    )
    semiopen_distractor_ids = [int(x) for x in semiopen_distractor_ids]
    selected_samples, selected_clip_ids, selected_observed, protocol_rows = _derive_protocol_samples(
        selected_clip_rows,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
        drop_ratio=drop_ratio,
        protocol_name=protocol_name,
        protocol_type=protocol_type,
        semiopen_distractor_ids=semiopen_distractor_ids,
    )
    target_eval_samples = _clone_target_only_eval_samples(selected_samples)

    prealign_result = train_prealign(
        output_root=protocol_root,
        materialized_samples=selected_samples,
        config=PrealignConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            device="cpu",
            seed=SMOKE_SEED,
            smoke=True,
            epochs=SMOKE_PREALIGN_EPOCHS,
            learning_rate=SMOKE_PREALIGN_LR,
            temperature=SMOKE_TEMPERATURE,
        ),
    )
    softem_result = run_soft_em(
        output_root=protocol_root,
        materialized_samples=selected_samples,
        config=SoftEMConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            mode="base_then_aug",
            device="cpu",
            seed=SMOKE_SEED,
            smoke=True,
            temperature=SMOKE_TEMPERATURE,
            em_subiterations=resolve_em_subiterations(smoke=True, explicit=None),
            base_epochs=SMOKE_BASE_EPOCHS,
            aug_epochs=SMOKE_AUG_EPOCHS,
            base_learning_rate=SMOKE_BASE_LR,
            aug_learning_rate=SMOKE_AUG_LR,
        ),
    )

    target_rows = {
        "prealign": _evaluate_rows(
            output_root=protocol_root,
            selected_samples=target_eval_samples,
            gt_lookup=gt_lookup,
            stage_id="prealign",
            checkpoint_path=protocol_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth",
            protocol_name=f"{protocol_name}_target_only",
        ),
        "softem_base": [],
        "softem_aug": [],
    }
    target_rows["softem_base"] = _evaluate_rows(
        output_root=protocol_root,
        selected_samples=target_eval_samples,
        gt_lookup=gt_lookup,
        stage_id="softem_base",
        checkpoint_path=protocol_root / "train" / "softem_base" / "checkpoints" / "softem_base_last.pth",
        protocol_name=f"{protocol_name}_target_only",
        previous_by_trajectory={row["trajectory_id"]: row for row in target_rows["prealign"]},
    )
    target_rows["softem_aug"] = _evaluate_rows(
        output_root=protocol_root,
        selected_samples=target_eval_samples,
        gt_lookup=gt_lookup,
        stage_id="softem_aug",
        checkpoint_path=protocol_root / "train" / "softem_aug" / "checkpoints" / "softem_aug_last.pth",
        protocol_name=f"{protocol_name}_target_only",
        previous_by_trajectory={row["trajectory_id"]: row for row in target_rows["softem_base"]},
    )

    semiopen_rows = {
        "prealign": _evaluate_rows(
            output_root=protocol_root,
            selected_samples=selected_samples,
            gt_lookup=gt_lookup,
            stage_id="prealign",
            checkpoint_path=protocol_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth",
            protocol_name=protocol_name,
        ),
        "softem_base": [],
        "softem_aug": [],
    }
    semiopen_rows["softem_base"] = _evaluate_rows(
        output_root=protocol_root,
        selected_samples=selected_samples,
        gt_lookup=gt_lookup,
        stage_id="softem_base",
        checkpoint_path=protocol_root / "train" / "softem_base" / "checkpoints" / "softem_base_last.pth",
        protocol_name=protocol_name,
        previous_by_trajectory={row["trajectory_id"]: row for row in semiopen_rows["prealign"]},
    )
    semiopen_rows["softem_aug"] = _evaluate_rows(
        output_root=protocol_root,
        selected_samples=selected_samples,
        gt_lookup=gt_lookup,
        stage_id="softem_aug",
        checkpoint_path=protocol_root / "train" / "softem_aug" / "checkpoints" / "softem_aug_last.pth",
        protocol_name=protocol_name,
        previous_by_trajectory={row["trajectory_id"]: row for row in semiopen_rows["softem_base"]},
    )

    target_by_stage = {stage_id: {row["trajectory_id"]: row for row in rows} for stage_id, rows in target_rows.items()}
    semiopen_merged_by_stage: Dict[str, List[Record]] = {}
    for stage_id, rows in semiopen_rows.items():
        merged: List[Record] = []
        for row in rows:
            target_row = target_by_stage[stage_id][row["trajectory_id"]]
            merged_row = dict(row)
            merged_row["gt_rank_target_set"] = target_row.get("gt_rank_stage_domain") or target_row.get("closed_set_gt_rank")
            merged_row["gt_rank_semiopen_set"] = row.get("gt_rank_stage_domain")
            merged_row["distractor_domination"] = bool(int(row.get("closed_set_top1_id") or -1) in set(semiopen_distractor_ids))
            merged_row["semiopen_distractor_ids"] = list(semiopen_distractor_ids)
            merged.append(merged_row)
        semiopen_merged_by_stage[stage_id] = merged

    target_ledger_rows = [*target_rows["prealign"], *target_rows["softem_base"], *target_rows["softem_aug"]]
    semiopen_ledger_rows = [*semiopen_merged_by_stage["prealign"], *semiopen_merged_by_stage["softem_base"], *semiopen_merged_by_stage["softem_aug"]]
    return (
        target_rows,
        semiopen_merged_by_stage,
        target_ledger_rows,
        semiopen_ledger_rows,
        list(selected_samples),
        protocol_rows,
        list(selected_clip_ids.keys()),
        list(selected_observed.items()),
        [prealign_result, softem_result],
    )


def _tier2_experiment_summary(
    *,
    repo_root: Path,
    selected_class_rows: Sequence[Mapping[str, Any]],
    selected_clip_rows: Sequence[Mapping[str, Any]],
    protocol_rows_25: Sequence[Mapping[str, Any]],
    protocol_rows_50: Sequence[Mapping[str, Any]],
    stage_rows_25: Mapping[str, Sequence[Mapping[str, Any]]],
    stage_rows_50: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    prior_tier1 = json.loads((repo_root / "codex" / "outputs" / "G7_training" / "g7_closed_set_drop_recovery_latest.json").read_text(encoding="utf-8"))
    prior_full = json.loads((repo_root / "codex" / "outputs" / "G7_training" / "g7_final_residual_drift_audit_latest.json").read_text(encoding="utf-8"))
    summary_25 = _aggregate_protocol_summary(
        protocol_name="tier2_25",
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        rows_by_stage=stage_rows_25,
        prior_tier1_summary=prior_tier1,
        prior_full_vocab_summary=prior_full,
    )
    summary_50 = _aggregate_protocol_summary(
        protocol_name="tier2_50",
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        rows_by_stage=stage_rows_50,
        prior_tier1_summary=prior_tier1,
        prior_full_vocab_summary=prior_full,
    )
    tier1_aug = prior_tier1.get("stage_summaries", {}).get("softem_aug", {})
    tier2_aug_25 = summary_25["stage_summaries"]["softem_aug"]
    tier2_aug_50 = summary_50["stage_summaries"]["softem_aug"]
    smooth_or_abrupt = "smooth" if float(summary_50["dropped_class_recovery_rate"]) <= float(summary_25["dropped_class_recovery_rate"]) else "abrupt"
    return {
        "status": "PASS",
        "task_id": "G7_training-task",
        "task_name": "g7_broader_bounded_validation",
        "dataset_name": "lvvis_train_base",
        "trajectory_source_branch": "mainline",
        "tier": "Tier-2",
        "class_count_target": TIER2_CLASS_COUNT,
        "clip_count_target": TIER2_CLIP_COUNT,
        "drop_ratio_targets": list(TIER2_DROP_RATIOS),
        "selected_class_count": int(len(selected_class_rows)),
        "selected_clip_count": int(len(selected_clip_rows)),
        "selected_class_ids": [int(row["class_id"]) for row in selected_class_rows],
        "selected_clip_ids": [str(row["clip_id"]) for row in selected_clip_rows],
        "selected_class_names": [str(row["class_name"]) for row in selected_class_rows],
        "selected_clip_names": [str(row["gt_class_name"]) for row in selected_clip_rows],
        "bounded_settings": {
            "smoke_max_trajectories": SMOKE_MAX_TRAJECTORIES,
            "prealign_epochs": SMOKE_PREALIGN_EPOCHS,
            "prealign_lr": SMOKE_PREALIGN_LR,
            "base_epochs": SMOKE_BASE_EPOCHS,
            "base_lr": SMOKE_BASE_LR,
            "aug_epochs": SMOKE_AUG_EPOCHS,
            "aug_lr": SMOKE_AUG_LR,
            "em_subiterations": resolve_em_subiterations(smoke=True, explicit=None),
            "temperature": SMOKE_TEMPERATURE,
            "topk": SMOKE_TOPK,
        },
        "protocol_summaries": {
            "tier2_25": summary_25,
            "tier2_50": summary_50,
        },
        "comparison_against_prior": {
            "tier1_closed_set": {
                "gt_top1_rate": tier1_aug.get("gt_top1_rate"),
                "mean_gt_rank": tier1_aug.get("mean_gt_rank"),
                "mean_gt_score": tier1_aug.get("mean_gt_score"),
            },
            "weak_full_vocab": {
                "gt_top1_rate": prior_full.get("stage_summaries", {}).get("softem_aug", {}).get("gt_top1_rate"),
                "mean_gt_rank": prior_full.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_rank"),
                "mean_gt_score": prior_full.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_score"),
            },
        },
        "overall_trend_from_tier1": smooth_or_abrupt,
        "semiopen_run": False,
        "semiopen_summary": None,
        "external_validity_statement": (
            "Tier-2 shows whether recovery remains meaningful when the closed-set grows to 10 classes and 32 clips. "
            "This does not establish unrestricted open-vocabulary robustness or formal G7 closure."
        ),
        "recommendation": (
            "proceed_to_broader_bounded_validation" if float(summary_50["dropped_class_recovery_rate"]) > 0.0 else "return_to_repair/training-strengthening"
        ),
    }


def _semiopen_protocol_summary(
    *,
    protocol_name: str,
    selected_class_rows: Sequence[Mapping[str, Any]],
    selected_clip_rows: Sequence[Mapping[str, Any]],
    selected_distractor_rows: Sequence[Mapping[str, Any]],
    target_rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]],
    semiopen_rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]],
    prior_tier1_summary: Mapping[str, Any],
    prior_tier2_summary: Mapping[str, Any],
    drop_ratio: float,
) -> Dict[str, Any]:
    tier2_protocol_key = "tier2_25" if protocol_name.endswith("_25") else "tier2_50"
    prior_tier2_protocol = prior_tier2_summary.get("protocol_summaries", {}).get(tier2_protocol_key, {})
    target_summary = _aggregate_protocol_summary(
        protocol_name=f"{protocol_name}_target_only",
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        rows_by_stage=target_rows_by_stage,
        prior_tier1_summary=prior_tier1_summary,
        prior_full_vocab_summary=prior_tier2_summary,
    )
    semiopen_summary = _aggregate_protocol_summary(
        protocol_name=protocol_name,
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        rows_by_stage=semiopen_rows_by_stage,
        prior_tier1_summary=prior_tier1_summary,
        prior_full_vocab_summary=prior_tier2_summary,
    )
    final_rows = list(semiopen_rows_by_stage.get("softem_aug", []))
    dropped_rows = [row for row in final_rows if bool(row.get("gt_dropped_from_observed"))]
    distractor_domination_rows = [row for row in dropped_rows if bool(row.get("distractor_domination"))]
    distractor_domination_rate = float(len(distractor_domination_rows) / len(dropped_rows)) if dropped_rows else 0.0
    target_aug = target_summary["stage_summaries"]["softem_aug"]
    semiopen_aug = semiopen_summary["stage_summaries"]["softem_aug"]
    tier2_aug = prior_tier2_protocol.get("stage_summaries", {}).get("softem_aug", {})
    tier1_aug = prior_tier1_summary.get("stage_summaries", {}).get("softem_aug", {})
    recovery_delta_vs_tier2 = float(semiopen_summary["dropped_class_recovery_rate"]) - float(prior_tier2_protocol.get("dropped_class_recovery_rate", 0.0))
    trend_from_tier2 = "smooth" if abs(recovery_delta_vs_tier2) <= 0.1 else "abrupt"
    return {
        "protocol_name": protocol_name,
        "status": "PASS" if dropped_rows else "EMPTY",
        "selected_class_count": int(len(selected_class_rows)),
        "selected_clip_count": int(len(selected_clip_rows)),
        "selected_class_ids": [int(row["class_id"]) for row in selected_class_rows],
        "selected_clip_ids": [str(row["clip_id"]) for row in selected_clip_rows],
        "selected_distractor_count": int(len(selected_distractor_rows)),
        "selected_distractor_ids": [int(row["class_id"]) for row in selected_distractor_rows],
        "selected_distractor_names": [str(row["class_name"]) for row in selected_distractor_rows],
        "drop_ratio": float(drop_ratio),
        "dropped_class_recovery_rate": semiopen_summary["dropped_class_recovery_rate"],
        "extra_gt_recall@K": semiopen_summary["extra_gt_recall@K"],
        "extra_precision@K": semiopen_summary["extra_precision@K"],
        "missing_class_recovery_rate": semiopen_summary["missing_class_recovery_rate"],
        "spurious_extra_rate": semiopen_summary["spurious_extra_rate"],
        "distractor_domination_rate": distractor_domination_rate,
        "trajectory_level_rows": int(len(final_rows)),
        "dropped_trajectory_rows": int(len(dropped_rows)),
        "recovered_trajectory_rows": int(len([row for row in dropped_rows if bool(row.get("gt_recovered_via_extra"))])),
        "target_set_stage_summaries": target_summary["stage_summaries"],
        "semiopen_set_stage_summaries": semiopen_summary["stage_summaries"],
        "comparison_against_prior": {
            "tier1_closed_set": {
                "gt_top1_rate": tier1_aug.get("gt_top1_rate"),
                "mean_gt_rank": tier1_aug.get("mean_gt_rank"),
                "mean_gt_score": tier1_aug.get("mean_gt_score"),
            },
            "tier2_broader_closed_set": {
                "gt_top1_rate": tier2_aug.get("gt_top1_rate"),
                "mean_gt_rank": tier2_aug.get("mean_gt_rank"),
                "mean_gt_score": tier2_aug.get("mean_gt_score"),
                "dropped_class_recovery_rate": prior_tier2_protocol.get("dropped_class_recovery_rate"),
            },
        },
        "trend_from_tier2": trend_from_tier2,
        "semiopen_run": True,
        "external_validity_statement": (
            "Semi-open validation tests whether the closed-set recovery signal survives a bounded distractor expansion. "
            "It does not establish unrestricted open-vocabulary robustness or formal G7 closure."
        ),
        "recommendation": (
            "continue_broader_bounded_validation"
            if semiopen_summary["dropped_class_recovery_rate"] > 0.0
            else "return_to_repair/training-strengthening"
        ),
    }


def _semiopen_experiment_summary(
    *,
    repo_root: Path,
    selected_class_rows: Sequence[Mapping[str, Any]],
    selected_clip_rows: Sequence[Mapping[str, Any]],
    selected_distractor_rows: Sequence[Mapping[str, Any]],
    protocol_rows_25: Sequence[Mapping[str, Any]],
    protocol_rows_50: Sequence[Mapping[str, Any]],
    target_rows_25: Mapping[str, Sequence[Mapping[str, Any]]],
    semiopen_rows_25: Mapping[str, Sequence[Mapping[str, Any]]],
    target_rows_50: Mapping[str, Sequence[Mapping[str, Any]]],
    semiopen_rows_50: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    prior_tier1 = json.loads((repo_root / "codex" / "outputs" / "G7_training" / "g7_closed_set_drop_recovery_latest.json").read_text(encoding="utf-8"))
    prior_tier2 = json.loads((repo_root / "codex" / "outputs" / "G7_training" / "g7_broader_bounded_validation_latest.json").read_text(encoding="utf-8"))
    prior_tier2_25 = prior_tier2.get("protocol_summaries", {}).get("tier2_25", {})
    prior_tier2_50 = prior_tier2.get("protocol_summaries", {}).get("tier2_50", {})
    summary_25 = _semiopen_protocol_summary(
        protocol_name="semiopen_25",
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        selected_distractor_rows=selected_distractor_rows,
        target_rows_by_stage=target_rows_25,
        semiopen_rows_by_stage=semiopen_rows_25,
        prior_tier1_summary=prior_tier1,
        prior_tier2_summary=prior_tier2,
        drop_ratio=0.25,
    )
    summary_50 = _semiopen_protocol_summary(
        protocol_name="semiopen_50",
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        selected_distractor_rows=selected_distractor_rows,
        target_rows_by_stage=target_rows_50,
        semiopen_rows_by_stage=semiopen_rows_50,
        prior_tier1_summary=prior_tier1,
        prior_tier2_summary=prior_tier2,
        drop_ratio=0.50,
    )
    recovery_delta_vs_tier2 = float(summary_50["dropped_class_recovery_rate"]) - float(prior_tier2_50.get("dropped_class_recovery_rate", 0.0))
    recovery_delta_vs_tier2_25 = float(summary_25["dropped_class_recovery_rate"]) - float(prior_tier2_25.get("dropped_class_recovery_rate", 0.0))
    trend_from_tier2 = "smooth" if abs(recovery_delta_vs_tier2) <= 0.1 and abs(recovery_delta_vs_tier2_25) <= 0.1 else "abrupt"
    selected_clip_ids = [str(row["clip_id"]) for row in selected_clip_rows]
    return {
        "status": "PASS",
        "task_id": "G7_training-task",
        "task_name": "g7_semiopen_bounded_validation",
        "dataset_name": "lvvis_train_base",
        "trajectory_source_branch": "mainline",
        "selected_class_count": int(len(selected_class_rows)),
        "selected_clip_count": int(len(selected_clip_rows)),
        "selected_class_ids": [int(row["class_id"]) for row in selected_class_rows],
        "selected_clip_ids": selected_clip_ids,
        "selected_class_names": [str(row["class_name"]) for row in selected_class_rows],
        "selected_clip_names": [str(row["gt_class_name"]) for row in selected_clip_rows],
        "selected_distractor_count": int(len(selected_distractor_rows)),
        "selected_distractor_ids": [int(row["class_id"]) for row in selected_distractor_rows],
        "selected_distractor_names": [str(row["class_name"]) for row in selected_distractor_rows],
        "protocol_summaries": {
            "semiopen_25": summary_25,
            "semiopen_50": summary_50,
        },
        "comparison_against_prior": {
            "tier1_closed_set": {
                "gt_top1_rate": prior_tier1.get("stage_summaries", {}).get("softem_aug", {}).get("gt_top1_rate"),
                "mean_gt_rank": prior_tier1.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_rank"),
                "mean_gt_score": prior_tier1.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_score"),
            },
            "tier2_broader_closed_set": {
                "gt_top1_rate": prior_tier2_50.get("stage_summaries", {}).get("softem_aug", {}).get("gt_top1_rate"),
                "mean_gt_rank": prior_tier2_50.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_rank"),
                "mean_gt_score": prior_tier2_50.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_score"),
                "dropped_class_recovery_rate": prior_tier2_50.get("dropped_class_recovery_rate"),
            },
            "tier2_broader_closed_set_25": {
                "gt_top1_rate": prior_tier2_25.get("stage_summaries", {}).get("softem_aug", {}).get("gt_top1_rate"),
                "mean_gt_rank": prior_tier2_25.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_rank"),
                "mean_gt_score": prior_tier2_25.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_score"),
                "dropped_class_recovery_rate": prior_tier2_25.get("dropped_class_recovery_rate"),
            },
        },
        "overall_trend_from_tier2": trend_from_tier2,
        "semiopen_run": True,
        "external_validity_statement": (
            "Semi-open validation tests whether the closed-set recovery signal survives a bounded distractor expansion. "
            "It does not establish unrestricted open-vocabulary robustness or formal G7 closure."
        ),
        "recommendation": (
            "continue_broader_bounded_validation"
            if float(summary_50["dropped_class_recovery_rate"]) > 0.0
            else "return_to_repair/training-strengthening"
        ),
    }


def _emit_semiopen_artifacts(
    *,
    repo_root: Path,
    summary: Mapping[str, Any],
    selected_distractor_rows: Sequence[Mapping[str, Any]],
    protocol_rows_25: Sequence[Mapping[str, Any]],
    protocol_rows_50: Sequence[Mapping[str, Any]],
    target_ledger_rows_25: Sequence[Mapping[str, Any]],
    target_ledger_rows_50: Sequence[Mapping[str, Any]],
    semiopen_ledger_rows_25: Sequence[Mapping[str, Any]],
    semiopen_ledger_rows_50: Sequence[Mapping[str, Any]],
    semiopen_casebook_rows_25: Sequence[Mapping[str, Any]],
    semiopen_casebook_rows_50: Sequence[Mapping[str, Any]],
) -> None:
    audit_root = repo_root / "train" / "audit"
    artifact_root = repo_root / "codex" / "outputs" / "G7_training"
    audit_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_json(audit_root / "semiopen_selected_distractors.json", {"status": "PASS", "selected_distractors": list(selected_distractor_rows)})
    _write_json(audit_root / "semiopen_drop_protocol_25.json", {"status": "PASS", "protocol_name": "semiopen_25", "drop_ratio": 0.25, "drop_protocol_rows": list(protocol_rows_25)})
    _write_json(audit_root / "semiopen_drop_protocol_50.json", {"status": "PASS", "protocol_name": "semiopen_50", "drop_ratio": 0.50, "drop_protocol_rows": list(protocol_rows_50)})
    _write_jsonl(audit_root / "semiopen_drop_recovery_ledger_25.jsonl", semiopen_ledger_rows_25)
    _write_jsonl(audit_root / "semiopen_drop_recovery_ledger_50.jsonl", semiopen_ledger_rows_50)
    _write_jsonl(audit_root / "semiopen_drop_recovery_ledger.jsonl", [*semiopen_ledger_rows_25, *semiopen_ledger_rows_50])
    _write_json(audit_root / "semiopen_drop_recovery_summary.json", summary)
    _write_jsonl(audit_root / "semiopen_drop_recovery_casebook.jsonl", [*semiopen_casebook_rows_25, *semiopen_casebook_rows_50])
    _write_json(artifact_root / "g7_semiopen_bounded_validation_latest.json", summary)
    (artifact_root / "g7_semiopen_bounded_validation_latest.md").write_text(
        "\n".join(
            [
                "# G7 Semi-Open Bounded Validation",
                "",
                f"- status: {summary['status']}",
                f"- selected_class_count: {summary['selected_class_count']}",
                f"- selected_clip_count: {summary['selected_clip_count']}",
                f"- selected_distractor_count: {summary['selected_distractor_count']}",
                f"- semiopen_25_recovery_rate: {summary['protocol_summaries']['semiopen_25']['dropped_class_recovery_rate']:.3f}",
                f"- semiopen_50_recovery_rate: {summary['protocol_summaries']['semiopen_50']['dropped_class_recovery_rate']:.3f}",
                f"- trend_from_tier2: {summary['overall_trend_from_tier2']}",
                "",
                "## External Validity",
                summary["external_validity_statement"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stressA_run_summary(
    *,
    repo_root: Path,
    split_id: str,
    setting: str,
    protocol_type: str,
    drop_ratio: Optional[float],
    distractor_count: int,
    selected_class_rows: Sequence[Mapping[str, Any]],
    selected_clip_rows: Sequence[Mapping[str, Any]],
    selected_distractor_rows: Sequence[Mapping[str, Any]],
    rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]],
    prior_tier1_summary: Mapping[str, Any],
    prior_tier2_summary: Mapping[str, Any],
    prior_semiopen_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    protocol_name = f"{split_id}_{setting}_{protocol_type}_{'d%02d' % int(distractor_count)}"
    final_rows = list(rows_by_stage.get("softem_aug", []))
    dropped_rows = [row for row in final_rows if bool(row.get("gt_dropped_from_observed"))]
    distractor_domination_rows = [row for row in dropped_rows if bool(row.get("distractor_domination"))]
    distractor_domination_rate = float(len(distractor_domination_rows) / len(dropped_rows)) if dropped_rows else 0.0
    base_summary = _aggregate_protocol_summary(
        protocol_name=protocol_name,
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        rows_by_stage=rows_by_stage,
        prior_tier1_summary=prior_tier1_summary,
        prior_full_vocab_summary=prior_tier2_summary,
    )
    tier1_aug = prior_tier1_summary.get("stage_summaries", {}).get("softem_aug", {})
    tier2_25 = prior_tier2_summary.get("protocol_summaries", {}).get("tier2_25", {})
    tier2_50 = prior_tier2_summary.get("protocol_summaries", {}).get("tier2_50", {})
    prior_semiopen_25 = prior_semiopen_summary.get("protocol_summaries", {}).get("semiopen_25", {})
    prior_semiopen_50 = prior_semiopen_summary.get("protocol_summaries", {}).get("semiopen_50", {})
    base_summary.update(
        {
            "split_id": str(split_id),
            "setting": str(setting),
            "protocol_type": str(protocol_type),
            "drop_ratio": float(drop_ratio) if drop_ratio is not None else None,
            "distractor_count": int(distractor_count),
            "selected_distractor_ids": [int(row["class_id"]) for row in selected_distractor_rows],
            "selected_distractor_names": [str(row["class_name"]) for row in selected_distractor_rows],
            "distractor_domination_rate": distractor_domination_rate if str(setting) == "semi_open" else 0.0,
            "comparison_against_prior": {
                "tier1_closed_set": {
                    "gt_top1_rate": tier1_aug.get("gt_top1_rate"),
                    "mean_gt_rank": tier1_aug.get("mean_gt_rank"),
                    "mean_gt_score": tier1_aug.get("mean_gt_score"),
                },
                "tier2_broader_bounded_25": {
                    "gt_top1_rate": tier2_25.get("stage_summaries", {}).get("softem_aug", {}).get("gt_top1_rate"),
                    "mean_gt_rank": tier2_25.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_rank"),
                    "mean_gt_score": tier2_25.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_score"),
                    "dropped_class_recovery_rate": tier2_25.get("dropped_class_recovery_rate"),
                },
                "tier2_broader_bounded_50": {
                    "gt_top1_rate": tier2_50.get("stage_summaries", {}).get("softem_aug", {}).get("gt_top1_rate"),
                    "mean_gt_rank": tier2_50.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_rank"),
                    "mean_gt_score": tier2_50.get("stage_summaries", {}).get("softem_aug", {}).get("mean_gt_score"),
                    "dropped_class_recovery_rate": tier2_50.get("dropped_class_recovery_rate"),
                },
                "prior_semiopen_bounded": {
                    "semiopen_25_recovery_rate": prior_semiopen_25.get("dropped_class_recovery_rate"),
                    "semiopen_50_recovery_rate": prior_semiopen_50.get("dropped_class_recovery_rate"),
                    "trend_from_tier2": prior_semiopen_summary.get("overall_trend_from_tier2"),
                },
            },
        }
    )
    return base_summary


def _aggregate_stressA_cross_summaries(run_summaries: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    by_setting: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_split: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for summary in run_summaries:
        by_setting[str(summary.get("setting"))].append(summary)
        by_split[str(summary.get("split_id"))].append(summary)

    def _mean(values: Sequence[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    cross_setting = {
        "status": "PASS" if run_summaries else "EMPTY",
        "setting_summaries": {},
    }
    for setting, rows in by_setting.items():
        recovery_rates = [float(row.get("dropped_class_recovery_rate", 0.0)) for row in rows]
        topk = [float(row.get("extra_gt_recall@K", 0.0)) for row in rows]
        precision = [float(row.get("extra_precision@K", 0.0)) for row in rows]
        cross_setting["setting_summaries"][setting] = {
            "run_count": len(rows),
            "mean_dropped_class_recovery_rate": _mean(recovery_rates),
            "mean_extra_gt_recall@K": _mean(topk),
            "mean_extra_precision@K": _mean(precision),
            "mean_spurious_extra_rate": _mean([float(row.get("spurious_extra_rate", 0.0)) for row in rows]),
            "mean_distractor_domination_rate": _mean([float(row.get("distractor_domination_rate", 0.0)) for row in rows]),
        }

    cross_split = {
        "status": "PASS" if run_summaries else "EMPTY",
        "split_summaries": {},
    }
    for split_id, rows in by_split.items():
        recovery_rates = [float(row.get("dropped_class_recovery_rate", 0.0)) for row in rows]
        cross_split["split_summaries"][split_id] = {
            "run_count": len(rows),
            "mean_dropped_class_recovery_rate": _mean(recovery_rates),
            "mean_extra_gt_recall@K": _mean([float(row.get("extra_gt_recall@K", 0.0)) for row in rows]),
            "mean_extra_precision@K": _mean([float(row.get("extra_precision@K", 0.0)) for row in rows]),
            "mean_spurious_extra_rate": _mean([float(row.get("spurious_extra_rate", 0.0)) for row in rows]),
            "mean_distractor_domination_rate": _mean([float(row.get("distractor_domination_rate", 0.0)) for row in rows]),
        }

    return cross_setting, cross_split


def _emit_stressA_artifacts(
    *,
    repo_root: Path,
    matrix_manifest: Mapping[str, Any],
    selected_targets_by_split: Mapping[str, Any],
    selected_clips_by_split: Mapping[str, Any],
    protocols_by_split: Mapping[str, Any],
    run_summaries: Sequence[Mapping[str, Any]],
    cross_setting_summary: Mapping[str, Any],
    cross_split_summary: Mapping[str, Any],
    casebook_rows: Sequence[Mapping[str, Any]],
) -> None:
    audit_root = repo_root / "train" / "audit"
    artifact_root = repo_root / "codex" / "outputs" / "G7_training"
    audit_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_json(audit_root / "stressA_matrix_manifest.json", matrix_manifest)
    _write_json(audit_root / "stressA_selected_targets_by_split.json", selected_targets_by_split)
    _write_json(audit_root / "stressA_selected_clips_by_split.json", selected_clips_by_split)
    _write_json(audit_root / "stressA_protocols_by_split.json", protocols_by_split)
    _write_json(audit_root / "stressA_run_summaries.json", {"status": "PASS", "run_summaries": list(run_summaries)})
    _write_json(audit_root / "stressA_cross_setting_summary.json", cross_setting_summary)
    _write_json(audit_root / "stressA_cross_split_robustness_summary.json", cross_split_summary)
    _write_jsonl(audit_root / "stressA_casebook.jsonl", casebook_rows)
    _write_json(
        artifact_root / "g7_stress_validation_trancheA_latest.json",
        {
            "status": "PASS",
            "matrix_manifest": matrix_manifest,
            "cross_setting_summary": cross_setting_summary,
            "cross_split_robustness_summary": cross_split_summary,
            "selected_target_splits": list(selected_targets_by_split.keys()),
            "selected_clip_splits": list(selected_clips_by_split.keys()),
            "summary_count": len(run_summaries),
        },
    )
    (artifact_root / "g7_stress_validation_trancheA_latest.md").write_text(
        "\n".join(
            [
                "# G7 Stress Validation Tranche A",
                "",
                f"- status: PASS",
                f"- split_count: {len(selected_targets_by_split)}",
                f"- run_count: {len(run_summaries)}",
                f"- cross_setting_status: {cross_setting_summary.get('status')}",
                f"- cross_split_status: {cross_split_summary.get('status')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _load_ratio_drop_baselines(repo_root: Path) -> Dict[str, Any]:
    prior_tier1 = json.loads((repo_root / "codex" / "outputs" / "G7_training" / "g7_closed_set_drop_recovery_latest.json").read_text(encoding="utf-8"))
    prior_tier2 = json.loads((repo_root / "codex" / "outputs" / "G7_training" / "g7_broader_bounded_validation_latest.json").read_text(encoding="utf-8"))
    prior_semiopen = json.loads((repo_root / "codex" / "outputs" / "G7_training" / "g7_semiopen_bounded_validation_latest.json").read_text(encoding="utf-8"))
    stressa_runs_path = repo_root / "train" / "audit" / "stressA_run_summaries.json"
    stressa_runs = json.loads(stressa_runs_path.read_text(encoding="utf-8")).get("run_summaries", []) if stressa_runs_path.is_file() else []
    ratio_drop_runs = [row for row in stressa_runs if str(row.get("protocol_type")) == "ratio_drop"]
    ratio_drop_by_setting: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in ratio_drop_runs:
        ratio_drop_by_setting[str(row.get("setting"))].append(row)

    def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        return float(np.mean([float(row.get(key, 0.0)) for row in rows])) if rows else None

    stressa_ratio_drop = {
        "status": "PASS" if ratio_drop_runs else "EMPTY",
        "setting_summaries": {
            setting: {
                "run_count": len(rows),
                "mean_dropped_class_recovery_rate": _mean(rows, "dropped_class_recovery_rate"),
                "mean_extra_gt_recall@K": _mean(rows, "extra_gt_recall@K"),
                "mean_extra_precision@K": _mean(rows, "extra_precision@K"),
                "mean_spurious_extra_rate": _mean(rows, "spurious_extra_rate"),
                "mean_distractor_domination_rate": _mean(rows, "distractor_domination_rate"),
            }
            for setting, rows in ratio_drop_by_setting.items()
        },
    }
    return {
        "tier1_closed_set": prior_tier1,
        "tier2_broader_bounded": prior_tier2,
        "semiopen_plus4": prior_semiopen,
        "stressA_ratio_drop": stressa_ratio_drop,
    }


def _mainline_ratio_drop_run_summary(
    *,
    stage_label: str,
    split_id: str,
    setting: str,
    protocol_type: str,
    drop_ratio: float,
    distractor_count: int,
    selected_class_rows: Sequence[Mapping[str, Any]],
    selected_clip_rows: Sequence[Mapping[str, Any]],
    selected_distractor_rows: Sequence[Mapping[str, Any]],
    rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]],
    prior_baselines: Mapping[str, Any],
) -> Dict[str, Any]:
    protocol_name = f"{stage_label}_{split_id}_{setting}_{protocol_type}_{'d%02d' % int(distractor_count)}"
    final_rows = list(rows_by_stage.get("softem_aug", []))
    dropped_rows = [row for row in final_rows if bool(row.get("gt_dropped_from_observed"))]
    distractor_domination_rows = [row for row in dropped_rows if bool(row.get("distractor_domination"))]
    distractor_domination_rate = float(len(distractor_domination_rows) / len(dropped_rows)) if dropped_rows else 0.0
    base_summary = _aggregate_protocol_summary(
        protocol_name=protocol_name,
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        rows_by_stage=rows_by_stage,
        prior_tier1_summary=prior_baselines["tier1_closed_set"],
        prior_full_vocab_summary=prior_baselines["tier2_broader_bounded"],
    )
    tier1_aug = prior_baselines["tier1_closed_set"].get("stage_summaries", {}).get("softem_aug", {})
    tier2_25 = prior_baselines["tier2_broader_bounded"].get("protocol_summaries", {}).get("tier2_25", {})
    tier2_50 = prior_baselines["tier2_broader_bounded"].get("protocol_summaries", {}).get("tier2_50", {})
    semiopen_25 = prior_baselines["semiopen_plus4"].get("protocol_summaries", {}).get("semiopen_25", {})
    semiopen_50 = prior_baselines["semiopen_plus4"].get("protocol_summaries", {}).get("semiopen_50", {})
    stressa_ratio_drop = prior_baselines["stressA_ratio_drop"].get("setting_summaries", {})
    base_summary.update(
        {
            "stage_label": str(stage_label),
            "split_id": str(split_id),
            "setting": str(setting),
            "protocol_type": str(protocol_type),
            "drop_ratio": float(drop_ratio),
            "distractor_count": int(distractor_count),
            "selected_distractor_ids": [int(row["class_id"]) for row in selected_distractor_rows],
            "selected_distractor_names": [str(row["class_name"]) for row in selected_distractor_rows],
            "distractor_domination_rate": distractor_domination_rate if str(setting) == "semi_open" else 0.0,
            "comparison_against_prior": {
                "tier1_closed_set": {
                    "gt_top1_rate": tier1_aug.get("gt_top1_rate"),
                    "mean_gt_rank": tier1_aug.get("mean_gt_rank"),
                    "mean_gt_score": tier1_aug.get("mean_gt_score"),
                },
                "tier2_broader_closed_set": {
                    "tier2_25_recovery_rate": tier2_25.get("dropped_class_recovery_rate"),
                    "tier2_50_recovery_rate": tier2_50.get("dropped_class_recovery_rate"),
                    "tier2_25_gt_top1_rate": tier2_25.get("stage_summaries", {}).get("softem_aug", {}).get("gt_top1_rate"),
                    "tier2_50_gt_top1_rate": tier2_50.get("stage_summaries", {}).get("softem_aug", {}).get("gt_top1_rate"),
                },
                "semiopen_plus4": {
                    "semiopen_25_recovery_rate": semiopen_25.get("dropped_class_recovery_rate"),
                    "semiopen_50_recovery_rate": semiopen_50.get("dropped_class_recovery_rate"),
                    "semiopen_25_distractor_domination_rate": semiopen_25.get("distractor_domination_rate"),
                    "semiopen_50_distractor_domination_rate": semiopen_50.get("distractor_domination_rate"),
                },
                "trancheA_ratio_drop": {
                    "closed_set_mean_recovery": stressa_ratio_drop.get("closed_set", {}).get("mean_dropped_class_recovery_rate"),
                    "semi_open_mean_recovery": stressa_ratio_drop.get("semi_open", {}).get("mean_dropped_class_recovery_rate"),
                    "closed_set_mean_distractor_domination_rate": stressa_ratio_drop.get("closed_set", {}).get("mean_distractor_domination_rate"),
                    "semi_open_mean_distractor_domination_rate": stressa_ratio_drop.get("semi_open", {}).get("mean_distractor_domination_rate"),
                },
            },
        }
    )
    base_summary["trend_from_prior"] = base_summary.get("trend_from_tier1")
    return base_summary


def _aggregate_mainline_ratio_drop_stage(run_summaries: Sequence[Mapping[str, Any]], stage_label: str) -> Dict[str, Any]:
    stage_rows = [row for row in run_summaries if str(row.get("stage_label")) == str(stage_label)]
    by_setting: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_split: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        by_setting[str(row.get("setting"))].append(row)
        by_split[str(row.get("split_id"))].append(row)

    def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        return float(np.mean([float(row.get(key, 0.0)) for row in rows])) if rows else None

    setting_summaries: Dict[str, Any] = {}
    for setting, rows in by_setting.items():
        setting_summaries[setting] = {
            "run_count": len(rows),
            "mean_dropped_class_recovery_rate": _mean(rows, "dropped_class_recovery_rate"),
            "mean_extra_gt_recall@K": _mean(rows, "extra_gt_recall@K"),
            "mean_extra_precision@K": _mean(rows, "extra_precision@K"),
            "mean_spurious_extra_rate": _mean(rows, "spurious_extra_rate"),
            "mean_distractor_domination_rate": _mean(rows, "distractor_domination_rate"),
        }
    split_summaries: Dict[str, Any] = {}
    for split_id, rows in by_split.items():
        split_summaries[split_id] = {
            "run_count": len(rows),
            "mean_dropped_class_recovery_rate": _mean(rows, "dropped_class_recovery_rate"),
            "mean_extra_gt_recall@K": _mean(rows, "extra_gt_recall@K"),
            "mean_extra_precision@K": _mean(rows, "extra_precision@K"),
            "mean_spurious_extra_rate": _mean(rows, "spurious_extra_rate"),
            "mean_distractor_domination_rate": _mean(rows, "distractor_domination_rate"),
        }
    return {
        "status": "PASS" if stage_rows else "EMPTY",
        "stage_label": str(stage_label),
        "run_count": len(stage_rows),
        "setting_summaries": setting_summaries,
        "split_summaries": split_summaries,
    }


def _emit_mainline_ratio_drop_artifacts(
    *,
    repo_root: Path,
    stage_label: str,
    matrix_manifest: Mapping[str, Any],
    selected_targets_by_split: Mapping[str, Any],
    selected_clips_by_split: Mapping[str, Any],
    protocols_by_split: Mapping[str, Any],
    run_summaries: Sequence[Mapping[str, Any]],
    cross_setting_summary: Mapping[str, Any],
    cross_split_summary: Mapping[str, Any],
    casebook_rows: Sequence[Mapping[str, Any]],
) -> None:
    audit_root = repo_root / "train" / "audit"
    artifact_root = repo_root / "codex" / "outputs" / "G7_training"
    audit_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_json(audit_root / f"mainline{stage_label}_selected_targets_by_split.json", selected_targets_by_split)
    _write_json(audit_root / f"mainline{stage_label}_selected_clips_by_split.json", selected_clips_by_split)
    _write_json(audit_root / f"mainline{stage_label}_protocols_by_split.json", protocols_by_split)
    _write_json(audit_root / f"mainline{stage_label}_run_summaries.json", {"status": "PASS", "run_summaries": list(run_summaries)})
    _write_json(audit_root / f"mainline{stage_label}_cross_setting_summary.json", cross_setting_summary)
    _write_json(audit_root / f"mainline{stage_label}_cross_split_robustness_summary.json", cross_split_summary)
    _write_json(audit_root / f"mainline{stage_label}_matrix_manifest.json", matrix_manifest)
    _write_json(
        artifact_root / f"g7_mainline_bounded_tranche{stage_label}_latest.json",
        {
            "status": "PASS",
            "stage_label": stage_label,
            "matrix_manifest": matrix_manifest,
            "cross_setting_summary": cross_setting_summary,
            "cross_split_robustness_summary": cross_split_summary,
            "selected_target_splits": list(selected_targets_by_split.keys()),
            "selected_clip_splits": list(selected_clips_by_split.keys()),
            "summary_count": len(run_summaries),
        },
    )
    (artifact_root / f"g7_mainline_bounded_tranche{stage_label}_latest.md").write_text(
        "\n".join(
            [
                f"# G7 Mainline Bounded Validation Tranche {stage_label}",
                "",
                f"- status: PASS",
                f"- split_count: {len(selected_targets_by_split)}",
                f"- run_count: {len(run_summaries)}",
                f"- cross_setting_status: {cross_setting_summary.get('status')}",
                f"- cross_split_status: {cross_split_summary.get('status')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_mainline_ratio_drop_stage(
    *,
    repo_root: Path,
    fixture_root: Path,
    stage_label: str,
    split_ids: Sequence[str],
    target_class_ids: Sequence[int],
    distractor_class_ids: Sequence[int],
    clip_count_target: int,
    start_clip_id: int,
    prior_baselines: Mapping[str, Any],
    distractor_counts: Sequence[int],
) -> Dict[str, Any]:
    fixture = _write_mainline_ratio_drop_fixture(
        fixture_root,
        split_ids=split_ids,
        target_class_ids=target_class_ids,
        distractor_class_ids=distractor_class_ids,
        clip_count_target=clip_count_target,
        start_clip_id=start_clip_id,
        generator_tag=f"synthetic_mainline_{stage_label}",
    )
    materialized = materialize_phase1_training_samples(
        fixture_root,
        Phase1MaterializationConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            smoke=True,
            smoke_max_trajectories=clip_count_target * len(split_ids),
        ),
    )
    gt_lookup = load_gt_sidecar_lookup(fixture_root, dataset_name="lvvis_train_base", trajectory_source_branch="mainline")
    class_name_lookup = dict(fixture["class_names"])
    candidate_map = _build_candidate_map(fixture_root)

    selected_targets_by_split: Dict[str, Any] = {}
    selected_clips_by_split: Dict[str, Any] = {}
    protocols_by_split: Dict[str, Any] = {}
    run_summaries: List[Dict[str, Any]] = []
    casebook_rows: List[Dict[str, Any]] = []
    matrix_run_specs: List[Dict[str, Any]] = []

    for split_id in split_ids:
        split_clip_ids = fixture["split_to_clip_ids"][split_id]
        split_selected_class_rows, split_selected_clip_rows, _ = _select_split_tier2_clips(
            materialized["samples"],
            gt_lookup,
            clip_ids=split_clip_ids,
            class_count_target=len(target_class_ids),
            clip_count_target=clip_count_target,
            class_name_lookup=class_name_lookup,
        )
        selected_targets_by_split[split_id] = {
            "status": "PASS",
            "selected_classes": list(split_selected_class_rows),
        }
        selected_clips_by_split[split_id] = {
            "status": "PASS",
            "selected_clips": list(split_selected_clip_rows),
        }
        protocols_by_split[split_id] = {
            "status": "PASS",
            "selected_class_ids": [int(row["class_id"]) for row in split_selected_class_rows],
            "selected_clip_ids": [str(row["clip_id"]) for row in split_selected_clip_rows],
            "protocol_specs": [],
        }

        for protocol_name, drop_ratio in [("ratio_drop_25", 0.25), ("ratio_drop_50", 0.50)]:
            closed_root_name = f"{stage_label}_{split_id}_closed_set_{protocol_name}"
            matrix_run_specs.append(
                {
                    "run_id": closed_root_name,
                    "stage_label": stage_label,
                    "split_id": split_id,
                    "setting": "closed_set",
                    "protocol_type": "ratio_drop",
                    "drop_ratio": drop_ratio,
                    "distractor_count": 0,
                }
            )
            stage_rows, ledger_rows, selected_samples, protocol_rows, _, _, _ = _run_protocol(
                repo_root=repo_root,
                fixture_root=fixture_root,
                protocol_name=closed_root_name,
                drop_ratio=drop_ratio,
                selected_clip_rows=split_selected_clip_rows,
                class_name_lookup=class_name_lookup,
                candidate_map=candidate_map,
                gt_lookup=gt_lookup,
                protocol_type="ratio_drop",
            )
            summary = _mainline_ratio_drop_run_summary(
                stage_label=stage_label,
                split_id=split_id,
                setting="closed_set",
                protocol_type="ratio_drop",
                drop_ratio=drop_ratio,
                distractor_count=0,
                selected_class_rows=split_selected_class_rows,
                selected_clip_rows=split_selected_clip_rows,
                selected_distractor_rows=[],
                rows_by_stage=stage_rows,
                prior_baselines=prior_baselines,
            )
            run_summary = {
                "run_id": closed_root_name,
                "stage_label": stage_label,
                "split_id": split_id,
                "setting": "closed_set",
                "protocol_type": "ratio_drop",
                "drop_ratio": drop_ratio,
                "distractor_count": 0,
                "selected_class_ids": [int(row["class_id"]) for row in split_selected_class_rows],
                "selected_clip_ids": [str(row["clip_id"]) for row in split_selected_clip_rows],
                "selected_distractor_ids": [],
                "selected_distractor_names": [],
                "stage_summaries": summary["stage_summaries"],
                "dropped_class_recovery_rate": summary["dropped_class_recovery_rate"],
                "extra_gt_recall@K": summary["extra_gt_recall@K"],
                "extra_precision@K": summary["extra_precision@K"],
                "missing_class_recovery_rate": summary["missing_class_recovery_rate"],
                "spurious_extra_rate": summary["spurious_extra_rate"],
                "distractor_domination_rate": 0.0,
                "comparison_against_prior": summary["comparison_against_prior"],
                "trend_from_prior": summary["trend_from_prior"],
                "top_failed_runs": [],
                "representative_recovered_examples": [],
                "representative_distractor_dominated_failures": [],
                "representative_single_observed_failures": [],
            }
            run_summaries.append(run_summary)
            casebook_rows.extend(
                {
                    **row,
                    "run_id": closed_root_name,
                    "stage_label": stage_label,
                    "split_id": split_id,
                    "setting": "closed_set",
                    "protocol_type": "ratio_drop",
                    "drop_ratio": drop_ratio,
                    "distractor_count": 0,
                }
                for row in _build_casebook(stage_rows, closed_root_name)
            )
            protocols_by_split[split_id]["protocol_specs"].append(
                {
                    "run_id": closed_root_name,
                    "setting": "closed_set",
                    "protocol_type": "ratio_drop",
                    "drop_ratio": drop_ratio,
                    "distractor_count": 0,
                    "protocol_rows": list(protocol_rows),
                    "selected_samples": list(selected_samples),
                }
            )

            for distractor_count in distractor_counts:
                semi_distractor_rows = _select_semiopen_distractors(
                    class_name_lookup,
                    count=int(distractor_count),
                    source_class_ids=distractor_class_ids,
                )
                semi_root_name = f"{stage_label}_{split_id}_semi_open_{protocol_name}_d{int(distractor_count)}"
                matrix_run_specs.append(
                    {
                        "run_id": semi_root_name,
                        "stage_label": stage_label,
                        "split_id": split_id,
                        "setting": "semi_open",
                        "protocol_type": "ratio_drop",
                        "drop_ratio": drop_ratio,
                        "distractor_count": int(distractor_count),
                    }
                )
                target_rows, semiopen_rows, target_ledger_rows, semiopen_ledger_rows, selected_samples_semi, semi_protocol_rows, _, _, _ = _run_semiopen_protocol(
                    repo_root=repo_root,
                    fixture_root=fixture_root,
                    protocol_name=semi_root_name,
                    drop_ratio=drop_ratio,
                    selected_clip_rows=split_selected_clip_rows,
                    class_name_lookup=class_name_lookup,
                    candidate_map=candidate_map,
                    gt_lookup=gt_lookup,
                    semiopen_distractor_ids=[int(row["class_id"]) for row in semi_distractor_rows],
                    protocol_type="ratio_drop",
                )
                semi_summary = _mainline_ratio_drop_run_summary(
                    stage_label=stage_label,
                    split_id=split_id,
                    setting="semi_open",
                    protocol_type="ratio_drop",
                    drop_ratio=drop_ratio,
                    distractor_count=int(distractor_count),
                    selected_class_rows=split_selected_class_rows,
                    selected_clip_rows=split_selected_clip_rows,
                    selected_distractor_rows=semi_distractor_rows,
                    rows_by_stage=semiopen_rows,
                    prior_baselines=prior_baselines,
                )
                run_summary = {
                    "run_id": semi_root_name,
                    "stage_label": stage_label,
                    "split_id": split_id,
                    "setting": "semi_open",
                    "protocol_type": "ratio_drop",
                    "drop_ratio": drop_ratio,
                    "distractor_count": int(distractor_count),
                    "selected_class_ids": [int(row["class_id"]) for row in split_selected_class_rows],
                    "selected_clip_ids": [str(row["clip_id"]) for row in split_selected_clip_rows],
                    "selected_distractor_ids": [int(row["class_id"]) for row in semi_distractor_rows],
                    "selected_distractor_names": [str(row["class_name"]) for row in semi_distractor_rows],
                    "stage_summaries": semi_summary["stage_summaries"],
                    "dropped_class_recovery_rate": semi_summary["dropped_class_recovery_rate"],
                    "extra_gt_recall@K": semi_summary["extra_gt_recall@K"],
                    "extra_precision@K": semi_summary["extra_precision@K"],
                    "missing_class_recovery_rate": semi_summary["missing_class_recovery_rate"],
                    "spurious_extra_rate": semi_summary["spurious_extra_rate"],
                    "distractor_domination_rate": semi_summary["distractor_domination_rate"],
                    "comparison_against_prior": semi_summary["comparison_against_prior"],
                    "trend_from_prior": semi_summary["trend_from_prior"],
                    "top_failed_runs": [],
                    "representative_recovered_examples": [],
                    "representative_distractor_dominated_failures": [],
                    "representative_single_observed_failures": [],
                }
                run_summaries.append(run_summary)
                casebook_rows.extend(
                    {
                        **row,
                        "run_id": semi_root_name,
                        "stage_label": stage_label,
                        "split_id": split_id,
                        "setting": "semi_open",
                        "protocol_type": "ratio_drop",
                        "drop_ratio": drop_ratio,
                        "distractor_count": int(distractor_count),
                    }
                    for row in _build_casebook(semiopen_rows, semi_root_name, distractor_ids=[int(row["class_id"]) for row in semi_distractor_rows])
                )
                protocols_by_split[split_id]["protocol_specs"].append(
                    {
                        "run_id": semi_root_name,
                        "setting": "semi_open",
                        "protocol_type": "ratio_drop",
                        "drop_ratio": drop_ratio,
                        "distractor_count": int(distractor_count),
                        "selected_distractor_ids": [int(row["class_id"]) for row in semi_distractor_rows],
                        "selected_distractor_names": [str(row["class_name"]) for row in semi_distractor_rows],
                        "target_rows": list(target_rows),
                        "semiopen_rows": list(semiopen_rows),
                        "target_ledger_rows": list(target_ledger_rows),
                        "semiopen_ledger_rows": list(semiopen_ledger_rows),
                        "selected_samples": list(selected_samples_semi),
                        "protocol_rows": list(semi_protocol_rows),
                    }
                )

    cross_setting_summary, cross_split_summary = _aggregate_stressA_cross_summaries(run_summaries)
    matrix_manifest = {
        "status": "PASS",
        "stage_label": stage_label,
        "task_name": "g7_mainline_bounded_trancheB",
        "dataset_name": "lvvis_train_base",
        "trajectory_source_branch": "mainline",
        "target_class_count": len(target_class_ids),
        "clip_count_target": clip_count_target,
        "split_ids": list(split_ids),
        "protocol_types": ["ratio_drop_25", "ratio_drop_50"],
        "distractor_count_targets": list(distractor_counts),
        "settings": ["closed_set", "semi_open"],
        "closed_set_distractor_count": 0,
        "semi_open_distractor_counts": list(distractor_counts),
        "run_count": int(len(run_summaries)),
        "closed_run_count": int(len([row for row in run_summaries if row.get("setting") == "closed_set"])),
        "semiopen_run_count": int(len([row for row in run_summaries if row.get("setting") == "semi_open"])),
        "run_specs": list(matrix_run_specs),
        "recovery_protocols": ["ratio_drop_25", "ratio_drop_50"],
    }
    _emit_mainline_ratio_drop_artifacts(
        repo_root=repo_root,
        stage_label=stage_label,
        matrix_manifest=matrix_manifest,
        selected_targets_by_split=selected_targets_by_split,
        selected_clips_by_split=selected_clips_by_split,
        protocols_by_split=protocols_by_split,
        run_summaries=run_summaries,
        cross_setting_summary=cross_setting_summary,
        cross_split_summary=cross_split_summary,
        casebook_rows=casebook_rows,
    )
    return {
        "stage_label": stage_label,
        "matrix_manifest": matrix_manifest,
        "selected_targets_by_split": selected_targets_by_split,
        "selected_clips_by_split": selected_clips_by_split,
        "protocols_by_split": protocols_by_split,
        "run_summaries": run_summaries,
        "cross_setting_summary": cross_setting_summary,
        "cross_split_summary": cross_split_summary,
        "casebook_rows": casebook_rows,
    }


def _build_mainline_ratio_drop_cross_stage_summary(
    *,
    stage_results: Sequence[Mapping[str, Any]],
    prior_baselines: Mapping[str, Any],
) -> Dict[str, Any]:
    stage_by_label: Dict[str, Mapping[str, Any]] = {str(result["stage_label"]): result for result in stage_results}
    b1 = stage_by_label.get("B1", {})
    b2 = stage_by_label.get("B2", {})
    b1_cross = b1.get("cross_setting_summary", {})
    b2_cross = b2.get("cross_setting_summary", {})
    b1_split = b1.get("cross_split_summary", {})
    b2_split = b2.get("cross_split_summary", {})
    b1_closed = b1_cross.get("setting_summaries", {}).get("closed_set", {})
    b2_closed = b2_cross.get("setting_summaries", {}).get("closed_set", {})
    b1_semi = b1_cross.get("setting_summaries", {}).get("semi_open", {})
    b2_semi = b2_cross.get("setting_summaries", {}).get("semi_open", {})
    delta_closed = float(b2_closed.get("mean_dropped_class_recovery_rate", 0.0)) - float(b1_closed.get("mean_dropped_class_recovery_rate", 0.0))
    delta_semi = float(b2_semi.get("mean_dropped_class_recovery_rate", 0.0)) - float(b1_semi.get("mean_dropped_class_recovery_rate", 0.0))
    trend_from_b1 = "smooth" if abs(delta_closed) <= 0.1 and abs(delta_semi) <= 0.1 else "abrupt"
    prior_tier1 = prior_baselines["tier1_closed_set"].get("stage_summaries", {}).get("softem_aug", {})
    prior_tier2 = prior_baselines["tier2_broader_bounded"].get("protocol_summaries", {})
    prior_semiopen = prior_baselines["semiopen_plus4"].get("protocol_summaries", {})
    stressa_ratio_drop = prior_baselines["stressA_ratio_drop"].get("setting_summaries", {})
    return {
        "status": "PASS" if stage_results else "EMPTY",
        "stage_summaries": {
            "B1": b1_cross,
            "B2": b2_cross,
        },
        "cross_split_summaries": {
            "B1": b1_split,
            "B2": b2_split,
        },
        "stage_comparison": {
            "closed_set_recovery_delta_B2_minus_B1": delta_closed,
            "semi_open_recovery_delta_B2_minus_B1": delta_semi,
            "trend_from_B1_to_B2": trend_from_b1,
            "comparison_against_prior": {
                "tier1_closed_set": {
                    "gt_top1_rate": prior_tier1.get("gt_top1_rate"),
                    "mean_gt_rank": prior_tier1.get("mean_gt_rank"),
                    "mean_gt_score": prior_tier1.get("mean_gt_score"),
                },
                "tier2_broader_bounded": {
                    "tier2_25_recovery_rate": prior_tier2.get("tier2_25", {}).get("dropped_class_recovery_rate"),
                    "tier2_50_recovery_rate": prior_tier2.get("tier2_50", {}).get("dropped_class_recovery_rate"),
                },
                "semiopen_plus4": {
                    "semiopen_25_recovery_rate": prior_semiopen.get("semiopen_25", {}).get("dropped_class_recovery_rate"),
                    "semiopen_50_recovery_rate": prior_semiopen.get("semiopen_50", {}).get("dropped_class_recovery_rate"),
                },
                "trancheA_ratio_drop": {
                    "closed_set_mean_recovery": stressa_ratio_drop.get("closed_set", {}).get("mean_dropped_class_recovery_rate"),
                    "semi_open_mean_recovery": stressa_ratio_drop.get("semi_open", {}).get("mean_dropped_class_recovery_rate"),
                },
            },
        },
    }


def _emit_mainline_ratio_drop_family_artifacts(
    *,
    repo_root: Path,
    stage_results: Sequence[Mapping[str, Any]],
    cross_stage_summary: Mapping[str, Any],
) -> None:
    audit_root = repo_root / "train" / "audit"
    artifact_root = repo_root / "codex" / "outputs" / "G7_training"
    audit_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    combined_casebook_rows: List[Mapping[str, Any]] = []
    for result in stage_results:
        combined_casebook_rows.extend(result.get("casebook_rows", []))
    _write_jsonl(audit_root / "mainline_ratio_drop_casebook.jsonl", combined_casebook_rows)
    _write_json(audit_root / "mainline_ratio_drop_cross_stage_summary.json", cross_stage_summary)
    _write_json(
        artifact_root / "g7_mainline_bounded_trancheB_latest.json",
        {
            "status": "PASS" if stage_results else "EMPTY",
            "stage_labels": [str(result.get("stage_label")) for result in stage_results],
            "cross_stage_summary": cross_stage_summary,
            "stage_result_count": len(stage_results),
        },
    )
    (artifact_root / "g7_mainline_bounded_trancheB_latest.md").write_text(
        "\n".join(
            [
                "# G7 Mainline Bounded Validation Tranche B",
                "",
                f"- status: {'PASS' if stage_results else 'EMPTY'}",
                f"- stage_labels: {[str(result.get('stage_label')) for result in stage_results]}",
                f"- stage_count: {len(stage_results)}",
                f"- cross_stage_status: {cross_stage_summary.get('status')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_tier2_selection_and_protocols_are_deterministic() -> None:
    fixture_root = Path(os.environ.get("PYTEST_TMPDIR", "/tmp")) / "g7_tier2_fixture_selection"
    fixture = _write_tier2_runtime_fixture(fixture_root)
    materialized = materialize_phase1_training_samples(
        fixture_root,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", trajectory_source_branch="mainline", smoke=True, smoke_max_trajectories=SMOKE_MAX_TRAJECTORIES),
    )
    gt_lookup = load_gt_sidecar_lookup(fixture_root, dataset_name="lvvis_train_base", trajectory_source_branch="mainline")
    class_name_lookup = dict(fixture["class_names"])
    candidate_map = _build_candidate_map(fixture_root)
    selected_class_rows_1, selected_clip_rows_1, _ = _select_tier2_clips(
        materialized["samples"], gt_lookup, class_count_target=TIER2_CLASS_COUNT, clip_count_target=TIER2_CLIP_COUNT, class_name_lookup=class_name_lookup
    )
    selected_class_rows_2, selected_clip_rows_2, _ = _select_tier2_clips(
        materialized["samples"], gt_lookup, class_count_target=TIER2_CLASS_COUNT, clip_count_target=TIER2_CLIP_COUNT, class_name_lookup=class_name_lookup
    )
    assert [row["class_id"] for row in selected_class_rows_1] == [row["class_id"] for row in selected_class_rows_2]
    assert [row["clip_id"] for row in selected_clip_rows_1] == [row["clip_id"] for row in selected_clip_rows_2]
    _, _, _, protocol_rows_25 = _derive_protocol_samples(
        selected_clip_rows_1,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
        drop_ratio=0.25,
        protocol_name="tier2_25",
    )
    _, _, _, protocol_rows_50 = _derive_protocol_samples(
        selected_clip_rows_1,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
        drop_ratio=0.50,
        protocol_name="tier2_50",
    )
    assert all(len(row["dropped_class_ids"]) == 1 for row in protocol_rows_25)
    assert all(len(row["dropped_class_ids"]) == 2 for row in protocol_rows_50)
    assert all(int(row["drop_class_id"]) not in row["observed_raw_ids_after_drop"] for row in protocol_rows_25)
    assert all(int(row["drop_class_id"]) not in row["observed_raw_ids_after_drop"] for row in protocol_rows_50)


def test_tier2_summary_aggregation_counts_recovery() -> None:
    rows = {
        "prealign": [
            {
                "trajectory_id": "traj_1",
                "clip_id": "1",
                "gt_available_for_audit": True,
                "gt_rank_full_vocab": 2,
                "gt_score": 0.2,
                "is_gt_top1": False,
                "is_gt_in_topk": True,
                "observed_top1_rate": False,
                "observed_mass_quality": 0.5,
                "margin_gt_minus_best_wrong": 0.1,
                "cosine_to_gt_text": 0.2,
                "cosine_to_best_wrong_text": 0.1,
            }
        ],
        "softem_base": [
            {
                "trajectory_id": "traj_1",
                "clip_id": "1",
                "gt_available_for_audit": True,
                "gt_rank_full_vocab": 1,
                "gt_score": 0.8,
                "is_gt_top1": True,
                "is_gt_in_topk": True,
                "observed_top1_rate": True,
                "observed_mass_quality": 0.7,
                "margin_gt_minus_best_wrong": 0.4,
                "cosine_to_gt_text": 0.8,
                "cosine_to_best_wrong_text": 0.2,
                "gt_dropped_from_observed": True,
                "gt_recovered_via_extra": False,
                "closed_set_is_gt_top1": False,
                "closed_set_gt_rank": 2,
                "closed_set_top1_id": 9,
            }
        ],
        "softem_aug": [
            {
                "trajectory_id": "traj_1",
                "clip_id": "1",
                "gt_available_for_audit": True,
                "gt_rank_full_vocab": 1,
                "gt_score": 0.9,
                "is_gt_top1": True,
                "is_gt_in_topk": True,
                "observed_top1_rate": True,
                "observed_mass_quality": 0.9,
                "margin_gt_minus_best_wrong": 0.8,
                "cosine_to_gt_text": 0.9,
                "cosine_to_best_wrong_text": 0.1,
                "gt_dropped_from_observed": True,
                "gt_recovered_via_extra": True,
                "closed_set_is_gt_top1": True,
                "closed_set_gt_rank": 1,
                "closed_set_top1_id": 1,
            }
        ],
    }
    summary = _aggregate_protocol_summary(
        protocol_name="tier2_25",
        selected_class_rows=[{"class_id": 1, "class_name": "class_1"}],
        selected_clip_rows=[{"clip_id": "1"}],
        rows_by_stage=rows,
        prior_tier1_summary={"stage_summaries": {"softem_aug": {"gt_top1_rate": 0.0, "mean_gt_rank": 4.0, "mean_gt_score": 0.1}}},
        prior_full_vocab_summary={"stage_summaries": {"softem_aug": {"gt_top1_rate": 0.0, "mean_gt_rank": 100.0, "mean_gt_score": 0.01}}},
    )
    assert summary["dropped_class_recovery_rate"] == 1.0
    assert summary["extra_gt_recall@K"] == 1.0
    assert summary["extra_precision@K"] == 1.0
    assert summary["missing_class_recovery_rate"] == 1.0
    assert summary["spurious_extra_rate"] == 0.0


def test_gt_is_audit_only_in_broader_validation(tmp_path: Path) -> None:
    fixture = _write_tier2_runtime_fixture(tmp_path)
    materialized = materialize_phase1_training_samples(
        tmp_path,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", trajectory_source_branch="mainline", smoke=True, smoke_max_trajectories=SMOKE_MAX_TRAJECTORIES),
    )
    gt_lookup = load_gt_sidecar_lookup(tmp_path, dataset_name="lvvis_train_base", trajectory_source_branch="mainline")
    class_name_lookup = dict(fixture["class_names"])
    candidate_map = _build_candidate_map(tmp_path)
    selected_class_rows, selected_clip_rows, _ = _select_tier2_clips(
        materialized["samples"], gt_lookup, class_count_target=TIER2_CLASS_COUNT, clip_count_target=TIER2_CLIP_COUNT, class_name_lookup=class_name_lookup
    )
    selected_samples_25, _, _, _ = _derive_protocol_samples(
        selected_clip_rows,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
        drop_ratio=0.25,
        protocol_name="tier2_25",
    )
    assert selected_samples_25
    assert "gt_class_id" not in selected_samples_25[0]["weak_label_record"]
    assert selected_samples_25[0]["candidate_ids_extra"] == [selected_samples_25[0]["controlled_drop_protocol"]["drop_class_ids"][0]]
    assert selected_samples_25[0]["controlled_drop_protocol"]["drop_class_ids"][0] not in selected_samples_25[0]["observed_raw_ids"]
    assert selected_samples_25[0]["sample_valid"] is True


def test_semiopen_distractor_selection_is_deterministic(tmp_path: Path) -> None:
    fixture = _write_tier2_runtime_fixture(tmp_path)
    selected_distractors_1 = _select_semiopen_distractors(fixture["class_names"])
    selected_distractors_2 = _select_semiopen_distractors(fixture["class_names"])
    assert [row["class_id"] for row in selected_distractors_1] == [21, 23, 25, 27]
    assert [row["class_id"] for row in selected_distractors_1] == [row["class_id"] for row in selected_distractors_2]
    assert not set(row["class_id"] for row in selected_distractors_1).intersection(TIER2_TARGET_CLASS_IDS)


def test_broader_bounded_validation_bounded_experiment_smoke(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fixture_root = tmp_path / "broader_bounded_fixture"
    fixture = _write_tier2_runtime_fixture(fixture_root)

    materialized = materialize_phase1_training_samples(
        fixture_root,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", trajectory_source_branch="mainline", smoke=True, smoke_max_trajectories=SMOKE_MAX_TRAJECTORIES),
    )
    gt_lookup = load_gt_sidecar_lookup(fixture_root, dataset_name="lvvis_train_base", trajectory_source_branch="mainline")
    class_name_lookup = dict(fixture["class_names"])
    candidate_map = _build_candidate_map(fixture_root)
    selected_class_rows, selected_clip_rows, _ = _select_tier2_clips(
        materialized["samples"],
        gt_lookup,
        class_count_target=TIER2_CLASS_COUNT,
        clip_count_target=TIER2_CLIP_COUNT,
        class_name_lookup=class_name_lookup,
    )

    stage_rows_25, ledger_25, selected_samples_25, protocol_rows_25, _, _, _ = _run_protocol(
        repo_root=repo_root,
        fixture_root=fixture_root,
        protocol_name="tier2_25",
        drop_ratio=0.25,
        selected_clip_rows=selected_clip_rows,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
        gt_lookup=gt_lookup,
    )
    stage_rows_50, ledger_50, selected_samples_50, protocol_rows_50, _, _, _ = _run_protocol(
        repo_root=repo_root,
        fixture_root=fixture_root,
        protocol_name="tier2_50",
        drop_ratio=0.50,
        selected_clip_rows=selected_clip_rows,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
        gt_lookup=gt_lookup,
    )

    summary = _tier2_experiment_summary(
        repo_root=repo_root,
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        protocol_rows_25=protocol_rows_25,
        protocol_rows_50=protocol_rows_50,
        stage_rows_25=stage_rows_25,
        stage_rows_50=stage_rows_50,
    )
    casebook_25 = _build_casebook(stage_rows_25, "tier2_25")
    casebook_50 = _build_casebook(stage_rows_50, "tier2_50")
    _emit_broader_validation_artifacts(
        repo_root=repo_root,
        summary=summary,
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        protocol_rows={"tier2_25": protocol_rows_25, "tier2_50": protocol_rows_50},
        ledger_rows={"tier2_25": ledger_25, "tier2_50": ledger_50},
        casebook_rows={"tier2_25": casebook_25, "tier2_50": casebook_50},
    )

    if os.environ.get("WSOVVIS_ENABLE_CANONICAL_WRITEBACK") == "1":
        _copytree(
            fixture_root,
            repo_root,
            [
                "train/prealign/train_state.json",
                "train/prealign/proxy_records.jsonl",
                "train/prealign/checkpoints/prealign_last.pth",
                "train/softem_base/train_state.json",
                "train/softem_base/responsibility_records.jsonl",
                "train/softem_base/checkpoints/softem_base_last.pth",
                "train/softem_aug/train_state.json",
                "train/softem_aug/responsibility_records.jsonl",
                "train/softem_aug/checkpoints/softem_aug_last.pth",
                "train/audit/tier2_selected_classes.json",
                "train/audit/tier2_selected_clips.json",
                "train/audit/tier2_drop_protocol_25.json",
                "train/audit/tier2_drop_protocol_50.json",
                "train/audit/tier2_drop_recovery_ledger_25.jsonl",
                "train/audit/tier2_drop_recovery_ledger_50.jsonl",
                "train/audit/tier2_drop_recovery_summary.json",
                "train/audit/tier2_drop_recovery_casebook.jsonl",
                "codex/outputs/G7_training/g7_broader_bounded_validation_latest.json",
                "codex/outputs/G7_training/g7_broader_bounded_validation_latest.md",
            ],
        )

    assert summary["status"] == "PASS"
    assert summary["selected_class_count"] == TIER2_CLASS_COUNT
    assert summary["selected_clip_count"] == TIER2_CLIP_COUNT
    assert summary["protocol_summaries"]["tier2_25"]["dropped_class_recovery_rate"] >= 0.0
    assert summary["protocol_summaries"]["tier2_50"]["dropped_class_recovery_rate"] >= 0.0
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_broader_bounded_validation_latest.json").is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_broader_bounded_validation_latest.md").is_file()
    assert (repo_root / "train" / "audit" / "tier2_drop_protocol_25.json").is_file()
    assert (repo_root / "train" / "audit" / "tier2_drop_protocol_50.json").is_file()


def test_semiopen_bounded_validation_bounded_experiment_smoke(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fixture_root = tmp_path / "semiopen_bounded_fixture"
    fixture = _write_tier2_runtime_fixture(fixture_root)

    materialized = materialize_phase1_training_samples(
        fixture_root,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", trajectory_source_branch="mainline", smoke=True, smoke_max_trajectories=SMOKE_MAX_TRAJECTORIES),
    )
    gt_lookup = load_gt_sidecar_lookup(fixture_root, dataset_name="lvvis_train_base", trajectory_source_branch="mainline")
    class_name_lookup = dict(fixture["class_names"])
    candidate_map = _build_candidate_map(fixture_root)
    selected_class_rows, selected_clip_rows, _ = _select_tier2_clips(
        materialized["samples"],
        gt_lookup,
        class_count_target=TIER2_CLASS_COUNT,
        clip_count_target=TIER2_CLIP_COUNT,
        class_name_lookup=class_name_lookup,
    )
    selected_distractor_rows = _select_semiopen_distractors(class_name_lookup)
    semiopen_distractor_ids = [int(row["class_id"]) for row in selected_distractor_rows]

    target_rows_25, semiopen_rows_25, target_ledger_25, semiopen_ledger_25, selected_samples_25, protocol_rows_25, _, _, _ = _run_semiopen_protocol(
        repo_root=repo_root,
        fixture_root=fixture_root,
        protocol_name="semiopen_25",
        drop_ratio=0.25,
        selected_clip_rows=selected_clip_rows,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
        gt_lookup=gt_lookup,
        semiopen_distractor_ids=semiopen_distractor_ids,
    )
    target_rows_50, semiopen_rows_50, target_ledger_50, semiopen_ledger_50, selected_samples_50, protocol_rows_50, _, _, _ = _run_semiopen_protocol(
        repo_root=repo_root,
        fixture_root=fixture_root,
        protocol_name="semiopen_50",
        drop_ratio=0.50,
        selected_clip_rows=selected_clip_rows,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
        gt_lookup=gt_lookup,
        semiopen_distractor_ids=semiopen_distractor_ids,
    )

    summary = _semiopen_experiment_summary(
        repo_root=repo_root,
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        selected_distractor_rows=selected_distractor_rows,
        protocol_rows_25=protocol_rows_25,
        protocol_rows_50=protocol_rows_50,
        target_rows_25=target_rows_25,
        semiopen_rows_25=semiopen_rows_25,
        target_rows_50=target_rows_50,
        semiopen_rows_50=semiopen_rows_50,
    )

    semiopen_casebook_25 = _build_casebook(semiopen_rows_25, "semiopen_25", distractor_ids=semiopen_distractor_ids)
    semiopen_casebook_50 = _build_casebook(semiopen_rows_50, "semiopen_50", distractor_ids=semiopen_distractor_ids)
    _emit_semiopen_artifacts(
        repo_root=repo_root,
        summary=summary,
        selected_distractor_rows=selected_distractor_rows,
        protocol_rows_25=protocol_rows_25,
        protocol_rows_50=protocol_rows_50,
        target_ledger_rows_25=target_ledger_25,
        target_ledger_rows_50=target_ledger_50,
        semiopen_ledger_rows_25=semiopen_ledger_25,
        semiopen_ledger_rows_50=semiopen_ledger_50,
        semiopen_casebook_rows_25=semiopen_casebook_25,
        semiopen_casebook_rows_50=semiopen_casebook_50,
    )

    assert summary["status"] == "PASS"
    assert summary["selected_distractor_count"] == SEMIOPEN_DISTRACTOR_COUNT
    assert summary["semiopen_run"] is True
    assert summary["selected_distractor_ids"] == semiopen_distractor_ids
    assert summary["overall_trend_from_tier2"] in {"smooth", "abrupt"}
    assert summary["protocol_summaries"]["semiopen_25"]["status"] == "PASS"
    assert summary["protocol_summaries"]["semiopen_50"]["status"] == "PASS"
    assert summary["protocol_summaries"]["semiopen_25"]["distractor_domination_rate"] >= 0.0
    assert summary["protocol_summaries"]["semiopen_50"]["distractor_domination_rate"] >= 0.0
    assert (repo_root / "train" / "audit" / "semiopen_selected_distractors.json").is_file()
    assert (repo_root / "train" / "audit" / "semiopen_drop_recovery_ledger_25.jsonl").is_file()
    assert (repo_root / "train" / "audit" / "semiopen_drop_recovery_ledger_50.jsonl").is_file()
    assert (repo_root / "train" / "audit" / "semiopen_drop_recovery_summary.json").is_file()
    assert (repo_root / "train" / "audit" / "semiopen_drop_recovery_casebook.jsonl").is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_semiopen_bounded_validation_latest.json").is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_semiopen_bounded_validation_latest.md").is_file()


def test_stressA_deterministic_split_and_protocol_construction(tmp_path: Path) -> None:
    fixture = _write_stressA_runtime_fixture(tmp_path)
    materialized = materialize_phase1_training_samples(
        tmp_path,
        Phase1MaterializationConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            smoke=True,
            smoke_max_trajectories=STRESSA_CLIP_COUNT * len(STRESSA_SPLIT_IDS),
        ),
    )
    gt_lookup = load_gt_sidecar_lookup(tmp_path, dataset_name="lvvis_train_base", trajectory_source_branch="mainline")
    class_name_lookup = dict(fixture["class_names"])
    candidate_map = _build_candidate_map(tmp_path)

    for split_id in STRESSA_SPLIT_IDS:
        selected_class_rows_1, selected_clip_rows_1, _ = _select_split_tier2_clips(
            materialized["samples"],
            gt_lookup,
            clip_ids=fixture["split_to_clip_ids"][split_id],
            class_count_target=STRESSA_CLASS_COUNT,
            clip_count_target=STRESSA_CLIP_COUNT,
            class_name_lookup=class_name_lookup,
        )
        selected_class_rows_2, selected_clip_rows_2, _ = _select_split_tier2_clips(
            materialized["samples"],
            gt_lookup,
            clip_ids=fixture["split_to_clip_ids"][split_id],
            class_count_target=STRESSA_CLASS_COUNT,
            clip_count_target=STRESSA_CLIP_COUNT,
            class_name_lookup=class_name_lookup,
        )
        assert [row["class_id"] for row in selected_class_rows_1] == [row["class_id"] for row in selected_class_rows_2]
        assert [row["clip_id"] for row in selected_clip_rows_1] == [row["clip_id"] for row in selected_clip_rows_2]

        _, _, _, protocol_rows_25 = _derive_protocol_samples(
            selected_clip_rows_1,
            class_name_lookup=class_name_lookup,
            candidate_map=candidate_map,
            drop_ratio=0.25,
            protocol_name=f"{split_id}_ratio_drop_25",
            protocol_type="ratio_drop",
        )
        _, _, _, protocol_rows_50 = _derive_protocol_samples(
            selected_clip_rows_1,
            class_name_lookup=class_name_lookup,
            candidate_map=candidate_map,
            drop_ratio=0.50,
            protocol_name=f"{split_id}_ratio_drop_50",
            protocol_type="ratio_drop",
        )
        _, _, _, keep_rows = _derive_protocol_samples(
            selected_clip_rows_1,
            class_name_lookup=class_name_lookup,
            candidate_map=candidate_map,
            drop_ratio=None,
            protocol_name=f"{split_id}_keep_one",
            protocol_type="keep_exactly_one_observed_per_clip",
        )
        assert all(int(row["drop_class_id"]) not in row["observed_raw_ids_after_drop"] for row in protocol_rows_25)
        assert all(int(row["drop_class_id"]) not in row["observed_raw_ids_after_drop"] for row in protocol_rows_50)
        assert all(len(row["observed_raw_ids_after_drop"]) == 1 for row in keep_rows)
        selected_distractors_8 = _select_semiopen_distractors(
            class_name_lookup,
            count=8,
            source_class_ids=STRESSA_DISTRACTOR_CLASS_IDS,
        )
        selected_distractors_12 = _select_semiopen_distractors(
            class_name_lookup,
            count=12,
            source_class_ids=STRESSA_DISTRACTOR_CLASS_IDS,
        )
        assert [row["class_id"] for row in selected_distractors_8] == STRESSA_DISTRACTOR_CLASS_IDS[:8]
        assert [row["class_id"] for row in selected_distractors_12] == STRESSA_DISTRACTOR_CLASS_IDS[:12]
        assert not set(row["class_id"] for row in selected_distractors_12).intersection(STRESSA_TARGET_CLASS_IDS)


def test_stressA_stronger_bounded_validation_tranche_a_matrix_smoke(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fixture_root = tmp_path / "stressA_bounded_fixture"
    fixture = _write_stressA_runtime_fixture(fixture_root)

    materialized = materialize_phase1_training_samples(
        fixture_root,
        Phase1MaterializationConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            smoke=True,
            smoke_max_trajectories=STRESSA_CLIP_COUNT * len(STRESSA_SPLIT_IDS),
        ),
    )
    gt_lookup = load_gt_sidecar_lookup(fixture_root, dataset_name="lvvis_train_base", trajectory_source_branch="mainline")
    class_name_lookup = dict(fixture["class_names"])
    candidate_map = _build_candidate_map(fixture_root)

    prior_tier1 = _load_json_if_exists(repo_root / "codex" / "outputs" / "G7_training" / "g7_closed_set_drop_recovery_latest.json")
    prior_tier2 = _load_json_if_exists(repo_root / "codex" / "outputs" / "G7_training" / "g7_broader_bounded_validation_latest.json")
    prior_semiopen = _load_json_if_exists(repo_root / "codex" / "outputs" / "G7_training" / "g7_semiopen_bounded_validation_latest.json")

    selected_targets_by_split: Dict[str, Any] = {}
    selected_clips_by_split: Dict[str, Any] = {}
    protocols_by_split: Dict[str, Any] = {}
    run_summaries: List[Dict[str, Any]] = []
    casebook_rows: List[Dict[str, Any]] = []
    matrix_run_specs: List[Dict[str, Any]] = []

    for split_id in STRESSA_SPLIT_IDS:
        split_clip_ids = fixture["split_to_clip_ids"][split_id]
        split_selected_class_rows, split_selected_clip_rows, _ = _select_split_tier2_clips(
            materialized["samples"],
            gt_lookup,
            clip_ids=split_clip_ids,
            class_count_target=STRESSA_CLASS_COUNT,
            clip_count_target=STRESSA_CLIP_COUNT,
            class_name_lookup=class_name_lookup,
        )
        selected_targets_by_split[split_id] = {
            "status": "PASS",
            "selected_classes": list(split_selected_class_rows),
        }
        selected_clips_by_split[split_id] = {
            "status": "PASS",
            "selected_clips": list(split_selected_clip_rows),
        }
        protocols_by_split[split_id] = {
            "status": "PASS",
            "selected_class_ids": [int(row["class_id"]) for row in split_selected_class_rows],
            "selected_clip_ids": [str(row["clip_id"]) for row in split_selected_clip_rows],
            "protocol_specs": [],
        }

        for protocol_type in STRESSA_PROTOCOL_TYPES:
            if protocol_type == "ratio_drop":
                protocol_configs = [("ratio_drop_25", 0.25), ("ratio_drop_50", 0.50)]
            else:
                protocol_configs = [("keep_exactly_one_observed_per_clip", None)]
            for protocol_name, drop_ratio in protocol_configs:
                closed_root_name = f"{split_id}_closed_set_{protocol_name}"
                matrix_run_specs.append(
                    {
                        "run_id": closed_root_name,
                        "split_id": split_id,
                        "setting": "closed_set",
                        "protocol_type": protocol_type,
                        "drop_ratio": drop_ratio,
                        "distractor_count": 0,
                    }
                )
                stage_rows, ledger_rows, selected_samples, protocol_rows, _, _, _ = _run_protocol(
                    repo_root=repo_root,
                    fixture_root=fixture_root,
                    protocol_name=closed_root_name,
                    drop_ratio=drop_ratio,
                    selected_clip_rows=split_selected_clip_rows,
                    class_name_lookup=class_name_lookup,
                    candidate_map=candidate_map,
                    gt_lookup=gt_lookup,
                    protocol_type=protocol_type,
                )
                summary = _stressA_run_summary(
                    repo_root=repo_root,
                    split_id=split_id,
                    setting="closed_set",
                    protocol_type=protocol_type,
                    drop_ratio=drop_ratio,
                    distractor_count=0,
                    selected_class_rows=split_selected_class_rows,
                    selected_clip_rows=split_selected_clip_rows,
                    selected_distractor_rows=[],
                    rows_by_stage=stage_rows,
                    prior_tier1_summary=prior_tier1,
                    prior_tier2_summary=prior_tier2,
                    prior_semiopen_summary=prior_semiopen,
                )
                run_summary = {
                    "run_id": closed_root_name,
                    "split_id": split_id,
                    "setting": "closed_set",
                    "protocol_type": protocol_type,
                    "drop_ratio": drop_ratio,
                    "distractor_count": 0,
                    "selected_class_ids": [int(row["class_id"]) for row in split_selected_class_rows],
                    "selected_clip_ids": [str(row["clip_id"]) for row in split_selected_clip_rows],
                    "selected_distractor_ids": [],
                    "selected_distractor_names": [],
                    "stage_summaries": summary["stage_summaries"],
                    "dropped_class_recovery_rate": summary["dropped_class_recovery_rate"],
                    "extra_gt_recall@K": summary["extra_gt_recall@K"],
                    "extra_precision@K": summary["extra_precision@K"],
                    "missing_class_recovery_rate": summary["missing_class_recovery_rate"],
                    "spurious_extra_rate": summary["spurious_extra_rate"],
                    "distractor_domination_rate": 0.0,
                    "comparison_against_prior": summary["comparison_against_prior"],
                    "trend_from_tier1": summary["trend_from_tier1"],
                    "top_failed_runs": [],
                    "representative_recovered_examples": [],
                    "representative_distractor_dominated_failures": [],
                    "representative_single_observed_failures": [],
                }
                run_summaries.append(run_summary)
                casebook_rows.extend(
                    {
                        **row,
                        "run_id": closed_root_name,
                        "split_id": split_id,
                        "setting": "closed_set",
                        "protocol_type": protocol_type,
                        "drop_ratio": drop_ratio,
                        "distractor_count": 0,
                    }
                    for row in _build_casebook(stage_rows, closed_root_name)
                )
                protocols_by_split[split_id]["protocol_specs"].append(
                    {
                        "run_id": closed_root_name,
                        "setting": "closed_set",
                        "protocol_type": protocol_type,
                        "drop_ratio": drop_ratio,
                        "distractor_count": 0,
                        "protocol_rows": list(protocol_rows),
                        "selected_samples": list(selected_samples),
                    }
                )

                for distractor_count in STRESSA_DISTRACTOR_COUNTS:
                    semi_distractor_rows = _select_semiopen_distractors(
                        class_name_lookup,
                        count=distractor_count,
                        source_class_ids=STRESSA_DISTRACTOR_CLASS_IDS,
                    )
                    semi_root_name = f"{split_id}_semi_open_{protocol_name}_d{int(distractor_count)}"
                    matrix_run_specs.append(
                        {
                            "run_id": semi_root_name,
                            "split_id": split_id,
                            "setting": "semi_open",
                            "protocol_type": protocol_type,
                            "drop_ratio": drop_ratio,
                            "distractor_count": int(distractor_count),
                        }
                    )
                    target_rows, semiopen_rows, target_ledger_rows, semiopen_ledger_rows, selected_samples_semi, semi_protocol_rows, _, _, _ = _run_semiopen_protocol(
                        repo_root=repo_root,
                        fixture_root=fixture_root,
                        protocol_name=semi_root_name,
                        drop_ratio=drop_ratio,
                        selected_clip_rows=split_selected_clip_rows,
                        class_name_lookup=class_name_lookup,
                        candidate_map=candidate_map,
                        gt_lookup=gt_lookup,
                        semiopen_distractor_ids=[int(row["class_id"]) for row in semi_distractor_rows],
                        protocol_type=protocol_type,
                    )
                    semi_summary = _stressA_run_summary(
                        repo_root=repo_root,
                        split_id=split_id,
                        setting="semi_open",
                        protocol_type=protocol_type,
                        drop_ratio=drop_ratio,
                        distractor_count=distractor_count,
                        selected_class_rows=split_selected_class_rows,
                        selected_clip_rows=split_selected_clip_rows,
                        selected_distractor_rows=semi_distractor_rows,
                        rows_by_stage=semiopen_rows,
                        prior_tier1_summary=prior_tier1,
                        prior_tier2_summary=prior_tier2,
                        prior_semiopen_summary=prior_semiopen,
                    )
                    run_summary = {
                        "run_id": semi_root_name,
                        "split_id": split_id,
                        "setting": "semi_open",
                        "protocol_type": protocol_type,
                        "drop_ratio": drop_ratio,
                        "distractor_count": int(distractor_count),
                        "selected_class_ids": [int(row["class_id"]) for row in split_selected_class_rows],
                        "selected_clip_ids": [str(row["clip_id"]) for row in split_selected_clip_rows],
                        "selected_distractor_ids": [int(row["class_id"]) for row in semi_distractor_rows],
                        "selected_distractor_names": [str(row["class_name"]) for row in semi_distractor_rows],
                        "stage_summaries": semi_summary["stage_summaries"],
                        "dropped_class_recovery_rate": semi_summary["dropped_class_recovery_rate"],
                        "extra_gt_recall@K": semi_summary["extra_gt_recall@K"],
                        "extra_precision@K": semi_summary["extra_precision@K"],
                        "missing_class_recovery_rate": semi_summary["missing_class_recovery_rate"],
                        "spurious_extra_rate": semi_summary["spurious_extra_rate"],
                        "distractor_domination_rate": semi_summary["distractor_domination_rate"],
                        "comparison_against_prior": semi_summary["comparison_against_prior"],
                        "trend_from_tier1": semi_summary["trend_from_tier1"],
                        "top_failed_runs": [],
                        "representative_recovered_examples": [],
                        "representative_distractor_dominated_failures": [],
                        "representative_single_observed_failures": [],
                    }
                    run_summaries.append(run_summary)
                    casebook_rows.extend(
                        {
                            **row,
                            "run_id": semi_root_name,
                            "split_id": split_id,
                            "setting": "semi_open",
                            "protocol_type": protocol_type,
                            "drop_ratio": drop_ratio,
                            "distractor_count": int(distractor_count),
                        }
                        for row in _build_casebook(semiopen_rows, semi_root_name, distractor_ids=[int(row["class_id"]) for row in semi_distractor_rows])
                    )
                    protocols_by_split[split_id]["protocol_specs"].append(
                        {
                            "run_id": semi_root_name,
                            "setting": "semi_open",
                            "protocol_type": protocol_type,
                            "drop_ratio": drop_ratio,
                            "distractor_count": int(distractor_count),
                            "selected_distractor_ids": [int(row["class_id"]) for row in semi_distractor_rows],
                            "selected_distractor_names": [str(row["class_name"]) for row in semi_distractor_rows],
                            "target_rows": list(target_rows),
                            "semiopen_rows": list(semiopen_rows),
                            "target_ledger_rows": list(target_ledger_rows),
                            "semiopen_ledger_rows": list(semiopen_ledger_rows),
                            "selected_samples": list(selected_samples_semi),
                            "protocol_rows": list(semi_protocol_rows),
                        }
                    )

    cross_setting_summary, cross_split_summary = _aggregate_stressA_cross_summaries(run_summaries)
    closed_runs = [row for row in run_summaries if row.get("setting") == "closed_set"]
    semi_runs = [row for row in run_summaries if row.get("setting") == "semi_open"]
    matrix_manifest = {
        "status": "PASS",
        "task_name": "g7_stress_validation_trancheA",
        "dataset_name": "lvvis_train_base",
        "trajectory_source_branch": "mainline",
        "target_class_count": STRESSA_CLASS_COUNT,
        "clip_count_target": STRESSA_CLIP_COUNT,
        "split_ids": list(STRESSA_SPLIT_IDS),
        "protocol_types": list(STRESSA_PROTOCOL_TYPES),
        "drop_ratio_targets": list(STRESSA_DROP_RATIOS),
        "distractor_count_targets": list(STRESSA_DISTRACTOR_COUNTS),
        "settings": ["closed_set", "semi_open"],
        "closed_set_distractor_count": 0,
        "semi_open_distractor_counts": list(STRESSA_DISTRACTOR_COUNTS),
        "run_count": int(len(run_summaries)),
        "closed_run_count": int(len(closed_runs)),
        "semiopen_run_count": int(len(semi_runs)),
        "run_specs": list(matrix_run_specs),
        "recovery_protocols": [
            "ratio_drop_25",
            "ratio_drop_50",
            "keep_exactly_one_observed_per_clip",
        ],
        "distractor_pressure_trend": {
            "closed_set_mean_recovery": float(np.mean([float(row["dropped_class_recovery_rate"]) for row in closed_runs])) if closed_runs else None,
            "semiopen_mean_recovery": float(np.mean([float(row["dropped_class_recovery_rate"]) for row in semi_runs])) if semi_runs else None,
            "mean_distractor_domination_rate": float(np.mean([float(row["distractor_domination_rate"]) for row in semi_runs])) if semi_runs else None,
        },
    }
    _emit_stressA_artifacts(
        repo_root=repo_root,
        matrix_manifest=matrix_manifest,
        selected_targets_by_split=selected_targets_by_split,
        selected_clips_by_split=selected_clips_by_split,
        protocols_by_split=protocols_by_split,
        run_summaries=run_summaries,
        cross_setting_summary=cross_setting_summary,
        cross_split_summary=cross_split_summary,
        casebook_rows=casebook_rows,
    )

    assert matrix_manifest["status"] == "PASS"
    assert len(run_summaries) == len(STRESSA_SPLIT_IDS) * (len(STRESSA_DROP_RATIOS) + 1) * (1 + len(STRESSA_DISTRACTOR_COUNTS))
    assert cross_setting_summary["status"] == "PASS"
    assert cross_split_summary["status"] == "PASS"
    assert (repo_root / "train" / "audit" / "stressA_matrix_manifest.json").is_file()
    assert (repo_root / "train" / "audit" / "stressA_selected_targets_by_split.json").is_file()
    assert (repo_root / "train" / "audit" / "stressA_selected_clips_by_split.json").is_file()
    assert (repo_root / "train" / "audit" / "stressA_protocols_by_split.json").is_file()
    assert (repo_root / "train" / "audit" / "stressA_run_summaries.json").is_file()
    assert (repo_root / "train" / "audit" / "stressA_cross_setting_summary.json").is_file()
    assert (repo_root / "train" / "audit" / "stressA_cross_split_robustness_summary.json").is_file()
    assert (repo_root / "train" / "audit" / "stressA_casebook.jsonl").is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_stress_validation_trancheA_latest.json").is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_stress_validation_trancheA_latest.md").is_file()


def test_mainline_ratio_drop_selection_and_protocols_are_deterministic() -> None:
    for stage_label, target_class_ids, clip_count_target in [
        ("B1", MAINLINEB_B1_TARGET_CLASS_IDS, MAINLINEB_B1_CLIP_COUNT),
        ("B2", MAINLINEB_B2_TARGET_CLASS_IDS, MAINLINEB_B2_CLIP_COUNT),
    ]:
        fixture_root = Path(os.environ.get("PYTEST_TMPDIR", "/tmp")) / f"g7_mainline_{stage_label.lower()}_fixture_selection"
        fixture = _write_mainline_ratio_drop_fixture(
            fixture_root,
            split_ids=MAINLINEB_SPLIT_IDS,
            target_class_ids=target_class_ids,
            distractor_class_ids=MAINLINEB_DISTRACTOR_CLASS_IDS,
            clip_count_target=clip_count_target,
            start_clip_id=MAINLINEB_B1_START_CLIP_ID if stage_label == "B1" else MAINLINEB_B2_START_CLIP_ID,
            generator_tag=f"synthetic_mainline_{stage_label}",
        )
        materialized = materialize_phase1_training_samples(
            fixture_root,
            Phase1MaterializationConfig(
                dataset_name="lvvis_train_base",
                trajectory_source_branch="mainline",
                smoke=True,
                smoke_max_trajectories=clip_count_target * len(MAINLINEB_SPLIT_IDS),
            ),
        )
        gt_lookup = load_gt_sidecar_lookup(fixture_root, dataset_name="lvvis_train_base", trajectory_source_branch="mainline")
        class_name_lookup = dict(fixture["class_names"])
        candidate_map = _build_candidate_map(fixture_root)

        for split_id in MAINLINEB_SPLIT_IDS:
            selected_class_rows_1, selected_clip_rows_1, _ = _select_split_tier2_clips(
                materialized["samples"],
                gt_lookup,
                clip_ids=fixture["split_to_clip_ids"][split_id],
                class_count_target=len(target_class_ids),
                clip_count_target=clip_count_target,
                class_name_lookup=class_name_lookup,
            )
            selected_class_rows_2, selected_clip_rows_2, _ = _select_split_tier2_clips(
                materialized["samples"],
                gt_lookup,
                clip_ids=fixture["split_to_clip_ids"][split_id],
                class_count_target=len(target_class_ids),
                clip_count_target=clip_count_target,
                class_name_lookup=class_name_lookup,
            )
            assert [row["class_id"] for row in selected_class_rows_1] == [row["class_id"] for row in selected_class_rows_2]
            assert [row["clip_id"] for row in selected_clip_rows_1] == [row["clip_id"] for row in selected_clip_rows_2]

            _, _, _, protocol_rows_25 = _derive_protocol_samples(
                selected_clip_rows_1,
                class_name_lookup=class_name_lookup,
                candidate_map=candidate_map,
                drop_ratio=0.25,
                protocol_name=f"{stage_label}_{split_id}_ratio_drop_25",
                protocol_type="ratio_drop",
            )
            _, _, _, protocol_rows_50 = _derive_protocol_samples(
                selected_clip_rows_1,
                class_name_lookup=class_name_lookup,
                candidate_map=candidate_map,
                drop_ratio=0.50,
                protocol_name=f"{stage_label}_{split_id}_ratio_drop_50",
                protocol_type="ratio_drop",
            )
            assert all(int(row["drop_class_id"]) not in row["observed_raw_ids_after_drop"] for row in protocol_rows_25)
            assert all(int(row["drop_class_id"]) not in row["observed_raw_ids_after_drop"] for row in protocol_rows_50)

            selected_distractors_8 = _select_semiopen_distractors(
                class_name_lookup,
                count=8,
                source_class_ids=MAINLINEB_DISTRACTOR_CLASS_IDS,
            )
            selected_distractors_16 = _select_semiopen_distractors(
                class_name_lookup,
                count=16,
                source_class_ids=MAINLINEB_DISTRACTOR_CLASS_IDS,
            )
            assert [row["class_id"] for row in selected_distractors_8] == MAINLINEB_DISTRACTOR_CLASS_IDS[:8]
            assert [row["class_id"] for row in selected_distractors_16] == MAINLINEB_DISTRACTOR_CLASS_IDS[:16]
            assert not set(row["class_id"] for row in selected_distractors_16).intersection(target_class_ids)


def test_mainline_ratio_drop_bounded_validation_tranche_b_smoke(tmp_path: Path) -> None:
    repo_root = _repo_root()
    prior_baselines = _load_ratio_drop_baselines(repo_root)

    b1_result = _run_mainline_ratio_drop_stage(
        repo_root=repo_root,
        fixture_root=tmp_path / "mainline_b1_fixture",
        stage_label="B1",
        split_ids=MAINLINEB_SPLIT_IDS,
        target_class_ids=MAINLINEB_B1_TARGET_CLASS_IDS,
        distractor_class_ids=MAINLINEB_DISTRACTOR_CLASS_IDS,
        clip_count_target=MAINLINEB_B1_CLIP_COUNT,
        start_clip_id=MAINLINEB_B1_START_CLIP_ID,
        prior_baselines=prior_baselines,
        distractor_counts=MAINLINEB_DISTRACTOR_COUNTS,
    )
    assert b1_result["matrix_manifest"]["status"] == "PASS"
    assert b1_result["cross_setting_summary"]["status"] == "PASS"
    assert b1_result["cross_split_summary"]["status"] == "PASS"

    stage_results = [b1_result]
    if b1_result["matrix_manifest"]["status"] == "PASS":
        b2_result = _run_mainline_ratio_drop_stage(
            repo_root=repo_root,
            fixture_root=tmp_path / "mainline_b2_fixture",
            stage_label="B2",
            split_ids=MAINLINEB_SPLIT_IDS,
        target_class_ids=MAINLINEB_B2_TARGET_CLASS_IDS,
        distractor_class_ids=MAINLINEB_DISTRACTOR_CLASS_IDS,
        clip_count_target=MAINLINEB_B2_CLIP_COUNT,
        start_clip_id=MAINLINEB_B2_START_CLIP_ID,
        prior_baselines=prior_baselines,
        distractor_counts=MAINLINEB_DISTRACTOR_COUNTS,
    )
        stage_results.append(b2_result)
        assert b2_result["matrix_manifest"]["status"] == "PASS"
        assert b2_result["cross_setting_summary"]["status"] == "PASS"
        assert b2_result["cross_split_summary"]["status"] == "PASS"

    cross_stage_summary = _build_mainline_ratio_drop_cross_stage_summary(
        stage_results=stage_results,
        prior_baselines=prior_baselines,
    )
    _emit_mainline_ratio_drop_family_artifacts(
        repo_root=repo_root,
        stage_results=stage_results,
        cross_stage_summary=cross_stage_summary,
    )

    assert cross_stage_summary["status"] == "PASS"
    assert (repo_root / "train" / "audit" / "mainline_ratio_drop_casebook.jsonl").is_file()
    assert (repo_root / "train" / "audit" / "mainline_ratio_drop_cross_stage_summary.json").is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_mainline_bounded_trancheB_latest.json").is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_mainline_bounded_trancheB_latest.md").is_file()
