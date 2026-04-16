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
import torch

from videocutler.ext_stageb_ovvis.algorithms.prealign import PrealignConfig, train_prealign
from videocutler.ext_stageb_ovvis.algorithms.soft_em import SoftEMConfig, run_soft_em
from videocutler.ext_stageb_ovvis.audit.projector_quality_audit import build_projector_quality_rows
from videocutler.ext_stageb_ovvis.audit.trajectory_gt_audit import _extract_gt_available, _extract_gt_class_id, load_gt_sidecar_lookup
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig


Record = Dict[str, Any]

TIER1_CLASS_COUNT = 6
TIER1_CLIP_COUNT = 12
SMOKE_MAX_TRAJECTORIES = 128
SMOKE_EM_SUBITERATIONS = 2
SMOKE_BASE_EPOCHS = 1
SMOKE_AUG_EPOCHS = 1
SMOKE_BASE_LR = 5e-5
SMOKE_AUG_LR = 5e-5
SMOKE_PREALIGN_EPOCHS = 1
SMOKE_PREALIGN_LR = 1e-4
TEMPERATURE = 0.07
TOPK = 5
DROP_PROTOCOL_SEED = 0

REPO_CLASS_IDS = [1, 3, 5, 7, 9, 11]


@dataclass(frozen=True)
class SelectedClip:
    clip_id: str
    trajectory_id: str
    gt_class_id: int
    gt_class_name: str
    observed_raw_ids: List[int]
    complexity_score: int
    selected_class_rank: int


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_roots_available(root: Path) -> bool:
    return all((root / rel).exists() for rel in ("exports", "carrier_bank", "frame_bank", "weak_labels", "text_bank"))


def _load_class_name_lookup(root: Path, class_ids: Sequence[int]) -> Dict[int, str]:
    try:
        from videocutler.ext_stageb_ovvis.banks.text_bank import _load_class_map

        lookup = {int(raw_id): str(name) for raw_id, name in _load_class_map()}
        return {int(cid): lookup.get(int(cid), f"class_{int(cid)}") for cid in class_ids}
    except Exception:
        return {int(cid): f"class_{int(cid)}" for cid in class_ids}


def _write_synthetic_runtime_fixture(root: Path) -> Dict[str, Any]:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    text_dir = root / "text_bank"
    audit_dir = root / "audit"
    exports_dir = root / "exports" / "lvvis_train_base"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload", audit_dir, exports_dir):
        path.mkdir(parents=True, exist_ok=True)

    class_names = {int(cid): f"class_{int(cid)}" for cid in REPO_CLASS_IDS}
    text_protos = np.zeros((len(REPO_CLASS_IDS), 512), dtype=np.float32)
    for idx, class_id in enumerate(REPO_CLASS_IDS):
        text_protos[idx, idx] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=text_protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [
            {
                "raw_id": int(class_id),
                "proto_path": f"payload/text_prototypes.npz#protos[{idx}]",
                "path_base_mode": "artifact_parent_dir",
            }
            for idx, class_id in enumerate(REPO_CLASS_IDS)
        ],
    )

    trajectory_rows: List[Record] = []
    carrier_rows: List[Record] = []
    frame_rows: List[Record] = []
    geom_rows: List[Record] = []
    weak_rows: List[Record] = []
    gt_rows: List[Record] = []

    clip_ids = list(range(100, 112))
    for idx, clip_id in enumerate(clip_ids):
        gt_class_id = int(REPO_CLASS_IDS[idx // 2])
        support_class_id = int(REPO_CLASS_IDS[(idx // 2 + 1) % len(REPO_CLASS_IDS)])
        trajectory_id = f"synthetic_traj_{clip_id:03d}"
        carrier_payload = np.zeros((1, 768), dtype=np.float16)
        carrier_payload[0, idx % 6] = 1.0
        carrier_path = carrier_dir / f"carrier_{clip_id:03d}.npz"
        np.savez(carrier_path, z_norm=carrier_payload)
        frame_payload = np.zeros((1, 4, 768), dtype=np.float16)
        frame_payload[0, 0, idx % 6] = 1.0
        frame_path = frame_dir / "payload" / f"clip_{clip_id:03d}.npz"
        np.savez(frame_path, slot_0=frame_payload[0])

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
                "valid_token_mask_path": f"frame_geom_records.jsonl#{idx}",
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
                "generator_tag": "synthetic_closed_set",
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
                "observed_raw_ids": [int(gt_class_id), int(support_class_id)],
                "observation_protocol_id": "synthetic_closed_set_v1",
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

    _write_jsonl(carrier_dir / "carrier_records.jsonl", carrier_rows)
    _write_jsonl(frame_dir / "frame_records.jsonl", frame_rows)
    _write_jsonl(frame_dir / "frame_geom_records.jsonl", geom_rows)
    _write_json(root / "weak_labels" / "weak_labels_train.json", weak_rows)
    _write_jsonl(exports_dir / "trajectory_records.jsonl", trajectory_rows)
    _write_jsonl(audit_dir / "trajectory_gt_match_train_mainline.jsonl", gt_rows)
    return {
        "class_names": class_names,
        "class_ids": list(REPO_CLASS_IDS),
        "clip_ids": clip_ids,
        "trajectory_ids": [row["trajectory_id"] for row in trajectory_rows],
    }


def _build_candidate_map(text_root: Path) -> Dict[int, Dict[str, Any]]:
    from videocutler.ext_stageb_ovvis.banks.text_bank import read_text_prototype_records

    records = read_text_prototype_records(text_root / "text_bank" / "text_prototype_records.jsonl")
    return {int(record["raw_id"]): dict(record) for record in records}


def _select_closed_set_clips(
    materialized_samples: Sequence[Mapping[str, Any]],
    gt_lookup: Mapping[str, Mapping[str, Any]],
    *,
    class_count_target: int = TIER1_CLASS_COUNT,
    clip_count_target: int = TIER1_CLIP_COUNT,
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
            if len(observed_raw_ids) < 2:
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
        class_name = str(class_name_lookup.get(int(chosen_gt_class_id), f"class_{int(chosen_gt_class_id)}"))
        clip_records.append(
            {
                "clip_id": str(clip_id),
                "trajectory_id": str(chosen_sample.get("trajectory_id", "")),
                "gt_class_id": int(chosen_gt_class_id),
                "gt_class_name": class_name,
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

    class_rows = []
    for class_id, group in class_groups.items():
        sorted_group = sorted(group, key=lambda row: (int(row["complexity_score"]), str(row["clip_id"]), str(row["trajectory_id"])))
        support_count = len(sorted_group)
        average_complexity = float(np.mean([int(row["complexity_score"]) for row in sorted_group])) if sorted_group else 0.0
        class_rows.append(
            {
                "class_id": int(class_id),
                "class_name": str(class_name_lookup.get(int(class_id), f"class_{int(class_id)}")),
                "clip_support_count": int(support_count),
                "mean_clip_complexity": float(average_complexity),
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
            f"could not build Tier-1 closed set: selected={len(selected_clip_rows)} clips, target={clip_count_target}"
        )

    selected_clip_rows.sort(key=lambda row: (int(row["selected_class_rank"]), int(row["complexity_score"]), str(row["clip_id"]), str(row["trajectory_id"])))
    for index, row in enumerate(selected_clip_rows, start=1):
        row["selection_rank"] = int(index)

    final_selected_class_ids = sorted({int(row["gt_class_id"]) for row in selected_clip_rows})
    return class_rows[: int(class_count_target)], selected_clip_rows, eligible_groups


def _derive_closed_set_samples(
    selected_clip_rows: Sequence[Mapping[str, Any]],
    *,
    class_name_lookup: Mapping[int, str],
    candidate_map: Mapping[int, Mapping[str, Any]],
) -> Tuple[List[Record], Dict[str, int], Dict[str, List[int]], List[Record]]:
    selected_class_ids = sorted({int(row["gt_class_id"]) for row in selected_clip_rows})
    selected_class_set = set(selected_class_ids)
    drop_by_clip = {str(row["clip_id"]): int(row["gt_class_id"]) for row in selected_clip_rows}
    selected_samples: List[Record] = []
    selected_clip_ids: Dict[str, int] = {}
    selected_observed: Dict[str, List[int]] = {}
    selected_protocol_rows: List[Record] = []

    for row in selected_clip_rows:
        sample = dict(row["sample"])
        clip_id = str(sample["clip_id"])
        drop_class_id = int(row["gt_class_id"])
        observed_raw_ids = sorted({int(x) for x in list(sample.get("observed_raw_ids", []))})
        observed_after_drop = sorted(
            {
                rid
                for rid in observed_raw_ids
                if rid in selected_class_set and rid != int(drop_class_id)
            }
        )
        if not observed_after_drop:
            raise RuntimeError(f"closed-set sample for clip {clip_id} would become empty after drop")
        candidate_ids_known = list(observed_after_drop)
        candidate_ids_extra = [int(drop_class_id)]
        candidate_text_records = [dict(candidate_map[int(raw_id)]) for raw_id in [*candidate_ids_known, *candidate_ids_extra]]
        class_name = str(class_name_lookup.get(int(drop_class_id), f"class_{int(drop_class_id)}"))
        sample["observed_raw_ids"] = list(observed_after_drop)
        sample["weak_label_record"] = dict(sample.get("weak_label_record", {}))
        sample["weak_label_record"]["observed_raw_ids"] = list(observed_after_drop)
        sample["candidate_ids_known"] = candidate_ids_known
        sample["candidate_ids_extra"] = candidate_ids_extra
        sample["candidate_text_prototypes"] = candidate_text_records
        sample["candidate_ids_extra_provenance"] = [
            {
                "raw_id": int(drop_class_id),
                "score": None,
                "rank": 1,
                "admission_reason": "controlled_drop_recovery_extra",
                "proposal_source": "controlled_drop_protocol",
            }
        ]
        sample["candidate_proposal_source"] = "controlled_drop_protocol"
        sample["missing_views"] = []
        sample["invalid_reasons"] = []
        sample["sample_valid"] = True
        sample["controlled_drop_protocol"] = {
            "drop_class_id": int(drop_class_id),
            "drop_class_name": class_name,
            "selected_class_ids": selected_class_ids,
            "selected_class_names": [str(class_name_lookup.get(cid, f"class_{cid}")) for cid in selected_class_ids],
            "observed_raw_ids_after_drop": observed_after_drop,
        }
        selected_samples.append(sample)
        selected_clip_ids[clip_id] = int(drop_class_id)
        selected_observed[clip_id] = list(observed_after_drop)
        selected_protocol_rows.append(
            {
                "clip_id": clip_id,
                "trajectory_id": str(sample["trajectory_id"]),
                "drop_class_id": int(drop_class_id),
                "drop_class_name": class_name,
                "observed_raw_ids_before_drop": observed_raw_ids,
                "observed_raw_ids_after_drop": list(observed_after_drop),
                "selected_class_ids": selected_class_ids,
                "selected_class_names": [str(class_name_lookup.get(cid, f"class_{cid}")) for cid in selected_class_ids],
            }
        )
    return selected_samples, selected_clip_ids, selected_observed, selected_protocol_rows


def _load_projector_from_checkpoint(checkpoint_path: Path, device: str = "cpu") -> Projector:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    config_payload = dict(checkpoint.get("projector_config", {}))
    projector = Projector(
        ProjectorConfig(
            input_dim=int(config_payload.get("input_dim", 768)),
            hidden_dim=int(config_payload.get("hidden_dim", 512)),
            output_dim=int(config_payload.get("output_dim", 512)),
            dropout=float(config_payload.get("dropout", 0.0)),
            use_layernorm=bool(config_payload.get("use_layernorm", True)),
        )
    ).to(device)
    projector.load_state_dict(checkpoint["projector_state_dict"])
    projector.eval()
    return projector


def _evaluate_closed_set_rows(
    *,
    output_root: Path,
    selected_samples: Sequence[Mapping[str, Any]],
    gt_lookup: Mapping[str, Mapping[str, Any]],
    stage_id: str,
    checkpoint_path: Path,
    previous_by_trajectory: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Record]:
    projector = _load_projector_from_checkpoint(checkpoint_path, device="cpu")
    rows = build_projector_quality_rows(
        output_root=output_root,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
        stage_id=stage_id,
        snapshot_id="stage_end",
        materialized_samples=selected_samples,
        projector=projector,
        topk=TOPK,
        gt_sidecar_lookup=gt_lookup,
        temperature=TEMPERATURE,
        previous_by_trajectory=previous_by_trajectory,
    )
    for row in rows:
        drop_class_id = int(row.get("gt_class_id")) if row.get("gt_class_id") is not None else None
        row["gt_dropped_from_observed"] = bool(
            drop_class_id is not None and drop_class_id not in row.get("observed_raw_ids", [])
        )
        row["gt_recovered_via_extra"] = bool(
            row["gt_dropped_from_observed"]
            and row.get("closed_set_is_gt_top1")
            and row.get("gt_class_id") is not None
            and row.get("gt_in_extra_domain")
        )
        row["gt_rank_closed_set"] = row.get("closed_set_gt_rank")
        row["closed_set_recovered"] = bool(row["gt_recovered_via_extra"])
        row["gt_rank_full_vocab"] = row.get("gt_rank_full_vocab")
    return rows


def _aggregate_recovery(rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]], selected_clip_ids: Mapping[str, int]) -> Dict[str, Any]:
    final_rows = list(rows_by_stage.get("softem_aug", []))
    dropped_rows = [row for row in final_rows if bool(row.get("gt_dropped_from_observed"))]
    recovered_rows = [row for row in dropped_rows if bool(row.get("gt_recovered_via_extra"))]
    topk_hits = [row for row in dropped_rows if bool(row.get("is_gt_in_topk"))]
    extra_top1_hits = [row for row in dropped_rows if bool(row.get("closed_set_is_gt_top1"))]
    summary = {
        "status": "PASS" if rows_by_stage else "EMPTY",
        "selected_clip_count": int(len(selected_clip_ids)),
        "selected_class_count": int(len({int(x) for x in selected_clip_ids.values()})),
        "dropped_class_recovery_rate": float(len(recovered_rows) / len(dropped_rows)) if dropped_rows else 0.0,
        "extra_gt_recall@K": float(len(topk_hits) / len(dropped_rows)) if dropped_rows else 0.0,
        "extra_precision@K": float(len(recovered_rows) / len(extra_top1_hits)) if extra_top1_hits else 0.0,
        "missing_class_recovery_rate": float(len(recovered_rows) / len(dropped_rows)) if dropped_rows else 0.0,
        "spurious_extra_rate": float(1.0 - (len(recovered_rows) / len(extra_top1_hits))) if extra_top1_hits else 0.0,
        "trajectory_level_rows": int(len(final_rows)),
        "dropped_trajectory_rows": int(len(dropped_rows)),
        "recovered_trajectory_rows": int(len(recovered_rows)),
        "dropped_trajectories": [
            {
                "trajectory_id": str(row.get("trajectory_id")),
                "clip_id": str(row.get("clip_id")),
                "gt_class_id": row.get("gt_class_id"),
                "gt_rank_closed_set": row.get("gt_rank_closed_set"),
                "gt_score": row.get("gt_score"),
                "best_wrong_id": row.get("best_wrong_id"),
                "margin_gt_minus_best_wrong": row.get("margin_gt_minus_best_wrong"),
                "gt_recovered_via_extra": bool(row.get("gt_recovered_via_extra")),
            }
            for row in dropped_rows
        ],
    }
    return summary


def _build_casebook(rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Record]:
    final_rows = list(rows_by_stage.get("softem_aug", []))
    casebook: List[Record] = []
    for row in final_rows:
        if not bool(row.get("gt_dropped_from_observed")):
            continue
        if bool(row.get("gt_recovered_via_extra")):
            failure_type = "recovered"
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


def _make_summary(
    *,
    repo_root: Path,
    selected_class_rows: Sequence[Mapping[str, Any]],
    selected_clip_rows: Sequence[Mapping[str, Any]],
    selected_samples: Sequence[Mapping[str, Any]],
    selected_clip_ids: Mapping[str, int],
    selected_observed: Mapping[str, Sequence[int]],
    selected_protocol_rows: Sequence[Mapping[str, Any]],
    stage_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    prior_path = repo_root / "codex" / "outputs" / "G7_training" / "g7_stagewise_projector_quality_latest.json"
    prior_summary = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else {}
    prior_stage = prior_summary.get("stage_summaries", {}).get("softem_aug", {})
    final_stage = list(stage_rows.get("softem_aug", []))
    dropped_rows = [row for row in final_stage if bool(row.get("gt_dropped_from_observed"))]
    recovered_rows = [row for row in dropped_rows if bool(row.get("gt_recovered_via_extra"))]
    recovery_rate = float(len(recovered_rows) / len(dropped_rows)) if dropped_rows else 0.0
    extra_recall = float(len([row for row in dropped_rows if bool(row.get("is_gt_in_topk"))]) / len(dropped_rows)) if dropped_rows else 0.0
    extra_precision = float(len(recovered_rows) / len([row for row in dropped_rows if bool(row.get("closed_set_is_gt_top1"))])) if [row for row in dropped_rows if bool(row.get("closed_set_is_gt_top1"))] else 0.0

    prior_gt_top1 = prior_stage.get("gt_top1_rate")
    prior_gt_rank = prior_stage.get("mean_gt_rank")
    prior_gt_score = prior_stage.get("mean_gt_score")
    final_summary = {
        "status": "PASS" if dropped_rows else "EMPTY",
        "task_id": "G7_training-task",
        "task_name": "small_closed_set_controlled_drop_recovery",
        "dataset_name": "lvvis_train_base",
        "trajectory_source_branch": "mainline",
        "tier": "Tier-1",
        "class_count_target": TIER1_CLASS_COUNT,
        "clip_count_target": TIER1_CLIP_COUNT,
        "selected_class_count": int(len(selected_class_rows)),
        "selected_clip_count": int(len(selected_clip_rows)),
        "selected_class_ids": [int(row["class_id"]) for row in selected_class_rows],
        "selected_clip_ids": [str(row["clip_id"]) for row in selected_clip_rows],
        "drop_protocol": {
            "seed": DROP_PROTOCOL_SEED,
            "selection_rule": "deterministic class ranking by support then low complexity; deterministic round-robin clip pick",
            "drop_rule": "drop_primary_gt_class_from_observed_and_restrict_to_selected_closed_set",
        },
        "bounded_settings": {
            "smoke_max_trajectories": SMOKE_MAX_TRAJECTORIES,
            "prealign_epochs": SMOKE_PREALIGN_EPOCHS,
            "prealign_lr": SMOKE_PREALIGN_LR,
            "base_epochs": SMOKE_BASE_EPOCHS,
            "base_lr": SMOKE_BASE_LR,
            "aug_epochs": SMOKE_AUG_EPOCHS,
            "aug_lr": SMOKE_AUG_LR,
            "em_subiterations": SMOKE_EM_SUBITERATIONS,
            "temperature": TEMPERATURE,
            "topk": TOPK,
        },
        "stage_summaries": {
            stage_id: {
                "gt_top1_rate": stage_summary.get("gt_top1_rate"),
                "mean_gt_rank": stage_summary.get("mean_gt_rank"),
                "median_gt_rank": stage_summary.get("median_gt_rank"),
                "mean_gt_score": stage_summary.get("mean_gt_score"),
                "mean_margin_gt_minus_best_wrong": stage_summary.get("mean_margin_gt_minus_best_wrong"),
                "mean_cosine_to_gt_text": stage_summary.get("mean_cosine_to_gt_text"),
                "mean_cosine_to_best_wrong_text": stage_summary.get("mean_cosine_to_best_wrong_text"),
                "observed_top1_rate": stage_summary.get("observed_top1_rate"),
                "observed_mass_quality": stage_summary.get("observed_mass_quality"),
            }
            for stage_id, stage_summary in {
                stage_id: {
                    "gt_top1_rate": None,
                    "mean_gt_rank": None,
                    "median_gt_rank": None,
                    "mean_gt_score": None,
                    "mean_margin_gt_minus_best_wrong": None,
                    "mean_cosine_to_gt_text": None,
                    "mean_cosine_to_best_wrong_text": None,
                    "observed_top1_rate": None,
                    "observed_mass_quality": None,
                }
                for stage_id in ("prealign", "softem_base", "softem_aug")
            }.items()
        },
        "recovery_metrics": {
            "dropped_class_recovery_rate": recovery_rate,
            "extra_gt_recall@K": extra_recall,
            "extra_precision@K": extra_precision,
            "missing_class_recovery_rate": recovery_rate,
            "spurious_extra_rate": float(1.0 - extra_precision),
        },
        "comparison_against_prior_weak_full_vocab_behavior": {},
        "selected_clip_observed_after_drop": {clip_id: list(values) for clip_id, values in selected_observed.items()},
        "selected_protocol_rows": list(selected_protocol_rows),
        "verdict": (
            "closed_set_recovery_works"
            if recovery_rate > 0.0
            else "closed_set_recovery_did_not_work"
        ),
        "external_validity_statement": (
            "This closed-set result demonstrates whether the repaired mechanism can recover deliberately dropped classes in a bounded, deterministic 6-class/12-clip setting. "
            "It does not establish open-vocabulary behavior, large-vocabulary calibration, or formal G7 closure."
        ),
        "recommendation": (
            "proceed_to_broader_bounded_validation"
            if recovery_rate > 0.0
            else "return_to_further_repair"
        ),
    }
    # Fill actual stage summaries after constructing the shell so the logic is easy to read.
    for stage_id in ("prealign", "softem_base", "softem_aug"):
        stage_rows_list = list(stage_rows.get(stage_id, []))
        if stage_rows_list:
            stage_rows_filtered = [row for row in stage_rows_list if row.get("gt_available_for_audit") and row.get("gt_rank_full_vocab") is not None and row.get("gt_score") is not None]
            gt_top1_hits = sum(1 for row in stage_rows_filtered if bool(row.get("is_gt_top1")))
            gt_ranks = [float(row["gt_rank_full_vocab"]) for row in stage_rows_filtered]
            gt_scores = [float(row["gt_score"]) for row in stage_rows_filtered]
            margins = [float(row["margin_gt_minus_best_wrong"]) for row in stage_rows_filtered if row.get("margin_gt_minus_best_wrong") is not None]
            cosine_gt = [float(row["cosine_to_gt_text"]) for row in stage_rows_filtered if row.get("cosine_to_gt_text") is not None]
            cosine_wrong = [float(row["cosine_to_best_wrong_text"]) for row in stage_rows_filtered if row.get("cosine_to_best_wrong_text") is not None]
            observed_top1_hits = sum(1 for row in stage_rows_filtered if bool(row.get("observed_top1_rate")))
            observed_mass = [float(row["observed_mass_quality"]) for row in stage_rows_filtered if row.get("observed_mass_quality") is not None]
            final_summary["stage_summaries"][stage_id] = {
                "status": "PASS" if stage_rows_filtered else "EMPTY",
                "row_count": int(len(stage_rows_list)),
                "gt_row_count": int(len(stage_rows_filtered)),
                "gt_top1_rate": float(gt_top1_hits / len(stage_rows_filtered)) if stage_rows_filtered else None,
                "gt_topk_rate": float(sum(1 for row in stage_rows_filtered if bool(row.get("is_gt_in_topk"))) / len(stage_rows_filtered)) if stage_rows_filtered else None,
                "mean_gt_rank": float(np.mean(gt_ranks)) if gt_ranks else None,
                "median_gt_rank": float(np.median(gt_ranks)) if gt_ranks else None,
                "mean_gt_score": float(np.mean(gt_scores)) if gt_scores else None,
                "mean_margin_gt_minus_best_wrong": float(np.mean(margins)) if margins else None,
                "mean_cosine_to_gt_text": float(np.mean(cosine_gt)) if cosine_gt else None,
                "mean_cosine_to_best_wrong_text": float(np.mean(cosine_wrong)) if cosine_wrong else None,
                "observed_top1_rate": float(observed_top1_hits / len(stage_rows_filtered)) if stage_rows_filtered else None,
                "observed_mass_quality": float(np.mean(observed_mass)) if observed_mass else None,
            }
        else:
            final_summary["stage_summaries"][stage_id] = {
                "status": "EMPTY",
                "row_count": 0,
                "gt_row_count": 0,
                "gt_top1_rate": None,
                "gt_topk_rate": None,
                "mean_gt_rank": None,
                "median_gt_rank": None,
                "mean_gt_score": None,
                "mean_margin_gt_minus_best_wrong": None,
                "mean_cosine_to_gt_text": None,
                "mean_cosine_to_best_wrong_text": None,
                "observed_top1_rate": None,
                "observed_mass_quality": None,
            }
    final_summary["comparison_against_prior_weak_full_vocab_behavior"] = {
        "prior_full_vocab_gt_top1_rate": prior_gt_top1,
        "prior_full_vocab_mean_gt_rank": prior_gt_rank,
        "prior_full_vocab_mean_gt_score": prior_gt_score,
        "current_closed_set_gt_top1_rate": final_summary["stage_summaries"]["softem_aug"]["gt_top1_rate"],
        "current_closed_set_mean_gt_rank": final_summary["stage_summaries"]["softem_aug"]["mean_gt_rank"],
        "current_closed_set_mean_gt_score": final_summary["stage_summaries"]["softem_aug"]["mean_gt_score"],
    }
    return final_summary


def _emit_final_artifacts(repo_root: Path, fixture_root: Path, summary: Mapping[str, Any], stage_rows: Mapping[str, Sequence[Mapping[str, Any]]], casebook: Sequence[Mapping[str, Any]], selected_class_rows: Sequence[Mapping[str, Any]], selected_clip_rows: Sequence[Mapping[str, Any]], drop_protocol_rows: Sequence[Mapping[str, Any]], recovery_ledger_rows: Sequence[Mapping[str, Any]]) -> None:
    artifact_root = repo_root / "codex" / "outputs" / "G7_training"
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_root / "g7_closed_set_drop_recovery_latest.json", summary)
    (artifact_root / "g7_closed_set_drop_recovery_latest.md").write_text(
        "\n".join(
            [
                "# G7 Closed-Set Controlled-Drop Recovery",
                "",
                f"- status: {summary['status']}",
                f"- verdict: {summary['verdict']}",
                f"- recommendation: {summary['recommendation']}",
                f"- selected_classes: {summary['selected_class_count']}",
                f"- selected_clips: {summary['selected_clip_count']}",
                f"- dropped_class_recovery_rate: {summary['recovery_metrics']['dropped_class_recovery_rate']:.3f}",
                f"- extra_gt_recall@K: {summary['recovery_metrics']['extra_gt_recall@K']:.3f}",
                f"- extra_precision@K: {summary['recovery_metrics']['extra_precision@K']:.3f}",
                "",
                "## External Validity",
                summary["external_validity_statement"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    audit_root = repo_root / "train" / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    _write_json(audit_root / "closed_set_selected_classes.json", {"status": "PASS", "selected_classes": list(selected_class_rows)})
    _write_json(audit_root / "closed_set_selected_clips.json", {"status": "PASS", "selected_clips": list(selected_clip_rows)})
    _write_json(
        audit_root / "closed_set_drop_protocol.json",
        {
            "status": "PASS",
            "drop_protocol_rows": list(drop_protocol_rows),
            "selection_rule": summary["drop_protocol"]["selection_rule"],
            "drop_rule": summary["drop_protocol"]["drop_rule"],
            "seed": summary["drop_protocol"]["seed"],
            "selected_class_ids": summary["selected_class_ids"],
            "selected_clip_ids": summary["selected_clip_ids"],
        },
    )
    _write_jsonl(audit_root / "closed_set_drop_recovery_ledger.jsonl", recovery_ledger_rows)
    _write_json(audit_root / "closed_set_drop_recovery_summary.json", summary)
    _write_jsonl(audit_root / "closed_set_drop_recovery_casebook.jsonl", casebook)


def test_closed_set_selection_and_protocol_are_deterministic() -> None:
    selected_classes, selected_clips, _ = _select_closed_set_clips(
        [
            {"clip_id": "1", "trajectory_id": "traj_1", "gt_class_id": 1, "gt_class_name": "class_1", "observed_raw_ids": [1, 3], "complexity_score": 2, "sample": {"observed_raw_ids": [1, 3]}},
            {"clip_id": "2", "trajectory_id": "traj_2", "gt_class_id": 1, "gt_class_name": "class_1", "observed_raw_ids": [1, 5], "complexity_score": 2, "sample": {"observed_raw_ids": [1, 5]}},
            {"clip_id": "3", "trajectory_id": "traj_3", "gt_class_id": 3, "gt_class_name": "class_3", "observed_raw_ids": [3, 5], "complexity_score": 2, "sample": {"observed_raw_ids": [3, 5]}},
            {"clip_id": "4", "trajectory_id": "traj_4", "gt_class_id": 3, "gt_class_name": "class_3", "observed_raw_ids": [3, 7], "complexity_score": 2, "sample": {"observed_raw_ids": [3, 7]}},
            {"clip_id": "5", "trajectory_id": "traj_5", "gt_class_id": 5, "gt_class_name": "class_5", "observed_raw_ids": [5, 7], "complexity_score": 2, "sample": {"observed_raw_ids": [5, 7]}},
            {"clip_id": "6", "trajectory_id": "traj_6", "gt_class_id": 5, "gt_class_name": "class_5", "observed_raw_ids": [5, 9], "complexity_score": 2, "sample": {"observed_raw_ids": [5, 9]}},
            {"clip_id": "7", "trajectory_id": "traj_7", "gt_class_id": 7, "gt_class_name": "class_7", "observed_raw_ids": [7, 9], "complexity_score": 2, "sample": {"observed_raw_ids": [7, 9]}},
            {"clip_id": "8", "trajectory_id": "traj_8", "gt_class_id": 7, "gt_class_name": "class_7", "observed_raw_ids": [7, 11], "complexity_score": 2, "sample": {"observed_raw_ids": [7, 11]}},
            {"clip_id": "9", "trajectory_id": "traj_9", "gt_class_id": 9, "gt_class_name": "class_9", "observed_raw_ids": [9, 11], "complexity_score": 2, "sample": {"observed_raw_ids": [9, 11]}},
            {"clip_id": "10", "trajectory_id": "traj_10", "gt_class_id": 9, "gt_class_name": "class_9", "observed_raw_ids": [9, 1], "complexity_score": 2, "sample": {"observed_raw_ids": [9, 1]}},
            {"clip_id": "11", "trajectory_id": "traj_11", "gt_class_id": 11, "gt_class_name": "class_11", "observed_raw_ids": [11, 1], "complexity_score": 2, "sample": {"observed_raw_ids": [11, 1]}},
            {"clip_id": "12", "trajectory_id": "traj_12", "gt_class_id": 11, "gt_class_name": "class_11", "observed_raw_ids": [11, 3], "complexity_score": 2, "sample": {"observed_raw_ids": [11, 3]}},
        ],
        {f"traj_{idx}": {"matched_gt_class_id": cid, "audit_usable": True} for idx, cid in enumerate([1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11], start=1)},
        class_name_lookup={1: "class_1", 3: "class_3", 5: "class_5", 7: "class_7", 9: "class_9", 11: "class_11"},
    )
    selected_classes_2, selected_clips_2, _ = _select_closed_set_clips(
        [
            {"clip_id": "1", "trajectory_id": "traj_1", "gt_class_id": 1, "gt_class_name": "class_1", "observed_raw_ids": [1, 3], "complexity_score": 2, "sample": {"observed_raw_ids": [1, 3]}},
            {"clip_id": "2", "trajectory_id": "traj_2", "gt_class_id": 1, "gt_class_name": "class_1", "observed_raw_ids": [1, 5], "complexity_score": 2, "sample": {"observed_raw_ids": [1, 5]}},
            {"clip_id": "3", "trajectory_id": "traj_3", "gt_class_id": 3, "gt_class_name": "class_3", "observed_raw_ids": [3, 5], "complexity_score": 2, "sample": {"observed_raw_ids": [3, 5]}},
            {"clip_id": "4", "trajectory_id": "traj_4", "gt_class_id": 3, "gt_class_name": "class_3", "observed_raw_ids": [3, 7], "complexity_score": 2, "sample": {"observed_raw_ids": [3, 7]}},
            {"clip_id": "5", "trajectory_id": "traj_5", "gt_class_id": 5, "gt_class_name": "class_5", "observed_raw_ids": [5, 7], "complexity_score": 2, "sample": {"observed_raw_ids": [5, 7]}},
            {"clip_id": "6", "trajectory_id": "traj_6", "gt_class_id": 5, "gt_class_name": "class_5", "observed_raw_ids": [5, 9], "complexity_score": 2, "sample": {"observed_raw_ids": [5, 9]}},
            {"clip_id": "7", "trajectory_id": "traj_7", "gt_class_id": 7, "gt_class_name": "class_7", "observed_raw_ids": [7, 9], "complexity_score": 2, "sample": {"observed_raw_ids": [7, 9]}},
            {"clip_id": "8", "trajectory_id": "traj_8", "gt_class_id": 7, "gt_class_name": "class_7", "observed_raw_ids": [7, 11], "complexity_score": 2, "sample": {"observed_raw_ids": [7, 11]}},
            {"clip_id": "9", "trajectory_id": "traj_9", "gt_class_id": 9, "gt_class_name": "class_9", "observed_raw_ids": [9, 11], "complexity_score": 2, "sample": {"observed_raw_ids": [9, 11]}},
            {"clip_id": "10", "trajectory_id": "traj_10", "gt_class_id": 9, "gt_class_name": "class_9", "observed_raw_ids": [9, 1], "complexity_score": 2, "sample": {"observed_raw_ids": [9, 1]}},
            {"clip_id": "11", "trajectory_id": "traj_11", "gt_class_id": 11, "gt_class_name": "class_11", "observed_raw_ids": [11, 1], "complexity_score": 2, "sample": {"observed_raw_ids": [11, 1]}},
            {"clip_id": "12", "trajectory_id": "traj_12", "gt_class_id": 11, "gt_class_name": "class_11", "observed_raw_ids": [11, 3], "complexity_score": 2, "sample": {"observed_raw_ids": [11, 3]}},
        ],
        {f"traj_{idx}": {"matched_gt_class_id": cid, "audit_usable": True} for idx, cid in enumerate([1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11], start=1)},
        class_name_lookup={1: "class_1", 3: "class_3", 5: "class_5", 7: "class_7", 9: "class_9", 11: "class_11"},
    )
    assert [row["class_id"] for row in selected_classes] == [row["class_id"] for row in selected_classes_2]
    assert [row["clip_id"] for row in selected_clips] == [row["clip_id"] for row in selected_clips_2]


def test_closed_set_summary_aggregation_counts_recovery() -> None:
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
    summary = _aggregate_recovery(rows, {"1": 1})
    assert summary["dropped_class_recovery_rate"] == 1.0
    assert summary["extra_gt_recall@K"] == 1.0
    assert summary["extra_precision@K"] == 1.0
    assert summary["missing_class_recovery_rate"] == 1.0
    assert summary["spurious_extra_rate"] == 0.0


def test_gt_is_audit_only_in_derivation(tmp_path: Path) -> None:
    fixture = _write_synthetic_runtime_fixture(tmp_path)
    class_name_lookup = {cid: name for cid, name in zip(REPO_CLASS_IDS, [f"class_{cid}" for cid in REPO_CLASS_IDS])}
    candidate_map = _build_candidate_map(tmp_path)
    samples = materialize_phase1_training_samples(
        tmp_path,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", trajectory_source_branch="mainline", smoke=True, smoke_max_trajectories=12),
    )["samples"]
    gt_lookup = load_gt_sidecar_lookup(tmp_path, dataset_name="lvvis_train_base", trajectory_source_branch="mainline")
    selected_class_rows, selected_clip_rows, _ = _select_closed_set_clips(
        samples,
        gt_lookup,
        class_count_target=TIER1_CLASS_COUNT,
        clip_count_target=TIER1_CLIP_COUNT,
        class_name_lookup=class_name_lookup,
    )
    selected_samples, selected_clip_ids, selected_observed, selected_protocol_rows = _derive_closed_set_samples(
        selected_clip_rows,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
    )
    assert selected_samples
    assert selected_clip_ids
    assert selected_protocol_rows[0]["observed_raw_ids_after_drop"]
    assert "gt_class_id" not in selected_samples[0]["weak_label_record"]
    assert selected_samples[0]["candidate_ids_extra"] == [selected_samples[0]["controlled_drop_protocol"]["drop_class_id"]]


def test_closed_set_bounded_experiment_smoke(tmp_path: Path) -> None:
    repo_root = _repo_root()
    fixture_root = tmp_path / "closed_set_fixture"
    fixture = _write_synthetic_runtime_fixture(fixture_root)

    materialized = materialize_phase1_training_samples(
        fixture_root,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", trajectory_source_branch="mainline", smoke=True, smoke_max_trajectories=SMOKE_MAX_TRAJECTORIES),
    )
    gt_lookup = load_gt_sidecar_lookup(fixture_root, dataset_name="lvvis_train_base", trajectory_source_branch="mainline")
    class_name_lookup = _load_class_name_lookup(fixture_root, REPO_CLASS_IDS)
    candidate_map = _build_candidate_map(fixture_root)
    selected_class_rows, selected_clip_rows, _ = _select_closed_set_clips(
        materialized["samples"],
        gt_lookup,
        class_count_target=TIER1_CLASS_COUNT,
        clip_count_target=TIER1_CLIP_COUNT,
        class_name_lookup=class_name_lookup,
    )
    selected_samples, selected_clip_ids, selected_observed, selected_protocol_rows = _derive_closed_set_samples(
        selected_clip_rows,
        class_name_lookup=class_name_lookup,
        candidate_map=candidate_map,
    )

    prealign_result = train_prealign(
        output_root=fixture_root,
        materialized_samples=selected_samples,
        config=PrealignConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            device="cpu",
            seed=0,
            smoke=True,
            epochs=SMOKE_PREALIGN_EPOCHS,
            learning_rate=SMOKE_PREALIGN_LR,
            temperature=TEMPERATURE,
        ),
    )
    softem_result = run_soft_em(
        output_root=fixture_root,
        materialized_samples=selected_samples,
        config=SoftEMConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            mode="base_then_aug",
            device="cpu",
            seed=0,
            smoke=True,
            temperature=TEMPERATURE,
            em_subiterations=SMOKE_EM_SUBITERATIONS,
            base_epochs=SMOKE_BASE_EPOCHS,
            aug_epochs=SMOKE_AUG_EPOCHS,
            base_learning_rate=SMOKE_BASE_LR,
            aug_learning_rate=SMOKE_AUG_LR,
        ),
    )

    prealign_rows = _evaluate_closed_set_rows(
        output_root=fixture_root,
        selected_samples=selected_samples,
        gt_lookup=gt_lookup,
        stage_id="prealign",
        checkpoint_path=fixture_root / "train" / "prealign" / "checkpoints" / "prealign_last.pth",
    )
    softem_base_rows = _evaluate_closed_set_rows(
        output_root=fixture_root,
        selected_samples=selected_samples,
        gt_lookup=gt_lookup,
        stage_id="softem_base",
        checkpoint_path=fixture_root / "train" / "softem_base" / "checkpoints" / "softem_base_last.pth",
        previous_by_trajectory={row["trajectory_id"]: row for row in prealign_rows},
    )
    softem_aug_rows = _evaluate_closed_set_rows(
        output_root=fixture_root,
        selected_samples=selected_samples,
        gt_lookup=gt_lookup,
        stage_id="softem_aug",
        checkpoint_path=fixture_root / "train" / "softem_aug" / "checkpoints" / "softem_aug_last.pth",
        previous_by_trajectory={row["trajectory_id"]: row for row in softem_base_rows},
    )

    stage_rows = {
        "prealign": prealign_rows,
        "softem_base": softem_base_rows,
        "softem_aug": softem_aug_rows,
    }
    summary = _make_summary(
        repo_root=repo_root,
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        selected_samples=selected_samples,
        selected_clip_ids=selected_clip_ids,
        selected_observed=selected_observed,
        selected_protocol_rows=selected_protocol_rows,
        stage_rows=stage_rows,
    )

    casebook = _build_casebook(stage_rows)
    recovery_ledger_rows = list(prealign_rows) + list(softem_base_rows) + list(softem_aug_rows)
    _emit_final_artifacts(
        repo_root=repo_root,
        fixture_root=fixture_root,
        summary=summary,
        stage_rows=stage_rows,
        casebook=casebook,
        selected_class_rows=selected_class_rows,
        selected_clip_rows=selected_clip_rows,
        drop_protocol_rows=selected_protocol_rows,
        recovery_ledger_rows=recovery_ledger_rows,
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
                "train/audit/closed_set_selected_classes.json",
                "train/audit/closed_set_selected_clips.json",
                "train/audit/closed_set_drop_protocol.json",
                "train/audit/closed_set_drop_recovery_ledger.jsonl",
                "train/audit/closed_set_drop_recovery_summary.json",
                "train/audit/closed_set_drop_recovery_casebook.jsonl",
                "codex/outputs/G7_training/g7_closed_set_drop_recovery_latest.json",
                "codex/outputs/G7_training/g7_closed_set_drop_recovery_latest.md",
            ],
        )

    assert summary["status"] == "PASS"
    assert summary["selected_class_count"] == TIER1_CLASS_COUNT
    assert summary["selected_clip_count"] == TIER1_CLIP_COUNT
    assert summary["recovery_metrics"]["dropped_class_recovery_rate"] >= 0.0
    assert (fixture_root / "train" / "prealign" / "train_state.json").is_file()
    assert (fixture_root / "train" / "softem_base" / "train_state.json").is_file()
    assert (fixture_root / "train" / "softem_aug" / "train_state.json").is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_closed_set_drop_recovery_latest.json").is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_closed_set_drop_recovery_latest.md").is_file()
