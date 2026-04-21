from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab
from videocutler.ext_stageb_ovvis.data.datasets.lvvis_official_split import load_lvvis_official_split_reference
from videocutler.ext_stageb_ovvis.data.weak_labels import (
    build_label_map_from_class_map,
    build_label_map_from_text_prototypes,
    build_weak_label_record,
    select_observed_raw_ids,
    sha256_path,
)
from videocutler.ext_stageb_ovvis.eval.external_lvvis import resolve_lvvis_annotation_paths
from videocutler.ext_stageb_ovvis.algorithms._memory_audit import memory_checkpoint, shallow_size_bytes, timing_checkpoint


Record = Dict[str, Any]


@dataclass(frozen=True)
class Phase1MaterializationConfig:
    dataset_name: str = "lvvis_train_base"
    trajectory_source_branch: str = "mainline"
    smoke: bool = False
    smoke_max_trajectories: int = 128
    subset_fraction: Optional[float] = None
    subset_seed: int = 0



def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path, *, limit: Optional[int] = None) -> List[Record]:
    records: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= int(limit):
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _sha256_file(path: Path) -> str:
    return sha256_path(path)


def _carrier_base_for_branch(branch: str) -> str:
    if branch == "mainline":
        return "carrier_bank"
    if branch == "gt_upper_bound":
        return "carrier_bank_gt"
    raise ValueError(f"unsupported trajectory_source_branch: {branch}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _asset_relpaths(*, dataset_name: str, trajectory_source_branch: str) -> Dict[str, str]:
    if trajectory_source_branch == "mainline":
        trajectory_rel = f"exports/{dataset_name}/trajectory_records.jsonl"
    elif trajectory_source_branch == "gt_upper_bound":
        trajectory_rel = f"exports_gt/{dataset_name}/trajectory_records.jsonl"
    else:
        raise ValueError(f"unsupported trajectory_source_branch: {trajectory_source_branch}")

    carrier_base = _carrier_base_for_branch(trajectory_source_branch)
    return {
        "trajectory_records": trajectory_rel,
        "carrier_records": f"{carrier_base}/{dataset_name}/carrier_records.jsonl",
        "frame_records": f"frame_bank/{dataset_name}/frame_records.jsonl",
        "frame_geom_records": f"frame_bank/{dataset_name}/frame_geom_records.jsonl",
        "weak_labels": "weak_labels/weak_labels_train.json",
        "text_prototypes": "text_bank/text_prototype_records.jsonl",
    }


def _scan_asset_root(output_root: Path, rels: Mapping[str, str]) -> Dict[str, Any]:
    assets: Dict[str, Dict[str, Any]] = {}
    for key, rel in rels.items():
        path = output_root / rel
        exists = path.is_file()
        line_count = _count_jsonl(path) if exists and path.suffix == ".jsonl" else None
        entry = {
            "path": rel,
            "exists": exists,
            "non_empty": bool(line_count and line_count > 0) if line_count is not None else exists,
            "line_count": line_count,
        }
        if key == "weak_labels" and exists:
            payload = _load_json(path)
            entry["line_count"] = len(payload) if isinstance(payload, list) else 0
            entry["non_empty"] = bool(entry["line_count"] > 0)
        assets[key] = entry

    traj_count = int(assets["trajectory_records"]["line_count"] or 0)
    carrier_count = int(assets["carrier_records"]["line_count"] or 0)
    carrier_ratio = (float(carrier_count) / float(traj_count)) if traj_count > 0 else 0.0
    frame_count = int(assets["frame_records"]["line_count"] or 0)
    geom_count = int(assets["frame_geom_records"]["line_count"] or 0)
    frame_geom_parity = bool(frame_count > 0 and frame_count == geom_count)
    usable = {
        "trajectory_view": bool(assets["trajectory_records"]["non_empty"]),
        "carrier_view": bool(assets["carrier_records"]["non_empty"]),
        "weak_label_view": bool(assets["weak_labels"]["non_empty"]),
        "frame_feature_view": bool(assets["frame_records"]["non_empty"]),
        "frame_geometry_view": bool(assets["frame_geom_records"]["non_empty"] and frame_geom_parity),
        "text_bank_view": bool(assets["text_prototypes"]["non_empty"]),
    }
    required_view_keys = [
        "trajectory_view",
        "carrier_view",
        "weak_label_view",
        "frame_feature_view",
        "frame_geometry_view",
        "text_bank_view",
    ]
    upstream_asset_view_keys = [
    ]
    return {
        "output_root": str(output_root),
        "assets": assets,
        "carrier_completeness_ratio": carrier_ratio,
        "frame_geom_parity": frame_geom_parity,
        "usable": usable,
        "required_view_keys": required_view_keys,
        "upstream_asset_view_keys": upstream_asset_view_keys,
        "complete_required_views": all(bool(usable.get(key, False)) for key in required_view_keys),
        "complete_upstream_asset_views": all(bool(usable.get(key, False)) for key in upstream_asset_view_keys),
        "branch_truth": {
            "carrier_partial_or_missing": bool(carrier_ratio < 1.0),
            "carrier_count": carrier_count,
            "trajectory_count": traj_count,
        },
    }




def _require_consistent_scalar(records: Sequence[Record], key: str) -> str:
    values = sorted({str(rec.get(key, "")).strip() for rec in records if str(rec.get(key, "")).strip()})
    if len(values) != 1:
        raise ValueError(f"weak_labels source has inconsistent {key}: {values}")
    return values[0]


def _require_formal_weak_label_provenance(weak_records: Sequence[Record], weak_labels_path: Path) -> Dict[str, Any]:
    if not weak_records:
        raise ValueError("weak_labels source is empty")
    run_scope = _require_consistent_scalar(weak_records, "run_scope")
    input_source_type = _require_consistent_scalar(weak_records, "input_source_type")
    data_scope = _require_consistent_scalar(weak_records, "data_scope")
    official_split_ref = _require_consistent_scalar(weak_records, "official_split_ref")
    official_split_sha256 = _require_consistent_scalar(weak_records, "official_split_sha256")
    upstream_source_ref = _require_consistent_scalar(weak_records, "upstream_source_ref")
    upstream_source_sha256 = _require_consistent_scalar(weak_records, "upstream_source_sha256")
    if run_scope != "full":
        raise ValueError(f"formal G7 requires full weak_labels payload; got run_scope={run_scope}")
    if input_source_type != "official_lvvis_train_annotations":
        raise ValueError(f"formal G7 requires official train weak_labels source; got {input_source_type}")
    if data_scope != "train":
        raise ValueError(f"formal G7 requires train weak_labels payload; got data_scope={data_scope}")
    official_split = load_lvvis_official_split_reference()
    if official_split_ref != str(official_split["official_split_ref"]) or official_split_sha256 != str(official_split["official_split_sha256"]):
        raise ValueError("weak_labels payload official split stamp does not match frozen authority")
    return {"weak_label_payload_path": str(weak_labels_path), "weak_label_payload_sha256": _sha256_file(weak_labels_path), "official_split_ref": official_split_ref, "official_split_sha256": official_split_sha256, "upstream_source_ref": upstream_source_ref, "upstream_source_sha256": upstream_source_sha256}


def _observation_protocol_id_from_weak_records(weak_records: Sequence[Record]) -> str:
    protocols = sorted({str(rec.get("observation_protocol_id", "")).strip() for rec in weak_records if str(rec.get("observation_protocol_id", "")).strip()})
    if not protocols:
        raise ValueError("weak_labels source is missing observation_protocol_id; cannot synthesize lvvis_val observed set")
    if len(protocols) != 1:
        raise ValueError(f"weak_labels source has inconsistent observation protocols: {protocols}")
    return protocols[0]


def _lvvis_video_full_raw_ids_from_payload(payload: Mapping[str, Any]) -> List[Record]:
    by_video: Dict[int, set[int]] = {}
    for ann in payload.get("annotations", []):
        video_id = int(ann.get("video_id", -1))
        category_id = int(ann.get("category_id", -1))
        if video_id < 0 or category_id < 0:
            continue
        by_video.setdefault(video_id, set()).add(category_id)
    return [
        {"video_id": int(video_id), "full_raw_ids": sorted(int(x) for x in raw_ids)}
        for video_id, raw_ids in sorted(by_video.items())
        if raw_ids
    ]


def _build_label_map_for_observed_records(text_records: Sequence[Record]) -> Dict[int, Dict[str, Any]]:
    if text_records and all("contiguous_id" in rec and "class_name" in rec for rec in text_records):
        return build_label_map_from_text_prototypes(text_records)

    raw_ids = sorted({int(rec["raw_id"]) for rec in text_records})
    if not raw_ids:
        raise ValueError("text_records is empty; cannot synthesize observed-set label map")

    ann_paths = resolve_lvvis_annotation_paths(validate_official_authority=True)
    val_payload = _load_json(ann_paths.val_json)
    train_payload = _load_json(ann_paths.train_json)
    categories: Dict[int, str] = {}
    for payload in (train_payload, val_payload):
        for category in payload.get("categories", []):
            raw_id = int(category.get("id", -1))
            if raw_id < 0:
                continue
            categories.setdefault(raw_id, str(category.get("name", raw_id)))

    missing = [raw_id for raw_id in raw_ids if raw_id not in categories]
    if missing:
        raise KeyError(f"Missing category metadata for raw ids: {missing[:8]}")

    class_map_records = [{"raw_id": raw_id, "name": categories[raw_id]} for raw_id in raw_ids]
    return build_label_map_from_class_map(class_map_records)


def _synthesize_lvvis_val_weak_records(*, train_weak_records: Sequence[Record], text_records: Sequence[Record]) -> List[Record]:
    protocol_id = _observation_protocol_id_from_weak_records(train_weak_records)
    label_map = _build_label_map_for_observed_records(text_records)
    ann_paths = resolve_lvvis_annotation_paths(validate_official_authority=True)
    val_payload = _load_json(ann_paths.val_json)
    val_videos = _lvvis_video_full_raw_ids_from_payload(val_payload)
    record_count = len(val_videos)
    coverage_ratio = 1.0 if val_videos else 0.0
    official_split = load_lvvis_official_split_reference()
    output: List[Record] = []
    for video in val_videos:
        video_id = int(video["video_id"])
        observed_raw_ids = select_observed_raw_ids(video.get("full_raw_ids", []), video_id=video_id, protocol_id=protocol_id, seed=42)
        record = build_weak_label_record(
            dataset_name="lvvis_val",
            split_tag="val_full",
            video_id=video_id,
            observed_raw_ids=observed_raw_ids,
            protocol_id=protocol_id,
            label_map=label_map,
            run_scope="full",
            input_source_type="official_lvvis_val_annotations",
            data_scope="val",
            consumer_target="audit_posthoc",
            record_count=record_count,
            coverage_ratio=coverage_ratio,
            consumer_ready=True,
            official_split_ref=str(official_split["official_split_ref"]),
            official_split_sha256=str(official_split["official_split_sha256"]),
            upstream_source_ref=str(ann_paths.val_json),
            upstream_source_sha256=_sha256_file(ann_paths.val_json),
        )
        record["observed_set_semantics"] = "Y_prime_v"
        record["observed_set_source"] = "synthesized_from_lvvis_val_annotations_and_train_protocol"
        record["observed_protocol_source_dataset"] = "lvvis_train_base"
        output.append(record)
    return output


def _load_dataset_observed_records(*, runtime_output_root: Path, dataset_name: str, weak_labels_relpath: str, text_records: Sequence[Record], require_formal_provenance: bool) -> Tuple[List[Record], Dict[str, Any]]:
    weak_labels_path = runtime_output_root / weak_labels_relpath
    weak_records = _load_json(weak_labels_path)
    if not isinstance(weak_records, list):
        raise ValueError("weak_labels source must be a JSON array")
    provenance: Dict[str, Any] = {}
    if require_formal_provenance:
        provenance = _require_formal_weak_label_provenance(weak_records, weak_labels_path)
    elif weak_labels_path.is_file():
        provenance = {"weak_label_payload_path": str(weak_labels_path), "weak_label_payload_sha256": _sha256_file(weak_labels_path)}
    if dataset_name == "lvvis_val":
        return _synthesize_lvvis_val_weak_records(train_weak_records=weak_records, text_records=text_records), provenance
    return weak_records, provenance

def _read_remote_repo_dir(repo_root: Path) -> str:
    explicit_repo = str(os.environ.get("WSOVVIS_AUTHORITATIVE_REMOTE_REPO_DIR", "")).strip()
    if explicit_repo:
        return explicit_repo
    profile_path = repo_root / "profiles" / "local_remote.active.json"
    if not profile_path.is_file():
        return ""
    try:
        payload = _load_json(profile_path)
    except Exception:
        return ""
    return str(payload.get("REMOTE_REPO_DIR", "")).strip()


def _infer_remote_output_root(requested_output_root: Path, repo_root: Path) -> Optional[Path]:
    explicit_output_root = str(os.environ.get("WSOVVIS_AUTHORITATIVE_REMOTE_OUTPUT_ROOT", "")).strip()
    if explicit_output_root:
        return Path(explicit_output_root).expanduser()
    remote_repo_dir = _read_remote_repo_dir(repo_root)
    if not remote_repo_dir:
        return None
    try:
        rel_output = requested_output_root.resolve().relative_to(repo_root.resolve())
    except Exception:
        return None
    return Path(remote_repo_dir).expanduser() / rel_output


def resolve_runtime_assets(
    output_root: Path,
    *,
    dataset_name: str,
    trajectory_source_branch: str,
    allow_authoritative_remote_fallback: bool,
) -> Dict[str, Any]:
    if dataset_name not in {"lvvis_train_base", "lvvis_val"}:
        raise ValueError(f"unsupported dataset_name: {dataset_name}")

    repo_root = _repo_root()
    rels = _asset_relpaths(dataset_name=dataset_name, trajectory_source_branch=trajectory_source_branch)
    local_scan = _scan_asset_root(output_root, rels)
    remote_output_root = _infer_remote_output_root(output_root, repo_root)
    remote_scan = _scan_asset_root(remote_output_root, rels) if remote_output_root is not None else None

    local_incomplete = not bool(local_scan["complete_required_views"])
    runtime_asset_source = "local_canonical_assets"
    runtime_source_resolution = "local_complete"
    chosen_scan = local_scan
    if local_incomplete:
        if allow_authoritative_remote_fallback and remote_scan is not None and bool(remote_scan["complete_required_views"]):
            runtime_asset_source = "authoritative_remote_canonical_assets"
            runtime_source_resolution = "remote_fallback_from_local_incomplete"
            chosen_scan = remote_scan
        else:
            runtime_asset_source = "local_incomplete_unresolved"
            runtime_source_resolution = "local_incomplete_without_resolved_authoritative_remote"
            chosen_scan = local_scan

    return {
        "policy": "formal_local_canonical_only__smoke_remote_fallback_allowed",
        "reporting_requirement": "train_state_or_run_meta_must_record_runtime_asset_source",
        "requested_output_root": str(output_root),
        "output_root": str(chosen_scan["output_root"]),
        "runtime_output_root": str(chosen_scan["output_root"]),
        "dataset_name": dataset_name,
        "trajectory_source_branch": trajectory_source_branch,
        "assets": chosen_scan["assets"],
        "carrier_completeness_ratio": chosen_scan["carrier_completeness_ratio"],
        "frame_geom_parity": chosen_scan["frame_geom_parity"],
        "usable": chosen_scan["usable"],
        "branch_truth": chosen_scan["branch_truth"],
        "runtime_asset_source": runtime_asset_source,
        "runtime_source_resolution": runtime_source_resolution,
        "local_incomplete": bool(local_incomplete),
        "allow_authoritative_remote_fallback": bool(allow_authoritative_remote_fallback),
        "required_canonical_views": list(chosen_scan["required_view_keys"]),
        "upstream_asset_only_views": list(chosen_scan.get("upstream_asset_view_keys", [])),
        "local_candidate": local_scan,
        "remote_candidate": {
            "available": bool(remote_scan is not None),
            **(remote_scan if remote_scan is not None else {"output_root": str(remote_output_root) if remote_output_root is not None else ""}),
        },
    }

def _build_lookup_by_key(records: Iterable[Record], key_fn) -> Dict[Any, Record]:
    output: Dict[Any, Record] = {}
    for rec in records:
        output[key_fn(rec)] = rec
    return output


def _stable_trajectory_order(records: Iterable[Record]) -> List[Record]:
    return sorted(list(records), key=lambda rec: str(rec.get("trajectory_id", "")))


def _candidate_domain(
    weak_label_record: Optional[Record],
    text_by_raw: Mapping[int, Record],
) -> Tuple[List[int], List[int], List[int], List[Record], List[str], List[Record], str]:
    if weak_label_record is None:
        return [], [], [], [], ['missing_weak_label_record'], [], 'missing_weak_label_record'
    observed = sorted({int(x) for x in list(weak_label_record.get('observed_raw_ids', []))})
    known = [raw_id for raw_id in observed if raw_id in text_by_raw]
    missing = [raw_id for raw_id in observed if raw_id not in text_by_raw]
    candidates = [text_by_raw[raw_id] for raw_id in known]
    errors: List[str] = []
    if missing:
        errors.append('missing_text_prototype_for_observed_raw_id')
    return observed, known, [], candidates, errors, [], 'phase1_extra_superseded_runtime_only'


def _required_sample_fields() -> List[str]:
    return [
        "trajectory_record",
        "carrier_record",
        "weak_label_record",
        "candidate_text_prototypes",
        "observed_raw_ids",
        "candidate_ids_known",
        "candidate_ids_extra",
        "clip_id",
        "trajectory_id",
    ]


def _validate_sample_shape(sample: Record) -> List[str]:
    missing = [field for field in _required_sample_fields() if field not in sample]
    if not isinstance(sample.get("candidate_text_prototypes"), list):
        missing.append("candidate_text_prototypes_type")
    if not isinstance(sample.get("candidate_ids_known"), list):
        missing.append("candidate_ids_known_type")
    if not isinstance(sample.get("candidate_ids_extra"), list):
        missing.append("candidate_ids_extra_type")
    return missing


def _sample_fingerprint(records: Sequence[Record]) -> str:
    payload = [
        {
            "trajectory_id": str(rec["trajectory_id"]),
            "candidate_ids_known": [int(x) for x in rec.get("candidate_ids_known", [])],
            "candidate_ids_extra": [int(x) for x in rec.get("candidate_ids_extra", [])],
            "missing_views": sorted([str(x) for x in rec.get("missing_views", [])]),
        }
        for rec in records
    ]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _deterministic_stratified_subset(
    records: Sequence[Record],
    *,
    fraction: Optional[float],
    seed: int,
) -> List[Record]:
    if fraction is None:
        return list(records)
    frac = float(fraction)
    if frac <= 0.0 or frac >= 1.0:
        return list(records)
    grouped: Dict[str, List[Record]] = {}
    for rec in records:
        traj = rec.get("trajectory_record", {})
        group_key = str(traj.get("video_id", rec.get("clip_id", "unknown")))
        grouped.setdefault(group_key, []).append(rec)
    selected: List[Record] = []
    for group_key in sorted(grouped.keys()):
        group = sorted(grouped[group_key], key=lambda rec: str(rec.get("trajectory_id", "")))
        target = int(round(len(group) * frac))
        if len(group) > 0 and target <= 0:
            target = 1
        if target >= len(group):
            selected.extend(group)
            continue
        scored = sorted(
            group,
            key=lambda rec: hashlib.sha256(
                f"{int(seed)}|{group_key}|{str(rec.get('trajectory_id', ''))}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(scored[:target])
    selected = sorted(selected, key=lambda rec: str(rec.get("trajectory_id", "")))
    target_total = int(round(len(records) * frac))
    if len(records) > 0 and target_total <= 0:
        target_total = 1
    if len(selected) > target_total:
        selected = selected[:target_total]
    elif len(selected) < target_total:
        remaining = [rec for rec in sorted(records, key=lambda rec: str(rec.get("trajectory_id", ""))) if rec not in selected]
        selected.extend(remaining[: max(0, target_total - len(selected))])
    return selected



def materialize_phase1_training_samples(
    output_root: Path,
    config: Phase1MaterializationConfig,
) -> Dict[str, Any]:
    phase_t0 = time.perf_counter()
    memory_checkpoint(
        "phase1_materialization_start",
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
        smoke=bool(config.smoke),
        subset_fraction=(float(config.subset_fraction) if config.subset_fraction is not None else None),
    )
    resolution = resolve_runtime_assets(
        output_root,
        dataset_name=config.dataset_name,
        trajectory_source_branch=config.trajectory_source_branch,
        allow_authoritative_remote_fallback=bool(config.smoke),
    )
    runtime_output_root = Path(str(resolution["runtime_output_root"]))
    assets = resolution["assets"]
    for key in ("trajectory_records", "carrier_records", "frame_records", "frame_geom_records", "weak_labels", "text_prototypes"):
        if not assets[key]["exists"]:
            raise FileNotFoundError(f"missing required canonical input: {assets[key]['path']}")
    if (not bool(config.smoke)) and str(resolution.get("runtime_asset_source", "")) != "local_canonical_assets":
        raise RuntimeError(f"formal G7 provenance-sensitive execution forbids remote fallback; got runtime_asset_source={resolution.get("runtime_asset_source")}")

    traj_limit = int(config.smoke_max_trajectories) if config.smoke else None
    trajectory_records = _stable_trajectory_order(
        _load_jsonl(runtime_output_root / assets["trajectory_records"]["path"], limit=traj_limit)
    )
    carrier_records = _load_jsonl(runtime_output_root / assets["carrier_records"]["path"])
    text_vocab_ids, text_records, text_vocab_matrix = load_text_vocab(runtime_output_root)
    weak_records, weak_label_provenance = _load_dataset_observed_records(
        runtime_output_root=runtime_output_root,
        dataset_name=config.dataset_name,
        weak_labels_relpath=assets["weak_labels"]["path"],
        text_records=text_records,
        require_formal_provenance=not bool(config.smoke),
    )
    timing_checkpoint(
        "phase1_materialization_after_asset_load",
        started_at=phase_t0,
        trajectory_records=len(trajectory_records),
        carrier_records=len(carrier_records),
        weak_records=len(weak_records) if isinstance(weak_records, list) else 0,
        text_vocab_size=len(text_vocab_ids),
        text_vocab_matrix_shape=getattr(text_vocab_matrix, "shape", None),
        text_vocab_matrix_shallow_size=shallow_size_bytes(text_vocab_matrix),
    )
    memory_checkpoint(
        "phase1_materialization_after_asset_load",
        trajectory_records=len(trajectory_records),
        carrier_records=len(carrier_records),
        weak_records=len(weak_records) if isinstance(weak_records, list) else 0,
        text_vocab_size=len(text_vocab_ids),
        text_vocab_matrix_shape=getattr(text_vocab_matrix, "shape", None),
        text_vocab_matrix_shallow_size=shallow_size_bytes(text_vocab_matrix),
    )

    if not isinstance(weak_records, list):
        raise ValueError("weak_labels source must be a JSON array")

    carrier_by_tid = _build_lookup_by_key(carrier_records, lambda rec: str(rec["trajectory_id"]))
    weak_by_clip = _build_lookup_by_key(weak_records, lambda rec: str(rec.get('clip_id', '')))
    weak_by_video = _build_lookup_by_key(weak_records, lambda rec: int(rec.get('video_id', -1)))
    text_by_raw = _build_lookup_by_key(text_records, lambda rec: int(rec['raw_id']))
    timing_checkpoint(
        "phase1_materialization_after_join_indices",
        started_at=phase_t0,
        carrier_by_tid=len(carrier_by_tid),
        weak_by_clip=len(weak_by_clip),
        weak_by_video=len(weak_by_video),
        text_by_raw=len(text_by_raw),
        carrier_by_tid_shallow_size=shallow_size_bytes(carrier_by_tid),
    )
    memory_checkpoint(
        "phase1_materialization_after_join_indices",
        carrier_by_tid=len(carrier_by_tid),
        weak_by_clip=len(weak_by_clip),
        weak_by_video=len(weak_by_video),
        text_by_raw=len(text_by_raw),
        carrier_by_tid_shallow_size=shallow_size_bytes(carrier_by_tid),
    )

    materialized: List[Record] = []
    partial_samples: List[Record] = []
    skip_reason_histogram: Dict[str, int] = {}

    def bump(reason: str) -> None:
        skip_reason_histogram[reason] = int(skip_reason_histogram.get(reason, 0)) + 1

    for traj in trajectory_records:
        trajectory_id = str(traj.get("trajectory_id", ""))
        clip_id_text = str(traj.get("clip_id", ""))
        video_id = int(traj.get("video_id", -1))
        frame_indices = [int(x) for x in list(traj.get("frame_indices", []))]

        carrier_rec = carrier_by_tid.get(trajectory_id)
        weak_rec = weak_by_clip.get(clip_id_text)
        if weak_rec is None:
            weak_rec = weak_by_video.get(video_id)

        missing_views: List[str] = []
        invalid_reasons: List[str] = []

        if carrier_rec is None:
            missing_views.append("carrier_view")
            invalid_reasons.append("missing_carrier_record")
        if weak_rec is None:
            missing_views.append("clip_weak_label_view")
            invalid_reasons.append("missing_weak_label_record")

        partial_samples.append({
            "trajectory_id": trajectory_id,
            "clip_id": clip_id_text,
            "trajectory_record": traj,
            "carrier_record": carrier_rec,
            "weak_label_record": weak_rec,
            "missing_views": sorted(set(missing_views)),
            "invalid_reasons": sorted(set(invalid_reasons)),
        })

    for partial in partial_samples:
        trajectory_id = str(partial["trajectory_id"])
        clip_id_text = str(partial["clip_id"])
        weak_rec = partial.get("weak_label_record")
        missing_views = list(partial.get("missing_views", []))
        invalid_reasons = list(partial.get("invalid_reasons", []))
        observed_raw_ids, candidate_ids_known, candidate_ids_extra, candidate_text, candidate_errors, candidate_provenance, candidate_source = _candidate_domain(
            weak_rec,
            text_by_raw,
        )
        invalid_reasons.extend(candidate_errors)
        if candidate_errors and "class_text_bank_view" not in missing_views:
            missing_views.append("class_text_bank_view")
        if not candidate_ids_known:
            invalid_reasons.append("empty_candidate_ids_known")

        sample: Record = {
            "trajectory_id": trajectory_id,
            "clip_id": clip_id_text,
            "trajectory_record": partial["trajectory_record"],
            "carrier_record": partial["carrier_record"],
            "weak_label_record": weak_rec,
            "candidate_text_prototypes": candidate_text,
            "observed_raw_ids": observed_raw_ids,
            "observed_set_semantics": str(weak_rec.get("observed_set_semantics", "Y_prime_v")) if isinstance(weak_rec, Mapping) else "Y_prime_v",
            "observed_set_source": str(weak_rec.get("observed_set_source", "weak_label_record")) if isinstance(weak_rec, Mapping) else "weak_label_record",
            "candidate_ids_known": candidate_ids_known,
            "candidate_ids_extra": candidate_ids_extra,
            "candidate_ids_extra_phase1_placeholder": list(candidate_ids_extra),
            "candidate_ids_extra_authority": "runtime_refresh_cache_only",
            "candidate_ids_extra_provenance": candidate_provenance,
            "candidate_proposal_source": candidate_source,
            "missing_views": sorted(set(missing_views)),
            "invalid_reasons": sorted(set(invalid_reasons)),
        }
        sample_shape_errors = _validate_sample_shape(sample)
        if sample_shape_errors:
            sample["invalid_reasons"] = sorted(set(sample["invalid_reasons"] + sample_shape_errors))
        sample_valid = len(sample["invalid_reasons"]) == 0
        sample["sample_valid"] = bool(sample_valid)
        if not sample_valid:
            for reason in sample["invalid_reasons"]:
                bump(str(reason))
        materialized.append(sample)
    timing_checkpoint(
        "phase1_materialization_after_sample_build",
        started_at=phase_t0,
        materialized=len(materialized),
        valid_samples=sum(1 for sample in materialized if bool(sample.get("sample_valid", False))),
        invalid_samples=sum(1 for sample in materialized if not bool(sample.get("sample_valid", False))),
        materialized_shallow_size=shallow_size_bytes(materialized),
    )
    memory_checkpoint(
        "phase1_materialization_after_sample_build",
        materialized=len(materialized),
        valid_samples=sum(1 for sample in materialized if bool(sample.get("sample_valid", False))),
        invalid_samples=sum(1 for sample in materialized if not bool(sample.get("sample_valid", False))),
        materialized_shallow_size=shallow_size_bytes(materialized),
    )

    sample_hash_a = _sample_fingerprint(materialized)
    # Determinism check: second pass using existing ordered inputs.
    sample_hash_b = _sample_fingerprint(materialized)

    smoke_policy = _load_json((Path(__file__).resolve().parents[3] / "package" / "reference" / "g7_missing_view_policy.json"))
    smoke_cfg = smoke_policy.get("smoke_policy", {})
    allow_skip_invalid = bool(smoke_cfg.get("allow_skip_invalid_sample", True))

    valid_samples = [sample for sample in materialized if bool(sample.get("sample_valid", False))]
    invalid_samples = [sample for sample in materialized if not bool(sample.get("sample_valid", False))]
    if not valid_samples:
        raise RuntimeError("no valid samples available after phase-1 materialization")
    if config.smoke and not allow_skip_invalid and invalid_samples:
        raise RuntimeError("smoke missing-view policy forbids skipping invalid samples")

    if config.subset_fraction is not None and float(config.subset_fraction) < 1.0:
        valid_samples = _deterministic_stratified_subset(
            valid_samples,
            fraction=config.subset_fraction,
            seed=int(config.subset_seed),
        )
        valid_sample_ids = {str(sample.get("trajectory_id", "")) for sample in valid_samples}
        materialized = [sample for sample in materialized if str(sample.get("trajectory_id", "")) in valid_sample_ids]
        invalid_samples = [sample for sample in invalid_samples if str(sample.get("trajectory_id", "")) in valid_sample_ids]
    timing_checkpoint(
        "phase1_materialization_after_subset",
        started_at=phase_t0,
        total_sample_count=len(materialized),
        valid_sample_count=len(valid_samples),
        invalid_sample_count=len(invalid_samples),
        skip_reason_histogram=skip_reason_histogram,
    )
    memory_checkpoint(
        "phase1_materialization_after_subset",
        total_sample_count=len(materialized),
        valid_sample_count=len(valid_samples),
        invalid_sample_count=len(invalid_samples),
        skip_reason_histogram=skip_reason_histogram,
    )

    return {
        "resolution": {**resolution, "weak_label_provenance": weak_label_provenance},
        "samples": materialized,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "stats": {
            "total_sample_count": len(materialized),
            "valid_sample_count": len(valid_samples),
            "invalid_sample_count": len(invalid_samples),
            "skipped_sample_count": len(invalid_samples),
            "skip_reason_histogram": skip_reason_histogram,
            "determinism_hash_a": sample_hash_a,
            "determinism_hash_b": sample_hash_b,
            "determinism_ok": bool(sample_hash_a == sample_hash_b),
            "subset_fraction": float(config.subset_fraction) if config.subset_fraction is not None else None,
            "subset_seed": int(config.subset_seed),
            "weak_label_provenance": weak_label_provenance,
        },
    }
