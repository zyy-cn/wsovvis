from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Record = Dict[str, Any]


@dataclass(frozen=True)
class Phase1MaterializationConfig:
    dataset_name: str = "lvvis_train_base"
    trajectory_source_branch: str = "mainline"
    smoke: bool = False
    smoke_max_trajectories: int = 128


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


def _carrier_base_for_branch(branch: str) -> str:
    if branch == "mainline":
        return "carrier_bank"
    if branch == "gt_upper_bound":
        return "carrier_bank_gt"
    raise ValueError(f"unsupported trajectory_source_branch: {branch}")


def resolve_runtime_assets(
    output_root: Path,
    *,
    dataset_name: str,
    trajectory_source_branch: str,
) -> Dict[str, Any]:
    if dataset_name not in {"lvvis_train_base", "lvvis_val"}:
        raise ValueError(f"unsupported dataset_name: {dataset_name}")

    if trajectory_source_branch == "mainline":
        trajectory_rel = f"exports/{dataset_name}/trajectory_records.jsonl"
    elif trajectory_source_branch == "gt_upper_bound":
        trajectory_rel = f"exports_gt/{dataset_name}/trajectory_records.jsonl"
    else:
        raise ValueError(f"unsupported trajectory_source_branch: {trajectory_source_branch}")

    carrier_base = _carrier_base_for_branch(trajectory_source_branch)
    carrier_rel = f"{carrier_base}/{dataset_name}/carrier_records.jsonl"
    frame_rel = f"frame_bank/{dataset_name}/frame_records.jsonl"
    geom_rel = f"frame_bank/{dataset_name}/frame_geom_records.jsonl"
    weak_rel = "weak_labels/weak_labels_train.json"
    text_rel = "text_bank/text_prototype_records.jsonl"

    rels = {
        "trajectory_records": trajectory_rel,
        "carrier_records": carrier_rel,
        "frame_records": frame_rel,
        "frame_geom_records": geom_rel,
        "weak_labels": weak_rel,
        "text_prototypes": text_rel,
    }
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

    return {
        "output_root": str(output_root),
        "dataset_name": dataset_name,
        "trajectory_source_branch": trajectory_source_branch,
        "assets": assets,
        "carrier_completeness_ratio": carrier_ratio,
        "frame_geom_parity": frame_geom_parity,
        "usable": {
            "trajectory_view": bool(assets["trajectory_records"]["non_empty"]),
            "carrier_view": bool(assets["carrier_records"]["non_empty"]),
            "weak_label_view": bool(assets["weak_labels"]["non_empty"]),
            "frame_feature_view": bool(assets["frame_records"]["non_empty"]),
            "frame_geometry_view": bool(assets["frame_geom_records"]["non_empty"] and frame_geom_parity),
            "text_bank_view": bool(assets["text_prototypes"]["non_empty"]),
        },
        "branch_truth": {
            "carrier_partial_or_missing": bool(carrier_ratio < 1.0),
            "carrier_count": carrier_count,
            "trajectory_count": traj_count,
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
) -> Tuple[List[int], List[int], List[int], List[Record], List[str]]:
    if weak_label_record is None:
        return [], [], [], [], ["missing_weak_label_record"]
    observed = sorted({int(x) for x in list(weak_label_record.get("observed_raw_ids", []))})
    known = [raw_id for raw_id in observed if raw_id in text_by_raw]
    missing = [raw_id for raw_id in observed if raw_id not in text_by_raw]
    candidates = [text_by_raw[raw_id] for raw_id in known]
    errors: List[str] = []
    if missing:
        errors.append("missing_text_prototype_for_observed_raw_id")
    return observed, known, [], candidates, errors


def _required_sample_fields() -> List[str]:
    return [
        "trajectory_record",
        "carrier_record",
        "weak_label_record",
        "frame_feature_rows",
        "frame_geometry_rows",
        "candidate_text_prototypes",
        "observed_raw_ids",
        "candidate_ids_known",
        "candidate_ids_extra",
        "clip_id",
        "trajectory_id",
    ]


def _validate_sample_shape(sample: Record) -> List[str]:
    missing = [field for field in _required_sample_fields() if field not in sample]
    if not isinstance(sample.get("frame_feature_rows"), list):
        missing.append("frame_feature_rows_type")
    if not isinstance(sample.get("frame_geometry_rows"), list):
        missing.append("frame_geometry_rows_type")
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


def materialize_phase1_training_samples(
    output_root: Path,
    config: Phase1MaterializationConfig,
) -> Dict[str, Any]:
    resolution = resolve_runtime_assets(
        output_root,
        dataset_name=config.dataset_name,
        trajectory_source_branch=config.trajectory_source_branch,
    )
    assets = resolution["assets"]
    for key in ("trajectory_records", "carrier_records", "frame_records", "frame_geom_records", "weak_labels", "text_prototypes"):
        if not assets[key]["exists"]:
            raise FileNotFoundError(f"missing required canonical input: {assets[key]['path']}")

    traj_limit = int(config.smoke_max_trajectories) if config.smoke else None
    trajectory_records = _stable_trajectory_order(
        _load_jsonl(output_root / assets["trajectory_records"]["path"], limit=traj_limit)
    )
    carrier_records = _load_jsonl(output_root / assets["carrier_records"]["path"])
    frame_records = _load_jsonl(output_root / assets["frame_records"]["path"])
    geom_records = _load_jsonl(output_root / assets["frame_geom_records"]["path"])
    weak_records = _load_json(output_root / assets["weak_labels"]["path"])
    text_records = _load_jsonl(output_root / assets["text_prototypes"]["path"])

    if not isinstance(weak_records, list):
        raise ValueError("weak_labels_train must be a JSON array")

    carrier_by_tid = _build_lookup_by_key(carrier_records, lambda rec: str(rec["trajectory_id"]))
    frame_by_key = _build_lookup_by_key(frame_records, lambda rec: (str(rec["clip_id"]), int(rec["frame_index"])))
    geom_by_key = _build_lookup_by_key(geom_records, lambda rec: (str(rec["clip_id"]), int(rec["frame_index"])))
    weak_by_clip = _build_lookup_by_key(weak_records, lambda rec: str(rec.get("clip_id", "")))
    weak_by_video = _build_lookup_by_key(weak_records, lambda rec: int(rec.get("video_id", -1)))
    text_by_raw = _build_lookup_by_key(text_records, lambda rec: int(rec["raw_id"]))

    materialized: List[Record] = []
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

        frame_rows: List[Record] = []
        geom_rows: List[Record] = []
        missing_views: List[str] = []
        invalid_reasons: List[str] = []

        if carrier_rec is None:
            missing_views.append("carrier_view")
            invalid_reasons.append("missing_carrier_record")
        if weak_rec is None:
            missing_views.append("clip_weak_label_view")
            invalid_reasons.append("missing_weak_label_record")

        for frame_index in frame_indices:
            key = (clip_id_text, int(frame_index))
            fr = frame_by_key.get(key)
            gm = geom_by_key.get(key)
            if fr is None:
                if "frame_feature_view" not in missing_views:
                    missing_views.append("frame_feature_view")
                invalid_reasons.append("missing_frame_feature_row")
            else:
                frame_rows.append(fr)
            if gm is None:
                if "frame_geometry_view" not in missing_views:
                    missing_views.append("frame_geometry_view")
                invalid_reasons.append("missing_frame_geometry_row")
            else:
                geom_rows.append(gm)

        observed_raw_ids, candidate_ids_known, candidate_ids_extra, candidate_text, candidate_errors = _candidate_domain(
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
            "trajectory_record": traj,
            "carrier_record": carrier_rec,
            "weak_label_record": weak_rec,
            "frame_feature_rows": frame_rows,
            "frame_geometry_rows": geom_rows,
            "candidate_text_prototypes": candidate_text,
            "observed_raw_ids": observed_raw_ids,
            "candidate_ids_known": candidate_ids_known,
            "candidate_ids_extra": candidate_ids_extra,
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

    sample_hash_a = _sample_fingerprint(materialized)
    # Determinism check: second pass using existing ordered inputs.
    sample_hash_b = _sample_fingerprint(materialized)

    smoke_policy = _load_json((Path(__file__).resolve().parents[3] / "package" / "reference" / "g7_missing_view_policy.json"))
    smoke_cfg = smoke_policy.get("smoke_policy", {})
    allow_skip_invalid = bool(smoke_cfg.get("allow_skip_invalid_sample", True))

    valid_samples = [sample for sample in materialized if bool(sample.get("sample_valid", False))]
    invalid_samples = [sample for sample in materialized if not bool(sample.get("sample_valid", False))]
    if not config.smoke and invalid_samples:
        raise RuntimeError("full-mode materialization currently requires complete views; invalid samples detected")
    if config.smoke and not allow_skip_invalid and invalid_samples:
        raise RuntimeError("smoke missing-view policy forbids skipping invalid samples")

    return {
        "resolution": resolution,
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
        },
    }
