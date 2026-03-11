from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wsovvis.features.track_dino_feature_v9 import load_track_dino_feature_cache_v9


SCHEMA_VERSION = "1.0.0"
PROTOTYPE_SCHEMA_NAME = "wsovvis.stagec.label_prototypes.v1"
SUMMARY_SCHEMA_NAME = "wsovvis.prototype_bank_summary_v9"
WORKED_EXAMPLE_SCHEMA_NAME = "wsovvis.prototype_bank_worked_example_v9"


class PrototypeBankError(RuntimeError):
    """Raised when the bounded G5 prototype-bank artifact is invalid."""


@dataclass(frozen=True)
class PrototypeBankConfig:
    observed_label_policy: str = "single_observed_label_only"
    track_selection_rule: str = "highest_objectness_then_member_count_then_lowest_global_track_id"
    prototype_aggregation_rule: str = "o_tau_weighted_mean_then_l2_normalize"
    support_weight_floor: float = 1e-6

    def canonical_dict(self) -> Dict[str, Any]:
        _require(
            self.observed_label_policy == "single_observed_label_only",
            "config.observed_label_policy",
            "must equal 'single_observed_label_only'",
        )
        _require(
            self.track_selection_rule == "highest_objectness_then_member_count_then_lowest_global_track_id",
            "config.track_selection_rule",
            "must equal the bounded G5 conservative rule",
        )
        _require(
            self.prototype_aggregation_rule == "o_tau_weighted_mean_then_l2_normalize",
            "config.prototype_aggregation_rule",
            "must equal the bounded G5 aggregation rule",
        )
        _require(
            isinstance(self.support_weight_floor, (int, float)) and float(self.support_weight_floor) > 0.0,
            "config.support_weight_floor",
            "must be numeric > 0",
        )
        return {
            "observed_label_policy": self.observed_label_policy,
            "track_selection_rule": self.track_selection_rule,
            "prototype_aggregation_rule": self.prototype_aggregation_rule,
            "support_weight_floor": float(self.support_weight_floor),
        }


@dataclass(frozen=True)
class PrototypeSupportRecord:
    video_id: str
    global_track_id: int
    representative_source_track_id: int | str
    o_tau: float
    member_count: int
    num_active_frames: int
    start_frame_idx: int
    end_frame_idx: int


@dataclass(frozen=True)
class PrototypeLabelRecord:
    label_id: int | str
    label_text: str
    row_index: int
    support_video_count: int
    support_track_count: int
    support_weight_sum: float
    support_refs: Tuple[PrototypeSupportRecord, ...]


@dataclass(frozen=True)
class PrototypeBankRecord:
    metadata: PrototypeLabelRecord
    prototype: np.ndarray


class PrototypeBankView:
    def __init__(self, root: Path, manifest_path: Path, manifest: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        self.manifest = manifest
        _require(
            manifest.get("schema_name") == PROTOTYPE_SCHEMA_NAME,
            "prototype_manifest.schema_name",
            f"must equal '{PROTOTYPE_SCHEMA_NAME}'",
        )
        major = _parse_major_version(manifest.get("schema_version"), "prototype_manifest.schema_version")
        _require(major == 1, "prototype_manifest.schema_version", "unsupported major version")
        _require(
            isinstance(manifest.get("embedding_dim"), int) and int(manifest["embedding_dim"]) > 0,
            "prototype_manifest.embedding_dim",
            "must be integer > 0",
        )
        _require(manifest.get("dtype") == "float32", "prototype_manifest.dtype", "must equal 'float32'")
        self.embedding_dim = int(manifest["embedding_dim"])
        self.dtype = str(manifest["dtype"])
        self.split = str(manifest.get("split", ""))
        self.array_key = str(manifest.get("array_key", "prototypes"))
        arrays_rel = _require_relative_path(manifest.get("arrays_path"), "prototype_manifest.arrays_path")
        self.arrays_path = self.root / Path(arrays_rel)
        _require(self.arrays_path.exists(), "prototype_manifest.arrays_path", f"missing arrays file: {self.arrays_path}")

        arrays = np.load(self.arrays_path, allow_pickle=False)
        _require(self.array_key in arrays.files, "prototype_manifest.array_key", f"missing key '{self.array_key}'")
        prototypes = np.asarray(arrays[self.array_key], dtype=np.float32)
        _require(prototypes.ndim == 2, "prototype_arrays", "must be rank-2 [N_label, D]")
        _require(prototypes.shape[1] == self.embedding_dim, "prototype_arrays", "embedding dim mismatch")
        _require(np.isfinite(prototypes).all(), "prototype_arrays", "must contain only finite values")
        self.prototypes = prototypes

        labels_raw = manifest.get("labels")
        _require(isinstance(labels_raw, list) and labels_raw, "prototype_manifest.labels", "must be a non-empty list")
        _require(
            len(labels_raw) == prototypes.shape[0],
            "prototype_manifest.labels",
            "length must match prototype row count",
        )
        self._rows: List[PrototypeLabelRecord] = []
        self._row_by_label: Dict[tuple[int, int | str], int] = {}
        for index, row in enumerate(labels_raw):
            rpath = f"prototype_manifest.labels[{index}]"
            _require(isinstance(row, dict), rpath, "must be an object")
            _require("label_id" in row, f"{rpath}.label_id", "required field missing")
            _require("label_text" in row, f"{rpath}.label_text", "required field missing")
            _require("row_index" in row, f"{rpath}.row_index", "required field missing")
            label_id = _normalize_label_id(row["label_id"], field_path=f"{rpath}.label_id")
            row_index = row["row_index"]
            _require(
                isinstance(row_index, int) and row_index == index,
                f"{rpath}.row_index",
                "must equal the row position",
            )
            label_text = row["label_text"]
            _require(
                isinstance(label_text, str) and bool(label_text),
                f"{rpath}.label_text",
                "must be a non-empty string",
            )
            support_refs_raw = row.get("support_refs", [])
            _require(isinstance(support_refs_raw, list), f"{rpath}.support_refs", "must be a list")
            support_refs: List[PrototypeSupportRecord] = []
            for sidx, support in enumerate(support_refs_raw):
                spath = f"{rpath}.support_refs[{sidx}]"
                _require(isinstance(support, dict), spath, "must be an object")
                support_refs.append(
                    PrototypeSupportRecord(
                        video_id=_normalize_video_id(support.get("video_id"), f"{spath}.video_id"),
                        global_track_id=_normalize_nonnegative_int(support.get("global_track_id"), f"{spath}.global_track_id"),
                        representative_source_track_id=_normalize_label_id(
                            support.get("representative_source_track_id"),
                            field_path=f"{spath}.representative_source_track_id",
                        ),
                        o_tau=_normalize_float(support.get("o_tau"), f"{spath}.o_tau"),
                        member_count=_normalize_nonnegative_int(support.get("member_count"), f"{spath}.member_count"),
                        num_active_frames=_normalize_nonnegative_int(
                            support.get("num_active_frames"), f"{spath}.num_active_frames"
                        ),
                        start_frame_idx=_normalize_nonnegative_int(
                            support.get("start_frame_idx"), f"{spath}.start_frame_idx"
                        ),
                        end_frame_idx=_normalize_nonnegative_int(support.get("end_frame_idx"), f"{spath}.end_frame_idx"),
                    )
                )
            key = _canonical_label_key(label_id)
            _require(key not in self._row_by_label, f"{rpath}.label_id", "duplicate canonical label id")
            self._row_by_label[key] = index
            self._rows.append(
                PrototypeLabelRecord(
                    label_id=label_id,
                    label_text=label_text,
                    row_index=index,
                    support_video_count=_normalize_nonnegative_int(
                        row.get("support_video_count"), f"{rpath}.support_video_count"
                    ),
                    support_track_count=_normalize_nonnegative_int(
                        row.get("support_track_count"), f"{rpath}.support_track_count"
                    ),
                    support_weight_sum=_normalize_float(row.get("support_weight_sum"), f"{rpath}.support_weight_sum"),
                    support_refs=tuple(support_refs),
                )
            )
        producer = manifest.get("producer")
        _require(isinstance(producer, dict), "prototype_manifest.producer", "must be an object")
        self.producer = producer

    def iter_records(self) -> Iterator[PrototypeBankRecord]:
        return (
            PrototypeBankRecord(metadata=metadata, prototype=self.prototypes[metadata.row_index])
            for metadata in self._rows
        )

    def get_record(self, label_id: int | str) -> PrototypeBankRecord:
        key = _canonical_label_key(_normalize_label_id(label_id, field_path="label_id"))
        _require(key in self._row_by_label, "label_id", f"unknown label_id '{label_id}'")
        row_index = self._row_by_label[key]
        metadata = self._rows[row_index]
        return PrototypeBankRecord(metadata=metadata, prototype=self.prototypes[row_index])


def _err(field_path: str, rule_summary: str) -> PrototypeBankError:
    return PrototypeBankError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path, rule_summary)


def _normalize_float(value: Any, field_path: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), field_path, "must be numeric")
    numeric = float(value)
    _require(np.isfinite(numeric), field_path, "must be finite")
    return numeric


def _normalize_nonnegative_int(value: Any, field_path: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, field_path, "must be integer >= 0")
    return int(value)


def _normalize_video_id(value: Any, field_path: str) -> str:
    _require(
        (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and bool(value)),
        field_path,
        "must be non-empty string or integer",
    )
    return str(value)


def _normalize_label_id(value: Any, *, field_path: str) -> int | str:
    _require(
        (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and bool(value)),
        field_path,
        "must be non-empty string or integer",
    )
    return value  # type: ignore[return-value]


def _canonical_label_key(label_id: int | str) -> tuple[int, int | str]:
    if isinstance(label_id, int) and not isinstance(label_id, bool):
        return (0, int(label_id))
    return (1, str(label_id))


def _canonical_label_sort_key(label_id: int | str) -> Tuple[int, str]:
    key = _canonical_label_key(label_id)
    return (key[0], str(key[1]).zfill(12) if key[0] == 0 else str(key[1]))


def _parse_major_version(value: Any, field_path: str) -> int:
    _require(isinstance(value, str), field_path, "must be a string")
    parts = value.split(".")
    _require(len(parts) == 3 and all(part.isdigit() for part in parts), field_path, "must follow MAJOR.MINOR.PATCH")
    return int(parts[0])


def _load_json(path: Path, file_label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PrototypeBankError(f"{file_label} missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PrototypeBankError(f"{file_label} invalid JSON at {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{file_label}.$", "top-level value must be an object")
    return payload


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _require_relative_path(path_value: Any, field_path: str) -> str:
    _require(isinstance(path_value, str) and bool(path_value), field_path, "must be a non-empty relative path")
    rel = PurePosixPath(path_value)
    _require(not rel.is_absolute(), field_path, "absolute path is forbidden")
    _require(".." not in rel.parts, field_path, "must not contain '..'")
    return str(rel)


def _discover_manifest_path(root: Path) -> Path:
    manifest_v1 = root / "prototype_manifest.v1.json"
    manifest_compat = root / "prototype_manifest.json"
    if manifest_v1.exists():
        return manifest_v1
    if manifest_compat.exists():
        return manifest_compat
    raise PrototypeBankError(
        f"prototype manifest missing under root {root}; expected prototype_manifest.v1.json or prototype_manifest.json"
    )


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64)
    denom = float(np.linalg.norm(vec))
    _require(denom > 0.0, "prototype", "must have non-zero norm")
    return np.asarray(vec / denom, dtype=np.float32)


def _select_track_for_clip(records: Sequence[Any]) -> Any:
    _require(len(records) > 0, "semantic_cache", "must provide at least one track for a selected clip")
    return min(
        records,
        key=lambda record: (
            -float(record.metadata.o_tau),
            -int(record.metadata.member_count),
            int(record.metadata.global_track_id),
        ),
    )


def build_prototype_bank_v9(
    semantic_cache_root: Path,
    protocol_output_json: Path,
    protocol_manifest_json: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    config: PrototypeBankConfig | None = None,
) -> Path:
    cfg = config or PrototypeBankConfig()
    cfg_dict = cfg.canonical_dict()
    view = load_track_dino_feature_cache_v9(Path(semantic_cache_root), eager_validate=True)
    protocol_output = _load_json(Path(protocol_output_json), "protocol_output_json")
    protocol_manifest = _load_json(Path(protocol_manifest_json), "protocol_manifest_json")
    clips_raw = protocol_output.get("clips")
    _require(isinstance(clips_raw, list), "protocol_output_json.clips", "must be a list")
    category_id_to_name_raw = protocol_manifest.get("category_id_to_name")
    _require(
        isinstance(category_id_to_name_raw, dict) and category_id_to_name_raw,
        "protocol_manifest_json.category_id_to_name",
        "must be a non-empty object",
    )
    category_id_to_name: Dict[int | str, str] = {}
    for key, value in category_id_to_name_raw.items():
        label_id = _normalize_label_id(int(key) if isinstance(key, str) and key.isdigit() else key, field_path="category_id_to_name")
        _require(isinstance(value, str) and bool(value), "category_id_to_name[*]", "values must be non-empty strings")
        category_id_to_name[label_id] = value

    seen_labels = {
        _normalize_label_id(label_id, field_path="protocol_output_json.clips[*].label_set_observed_ids[*]")
        for clip in clips_raw
        for label_id in clip.get("label_set_observed_ids", [])
    }
    support_embeddings_by_label: Dict[tuple[int, int | str], List[np.ndarray]] = {}
    support_weights_by_label: Dict[tuple[int, int | str], List[float]] = {}
    support_refs_by_label: Dict[tuple[int, int | str], List[PrototypeSupportRecord]] = {}
    eligible_single_label_clips = 0
    clips_with_tracks = 0
    for index, clip in enumerate(clips_raw):
        cpath = f"protocol_output_json.clips[{index}]"
        _require(isinstance(clip, dict), cpath, "must be an object")
        video_id = _normalize_video_id(clip.get("video_id"), f"{cpath}.video_id")
        observed_ids_raw = clip.get("label_set_observed_ids", [])
        _require(isinstance(observed_ids_raw, list), f"{cpath}.label_set_observed_ids", "must be a list")
        observed_ids = [
            _normalize_label_id(label_id, field_path=f"{cpath}.label_set_observed_ids[{i}]")
            for i, label_id in enumerate(observed_ids_raw)
        ]
        if len(observed_ids) != 1:
            continue
        eligible_single_label_clips += 1
        label_id = observed_ids[0]
        tracks = list(view.iter_tracks(video_id))
        if not tracks:
            continue
        clips_with_tracks += 1
        selected = _select_track_for_clip(tracks)
        key = _canonical_label_key(label_id)
        support_embeddings_by_label.setdefault(key, []).append(np.asarray(selected.z_tau, dtype=np.float32))
        support_weights_by_label.setdefault(key, []).append(max(float(selected.metadata.o_tau), float(cfg.support_weight_floor)))
        support_refs_by_label.setdefault(key, []).append(
            PrototypeSupportRecord(
                video_id=video_id,
                global_track_id=int(selected.metadata.global_track_id),
                representative_source_track_id=selected.metadata.representative_source_track_id,
                o_tau=float(selected.metadata.o_tau),
                member_count=int(selected.metadata.member_count),
                num_active_frames=int(selected.metadata.num_active_frames),
                start_frame_idx=int(selected.metadata.start_frame_idx),
                end_frame_idx=int(selected.metadata.end_frame_idx),
            )
        )

    ordered_keys = sorted(support_embeddings_by_label.keys(), key=lambda item: _canonical_label_sort_key(item[1]))
    _require(ordered_keys, "prototype_bank", "no prototype-supporting single-label clips were found")
    prototype_rows: List[np.ndarray] = []
    label_rows: List[Dict[str, Any]] = []
    for row_index, key in enumerate(ordered_keys):
        label_id = key[1]
        label_text = category_id_to_name.get(label_id)
        _require(label_text is not None, "category_id_to_name", f"missing label text for label_id '{label_id}'")
        embeddings = np.stack(support_embeddings_by_label[key], axis=0).astype(np.float64)
        weights = np.asarray(support_weights_by_label[key], dtype=np.float64)
        prototype = _l2_normalize(np.sum(embeddings * weights[:, None], axis=0) / float(weights.sum()))
        support_refs = support_refs_by_label[key]
        prototype_rows.append(prototype)
        label_rows.append(
            {
                "label_id": label_id,
                "label_text": label_text,
                "row_index": row_index,
                "support_video_count": len({record.video_id for record in support_refs}),
                "support_track_count": len(support_refs),
                "support_weight_sum": float(weights.sum()),
                "support_refs": [
                    {
                        "video_id": record.video_id,
                        "global_track_id": int(record.global_track_id),
                        "representative_source_track_id": record.representative_source_track_id,
                        "o_tau": float(record.o_tau),
                        "member_count": int(record.member_count),
                        "num_active_frames": int(record.num_active_frames),
                        "start_frame_idx": int(record.start_frame_idx),
                        "end_frame_idx": int(record.end_frame_idx),
                    }
                    for record in support_refs
                ],
            }
        )

    seen_labels_sorted = sorted(seen_labels, key=_canonical_label_sort_key)
    prototype_array = np.asarray(np.stack(prototype_rows, axis=0), dtype=np.float32)
    total_seen_labels = len(seen_labels_sorted)
    selected_label_id = max(
        label_rows,
        key=lambda row: (
            int(row["support_video_count"]),
            float(row["support_weight_sum"]),
            -_canonical_label_sort_key(row["label_id"])[0],
            str(row["label_id"]),
        ),
    )["label_id"]
    manifest = {
        "schema_name": PROTOTYPE_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "prototype_source": "g5_seen_visual_prototypes_v9",
        "split": view.split,
        "embedding_dim": int(view.embedding_dim),
        "dtype": "float32",
        "array_key": "prototypes",
        "arrays_path": "prototype_arrays.v1.npz",
        "producer": {
            "semantic_cache_ref": str(Path(semantic_cache_root)),
            "protocol_output_json_ref": str(Path(protocol_output_json)),
            "protocol_manifest_json_ref": str(Path(protocol_manifest_json)),
            "semantic_cache_schema_name": str(view.manifest.get("schema_name")),
            "protocol_name": protocol_output.get("protocol"),
            "missing_rate": protocol_output.get("missing_rate"),
            **cfg_dict,
        },
        "coverage": {
            "num_seen_labels_total": total_seen_labels,
            "num_labels_with_prototype": len(label_rows),
            "coverage_ratio": (float(len(label_rows)) / float(total_seen_labels)) if total_seen_labels > 0 else 0.0,
            "eligible_single_label_clips": eligible_single_label_clips,
            "eligible_single_label_clips_with_tracks": clips_with_tracks,
            "selected_label_id": selected_label_id,
        },
        "labels": label_rows,
    }
    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise PrototypeBankError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)
    temp_dir = Path(tempfile.mkdtemp(prefix="prototype_bank_v9.", dir=str(output_root.parent if output_root.parent.exists() else Path.cwd())))
    try:
        np.savez_compressed(temp_dir / "prototype_arrays.v1.npz", prototypes=prototype_array)
        _dump_json(temp_dir / "prototype_manifest.v1.json", manifest)
        temp_dir.replace(output_root)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_root


def load_prototype_bank_v9(root: Path) -> PrototypeBankView:
    manifest_path = _discover_manifest_path(Path(root))
    manifest = _load_json(manifest_path, "prototype_manifest.v1.json")
    return PrototypeBankView(Path(root), manifest_path, manifest)


def summarize_prototype_bank_v9(root: Path) -> Dict[str, Any]:
    view = load_prototype_bank_v9(root)
    support_video_counts = np.asarray(
        [record.metadata.support_video_count for record in view.iter_records()],
        dtype=np.float64,
    )
    support_track_counts = np.asarray(
        [record.metadata.support_track_count for record in view.iter_records()],
        dtype=np.float64,
    )
    prototype_norms = np.asarray(
        [float(np.linalg.norm(record.prototype)) for record in view.iter_records()],
        dtype=np.float64,
    )
    coverage = dict(view.manifest.get("coverage", {}))
    return {
        "schema_name": SUMMARY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "split": view.split,
        "prototype_bank_coverage": {
            **coverage,
            "support_video_count": {
                "min": int(support_video_counts.min()) if support_video_counts.size else 0,
                "max": int(support_video_counts.max()) if support_video_counts.size else 0,
                "mean": float(support_video_counts.mean()) if support_video_counts.size else 0.0,
                "median": float(np.median(support_video_counts)) if support_video_counts.size else 0.0,
            },
            "support_track_count": {
                "min": int(support_track_counts.min()) if support_track_counts.size else 0,
                "max": int(support_track_counts.max()) if support_track_counts.size else 0,
                "mean": float(support_track_counts.mean()) if support_track_counts.size else 0.0,
                "median": float(np.median(support_track_counts)) if support_track_counts.size else 0.0,
            },
            "prototype_norm_mean": float(prototype_norms.mean()) if prototype_norms.size else 0.0,
        },
        "selected_label_id": coverage.get("selected_label_id"),
        "selected_label_text": _label_text_for_id(view, coverage.get("selected_label_id")),
    }


def _label_text_for_id(view: PrototypeBankView, label_id: Any) -> Optional[str]:
    if label_id is None:
        return None
    try:
        record = view.get_record(label_id)
    except PrototypeBankError:
        return None
    return record.metadata.label_text


def build_prototype_bank_v9_worked_example(
    root: Path,
    *,
    selected_label_id: int | str | None = None,
) -> Dict[str, Any]:
    view = load_prototype_bank_v9(root)
    summary = summarize_prototype_bank_v9(root)
    label_id = selected_label_id if selected_label_id is not None else summary["selected_label_id"]
    record = view.get_record(label_id)
    similarities = view.prototypes @ record.prototype
    ranked = sorted(
        (
            (
                other.metadata.label_id,
                other.metadata.label_text,
                float(similarities[other.metadata.row_index]),
            )
            for other in view.iter_records()
        ),
        key=lambda item: (-item[2], _canonical_label_sort_key(item[0])),
    )
    representative_support = min(
        record.metadata.support_refs,
        key=lambda support: (-float(support.o_tau), -int(support.member_count), str(support.video_id), int(support.global_track_id)),
    )
    return {
        "schema_name": WORKED_EXAMPLE_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "selected_label_id": record.metadata.label_id,
        "selected_label_text": record.metadata.label_text,
        "prototype_bank": {
            "support_video_count": int(record.metadata.support_video_count),
            "support_track_count": int(record.metadata.support_track_count),
            "support_weight_sum": float(record.metadata.support_weight_sum),
            "prototype_dim": int(record.prototype.shape[0]),
            "prototype_l2_norm": float(np.linalg.norm(record.prototype)),
        },
        "representative_support": {
            "video_id": representative_support.video_id,
            "global_track_id": int(representative_support.global_track_id),
            "representative_source_track_id": representative_support.representative_source_track_id,
            "o_tau": float(representative_support.o_tau),
            "member_count": int(representative_support.member_count),
            "num_active_frames": int(representative_support.num_active_frames),
            "start_frame_idx": int(representative_support.start_frame_idx),
            "end_frame_idx": int(representative_support.end_frame_idx),
        },
        "nearest_visual_prototypes": [
            {
                "label_id": label,
                "label_text": text,
                "cosine_similarity": cosine,
            }
            for label, text, cosine in ranked[:5]
        ],
    }
