from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import numpy as np

from wsovvis.features.track_dino_feature_v9 import load_track_dino_feature_cache_v9
from wsovvis.metrics.ws_metrics import set_coverage_recall
from wsovvis.metrics.ws_metrics_reporting_v1 import build_ws_metrics_summary_v1
from wsovvis.track_feature_export.stagec1_attribution_mil_v1 import load_stagec_label_prototype_inventory_v1


SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_NAME = "wsovvis.openworld_core_v9"
SUMMARY_SCHEMA_NAME = "wsovvis.g6_openworld_core_summary_v9"
WORKED_EXAMPLE_SCHEMA_NAME = "wsovvis.g6_openworld_core_worked_example_v9"

SPECIAL_BG = "__bg__"
SPECIAL_UNK = "__unk__"


class OpenWorldCoreError(RuntimeError):
    """Raised when the bounded G6 core attribution artifact is invalid."""


@dataclass(frozen=True)
class OpenWorldCoreConfig:
    assignment_backend: str = "cosine_gap_core_v1"
    protocol_alignment_policy: str = "mapped_text_intersection"
    closed_world_policy: str = "observed_only_with_bg_gate"
    open_world_policy: str = "all_mapped_labels_with_bg_unk_gate"
    bg_score_threshold: float = 0.34
    observed_min_score: float = 0.34
    unknown_min_score: float = 0.40
    unknown_margin: float = 0.08
    unknown_min_objectness: float = 0.75
    default_off_modules_enabled: Tuple[str, ...] = ()

    def canonical_dict(self) -> Dict[str, Any]:
        _require(
            self.assignment_backend == "cosine_gap_core_v1",
            "config.assignment_backend",
            "must equal 'cosine_gap_core_v1'",
        )
        _require(
            self.protocol_alignment_policy == "mapped_text_intersection",
            "config.protocol_alignment_policy",
            "must equal 'mapped_text_intersection'",
        )
        _require(
            self.closed_world_policy == "observed_only_with_bg_gate",
            "config.closed_world_policy",
            "must equal 'observed_only_with_bg_gate'",
        )
        _require(
            self.open_world_policy == "all_mapped_labels_with_bg_unk_gate",
            "config.open_world_policy",
            "must equal 'all_mapped_labels_with_bg_unk_gate'",
        )
        _require(_is_number(self.bg_score_threshold), "config.bg_score_threshold", "must be numeric")
        _require(_is_number(self.observed_min_score), "config.observed_min_score", "must be numeric")
        _require(_is_number(self.unknown_min_score), "config.unknown_min_score", "must be numeric")
        _require(_is_number(self.unknown_margin), "config.unknown_margin", "must be numeric")
        _require(_is_number(self.unknown_min_objectness), "config.unknown_min_objectness", "must be numeric")
        _require(
            0.0 <= float(self.unknown_min_objectness) <= 1.0,
            "config.unknown_min_objectness",
            "must be in [0,1]",
        )
        _require(
            isinstance(self.default_off_modules_enabled, tuple),
            "config.default_off_modules_enabled",
            "must be a tuple",
        )
        _require(
            len(self.default_off_modules_enabled) == 0,
            "config.default_off_modules_enabled",
            "must stay empty for bounded G6",
        )
        return {
            "assignment_backend": self.assignment_backend,
            "protocol_alignment_policy": self.protocol_alignment_policy,
            "closed_world_policy": self.closed_world_policy,
            "open_world_policy": self.open_world_policy,
            "bg_score_threshold": float(self.bg_score_threshold),
            "observed_min_score": float(self.observed_min_score),
            "unknown_min_score": float(self.unknown_min_score),
            "unknown_margin": float(self.unknown_margin),
            "unknown_min_objectness": float(self.unknown_min_objectness),
            "default_off_modules_enabled": list(self.default_off_modules_enabled),
        }


@dataclass(frozen=True)
class _ProtocolClip:
    video_id: str
    label_set_full_ids: Tuple[int, ...]
    label_set_observed_ids: Tuple[int, ...]
    missing_rate: float


@dataclass(frozen=True)
class _MappedTextIndex:
    text_map_root: Path
    text_map_manifest_path: Path
    mapped_manifest_path: Path
    label_ids_by_row: Tuple[int, ...]
    label_text_by_id: Mapping[int, str]
    row_index_by_label_id: Mapping[int, int]
    prototypes: np.ndarray


def _err(field_path: str, rule_summary: str) -> OpenWorldCoreError:
    return OpenWorldCoreError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path, rule_summary)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_int_label_id(value: Any, field_path: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), field_path, "must be integer")
    return int(value)


def _normalize_video_id(value: Any, field_path: str) -> str:
    _require(
        (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and bool(value)),
        field_path,
        "must be integer or non-empty string",
    )
    return str(value)


def _parse_major_version(value: Any, field_path: str) -> int:
    _require(isinstance(value, str), field_path, "must be a string")
    parts = value.split(".")
    _require(len(parts) == 3 and all(part.isdigit() for part in parts), field_path, "must follow MAJOR.MINOR.PATCH")
    return int(parts[0])


def _load_json(path: Path, file_label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _err(file_label, f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise _err(file_label, f"invalid JSON at {path}: {exc}") from exc
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


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    _require(arr.ndim == 2, "matrix", "must be rank-2")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return np.asarray(arr / norms, dtype=np.float32)


def _sorted_unique_ints(values: Iterable[int]) -> List[int]:
    return sorted(set(int(value) for value in values))


def _load_protocol(protocol_output_json: Path, protocol_manifest_json: Path) -> tuple[Dict[str, _ProtocolClip], Dict[int, str], float]:
    output = _load_json(protocol_output_json, "protocol_output_json")
    manifest = _load_json(protocol_manifest_json, "protocol_manifest_json")
    _require(output.get("version") == "wsovvis-labelset-protocol-v1", "protocol_output_json.version", "unexpected version")
    _require(manifest.get("version") == "wsovvis-labelset-protocol-v1", "protocol_manifest_json.version", "unexpected version")
    clips_raw = output.get("clips")
    _require(isinstance(clips_raw, list) and clips_raw, "protocol_output_json.clips", "must be a non-empty list")
    category_raw = manifest.get("category_id_to_name")
    _require(isinstance(category_raw, dict), "protocol_manifest_json.category_id_to_name", "must be an object")
    category_name_by_id: Dict[int, str] = {}
    for key, value in category_raw.items():
        _require(isinstance(key, str) and key.isdigit(), "protocol_manifest_json.category_id_to_name", "keys must be decimal strings")
        _require(isinstance(value, str) and bool(value), "protocol_manifest_json.category_id_to_name", "values must be non-empty strings")
        category_name_by_id[int(key)] = value

    missing_rate = float(output.get("missing_rate", 0.0))
    clips_by_video: Dict[str, _ProtocolClip] = {}
    for index, row in enumerate(clips_raw):
        rpath = f"protocol_output_json.clips[{index}]"
        _require(isinstance(row, dict), rpath, "must be an object")
        video_id = _normalize_video_id(row.get("video_id"), f"{rpath}.video_id")
        full_ids = tuple(_sorted_unique_ints(_normalize_int_label_id(v, f"{rpath}.label_set_full_ids[]") for v in row.get("label_set_full_ids", ())))
        observed_ids = tuple(
            _sorted_unique_ints(_normalize_int_label_id(v, f"{rpath}.label_set_observed_ids[]") for v in row.get("label_set_observed_ids", ()))
        )
        _require(video_id not in clips_by_video, f"{rpath}.video_id", "duplicate video_id")
        clips_by_video[video_id] = _ProtocolClip(
            video_id=video_id,
            label_set_full_ids=full_ids,
            label_set_observed_ids=observed_ids,
            missing_rate=missing_rate,
        )
    return clips_by_video, category_name_by_id, missing_rate


def _load_mapped_text_index(text_map_root: Path) -> _MappedTextIndex:
    text_map_root = Path(text_map_root)
    manifest_path = text_map_root / "text_map_manifest.v1.json"
    manifest = _load_json(manifest_path, "text_map_manifest.v1.json")
    _require(manifest.get("schema_name") == "wsovvis.text_map_v9", "text_map_manifest.v1.json.schema_name", "unexpected schema")
    _require(_parse_major_version(manifest.get("schema_version"), "text_map_manifest.v1.json.schema_version") == 1, "text_map_manifest.v1.json.schema_version", "unsupported major version")
    mapped_manifest_rel = _require_relative_path(manifest.get("mapped_text_manifest_path"), "text_map_manifest.v1.json.mapped_text_manifest_path")
    mapped_manifest_path = text_map_root / Path(mapped_manifest_rel)
    inventory = load_stagec_label_prototype_inventory_v1(mapped_manifest_path)
    prototypes = _l2_normalize_rows(np.asarray(inventory.prototypes, dtype=np.float32))

    labels_raw = manifest.get("labels")
    _require(isinstance(labels_raw, list) and labels_raw, "text_map_manifest.v1.json.labels", "must be a non-empty list")
    row_index_by_label_id: Dict[int, int] = {}
    label_text_by_id: Dict[int, str] = {}
    ordered_label_ids: List[int] = []
    for index, row in enumerate(labels_raw):
        rpath = f"text_map_manifest.v1.json.labels[{index}]"
        _require(isinstance(row, dict), rpath, "must be an object")
        label_id = _normalize_int_label_id(row.get("label_id"), f"{rpath}.label_id")
        row_index = row.get("row_index")
        _require(isinstance(row_index, int) and row_index == index, f"{rpath}.row_index", "must equal row position")
        label_text = row.get("label_text")
        _require(isinstance(label_text, str) and bool(label_text), f"{rpath}.label_text", "must be non-empty string")
        row_index_by_label_id[label_id] = index
        label_text_by_id[label_id] = label_text
        ordered_label_ids.append(label_id)

    _require(len(ordered_label_ids) == prototypes.shape[0], "mapped_text_prototypes", "row count must match text_map_manifest labels")
    return _MappedTextIndex(
        text_map_root=text_map_root,
        text_map_manifest_path=manifest_path,
        mapped_manifest_path=mapped_manifest_path,
        label_ids_by_row=tuple(ordered_label_ids),
        label_text_by_id=label_text_by_id,
        row_index_by_label_id=row_index_by_label_id,
        prototypes=prototypes,
    )


def _pick_best_label(scores: np.ndarray, row_label_ids: Sequence[int]) -> tuple[int, float]:
    _require(scores.ndim == 1 and scores.shape[0] == len(row_label_ids), "scores", "shape mismatch")
    best_index = int(np.argmax(scores))
    return int(row_label_ids[best_index]), float(scores[best_index])


def _pick_best_from_subset(scores: np.ndarray, row_label_ids: Sequence[int], accepted: Sequence[int]) -> tuple[int | None, float | None]:
    if not accepted:
        return None, None
    accepted_rows = [index for index, label_id in enumerate(row_label_ids) if int(label_id) in set(int(v) for v in accepted)]
    if not accepted_rows:
        return None, None
    subset_scores = scores[np.asarray(accepted_rows, dtype=np.int64)]
    local_index = int(np.argmax(subset_scores))
    row_index = accepted_rows[local_index]
    return int(row_label_ids[row_index]), float(scores[row_index])


def _build_eval_bundle(
    *,
    gt_entities: Sequence[int],
    observed_entities: Sequence[int],
    predicted_entities: Sequence[int],
    unknown_attributed_entities: Sequence[int],
    missing_rate: float,
) -> Dict[str, Any]:
    return {
        "gt_entities": [int(v) for v in gt_entities],
        "observed_entities": [int(v) for v in observed_entities],
        "predicted_entities": [int(v) for v in predicted_entities],
        "hidden_positive_entities": [int(v) for v in gt_entities if int(v) not in set(int(o) for o in observed_entities)],
        "unknown_attributed_entities": [int(v) for v in unknown_attributed_entities],
        "predictions_by_missing_rate": {
            f"{float(missing_rate):.1f}": [int(v) for v in predicted_entities],
        },
    }


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _normalize_track_id(value: Any, field_path: str) -> int | str:
    _require(
        (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and bool(value)),
        field_path,
        "must be integer or non-empty string",
    )
    return value


def _score_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _build_summary_payload(*, manifest: Dict[str, Any], video_diagnostics: Sequence[Dict[str, Any]], missing_rate: float) -> Dict[str, Any]:
    closed_scr: List[float] = []
    open_scr: List[float] = []
    closed_aurc: List[float] = []
    open_aurc: List[float] = []
    closed_obs: List[float] = []
    open_obs: List[float] = []
    closed_hpr_hidden: List[float] = []
    open_hpr_hidden: List[float] = []
    closed_uar_hidden: List[float] = []
    open_uar_hidden: List[float] = []
    hidden_eval_videos = 0

    open_known_track_count = 0
    open_background_track_count = 0
    open_unknown_track_count = 0
    closed_known_track_count = 0
    closed_background_track_count = 0

    for video in video_diagnostics:
        metrics_closed = video["closed_world"]["metrics"]
        metrics_open = video["open_world"]["metrics"]
        closed_scr.append(float(metrics_closed["scr"]))
        open_scr.append(float(metrics_open["scr"]))
        closed_aurc.append(float(metrics_closed["aurc"]))
        open_aurc.append(float(metrics_open["aurc"]))
        closed_obs.append(float(video["closed_world"]["observed_recall"]))
        open_obs.append(float(video["open_world"]["observed_recall"]))

        hidden_positive_ids = video["aligned_hidden_positive_label_ids"]
        if hidden_positive_ids:
            hidden_eval_videos += 1
            closed_hpr_hidden.append(float(metrics_closed.get("hpr", 0.0)))
            open_hpr_hidden.append(float(metrics_open.get("hpr", 0.0)))
            closed_uar_hidden.append(float(metrics_closed.get("uar", 0.0)))
            open_uar_hidden.append(float(metrics_open.get("uar", 0.0)))

        closed_counts = video["closed_world"]["allocation_counts"]
        open_counts = video["open_world"]["allocation_counts"]
        closed_known_track_count += int(closed_counts["known"])
        closed_background_track_count += int(closed_counts["background"])
        open_known_track_count += int(open_counts["known"])
        open_background_track_count += int(open_counts["background"])
        open_unknown_track_count += int(open_counts["unknown"])

    evaluated_videos = len(video_diagnostics)
    open_total_tracks = open_known_track_count + open_background_track_count + open_unknown_track_count
    closed_total_tracks = closed_known_track_count + closed_background_track_count
    return {
        "schema_name": SUMMARY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "missing_rate": float(missing_rate),
        "evaluated_video_count": int(evaluated_videos),
        "hidden_positive_video_count": int(hidden_eval_videos),
        "core_path_proof": {
            "active_entrypoint": "g6_openworld_core_v9",
            "default_off_modules_enabled": list(manifest["config"]["default_off_modules_enabled"]),
            "bounded_candidate_space": "all mapped text prototypes only",
            "assignment_backend": manifest["config"]["assignment_backend"],
            "uses_retrieval": False,
            "uses_warmup_bce": False,
            "uses_temporal_consistency_module": False,
            "uses_unknown_fallback": False,
            "uses_quality_refinement": False,
        },
        "protocol_alignment": {
            "policy": manifest["config"]["protocol_alignment_policy"],
            "num_text_map_labels": int(manifest["num_text_map_labels"]),
            "num_protocol_clips_total": int(manifest["num_protocol_clips_total"]),
            "num_protocol_clips_with_tracks_and_alignment": int(evaluated_videos),
        },
        "closed_world_metrics": {
            "macro_scr": _mean(closed_scr),
            "macro_aurc_single_point": _mean(closed_aurc),
            "macro_observed_recall": _mean(closed_obs),
            "macro_hpr_hidden_positive_only": _mean(closed_hpr_hidden),
            "macro_uar_hidden_positive_only": _mean(closed_uar_hidden),
        },
        "open_world_metrics": {
            "macro_scr": _mean(open_scr),
            "macro_aurc_single_point": _mean(open_aurc),
            "macro_observed_recall": _mean(open_obs),
            "macro_hpr_hidden_positive_only": _mean(open_hpr_hidden),
            "macro_uar_hidden_positive_only": _mean(open_uar_hidden),
        },
        "allocation_metrics": {
            "closed_world": {
                "known_track_count": int(closed_known_track_count),
                "background_track_count": int(closed_background_track_count),
                "total_track_count": int(closed_total_tracks),
            },
            "open_world": {
                "known_track_count": int(open_known_track_count),
                "background_track_count": int(open_background_track_count),
                "unknown_track_count": int(open_unknown_track_count),
                "total_track_count": int(open_total_tracks),
                "known_fraction": float(open_known_track_count / open_total_tracks) if open_total_tracks > 0 else 0.0,
                "background_fraction": float(open_background_track_count / open_total_tracks) if open_total_tracks > 0 else 0.0,
                "unknown_fraction": float(open_unknown_track_count / open_total_tracks) if open_total_tracks > 0 else 0.0,
            },
        },
        "comparison": {
            "delta_macro_scr": (_mean(open_scr) or 0.0) - (_mean(closed_scr) or 0.0),
            "delta_macro_aurc_single_point": (_mean(open_aurc) or 0.0) - (_mean(closed_aurc) or 0.0),
            "delta_macro_observed_recall": (_mean(open_obs) or 0.0) - (_mean(closed_obs) or 0.0),
            "delta_macro_hpr_hidden_positive_only": (_mean(open_hpr_hidden) or 0.0) - (_mean(closed_hpr_hidden) or 0.0),
            "delta_macro_uar_hidden_positive_only": (_mean(open_uar_hidden) or 0.0) - (_mean(closed_uar_hidden) or 0.0),
        },
        "selected_video_id": manifest["selected_video_id"],
    }


def build_openworld_core_v9(
    semantic_cache_root: Path,
    text_map_root: Path,
    protocol_output_json: Path,
    protocol_manifest_json: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    config: OpenWorldCoreConfig | None = None,
    selected_video_id: str | None = None,
) -> Path:
    cfg = config or OpenWorldCoreConfig()
    cfg_dict = cfg.canonical_dict()
    semantic_view = load_track_dino_feature_cache_v9(Path(semantic_cache_root), eager_validate=True)
    mapped = _load_mapped_text_index(Path(text_map_root))
    clips_by_video, category_name_by_id, missing_rate = _load_protocol(Path(protocol_output_json), Path(protocol_manifest_json))

    row_label_ids = list(mapped.label_ids_by_row)
    prototype_matrix = np.asarray(mapped.prototypes, dtype=np.float32)
    available_label_set = set(int(label_id) for label_id in row_label_ids)
    closed_rows: List[Dict[str, Any]] = []
    open_rows: List[Dict[str, Any]] = []
    video_diagnostics: List[Dict[str, Any]] = []

    for video in semantic_view.iter_videos(include_statuses=("processed_with_tracks",)):
        clip = clips_by_video.get(video.video_id)
        if clip is None:
            continue
        tracks = list(semantic_view.iter_tracks(video.video_id))
        if not tracks:
            continue

        aligned_full = [label_id for label_id in clip.label_set_full_ids if int(label_id) in available_label_set]
        aligned_observed = [label_id for label_id in clip.label_set_observed_ids if int(label_id) in available_label_set]
        aligned_hidden = [label_id for label_id in aligned_full if int(label_id) not in set(aligned_observed)]
        if not aligned_full:
            continue

        observed_set = set(int(label_id) for label_id in aligned_observed)
        track_rows_closed: List[Dict[str, Any]] = []
        track_rows_open: List[Dict[str, Any]] = []
        closed_predicted_entities: List[int] = []
        open_predicted_entities: List[int] = []
        unknown_attributed_entities: List[int] = []
        score_matrix_rows: List[List[float]] = []

        for track in tracks:
            metadata = track.metadata
            z_tau = np.asarray(track.z_tau, dtype=np.float32).reshape(1, -1)
            z_tau = _l2_normalize_rows(z_tau)[0]
            scores = np.asarray(z_tau @ prototype_matrix.T, dtype=np.float32)
            best_all_label_id, best_all_score = _pick_best_label(scores, row_label_ids)
            best_observed_label_id, best_observed_score = _pick_best_from_subset(scores, row_label_ids, aligned_observed)
            top_observed_score = -1.0 if best_observed_score is None else float(best_observed_score)
            hidden_margin = float(best_all_score - top_observed_score)
            assigned_closed_label_id: int | None = None
            closed_source = "bg"
            if best_observed_label_id is not None and float(best_observed_score) >= float(cfg.observed_min_score):
                assigned_closed_label_id = int(best_observed_label_id)
                closed_source = "observed"
                closed_predicted_entities.append(int(assigned_closed_label_id))

            assigned_open_label_id: int | None = None
            open_source = "bg"
            if float(best_all_score) < float(cfg.bg_score_threshold):
                assigned_open_label_id = None
                open_source = "bg"
            elif int(best_all_label_id) in observed_set and float(best_all_score) >= float(cfg.observed_min_score):
                assigned_open_label_id = int(best_all_label_id)
                open_source = "observed"
                open_predicted_entities.append(int(assigned_open_label_id))
            elif (
                int(best_all_label_id) not in observed_set
                and float(best_all_score) >= float(cfg.unknown_min_score)
                and float(hidden_margin) >= float(cfg.unknown_margin)
                and float(metadata.o_tau) >= float(cfg.unknown_min_objectness)
            ):
                assigned_open_label_id = int(best_all_label_id)
                open_source = "unknown_resolved"
                open_predicted_entities.append(int(assigned_open_label_id))
                unknown_attributed_entities.append(int(assigned_open_label_id))
            elif best_observed_label_id is not None and float(best_observed_score) >= float(cfg.observed_min_score):
                assigned_open_label_id = int(best_observed_label_id)
                open_source = "observed_fallback"
                open_predicted_entities.append(int(assigned_open_label_id))

            row_base = {
                "video_id": str(video.video_id),
                "global_track_id": int(metadata.global_track_id),
                "row_index": int(metadata.row_index),
                "representative_source_track_id": _normalize_track_id(
                    metadata.representative_source_track_id,
                    "track.metadata.representative_source_track_id",
                ),
                "o_tau": float(metadata.o_tau),
                "num_active_frames": int(metadata.num_active_frames),
                "member_count": int(metadata.member_count),
                "best_all_label_id": int(best_all_label_id),
                "best_all_label_text": mapped.label_text_by_id[int(best_all_label_id)],
                "best_all_score": float(best_all_score),
                "best_observed_label_id": None if best_observed_label_id is None else int(best_observed_label_id),
                "best_observed_label_text": None if best_observed_label_id is None else mapped.label_text_by_id[int(best_observed_label_id)],
                "best_observed_score": None if best_observed_score is None else float(best_observed_score),
                "hidden_margin_over_observed": float(hidden_margin),
            }
            closed_row = dict(row_base)
            closed_row["assigned_label_id"] = assigned_closed_label_id
            closed_row["assigned_label_text"] = None if assigned_closed_label_id is None else mapped.label_text_by_id[int(assigned_closed_label_id)]
            closed_row["assignment_source"] = closed_source
            open_row = dict(row_base)
            open_row["assigned_label_id"] = assigned_open_label_id
            open_row["assigned_label_text"] = None if assigned_open_label_id is None else mapped.label_text_by_id[int(assigned_open_label_id)]
            open_row["assignment_source"] = open_source

            track_rows_closed.append(closed_row)
            track_rows_open.append(open_row)
            closed_rows.append(closed_row)
            open_rows.append(open_row)
            score_matrix_rows.append([float(score) for score in scores.tolist()])

        # Preserve the observed bag as protected clip-level evidence for both the aligned
        # closed-world comparator and the bounded open-world path.
        closed_predicted_entities = _sorted_unique_ints(list(closed_predicted_entities) + list(aligned_observed))
        open_predicted_entities = _sorted_unique_ints(list(open_predicted_entities) + list(aligned_observed))
        unknown_attributed_entities = _sorted_unique_ints(unknown_attributed_entities)

        closed_bundle = _build_eval_bundle(
            gt_entities=aligned_full,
            observed_entities=aligned_observed,
            predicted_entities=closed_predicted_entities,
            unknown_attributed_entities=[],
            missing_rate=missing_rate,
        )
        open_bundle = _build_eval_bundle(
            gt_entities=aligned_full,
            observed_entities=aligned_observed,
            predicted_entities=open_predicted_entities,
            unknown_attributed_entities=unknown_attributed_entities,
            missing_rate=missing_rate,
        )
        metrics_closed = build_ws_metrics_summary_v1(closed_bundle)["metrics"]
        metrics_open = build_ws_metrics_summary_v1(open_bundle)["metrics"]
        video_diagnostics.append(
            {
                "video_id": str(video.video_id),
                "aligned_full_label_ids": [int(v) for v in aligned_full],
                "aligned_observed_label_ids": [int(v) for v in aligned_observed],
                "aligned_hidden_positive_label_ids": [int(v) for v in aligned_hidden],
                "aligned_full_label_texts": [category_name_by_id.get(int(v), f"class_{v}") for v in aligned_full],
                "aligned_observed_label_texts": [category_name_by_id.get(int(v), f"class_{v}") for v in aligned_observed],
                "aligned_hidden_positive_label_texts": [category_name_by_id.get(int(v), f"class_{v}") for v in aligned_hidden],
                "closed_world": {
                    "predicted_entities": closed_predicted_entities,
                    "predicted_texts": [mapped.label_text_by_id[int(v)] for v in closed_predicted_entities],
                    "metrics": metrics_closed,
                    "observed_recall": float(set_coverage_recall(aligned_observed, closed_predicted_entities)),
                    "allocation_counts": {
                        "known": sum(1 for row in track_rows_closed if row["assigned_label_id"] is not None),
                        "background": sum(1 for row in track_rows_closed if row["assigned_label_id"] is None),
                    },
                },
                "open_world": {
                    "predicted_entities": open_predicted_entities,
                    "predicted_texts": [mapped.label_text_by_id[int(v)] for v in open_predicted_entities],
                    "unknown_attributed_entities": unknown_attributed_entities,
                    "unknown_attributed_texts": [mapped.label_text_by_id[int(v)] for v in unknown_attributed_entities],
                    "metrics": metrics_open,
                    "observed_recall": float(set_coverage_recall(aligned_observed, open_predicted_entities)),
                    "allocation_counts": {
                        "known": sum(1 for row in track_rows_open if row["assignment_source"] in {"observed", "observed_fallback"}),
                        "background": sum(1 for row in track_rows_open if row["assignment_source"] == "bg"),
                        "unknown": sum(1 for row in track_rows_open if row["assignment_source"] == "unknown_resolved"),
                    },
                },
                "track_assignments": {
                    "closed_world": track_rows_closed,
                    "open_world": track_rows_open,
                },
                "score_matrix": {
                    "row_global_track_ids": [int(row["global_track_id"]) for row in track_rows_open],
                    "column_label_ids": [int(v) for v in row_label_ids],
                    "column_label_texts": [mapped.label_text_by_id[int(v)] for v in row_label_ids],
                    "scores": score_matrix_rows,
                },
            }
        )

    _require(video_diagnostics, "video_diagnostics", "no aligned protocol clips with tracks were found")
    if selected_video_id is None:
        selected = sorted(
            video_diagnostics,
            key=lambda row: (
                -len(row["open_world"]["unknown_attributed_entities"]),
                -len(row["aligned_hidden_positive_label_ids"]),
                -float(row["open_world"]["metrics"].get("hpr", 0.0)),
                str(row["video_id"]),
            ),
        )[0]
        selected_video_id = str(selected["video_id"])

    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise OpenWorldCoreError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)
    temp_dir = Path(tempfile.mkdtemp(prefix="openworld_core_v9.", dir=str(output_root.parent if output_root.parent.exists() else Path.cwd())))
    try:
        def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(dict(row), sort_keys=True) + "\n")

        _write_jsonl(temp_dir / "closed_world_assignments.jsonl", closed_rows)
        _write_jsonl(temp_dir / "open_world_assignments.jsonl", open_rows)
        _dump_json(temp_dir / "video_diagnostics.v1.json", {"videos": list(video_diagnostics)})

        semantic_root_rel = os.path.relpath(Path(semantic_cache_root), start=output_root)
        text_map_root_rel = os.path.relpath(Path(text_map_root), start=output_root)
        protocol_output_rel = os.path.relpath(Path(protocol_output_json), start=output_root)
        protocol_manifest_rel = os.path.relpath(Path(protocol_manifest_json), start=output_root)
        manifest = {
            "schema_name": MANIFEST_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "split": str(semantic_view.split),
            "selected_video_id": str(selected_video_id),
            "num_text_map_labels": int(len(row_label_ids)),
            "num_protocol_clips_total": int(len(clips_by_video)),
            "semantic_cache_root_rel": semantic_root_rel,
            "text_map_root_rel": text_map_root_rel,
            "protocol_output_json_rel": protocol_output_rel,
            "protocol_manifest_json_rel": protocol_manifest_rel,
            "config": cfg_dict,
            "artifacts": {
                "closed_world_assignments_path": "closed_world_assignments.jsonl",
                "open_world_assignments_path": "open_world_assignments.jsonl",
                "video_diagnostics_path": "video_diagnostics.v1.json",
                "summary_path": "summary.v1.json",
            },
        }
        summary = _build_summary_payload(manifest=manifest, video_diagnostics=video_diagnostics, missing_rate=missing_rate)
        _dump_json(temp_dir / "summary.v1.json", summary)
        _dump_json(temp_dir / "openworld_core_manifest.v1.json", manifest)
        temp_dir.replace(output_root)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_root


def summarize_openworld_core_v9(output_root: Path) -> Dict[str, Any]:
    summary_path = Path(output_root) / "summary.v1.json"
    return _load_json(summary_path, "summary.v1.json")


def build_openworld_core_v9_worked_example(
    output_root: Path,
    *,
    selected_video_id: str | None = None,
) -> Dict[str, Any]:
    manifest = _load_json(Path(output_root) / "openworld_core_manifest.v1.json", "openworld_core_manifest.v1.json")
    videos_payload = _load_json(Path(output_root) / "video_diagnostics.v1.json", "video_diagnostics.v1.json")
    videos = videos_payload.get("videos")
    _require(isinstance(videos, list) and videos, "video_diagnostics.v1.json.videos", "must be a non-empty list")
    target_video_id = str(selected_video_id or manifest.get("selected_video_id"))
    selected = None
    for row in videos:
        if isinstance(row, dict) and str(row.get("video_id")) == target_video_id:
            selected = row
            break
    _require(selected is not None, "selected_video_id", f"unknown video_id '{target_video_id}'")

    column_label_ids = [int(value) for value in selected["score_matrix"]["column_label_ids"]]
    observed_set = set(int(v) for v in selected["aligned_observed_label_ids"])
    hidden_set = set(int(v) for v in selected["aligned_hidden_positive_label_ids"])
    selected_columns = list(selected["aligned_observed_label_ids"])
    for label_id in selected["open_world"]["unknown_attributed_entities"]:
        if int(label_id) not in set(int(v) for v in selected_columns):
            selected_columns.append(int(label_id))
    if not selected_columns:
        selected_columns = column_label_ids[: min(5, len(column_label_ids))]

    column_indices = [column_label_ids.index(int(label_id)) for label_id in selected_columns]
    score_rows = np.asarray(selected["score_matrix"]["scores"], dtype=np.float64)
    reduced_score_matrix = score_rows[:, np.asarray(column_indices, dtype=np.int64)]
    reduced_cost_matrix = np.asarray(1.0 - reduced_score_matrix, dtype=np.float64)

    assignment_rows: List[Dict[str, Any]] = []
    open_tracks = {int(row["global_track_id"]): row for row in selected["track_assignments"]["open_world"]}
    closed_tracks = {int(row["global_track_id"]): row for row in selected["track_assignments"]["closed_world"]}
    for global_track_id in selected["score_matrix"]["row_global_track_ids"]:
        open_row = open_tracks[int(global_track_id)]
        closed_row = closed_tracks[int(global_track_id)]
        assignment_rows.append(
            {
                "global_track_id": int(global_track_id),
                "o_tau": float(open_row["o_tau"]),
                "best_observed_label_id": open_row["best_observed_label_id"],
                "best_observed_score": open_row["best_observed_score"],
                "best_all_label_id": int(open_row["best_all_label_id"]),
                "best_all_score": float(open_row["best_all_score"]),
                "hidden_margin_over_observed": float(open_row["hidden_margin_over_observed"]),
                "closed_world_assignment": {
                    "label_id": closed_row["assigned_label_id"],
                    "assignment_source": closed_row["assignment_source"],
                },
                "open_world_assignment": {
                    "label_id": open_row["assigned_label_id"],
                    "assignment_source": open_row["assignment_source"],
                },
            }
        )

    return {
        "schema_name": WORKED_EXAMPLE_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "selected_video_id": str(selected["video_id"]),
        "aligned_full_label_ids": [int(v) for v in selected["aligned_full_label_ids"]],
        "aligned_full_label_texts": list(selected["aligned_full_label_texts"]),
        "aligned_observed_label_ids": [int(v) for v in selected["aligned_observed_label_ids"]],
        "aligned_observed_label_texts": list(selected["aligned_observed_label_texts"]),
        "aligned_hidden_positive_label_ids": [int(v) for v in selected["aligned_hidden_positive_label_ids"]],
        "aligned_hidden_positive_label_texts": list(selected["aligned_hidden_positive_label_texts"]),
        "closed_world_predicted_entities": [int(v) for v in selected["closed_world"]["predicted_entities"]],
        "open_world_predicted_entities": [int(v) for v in selected["open_world"]["predicted_entities"]],
        "open_world_unknown_attributed_entities": [int(v) for v in selected["open_world"]["unknown_attributed_entities"]],
        "closed_world_metrics": dict(selected["closed_world"]["metrics"]),
        "open_world_metrics": dict(selected["open_world"]["metrics"]),
        "assignment_matrix": {
            "column_label_ids": [int(v) for v in selected_columns],
            "column_roles": [
                "observed" if int(label_id) in observed_set else ("hidden_positive" if int(label_id) in hidden_set else "open_world_candidate")
                for label_id in selected_columns
            ],
            "row_global_track_ids": [int(v) for v in selected["score_matrix"]["row_global_track_ids"]],
            "score_matrix": [[float(value) for value in row] for row in reduced_score_matrix.tolist()],
            "cost_matrix": [[float(value) for value in row] for row in reduced_cost_matrix.tolist()],
            "special_columns": {
                "bg_score_threshold": float(manifest["config"]["bg_score_threshold"]),
                "unknown_min_score": float(manifest["config"]["unknown_min_score"]),
                "unknown_margin": float(manifest["config"]["unknown_margin"]),
                "unknown_min_objectness": float(manifest["config"]["unknown_min_objectness"]),
                "bg_label": SPECIAL_BG,
                "unknown_label": SPECIAL_UNK,
            },
        },
        "assignment_summary": assignment_rows,
    }
