from __future__ import annotations

import html
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from wsovvis.attribution.openworld_core_v9 import (
    _build_eval_bundle,
    _dump_json,
    _l2_normalize_rows,
    _load_json,
    _load_mapped_text_index,
    _load_protocol,
    _mean,
    _normalize_track_id,
    _score_stats,
    _sorted_unique_ints,
)
from wsovvis.features.track_dino_feature_v9 import load_track_dino_feature_cache_v9
from wsovvis.metrics.ws_metrics import set_coverage_recall
from wsovvis.metrics.ws_metrics_reporting_v1 import build_ws_metrics_summary_v1


SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_NAME = "wsovvis.bagfree_inference_v9"
SUMMARY_SCHEMA_NAME = "wsovvis.g7_bagfree_eval_summary_v9"
WORKED_EXAMPLE_SCHEMA_NAME = "wsovvis.g7_bagfree_eval_worked_example_v9"

SPECIAL_BG = "__bg__"
SPECIAL_UNK = "__unk__"


class BagFreeInferenceError(RuntimeError):
    """Raised when the bounded G7 bag-free inference artifact is invalid."""


@dataclass(frozen=True)
class BagFreeInferenceConfig:
    assignment_backend: str = "bagfree_cosine_gap_v1"
    candidate_space_policy: str = "all_mapped_text_labels_only"
    inference_policy: str = "bagfree_all_labels_with_bg_unk_gate"
    bg_score_threshold: float = 0.34
    direct_min_score: float = 0.40
    direct_margin: float = 0.10
    unknown_min_score: float = 0.34
    unknown_min_objectness: float = 0.75
    default_off_modules_enabled: Tuple[str, ...] = ()

    def canonical_dict(self) -> Dict[str, Any]:
        _require(
            self.assignment_backend == "bagfree_cosine_gap_v1",
            "config.assignment_backend",
            "must equal 'bagfree_cosine_gap_v1'",
        )
        _require(
            self.candidate_space_policy == "all_mapped_text_labels_only",
            "config.candidate_space_policy",
            "must equal 'all_mapped_text_labels_only'",
        )
        _require(
            self.inference_policy == "bagfree_all_labels_with_bg_unk_gate",
            "config.inference_policy",
            "must equal 'bagfree_all_labels_with_bg_unk_gate'",
        )
        _require(_is_number(self.bg_score_threshold), "config.bg_score_threshold", "must be numeric")
        _require(_is_number(self.direct_min_score), "config.direct_min_score", "must be numeric")
        _require(_is_number(self.direct_margin), "config.direct_margin", "must be numeric")
        _require(_is_number(self.unknown_min_score), "config.unknown_min_score", "must be numeric")
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
            "must stay empty for bounded G7",
        )
        return {
            "assignment_backend": self.assignment_backend,
            "candidate_space_policy": self.candidate_space_policy,
            "inference_policy": self.inference_policy,
            "bg_score_threshold": float(self.bg_score_threshold),
            "direct_min_score": float(self.direct_min_score),
            "direct_margin": float(self.direct_margin),
            "unknown_min_score": float(self.unknown_min_score),
            "unknown_min_objectness": float(self.unknown_min_objectness),
            "default_off_modules_enabled": list(self.default_off_modules_enabled),
            "requires_observed_label_bag_at_test_time": False,
        }


def _err(field_path: str, rule_summary: str) -> BagFreeInferenceError:
    return BagFreeInferenceError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path, rule_summary)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _pick_best_and_second(scores: np.ndarray, row_label_ids: Sequence[int]) -> tuple[int, float, int | None, float | None, float]:
    _require(scores.ndim == 1 and scores.shape[0] == len(row_label_ids), "scores", "shape mismatch")
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    best_index = int(order[0])
    best_label_id = int(row_label_ids[best_index])
    best_score = float(scores[best_index])
    if len(order) == 1:
        return best_label_id, best_score, None, None, float(best_score)
    second_index = int(order[1])
    second_label_id = int(row_label_ids[second_index])
    second_score = float(scores[second_index])
    return best_label_id, best_score, second_label_id, second_score, float(best_score - second_score)


def _top_ranked_labels(
    label_scores: np.ndarray,
    row_label_ids: Sequence[int],
    label_text_by_id: Mapping[int, str],
    *,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    if label_scores.size == 0:
        return []
    order = np.argsort(np.asarray(label_scores, dtype=np.float64))[::-1]
    rows: List[Dict[str, Any]] = []
    for rank, row_index in enumerate(order[:limit], start=1):
        label_id = int(row_label_ids[int(row_index)])
        rows.append(
            {
                "rank": int(rank),
                "label_id": int(label_id),
                "label_text": str(label_text_by_id[int(label_id)]),
                "score": float(label_scores[int(row_index)]),
            }
        )
    return rows


def _average_precision_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    truth = np.asarray([1 if int(v) else 0 for v in y_true], dtype=np.int64)
    scores = np.asarray([float(v) for v in y_score], dtype=np.float64)
    positives = int(truth.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    precision_sum = 0.0
    hits = 0
    for rank, index in enumerate(order, start=1):
        if int(truth[int(index)]) == 1:
            hits += 1
            precision_sum += float(hits) / float(rank)
    return float(precision_sum / float(positives))


def _build_map_metrics(
    video_diagnostics: Sequence[Mapping[str, Any]],
    row_label_ids: Sequence[int],
    label_text_by_id: Mapping[int, str],
) -> Dict[str, Any]:
    ap_rows: List[Dict[str, Any]] = []
    for label_id in row_label_ids:
        y_true: List[int] = []
        y_score: List[float] = []
        positive_count = 0
        for video in video_diagnostics:
            full = set(int(v) for v in video["aligned_full_label_ids"])
            label_scores = video["label_score_vector"]
            y_true.append(1 if int(label_id) in full else 0)
            if int(label_id) in full:
                positive_count += 1
            y_score.append(float(label_scores[str(int(label_id))]))
        ap = _average_precision_binary(y_true, y_score)
        if ap is None:
            continue
        ap_rows.append(
            {
                "label_id": int(label_id),
                "label_text": str(label_text_by_id[int(label_id)]),
                "positive_video_count": int(positive_count),
                "average_precision": float(ap),
            }
        )
    aps = [float(row["average_precision"]) for row in ap_rows]
    ap_rows_sorted = sorted(ap_rows, key=lambda row: (-float(row["average_precision"]), -int(row["positive_video_count"]), int(row["label_id"])))
    return {
        "macro_map": _mean(aps),
        "num_labels_with_positive_support": int(len(ap_rows)),
        "num_labels_scored": int(len(row_label_ids)),
        "average_precision_stats": _score_stats(aps),
        "top_labels_by_ap": ap_rows_sorted[:5],
        "bottom_labels_by_ap": list(reversed(ap_rows_sorted[-5:])) if ap_rows_sorted else [],
    }


def _build_summary_payload(
    *,
    manifest: Dict[str, Any],
    video_diagnostics: Sequence[Dict[str, Any]],
    row_label_ids: Sequence[int],
    label_text_by_id: Mapping[int, str],
    missing_rate: float,
    qualitative_paths: Sequence[str],
) -> Dict[str, Any]:
    scrs: List[float] = []
    aurcs: List[float] = []
    observed_recalls: List[float] = []
    hpr_hidden: List[float] = []
    uar_hidden: List[float] = []
    hidden_eval_videos = 0
    direct_track_count = 0
    background_track_count = 0
    unknown_track_count = 0

    for video in video_diagnostics:
        metrics = video["metrics"]
        scrs.append(float(metrics["scr"]))
        aurcs.append(float(metrics["aurc"]))
        observed_recalls.append(float(video["observed_recall"]))
        if video["aligned_hidden_positive_label_ids"]:
            hidden_eval_videos += 1
            hpr_hidden.append(float(metrics.get("hpr", 0.0)))
            uar_hidden.append(float(metrics.get("uar", 0.0)))
        counts = video["allocation_counts"]
        direct_track_count += int(counts["direct"])
        background_track_count += int(counts["background"])
        unknown_track_count += int(counts["unknown"])

    total_tracks = direct_track_count + background_track_count + unknown_track_count
    map_metrics = _build_map_metrics(video_diagnostics, row_label_ids, label_text_by_id)
    macro_scr = _mean(scrs)
    macro_aurc = _mean(aurcs)
    macro_hpr = _mean(hpr_hidden)
    macro_uar = _mean(uar_hidden)
    return {
        "schema_name": SUMMARY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "missing_rate": float(missing_rate),
        "evaluated_video_count": int(len(video_diagnostics)),
        "hidden_positive_video_count": int(hidden_eval_videos),
        "main_result_table": {
            "mAP": map_metrics["macro_map"],
            "HPR_hidden_positive_only": macro_hpr,
            "UAR_hidden_positive_only": macro_uar,
            "macro_scr": macro_scr,
            "macro_observed_recall": _mean(observed_recalls),
            "macro_aurc_single_point": macro_aurc,
        },
        "map_metrics": map_metrics,
        "bagfree_metrics": {
            "macro_scr": macro_scr,
            "macro_aurc_single_point": macro_aurc,
            "macro_observed_recall": _mean(observed_recalls),
            "macro_hpr_hidden_positive_only": macro_hpr,
            "macro_uar_hidden_positive_only": macro_uar,
        },
        "allocation_metrics": {
            "direct_track_count": int(direct_track_count),
            "background_track_count": int(background_track_count),
            "unknown_track_count": int(unknown_track_count),
            "total_track_count": int(total_tracks),
            "direct_fraction": float(direct_track_count / total_tracks) if total_tracks > 0 else 0.0,
            "background_fraction": float(background_track_count / total_tracks) if total_tracks > 0 else 0.0,
            "unknown_fraction": float(unknown_track_count / total_tracks) if total_tracks > 0 else 0.0,
        },
        "robustness_metrics": {
            "metric_name": "macro_aurc_single_point",
            "missing_rate_points": [float(missing_rate)],
            "macro_aurc_single_point": macro_aurc,
            "macro_scr": macro_scr,
        },
        "bagfree_path_proof": {
            "active_entrypoint": "g7_bagfree_inference_v9",
            "assignment_backend": manifest["config"]["assignment_backend"],
            "candidate_space_policy": manifest["config"]["candidate_space_policy"],
            "inference_policy": manifest["config"]["inference_policy"],
            "requires_observed_label_bag_at_test_time": False,
            "observed_label_bag_used_during_inference": False,
            "observed_label_bag_used_for_evaluation_only": True,
            "default_off_modules_enabled": list(manifest["config"]["default_off_modules_enabled"]),
            "uses_retrieval": False,
            "uses_warmup_bce": False,
            "uses_temporal_consistency_module": False,
            "uses_unknown_fallback": False,
            "uses_quality_refinement": False,
        },
        "qualitative_visualizations": [str(path) for path in qualitative_paths],
        "selected_video_id": str(manifest["selected_video_id"]),
    }


def _render_svg_lines(path: Path, title: str, lines: Sequence[str]) -> None:
    width = 1080
    line_height = 22
    height = 48 + line_height * max(1, len(lines)) + 24
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff9ef"/>',
        f'<text x="24" y="34" font-family="Courier New, monospace" font-size="20" fill="#1f2937">{html.escape(title)}</text>',
    ]
    for index, line in enumerate(lines, start=0):
        y = 62 + index * line_height
        rows.append(
            f'<text x="24" y="{y}" font-family="Courier New, monospace" font-size="16" fill="#374151">{html.escape(line)}</text>'
        )
    rows.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def _render_video_prediction_svg(video: Mapping[str, Any], path: Path) -> None:
    track_rows = list(video["track_assignments"])[:10]
    lines = [
        f"video_id={video['video_id']}",
        f"gt_full={video['aligned_full_label_texts']}",
        f"observed_eval_only={video['aligned_observed_label_texts']}",
        f"hidden_positive={video['aligned_hidden_positive_label_texts']}",
        f"predicted={video['predicted_texts']}",
        f"unknown_attributed={video['unknown_attributed_texts']}",
        f"metrics scr={video['metrics']['scr']:.4f} hpr={float(video['metrics'].get('hpr', 0.0)):.4f} uar={float(video['metrics'].get('uar', 0.0)):.4f} aurc={video['metrics']['aurc']:.4f}",
        "top_labels="
        + ", ".join(f"{row['label_text']}:{row['score']:.3f}" for row in video["top_ranked_labels"][:5]),
        "track_assignments:",
    ]
    for row in track_rows:
        lines.append(
            f"  tau={row['global_track_id']} src={row['assignment_source']} label={row['assigned_label_text']} score={row['best_score']:.3f} margin={row['best_margin']:.3f} o_tau={row['o_tau']:.3f}"
        )
    _render_svg_lines(path, f"G7 bag-free prediction visualization for video {video['video_id']}", lines)


def build_bagfree_inference_v9(
    semantic_cache_root: Path,
    text_map_root: Path,
    protocol_output_json: Path,
    protocol_manifest_json: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    config: BagFreeInferenceConfig | None = None,
    selected_video_id: str | None = None,
    num_qualitative_videos: int = 3,
) -> Path:
    cfg = config or BagFreeInferenceConfig()
    cfg_dict = cfg.canonical_dict()
    semantic_view = load_track_dino_feature_cache_v9(Path(semantic_cache_root), eager_validate=True)
    mapped = _load_mapped_text_index(Path(text_map_root))
    clips_by_video, category_name_by_id, missing_rate = _load_protocol(Path(protocol_output_json), Path(protocol_manifest_json))

    row_label_ids = list(mapped.label_ids_by_row)
    prototype_matrix = np.asarray(mapped.prototypes, dtype=np.float32)
    available_label_set = set(int(label_id) for label_id in row_label_ids)
    video_diagnostics: List[Dict[str, Any]] = []
    track_assignment_rows: List[Dict[str, Any]] = []
    video_prediction_rows: List[Dict[str, Any]] = []

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

        track_rows: List[Dict[str, Any]] = []
        predicted_entities: List[int] = []
        unknown_attributed_entities: List[int] = []
        score_matrix_rows: List[List[float]] = []
        video_label_scores = np.full((len(row_label_ids),), -1.0, dtype=np.float32)

        for track in tracks:
            metadata = track.metadata
            z_tau = np.asarray(track.z_tau, dtype=np.float32).reshape(1, -1)
            z_tau = _l2_normalize_rows(z_tau)[0]
            scores = np.asarray(z_tau @ prototype_matrix.T, dtype=np.float32)
            best_label_id, best_score, second_label_id, second_score, best_margin = _pick_best_and_second(scores, row_label_ids)
            video_label_scores = np.maximum(video_label_scores, scores)

            assigned_label_id: int | None = None
            assignment_source = "bg"
            if float(best_score) < float(cfg.bg_score_threshold):
                assigned_label_id = None
                assignment_source = "bg"
            elif float(best_score) >= float(cfg.direct_min_score) and float(best_margin) >= float(cfg.direct_margin):
                assigned_label_id = int(best_label_id)
                assignment_source = "direct"
                predicted_entities.append(int(assigned_label_id))
            elif float(best_score) >= float(cfg.unknown_min_score) and float(metadata.o_tau) >= float(cfg.unknown_min_objectness):
                assigned_label_id = int(best_label_id)
                assignment_source = "unknown_resolved"
                predicted_entities.append(int(assigned_label_id))
                unknown_attributed_entities.append(int(assigned_label_id))

            row = {
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
                "best_label_id": int(best_label_id),
                "best_label_text": mapped.label_text_by_id[int(best_label_id)],
                "best_score": float(best_score),
                "second_best_label_id": None if second_label_id is None else int(second_label_id),
                "second_best_label_text": None if second_label_id is None else mapped.label_text_by_id[int(second_label_id)],
                "second_best_score": None if second_score is None else float(second_score),
                "best_margin": float(best_margin),
                "assigned_label_id": assigned_label_id,
                "assigned_label_text": None if assigned_label_id is None else mapped.label_text_by_id[int(assigned_label_id)],
                "assignment_source": assignment_source,
            }
            track_rows.append(row)
            track_assignment_rows.append(dict(row))
            score_matrix_rows.append([float(value) for value in scores.tolist()])

        predicted_entities = _sorted_unique_ints(predicted_entities)
        unknown_attributed_entities = _sorted_unique_ints(unknown_attributed_entities)
        top_ranked_labels = _top_ranked_labels(video_label_scores, row_label_ids, mapped.label_text_by_id)

        # The observed bag is used only for evaluation metrics and never for the
        # bag-free assignment decisions above.
        eval_bundle = _build_eval_bundle(
            gt_entities=aligned_full,
            observed_entities=aligned_observed,
            predicted_entities=predicted_entities,
            unknown_attributed_entities=unknown_attributed_entities,
            missing_rate=missing_rate,
        )
        metrics = build_ws_metrics_summary_v1(eval_bundle)["metrics"]
        observed_recall = float(set_coverage_recall(aligned_observed, predicted_entities))
        video_row = {
            "video_id": str(video.video_id),
            "aligned_full_label_ids": [int(v) for v in aligned_full],
            "aligned_full_label_texts": [category_name_by_id.get(int(v), f"class_{v}") for v in aligned_full],
            "aligned_observed_label_ids": [int(v) for v in aligned_observed],
            "aligned_observed_label_texts": [category_name_by_id.get(int(v), f"class_{v}") for v in aligned_observed],
            "aligned_hidden_positive_label_ids": [int(v) for v in aligned_hidden],
            "aligned_hidden_positive_label_texts": [category_name_by_id.get(int(v), f"class_{v}") for v in aligned_hidden],
            "predicted_entities": [int(v) for v in predicted_entities],
            "predicted_texts": [mapped.label_text_by_id[int(v)] for v in predicted_entities],
            "unknown_attributed_entities": [int(v) for v in unknown_attributed_entities],
            "unknown_attributed_texts": [mapped.label_text_by_id[int(v)] for v in unknown_attributed_entities],
            "metrics": metrics,
            "observed_recall": observed_recall,
            "allocation_counts": {
                "direct": sum(1 for row in track_rows if row["assignment_source"] == "direct"),
                "background": sum(1 for row in track_rows if row["assignment_source"] == "bg"),
                "unknown": sum(1 for row in track_rows if row["assignment_source"] == "unknown_resolved"),
            },
            "top_ranked_labels": top_ranked_labels,
            "track_assignments": track_rows,
            "score_matrix": {
                "row_global_track_ids": [int(row["global_track_id"]) for row in track_rows],
                "column_label_ids": [int(v) for v in row_label_ids],
                "column_label_texts": [mapped.label_text_by_id[int(v)] for v in row_label_ids],
                "scores": score_matrix_rows,
            },
            "label_score_vector": {str(int(label_id)): float(video_label_scores[index]) for index, label_id in enumerate(row_label_ids)},
        }
        video_diagnostics.append(video_row)
        video_prediction_rows.append(
            {
                "video_id": str(video.video_id),
                "predicted_entities": [int(v) for v in predicted_entities],
                "unknown_attributed_entities": [int(v) for v in unknown_attributed_entities],
                "top_ranked_labels": top_ranked_labels,
            }
        )

    _require(video_diagnostics, "video_diagnostics", "no aligned protocol clips with tracks were found")
    ranked_videos = sorted(
        video_diagnostics,
        key=lambda row: (
            -len(row["aligned_hidden_positive_label_ids"]),
            -float(row["metrics"].get("hpr", 0.0)),
            -len(row["unknown_attributed_entities"]),
            -len(row["predicted_entities"]),
            str(row["video_id"]),
        ),
    )
    if selected_video_id is None:
        selected_video_id = str(ranked_videos[0]["video_id"])

    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise BagFreeInferenceError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)
    temp_dir = Path(tempfile.mkdtemp(prefix="bagfree_inference_v9.", dir=str(output_root.parent if output_root.parent.exists() else Path.cwd())))
    try:
        def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(dict(row), sort_keys=True) + "\n")

        qualitative_video_ids = [str(row["video_id"]) for row in ranked_videos[: max(1, int(num_qualitative_videos))]]
        qualitative_paths = [f"qualitative/video_{video_id}.svg" for video_id in qualitative_video_ids]

        _write_jsonl(temp_dir / "track_assignments.jsonl", track_assignment_rows)
        _write_jsonl(temp_dir / "video_predictions.jsonl", video_prediction_rows)
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
            "bagfree_path_proof": {
                "requires_observed_label_bag_at_test_time": False,
                "observed_label_bag_used_during_inference": False,
                "observed_label_bag_used_for_evaluation_only": True,
                "inference_inputs": [
                    "semantic_cache_root_rel",
                    "text_map_root_rel",
                ],
                "evaluation_only_inputs": [
                    "protocol_output_json_rel",
                    "protocol_manifest_json_rel",
                ],
                "default_off_modules_enabled": list(cfg_dict["default_off_modules_enabled"]),
            },
            "artifacts": {
                "track_assignments_path": "track_assignments.jsonl",
                "video_predictions_path": "video_predictions.jsonl",
                "video_diagnostics_path": "video_diagnostics.v1.json",
                "summary_path": "summary.v1.json",
                "qualitative_visualization_paths": qualitative_paths,
            },
        }
        for video_id in qualitative_video_ids:
            selected = next(row for row in video_diagnostics if str(row["video_id"]) == str(video_id))
            _render_video_prediction_svg(selected, temp_dir / "qualitative" / f"video_{video_id}.svg")
        summary = _build_summary_payload(
            manifest=manifest,
            video_diagnostics=video_diagnostics,
            row_label_ids=row_label_ids,
            label_text_by_id=mapped.label_text_by_id,
            missing_rate=missing_rate,
            qualitative_paths=qualitative_paths,
        )
        _dump_json(temp_dir / "summary.v1.json", summary)
        _dump_json(temp_dir / "bagfree_eval_manifest.v1.json", manifest)
        temp_dir.replace(output_root)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_root


def summarize_bagfree_inference_v9(output_root: Path) -> Dict[str, Any]:
    summary_path = Path(output_root) / "summary.v1.json"
    return _load_json(summary_path, "summary.v1.json")


def build_bagfree_inference_v9_worked_example(
    output_root: Path,
    *,
    selected_video_id: str | None = None,
) -> Dict[str, Any]:
    manifest = _load_json(Path(output_root) / "bagfree_eval_manifest.v1.json", "bagfree_eval_manifest.v1.json")
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
    full_set = set(int(v) for v in selected["aligned_full_label_ids"])
    observed_set = set(int(v) for v in selected["aligned_observed_label_ids"])
    hidden_set = set(int(v) for v in selected["aligned_hidden_positive_label_ids"])
    selected_columns = [int(v) for v in selected["aligned_full_label_ids"]]
    for label_id in selected["predicted_entities"]:
        if int(label_id) not in set(selected_columns):
            selected_columns.append(int(label_id))
    if not selected_columns:
        selected_columns = column_label_ids[: min(5, len(column_label_ids))]

    column_indices = [column_label_ids.index(int(label_id)) for label_id in selected_columns]
    score_rows = np.asarray(selected["score_matrix"]["scores"], dtype=np.float64)
    reduced_score_matrix = score_rows[:, np.asarray(column_indices, dtype=np.int64)]
    reduced_cost_matrix = np.asarray(1.0 - reduced_score_matrix, dtype=np.float64)

    assignment_rows: List[Dict[str, Any]] = []
    for row in selected["track_assignments"]:
        assignment_rows.append(
            {
                "global_track_id": int(row["global_track_id"]),
                "o_tau": float(row["o_tau"]),
                "best_label_id": int(row["best_label_id"]),
                "best_label_text": str(row["best_label_text"]),
                "best_score": float(row["best_score"]),
                "second_best_label_id": row["second_best_label_id"],
                "second_best_score": row["second_best_score"],
                "best_margin": float(row["best_margin"]),
                "assigned_label_id": row["assigned_label_id"],
                "assigned_label_text": row["assigned_label_text"],
                "assignment_source": row["assignment_source"],
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
        "final_predicted_entities": [int(v) for v in selected["predicted_entities"]],
        "final_predicted_texts": list(selected["predicted_texts"]),
        "unknown_attributed_entities": [int(v) for v in selected["unknown_attributed_entities"]],
        "unknown_attributed_texts": list(selected["unknown_attributed_texts"]),
        "metrics": dict(selected["metrics"]),
        "top_ranked_labels": list(selected["top_ranked_labels"]),
        "assignment_matrix": {
            "column_label_ids": [int(v) for v in selected_columns],
            "column_roles": [
                "observed_eval_only"
                if int(label_id) in observed_set
                else ("hidden_positive" if int(label_id) in hidden_set else ("gt_full" if int(label_id) in full_set else "predicted_only"))
                for label_id in selected_columns
            ],
            "row_global_track_ids": [int(v) for v in selected["score_matrix"]["row_global_track_ids"]],
            "score_matrix": [[float(value) for value in row] for row in reduced_score_matrix.tolist()],
            "cost_matrix": [[float(value) for value in row] for row in reduced_cost_matrix.tolist()],
            "special_columns": {
                "bg_score_threshold": float(manifest["config"]["bg_score_threshold"]),
                "direct_min_score": float(manifest["config"]["direct_min_score"]),
                "direct_margin": float(manifest["config"]["direct_margin"]),
                "unknown_min_score": float(manifest["config"]["unknown_min_score"]),
                "unknown_min_objectness": float(manifest["config"]["unknown_min_objectness"]),
                "bg_label": SPECIAL_BG,
                "unknown_label": SPECIAL_UNK,
            },
        },
        "assignment_summary": assignment_rows,
        "bagfree_path_proof": dict(manifest["bagfree_path_proof"]),
    }


def render_bagfree_prediction_svg(
    output_root: Path,
    svg_path: Path,
    *,
    selected_video_id: str | None = None,
) -> None:
    worked_example = build_bagfree_inference_v9_worked_example(output_root, selected_video_id=selected_video_id)
    lines = [
        f"selected_video_id={worked_example['selected_video_id']}",
        f"gt_full={worked_example['aligned_full_label_texts']}",
        f"observed_eval_only={worked_example['aligned_observed_label_texts']}",
        f"hidden_positive={worked_example['aligned_hidden_positive_label_texts']}",
        f"predicted={worked_example['final_predicted_texts']}",
        f"unknown_attributed={worked_example['unknown_attributed_texts']}",
        "top_predictions="
        + ", ".join(f"{row['label_text']}:{row['score']:.3f}" for row in worked_example["top_ranked_labels"][:5]),
    ]
    for row in worked_example["assignment_summary"][:10]:
        lines.append(
            f"tau={row['global_track_id']} src={row['assignment_source']} assigned={row['assigned_label_text']} best={row['best_label_text']} best_score={row['best_score']:.3f} margin={row['best_margin']:.3f} o_tau={row['o_tau']:.3f}"
        )
    _render_svg_lines(svg_path, f"G7 bag-free worked example for video {worked_example['selected_video_id']}", lines)
