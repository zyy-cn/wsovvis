from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover
        raise ValueError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ValueError(f"invalid JSON at {path}: {exc}") from exc


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mask_utils():
    try:
        import pycocotools.mask as mask_utils
    except Exception as exc:  # pragma: no cover
        raise ValueError("pycocotools.mask is required for S1 structure evaluation") from exc
    return mask_utils


def _canonical_video_name(video_payload: Mapping[str, Any]) -> str:
    name = video_payload.get("name")
    if isinstance(name, str) and name:
        return name
    file_names = video_payload.get("file_names")
    if isinstance(file_names, list) and file_names:
        first = file_names[0]
        if isinstance(first, str) and first:
            return Path(first).parent.name or Path(first).stem
    raw_id = video_payload.get("id")
    if raw_id is None:
        raise ValueError("video record missing canonical name and id")
    return str(raw_id)


def _tube_id(row: Mapping[str, Any]) -> str:
    if "track_id" in row:
        return str(row["track_id"])
    if "id" in row:
        return str(row["id"])
    raise ValueError("tube row missing track_id/id")


def _to_float(value: Any) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list):
        if not value:
            return 0.0
        value = value[0]
    return float(value)


def _seg_area(segmentation: Any) -> float:
    if not segmentation:
        return 0.0
    return _to_float(_mask_utils().area(segmentation))


def _seg_iou_and_union(pred_seg: Any, gt_seg: Any) -> Tuple[float, float, float]:
    pred_area = _seg_area(pred_seg)
    gt_area = _seg_area(gt_seg)
    if not pred_seg and not gt_seg:
        return 0.0, 0.0, 0.0
    if pred_seg and gt_seg:
        iou = _to_float(_mask_utils().iou([pred_seg], [gt_seg], [0]))
        if iou <= 0.0:
            inter = 0.0
        else:
            inter = iou * (pred_area + gt_area) / (1.0 + iou)
        union = pred_area + gt_area - inter
        return float(iou), float(inter), float(union)
    return 0.0, 0.0, float(pred_area + gt_area)


def _active_frames(segmentations: Sequence[Any]) -> List[int]:
    return [index for index, segmentation in enumerate(segmentations) if segmentation]


def _intervals(segmentations: Sequence[Any]) -> List[Tuple[int, int]]:
    active = _active_frames(segmentations)
    if not active:
        return []
    start = active[0]
    previous = active[0]
    intervals: List[Tuple[int, int]] = []
    for frame_idx in active[1:]:
        if frame_idx == previous + 1:
            previous = frame_idx
            continue
        intervals.append((start, previous))
        start = frame_idx
        previous = frame_idx
    intervals.append((start, previous))
    return intervals


def _segment_count(segmentations: Sequence[Any]) -> int:
    return len(_intervals(segmentations))


@dataclass(frozen=True)
class VideoInfo:
    canonical_name: str
    raw_video_id: str
    length: int
    width: int
    height: int


@dataclass(frozen=True)
class TubeRecord:
    comparator_label: str
    canonical_video_name: str
    tube_id: str
    raw_video_id: str
    segmentations: Tuple[Any, ...]
    score: float

    @property
    def active_frames(self) -> List[int]:
        return _active_frames(self.segmentations)

    @property
    def duration(self) -> int:
        return len(self.active_frames)

    @property
    def num_segments(self) -> int:
        return _segment_count(self.segmentations)

    @property
    def intervals(self) -> List[Tuple[int, int]]:
        return _intervals(self.segmentations)


@dataclass(frozen=True)
class PairStats:
    pred_tube_id: str
    gt_tube_id: str
    tube_iou: float
    matched_frame_count: int
    temporal_overlap_frames: int
    max_frame_iou: float


@dataclass(frozen=True)
class GTEvalRecord:
    canonical_video_name: str
    gt_tube_id: str
    best_iou: float
    best_pred_tube_id: Optional[str]
    best_pred_score: float
    fragment_count: int
    fragmentation: float
    matched_pred_tube_ids: Tuple[str, ...]
    best_pair: Optional[PairStats]
    pair_stats: Tuple[PairStats, ...]


@dataclass(frozen=True)
class ComparatorEvalResult:
    label: str
    summary: Mapping[str, Any]
    gt_records: Tuple[GTEvalRecord, ...]
    tube_inventory: Mapping[str, Tuple[TubeRecord, ...]]


class _GroundTruthView:
    def __init__(self, json_path: Path) -> None:
        payload = _load_json(json_path)
        if not isinstance(payload, dict):
            raise ValueError("ground-truth JSON must be an object")
        videos = payload.get("videos")
        annotations = payload.get("annotations")
        if not isinstance(videos, list) or not isinstance(annotations, list):
            raise ValueError("ground-truth JSON must contain videos and annotations lists")

        self.videos_by_name: Dict[str, VideoInfo] = {}
        self.video_name_by_id: Dict[str, str] = {}
        for row in videos:
            if not isinstance(row, dict):
                continue
            canonical_name = _canonical_video_name(row)
            raw_video_id = str(row.get("id"))
            info = VideoInfo(
                canonical_name=canonical_name,
                raw_video_id=raw_video_id,
                length=int(row.get("length", 0)),
                width=int(row.get("width", 0)),
                height=int(row.get("height", 0)),
            )
            self.videos_by_name[canonical_name] = info
            self.video_name_by_id[raw_video_id] = canonical_name

        self.gt_by_video: Dict[str, List[TubeRecord]] = {key: [] for key in self.videos_by_name}
        for row in annotations:
            if not isinstance(row, dict):
                continue
            raw_video_id = str(row.get("video_id"))
            canonical_name = self.video_name_by_id.get(raw_video_id)
            if canonical_name is None:
                continue
            segmentations = row.get("segmentations")
            if not isinstance(segmentations, list):
                continue
            self.gt_by_video[canonical_name].append(
                TubeRecord(
                    comparator_label="ground_truth",
                    canonical_video_name=canonical_name,
                    tube_id=_tube_id(row),
                    raw_video_id=raw_video_id,
                    segmentations=tuple(segmentations),
                    score=1.0,
                )
            )


def _load_candidate_tubes(
    *,
    json_path: Path,
    comparator_label: str,
    gt_view: _GroundTruthView,
) -> Dict[str, List[TubeRecord]]:
    payload = _load_json(json_path)
    candidate_video_name_by_id: Dict[str, str] = {}

    if isinstance(payload, dict):
        videos = payload.get("videos")
        if isinstance(videos, list):
            for row in videos:
                if not isinstance(row, dict):
                    continue
                try:
                    candidate_video_name_by_id[str(row.get("id"))] = _canonical_video_name(row)
                except ValueError:
                    continue
        rows = payload.get("annotations")
        if not isinstance(rows, list):
            rows = payload.get("predictions")
        if not isinstance(rows, list):
            raise ValueError(f"{json_path} must contain annotations or predictions list")
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{json_path} must be a list or object")

    result: Dict[str, List[TubeRecord]] = {key: [] for key in gt_view.videos_by_name}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_video_id = str(row.get("video_id"))
        canonical_name = candidate_video_name_by_id.get(raw_video_id)
        if canonical_name is None:
            canonical_name = gt_view.video_name_by_id.get(raw_video_id)
        if canonical_name is None or canonical_name not in gt_view.videos_by_name:
            continue
        segmentations = row.get("segmentations")
        if not isinstance(segmentations, list):
            continue
        result[canonical_name].append(
            TubeRecord(
                comparator_label=comparator_label,
                canonical_video_name=canonical_name,
                tube_id=_tube_id(row),
                raw_video_id=raw_video_id,
                segmentations=tuple(segmentations),
                score=float(row.get("score", 1.0)),
            )
        )
    for tubes in result.values():
        tubes.sort(key=lambda tube: (-tube.score, tube.tube_id))
    return result


def _pair_stats(pred: TubeRecord, gt: TubeRecord, *, frame_match_iou_threshold: float) -> PairStats:
    max_len = max(len(pred.segmentations), len(gt.segmentations))
    total_inter = 0.0
    total_union = 0.0
    matched_frame_count = 0
    temporal_overlap_frames = 0
    max_frame_iou = 0.0
    for index in range(max_len):
        pred_seg = pred.segmentations[index] if index < len(pred.segmentations) else None
        gt_seg = gt.segmentations[index] if index < len(gt.segmentations) else None
        frame_iou, inter, union = _seg_iou_and_union(pred_seg, gt_seg)
        total_inter += inter
        total_union += union
        if pred_seg and gt_seg:
            temporal_overlap_frames += 1
            if frame_iou >= frame_match_iou_threshold:
                matched_frame_count += 1
            if frame_iou > max_frame_iou:
                max_frame_iou = frame_iou
    tube_iou = 0.0 if total_union <= 0.0 else total_inter / total_union
    return PairStats(
        pred_tube_id=pred.tube_id,
        gt_tube_id=gt.tube_id,
        tube_iou=float(tube_iou),
        matched_frame_count=int(matched_frame_count),
        temporal_overlap_frames=int(temporal_overlap_frames),
        max_frame_iou=float(max_frame_iou),
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _round_metrics(payload: MutableMapping[str, Any]) -> Dict[str, Any]:
    rounded: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def _evaluate_comparator(
    *,
    gt_view: _GroundTruthView,
    comparator_label: str,
    candidate_tubes: Mapping[str, Sequence[TubeRecord]],
    frame_match_iou_threshold: float,
    short_track_threshold: int,
) -> ComparatorEvalResult:
    gt_records: List[GTEvalRecord] = []
    all_pred_tubes: List[TubeRecord] = []

    for tubes in candidate_tubes.values():
        all_pred_tubes.extend(tubes)

    durations = [tube.duration for tube in all_pred_tubes]
    broken_flags = [tube.num_segments > 1 for tube in all_pred_tubes]
    temporal_inconsistency_values = [
        0.0 if tube.duration <= 0 else float(max(0, tube.num_segments - 1)) / float(max(1, tube.duration))
        for tube in all_pred_tubes
    ]

    for canonical_video_name, gt_tubes in gt_view.gt_by_video.items():
        pred_tubes = list(candidate_tubes.get(canonical_video_name, ()))
        for gt_tube in gt_tubes:
            pair_rows = [
                _pair_stats(pred_tube, gt_tube, frame_match_iou_threshold=frame_match_iou_threshold)
                for pred_tube in pred_tubes
            ]
            pair_rows.sort(key=lambda row: (-row.tube_iou, -row.matched_frame_count, row.pred_tube_id))
            best_pair = pair_rows[0] if pair_rows else None
            matched_pairs = [row for row in pair_rows if row.matched_frame_count > 0]
            matched_pairs.sort(key=lambda row: (-row.tube_iou, -row.matched_frame_count, row.pred_tube_id))
            best_pred_id = best_pair.pred_tube_id if best_pair is not None else None
            best_pred_score = 0.0
            if best_pred_id is not None:
                for pred in pred_tubes:
                    if pred.tube_id == best_pred_id:
                        best_pred_score = pred.score
                        break
            gt_records.append(
                GTEvalRecord(
                    canonical_video_name=canonical_video_name,
                    gt_tube_id=gt_tube.tube_id,
                    best_iou=0.0 if best_pair is None else best_pair.tube_iou,
                    best_pred_tube_id=best_pred_id,
                    best_pred_score=float(best_pred_score),
                    fragment_count=len(matched_pairs),
                    fragmentation=float(max(0, len(matched_pairs) - 1)),
                    matched_pred_tube_ids=tuple(row.pred_tube_id for row in matched_pairs),
                    best_pair=best_pair,
                    pair_stats=tuple(pair_rows),
                )
            )

    mean_best_iou = _mean(record.best_iou for record in gt_records)
    recall_at_05 = _mean(1.0 if record.best_iou >= 0.5 else 0.0 for record in gt_records)
    recall_at_07 = _mean(1.0 if record.best_iou >= 0.7 else 0.0 for record in gt_records)
    fragmentation = _mean(record.fragmentation for record in gt_records)
    unmatched_gt_ratio = _mean(1.0 if record.best_iou < 0.5 else 0.0 for record in gt_records)
    over_segmentation = _mean(1.0 if record.fragment_count > 1 else 0.0 for record in gt_records)
    under_coverage = _mean(1.0 - record.best_iou for record in gt_records)

    summary = _round_metrics(
        {
            "comparator_label": comparator_label,
            "videos_total": len(gt_view.videos_by_name),
            "gt_instances_total": len(gt_records),
            "predicted_tracks_total": len(all_pred_tubes),
            "mean_best_iou": mean_best_iou,
            "recall_at_0.5": recall_at_05,
            "recall_at_0.7": recall_at_07,
            "fragmentation_per_gt_instance": fragmentation,
            "short_track_ratio": _mean(1.0 if duration <= short_track_threshold else 0.0 for duration in durations),
            "broken_track_ratio": _mean(1.0 if flag else 0.0 for flag in broken_flags),
            "average_track_duration": _mean(float(duration) for duration in durations),
            "temporal_inconsistency": _mean(temporal_inconsistency_values),
            "unmatched_gt_ratio": unmatched_gt_ratio,
            "over_segmentation_tendency": over_segmentation,
            "under_coverage_tendency": under_coverage,
            "frame_match_iou_threshold": float(frame_match_iou_threshold),
            "short_track_threshold": int(short_track_threshold),
        }
    )

    tube_inventory: Dict[str, Tuple[TubeRecord, ...]] = {
        video_name: tuple(tubes) for video_name, tubes in candidate_tubes.items()
    }
    return ComparatorEvalResult(
        label=comparator_label,
        summary=summary,
        gt_records=tuple(gt_records),
        tube_inventory=tube_inventory,
    )


def _record_index(records: Sequence[GTEvalRecord]) -> Dict[Tuple[str, str], GTEvalRecord]:
    return {(record.canonical_video_name, record.gt_tube_id): record for record in records}


def _tube_inventory_index(inventory: Mapping[str, Sequence[TubeRecord]]) -> Dict[Tuple[str, str], TubeRecord]:
    table: Dict[Tuple[str, str], TubeRecord] = {}
    for video_name, tubes in inventory.items():
        for tube in tubes:
            table[(video_name, tube.tube_id)] = tube
    return table


def _example_rows(
    *,
    gt_tube: TubeRecord,
    comparator_name: str,
    eval_record: GTEvalRecord,
    tube_lookup: Mapping[Tuple[str, str], TubeRecord],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {
            "role": "gt",
            "label": "ground_truth",
            "tube_id": gt_tube.tube_id,
            "intervals": gt_tube.intervals,
            "duration": gt_tube.duration,
        }
    ]
    if eval_record.best_pred_tube_id is not None:
        best_tube = tube_lookup[(gt_tube.canonical_video_name, eval_record.best_pred_tube_id)]
        rows.append(
            {
                "role": f"{comparator_name}_best",
                "label": comparator_name,
                "tube_id": best_tube.tube_id,
                "intervals": best_tube.intervals,
                "duration": best_tube.duration,
                "tube_iou": eval_record.best_iou,
            }
        )
    for pred_tube_id in eval_record.matched_pred_tube_ids:
        if pred_tube_id == eval_record.best_pred_tube_id:
            continue
        matched_tube = tube_lookup[(gt_tube.canonical_video_name, pred_tube_id)]
        pair_row = next((row for row in eval_record.pair_stats if row.pred_tube_id == pred_tube_id), None)
        rows.append(
            {
                "role": f"{comparator_name}_fragment",
                "label": comparator_name,
                "tube_id": matched_tube.tube_id,
                "intervals": matched_tube.intervals,
                "duration": matched_tube.duration,
                "tube_iou": None if pair_row is None else pair_row.tube_iou,
            }
        )
    return rows


def _build_example_payload(
    *,
    example_kind: str,
    selection_key: Tuple[str, str],
    gt_view: _GroundTruthView,
    raw_eval: ComparatorEvalResult,
    refined_eval: ComparatorEvalResult,
) -> Dict[str, Any]:
    gt_records = _record_index(raw_eval.gt_records)
    refined_records = _record_index(refined_eval.gt_records)
    gt_record_raw = gt_records[selection_key]
    gt_record_refined = refined_records[selection_key]
    gt_tube = next(
        tube
        for tube in gt_view.gt_by_video[selection_key[0]]
        if tube.tube_id == selection_key[1]
    )
    raw_lookup = _tube_inventory_index(raw_eval.tube_inventory)
    refined_lookup = _tube_inventory_index(refined_eval.tube_inventory)

    return {
        "example_kind": example_kind,
        "video_id": selection_key[0],
        "gt_tube_id": selection_key[1],
        "video_length": gt_view.videos_by_name[selection_key[0]].length,
        "raw": {
            "best_iou": gt_record_raw.best_iou,
            "best_pred_tube_id": gt_record_raw.best_pred_tube_id,
            "fragment_count": gt_record_raw.fragment_count,
            "fragmentation": gt_record_raw.fragmentation,
            "rows": _example_rows(
                gt_tube=gt_tube,
                comparator_name="raw",
                eval_record=gt_record_raw,
                tube_lookup=raw_lookup,
            ),
        },
        "refined": {
            "best_iou": gt_record_refined.best_iou,
            "best_pred_tube_id": gt_record_refined.best_pred_tube_id,
            "fragment_count": gt_record_refined.fragment_count,
            "fragmentation": gt_record_refined.fragmentation,
            "rows": _example_rows(
                gt_tube=gt_tube,
                comparator_name="refined",
                eval_record=gt_record_refined,
                tube_lookup=refined_lookup,
            ),
        },
        "delta": {
            "best_iou_delta": round(gt_record_refined.best_iou - gt_record_raw.best_iou, 6),
            "fragmentation_delta": round(gt_record_refined.fragmentation - gt_record_raw.fragmentation, 6),
        },
    }


def _pick_examples(
    *,
    raw_eval: ComparatorEvalResult,
    refined_eval: ComparatorEvalResult,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    raw_index = _record_index(raw_eval.gt_records)
    refined_index = _record_index(refined_eval.gt_records)
    keys = sorted(set(raw_index) & set(refined_index))
    if not keys:
        raise ValueError("no overlapping GT evaluation records for worked example selection")

    def _success_rank(key: Tuple[str, str]) -> Tuple[float, float, float, str, str]:
        raw_row = raw_index[key]
        refined_row = refined_index[key]
        return (
            refined_row.best_iou - raw_row.best_iou,
            raw_row.fragmentation - refined_row.fragmentation,
            refined_row.best_iou,
            key[0],
            key[1],
        )

    def _failure_rank(key: Tuple[str, str]) -> Tuple[float, float, float, str, str]:
        raw_row = raw_index[key]
        refined_row = refined_index[key]
        return (
            raw_row.best_iou - refined_row.best_iou,
            refined_row.fragmentation - raw_row.fragmentation,
            raw_row.best_iou,
            key[0],
            key[1],
        )

    paired_key = max(keys, key=_success_rank)
    failure_key = max(keys, key=_failure_rank)
    return paired_key, failure_key


def _render_example_svg(*, title: str, payload: Mapping[str, Any], output_path: Path) -> None:
    video_length = max(1, int(payload["video_length"]))
    panel_width = 1100
    left_margin = 170
    right_margin = 30
    row_height = 34
    box_height = 18
    timeline_width = panel_width - left_margin - right_margin
    rows: List[Mapping[str, Any]] = []
    rows.extend(payload["raw"]["rows"])
    rows.extend(payload["refined"]["rows"][1:])
    total_rows = len(rows)
    svg_height = 90 + total_rows * row_height
    colors = {
        "ground_truth": "#1F2933",
        "raw": "#D64545",
        "refined": "#2F855A",
    }

    def _x(frame_idx: int) -> float:
        if video_length <= 1:
            return float(left_margin)
        return left_margin + (timeline_width * float(frame_idx) / float(video_length - 1))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_width}" height="{svg_height}" viewBox="0 0 {panel_width} {svg_height}">',
        '<rect width="100%" height="100%" fill="#F7FAFC" />',
        f'<text x="{left_margin}" y="28" font-family="monospace" font-size="16" fill="#102A43">{title}</text>',
        f'<text x="{left_margin}" y="50" font-family="monospace" font-size="12" fill="#486581">video={payload["video_id"]} gt={payload["gt_tube_id"]} raw_iou={payload["raw"]["best_iou"]:.3f} refined_iou={payload["refined"]["best_iou"]:.3f}</text>',
    ]
    for tick in range(0, video_length, max(1, video_length // 10)):
        xpos = _x(tick)
        lines.append(f'<line x1="{xpos:.2f}" y1="64" x2="{xpos:.2f}" y2="{svg_height - 12}" stroke="#CBD2D9" stroke-width="1" />')
        lines.append(f'<text x="{xpos:.2f}" y="{svg_height - 4}" text-anchor="middle" font-family="monospace" font-size="10" fill="#7B8794">{tick}</text>')

    for row_index, row in enumerate(rows):
        ypos = 76 + row_index * row_height
        label = str(row["role"])
        label_color = colors["ground_truth"]
        if label.startswith("raw"):
            label_color = colors["raw"]
        elif label.startswith("refined"):
            label_color = colors["refined"]
        lines.append(
            f'<text x="18" y="{ypos + 13}" font-family="monospace" font-size="12" fill="{label_color}">{label}:{row["tube_id"]}</text>'
        )
        lines.append(
            f'<line x1="{left_margin}" y1="{ypos + box_height / 2}" x2="{left_margin + timeline_width}" y2="{ypos + box_height / 2}" stroke="#9FB3C8" stroke-width="1" />'
        )
        for start, end in row["intervals"]:
            x0 = _x(start)
            x1 = _x(end)
            width = max(3.0, x1 - x0 + 3.0)
            lines.append(
                f'<rect x="{x0:.2f}" y="{ypos:.2f}" width="{width:.2f}" height="{box_height}" rx="3" fill="{label_color}" opacity="0.82" />'
            )
        if "tube_iou" in row and row["tube_iou"] is not None:
            lines.append(
                f'<text x="{panel_width - 18}" y="{ypos + 13}" text-anchor="end" font-family="monospace" font-size="11" fill="#334E68">IoU={float(row["tube_iou"]):.3f}</text>'
            )
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_comparator_table(
    *,
    raw_summary: Mapping[str, Any],
    refined_summary: Mapping[str, Any],
) -> str:
    metrics = [
        "mean_best_iou",
        "recall_at_0.5",
        "fragmentation_per_gt_instance",
        "short_track_ratio",
        "broken_track_ratio",
        "average_track_duration",
        "temporal_inconsistency",
        "unmatched_gt_ratio",
        "over_segmentation_tendency",
        "under_coverage_tendency",
    ]
    lines = [
        "| metric | raw_pseudo_tube | refined_basis | refined_minus_raw |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        raw_value = float(raw_summary.get(metric, 0.0))
        refined_value = float(refined_summary.get(metric, 0.0))
        lines.append(
            f"| {metric} | {raw_value:.6f} | {refined_value:.6f} | {(refined_value - raw_value):.6f} |"
        )
    return "\n".join(lines) + "\n"


def run_s1_basis_superiority_eval(
    *,
    gt_json_path: Path,
    raw_json_path: Path,
    refined_json_path: Path,
    output_root: Path,
    raw_label: str = "raw_pseudo_tube",
    refined_label: str = "refined_basis",
    frame_match_iou_threshold: float = 0.5,
    short_track_threshold: int = 5,
) -> Dict[str, Any]:
    gt_view = _GroundTruthView(gt_json_path)
    raw_inventory = _load_candidate_tubes(json_path=raw_json_path, comparator_label=raw_label, gt_view=gt_view)
    refined_inventory = _load_candidate_tubes(json_path=refined_json_path, comparator_label=refined_label, gt_view=gt_view)

    raw_eval = _evaluate_comparator(
        gt_view=gt_view,
        comparator_label=raw_label,
        candidate_tubes=raw_inventory,
        frame_match_iou_threshold=frame_match_iou_threshold,
        short_track_threshold=short_track_threshold,
    )
    refined_eval = _evaluate_comparator(
        gt_view=gt_view,
        comparator_label=refined_label,
        candidate_tubes=refined_inventory,
        frame_match_iou_threshold=frame_match_iou_threshold,
        short_track_threshold=short_track_threshold,
    )

    paired_key, failure_key = _pick_examples(raw_eval=raw_eval, refined_eval=refined_eval)
    paired_payload = _build_example_payload(
        example_kind="paired_worked_example",
        selection_key=paired_key,
        gt_view=gt_view,
        raw_eval=raw_eval,
        refined_eval=refined_eval,
    )
    failure_payload = _build_example_payload(
        example_kind="failure_case_example",
        selection_key=failure_key,
        gt_view=gt_view,
        raw_eval=raw_eval,
        refined_eval=refined_eval,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    comparator_table = _render_comparator_table(raw_summary=raw_eval.summary, refined_summary=refined_eval.summary)
    (output_root / "comparator_table.md").write_text(comparator_table, encoding="utf-8")
    _dump_json(output_root / "evaluator_spec.json", {
        "gate": "S1",
        "scope": "structure_only",
        "split": "val",
        "comparators": {
            "raw": raw_label,
            "refined": refined_label,
        },
        "ground_truth": str(gt_json_path),
        "primary_metrics": [
            "mean_best_iou",
            "recall_at_0.5",
            "fragmentation_per_gt_instance",
        ],
        "diagnostic_metrics": [
            "short_track_ratio",
            "broken_track_ratio",
            "average_track_duration",
            "temporal_inconsistency",
            "unmatched_gt_ratio",
            "over_segmentation_tendency",
            "under_coverage_tendency",
        ],
        "metric_definitions": {
            "mean_best_iou": "mean over GT instances of the best spatio-temporal tube IoU against same-video predicted tracks",
            "recall_at_0.5": "fraction of GT instances with best tube IoU >= 0.5",
            "fragmentation_per_gt_instance": "mean over GT instances of max(0, matched_pred_tracks_with_frame_iou>=threshold - 1)",
            "short_track_ratio": f"fraction of predicted tracks with active duration <= {short_track_threshold} frames",
            "broken_track_ratio": "fraction of predicted tracks with more than one active temporal segment",
            "average_track_duration": "mean number of active frames per predicted track",
            "temporal_inconsistency": "mean extra-active-segment count normalized by track duration",
            "unmatched_gt_ratio": "fraction of GT instances with best tube IoU < 0.5",
            "over_segmentation_tendency": "fraction of GT instances with more than one matched predicted fragment",
            "under_coverage_tendency": "mean of 1 - best tube IoU over GT instances",
        },
        "thresholds": {
            "frame_match_iou_threshold": frame_match_iou_threshold,
            "short_track_threshold": short_track_threshold,
        },
    })
    _dump_json(output_root / "summary.json", {
        "gate": "S1",
        "raw": dict(raw_eval.summary),
        "refined": dict(refined_eval.summary),
        "delta": _round_metrics({
            key: float(refined_eval.summary.get(key, 0.0)) - float(raw_eval.summary.get(key, 0.0))
            for key in (
                "mean_best_iou",
                "recall_at_0.5",
                "recall_at_0.7",
                "fragmentation_per_gt_instance",
                "short_track_ratio",
                "broken_track_ratio",
                "average_track_duration",
                "temporal_inconsistency",
                "unmatched_gt_ratio",
                "over_segmentation_tendency",
                "under_coverage_tendency",
            )
        }),
        "selected_examples": {
            "paired_worked_example": {
                "video_id": paired_payload["video_id"],
                "gt_tube_id": paired_payload["gt_tube_id"],
            },
            "failure_case_example": {
                "video_id": failure_payload["video_id"],
                "gt_tube_id": failure_payload["gt_tube_id"],
            },
        },
    })
    _dump_json(output_root / "paired_worked_example.json", paired_payload)
    _dump_json(output_root / "failure_case_example.json", failure_payload)
    _render_example_svg(
        title="S1 paired worked example: raw pseudo tube vs refined basis vs GT",
        payload=paired_payload,
        output_path=output_root / "paired_worked_example.svg",
    )
    _render_example_svg(
        title="S1 failure-case example: raw pseudo tube vs refined basis vs GT",
        payload=failure_payload,
        output_path=output_root / "failure_case_example.svg",
    )

    manifest = {
        "schema_name": "wsovvis.s1_basis_superiority_eval",
        "schema_version": "1.0.0",
        "gt_json": str(gt_json_path),
        "raw_json": str(raw_json_path),
        "refined_json": str(refined_json_path),
        "artifacts": {
            "summary_json": "summary.json",
            "comparator_table_md": "comparator_table.md",
            "evaluator_spec_json": "evaluator_spec.json",
            "paired_worked_example_json": "paired_worked_example.json",
            "paired_worked_example_svg": "paired_worked_example.svg",
            "failure_case_example_json": "failure_case_example.json",
            "failure_case_example_svg": "failure_case_example.svg",
        },
    }
    _dump_json(output_root / "s1_eval_manifest.json", manifest)
    return manifest
