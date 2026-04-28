#!/usr/bin/env python3
"""Read-only VideoCutLER GT trajectory coverage audit.

This audit answers whether frozen VideoCutLER trajectories cover LV-VIS GT
instances/classes, and whether the Y′ no-carrier gap is caused by proposal
coverage or by downstream sidecar/binding policy.

The implementation is intentionally schema-tolerant: it can compute exact tube
IoU when GT and trajectory masks are available, and it can still emit sidecar /
metadata summaries when exact masks are unavailable. It never modifies training
artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

try:
    from pycocotools import mask as mask_utils  # type: ignore
except Exception:  # pragma: no cover
    mask_utils = None


THRESHOLDS_DEFAULT = (0.1, 0.3, 0.5, 0.7)
ID_KEYS = ("video_id", "clip_id", "lvvis_video_id", "vid", "id")
TRAJ_VIDEO_KEYS = ("video_id", "clip_id", "lvvis_video_id", "source_video_id")
TRAJ_ID_KEYS = ("trajectory_id", "traj_id", "track_id", "id")
CATEGORY_KEYS = ("category_id", "raw_id", "gt_raw_id", "matched_gt_raw_id", "matched_gt_raw_id_canonical")
MATCH_IOU_KEYS = ("matched_gt_iou", "best_iou", "gt_iou", "max_iou", "tube_iou")
SEGMENTATION_KEYS = ("segmentations", "segmentation", "masks", "mask_rles", "rles", "pred_masks")
FRAME_INDEX_KEYS = ("frame_indices", "frames", "frame_ids", "sampled_frame_indices", "valid_frame_indices")


@dataclass
class GTInstance:
    ann_id: str
    video_id: str
    raw_id: int
    class_name: str
    segmentations: List[Any] = field(default_factory=list)
    frame_indices: Optional[List[int]] = None
    width: Optional[int] = None
    height: Optional[int] = None
    best_iou: float = 0.0
    best_traj_id: Optional[str] = None
    best_traj_raw_id: Optional[int] = None


@dataclass
class Trajectory:
    trajectory_id: str
    video_id: str
    segmentations: List[Any] = field(default_factory=list)
    frame_indices: Optional[List[int]] = None
    matched_raw_id: Optional[int] = None
    matched_iou: Optional[float] = None
    raw: Mapping[str, Any] = field(default_factory=dict)


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return bool(x)


def _as_int(x: Any) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def _as_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        f = float(x)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _first(row: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, Mapping):
                yield obj


def _safe_name(name: Any) -> str:
    return str(name).replace("\n", " ").replace("\r", " ").strip()


def _resolve_first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            pass
    return None


def _load_official_base_ids(repo_root: Path) -> Tuple[set[int], Dict[str, Any]]:
    path = repo_root / "package" / "reference" / "lvvis_official_base_novel_split.json"
    meta: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return set(), meta
    obj = _read_json(path)
    base = None
    novel = None
    for k in ("base_raw_ids", "base_ids", "base", "base_categories", "base_category_ids"):
        if isinstance(obj.get(k), list):
            base = obj[k]
            break
    for k in ("novel_raw_ids", "novel_ids", "novel", "novel_categories", "novel_category_ids"):
        if isinstance(obj.get(k), list):
            novel = obj[k]
            break
    base_ids: set[int] = set()
    if base is not None:
        for x in base:
            if isinstance(x, Mapping):
                v = _as_int(_first(x, ("raw_id", "id", "category_id")))
            else:
                v = _as_int(x)
            if v is not None:
                base_ids.add(v)
    novel_count = 0
    if novel is not None:
        novel_count = sum(1 for x in novel if (_as_int(_first(x, ("raw_id", "id", "category_id")) if isinstance(x, Mapping) else x) is not None))
    meta.update({"base_count": len(base_ids), "novel_count": novel_count})
    return base_ids, meta


def _category_maps(ann: Mapping[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for c in ann.get("categories", []) or []:
        if not isinstance(c, Mapping):
            continue
        rid = _as_int(_first(c, ("id", "raw_id", "category_id")))
        if rid is not None:
            out[rid] = _safe_name(c.get("name", f"class_{rid}"))
    return out


def _video_meta(ann: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    meta: Dict[str, Mapping[str, Any]] = {}
    for v in ann.get("videos", []) or []:
        if isinstance(v, Mapping):
            vid = _first(v, ID_KEYS)
            if vid is not None:
                meta[str(vid)] = v
    for im in ann.get("images", []) or []:
        if isinstance(im, Mapping):
            vid = _first(im, ("video_id", "clip_id", "id"))
            if vid is not None:
                meta.setdefault(str(vid), im)
    return meta


def _extract_dims(obj: Mapping[str, Any], fallback: Optional[Mapping[str, Any]] = None) -> Tuple[Optional[int], Optional[int]]:
    w = _as_int(obj.get("width"))
    h = _as_int(obj.get("height"))
    if (w is None or h is None) and fallback is not None:
        w = w if w is not None else _as_int(fallback.get("width"))
        h = h if h is not None else _as_int(fallback.get("height"))
    return w, h


def _extract_segmentations(row: Mapping[str, Any]) -> List[Any]:
    for k in SEGMENTATION_KEYS:
        val = row.get(k)
        if val is None:
            continue
        if isinstance(val, list):
            return val
        # single segmentation dict/polygon
        return [val]
    frames = row.get("frames")
    if isinstance(frames, list):
        segs = []
        for fr in frames:
            if isinstance(fr, Mapping):
                segs.append(_first(fr, SEGMENTATION_KEYS))
        if segs:
            return segs
    return []


def _extract_frame_indices(row: Mapping[str, Any], n: int) -> Optional[List[int]]:
    for k in FRAME_INDEX_KEYS:
        val = row.get(k)
        if val is None:
            continue
        if isinstance(val, list):
            idx: List[int] = []
            ok = True
            for x in val:
                if isinstance(x, Mapping):
                    x = _first(x, ("frame_index", "frame_id", "index", "id"))
                v = _as_int(x)
                if v is None:
                    ok = False
                    break
                idx.append(v)
            if ok and idx:
                return idx
    if n > 0:
        return list(range(n))
    return None


def _load_gt_instances(annotation_json: Path, base_ids: Optional[set[int]], base_only: bool) -> Tuple[List[GTInstance], Dict[str, set[int]], Dict[str, Any], Dict[int, str]]:
    ann = _read_json(annotation_json)
    names = _category_maps(ann)
    videos = _video_meta(ann)
    gts: List[GTInstance] = []
    gt_classes_by_video: Dict[str, set[int]] = defaultdict(set)
    for a in ann.get("annotations", []) or []:
        if not isinstance(a, Mapping):
            continue
        rid = _as_int(_first(a, CATEGORY_KEYS))
        if rid is None:
            continue
        if base_only and base_ids and rid not in base_ids:
            continue
        vid = _first(a, ("video_id", "clip_id", "image_id"))
        if vid is None:
            continue
        vid_s = str(vid)
        vmeta = videos.get(vid_s)
        w, h = _extract_dims(a, vmeta)
        segs = _extract_segmentations(a)
        frame_idx = _extract_frame_indices(a, len(segs))
        ann_id = str(_first(a, ("id", "ann_id", "instance_id"), f"{vid_s}:{len(gts)}"))
        gts.append(GTInstance(
            ann_id=ann_id,
            video_id=vid_s,
            raw_id=rid,
            class_name=names.get(rid, f"class_{rid}"),
            segmentations=segs,
            frame_indices=frame_idx,
            width=w,
            height=h,
        ))
        gt_classes_by_video[vid_s].add(rid)
    meta = {"annotation_json": str(annotation_json), "gt_instance_count": len(gts), "video_count": len(gt_classes_by_video)}
    return gts, gt_classes_by_video, meta, names


def _load_trajectories(path: Path) -> Tuple[Dict[str, List[Trajectory]], Dict[str, Any]]:
    by_video: Dict[str, List[Trajectory]] = defaultdict(list)
    count = 0
    with_masks = 0
    with_sidecar = 0
    for rec in _iter_jsonl(path):
        vid = _first(rec, TRAJ_VIDEO_KEYS)
        tid = _first(rec, TRAJ_ID_KEYS)
        if vid is None or tid is None:
            continue
        segs = _extract_segmentations(rec)
        if segs:
            with_masks += 1
        frame_idx = _extract_frame_indices(rec, len(segs))
        matched_raw = _as_int(_first(rec, CATEGORY_KEYS))
        matched_iou = _as_float(_first(rec, MATCH_IOU_KEYS))
        if matched_raw is not None or matched_iou is not None:
            with_sidecar += 1
        by_video[str(vid)].append(Trajectory(
            trajectory_id=str(tid),
            video_id=str(vid),
            segmentations=segs,
            frame_indices=frame_idx,
            matched_raw_id=matched_raw,
            matched_iou=matched_iou,
            raw=rec,
        ))
        count += 1
    return by_video, {"trajectory_records_jsonl": str(path), "trajectory_count": count, "with_masks": with_masks, "with_sidecar_fields": with_sidecar, "video_count": len(by_video)}


def _normalize_rle(seg: Any, height: Optional[int], width: Optional[int]) -> Optional[Mapping[str, Any]]:
    if seg is None or seg == []:
        return None
    if mask_utils is None:
        return None
    if isinstance(seg, Mapping) and "counts" in seg and "size" in seg:
        rle = dict(seg)
        counts = rle.get("counts")
        if isinstance(counts, str):
            rle["counts"] = counts.encode("utf-8")
        return rle
    if height is None or width is None:
        return None
    try:
        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, int(height), int(width))
            if isinstance(rles, list):
                return mask_utils.merge(rles)
            return rles
    except Exception:
        return None
    return None


def _rle_area(rle: Mapping[str, Any]) -> float:
    if mask_utils is None:
        return 0.0
    try:
        return float(mask_utils.area(rle))
    except Exception:
        return 0.0


def _rle_intersection(rle_a: Mapping[str, Any], rle_b: Mapping[str, Any]) -> float:
    if mask_utils is None:
        return 0.0
    try:
        return float(mask_utils.area(mask_utils.merge([rle_a, rle_b], intersect=True)))
    except Exception:
        return 0.0


def _seg_by_frame(segs: Sequence[Any], frame_indices: Optional[Sequence[int]]) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    if not segs:
        return out
    if frame_indices and len(frame_indices) == len(segs):
        for idx, seg in zip(frame_indices, segs):
            if seg is not None and seg != []:
                out[int(idx)] = seg
    else:
        for idx, seg in enumerate(segs):
            if seg is not None and seg != []:
                out[int(idx)] = seg
    return out


def _tube_iou(gt: GTInstance, tr: Trajectory) -> Optional[float]:
    if not gt.segmentations or not tr.segmentations or mask_utils is None:
        return None
    gmap = _seg_by_frame(gt.segmentations, gt.frame_indices)
    tmap = _seg_by_frame(tr.segmentations, tr.frame_indices)
    frames = sorted(set(gmap) | set(tmap))
    if not frames:
        return None
    inter = 0.0
    union = 0.0
    for fi in frames:
        gs = gmap.get(fi)
        ts = tmap.get(fi)
        gr = _normalize_rle(gs, gt.height, gt.width) if gs is not None else None
        trle = _normalize_rle(ts, gt.height, gt.width) if ts is not None else None
        if gr is None and trle is None:
            continue
        if gr is None:
            union += _rle_area(trle) if trle is not None else 0.0
            continue
        if trle is None:
            union += _rle_area(gr)
            continue
        ia = _rle_intersection(gr, trle)
        aa = _rle_area(gr)
        ba = _rle_area(trle)
        inter += ia
        union += aa + ba - ia
    if union <= 0:
        return None
    return float(inter / union)


def _compute_best_iou(gts: List[GTInstance], traj_by_video: Mapping[str, List[Trajectory]], show_progress: bool) -> Dict[str, Any]:
    exact_pairs = 0
    sidecar_pairs = 0
    missing_masks = 0
    iterator: Iterable[GTInstance] = gts
    if show_progress and tqdm is not None:
        iterator = tqdm(gts, desc="best-IoU GT instances", dynamic_ncols=True)
    for gt in iterator:
        best_iou = 0.0
        best_tid: Optional[str] = None
        best_raw: Optional[int] = None
        trs = traj_by_video.get(gt.video_id, [])
        if not trs:
            continue
        for tr in trs:
            iou = _tube_iou(gt, tr)
            if iou is None:
                missing_masks += 1
                # Fallback only for sidecar exact category match; this does NOT prove oracle IoU.
                if tr.matched_raw_id == gt.raw_id and tr.matched_iou is not None:
                    iou = tr.matched_iou
                    sidecar_pairs += 1
                else:
                    continue
            else:
                exact_pairs += 1
            if iou > best_iou:
                best_iou = float(iou)
                best_tid = tr.trajectory_id
                best_raw = tr.matched_raw_id
        gt.best_iou = best_iou
        gt.best_traj_id = best_tid
        gt.best_traj_raw_id = best_raw
    return {"exact_iou_pairs": exact_pairs, "sidecar_fallback_pairs": sidecar_pairs, "missing_mask_pair_attempts": missing_masks, "pycocotools_available": mask_utils is not None}


def _load_yprime_rows(path: Optional[Path]) -> Tuple[List[Mapping[str, Any]], Dict[Tuple[str, int], bool]]:
    rows: List[Mapping[str, Any]] = []
    support: Dict[Tuple[str, int], bool] = {}
    if path is None or not path.exists():
        return rows, support
    for r in _iter_jsonl(path):
        if r.get("record_type") not in (None, "yprime_pair"):
            continue
        clip = _first(r, ("clip_id", "video_id"))
        raw = _as_int(_first(r, ("raw_id", "yprime_raw_id", "category_id")))
        if clip is None or raw is None:
            continue
        rows.append(r)
        has = r.get("has_carrier_support")
        if has is not None:
            support[(str(clip), raw)] = _as_bool(has)
    return rows, support


def _load_gt_yprime_rows(path: Optional[Path]) -> Tuple[List[Mapping[str, Any]], set[Tuple[str, int]], set[Tuple[str, int]]]:
    yprime_pairs: set[Tuple[str, int]] = set()
    gt_pairs: set[Tuple[str, int]] = set()
    rows: List[Mapping[str, Any]] = []
    if path is None or not path.exists():
        return rows, yprime_pairs, gt_pairs
    for r in _iter_jsonl(path):
        rows.append(r)
        clip = _first(r, ("clip_id", "video_id"))
        raw = _as_int(_first(r, ("raw_id", "yprime_raw_id", "gt_raw_id", "category_id")))
        if clip is None or raw is None:
            continue
        if r.get("record_type") == "gt_missing_from_yprime":
            gt_pairs.add((str(clip), raw))
        elif r.get("record_type") == "yprime_pair":
            yprime_pairs.add((str(clip), raw))
            if _as_bool(r.get("in_gt_annotation", True)):
                gt_pairs.add((str(clip), raw))
    return rows, yprime_pairs, gt_pairs


def _mean(vals: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _median(vals: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def _rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num / den)


def _write_csv(path: Path, rows: List[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo_root = Path(args.runtime_output_root).resolve()
    run_root = Path(args.run_root).resolve()
    out_dir = run_root / "analysis" / "videocutler_gt_trajectory_coverage" / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    annotation_json = Path(args.annotation_json) if args.annotation_json else _resolve_first_existing([
        Path("/home/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations/train_instances.json"),
        repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json",
        repo_root / "datasets" / "LV-VIS" / "annotations" / "train_instances.json",
    ])
    if annotation_json is None:
        raise SystemExit("annotation_json not found; pass --annotation_json")

    trajectory_jsonl = Path(args.trajectory_jsonl) if args.trajectory_jsonl else _resolve_first_existing([
        Path("/home/zyy/code/wsovvis_asserts/exports") / args.dataset_name / "trajectory_records.jsonl",
        repo_root / "exports" / args.dataset_name / "trajectory_records.jsonl",
        repo_root / "videocutler" / "exports" / args.dataset_name / "trajectory_records.jsonl",
    ])
    if trajectory_jsonl is None:
        raise SystemExit("trajectory_jsonl not found; pass --trajectory_jsonl")

    yprime_rows_path = Path(args.yprime_support_rows) if args.yprime_support_rows else _resolve_first_existing([
        run_root / "analysis" / "yprime_support_coverage" / args.dataset_name / "softem_aug" / "clip_yprime_support_rows.jsonl",
        run_root / "analysis" / "yprime_support_coverage" / args.dataset_name / "prealign" / "clip_yprime_support_rows.jsonl",
    ])
    gt_yprime_rows_path = Path(args.gt_yprime_rows) if args.gt_yprime_rows else _resolve_first_existing([
        run_root / "analysis" / "gt_yprime_coverage" / args.dataset_name / "clip_gt_yprime_rows.jsonl",
    ])

    base_ids, split_meta = _load_official_base_ids(repo_root)
    gts, gt_classes_by_video, ann_meta, class_names = _load_gt_instances(annotation_json, base_ids, _as_bool(args.base_only))
    traj_by_video, traj_meta = _load_trajectories(trajectory_jsonl)
    iou_meta = _compute_best_iou(gts, traj_by_video, _as_bool(args.show_progress))

    thresholds = [float(x) for x in str(args.iou_thresholds).split(",") if x.strip()]
    gt_ious = [g.best_iou for g in gts]
    instance_recall = {f"gt_instance_recall_at_{str(t).replace('.', '_')}": _rate(sum(1 for g in gts if g.best_iou >= t), len(gts)) for t in thresholds}
    instance_summary = {
        "gt_instance_count": len(gts),
        "mean_best_iou": _mean(gt_ious),
        "median_best_iou": _median(gt_ious),
        **instance_recall,
        "unmatched_gt_instance_rate_at_0_5": _rate(sum(1 for g in gts if g.best_iou < 0.5), len(gts)),
    }

    # GT class-pair oracle support.
    gt_by_video_class: Dict[Tuple[str, int], List[GTInstance]] = defaultdict(list)
    for g in gts:
        gt_by_video_class[(g.video_id, g.raw_id)].append(g)
    gt_pairs = set(gt_by_video_class)
    class_pair_support: Dict[float, set[Tuple[str, int]]] = {t: set() for t in thresholds}
    for pair, insts in gt_by_video_class.items():
        best = max((g.best_iou for g in insts), default=0.0)
        for t in thresholds:
            if best >= t:
                class_pair_support[t].add(pair)
    gt_class_pair_summary = {f"clip_gt_class_support_rate_at_{str(t).replace('.', '_')}": _rate(len(class_pair_support[t]), len(gt_pairs)) for t in thresholds}

    yprime_rows, sidecar_support = _load_yprime_rows(yprime_rows_path)
    _, yprime_pairs_from_gt_rows, gt_pairs_from_gt_yprime = _load_gt_yprime_rows(gt_yprime_rows_path)
    yprime_pairs = {(str(_first(r, ("clip_id", "video_id"))), int(_as_int(_first(r, ("raw_id", "yprime_raw_id", "category_id"))) or -1)) for r in yprime_rows}
    yprime_pairs = {p for p in yprime_pairs if p[0] not in ("None", "") and p[1] >= 0}
    if not yprime_pairs and yprime_pairs_from_gt_rows:
        yprime_pairs = yprime_pairs_from_gt_rows

    yprime_support_summary: Dict[str, Any] = {"clip_yprime_pair_count": len(yprime_pairs)}
    for t in thresholds:
        supported = sum(1 for p in yprime_pairs if p in class_pair_support[t])
        yprime_support_summary[f"clip_yprime_support_rate_at_{str(t).replace('.', '_')}"] = _rate(supported, len(yprime_pairs))
    if yprime_pairs:
        no_oracle_05 = sum(1 for p in yprime_pairs if p not in class_pair_support.get(0.5, set()))
        yprime_support_summary["yprime_in_gt_but_no_oracle_trajectory_rate_at_0_5"] = _rate(no_oracle_05, len(yprime_pairs))

    # Sidecar / existing auditable support gap.
    sidecar_pairs_supported = {p for p, has in sidecar_support.items() if has}
    sidecar_support_rate = _rate(len(sidecar_pairs_supported & yprime_pairs), len(yprime_pairs)) if yprime_pairs else None
    oracle05 = class_pair_support.get(0.5, set())
    oracle03 = class_pair_support.get(0.3, set())
    sidecar_missing_but_oracle05 = (oracle05 & yprime_pairs) - sidecar_pairs_supported
    sidecar_missing_but_oracle03 = (oracle03 & yprime_pairs) - sidecar_pairs_supported
    oracle_missing05 = yprime_pairs - oracle05

    sidecar_oracle_summary = {
        "sidecar_support_rate": sidecar_support_rate,
        "oracle_support_rate_at_0_3": _rate(len(oracle03 & yprime_pairs), len(yprime_pairs)) if yprime_pairs else None,
        "oracle_support_rate_at_0_5": _rate(len(oracle05 & yprime_pairs), len(yprime_pairs)) if yprime_pairs else None,
        "sidecar_missing_but_oracle_found_count_at_0_5": len(sidecar_missing_but_oracle05),
        "sidecar_missing_but_oracle_found_rate_at_0_5": _rate(len(sidecar_missing_but_oracle05), len(yprime_pairs)) if yprime_pairs else None,
        "sidecar_missing_but_oracle_found_count_at_0_3": len(sidecar_missing_but_oracle03),
        "sidecar_missing_but_oracle_found_rate_at_0_3": _rate(len(sidecar_missing_but_oracle03), len(yprime_pairs)) if yprime_pairs else None,
        "oracle_missing_count_at_0_5": len(oracle_missing05),
        "oracle_missing_rate_at_0_5": _rate(len(oracle_missing05), len(yprime_pairs)) if yprime_pairs else None,
    }

    # Class-level summary.
    yprime_pair_count_by_class = Counter(raw for _, raw in yprime_pairs)
    gt_pair_count_by_class = Counter(raw for _, raw in gt_pairs)
    sidecar_supported_by_class = Counter(raw for _, raw in sidecar_pairs_supported & yprime_pairs)
    oracle03_by_class = Counter(raw for _, raw in oracle03 & yprime_pairs)
    oracle05_by_class = Counter(raw for _, raw in oracle05 & yprime_pairs)
    best_iou_by_class: Dict[int, List[float]] = defaultdict(list)
    inst_count_by_class = Counter()
    for g in gts:
        best_iou_by_class[g.raw_id].append(g.best_iou)
        inst_count_by_class[g.raw_id] += 1

    class_rows: List[Dict[str, Any]] = []
    for raw in sorted(set(inst_count_by_class) | set(yprime_pair_count_by_class) | set(gt_pair_count_by_class)):
        ypc = yprime_pair_count_by_class[raw]
        gpc = gt_pair_count_by_class[raw]
        row = {
            "raw_id": raw,
            "class_name": class_names.get(raw, f"class_{raw}"),
            "gt_instance_count": inst_count_by_class[raw],
            "gt_class_pair_count": gpc,
            "yprime_pair_count": ypc,
            "sidecar_support_rate": _rate(sidecar_supported_by_class[raw], ypc) if ypc else None,
            "oracle_support_rate_at_0_3": _rate(oracle03_by_class[raw], ypc) if ypc else None,
            "oracle_support_rate_at_0_5": _rate(oracle05_by_class[raw], ypc) if ypc else None,
            "mean_best_iou": _mean(best_iou_by_class.get(raw, [])),
            "median_best_iou": _median(best_iou_by_class.get(raw, [])),
            "sidecar_oracle_gap_at_0_5": ( _rate(oracle05_by_class[raw], ypc) - _rate(sidecar_supported_by_class[raw], ypc) ) if ypc and _rate(oracle05_by_class[raw], ypc) is not None and _rate(sidecar_supported_by_class[raw], ypc) is not None else None,
        }
        class_rows.append(row)
    class_rows_sorted = sorted(class_rows, key=lambda r: ((r["oracle_support_rate_at_0_5"] is None, r["oracle_support_rate_at_0_5"] if r["oracle_support_rate_at_0_5"] is not None else 999), -int(r.get("yprime_pair_count") or 0)))

    # Failure buckets for Y′ class pairs.
    bucket_counts = Counter()
    examples: List[Dict[str, Any]] = []
    for p in sorted(yprime_pairs):
        side = p in sidecar_pairs_supported
        oracle5 = p in oracle05
        oracle3 = p in oracle03
        if side:
            b = "sidecar_supported"
        elif oracle5:
            b = "sidecar_missing_but_oracle_iou_ge_0_5"
        elif oracle3:
            b = "sidecar_missing_but_oracle_iou_0_3_to_0_5"
        else:
            b = "oracle_low_iou_below_0_3_or_no_candidate"
        bucket_counts[b] += 1
        if len(examples) < int(args.top_examples):
            vid, raw = p
            insts = gt_by_video_class.get(p, [])
            examples.append({
                "video_id": vid,
                "raw_id": raw,
                "class_name": class_names.get(raw, f"class_{raw}"),
                "bucket": b,
                "sidecar_supported": side,
                "oracle_best_iou_for_class_pair": max((g.best_iou for g in insts), default=0.0),
                "best_traj_id": max(insts, key=lambda g: g.best_iou).best_traj_id if insts else None,
                "gt_instance_count_for_pair": len(insts),
            })

    failure_rows = []
    for b, c in bucket_counts.most_common():
        failure_rows.append({"bucket": b, "count": c, "rate_vs_yprime_pairs": _rate(c, len(yprime_pairs)) if yprime_pairs else None})

    summary: Dict[str, Any] = {
        "status": "PASS",
        "dataset_name": args.dataset_name,
        "base_only": _as_bool(args.base_only),
        "output_dir": str(out_dir),
        "split_meta": split_meta,
        "annotation_meta": ann_meta,
        "trajectory_meta": traj_meta,
        "iou_meta": iou_meta,
        **instance_summary,
        "clip_gt_class_pair_count": len(gt_pairs),
        **gt_class_pair_summary,
        **yprime_support_summary,
        **sidecar_oracle_summary,
        "failure_bucket_counts": dict(bucket_counts),
        "interpretation": {},
    }
    # Compact verdict.
    if yprime_pairs and sidecar_support_rate is not None:
        o5 = sidecar_oracle_summary.get("oracle_support_rate_at_0_5") or 0.0
        o3 = sidecar_oracle_summary.get("oracle_support_rate_at_0_3") or 0.0
        if o5 > (sidecar_support_rate + 0.10):
            verdict = "sidecar_binding_gap_large_oracle_coverage_higher"
        elif o3 < 0.60:
            verdict = "proposal_or_oracle_coverage_gap_large"
        else:
            verdict = "trajectory_coverage_reasonable_under_oracle"
    else:
        verdict = "insufficient_yprime_sidecar_join_for_verdict"
    summary["interpretation"] = {
        "verdict": verdict,
        "note": "Use oracle IoU support, not previous sidecar-only support, to decide whether the 52.64% no-carrier gap is proposal missing or sidecar/binding missing.",
    }

    # Write outputs.
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_dir / "class_trajectory_coverage_summary.csv", class_rows_sorted, [
        "raw_id", "class_name", "gt_instance_count", "gt_class_pair_count", "yprime_pair_count",
        "sidecar_support_rate", "oracle_support_rate_at_0_3", "oracle_support_rate_at_0_5",
        "mean_best_iou", "median_best_iou", "sidecar_oracle_gap_at_0_5",
    ])
    worst_rows = [r for r in class_rows_sorted if (r.get("yprime_pair_count") or 0) > 0][:100]
    _write_csv(out_dir / "worst_classes.csv", worst_rows, [
        "raw_id", "class_name", "gt_instance_count", "gt_class_pair_count", "yprime_pair_count",
        "sidecar_support_rate", "oracle_support_rate_at_0_3", "oracle_support_rate_at_0_5",
        "mean_best_iou", "median_best_iou", "sidecar_oracle_gap_at_0_5",
    ])
    _write_csv(out_dir / "failure_bucket_summary.csv", failure_rows, ["bucket", "count", "rate_vs_yprime_pairs"])
    with (out_dir / "coverage_examples.jsonl").open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    takeover = [
        "# VideoCutLER GT Trajectory Coverage Audit",
        "",
        f"- status: {summary['status']}",
        f"- verdict: {verdict}",
        f"- dataset: {args.dataset_name}",
        f"- gt_instance_count: {summary.get('gt_instance_count')}",
        f"- mean_best_iou: {summary.get('mean_best_iou')}",
        f"- median_best_iou: {summary.get('median_best_iou')}",
        f"- gt_instance_recall_at_0_5: {summary.get('gt_instance_recall_at_0_5')}",
        f"- clip_gt_class_support_rate_at_0_5: {summary.get('clip_gt_class_support_rate_at_0_5')}",
        f"- clip_yprime_support_rate_at_0_5: {summary.get('clip_yprime_support_rate_at_0_5')}",
        f"- sidecar_support_rate: {summary.get('sidecar_support_rate')}",
        f"- oracle_support_rate_at_0_5: {summary.get('oracle_support_rate_at_0_5')}",
        f"- sidecar_missing_but_oracle_found_rate_at_0_5: {summary.get('sidecar_missing_but_oracle_found_rate_at_0_5')}",
        "",
        "## Outputs",
        f"- summary: `{out_dir / 'summary.json'}`",
        f"- class summary: `{out_dir / 'class_trajectory_coverage_summary.csv'}`",
        f"- worst classes: `{out_dir / 'worst_classes.csv'}`",
        f"- buckets: `{out_dir / 'failure_bucket_summary.csv'}`",
        f"- examples: `{out_dir / 'coverage_examples.jsonl'}`",
    ]
    (out_dir / "VIDEOCUTLER_GT_TRAJECTORY_COVERAGE_TAKEOVER.md").write_text("\n".join(takeover) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only VideoCutLER GT trajectory coverage audit")
    p.add_argument("--run_root", required=True)
    p.add_argument("--runtime_output_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--annotation_json", default=None)
    p.add_argument("--trajectory_jsonl", default=None)
    p.add_argument("--yprime_support_rows", default=None)
    p.add_argument("--gt_yprime_rows", default=None)
    p.add_argument("--base_only", default="true")
    p.add_argument("--iou_thresholds", default="0.1,0.3,0.5,0.7")
    p.add_argument("--top_examples", type=int, default=128)
    p.add_argument("--show_progress", default="true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps({
        "status": summary.get("status"),
        "output_dir": summary.get("output_dir"),
        "verdict": summary.get("interpretation", {}).get("verdict"),
        "gt_instance_recall_at_0_5": summary.get("gt_instance_recall_at_0_5"),
        "clip_yprime_support_rate_at_0_5": summary.get("clip_yprime_support_rate_at_0_5"),
        "sidecar_missing_but_oracle_found_rate_at_0_5": summary.get("sidecar_missing_but_oracle_found_rate_at_0_5"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
