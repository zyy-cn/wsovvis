#!/usr/bin/env python3
"""
Read-only VideoCutLER multiplicity / precision audit.

Purpose:
  1) GT-centric: how many VideoCutLER trajectories cover each GT instance.
  2) Y'-centric: how many trajectories support each (clip, Y' class) pair.
  3) trajectory-centric: how many VideoCutLER trajectories match any GT, match Y',
     match hidden GT, or match no GT.

This script does NOT change training outputs and does NOT use GT to filter training.
It is an offline audit only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from pycocotools import mask as mask_utils  # type: ignore
    PYCOCOTOOLS_AVAILABLE = True
except Exception:
    mask_utils = None
    PYCOCOTOOLS_AVAILABLE = False


def _json_load(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                # Normalize the common VideoCutLER export key used by the validated
                # coverage audit so the exact same mask decoding path can read the
                # raw trajectory surface.
                if 'masks_rle' in obj and 'mask_rles' not in obj:
                    obj = dict(obj)
                    obj['mask_rles'] = obj['masks_rle']
                yield obj


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(str(x))
        except Exception:
            return None


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x))
        except Exception:
            return None


def _get_first(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default


def _extract_dims(obj: Dict[str, Any], fallback: Optional[Dict[str, Any]] = None) -> Tuple[Optional[int], Optional[int]]:
    w = _as_int(obj.get('width'))
    h = _as_int(obj.get('height'))
    if (w is None or h is None) and fallback is not None:
        w = w if w is not None else _as_int(fallback.get('width'))
        h = h if h is not None else _as_int(fallback.get('height'))
    return w, h


def _extract_frame_indices(obj: Dict[str, Any], n: int) -> Optional[List[int]]:
    for key in ['frame_indices', 'frames', 'frame_ids', 'sampled_frame_indices', 'valid_frame_indices']:
        val = obj.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            idx: List[int] = []
            ok = True
            for x in val:
                if isinstance(x, dict):
                    x = _get_first(x, ['frame_index', 'frame_id', 'index', 'id'])
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


def _video_id(obj: Dict[str, Any]) -> Optional[str]:
    v = _get_first(obj, ['video_id', 'clip_id', 'vid', 'video', 'video_idx', 'lvvis_video_id'])
    if isinstance(v, dict):
        v = _get_first(v, ['id', 'video_id', 'clip_id'])
    if v is None:
        return None
    return str(v)


def _traj_id(obj: Dict[str, Any], fallback_idx: int = 0) -> str:
    v = _get_first(obj, ['trajectory_id', 'traj_id', 'track_id', 'id', 'instance_id', 'proposal_id'])
    if v is None:
        vid = _video_id(obj) or 'unknown'
        v = f'{vid}:{fallback_idx:06d}'
    return str(v)


def _cat_id(obj: Dict[str, Any]) -> Optional[int]:
    return _as_int(_get_first(obj, ['category_id', 'raw_id', 'class_id', 'matched_gt_raw_id_canonical', 'matched_gt_raw_id']))


def _normalise_rle(rle: Any, height: Optional[int] = None, width: Optional[int] = None) -> Optional[Dict[str, Any]]:
    if not PYCOCOTOOLS_AVAILABLE or rle is None:
        return None
    if isinstance(rle, dict):
        if 'counts' in rle and 'size' in rle:
            out = dict(rle)
            counts = out.get('counts')
            if isinstance(counts, str):
                out['counts'] = counts.encode('utf-8')
            return out
        for k in ['rle', 'segmentation', 'mask']:
            if k in rle:
                return _normalise_rle(rle.get(k), height, width)
        return None
    if isinstance(rle, list) and height and width:
        try:
            rr = mask_utils.frPyObjects(rle, int(height), int(width))
            if isinstance(rr, list):
                rr = mask_utils.merge(rr)
            return rr
        except Exception:
            return None
    return None


def _extract_mask_sequence(obj: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    """Extract a frame-aligned sequence of RLE masks from a GT or trajectory object."""
    seq = _get_first(obj, [
        'segmentations', 'segmentation', 'masks', 'mask', 'rles', 'rle',
        'frame_masks', 'pred_masks', 'trajectory_masks', 'mask_rles', 'masks_rle'
    ])
    height = _as_int(_get_first(obj, ['height', 'h']))
    width = _as_int(_get_first(obj, ['width', 'w']))
    if seq is None:
        return []
    if isinstance(seq, dict):
        if 'counts' in seq and 'size' in seq:
            return [_normalise_rle(seq, height, width)]
        vals = []
        for k in sorted(seq.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            vals.append(_normalise_rle(seq[k], height, width))
        return vals
    if isinstance(seq, list):
        return [_normalise_rle(x, height, width) for x in seq]
    return []


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


def _area(rle: Optional[Dict[str, Any]]) -> float:
    if not PYCOCOTOOLS_AVAILABLE or rle is None:
        return 0.0
    try:
        return float(mask_utils.area(rle))
    except Exception:
        return 0.0


def _intersection(r1: Optional[Dict[str, Any]], r2: Optional[Dict[str, Any]]) -> float:
    if not PYCOCOTOOLS_AVAILABLE or r1 is None or r2 is None:
        return 0.0
    try:
        inter = mask_utils.merge([r1, r2], intersect=True)
        return float(mask_utils.area(inter))
    except Exception:
        return 0.0


def _tube_iou(gt: Dict[str, Any], tr: Dict[str, Any]) -> float:
    if not gt.get('masks') or not tr.get('masks') or mask_utils is None:
        return 0.0
    gmap = _seg_by_frame(gt.get('masks') or [], gt.get('frame_indices'))
    tmap = _seg_by_frame(tr.get('masks') or [], tr.get('frame_indices'))
    frames = sorted(set(gmap) | set(tmap))
    if not frames:
        return 0.0
    inter = 0.0
    union = 0.0
    gt_w = _as_int(gt.get('width'))
    gt_h = _as_int(gt.get('height'))
    tr_w = _as_int(tr.get('width'))
    tr_h = _as_int(tr.get('height'))
    width = gt_w if gt_w is not None else tr_w
    height = gt_h if gt_h is not None else tr_h
    for fi in frames:
        gs = gmap.get(fi)
        ts = tmap.get(fi)
        gr = _normalise_rle(gs, height, width) if gs is not None else None
        trle = _normalise_rle(ts, height, width) if ts is not None else None
        if gr is None and trle is None:
            continue
        if gr is None:
            union += _area(trle) if trle is not None else 0.0
            continue
        if trle is None:
            union += _area(gr)
            continue
        ia = _intersection(gr, trle)
        aa = _area(gr)
        bb = _area(trle)
        inter += ia
        union += aa + bb - ia
    return float(inter / union) if union > 0 else 0.0


def load_categories(ann: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for c in ann.get('categories', []) or []:
        cid = _as_int(c.get('id'))
        if cid is not None:
            out[cid] = str(c.get('name', cid))
    return out


def load_gt_annotations(annotation_json: Path, base_ids: Optional[set[int]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[int, str]]:
    ann = _json_load(annotation_json)
    cats = load_categories(ann)
    by_vid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in ann.get('annotations', []) or []:
        cid = _cat_id(a)
        if cid is None:
            continue
        if base_ids is not None and cid not in base_ids:
            continue
        vid = _video_id(a)
        if vid is None:
            continue
        masks = _extract_mask_sequence(a)
        frame_idx = _extract_frame_indices(a, len(masks))
        w, h = _extract_dims(a)
        by_vid[vid].append({
            'gt_id': str(_get_first(a, ['id', 'instance_id', 'track_id'], f'{vid}:{len(by_vid[vid])}')),
            'video_id': vid,
            'raw_id': cid,
            'name': cats.get(cid, str(cid)),
            'masks': masks,
            'frame_indices': frame_idx,
            'width': w,
            'height': h,
        })
    return by_vid, cats


def load_trajectories(trajectory_jsonl: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_vid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, r in enumerate(_iter_jsonl(trajectory_jsonl)):
        vid = _video_id(r)
        if vid is None:
            continue
        masks = _extract_mask_sequence(r)
        frame_idx = _extract_frame_indices(r, len(masks))
        w, h = _extract_dims(r)
        by_vid[vid].append({
            'trajectory_id': _traj_id(r, idx),
            'video_id': vid,
            'masks': masks,
            'frame_indices': frame_idx,
            'width': w,
            'height': h,
            'raw': r,
        })
    return by_vid


def load_official_base_ids(path: Optional[Path]) -> Optional[set[int]]:
    if not path or not path.exists():
        return None
    obj = _json_load(path)
    # Common shapes.
    for key in ['base_raw_ids', 'base_ids', 'base', 'base_categories', 'base_category_ids']:
        if key in obj and isinstance(obj[key], list):
            ids = set()
            for x in obj[key]:
                if isinstance(x, dict):
                    v = _as_int(_get_first(x, ['raw_id', 'id', 'category_id']))
                else:
                    v = _as_int(x)
                if v is not None:
                    ids.add(v)
            if ids:
                return ids
    return None


def load_weak_labels(path: Path, base_ids: Optional[set[int]]) -> Dict[str, set[int]]:
    obj = _json_load(path)
    rows: Iterable[Any]
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        # Could be mapping clip_id -> labels or contain records.
        for key in ['records', 'weak_labels', 'items', 'data']:
            if isinstance(obj.get(key), list):
                rows = obj[key]
                break
        else:
            out: Dict[str, set[int]] = {}
            for k, v in obj.items():
                if isinstance(v, dict):
                    vals = _get_first(v, ['observed_raw_ids', 'weak_raw_ids', 'raw_ids', 'labels', 'category_ids'], [])
                else:
                    vals = v
                ids = {_as_int(x) for x in vals} if isinstance(vals, list) else set()
                clean = {int(x) for x in ids if x is not None and (base_ids is None or int(x) in base_ids)}
                out[str(k)] = clean
            return out
    else:
        rows = []
    out: Dict[str, set[int]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        vid = _video_id(r)
        if vid is None:
            continue
        vals = _get_first(r, ['observed_raw_ids', 'weak_raw_ids', 'raw_ids', 'labels', 'category_ids', 'candidate_ids_known'], [])
        ids = set()
        if isinstance(vals, list):
            for x in vals:
                v = _as_int(x.get('raw_id') if isinstance(x, dict) else x)
                if v is not None and (base_ids is None or v in base_ids):
                    ids.add(v)
        out[vid] = ids
    return out


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None


def _median(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    if n % 2:
        return float(ys[n // 2])
    return float((ys[n // 2 - 1] + ys[n // 2]) / 2.0)


def _rate(num: float, den: float) -> Optional[float]:
    return float(num / den) if den else None


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    keys.append(k); seen.add(k)
        fieldnames = keys
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})


def _find_validated_coverage_summary(run_root: Path, dataset_name: str) -> Optional[Path]:
    candidates = [
        run_root / 'analysis' / 'videocutler_gt_trajectory_coverage_validated' / dataset_name / 'summary.json',
        run_root / 'analysis' / 'videocutler_gt_trajectory_coverage' / dataset_name / 'summary.json',
    ]
    for p in candidates:
        if p.exists():
            return p
    # Lightweight fallback search.
    for base in [
        run_root / 'analysis',
        run_root / 'analysis' / 'videocutler_gt_trajectory_coverage_validated',
        run_root / 'analysis' / 'videocutler_gt_trajectory_coverage',
    ]:
        if not base.exists():
            continue
        for p in sorted(base.rglob('summary.json')):
            if 'videocutler_gt_trajectory_coverage' in str(p.parent):
                return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_root', required=True)
    ap.add_argument('--runtime_output_root', default='.')
    ap.add_argument('--dataset_name', default='lvvis_train_base')
    ap.add_argument('--annotation_json', required=True)
    ap.add_argument('--trajectory_jsonl', required=True)
    ap.add_argument('--weak_labels_json', required=True)
    ap.add_argument('--official_split_json', default=None)
    ap.add_argument('--base_only', default='true')
    ap.add_argument('--iou_thresholds', default='0.1,0.3,0.5,0.7')
    ap.add_argument('--top_examples', type=int, default=128)
    ap.add_argument('--show_progress', default='false')
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = run_root / 'analysis' / 'videocutler_multiplicity_precision' / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in args.iou_thresholds.split(',') if x.strip()]
    base_ids = load_official_base_ids(Path(args.official_split_json)) if args.official_split_json else None
    if str(args.base_only).lower() not in ('1','true','yes'):
        base_ids = None
    validated_coverage_summary_path = _find_validated_coverage_summary(run_root, args.dataset_name)
    validated_summary: Dict[str, Any] = {}
    if validated_coverage_summary_path and validated_coverage_summary_path.exists():
        try:
            validated_summary = _json_load(validated_coverage_summary_path)
        except Exception:
            validated_summary = {}

    gt_by_vid, cats = load_gt_annotations(Path(args.annotation_json), base_ids)
    traj_by_vid = load_trajectories(Path(args.trajectory_jsonl))
    weak_by_vid = load_weak_labels(Path(args.weak_labels_json), base_ids)

    videos = sorted(set(gt_by_vid) | set(traj_by_vid) | set(weak_by_vid), key=lambda x: int(x) if str(x).isdigit() else x)

    gt_rows: List[Dict[str, Any]] = []
    yprime_pair_rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    traj_rows: List[Dict[str, Any]] = []
    class_stats: Dict[int, Dict[str, Any]] = defaultdict(lambda: Counter())  # type: ignore
    examples: List[Dict[str, Any]] = []

    exact_iou_pairs = 0
    videos_done = 0
    for vid in videos:
        videos_done += 1
        if str(args.show_progress).lower() in ('1','true','yes') and videos_done % 200 == 0:
            print(f'[progress] videos {videos_done}/{len(videos)}', flush=True)
        gts = gt_by_vid.get(vid, [])
        trjs = traj_by_vid.get(vid, [])
        yprime = weak_by_vid.get(vid, set())
        gt_classes = {g['raw_id'] for g in gts}

        # Pairwise IoU matrix by explicit loops; VideoCutLER has ~10 traj/video so fine.
        gt_cover_counts_by_thr: Dict[str, Dict[float, int]] = {}
        gt_best: Dict[str, Tuple[float, Optional[str]]] = {}
        traj_best: Dict[str, Tuple[float, Optional[str], Optional[int]]] = {}
        class_cover_counts = {c: {t: 0 for t in thresholds} for c in gt_classes | yprime}
        class_best_iou = {c: 0.0 for c in gt_classes | yprime}

        for tr in trjs:
            traj_best[tr['trajectory_id']] = (0.0, None, None)

        for g in gts:
            gid = g['gt_id']
            cls = int(g['raw_id'])
            counts = {t: 0 for t in thresholds}
            best_iou = 0.0
            best_tid = None
            for tr in trjs:
                iou = _tube_iou(g, tr)
                exact_iou_pairs += 1
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tr['trajectory_id']
                old = traj_best.get(tr['trajectory_id'], (0.0, None, None))
                if iou > old[0]:
                    traj_best[tr['trajectory_id']] = (iou, gid, cls)
                for th in thresholds:
                    if iou >= th:
                        counts[th] += 1
            gt_cover_counts_by_thr[gid] = counts
            gt_best[gid] = (best_iou, best_tid)
            for th in thresholds:
                if counts[th] > 0:
                    class_cover_counts.setdefault(cls, {t: 0 for t in thresholds})[th] += 1
            class_best_iou[cls] = max(class_best_iou.get(cls, 0.0), best_iou)
            row = {
                'video_id': vid, 'gt_id': gid, 'raw_id': cls, 'class_name': g['name'],
                'best_iou': best_iou, 'best_trajectory_id': best_tid,
            }
            for th in thresholds:
                row[f'num_traj_covering_at_{th:g}'] = counts[th]
                row[f'multi_covered_at_{th:g}'] = counts[th] >= 2
                row[f'covered_at_{th:g}'] = counts[th] >= 1
            gt_rows.append(row)
            st = class_stats[cls]
            st['gt_instance_count'] += 1
            st['gt_class_pair_videos'].add(vid) if isinstance(st.get('gt_class_pair_videos'), set) else None

        # class-pair coverage counts for GT and Y'.
        for cls in gt_classes | yprime:
            gts_cls = [g for g in gts if int(g['raw_id']) == int(cls)]
            cover_count = {t: 0 for t in thresholds}
            best_cls_iou = 0.0
            # Count unique trajectories supporting this class-pair if they cover any instance of that class.
            for tr in trjs:
                best_for_tr_cls = 0.0
                for g in gts_cls:
                    best_for_tr_cls = max(best_for_tr_cls, _tube_iou(g, tr))
                best_cls_iou = max(best_cls_iou, best_for_tr_cls)
                for th in thresholds:
                    if best_for_tr_cls >= th:
                        cover_count[th] += 1
            if cls in yprime:
                yrow = {
                    'video_id': vid, 'raw_id': cls, 'class_name': cats.get(int(cls), str(cls)),
                    'in_gt': cls in gt_classes,
                    'gt_instance_count_for_class': len(gts_cls),
                    'best_iou_for_class_pair': best_cls_iou,
                }
                for th in thresholds:
                    yrow[f'num_traj_supporting_at_{th:g}'] = cover_count[th]
                    yrow[f'multi_supported_at_{th:g}'] = cover_count[th] >= 2
                    yrow[f'zero_supported_at_{th:g}'] = cover_count[th] == 0
                    yrow[f'supported_at_{th:g}'] = cover_count[th] >= 1
                yprime_pair_rows[(vid, int(cls))] = yrow
            st = class_stats[int(cls)]
            st['raw_id'] = int(cls)
            st['class_name'] = cats.get(int(cls), str(cls))
            if cls in yprime:
                st['yprime_pair_count'] += 1
            if cls in gt_classes:
                st['gt_class_pair_count'] += 1
            for th in thresholds:
                if cls in yprime:
                    st[f'yprime_support_pair_count_at_{th:g}'] += int(cover_count[th] >= 1)
                    st[f'yprime_multi_pair_count_at_{th:g}'] += int(cover_count[th] >= 2)
                    st[f'yprime_num_support_sum_at_{th:g}'] += cover_count[th]
                if cls in gt_classes:
                    st[f'gt_class_support_pair_count_at_{th:g}'] += int(cover_count[th] >= 1)

        # trajectory-centric rows.
        for tr in trjs:
            best_iou, gid, cls = traj_best.get(tr['trajectory_id'], (0.0, None, None))
            cls_int = int(cls) if cls is not None else None
            best_class_in_yprime = (cls_int in yprime) if cls_int is not None else False
            best_class_in_gt = (cls_int in gt_classes) if cls_int is not None else False
            row = {
                'video_id': vid,
                'trajectory_id': tr['trajectory_id'],
                'best_gt_iou': best_iou,
                'best_gt_id': gid,
                'best_gt_raw_id': cls_int,
                'best_gt_class_name': cats.get(cls_int, str(cls_int)) if cls_int is not None else '',
                'best_gt_class_in_yprime': best_class_in_yprime,
                'best_gt_class_in_gt': best_class_in_gt,
            }
            for th in thresholds:
                matched = best_iou >= th
                row[f'matched_any_gt_at_{th:g}'] = matched
                row[f'unmatched_any_gt_at_{th:g}'] = not matched
                row[f'matched_yprime_gt_at_{th:g}'] = matched and best_class_in_yprime
                row[f'matched_hidden_gt_at_{th:g}'] = matched and (not best_class_in_yprime) and best_class_in_gt
                row[f'not_yprime_not_gt_at_{th:g}'] = not matched
            traj_rows.append(row)
            if len(examples) < args.top_examples and (best_iou < 0.3 or (best_iou >= 0.5 and not best_class_in_yprime)):
                examples.append(row)

    # Aggregate.
    gt_count = len(gt_rows)
    yprime_rows_list = list(yprime_pair_rows.values())
    traj_count = len(traj_rows)
    gt_best_ious = [float(r['best_iou']) for r in gt_rows]
    validated_mean_best_iou = _as_float(validated_summary.get('mean_best_iou'))
    validated_gt_instance_recall_05 = _as_float(validated_summary.get('gt_instance_recall_at_0_5'))
    validated_oracle_support_05 = _as_float(validated_summary.get('oracle_support_rate_at_0_5'))
    mean_best_iou_per_gt = _mean(gt_best_ious)
    gt_zero_covered_rate_05 = _rate(sum(int(r['num_traj_covering_at_0.5']) == 0 for r in gt_rows), gt_count)
    yprime_zero_supported_rate_05 = _rate(sum(int(r['num_traj_supporting_at_0.5']) == 0 for r in yprime_rows_list), len(yprime_rows_list))
    gt_recall_05 = _rate(sum(int(r['num_traj_covering_at_0.5']) >= 1 for r in gt_rows), gt_count)
    yprime_support_05 = _rate(sum(int(r['num_traj_supporting_at_0.5']) >= 1 for r in yprime_rows_list), len(yprime_rows_list))
    consistency_diffs = {
        'mean_best_iou_abs_diff': None if validated_mean_best_iou is None or mean_best_iou_per_gt is None else abs(mean_best_iou_per_gt - validated_mean_best_iou),
        'gt_zero_covered_rate_at_0.5_abs_diff_vs_1_minus_gt_recall': None if validated_gt_instance_recall_05 is None or gt_zero_covered_rate_05 is None else abs(gt_zero_covered_rate_05 - (1.0 - validated_gt_instance_recall_05)),
        'yprime_zero_supported_rate_at_0.5_abs_diff_vs_1_minus_oracle_support': None if validated_oracle_support_05 is None or yprime_zero_supported_rate_05 is None else abs(yprime_zero_supported_rate_05 - (1.0 - validated_oracle_support_05)),
    }
    consistency_ok = (
        validated_coverage_summary_path is not None
        and validated_mean_best_iou is not None
        and validated_gt_instance_recall_05 is not None
        and validated_oracle_support_05 is not None
        and (consistency_diffs['mean_best_iou_abs_diff'] is not None and consistency_diffs['mean_best_iou_abs_diff'] <= 0.03)
        and (consistency_diffs['gt_zero_covered_rate_at_0.5_abs_diff_vs_1_minus_gt_recall'] is not None and consistency_diffs['gt_zero_covered_rate_at_0.5_abs_diff_vs_1_minus_gt_recall'] <= 0.02)
        and (consistency_diffs['yprime_zero_supported_rate_at_0.5_abs_diff_vs_1_minus_oracle_support'] is not None and consistency_diffs['yprime_zero_supported_rate_at_0.5_abs_diff_vs_1_minus_oracle_support'] <= 0.02)
    )
    consistency_status = 'PASS' if consistency_ok else 'INVALID_IOMATCH_SURFACE' if validated_coverage_summary_path else 'INVALID_VALIDATED_COVERAGE_AUTHORITY_NOT_FOUND'
    summary: Dict[str, Any] = {
        'status': 'PASS' if PYCOCOTOOLS_AVAILABLE and exact_iou_pairs > 0 and consistency_ok else ('INVALID_INPUT_SURFACE' if exact_iou_pairs == 0 else consistency_status),
        'dataset_name': args.dataset_name,
        'pycocotools_available': PYCOCOTOOLS_AVAILABLE,
        'exact_iou_pairs': exact_iou_pairs,
        'video_count_union': len(videos),
        'gt_instance_count': gt_count,
        'trajectory_count': traj_count,
        'yprime_class_pair_count': len(yprime_rows_list),
        'mean_best_iou_per_gt': mean_best_iou_per_gt,
        'median_best_iou_per_gt': _median(gt_best_ious),
        'validated_coverage_summary_path': str(validated_coverage_summary_path) if validated_coverage_summary_path else None,
        'validated_mean_best_iou': validated_mean_best_iou,
        'validated_gt_instance_recall_at_0.5': validated_gt_instance_recall_05,
        'validated_oracle_support_rate_at_0.5': validated_oracle_support_05,
        'consistency_check_status': consistency_status,
        'consistency_check_diffs': consistency_diffs,
    }
    for th in thresholds:
        gt_cover_counts = [int(r[f'num_traj_covering_at_{th:g}']) for r in gt_rows]
        y_support_counts = [int(r[f'num_traj_supporting_at_{th:g}']) for r in yprime_rows_list]
        traj_matched = [bool(r[f'matched_any_gt_at_{th:g}']) for r in traj_rows]
        traj_yprime = [bool(r[f'matched_yprime_gt_at_{th:g}']) for r in traj_rows]
        traj_hidden = [bool(r[f'matched_hidden_gt_at_{th:g}']) for r in traj_rows]
        traj_none = [bool(r[f'not_yprime_not_gt_at_{th:g}']) for r in traj_rows]
        summary.update({
            f'mean_num_traj_per_gt_at_{th:g}': _mean([float(x) for x in gt_cover_counts]),
            f'median_num_traj_per_gt_at_{th:g}': _median([float(x) for x in gt_cover_counts]),
            f'gt_multi_covered_rate_at_{th:g}': _rate(sum(x >= 2 for x in gt_cover_counts), gt_count),
            f'gt_zero_covered_rate_at_{th:g}': _rate(sum(x == 0 for x in gt_cover_counts), gt_count),
            f'gt_covered_rate_at_{th:g}': _rate(sum(x >= 1 for x in gt_cover_counts), gt_count),
            f'mean_num_traj_per_yprime_pair_at_{th:g}': _mean([float(x) for x in y_support_counts]),
            f'median_num_traj_per_yprime_pair_at_{th:g}': _median([float(x) for x in y_support_counts]),
            f'yprime_multi_supported_rate_at_{th:g}': _rate(sum(x >= 2 for x in y_support_counts), len(y_support_counts)),
            f'yprime_zero_supported_rate_at_{th:g}': _rate(sum(x == 0 for x in y_support_counts), len(y_support_counts)),
            f'yprime_supported_rate_at_{th:g}': _rate(sum(x >= 1 for x in y_support_counts), len(y_support_counts)),
            f'trajectory_matched_to_any_gt_rate_at_{th:g}': _rate(sum(traj_matched), traj_count),
            f'trajectory_unmatched_to_gt_rate_at_{th:g}': _rate(sum(traj_none), traj_count),
            f'trajectory_matched_to_yprime_gt_rate_at_{th:g}': _rate(sum(traj_yprime), traj_count),
            f'trajectory_matched_to_hidden_gt_rate_at_{th:g}': _rate(sum(traj_hidden), traj_count),
            f'trajectory_not_yprime_not_gt_rate_at_{th:g}': _rate(sum(traj_none), traj_count),
        })

    # Mirror the validated coverage authority at the main threshold for clear comparison.
    summary['gt_recall_at_0.5'] = gt_recall_05
    summary['yprime_support_rate_at_0.5'] = yprime_support_05

    # Class rows.
    class_rows: List[Dict[str, Any]] = []
    for cls, st in class_stats.items():
        row = {'raw_id': cls, 'class_name': cats.get(cls, str(cls))}
        for k, v in st.items():
            if isinstance(v, (str, int, float)):
                row[k] = v
        for th in thresholds:
            yden = int(st.get('yprime_pair_count', 0))
            gden = int(st.get('gt_class_pair_count', 0))
            row[f'yprime_supported_rate_at_{th:g}'] = _rate(float(st.get(f'yprime_support_pair_count_at_{th:g}', 0)), yden)
            row[f'yprime_multi_supported_rate_at_{th:g}'] = _rate(float(st.get(f'yprime_multi_pair_count_at_{th:g}', 0)), yden)
            row[f'mean_num_traj_per_yprime_pair_at_{th:g}'] = _rate(float(st.get(f'yprime_num_support_sum_at_{th:g}', 0)), yden)
            row[f'gt_class_support_rate_at_{th:g}'] = _rate(float(st.get(f'gt_class_support_pair_count_at_{th:g}', 0)), gden)
        class_rows.append(row)
    class_rows.sort(key=lambda r: (r.get('yprime_supported_rate_at_0.5') if r.get('yprime_supported_rate_at_0.5') is not None else 999, -int(r.get('yprime_pair_count', 0))))

    # Bucket summary at 0.5 if available; otherwise first threshold.
    main_th = 0.5 if 0.5 in thresholds else thresholds[0]
    bucket_counts = Counter()
    for r in traj_rows:
        if r[f'matched_yprime_gt_at_{main_th:g}']:
            bucket_counts['trajectory_matches_yprime_gt'] += 1
        elif r[f'matched_hidden_gt_at_{main_th:g}']:
            bucket_counts['trajectory_matches_hidden_gt_not_yprime'] += 1
        elif r[f'not_yprime_not_gt_at_{main_th:g}']:
            bucket_counts['trajectory_matches_no_gt_background_or_low_iou'] += 1
        else:
            bucket_counts['other'] += 1
    bucket_rows = [{'bucket': k, 'count': v, 'rate': _rate(v, traj_count)} for k, v in bucket_counts.most_common()]

    summary['bucket_counts_at_main_threshold'] = dict(bucket_counts)
    summary['main_threshold'] = main_th
    if summary['status'] == 'PASS':
        summary['interpretation'] = {
            'gt_multi_cover_claim_safe': 'Use gt_multi_covered_rate; do not claim multi-cover unless rate is high.',
            'yprime_multi_support_claim_safe': 'Use yprime_multi_supported_rate; prior evidence does not support every Yprime class having multiple supports.',
            'trajectory_background_claim_safe': 'Use trajectory_unmatched_to_gt_rate and hidden/yprime split.',
        }
    else:
        summary['interpretation'] = {
            'verdict': summary['status'],
            'validated_coverage_summary_path': summary.get('validated_coverage_summary_path'),
        }

    # Writes.
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    write_csv(out_dir / 'gt_multiplicity_rows.csv', gt_rows[:200000])
    write_csv(out_dir / 'yprime_multiplicity_rows.csv', yprime_rows_list[:200000])
    write_csv(out_dir / 'trajectory_precision_rows.csv', traj_rows[:200000])
    write_csv(out_dir / 'class_multiplicity_precision_summary.csv', class_rows)
    write_csv(out_dir / 'failure_bucket_summary.csv', bucket_rows)
    with (out_dir / 'multiplicity_precision_examples.jsonl').open('w', encoding='utf-8') as f:
        for e in examples[:args.top_examples]:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    # Compact takeover.
    lines = [
        '# VideoCutLER Multiplicity / Precision Audit', '',
        f'- status: {summary.get("status")}',
        f'- dataset: {args.dataset_name}',
        f'- exact_iou_pairs: {exact_iou_pairs}',
        f'- gt_instance_count: {gt_count}',
        f'- trajectory_count: {traj_count}',
        f'- yprime_class_pair_count: {len(yprime_rows_list)}',
        f'- mean_best_iou_per_gt: {summary.get("mean_best_iou_per_gt")}',
        f'- gt_multi_covered_rate_at_{main_th:g}: {summary.get(f"gt_multi_covered_rate_at_{main_th:g}")}',
        f'- yprime_multi_supported_rate_at_{main_th:g}: {summary.get(f"yprime_multi_supported_rate_at_{main_th:g}")}',
        f'- yprime_zero_supported_rate_at_{main_th:g}: {summary.get(f"yprime_zero_supported_rate_at_{main_th:g}")}',
        f'- trajectory_matched_to_any_gt_rate_at_{main_th:g}: {summary.get(f"trajectory_matched_to_any_gt_rate_at_{main_th:g}")}',
        f'- trajectory_matched_to_yprime_gt_rate_at_{main_th:g}: {summary.get(f"trajectory_matched_to_yprime_gt_rate_at_{main_th:g}")}',
        f'- trajectory_matched_to_hidden_gt_rate_at_{main_th:g}: {summary.get(f"trajectory_matched_to_hidden_gt_rate_at_{main_th:g}")}',
        f'- trajectory_unmatched_to_gt_rate_at_{main_th:g}: {summary.get(f"trajectory_unmatched_to_gt_rate_at_{main_th:g}")}',
        '', '## Outputs',
        f'- summary: `{out_dir / "summary.json"}`',
        f'- GT rows: `{out_dir / "gt_multiplicity_rows.csv"}`',
        f'- Yprime rows: `{out_dir / "yprime_multiplicity_rows.csv"}`',
        f'- trajectory rows: `{out_dir / "trajectory_precision_rows.csv"}`',
        f'- class summary: `{out_dir / "class_multiplicity_precision_summary.csv"}`',
        f'- buckets: `{out_dir / "failure_bucket_summary.csv"}`',
    ]
    (out_dir / 'VIDEOCUTLER_MULTIPLICITY_PRECISION_TAKEOVER.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
