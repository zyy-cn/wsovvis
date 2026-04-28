#!/usr/bin/env python3
"""
WSOVVIS assignment non-identifiability audit.

Read-only posthoc audit.  It aggregates existing Stage-B audit artifacts to
formalize whether the current observed evidence is sufficient to identify the
latent trajectory-to-class assignment.

It intentionally does not train, infer, rewrite checkpoints, create sidecars, or
hard-code any hub prior into training.  Optional diagnostic hubs are used only as
posthoc analysis labels; the script also computes data-driven hubs from the
responsibility records.

Expected upstream artifacts, when available:
  <run_root>/analysis/yprime_support_coverage/<dataset>/<stage>/summary.json
  <run_root>/analysis/yprime_support_coverage/<dataset>/<stage>/clip_yprime_support_rows.jsonl
  <run_root>/analysis/text_projector_hubness/<dataset>/<stage>/row_text_margin_diagnostics.jsonl
  <run_root>/analysis/text_projector_hubness/<dataset>/<stage>/stage_comparison_summary.csv
  <run_root>/analysis/hub_carrier_separability/<dataset>/<stage>/row_geometry_diagnostics.jsonl
  <run_root>/train/<stage>/responsibility_records.jsonl
  <run_root>/analysis/support_null_responsibility/<dataset>/<stage>/summary.json

Outputs:
  <run_root>/analysis/assignment_nonidentifiability/<dataset>/<stage>/summary.json
  <run_root>/analysis/assignment_nonidentifiability/<dataset>/<stage>/NONIDENTIFIABILITY_TAKEOVER.md
  <run_root>/analysis/assignment_nonidentifiability/<dataset>/<stage>/oracle_support_upper_bound.csv
  <run_root>/analysis/assignment_nonidentifiability/<dataset>/<stage>/score_ambiguity_summary.csv
  <run_root>/analysis/assignment_nonidentifiability/<dataset>/<stage>/observational_equivalence_neighbors.csv
  <run_root>/analysis/assignment_nonidentifiability/<dataset>/<stage>/observational_equivalence_summary.json
  <run_root>/analysis/assignment_nonidentifiability/<dataset>/<stage>/data_driven_hub_summary.csv
  <run_root>/analysis/assignment_nonidentifiability/<dataset>/<stage>/cooccurrence_shortcut_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

Record = Dict[str, Any]


def _parse_int_tuple(value: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in str(value or '').replace(';', ',').split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    return tuple(sorted(set(out)))


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == '':
            return default
        v = float(value)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == '':
            return default
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [_safe_float(v) for v in values]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _median(values: Iterable[Any]) -> Optional[float]:
    vals = sorted(float(v) for v in (_safe_float(x) for x in values) if v is not None)
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _rate(numer: Any, denom: Any = None) -> Optional[float]:
    if denom is None:
        vals = [bool(x) for x in numer]
        if not vals:
            return None
        return float(sum(1 for x in vals if x) / len(vals))
    try:
        d = float(denom)
        if d <= 0:
            return None
        return float(float(numer) / d)
    except Exception:
        return None


def _read_json(path: Path) -> Record:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _iter_jsonl(path: Path, *, max_rows: int = 0) -> Iterable[Record]:
    if not path.is_file():
        return
    count = 0
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if max_rows and count >= int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                count += 1
                yield obj


def _read_csv(path: Path, *, max_rows: int = 0) -> List[Record]:
    if not path.is_file():
        return []
    rows: List[Record] = []
    try:
        with path.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(dict(row))
                if max_rows and len(rows) >= int(max_rows):
                    break
    except Exception:
        return []
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=False) + '\n', encoding='utf-8')


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k); keys.append(str(k))
        fieldnames = keys
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in {'1', 'true', 't', 'yes', 'y', 'on'}


def _flatten_stage_row(stage_csv_rows: Sequence[Mapping[str, Any]]) -> Record:
    return dict(stage_csv_rows[0]) if stage_csv_rows else {}


def _oracle_support_upper_bound(
    *,
    support_summary: Mapping[str, Any],
    class_support_rows: Sequence[Mapping[str, Any]],
    iou_threshold_label: str,
) -> Tuple[Record, List[Record]]:
    pair_count = _safe_int(support_summary.get('clip_yprime_pair_count'))
    support_rate = _safe_float(support_summary.get('yprime_trajectory_support_rate'))
    clip_all_rate = _safe_float(support_summary.get('clip_all_yprime_supported_rate'))
    zero_rate = None if support_rate is None else float(1.0 - support_rate)
    upper = {
        'threshold': str(iou_threshold_label),
        'clip_yprime_pair_count': pair_count,
        'yprime_support_rate': support_rate,
        'yprime_zero_support_rate': zero_rate,
        'clip_all_yprime_supported_rate': clip_all_rate,
        'interpretation': 'Any trajectory-assignment method is upper-bounded by proposal support for Yprime pairs at this IoU threshold.',
    }
    worst_rows = []
    for row in class_support_rows:
        sr = _safe_float(row.get('support_rate'))
        if sr is None:
            continue
        worst_rows.append({
            'raw_id': row.get('raw_id'),
            'name': row.get('name'),
            'clip_yprime_count': row.get('clip_yprime_count'),
            'support_rate': sr,
            'zero_support_count': row.get('zero_support_count'),
            'supported_pair_count': row.get('supported_pair_count'),
        })
    worst_rows = sorted(worst_rows, key=lambda r: (float(r.get('support_rate', 0.0)), -int(_safe_int(r.get('clip_yprime_count'), 0) or 0)))[:50]
    return upper, worst_rows


def _score_ambiguity_summary(
    *,
    yprime_rows: Sequence[Mapping[str, Any]],
    text_rows: Sequence[Mapping[str, Any]],
    stage_summary_row: Mapping[str, Any],
    resp_summary: Mapping[str, Any],
) -> Tuple[Record, List[Record]]:
    supported = [r for r in yprime_rows if _truthy(r.get('has_trajectory_support'))]
    resp_available = [r for r in yprime_rows if _truthy(r.get('responsibility_available_for_y'))]
    score_summary = {
        'supported_yprime_pair_count': int(len(supported)),
        'all_yprime_pair_count': int(len(yprime_rows)),
        'supported_rank1_among_yprime_rate': _rate([_safe_int(r.get('best_rank_of_y_among_yprime')) == 1 for r in supported if r.get('best_rank_of_y_among_yprime') is not None]),
        'supported_rank1_full_vocab_rate': _rate([_safe_int(r.get('best_rank_of_y_among_vocab')) == 1 for r in supported if r.get('best_rank_of_y_among_vocab') is not None]),
        'supported_top5_full_vocab_rate': _rate([(_safe_int(r.get('best_rank_of_y_among_vocab'), 10**9) or 10**9) <= 5 for r in supported if r.get('best_rank_of_y_among_vocab') is not None]),
        'supported_top20_full_vocab_rate': _rate([(_safe_int(r.get('best_rank_of_y_among_vocab'), 10**9) or 10**9) <= 20 for r in supported if r.get('best_rank_of_y_among_vocab') is not None]),
        'supported_best_yprime_rank_mean': _mean([r.get('best_rank_of_y_among_yprime') for r in supported]),
        'supported_best_vocab_rank_mean': _mean([r.get('best_rank_of_y_among_vocab') for r in supported]),
        'supported_best_margin_vs_diagnostic_hub_mean': _mean([r.get('best_margin_vs_hub') for r in supported]),
        'supported_best_margin_vs_diagnostic_hub_positive_rate': _rate([(_safe_float(r.get('best_margin_vs_hub')) or -1e9) > 0.0 for r in supported if r.get('best_margin_vs_hub') is not None]),
        'supported_score_bad_rate': _rate([_truthy(r.get('support_exists_but_text_score_bad')) for r in supported]),
        'supported_diagnostic_hub_higher_rate': _rate([_truthy(r.get('support_exists_but_person_higher')) for r in supported]),
        'responsibility_true_support_mass_mean': resp_summary.get('sinkhorn_yprime_true_support_mass_mean'),
        'responsibility_true_support_top1_rate': resp_summary.get('sinkhorn_yprime_true_support_top1_rate'),
        'responsibility_hub_hijack_rate': resp_summary.get('sinkhorn_yprime_hub_hijack_rate'),
        'stage_positive_text_margin_gt_vs_diagnostic_hub_rate': stage_summary_row.get('positive_text_margin_gt_vs_nearest_hub_rate'),
        'stage_text_top1_is_diagnostic_hub_rate': stage_summary_row.get('text_top1_is_hub_rate'),
        'stage_text_gt_rank_full_vocab_mean': stage_summary_row.get('mean_text_gt_rank_full_vocab'),
        'stage_text_gt_rank_full_vocab_median': stage_summary_row.get('median_text_gt_rank_full_vocab'),
    }
    # Per-class ambiguity from yprime rows.
    by_y: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for r in supported:
        y = _safe_int(r.get('yprime_raw_id'))
        if y is not None:
            by_y[int(y)].append(r)
    class_rows = []
    for y, rs in sorted(by_y.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        class_rows.append({
            'yprime_raw_id': int(y),
            'name': next((r.get('yprime_name') for r in rs if r.get('yprime_name')), None),
            'supported_pair_count': int(len(rs)),
            'rank1_yprime_rate': _rate([_safe_int(r.get('best_rank_of_y_among_yprime')) == 1 for r in rs if r.get('best_rank_of_y_among_yprime') is not None]),
            'top20_vocab_rate': _rate([(_safe_int(r.get('best_rank_of_y_among_vocab'), 10**9) or 10**9) <= 20 for r in rs if r.get('best_rank_of_y_among_vocab') is not None]),
            'diagnostic_hub_higher_rate': _rate([_truthy(r.get('support_exists_but_person_higher')) for r in rs]),
            'assignment_hijacked_rate': _rate([not _truthy(r.get('responsibility_true_support_top1')) for r in rs if _truthy(r.get('responsibility_available_for_y'))]),
        })
    return score_summary, class_rows


def _load_responsibility_hubs(resp_path: Path, *, top_hubs: int) -> Tuple[List[Record], Record]:
    top1_counter: Counter[int] = Counter()
    mass_counter: Counter[int] = Counter()
    row_count = 0
    rows_with_nonnull = 0
    null_top1_count = 0
    for row in _iter_jsonl(resp_path):
        row_count += 1
        r_final = row.get('r_final') if isinstance(row.get('r_final'), Mapping) else {}
        if not isinstance(r_final, Mapping) or not r_final:
            continue
        parsed = []
        for k, v in r_final.items():
            rid = _safe_int(k)
            val = _safe_float(v)
            if rid is None or val is None:
                continue
            parsed.append((int(rid), float(val)))
        if not parsed:
            continue
        top_rid, top_val = max(parsed, key=lambda kv: kv[1])
        if int(top_rid) == -1:
            null_top1_count += 1
        nonnull = [(rid, val) for rid, val in parsed if int(rid) != -1]
        if nonnull:
            rows_with_nonnull += 1
            nn_top, _ = max(nonnull, key=lambda kv: kv[1])
            top1_counter[int(nn_top)] += 1
            for rid, val in nonnull:
                mass_counter[int(rid)] += float(val)
    hub_ids = set()
    for rid, _ in top1_counter.most_common(int(top_hubs)):
        hub_ids.add(int(rid))
    for rid, _ in mass_counter.most_common(int(top_hubs)):
        hub_ids.add(int(rid))
    hub_rows = []
    for rid in sorted(hub_ids, key=lambda x: (-(top1_counter[x] + mass_counter[x]), x))[: int(top_hubs)]:
        hub_rows.append({
            'raw_id': int(rid),
            'nonnull_top1_count': int(top1_counter[rid]),
            'nonnull_top1_rate': _rate(int(top1_counter[rid]), rows_with_nonnull),
            'nonnull_mass_sum': float(mass_counter[rid]),
            'data_driven_hub_score': float(top1_counter[rid]) + float(mass_counter[rid]),
        })
    meta = {
        'responsibility_path': str(resp_path),
        'row_count': int(row_count),
        'rows_with_nonnull': int(rows_with_nonnull),
        'null_top1_rate': _rate(null_top1_count, row_count),
        'data_driven_hub_count': int(len(hub_rows)),
    }
    return hub_rows, meta


def _cooccurrence_shortcut_summary(
    *,
    yprime_rows: Sequence[Mapping[str, Any]],
    data_hub_ids: Sequence[int],
    diagnostic_hub_ids: Sequence[int],
) -> Tuple[List[Record], Record]:
    by_clip: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for r in yprime_rows:
        clip = _safe_int(r.get('clip_id'))
        if clip is not None:
            by_clip[int(clip)].append(r)
    data_hubs = {int(x) for x in data_hub_ids}
    diag_hubs = {int(x) for x in diagnostic_hub_ids}
    hub_sets = {
        'data_driven_hubs': data_hubs,
        'diagnostic_hubs': diag_hubs,
    }
    summary_rows: List[Record] = []
    for name, hubs in hub_sets.items():
        if not hubs:
            summary_rows.append({'hub_set': name, 'hub_count': 0, 'status': 'EMPTY'})
            continue
        cooccur_flags = []
        hijack_flags = []
        supported_flags = []
        rows = []
        for clip, rs in by_clip.items():
            yset = {_safe_int(r.get('yprime_raw_id')) for r in rs}
            yset = {int(x) for x in yset if x is not None}
            clip_has_hub = bool(yset & hubs)
            for r in rs:
                y = _safe_int(r.get('yprime_raw_id'))
                if y is None or int(y) in hubs:
                    continue
                best_gt = _safe_int(r.get('responsibility_best_mass_gt_raw_id'))
                # Oracle evaluation of whether the row selected a trajectory whose GT is a hub class.
                hijacked_by_hub_gt = bool(best_gt is not None and int(best_gt) in hubs and not _truthy(r.get('responsibility_true_support_top1')))
                rec = {
                    'clip_has_hub_in_yprime': bool(clip_has_hub),
                    'hijacked_by_hub_gt': bool(hijacked_by_hub_gt),
                    'has_trajectory_support': _truthy(r.get('has_trajectory_support')),
                }
                rows.append(rec)
                cooccur_flags.append(bool(clip_has_hub))
                hijack_flags.append(bool(hijacked_by_hub_gt))
                supported_flags.append(_truthy(r.get('has_trajectory_support')))
        co = [r for r in rows if r['clip_has_hub_in_yprime']]
        no = [r for r in rows if not r['clip_has_hub_in_yprime']]
        hij_co = _rate([r['hijacked_by_hub_gt'] for r in co])
        hij_no = _rate([r['hijacked_by_hub_gt'] for r in no])
        odds_ratio = None
        if hij_co is not None and hij_no is not None:
            # Smoothed odds ratio.
            a = sum(1 for r in co if r['hijacked_by_hub_gt']) + 0.5
            b = sum(1 for r in co if not r['hijacked_by_hub_gt']) + 0.5
            c = sum(1 for r in no if r['hijacked_by_hub_gt']) + 0.5
            d = sum(1 for r in no if not r['hijacked_by_hub_gt']) + 0.5
            odds_ratio = float((a / b) / (c / d))
        summary_rows.append({
            'hub_set': name,
            'hub_count': int(len(hubs)),
            'evaluated_yprime_pair_count': int(len(rows)),
            'cooccur_pair_count': int(len(co)),
            'noncooccur_pair_count': int(len(no)),
            'hub_hijack_rate_given_cooccur': hij_co,
            'hub_hijack_rate_without_cooccur': hij_no,
            'hub_hijack_odds_ratio_cooccur_vs_noncooccur': odds_ratio,
            'overall_hub_hijack_rate': _rate(hijack_flags),
            'support_rate_in_evaluated_pairs': _rate(supported_flags),
        })
    meta = {
        'clip_count': int(len(by_clip)),
        'note': 'Hub sets are audit-only. Data-driven hubs are derived from responsibility records; diagnostic hubs are optional CLI inputs.',
    }
    return summary_rows, meta


def _observational_equivalence(
    *,
    text_rows: Sequence[Mapping[str, Any]],
    carrier_rows: Sequence[Mapping[str, Any]],
    neighbor_k: int,
    max_rows: int,
    distance_eps_quantile: float,
) -> Tuple[Record, List[Record]]:
    if np is None:
        return {'status': 'SKIPPED', 'reason': 'numpy_unavailable'}, []
    carrier_by_tid = {str(r.get('trajectory_id')): r for r in carrier_rows if r.get('trajectory_id') is not None}
    features: List[List[float]] = []
    metas: List[Record] = []
    feature_names = [
        'text_score_gt',
        'nearest_hub_text_score',
        'text_margin_gt_minus_nearest_hub',
        'text_gt_rank_full_vocab_log',
        'carrier_cos_to_gt_centroid',
        'nearest_non_gt_centroid_cos',
        'carrier_gt_vs_nearest_non_gt_margin',
        'carrier_gt_vs_nearest_hub_margin',
    ]
    for r in text_rows:
        tid = str(r.get('trajectory_id'))
        c = carrier_by_tid.get(tid, {})
        vals: Dict[str, Optional[float]] = {
            'text_score_gt': _safe_float(r.get('text_score_gt')),
            'nearest_hub_text_score': _safe_float(r.get('nearest_hub_text_score')),
            'text_margin_gt_minus_nearest_hub': _safe_float(r.get('text_margin_gt_minus_nearest_hub')),
            'text_gt_rank_full_vocab_log': math.log1p(float(_safe_int(r.get('text_gt_rank_full_vocab'), 99999) or 99999)),
            'carrier_cos_to_gt_centroid': _safe_float(c.get('carrier_cos_to_gt_centroid')),
            'nearest_non_gt_centroid_cos': _safe_float(c.get('nearest_non_gt_centroid_cos')),
            'carrier_gt_vs_nearest_non_gt_margin': _safe_float(c.get('carrier_gt_vs_nearest_non_gt_margin')),
            'carrier_gt_vs_nearest_hub_margin': _safe_float(c.get('carrier_gt_vs_nearest_hub_margin')),
        }
        if sum(1 for v in vals.values() if v is not None) < 4:
            continue
        feat = [float(vals.get(name) if vals.get(name) is not None else 0.0) for name in feature_names]
        features.append(feat)
        metas.append({
            'trajectory_id': tid,
            'clip_id': r.get('clip_id'),
            'gt_raw_id': _safe_int(r.get('gt_raw_id')),
            'gt_name': r.get('gt_name'),
            'text_top1_raw_id': _safe_int(r.get('text_top1_raw_id')),
            'text_top1_is_hub': _truthy(r.get('text_top1_is_hub')),
            'text_top1_is_gt': _truthy(r.get('text_top1_is_gt')),
        })
        if max_rows and len(features) >= int(max_rows):
            break
    n = len(features)
    if n < 5:
        return {'status': 'SKIPPED', 'reason': 'not_enough_rows', 'row_count': int(n)}, []
    X = np.asarray(features, dtype=np.float64)
    # z-score, with zero-variance guard. This is an empirical signature of current score/geometry evidence.
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    X = (X - mu) / sd
    labels = np.asarray([int(m['gt_raw_id']) if m.get('gt_raw_id') is not None else -999999 for m in metas], dtype=np.int64)
    clips = np.asarray([str(m.get('clip_id')) for m in metas], dtype=object)
    k = max(1, int(neighbor_k))
    neighbor_rows: List[Record] = []
    top1_same = []
    topk_same_rates = []
    top1_leave_clip_same = []
    distances_top1 = []
    # n is normally ~4k; full distance is acceptable.
    for i in range(n):
        diff = X - X[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        dist[i] = np.inf
        order = np.argsort(dist, kind='stable')
        order = [int(j) for j in order if np.isfinite(dist[int(j)])]
        if not order:
            continue
        order_lco = [j for j in order if str(clips[j]) != str(clips[i])]
        top = order[: min(k, len(order))]
        same_arr = [bool(labels[j] == labels[i]) for j in top]
        top1_same.append(same_arr[0])
        topk_same_rates.append(float(sum(1 for x in same_arr if x) / len(same_arr)))
        distances_top1.append(float(dist[top[0]]))
        lco_same = None
        if order_lco:
            lco_same = bool(labels[order_lco[0]] == labels[i])
            top1_leave_clip_same.append(lco_same)
        if len(neighbor_rows) < 1000:
            neighbor_rows.append({
                'trajectory_id': metas[i].get('trajectory_id'),
                'clip_id': metas[i].get('clip_id'),
                'gt_raw_id': int(labels[i]),
                'gt_name': metas[i].get('gt_name'),
                'nearest_trajectory_id': metas[top[0]].get('trajectory_id'),
                'nearest_clip_id': metas[top[0]].get('clip_id'),
                'nearest_gt_raw_id': int(labels[top[0]]),
                'nearest_gt_name': metas[top[0]].get('gt_name'),
                'nearest_distance': float(dist[top[0]]),
                'nearest_same_gt': bool(labels[top[0]] == labels[i]),
                'topk_same_gt_rate': float(sum(1 for x in same_arr if x) / len(same_arr)),
                'leave_clip_nearest_same_gt': lco_same,
            })
    eps = None
    near_diff_rate = None
    if distances_top1:
        eps = float(np.quantile(np.asarray(distances_top1, dtype=np.float64), float(distance_eps_quantile)))
        near_indices = [idx for idx, row in enumerate(neighbor_rows) if _safe_float(row.get('nearest_distance'), 1e9) <= eps]
        if near_indices:
            near_diff_rate = _rate([not bool(neighbor_rows[idx].get('nearest_same_gt')) for idx in near_indices])
    local_majority_error_proxy = None
    if topk_same_rates:
        # If same-GT purity is low in local balls, even a nearest-neighbor style resolver has a high local ambiguity floor.
        local_majority_error_proxy = float(1.0 - _mean(topk_same_rates)) if _mean(topk_same_rates) is not None else None
    summary = {
        'status': 'PASS',
        'row_count': int(n),
        'feature_names': feature_names,
        'neighbor_k': int(k),
        'top1_neighbor_same_gt_rate': _rate(top1_same),
        'top1_neighbor_diff_gt_rate': None if _rate(top1_same) is None else float(1.0 - float(_rate(top1_same))),
        'topk_same_gt_purity_mean': _mean(topk_same_rates),
        'topk_local_majority_error_proxy': local_majority_error_proxy,
        'leave_one_clip_top1_same_gt_rate': _rate(top1_leave_clip_same),
        'near_distance_eps_quantile': float(distance_eps_quantile),
        'near_distance_eps': eps,
        'near_duplicate_diff_gt_rate': near_diff_rate,
        'interpretation': 'Low same-GT neighbor rates mean current score/geometry signatures admit visually/statistically similar rows with different GT assignments.',
    }
    return summary, neighbor_rows


def _derive_verdict(summary: Mapping[str, Any]) -> Tuple[str, List[str]]:
    signals: List[str] = []
    oracle = summary.get('oracle_support_upper_bound') if isinstance(summary.get('oracle_support_upper_bound'), Mapping) else {}
    score = summary.get('score_ambiguity') if isinstance(summary.get('score_ambiguity'), Mapping) else {}
    obs = summary.get('observational_equivalence') if isinstance(summary.get('observational_equivalence'), Mapping) else {}
    co_rows = summary.get('cooccurrence_shortcut_rows') if isinstance(summary.get('cooccurrence_shortcut_rows'), list) else []
    sr = _safe_float(oracle.get('yprime_support_rate'))
    if sr is not None and sr < 0.60:
        signals.append('low_proposal_support_upper_bound')
    rank1 = _safe_float(score.get('supported_rank1_full_vocab_rate'))
    if rank1 is not None and rank1 < 0.30:
        signals.append('true_class_not_top_scored_for_many_supported_pairs')
    top1_hub = _safe_float(score.get('stage_text_top1_is_diagnostic_hub_rate'))
    if top1_hub is not None and top1_hub > 0.30:
        signals.append('diagnostic_hub_top1_high')
    same = _safe_float(obs.get('top1_neighbor_same_gt_rate'))
    if same is not None and same < 0.40:
        signals.append('observational_neighbors_often_different_gt')
    for r in co_rows:
        orv = _safe_float(r.get('hub_hijack_odds_ratio_cooccur_vs_noncooccur'))
        if orv is not None and orv > 1.50:
            signals.append(f"cooccurrence_hub_shortcut_{r.get('hub_set')}")
    if len(signals) >= 3:
        verdict = 'strong_empirical_nonidentifiability_evidence'
    elif len(signals) >= 2:
        verdict = 'moderate_empirical_nonidentifiability_evidence'
    else:
        verdict = 'mixed_or_insufficient_nonidentifiability_evidence'
    return verdict, signals


def run(args: argparse.Namespace) -> Record:
    run_root = Path(args.run_root)
    dataset = str(args.dataset_name)
    stage = str(args.stage)
    out = Path(args.output_dir) if args.output_dir else run_root / 'analysis' / 'assignment_nonidentifiability' / dataset / stage
    out.mkdir(parents=True, exist_ok=True)

    yprime_dir = run_root / 'analysis' / 'yprime_support_coverage' / dataset / stage
    text_dir = run_root / 'analysis' / 'text_projector_hubness' / dataset / stage
    carrier_dir = run_root / 'analysis' / 'hub_carrier_separability' / dataset / stage
    resp_dir = run_root / 'analysis' / 'support_null_responsibility' / dataset / stage
    resp_path = run_root / 'train' / stage / 'responsibility_records.jsonl'

    yprime_summary = _read_json(yprime_dir / 'summary.json')
    yprime_rows = list(_iter_jsonl(yprime_dir / 'clip_yprime_support_rows.jsonl'))
    class_support_rows = _read_csv(yprime_dir / 'class_support_summary.csv')
    text_summary_rows = _read_csv(text_dir / 'stage_comparison_summary.csv')
    text_stage_summary = _flatten_stage_row(text_summary_rows)
    text_rows = list(_iter_jsonl(text_dir / 'row_text_margin_diagnostics.jsonl', max_rows=int(args.max_rows or 0)))
    carrier_rows = list(_iter_jsonl(carrier_dir / 'row_geometry_diagnostics.jsonl', max_rows=int(args.max_rows or 0)))
    resp_summary = _read_json(resp_dir / 'summary.json')

    oracle_upper, worst_support_rows = _oracle_support_upper_bound(
        support_summary=yprime_summary,
        class_support_rows=class_support_rows,
        iou_threshold_label=str(args.iou_threshold_label),
    )
    score_summary, score_class_rows = _score_ambiguity_summary(
        yprime_rows=yprime_rows,
        text_rows=text_rows,
        stage_summary_row=text_stage_summary,
        resp_summary=resp_summary,
    )
    hub_rows, hub_meta = _load_responsibility_hubs(resp_path, top_hubs=int(args.top_hubs))
    data_hub_ids = [int(r['raw_id']) for r in hub_rows]
    diagnostic_hub_ids = _parse_int_tuple(str(args.diagnostic_hub_raw_ids))
    co_rows, co_meta = _cooccurrence_shortcut_summary(
        yprime_rows=yprime_rows,
        data_hub_ids=data_hub_ids,
        diagnostic_hub_ids=diagnostic_hub_ids,
    )
    obs_summary, obs_neighbor_rows = _observational_equivalence(
        text_rows=text_rows,
        carrier_rows=carrier_rows,
        neighbor_k=int(args.neighbor_k),
        max_rows=int(args.max_rows or 0),
        distance_eps_quantile=float(args.distance_eps_quantile),
    )

    _write_csv(out / 'oracle_support_upper_bound.csv', [oracle_upper])
    _write_csv(out / 'class_support_lowest50.csv', worst_support_rows)
    _write_csv(out / 'score_ambiguity_summary.csv', [score_summary])
    _write_csv(out / 'score_ambiguity_by_class.csv', score_class_rows[:200])
    _write_json(out / 'observational_equivalence_summary.json', obs_summary)
    _write_csv(out / 'observational_equivalence_neighbors.csv', obs_neighbor_rows)
    _write_csv(out / 'data_driven_hub_summary.csv', hub_rows)
    _write_csv(out / 'cooccurrence_shortcut_summary.csv', co_rows)

    summary: Record = {
        'status': 'PASS',
        'audit_name': 'assignment_nonidentifiability',
        'run_root': str(run_root),
        'dataset_name': dataset,
        'stage': stage,
        'input_artifacts': {
            'yprime_support_summary': str(yprime_dir / 'summary.json'),
            'yprime_support_rows': str(yprime_dir / 'clip_yprime_support_rows.jsonl'),
            'text_hubness_stage_summary': str(text_dir / 'stage_comparison_summary.csv'),
            'text_row_diagnostics': str(text_dir / 'row_text_margin_diagnostics.jsonl'),
            'carrier_row_geometry': str(carrier_dir / 'row_geometry_diagnostics.jsonl'),
            'responsibility_records': str(resp_path),
            'support_null_responsibility_summary': str(resp_dir / 'summary.json'),
        },
        'artifact_existence': {
            'yprime_support_summary': bool((yprime_dir / 'summary.json').is_file()),
            'yprime_support_rows': bool((yprime_dir / 'clip_yprime_support_rows.jsonl').is_file()),
            'text_stage_summary': bool((text_dir / 'stage_comparison_summary.csv').is_file()),
            'text_row_diagnostics': bool((text_dir / 'row_text_margin_diagnostics.jsonl').is_file()),
            'carrier_row_geometry': bool((carrier_dir / 'row_geometry_diagnostics.jsonl').is_file()),
            'responsibility_records': bool(resp_path.is_file()),
            'support_null_responsibility_summary': bool((resp_dir / 'summary.json').is_file()),
        },
        'oracle_support_upper_bound': oracle_upper,
        'score_ambiguity': score_summary,
        'observational_equivalence': obs_summary,
        'data_driven_hub_meta': hub_meta,
        'data_driven_top_hubs': hub_rows[: int(args.top_hubs)],
        'cooccurrence_shortcut_rows': co_rows,
        'cooccurrence_shortcut_meta': co_meta,
        'limitations': [
            'This is an empirical non-identifiability audit, not a mathematical proof by itself.',
            'Observational-equivalence features are posthoc score/geometry signatures; strict proof still requires a formal two-world construction.',
            'Diagnostic hub raw ids are audit-only and must not be used as training priors.',
        ],
        'outputs': {
            'summary': str(out / 'summary.json'),
            'takeover': str(out / 'NONIDENTIFIABILITY_TAKEOVER.md'),
            'oracle_support_upper_bound': str(out / 'oracle_support_upper_bound.csv'),
            'class_support_lowest50': str(out / 'class_support_lowest50.csv'),
            'score_ambiguity_summary': str(out / 'score_ambiguity_summary.csv'),
            'score_ambiguity_by_class': str(out / 'score_ambiguity_by_class.csv'),
            'observational_equivalence_summary': str(out / 'observational_equivalence_summary.json'),
            'observational_equivalence_neighbors': str(out / 'observational_equivalence_neighbors.csv'),
            'data_driven_hub_summary': str(out / 'data_driven_hub_summary.csv'),
            'cooccurrence_shortcut_summary': str(out / 'cooccurrence_shortcut_summary.csv'),
        },
    }
    verdict, signals = _derive_verdict(summary)
    summary['verdict'] = verdict
    summary['signals'] = signals
    _write_json(out / 'summary.json', summary)

    md: List[str] = []
    md.append('# Assignment Non-identifiability Audit')
    md.append('')
    md.append(f"- status: `PASS`")
    md.append(f"- dataset: `{dataset}`")
    md.append(f"- stage: `{stage}`")
    md.append(f"- verdict: `{verdict}`")
    md.append(f"- signals: `{', '.join(signals) if signals else 'none'}`")
    md.append('')
    md.append('## 1. Oracle proposal support upper bound')
    for k in ['clip_yprime_pair_count', 'yprime_support_rate', 'yprime_zero_support_rate', 'clip_all_yprime_supported_rate']:
        md.append(f"- {k}: `{oracle_upper.get(k)}`")
    md.append('')
    md.append('## 2. Score ambiguity')
    for k in [
        'supported_rank1_among_yprime_rate',
        'supported_rank1_full_vocab_rate',
        'supported_top20_full_vocab_rate',
        'supported_diagnostic_hub_higher_rate',
        'responsibility_true_support_mass_mean',
        'responsibility_true_support_top1_rate',
        'responsibility_hub_hijack_rate',
        'stage_positive_text_margin_gt_vs_diagnostic_hub_rate',
        'stage_text_top1_is_diagnostic_hub_rate',
    ]:
        md.append(f"- {k}: `{score_summary.get(k)}`")
    md.append('')
    md.append('## 3. Observational equivalence proxy')
    for k in [
        'row_count',
        'top1_neighbor_same_gt_rate',
        'top1_neighbor_diff_gt_rate',
        'topk_same_gt_purity_mean',
        'leave_one_clip_top1_same_gt_rate',
        'near_duplicate_diff_gt_rate',
    ]:
        md.append(f"- {k}: `{obs_summary.get(k)}`")
    md.append('')
    md.append('## 4. Co-occurrence / data-driven hub shortcut')
    for row in co_rows:
        md.append(
            f"- {row.get('hub_set')}: hub_count=`{row.get('hub_count')}`, "
            f"hijack_given_cooccur=`{row.get('hub_hijack_rate_given_cooccur')}`, "
            f"hijack_without_cooccur=`{row.get('hub_hijack_rate_without_cooccur')}`, "
            f"odds_ratio=`{row.get('hub_hijack_odds_ratio_cooccur_vs_noncooccur')}`"
        )
    md.append('')
    md.append('## Reading guide')
    md.append('- Low proposal support rate is a hard upper bound for trajectory-only assignment coverage.')
    md.append('- Low true-class score/rank and high hub top1 indicate the score function itself does not identify the correct class reliably.')
    md.append('- Low same-GT rate among nearest observational signatures is empirical evidence that similar observed evidence maps to different GT assignments.')
    md.append('- Data-driven hub/co-occurrence effects indicate shortcut explanations, not merely random errors.')
    md.append('- Diagnostic hub ids are audit-only; do not use them as training priors.')
    md.append('')
    md.append('## Outputs')
    for k, v in summary['outputs'].items():
        md.append(f"- {k}: `{v}`")
    (out / 'NONIDENTIFIABILITY_TAKEOVER.md').write_text('\n'.join(md) + '\n', encoding='utf-8')

    print(json.dumps({
        'status': 'PASS',
        'verdict': verdict,
        'signals': signals,
        'output_dir': str(out),
        'yprime_support_rate': oracle_upper.get('yprime_support_rate'),
        'stage_text_top1_is_diagnostic_hub_rate': score_summary.get('stage_text_top1_is_diagnostic_hub_rate'),
        'top1_neighbor_same_gt_rate': obs_summary.get('top1_neighbor_same_gt_rate'),
    }, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='WSOVVIS assignment non-identifiability posthoc audit')
    parser.add_argument('--run_root', required=True)
    parser.add_argument('--dataset_name', default='lvvis_train_base')
    parser.add_argument('--stage', default='prealign')
    parser.add_argument('--trajectory_source_branch', default='mainline', help='reserved for compatibility; this posthoc audit reads existing artifacts')
    parser.add_argument('--diagnostic_hub_raw_ids', default='', help='optional audit-only diagnostic hubs, e.g. 773; not used as a training prior')
    parser.add_argument('--top_hubs', type=int, default=20)
    parser.add_argument('--neighbor_k', type=int, default=10)
    parser.add_argument('--max_rows', type=int, default=0, help='limit rows for observational equivalence; 0 means all available')
    parser.add_argument('--distance_eps_quantile', type=float, default=0.10)
    parser.add_argument('--iou_threshold_label', default='current_sidecar_threshold')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
