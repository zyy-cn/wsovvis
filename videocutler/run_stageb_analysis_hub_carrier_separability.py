#!/usr/bin/env python3
"""
WSOVVIS hub carrier separability audit.

Read-only analysis script. It reuses the existing G8/minimal-split materialization
and carrier-loading path, then tests whether co-occurrence hub classes and the GT
classes they suppress are already inseparable in trajectory-carrier space.

Primary question:
  Are hub-related failures caused by carrier geometry collapse, or by later
  clip-level source leakage / TopK slot allocation / E-step attribution collapse?

Inputs reused from the current tree:
  - selected_for_infer / runtime asset resolution under --run_root
  - GT sidecars already used by g8_minimal_split_audit
  - training stage rows: train/<stage>/{proxy_records|responsibility_records}.jsonl
  - optional extra_mining_recall_diagnosis formal rows for failure buckets/source fields

Outputs:
  <run_root>/analysis/hub_carrier_separability/<dataset>/<stage>/summary.json
  <run_root>/analysis/hub_carrier_separability/<dataset>/<stage>/row_geometry_diagnostics.jsonl
  <run_root>/analysis/hub_carrier_separability/<dataset>/<stage>/bucket_summary.json
  <run_root>/analysis/hub_carrier_separability/<dataset>/<stage>/class_centroid_summary.csv
  <run_root>/analysis/hub_carrier_separability/<dataset>/<stage>/hub_pair_margin_summary.csv
  <run_root>/analysis/hub_carrier_separability/<dataset>/<stage>/pairwise_class_cosine.csv
  <run_root>/analysis/hub_carrier_separability/<dataset>/<stage>/knn_purity_summary.json
  <run_root>/analysis/hub_carrier_separability/<dataset>/<stage>/source_leakage_counterfactual_summary.json
  <run_root>/analysis/hub_carrier_separability/<dataset>/<stage>/HUB_CARRIER_SEPARABILITY_TAKEOVER.md

This script does not train, infer, rewrite checkpoints, or generate sidecars by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_carrier_evidence  # noqa: E402
from videocutler.ext_stageb_ovvis.audit.g8_minimal_split_audit import (  # noqa: E402
    MinimalSplitAuditConfig,
    _canonical_sidecar_gt_raw_id,
    _load_proxy_observed_lookup,
    _materialize_shared_inputs,
    _split_order_for_dataset,
    _stage_row_source_path,
)
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import _all_gt_split_label  # noqa: E402


Record = Dict[str, Any]
DEFAULT_KNN_KS: Tuple[int, ...] = (1, 5, 10)
DEFAULT_PERSON_RAW_ID = 773


@dataclass(frozen=True)
class AuditConfig:
    run_root: Path
    dataset_name: str
    stage: str
    trajectory_source_branch: str
    output_dir: Optional[Path]
    diagnosis_dir: Optional[Path]
    splits: Tuple[str, ...]
    hub_raw_ids: Tuple[int, ...]
    min_class_count: int
    knn_ks: Tuple[int, ...]
    max_rows: int
    max_pairwise_classes: int
    top_examples: int
    source_leakage_quantile: float
    all_gt_generate_sidecars_if_missing: bool
    show_progress: bool


def _parse_int_tuple(value: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in str(value).replace(';', ',').split(','):
        part = part.strip()
        if part:
            out.append(int(part))
    return tuple(sorted(set(out)))


def _parse_str_tuple(value: str) -> Tuple[str, ...]:
    out: List[str] = []
    for part in str(value).replace(';', ',').split(','):
        part = part.strip()
        if part:
            out.append(part)
    return tuple(out)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if s in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'expected boolean, got {value!r}')


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        v = float(value)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.asarray(vals, dtype=np.float64).mean())


def _median(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.quantile(np.asarray(vals, dtype=np.float64), float(q)))


def _rate(values: Sequence[bool]) -> Optional[float]:
    if not values:
        return None
    return float(np.asarray([1.0 if bool(v) else 0.0 for v in values], dtype=np.float64).mean())


def _l2_normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def _iter_jsonl(path: Path) -> Iterable[Record]:
    if not path.is_file():
        return
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield dict(row)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=False) + '\n', encoding='utf-8')


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=False) + '\n')


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = [dict(r) for r in rows]
    if not rows_list:
        path.write_text('', encoding='utf-8')
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows_list:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def _default_output_dir(run_root: Path, dataset_name: str, stage: str) -> Path:
    return run_root / 'analysis' / 'hub_carrier_separability' / dataset_name / stage


def _default_diagnosis_dir(run_root: Path, dataset_name: str, stage: str) -> Path:
    return run_root / 'analysis' / 'extra_mining_recall_diagnosis' / dataset_name / stage


def _class_name_map(prepared: Mapping[str, Any]) -> Dict[int, str]:
    gt_payload = dict(prepared.get('gt_payload', {}))
    out: Dict[int, str] = {}
    for cat in gt_payload.get('categories', []) or []:
        if not isinstance(cat, Mapping):
            continue
        rid = _safe_int(cat.get('id'))
        if rid is None:
            continue
        name = cat.get('name') or cat.get('category_name') or cat.get('synset')
        out[int(rid)] = str(name) if name is not None else str(rid)
    return out


def _name(class_names: Mapping[int, str], raw_id: Any) -> Optional[str]:
    rid = _safe_int(raw_id)
    if rid is None:
        return None
    return class_names.get(int(rid))


def _load_diagnosis_rows(diag_dir: Path) -> Tuple[Dict[str, Record], Dict[str, Any]]:
    candidates = [
        diag_dir / 'formal_aligned_row_diagnostics.jsonl',
        diag_dir / 'row_diagnostics.jsonl',
        diag_dir / 'formal_aligned_rows.jsonl',
    ]
    used: Optional[Path] = None
    by_tid: Dict[str, Record] = {}
    total = 0
    for path in candidates:
        if not path.is_file():
            continue
        used = path
        for row in _iter_jsonl(path):
            total += 1
            tid = str(row.get('trajectory_id') or row.get('traj_id') or row.get('row_trajectory_id') or '').strip()
            if tid:
                by_tid[tid] = dict(row)
        break
    return by_tid, {
        'path': str(used) if used is not None else None,
        'exists': used is not None,
        'record_count': int(total),
        'by_trajectory_id_count': int(len(by_tid)),
    }


def _first(row: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return default


def _as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (int, float, str)):
        iv = _safe_int(value)
        return [] if iv is None else [int(iv)]
    out: List[int] = []
    seen: set[int] = set()
    try:
        for item in value:
            iv = _safe_int(item)
            if iv is None or iv in seen:
                continue
            seen.add(int(iv))
            out.append(int(iv))
    except Exception:
        return []
    return out


def _active_raw_contains(diag: Mapping[str, Any], gt_raw_id: int) -> Optional[bool]:
    if 'active_raw_contains' in diag:
        return bool(diag.get('active_raw_contains'))
    if 'gt_in_active_extra' in diag:
        return bool(diag.get('gt_in_active_extra'))
    extra = _as_int_list(_first(diag, ['active_extra_raw_ids', 'candidate_ids_extra', 'candidate_ids_extra_raw'], None))
    known = set(_as_int_list(_first(diag, ['candidate_ids_known', 'candidate_ids_yprime', 'observed_raw_ids'], None)))
    if extra:
        return bool(int(gt_raw_id) in (set(extra) - known))
    return None


def _winner_domain(diag: Mapping[str, Any]) -> str:
    value = _first(diag, ['final_winner_domain', 'winner_domain', 'pred_winner_domain', 'r_winner_domain'], None)
    if value is None:
        return 'unknown_or_missing'
    return str(value)


def _suppressor_raw_id(diag: Mapping[str, Any]) -> Optional[int]:
    return _safe_int(_first(
        diag,
        [
            'gt_suppressor_raw_id',
            'suppressor_raw_id',
            'top_suppressor_raw_id',
            'best_suppressor_raw_id',
            'wrong_extra_winner_raw_id',
            'final_winner_raw_id',
            'winner_raw_id',
            'r_winner_raw_id',
        ],
        None,
    ))


def _source_trajectory_id(diag: Mapping[str, Any]) -> Optional[str]:
    val = _first(
        diag,
        [
            'hub_clip_argmax_trajectory_id',
            'suppressor_argmax_trajectory_id',
            'selected_extra_source_trajectory_id',
            'winner_source_trajectory_id',
            'clip_max_source_trajectory_id',
            'selected_argmax_trajectory_id',
        ],
        None,
    )
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _row_bucket(diag: Mapping[str, Any], *, active_raw: Optional[bool], final_top1_is_gt: Optional[bool]) -> str:
    fb = str(_first(diag, ['failure_bucket', 'failure_mode', 'primary_failure_bucket'], '') or '')
    if fb and fb.lower() not in {'', 'none', 'nan'}:
        return fb
    if active_raw is False:
        return 'gt_not_active_extra'
    if final_top1_is_gt is True:
        return 'active_gt_win'
    wd = _winner_domain(diag)
    if active_raw is True and wd in {'Yprime', 'known', 'observed'}:
        return 'active_yprime_win'
    if active_raw is True and 'extra' in wd.lower():
        return 'active_wrong_extra_win'
    if active_raw is True:
        return 'active_non_gt_win'
    return 'bucket_unknown'


def _materialize_rows_and_carriers(config: AuditConfig) -> Tuple[List[Record], np.ndarray, Dict[str, Any]]:
    ms_config = MinimalSplitAuditConfig(
        dataset_name=config.dataset_name,
        output_root=config.run_root,
        stage=config.stage,
        device=torch.device('cpu'),
        trajectory_source_branch=config.trajectory_source_branch,
        all_gt_generate_sidecars_if_missing=bool(config.all_gt_generate_sidecars_if_missing),
        heartbeat_every_rows=512,
        batch_size_rows=128,
        candidate_chunk_size=0,
    )
    prepared = dict(_materialize_shared_inputs(ms_config))
    class_names = _class_name_map(prepared)
    gt_sidecar_lookup = dict(prepared.get('gt_sidecar_lookup', {}))
    asset_roots = prepared['asset_roots']
    sample_by_tid: Dict[str, Mapping[str, Any]] = {
        str(sample.get('trajectory_id', sample.get('trajectory_record', {}).get('trajectory_id', ''))).strip(): sample
        for sample in prepared.get('samples', [])
        if str(sample.get('trajectory_id', sample.get('trajectory_record', {}).get('trajectory_id', ''))).strip()
    }

    rows_source_meta: Dict[str, Any] = {}
    candidate_tids: List[str] = []
    stage_path = _stage_row_source_path(config.run_root, config.dataset_name, config.stage)
    if stage_path is not None and stage_path.is_file():
        for row in _iter_jsonl(stage_path):
            tid = str(row.get('trajectory_id') or '').strip()
            if tid and tid in sample_by_tid:
                candidate_tids.append(tid)
        rows_source_meta = {'source': 'train_stage_rows', 'path': str(stage_path), 'count': int(len(candidate_tids))}
    else:
        candidate_tids = sorted(sample_by_tid.keys())
        rows_source_meta = {'source': 'materialized_samples', 'path': str(stage_path) if stage_path is not None else None, 'count': int(len(candidate_tids))}

    if config.max_rows and int(config.max_rows) > 0:
        candidate_tids = candidate_tids[: int(config.max_rows)]

    split_order = _split_order_for_dataset(config.dataset_name)
    requested_splits = set(split_order if not config.splits or config.splits == ('all',) else config.splits)

    rows: List[Record] = []
    vectors: List[np.ndarray] = []
    missing_carrier = 0
    skipped_no_gt = 0
    skipped_split = 0
    for idx, tid in enumerate(candidate_tids, start=1):
        sample = sample_by_tid.get(tid)
        if sample is None:
            continue
        sidecar = dict(gt_sidecar_lookup.get(tid, {}))
        gt_raw_id = _canonical_sidecar_gt_raw_id(sidecar)
        if not bool(sidecar.get('audit_usable', False)) or gt_raw_id is None:
            skipped_no_gt += 1
            continue
        split = _all_gt_split_label(config.dataset_name, int(gt_raw_id), prepared['base_vocab_ids'])
        if split not in requested_splits:
            skipped_split += 1
            continue
        try:
            vec = load_carrier_evidence(
                sample,
                output_root=asset_roots.asset_root,
                dataset_name=config.dataset_name,
                trajectory_source_branch=config.trajectory_source_branch,
            )
        except Exception:
            missing_carrier += 1
            continue
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.isfinite(arr).all():
            missing_carrier += 1
            continue
        arr = _l2_normalize_np(arr)
        clip_id = sample.get('clip_id') or sample.get('trajectory_record', {}).get('clip_id')
        rows.append({
            'row_index': int(len(rows)),
            'trajectory_id': str(tid),
            'clip_id': str(clip_id) if clip_id is not None else None,
            'gt_raw_id': int(gt_raw_id),
            'gt_name': _name(class_names, int(gt_raw_id)),
            'split': str(split),
        })
        vectors.append(arr.astype(np.float32))
        if config.show_progress and idx % 1024 == 0:
            print(f'[hub-carrier-separability] materialized {len(rows)} usable rows / scanned {idx}', flush=True)

    if not rows:
        raise RuntimeError(
            f'HUB_CARRIER_SEPARABILITY_EMPTY: dataset={config.dataset_name} stage={config.stage} '
            f'splits={sorted(requested_splits)} run_root={config.run_root}'
        )
    metadata = {
        'prepared_sample_count': int(len(sample_by_tid)),
        'candidate_tid_count': int(len(candidate_tids)),
        'usable_row_count': int(len(rows)),
        'skipped_no_gt_count': int(skipped_no_gt),
        'skipped_split_count': int(skipped_split),
        'missing_carrier_count': int(missing_carrier),
        'rows_source': rows_source_meta,
        'class_names': {str(k): v for k, v in class_names.items()},
        'asset_root': str(asset_roots.asset_root),
    }
    return rows, np.stack(vectors, axis=0).astype(np.float32), metadata


def _attach_diagnosis(rows: List[Record], diag_by_tid: Mapping[str, Mapping[str, Any]], hub_set: set[int]) -> Dict[str, Any]:
    joined = 0
    for row in rows:
        tid = str(row['trajectory_id'])
        diag = dict(diag_by_tid.get(tid, {}))
        if diag:
            joined += 1
        gt = int(row['gt_raw_id'])
        active = _active_raw_contains(diag, gt) if diag else None
        top1 = None
        if 'final_top1_is_gt' in diag:
            top1 = bool(diag.get('final_top1_is_gt'))
        elif 'r_final_gt_winner' in diag:
            top1 = bool(diag.get('r_final_gt_winner'))
        suppressor = _suppressor_raw_id(diag) if diag else None
        source_tid = _source_trajectory_id(diag) if diag else None
        row.update({
            'diagnosis_joined': bool(diag),
            'active_raw_contains': active,
            'final_top1_is_gt': top1,
            'winner_domain': _winner_domain(diag) if diag else 'unknown_or_missing',
            'failure_bucket': _row_bucket(diag, active_raw=active, final_top1_is_gt=top1),
            'suppressor_raw_id': int(suppressor) if suppressor is not None else None,
            'suppressor_name': None,
            'suppressor_is_config_hub': bool(suppressor is not None and int(suppressor) in hub_set),
            'source_trajectory_id': source_tid,
            'source_is_other_trajectory': bool(source_tid is not None and str(source_tid) != str(tid)),
        })
    return {'joined_row_count': int(joined), 'join_rate': float(joined / max(len(rows), 1))}


def _class_stats(rows: Sequence[Mapping[str, Any]], vectors: np.ndarray, class_names: Mapping[int, str]) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_class[int(row['gt_raw_id'])].append(int(i))
    stats: Dict[int, Dict[str, Any]] = {}
    out_rows: List[Dict[str, Any]] = []
    for raw_id, indices in sorted(by_class.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        X = np.asarray(vectors[indices], dtype=np.float32)
        n = int(X.shape[0])
        mean_vec = X.mean(axis=0)
        centroid = _l2_normalize_np(mean_vec)
        if n >= 2:
            sum_vec = X.sum(axis=0)
            within = float((float(np.dot(sum_vec, sum_vec)) - n) / max(n * (n - 1), 1))
        else:
            within = None
        stats[int(raw_id)] = {
            'raw_id': int(raw_id),
            'name': class_names.get(int(raw_id)),
            'count': n,
            'indices': indices,
            'mean_vec': mean_vec.astype(np.float32),
            'centroid': centroid.astype(np.float32),
            'within_class_mean_cosine': within,
            'mean_vector_norm': float(np.linalg.norm(mean_vec)),
        }
        out_rows.append({
            'raw_id': int(raw_id),
            'name': class_names.get(int(raw_id)),
            'count': n,
            'within_class_mean_cosine': within,
            'mean_vector_norm': float(np.linalg.norm(mean_vec)),
        })
    return stats, out_rows


def _target_class_set(rows: Sequence[Mapping[str, Any]], class_stats: Mapping[int, Mapping[str, Any]], hub_set: set[int], *, max_pairwise_classes: int) -> set[int]:
    impacted = {
        int(r['gt_raw_id'])
        for r in rows
        if (r.get('suppressor_raw_id') is not None and int(r.get('suppressor_raw_id')) in hub_set)
        or bool(r.get('source_is_other_trajectory'))
        or str(r.get('failure_bucket')) in {'gt_not_active_extra', 'active_yprime_win', 'active_wrong_extra_win'}
    }
    targets = set(hub_set) | impacted
    if len(targets) < 2:
        # Fallback: include the most frequent GT classes.
        frequent = [int(k) for k, _ in sorted(class_stats.items(), key=lambda kv: int(kv[1].get('count', 0)), reverse=True)]
        targets.update(frequent[: max(2, int(max_pairwise_classes))])
    if max_pairwise_classes > 0 and len(targets) > max_pairwise_classes:
        ranked = sorted(targets, key=lambda rid: int(class_stats.get(rid, {}).get('count', 0)), reverse=True)
        targets = set(ranked[: int(max_pairwise_classes)]) | set(hub_set)
    return {int(x) for x in targets if int(x) in class_stats}


def _pairwise_rows(class_stats: Mapping[int, Mapping[str, Any]], target_classes: Sequence[int], hub_set: set[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    targets = [int(x) for x in sorted(set(target_classes)) if int(x) in class_stats]
    for i, a in enumerate(targets):
        sa = class_stats[int(a)]
        for b in targets[i + 1:]:
            sb = class_stats[int(b)]
            mean_pair_cos = float(np.dot(np.asarray(sa['mean_vec'], dtype=np.float32), np.asarray(sb['mean_vec'], dtype=np.float32)))
            centroid_cos = float(np.dot(np.asarray(sa['centroid'], dtype=np.float32), np.asarray(sb['centroid'], dtype=np.float32)))
            rows.append({
                'class_a_raw_id': int(a),
                'class_a_name': sa.get('name'),
                'class_a_count': int(sa.get('count', 0)),
                'class_a_is_hub': bool(int(a) in hub_set),
                'class_b_raw_id': int(b),
                'class_b_name': sb.get('name'),
                'class_b_count': int(sb.get('count', 0)),
                'class_b_is_hub': bool(int(b) in hub_set),
                'sample_mean_between_cosine': mean_pair_cos,
                'centroid_cosine': centroid_cos,
                'within_a_minus_between_mean': None if sa.get('within_class_mean_cosine') is None else float(sa['within_class_mean_cosine']) - mean_pair_cos,
                'within_b_minus_between_mean': None if sb.get('within_class_mean_cosine') is None else float(sb['within_class_mean_cosine']) - mean_pair_cos,
            })
    return sorted(rows, key=lambda r: float(r['centroid_cosine']), reverse=True)


def _compute_row_geometry(rows: List[Record], vectors: np.ndarray, class_stats: Mapping[int, Mapping[str, Any]], hub_set: set[int]) -> None:
    valid_centroids = {int(k): np.asarray(v['centroid'], dtype=np.float32) for k, v in class_stats.items() if int(v.get('count', 0)) >= 1}
    all_ids = sorted(valid_centroids)
    all_matrix = np.stack([valid_centroids[rid] for rid in all_ids], axis=0) if all_ids else np.zeros((0, vectors.shape[1]), dtype=np.float32)
    hub_ids = [int(x) for x in sorted(hub_set) if int(x) in valid_centroids]
    hub_matrix = np.stack([valid_centroids[rid] for rid in hub_ids], axis=0) if hub_ids else np.zeros((0, vectors.shape[1]), dtype=np.float32)

    for i, row in enumerate(rows):
        z = np.asarray(vectors[i], dtype=np.float32)
        gt = int(row['gt_raw_id'])
        gt_centroid = valid_centroids.get(gt)
        cos_gt = float(np.dot(z, gt_centroid)) if gt_centroid is not None else None

        nearest_non_gt_id: Optional[int] = None
        nearest_non_gt_cos: Optional[float] = None
        if all_matrix.shape[0]:
            sims = np.matmul(all_matrix, z)
            for pos in np.argsort(-sims, kind='stable'):
                rid = all_ids[int(pos)]
                if rid == gt:
                    continue
                nearest_non_gt_id = int(rid)
                nearest_non_gt_cos = float(sims[int(pos)])
                break

        nearest_hub_id: Optional[int] = None
        nearest_hub_cos: Optional[float] = None
        if hub_matrix.shape[0]:
            hsims = np.matmul(hub_matrix, z)
            hpos = int(np.argmax(hsims))
            nearest_hub_id = int(hub_ids[hpos])
            nearest_hub_cos = float(hsims[hpos])

        suppressor = _safe_int(row.get('suppressor_raw_id'))
        suppressor_cos: Optional[float] = None
        if suppressor is not None and int(suppressor) in valid_centroids:
            suppressor_cos = float(np.dot(z, valid_centroids[int(suppressor)]))

        row.update({
            'carrier_cos_to_gt_centroid': cos_gt,
            'nearest_non_gt_centroid_raw_id': nearest_non_gt_id,
            'nearest_non_gt_centroid_name': class_stats.get(int(nearest_non_gt_id), {}).get('name') if nearest_non_gt_id is not None else None,
            'nearest_non_gt_centroid_cos': nearest_non_gt_cos,
            'carrier_gt_vs_nearest_non_gt_margin': None if cos_gt is None or nearest_non_gt_cos is None else float(cos_gt - nearest_non_gt_cos),
            'nearest_hub_centroid_raw_id': nearest_hub_id,
            'nearest_hub_centroid_name': class_stats.get(int(nearest_hub_id), {}).get('name') if nearest_hub_id is not None else None,
            'nearest_hub_centroid_cos': nearest_hub_cos,
            'carrier_gt_vs_nearest_hub_margin': None if cos_gt is None or nearest_hub_cos is None else float(cos_gt - nearest_hub_cos),
            'carrier_cos_to_suppressor_centroid': suppressor_cos,
            'carrier_gt_vs_suppressor_margin': None if cos_gt is None or suppressor_cos is None else float(cos_gt - suppressor_cos),
            'carrier_geometry_gt_centroid_wins_non_gt': bool(cos_gt is not None and nearest_non_gt_cos is not None and cos_gt > nearest_non_gt_cos),
            'carrier_geometry_gt_centroid_wins_hub': bool(cos_gt is not None and nearest_hub_cos is not None and cos_gt > nearest_hub_cos),
            'carrier_geometry_gt_centroid_wins_suppressor': None if suppressor_cos is None or cos_gt is None else bool(cos_gt > suppressor_cos),
        })


def _summarize_bucket(rows: Sequence[Mapping[str, Any]], name: str) -> Dict[str, Any]:
    seq = [dict(r) for r in rows]
    margins_non_gt = [float(r['carrier_gt_vs_nearest_non_gt_margin']) for r in seq if r.get('carrier_gt_vs_nearest_non_gt_margin') is not None]
    margins_hub = [float(r['carrier_gt_vs_nearest_hub_margin']) for r in seq if r.get('carrier_gt_vs_nearest_hub_margin') is not None]
    margins_suppressor = [float(r['carrier_gt_vs_suppressor_margin']) for r in seq if r.get('carrier_gt_vs_suppressor_margin') is not None]
    return {
        'bucket': str(name),
        'count': int(len(seq)),
        'diagnosis_join_rate': _rate([bool(r.get('diagnosis_joined')) for r in seq]),
        'active_raw_contains_rate': _rate([bool(r.get('active_raw_contains')) for r in seq if r.get('active_raw_contains') is not None]),
        'final_top1_is_gt_rate': _rate([bool(r.get('final_top1_is_gt')) for r in seq if r.get('final_top1_is_gt') is not None]),
        'source_is_other_trajectory_rate': _rate([bool(r.get('source_is_other_trajectory')) for r in seq if r.get('source_trajectory_id') is not None]),
        'mean_margin_gt_vs_nearest_non_gt': _mean(margins_non_gt),
        'median_margin_gt_vs_nearest_non_gt': _median(margins_non_gt),
        'p10_margin_gt_vs_nearest_non_gt': _quantile(margins_non_gt, 0.10),
        'margin_gt_vs_nearest_non_gt_positive_rate': _rate([float(x) > 0.0 for x in margins_non_gt]),
        'mean_margin_gt_vs_nearest_hub': _mean(margins_hub),
        'median_margin_gt_vs_nearest_hub': _median(margins_hub),
        'p10_margin_gt_vs_nearest_hub': _quantile(margins_hub, 0.10),
        'margin_gt_vs_nearest_hub_positive_rate': _rate([float(x) > 0.0 for x in margins_hub]),
        'mean_margin_gt_vs_suppressor': _mean(margins_suppressor),
        'median_margin_gt_vs_suppressor': _median(margins_suppressor),
        'p10_margin_gt_vs_suppressor': _quantile(margins_suppressor, 0.10),
        'margin_gt_vs_suppressor_positive_rate': _rate([float(x) > 0.0 for x in margins_suppressor]),
        'suppressor_histogram_top20': [
            {'raw_id': int(k), 'count': int(v)}
            for k, v in Counter(int(r['suppressor_raw_id']) for r in seq if r.get('suppressor_raw_id') is not None).most_common(20)
        ],
    }


def _bucket_summary(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get('failure_bucket') or 'bucket_unknown')].append(row)
    payload_rows = [_summarize_bucket(group, name) for name, group in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
    payload = {
        'status': 'PASS',
        'row_count': int(len(rows)),
        'bucket_count': int(len(payload_rows)),
        'buckets': payload_rows,
    }
    return payload, payload_rows


def _hub_pair_margin_rows(rows: Sequence[Mapping[str, Any]], class_stats: Mapping[int, Mapping[str, Any]], hub_set: set[int]) -> List[Dict[str, Any]]:
    by_pair: Dict[Tuple[int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        gt = int(row['gt_raw_id'])
        sup = _safe_int(row.get('suppressor_raw_id'))
        if sup is not None and int(sup) in hub_set and int(sup) != gt:
            by_pair[(gt, int(sup))].append(row)
        else:
            nh = _safe_int(row.get('nearest_hub_centroid_raw_id'))
            if nh is not None and int(nh) != gt:
                by_pair[(gt, int(nh))].append(row)
    out: List[Dict[str, Any]] = []
    for (gt, hub), seq in sorted(by_pair.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        margins = [float(r['carrier_gt_vs_suppressor_margin']) for r in seq if r.get('suppressor_raw_id') == hub and r.get('carrier_gt_vs_suppressor_margin') is not None]
        if not margins:
            margins = [float(r['carrier_gt_vs_nearest_hub_margin']) for r in seq if r.get('nearest_hub_centroid_raw_id') == hub and r.get('carrier_gt_vs_nearest_hub_margin') is not None]
        out.append({
            'gt_raw_id': int(gt),
            'gt_name': class_stats.get(gt, {}).get('name'),
            'hub_raw_id': int(hub),
            'hub_name': class_stats.get(hub, {}).get('name'),
            'row_count': int(len(seq)),
            'mean_carrier_margin_gt_minus_hub': _mean(margins),
            'median_carrier_margin_gt_minus_hub': _median(margins),
            'p10_carrier_margin_gt_minus_hub': _quantile(margins, 0.10),
            'positive_margin_rate': _rate([float(x) > 0.0 for x in margins]),
            'active_raw_contains_rate': _rate([bool(r.get('active_raw_contains')) for r in seq if r.get('active_raw_contains') is not None]),
            'final_top1_is_gt_rate': _rate([bool(r.get('final_top1_is_gt')) for r in seq if r.get('final_top1_is_gt') is not None]),
            'source_is_other_trajectory_rate': _rate([bool(r.get('source_is_other_trajectory')) for r in seq if r.get('source_trajectory_id') is not None]),
        })
    return out


def _knn_purity(rows: List[Record], vectors: np.ndarray, ks: Sequence[int], *, chunk_size: int = 512) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    n = int(vectors.shape[0])
    labels = np.asarray([int(r['gt_raw_id']) for r in rows], dtype=np.int64)
    clips = np.asarray([str(r.get('clip_id')) for r in rows], dtype=object)
    max_k = max([int(k) for k in ks] + [1])
    max_k = min(max_k, max(1, n - 1))

    per_row: Dict[int, Dict[str, Any]] = {i: {} for i in range(n)}
    same_rates: Dict[int, List[float]] = {int(k): [] for k in ks}
    leave_clip_same_rates: Dict[int, List[float]] = {int(k): [] for k in ks}
    top1_same: List[bool] = []
    top1_leave_clip_same: List[bool] = []

    X = np.asarray(vectors, dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = np.matmul(X[start:end], X.T)
        for local_i, global_i in enumerate(range(start, end)):
            row_sims = sims[local_i].astype(np.float32)
            row_sims[global_i] = -np.inf
            order = np.argsort(-row_sims, kind='stable')
            order = [int(j) for j in order if np.isfinite(row_sims[int(j)])]
            order_lco = [int(j) for j in order if str(clips[j]) != str(clips[global_i])]
            if order:
                top1_same.append(bool(labels[order[0]] == labels[global_i]))
                per_row[global_i]['knn_top1_same_gt'] = bool(labels[order[0]] == labels[global_i])
                per_row[global_i]['knn_top1_raw_id'] = int(labels[order[0]])
                per_row[global_i]['knn_top1_similarity'] = float(row_sims[order[0]])
            if order_lco:
                top1_leave_clip_same.append(bool(labels[order_lco[0]] == labels[global_i]))
                per_row[global_i]['knn_lco_top1_same_gt'] = bool(labels[order_lco[0]] == labels[global_i])
                per_row[global_i]['knn_lco_top1_raw_id'] = int(labels[order_lco[0]])
                per_row[global_i]['knn_lco_top1_similarity'] = float(row_sims[order_lco[0]])
            for k in ks:
                kk = min(int(k), len(order))
                if kk > 0:
                    same = float(np.asarray(labels[order[:kk]] == labels[global_i], dtype=np.float32).mean())
                    same_rates[int(k)].append(same)
                    per_row[global_i][f'knn_top{k}_same_gt_purity'] = same
                kk_lco = min(int(k), len(order_lco))
                if kk_lco > 0:
                    same_lco = float(np.asarray(labels[order_lco[:kk_lco]] == labels[global_i], dtype=np.float32).mean())
                    leave_clip_same_rates[int(k)].append(same_lco)
                    per_row[global_i][f'knn_lco_top{k}_same_gt_purity'] = same_lco
    summary = {
        'status': 'PASS',
        'row_count': int(n),
        'ks': [int(k) for k in ks],
        'top1_same_gt_rate': _rate(top1_same),
        'leave_one_clip_top1_same_gt_rate': _rate(top1_leave_clip_same),
        'topk_same_gt_purity': {str(k): _mean(v) for k, v in same_rates.items()},
        'leave_one_clip_topk_same_gt_purity': {str(k): _mean(v) for k, v in leave_clip_same_rates.items()},
    }
    return summary, per_row


def _source_leakage_counterfactual(rows: Sequence[Mapping[str, Any]], vectors: np.ndarray, class_stats: Mapping[int, Mapping[str, Any]], *, q: float, top_examples: int) -> Dict[str, Any]:
    valid_centroids = {int(k): np.asarray(v['centroid'], dtype=np.float32) for k, v in class_stats.items() if int(v.get('count', 0)) >= 1}
    row_index_by_tid = {str(r['trajectory_id']): i for i, r in enumerate(rows)}
    source_suppressor_sims: List[float] = []
    cases: List[Dict[str, Any]] = []
    for row in rows:
        source_tid = row.get('source_trajectory_id')
        sup = _safe_int(row.get('suppressor_raw_id'))
        if not source_tid or sup is None or int(sup) not in valid_centroids:
            continue
        src_idx = row_index_by_tid.get(str(source_tid))
        if src_idx is None:
            continue
        source_sim = float(np.dot(np.asarray(vectors[src_idx], dtype=np.float32), valid_centroids[int(sup)]))
        source_suppressor_sims.append(source_sim)
    threshold = _quantile(source_suppressor_sims, q) if source_suppressor_sims else None

    for row in rows:
        source_tid = row.get('source_trajectory_id')
        sup = _safe_int(row.get('suppressor_raw_id'))
        if not source_tid or sup is None or int(sup) not in valid_centroids:
            continue
        src_idx = row_index_by_tid.get(str(source_tid))
        if src_idx is None:
            continue
        target_margin = _safe_float(row.get('carrier_gt_vs_suppressor_margin'))
        source_sim = float(np.dot(np.asarray(vectors[src_idx], dtype=np.float32), valid_centroids[int(sup)]))
        target_sup = _safe_float(row.get('carrier_cos_to_suppressor_centroid'))
        cases.append({
            'trajectory_id': row.get('trajectory_id'),
            'clip_id': row.get('clip_id'),
            'gt_raw_id': row.get('gt_raw_id'),
            'gt_name': row.get('gt_name'),
            'suppressor_raw_id': int(sup),
            'suppressor_name': row.get('suppressor_name'),
            'source_trajectory_id': source_tid,
            'source_is_other_trajectory': bool(str(source_tid) != str(row.get('trajectory_id'))),
            'target_margin_gt_minus_suppressor': target_margin,
            'target_suppressor_cos': target_sup,
            'source_suppressor_cos': source_sim,
            'target_carrier_separable_but_source_hijacked': bool(
                str(source_tid) != str(row.get('trajectory_id'))
                and target_margin is not None
                and target_margin > 0.0
                and threshold is not None
                and source_sim >= float(threshold)
            ),
        })
    positive = [c for c in cases if bool(c['target_carrier_separable_but_source_hijacked'])]
    cases_sorted = sorted(cases, key=lambda c: (not bool(c['target_carrier_separable_but_source_hijacked']), -float(c.get('source_suppressor_cos') or -1e9)))
    return {
        'status': 'PASS' if cases else 'NO_SOURCE_FIELDS_OR_NO_SOURCE_CARRIERS',
        'definition': 'If target carrier is closer to its GT centroid than to the suppressor centroid, but an other-source trajectory is very close to that suppressor centroid, this supports source leakage rather than carrier inseparability.',
        'case_count': int(len(cases)),
        'source_suppressor_similarity_quantile': float(q),
        'source_suppressor_similarity_threshold': threshold,
        'target_separable_but_source_hijacked_count': int(len(positive)),
        'target_separable_but_source_hijacked_rate': float(len(positive) / max(len(cases), 1)) if cases else None,
        'source_is_other_trajectory_rate': _rate([bool(c.get('source_is_other_trajectory')) for c in cases]),
        'mean_target_margin_gt_minus_suppressor': _mean([float(c['target_margin_gt_minus_suppressor']) for c in cases if c.get('target_margin_gt_minus_suppressor') is not None]),
        'mean_source_suppressor_cos': _mean([float(c['source_suppressor_cos']) for c in cases if c.get('source_suppressor_cos') is not None]),
        'top_examples': cases_sorted[: max(0, int(top_examples))],
    }


def _interpretation(summary: Mapping[str, Any], bucket_payload: Mapping[str, Any], source_payload: Mapping[str, Any], knn_payload: Mapping[str, Any]) -> Dict[str, Any]:
    all_bucket = next((b for b in bucket_payload.get('buckets', []) if b.get('bucket') == 'ALL'), None)
    if all_bucket is None:
        all_bucket = _summarize_bucket([], 'ALL')
    pos_non_gt = _safe_float(all_bucket.get('margin_gt_vs_nearest_non_gt_positive_rate'))
    pos_hub = _safe_float(all_bucket.get('margin_gt_vs_nearest_hub_positive_rate'))
    source_rate = _safe_float(source_payload.get('target_separable_but_source_hijacked_rate'))
    knn_top1 = _safe_float(knn_payload.get('top1_same_gt_rate'))
    lco_top1 = _safe_float(knn_payload.get('leave_one_clip_top1_same_gt_rate'))

    if pos_hub is not None and pos_hub < 0.35 and knn_top1 is not None and knn_top1 < 0.25:
        verdict = 'carrier_geometry_inseparability_likely_major_factor'
    elif source_rate is not None and source_rate >= 0.30:
        verdict = 'source_leakage_likely_major_factor_carrier_not_sufficiently_exonerated'
    elif pos_hub is not None and pos_hub >= 0.50:
        verdict = 'carrier_has_substantial_gt_vs_hub_separability_aggregation_or_estep_likely_major_factor'
    else:
        verdict = 'mixed_or_insufficient_geometry_evidence'
    return {
        'verdict': verdict,
        'key_rates': {
            'overall_positive_margin_gt_vs_nearest_non_gt_rate': pos_non_gt,
            'overall_positive_margin_gt_vs_nearest_hub_rate': pos_hub,
            'source_leakage_target_separable_but_source_hijacked_rate': source_rate,
            'knn_top1_same_gt_rate': knn_top1,
            'leave_one_clip_knn_top1_same_gt_rate': lco_top1,
        },
        'decision_matrix': {
            'carrier_inseparable_if': 'low GT-vs-hub centroid positive-margin rate plus low same-class kNN purity, especially in failed/person-suppressed buckets',
            'source_leakage_if': 'target GT-vs-suppressor carrier margin is positive but suppressor source trajectory is an other trajectory with high suppressor-centroid cosine',
            'aggregation_estep_if': 'carrier margins and kNN purity are nontrivial but active/top1 failure remains high in diagnosis buckets',
        },
    }


def _write_takeover(path: Path, *, config: AuditConfig, summary: Mapping[str, Any], interpretation: Mapping[str, Any], output_files: Mapping[str, str]) -> None:
    lines = [
        '# Hub Carrier Separability Audit Takeover',
        '',
        '## Scope',
        f'- dataset: `{config.dataset_name}`',
        f'- stage: `{config.stage}`',
        f'- splits: `{",".join(config.splits) if config.splits else "all"}`',
        f'- hub_raw_ids: `{list(config.hub_raw_ids)}`',
        '',
        '## Status',
        f'- status: `{summary.get("status")}`',
        f'- usable rows: `{summary.get("row_count")}`',
        f'- diagnosis join rate: `{summary.get("diagnosis", {}).get("join_rate")}`',
        f'- interpretation verdict: `{interpretation.get("verdict")}`',
        '',
        '## Key rates',
    ]
    for key, value in dict(interpretation.get('key_rates', {})).items():
        lines.append(f'- {key}: `{value}`')
    lines.extend(['', '## Output files'])
    for key, value in output_files.items():
        lines.append(f'- {key}: `{value}`')
    lines.extend([
        '',
        '## Reading guide',
        '- If failed/person-suppressed rows still have positive GT-vs-hub carrier margins, the dominant issue is not carrier geometry alone; check source leakage and E-step/slot allocation.',
        '- If failed/person-suppressed rows have negative GT-vs-hub carrier margins and low kNN purity, carrier-level inseparability is a major factor and frame/local evidence becomes a necessary repair direction.',
        '- This audit is read-only; it does not change training, checkpoints, predictions, or sidecars.',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


def run_audit(config: AuditConfig) -> Dict[str, Any]:
    output_dir = Path(config.output_dir) if config.output_dir is not None else _default_output_dir(config.run_root, config.dataset_name, config.stage)
    diag_dir = Path(config.diagnosis_dir) if config.diagnosis_dir is not None else _default_diagnosis_dir(config.run_root, config.dataset_name, config.stage)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, vectors, materialization_meta = _materialize_rows_and_carriers(config)
    class_names = {int(k): str(v) for k, v in materialization_meta.get('class_names', {}).items()}
    hub_set = {int(x) for x in config.hub_raw_ids}
    diag_by_tid, diag_meta = _load_diagnosis_rows(diag_dir)
    diag_join_meta = _attach_diagnosis(rows, diag_by_tid, hub_set)
    for row in rows:
        sup = _safe_int(row.get('suppressor_raw_id'))
        if sup is not None:
            row['suppressor_name'] = class_names.get(int(sup))

    class_stats, class_rows = _class_stats(rows, vectors, class_names)
    target_classes = _target_class_set(rows, class_stats, hub_set, max_pairwise_classes=config.max_pairwise_classes)
    pairwise_rows = _pairwise_rows(class_stats, sorted(target_classes), hub_set)
    _compute_row_geometry(rows, vectors, class_stats, hub_set)

    knn_summary, knn_per_row = _knn_purity(rows, vectors, config.knn_ks)
    for i, row in enumerate(rows):
        row.update(knn_per_row.get(i, {}))

    all_bucket = _summarize_bucket(rows, 'ALL')
    bucket_payload, bucket_rows = _bucket_summary(rows)
    bucket_payload['buckets'] = [all_bucket] + list(bucket_payload.get('buckets', []))
    bucket_rows = [all_bucket] + bucket_rows

    hub_pair_rows = _hub_pair_margin_rows(rows, class_stats, hub_set)
    source_payload = _source_leakage_counterfactual(rows, vectors, class_stats, q=config.source_leakage_quantile, top_examples=config.top_examples)

    summary: Dict[str, Any] = {
        'status': 'PASS',
        'audit_name': 'hub_carrier_separability',
        'question': 'Do hub classes and the classes they suppress collapse already at trajectory-carrier geometry level?',
        'run_root': str(config.run_root),
        'dataset_name': str(config.dataset_name),
        'stage': str(config.stage),
        'splits': list(config.splits),
        'hub_raw_ids': [int(x) for x in sorted(hub_set)],
        'row_count': int(len(rows)),
        'carrier_dim': int(vectors.shape[1]),
        'class_count': int(len(class_stats)),
        'target_pairwise_class_count': int(len(target_classes)),
        'materialization': materialization_meta,
        'diagnosis': {**diag_meta, **diag_join_meta},
        'overall_geometry': all_bucket,
        'knn': knn_summary,
        'source_leakage_counterfactual': {k: v for k, v in source_payload.items() if k != 'top_examples'},
    }
    interpretation = _interpretation(summary, bucket_payload, source_payload, knn_summary)
    summary['interpretation'] = interpretation

    paths = {
        'summary': output_dir / 'summary.json',
        'row_geometry_diagnostics': output_dir / 'row_geometry_diagnostics.jsonl',
        'bucket_summary': output_dir / 'bucket_summary.json',
        'bucket_summary_csv': output_dir / 'bucket_summary.csv',
        'class_centroid_summary': output_dir / 'class_centroid_summary.csv',
        'pairwise_class_cosine': output_dir / 'pairwise_class_cosine.csv',
        'hub_pair_margin_summary': output_dir / 'hub_pair_margin_summary.csv',
        'knn_purity_summary': output_dir / 'knn_purity_summary.json',
        'source_leakage_counterfactual_summary': output_dir / 'source_leakage_counterfactual_summary.json',
        'source_leakage_top_examples': output_dir / 'source_leakage_top_examples.jsonl',
        'takeover': output_dir / 'HUB_CARRIER_SEPARABILITY_TAKEOVER.md',
    }

    _write_json(paths['summary'], summary)
    _write_jsonl(paths['row_geometry_diagnostics'], rows)
    _write_json(paths['bucket_summary'], bucket_payload)
    _write_csv(paths['bucket_summary_csv'], bucket_rows)
    _write_csv(paths['class_centroid_summary'], class_rows)
    _write_csv(paths['pairwise_class_cosine'], pairwise_rows)
    _write_csv(paths['hub_pair_margin_summary'], hub_pair_rows)
    _write_json(paths['knn_purity_summary'], knn_summary)
    _write_json(paths['source_leakage_counterfactual_summary'], source_payload)
    _write_jsonl(paths['source_leakage_top_examples'], source_payload.get('top_examples', []))
    _write_takeover(
        paths['takeover'],
        config=config,
        summary=summary,
        interpretation=interpretation,
        output_files={k: str(v) for k, v in paths.items()},
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Audit hub/affected-class separability in trajectory-carrier space')
    p.add_argument('--run_root', type=Path, required=True, help='Experiment output root, e.g. codex/outputs/G8_inference_and_eval/<exp>')
    p.add_argument('--dataset_name', type=str, default='lvvis_train_base')
    p.add_argument('--stage', type=str, default='softem_aug', choices=['prealign', 'softem_base', 'softem_aug'])
    p.add_argument('--trajectory_source_branch', type=str, default='mainline')
    p.add_argument('--output_dir', type=Path, default=None)
    p.add_argument('--diagnosis_dir', type=Path, default=None, help='Optional extra_mining_recall_diagnosis dir; defaults under run_root')
    p.add_argument('--splits', type=_parse_str_tuple, default=('base_unobserved',), help='Comma-separated splits or all. Default: base_unobserved')
    p.add_argument('--hub_raw_ids', type=_parse_int_tuple, default=(DEFAULT_PERSON_RAW_ID,), help='Comma-separated hub raw ids. Default: 773 (LV-VIS person)')
    p.add_argument('--min_class_count', type=int, default=3)
    p.add_argument('--knn_ks', type=_parse_int_tuple, default=DEFAULT_KNN_KS)
    p.add_argument('--max_rows', type=int, default=0, help='Optional debug cap; 0 means all rows')
    p.add_argument('--max_pairwise_classes', type=int, default=80)
    p.add_argument('--top_examples', type=int, default=80)
    p.add_argument('--source_leakage_quantile', type=float, default=0.75)
    p.add_argument('--all_gt_generate_sidecars_if_missing', type=_parse_bool, default=False)
    p.add_argument('--show_progress', type=_parse_bool, default=True)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    config = AuditConfig(
        run_root=Path(args.run_root).expanduser().resolve(),
        dataset_name=str(args.dataset_name),
        stage=str(args.stage),
        trajectory_source_branch=str(args.trajectory_source_branch),
        output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir is not None else None,
        diagnosis_dir=Path(args.diagnosis_dir).expanduser().resolve() if args.diagnosis_dir is not None else None,
        splits=tuple(str(x) for x in args.splits),
        hub_raw_ids=tuple(int(x) for x in args.hub_raw_ids),
        min_class_count=max(1, int(args.min_class_count)),
        knn_ks=tuple(sorted({max(1, int(x)) for x in args.knn_ks})),
        max_rows=max(0, int(args.max_rows)),
        max_pairwise_classes=max(0, int(args.max_pairwise_classes)),
        top_examples=max(0, int(args.top_examples)),
        source_leakage_quantile=min(1.0, max(0.0, float(args.source_leakage_quantile))),
        all_gt_generate_sidecars_if_missing=bool(args.all_gt_generate_sidecars_if_missing),
        show_progress=bool(args.show_progress),
    )
    summary = run_audit(config)
    print(json.dumps({
        'status': summary.get('status'),
        'row_count': summary.get('row_count'),
        'output_dir': str(config.output_dir or _default_output_dir(config.run_root, config.dataset_name, config.stage)),
        'verdict': summary.get('interpretation', {}).get('verdict'),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
