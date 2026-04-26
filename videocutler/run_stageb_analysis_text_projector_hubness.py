#!/usr/bin/env python3
"""
WSOVVIS text-projector / prototype hubness audit.

Read-only analysis. This script tests the mechanism hypothesis raised after the
hub-carrier separability audit:

  Carrier features for hub-affected GT classes may still be separable from hubs
  such as person, but the trained text-side projector and clip-level
  carrier-to-text scoring may make projected text anchors / scores hub-biased,
  causing extra-mining slots to be occupied by hubs and preventing hidden GT
  classes from receiving an E/M reinforcement path.

It does not train, infer, rewrite checkpoints, or modify existing artifacts.

Default outputs:
  <run_root>/analysis/text_projector_hubness/<dataset>/<stage>/summary.json
  <run_root>/analysis/text_projector_hubness/<dataset>/<stage>/TEXT_PROJECTOR_HUBNESS_TAKEOVER.md
  <run_root>/analysis/text_projector_hubness/<dataset>/<stage>/row_text_margin_diagnostics.jsonl
  <run_root>/analysis/text_projector_hubness/<dataset>/<stage>/pair_text_projection_summary.csv
  <run_root>/analysis/text_projector_hubness/<dataset>/<stage>/class_text_hubness_summary.csv
  <run_root>/analysis/text_projector_hubness/<dataset>/<stage>/stage_comparison_summary.csv
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
import torch.nn.functional as F


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_carrier_evidence, load_text_vocab  # noqa: E402
from videocutler.ext_stageb_ovvis.audit.g8_minimal_split_audit import (  # noqa: E402
    MinimalSplitAuditConfig,
    _canonical_sidecar_gt_raw_id,
    _materialize_shared_inputs,
    _split_order_for_dataset,
    _stage_row_source_path,
)
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import _all_gt_split_label  # noqa: E402
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig  # noqa: E402


Record = Dict[str, Any]
DEFAULT_PERSON_RAW_ID = 773
STAGE_CHECKPOINTS: Dict[str, str] = {
    'prealign': 'train/prealign/checkpoints/prealign_last.pth',
    'softem_base': 'train/softem_base/checkpoints/softem_base_last.pth',
    'softem_aug': 'train/softem_aug/checkpoints/softem_aug_last.pth',
}


@dataclass(frozen=True)
class AuditConfig:
    run_root: Path
    dataset_name: str
    stage: str
    compare_stages: Tuple[str, ...]
    trajectory_source_branch: str
    output_dir: Optional[Path]
    splits: Tuple[str, ...]
    hub_raw_ids: Tuple[int, ...]
    min_class_count: int
    max_rows: int
    top_pairs: int
    all_gt_generate_sidecars_if_missing: bool
    device: str
    batch_size_rows: int
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


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.asarray(vals, dtype=np.float64).mean()) if vals else None


def _median(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(np.asarray(vals, dtype=np.float64))) if vals else None


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.quantile(np.asarray(vals, dtype=np.float64), float(q))) if vals else None


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
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def _write_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(str(x) for x in lines).rstrip() + '\n', encoding='utf-8')


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
    return class_names.get(int(rid), f'raw_id_{int(rid)}')


def _load_projector_from_checkpoint(checkpoint_path: Path, *, device: torch.device) -> Tuple[Projector, float, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'text_projector_state_dict' not in checkpoint or 'text_projector_config' not in checkpoint:
        raise RuntimeError(f'incompatible text-projector checkpoint: {checkpoint_path}')
    cfg = dict(checkpoint.get('text_projector_config', {}))
    projector = Projector(
        ProjectorConfig(
            input_dim=int(cfg.get('input_dim', 512)),
            hidden_dim=int(cfg.get('hidden_dim', 1024)),
            output_dim=int(cfg.get('output_dim', 768)),
            dropout=float(cfg.get('dropout', 0.0)),
            use_layernorm=bool(cfg.get('use_layernorm', True)),
        )
    ).to(device)
    projector.load_state_dict(checkpoint['text_projector_state_dict'])
    projector.eval()
    theta_t = float(checkpoint.get('theta_T', 0.07))
    t_dis = float(F.softplus(torch.tensor(theta_t, dtype=torch.float32)).item() + 1e-4)
    meta = {
        'checkpoint_path': str(checkpoint_path),
        'stage_id': checkpoint.get('stage_id'),
        'epoch': checkpoint.get('epoch'),
        'theta_T': theta_t,
        'T_dis': t_dis,
        'pipeline': checkpoint.get('pipeline'),
        'training_semantics': checkpoint.get('training_semantics'),
        'extra_selection_mode': checkpoint.get('extra_selection_mode'),
    }
    return projector, t_dis, meta


@torch.no_grad()
def _project_text_matrix(projector: Projector, text_matrix: np.ndarray, *, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    chunks: List[np.ndarray] = []
    arr = np.asarray(text_matrix, dtype=np.float32)
    for start in range(0, int(arr.shape[0]), int(batch_size)):
        chunk = torch.from_numpy(arr[start:start + int(batch_size)]).to(device=device, dtype=torch.float32)
        proj = projector(chunk)
        proj = F.normalize(proj, p=2.0, dim=-1)
        chunks.append(proj.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        raise RuntimeError('empty text matrix')
    return np.concatenate(chunks, axis=0).astype(np.float32)


def _stage_checkpoint_path(run_root: Path, stage: str) -> Path:
    if str(stage) not in STAGE_CHECKPOINTS:
        raise ValueError(f'unsupported stage={stage!r}; expected one of {sorted(STAGE_CHECKPOINTS)}')
    return run_root / STAGE_CHECKPOINTS[str(stage)]


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
        observed_raw_ids = [int(x) for x in list(sample.get('observed_raw_ids', []))]
        split = _all_gt_split_label(
            dataset_name=config.dataset_name,
            gt_raw_id=int(gt_raw_id),
            observed_raw_ids=observed_raw_ids,
            base_vocab_ids=prepared['base_vocab_ids'],
        )
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
            'observed_raw_ids': observed_raw_ids,
        })
        vectors.append(arr.astype(np.float32))
        if config.show_progress and idx % 1024 == 0:
            print(f'[text-projector-hubness] materialized {len(rows)} usable rows / scanned {idx}', flush=True)

    if not rows:
        raise RuntimeError(
            f'TEXT_PROJECTOR_HUBNESS_EMPTY: dataset={config.dataset_name} stage={config.stage} '
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
        'text_vocab_root': str(asset_roots.asset_root if (asset_roots.asset_root / 'text_bank' / 'text_prototype_records.jsonl').is_file() else config.run_root),
    }
    return rows, np.stack(vectors, axis=0).astype(np.float32), metadata


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
            'name': class_names.get(int(raw_id), f'raw_id_{int(raw_id)}'),
            'count': n,
            'indices': indices,
            'centroid': centroid.astype(np.float32),
            'within_class_mean_cosine': within,
        }
        out_rows.append({
            'raw_id': int(raw_id),
            'name': class_names.get(int(raw_id), f'raw_id_{int(raw_id)}'),
            'count': n,
            'within_class_mean_cosine': within,
        })
    return stats, out_rows


def _load_text_vocab_for_audit(asset_root: Path, run_root: Path) -> Tuple[List[int], List[Record], np.ndarray, Path]:
    if (asset_root / 'text_bank' / 'text_prototype_records.jsonl').is_file():
        raw_ids, records, matrix = load_text_vocab(asset_root)
        return raw_ids, records, matrix, asset_root
    raw_ids, records, matrix = load_text_vocab(run_root)
    return raw_ids, records, matrix, run_root


def _rank_desc(scores: np.ndarray, target_index: int) -> int:
    scores = np.asarray(scores, dtype=np.float32)
    if target_index < 0 or target_index >= scores.shape[0]:
        return -1
    target = float(scores[int(target_index)])
    return int(np.sum(scores > target)) + 1


def _top_raw_ids_from_scores(scores: np.ndarray, raw_ids: Sequence[int], k: int, *, exclude_raw_id: Optional[int] = None) -> List[int]:
    arr = np.asarray(scores, dtype=np.float32).copy()
    if exclude_raw_id is not None:
        try:
            idx = list(raw_ids).index(int(exclude_raw_id))
            arr[idx] = -np.inf
        except ValueError:
            pass
    k = min(int(k), int(arr.shape[0]))
    if k <= 0:
        return []
    idxs = np.argpartition(-arr, kth=k - 1)[:k]
    idxs = idxs[np.argsort(-arr[idxs])]
    return [int(raw_ids[int(i)]) for i in idxs.tolist()]


@torch.no_grad()
def _compute_stage_row_text_margins(
    *,
    rows: List[Record],
    vectors: np.ndarray,
    raw_ids: Sequence[int],
    raw_id_to_idx: Mapping[int, int],
    projected_text: np.ndarray,
    hub_ids: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Tuple[List[Record], Dict[str, Any]]:
    Z = np.asarray(vectors, dtype=np.float32)
    A = np.asarray(projected_text, dtype=np.float32)
    hub_indices = [int(raw_id_to_idx[int(h)]) for h in hub_ids if int(h) in raw_id_to_idx]
    hub_ids_valid = [int(h) for h in hub_ids if int(h) in raw_id_to_idx]
    if not hub_indices:
        raise RuntimeError(f'none of requested hub ids {hub_ids} exist in text vocab')
    A_t = torch.from_numpy(A).to(device=device, dtype=torch.float32)
    out_rows: List[Record] = []
    gt_minus_nearest_hub: List[float] = []
    gt_minus_person: List[float] = []
    gt_rank_values: List[int] = []
    text_hub_winner: List[bool] = []
    for start in range(0, int(Z.shape[0]), int(batch_size)):
        end = min(int(Z.shape[0]), start + int(batch_size))
        z = torch.from_numpy(Z[start:end]).to(device=device, dtype=torch.float32)
        z = F.normalize(z, p=2.0, dim=-1)
        scores = torch.matmul(z, A_t.t()).detach().cpu().numpy().astype(np.float32)
        hub_scores = scores[:, hub_indices]
        nearest_hub_local = np.argmax(hub_scores, axis=1)
        nearest_hub_scores = hub_scores[np.arange(hub_scores.shape[0]), nearest_hub_local]
        top1_indices = np.argmax(scores, axis=1)
        for local_i, row_index in enumerate(range(start, end)):
            row = dict(rows[row_index])
            gt = int(row['gt_raw_id'])
            gt_idx = int(raw_id_to_idx[gt]) if gt in raw_id_to_idx else None
            if gt_idx is None:
                continue
            gt_score = float(scores[local_i, gt_idx])
            nearest_hub_idx = int(hub_indices[int(nearest_hub_local[local_i])])
            nearest_hub_raw = int(raw_ids[nearest_hub_idx])
            nearest_hub_score = float(nearest_hub_scores[local_i])
            margin_nearest = float(gt_score - nearest_hub_score)
            person_score = None
            person_margin = None
            if DEFAULT_PERSON_RAW_ID in raw_id_to_idx:
                person_score = float(scores[local_i, int(raw_id_to_idx[DEFAULT_PERSON_RAW_ID])])
                person_margin = float(gt_score - person_score)
                gt_minus_person.append(person_margin)
            gt_rank = _rank_desc(scores[local_i], gt_idx)
            top1_raw = int(raw_ids[int(top1_indices[local_i])])
            top1_is_hub = bool(top1_raw in set(hub_ids_valid))
            gt_minus_nearest_hub.append(margin_nearest)
            gt_rank_values.append(int(gt_rank))
            text_hub_winner.append(top1_is_hub)
            out_rows.append({
                'trajectory_id': row.get('trajectory_id'),
                'clip_id': row.get('clip_id'),
                'gt_raw_id': gt,
                'gt_name': row.get('gt_name'),
                'text_score_gt': gt_score,
                'nearest_hub_raw_id': nearest_hub_raw,
                'nearest_hub_text_score': nearest_hub_score,
                'text_margin_gt_minus_nearest_hub': margin_nearest,
                'person_text_score': person_score,
                'text_margin_gt_minus_person': person_margin,
                'text_gt_rank_full_vocab': int(gt_rank),
                'text_top1_raw_id': top1_raw,
                'text_top1_is_gt': bool(top1_raw == gt),
                'text_top1_is_hub': top1_is_hub,
            })
    summary = {
        'row_count': int(len(out_rows)),
        'positive_text_margin_gt_vs_nearest_hub_rate': _rate([x > 0.0 for x in gt_minus_nearest_hub]),
        'mean_text_margin_gt_vs_nearest_hub': _mean(gt_minus_nearest_hub),
        'median_text_margin_gt_vs_nearest_hub': _median(gt_minus_nearest_hub),
        'p10_text_margin_gt_vs_nearest_hub': _quantile(gt_minus_nearest_hub, 0.10),
        'positive_text_margin_gt_vs_person_rate': _rate([x > 0.0 for x in gt_minus_person]),
        'mean_text_margin_gt_vs_person': _mean(gt_minus_person),
        'median_text_margin_gt_vs_person': _median(gt_minus_person),
        'mean_text_gt_rank_full_vocab': _mean([float(x) for x in gt_rank_values]),
        'median_text_gt_rank_full_vocab': _median([float(x) for x in gt_rank_values]),
        'text_top1_is_hub_rate': _rate(text_hub_winner),
    }
    return out_rows, summary


def _pair_projection_rows(
    *,
    class_stats: Mapping[int, Mapping[str, Any]],
    class_names: Mapping[int, str],
    raw_ids: Sequence[int],
    raw_id_to_idx: Mapping[int, int],
    raw_text: np.ndarray,
    projected_by_stage: Mapping[str, np.ndarray],
    hub_ids: Sequence[int],
    min_class_count: int,
    top_pairs: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    valid_gt_ids = [int(rid) for rid, st in class_stats.items() if int(st.get('count', 0)) >= int(min_class_count) and int(rid) in raw_id_to_idx]
    for gt in valid_gt_ids:
        gt_idx = int(raw_id_to_idx[gt])
        gt_centroid = np.asarray(class_stats[gt]['centroid'], dtype=np.float32)
        for hub in hub_ids:
            if int(hub) == int(gt) or int(hub) not in raw_id_to_idx or int(hub) not in class_stats:
                # hub centroid may be absent from this split; text rows are still useful,
                # but carrier centroid margin is unavailable.
                pass
            hub_idx = int(raw_id_to_idx[int(hub)]) if int(hub) in raw_id_to_idx else None
            if hub_idx is None:
                continue
            raw_text_cos = float(np.dot(raw_text[gt_idx], raw_text[hub_idx]))
            row: Dict[str, Any] = {
                'gt_raw_id': int(gt),
                'gt_name': class_names.get(int(gt), f'raw_id_{int(gt)}'),
                'hub_raw_id': int(hub),
                'hub_name': class_names.get(int(hub), f'raw_id_{int(hub)}'),
                'row_count': int(class_stats[gt].get('count', 0)),
                'raw_clip_text_cos_gt_hub': raw_text_cos,
                'carrier_centroid_cos_gt_hub': None,
            }
            if int(hub) in class_stats:
                hub_centroid = np.asarray(class_stats[int(hub)]['centroid'], dtype=np.float32)
                row['carrier_centroid_cos_gt_hub'] = float(np.dot(gt_centroid, hub_centroid))
            prev_cos = None
            for stage, A in projected_by_stage.items():
                proj_cos = float(np.dot(A[gt_idx], A[hub_idx]))
                row[f'{stage}_projected_text_cos_gt_hub'] = proj_cos
                row[f'{stage}_projected_minus_raw_text_cos'] = float(proj_cos - raw_text_cos)
                if prev_cos is not None:
                    row[f'{stage}_minus_previous_stage_projected_cos'] = float(proj_cos - prev_cos)
                prev_cos = proj_cos
            rows.append(row)
    # Rank by strongest final-stage hub pull.
    primary_stage = list(projected_by_stage.keys())[-1] if projected_by_stage else ''
    key = f'{primary_stage}_projected_text_cos_gt_hub'
    rows = sorted(rows, key=lambda r: float(r.get(key, -999.0)), reverse=True)
    return rows[: int(top_pairs)] if int(top_pairs) > 0 else rows


def _class_text_hubness_rows(
    *,
    class_stats: Mapping[int, Mapping[str, Any]],
    class_names: Mapping[int, str],
    raw_ids: Sequence[int],
    raw_id_to_idx: Mapping[int, int],
    raw_text: np.ndarray,
    projected_by_stage: Mapping[str, np.ndarray],
    hub_ids: Sequence[int],
    min_class_count: int,
    top_k_neighbors: int = 5,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    valid_class_ids = [int(rid) for rid, st in class_stats.items() if int(st.get('count', 0)) >= int(min_class_count) and int(rid) in raw_id_to_idx]
    hub_set = set(int(x) for x in hub_ids)
    raw_sim = raw_text @ raw_text.T
    projected_sims = {stage: A @ A.T for stage, A in projected_by_stage.items()}
    for rid in valid_class_ids:
        idx = int(raw_id_to_idx[rid])
        raw_neighbors = _top_raw_ids_from_scores(raw_sim[idx], raw_ids, top_k_neighbors, exclude_raw_id=rid)
        row: Dict[str, Any] = {
            'raw_id': int(rid),
            'name': class_names.get(int(rid), f'raw_id_{int(rid)}'),
            'row_count': int(class_stats[int(rid)].get('count', 0)),
            'is_config_hub': bool(int(rid) in hub_set),
            'raw_text_mean_cos_to_vocab': float((np.sum(raw_sim[idx]) - 1.0) / max(len(raw_ids) - 1, 1)),
            'raw_text_top_neighbors': raw_neighbors,
            'raw_text_top_neighbor_is_hub': bool(any(int(x) in hub_set for x in raw_neighbors[:1])),
        }
        for stage, sim in projected_sims.items():
            neigh = _top_raw_ids_from_scores(sim[idx], raw_ids, top_k_neighbors, exclude_raw_id=rid)
            row[f'{stage}_projected_text_mean_cos_to_vocab'] = float((np.sum(sim[idx]) - 1.0) / max(len(raw_ids) - 1, 1))
            row[f'{stage}_projected_text_top_neighbors'] = neigh
            row[f'{stage}_projected_text_top_neighbor_is_hub'] = bool(any(int(x) in hub_set for x in neigh[:1]))
            row[f'{stage}_projected_text_any_top{top_k_neighbors}_neighbor_is_hub'] = bool(any(int(x) in hub_set for x in neigh))
        rows.append(row)
    primary_stage = list(projected_by_stage.keys())[-1] if projected_by_stage else ''
    rows = sorted(rows, key=lambda r: float(r.get(f'{primary_stage}_projected_text_mean_cos_to_vocab', -999.0)), reverse=True)
    return rows


def _summarize_class_text_hubness(rows: Sequence[Mapping[str, Any]], stages: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {'class_count': int(len(rows))}
    out['raw_text_top1_neighbor_is_hub_rate'] = _rate([bool(r.get('raw_text_top_neighbor_is_hub')) for r in rows])
    for stage in stages:
        out[f'{stage}_projected_text_top1_neighbor_is_hub_rate'] = _rate([bool(r.get(f'{stage}_projected_text_top_neighbor_is_hub')) for r in rows])
        out[f'{stage}_projected_text_any_top5_neighbor_is_hub_rate'] = _rate([bool(r.get(f'{stage}_projected_text_any_top5_neighbor_is_hub')) for r in rows])
        raw_cent = [float(r.get('raw_text_mean_cos_to_vocab')) for r in rows if r.get('raw_text_mean_cos_to_vocab') is not None]
        proj_cent = [float(r.get(f'{stage}_projected_text_mean_cos_to_vocab')) for r in rows if r.get(f'{stage}_projected_text_mean_cos_to_vocab') is not None]
        out[f'{stage}_projected_minus_raw_mean_cos_to_vocab_mean_delta'] = None if not raw_cent or not proj_cent else float(np.mean(np.asarray(proj_cent) - np.asarray(raw_cent)))
    return out


def _load_carrier_separability_reference(run_root: Path, dataset_name: str, stage: str) -> Dict[str, Any]:
    path = run_root / 'analysis' / 'hub_carrier_separability' / dataset_name / stage / 'summary.json'
    if not path.is_file():
        return {'exists': False, 'path': str(path)}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return {'exists': False, 'path': str(path), 'load_error': str(exc)}
    interp = payload.get('interpretation') if isinstance(payload.get('interpretation'), Mapping) else {}
    key_rates = interp.get('key_rates') if isinstance(interp.get('key_rates'), Mapping) else {}
    return {
        'exists': True,
        'path': str(path),
        'verdict': interp.get('verdict') or payload.get('verdict'),
        'overall_positive_margin_gt_vs_nearest_hub_rate': key_rates.get('overall_positive_margin_gt_vs_nearest_hub_rate'),
        'knn_top1_same_gt_rate': key_rates.get('knn_top1_same_gt_rate'),
        'leave_one_clip_knn_top1_same_gt_rate': key_rates.get('leave_one_clip_knn_top1_same_gt_rate'),
    }


def _derive_verdict(*, carrier_ref: Mapping[str, Any], stage_text_summary: Mapping[str, Any], class_hubness_summary: Mapping[str, Any], stage: str) -> str:
    carrier_sep = carrier_ref.get('overall_positive_margin_gt_vs_nearest_hub_rate')
    text_pos = stage_text_summary.get('positive_text_margin_gt_vs_nearest_hub_rate')
    text_hub_top1 = stage_text_summary.get('text_top1_is_hub_rate')
    proj_hub_neigh = class_hubness_summary.get(f'{stage}_projected_text_any_top5_neighbor_is_hub_rate')
    raw_hub_neigh = class_hubness_summary.get('raw_text_top1_neighbor_is_hub_rate')
    try:
        if carrier_sep is not None and text_pos is not None:
            if float(carrier_sep) >= 0.55 and float(text_pos) < 0.50:
                return 'carrier_separable_but_projected_text_scoring_hub_biased_likely'
            if float(text_pos) >= 0.55 and float(text_hub_top1 or 0.0) < 0.30:
                return 'projected_text_scoring_not_primary_bottleneck_aggregation_or_estep_likely'
        if proj_hub_neigh is not None and raw_hub_neigh is not None and float(proj_hub_neigh) > max(0.25, float(raw_hub_neigh) + 0.10):
            return 'projector_increases_hub_adjacency_likely'
    except Exception:
        pass
    return 'mixed_or_inconclusive_text_projector_hubness_needs_pairwise_review'


def run_audit(config: AuditConfig) -> Dict[str, Any]:
    rows, vectors, meta = _materialize_rows_and_carriers(config)
    class_names = {int(k): str(v) for k, v in dict(meta.get('class_names', {})).items()}
    class_stats, class_centroid_rows = _class_stats(rows, vectors, class_names)
    asset_root = Path(str(meta.get('asset_root')))
    raw_ids, _records, text_matrix, text_root = _load_text_vocab_for_audit(asset_root, config.run_root)
    raw_ids = [int(x) for x in raw_ids]
    raw_id_to_idx = {int(rid): idx for idx, rid in enumerate(raw_ids)}
    raw_text = _l2_normalize_np(np.asarray(text_matrix, dtype=np.float32), axis=1)

    device = torch.device(config.device if str(config.device).startswith('cuda') and torch.cuda.is_available() else 'cpu')
    compare_stages = tuple(dict.fromkeys([*config.compare_stages, config.stage]))
    projected_by_stage: Dict[str, np.ndarray] = {}
    checkpoint_meta: Dict[str, Any] = {}
    for stage in compare_stages:
        ckpt = _stage_checkpoint_path(config.run_root, stage)
        if not ckpt.is_file():
            if stage == config.stage:
                raise FileNotFoundError(f'missing requested checkpoint for stage={stage}: {ckpt}')
            continue
        projector, _temp, ckpt_meta = _load_projector_from_checkpoint(ckpt, device=device)
        projected_by_stage[str(stage)] = _project_text_matrix(projector, raw_text, device=device)
        checkpoint_meta[str(stage)] = ckpt_meta

    if config.stage not in projected_by_stage:
        raise RuntimeError(f'no projected text matrix for requested stage={config.stage}')

    stage_row_diagnostics: Dict[str, List[Record]] = {}
    stage_row_summaries: Dict[str, Dict[str, Any]] = {}
    for stage, A in projected_by_stage.items():
        row_diag, row_summary = _compute_stage_row_text_margins(
            rows=rows,
            vectors=vectors,
            raw_ids=raw_ids,
            raw_id_to_idx=raw_id_to_idx,
            projected_text=A,
            hub_ids=config.hub_raw_ids,
            device=device,
            batch_size=config.batch_size_rows,
        )
        stage_row_diagnostics[stage] = row_diag
        stage_row_summaries[stage] = row_summary

    pair_rows = _pair_projection_rows(
        class_stats=class_stats,
        class_names=class_names,
        raw_ids=raw_ids,
        raw_id_to_idx=raw_id_to_idx,
        raw_text=raw_text,
        projected_by_stage=projected_by_stage,
        hub_ids=config.hub_raw_ids,
        min_class_count=config.min_class_count,
        top_pairs=config.top_pairs,
    )
    class_text_rows = _class_text_hubness_rows(
        class_stats=class_stats,
        class_names=class_names,
        raw_ids=raw_ids,
        raw_id_to_idx=raw_id_to_idx,
        raw_text=raw_text,
        projected_by_stage=projected_by_stage,
        hub_ids=config.hub_raw_ids,
        min_class_count=config.min_class_count,
    )
    class_text_summary = _summarize_class_text_hubness(class_text_rows, list(projected_by_stage.keys()))

    # Cross-signal: carrier-separable but text margin negative for requested stage.
    requested_diag = stage_row_diagnostics[config.stage]
    by_tid_requested = {str(r['trajectory_id']): r for r in requested_diag}
    carrier_ref = _load_carrier_separability_reference(config.run_root, config.dataset_name, config.stage)

    # Compute carrier centroid GT-vs-nearest hub margin on the same rows for a direct row-level contrast.
    hub_ids = [int(h) for h in config.hub_raw_ids if int(h) in class_stats]
    hub_centroids = {int(h): np.asarray(class_stats[int(h)]['centroid'], dtype=np.float32) for h in hub_ids}
    carrier_pos_text_neg_flags: List[bool] = []
    direct_contrast_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        gt = int(row['gt_raw_id'])
        gt_cent = np.asarray(class_stats[gt]['centroid'], dtype=np.float32) if gt in class_stats else None
        if gt_cent is None or not hub_centroids:
            continue
        z = np.asarray(vectors[i], dtype=np.float32)
        gt_carrier_score = float(np.dot(z, gt_cent))
        hub_scores = [(h, float(np.dot(z, cent))) for h, cent in hub_centroids.items()]
        nearest_hub, nearest_hub_score = max(hub_scores, key=lambda x: x[1])
        carrier_margin = float(gt_carrier_score - nearest_hub_score)
        text_row = by_tid_requested.get(str(row['trajectory_id']), {})
        text_margin = text_row.get('text_margin_gt_minus_nearest_hub')
        if text_margin is None:
            continue
        text_margin_f = float(text_margin)
        flag = bool(carrier_margin > 0.0 and text_margin_f <= 0.0)
        carrier_pos_text_neg_flags.append(flag)
        direct_contrast_rows.append({
            'trajectory_id': row.get('trajectory_id'),
            'gt_raw_id': gt,
            'gt_name': row.get('gt_name'),
            'nearest_hub_raw_id_carrier': int(nearest_hub),
            'carrier_margin_gt_minus_nearest_hub': carrier_margin,
            'text_margin_gt_minus_nearest_hub': text_margin_f,
            'carrier_positive_text_negative': flag,
            'text_gt_rank_full_vocab': text_row.get('text_gt_rank_full_vocab'),
            'text_top1_raw_id': text_row.get('text_top1_raw_id'),
            'text_top1_is_hub': text_row.get('text_top1_is_hub'),
        })

    direct_contrast_summary = {
        'case_count': int(len(carrier_pos_text_neg_flags)),
        'carrier_positive_but_text_negative_rate': _rate(carrier_pos_text_neg_flags),
        'carrier_positive_but_text_negative_count': int(sum(1 for x in carrier_pos_text_neg_flags if x)),
    }

    verdict = _derive_verdict(
        carrier_ref=carrier_ref,
        stage_text_summary=stage_row_summaries[config.stage],
        class_hubness_summary=class_text_summary,
        stage=config.stage,
    )

    output_dir = config.output_dir or (config.run_root / 'analysis' / 'text_projector_hubness' / config.dataset_name / config.stage)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Requested stage row diagnostics are the primary per-row artifact; other stages are summary-only.
    _write_jsonl(output_dir / 'row_text_margin_diagnostics.jsonl', stage_row_diagnostics[config.stage])
    _write_jsonl(output_dir / 'carrier_text_contrast_diagnostics.jsonl', direct_contrast_rows)
    _write_csv(output_dir / 'pair_text_projection_summary.csv', pair_rows)
    _write_csv(output_dir / 'class_text_hubness_summary.csv', class_text_rows)
    _write_csv(output_dir / 'class_centroid_summary.csv', class_centroid_rows)
    stage_comparison_rows = []
    for stage, summary in stage_row_summaries.items():
        row = {'stage': stage}
        row.update(summary)
        row.update({f'checkpoint_{k}': v for k, v in dict(checkpoint_meta.get(stage, {})).items()})
        stage_comparison_rows.append(row)
    _write_csv(output_dir / 'stage_comparison_summary.csv', stage_comparison_rows)

    summary = {
        'status': 'PASS',
        'run_root': str(config.run_root),
        'dataset_name': config.dataset_name,
        'stage': config.stage,
        'compare_stages': list(projected_by_stage.keys()),
        'row_count': int(len(rows)),
        'split_filter': list(config.splits),
        'hub_raw_ids': list(config.hub_raw_ids),
        'asset_root': str(asset_root),
        'text_vocab_root': str(text_root),
        'materialization': meta,
        'checkpoint_meta': checkpoint_meta,
        'stage_row_text_margin_summary': stage_row_summaries,
        'class_text_hubness_summary': class_text_summary,
        'carrier_text_direct_contrast_summary': direct_contrast_summary,
        'carrier_separability_reference': carrier_ref,
        'interpretation': {
            'verdict': verdict,
            'main_question': 'Are projected text anchors / carrier-to-text scores more hub-biased than carrier geometry itself?',
            'key_rates': {
                'carrier_positive_margin_gt_vs_nearest_hub_rate_from_reference': carrier_ref.get('overall_positive_margin_gt_vs_nearest_hub_rate'),
                f'{config.stage}_positive_text_margin_gt_vs_nearest_hub_rate': stage_row_summaries[config.stage].get('positive_text_margin_gt_vs_nearest_hub_rate'),
                f'{config.stage}_text_top1_is_hub_rate': stage_row_summaries[config.stage].get('text_top1_is_hub_rate'),
                'carrier_positive_but_text_negative_rate': direct_contrast_summary.get('carrier_positive_but_text_negative_rate'),
                'raw_text_top1_neighbor_is_hub_rate': class_text_summary.get('raw_text_top1_neighbor_is_hub_rate'),
                f'{config.stage}_projected_text_any_top5_neighbor_is_hub_rate': class_text_summary.get(f'{config.stage}_projected_text_any_top5_neighbor_is_hub_rate'),
            },
            'decision_matrix': {
                'projector_hub_pull_if': 'projected GT-hub text cosine increases over raw CLIP text cosine, especially for hub-affected GT classes',
                'text_scoring_hub_bias_if': 'carrier GT-vs-hub margin is often positive but carrier-to-projected-text GT-vs-hub margin is often non-positive',
                'aggregation_estep_if': 'both carrier and carrier-to-text margins are nontrivial but final top1/extra admission still fails',
            },
        },
        'output_files': {
            'summary': str(output_dir / 'summary.json'),
            'takeover': str(output_dir / 'TEXT_PROJECTOR_HUBNESS_TAKEOVER.md'),
            'row_text_margin_diagnostics': str(output_dir / 'row_text_margin_diagnostics.jsonl'),
            'carrier_text_contrast_diagnostics': str(output_dir / 'carrier_text_contrast_diagnostics.jsonl'),
            'pair_text_projection_summary': str(output_dir / 'pair_text_projection_summary.csv'),
            'class_text_hubness_summary': str(output_dir / 'class_text_hubness_summary.csv'),
            'stage_comparison_summary': str(output_dir / 'stage_comparison_summary.csv'),
        },
    }
    _write_json(output_dir / 'summary.json', summary)

    md = [
        '# Text Projector Hubness Audit Takeover',
        '',
        '## Scope',
        f'- dataset: `{config.dataset_name}`',
        f'- stage: `{config.stage}`',
        f'- splits: `{",".join(config.splits)}`',
        f'- hub_raw_ids: `{list(config.hub_raw_ids)}`',
        '',
        '## Status',
        '- status: `PASS`',
        f'- usable rows: `{len(rows)}`',
        f'- interpretation verdict: `{verdict}`',
        '',
        '## Key rates',
    ]
    key_rates = summary['interpretation']['key_rates']
    for k, v in key_rates.items():
        md.append(f'- {k}: `{v}`')
    md.extend([
        '',
        '## Reading guide',
        '- If carrier-positive/text-negative is high, carrier geometry contains GT-vs-hub signal but projected text scoring is hub-biased.',
        '- If projected GT-hub text cosine rises over raw CLIP text cosine, the text projector may be pulling impacted classes toward hubs.',
        '- If projected scoring is not hub-biased but final top1 is still poor, prioritize clip-level aggregation, slot allocation, and E-step conversion.',
        '',
        '## Output files',
    ])
    for k, v in summary['output_files'].items():
        md.append(f'- {k}: `{v}`')
    _write_md(output_dir / 'TEXT_PROJECTOR_HUBNESS_TAKEOVER.md', md)

    print(json.dumps({'status': 'PASS', 'row_count': len(rows), 'output_dir': str(output_dir), 'verdict': verdict}, ensure_ascii=False, indent=2), flush=True)
    return summary


def _self_test() -> int:
    rng = np.random.default_rng(3407)
    X = rng.normal(size=(16, 8)).astype(np.float32)
    X = _l2_normalize_np(X, axis=1)
    Y = rng.normal(size=(6, 8)).astype(np.float32)
    Y = _l2_normalize_np(Y, axis=1)
    assert X.shape == (16, 8)
    assert Y.shape == (6, 8)
    assert abs(float(np.linalg.norm(X[0])) - 1.0) < 1e-5
    print(json.dumps({'status': 'PASS', 'self_test': 'synthetic_l2_normalization'}, indent=2), flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Read-only text-projector / prototype hubness audit for WSOVVIS Stage B.')
    parser.add_argument('--self_test', action='store_true', help='run a tiny synthetic self-test and exit')
    parser.add_argument('--run_root', type=Path, default=Path('codex/outputs/G8_inference_and_eval/extra_consensus_aggressive_20260424/K3_cons_l015_penalty050'))
    parser.add_argument('--dataset_name', type=str, default='lvvis_train_base')
    parser.add_argument('--stage', type=str, default='softem_aug', choices=tuple(STAGE_CHECKPOINTS.keys()))
    parser.add_argument('--compare_stages', type=str, default='prealign,softem_base,softem_aug')
    parser.add_argument('--trajectory_source_branch', type=str, default='mainline')
    parser.add_argument('--output_dir', type=Path, default=None)
    parser.add_argument('--splits', type=str, default='base_unobserved')
    parser.add_argument('--hub_raw_ids', type=str, default=str(DEFAULT_PERSON_RAW_ID))
    parser.add_argument('--min_class_count', type=int, default=3)
    parser.add_argument('--max_rows', type=int, default=0)
    parser.add_argument('--top_pairs', type=int, default=200)
    parser.add_argument('--all_gt_generate_sidecars_if_missing', type=_parse_bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size_rows', type=int, default=512)
    parser.add_argument('--show_progress', type=_parse_bool, default=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if bool(args.self_test):
        return _self_test()
    config = AuditConfig(
        run_root=Path(args.run_root).expanduser().resolve(),
        dataset_name=str(args.dataset_name),
        stage=str(args.stage),
        compare_stages=_parse_str_tuple(str(args.compare_stages)),
        trajectory_source_branch=str(args.trajectory_source_branch),
        output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
        splits=_parse_str_tuple(str(args.splits)),
        hub_raw_ids=_parse_int_tuple(str(args.hub_raw_ids)),
        min_class_count=int(args.min_class_count),
        max_rows=int(args.max_rows),
        top_pairs=int(args.top_pairs),
        all_gt_generate_sidecars_if_missing=bool(args.all_gt_generate_sidecars_if_missing),
        device=str(args.device),
        batch_size_rows=max(1, int(args.batch_size_rows)),
        show_progress=bool(args.show_progress),
    )
    run_audit(config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
