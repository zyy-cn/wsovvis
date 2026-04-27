from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Literal

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm as _tqdm_cls
except Exception:  # pragma: no cover - tqdm is optional in smoke environments
    _tqdm_cls = None

from videocutler.ext_stageb_ovvis.banks.responsibility_cache import ResponsibilityCache
from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    build_stage_domain_indices,
    score_carrier_logits,
    score_carrier_logits_torch,
    load_carrier_evidence,
    load_text_vocab,
    refine_responsibilities,
)
from videocutler.ext_stageb_ovvis.algorithms._memory_audit import memory_checkpoint, shallow_size_bytes, timing_checkpoint
from videocutler.ext_stageb_ovvis.banks.text_bank import resolve_text_prototype
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig
from videocutler.ext_stageb_ovvis.algorithms._training_budget import build_dynamic_microbatches, resolve_default_batch_budget


Record = Dict[str, Any]

UNKNOWN_INIT_MODE_LEGACY = 'legacy'
UNKNOWN_INIT_MODE_PRIOR = 'unknown_prior'
UNKNOWN_INIT_MODE_CHOICES = (UNKNOWN_INIT_MODE_LEGACY, UNKNOWN_INIT_MODE_PRIOR)
UNKNOWN_INIT_PRIOR_ALPHA = 0.2


@dataclass(frozen=True)
class SoftEMStageConfig:
    stage_id: str
    selected_for_infer: str
    checkpoint_name: str
    responsibility_relpath: str
    train_state_relpath: str
    learning_rate: float
    epochs: int


@dataclass(frozen=True)
class SoftEMConfig:
    dataset_name: str
    trajectory_source_branch: str = 'mainline'
    mode: str = 'base_then_aug'
    device: str = 'cpu'
    seed: int = 0
    smoke: bool = False
    lambda_frame: float = 0.25
    lambda_cov: float = 1.0
    t_dis_init: float = 0.07
    b_u_init: float = 0.0
    weight_decay: float = 1e-2
    em_subiterations: int = 2
    projector: ProjectorConfig = ProjectorConfig()
    base_epochs: int = 1
    aug_epochs: int = 1
    base_learning_rate: float = 5e-5
    aug_learning_rate: float = 5e-5
    k_extra: int = 2
    extra_alpha: float = 0.25
    extra_refresh_interval_iters: Optional[int] = None
    runtime_asset_source: str = 'local_canonical_assets'
    runtime_asset_source_local_incomplete: bool = False
    runtime_asset_output_root: str = ''
    batch_budget: int | None = None
    show_progress: bool = True
    log_every: int = 10
    write_runtime_metrics_jsonl: bool = True
    print_epoch_summary: bool = True
    unknown_init_mode: str = 'legacy'


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_jsonl(path: Path) -> List[Record]:
    rows: List[Record] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _append_jsonl(path: Path, row: Record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _make_progress_bar(*, total: int, desc: str, enabled: bool):
    if enabled and _tqdm_cls is not None:
        return _tqdm_cls(total=max(1, int(total)), unit='batch', dynamic_ncols=True, leave=True, desc=desc)

    class _SilentProgress:
        def __init__(self, total: int, desc: str) -> None:
            self.total = max(1, int(total))
            self.desc = desc
            self.n = 0

        def __enter__(self) -> '_SilentProgress':
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def update(self, n: int = 1) -> None:
            self.n += int(n)

        def set_postfix(self, values: Optional[Dict[str, Any]] = None, refresh: bool = True) -> None:  # noqa: ARG002
            return None

    return _SilentProgress(total=total, desc=desc)


def _should_log_microbatch(log_every: int, microbatch_index: int, total_microbatches: int) -> bool:
    if int(log_every) <= 0:
        return False
    if int(microbatch_index) <= 1:
        return True
    if int(microbatch_index) >= int(total_microbatches):
        return True
    return int(microbatch_index) % int(log_every) == 0


def _runtime_metrics_path(output_root: Path, stage_id: str) -> Path:
    return output_root / 'train' / stage_id / 'runtime_metrics.jsonl'


def _quantile_snapshot(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size <= 0:
        return {
            'min': 0.0,
            'p10': 0.0,
            'p50': 0.0,
            'p90': 0.0,
            'max': 0.0,
        }
    return {
        'min': float(np.min(arr)),
        'p10': float(np.quantile(arr, 0.10)),
        'p50': float(np.quantile(arr, 0.50)),
        'p90': float(np.quantile(arr, 0.90)),
        'max': float(np.max(arr)),
    }


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(list(values), dtype=np.float32))) if values else 0.0


def _format_epoch_summary(stage_id: str, summary: Mapping[str, Any]) -> str:
    keys = [
        'epoch',
        'microbatch_count',
        'loss_mean',
        'loss_last',
        'optimization_loss_mean',
        'optimization_loss_last',
        'effective_responsibility_unit_count_total',
        'effective_responsibility_unit_count_mean',
        'unknown_mean_responsibility_epoch',
        'observed_mean_responsibility_epoch',
        'responsibility_entropy_epoch',
    ]
    if str(stage_id) == 'softem_aug':
        keys.insert(11, 'extra_mean_responsibility_epoch')
    keys.extend([
        'unknown_resp_min',
        'unknown_resp_p10',
        'unknown_resp_p50',
        'unknown_resp_p90',
        'unknown_resp_max',
        'observed_resp_min',
        'observed_resp_p10',
        'observed_resp_p50',
        'observed_resp_p90',
        'observed_resp_max',
    ])
    if str(stage_id) == 'softem_aug':
        keys.extend([
            'extra_resp_min',
            'extra_resp_p10',
            'extra_resp_p50',
            'extra_resp_p90',
            'extra_resp_max',
        ])
    keys.extend([
        'responsibility_entropy_min',
        'responsibility_entropy_p10',
        'responsibility_entropy_p50',
        'responsibility_entropy_p90',
        'responsibility_entropy_max',
    ])
    parts = [f'[{stage_id}] epoch_summary']
    for key in keys:
        if key in summary:
            parts.append(f'{key}={summary[key]}')
    return ' '.join(parts)


def _prepare_examples(
    materialized_samples: Sequence[Record],
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
) -> Dict[str, Any]:
    examples: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    text_records_path = output_root / 'text_bank' / 'text_prototype_records.jsonl'

    def bump(reason: str) -> None:
        skipped[reason] = int(skipped.get(reason, 0)) + 1

    for sample in materialized_samples:
        if not bool(sample.get('sample_valid', False)):
            bump('sample_not_valid_from_phase1')
            continue
        try:
            memory_checkpoint(
                "softem_prepare_before_load_combined_evidence",
                trajectory_id=str(sample.get('trajectory_id', '')),
                observed_raw_ids=len(list(sample.get('observed_raw_ids', []))),
            )
            carrier_vec = load_carrier_evidence(
                sample,
                output_root=output_root,
                dataset_name=dataset_name,
                trajectory_source_branch=trajectory_source_branch,
            )
            memory_checkpoint(
                "softem_prepare_after_load_combined_evidence",
                trajectory_id=str(sample.get('trajectory_id', '')),
                carrier_vec_shallow_size=shallow_size_bytes(carrier_vec),
            )
        except Exception:
            bump('missing_carrier_evidence')
            continue
        candidate_records = list(sample.get('candidate_text_prototypes', []))
        if not candidate_records:
            bump('empty_candidate_text_prototypes')
            continue
        candidate_ids_known = [int(x) for x in list(sample.get('candidate_ids_known', []))]
        candidate_ids_extra = [int(x) for x in list(sample.get('candidate_ids_extra', []))]
        if len(candidate_ids_known) + len(candidate_ids_extra) != len(candidate_records):
            bump('candidate_id_vector_length_mismatch')
            continue
        observed_set = {int(x) for x in list(sample.get('observed_raw_ids', []))}
        if not candidate_ids_known:
            bump('empty_candidate_ids_known')
            continue
        try:
            candidate_matrix = [resolve_text_prototype(text_records_path, rec) for rec in candidate_records]
        except Exception:
            bump('invalid_text_prototype_locator')
            continue
        examples.append(
            {
                'trajectory_id': str(sample.get('trajectory_id', '')),
                'clip_id': int(sample.get('clip_id', -1)),
                'video_id': int(sample.get('trajectory_record', {}).get('video_id', -1)),
                'observed_raw_ids': sorted(observed_set),
                'candidate_ids_known': candidate_ids_known,
                'candidate_ids_extra': candidate_ids_extra,
                'candidate_matrix': np.asarray(candidate_matrix, dtype=np.float32),
                'carrier_vec': np.asarray(carrier_vec, dtype=np.float32),
                'candidate_records': candidate_records,
            }
        )
    return {'examples': examples, 'skipped_reason_histogram': skipped}


def _inverse_softplus(value: float) -> float:
    target = max(float(value), 1e-6)
    return float(math.log(math.expm1(target)))


def _compute_t_dis(theta_t: torch.nn.Parameter) -> torch.Tensor:
    return F.softplus(theta_t) + 1e-4



def _load_projector_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> Tuple[Projector, torch.nn.Parameter, torch.nn.Parameter, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'text_projector_state_dict' not in checkpoint or 'text_projector_config' not in checkpoint:
        raise RuntimeError(
            f'incompatible checkpoint at {checkpoint_path}: expected text_projector_state_dict/text_projector_config under the new authority'
        )
    config_payload = dict(checkpoint.get('text_projector_config', {}))
    projector = Projector(
        ProjectorConfig(
            input_dim=int(config_payload.get('input_dim', 512)),
            hidden_dim=int(config_payload.get('hidden_dim', 1024)),
            output_dim=int(config_payload.get('output_dim', 768)),
            dropout=float(config_payload.get('dropout', 0.0)),
            use_layernorm=bool(config_payload.get('use_layernorm', True)),
        )
    ).to(device)
    projector.load_state_dict(checkpoint['text_projector_state_dict'])
    theta_t = torch.nn.Parameter(torch.tensor(float(checkpoint.get('theta_T', _inverse_softplus(0.07))), device=device, dtype=torch.float32))
    b_u = torch.nn.Parameter(torch.tensor(float(checkpoint.get('b_u', 0.0)), device=device, dtype=torch.float32))
    return projector, theta_t, b_u, checkpoint


def _normalize_mass(mass: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total = 0.0
    for key, value in mass.items():
        v = max(0.0, float(value))
        out[str(key)] = v
        total += v
    if total <= 0.0:
        return {'unknown': 1.0}
    return {key: float(value / total) for key, value in out.items()}


def _stage_cfg(config: SoftEMConfig) -> List[SoftEMStageConfig]:
    if config.mode == 'base_only':
        return [
            SoftEMStageConfig(
                stage_id='softem_base',
                selected_for_infer='base_only',
                checkpoint_name='softem_base_last.pth',
                responsibility_relpath='train/softem_base/responsibility_records.jsonl',
                train_state_relpath='train/softem_base/train_state.json',
                learning_rate=float(config.base_learning_rate),
                epochs=int(config.base_epochs),
            )
        ]
    if config.mode == 'aug_only':
        return [
            SoftEMStageConfig(
                stage_id='softem_aug',
                selected_for_infer='augmented',
                checkpoint_name='softem_aug_last.pth',
                responsibility_relpath='train/softem_aug/responsibility_records.jsonl',
                train_state_relpath='train/softem_aug/train_state.json',
                learning_rate=float(config.aug_learning_rate),
                epochs=int(config.aug_epochs),
            )
        ]
    if config.mode == 'base_then_aug':
        return [
            SoftEMStageConfig(
                stage_id='softem_base',
                selected_for_infer='base_only',
                checkpoint_name='softem_base_last.pth',
                responsibility_relpath='train/softem_base/responsibility_records.jsonl',
                train_state_relpath='train/softem_base/train_state.json',
                learning_rate=float(config.base_learning_rate),
                epochs=int(config.base_epochs),
            ),
            SoftEMStageConfig(
                stage_id='softem_aug',
                selected_for_infer='augmented',
                checkpoint_name='softem_aug_last.pth',
                responsibility_relpath='train/softem_aug/responsibility_records.jsonl',
                train_state_relpath='train/softem_aug/train_state.json',
                learning_rate=float(config.aug_learning_rate),
                epochs=int(config.aug_epochs),
            ),
        ]
    raise ValueError(f'unsupported soft-em mode: {config.mode}')


def _stage_domain_for_stage(stage_id: str, example: Mapping[str, Any]) -> Tuple[List[int], List[int], List[int]]:
    return build_stage_domain_indices(
        example.get('candidate_ids_known', []),
        example.get('candidate_ids_extra', []),
        stage_id=stage_id,
    )


def _build_runtime_extra_cache(
    *,
    examples: Sequence[Mapping[str, Any]],
    text_projector: Projector,
    theta_t: torch.nn.Parameter,
    output_root: Path,
    k_extra: int,
    alpha: float,
    lambda_frame: float,
    device: torch.device,
    extra_margin_gate: float | None = None,
    allowed_extra_raw_ids: Optional[Iterable[int]] = None,
    extra_vocab_scope_policy: str = 'legacy_full',
    strict_check: bool = False,
) -> Dict[int, Dict[str, Any]]:
    if int(k_extra) <= 0:
        return {}
    audit_t0 = time.perf_counter()
    full_vocab_ids, vocab_records, full_vocab_matrix = load_text_vocab(output_root)
    full_idx_by_raw = {int(raw_id): idx for idx, raw_id in enumerate(full_vocab_ids)}
    if allowed_extra_raw_ids is None:
        vocab_ids = [int(raw_id) for raw_id in full_vocab_ids]
    else:
        allowed_set = {int(raw_id) for raw_id in allowed_extra_raw_ids}
        missing_allowed = sorted(raw_id for raw_id in allowed_set if raw_id not in full_idx_by_raw)
        if missing_allowed:
            raise KeyError(f'allowed extra raw ids missing from text bank: {missing_allowed[:16]}')
        vocab_ids = [int(raw_id) for raw_id in full_vocab_ids if int(raw_id) in allowed_set]
    if not vocab_ids:
        raise ValueError('runtime extra cache has empty scoped vocabulary')
    scoped_indices = [full_idx_by_raw[int(raw_id)] for raw_id in vocab_ids]
    vocab_matrix = np.asarray(full_vocab_matrix, dtype=np.float32)[scoped_indices]
    idx_by_raw = {int(raw_id): idx for idx, raw_id in enumerate(vocab_ids)}
    text_lookup = {int(rec['raw_id']): dict(rec) for rec in vocab_records}
    t_dis = _compute_t_dis(theta_t)
    with torch.no_grad():
        text_tensor = torch.from_numpy(np.asarray(vocab_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
        projected_text = F.normalize(text_projector(text_tensor), p=2.0, dim=-1)
        text_text_sim = torch.matmul(projected_text, projected_text.t()).detach().cpu().numpy().astype(np.float32)
        examples_by_clip: Dict[int, List[Tuple[int, Mapping[str, Any]]]] = {}
        for row_idx, ex in enumerate(examples):
            examples_by_clip.setdefault(int(ex['clip_id']), []).append((row_idx, ex))
    memory_checkpoint(
        "softem_before_runtime_extra_cache_build",
        example_count=len(examples),
        clip_group_count=len(examples_by_clip),
        vocab_size=len(vocab_ids),
        text_text_sim_shape=getattr(text_text_sim, "shape", None),
    )
    timing_checkpoint(
        "softem_before_runtime_extra_cache_build",
        started_at=audit_t0,
        example_count=len(examples),
        clip_group_count=len(examples_by_clip),
        vocab_size=len(vocab_ids),
        text_text_sim_shape=getattr(text_text_sim, "shape", None),
    )
    cache: Dict[int, Dict[str, Any]] = {}
    for clip_id, grouped in examples_by_clip.items():
        observed = sorted({int(x) for _, ex in grouped for x in list(ex.get('observed_raw_ids', []))})
        observed_set = set(observed)
        outside_observed = sorted(raw_id for raw_id in observed_set if raw_id not in idx_by_raw)
        if outside_observed and bool(strict_check):
            raise RuntimeError(f'extra vocab scope violation: observed ids outside scoped training vocab: {outside_observed[:16]}')
        candidate_ids = [int(raw_id) for raw_id in vocab_ids if int(raw_id) not in observed_set]
        if not candidate_ids:
            cache[int(clip_id)] = {
                'candidate_ids_extra': [],
                'candidate_ids_extra_provenance': [],
                'candidate_ids_extra_runtime_authoritative': [],
                'candidate_ids_extra_authority': 'runtime_refresh_cache_only',
                'text_lookup': text_lookup,
                'extra_vocab_scope_policy': str(extra_vocab_scope_policy),
                'extra_allowed_vocab_count': int(len(vocab_ids)),
                'candidate_ids_extra_outside_scope_count': 0,
            }
            continue
        row_indices = [row_idx for row_idx, _ in grouped]
        clip_fused_rows: List[np.ndarray] = []
        for _, ex in grouped:
            fused_logits = score_carrier_logits(
                projector=text_projector,
                carrier_vec=np.asarray(ex['carrier_vec'], dtype=np.float32),
                candidate_matrix=vocab_matrix,
                temperature=t_dis,
            )
            clip_fused_rows.append(np.asarray(fused_logits, dtype=np.float32))
        score_slice = np.asarray(clip_fused_rows, dtype=np.float32)
        support_count: Dict[int, int] = {}
        non_observed_indices = [idx_by_raw[int(raw_id)] for raw_id in candidate_ids]
        for local_row in score_slice:
            local_non_observed = local_row[non_observed_indices]
            winner_raw = int(candidate_ids[int(np.argmax(local_non_observed))])
            support_count[winner_raw] = int(support_count.get(winner_raw, 0)) + 1
        per_class: List[Tuple[float, int, int]] = []
        for raw_id in candidate_ids:
            vocab_idx = idx_by_raw[int(raw_id)]
            class_scores = score_slice[:, vocab_idx]
            argmax_local = int(np.argmax(class_scores))
            s_max = float(class_scores[argmax_local])
            debias = 0.0
            if observed:
                debias = float(max(text_text_sim[vocab_idx, idx_by_raw[int(obs)]] for obs in observed if int(obs) in idx_by_raw))
            per_class.append((float(s_max - float(alpha) * debias), int(raw_id), int(row_indices[argmax_local])))
        per_class.sort(key=lambda item: (-item[0], item[1]))
        pre_gate_count = int(len(per_class))
        best_observed_score = float('-inf')
        if observed:
            observed_indices = [idx_by_raw[int(raw_id)] for raw_id in observed if int(raw_id) in idx_by_raw]
            if observed_indices:
                best_observed_score = float(np.max(score_slice[:, observed_indices]))
        gate = None if extra_margin_gate is None else float(extra_margin_gate)
        if gate is not None and gate > 0.0 and best_observed_score != float('-inf'):
            selected = [item for item in per_class if float(item[0]) - float(best_observed_score) > float(gate)]
        else:
            selected = list(per_class)
        selected = selected[: int(k_extra)]
        cache[int(clip_id)] = {
            'candidate_ids_extra': [int(raw_id) for _, raw_id, _ in selected],
            'candidate_ids_extra_provenance': [
                {
                    'raw_id': int(raw_id),
                    'score': float(score),
                    'argmax_trajectory_id': str(examples[argmax_row]['trajectory_id']),
                    'support_count': int(support_count.get(int(raw_id), 0)),
                    'admission_reason': 'fused_score_class_level_max_with_observed_neighbor_penalty',
                }
                for score, raw_id, argmax_row in selected
            ],
            'candidate_ids_extra_runtime_authoritative': [int(raw_id) for _, raw_id, _ in selected],
            'candidate_ids_extra_authority': 'runtime_refresh_cache_only',
            'text_lookup': text_lookup,
            'candidate_ids_extra_pre_gate_count': int(pre_gate_count),
            'candidate_ids_extra_retained_count': int(len(selected)),
            'sinkhorn_extra_margin_gate': None if gate is None else float(gate),
            'best_observed_score': None if best_observed_score == float('-inf') else float(best_observed_score),
            'extra_vocab_scope_policy': str(extra_vocab_scope_policy),
            'extra_allowed_vocab_count': int(len(vocab_ids)),
            'candidate_ids_extra_outside_scope_count': int(sum(1 for _, raw_id, _ in selected if int(raw_id) not in idx_by_raw)),
        }
    memory_checkpoint(
        "softem_after_runtime_extra_cache_build",
        clip_group_count=len(examples_by_clip),
        example_count=len(examples),
        runtime_extra_cache_clip_count=len(cache),
    )
    timing_checkpoint(
        "softem_after_runtime_extra_cache_build",
        started_at=audit_t0,
        clip_group_count=len(examples_by_clip),
        example_count=len(examples),
        runtime_extra_cache_clip_count=len(cache),
    )
    return cache


def _apply_runtime_extra_cache(
    examples: Sequence[Mapping[str, Any]],
    *,
    runtime_extra_cache: Mapping[int, Mapping[str, Any]],
    output_root: Path,
) -> List[Dict[str, Any]]:
    augmented: List[Dict[str, Any]] = []
    text_records_path = output_root / 'text_bank' / 'text_prototype_records.jsonl'
    for ex in examples:
        cache_entry = dict(runtime_extra_cache.get(int(ex['clip_id']), {}))
        extra_ids = [int(x) for x in list(cache_entry.get('candidate_ids_extra', []))]
        text_lookup = dict(cache_entry.get('text_lookup', {}))
        extra_records = [dict(text_lookup[int(raw_id)]) for raw_id in extra_ids if int(raw_id) in text_lookup]
        if extra_records:
            extra_matrix = np.asarray([resolve_text_prototype(text_records_path, rec) for rec in extra_records], dtype=np.float32)
            candidate_matrix = np.concatenate([np.asarray(ex['candidate_matrix'], dtype=np.float32), extra_matrix], axis=0)
        else:
            candidate_matrix = np.asarray(ex['candidate_matrix'], dtype=np.float32)
        row = dict(ex)
        row['candidate_ids_extra_phase1_placeholder'] = list(ex.get('candidate_ids_extra', []))
        row['candidate_ids_extra'] = list(extra_ids)
        row['candidate_ids_extra_runtime_authoritative'] = list(cache_entry.get('candidate_ids_extra_runtime_authoritative', extra_ids))
        row['candidate_ids_extra_authority'] = str(cache_entry.get('candidate_ids_extra_authority', 'runtime_refresh_cache_only'))
        row['candidate_records'] = [*list(ex['candidate_records']), *extra_records]
        row['candidate_matrix'] = candidate_matrix
        row['candidate_ids_extra_provenance'] = list(cache_entry.get('candidate_ids_extra_provenance', []))
        row['candidate_ids_extra_pre_gate_count'] = int(cache_entry.get('candidate_ids_extra_pre_gate_count', len(extra_ids)))
        row['candidate_ids_extra_retained_count'] = int(cache_entry.get('candidate_ids_extra_retained_count', len(extra_ids)))
        row['sinkhorn_extra_margin_gate'] = cache_entry.get('sinkhorn_extra_margin_gate')
        row['best_observed_score'] = cache_entry.get('best_observed_score')
        augmented.append(row)
    return augmented


def _load_proxy_rows(output_root: Path) -> List[Record]:
    proxy_path = output_root / 'train' / 'prealign' / 'proxy_records.jsonl'
    if not proxy_path.is_file():
        raise FileNotFoundError('missing prealign proxy records: train/prealign/proxy_records.jsonl')
    return _load_jsonl(proxy_path)


def _initial_checkpoint_path(output_root: Path, *, mode: str) -> Path:
    if mode == 'aug_only':
        aug_path = output_root / 'train' / 'softem_aug' / 'checkpoints' / 'softem_aug_last.pth'
        if aug_path.is_file():
            return aug_path
    return output_root / 'train' / 'prealign' / 'checkpoints' / 'prealign_last.pth'


def _project_init_mass_to_domain(init_mass: Mapping[str, float], domain_ids: Sequence[int]) -> Dict[str, float]:
    projected = {'unknown': float(init_mass.get('unknown', 0.0))}
    for raw_id in domain_ids:
        projected[str(int(raw_id))] = float(init_mass.get(str(int(raw_id)), 0.0))
    return _normalize_mass(projected)


def _apply_unknown_prior_to_init_mass(
    projected_init_mass: Mapping[str, float],
    *,
    alpha: float = UNKNOWN_INIT_PRIOR_ALPHA,
) -> Dict[str, float]:
    bounded_alpha = min(max(float(alpha), 0.0), 1.0)
    if bounded_alpha <= 0.0:
        return _normalize_mass(projected_init_mass)
    remixed = {str(key): (1.0 - bounded_alpha) * float(value) for key, value in projected_init_mass.items()}
    remixed['unknown'] = float(remixed.get('unknown', 0.0) + bounded_alpha)
    return _normalize_mass(remixed)


def _explicit_init_mass_for_aug_stage(
    *,
    domain_ids: Sequence[int],
    known_ids: Sequence[int],
    extra_ids: Sequence[int],
    stage_logits_candidates: torch.Tensor,
    b_u: torch.nn.Parameter,
    delta_extra: float = 0.1,
) -> Dict[str, float]:
    ordered_scores: List[float] = [float(b_u.detach().cpu().item())]
    ordered_keys: List[str] = ['unknown']
    domain_id_list = [int(raw_id) for raw_id in domain_ids]
    known_set = {int(raw_id) for raw_id in known_ids}
    extra_set = {int(raw_id) for raw_id in extra_ids}
    for idx, raw_id in enumerate(domain_id_list):
        score = float(stage_logits_candidates[idx].detach().cpu().item())
        if int(raw_id) in extra_set and int(raw_id) not in known_set:
            score -= float(delta_extra)
        ordered_scores.append(score)
        ordered_keys.append(str(int(raw_id)))
    probs = torch.softmax(torch.tensor(ordered_scores, dtype=torch.float64), dim=0).cpu().numpy().astype(np.float64)
    return _normalize_mass({key: float(prob) for key, prob in zip(ordered_keys, probs.tolist())})


def _compute_initial_mass_for_stage(
    *,
    stage_id: str,
    base_cache: ResponsibilityCache,
    trajectory_id: str,
    domain_ids: Sequence[int],
    known_ids: Sequence[int],
    extra_ids: Sequence[int],
    stage_logits_candidates: torch.Tensor,
    b_u: torch.nn.Parameter,
    unknown_init_mode: str = UNKNOWN_INIT_MODE_LEGACY,
) -> Dict[str, float]:
    if str(stage_id) == 'softem_aug' and len(extra_ids) > 0:
        return _explicit_init_mass_for_aug_stage(
            domain_ids=domain_ids,
            known_ids=known_ids,
            extra_ids=extra_ids,
            stage_logits_candidates=stage_logits_candidates,
            b_u=b_u,
        )
    projected = _project_init_mass_to_domain(base_cache.get_init_mass(trajectory_id), domain_ids)
    if str(stage_id) == 'softem_base' and str(unknown_init_mode) == UNKNOWN_INIT_MODE_PRIOR:
        return _apply_unknown_prior_to_init_mass(projected, alpha=UNKNOWN_INIT_PRIOR_ALPHA)
    return projected


def _build_clip_coverage_context(current_masses_by_tid: Mapping[str, Mapping[str, float]], clip_examples: Sequence[Mapping[str, Any]], known_ids: Sequence[int]) -> Dict[str, float]:
    context: Dict[str, float] = {}
    for raw_id in known_ids:
        key = str(int(raw_id))
        total = 0.0
        for ex in clip_examples:
            total += float(current_masses_by_tid[str(ex['trajectory_id'])].get(key, 0.0))
        context[key] = float(total)
    return context


def _compute_clip_refinement_rows(
    *,
    stage_id: str,
    unknown_init_mode: str,
    clip_examples: Sequence[Mapping[str, Any]],
    base_cache: ResponsibilityCache,
    text_projector: Projector,
    theta_t: torch.nn.Parameter,
    b_u: torch.nn.Parameter,
    em_subiterations: int,
    lambda_frame: float,
    lambda_cov: float = 1.0,
    device: torch.device,
) -> Tuple[List[Record], List[Dict[str, Any]]]:
    t_dis = _compute_t_dis(theta_t)
    per_tid_model_logits: Dict[str, np.ndarray] = {}
    per_tid_domain_ids: Dict[str, List[int]] = {}
    per_tid_known_ids: Dict[str, List[int]] = {}
    per_tid_extra_ids: Dict[str, List[int]] = {}
    initial_masses_by_tid: Dict[str, Dict[str, float]] = {}
    trace_by_tid: Dict[str, List[Dict[str, Any]]] = {str(ex['trajectory_id']): [] for ex in clip_examples}

    with torch.no_grad():
        for ex in clip_examples:
            tid = str(ex['trajectory_id'])
            domain_ids, known_ids, extra_ids = build_stage_domain_indices(ex.get('candidate_ids_known', []), ex.get('candidate_ids_extra', []), stage_id=stage_id)
            logits_known_extra = score_carrier_logits_torch(
                projector=text_projector,
                carrier_vec=ex['carrier_vec'],
                candidate_matrix=ex['candidate_matrix'],
                temperature=t_dis,
            )
            stage_logits_candidates = logits_known_extra[: len(domain_ids)]
            model_logits = stage_logits_candidates.detach().cpu().numpy().astype(np.float64)
            per_tid_model_logits[tid] = model_logits
            per_tid_domain_ids[tid] = list(domain_ids)
            per_tid_known_ids[tid] = list(known_ids)
            per_tid_extra_ids[tid] = list(extra_ids)
            initial_masses_by_tid[tid] = _compute_initial_mass_for_stage(
                stage_id=stage_id,
                base_cache=base_cache,
                trajectory_id=tid,
                domain_ids=domain_ids,
                known_ids=known_ids,
                extra_ids=extra_ids,
                stage_logits_candidates=stage_logits_candidates,
                b_u=b_u,
                unknown_init_mode=str(unknown_init_mode),
            )

    current_masses_by_tid = {tid: dict(mass) for tid, mass in initial_masses_by_tid.items()}
    b_u_value = float(b_u.detach().cpu().item())
    subiterations = max(1, int(em_subiterations))

    for subiter_idx in range(subiterations):
        next_masses_by_tid: Dict[str, Dict[str, float]] = {}
        for ex in clip_examples:
            tid = str(ex['trajectory_id'])
            coverage_context = _build_clip_coverage_context(
                current_masses_by_tid=current_masses_by_tid,
                clip_examples=clip_examples,
                known_ids=per_tid_known_ids[tid],
            )
            refined_init, refined_final, refine_trace = refine_responsibilities(
                initial_mass=current_masses_by_tid[tid],
                model_logits=per_tid_model_logits[tid],
                candidate_ids_known=per_tid_known_ids[tid],
                candidate_ids_extra=per_tid_extra_ids[tid],
                stage_id=stage_id,
                coverage_bonus=0.1 * float(lambda_cov),
                coverage_epsilon=1.0,
                extra_penalty=0.1 * float(lambda_cov),
                coverage_context=coverage_context,
                b_u_value=b_u_value,
            )
            trace_by_tid[tid].append(
                {
                    'subiteration_index': int(subiter_idx),
                    'r_init': dict(refined_init),
                    'r_final': dict(refined_final),
                    'coverage_context': dict(coverage_context),
                    'coverage_bonus_applied_to': list(refine_trace.get('coverage_bonus_applied_to', [])),
                    'refine_trace': dict(refine_trace),
                }
            )
            next_masses_by_tid[tid] = dict(refined_final)
        current_masses_by_tid = next_masses_by_tid

    rows: List[Record] = []
    for ex in clip_examples:
        tid = str(ex['trajectory_id'])
        rows.append(
            {
                'dataset_name': str(ex.get('dataset_name', 'lvvis_train_base')),
                'clip_id': int(ex['clip_id']),
                'video_id': int(ex['video_id']),
                'trajectory_id': tid,
                'candidate_ids_known': [int(x) for x in ex['candidate_ids_known']],
                'candidate_ids_extra': [int(x) for x in ex['candidate_ids_extra']],
                'unknown_slot': 'unknown',
                'r_init': dict(initial_masses_by_tid[tid]),
                'r_final': dict(current_masses_by_tid[tid]),
                'coverage_bonus_applied_to': list(trace_by_tid[tid][-1]['coverage_bonus_applied_to']) if trace_by_tid[tid] else [],
                'refine_trace': dict(trace_by_tid[tid][-1]['refine_trace']) if trace_by_tid[tid] else {},
                'em_subiterations': int(subiterations),
                'em_subiteration_count': int(len(trace_by_tid[tid])),
                'join_key': tid,
                'subiteration_trace': list(trace_by_tid[tid]),
            }
        )
    sample_trace = list(trace_by_tid[str(clip_examples[0]['trajectory_id'])]) if clip_examples else []
    return rows, sample_trace


def _refresh_stage_runtime_state(
    *,
    stage_id: str,
    unknown_init_mode: str,
    base_examples: Sequence[Mapping[str, Any]],
    base_cache: ResponsibilityCache,
    text_projector: Projector,
    theta_t: torch.nn.Parameter,
    b_u: torch.nn.Parameter,
    output_root: Path,
    k_extra: int,
    extra_alpha: float,
    lambda_frame: float,
    lambda_cov: float,
    em_subiterations: int,
    device: torch.device,
) -> Tuple[Dict[str, Dict[str, Any]], ResponsibilityCache, Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    runtime_extra_cache: Dict[int, Dict[str, Any]] = {}
    stage_examples = list(base_examples)
    if str(stage_id) == 'softem_aug':
        runtime_extra_cache = _build_runtime_extra_cache(
            examples=base_examples,
            text_projector=text_projector,
            theta_t=theta_t,
            output_root=output_root,
            k_extra=int(k_extra),
            alpha=float(extra_alpha),
            lambda_frame=float(lambda_frame),
            device=device,
        )
        stage_examples = _apply_runtime_extra_cache(base_examples, runtime_extra_cache=runtime_extra_cache, output_root=output_root)

    stage_examples_by_clip: Dict[int, List[Dict[str, Any]]] = {}
    stage_examples_by_tid: Dict[str, Dict[str, Any]] = {}
    for ex in stage_examples:
        stage_examples_by_clip.setdefault(int(ex['clip_id']), []).append(dict(ex))
        stage_examples_by_tid[str(ex['trajectory_id'])] = dict(ex)

    refreshed_rows: List[Record] = []
    stage_trace_sample: List[Dict[str, Any]] = []
    for clip_id in sorted(stage_examples_by_clip.keys()):
        rows, sample_trace = _compute_clip_refinement_rows(
            stage_id=stage_id,
            unknown_init_mode=str(unknown_init_mode),
            clip_examples=stage_examples_by_clip[int(clip_id)],
            base_cache=base_cache,
            text_projector=text_projector,
            theta_t=theta_t,
            b_u=b_u,
            em_subiterations=em_subiterations,
            lambda_frame=lambda_frame,
            lambda_cov=lambda_cov,
            device=device,
        )
        refreshed_rows.extend(rows)
        if not stage_trace_sample:
            stage_trace_sample = sample_trace
    refreshed_cache = ResponsibilityCache.from_records(stage_id=str(stage_id), records=refreshed_rows)
    return stage_examples_by_tid, refreshed_cache, runtime_extra_cache, stage_trace_sample


def run_soft_em(
    *,
    output_root: Path,
    materialized_samples: Sequence[Record],
    config: SoftEMConfig,
    audit_callback: Any = None,
) -> Dict[str, Any]:
    if config.dataset_name not in {'lvvis_train_base', 'lvvis_val'}:
        raise ValueError("soft-EM implementation currently supports dataset_name=lvvis_train_base or lvvis_val")
    _set_seed(int(config.seed))
    device = torch.device(str(config.device))
    audit_t0 = time.perf_counter()
    memory_checkpoint(
        "softem_start",
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
        mode=str(config.mode),
        batch_budget=(int(config.batch_budget) if config.batch_budget is not None else None),
        lambda_frame=float(config.lambda_frame),
        lambda_cov=float(config.lambda_cov),
    )
    prepared = _prepare_examples(
        materialized_samples,
        output_root=output_root,
        dataset_name=config.dataset_name,
        trajectory_source_branch=config.trajectory_source_branch,
    )
    examples = list(prepared['examples'])
    skipped = dict(prepared['skipped_reason_histogram'])
    memory_checkpoint(
        "softem_after_prepare_examples",
        materialized_samples=len(materialized_samples),
        trainable_examples=len(examples),
        skipped_reason_histogram=skipped,
        total_candidate_rows=sum(int(np.asarray(ex.get('candidate_matrix')).shape[0]) if ex.get('candidate_matrix') is not None else 0 for ex in examples),
    )
    timing_checkpoint(
        "softem_after_prepare_examples",
        started_at=audit_t0,
        materialized_samples=len(materialized_samples),
        trainable_examples=len(examples),
        skipped_reason_histogram=skipped,
        total_candidate_rows=sum(int(np.asarray(ex.get('candidate_matrix')).shape[0]) if ex.get('candidate_matrix') is not None else 0 for ex in examples),
    )
    if not examples:
        raise RuntimeError('no trainable examples available for soft-EM')
    batch_budget = resolve_default_batch_budget(smoke=bool(config.smoke), explicit=config.batch_budget)
    proxy_rows = _load_proxy_rows(output_root)
    cache = ResponsibilityCache.from_proxy_records(proxy_rows, stage_id='prealign_proxy')
    stage_reports: List[Dict[str, Any]] = []
    current_checkpoint = _initial_checkpoint_path(output_root, mode=config.mode)
    if not current_checkpoint.is_file():
        raise FileNotFoundError(f'missing prealign checkpoint for soft-EM bootstrap: {current_checkpoint}')
    memory_checkpoint(
        "softem_before_stage_loop",
        examples=len(examples),
        batch_budget=int(batch_budget),
        initial_checkpoint=str(current_checkpoint),
    )
    timing_checkpoint(
        "softem_before_stage_loop",
        started_at=audit_t0,
        examples=len(examples),
        batch_budget=int(batch_budget),
        initial_checkpoint=str(current_checkpoint),
    )

    for stage in _stage_cfg(config):
        text_projector, theta_t, b_u, ckpt = _load_projector_from_checkpoint(current_checkpoint, device=device)
        text_projector.train()
        optimizer = torch.optim.AdamW(
            [*text_projector.parameters(), theta_t, b_u],
            lr=float(stage.learning_rate),
            weight_decay=float(config.weight_decay),
        )
        losses: List[float] = []
        optimization_losses: List[float] = []
        iteration_index = int(ckpt.get('global_step', 0))
        refresh_interval = int(config.extra_refresh_interval_iters) if config.extra_refresh_interval_iters is not None else max(len(examples), 1)
        refresh_interval = max(int(refresh_interval), 1)
        stage_trace_sample: List[Dict[str, Any]] = []

        active_examples_by_tid, cache, runtime_extra_cache, stage_trace_sample = _refresh_stage_runtime_state(
            stage_id=stage.stage_id,
            unknown_init_mode=str(config.unknown_init_mode),
            base_examples=examples,
            base_cache=cache,
            text_projector=text_projector,
            theta_t=theta_t,
            b_u=b_u,
            output_root=output_root,
            k_extra=int(config.k_extra),
            extra_alpha=float(config.extra_alpha),
            lambda_frame=float(config.lambda_frame),
            lambda_cov=float(config.lambda_cov),
            em_subiterations=max(1, int(config.em_subiterations)),
            device=device,
        )
        memory_checkpoint(
            f"softem_after_stage_refresh_{stage.stage_id}",
            stage_id=str(stage.stage_id),
            active_examples=len(active_examples_by_tid),
            runtime_extra_cache=len(runtime_extra_cache),
            cache_size=len(cache.by_trajectory_id),
            trace_sample_len=len(stage_trace_sample),
        )
        timing_checkpoint(
            f"softem_after_stage_refresh_{stage.stage_id}",
            started_at=audit_t0,
            stage_id=str(stage.stage_id),
            active_examples=len(active_examples_by_tid),
            runtime_extra_cache=len(runtime_extra_cache),
            cache_size=len(cache.by_trajectory_id),
            trace_sample_len=len(stage_trace_sample),
        )

        if audit_callback is not None:
            audit_callback(
                {
                    'dataset_name': str(config.dataset_name),
                    'trajectory_source_branch': str(config.trajectory_source_branch),
                    'stage_id': str(stage.stage_id),
                    'snapshot_id': 'stage_start',
                    'phase': 'stage_start',
                    'output_root': output_root,
                    'materialized_samples': materialized_samples,
                    'text_projector': text_projector,
                    'projector': text_projector,
                    'theta_T': theta_t,
                    'b_u': b_u,
                    'responsibility_cache': cache,
                    'device': str(device),
                    'temperature': float(_compute_t_dis(theta_t).detach().cpu().item()),
                    'seed': int(config.seed),
                    'mode': str(config.mode),
                }
            )

        since_refresh = 0
        current_refresh_interval = int(refresh_interval)
        final_epoch_plan = None
        runtime_metrics_path = _runtime_metrics_path(output_root, stage.stage_id)
        for epoch_index in range(int(stage.epochs)):
            stage_examples = [active_examples_by_tid[tid] for tid in active_examples_by_tid.keys()]
            random.Random(int(config.seed) + int(epoch_index)).shuffle(stage_examples)
            epoch_plan = build_dynamic_microbatches(
                stage_examples,
                batch_budget=batch_budget,
                cost_fn=lambda ex: len(_stage_domain_for_stage(stage.stage_id, ex)[0]),
                bucket_key_fn=lambda ex: (1, len(_stage_domain_for_stage(stage.stage_id, ex)[0])),
            )
            final_epoch_plan = epoch_plan
            current_refresh_interval = int(config.extra_refresh_interval_iters) if config.extra_refresh_interval_iters is not None else max(1, int(epoch_plan.batch_count))
            epoch_losses: List[float] = []
            epoch_batch_losses: List[float] = []
            epoch_effective_counts: List[int] = []
            epoch_unknown_responsibilities: List[float] = []
            epoch_observed_responsibilities: List[float] = []
            epoch_extra_responsibilities: List[float] = []
            epoch_entropies: List[float] = []
            progress_enabled = bool(config.show_progress)
            with _make_progress_bar(
                total=int(epoch_plan.batch_count),
                desc=f"{stage.stage_id} epoch {int(epoch_index) + 1}/{int(stage.epochs)}",
                enabled=progress_enabled,
            ) as progress:
                for microbatch_index, batch_indices in enumerate(epoch_plan.batches, start=1):
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss_accum: torch.Tensor | None = None
                    effective_responsibility_unit_count = 0
                    sample_losses: List[float] = []
                    sample_unknown_responsibilities: List[float] = []
                    sample_observed_responsibilities: List[float] = []
                    sample_extra_responsibilities: List[float] = []
                    sample_entropies: List[float] = []
                    for batch_index in batch_indices:
                        ex = stage_examples[int(batch_index)]
                        tid = str(ex['trajectory_id'])
                        domain_ids, known_ids, extra_ids = _stage_domain_for_stage(stage.stage_id, ex)
                        extra_id_set = {int(x) for x in extra_ids}
                        current_t_dis = _compute_t_dis(theta_t)
                        logits_known_extra = score_carrier_logits_torch(
                            projector=text_projector,
                            carrier_vec=ex['carrier_vec'],
                            candidate_matrix=ex['candidate_matrix'],
                            temperature=current_t_dis,
                        )
                        stage_candidate_count = len(domain_ids)
                        if stage_candidate_count <= 0:
                            raise RuntimeError(f'empty candidate domain for stage {stage.stage_id}')
                        stage_logits_candidates = logits_known_extra[:stage_candidate_count]
                        stage_logits = torch.cat([b_u.reshape(1), stage_logits_candidates], dim=0)
                        target_row = cache.by_trajectory_id[str(tid)]
                        target = torch.tensor(
                            [target_row['r_final']['unknown'], *[target_row['r_final'][str(int(raw_id))] for raw_id in domain_ids]],
                            device=device,
                            dtype=torch.float32,
                        )
                        sample_loss = -(target * torch.log_softmax(stage_logits, dim=0)).sum()
                        losses.append(float(sample_loss.detach().cpu().item()))
                        sample_loss_value = float(sample_loss.detach().cpu().item())
                        epoch_losses.append(sample_loss_value)
                        batch_loss_accum = sample_loss if batch_loss_accum is None else (batch_loss_accum + sample_loss)
                        effective_responsibility_unit_count += 1
                        sample_losses.append(sample_loss_value)
                        unknown_resp_value = float(target[0].detach().cpu().item())
                        observed_resp_value = float(target[1:].mean().detach().cpu().item()) if target.shape[0] > 1 else 0.0
                        sample_unknown_responsibilities.append(unknown_resp_value)
                        sample_observed_responsibilities.append(observed_resp_value)
                        extra_positions = [idx for idx, raw_id in enumerate(domain_ids) if int(raw_id) in extra_id_set]
                        extra_resp_value = float(target[[idx + 1 for idx in extra_positions]].mean().detach().cpu().item()) if extra_positions else 0.0
                        sample_extra_responsibilities.append(extra_resp_value)
                        entropy_value = float((-(target * torch.log(target.clamp_min(1e-12)))).sum().detach().cpu().item())
                        sample_entropies.append(entropy_value)
                        epoch_unknown_responsibilities.append(unknown_resp_value)
                        epoch_observed_responsibilities.append(observed_resp_value)
                        epoch_extra_responsibilities.append(extra_resp_value)
                        epoch_entropies.append(entropy_value)
                    if batch_loss_accum is None or effective_responsibility_unit_count <= 0:
                        continue
                    batch_loss = batch_loss_accum / float(effective_responsibility_unit_count)
                    batch_loss.backward()
                    optimizer.step()
                    batch_loss_value = float(batch_loss.detach().cpu().item())
                    optimization_losses.append(batch_loss_value)
                    epoch_batch_losses.append(batch_loss_value)
                    epoch_effective_counts.append(int(effective_responsibility_unit_count))
                    iteration_index += 1
                    since_refresh += 1
                    progress.update(1)
                    progress.set_postfix(
                        {
                            'loss': f'{float(np.mean(sample_losses)):.4f}',
                            'opt_loss': f'{batch_loss_value:.4f}',
                            'units': effective_responsibility_unit_count,
                        },
                        refresh=False,
                    )
                    if _should_log_microbatch(int(config.log_every), microbatch_index, int(epoch_plan.batch_count)):
                        base_line = (
                            f"[{stage.stage_id}] epoch={int(epoch_index) + 1}/{int(stage.epochs)} "
                            f"microbatch={microbatch_index}/{int(epoch_plan.batch_count)} "
                            f"loss={float(np.mean(sample_losses)):.6f} "
                            f"opt_loss={batch_loss_value:.6f} "
                            f"effective_responsibility_unit_count={effective_responsibility_unit_count} "
                            f"unknown_mean_responsibility={float(np.mean(sample_unknown_responsibilities)):.6f} "
                            f"observed_mean_responsibility={float(np.mean(sample_observed_responsibilities)):.6f} "
                            f"responsibility_entropy={float(np.mean(sample_entropies)):.6f}"
                        )
                        if stage.stage_id == 'softem_aug':
                            base_line += (
                                f" extra_mean_responsibility={float(np.mean(sample_extra_responsibilities)):.6f}"
                            )
                        print(base_line, file=sys.stderr, flush=True)
                    if bool(config.write_runtime_metrics_jsonl):
                        metric_row = {
                            'row_type': 'microbatch',
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'stage': str(stage.stage_id),
                            'epoch': int(epoch_index) + 1,
                            'microbatch_idx': int(microbatch_index),
                            'microbatch_total': int(epoch_plan.batch_count),
                            'loss': float(np.mean(sample_losses)),
                            'optimization_loss': float(batch_loss_value),
                            'effective_responsibility_unit_count': int(effective_responsibility_unit_count),
                            'unknown_mean_responsibility': float(np.mean(sample_unknown_responsibilities)),
                            'observed_mean_responsibility': float(np.mean(sample_observed_responsibilities)),
                            'responsibility_entropy': float(np.mean(sample_entropies)),
                        }
                        if stage.stage_id == 'softem_aug':
                            metric_row['extra_mean_responsibility'] = float(np.mean(sample_extra_responsibilities))
                        _append_jsonl(runtime_metrics_path, metric_row)
                    if since_refresh >= current_refresh_interval:
                        active_examples_by_tid, cache, runtime_extra_cache, refreshed_trace = _refresh_stage_runtime_state(
                            stage_id=stage.stage_id,
                            unknown_init_mode=str(config.unknown_init_mode),
                            base_examples=examples,
                            base_cache=cache,
                            text_projector=text_projector,
                            theta_t=theta_t,
                            b_u=b_u,
                            output_root=output_root,
                            k_extra=int(config.k_extra),
                            extra_alpha=float(config.extra_alpha),
                            lambda_frame=float(config.lambda_frame),
                            lambda_cov=float(config.lambda_cov),
                            em_subiterations=max(1, int(config.em_subiterations)),
                            device=device,
                        )
                        if not stage_trace_sample:
                            stage_trace_sample = refreshed_trace
                        since_refresh = 0
            epoch_summary = {
                'stage': str(stage.stage_id),
                'epoch': int(epoch_index) + 1,
                'microbatch_count': int(len(epoch_batch_losses)),
                'loss_mean': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                'loss_last': float(epoch_losses[-1]) if epoch_losses else 0.0,
                'optimization_loss_mean': float(np.mean(epoch_batch_losses)) if epoch_batch_losses else 0.0,
                'optimization_loss_last': float(epoch_batch_losses[-1]) if epoch_batch_losses else 0.0,
                'effective_responsibility_unit_count_total': int(np.sum(epoch_effective_counts)) if epoch_effective_counts else 0,
                'effective_responsibility_unit_count_mean': float(np.mean(epoch_effective_counts)) if epoch_effective_counts else 0.0,
                'unknown_mean_responsibility_epoch': float(np.mean(epoch_unknown_responsibilities)) if epoch_unknown_responsibilities else 0.0,
                'observed_mean_responsibility_epoch': float(np.mean(epoch_observed_responsibilities)) if epoch_observed_responsibilities else 0.0,
                'responsibility_entropy_epoch': float(np.mean(epoch_entropies)) if epoch_entropies else 0.0,
            }
            if str(stage.stage_id) == 'softem_aug':
                epoch_summary['extra_mean_responsibility_epoch'] = float(np.mean(epoch_extra_responsibilities)) if epoch_extra_responsibilities else 0.0
            unknown_quantiles = _quantile_snapshot(epoch_unknown_responsibilities)
            observed_quantiles = _quantile_snapshot(epoch_observed_responsibilities)
            entropy_quantiles = _quantile_snapshot(epoch_entropies)
            for prefix, values in (
                ('unknown_resp', unknown_quantiles),
                ('observed_resp', observed_quantiles),
                ('responsibility_entropy', entropy_quantiles),
            ):
                for key, value in values.items():
                    epoch_summary[f'{prefix}_{key}'] = float(value)
            if str(stage.stage_id) == 'softem_aug':
                extra_quantiles = _quantile_snapshot(epoch_extra_responsibilities)
                for key, value in extra_quantiles.items():
                    epoch_summary[f'extra_resp_{key}'] = float(value)
            if bool(config.print_epoch_summary):
                print(_format_epoch_summary(str(stage.stage_id), epoch_summary), file=sys.stderr, flush=True)
            if bool(config.write_runtime_metrics_jsonl):
                _append_jsonl(
                    runtime_metrics_path,
                    {
                        'row_type': 'epoch_summary',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        **epoch_summary,
                    },
                )
            if audit_callback is not None:
                audit_callback(
                    {
                        'dataset_name': str(config.dataset_name),
                        'trajectory_source_branch': str(config.trajectory_source_branch),
                        'stage_id': str(stage.stage_id),
                        'snapshot_id': f'epoch_{int(epoch_index) + 1:03d}',
                        'phase': 'epoch_end',
                        'output_root': output_root,
                        'materialized_samples': materialized_samples,
                        'text_projector': text_projector,
                        'projector': text_projector,
                        'theta_T': theta_t,
                        'b_u': b_u,
                        'responsibility_cache': cache,
                        'device': str(device),
                        'temperature': float(_compute_t_dis(theta_t).detach().cpu().item()),
                        'seed': int(config.seed),
                        'mode': str(config.mode),
                    }
                )
            memory_checkpoint(
                f"softem_after_epoch_{stage.stage_id}_{int(epoch_index) + 1}",
                stage_id=str(stage.stage_id),
                epoch=int(epoch_index) + 1,
                losses=len(losses),
                optimization_losses=len(optimization_losses),
                active_examples=len(stage_examples),
                cache_size=len(cache.by_trajectory_id),
            )
            timing_checkpoint(
                f"softem_after_epoch_{stage.stage_id}_{int(epoch_index) + 1}",
                started_at=audit_t0,
                stage_id=str(stage.stage_id),
                epoch=int(epoch_index) + 1,
                losses=len(losses),
                optimization_losses=len(optimization_losses),
                active_examples=len(stage_examples),
                cache_size=len(cache.by_trajectory_id),
            )

        if since_refresh > 0:
            active_examples_by_tid, cache, runtime_extra_cache, refreshed_trace = _refresh_stage_runtime_state(
                stage_id=stage.stage_id,
                unknown_init_mode=str(config.unknown_init_mode),
                base_examples=examples,
                base_cache=cache,
                text_projector=text_projector,
                theta_t=theta_t,
                b_u=b_u,
                output_root=output_root,
                k_extra=int(config.k_extra),
                extra_alpha=float(config.extra_alpha),
                lambda_frame=float(config.lambda_frame),
                lambda_cov=float(config.lambda_cov),
                em_subiterations=max(1, int(config.em_subiterations)),
                device=device,
            )
            if not stage_trace_sample:
                stage_trace_sample = refreshed_trace
        memory_checkpoint(
            f"softem_after_stage_end_{stage.stage_id}",
            stage_id=str(stage.stage_id),
            cache_size=len(cache.by_trajectory_id),
            stage_reports=len(stage_reports),
            current_checkpoint=str(current_checkpoint),
        )
        timing_checkpoint(
            f"softem_after_stage_end_{stage.stage_id}",
            started_at=audit_t0,
            stage_id=str(stage.stage_id),
            cache_size=len(cache.by_trajectory_id),
            stage_reports=len(stage_reports),
            current_checkpoint=str(current_checkpoint),
        )

        stage_dir = output_root / 'train' / stage.stage_id
        ckpt_dir = stage_dir / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        responsibility_path = output_root / stage.responsibility_relpath
        train_state_path = output_root / stage.train_state_relpath
        checkpoint_path = ckpt_dir / stage.checkpoint_name
        cache.stage_id = stage.stage_id
        cache.write_jsonl(responsibility_path)
        train_state = {
            'stage_id': stage.stage_id,
            'epoch': int(stage.epochs),
            'selected_for_infer': stage.selected_for_infer,
            'selected_for_infer_authority': 'explicit_train_state_field',
            'checkpoint_last': str((Path('train') / stage.stage_id / 'checkpoints' / stage.checkpoint_name).as_posix()),
            'checkpoint_selected': str((Path('train') / stage.stage_id / 'checkpoints' / stage.checkpoint_name).as_posix()),
            'global_step': int(iteration_index),
            'extra_refresh_interval_iters': int(current_refresh_interval),
            'runtime_asset_source': str(config.runtime_asset_source),
            'runtime_asset_source_local_incomplete': bool(config.runtime_asset_source_local_incomplete),
            'runtime_asset_output_root': str(config.runtime_asset_output_root),
        }
        _write_json(train_state_path, train_state)
        torch.save(
            {
                'stage_id': stage.stage_id,
                'epoch': int(stage.epochs),
                'text_projector_state_dict': text_projector.state_dict(),
                'text_projector_config': {
                    'input_dim': int(config.projector.input_dim),
                    'hidden_dim': int(config.projector.hidden_dim),
                    'output_dim': int(config.projector.output_dim),
                    'dropout': float(config.projector.dropout),
                    'use_layernorm': bool(config.projector.use_layernorm),
                },
                'theta_T': float(theta_t.detach().cpu().item()),
                'b_u': float(b_u.detach().cpu().item()),
                'seed': int(config.seed),
                'mode': str(config.mode),
                'global_step': int(iteration_index),
                'extra_refresh_interval_iters': int(current_refresh_interval),
                'unknown_init_mode': str(config.unknown_init_mode),
            },
            checkpoint_path,
        )
        stage_reports.append(
            {
                'stage_id': stage.stage_id,
                'responsibility_records_path': stage.responsibility_relpath,
                'train_state_path': stage.train_state_relpath,
                'checkpoint_last_path': str((Path('train') / stage.stage_id / 'checkpoints' / stage.checkpoint_name).as_posix()),
                'record_count_output': int(len(cache.by_trajectory_id)),
                'loss_mean': float(np.mean(losses)) if losses else 0.0,
                'loss_last': float(losses[-1]) if losses else 0.0,
                'optimization_loss_mean': float(np.mean(optimization_losses)) if optimization_losses else 0.0,
                'optimization_loss_last': float(optimization_losses[-1]) if optimization_losses else 0.0,
                'em_subiterations': int(max(1, int(config.em_subiterations))),
                'subiteration_trace_sample': stage_trace_sample,
                'runtime_extra_cache_enabled': bool(str(stage.stage_id) == 'softem_aug'),
                'runtime_extra_cache_clip_count': int(len(runtime_extra_cache)),
                'extra_refresh_interval_iters': int(current_refresh_interval),
                'unknown_init_mode': str(config.unknown_init_mode),
                'batch_budget': int(batch_budget),
                'budget_policy': 'dynamic_sum_Tv_times_Kv',
                'loss_normalization': 'effective_responsibility_unit_count',
                'micro_batch_count_per_epoch': int(final_epoch_plan.batch_count) if final_epoch_plan is not None else 0,
                'unknown_init_mode': str(config.unknown_init_mode),
                'unknown_init_prior_alpha': float(UNKNOWN_INIT_PRIOR_ALPHA) if str(config.unknown_init_mode) == UNKNOWN_INIT_MODE_PRIOR else 0.0,
            }
        )
        current_checkpoint = checkpoint_path

        if audit_callback is not None:
            audit_callback(
                {
                    'dataset_name': str(config.dataset_name),
                    'trajectory_source_branch': str(config.trajectory_source_branch),
                    'stage_id': str(stage.stage_id),
                    'snapshot_id': 'stage_end',
                    'phase': 'stage_end',
                    'output_root': output_root,
                    'materialized_samples': materialized_samples,
                    'text_projector': text_projector,
                    'projector': text_projector,
                    'theta_T': theta_t,
                    'b_u': b_u,
                    'responsibility_cache': cache,
                    'device': str(device),
                    'temperature': float(_compute_t_dis(theta_t).detach().cpu().item()),
                    'seed': int(config.seed),
                    'mode': str(config.mode),
                    'train_state': train_state,
                }
            )

    selected_checkpoint_path = stage_reports[-1]['checkpoint_last_path'] if stage_reports else ''
    return {
        'stage_reports': stage_reports,
        'record_count_input': int(len(materialized_samples)),
        'record_count_trainable': int(len(examples)),
        'record_count_output': int(len(cache.by_trajectory_id)),
        'coverage_ratio_trainable': float(len(examples) / float(len(materialized_samples))) if materialized_samples else 0.0,
        'skipped_reason_histogram': skipped,
        'selected_checkpoint_path': selected_checkpoint_path,
    }
