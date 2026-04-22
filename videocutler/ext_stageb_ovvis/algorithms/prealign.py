from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm as _tqdm_cls
except Exception:  # pragma: no cover - tqdm is optional in smoke environments
    _tqdm_cls = None

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    score_carrier_logits,
    score_carrier_logits_torch,
    load_carrier_evidence,
    load_text_vocab,
    observed_mass_loss,
)
from videocutler.ext_stageb_ovvis.algorithms._memory_audit import memory_checkpoint, shallow_size_bytes, timing_checkpoint
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig
from videocutler.ext_stageb_ovvis.algorithms._training_budget import build_dynamic_microbatches, resolve_default_batch_budget


Record = Dict[str, Any]


@dataclass(frozen=True)
class PrealignConfig:
    dataset_name: str
    trajectory_source_branch: str = 'mainline'
    device: str = 'cpu'
    seed: int = 0
    smoke: bool = False
    lambda_frame: float = 0.25
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    t_dis_init: float = 0.07
    b_u_init: float = 0.0
    projector: ProjectorConfig = ProjectorConfig()
    runtime_asset_source: str = 'local_canonical_assets'
    runtime_asset_source_local_incomplete: bool = False
    runtime_asset_output_root: str = ''
    batch_budget: int | None = None
    show_progress: bool = True
    log_every: int = 10
    write_runtime_metrics_jsonl: bool = True
    print_epoch_summary: bool = True


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _write_jsonl(path: Path, rows: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _append_jsonl(path: Path, row: Record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _make_progress_bar(*, total: int, desc: str, enabled: bool):
    if enabled and _tqdm_cls is not None and sys.stderr.isatty():
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

        def set_postfix(self, values: Dict[str, Any] | None = None, refresh: bool = True) -> None:  # noqa: ARG002
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


def _runtime_metrics_path(output_root: Path) -> Path:
    return output_root / 'train' / 'prealign' / 'runtime_metrics.jsonl'


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


def _format_epoch_summary(prefix: str, summary: Mapping[str, Any]) -> str:
    keys = [
        'epoch',
        'microbatch_count',
        'loss_mean',
        'loss_last',
        'optimization_loss_mean',
        'optimization_loss_last',
        'effective_trajectory_count_total',
        'positive_unit_count_total',
        'unknown_mass_mean_epoch',
        'observed_mass_mean_epoch',
        'unknown_mass_min',
        'unknown_mass_p10',
        'unknown_mass_p50',
        'unknown_mass_p90',
        'unknown_mass_max',
        'observed_mass_min',
        'observed_mass_p10',
        'observed_mass_p50',
        'observed_mass_p90',
        'observed_mass_max',
    ]
    parts = [f'[{prefix}] epoch_summary']
    for key in keys:
        if key in summary:
            parts.append(f'{key}={summary[key]}')
    if 'skipped_reason_histogram_epoch' in summary:
        parts.append(f"skipped_reason_histogram_epoch={json.dumps(summary['skipped_reason_histogram_epoch'], sort_keys=True)}")
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

    def bump(reason: str) -> None:
        skipped[reason] = int(skipped.get(reason, 0)) + 1

    for sample in materialized_samples:
        if not bool(sample.get('sample_valid', False)):
            bump('sample_not_valid_from_phase1')
            continue
        try:
            memory_checkpoint(
                "prealign_prepare_before_load_combined_evidence",
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
                "prealign_prepare_after_load_combined_evidence",
                trajectory_id=str(sample.get('trajectory_id', '')),
                carrier_vec_shallow_size=shallow_size_bytes(carrier_vec),
            )
        except Exception:
            bump('missing_carrier_evidence')
            continue
        examples.append(
            {
                'trajectory_id': str(sample['trajectory_id']),
                'clip_id': int(sample['clip_id']),
                'video_id': int(sample['trajectory_record']['video_id']),
                'observed_raw_ids': sorted({int(x) for x in list(sample.get('observed_raw_ids', []))}),
                'carrier_vec': np.asarray(carrier_vec, dtype=np.float32),
            }
        )
    return {'examples': examples, 'skipped_reason_histogram': skipped}


def _inverse_softplus(value: float) -> float:
    target = max(float(value), 1e-6)
    return float(math.log(math.expm1(target)))


def _compute_t_dis(theta_t: torch.nn.Parameter) -> torch.Tensor:
    return F.softplus(theta_t) + 1e-4


def train_prealign(
    *,
    output_root: Path,
    materialized_samples: Sequence[Record],
    config: PrealignConfig,
    audit_callback: Any = None,
) -> Dict[str, Any]:
    _set_seed(int(config.seed))
    device = torch.device(str(config.device))
    audit_t0 = time.perf_counter()
    memory_checkpoint(
        "prealign_start",
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
        smoke=bool(config.smoke),
        batch_budget=(int(config.batch_budget) if config.batch_budget is not None else None),
        lambda_frame=float(config.lambda_frame),
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
        "prealign_after_prepare_examples",
        materialized_samples=len(materialized_samples),
        trainable_examples=len(examples),
        skipped_reason_histogram=skipped,
        total_observed_ids=sum(len(ex.get('observed_raw_ids', [])) for ex in examples),
    )
    timing_checkpoint(
        "prealign_after_prepare_examples",
        started_at=audit_t0,
        materialized_samples=len(materialized_samples),
        trainable_examples=len(examples),
        skipped_reason_histogram=skipped,
        total_observed_ids=sum(len(ex.get('observed_raw_ids', [])) for ex in examples),
    )
    total_samples = len(materialized_samples)
    if not examples:
        raise RuntimeError('no valid trainable prealign examples after phase-1 filtering')
    text_vocab_ids, text_vocab_records, text_vocab_matrix = load_text_vocab(output_root)
    memory_checkpoint(
        "prealign_after_text_vocab_load",
        text_vocab_size=len(text_vocab_ids),
        text_vocab_records=len(text_vocab_records),
        text_vocab_matrix_shape=getattr(text_vocab_matrix, "shape", None),
        text_vocab_matrix_shallow_size=shallow_size_bytes(text_vocab_matrix),
    )
    timing_checkpoint(
        "prealign_after_text_vocab_load",
        started_at=audit_t0,
        text_vocab_size=len(text_vocab_ids),
        text_vocab_records=len(text_vocab_records),
        text_vocab_matrix_shape=getattr(text_vocab_matrix, "shape", None),
        text_vocab_matrix_shallow_size=shallow_size_bytes(text_vocab_matrix),
    )

    text_projector = Projector(config.projector).to(device)
    text_projector.train()
    theta_t = torch.nn.Parameter(
        torch.tensor(_inverse_softplus(max(float(config.t_dis_init) - 1e-4, 1e-6)), device=device, dtype=torch.float32)
    )
    b_u = torch.nn.Parameter(torch.tensor(float(config.b_u_init), device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        [*text_projector.parameters(), theta_t, b_u],
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    losses: List[float] = []
    batch_losses: List[float] = []
    global_step = 0
    text_candidate_matrix = np.asarray(text_vocab_matrix, dtype=np.float32)
    batch_budget = resolve_default_batch_budget(smoke=bool(config.smoke), explicit=config.batch_budget)
    example_plan = build_dynamic_microbatches(
        examples,
        batch_budget=batch_budget,
        cost_fn=lambda ex: len(text_vocab_ids),
        bucket_key_fn=lambda ex: (1, len(text_vocab_ids)),
    )
    runtime_metrics_path = _runtime_metrics_path(output_root)
    memory_checkpoint(
        "prealign_before_first_epoch",
        batch_budget=int(batch_budget),
        batch_count=int(example_plan.batch_count),
        examples=len(examples),
    )
    timing_checkpoint(
        "prealign_before_first_epoch",
        started_at=audit_t0,
        batch_budget=int(batch_budget),
        batch_count=int(example_plan.batch_count),
        examples=len(examples),
    )

    if audit_callback is not None:
        audit_callback(
            {
                'dataset_name': str(config.dataset_name),
                'trajectory_source_branch': str(config.trajectory_source_branch),
                'stage_id': 'prealign',
                'snapshot_id': 'stage_start',
                'phase': 'stage_start',
                'output_root': output_root,
                'materialized_samples': materialized_samples,
                'text_projector': text_projector,
                'projector': text_projector,
                'theta_T': theta_t,
                'b_u': b_u,
                'device': str(device),
                'temperature': float(_compute_t_dis(theta_t).detach().cpu().item()),
                'seed': int(config.seed),
                'mode': 'prealign',
            }
        )

    for epoch_index in range(int(config.epochs)):
        shuffled_examples = list(examples)
        random.Random(int(config.seed) + int(epoch_index)).shuffle(shuffled_examples)
        epoch_plan = build_dynamic_microbatches(
            shuffled_examples,
            batch_budget=batch_budget,
            cost_fn=lambda ex: len(text_vocab_ids),
            bucket_key_fn=lambda ex: (1, len(text_vocab_ids)),
        )
        epoch_losses: List[float] = []
        epoch_batch_losses: List[float] = []
        epoch_effective_trajectory_counts: List[int] = []
        epoch_positive_unit_counts: List[float] = []
        epoch_unknown_masses: List[float] = []
        epoch_observed_masses: List[float] = []
        progress_enabled = bool(config.show_progress)
        with _make_progress_bar(
            total=int(epoch_plan.batch_count),
            desc=f"prealign epoch {int(epoch_index) + 1}/{int(config.epochs)}",
            enabled=progress_enabled,
        ) as progress:
            for microbatch_index, batch_indices in enumerate(epoch_plan.batches, start=1):
                optimizer.zero_grad(set_to_none=True)
                batch_loss_accum: torch.Tensor | None = None
                effective_trajectory_count = 0
                sample_losses: List[float] = []
                sample_unknown_masses: List[float] = []
                sample_observed_masses: List[float] = []
                sample_positive_counts: List[int] = []
                for batch_index in batch_indices:
                    ex = shuffled_examples[int(batch_index)]
                    current_t_dis = _compute_t_dis(theta_t)
                    logits = score_carrier_logits_torch(
                        projector=text_projector,
                        carrier_vec=ex['carrier_vec'],
                        candidate_matrix=text_candidate_matrix,
                        temperature=current_t_dis,
                    )
                    observed_raw_ids = [int(x) for x in ex['observed_raw_ids']]
                    positive = [idx for idx, raw_id in enumerate(text_vocab_ids) if int(raw_id) in observed_raw_ids]
                    if not positive:
                        raise RuntimeError(f"no observed raw ids found in text vocab for trajectory {ex['trajectory_id']}")
                    logits_full = torch.cat([b_u.reshape(1), logits], dim=0)
                    probs = torch.softmax(logits_full, dim=0)
                    sample_loss = observed_mass_loss(logits, positive, unknown_logit=b_u)
                    sample_losses.append(float(sample_loss.detach().cpu().item()))
                    sample_unknown_masses.append(float(probs[0].detach().cpu().item()))
                    sample_observed_masses.append(float(probs[[idx + 1 for idx in positive]].sum().detach().cpu().item()))
                    sample_positive_counts.append(int(len(positive)))
                    losses.append(float(sample_loss.detach().cpu().item()))
                    epoch_losses.append(float(sample_loss.detach().cpu().item()))
                    epoch_unknown_masses.append(float(probs[0].detach().cpu().item()))
                    epoch_observed_masses.append(float(probs[[idx + 1 for idx in positive]].sum().detach().cpu().item()))
                    epoch_positive_unit_counts.append(float(len(positive)))
                    batch_loss_accum = sample_loss if batch_loss_accum is None else (batch_loss_accum + sample_loss)
                    effective_trajectory_count += 1
                if batch_loss_accum is None or effective_trajectory_count <= 0:
                    continue
                batch_loss = batch_loss_accum / float(effective_trajectory_count)
                batch_loss.backward()
                optimizer.step()
                batch_loss_value = float(batch_loss.detach().cpu().item())
                batch_losses.append(batch_loss_value)
                epoch_batch_losses.append(batch_loss_value)
                epoch_effective_trajectory_counts.append(int(effective_trajectory_count))
                global_step += 1
                progress.update(1)
                progress.set_postfix(
                    {
                        'loss': f'{float(np.mean(sample_losses)):.4f}',
                        'opt_loss': f'{batch_loss_value:.4f}',
                        'traj': effective_trajectory_count,
                    },
                    refresh=False,
                )
                if _should_log_microbatch(int(config.log_every), microbatch_index, int(epoch_plan.batch_count)):
                    print(
                        (
                            f"[prealign] epoch={int(epoch_index) + 1}/{int(config.epochs)} "
                            f"microbatch={microbatch_index}/{int(epoch_plan.batch_count)} "
                            f"loss={float(np.mean(sample_losses)):.6f} "
                            f"opt_loss={batch_loss_value:.6f} "
                            f"effective_trajectory_count={effective_trajectory_count} "
                            f"positive_unit_count={float(np.mean(sample_positive_counts)):.2f} "
                            f"batch_budget_used={len(batch_indices)} "
                            f"unknown_mass_mean={float(np.mean(sample_unknown_masses)):.6f} "
                            f"observed_mass_mean={float(np.mean(sample_observed_masses)):.6f}"
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                if bool(config.write_runtime_metrics_jsonl):
                    _append_jsonl(
                        runtime_metrics_path,
                        {
                            'row_type': 'microbatch',
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'stage': 'prealign',
                            'epoch': int(epoch_index) + 1,
                            'microbatch_idx': int(microbatch_index),
                            'microbatch_total': int(epoch_plan.batch_count),
                            'loss': float(np.mean(sample_losses)),
                            'optimization_loss': float(batch_loss_value),
                            'effective_trajectory_count': int(effective_trajectory_count),
                            'positive_unit_count': float(np.mean(sample_positive_counts)),
                            'batch_budget_used': int(len(batch_indices)),
                            'unknown_mass_mean': float(np.mean(sample_unknown_masses)),
                            'observed_mass_mean': float(np.mean(sample_observed_masses)),
                        },
                    )
        epoch_summary = {
            'stage': 'prealign',
            'epoch': int(epoch_index) + 1,
            'microbatch_count': int(len(epoch_batch_losses)),
            'loss_mean': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            'loss_last': float(epoch_losses[-1]) if epoch_losses else 0.0,
            'optimization_loss_mean': float(np.mean(epoch_batch_losses)) if epoch_batch_losses else 0.0,
            'optimization_loss_last': float(epoch_batch_losses[-1]) if epoch_batch_losses else 0.0,
            'effective_trajectory_count_total': int(np.sum(epoch_effective_trajectory_counts)) if epoch_effective_trajectory_counts else 0,
            'effective_trajectory_count_mean': float(np.mean(epoch_effective_trajectory_counts)) if epoch_effective_trajectory_counts else 0.0,
            'positive_unit_count_total': float(np.sum(epoch_positive_unit_counts)) if epoch_positive_unit_counts else 0.0,
            'positive_unit_count_mean': float(np.mean(epoch_positive_unit_counts)) if epoch_positive_unit_counts else 0.0,
            'unknown_mass_mean_epoch': float(np.mean(epoch_unknown_masses)) if epoch_unknown_masses else 0.0,
            'observed_mass_mean_epoch': float(np.mean(epoch_observed_masses)) if epoch_observed_masses else 0.0,
            'skipped_reason_histogram_epoch': dict(sorted(skipped.items())),
        }
        unknown_quantiles = _quantile_snapshot(epoch_unknown_masses)
        observed_quantiles = _quantile_snapshot(epoch_observed_masses)
        for prefix, values in (('unknown_mass', unknown_quantiles), ('observed_mass', observed_quantiles)):
            for key, value in values.items():
                epoch_summary[f'{prefix}_{key}'] = float(value)
        if bool(config.print_epoch_summary):
            print(_format_epoch_summary('prealign', epoch_summary), file=sys.stderr, flush=True)
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
                    'stage_id': 'prealign',
                    'snapshot_id': f'epoch_{int(epoch_index) + 1:03d}',
                    'phase': 'epoch_end',
                    'output_root': output_root,
                    'materialized_samples': materialized_samples,
                    'text_projector': text_projector,
                    'projector': text_projector,
                    'theta_T': theta_t,
                    'b_u': b_u,
                    'device': str(device),
                    'temperature': float(_compute_t_dis(theta_t).detach().cpu().item()),
                    'seed': int(config.seed),
                    'mode': 'prealign',
                }
            )
        memory_checkpoint(
            f"prealign_after_epoch_{int(epoch_index) + 1}",
            epoch=int(epoch_index) + 1,
            losses=len(losses),
            batch_losses=len(batch_losses),
            trainable_examples=len(examples),
            proxy_rows_planned=len(examples),
        )
        timing_checkpoint(
            f"prealign_after_epoch_{int(epoch_index) + 1}",
            started_at=audit_t0,
            epoch=int(epoch_index) + 1,
            losses=len(losses),
            batch_losses=len(batch_losses),
            trainable_examples=len(examples),
            proxy_rows_planned=len(examples),
        )

    text_projector.eval()
    proxy_rows: List[Record] = []
    with torch.no_grad():
        current_t_dis = _compute_t_dis(theta_t)
        for ex in sorted(examples, key=lambda row: str(row['trajectory_id'])):
            logits_np = score_carrier_logits(
                projector=text_projector,
                carrier_vec=ex['carrier_vec'],
                candidate_matrix=text_candidate_matrix,
                temperature=current_t_dis,
            )
            logits = torch.from_numpy(np.asarray(logits_np, dtype=np.float32)).to(device=device, dtype=torch.float32)
            logits_full = torch.cat([b_u.detach().reshape(1), logits], dim=0)
            probs = torch.softmax(logits_full, dim=0).detach().cpu().numpy().astype(np.float64)
            proxy_mass = {'unknown': float(probs[0])}
            for idx, raw_id in enumerate(text_vocab_ids):
                proxy_mass[str(int(raw_id))] = float(probs[idx + 1])
            proxy_rows.append(
                {
                    'dataset_name': str(config.dataset_name),
                    'clip_id': int(ex['clip_id']),
                    'video_id': int(ex['video_id']),
                    'trajectory_id': str(ex['trajectory_id']),
                    'observed_raw_ids': [int(x) for x in ex['observed_raw_ids']],
                    'proxy_mass': proxy_mass,
                    'join_key': str(ex['trajectory_id']),
                }
            )

    train_dir = output_root / 'train' / 'prealign'
    ckpt_dir = train_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = train_dir / 'proxy_records.jsonl'
    train_state_path = train_dir / 'train_state.json'
    ckpt_last_path = ckpt_dir / 'prealign_last.pth'
    torch.save(
        {
            'stage_id': 'prealign',
            'epoch': int(config.epochs),
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
            'global_step': int(global_step),
        },
        ckpt_last_path,
    )
    _write_jsonl(proxy_path, proxy_rows)
    memory_checkpoint(
        "prealign_after_proxy_materialization",
        proxy_rows=len(proxy_rows),
        proxy_rows_shallow_size=len(proxy_rows) if hasattr(proxy_rows, "__len__") else None,
    )
    timing_checkpoint(
        "prealign_after_proxy_materialization",
        started_at=audit_t0,
        proxy_rows=len(proxy_rows),
        proxy_rows_shallow_size=len(proxy_rows) if hasattr(proxy_rows, "__len__") else None,
    )
    timing_checkpoint(
        "prealign_after_proxy_materialization",
        started_at=audit_t0,
        proxy_rows=len(proxy_rows),
        proxy_rows_shallow_size=len(proxy_rows) if hasattr(proxy_rows, "__len__") else None,
    )
    train_state = {
        'stage_id': 'prealign',
        'epoch': int(config.epochs),
        'selected_for_infer': 'prealign_only',
        'selected_for_infer_authority': 'explicit_train_state_field',
        'checkpoint_last': 'train/prealign/checkpoints/prealign_last.pth',
        'checkpoint_selected': 'train/prealign/checkpoints/prealign_last.pth',
        'global_step': int(global_step),
        'runtime_asset_source': str(config.runtime_asset_source),
        'runtime_asset_source_local_incomplete': bool(config.runtime_asset_source_local_incomplete),
        'runtime_asset_output_root': str(config.runtime_asset_output_root),
    }
    _write_json(train_state_path, train_state)
    memory_checkpoint(
        "prealign_stage_end",
        train_state_keys=len(train_state),
        proxy_rows=len(proxy_rows),
        record_count_input=total_samples,
        record_count_trainable=len(examples),
        record_count_output=len(proxy_rows),
    )
    timing_checkpoint(
        "prealign_stage_end",
        started_at=audit_t0,
        train_state_keys=len(train_state),
        proxy_rows=len(proxy_rows),
        record_count_input=total_samples,
        record_count_trainable=len(examples),
        record_count_output=len(proxy_rows),
    )
    timing_checkpoint(
        "prealign_stage_end",
        started_at=audit_t0,
        train_state_keys=len(train_state),
        proxy_rows=len(proxy_rows),
        record_count_input=total_samples,
        record_count_trainable=len(examples),
        record_count_output=len(proxy_rows),
    )

    if audit_callback is not None:
        audit_callback(
            {
                'dataset_name': str(config.dataset_name),
                'trajectory_source_branch': str(config.trajectory_source_branch),
                'stage_id': 'prealign',
                'snapshot_id': 'stage_end',
                'phase': 'stage_end',
                'output_root': output_root,
                'materialized_samples': materialized_samples,
                'text_projector': text_projector,
                'projector': text_projector,
                'theta_T': theta_t,
                'b_u': b_u,
                'device': str(device),
                'temperature': float(_compute_t_dis(theta_t).detach().cpu().item()),
                'seed': int(config.seed),
                'mode': 'prealign',
                'train_state': train_state,
            }
        )

    return {
        'proxy_records_path': proxy_path,
        'train_state_path': train_state_path,
        'checkpoint_last_path': ckpt_last_path,
        'record_count_input': int(total_samples),
        'record_count_trainable': int(len(examples)),
        'record_count_output': int(len(proxy_rows)),
        'coverage_ratio_trainable': float(len(examples) / float(total_samples)) if total_samples > 0 else 0.0,
        'skipped_reason_histogram': skipped,
        'loss_mean': float(np.mean(losses)) if losses else 0.0,
        'loss_last': float(losses[-1]) if losses else 0.0,
        'optimization_loss_mean': float(np.mean(batch_losses)) if batch_losses else 0.0,
        'optimization_loss_last': float(batch_losses[-1]) if batch_losses else 0.0,
        'batch_budget': int(batch_budget),
        'micro_batch_count_per_epoch': int(example_plan.batch_count),
        'budget_policy': 'dynamic_sum_Tv_times_Kv',
        'loss_normalization': 'effective_trajectory_count',
        'train_state': train_state,
    }
