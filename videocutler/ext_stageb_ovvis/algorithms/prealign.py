from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    fuse_carrier_frame_logits,
    fuse_carrier_frame_logits_torch,
    load_combined_evidence,
    load_text_vocab,
    observed_mass_loss,
)
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
            carrier_vec, frame_vectors, frame_vec, combined_vec = load_combined_evidence(
                sample,
                output_root=output_root,
                dataset_name=dataset_name,
                trajectory_source_branch=trajectory_source_branch,
            )
        except Exception:
            bump('missing_frame_evidence')
            continue
        examples.append(
            {
                'trajectory_id': str(sample['trajectory_id']),
                'clip_id': int(sample['clip_id']),
                'video_id': int(sample['trajectory_record']['video_id']),
                'observed_raw_ids': sorted({int(x) for x in list(sample.get('observed_raw_ids', []))}),
                'carrier_vec': np.asarray(carrier_vec, dtype=np.float32),
                'frame_vectors': [np.asarray(vec, dtype=np.float32) for vec in frame_vectors],
                'frame_vec': np.asarray(frame_vec, dtype=np.float32),
                'combined_vec': np.asarray(combined_vec, dtype=np.float32),
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
    prepared = _prepare_examples(
        materialized_samples,
        output_root=output_root,
        dataset_name=config.dataset_name,
        trajectory_source_branch=config.trajectory_source_branch,
    )
    examples = list(prepared['examples'])
    skipped = dict(prepared['skipped_reason_histogram'])
    total_samples = len(materialized_samples)
    if not examples:
        raise RuntimeError('no valid trainable prealign examples after phase-1 filtering')
    text_vocab_ids, text_vocab_records, text_vocab_matrix = load_text_vocab(output_root)

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
        for batch_indices in epoch_plan.batches:
            optimizer.zero_grad(set_to_none=True)
            batch_loss_accum: torch.Tensor | None = None
            effective_trajectory_count = 0
            for batch_index in batch_indices:
                ex = shuffled_examples[int(batch_index)]
                current_t_dis = _compute_t_dis(theta_t)
                _, _, logits = fuse_carrier_frame_logits_torch(
                    projector=text_projector,
                    carrier_vec=ex['carrier_vec'],
                    frame_vec=ex['frame_vec'],
                    candidate_matrix=text_candidate_matrix,
                    temperature=current_t_dis,
                    frame_vectors=ex['frame_vectors'],
                )
                observed_raw_ids = [int(x) for x in ex['observed_raw_ids']]
                positive = [idx for idx, raw_id in enumerate(text_vocab_ids) if int(raw_id) in observed_raw_ids]
                if not positive:
                    raise RuntimeError(f"no observed raw ids found in text vocab for trajectory {ex['trajectory_id']}")
                sample_loss = observed_mass_loss(logits, positive, unknown_logit=b_u)
                losses.append(float(sample_loss.detach().cpu().item()))
                batch_loss_accum = sample_loss if batch_loss_accum is None else (batch_loss_accum + sample_loss)
                effective_trajectory_count += 1
            if batch_loss_accum is None or effective_trajectory_count <= 0:
                continue
            batch_loss = batch_loss_accum / float(effective_trajectory_count)
            batch_loss.backward()
            optimizer.step()
            batch_losses.append(float(batch_loss.detach().cpu().item()))
            global_step += 1
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

    text_projector.eval()
    proxy_rows: List[Record] = []
    with torch.no_grad():
        current_t_dis = _compute_t_dis(theta_t)
        for ex in sorted(examples, key=lambda row: str(row['trajectory_id'])):
            _, _, logits_np = fuse_carrier_frame_logits(
                projector=text_projector,
                carrier_vec=ex['carrier_vec'],
                frame_vec=ex['frame_vec'],
                candidate_matrix=text_candidate_matrix,
                temperature=current_t_dis,
                frame_vectors=ex['frame_vectors'],
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
