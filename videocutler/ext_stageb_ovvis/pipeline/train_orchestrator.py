from __future__ import annotations
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Sequence
import torch
from videocutler.ext_stageb_ovvis.pipeline.plans import TrainPlan
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import Phase1MaterializationConfig, materialize_phase1_training_samples
from videocutler.ext_stageb_ovvis.algorithms.prealign import PrealignConfig, train_prealign
from videocutler.ext_stageb_ovvis.algorithms.soft_em import SoftEMConfig, run_soft_em
from videocutler.ext_stageb_ovvis.algorithms.reservoir_v1 import ReservoirPrealignConfig, ReservoirSoftEMConfig, train_reservoir_prealign, run_reservoir_soft_em
from videocutler.ext_stageb_ovvis.audit.g8_minimal_split_audit import MinimalSplitAuditConfig, run_minimal_split_audit
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import TRAIN_DATASETS

REPO_ASSET_LINK_NAMES = ('exports', 'exports_gt', 'carrier_bank', 'carrier_bank_gt', 'frame_bank', 'text_bank', 'gt_sidecar_bank', 'weak_labels', 'weights', 'dataset', 'eval')


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _safe_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _bootstrap_asset_links(target_root: Path, asset_root: Path) -> None:
    if not asset_root.is_dir():
        return
    target_root.mkdir(parents=True, exist_ok=True)
    for name in REPO_ASSET_LINK_NAMES:
        src = asset_root / name
        dst = target_root / name
        if src.exists() and not dst.exists() and not dst.is_symlink():
            try:
                _safe_link(src, dst)
            except Exception:
                pass


def _bootstrap_repo_assets(plan: TrainPlan) -> None:
    asset_root = Path(plan.asset_root)
    _bootstrap_asset_links(Path(plan.repo_root), asset_root)
    _bootstrap_asset_links(Path(plan.output_root), asset_root)


def _materialize(plan: TrainPlan):
    _bootstrap_repo_assets(plan)
    with _pushd(plan.repo_root):
        return materialize_phase1_training_samples(
            plan.repo_root,
            Phase1MaterializationConfig(
                dataset_name=plan.dataset_name,
                trajectory_source_branch=plan.trajectory_source_branch,
                smoke=plan.smoke,
                smoke_max_trajectories=plan.smoke_max_trajectories,
                subset_fraction=plan.subset_fraction,
                subset_seed=plan.seed,
            ),
        )


def _resolve_materialized_samples(materialized: Dict[str, Any], *, prefer_valid: bool) -> Sequence[Dict[str, Any]]:
    if prefer_valid:
        valid_samples = materialized.get('valid_samples')
        if isinstance(valid_samples, list) and valid_samples:
            return valid_samples
    samples = materialized.get('samples')
    if isinstance(samples, list):
        filtered = [sample for sample in samples if bool(sample.get('sample_valid', False))]
        if filtered:
            return filtered
        return samples
    return []


def _run_post_train_minimal_split_audit(plan: TrainPlan) -> Dict[str, Any]:
    if str(plan.dataset_name) not in TRAIN_DATASETS:
        return {'status': 'SKIPPED', 'reason': 'dataset_not_supported'}
    try:
        payload = run_minimal_split_audit(
            MinimalSplitAuditConfig(
                dataset_name=str(plan.dataset_name),
                output_root=plan.output_root,
                stage='all',
                device=torch.device(str(plan.device)),
                trajectory_source_branch=str(plan.trajectory_source_branch),
                all_gt_generate_sidecars_if_missing=False,
            )
        )
        by_split = payload.get('comparison_by_split', {}).get('by_split', {})
        compact: Dict[str, Any] = {}
        for split_name, stage_map in by_split.items():
            compact[str(split_name)] = {}
            for stage_name, metrics in stage_map.items():
                compact[str(split_name)][str(stage_name)] = {
                    'gt_count': metrics.get('gt_count'),
                    'mean_normalized_gt_rank': metrics.get('mean_normalized_gt_rank'),
                    'gt_top1_hit_rate': metrics.get('gt_top1_hit_rate'),
                    'status': metrics.get('status'),
                }
        dataset_summary_path = plan.output_root / 'audit' / 'minimal_split' / str(plan.dataset_name) / 'train_minimal_split_summary.json'
        comparison_path = plan.output_root / 'audit' / 'minimal_split' / str(plan.dataset_name) / 'minimal_split_comparison.json'
        return {
            'status': 'PASS',
            'audit_type': 'train_minimal_split',
            'summary_path': str(dataset_summary_path),
            'comparison_path': str(comparison_path),
            'split_order': list(payload.get('split_order', [])),
            'stage_scope': list(payload.get('stage_scope', [])),
            'comparison_by_split': compact,
        }
    except Exception as e:
        return {
            'status': 'FAIL',
            'audit_type': 'train_minimal_split',
            'error': repr(e),
        }


def run_train_pipeline(plan: TrainPlan) -> Dict[str, Any]:
    materialized = _materialize(plan)
    summary = {
        'exp_name': plan.exp_name,
        'pipeline': plan.pipeline,
        'stage_scope': plan.stage_scope,
        'ablation_flags': {
            'ablate_skip_base': bool(getattr(plan, 'ablate_skip_base', False)),
            'ablate_no_yprime_reward': bool(getattr(plan, 'ablate_no_yprime_reward', False)),
        },
        'dataset_name': plan.dataset_name,
        'trajectory_source_branch': plan.trajectory_source_branch,
        'smoke': plan.smoke,
        'repo_root': str(plan.repo_root),
        'asset_root': str(plan.asset_root),
        'resolution': materialized['resolution'],
        'materialization_stats': materialized['stats'],
        'stages': {},
    }
    if plan.pipeline == 'legacy':
        pre = train_prealign(
            output_root=plan.output_root,
            materialized_samples=materialized['samples'],
            config=PrealignConfig(
                dataset_name=plan.dataset_name, trajectory_source_branch=plan.trajectory_source_branch, device=plan.device,
                seed=plan.seed, smoke=plan.smoke, epochs=(1 if plan.smoke else 5) if plan.prealign_epochs is None else plan.prealign_epochs,
                learning_rate=(1e-4 if plan.prealign_learning_rate is None else plan.prealign_learning_rate), weight_decay=plan.weight_decay,
                t_dis_init=plan.t_dis_init, lambda_frame=(0.25 if plan.lambda_frame is None else plan.lambda_frame),
                runtime_asset_source=str(materialized['resolution'].get('runtime_asset_source','local_canonical_assets')),
                runtime_asset_source_local_incomplete=bool(materialized['resolution'].get('local_incomplete',False)),
                runtime_asset_output_root=str(materialized['resolution'].get('runtime_output_root', str(plan.repo_root))),
                batch_budget=plan.batch_budget, show_progress=plan.show_progress, log_every=plan.log_every,
                write_runtime_metrics_jsonl=plan.write_runtime_metrics_jsonl, print_epoch_summary=plan.print_epoch_summary,
            ),
        )
        summary['stages']['prealign'] = pre
        if plan.stage_scope != 'prealign_only':
            mode = 'base_only' if plan.stage_scope == 'prealign_base' else 'base_then_aug'
            soft = run_soft_em(
                output_root=plan.output_root,
                materialized_samples=materialized['samples'],
                config=SoftEMConfig(
                    dataset_name=plan.dataset_name, trajectory_source_branch=plan.trajectory_source_branch, mode=mode,
                    device=plan.device, seed=plan.seed, smoke=plan.smoke, lambda_frame=(0.25 if plan.lambda_frame is None else plan.lambda_frame),
                    lambda_cov=plan.lambda_cov, t_dis_init=plan.t_dis_init, weight_decay=plan.weight_decay,
                    base_epochs=(1 if plan.smoke else 5) if plan.base_epochs is None else plan.base_epochs,
                    aug_epochs=(1 if plan.smoke else 5) if plan.aug_epochs is None else plan.aug_epochs,
                    base_learning_rate=(5e-5 if plan.base_learning_rate is None else plan.base_learning_rate),
                    aug_learning_rate=(5e-5 if plan.aug_learning_rate is None else plan.aug_learning_rate),
                    runtime_asset_source=str(materialized['resolution'].get('runtime_asset_source','local_canonical_assets')),
                    runtime_asset_source_local_incomplete=bool(materialized['resolution'].get('local_incomplete',False)),
                    runtime_asset_output_root=str(materialized['resolution'].get('runtime_output_root', str(plan.repo_root))),
                    batch_budget=plan.batch_budget, show_progress=plan.show_progress, log_every=plan.log_every,
                    write_runtime_metrics_jsonl=plan.write_runtime_metrics_jsonl, print_epoch_summary=plan.print_epoch_summary,
                ),
            )
            summary['stages']['softem'] = soft
    else:
        reservoir_samples = _resolve_materialized_samples(materialized, prefer_valid=True)
        pre = train_reservoir_prealign(
            output_root=plan.output_root,
            materialized_samples=reservoir_samples,
            config=ReservoirPrealignConfig(
                dataset_name=plan.dataset_name, trajectory_source_branch=plan.trajectory_source_branch, device=plan.device,
                seed=plan.seed, smoke=plan.smoke, epochs=(1 if plan.smoke else 5) if plan.prealign_epochs is None else plan.prealign_epochs,
                learning_rate=(1e-4 if plan.prealign_learning_rate is None else plan.prealign_learning_rate), weight_decay=plan.weight_decay,
                t_dis_init=plan.t_dis_init, lambda_frame=(0.25 if plan.lambda_frame is None else plan.lambda_frame),
                runtime_asset_source=str(materialized['resolution'].get('runtime_asset_source','local_canonical_assets')),
                runtime_asset_source_local_incomplete=bool(materialized['resolution'].get('local_incomplete',False)),
                runtime_asset_output_root=str(materialized['resolution'].get('runtime_output_root', str(plan.repo_root))),
                batch_budget=plan.batch_budget, show_progress=plan.show_progress, log_every=plan.log_every,
                write_runtime_metrics_jsonl=plan.write_runtime_metrics_jsonl, print_epoch_summary=plan.print_epoch_summary,
            ),
        )
        summary['stages']['prealign'] = pre
        if plan.stage_scope != 'prealign_only':
            mode = 'base_only' if plan.stage_scope == 'prealign_base' else 'base_then_aug'
            soft = run_reservoir_soft_em(
                output_root=plan.output_root,
                materialized_samples=reservoir_samples,
                config=ReservoirSoftEMConfig(
                    dataset_name=plan.dataset_name, trajectory_source_branch=plan.trajectory_source_branch, mode=mode,
                    device=plan.device, seed=plan.seed, smoke=plan.smoke, lambda_frame=(0.25 if plan.lambda_frame is None else plan.lambda_frame),
                    t_dis_init=plan.t_dis_init, weight_decay=plan.weight_decay,
                    base_epochs=(1 if plan.smoke else 5) if plan.base_epochs is None else plan.base_epochs,
                    aug_epochs=(1 if plan.smoke else 5) if plan.aug_epochs is None else plan.aug_epochs,
                    base_learning_rate=(5e-5 if plan.base_learning_rate is None else plan.base_learning_rate),
                    aug_learning_rate=(5e-5 if plan.aug_learning_rate is None else plan.aug_learning_rate),
                    runtime_asset_source=str(materialized['resolution'].get('runtime_asset_source','local_canonical_assets')),
                    runtime_asset_source_local_incomplete=bool(materialized['resolution'].get('local_incomplete',False)),
                    runtime_asset_output_root=str(materialized['resolution'].get('runtime_output_root', str(plan.repo_root))),
                    batch_budget=plan.batch_budget, k_extra=int(plan.k_extra), extra_alpha=float(plan.extra_alpha), base_release_margin=float(plan.base_release_margin),
                    ablate_skip_base=bool(getattr(plan, 'ablate_skip_base', False)),
                    ablate_no_yprime_reward=bool(getattr(plan, 'ablate_no_yprime_reward', False)),
                    show_progress=plan.show_progress, log_every=plan.log_every,
                    write_runtime_metrics_jsonl=plan.write_runtime_metrics_jsonl, print_epoch_summary=plan.print_epoch_summary,
                ),
            )
            summary['stages']['softem'] = soft
            summary['selected_checkpoint_path'] = soft.get('selected_checkpoint_path')
            summary['unknown_metrics'] = soft.get('unknown_metrics', {})
            if bool(getattr(plan, 'ablate_skip_base', False)):
                summary['stages']['softem_base'] = {
                    'stage_id': 'softem_base',
                    'status': 'SKIPPED',
                    'reason': 'ablate_skip_base',
                }
    summary['post_train_audit'] = _run_post_train_minimal_split_audit(plan)
    out_path = plan.output_root / 'train' / 'pipeline_train_summary.json'
    _write_json(out_path, summary)
    return {'status':'PASS','summary_path':str(out_path),'summary':summary}
