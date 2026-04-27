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
from videocutler.ext_stageb_ovvis.algorithms.reservoir_v1 import ReservoirPrealignConfig, ReservoirSoftEMConfig, ReservoirSinkhornNoUnknownConfig, train_reservoir_prealign, run_reservoir_soft_em, run_reservoir_sinkhorn_no_unknown
from videocutler.ext_stageb_ovvis.algorithms.legacy_scta_backend import LegacySCTABackendConfig, run_legacy_scta_soft_em_via_reservoir
from videocutler.ext_stageb_ovvis.audit.g8_minimal_split_audit import MinimalSplitAuditConfig, run_minimal_split_audit
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import TRAIN_DATASETS

REPO_ASSET_LINK_NAMES = ('exports', 'exports_gt', 'carrier_bank', 'carrier_bank_gt', 'frame_bank', 'text_bank', 'gt_sidecar_bank', 'weak_labels', 'weights', 'dataset', 'eval')


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')




def _canonical_mirror_output_root(plan: TrainPlan) -> Path:
    default_root = Path(plan.repo_root) / 'codex' / 'outputs' / 'G8_inference_and_eval' / str(plan.exp_name)
    try:
        return default_root.resolve()
    except Exception:
        return default_root


def _iter_writeback_roots(plan: TrainPlan) -> Sequence[Path]:
    roots: list[Path] = []
    primary = Path(plan.output_root).expanduser().resolve()
    roots.append(primary)
    mirror = _canonical_mirror_output_root(plan)
    if mirror not in roots:
        roots.append(mirror)
    return roots


def _minimal_prealign_stage_summary(plan: TrainPlan, pre: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'stage_id': 'prealign',
        'pipeline': str(plan.pipeline),
        'training_semantics': str(getattr(plan, 'training_semantics', 'legacy_scta')),
        'loss_mean': pre.get('loss_mean', 0.0),
        'loss_last': pre.get('loss_last', 0.0),
        'optimization_loss_mean': pre.get('optimization_loss_mean', 0.0),
        'optimization_loss_last': pre.get('optimization_loss_last', 0.0),
        'unknown_metrics': dict(pre.get('unknown_metrics', {})),
    }


def _ensure_train_writeback(plan: TrainPlan, summary: Dict[str, Any], *, pre: Dict[str, Any] | None = None, soft: Dict[str, Any] | None = None) -> Path:
    roots = _iter_writeback_roots(plan)
    for root in roots:
        (root / 'train').mkdir(parents=True, exist_ok=True)
        if isinstance(pre, dict) and pre:
            stage_summary_path = root / 'train' / 'prealign' / 'stage_summary.json'
            if not stage_summary_path.exists():
                _write_json(stage_summary_path, _minimal_prealign_stage_summary(plan, pre))
        if isinstance(soft, dict) and soft:
            for stage_report in soft.get('stage_reports', []) or []:
                if not isinstance(stage_report, dict):
                    continue
                relpath = stage_report.get('stage_summary_relpath')
                payload = stage_report.get('stage_summary_payload')
                if relpath and isinstance(payload, dict):
                    stage_summary_path = root / str(relpath)
                    if not stage_summary_path.exists():
                        _write_json(stage_summary_path, payload)
        out_path = root / 'train' / 'pipeline_train_summary.json'
        _write_json(out_path, summary)
    return roots[0] / 'train' / 'pipeline_train_summary.json'


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
    sinkhorn_scopes = {'sinkhorn_prealign_only', 'sinkhorn_preaug_no_unknown'}
    if str(plan.pipeline) == 'reservoir_v1_sinkhorn_no_unknown' and str(plan.stage_scope) not in sinkhorn_scopes:
        raise ValueError('reservoir_v1_sinkhorn_no_unknown requires stage_scope sinkhorn_prealign_only or sinkhorn_preaug_no_unknown')
    if str(plan.pipeline) != 'reservoir_v1_sinkhorn_no_unknown' and str(plan.stage_scope) in sinkhorn_scopes:
        raise ValueError('sinkhorn stage_scope is only valid with pipeline reservoir_v1_sinkhorn_no_unknown')
    materialized = _materialize(plan)
    summary = {
        'exp_name': plan.exp_name,
        'pipeline': plan.pipeline,
        'training_semantics': str(getattr(plan, 'training_semantics', 'legacy_scta')),
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
    elif plan.pipeline == 'reservoir_v1_sinkhorn_no_unknown':
        reservoir_samples = _resolve_materialized_samples(materialized, prefer_valid=True)
        sink = run_reservoir_sinkhorn_no_unknown(
            output_root=plan.output_root,
            materialized_samples=reservoir_samples,
            stage_scope=str(plan.stage_scope),
            config=ReservoirSinkhornNoUnknownConfig(
                dataset_name=plan.dataset_name, trajectory_source_branch=plan.trajectory_source_branch, device=plan.device,
                seed=plan.seed, smoke=plan.smoke, prealign_epochs=(1 if plan.smoke else 5) if plan.prealign_epochs is None else plan.prealign_epochs,
                aug_epochs=(1 if plan.smoke else 5) if plan.aug_epochs is None else plan.aug_epochs,
                prealign_learning_rate=(1e-4 if plan.prealign_learning_rate is None else plan.prealign_learning_rate),
                aug_learning_rate=(5e-5 if plan.aug_learning_rate is None else plan.aug_learning_rate),
                weight_decay=plan.weight_decay, t_dis_init=plan.t_dis_init, lambda_frame=(0.25 if plan.lambda_frame is None else plan.lambda_frame),
                runtime_asset_source=str(materialized['resolution'].get('runtime_asset_source','local_canonical_assets')),
                runtime_asset_source_local_incomplete=bool(materialized['resolution'].get('local_incomplete',False)),
                runtime_asset_output_root=str(materialized['resolution'].get('runtime_output_root', str(plan.repo_root))),
                batch_budget=plan.batch_budget, k_extra=int(getattr(plan, 'k_extra', 2)), extra_alpha=float(getattr(plan, 'extra_alpha', 0.25)),
                sinkhorn_tau=float(getattr(plan, 'sinkhorn_tau', 0.15)), sinkhorn_iters=int(getattr(plan, 'sinkhorn_iters', 5)),
                sinkhorn_row_cap_scale=float(getattr(plan, 'sinkhorn_row_cap_scale', 2.0)), sinkhorn_extra_demand=float(getattr(plan, 'sinkhorn_extra_demand', 0.25)),
                sinkhorn_aug_extra_lambda=float(getattr(plan, 'sinkhorn_aug_extra_lambda', 0.2)), sinkhorn_assignment_stopgrad=bool(getattr(plan, 'sinkhorn_assignment_stopgrad', True)),
                sinkhorn_safe_negatives=bool(getattr(plan, 'sinkhorn_safe_negatives', False)), sinkhorn_safe_neg_count=int(getattr(plan, 'sinkhorn_safe_neg_count', 64)),
                sinkhorn_safe_neg_weight=float(getattr(plan, 'sinkhorn_safe_neg_weight', 0.25)), sinkhorn_safe_neg_text_sim_threshold=float(getattr(plan, 'sinkhorn_safe_neg_text_sim_threshold', 0.50)),
                sinkhorn_safe_neg_exclude_model_topk=int(getattr(plan, 'sinkhorn_safe_neg_exclude_model_topk', 100)), sinkhorn_safe_neg_seed=int(getattr(plan, 'sinkhorn_safe_neg_seed', 3407)),
                show_progress=plan.show_progress, log_every=plan.log_every,
                write_runtime_metrics_jsonl=plan.write_runtime_metrics_jsonl, print_epoch_summary=plan.print_epoch_summary,
            ),
        )
        summary['stages']['sinkhorn_no_unknown'] = sink
        summary['selected_checkpoint_path'] = sink.get('selected_checkpoint_path')
        summary['unknown_disabled'] = True
        summary['softem_base_skipped'] = True
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
            if str(getattr(plan, 'training_semantics', 'legacy_scta')) == 'legacy_scta':
                soft = run_legacy_scta_soft_em_via_reservoir(
                    output_root=plan.output_root,
                    materialized_samples=reservoir_samples,
                    config=LegacySCTABackendConfig(
                        dataset_name=plan.dataset_name, trajectory_source_branch=plan.trajectory_source_branch, mode=mode,
                        device=plan.device, seed=plan.seed, smoke=plan.smoke, lambda_frame=(0.25 if plan.lambda_frame is None else plan.lambda_frame),
                        lambda_cov=plan.lambda_cov, t_dis_init=plan.t_dis_init, weight_decay=plan.weight_decay,
                        base_epochs=(1 if plan.smoke else 5) if plan.base_epochs is None else plan.base_epochs,
                        aug_epochs=(1 if plan.smoke else 5) if plan.aug_epochs is None else plan.aug_epochs,
                        base_learning_rate=(5e-5 if plan.base_learning_rate is None else plan.base_learning_rate),
                        aug_learning_rate=(5e-5 if plan.aug_learning_rate is None else plan.aug_learning_rate),
                        k_extra=int(getattr(plan, 'k_extra', 2)), extra_alpha=float(getattr(plan, 'extra_alpha', 0.25)), extra_selection_mode=str(getattr(plan, 'extra_selection_mode', 'trajectory_epoch_topk_nonYprime')), clip_extra_obs_sim_max=float(getattr(plan, 'clip_extra_obs_sim_max', 0.90)), clip_extra_allow_empty=bool(getattr(plan, 'clip_extra_allow_empty', True)), extra_activation_mode=str(getattr(plan, 'extra_activation_mode', 'always')), extra_activation_margin=float(getattr(plan, 'extra_activation_margin', 0.0)), extra_penalty_scale=float(getattr(plan, 'extra_penalty_scale', 1.0)), extra_consensus_bonus_lambda=float(getattr(plan, 'extra_consensus_bonus_lambda', 0.0)), extra_coverage_mode=str(getattr(plan, 'extra_coverage_mode', 'observed_only')), extra_coverage_scale=float(getattr(plan, 'extra_coverage_scale', 0.0)), aug_loss_mode=str(getattr(plan, 'aug_loss_mode', 'soft_ce')), aug_nce_tau=float(getattr(plan, 'aug_nce_tau', 0.07)), aug_nce_positive_min_resp=float(getattr(plan, 'aug_nce_positive_min_resp', 0.0)), aug_nce_include_unknown=bool(getattr(plan, 'aug_nce_include_unknown', False)), aug_soft_ce_impl=str(getattr(plan, 'aug_soft_ce_impl', 'legacy_loop')), aug_soft_ce_equivalence_check_batches=max(0, int(getattr(plan, 'aug_soft_ce_equivalence_check_batches', 0))),
                        em_subiterations=max(0, int(getattr(plan, 'em_subiterations', 2))),
                        base_em_refresh_policy=str(getattr(plan, 'base_em_refresh_policy', 'stage_once')),
                        unknown_mode=str(getattr(plan, 'unknown_mode', 'prototype')),
                        runtime_asset_source=str(materialized['resolution'].get('runtime_asset_source','local_canonical_assets')),
                        runtime_asset_source_local_incomplete=bool(materialized['resolution'].get('local_incomplete',False)),
                        runtime_asset_output_root=str(materialized['resolution'].get('runtime_output_root', str(plan.repo_root))),
                        batch_budget=plan.batch_budget, ablate_skip_base=bool(getattr(plan, 'ablate_skip_base', False)),
                        show_progress=plan.show_progress, log_every=plan.log_every,
                        write_runtime_metrics_jsonl=plan.write_runtime_metrics_jsonl, print_epoch_summary=plan.print_epoch_summary,
                    ),
                )
            else:
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
                        batch_budget=plan.batch_budget, base_release_margin=float(plan.base_release_margin),
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
    if str(plan.pipeline) == 'reservoir_v1_sinkhorn_no_unknown':
        summary['post_train_audit'] = {'status': 'SKIPPED', 'reason': 'sinkhorn_no_unknown_experimental_branch'}
    else:
        summary['post_train_audit'] = _run_post_train_minimal_split_audit(plan)
    out_path = _ensure_train_writeback(plan, summary, pre=summary['stages'].get('prealign'), soft=summary['stages'].get('softem'))
    return {'status':'PASS','summary_path':str(out_path),'summary':summary}
