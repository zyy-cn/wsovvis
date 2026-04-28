from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json

def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
_bootstrap_repo_root_for_direct_cli()
from videocutler.ext_stageb_ovvis.pipeline.plans import resolve_train_plan
from videocutler.ext_stageb_ovvis.pipeline.train_orchestrator import run_train_pipeline

def _parse_bool(value):
    if isinstance(value, bool): return value
    normalized = str(value).strip().lower()
    if normalized in {'1','true','t','yes','y','on'}: return True
    if normalized in {'0','false','f','no','n','off'}: return False
    raise argparse.ArgumentTypeError(f'expected boolean, got {value!r}')

def _merge_scope_contract_into_pipeline_summary(summary_path: Path) -> None:
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception:
        return
    if not isinstance(summary, dict):
        return
    for stage_rel in (
        summary_path.parent / 'softem_aug' / 'stage_summary.json',
        summary_path.parent / 'prealign' / 'stage_summary.json',
    ):
        try:
            if not stage_rel.exists():
                continue
            stage_summary = json.loads(stage_rel.read_text(encoding='utf-8'))
            if not isinstance(stage_summary, dict):
                continue
            for key in (
                'vocab_scope_policy',
                'weak_vocab_count',
                'full_text_vocab_count',
                'extra_scope',
                'safe_neg_scope',
                'model_topk_scope',
                'extra_outside_weak_count',
                'safe_neg_outside_weak_count',
                'model_topk_outside_weak_count',
                'denominator_outside_weak_count',
                'responsibility_candidate_outside_weak_count',
            ):
                if key in stage_summary:
                    summary[key] = stage_summary.get(key)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
            return
        except Exception:
            continue

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Unified Stage-B training entrypoint.')
    p.add_argument('--exp_name', required=True); p.add_argument('--output_root', required=True); p.add_argument('--device', required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--smoke', action='store_true')
    p.add_argument('--dataset_name', default='lvvis_train_base', choices=('lvvis_train_base','lvvis_val')); p.add_argument('--trajectory_source_branch', default='mainline', choices=('mainline','gt_upper_bound')); p.add_argument('--pipeline', default='legacy', choices=('legacy','reservoir_v1','reservoir_v1_sinkhorn_no_unknown')); p.add_argument('--training_semantics', default='legacy_scta', choices=('legacy_scta','reservoir_release')); p.add_argument('--stage_scope', default='prealign_base_aug', choices=('prealign_only','prealign_base','prealign_base_aug','sinkhorn_prealign_only','sinkhorn_preaug_no_unknown','support_null_prealign_base_only')); p.add_argument('--smoke_max_trajectories', type=int, default=128)
    p.add_argument('--repo_root', default=None); p.add_argument('--asset_root', default=None)
    p.add_argument('--prealign_epochs', type=int, default=None); p.add_argument('--base_epochs', type=int, default=None); p.add_argument('--aug_epochs', type=int, default=None)
    p.add_argument('--prealign_learning_rate', type=float, default=None); p.add_argument('--base_learning_rate', type=float, default=None); p.add_argument('--aug_learning_rate', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-2); p.add_argument('--unknown_mode', default='prototype', choices=('prototype','scalar_bias')); p.add_argument('--t_dis_init', type=float, default=0.07); p.add_argument('--lambda_frame', type=float, default=None); p.add_argument('--lambda_cov', type=float, default=1.0); p.add_argument('--subset_fraction', type=float, default=None); p.add_argument('--batch_budget', type=int, default=None); p.add_argument('--k_extra', type=int, default=2); p.add_argument('--extra_alpha', type=float, default=0.25); p.add_argument('--extra_selection_mode', default='trajectory_epoch_topk_nonYprime', choices=('trajectory_epoch_topk_nonYprime','clip_observed_dissimilar_topk')); p.add_argument('--clip_extra_obs_sim_max', type=float, default=0.90); p.add_argument('--clip_extra_allow_empty', type=_parse_bool, default=True); p.add_argument('--extra_activation_mode', default='always', choices=('always','margin_over_yprime')); p.add_argument('--extra_activation_margin', type=float, default=0.0); p.add_argument('--extra_penalty_scale', type=float, default=1.0); p.add_argument('--extra_consensus_bonus_lambda', type=float, default=0.0); p.add_argument('--extra_coverage_mode', default='observed_only', choices=('observed_only','unified_with_yprime')); p.add_argument('--extra_coverage_scale', type=float, default=0.0); p.add_argument('--base_release_margin', type=float, default=0.0); p.add_argument('--em_subiterations', type=int, default=2); p.add_argument('--base_em_refresh_policy', default='stage_once', choices=('stage_once','epoch_start')); p.add_argument('--ablate_skip_base', type=_parse_bool, default=False); p.add_argument('--ablate_no_yprime_reward', type=_parse_bool, default=False); p.add_argument('--show_progress', type=_parse_bool, default=True); p.add_argument('--log_every', type=int, default=10); p.add_argument('--write_runtime_metrics_jsonl', type=_parse_bool, default=True); p.add_argument('--print_epoch_summary', type=_parse_bool, default=True)
    p.add_argument('--aug_loss_mode', default='soft_ce', choices=('soft_ce','hard_candidate_nce'))
    p.add_argument('--aug_nce_tau', type=float, default=0.07)
    p.add_argument('--aug_nce_positive_min_resp', type=float, default=0.0)
    p.add_argument('--aug_nce_include_unknown', type=_parse_bool, default=False)
    p.add_argument('--aug_soft_ce_impl', default='legacy_loop', choices=('legacy_loop','batched'))
    p.add_argument('--aug_soft_ce_equivalence_check_batches', type=int, default=0)
    p.add_argument('--sinkhorn_tau', type=float, default=0.15)
    p.add_argument('--sinkhorn_iters', type=int, default=5)
    p.add_argument('--sinkhorn_row_cap_scale', type=float, default=2.0)
    p.add_argument('--sinkhorn_extra_demand', type=float, default=0.25)
    p.add_argument('--sinkhorn_aug_extra_lambda', type=float, default=0.2)
    p.add_argument('--sinkhorn_assignment_stopgrad', type=_parse_bool, default=True)

    p.add_argument('--sinkhorn_safe_negatives', type=_parse_bool, default=False)
    p.add_argument('--sinkhorn_safe_neg_count', type=int, default=64)
    p.add_argument('--sinkhorn_safe_neg_weight', type=float, default=0.25)
    p.add_argument('--sinkhorn_safe_neg_text_sim_threshold', type=float, default=0.50)
    p.add_argument('--sinkhorn_safe_neg_exclude_model_topk', type=int, default=100)
    p.add_argument('--sinkhorn_safe_neg_seed', type=int, default=3407)
    p.add_argument('--sinkhorn_extra_margin_gate', type=float, default=None)
    p.add_argument('--sinkhorn_final_rerank_lambda_r', type=float, default=0.0)
    p.add_argument('--sinkhorn_vocab_scope_policy', default='weak_label_only', choices=('weak_label_only','legacy_full'), help='Sinkhorn/no-unknown training category scope. weak_label_only restricts extra, safe negatives, model-topK, denominator, and responsibility candidates to union(Y-prime) from weak_labels_train.json. legacy_full preserves the retired full-vocab behavior for old-result reproduction only.')
    p.add_argument('--sinkhorn_vocab_scope_strict_check', type=_parse_bool, default=True)
    # Support-null prealign/base branch. Defaults preserve old behavior.
    p.add_argument('--sinkhorn_enable_null_column', type=_parse_bool, default=False)
    p.add_argument('--sinkhorn_null_logit_bias', type=float, default=0.0)
    p.add_argument('--sinkhorn_null_residual', type=_parse_bool, default=False)
    p.add_argument('--sinkhorn_null_demand_cap_ratio', type=float, default=1.0, help='Optional cap for NULL demand as a ratio of valid trajectory count. 1.0 preserves residual behavior; <1 bounds dustbin mass.')
    p.add_argument('--sinkhorn_support_warmup_epochs', type=int, default=0)
    p.add_argument('--sinkhorn_yprime_demand_mode', default='fixed', choices=('fixed','support_ema','relative_margin_ema'), help='fixed: old behavior; support_ema: absolute top-k score confidence; relative_margin_ema: top-k margin vs competing Y-prime/null confidence.')
    p.add_argument('--sinkhorn_yprime_demand_min', type=float, default=0.10)
    p.add_argument('--sinkhorn_yprime_support_topk', type=int, default=2)
    p.add_argument('--sinkhorn_yprime_support_temp', type=float, default=0.25)
    p.add_argument('--sinkhorn_yprime_support_ema', type=float, default=0.90)
    p.add_argument('--sinkhorn_null_collapse_max', type=float, default=0.85)
    p.add_argument('--sinkhorn_yprime_demand_min_guard', type=float, default=0.20)
    # V2-C positive-support protection. Defaults preserve old/V2-B behavior.
    p.add_argument('--sinkhorn_enable_positive_protection', type=_parse_bool, default=False)
    p.add_argument('--sinkhorn_positive_margin_threshold', type=float, default=0.15)
    p.add_argument('--sinkhorn_positive_margin_temp', type=float, default=0.10)
    p.add_argument('--sinkhorn_positive_null_cap', type=float, default=0.40)
    p.add_argument('--sinkhorn_positive_redistribute_mode', default='best_y', choices=('best_y',))
    return p.parse_args()
def main() -> int:
    result=run_train_pipeline(resolve_train_plan(parse_args()))
    _merge_scope_contract_into_pipeline_summary(Path(result['summary_path']))
    print(result['summary_path'])
    return 0
if __name__=='__main__': raise SystemExit(main())
