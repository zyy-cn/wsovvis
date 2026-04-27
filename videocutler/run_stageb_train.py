from __future__ import annotations
import argparse
from pathlib import Path
import sys

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

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Unified Stage-B training entrypoint.')
    p.add_argument('--exp_name', required=True); p.add_argument('--output_root', required=True); p.add_argument('--device', required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--smoke', action='store_true')
    p.add_argument('--dataset_name', default='lvvis_train_base', choices=('lvvis_train_base','lvvis_val')); p.add_argument('--trajectory_source_branch', default='mainline', choices=('mainline','gt_upper_bound')); p.add_argument('--pipeline', default='legacy', choices=('legacy','reservoir_v1','reservoir_v1_sinkhorn_no_unknown')); p.add_argument('--training_semantics', default='legacy_scta', choices=('legacy_scta','reservoir_release')); p.add_argument('--stage_scope', default='prealign_base_aug', choices=('prealign_only','prealign_base','prealign_base_aug','sinkhorn_prealign_only','sinkhorn_preaug_no_unknown')); p.add_argument('--smoke_max_trajectories', type=int, default=128)
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
    return p.parse_args()
def main() -> int:
    result=run_train_pipeline(resolve_train_plan(parse_args())); print(result['summary_path']); return 0
if __name__=='__main__': raise SystemExit(main())
