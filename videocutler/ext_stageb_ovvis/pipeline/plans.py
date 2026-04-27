from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ALLOWED_PIPELINES=('legacy','reservoir_v1','reservoir_v1_sinkhorn_no_unknown')
_ALLOWED_TRAINING_SEMANTICS=('legacy_scta','reservoir_release')
_ALLOWED_BASE_EM_REFRESH_POLICIES=('stage_once','epoch_start')
_ALLOWED_UNKNOWN_MODES=('prototype','scalar_bias')
_ALLOWED_EXTRA_SELECTION_MODES=('trajectory_epoch_topk_nonYprime','clip_observed_dissimilar_topk')
_ALLOWED_EXTRA_ACTIVATION_MODES=('always','margin_over_yprime')
_ALLOWED_EXTRA_COVERAGE_MODES=('observed_only','unified_with_yprime')
_ALLOWED_AUG_LOSS_MODES=('soft_ce','hard_candidate_nce')
_ALLOWED_STAGE_SCOPES=('prealign_only','prealign_base','prealign_base_aug','sinkhorn_prealign_only','sinkhorn_preaug_no_unknown')
_ALLOWED_METRICS_PROFILES=('default','formal')
_ALLOWED_BENCHMARKS=('lvvis',)
_ALLOWED_SINKHORN_VOCAB_SCOPE_POLICIES=('weak_label_only','legacy_full')


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_asset_root(repo_root: Path) -> Path:
    return repo_root.parent / 'wsovvis_asserts'


@dataclass(frozen=True)
class TrainPlan:
    exp_name:str
    output_root:Path
    device:str
    seed:int
    smoke:bool
    dataset_name:str
    trajectory_source_branch:str
    pipeline:str
    stage_scope:str
    repo_root:Path
    asset_root:Path
    training_semantics:str='legacy_scta'
    batch_budget:Optional[int]=None
    prealign_epochs:Optional[int]=None
    base_epochs:Optional[int]=None
    aug_epochs:Optional[int]=None
    prealign_learning_rate:Optional[float]=None
    base_learning_rate:Optional[float]=None
    aug_learning_rate:Optional[float]=None
    weight_decay:float=1e-2
    lambda_frame:Optional[float]=None
    lambda_cov:float=1.0
    t_dis_init:float=0.07
    smoke_max_trajectories:int=128
    subset_fraction:Optional[float]=None
    k_extra:int=2
    extra_alpha:float=0.25
    extra_selection_mode:str='trajectory_epoch_topk_nonYprime'
    clip_extra_obs_sim_max:float=0.90
    clip_extra_allow_empty:bool=True
    extra_activation_mode:str='always'
    extra_activation_margin:float=0.0
    extra_penalty_scale:float=1.0
    extra_consensus_bonus_lambda:float=0.0
    extra_coverage_mode:str='observed_only'
    extra_coverage_scale:float=0.0
    aug_loss_mode:str='soft_ce'
    aug_nce_tau:float=0.07
    aug_nce_positive_min_resp:float=0.0
    aug_nce_include_unknown:bool=False
    aug_soft_ce_impl:str='legacy_loop'
    aug_soft_ce_equivalence_check_batches:int=0
    base_release_margin:float=0.0
    em_subiterations:int=2
    base_em_refresh_policy:str='stage_once'
    unknown_mode:str='prototype'
    ablate_skip_base:bool=False
    ablate_no_yprime_reward:bool=False
    show_progress:bool=True
    log_every:int=10
    write_runtime_metrics_jsonl:bool=True
    print_epoch_summary:bool=True
    sinkhorn_tau:float=0.15
    sinkhorn_iters:int=5
    sinkhorn_row_cap_scale:float=2.0
    sinkhorn_extra_demand:float=0.25
    sinkhorn_aug_extra_lambda:float=0.2
    sinkhorn_assignment_stopgrad:bool=True
    sinkhorn_safe_negatives:bool=False
    sinkhorn_safe_neg_count:int=64
    sinkhorn_safe_neg_weight:float=0.25
    sinkhorn_safe_neg_text_sim_threshold:float=0.50
    sinkhorn_safe_neg_exclude_model_topk:int=100
    sinkhorn_safe_neg_seed:int=3407
    sinkhorn_extra_margin_gate:Optional[float]=None
    sinkhorn_final_rerank_lambda_r:float=0.0
    sinkhorn_vocab_scope_policy:str='weak_label_only'
    sinkhorn_vocab_scope_strict_check:bool=True


    def __post_init__(self):
        object.__setattr__(self, 'output_root', Path(self.output_root).expanduser().resolve())
        object.__setattr__(self, 'repo_root', Path(self.repo_root).expanduser().resolve())
        object.__setattr__(self, 'asset_root', Path(self.asset_root).expanduser().resolve())

@dataclass(frozen=True)
class TestPlan:
    exp_name:str
    output_root:Path
    device:str
    seed:int
    smoke:bool
    pipeline:str
    stage_scope:str
    dataset_name:str
    benchmark:str
    repo_root:Path
    asset_root:Path
    metrics_profile:str='default'
    logit_chunk_size:int=256
    ckpt_path:Optional[str]=None


    def __post_init__(self):
        object.__setattr__(self, 'output_root', Path(self.output_root).expanduser().resolve())
        object.__setattr__(self, 'repo_root', Path(self.repo_root).expanduser().resolve())
        object.__setattr__(self, 'asset_root', Path(self.asset_root).expanduser().resolve())

def _require(v:str, allowed:tuple[str,...], name:str)->str:
    if v not in allowed:
        raise ValueError(f'{name} must be one of {allowed}, got {v!r}')
    return v


def resolve_train_plan(args)->TrainPlan:
    repo_root = Path(getattr(args, 'repo_root', None) or _default_repo_root()).expanduser().resolve()
    asset_root = Path(getattr(args, 'asset_root', None) or _default_asset_root(repo_root)).expanduser().resolve()
    return TrainPlan(
        exp_name=str(args.exp_name), output_root=Path(args.output_root).expanduser().resolve(), device=str(args.device),
        seed=int(args.seed), smoke=bool(args.smoke), dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(args.trajectory_source_branch), pipeline=_require(str(args.pipeline), _ALLOWED_PIPELINES, 'pipeline'),
        training_semantics=_require(str(getattr(args, 'training_semantics', 'legacy_scta')), _ALLOWED_TRAINING_SEMANTICS, 'training_semantics'),
        stage_scope=_require(str(args.stage_scope), _ALLOWED_STAGE_SCOPES, 'stage_scope'), repo_root=repo_root, asset_root=asset_root,
        batch_budget=None if args.batch_budget is None else int(args.batch_budget),
        prealign_epochs=None if args.prealign_epochs is None else int(args.prealign_epochs),
        base_epochs=None if args.base_epochs is None else int(args.base_epochs),
        aug_epochs=None if args.aug_epochs is None else int(args.aug_epochs),
        prealign_learning_rate=None if args.prealign_learning_rate is None else float(args.prealign_learning_rate),
        base_learning_rate=None if args.base_learning_rate is None else float(args.base_learning_rate),
        aug_learning_rate=None if args.aug_learning_rate is None else float(args.aug_learning_rate),
        weight_decay=float(args.weight_decay), lambda_frame=None if args.lambda_frame is None else float(args.lambda_frame),
        lambda_cov=float(args.lambda_cov), t_dis_init=float(args.t_dis_init), smoke_max_trajectories=int(args.smoke_max_trajectories),
        subset_fraction=None if args.subset_fraction is None else float(args.subset_fraction), k_extra=int(getattr(args, 'k_extra', 2)), extra_alpha=float(getattr(args, 'extra_alpha', 0.25)), extra_selection_mode=_require(str(getattr(args, 'extra_selection_mode', 'trajectory_epoch_topk_nonYprime')), _ALLOWED_EXTRA_SELECTION_MODES, 'extra_selection_mode'), clip_extra_obs_sim_max=float(getattr(args, 'clip_extra_obs_sim_max', 0.90)), clip_extra_allow_empty=bool(getattr(args, 'clip_extra_allow_empty', True)), extra_activation_mode=_require(str(getattr(args, 'extra_activation_mode', 'always')), _ALLOWED_EXTRA_ACTIVATION_MODES, 'extra_activation_mode'), extra_activation_margin=float(getattr(args, 'extra_activation_margin', 0.0)), extra_penalty_scale=float(getattr(args, 'extra_penalty_scale', 1.0)), extra_consensus_bonus_lambda=float(getattr(args, 'extra_consensus_bonus_lambda', 0.0)), extra_coverage_mode=_require(str(getattr(args, 'extra_coverage_mode', 'observed_only')), _ALLOWED_EXTRA_COVERAGE_MODES, 'extra_coverage_mode'), extra_coverage_scale=float(getattr(args, 'extra_coverage_scale', 0.0)), aug_loss_mode=_require(str(getattr(args, 'aug_loss_mode', 'soft_ce')), _ALLOWED_AUG_LOSS_MODES, 'aug_loss_mode'), aug_nce_tau=float(getattr(args, 'aug_nce_tau', 0.07)), aug_nce_positive_min_resp=float(getattr(args, 'aug_nce_positive_min_resp', 0.0)), aug_nce_include_unknown=bool(getattr(args, 'aug_nce_include_unknown', False)), aug_soft_ce_impl=_require(str(getattr(args, 'aug_soft_ce_impl', 'legacy_loop')), {'legacy_loop','batched'}, 'aug_soft_ce_impl'), aug_soft_ce_equivalence_check_batches=max(0, int(getattr(args, 'aug_soft_ce_equivalence_check_batches', 0))), base_release_margin=float(args.base_release_margin),
        em_subiterations=max(0, int(getattr(args, 'em_subiterations', 2))),
        base_em_refresh_policy=_require(str(getattr(args, 'base_em_refresh_policy', 'stage_once')), _ALLOWED_BASE_EM_REFRESH_POLICIES, 'base_em_refresh_policy'),
        unknown_mode=_require(str(getattr(args, 'unknown_mode', 'prototype')), _ALLOWED_UNKNOWN_MODES, 'unknown_mode'),
        ablate_skip_base=bool(getattr(args, 'ablate_skip_base', False)),
        ablate_no_yprime_reward=bool(getattr(args, 'ablate_no_yprime_reward', False)), show_progress=bool(args.show_progress),
        log_every=int(args.log_every), write_runtime_metrics_jsonl=bool(args.write_runtime_metrics_jsonl),
        print_epoch_summary=bool(args.print_epoch_summary),
        sinkhorn_tau=float(getattr(args, 'sinkhorn_tau', 0.15)),
        sinkhorn_iters=max(1, int(getattr(args, 'sinkhorn_iters', 5))),
        sinkhorn_row_cap_scale=float(getattr(args, 'sinkhorn_row_cap_scale', 2.0)),
        sinkhorn_extra_demand=float(getattr(args, 'sinkhorn_extra_demand', 0.25)),
        sinkhorn_aug_extra_lambda=float(getattr(args, 'sinkhorn_aug_extra_lambda', 0.2)),
        sinkhorn_assignment_stopgrad=bool(getattr(args, 'sinkhorn_assignment_stopgrad', True)),
        sinkhorn_safe_negatives=bool(getattr(args, 'sinkhorn_safe_negatives', False)),
        sinkhorn_safe_neg_count=max(0, int(getattr(args, 'sinkhorn_safe_neg_count', 64))),
        sinkhorn_safe_neg_weight=float(getattr(args, 'sinkhorn_safe_neg_weight', 0.25)),
        sinkhorn_safe_neg_text_sim_threshold=float(getattr(args, 'sinkhorn_safe_neg_text_sim_threshold', 0.50)),
        sinkhorn_safe_neg_exclude_model_topk=max(0, int(getattr(args, 'sinkhorn_safe_neg_exclude_model_topk', 100))),
        sinkhorn_safe_neg_seed=int(getattr(args, 'sinkhorn_safe_neg_seed', 3407)),
        sinkhorn_extra_margin_gate=None if getattr(args, 'sinkhorn_extra_margin_gate', None) is None else float(getattr(args, 'sinkhorn_extra_margin_gate', None)),
        sinkhorn_final_rerank_lambda_r=float(getattr(args, 'sinkhorn_final_rerank_lambda_r', 0.0)),
        sinkhorn_vocab_scope_policy=_require(str(getattr(args, 'sinkhorn_vocab_scope_policy', 'weak_label_only')), _ALLOWED_SINKHORN_VOCAB_SCOPE_POLICIES, 'sinkhorn_vocab_scope_policy'),
        sinkhorn_vocab_scope_strict_check=bool(getattr(args, 'sinkhorn_vocab_scope_strict_check', True)),
    )


def resolve_test_plan(args)->TestPlan:
    repo_root = Path(getattr(args, 'repo_root', None) or _default_repo_root()).expanduser().resolve()
    asset_root = Path(getattr(args, 'asset_root', None) or _default_asset_root(repo_root)).expanduser().resolve()
    return TestPlan(
        exp_name=str(args.exp_name), output_root=Path(args.output_root).expanduser().resolve(), device=str(args.device),
        seed=int(args.seed), smoke=bool(args.smoke), pipeline=_require(str(args.pipeline), _ALLOWED_PIPELINES, 'pipeline'),
        stage_scope=_require(str(args.stage_scope), _ALLOWED_STAGE_SCOPES, 'stage_scope'), dataset_name=str(args.dataset_name),
        benchmark=_require(str(args.benchmark), _ALLOWED_BENCHMARKS, 'benchmark'), repo_root=repo_root, asset_root=asset_root,
        metrics_profile=_require(str(args.metrics_profile), _ALLOWED_METRICS_PROFILES, 'metrics_profile'),
        logit_chunk_size=int(args.logit_chunk_size), ckpt_path=None if args.ckpt_path is None else str(args.ckpt_path),
    )
