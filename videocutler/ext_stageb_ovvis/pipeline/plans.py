from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ALLOWED_PIPELINES=('legacy','reservoir_v1')
_ALLOWED_STAGE_SCOPES=('prealign_only','prealign_base','prealign_base_aug')
_ALLOWED_METRICS_PROFILES=('default','formal')
_ALLOWED_BENCHMARKS=('lvvis',)


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
    base_release_margin:float=0.0
    ablate_skip_base:bool=False
    ablate_no_yprime_reward:bool=False
    show_progress:bool=True
    log_every:int=10
    write_runtime_metrics_jsonl:bool=True
    print_epoch_summary:bool=True


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
        subset_fraction=None if args.subset_fraction is None else float(args.subset_fraction), base_release_margin=float(args.base_release_margin),
        ablate_skip_base=bool(getattr(args, 'ablate_skip_base', False)),
        ablate_no_yprime_reward=bool(getattr(args, 'ablate_no_yprime_reward', False)), show_progress=bool(args.show_progress),
        log_every=int(args.log_every), write_runtime_metrics_jsonl=bool(args.write_runtime_metrics_jsonl),
        print_epoch_summary=bool(args.print_epoch_summary),
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
