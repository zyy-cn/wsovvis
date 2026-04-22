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
    p.add_argument('--dataset_name', default='lvvis_train_base', choices=('lvvis_train_base','lvvis_val')); p.add_argument('--trajectory_source_branch', default='mainline', choices=('mainline','gt_upper_bound')); p.add_argument('--pipeline', default='legacy', choices=('legacy','reservoir_v1')); p.add_argument('--stage_scope', default='prealign_base_aug', choices=('prealign_only','prealign_base','prealign_base_aug')); p.add_argument('--smoke_max_trajectories', type=int, default=128)
    p.add_argument('--prealign_epochs', type=int, default=None); p.add_argument('--base_epochs', type=int, default=None); p.add_argument('--aug_epochs', type=int, default=None)
    p.add_argument('--prealign_learning_rate', type=float, default=None); p.add_argument('--base_learning_rate', type=float, default=None); p.add_argument('--aug_learning_rate', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=1e-2); p.add_argument('--t_dis_init', type=float, default=0.07); p.add_argument('--lambda_frame', type=float, default=None); p.add_argument('--lambda_cov', type=float, default=1.0); p.add_argument('--subset_fraction', type=float, default=None); p.add_argument('--batch_budget', type=int, default=None); p.add_argument('--show_progress', type=_parse_bool, default=True); p.add_argument('--log_every', type=int, default=10); p.add_argument('--write_runtime_metrics_jsonl', type=_parse_bool, default=True); p.add_argument('--print_epoch_summary', type=_parse_bool, default=True)
    return p.parse_args()
def main() -> int:
    result=run_train_pipeline(resolve_train_plan(parse_args())); print(result['summary_path']); return 0
if __name__=='__main__': raise SystemExit(main())
