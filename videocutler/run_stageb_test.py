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
from videocutler.ext_stageb_ovvis.pipeline.plans import resolve_test_plan
from videocutler.ext_stageb_ovvis.pipeline.test_orchestrator import run_test_pipeline

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Unified Stage-B testing entrypoint.')
    p.add_argument('--exp_name', required=True); p.add_argument('--output_root', required=True); p.add_argument('--device', required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--smoke', action='store_true')
    p.add_argument('--pipeline', default='legacy', choices=('legacy','reservoir_v1')); p.add_argument('--stage_scope', default='prealign_base_aug', choices=('prealign_only','prealign_base','prealign_base_aug')); p.add_argument('--dataset_name', default='lvvis_val', choices=('lvvis_val','ytvis_2019_val')); p.add_argument('--benchmark', default='lvvis', choices=('lvvis',)); p.add_argument('--metrics_profile', default='default', choices=('default','formal')); p.add_argument('--logit_chunk_size', type=int, default=256); p.add_argument('--ckpt_path', default=None)
    return p.parse_args()
def main() -> int:
    result=run_test_pipeline(resolve_test_plan(parse_args())); print(result['summary_path']); return 0
if __name__=='__main__': raise SystemExit(main())
