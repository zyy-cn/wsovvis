from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.audit.train_gt_rank_audit import TrainGtRankAuditConfig, run_train_gt_rank_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G7 train-time GT rank audit on matched trajectories.")
    parser.add_argument("--dataset_name", default="lvvis_train_base", choices=("lvvis_train_base",))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline", "gt_upper_bound"))
    parser.add_argument("--stage", default="all", choices=("prealign", "softem_base", "softem_aug", "all"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--logit_chunk_size", type=int, default=256)
    parser.add_argument("--generate_sidecars", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_train_gt_rank_audit(
        TrainGtRankAuditConfig(
            output_root=Path(args.output_root),
            dataset_name=str(args.dataset_name),
            trajectory_source_branch=str(args.trajectory_source_branch),
            stage=str(args.stage),
            device=str(args.device),
            logit_chunk_size=int(args.logit_chunk_size),
            generate_sidecars=bool(args.generate_sidecars),
        )
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
