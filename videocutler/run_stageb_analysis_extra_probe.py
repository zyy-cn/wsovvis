from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.analysis.extra_attribution_probe import (  # noqa: E402
    ExtraAttributionProbeConfig,
    STAGE_ORDER,
    run_extra_attribution_probe,
)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run extra attribution diagnostic probe on existing stage checkpoints")
    p.add_argument("--run_root", required=True)
    p.add_argument("--runtime_output_root", default=None)
    p.add_argument("--dataset_name", default="lvvis_val", choices=("lvvis_train_base", "lvvis_val"))
    p.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline", "gt_upper_bound"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", type=_parse_bool, default=False)
    p.add_argument("--smoke_max_trajectories", type=int, default=128)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--stage_scope", nargs="*", default=list(STAGE_ORDER))
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--extra_candidate_topk", type=int, default=5)
    p.add_argument("--top_failure_cases", type=int, default=64)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--sidecar_root", default=None)
    p.add_argument("--repo_root", default=None)
    p.add_argument("--show_progress", type=_parse_bool, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root is not None else Path(__file__).resolve().parents[1]
    runtime_output_root = Path(args.runtime_output_root).expanduser().resolve() if args.runtime_output_root else repo_root
    stage_scope = tuple(str(x) for x in list(args.stage_scope or STAGE_ORDER))
    result = run_extra_attribution_probe(
        ExtraAttributionProbeConfig(
            run_root=Path(args.run_root).expanduser().resolve(),
            runtime_output_root=runtime_output_root,
            dataset_name=str(args.dataset_name),
            trajectory_source_branch=str(args.trajectory_source_branch),
            device=str(args.device),
            smoke=bool(args.smoke),
            smoke_max_trajectories=int(args.smoke_max_trajectories),
            subset_fraction=None if args.subset_fraction is None else float(args.subset_fraction),
            stage_scope=stage_scope,
            batch_size=max(1, int(args.batch_size)),
            extra_candidate_topk=max(1, int(args.extra_candidate_topk)),
            top_failure_cases=max(1, int(args.top_failure_cases)),
            output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
            sidecar_root=Path(args.sidecar_root).expanduser().resolve() if args.sidecar_root else None,
            show_progress=bool(args.show_progress),
        )
    )
    print(result["comparison_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
