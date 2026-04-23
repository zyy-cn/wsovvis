
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

from videocutler.ext_stageb_ovvis.analysis.prealign_mass_probe import (  # noqa: E402
    PrealignMassProbeConfig,
    run_prealign_mass_probe,
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
    parser = argparse.ArgumentParser(description="Prealign GT-set mass probe.")
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--runtime_output_root", required=False, default=None)
    parser.add_argument("--dataset_name", default="lvvis_train_base", choices=("lvvis_train_base", "lvvis_val"))
    parser.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline", "gt_upper_bound"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", type=_parse_bool, default=False)
    parser.add_argument("--smoke_max_trajectories", type=int, default=128)
    parser.add_argument("--subset_fraction", type=float, default=None)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--sidecar_root", default=None)
    parser.add_argument("--repo_root", default=None)
    parser.add_argument("--show_progress", type=_parse_bool, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root is not None else Path(__file__).resolve().parents[1]
    runtime_output_root = Path(args.runtime_output_root).expanduser().resolve() if args.runtime_output_root is not None else repo_root
    result = run_prealign_mass_probe(
        PrealignMassProbeConfig(
            run_root=Path(args.run_root).expanduser().resolve(),
            runtime_output_root=runtime_output_root,
            dataset_name=str(args.dataset_name),
            trajectory_source_branch=str(args.trajectory_source_branch),
            device=str(args.device),
            smoke=bool(args.smoke),
            smoke_max_trajectories=int(args.smoke_max_trajectories),
            subset_fraction=None if args.subset_fraction is None else float(args.subset_fraction),
            checkpoint_path=None if args.checkpoint_path is None else Path(args.checkpoint_path).expanduser().resolve(),
            output_dir=None if args.output_dir is None else Path(args.output_dir).expanduser().resolve(),
            sidecar_root=None if args.sidecar_root is None else Path(args.sidecar_root).expanduser().resolve(),
            show_progress=bool(args.show_progress),
        )
    )
    print(result["summary_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
