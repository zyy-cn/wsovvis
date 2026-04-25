from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.analysis.extra_mining_simulator import (  # noqa: E402
    DEFAULT_EXPECTED_EXISTING_RATE,
    DEFAULT_EXPECTED_FORMAL_GT_COUNT,
    DEFAULT_K_VALUES,
    ExtraMiningSimulatorConfig,
    run_extra_mining_simulator,
    synthetic_self_test,
)
from videocutler.ext_stageb_ovvis.analysis.extra_mining_score_registry import build_default_variants  # noqa: E402


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _parse_csv_ints(value: str):
    if isinstance(value, tuple):
        return value
    return tuple(int(x.strip()) for x in str(value).split(',') if x.strip())


def _parse_csv_floats(value: str):
    if isinstance(value, tuple):
        return value
    return tuple(float(x.strip()) for x in str(value).split(',') if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run formal-aligned extra mining score simulator on existing WSOVVIS run outputs")
    p.add_argument("--run_root", default=None)
    p.add_argument("--runtime_output_root", default=None)
    p.add_argument("--repo_root", default=None)
    p.add_argument("--dataset_name", default="lvvis_train_base", choices=("lvvis_train_base", "lvvis_val"))
    p.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline", "gt_upper_bound"))
    p.add_argument("--formal_split", default="base_unobserved")
    p.add_argument("--device", default="cpu")
    p.add_argument("--stage_id", default="softem_aug")
    p.add_argument("--checkpoint_stage", default="softem_aug", choices=("prealign", "softem_base", "softem_aug"))
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--sidecar_root", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--smoke", type=_parse_bool, default=False)
    p.add_argument("--smoke_max_trajectories", type=int, default=128)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--batch_size_clips", type=int, default=16)
    p.add_argument("--k_values", type=_parse_csv_ints, default=DEFAULT_K_VALUES)
    p.add_argument("--primary_k", type=int, default=3)
    p.add_argument("--variant_names", nargs="*", default=None)
    p.add_argument("--list_variants", action="store_true")
    p.add_argument("--expected_formal_gt_count", type=int, default=DEFAULT_EXPECTED_FORMAL_GT_COUNT)
    p.add_argument("--expected_existing_gt_in_extra_rate", type=float, default=DEFAULT_EXPECTED_EXISTING_RATE)
    p.add_argument("--expected_rate_tolerance", type=float, default=0.003)
    p.add_argument("--enforce_expected_baseline", type=_parse_bool, default=True)
    p.add_argument("--obs_sim_max", type=float, default=0.90)
    p.add_argument("--alpha_obs", type=float, default=0.25)
    p.add_argument("--topm_values", type=_parse_csv_ints, default=(2, 3, 5))
    p.add_argument("--lse_taus", type=_parse_csv_floats, default=(0.05, 0.10, 0.20))
    p.add_argument("--hub_prior_topm", type=int, default=20)
    p.add_argument("--density_k_values", type=_parse_csv_ints, default=(10, 20))
    p.add_argument("--mmr_top_l", type=int, default=50)
    p.add_argument("--show_progress", type=_parse_bool, default=True)
    p.add_argument("--write_row_level_debug", type=_parse_bool, default=False)
    p.add_argument("--self_test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_variants:
        print(json.dumps([v.__dict__ for v in build_default_variants()], ensure_ascii=False, indent=2))
        return 0
    if args.self_test:
        print(json.dumps(synthetic_self_test(), ensure_ascii=False, indent=2))
        return 0
    if not args.run_root:
        raise SystemExit("--run_root is required unless --self_test or --list_variants is used")
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    runtime_output_root = Path(args.runtime_output_root).expanduser().resolve() if args.runtime_output_root else repo_root
    config = ExtraMiningSimulatorConfig(
        run_root=Path(args.run_root).expanduser().resolve(),
        runtime_output_root=runtime_output_root,
        dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(args.trajectory_source_branch),
        formal_split=str(args.formal_split),
        device=str(args.device),
        stage_id=str(args.stage_id),
        checkpoint_stage=str(args.checkpoint_stage),
        checkpoint_path=Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else None,
        sidecar_root=Path(args.sidecar_root).expanduser().resolve() if args.sidecar_root else None,
        output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
        subset_fraction=None if args.subset_fraction is None else float(args.subset_fraction),
        batch_size_clips=max(1, int(args.batch_size_clips)),
        k_values=tuple(int(x) for x in args.k_values),
        primary_k=int(args.primary_k),
        variant_names=tuple(str(x) for x in args.variant_names) if args.variant_names else None,
        expected_formal_gt_count=None if args.expected_formal_gt_count < 0 else int(args.expected_formal_gt_count),
        expected_existing_gt_in_extra_rate=None if args.expected_existing_gt_in_extra_rate < 0 else float(args.expected_existing_gt_in_extra_rate),
        expected_rate_tolerance=float(args.expected_rate_tolerance),
        enforce_expected_baseline=bool(args.enforce_expected_baseline),
        obs_sim_max=float(args.obs_sim_max),
        alpha_obs=float(args.alpha_obs),
        topm_values=tuple(int(x) for x in args.topm_values),
        lse_taus=tuple(float(x) for x in args.lse_taus),
        hub_prior_topm=int(args.hub_prior_topm),
        density_k_values=tuple(int(x) for x in args.density_k_values),
        mmr_top_l=int(args.mmr_top_l),
        show_progress=bool(args.show_progress),
        write_row_level_debug=bool(args.write_row_level_debug),
    )
    result = run_extra_mining_simulator(config)
    print(result["variant_table_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
