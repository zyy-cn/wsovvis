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

from videocutler.ext_stageb_ovvis.eval.external_lvvis import ExternalLVVISEvalConfig, run_external_lvvis_eval
from videocutler.ext_stageb_ovvis.eval.g8_bridge import build_cli_contract_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G8 LV-VIS external evaluation CLI.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_external_lvvis_eval(
        ExternalLVVISEvalConfig(
            exp_name=args.exp_name,
            output_root=Path(args.output_root).expanduser().resolve(),
            seed=int(args.seed),
            smoke=bool(args.smoke),
        )
    )
    payload = {
        **result,
        "cli": build_cli_contract_summary("contracts/cli/run_stageb_eval_lvvis.cli_contract.json"),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
