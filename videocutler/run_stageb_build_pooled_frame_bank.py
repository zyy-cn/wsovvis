from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from videocutler.ext_stageb_ovvis.banks.frame_pooled_bank import build_pooled_frame_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trajectory-level pooled frame asset for G7 training.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--dataset_name", default="lvvis_train_base", choices=("lvvis_train_base", "lvvis_val"))
    parser.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline", "gt_upper_bound"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_max_trajectories", type=int, default=128)
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    requested_output_root = Path(args.output_root).expanduser()
    if requested_output_root.is_absolute():
        output_root = requested_output_root
    else:
        cwd_text = os.environ.get("PWD", "").strip()
        cwd = Path(cwd_text) if cwd_text else Path.cwd()
        output_root = cwd / requested_output_root
    result = build_pooled_frame_bank(
        output_root=output_root,
        dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(args.trajectory_source_branch),
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
    )
    summary_path = _repo_root() / "codex" / "outputs" / "G7_training" / "g7_pooled_frame_bank_summary.json"
    _write_json(summary_path, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
